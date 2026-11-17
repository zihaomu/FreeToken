// The QSA indexer's compressed key: one 128-dim vector per four tokens of key history.
//
// Qwen3.8-Flash-Next's full-attention layers use a small MQA indexer that scores 4-token blocks of
// the key history; attention reads only the best 512, so its cost is constant past 2051 tokens.
// This kernel builds the block keys those scores use (docs/QSA.md has the decomposition).
//
// The reference (`Qwen4ExpTextQSAIndexer`) recomputes every block key per query. A block key does
// not depend on the query and is fixed once its four tokens exist, so it is computed once, when
// the fourth token arrives, and cached (64 bytes per token per layer instead of a pass over the
// history every step).
//
// Arithmetic, in the reference's order and rounding:
//
//   pooled = mean over the block's `ratio` raw keys, accumulated in f32 and rounded to bf16
//   normed = RMSNorm(pooled) over `n`, gain (wadd + w)
//   out    = rope(normed) at the position of the block's first token
//
// The mean is rounded to bf16 before the norm, as in the reference
// (`key_groups.float().mean(dim=1).to(raw_keys.dtype)`); carrying f32 would give systematically
// different values.
//
// The rotation matches r4d_qk_norm_rope_gate: `1/powf(theta, 2i/rot)`, position divided by
// `rscale` in f32, `sincosf`, neox pairing at stride `rot/2`. sincos is computed rather than read
// from that kernel's table, which holds only this step's positions (a block completing at decode
// starts three positions back).
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "rad_yarn.h"
#include "r4d_emit.h"
#include "r4d_common.h"   // r4d_wave_sum

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kMaxN = 256;      // the index head dim; 128 on this model, bounded for the LDS

/* Position component read by frequency `i` for multi-component rotaries: mode 2 is contiguous
 * sections, 3 Qwen3-VL's interleaved ones (as in r4d_model_bf16.hip's rope). */
__device__ __forceinline__ int qbk_comp(int mode, int i, int s0, int s1, int s2) {
    if (mode == 3) {
        if (i % 3 == 1 && i < 3 * s1) return 1;
        if (i % 3 == 2 && i < 3 * s2) return 2;
        return 0;
    }
    return i < s0 ? 0 : (i < s0 + s1 ? 1 : 2);
}

#define QSA_BLOCK_KEY_PARAMS                                                                    \
    const uint16_t* __restrict__ k, const float* __restrict__ w,                                \
    const int32_t* __restrict__ page, const int32_t* __restrict__ bpos,                         \
    uint16_t* __restrict__ out, int n, int ratio, int rot, float eps, float wadd,               \
    float theta, float rscale, long long pos0, int k_ld, int o_ld, int mode, int s0, int s1,    \
    int s2, int bstep, int mc, RadYarn yarn
__device__ __forceinline__ void qsa_block_key_body(QSA_BLOCK_KEY_PARAMS) {

    __shared__ float sh[kMaxN];
    __shared__ float red[kMaxN / 32];

    const int b = blockIdx.x;
    const int t = threadIdx.x;

    /* Mean in f32 over the block's rows, rounded to bf16 once. */
    float acc = 0.0f;
    if (t < n) {
        const size_t base = (size_t)b * ratio;
        for (int j = 0; j < ratio; ++j)
            acc += __bfloat162float(((const __hip_bfloat16*)k)[(base + j) * k_ld + t]);
        acc = __bfloat162float(__float2bfloat16(acc / (float)ratio));
    }

    /* RMSNorm over n: a block reduction, since n (128) spans several waves. */
    float ss = t < n ? acc * acc : 0.0f;
    ss = r4d_wave_sum(ss);
    if ((t & 31) == 0) red[t >> 5] = ss;
    __syncthreads();
    float sum = 0.0f;
    const int nw = (blockDim.x + 31) / 32;
    for (int i = 0; i < nw; ++i) sum += red[i];
    const float inv = rsqrtf(sum / (float)n + eps);

    /* f32 through the norm and rotation, rounded once at the store, matching
     * r4d_qk_norm_rope_gate for the model's own K. */
    if (t < n) sh[t] = acc * inv * (wadd + w[t]);
    __syncthreads();

    if (t >= n) return;
    float res = sh[t];
    if (t < rot) {
        const int half = rot >> 1;
        const bool first = t < half;
        const int pt = first ? t + half : t - half;
        const int ci = first ? t : t - half;
        /* The block's position: from the work list (bpos) when given, so one launch serves blocks
         * of different sequences (decode); otherwise the pos0 + b*ratio tiling (a prefill chunk's
         * contiguous run). With `mc` the list holds a block's three components side by side,
         * blocks `bstep` apart, and each frequency reads its section's component. */
        const int comp = mc ? qbk_comp(mode, ci, s0, s1, s2) : 0;
        const long long bp = bpos ? (long long)bpos[(size_t)b * bstep + comp]
                                  : (pos0 + (long long)b * ratio);
        const float p = (float)bp / (rscale != 0.0f ? rscale : 1.0f);
        const float iv = yarn.frequency(1.0f / powf(theta, (float)(2 * ci) / (float)rot), ci);
        float s, c;
        sincosf(p * iv, &s, &c);
        if (yarn.factor != 0.0f) { c *= yarn.magnitude; s *= yarn.magnitude; }
        const float xp = sh[pt];
        res = first ? (res * c - xp * s) : (xp * s + res * c);
    }
    /* With `page`, block b is written to its page of the paged block-key store (docs/QSA.md);
     * a negative entry (no page) is skipped. Without it the write is contiguous. */
    long long orow = b;
    if (page) {
        const int32_t p = page[b];
        if (p < 0) return;
        orow = p;
    }
    ((__hip_bfloat16*)out)[(size_t)orow * o_ld + t] = __float2bfloat16(res);
}

/* The kernel around the body, whose blocks and threads return where their work ends: CH `r4dchain`
 * is the chained form (r4d_chain.h). */
template <class CH>
__global__ __launch_bounds__(kMaxN)
void qsa_block_key_kernel(QSA_BLOCK_KEY_PARAMS) {
    if (CH::on) r4d_chain_wait();
    qsa_block_key_body(k, w, page, bpos, out, n, ratio, rot, eps, wadd, theta, rscale, pos0, k_ld,
                       o_ld, mode, s0, s1, s2, bstep, mc, yarn);
    if (CH::on) r4d_chain_done();
}
#undef QSA_BLOCK_KEY_PARAMS

}  // namespace

//   k    [M*ratio][n]  bf16, `k_ld` elements a row (0 = n)
//   w    [n]           f32
//   out  [M][n]        bf16, `o_ld` elements a row (0 = n)
//
// Without `bpos`, `pos0` is the position of the first block's first token and block i starts at
// `pos0 + i * ratio`.
static int r4d_qsa_block_key_bf16_emit(long k, long w, long page, long bpos, long out, int M,
                                      int n, int ratio, int rot, float eps, float wadd,
                                      float theta, float rscale, long long pos0, int k_ld,
                                      int o_ld, int mode, int s0, int s1, int s2, int bstep, int mc,
                                      RadStream stream, RadYarn yarn, bool chain) {
    if (!k || !w || !out || (pos0 < 0 && !bpos)) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n <= 0 || n > kMaxN || ratio <= 0) return R4D_NO(RAD_E_SHAPE);
    if (rot < 0 || rot > n || (rot & 1)) return R4D_NO(RAD_E_SHAPE);
    if (!k_ld) k_ld = n;
    if (!o_ld) o_ld = n;
    /* n threads rounded up to whole waves for the block reduction (extra threads contribute 0). */
    const int thr = ((n + 31) / 32) * 32;
    /* Each form named at its own launch: a test reads the kernel a launch site names. */
#define R4D_QBKEY(CH_)                                                                        \
    R4D_EMIT(stream, qsa_block_key_kernel<CH_>, dim3((unsigned)M), dim3((unsigned)thr), 0,    \
             (const uint16_t*)k, (const float*)w, (const int32_t*)page, (const int32_t*)bpos, \
             (uint16_t*)out, n, ratio, rot, eps, wadd, theta, rscale, pos0, k_ld, o_ld, mode, \
             s0, s1, s2, bstep, mc, yarn)
    const int rc = chain ? R4D_QBKEY(r4dchain) : R4D_QBKEY(R4dUnchained);
#undef R4D_QBKEY
    return rc;
}

// ============================================================== the plugin ABI entry point

enum { QBK_K = 0, QBK_W = 1, QBK_PAGE = 2, QBK_POS = 3, QBK_OUT = 4 };

extern "C" int r4d_rad_qsa_block_key(const RadArgs* a, RadStream stream) {
    const RadTensor* k = r4d_opt(a, QBK_K);
    const RadTensor* w = r4d_opt(a, QBK_W);
    const RadTensor* pg = r4d_opt(a, QBK_PAGE);      /* optional: absent is a contiguous write */
    const RadTensor* bp = r4d_opt(a, QBK_POS);       /* optional: absent is the pos0 tiling */
    const RadTensor* o = r4d_opt(a, QBK_OUT);
    if (!k || !w || !o) return R4D_NO(RAD_E_INVAL);
    /* The destination may be a rank-4 KV cache: the block-key store is a KV group with kv_heads 1,
     * block_size = ratio and head_dim n/(2*ratio), so a page holds exactly n values (docs/QSA.md).
     * It is checked as [pages, n] by the product of the inner dims, not the rank. */
    if (k->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (o->rank < 2) return R4D_NO(RAD_E_SHAPE);
    const long long n     = r4d_geti_or(a, "n", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long rot   = r4d_geti_or(a, "rotary_dim", 0);
    const long long pos0  = r4d_geti_or(a, "pos0", -1);
    if (n <= 0 || ratio <= 0 || (pos0 < 0 && !bp)) return R4D_NO(RAD_E_INVAL);
    /* block_size sizes the block-key KV group and must equal the compress ratio, so a page holds
     * exactly one block. */
    const long long bs = r4d_geti_or(a, "block_size", ratio);
    if (bs != ratio) return R4D_NO(RAD_E_SHAPE);
    /* With a page list, M is the number of blocks completing this step, not the store's height
     * (the whole pool). */
    const long long M = pg ? r4d_numel(pg) : o->shape[0];
    const long long ow = o->shape[0] > 0 ? r4d_numel(o) / o->shape[0] : 0;
    if (M < 1 || ow != n) return R4D_NO(RAD_E_SHAPE);
    if (pg && o->shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    /* `pos` is one position per block, or a rank-2 [>= M, 1 or 3] column slice with a block's
     * components side by side and blocks a row pitch apart (qsa_work's output). */
    int bstep = 1;
    if (bp && bp->rank == 2) {
        if (bp->shape[0] < M || bp->stride[1] != 1 || bp->stride[0] < bp->shape[1])
            return R4D_NO(RAD_E_SHAPE);
        bstep = (int)bp->stride[0];
    } else if (bp && r4d_numel(bp) < M) {
        return R4D_NO(RAD_E_SHAPE);
    }
    /* Multi-component rotary: mode mrope or imrope with `sections` and three columns of `pos`
     * (qsa_work's output with its `rope` operand). A single column is read as NeoX in any mode. */
    const char* mode_s = r4d_gets(a, "mode");
    int mode = 0, sec[3] = { 0, 0, 0 }, n_sec = 0, mc = 0;
    if (mode_s && !std::strcmp(mode_s, "mrope")) mode = 2;
    else if (mode_s && !std::strcmp(mode_s, "imrope")) mode = 3;
    else if (mode_s && std::strcmp(mode_s, "neox")) return R4D_NO(RAD_E_SCHEMA);
    for (const char* v = r4d_gets(a, "sections"); v && *v;) {
        char* e = nullptr;
        const long x = std::strtol(v, &e, 10);
        if (e == v) break;
        if (x <= 0 || n_sec >= 3) return R4D_NO(RAD_E_SHAPE);
        sec[n_sec++] = (int)x;
        v = e;
    }
    if (mode && n_sec > 0 && bp && bp->rank == 2 && bp->shape[1] >= 3) mc = 1;
    /* The key operand must be exactly M * ratio rows; a wrong height would silently pool rows
     * from neighbouring blocks. */
    if (k->shape[0] != M * ratio || k->shape[1] != n) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(w) < n) return R4D_NO(RAD_E_SHAPE);
    const long long kp = r4d_row_pitch(k), op = r4d_row_pitch(o);
    if (kp < 0 || op < 0) return R4D_NO(RAD_E_STRIDE);
    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn) ||
        (yarn.factor != 0.0f && (rot <= 0 || r4d_getf(a, "scale", 1.0) != 1.0)))
        return R4D_NO(RAD_E_INVAL);
    return r4d_qsa_block_key_bf16_emit(
        r4d_addr(k), r4d_addr(w), r4d_addr(pg), r4d_addr(bp), r4d_addr(o),
        (int)M, (int)n, (int)ratio, (int)rot,
        (float)r4d_getf(a, "eps", 1e-6), (float)r4d_getf(a, "wadd", 1.0),
        (float)r4d_getf(a, "theta", 10000.0), (float)r4d_getf(a, "scale", 1.0),
        pos0, (int)kp, (int)op, mode, sec[0], sec[1], sec[2], bstep, mc, stream, yarn,
        r4d_geti_or(a, "chain", 0) != 0);
}

