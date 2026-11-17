// r4d_qkv_norm_rope.hip - r4d_qkv_norm_rope: the per-head norms and the rotary of separate q, k and
// v projections in one dispatch -- Gemma 4's attention prologue.
//
// Between its projections and the cache store a Gemma 4 attention block runs five single-pass
// kernels over a few dozen heads: the weightless value norm (rmsnorm_p), the q and k norms
// (rmsnorm, one row per head) and two rotaries (rope). At decode each is a handful of workgroups,
// so this fuses the five into one dispatch, one workgroup per (token, head).
//
// Byte-identical to the five (see r4d_rmsnorm_quant_i8.hip for why that matters), because:
//
//   * each norm's sum of squares uses its standalone kernel's partition. rmsnorm folds a row over
//     kR4dSumLanes' 256 lanes whatever its width (r4d_common.h) and rmsnorm_p runs at 256; lane
//     t < min(256, head_dim) folds columns t, t+256, ... in order and the waves are summed in
//     order through LDS. Here a workgroup is head_dim threads, and thread t < min(256, head_dim)
//     folds the same columns in the same order while the rest hold zero, as the standalone
//     kernel's surplus lanes do;
//   * the norm output is rounded to bf16 as the standalone norm stores it, and the rotary reads that;
//   * the rotary's per-pair arithmetic is rope_kernel's (powf ladder, sincosf, `x0*c - x1*s`,
//     `x0*s + x1*c`), under the library's -ffp-contract=off.
//
// `v_from_k` is Gemma's global class: k == v, so v is the value norm of the raw k, taken before the
// k norm overwrites it, in the order the unfused block issues them. `v_norm` 0 leaves v as it is
// (a Qwen3-shaped layer, the DFlash2 drafter's: q and k normed, v not), so the launch is the two
// norms and two rotaries, and with the store v's heads only store.
//
// `qkv_norm_rope_kv_store` is the same launch followed by `kv_store`: the block of a k head (and of
// a v head) ends holding its token's finished head, so it also writes it into the paged cache at
// the token's slot, converted as kv_store converts it (the int8 cache: the head's codes against its
// absmax / 127 as an f16, kv_store_i8's bytes; bf16: the values as they are). The head is staged in
// LDS for that, since the int8 store takes the head's absmax first. Byte-identical to the pair at
// either cache; the k, v and q rows are still written, as the pair leaves them.
//
// Returns a negative code instead of throwing: a kernel entry point has no caller to catch.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_chain.h"
#include <cstdint>
#include <algorithm>
#include <cstring>

namespace {

constexpr uint32_t kWave = 32;

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

// The standalone norm's partition of one head's sum of squares inside a head_dim-wide workgroup:
// thread t < P = min(256, HD) folds t, t+P, ...; the waves are then summed in order.
template <uint32_t HD>
__device__ __forceinline__ float head_ss(const uint16_t* __restrict__ x, uint32_t P, float* lds) {
    float ss = 0.0f;
    if (threadIdx.x < P)
        for (uint32_t i = threadIdx.x; i < HD; i += P) { const float v = bf2f(x[i]); ss += v * v; }
    ss = r4d_wave_sum(ss);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = ss;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < HD / kWave; ++w) s += lds[w];
    return s;
}

// The rotary over a normed head staged in `nb`, written to `out` (and to `st` when the head is also
// stored): rope_kernel's body for pair i.
template <uint32_t HD>
__device__ __forceinline__ void rotate(const uint16_t* __restrict__ nb, uint16_t* __restrict__ out,
                                       float p0, uint32_t rot, uint32_t nfreq, int den, float theta,
                                       uint16_t* __restrict__ st) {
    const uint32_t half = rot / 2, t = threadIdx.x;
    if (t < half) {
        float inv = 1.0f / powf(theta, (float)(2 * t) / (float)den);
        if (t >= nfreq) inv = 0.0f;
        const float ang = p0 * inv;
        float c, s;
        sincosf(ang, &s, &c);
        const float x0 = bf2f(nb[t]), x1 = bf2f(nb[t + half]);
        const uint16_t y0 = f32_to_bf16(x0 * c - x1 * s), y1 = f32_to_bf16(x0 * s + x1 * c);
        out[t]        = y0;
        out[t + half] = y1;
        if (st) { st[t] = y0; st[t + half] = y1; }
    }
    for (uint32_t i = rot + t; i < HD; i += HD) {   // the unrotated tail
        out[i] = nb[i];
        if (st) st[i] = nb[i];
    }
}

// One head into the paged cache (r4d_model_bf16.hip's kv_store kernels). `src` is the head in LDS,
// complete before the call. KV 2, a bf16 cache: the values as they are, at element offset `off`.
// KV 1, the int8 cache: the head's int8 codes at byte offset `off` and its f16 scale (absmax / 127,
// the codes taken against the rounded scale) at byte offset `soff`, as kv_store_i8 writes them;
// `red` is a wave's slot each for the block's absmax, and the call holds a barrier, so the whole
// block makes it.
template <uint32_t HD, int KV>
__device__ __forceinline__ void kv_put(void* __restrict__ kv, size_t off, size_t soff,
                                       const uint16_t* __restrict__ src, float* red) {
    const uint32_t t = threadIdx.x;
    if (KV == 1) {
        const float f = bf2f(src[t]);
        float am = r4d_wave_max(fabsf(f));
        if ((t & 31u) == 0u) red[t >> 5] = am;
        __syncthreads();
        am = 0.0f;
#pragma unroll
        for (uint32_t w = 0; w < HD / kWave; ++w) am = fmaxf(am, red[w]);
        const __fp16 s16 = (__fp16)fminf(am / 127.0f, 515.0f);
        const float sc = (float)s16;
        const float r = sc > 0.0f ? rintf(f / sc) : 0.0f;
        ((uint8_t*)kv)[off + t] = (uint8_t)(int8_t)(int)fminf(fmaxf(r, -127.0f), 127.0f);
        if (t == 0u) *(uint16_t*)((uint8_t*)kv + soff) = __builtin_bit_cast(uint16_t, s16);
    } else if (KV == 2) {
        for (uint32_t i = t; i < HD; i += HD) ((uint16_t*)kv)[off + i] = src[i];
    }
}

// blockIdx.y: [0, nq) a query head, [nq, nq+nkv) a key head (and with VK its value, from the raw
// key), [nq+nkv, nq+2nkv) a value head normalised in place (without VK).
//
// KV 1 or 2 also stores each k and v head into the paged cache [n_blocks, nkv, block_size, row] at
// slot[m] (k at column 0, v at HD; row 2*HD, or 2*HD + 4 at the int8 cache, whose two scales
// follow the codes), for rows m < n_slot whose slot is in range; KV 0 stores nothing and ignores
// the cache arguments.
// The body: its blocks return early (rows past the cliff, value heads), so the chained form
// (r4d_chain.h) wraps it below rather than bracketing it in place.
template <uint32_t HD, bool WF32, bool VK, int KV>
__device__ __forceinline__ void qkv_norm_rope_body(
    uint16_t* __restrict__ q, uint16_t* __restrict__ k, uint16_t* __restrict__ v,
    const int32_t* __restrict__ pos, const void* __restrict__ qw, const void* __restrict__ kw,
    uint32_t nq, uint32_t nkv, uint32_t pq, uint32_t pk, uint32_t pv,
    uint32_t q_ld, uint32_t k_ld, uint32_t v_ld, float eps, float wadd,
    uint32_t rot, uint32_t nfreq, int den, float theta, float rscale, uint32_t n_pos,
    uint32_t m_rows, const int32_t* __restrict__ slot, void* __restrict__ kvc, uint32_t n_slot,
    uint32_t block_size, uint32_t n_blocks, uint32_t vnorm) {
    __shared__ float l1[HD / kWave], l2[HD / kWave];
    __shared__ uint16_t nb[HD];
    __shared__ uint16_t sb[KV ? HD : 1];           // the head this block stores
    __shared__ uint16_t vb[(KV && VK) ? HD : 1];   // and, under VK, the value from the raw k
    const uint32_t m = blockIdx.x, hh = blockIdx.y, t = threadIdx.x;
    if (m >= m_rows) return;   // a row the launch added past the full-card cliff (r4d_cliff_rows)
    const float div = rscale != 0.0f ? rscale : 1.0f;
    // rope rotates only the rows its positions cover; a row past them is normed and not rotated
    const float p0 = m < n_pos ? (float)pos[m] / div : 0.0f;
    // The row's cache slot, as kv_store reads it: a negative or out-of-range slot stores nothing.
    // Uniform over the block, so the barriers below are too. ROW is a token's row in a head.
    constexpr size_t ROW = KV == 1 ? 2 * HD + 4 : 2 * HD;
    size_t base = 0;
    bool put = false;
    if (KV && m < n_slot) {
        const int32_t sl = slot[m];
        if (sl >= 0 && (uint32_t)sl / block_size < n_blocks) {
            put = true;
            base = ((size_t)((uint32_t)sl / block_size) * nkv * block_size +
                    (size_t)((uint32_t)sl % block_size)) * ROW;
        }
    }
    const size_t head_step = (size_t)block_size * ROW;

    if (hh >= nq + nkv) {   // a value head, in place: rmsnorm_p (or, without v_norm, only stored)
        const uint32_t h = hh - nq - nkv;
        uint16_t* __restrict__ x = v + (size_t)m * v_ld + (size_t)h * HD;
        if (vnorm) {
            const float s = rsqrtf(head_ss<HD>(x, pv, l1) / (float)HD + eps);
            __syncthreads();   // every thread has read x before any writes it
            for (uint32_t i = t; i < HD; i += HD) {
                const uint16_t y = f32_to_bf16(bf2f(x[i]) * s);
                x[i] = y;
                if (KV) sb[i] = y;
            }
        } else if (KV) {
            for (uint32_t i = t; i < HD; i += HD) sb[i] = x[i];
        }
        if (KV && put) {
            __syncthreads();
            const size_t r0 = base + (size_t)h * head_step;
            kv_put<HD, KV>(kvc, r0 + HD, r0 + 2 * HD + 2, sb, l1);
        }
        return;
    }
    const bool isq = hh < nq;
    const uint32_t h = isq ? hh : hh - nq;
    uint16_t* __restrict__ x = isq ? q + (size_t)m * q_ld + (size_t)h * HD
                                   : k + (size_t)m * k_ld + (size_t)h * HD;
    const void* __restrict__ w = isq ? qw : kw;

    if (VK && !isq) {   // v = rmsnorm_p(raw k), before the k norm rewrites k
        uint16_t* __restrict__ vo = v + (size_t)m * v_ld + (size_t)h * HD;
        const float sv = rsqrtf(head_ss<HD>(x, pv, l2) / (float)HD + eps);
        for (uint32_t i = t; i < HD; i += HD) {
            const uint16_t y = f32_to_bf16(bf2f(x[i]) * sv);
            vo[i] = y;
            if (KV && VK) vb[i] = y;
        }
    }
    const float s = rsqrtf(head_ss<HD>(x, isq ? pq : pk, l1) / (float)HD + eps);
    for (uint32_t i = t; i < HD; i += HD) {
        const float g = WF32 ? ((const float*)w)[i] : bf2f(((const uint16_t*)w)[i]);
        nb[i] = f32_to_bf16(bf2f(x[i]) * s * (g + wadd));
    }
    __syncthreads();   // the normed head is complete before any pair reads across it
    uint16_t* const st = (KV && !isq) ? sb : nullptr;
    if (m < n_pos) rotate<HD>(nb, x, p0, rot, nfreq, den, theta, st);
    else for (uint32_t i = t; i < HD; i += HD) { x[i] = nb[i]; if (st) st[i] = nb[i]; }
    if (KV && !isq && put) {
        __syncthreads();   // the stored head (and VK's value) complete in LDS
        const size_t r0 = base + (size_t)h * head_step;
        kv_put<HD, KV>(kvc, r0, r0 + 2 * HD, sb, l1);
        if (VK) kv_put<HD, KV>(kvc, r0 + HD, r0 + 2 * HD + 2, vb, l2);
    }
}

#define R4D_QNR_PARAMS uint16_t* __restrict__ q, uint16_t* __restrict__ k, uint16_t* __restrict__ v, \
    const int32_t* __restrict__ pos, const void* __restrict__ qw, const void* __restrict__ kw,        \
    uint32_t nq, uint32_t nkv, uint32_t pq, uint32_t pk, uint32_t pv,                                \
    uint32_t q_ld, uint32_t k_ld, uint32_t v_ld, float eps, float wadd,                              \
    uint32_t rot, uint32_t nfreq, int den, float theta, float rscale, uint32_t n_pos,                \
    uint32_t m_rows, const int32_t* __restrict__ slot, void* __restrict__ kvc, uint32_t n_slot,      \
    uint32_t block_size, uint32_t n_blocks, uint32_t vnorm
#define R4D_QNR_ARGS q, k, v, pos, qw, kw, nq, nkv, pq, pk, pv, q_ld, k_ld, v_ld, eps, wadd, rot, nfreq, \
    den, theta, rscale, n_pos, m_rows, slot, kvc, n_slot, block_size, n_blocks, vnorm
// The plain kernel, and the chained one (r4dchain*, r4d_chain.h) wrapping the same body.
template <uint32_t HD, bool WF32, bool VK, int KV>
__global__ __launch_bounds__(HD) void qkv_norm_rope_kernel(R4D_QNR_PARAMS) {
    qkv_norm_rope_body<HD, WF32, VK, KV>(R4D_QNR_ARGS);
}
template <uint32_t HD, bool WF32, bool VK, int KV>
__global__ __launch_bounds__(HD) void r4dchain_qkv_norm_rope(R4D_QNR_PARAMS) {
    r4d_chain_wait();
    qkv_norm_rope_body<HD, WF32, VK, KV>(R4D_QNR_ARGS);
    r4d_chain_done();
}
#undef R4D_QNR_ARGS
#undef R4D_QNR_PARAMS

}  // namespace

// ============================================================== the plugin ABI entry points
enum { QN_Q = 0, QN_K, QN_V, QN_POS, QN_QW, QN_KW, QN_SLOT, QN_CACHE };

// Both ops' launch: `store` reads qkv_norm_rope_kv_store's slot mapping and cache (operands 6, 7)
// and its kv_store parameters.
static int qkv_norm_rope_launch(const RadArgs* a, RadStream stream, bool store) {
    int emit_rc = RAD_OK;
    const RadTensor* q  = r4d_opt(a, QN_Q);
    const RadTensor* k  = r4d_opt(a, QN_K);
    const RadTensor* v  = r4d_opt(a, QN_V);
    const RadTensor* ps = r4d_opt(a, QN_POS);
    const RadTensor* qw = r4d_opt(a, QN_QW);
    const RadTensor* kw = r4d_opt(a, QN_KW);
    if (!q || !k || !v || !ps || !qw || !kw) return R4D_NO(RAD_E_INVAL);
    if (q->dtype != RAD_BF16 || k->dtype != RAD_BF16 || v->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (ps->dtype != RAD_I32 || ps->rank != 1 || !r4d_contig(ps)) return R4D_NO(RAD_E_SHAPE);
    if (qw->dtype != kw->dtype || (qw->dtype != RAD_BF16 && qw->dtype != RAD_F32)) return R4D_NO(RAD_E_DTYPE);
    const long long hd  = r4d_geti_or(a, "head_dim", 0);
    const long long nq  = r4d_geti_or(a, "n_head", 0);
    const long long nkv = r4d_geti_or(a, "n_head_kv", 0);
    const long long rot = r4d_geti_or(a, "rotary_dim", hd);
    const long long vk  = r4d_geti_or(a, "v_from_k", 0);
    // v_norm 0: no value norm (Gemma 3; a Qwen3-shaped layer such as the DFlash2 drafter's): the v
    // heads are left out of the grid, or with the store only store.
    const long long vn  = r4d_geti_or(a, "v_norm", 1);
    if (hd != 128 && hd != 256 && hd != 512) return R4D_NO(RAD_E_SHAPE);
    if (vk && !vn) return R4D_NO(RAD_E_INVAL);   // v from k is v's norm
    /* rot 0 is the NoPE layer: the norm still runs, and `rotate` with half = 0 only copies the
     * unrotated tail, which is the whole head. */
    if (nq < 0 || nkv <= 0 || rot < 0 || rot > hd || (rot & 1)) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(qw) != hd || r4d_numel(kw) != hd) return R4D_NO(RAD_E_SHAPE);
    const char* mode = r4d_gets(a, "mode");
    if (mode && std::strcmp(mode, "neox")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (r4d_getf(a, "yarn_factor", 0.0) != 0.0) return R4D_NO(RAD_E_UNSUPPORTED);
    // rope's ladder, resolved as r4d_rad_rope resolves it
    const char* lad = r4d_gets(a, "ladder");
    long long den = rot;
    if (lad) {
        if (!std::strcmp(lad, "head")) den = hd;
        else if (std::strcmp(lad, "rotary")) return R4D_NO(RAD_E_UNSUPPORTED);
    }
    const long long nfreq = rot / 2;
    const long long rot_l = (den == hd && rot < hd) ? hd : rot;

    long long M = 0, Mk = 0, Mv = 0;
    /* n_head 0: the k (and v) heads alone, q neither read nor written (a context pass's keys) */
    const long long q_ld = nq ? r4d_rows_pitch(q, nq * hd, &M) : 0;
    const long long k_ld = r4d_rows_pitch(k, nkv * hd, &Mk);
    const long long v_ld = r4d_rows_pitch(v, nkv * hd, &Mv);
    if (q_ld < 0 || k_ld < 0 || v_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (!nq) M = Mk;
    const long long Mp = r4d_numel(ps);
    if (Mp <= 0 || M < Mp || Mk != M || Mv != M) return R4D_NO(RAD_E_SHAPE);

    // The standalone kernels' partitions: rmsnorm (one row per head) folds over kR4dSumLanes'
    // lanes at any width, rmsnorm_p runs at 256.
    const uint32_t pq = nq ? (uint32_t)std::min<long long>(kR4dSumLanes, hd) : (uint32_t)hd;
    const uint32_t pk = (uint32_t)std::min<long long>(kR4dSumLanes, hd);
    const uint32_t pv = (uint32_t)std::min<long long>(256, hd);
    const float eps   = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd  = (float)r4d_getf(a, "wadd", 0.0);
    const float theta = (float)r4d_getf(a, "theta", 10000.0);
    const float rsc   = (float)r4d_getf(a, "scale", 1.0);
    /* v's own heads: normed (and stored), stored only, or with neither nothing to do */
    const uint32_t heads = (uint32_t)(nq + nkv + ((vk || (!vn && !store)) ? 0 : nkv));
    // A tp-4 verify of the 31B is 16 rows x 16 heads of 8 wavefronts: exactly a full card, the
    // cliff r4d_cliff_rows steps past with one idle row.
    const dim3 grid(r4d_cliff_rows((uint32_t)M, heads, (uint32_t)(hd / kWave)), heads);
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;   /* RadKernelChainFn, decode */

    // The store's geometry, checked as kv_store_geom checks it (r4d_model_bf16.hip).
    int kvw = 0;   // 0 none, 1 the int8 cache, 2 bf16
    const RadTensor* sm = nullptr;
    const RadTensor* kv = nullptr;
    long long bs = 0, nblk = 0, nslot = 0;
    if (store) {
        sm = r4d_opt(a, QN_SLOT);
        kv = r4d_opt(a, QN_CACHE);
        if (!sm || !kv) return R4D_NO(RAD_E_INVAL);
        if (sm->dtype != RAD_I32 || !r4d_contig(sm) || !r4d_contig(kv)) return R4D_NO(RAD_E_STRIDE);
        if (kv->dtype == RAD_I8)        kvw = 1;
        else if (kv->dtype == RAD_BF16) kvw = 2;
        else return R4D_NO(RAD_E_DTYPE);
        bs = r4d_geti_or(a, "block_size", 0);
        if (bs <= 0 || kv->rank != 4 || kv->shape[1] != nkv || kv->shape[2] != bs ||
            kv->shape[3] != (kvw == 1 ? 2 * hd + 4 : 2 * hd))
            return R4D_NO(RAD_E_SHAPE);
        nblk  = kv->shape[0];
        nslot = r4d_numel(sm);
        if (nslot <= 0 || nslot > M) return R4D_NO(RAD_E_SHAPE);
    }

#define R4D_QNR_T(HD, WF, VK, KV) emit_rc = chain ? R4D_QNR_C(HD, WF, VK, KV, true) : R4D_QNR_C(HD, WF, VK, KV, false)
#define R4D_QNR_C(HD, WF, VK, KV, CH) R4D_EMIT(stream, (CH ? r4dchain_qkv_norm_rope<HD, WF, VK, KV> : qkv_norm_rope_kernel<HD, WF, VK, KV>), grid, dim3(HD), 0, \
    (uint16_t*)r4d_wp(q), (uint16_t*)r4d_wp(k), (uint16_t*)r4d_wp(v), (const int32_t*)r4d_p(ps),           \
    r4d_p(qw), r4d_p(kw), (uint32_t)nq, (uint32_t)nkv, pq, pk, pv, (uint32_t)q_ld, (uint32_t)k_ld,          \
    (uint32_t)v_ld, eps, wadd, (uint32_t)rot_l, (uint32_t)nfreq, (int)den, theta, rsc, (uint32_t)Mp,        \
    (uint32_t)M, (const int32_t*)(sm ? r4d_p(sm) : nullptr), (void*)(kv ? r4d_wp(kv) : nullptr),           \
    (uint32_t)nslot, (uint32_t)bs, (uint32_t)nblk, (uint32_t)(vn ? 1 : 0))
#define R4D_QNR_KV(HD, WF, VK)                                                                          \
    do { if (kvw == 1) R4D_QNR_T(HD, WF, VK, 1); else if (kvw == 2) R4D_QNR_T(HD, WF, VK, 2);           \
         else R4D_QNR_T(HD, WF, VK, 0); } while (0)
#define R4D_QNR(HD)                                                                                     \
    do { if (qw->dtype == RAD_F32) { if (vk) R4D_QNR_KV(HD, true, true);  else R4D_QNR_KV(HD, true, false); } \
         else                      { if (vk) R4D_QNR_KV(HD, false, true); else R4D_QNR_KV(HD, false, false); } } while (0)
    if (hd == 128)      R4D_QNR(128);
    else if (hd == 256) R4D_QNR(256);
    else                R4D_QNR(512);
#undef R4D_QNR
#undef R4D_QNR_KV
#undef R4D_QNR_T
#undef R4D_QNR_C
    return emit_rc;
}

extern "C" int r4d_rad_qkv_norm_rope(const RadArgs* a, RadStream stream) {
    return qkv_norm_rope_launch(a, stream, false);
}

extern "C" int r4d_rad_qkv_norm_rope_kv_store(const RadArgs* a, RadStream stream) {
    return qkv_norm_rope_launch(a, stream, true);
}
