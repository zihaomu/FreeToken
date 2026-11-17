// r4d_ple_conv_bf16.hip -- Per-Layer Embedding's dilated depthwise convolution, its silu, and the
// residual add that finishes the layer; and, as `causal_conv1d`, the same convolution with no
// activation, no residual and no dilation, which is LFM2's short convolution.
//
//   y[t][c] = resid[t][c] + silu( sum_j w[c][j] * x[t - (width-1-j)*dilation][c] )
//
// libref/ref_ple.cpp is the oracle and docs/OPS.md the contract. The taps are `dilation` apart, so
// at width 4 and dilation 3 they sit 9, 6, 3 and 0 timesteps back and a sequence carries a history
// of (width-1)*dilation = 9, not 3. An off-by-one in a convolution cache yields fluent output from
// a slightly wrong context, so the indexing mirrors ref's line for line: state slot `i` holds
// relative position `i - hist`, a read at `src < 0` becomes `cs[slot][c][off + src + hist]`, and
// `has_init` 0 means that window is zeros, not the slot's stale contents.
//
// One thread per channel: a depthwise convolution never mixes channels, so only time must stay
// together. A thread walks a tile of tokens; consecutive threads own consecutive channels, so a
// wave loads one contiguous 512-byte slice of a token row.
//
// The grid is (channel blocks, token tiles, sequences). Token tiles are counted from the batch's
// total token count, an upper bound on any one sequence's; a short sequence's extra blocks find
// `t0 >= slen` and return. A per-sequence count would mean reading `cu` on the host.
//
// The taps are read from memory, not kept in a shift register: the window would need `hist + 1`
// registers indexed by a runtime value, which compiles to a select tree (see CU_STATIC in
// r4d_gdn_conv_w4_h128_bf16.hip), and a compile-time window would need a specialisation per
// (width, dilation). The re-reads hit a tile working set of (BT + hist) * 512 bytes (21 KB at
// BT 32, L0-resident), and the op is bound by the three arrays it must touch anyway.
//
// The window write-back is in the same launch; see the note at the end of the kernel. It relies
// on `hist < kPcBT`, asserted below.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"

#include <cstdint>

namespace {

enum {
    kPcThreads = 256,
    kPcBT      = 32,   /* tokens a thread walks; the tile's working set is (BT + hist) * 512 B */
    kPcMaxW    = 8,    /* taps held in registers, so the weight loop unrolls to a static bound */
    kPcMaxHist = 16,   /* (width-1)*dilation, the write-back's register window; see cPleConv */
};

/* The write-back lives in tile 0, which is sound only while the deepest tap is inside the first
 * tile. At hist >= BT a second block would read history and the write-back would overwrite bytes
 * it is still reading, silently corrupting a sequence's first tokens. The constraint table caps
 * hist <= 16; the tile size is the other half of that bound. */
static_assert(kPcMaxHist < kPcBT, "the conv history must fit inside tile 0 for the fused write-back");

/* expf rather than __expf: the op is bound by memory, not the transcendental, and the oracle uses
 * std::exp. The instruction-bound GDN conv makes the opposite trade. */
__device__ __forceinline__ float pc_silu(float z) { return z / (1.0f + expf(-z)); }

/* SILU is ple_conv's activation; causal_conv1d (LFM2's short convolution) is the same window with
 * no activation, no residual and dilation one. */
#define PLE_CONV_PARAMS                                                                         \
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,                             \
    uint16_t* __restrict__ cs, const int32_t* __restrict__ cu,                                  \
    const uint16_t* __restrict__ resid, const int32_t* __restrict__ sidx,                       \
    const int32_t* __restrict__ hini, const int32_t* __restrict__ nacc,                         \
    uint16_t* __restrict__ y, const uint16_t* __restrict__ bias, uint32_t n, uint32_t wid,      \
    uint32_t dil, uint32_t hist, uint32_t xld, uint32_t rld, uint32_t yld, uint32_t wld,        \
    long long cs_slot, long long cs_ch, long long cs_tok, long long state_len
template <bool SILU>
__device__ __forceinline__ void ple_conv_body(PLE_CONV_PARAMS) {

    const uint32_t c = blockIdx.x * kPcThreads + threadIdx.x;
    if (c >= n) return;
    const uint32_t s = blockIdx.z;
    const int32_t bos = cu[s];
    const int32_t slen = cu[s + 1] - bos;
    if (slen <= 0) return;
    const int32_t t0 = (int32_t)(blockIdx.y * kPcBT);
    if (t0 >= slen) return;
    const int32_t t1 = min(t0 + (int32_t)kPcBT, slen);

    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    if (slot < 0) return;
    /* A cold slot's window is zeros: a fresh sequence on a recycled slot would otherwise convolve
     * the previous sequence's tail into its first `hist` tokens. */
    const bool warm = cs && (!hini || hini[s] != 0);
    const long long off = nacc ? (long long)nacc[s] - 1 : 0;

    /* This channel's taps, read once for the whole tile. */
    float wt[kPcMaxW];
#pragma unroll
    for (uint32_t j = 0; j < kPcMaxW; ++j) wt[j] = j < wid ? bf16_to_f32(w[(size_t)c * wld + j]) : 0.0f;

    const uint16_t* __restrict__ xc = x + (size_t)bos * xld + c;
    const uint16_t* __restrict__ rc = resid ? resid + (size_t)bos * rld + c : nullptr;
    uint16_t* __restrict__ dst = cs ? cs + (size_t)slot * cs_slot + (size_t)c * cs_ch : nullptr;
    const uint16_t* __restrict__ hc = warm ? dst + off * cs_tok : nullptr;
    uint16_t* __restrict__ yc = y + (size_t)bos * yld + c;

    const float b0 = bias ? bf16_to_f32(bias[c]) : 0.0f;
    for (int32_t t = t0; t < t1; ++t) {
        float acc = b0;
#pragma unroll
        for (uint32_t j = 0; j < kPcMaxW; ++j) {
            if (j < wid) {
                const int32_t src = t - (int32_t)hist + (int32_t)(j * dil);
                const float v = src >= 0 ? bf16_to_f32(xc[(size_t)src * xld])
                                         : (hc ? bf16_to_f32(hc[(src + (int32_t)hist) * cs_tok]) : 0.0f);
                acc += v * wt[j];
            }
        }
        const float r = rc ? bf16_to_f32(rc[(size_t)t * rld]) : 0.0f;
        yc[(size_t)t * yld] = f32_to_bf16(r + (SILU ? pc_silu(acc) : acc));
    }

    /* The window this step leaves, in the same launch with no barrier. History is read only where
     * `t - hist + j*dilation < 0`, i.e. by tokens below `hist` (9, against a 32-token tile), so
     * tile 0 is the only block touching state[slot][c]. One thread owns the whole channel there, so
     * its reads precede its writes in program order. At decode the op's cost is the dispatch itself
     * (flat from M=1 to M=8), so a second launch would roughly double it.
     *
     * The layout is libref/ref_ple.cpp's, chosen by `num_accepted`. Present, this is a decode step
     * whose drafts a verify may reject: the history read shifts down by one and every input of the
     * step follows it, slot i holding position i + 1 of (window read ++ step), so a next step that
     * kept k tokens reads at k - 1. Absent, every token commits and the last `hist` inputs land at
     * offset zero. The two agree at one token. Slots past `state_len` are not written; the manager
     * sizes the window hist + n_spec, enough for a step of 1 + n_spec tokens.
     *
     * Values are copied as bf16 bits, the same tensor's bytes on both sides. The step's rows come
     * from the read-only `x`, so other tiles reading them is no hazard. Every history value is
     * loaded before the first store. */
    if (!dst || blockIdx.y != 0) return;
    uint16_t keep[kPcMaxHist];
    if (nacc) {
#pragma unroll
        for (uint32_t i = 0; i + 1 < kPcMaxHist; ++i)
            if (i + 1 < hist) keep[i] = hc ? hc[(size_t)(i + 1) * cs_tok] : (uint16_t)0;
#pragma unroll
        for (uint32_t i = 0; i + 1 < kPcMaxHist; ++i)
            if (i + 1 < hist) dst[(size_t)i * cs_tok] = keep[i];
        for (int32_t t = 0; t < slen && (long long)hist - 1 + t < state_len; ++t)
            dst[(size_t)(hist - 1 + (uint32_t)t) * cs_tok] = xc[(size_t)t * xld];
        return;
    }
#pragma unroll
    for (uint32_t i = 0; i < kPcMaxHist; ++i) {
        if (i < hist) {
            const int32_t src = slen - (int32_t)hist + (int32_t)i;
            keep[i] = src >= 0 ? xc[(size_t)src * xld]
                               : (hc ? hc[(src + (int32_t)hist) * cs_tok] : (uint16_t)0);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < kPcMaxHist; ++i)
        if (i < hist) dst[(size_t)i * cs_tok] = keep[i];
}

/* The kernel around the body, whose blocks return where their work ends: CH `r4dchain` is the
 * chained form (r4d_chain.h), taken at decode heights. */
template <bool SILU, class CH>
__global__ __launch_bounds__(kPcThreads)
void ple_conv_kernel(PLE_CONV_PARAMS) {
    if (CH::on) r4d_chain_wait();
    ple_conv_body<SILU>(x, w, cs, cu, resid, sidx, hini, nacc, y, bias, n, wid, dil, hist, xld, rld,
                        yld, wld, cs_slot, cs_ch, cs_tok, state_len);
    if (CH::on) r4d_chain_done();
}
#undef PLE_CONV_PARAMS

enum { PC_X = 0, PC_W, PC_CS, PC_CU, PC_RESID, PC_CIDX, PC_HINIT, PC_NACC, PC_Y };
/* ple_conv_bias: ple_conv's operands with `b`, the per-channel bias inside the silu (Mamba-2's
 * conv1d), before `y`. */
enum { PB_B = PC_Y, PB_Y };
/* causal_conv1d's operands: ple_conv's without `resid`. */
enum { CC_X = 0, CC_W, CC_CS, CC_CU, CC_CIDX, CC_HINIT, CC_NACC, CC_Y };

int conv_launch(const RadArgs* a, RadStream s, bool silu, const RadTensor* x, const RadTensor* w,
                const RadTensor* cs, const RadTensor* cu, const RadTensor* resid,
                const RadTensor* cidx, const RadTensor* hini, const RadTensor* nacc,
                const RadTensor* y, const RadTensor* b = nullptr) {
    if (!x || !w || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || w->dtype != RAD_BF16 || y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (resid && resid->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (b && b->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (cs && cs->data && cs->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (cidx && cidx->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    /* `has_init` must be int32 (RadBatch::ctx_lens, like every derived buffer): read as bytes, a
     * ctx_len of 512 has a zero low byte and the history would be dropped. Anything else is
     * refused. */
    if (hini && hini->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (nacc && nacc->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    const long long n   = r4d_geti_or(a, "n", 0);
    const long long wid = r4d_geti_or(a, "width", 0);
    const long long dil = r4d_geti_or(a, "dilation", 1);
    if (n <= 0 || wid <= 0 || dil <= 0) return R4D_NO(RAD_E_SHAPE);
    if (wid > kPcMaxW) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long hist = (wid - 1) * dil;
    if (hist > kPcMaxHist) return R4D_NO(RAD_E_UNSUPPORTED);

    long long T = 0, ty = 0, tr = 0, wrows = 0;
    const long long xld = r4d_rows_pitch(x, n, &T);
    const long long yld = r4d_rows_pitch(y, n, &ty);
    const long long rld = resid ? r4d_rows_pitch(resid, n, &tr) : 0;
    const long long wld = r4d_rows_pitch(w, wid, &wrows);
    if (xld < 0 || yld < 0 || wld < 0 || (resid && rld < 0)) return R4D_NO(RAD_E_STRIDE);
    /* The row count comes from the operands, and they must agree: `M` is ranged, so the arg holds
     * the band's upper bound and a grid sized by it would run off the arena. */
    if (T <= 0 || ty != T || (resid && tr != T)) return R4D_NO(RAD_E_SHAPE);
    if (wrows < n) return R4D_NO(RAD_E_SHAPE);
    if (b && (r4d_numel(b) != n || !r4d_contig(b))) return R4D_NO(RAD_E_SHAPE);

    const long long N = r4d_numel(cu) - 1;
    if (N <= 0) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);

    /* [n_slots, n, state_len]. A state too shallow for the deepest tap is refused here; the
     * rollback offset cannot be checked without reading `num_accepted` on the host, which is why
     * the manager sizes the window hist + n_spec. */
    long long cs_slot = 0, cs_ch = 0, cs_tok = 0, state_len = 0;
    const bool have_cs = cs && cs->data && hist > 0;
    if (have_cs) {
        if (cs->rank != 3) return R4D_NO(RAD_E_SHAPE);
        if (r4d_dim(cs, 1) < n || r4d_dim(cs, 2) < hist) return R4D_NO(RAD_E_SHAPE);
        cs_slot   = r4d_stride(cs, 0);
        cs_ch     = r4d_stride(cs, 1);
        cs_tok    = r4d_stride(cs, 2);
        state_len = r4d_dim(cs, 2);
    }

    const uint32_t cb = (uint32_t)((n + kPcThreads - 1) / kPcThreads);
    const uint32_t tb = (uint32_t)((T + kPcBT - 1) / kPcBT);
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && T <= 64;
#define R4D_PC(S_, CH_)                                                                         \
    hipLaunchKernelGGL((ple_conv_kernel<S_, CH_>), dim3(cb, tb, (uint32_t)N),                   \
                       dim3(kPcThreads), 0, (hipStream_t)s,                                      \
                       (const uint16_t*)r4d_addr(x), (const uint16_t*)r4d_addr(w),               \
                       have_cs ? (uint16_t*)r4d_addr(cs) : nullptr,                              \
                       (const int32_t*)r4d_addr(cu),                                             \
                       resid ? (const uint16_t*)r4d_addr(resid) : nullptr,                       \
                       cidx ? (const int32_t*)r4d_addr(cidx) : nullptr,                          \
                       hini ? (const int32_t*)r4d_addr(hini) : nullptr,                          \
                       nacc ? (const int32_t*)r4d_addr(nacc) : nullptr,                          \
                       (uint16_t*)r4d_addr(y), b ? (const uint16_t*)r4d_addr(b) : nullptr,       \
                       (uint32_t)n, (uint32_t)wid, (uint32_t)dil,                                \
                       (uint32_t)hist, (uint32_t)xld, (uint32_t)rld, (uint32_t)yld, (uint32_t)wld, \
                       cs_slot, cs_ch, cs_tok, state_len)
    if (silu) { if (chain) R4D_PC(true, r4dchain);  else R4D_PC(true, R4dUnchained); }
    else      { if (chain) R4D_PC(false, r4dchain); else R4D_PC(false, R4dUnchained); }
#undef R4D_PC
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

}  // namespace

extern "C" int r4d_rad_ple_conv(const RadArgs* a, RadStream s) {
    return conv_launch(a, s, true, r4d_opt(a, PC_X), r4d_opt(a, PC_W), r4d_opt(a, PC_CS),
                       r4d_opt(a, PC_CU), r4d_opt(a, PC_RESID), r4d_opt(a, PC_CIDX),
                       r4d_opt(a, PC_HINIT), r4d_opt(a, PC_NACC), r4d_opt(a, PC_Y));
}

extern "C" int r4d_rad_ple_conv_bias(const RadArgs* a, RadStream s) {
    const RadTensor* b = r4d_opt(a, PB_B);
    if (!b) return R4D_NO(RAD_E_INVAL);
    return conv_launch(a, s, true, r4d_opt(a, PC_X), r4d_opt(a, PC_W), r4d_opt(a, PC_CS),
                       r4d_opt(a, PC_CU), r4d_opt(a, PC_RESID), r4d_opt(a, PC_CIDX),
                       r4d_opt(a, PC_HINIT), r4d_opt(a, PC_NACC), r4d_opt(a, PB_Y), b);
}

extern "C" int r4d_rad_causal_conv1d(const RadArgs* a, RadStream s) {
    if (r4d_geti_or(a, "dilation", 1) != 1) return R4D_NO(RAD_E_INVAL);
    return conv_launch(a, s, false, r4d_opt(a, CC_X), r4d_opt(a, CC_W), r4d_opt(a, CC_CS),
                       r4d_opt(a, CC_CU), nullptr, r4d_opt(a, CC_CIDX), r4d_opt(a, CC_HINIT),
                       r4d_opt(a, CC_NACC), r4d_opt(a, CC_Y));
}
