// r4d_vit_bf16.hip -- vision-tower ops: layer norm with bias, learned-grid position embedding, and
// a biased GEMM with the activation and residual folded into its store.
//
// Arithmetic matches libref (ref_elementwise.cpp, ref_gemm.cpp), rounding where it rounds: f32
// accumulation, RTNE bf16 stores, and the GEMM epilogue rounds after the biased product, after the
// activation and after the residual, as the unfused chain would.
//
// Not gemm_bf16_nt_tiled: that kernel needs K in whole 64-element tiles (a tower MLP's 4304 is 16
// past one) and has no bias. Two kernels, both with a zero-staged K tail, no split-K, and the
// epilogue in the store: a 64 x 64 tile with both operands staged through LDS for small grids, and
// for large ones (tall activations, thousands of patches) a 128 x 128 tile that reads the weight
// straight into registers and stages only the activation.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"

#include <cstdint>

namespace {

constexpr int kThr = 256;

__device__ __forceinline__ float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}
__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}

template <uint32_t NW>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    __syncthreads();
    return s;
}

template <bool F32>
__device__ __forceinline__ float ldw(const void* p, uint32_t i) {
    return F32 ? ((const float*)p)[i] : bf2f(((const unsigned short*)p)[i]);
}

// ---------------------------------------------------------------------------- layernorm
// ref_layernorm: y = (x - mean) * rsqrt(var + eps) * w + b, biased variance, two passes (mean, then
// squared deviations); one-pass E[x^2] - E[x]^2 cancels badly when the mean is large relative to
// the spread. One block per row. `TX` is the row's element: bf16, or f32 for a residual stream kept
// in f32 (a video decoder's), read as it is; the output is bf16 either way.
__device__ __forceinline__ float ldx(const unsigned short* p, uint32_t i) { return bf2f(p[i]); }
__device__ __forceinline__ float ldx(const float* p, uint32_t i) { return p[i]; }

template <bool WF32, typename TX>
__global__ __launch_bounds__(kThr) void layernorm_kernel(
    const TX* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b,
    unsigned short* __restrict__ y, uint32_t n, float eps, uint32_t x_ld, uint32_t y_ld) {
    __shared__ float lds[kThr / 32];
    const uint32_t m = blockIdx.x;
    const TX* __restrict__ xr = x + (size_t)m * x_ld;
    unsigned short* __restrict__ yr = y + (size_t)m * y_ld;
    float s = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) s += ldx(xr, i);
    const float mean = block_sum<kThr / 32>(s, lds) / (float)n;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        const float d = ldx(xr, i) - mean;
        ss += d * d;
    }
    const float r = rsqrtf(block_sum<kThr / 32>(ss, lds) / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float v = (ldx(xr, i) - mean) * r * ldw<WF32>(w, i);
        if (b) v += ldw<WF32>(b, i);
        yr[i] = f32_to_bf16(v);
    }
}

// ---------------------------------------------------------------------------- grid_embed
// ref_grid_embed: a side x side table resampled bilinearly to row m's grid (corners aligned, edge
// taps clamped), four taps summed in f32 in libref's order, rounded, added to x and rounded again.
// `crd` is [4, planes]: row, column, grid height, grid width.
template <bool TF32>
__global__ __launch_bounds__(kThr) void grid_embed_kernel(
    unsigned short* __restrict__ x, const void* __restrict__ tab, const int32_t* __restrict__ crd,
    uint32_t n, int side, uint32_t x_ld, uint32_t plane) {
    const uint32_t m = blockIdx.x;
    const int g[4] = { crd[m], crd[plane + m], crd[2 * plane + m], crd[3 * plane + m] };
    int th[2], tw[2];
    float wh[2], ww[2];
    auto axis = [&](int idx, int size, int* t, float* wt) {
        const float src = (float)idx * (float)(side - 1) / (float)(size - 1 > 1 ? size - 1 : 1);
        const float fl = floorf(src);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int tap = (int)fl + k;
            t[k] = tap < 0 ? 0 : (tap > side - 1 ? side - 1 : tap);
            const float d = fabsf(src - fl - (float)k);
            wt[k] = 1.0f - d > 0.0f ? 1.0f - d : 0.0f;
        }
    };
    axis(g[0], g[2], th, wh);
    axis(g[1], g[3], tw, ww);
    unsigned short* __restrict__ xr = x + (size_t)m * x_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float acc = 0.0f;
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
#pragma unroll
            for (int kj = 0; kj < 2; ++kj)
                acc += ldw<TF32>(tab, (uint32_t)(th[ki] * side + tw[kj]) * n + i) *
                       (wh[ki] * ww[kj]);
        const float pe = bf2f(f32_to_bf16(acc));
        xr[i] = f32_to_bf16(bf2f(xr[i]) + pe);
    }
}

// ---------------------------------------------------------------------------- grid_embed_aa
// ref_grid_embed_aa: torch's antialiased half-pixel bilinear resample (SigLIP2's position table).
// Thread 0 builds the row's two tap lists (at most `side` each) in LDS; threads then take channels,
// summing taps in libref's order (columns within a row, then rows), rounded once before the add.
__device__ __forceinline__ int aa_taps(int at, int size, int side, float* w, int* lo) {
    size = size < 1 ? 1 : size;
    at = at < 0 ? 0 : (at >= size ? size - 1 : at);
    const float scale = (float)side / (float)size;
    const float support = scale > 1.0f ? scale : 1.0f, inv = 1.0f / support;
    const float center = scale * ((float)at + 0.5f);
    int a = (int)(center - support + 0.5f), b = (int)(center + support + 0.5f);
    a = a < 0 ? 0 : a;
    b = b > side ? side : b;
    float total = 0.0f;
    for (int j = a; j < b; ++j) {
        const float d = fabsf(((float)j - center + 0.5f) * inv);
        w[j - a] = d < 1.0f ? 1.0f - d : 0.0f;
        total += w[j - a];
    }
    for (int j = a; j < b; ++j) w[j - a] = total != 0.0f ? w[j - a] / total : 0.0f;
    *lo = a;
    return b > a ? b - a : 0;
}

template <bool TF32>
__global__ __launch_bounds__(kThr) void grid_embed_aa_kernel(
    unsigned short* __restrict__ x, const void* __restrict__ tab, const int32_t* __restrict__ crd,
    uint32_t n, int side, uint32_t x_ld, uint32_t plane) {
    extern __shared__ float aa_w[];           // [2 * side]: the row taps, then the column taps
    __shared__ int aa_span[4];                // h0, nh, w0, nw
    const uint32_t m = blockIdx.x;
    if (threadIdx.x == 0) {
        aa_span[1] = aa_taps(crd[m], crd[2 * plane + m], side, aa_w, &aa_span[0]);
        aa_span[3] = aa_taps(crd[plane + m], crd[3 * plane + m], side, aa_w + side, &aa_span[2]);
    }
    __syncthreads();
    const int h0 = aa_span[0], nh = aa_span[1], w0 = aa_span[2], nw = aa_span[3];
    unsigned short* __restrict__ xr = x + (size_t)m * x_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float acc = 0.0f;
        for (int y = 0; y < nh; ++y) {
            float row = 0.0f;
            for (int c = 0; c < nw; ++c)
                row += ldw<TF32>(tab, (uint32_t)((h0 + y) * side + w0 + c) * n + i) * aa_w[side + c];
            acc += row * aa_w[y];
        }
        const float pe = bf2f(f32_to_bf16(acc));
        xr[i] = f32_to_bf16(bf2f(xr[i]) + pe);
    }
}

// ---------------------------------------------------------------------------- grid_embed_cpe
// ref_grid_embed_cpe: RADIO's cropped position embedding, torch's half-pixel bilinear without
// antialias onto a square of the grid's larger side, cropped to the grid. Two taps an axis, the
// row blended first as libref does; rounded once before the add.
__device__ __forceinline__ void cpe_taps(int at, int d, int side, int* t, float* w) {
    d = d < 1 ? 1 : d;
    at = at < 0 ? 0 : (at >= d ? d - 1 : at);
    float src = (float)side / (float)d * ((float)at + 0.5f) - 0.5f;
    src = src < 0.0f ? 0.0f : src;
    int i0 = (int)src;
    i0 = i0 > side - 1 ? side - 1 : i0;
    t[0] = i0;
    t[1] = i0 < side - 1 ? i0 + 1 : i0;
    w[1] = src - (float)i0;
    w[0] = 1.0f - w[1];
}

template <bool TF32>
__global__ __launch_bounds__(kThr) void grid_embed_cpe_kernel(
    unsigned short* __restrict__ x, const void* __restrict__ tab, const int32_t* __restrict__ crd,
    uint32_t n, int side, uint32_t x_ld, uint32_t plane, const unsigned short* __restrict__ pre,
    int npre) {
    const uint32_t m = blockIdx.x;
    const int g[4] = { crd[m], crd[plane + m], crd[2 * plane + m], crd[3 * plane + m] };
    unsigned short* __restrict__ xr = x + (size_t)m * x_ld;
    if (g[0] < 0) {
        // A row ahead of its segment's patches: the learned bf16 row its column names.
        if (!pre || npre <= 0) return;
        const int k = g[1] < 0 ? 0 : (g[1] >= npre ? npre - 1 : g[1]);
        for (uint32_t i = threadIdx.x; i < n; i += kThr)
            xr[i] = f32_to_bf16(bf2f(xr[i]) + bf2f(pre[(size_t)k * n + i]));
        return;
    }
    const int d = g[2] > g[3] ? g[2] : g[3];
    int th[2], tw[2];
    float wh[2], ww[2];
    cpe_taps(g[0], d, side, th, wh);
    cpe_taps(g[1], d, side, tw, ww);
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        const float r0 = ww[0] * ldw<TF32>(tab, (uint32_t)(th[0] * side + tw[0]) * n + i) +
                         ww[1] * ldw<TF32>(tab, (uint32_t)(th[0] * side + tw[1]) * n + i);
        const float r1 = ww[0] * ldw<TF32>(tab, (uint32_t)(th[1] * side + tw[0]) * n + i) +
                         ww[1] * ldw<TF32>(tab, (uint32_t)(th[1] * side + tw[1]) * n + i);
        const float pe = bf2f(f32_to_bf16(wh[0] * r0 + wh[1] * r1));
        xr[i] = f32_to_bf16(bf2f(xr[i]) + pe);
    }
}

// ---------------------------------------------------------------------------- gemm_nt_bias
#define VB_BK 64
#define VB_PITCH (VB_BK + 8)      // 144-byte LDS rows, as in gemm_bf16_nt_tiled (bank spread)

// TM x TN WMMA tiles per wave, eight waves as 2 x 4: BM = 2*TM*16, BN = 4*TN*16. ACT: 0 none,
// 1 exact GELU, 2 tanh GELU. Serves grids under 32 blocks of 128 x 128 at TM 2, TN 1 (64 x 64);
// larger grids take vit_gemm_wa_kernel below.
template <int TM, int TN, int ACT, bool RES, bool BF32>
__global__ __launch_bounds__(256) void vit_gemm_kernel(
    const unsigned short* __restrict__ A, unsigned lda, const unsigned short* __restrict__ W,
    unsigned ldw_, const void* __restrict__ bias, const unsigned short* R, unsigned ldr,
    unsigned short* C, unsigned ldc, int M, int K, int N) {
    constexpr int BM = 2 * TM * 16, BN = 4 * TN * 16;
    constexpr int CPR = VB_BK / 8;                      // 16-byte chunks a staged row
    constexpr int AL = BM * CPR / 256, WL = BN * CPR / 256;
    static_assert(AL >= 1 && WL >= 1 && BM * CPR % 256 == 0 && BN * CPR % 256 == 0,
                  "a tile is whole 16-byte chunks for every thread");
    __shared__ __attribute__((aligned(16))) unsigned short sA[BM * VB_PITCH];
    __shared__ __attribute__((aligned(16))) unsigned short sW[BN * VB_PITCH];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave / 4, wn = wave % 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    // Staging: chunk c = tid + 256 i is row c / CPR, elements (c % CPR) * 8. Rows past M or N
    // re-read the last valid row; their results are not stored.
    unsigned aoff[AL], woff[WL];
    int ae[AL], we[WL];
    unsigned short* adst[AL];
    unsigned short* wdst[WL];
#pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int gr = m0 + r < M ? m0 + r : M - 1;
        aoff[i] = (unsigned)gr * lda + (unsigned)e;
        ae[i] = e;
        adst[i] = sA + r * VB_PITCH + e;
    }
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int gr = n0 + r < N ? n0 + r : N - 1;
        woff[i] = (unsigned)gr * ldw_ + (unsigned)e;
        we[i] = e;
        wdst[i] = sW + r * VB_PITCH + e;
    }
    const unsigned short* fa = sA + (wm * TM * 16 + (lane & 15)) * VB_PITCH + 4 * (lane >> 4);
    const unsigned short* fw = sW + (wn * TN * 16 + (lane & 15)) * VB_PITCH + 4 * (lane >> 4);

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

    // The K tail is staged as zeros: K is a multiple of 8, so a 16-byte chunk is wholly inside or
    // wholly past K.
    const r4d_u32x4 zero = (r4d_u32x4)(0u);
    r4d_u32x4 ra[AL], rw[WL];                       // registers; uint4 would sit in scratch
#pragma unroll
    for (int i = 0; i < AL; ++i) ra[i] = ae[i] < K ? *(const r4d_u32x4*)(A + aoff[i]) : zero;
#pragma unroll
    for (int i = 0; i < WL; ++i) rw[i] = we[i] < K ? *(const r4d_u32x4*)(W + woff[i]) : zero;

    const int nk = (K + VB_BK - 1) / VB_BK;
    for (int kt = 0; kt < nk; ++kt) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < AL; ++i) *(r4d_u32x4*)adst[i] = ra[i];
#pragma unroll
        for (int i = 0; i < WL; ++i) *(r4d_u32x4*)wdst[i] = rw[i];
        __syncthreads();
        if (kt + 1 < nk) {
            const int k1 = (kt + 1) * VB_BK;
#pragma unroll
            for (int i = 0; i < AL; ++i)
                ra[i] = k1 + ae[i] < K ? *(const r4d_u32x4*)(A + aoff[i] + k1) : zero;
#pragma unroll
            for (int i = 0; i < WL; ++i)
                rw[i] = k1 + we[i] < K ? *(const r4d_u32x4*)(W + woff[i] + k1) : zero;
        }
#pragma unroll
        for (int s = 0; s < VB_BK / 16; ++s) {
            v8bf af[TM], bfr[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) af[i] = fragP(fa + i * 16 * VB_PITCH + s * 16);
#pragma unroll
            for (int j = 0; j < TN; ++j) bfr[j] = fragP(fw + j * 16 * VB_PITCH + s * 16);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = mma(af[i], bfr[j], acc[i][j]);
        }
    }

    // Lane l holds column (l & 15) of a tile and rows 8 (l >> 4) + e. The residual is read by the
    // thread that stores the same element, so R may alias C.
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
            const int mb = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
            if (n >= N) continue;
            const float bv = ldw<BF32>(bias, (uint32_t)n);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = mb + e;
                if (m >= M) continue;
                unsigned short t = f32_to_bf16(acc[i][j][e] + bv);
                if constexpr (ACT == 1) t = f32_to_bf16(gelu_erf(bf2f(t)));
                if constexpr (ACT == 2) t = f32_to_bf16(gelu_tanh(bf2f(t)));
                if constexpr (RES) t = f32_to_bf16(bf2f(R[(size_t)m * ldr + n]) + bf2f(t));
                C[(size_t)m * ldc + n] = t;
            }
        }
}


// ---------------------------------------------------------------------------- gemm_nt_bias, large grids
//
// 128 weight rows x 128 activation rows a block, eight waves as kVWW (weight) x kVWA
// (activation), a wave kVTW weight tiles x kVTA activation tiles. The weight is the WMMA A operand,
// read from global straight into registers; the activation is the B operand, staged through LDS
// and read by the four waves that share its rows. A lane's accumulator is then eight consecutive
// weight rows of one activation row, so the bias is eight consecutive values and the store (with
// its residual read) is one 16-byte access.
//
// Staging both operands through LDS with a 4 x 2 tile a wave (the kernel above at 128 x 128)
// costs 384 bytes of fragment reads and 128 of stores per bf16 WMMA: all of gfx1201's
// LDS bandwidth, 64 bytes a clock per CU against one bf16 WMMA every 16 clocks per SIMD
// (libr4d/isa/wmmamix.hip). Here it is 256 and 64, and the weight's 128 bytes come through the
// vector cache.
//
// K is relabelled inside a staged tile so that a lane's operand bytes are contiguous: lane half h,
// k step s and element e hold k = 32h + 8s + e of the tile, for both operands alike. That is a
// permutation of K applied to A and B together, so the product is the same sum; its rounding inside
// the WMMA differs from the canonical order in the last bit (rel_l2 ~4e-5 against it, well inside a
// bf16 output). It makes a lane's weight fragments for the tile one 64-byte run of its row, four
// 16-byte loads, and its activation fragment one 16-byte LDS read.
//
// A weight row stride that is a multiple of 8 KiB (K = 4096 or 8192) puts all sixteen rows a
// fragment load touches in one vector-cache set, and every load refetches them. The block's first K
// tile is therefore rotated by its grid position, which spreads the blocks resident at once over
// the K columns: +5% at K 8192, -1% at K 2048 (noise-level), the summation order changing with it.
// The real fix is a weight layout without the stride (a padded row reached 139 TFLOP/s at K 8192
// against 115 here), which needs a layout hook.
//
// Measured at the video decoder's shapes (M 25158, 300 W; the kernel above 93-98 TFLOP/s): this
// tile 133-137 at K 2048, 115 at K 8192. Rejected: 2 x 4 waves of 4 weight x 2 activation tiles
// (spills, -20%); 2 x 2 waves of 4 x 4 tiles at a 32-wide K tile (-22%: half-line weight loads,
// twice the barriers); 4 x 1 waves of 2 x 8 at a 32-wide K tile (-8%); eight waves a SIMD at a
// 32-wide K tile (-4%). kVGroup 1, 2, 4 and 16 are within 2% of 8 except at N 16384 (-9, -5, -3
// and -3%); 32 is 5-15% slower. The kernel above with the grouped raster: +3% at N 16384, -2% at
// K 8192. A uniform block base and 32-bit lane offsets for the loads: neutral. The weight through
// raw buffer loads: +1% at K 2048 and +4% at K 8192, but the same loads cost the fp8 prefill tile
// 20%, so neither tile uses them. The weight's loads nontemporal: -25..-50%.
constexpr uint32_t kVTW = 2, kVTA = 4, kVWW = 4, kVWA = 2;
constexpr uint32_t kVBN = kVWW * kVTW * 16, kVBM = kVWA * kVTA * 16;   // weight x activation rows
constexpr uint32_t kVThreads = kVWW * kVWA * 32;
constexpr uint32_t kVFK = 64;                                          // K elements a staged tile
constexpr uint32_t kVPB = kVFK * 2 + 16;                               // LDS row pitch, bytes
constexpr uint32_t kVCPR = kVFK * 2 / 16;                              // 16-byte chunks a row
constexpr uint32_t kVSPT = kVBM * kVCPR / kVThreads;                   // chunks a thread stages
constexpr uint32_t kVWC = kVFK / 16;                                   // a lane's weight chunks a tile
// Activation bands per raster group; see gemm_8a8_prefill_kernel (r4d_gemm_fp8a8.hip).
constexpr uint32_t kVGroup = 8;
static_assert(kVBM * kVCPR % kVThreads == 0, "whole chunks a thread");

// The store's three roundings, libref's order: the biased product, the activation, the residual.
template <int ACT, bool RES>
__device__ __forceinline__ unsigned short vit_finish(float acc, float bv, unsigned short r) {
    unsigned short t = f32_to_bf16(acc + bv);
    if constexpr (ACT == 1) t = f32_to_bf16(gelu_erf(bf2f(t)));
    if constexpr (ACT == 2) t = f32_to_bf16(gelu_tanh(bf2f(t)));
    if constexpr (RES) t = f32_to_bf16(bf2f(r) + bf2f(t));
    return t;
}

/* `vec` (host-checked): C, R and the bias take 16-byte accesses at every eight-column boundary. */
template <int ACT, bool RES, bool BF32>
__global__ __launch_bounds__(kVThreads) R4D_WAVES(6) void vit_gemm_wa_kernel(
    const unsigned short* __restrict__ A, unsigned lda, const unsigned short* __restrict__ W,
    unsigned ldw_, const void* __restrict__ bias, const unsigned short* R, unsigned ldr,
    unsigned short* C, unsigned ldc, int M, int K, int N, unsigned vec) {
    typedef r4d_u32x4 U4;
    // Double-buffered: one barrier a staged tile, which publishes it.
    __shared__ __attribute__((aligned(16))) unsigned char s_a[2][kVBM * kVPB];

    const uint32_t tid = threadIdx.x, lane = tid & 31u;
    // Uniform per wave, and through readfirstlane known so, so what derives from it stays scalar.
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const uint32_t ww = wave % kVWW, wa = wave / kVWW;

    // Grouped raster, as in gemm_8a8_prefill_kernel: kVGroup activation bands sweep N together.
    uint32_t bm = blockIdx.x, bn = blockIdx.y;
    if (gridDim.x > kVGroup) {
        const uint32_t per = kVGroup * gridDim.y;
        const uint32_t lin = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t first = (lin / per) * kVGroup, r = lin % per;
        const uint32_t gm = min(gridDim.x - first, kVGroup);
        bm = first + r % gm;
        bn = r / gm;
    }
    const uint32_t m0 = bm * kVBM, n0 = bn * kVBN;

    // Staging: eight consecutive threads cover a row's 128 bytes, so a load is whole cachelines.
    // Rows past M re-read row M-1; nothing past M is stored.
    const unsigned char* asrc[kVSPT];
    uint32_t adst[kVSPT], aoff[kVSPT];
#pragma unroll
    for (uint32_t i = 0; i < kVSPT; ++i) {
        const uint32_t c = tid + kVThreads * i, r = c / kVCPR, ch = c % kVCPR;
        asrc[i] = (const unsigned char*)(A + (size_t)min(m0 + r, (uint32_t)M - 1u) * lda);
        aoff[i] = ch * 8u;                                        // elements into the tile
        adst[i] = r * kVPB + ch * 16u;
    }
    // Weight: lane row (l & 15) of each tile, elements [32h, 32h + 32) of the staged tile. Rows past
    // N re-read row N-1; nothing past N is stored.
    const unsigned short* wsrc[kVTW];
#pragma unroll
    for (uint32_t i = 0; i < kVTW; ++i)
        wsrc[i] = W + (size_t)min(n0 + (ww * kVTW + i) * 16u + (lane & 15u), (uint32_t)N - 1u) * ldw_ +
                  (lane >> 4) * (kVFK / 2u);
    // This lane's activation fragment row and half in the staged tile.
    const uint32_t frow = (wa * kVTA * 16u + (lane & 15u)) * kVPB + (lane >> 4) * kVFK;

    v8f acc[kVTW][kVTA];
#pragma unroll
    for (uint32_t i = 0; i < kVTW; ++i)
#pragma unroll
        for (uint32_t j = 0; j < kVTA; ++j) acc[i][j] = (v8f)(0.f);

    /* Four k steps of one staged tile: the weight from `cur`, the activation from LDS. */
    auto mma_tile = [&](const unsigned char* sa, const U4 (&cur)[kVTW][kVWC]) {
#pragma unroll
        for (uint32_t s = 0; s < kVWC; ++s) {
            v8bf bfr[kVTA];
#pragma unroll
            for (uint32_t j = 0; j < kVTA; ++j)
                bfr[j] = __builtin_bit_cast(v8bf, *(const U4*)(sa + frow + j * 16u * kVPB + s * 16u));
#pragma unroll
            for (uint32_t i = 0; i < kVTW; ++i) {
                const v8bf af = __builtin_bit_cast(v8bf, cur[i][s]);
#pragma unroll
                for (uint32_t j = 0; j < kVTA; ++j) acc[i][j] = mma(af, bfr[j], acc[i][j]);
            }
        }
    };

    // Whole tiles in a rotated order, then the K tail (K is a multiple of 8, so a 16-byte chunk
    // is wholly inside or wholly past K) staged with zeros.
    const uint32_t nfull = (uint32_t)K / kVFK;
    if (nfull > 0u) {
        const uint32_t t0 = (bm + bn) % nfull;
        auto kt = [&](uint32_t t) { const uint32_t u = t0 + t; return (u >= nfull ? u - nfull : u) * kVFK; };
        U4 pa[kVSPT], w0[kVTW][kVWC], w1[kVTW][kVWC];
        /* Every load is issued unconditionally (the last tile's prefetch re-reads the last tile),
         * so the loop has no branch and its waits count exactly. */
        auto load = [&](uint32_t k0, U4 (&w)[kVTW][kVWC]) {
#pragma unroll
            for (uint32_t i = 0; i < kVSPT; ++i) pa[i] = *(const U4*)(asrc[i] + (size_t)(k0 + aoff[i]) * 2u);
#pragma unroll
            for (uint32_t i = 0; i < kVTW; ++i)
#pragma unroll
                for (uint32_t c = 0; c < kVWC; ++c) w[i][c] = *(const U4*)(wsrc[i] + k0 + c * 8u);
        };
        auto tile = [&](uint32_t t, uint32_t buf, U4 (&cur)[kVTW][kVWC], U4 (&nxt)[kVTW][kVWC]) {
#pragma unroll
            for (uint32_t i = 0; i < kVSPT; ++i) *(U4*)(s_a[buf] + adst[i]) = pa[i];
            lds_barrier();
            load(kt(min(t + 1u, nfull - 1u)), nxt);
            mma_tile(s_a[buf], cur);
        };
        load(kt(0), w0);
        uint32_t t = 0;
        for (; t + 1u < nfull; t += 2u) {
            tile(t, 0, w0, w1);
            tile(t + 1u, 1, w1, w0);
        }
        if (t < nfull) tile(t, 0, w0, w1);
    }
    if ((uint32_t)K % kVFK) {
        // The buffer the last whole tile did not use, last read two tiles back.
        unsigned char* const sa = s_a[nfull % 2u];
        const uint32_t k0 = nfull * kVFK;
        const U4 zero = (U4)(0u);
#pragma unroll
        for (uint32_t i = 0; i < kVSPT; ++i)
            *(U4*)(sa + adst[i]) = k0 + aoff[i] < (uint32_t)K
                                       ? *(const U4*)(asrc[i] + (size_t)(k0 + aoff[i]) * 2u) : zero;
        U4 w[kVTW][kVWC];
#pragma unroll
        for (uint32_t i = 0; i < kVTW; ++i)
#pragma unroll
            for (uint32_t c = 0; c < kVWC; ++c)
                w[i][c] = k0 + (lane >> 4) * (kVFK / 2u) + c * 8u < (uint32_t)K
                              ? *(const U4*)(wsrc[i] + k0 + c * 8u) : zero;
        lds_barrier();
        mma_tile(sa, w);
    }

    // D is [weight row][activation row]: eight consecutive columns of one C row a lane a tile.
    // The residual is read by the thread that stores the same element, so R may alias C.
#pragma unroll
    for (uint32_t i = 0; i < kVTW; ++i)
#pragma unroll
        for (uint32_t j = 0; j < kVTA; ++j) {
            const uint32_t m = m0 + (wa * kVTA + j) * 16u + (lane & 15u);
            const uint32_t n = n0 + (ww * kVTW + i) * 16u + (lane >> 4) * 8u;
            if (m >= (uint32_t)M) continue;
            const v8f v = acc[i][j];
            if (vec && n + 8u <= (uint32_t)N) {
                float bv[8];
                if constexpr (BF32) {
                    const float4 b0 = *(const float4*)((const float*)bias + n);
                    const float4 b1 = *(const float4*)((const float*)bias + n + 4);
                    bv[0] = b0.x; bv[1] = b0.y; bv[2] = b0.z; bv[3] = b0.w;
                    bv[4] = b1.x; bv[5] = b1.y; bv[6] = b1.z; bv[7] = b1.w;
                } else {
                    const uint4 b = *(const uint4*)((const unsigned short*)bias + n);
                    const uint32_t d[4] = { b.x, b.y, b.z, b.w };
#pragma unroll
                    for (uint32_t e = 0; e < 8; ++e) bv[e] = bf2f((unsigned short)(d[e / 2] >> (16 * (e % 2))));
                }
                uint32_t rr[4] = { 0u, 0u, 0u, 0u };
                if constexpr (RES) {
                    const uint4 r4 = *(const uint4*)(R + (size_t)m * ldr + n);
                    rr[0] = r4.x; rr[1] = r4.y; rr[2] = r4.z; rr[3] = r4.w;
                }
                uint32_t o[4];
#pragma unroll
                for (uint32_t e = 0; e < 8; e += 2)
                    o[e / 2] = (uint32_t)vit_finish<ACT, RES>(v[e], bv[e], (unsigned short)rr[e / 2]) |
                               ((uint32_t)vit_finish<ACT, RES>(v[e + 1], bv[e + 1],
                                                               (unsigned short)(rr[e / 2] >> 16)) << 16);
                *(uint4*)(C + (size_t)m * ldc + n) = make_uint4(o[0], o[1], o[2], o[3]);
            } else {
#pragma unroll
                for (uint32_t e = 0; e < 8; ++e) {
                    if (n + e >= (uint32_t)N) continue;
                    const unsigned short r = RES ? R[(size_t)m * ldr + n + e] : (unsigned short)0;
                    C[(size_t)m * ldc + n + e] = vit_finish<ACT, RES>(v[e], ldw<BF32>(bias, n + e), r);
                }
            }
        }
}

}  // namespace

// ============================================================== the plugin ABI entry points

/* ------------------------------------------------------------------ layernorm */
enum { LN_X = 0, LN_W, LN_B, LN_Y };

extern "C" int r4d_rad_layernorm(const RadArgs* a, RadStream stream) {
    const RadTensor* x = r4d_opt(a, LN_X);
    const RadTensor* w = r4d_opt(a, LN_W);
    const RadTensor* b = r4d_opt(a, LN_B);
    const RadTensor* y = r4d_opt(a, LN_Y);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    /* The row bf16 or f32, the output bf16. */
    if ((x->dtype != RAD_BF16 && x->dtype != RAD_F32) || y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    /* Gain and bias share one dtype, f32 or bf16. */
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (b && b->dtype != w->dtype) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(w) || (b && !r4d_contig(b))) return R4D_NO(RAD_E_STRIDE);

    const long long n = r4d_geti_or(a, "n", r4d_dim(w, (int)w->rank - 1));
    if (n <= 0 || r4d_numel(w) < n || (b && r4d_numel(b) < n)) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    const float eps = (float)r4d_getf(a, "eps", 1e-5);
#define R4D_LN(WF, TX)                                                                              \
    return R4D_EMIT(stream, (layernorm_kernel<WF, TX>), dim3((uint32_t)M), dim3(kThr), 0,          \
                    (const TX*)r4d_p(x), r4d_p(w), r4d_p(b), (unsigned short*)r4d_wp(y),           \
                    (uint32_t)n, eps, (uint32_t)x_ld, (uint32_t)y_ld)
    if (x->dtype == RAD_F32) {
        if (w->dtype == RAD_F32) R4D_LN(true, float);
        R4D_LN(false, float);
    }
    if (w->dtype == RAD_F32) R4D_LN(true, unsigned short);
    R4D_LN(false, unsigned short);
#undef R4D_LN
}

/* ------------------------------------------------------------------ grid_embed */
enum { GE_X = 0, GE_TAB, GE_CRD };

extern "C" int r4d_rad_grid_embed(const RadArgs* a, RadStream stream) {
    const RadTensor* x   = r4d_opt(a, GE_X);
    const RadTensor* tab = r4d_opt(a, GE_TAB);
    const RadTensor* crd = r4d_opt(a, GE_CRD);
    if (!x || !tab || !crd) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || crd->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (tab->dtype != RAD_BF16 && tab->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(tab)) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(tab, (int)tab->rank - 1));
    const long long side = r4d_geti_or(a, "side", 0);
    if (n <= 0 || side <= 0 || r4d_numel(tab) < side * side * n) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    /* Four planes of coordinates, dense: [4, planes] with planes >= M. */
    if (crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < M || !r4d_contig(crd))
        return R4D_NO(RAD_E_SHAPE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const uint32_t plane = (uint32_t)crd->shape[1];
    if (tab->dtype == RAD_F32)
        return R4D_EMIT(stream, grid_embed_kernel<true>, dim3((uint32_t)M), dim3(kThr), 0,
                        (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                        (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
    return R4D_EMIT(stream, grid_embed_kernel<false>, dim3((uint32_t)M), dim3(kThr), 0,
                    (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                    (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
}

/* grid_embed_aa: grid_embed's operands and checks, the antialiased resample. */
extern "C" int r4d_rad_grid_embed_aa(const RadArgs* a, RadStream stream) {
    const RadTensor* x   = r4d_opt(a, GE_X);
    const RadTensor* tab = r4d_opt(a, GE_TAB);
    const RadTensor* crd = r4d_opt(a, GE_CRD);
    if (!x || !tab || !crd) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || crd->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (tab->dtype != RAD_BF16 && tab->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(tab)) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(tab, (int)tab->rank - 1));
    const long long side = r4d_geti_or(a, "side", 0);
    if (n <= 0 || side <= 0 || side > 1024 || r4d_numel(tab) < side * side * n)
        return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < M || !r4d_contig(crd))
        return R4D_NO(RAD_E_SHAPE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const uint32_t plane = (uint32_t)crd->shape[1];
    const size_t lds = (size_t)(2 * side) * sizeof(float);
    if (tab->dtype == RAD_F32)
        return R4D_EMIT(stream, grid_embed_aa_kernel<true>, dim3((uint32_t)M), dim3(kThr), lds,
                        (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                        (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
    return R4D_EMIT(stream, grid_embed_aa_kernel<false>, dim3((uint32_t)M), dim3(kThr), lds,
                    (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                    (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
}

/* grid_embed_cpe: grid_embed's operands and checks, RADIO's cropped resample. */
extern "C" int r4d_rad_grid_embed_cpe(const RadArgs* a, RadStream stream) {
    const RadTensor* x   = r4d_opt(a, GE_X);
    const RadTensor* tab = r4d_opt(a, GE_TAB);
    const RadTensor* crd = r4d_opt(a, GE_CRD);
    if (!x || !tab || !crd) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || crd->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (tab->dtype != RAD_BF16 && tab->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(tab)) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(tab, (int)tab->rank - 1));
    const long long side = r4d_geti_or(a, "side", 0);
    if (n <= 0 || side <= 0 || r4d_numel(tab) < side * side * n) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < M || !r4d_contig(crd))
        return R4D_NO(RAD_E_SHAPE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const uint32_t plane = (uint32_t)crd->shape[1];
    /* The prefix rows: bf16, contiguous, n_prefix rows of n. */
    const long long npre = r4d_geti_or(a, "n_prefix", 0);
    const RadTensor* pre = npre > 0 ? r4d_opt(a, 3) : nullptr;
    if (npre > 0 && (!pre || pre->dtype != RAD_BF16 || !r4d_contig(pre) ||
                     r4d_numel(pre) < npre * n))
        return R4D_NO(pre && pre->dtype != RAD_BF16 ? RAD_E_DTYPE : RAD_E_SHAPE);
    const unsigned short* pp = pre ? (const unsigned short*)r4d_p(pre) : nullptr;
    if (tab->dtype == RAD_F32)
        return R4D_EMIT(stream, grid_embed_cpe_kernel<true>, dim3((uint32_t)M), dim3(kThr), 0,
                        (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                        (uint32_t)n, (int)side, (uint32_t)x_ld, plane, pp, (int)npre);
    return R4D_EMIT(stream, grid_embed_cpe_kernel<false>, dim3((uint32_t)M), dim3(kThr), 0,
                    (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                    (uint32_t)n, (int)side, (uint32_t)x_ld, plane, pp, (int)npre);
}

/* ------------------------------------------------------------------ gemm_nt_bias */
enum { GB_A = 0, GB_W, GB_BIAS, GB_RES, GB_Y };

extern "C" int r4d_rad_gemm_nt_bias(const RadArgs* a, RadStream stream) {
    const RadTensor* A = r4d_opt(a, GB_A);
    const RadTensor* W = r4d_opt(a, GB_W);
    const RadTensor* B = r4d_opt(a, GB_BIAS);
    const RadTensor* R = r4d_opt(a, GB_RES);
    const RadTensor* Y = r4d_opt(a, GB_Y);
    if (!A || !W || !B || !Y) return R4D_NO(RAD_E_INVAL);
    if (A->dtype != RAD_BF16 || W->dtype != RAD_BF16 || Y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_BF16 && B->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (R && R->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(W) || !r4d_contig(B)) return R4D_NO(RAD_E_STRIDE);

    const long long K = r4d_geti_or(a, "K", r4d_dim(W, (int)W->rank - 1));
    const long long N = r4d_geti_or(a, "N", r4d_numel(W) / (K > 0 ? K : 1));
    if (K <= 0 || N <= 0 || (K % 8) || r4d_numel(W) < N * K || r4d_numel(B) < N)
        return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0, Mr = 0;
    const long long ldc = r4d_rows_pitch(Y, N, &M);
    const long long lda = r4d_rows_pitch(A, K, &Ma);
    const long long ldr = R ? r4d_rows_pitch(R, N, &Mr) : 0;
    if (ldc < 0 || lda < 0 || ldr < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M || (R && Mr < M)) return R4D_NO(RAD_E_SHAPE);
    /* 16-byte staged chunks: the operand rows must start on 16-byte boundaries. */
    if ((lda % 8) || !r4d_aligned(r4d_p(A), 16) || !r4d_aligned(r4d_p(W), 16))
        return R4D_NO(RAD_E_STRIDE);
    if ((long long)M * lda >= (1ll << 32) || (long long)N * K >= (1ll << 32))
        return R4D_NO(RAD_E_SHAPE);

    const char* act_s = r4d_gets(a, "act");
    int act = 0;
    if (act_s && *act_s) {
        if (!std::strcmp(act_s, "gelu")) act = 1;
        else if (!std::strcmp(act_s, "gelu_tanh")) act = 2;
        else if (std::strcmp(act_s, "none")) return R4D_NO(RAD_E_SCHEMA);
    }
    const bool res = R != nullptr, bf32 = B->dtype == RAD_F32;

    /* The 128 x 128 kernel once its grid reaches 32 workgroups, as gemm_bf16_nt_tiled chooses;
     * below that the 64 x 64 tile, whose four times the blocks fill more of the card. */
    const long long g128 = ((N + 127) / 128) * ((M + 127) / 128);
    const bool big = g128 >= 32;
    auto Ap = (const unsigned short*)r4d_p(A);
    auto Wp = (const unsigned short*)r4d_p(W);
    auto Rp = (const unsigned short*)r4d_p(R);
    auto Cp = (unsigned short*)r4d_wp(Y);
    const void* Bp = r4d_p(B);
    if (big) {
        /* Sixteen-byte stores, residual reads and bias reads at every eight-column boundary. */
        const unsigned vec = (ldc % 8) == 0 && r4d_aligned(Cp, 16) && r4d_aligned(Bp, 16) &&
                             (!res || ((ldr % 8) == 0 && r4d_aligned(Rp, 16)));
        const dim3 grid((unsigned)((M + kVBM - 1) / kVBM), (unsigned)((N + kVBN - 1) / kVBN));
#define VB_L(AC, RS, BF)                                                                          \
        return R4D_EMIT(stream, (vit_gemm_wa_kernel<AC, RS, BF>), grid, dim3(kVThreads), 0, Ap,  \
                        (unsigned)lda, Wp, (unsigned)K, Bp, Rp, (unsigned)ldr, Cp, (unsigned)ldc,  \
                        (int)M, (int)K, (int)N, vec)
#define VB_ACT(RS, BF)                                                                            \
        do {                                                                                      \
            if (act == 1) VB_L(1, RS, BF);                                                        \
            if (act == 2) VB_L(2, RS, BF);                                                        \
            VB_L(0, RS, BF);                                                                      \
        } while (0)
#define VB_RES(BF)                                                                                \
        do { if (res) VB_ACT(true, BF); VB_ACT(false, BF); } while (0)
        if (bf32) VB_RES(true);
        VB_RES(false);
#undef VB_RES
#undef VB_ACT
#undef VB_L
    }
    const dim3 grid((unsigned)((N + 63) / 64), (unsigned)((M + 63) / 64)), block(256);
#define VB_L(AC, RS, BF)                                                                          \
    return R4D_EMIT(stream, (vit_gemm_kernel<2, 1, AC, RS, BF>), grid, block, 0, Ap,             \
                    (unsigned)lda, Wp, (unsigned)K, Bp, Rp, (unsigned)ldr, Cp, (unsigned)ldc,      \
                    (int)M, (int)K, (int)N)
#define VB_ACT(RS, BF)                                                                            \
    do {                                                                                          \
        if (act == 1) VB_L(1, RS, BF);                                                            \
        if (act == 2) VB_L(2, RS, BF);                                                            \
        VB_L(0, RS, BF);                                                                          \
    } while (0)
#define VB_RES(BF)                                                                                \
    do { if (res) VB_ACT(true, BF); VB_ACT(false, BF); } while (0)
    if (bf32) VB_RES(true);
    VB_RES(false);
#undef VB_RES
#undef VB_ACT
#undef VB_L
}
