// r4d_rmsnorm_quant_i8.hip - r4d_rmsnorm_quant_i8: the RMS norm and the per-row int8 quantiser
// r4d_gemm_w4a8_nt_m64 reads, in one pass over the row -- optionally with the residual junction
// in front of it: a post-norm on the branch output, the residual add and a scalar on the sum.
//
// A w4a8 linear needs its activation as int8 codes in the GEMM's fragment order with one f32 scale
// per row (r4d_quant_act_i8), produced by a norm. Unfused that is two dispatches, and a
// sandwich-norm model (Gemma 2/3/4) issues four or five at every residual junction:
//
//     rmsnorm(branch; w_post) -> add(residual) -> [scale] -> rmsnorm(residual; w) -> quant_act_i8
//
// each a single workgroup per row at decode, so the step pays for dispatches, not bytes. This is
// the chain as one dispatch.
//
// Byte-identical to the chain, because the engine's greedy output is compared byte for byte across
// builds:
//
//   * every intermediate the chain stores in bf16 is rounded to bf16 here and the next stage reads
//     the rounded value: the post-norm output, the residual sum, the scaled residual and the norm
//     output the quantiser reduces over;
//   * both sums of squares use the standalone norm's partition (kR4dSumLanes, r4d_common.h): lane
//     v < 256 folds columns v, v+256, ... in order, the lanes past it add nothing, then r4d_wave_sum
//     and the waves in order through LDS (copied from r4d_model_bf16.hip's rmsnorm_kernel and
//     block_sum, which live in another unit). At 256 threads a thread's own columns are that
//     partition; a wider block stages the row in LDS and its first 256 lanes fold it there;
//   * each element's arithmetic matches the standalone kernels (`x * s * (g + wadd)`,
//     `a + b`, `x * scalar`, `fmaxf(m, 1e-8f) / 127.f`), under -ffp-contract=off.
//
// The residual is read and written; each thread rewrites only the columns it read, so the second
// pass reads its own writes back.
//
// `rmsnorm_quant_i8g` and its two-rank all-reduce form are the same junction with quant_act_i8g's
// quantiser instead (GRP): row-major codes and an f32 scale per (row, 128 columns), what gemm_nt_q
// reads at i8a8, w4a8i8_asym and w5a8i8. Only the epilogue differs; it reads the same bf16 norm
// output from LDS, so the codes are the norm + quant_act_i8g pair's bytes.
//
// Returns a negative code instead of throwing, for the reason given in r4d_gemm_fp8a8.hip.
#include "r4d_chain.h"
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_ar_seq.h"
#include "r4d_ar_wide.h"
#include "r4d_q8.h"
#include <cstdint>

namespace {

constexpr uint32_t kWave = 32;

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

// The standalone norm's reduction, verbatim (r4d_model_bf16.hip): the wave butterfly, then the
// waves summed in order by every thread from LDS. `lds` must not be reused by a second reduction
// without a barrier in between; the kernel below gives each its own array.
template <uint32_t NW>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    return s;
}

// A row's sum of squares in the standalone norm's partition (kR4dSumLanes): lane v < 256 adds
// columns v, v + 256, ... of `row` in order, the lanes past it add nothing, so a row's bits do not
// depend on the block's width. `row` holds the row's bf16 values; when the block's threads wrote
// it, the caller issues the barrier first.
__device__ __forceinline__ float lanes_sumsq(const uint16_t* row, uint32_t n) {
    return r4d_lanes_sumsq(row, n, [](uint16_t h) { return bf2f(h); });
}

__device__ __forceinline__ float gain(const void* w, uint32_t i, bool f32) {
    return f32 ? ((const float*)w)[i] : bf2f(((const uint16_t*)w)[i]);
}

// quant_act_i8g's group, the width of one scale (r4d_quant_act_fp8.hip's kGroup).
constexpr uint32_t kGroupI8g = 128;

// RES:  a residual is present; the norm is taken of `res + d` (stored back into `res`), where d is
//       the branch `x` itself or, with POST, rmsnorm(x; w_post).  Without RES the norm is of `x`.
// SCL:  the sum is multiplied by the [1] scalar before it is stored and normalised.
// GRP:  quant_act_i8g's codes and scales (r4d_q8.h's q8_row_i8g) instead of quant_act_i8's; `Sr`
//       then holds n / 128 scales.
// `ybuf` holds the row's norm output for the code store, which reads across threads.
//
// The row body from the junction on: `s1` is the post-norm's factor, already reduced over `xr` by
// the caller (the plain kernel takes it off the row; the all-reduce form off the reduced sum it has
// just stored). l2 and lm are this function's own reduction planes.
template <bool WF32, bool RES, bool POST, bool SCL, uint32_t THR, bool GRP = false>
__device__ __forceinline__ void junction_row(
    const uint16_t* __restrict__ xr, uint16_t* __restrict__ rr, const void* __restrict__ w,
    const void* __restrict__ wp, float sv, signed char* __restrict__ qr, float* __restrict__ Sr,
    uint16_t* __restrict__ yr, uint32_t n, float eps, float wadd, float s1,
    uint16_t* ybuf, float* l2, float* lm) {
    // ---- the junction (post-norm, add, scalar), stored, and the norm's sum of squares over it
    float ss2 = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        uint16_t v = xr[i];
        if (RES) {
            if (POST) v = f32_to_bf16(bf2f(v) * s1 * (gain(wp, i, WF32) + wadd));
            v = f32_to_bf16(bf2f(rr[i]) + bf2f(v));
            if (SCL) v = f32_to_bf16(bf2f(v) * sv);
            rr[i] = v;
        }
        if constexpr (THR == kR4dSumLanes) { const float f = bf2f(v); ss2 += f * f; }
        else ybuf[i] = v;
    }
    if constexpr (THR != kR4dSumLanes) { __syncthreads(); ss2 = lanes_sumsq(ybuf, n); }
    ss2 = block_sum<THR / kWave>(ss2, l2);
    const float s2 = rsqrtf(ss2 / (float)n + eps);

    // ---- the norm output, rounded to bf16 as the standalone norm stores it, and its absmax
    const uint16_t* __restrict__ nr = RES ? rr : xr;
    float m = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const uint16_t y = f32_to_bf16(bf2f(nr[i]) * s2 * (gain(w, i, WF32) + wadd));
        ybuf[i] = y;
        if (yr) yr[i] = y;
        m = fmaxf(m, fabsf(bf2f(y)));
    }
    if constexpr (GRP) {
        __syncthreads();   // publishes ybuf to every thread
        q8_row_i8g<THR>(ybuf, qr, Sr, n);
        return;
    }
    m = r4d_wave_max(m);
    if ((threadIdx.x & 31) == 0) lm[threadIdx.x >> 5] = m;
    __syncthreads();   // also publishes ybuf to every thread
    m = lm[0];
#pragma unroll
    for (uint32_t i = 1; i < THR / kWave; i++) m = fmaxf(m, lm[i]);
    const float scale = fmaxf(m, 1e-8f) / 127.f;
    const float inv = 1.f / scale;
    if (threadIdx.x == 0) *Sr = scale;

    // ---- the codes, in r4d_quant_act_i8's fragment order: byte j of a k step comes from
    // lane-half j>>3 and element j&7, i.e. source 8*((j&7)>>2) + 4*(j>>3) + (j&3) inside the step.
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const uint32_t s = i >> 4, j = i & 15;
        const uint32_t src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
        const int t = __float2int_rn(bf2f(ybuf[src]) * inv);
        qr[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
    }
}

__device__ __forceinline__ float scalar_of(const void* sc, uint32_t f32) {
    return f32 ? ((const float*)sc)[0] : bf2f(((const uint16_t*)sc)[0]);
}

// The exact all-reduce's fold of one 16-byte word (eight bf16): every source's word loaded first,
// then per element fp32 from 0.0f, the sources in ascending rank, one rounding through
// bf16_hi_rne_nan -- ar_vsum's arithmetic, so the word is the standalone exact all-reduce's.
template <int WS>
__device__ __forceinline__ int4 fold_word8(const int4* const* src4, size_t k) {
    int4 v[WS];
#pragma unroll
    for (int r = 0; r < WS; ++r) v[r] = src4[r][k];
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < WS; ++r) {
        const uint16_t* pv = reinterpret_cast<const uint16_t*>(&v[r]);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] += bf2f(pv[j]);
    }
    int4 a;
    uint16_t* pa = reinterpret_cast<uint16_t*>(&a);
#pragma unroll
    for (int j = 0; j < 8; ++j) pa[j] = (uint16_t)(bf16_hi_rne_nan(acc[j]) >> 16);
    return a;
}

// A row's scales: one, or n / 128 at quant_act_i8g's grid.
template <bool GRP>
__device__ __forceinline__ uint32_t scales_a_row(uint32_t n) {
    return GRP ? n / kGroupI8g : 1u;
}

template <bool WF32, bool RES, bool POST, bool SCL, uint32_t THR, bool GRP = false>
__global__ __launch_bounds__(THR) void rmsnorm_quant_i8_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    const void* __restrict__ wp, const void* __restrict__ sc, uint32_t sc_f32,
    signed char* __restrict__ q, float* __restrict__ S, uint16_t* __restrict__ yo,
    uint32_t n, float eps, float wadd, uint32_t x_ld, uint32_t r_ld, uint32_t y_ld) {
    extern __shared__ uint16_t ybuf[];
    __shared__ float l1[THR / kWave], l2[THR / kWave], lm[THR / kWave];
    const uint32_t row = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;

    // ---- the post-norm's sum of squares, over the branch output
    float s1 = 0.0f;
    if (POST) {
        float ss = 0.0f;
        if constexpr (THR == kR4dSumLanes)
            for (uint32_t i = threadIdx.x; i < n; i += THR) { const float v = bf2f(xr[i]); ss += v * v; }
        else
            ss = lanes_sumsq(xr, n);   // the branch output, read-only here
        ss = block_sum<THR / kWave>(ss, l1);
        s1 = rsqrtf(ss / (float)n + eps);
    }
    junction_row<WF32, RES, POST, SCL, THR, GRP>(
        xr, RES ? res + (size_t)row * r_ld : nullptr, w, wp, SCL ? scalar_of(sc, sc_f32) : 1.0f,
        q + (size_t)row * n, S + (size_t)row * scales_a_row<GRP>(n),
        yo ? yo + (size_t)row * y_ld : nullptr, n, eps, wadd, s1, ybuf, l2, lm);
}

// The same kernel at decode heights (THR 1024, n <= 1024 * EPT), with every operand of the row
// loaded in one go before the first reduction. The kernel above reads the row again in each pass,
// each read behind a barrier, so a row costs three dependent memory round trips; on gfx1201 at
// 1 to 16 rows of 5376 that is 11 us a launch against 8.2 us for this form. Every elementwise pass
// partitions the row the same way (thread t takes columns t, t+THR, ...), so each thread keeps its
// own columns in registers; the sums and the fragment-order store cross threads through `ybuf`,
// which holds the row each sum folds in the standalone norm's partition. The sums, their order and
// every element's arithmetic are junction_row's, so the output bytes are the same. One row of it, for this kernel and for the all-reduce's tail
// (ar_rmsnorm_quant_i8_tail_kernel). GRP as in junction_row.
template <bool WF32, bool RES, bool POST, bool SCL, uint32_t THR, uint32_t EPT, bool GRP = false>
__device__ __forceinline__ void junction_reg_row(
    const uint16_t* __restrict__ xr, uint16_t* __restrict__ rr, const void* __restrict__ w,
    const void* __restrict__ wp, const void* __restrict__ sc, uint32_t sc_f32,
    signed char* __restrict__ qr, float* __restrict__ Sr, uint16_t* __restrict__ yr, uint32_t n,
    float eps, float wadd, uint16_t* ybuf, float* l1, float* l2, float* lm) {
    const uint32_t tid = threadIdx.x;
    uint16_t xv[EPT], rv[EPT];
    float gp[EPT], gw[EPT];
#pragma unroll
    for (uint32_t e = 0; e < EPT; ++e) {
        const uint32_t i = tid + e * THR;
        xv[e] = 0; rv[e] = 0; gp[e] = 0.0f; gw[e] = 0.0f;
        if (i < n) {
            xv[e] = xr[i];
            if (RES) rv[e] = rr[i];
            if (POST) gp[e] = gain(wp, i, WF32) + wadd;
            gw[e] = gain(w, i, WF32) + wadd;
        }
    }
    const float sv = SCL ? scalar_of(sc, sc_f32) : 1.0f;

    // ---- the post-norm's sum of squares, over the branch output
    float s1 = 0.0f;
    if (POST) {
        float ss = 0.0f;
#pragma unroll
        for (uint32_t e = 0; e < EPT; ++e)
            if (tid + e * THR < n) {
                if constexpr (THR == kR4dSumLanes) { const float v = bf2f(xv[e]); ss += v * v; }
                else ybuf[tid + e * THR] = xv[e];
            }
        if constexpr (THR != kR4dSumLanes) { __syncthreads(); ss = lanes_sumsq(ybuf, n); }
        s1 = rsqrtf(block_sum<THR / kWave>(ss, l1) / (float)n + eps);
    }

    // ---- the junction (post-norm, add, scalar), stored, and the norm's sum of squares over it
    float ss2 = 0.0f;
#pragma unroll
    for (uint32_t e = 0; e < EPT; ++e) {
        const uint32_t i = tid + e * THR;
        if (i < n) {
            uint16_t v = xv[e];
            if (RES) {
                if (POST) v = f32_to_bf16(bf2f(v) * s1 * gp[e]);
                v = f32_to_bf16(bf2f(rv[e]) + bf2f(v));
                if (SCL) v = f32_to_bf16(bf2f(v) * sv);
                rr[i] = v;
            }
            xv[e] = v;
            if constexpr (THR == kR4dSumLanes) { const float f = bf2f(v); ss2 += f * f; }
            else ybuf[i] = v;
        }
    }
    if constexpr (THR != kR4dSumLanes) { __syncthreads(); ss2 = lanes_sumsq(ybuf, n); }
    const float s2 = rsqrtf(block_sum<THR / kWave>(ss2, l2) / (float)n + eps);

    // ---- the norm output, rounded to bf16 as the standalone norm stores it, and its absmax
    float m = 0.0f;
#pragma unroll
    for (uint32_t e = 0; e < EPT; ++e) {
        const uint32_t i = tid + e * THR;
        if (i < n) {
            const uint16_t y = f32_to_bf16(bf2f(xv[e]) * s2 * gw[e]);
            ybuf[i] = y;
            if (yr) yr[i] = y;
            m = fmaxf(m, fabsf(bf2f(y)));
        }
    }
    if constexpr (GRP) {
        __syncthreads();   // publishes ybuf to every thread
        q8_row_i8g<THR>(ybuf, qr, Sr, n);
        return;
    }
    m = r4d_wave_max(m);
    if ((tid & 31) == 0) lm[tid >> 5] = m;
    __syncthreads();   // also publishes ybuf to every thread
    m = lm[0];
#pragma unroll
    for (uint32_t i = 1; i < THR / kWave; i++) m = fmaxf(m, lm[i]);
    const float scale = fmaxf(m, 1e-8f) / 127.f;
    const float inv = 1.f / scale;
    if (tid == 0) *Sr = scale;

    // ---- the codes, in junction_row's fragment order
    for (uint32_t i = tid; i < n; i += THR) {
        const uint32_t s = i >> 4, j = i & 15;
        const uint32_t src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
        const int t = __float2int_rn(bf2f(ybuf[src]) * inv);
        qr[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
    }
}

template <bool WF32, bool RES, bool POST, bool SCL, uint32_t THR, uint32_t EPT, bool GRP = false>
__global__ __launch_bounds__(THR) void rmsnorm_quant_i8_reg_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    const void* __restrict__ wp, const void* __restrict__ sc, uint32_t sc_f32,
    signed char* __restrict__ q, float* __restrict__ S, uint16_t* __restrict__ yo,
    uint32_t n, float eps, float wadd, uint32_t x_ld, uint32_t r_ld, uint32_t y_ld) {
    extern __shared__ uint16_t ybuf[];
    __shared__ float l1[THR / kWave], l2[THR / kWave], lm[THR / kWave];
    const uint32_t row = blockIdx.x;
    junction_reg_row<WF32, RES, POST, SCL, THR, EPT, GRP>(
        x + (size_t)row * x_ld, RES ? res + (size_t)row * r_ld : nullptr, w, wp, sc, sc_f32,
        q + (size_t)row * n, S + (size_t)row * scales_a_row<GRP>(n),
        yo ? yo + (size_t)row * y_ld : nullptr, n, eps, wadd, ybuf, l1, l2, lm);
}

// ---------------------------------------------------------------------------------------------
// The same junction with the two-rank all-reduce that precedes it: the branch output is a rank's
// partial sum of a column-sharded linear (o_proj, down_proj), and the post-norm is of the whole
// sum. ar_rmsnorm_quant_fp8's handshake, exact wire only: each block pushes its rows into the
// peer's scratch slot as 16B words, publishes its flag, waits for the peer's, and reduces
// `f2bf(mine + peer)` back into `x` (the standalone exact all-reduce's bytes; the fp32 add
// commutes, so both ranks agree) during the post-norm's sum-of-squares pass. A block owns whole
// rows, one per block at decode.
#define R4D_ARQ8_SPIN_MAX 4000000000ULL
/* The all-reduce's tail form serves up to this many rows: a block a row beside the transport's,
 * far under one block a CU on the cards it runs on. */
#define R4D_ARQ8_TAIL_MAX_ROWS 32

__device__ __forceinline__ void arq8_store_rel(unsigned int* p, unsigned int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int arq8_load_acq(const unsigned int* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// CH `r4dchain`: the chained form (r4d_chain.h); no block returns early.
template <bool WF32, bool RES, bool POST, bool SCL, uint32_t THR, bool GRP = false,
          class CH = R4dUnchained>
__global__ __launch_bounds__(THR) void ar_rmsnorm_quant_i8_kernel(
    uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    const void* __restrict__ wp, const void* __restrict__ sc, uint32_t sc_f32,
    signed char* __restrict__ q, float* __restrict__ S, uint16_t* __restrict__ yo,
    int4* peer_base, const int4* my_base, unsigned int* peer_flags, unsigned int* my_flags,
    unsigned int* seq_ctrs, int M, uint32_t n, float eps, float wadd, uint32_t r_ld, uint32_t y_ld,
    int slot_stride16, int drain, int acq, unsigned int nbw, long stage) {
    if (CH::on) r4d_chain_wait();
    extern __shared__ uint16_t ybuf[];
    __shared__ float l1[THR / kWave], l2[THR / kWave], lm[THR / kWave];
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, nb = (int)gridDim.x;
    const uint32_t tid = threadIdx.x;
    const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);

    r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   /* device-resident, replay-safe */
    const unsigned int sq = s_seq;
    int4* const ps = peer_base + (size_t)s_slot * slot_stride16;
    const int4* ms = my_base + (size_t)s_slot * slot_stride16;
    const long md = r4d_archk_md(stage, my_base);   // the host mailbox's VRAM mirror (r4d_ar_seq.h)

    /* push my rows into the peer's slot; `x` is a tight [M][n] of bf16 */
    const int wpr = (int)(n >> 3);
    const int4* const in4 = (const int4*)x;
    unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)
    for (int k = r0 * wpr + (int)tid; k < r1 * wpr; k += (int)THR) {
        const int4 v = in4[k];
        ps[k] = v;
        hck += r4d_archk_u4(v, 4u * (unsigned)k);
    }
    r4d_archk_put(hck, peer_flags, nbw, s_slot, (unsigned)b, sq);
    if (drain == 1) __threadfence_system();
    else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (drain == 2) __threadfence_system();
        arq8_store_rel(&peer_flags[b], sq);
        unsigned long long z = 0;
        while (r4d_ar_seq_behind(arq8_load_acq(&my_flags[b]), sq)) {
            if (++z > R4D_ARQ8_SPIN_MAX) __builtin_trap();
        }
    }
    __syncthreads();
    if (acq) __threadfence_system();
    r4d_archk_wait(my_flags, nbw, s_slot, (unsigned)b, sq, [&](unsigned int& h) {
        for (int k = r0 * wpr + (int)tid; k < r1 * wpr; k += (int)THR)
            h += r4d_archk_u4(r4d_archk_keep(&ms[k], md), 4u * (unsigned)k);
    });
    ms = r4d_archk_moved(ms, md);

    const uint16_t* __restrict__ P = (const uint16_t*)ms;   /* the peer's rows, in my scratch */
    const float sv = SCL ? scalar_of(sc, sc_f32) : 1.0f;
    for (int row = r0; row < r1; ++row) {
        uint16_t* __restrict__ xr = x + (size_t)row * n;
        const uint16_t* __restrict__ pr = P + (size_t)row * n;
        float ss = 0.0f;
        for (uint32_t i = tid; i < n; i += THR) {
            const uint16_t a = f32_to_bf16(bf2f(xr[i]) + bf2f(pr[i]));
            xr[i] = a;
            if (POST) {
                if constexpr (THR == kR4dSumLanes) { const float v = bf2f(a); ss += v * v; }
                else ybuf[i] = a;
            }
        }
        if constexpr (THR != kR4dSumLanes)
            if (POST) { __syncthreads(); ss = lanes_sumsq(ybuf, n); }
        float s1 = 0.0f;
        if (POST) s1 = rsqrtf(block_sum<THR / kWave>(ss, l1) / (float)n + eps);
        junction_row<WF32, RES, POST, SCL, THR, GRP>(
            xr, RES ? res + (size_t)row * r_ld : nullptr, w, wp, sv, q + (size_t)row * n,
            S + (size_t)row * scales_a_row<GRP>(n), yo ? yo + (size_t)row * y_ld : nullptr, n, eps,
            wadd, s1, ybuf, l2, lm);
        __syncthreads();   /* the planes and ybuf are re-used by the next row this block owns */
    }
    if (CH::on) r4d_chain_done();
}

// ---------------------------------------------------------------------------------------------
// The same junction with the four-rank all-reduce in front: the wide family's exact one-shot
// (r4d_ar_oneshot_Nrank_exact.hip) with the junction as its epilogue, as ar_hc_write_4rank is that
// one-shot with hc_write. Up to the reduce it is that one-shot's transport step for step (the same
// seq take over R4D_AR_WIDE_MAX_BLOCKS counters, the same handshake value 2*s, the same (parity,
// source) slots), so it shares an instance with the standalone all_reduce on one stream
// (r4d_ar_wide.h). Two things differ, both for the junction:
//
//   * A block owns whole rows, the two-rank fused form's partition, since a post-norm's sum of
//     squares closes over one row. Every rank runs the same grid over the same M, so block b's
//     words are the same words on every rank and the per-block handshake pairs them.
//   * The reduce is element by element, stored back into `x`, and the post-norm's sum of squares
//     folds the stored row in the standalone norm's partition (lanes_sumsq). Each element is ar_vsum's
//     arithmetic exactly: fp32 from 0.0f, the ranks in ascending order, one rounding to bf16 as the
//     __hip_bfloat16 cast rounds (bf16_hi_rne_nan). So the reduced row is the standalone exact
//     all-reduce's to the bit, and what follows is rmsnorm_quant_i8's junction over it.
//
// In place is safe for the reason the hc_write form gives: a thread reads every source of an element
// before it stores it, its block pushed that element to every peer before the handshake, and no peer
// reads this rank's `x`.
template <bool WF32, bool RES, bool POST, bool SCL, uint32_t THR, int WS>
__global__ __launch_bounds__(THR) void ar_rmsnorm_quant_i8_wide_kernel(
    uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    const void* __restrict__ wp, const void* __restrict__ sc, uint32_t sc_f32,
    signed char* __restrict__ q, float* __restrict__ S, uint16_t* __restrict__ yo,
    R4DArWidePeers<int4, WS - 1> peers, const int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs, int M, uint32_t n,
    float eps, float wadd, uint32_t r_ld, uint32_t y_ld, int slot_stride16, int rank, int drain,
    int acq, int pub, long long* gate, unsigned int* gate_ctr) {
    constexpr int NPEERS = WS - 1;
    extern __shared__ uint16_t ybuf[];
    __shared__ float l1[THR / kWave], l2[THR / kWave], lm[THR / kWave];
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, nb = (int)gridDim.x;
    const uint32_t tid = threadIdx.x;
    const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);
    const long long gate_word = r4d_ar_link_word(gate, tid);

    r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);   /* device-resident, replay-safe */
    const unsigned int s = s_seq;
    const int parity = (int)s_slot;

    /* push my rows into every peer's (parity, my rank) slot; `x` is a tight [M][n] of bf16 */
    int4* ps[NPEERS];
#pragma unroll
    for (int p = 0; p < NPEERS; ++p)
        ps[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;
    const int wpr = (int)(n >> 3);
    const int4* const in4 = (const int4*)x;
    for (int k = r0 * wpr + (int)tid; k < r1 * wpr; k += (int)THR) {
        const int4 v = in4[k];
#pragma unroll
        for (int p = 0; p < NPEERS; ++p) ps[p][k] = v;
    }

    r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
    /* The one-shot's release: the fold and the junction behind it are local VRAM traffic, so the
     * last block past its handshake ends the engine's hold on the link (r4d_ar_seq.h). Every block
     * owns at least one row and takes exactly one handshake, so every path reaches it. */
    r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, tid);

    /* the sources in ascending global rank, this rank's own from `x` (its own slot is never
     * written), the peers' from their slots in my scratch */
    const int4* src4[WS];
#pragma unroll
    for (int r = 0; r < WS; ++r)
        src4[r] = r == rank ? (const int4*)x
                            : my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16;
    const float sv = SCL ? scalar_of(sc, sc_f32) : 1.0f;
    for (int row = r0; row < r1; ++row) {
        const size_t o = (size_t)row * n;
        uint16_t* __restrict__ xr = x + o;
        /* The fold a 16-byte word a thread, every source's word in flight at once (the peers' are
         * in fine-grained memory, a round trip each), stored back into `x`; then the post-norm's
         * sum of squares over the stored row in the standalone norm's partition. Per element it is
         * the fold fold_word8 documents, and the sum reads the rounded values the standalone
         * all-reduce stores, so the bytes are those of the all-reduce and the norm taken apart. */
        for (int k = (int)tid; k < wpr; k += (int)THR)
            ((int4*)xr)[k] = fold_word8<WS>(src4, (size_t)row * wpr + k);
        float ss = 0.0f;
        if (POST) {
            __syncthreads();   /* the folded row, read across the block's threads */
            ss = lanes_sumsq(xr, n);
        }
        float s1 = 0.0f;
        if (POST) s1 = rsqrtf(block_sum<THR / kWave>(ss, l1) / (float)n + eps);
        else __syncthreads();   /* junction_row reads the folded row across threads too */
        junction_row<WF32, RES, POST, SCL, THR>(
            xr, RES ? res + (size_t)row * r_ld : nullptr, w, wp, sv, q + (size_t)row * n, S + row,
            yo ? yo + (size_t)row * y_ld : nullptr, n, eps, wadd, s1, ybuf, l2, lm);
        __syncthreads();   /* the planes and ybuf are re-used by the next row this block owns */
    }
}

// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
// The junction as the tail of the standalone exact two-shot, in one launch. Blocks [0, nbt) run
// r4d_ar_twoshot_wide_exact_kernel's body over x in place (r4d_ar_twoshot_Nrank_exact.hip): the same
// shards, words, fold (fp32 from 0.0f in ascending rank, one rounding) and handshakes, so x ends as
// the standalone all-reduce leaves it; each then counts itself in on a word of the instance's
// counter allocation, at agent scope behind a release. Blocks [nbt, nbt + M) are the junction, one
// row each as rmsnorm_quant_i8_reg_kernel runs it (junction_reg_row, the same width and partition):
// each waits for every transport block's count, acquires, and runs its row; the last of them zeroes
// both counts for the next launch. So the junction takes neither a cross-card handshake nor a
// launch of its own: the pair's bytes, one dependent launch fewer. Blocks are dispatched in index
// order, so the transport blocks are resident before any junction block spins; the grid is kept far
// under one block a CU.
//
// tail[0] counts the transport blocks done, tail[1] the junction blocks done; both start at zero
// (the counter allocation is zeroed at init) and the last junction block returns them there.
// The body: transport blocks return after their part, so the chained form (r4d_chain.h) wraps
// it below. The last junction block resets the tail counters inside the body, before the chained
// form's add, so the next launch on the instance never sees them mid-reset.
template <bool WF32, bool RES, bool POST, bool SCL, uint32_t EPT, int WS>
__device__ __forceinline__ void ar_rmsnorm_quant_i8_tail_body(
    uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    const void* __restrict__ wp, const void* __restrict__ sc, uint32_t sc_f32,
    signed char* __restrict__ q, float* __restrict__ S, uint16_t* __restrict__ yo,
    R4DArWidePeers<int4, WS - 1> peers, const int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs,
    unsigned int* __restrict__ tail, int M, uint32_t n, float eps, float wadd, uint32_t r_ld,
    uint32_t y_ld, int slot_stride16, int rank, int nbt, int drain, int acq, int pub,
    long long* gate, unsigned int* gate_ctr) {
    constexpr uint32_t THR = 1024;
    constexpr int NPEERS = WS - 1;
    extern __shared__ uint16_t ybuf[];
    __shared__ float l1[THR / kWave], l2[THR / kWave], lm[THR / kWave];
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, tid = (int)threadIdx.x, nt = (int)THR;

    if (b < nbt) {
        // ---- the standalone exact two-shot, x in place, block b of nbt
        const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
        r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
        const unsigned int s = s_seq;
        const int parity = (int)s_slot;
        int4* const x4 = (int4*)x;
        const int n16 = (int)(((long)M * n) >> 3);
        const int base = n16 / WS, rem = n16 % WS;
        const int sh_max = base + (rem ? 1 : 0);
#define R4D_TT_OFF(o) ((o) * base + ((o) < rem ? (o) : rem))
#define R4D_TT_LEN(o) (base + ((o) < rem ? 1 : 0))
        const int wd = (sh_max + nbt - 1) / nbt;
        const int w0 = b * wd;
        int w1 = w0 + wd; if (w1 > sh_max) w1 = sh_max;
        int4* pa[NPEERS];
#pragma unroll
        for (int p = 0; p < NPEERS; ++p)
            pa[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;
        for (int p = 0; p < NPEERS; ++p) {
            const int owner = peers.rank[p];
            const int4* src = x4 + (size_t)R4D_TT_OFF(owner);
            int4* dst = pa[p];
            const int lim = w1 < R4D_TT_LEN(owner) ? w1 : R4D_TT_LEN(owner);
            for (int k = w0 + tid; k < lim; k += nt) dst[k] = src[k];
        }
        r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s - 1u, drain, acq, pub);
        const int4* src[WS];
#pragma unroll
        for (int r = 0; r < WS; ++r)
            src[r] = (r == rank) ? (x4 + (size_t)R4D_TT_OFF(rank))
                                 : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);
        int4* my_out = x4 + (size_t)R4D_TT_OFF(rank);
        const int my_lim = w1 < R4D_TT_LEN(rank) ? w1 : R4D_TT_LEN(rank);
        for (int k = w0 + tid; k < my_lim; k += nt) {
            float acc[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
#pragma unroll
            for (int r = 0; r < WS; ++r) {
                const int4 v = src[r][k];
                const __hip_bfloat16* pv = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[j] += r4d_ar_to_f(pv[j]);
            }
            int4 o;
            __hip_bfloat16* po = reinterpret_cast<__hip_bfloat16*>(&o);
#pragma unroll
            for (int j = 0; j < 8; ++j) r4d_ar_from_f(acc[j], po[j]);
            my_out[k] = o;
#pragma unroll
            for (int p = 0; p < NPEERS; ++p) pa[p][sh_max + k] = o;
        }
        r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
        r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nbt, (unsigned int)tid);
        for (int p = 0; p < NPEERS; ++p) {
            const int owner = peers.rank[p];
            const int4* srcB = my_base + ((size_t)parity * WS + owner) * (size_t)slot_stride16 + sh_max;
            int4* dst = x4 + (size_t)R4D_TT_OFF(owner);
            const int lim = w1 < R4D_TT_LEN(owner) ? w1 : R4D_TT_LEN(owner);
            for (int k = w0 + tid; k < lim; k += nt) dst[k] = srcB[k];
        }
#undef R4D_TT_OFF
#undef R4D_TT_LEN
        // ---- every word this block stored in x, published, then counted in
        __syncthreads();
        if (tid == 0) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __hip_atomic_fetch_add(&tail[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        return;
    }

    // ---- the junction, row b - nbt, once x is whole on this card
    const int row = b - nbt;
    if (tid == 0) {
        unsigned long long z = 0;
        while (__hip_atomic_load(&tail[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
               (unsigned int)nbt) {
            __builtin_amdgcn_s_sleep(1);
            if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
        }
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    junction_reg_row<WF32, RES, POST, SCL, THR, EPT>(
        x + (size_t)row * n, RES ? res + (size_t)row * r_ld : nullptr, w, wp, sc, sc_f32,
        q + (size_t)row * n, S + row, yo ? yo + (size_t)row * y_ld : nullptr, n, eps, wadd, ybuf,
        l1, l2, lm);
    __syncthreads();
    if (tid == 0) {
        const unsigned int prev =
            __hip_atomic_fetch_add(&tail[1], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (prev + 1u == (unsigned int)M) {
            __hip_atomic_store(&tail[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&tail[1], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

#define R4D_ARQ8T_PARAMS uint16_t* __restrict__ x, uint16_t* __restrict__ res,                    \
    const void* __restrict__ w, const void* __restrict__ wp, const void* __restrict__ sc,          \
    uint32_t sc_f32, signed char* __restrict__ q, float* __restrict__ S,                           \
    uint16_t* __restrict__ yo, R4DArWidePeers<int4, WS - 1> peers,                                 \
    const int4* __restrict__ my_base, unsigned int* __restrict__ my_flags,                         \
    unsigned int* __restrict__ seq_ctrs, unsigned int* __restrict__ tail, int M, uint32_t n,       \
    float eps, float wadd, uint32_t r_ld, uint32_t y_ld, int slot_stride16, int rank, int nbt,     \
    int drain, int acq, int pub, long long* gate, unsigned int* gate_ctr
#define R4D_ARQ8T_ARGS x, res, w, wp, sc, sc_f32, q, S, yo, peers, my_base, my_flags, seq_ctrs, tail, \
    M, n, eps, wadd, r_ld, y_ld, slot_stride16, rank, nbt, drain, acq, pub, gate, gate_ctr
// The plain kernel, and the chained one (r4dchain*): its wait comes before anything, the sequence
// take included, so a block of the next launch on the instance never starts ahead of this one's.
template <bool WF32, bool RES, bool POST, bool SCL, uint32_t EPT, int WS>
__global__ __launch_bounds__(1024) void ar_rmsnorm_quant_i8_tail_kernel(R4D_ARQ8T_PARAMS) {
    ar_rmsnorm_quant_i8_tail_body<WF32, RES, POST, SCL, EPT, WS>(R4D_ARQ8T_ARGS);
}
template <bool WF32, bool RES, bool POST, bool SCL, uint32_t EPT, int WS>
__global__ __launch_bounds__(1024) void r4dchain_ar_rmsnorm_quant_i8_tail(R4D_ARQ8T_PARAMS) {
    r4d_chain_wait();
    ar_rmsnorm_quant_i8_tail_body<WF32, RES, POST, SCL, EPT, WS>(R4D_ARQ8T_ARGS);
    r4d_chain_done();
}
#undef R4D_ARQ8T_ARGS
#undef R4D_ARQ8T_PARAMS

// ---------------------------------------------------------------------------------------------
// The junction with the all-reduce split around it (`sp`): reduce-scatter, the junction on the rows
// this rank owns, all-gather of what the next GEMM reads. Rank r owns rows [r0(r), r0(r) + nr(r)),
// M split as evenly as the rank order allows, and block b serves local row b of every rank:
//
//   1. it pushes this rank's partial sums of each peer's local row b into that peer's slot (bf16,
//      the two-shot's first hop) and takes the handshake;
//   2. on its own row it folds the four partials in rank order -- r4d_ar_to_f, f32, rank 0 first,
//      r4d_ar_from_f, the standalone exact all-reduce's fold, so the same bf16 sum -- and runs the
//      junction on it in the register form, each sum folded in the standalone norm's partition
//      (lanes_sumsq), so every sum is the standalone's; it
//      stores the residual row and pushes the row's int8 codes and scale (and with `gres` the
//      residual row) into every peer's slot past the first hop's region, then takes the handshake;
//   3. it copies the peers' local row b of codes and scales (and residual) into place.
//
// So every rank leaves q and S on every row, byte for byte the chain's (all_reduce, then
// rmsnorm_quant_i8), and the residual on its own rows only -- on every row with `gres`. `x` keeps
// this rank's partial sums: nothing past the junction reads the reduced branch output. Against the
// tail form the second hop carries int8 instead of bf16 and no block waits beside the transport;
// one launch at a time on 4x R9700 it takes 22.9 against 24.5 us at 8 rows of 5376,
// 30.8 against 34.9 at 16, 72.3 against 89.2 at 64, and 19.8 against the one-shot's 21.9 at 4.
template <uint32_t EPT, uint32_t THR, int WS>
__device__ __forceinline__ void ar_rmsnorm_quant_i8_sp_body(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    const void* __restrict__ wp, const void* __restrict__ sc, uint32_t sc_f32, uint32_t wf32,
    uint32_t gres, signed char* __restrict__ q, float* __restrict__ S,
    R4DArWidePeers<int4, WS - 1> peers, const int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs, int M, uint32_t n,
    float eps, float wadd, uint32_t r_ld, int slot_stride16, int rank, int drain, int acq, int pub,
    long long* gate, unsigned int* gate_ctr) {
    constexpr int NPEERS = WS - 1;
    extern __shared__ uint16_t ybuf[];
    signed char* const cbuf = (signed char*)(ybuf + n);
    __shared__ float l1[THR / kWave], l2[THR / kWave], lm[THR / kWave];
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const long long gate_word = r4d_ar_link_word(gate, tid);
    r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
    const unsigned int s = s_seq;
    const int parity = (int)s_slot;
    const int base = M / WS, rem = M % WS;
#define R4D_SP_R0(r) ((r) * base + ((r) < rem ? (r) : rem))
#define R4D_SP_NR(r) (base + ((r) < rem ? 1 : 0))
    const int n16 = (int)(n >> 3), c16 = (int)(n >> 4);
    const size_t offB = (size_t)(base + (rem ? 1 : 0)) * n16;   /* past the first hop's rows */
    const size_t rowB = (size_t)c16 + 1 + (gres ? n16 : 0);    /* codes, a scale word, residual */
    const size_t my_slot = ((size_t)parity * WS + rank) * (size_t)slot_stride16;

    // ---- the first hop: this rank's partials of each peer's local row b
    for (int p = 0; p < NPEERS; ++p) {
        const int d = peers.rank[p];
        if (b >= R4D_SP_NR(d)) continue;
        const int4* src = (const int4*)(x + (size_t)(R4D_SP_R0(d) + b) * n);
        int4* dst = peers.scratch[p] + my_slot + (size_t)b * n16;
        for (int k = (int)tid; k < n16; k += (int)THR) dst[k] = src[k];
    }
    r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s - 1u, drain, acq, pub);

    if (b < R4D_SP_NR(rank)) {
        const int row = R4D_SP_R0(rank) + b;
        const __hip_bfloat16* srcs[WS];
#pragma unroll
        for (int r = 0; r < WS; ++r)
            srcs[r] = r == rank ? (const __hip_bfloat16*)(x + (size_t)row * n)
                                : (const __hip_bfloat16*)(my_base + ((size_t)parity * WS + r) *
                                                          (size_t)slot_stride16 + (size_t)b * n16);
        uint16_t* const rr = res + (size_t)row * r_ld;
        uint16_t xv[EPT], rv[EPT];
        float gp[EPT], gw[EPT];
#pragma unroll
        for (uint32_t e = 0; e < EPT; ++e) {
            const uint32_t i = tid + e * THR;
            xv[e] = 0; rv[e] = 0; gp[e] = 0.0f; gw[e] = 0.0f;
            if (i < n) {
                float acc = 0.0f;
#pragma unroll
                for (int r = 0; r < WS; ++r) acc += r4d_ar_to_f(srcs[r][i]);
                __hip_bfloat16 o;
                r4d_ar_from_f(acc, o);
                __builtin_memcpy(&xv[e], &o, 2);
                rv[e] = rr[i];
                gp[e] = gain(wp, i, wf32) + wadd;
                gw[e] = gain(w, i, wf32) + wadd;
            }
        }
        const float sv = sc ? scalar_of(sc, sc_f32) : 1.0f;

        // ---- junction_reg_row's passes over the folded row: the post-norm's sum of squares,
        // the post-norm, add and scalar with the norm's sum of squares, the norm and its absmax
        float ss = 0.0f;
#pragma unroll
        for (uint32_t e = 0; e < EPT; ++e)
            if (tid + e * THR < n) {
                if constexpr (THR == kR4dSumLanes) { const float v = bf2f(xv[e]); ss += v * v; }
                else ybuf[tid + e * THR] = xv[e];
            }
        if constexpr (THR != kR4dSumLanes) { __syncthreads(); ss = lanes_sumsq(ybuf, n); }
        const float s1 = rsqrtf(block_sum<THR / kWave>(ss, l1) / (float)n + eps);
        float ss2 = 0.0f;
#pragma unroll
        for (uint32_t e = 0; e < EPT; ++e) {
            const uint32_t i = tid + e * THR;
            if (i < n) {
                uint16_t v = f32_to_bf16(bf2f(xv[e]) * s1 * gp[e]);
                v = f32_to_bf16(bf2f(rv[e]) + bf2f(v));
                if (sc) v = f32_to_bf16(bf2f(v) * sv);
                rr[i] = v;
                xv[e] = v;
                if constexpr (THR == kR4dSumLanes) { const float f = bf2f(v); ss2 += f * f; }
                else ybuf[i] = v;
            }
        }
        if constexpr (THR != kR4dSumLanes) { __syncthreads(); ss2 = lanes_sumsq(ybuf, n); }
        const float s2 = rsqrtf(block_sum<THR / kWave>(ss2, l2) / (float)n + eps);
        float m = 0.0f;
#pragma unroll
        for (uint32_t e = 0; e < EPT; ++e) {
            const uint32_t i = tid + e * THR;
            if (i < n) {
                const uint16_t y = f32_to_bf16(bf2f(xv[e]) * s2 * gw[e]);
                ybuf[i] = y;
                m = fmaxf(m, fabsf(bf2f(y)));
            }
        }
        m = r4d_wave_max(m);
        if ((tid & 31) == 0) lm[tid >> 5] = m;
        __syncthreads();   // also publishes ybuf to every thread
        m = lm[0];
#pragma unroll
        for (uint32_t i = 1; i < THR / kWave; i++) m = fmaxf(m, lm[i]);
        const float scale = fmaxf(m, 1e-8f) / 127.f;
        const float inv = 1.f / scale;
        // ---- the codes, in junction_row's fragment order, into LDS
        for (uint32_t i = tid; i < n; i += THR) {
            const uint32_t sg = i >> 4, j = i & 15;
            const uint32_t src = (sg << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
            const int t = __float2int_rn(bf2f(ybuf[src]) * inv);
            cbuf[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
        }
        __syncthreads();   // the codes, and this block's residual row in memory, whole
        // ---- the second hop: the row's codes and scale (and residual) to every peer, and here
        const int4* const c4 = (const int4*)cbuf;
        int4* const q4 = (int4*)(q + (size_t)row * n);
        int4* dB[NPEERS];
#pragma unroll
        for (int p = 0; p < NPEERS; ++p) dB[p] = peers.scratch[p] + my_slot + offB + (size_t)b * rowB;
        for (int k = (int)tid; k < c16; k += (int)THR) {
            const int4 v = c4[k];
            q4[k] = v;
#pragma unroll
            for (int p = 0; p < NPEERS; ++p) dB[p][k] = v;
        }
        if (gres) {
            const int4* const r4 = (const int4*)rr;
            for (int k = (int)tid; k < n16; k += (int)THR) {
                const int4 v = r4[k];
#pragma unroll
                for (int p = 0; p < NPEERS; ++p) dB[p][c16 + 1 + k] = v;
            }
        }
        if (tid == 0) {
            S[row] = scale;
            const int4 sw = make_int4(__float_as_int(scale), 0, 0, 0);
#pragma unroll
            for (int p = 0; p < NPEERS; ++p) dB[p][c16] = sw;
        }
    }
    r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
    r4d_ar_link_landed(gate, gate_word, gate_ctr, gridDim.x, tid);
    // ---- the peers' local row b, codes and scale (and residual), into place
    for (int p = 0; p < NPEERS; ++p) {
        const int sr = peers.rank[p];
        if (b >= R4D_SP_NR(sr)) continue;
        const int4* const src = my_base + ((size_t)parity * WS + sr) * (size_t)slot_stride16 + offB +
                                (size_t)b * rowB;
        const int row = R4D_SP_R0(sr) + b;
        int4* const q4 = (int4*)(q + (size_t)row * n);
        for (int k = (int)tid; k < c16; k += (int)THR) q4[k] = src[k];
        if (gres) {
            int4* const r4 = (int4*)(res + (size_t)row * r_ld);
            for (int k = (int)tid; k < n16; k += (int)THR) r4[k] = src[c16 + 1 + k];
        }
        if (tid == 0) S[row] = __int_as_float(src[c16].x);
    }
#undef R4D_SP_R0
#undef R4D_SP_NR
}

#define R4D_ARQ8S_PARAMS const uint16_t* __restrict__ x, uint16_t* __restrict__ res,              \
    const void* __restrict__ w, const void* __restrict__ wp, const void* __restrict__ sc,          \
    uint32_t sc_f32, uint32_t wf32, uint32_t gres, signed char* __restrict__ q,                    \
    float* __restrict__ S, R4DArWidePeers<int4, WS - 1> peers, const int4* __restrict__ my_base,   \
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs, int M, uint32_t n,   \
    float eps, float wadd, uint32_t r_ld, int slot_stride16, int rank, int drain, int acq, int pub, \
    long long* gate, unsigned int* gate_ctr
#define R4D_ARQ8S_ARGS x, res, w, wp, sc, sc_f32, wf32, gres, q, S, peers, my_base, my_flags,     \
    seq_ctrs, M, n, eps, wadd, r_ld, slot_stride16, rank, drain, acq, pub, gate, gate_ctr
template <uint32_t EPT, uint32_t THR, int WS>
__global__ __launch_bounds__(THR) void ar_rmsnorm_quant_i8_sp_kernel(R4D_ARQ8S_PARAMS) {
    ar_rmsnorm_quant_i8_sp_body<EPT, THR, WS>(R4D_ARQ8S_ARGS);
}
template <uint32_t EPT, uint32_t THR, int WS>
__global__ __launch_bounds__(THR) void r4dchain_ar_rmsnorm_quant_i8_sp(R4D_ARQ8S_PARAMS) {
    r4d_chain_wait();
    ar_rmsnorm_quant_i8_sp_body<EPT, THR, WS>(R4D_ARQ8S_ARGS);
    r4d_chain_done();
}
#undef R4D_ARQ8S_ARGS
#undef R4D_ARQ8S_PARAMS

}  // namespace

// ============================================================== the plugin ABI entry point
//
// Operands, in the schema's order (r4d_rows.cpp, oRmsNormQuantI8):
//   0 x         [M][n] bf16           the norm's input, or the branch output with a residual
//   1 residual  [M][n] bf16, optional READ AND WRITTEN
//   2 w         [n]    bf16 or f32    the norm's gain
//   3 q         [M][n] int8           fragment order, unpitched
//   4 scale     [M][1] f32            one per row
//   5 out_bf16  [M][n] bf16, optional the norm output the unfused norm would have stored
//   6 w_post    [n]    optional       a post-norm on x before the add; needs the residual
//   7 scalar    [1]    optional       multiplies the sum; needs the residual
// rmsnorm_quant_i8g (`grp`) takes the same operands with quant_act_i8g's grid: q row-major, scale
// [M][n / 128], n a multiple of 128.
enum { RQ_X = 0, RQ_RES, RQ_W, RQ_Q, RQ_S, RQ_Y, RQ_WP, RQ_SC };

static int rad_rmsnorm_quant_i8(const RadArgs* a, RadStream stream, bool grp) {
    int emit_rc = RAD_OK;
    const RadTensor* x  = r4d_opt(a, RQ_X);
    const RadTensor* rs = r4d_opt(a, RQ_RES);
    const RadTensor* w  = r4d_opt(a, RQ_W);
    const RadTensor* q  = r4d_opt(a, RQ_Q);
    const RadTensor* s  = r4d_opt(a, RQ_S);
    const RadTensor* y  = r4d_opt(a, RQ_Y);
    const RadTensor* wp = r4d_opt(a, RQ_WP);
    const RadTensor* sc = r4d_opt(a, RQ_SC);
    if (!x || !w || !q || !s) return R4D_NO(RAD_E_INVAL);
    if ((wp || sc) && !rs) return R4D_NO(RAD_E_INVAL);   // a junction without a stream to add into
    if (x->dtype != RAD_BF16 || q->dtype != RAD_I8 || s->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    long long M = 0, n = 0;
    if (r4d_rows_of(x, &M, &n) != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    if (n % 16) return R4D_NO(RAD_E_SHAPE);                  // a k step is sixteen bytes of one row
    if (r4d_numel(w) != n) return R4D_NO(RAD_E_SHAPE);
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (wp && (r4d_numel(wp) != n || wp->dtype != w->dtype)) return R4D_NO(RAD_E_SHAPE);
    if (sc && (r4d_numel(sc) != 1 || (sc->dtype != RAD_F32 && sc->dtype != RAD_BF16)))
        return R4D_NO(RAD_E_SHAPE);
    if (grp && n % kGroupI8g) return R4D_NO(RAD_E_SHAPE);   // whole scale groups
    if (r4d_numel(q) != M * n || !r4d_contig(q)) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(s) != M * (grp ? n / kGroupI8g : 1) || !r4d_contig(s)) return R4D_NO(RAD_E_SHAPE);
    if (y && r4d_numel(y) != M * n) return R4D_NO(RAD_E_SHAPE);
    if (rs && r4d_numel(rs) != M * n) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), rp = rs ? r4d_row_pitch(rs) : n, yp = y ? r4d_row_pitch(y) : n;
    if (xp < 0 || rp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    // The row's norm output is staged in LDS for the cross-thread code store.
    const size_t lds = (size_t)n * sizeof(uint16_t);
    if (lds > 60 * 1024) return R4D_NO(RAD_E_SHAPE);

    // The same width rmsnorm_bf16 and quant_act_i8 pick, from the same function. The sums fold in
    // kR4dSumLanes' partition at any width, so the width changes speed, not bytes.
    const int thr = r4d_act_threads((int)M, (int)n);
    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    const uint32_t scf = sc ? (sc->dtype == RAD_F32) : 0u;

    // The decode form keeps a thread's columns in registers, EPT of them: at 1024 threads (to 32
    // rows) and at 512 (33 to 72-120 rows, r4d_act_threads), the same partition and arithmetic as
    // the re-read form at that width, so the same bytes. At 512 it serves the multi-user verify
    // (five to fifteen users of a DFlash2 verify, 8 rows a user), where the re-read form took ~14 us
    // a junction under --profile-ops at 64 rows of 5376.
    const int ept = !r4d_cfg_rmsq8_reg(a) ? 0
                  : thr == 1024 ? (n <= 4096 ? 4 : n <= 6144 ? 6 : n <= 8192 ? 8 : 0)
                  : thr == 512  ? (n <= 4096 ? 8 : n <= 6144 ? 12 : n <= 8192 ? 16 : 0)
                  : 0;

#define R4D_RQI8_T(WF, RS, PO, SC, T, G) emit_rc = R4D_EMIT(stream,                              \
    (rmsnorm_quant_i8_kernel<WF, RS, PO, SC, T, G>), dim3((uint32_t)M), dim3(T), lds,            \
    (const uint16_t*)r4d_p(x), (uint16_t*)r4d_wp(rs), r4d_p(w), r4d_p(wp), r4d_p(sc), scf,       \
    (signed char*)r4d_wp(q), (float*)r4d_wp(s), (uint16_t*)r4d_wp(y), (uint32_t)n, eps, wadd,    \
    (uint32_t)xp, (uint32_t)rp, (uint32_t)yp)
#define R4D_RQI8_R(WF, RS, PO, SC, T, E, G) emit_rc = R4D_EMIT(stream,                           \
    (rmsnorm_quant_i8_reg_kernel<WF, RS, PO, SC, T, E, G>), dim3((uint32_t)M), dim3(T), lds,     \
    (const uint16_t*)r4d_p(x), (uint16_t*)r4d_wp(rs), r4d_p(w), r4d_p(wp), r4d_p(sc), scf,       \
    (signed char*)r4d_wp(q), (float*)r4d_wp(s), (uint16_t*)r4d_wp(y), (uint32_t)n, eps, wadd,    \
    (uint32_t)xp, (uint32_t)rp, (uint32_t)yp)
#define R4D_RQI8(WF, RS, PO, SC, G)                                                               \
    do { if (thr == 1024 && ept == 4)      R4D_RQI8_R(WF, RS, PO, SC, 1024, 4, G);                \
         else if (thr == 1024 && ept == 6) R4D_RQI8_R(WF, RS, PO, SC, 1024, 6, G);                \
         else if (thr == 1024 && ept == 8) R4D_RQI8_R(WF, RS, PO, SC, 1024, 8, G);                \
         else if (thr == 512 && ept == 8)  R4D_RQI8_R(WF, RS, PO, SC, 512, 8, G);                 \
         else if (thr == 512 && ept == 12) R4D_RQI8_R(WF, RS, PO, SC, 512, 12, G);                \
         else if (thr == 512 && ept == 16) R4D_RQI8_R(WF, RS, PO, SC, 512, 16, G);                \
         else if (thr == 1024) R4D_RQI8_T(WF, RS, PO, SC, 1024, G);                               \
         else if (thr == 512) R4D_RQI8_T(WF, RS, PO, SC, 512, G);                                 \
         else if (thr == 256) R4D_RQI8_T(WF, RS, PO, SC, 256, G);                                 \
         else return R4D_NO(RAD_E_INVAL); } while (0)
#define R4D_RQI8_W(WF, G)                                                                         \
    do { if (!rs)            R4D_RQI8(WF, false, false, false, G);                                \
         else if (!wp && !sc) R4D_RQI8(WF, true, false, false, G);                                \
         else if (wp && !sc)  R4D_RQI8(WF, true, true, false, G);                                 \
         else if (wp && sc)   R4D_RQI8(WF, true, true, true, G);                                  \
         else                 R4D_RQI8(WF, true, false, true, G); } while (0)
    if (grp) {
        if (w->dtype == RAD_F32) R4D_RQI8_W(true, true);
        else                     R4D_RQI8_W(false, true);
    } else {
        if (w->dtype == RAD_F32) R4D_RQI8_W(true, false);
        else                     R4D_RQI8_W(false, false);
    }
#undef R4D_RQI8_W
#undef R4D_RQI8
#undef R4D_RQI8_R
#undef R4D_RQI8_T
    return emit_rc;
}

extern "C" int r4d_rad_rmsnorm_quant_i8(const RadArgs* a, RadStream stream) {
    return rad_rmsnorm_quant_i8(a, stream, false);
}

/* quant_act_i8g's grid: a separate op, not a parameter, since its scale plane has another shape and
 * its codes another order (as quant_act_i8 and quant_act_i8g are two ops). */
extern "C" int r4d_rad_rmsnorm_quant_i8g(const RadArgs* a, RadStream stream) {
    return rad_rmsnorm_quant_i8(a, stream, true);
}

// The fused all-reduce form's launch. The IPC arguments are r4d_ar_oneshot_2rank_exact's -- the
// double-buffered scratch, the per-block flags and the device-resident sequence counters -- and the
// rest is the junction's. `x` is the message and takes no pitch; `nblocks` is clamped to M (a block
// owns whole rows). `grp` takes quant_act_i8g's grid (ar_rmsnorm_quant_i8g): `s` is [M][n / 128]
// and n a multiple of 128. `chain` takes the chained form (r4d_chain.h), which the launch has at
// the decode width of 1024 threads. Returns 0, or a negative code for a shape it does not serve.
extern "C" int r4d_ar_rmsnorm_quant_i8(long peer_scratch, long my_scratch, long peer_flags,
                                       long my_flags, long seq_ctrs, long slot_stride16,
                                       long x, long res, long w, long wp, long sc, int sc_f32,
                                       long y, long q, long s, int M, int n, float eps, float wadd,
                                       int w_f32, int r_ld, int y_ld, long stream, int nblocks,
                                       int drain, int acq, int grp, long stage, int chain) {
    if (!x || !w || !q || !s) return -1;
    if (!peer_scratch || !my_scratch || !peer_flags || !my_flags || !seq_ctrs) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 16 || (grp && n % (int)kGroupI8g)) return -3;
    if ((wp || sc) && !res) return -5;
    if ((size_t)n * sizeof(uint16_t) > 60 * 1024) return -6;
    if (!r_ld) r_ld = n;
    if (!y_ld) y_ld = n;
    int nb = nblocks < 1 ? 1 : nblocks;
    if (nb > M) nb = M;
    const int nbw = r4d_ar_max_blocks();
    if (nb > nbw) nb = nbw;
    const int thr = r4d_act_threads(M, n);
    const size_t lds = (size_t)n * sizeof(uint16_t);
#define R4D_ARQ8_T(WF, RS, PO, SC, T, G, CH)                                                      \
    hipLaunchKernelGGL((ar_rmsnorm_quant_i8_kernel<WF, RS, PO, SC, T, G, CH>), dim3((uint32_t)nb), \
                       dim3(T), lds, (hipStream_t)stream, (uint16_t*)x, (uint16_t*)res,          \
                       (const void*)w, (const void*)wp, (const void*)sc, (uint32_t)sc_f32,       \
                       (signed char*)q, (float*)s, (uint16_t*)y, (int4*)peer_scratch,            \
                       (const int4*)my_scratch, (unsigned int*)peer_flags,                       \
                       (unsigned int*)my_flags, (unsigned int*)seq_ctrs, M, (uint32_t)n, eps,    \
                       wadd, (uint32_t)r_ld, (uint32_t)y_ld, (int)slot_stride16, drain, acq,      \
                       (unsigned)nbw, stage)
#define R4D_ARQ8(WF, RS, PO, SC, G)                                                               \
    do { if (thr == 1024 && chain) R4D_ARQ8_T(WF, RS, PO, SC, 1024, G, r4dchain);                 \
         else if (thr == 1024) R4D_ARQ8_T(WF, RS, PO, SC, 1024, G, R4dUnchained);                 \
         else if (thr == 512) R4D_ARQ8_T(WF, RS, PO, SC, 512, G, R4dUnchained);                   \
         else                 R4D_ARQ8_T(WF, RS, PO, SC, 256, G, R4dUnchained); } while (0)
#define R4D_ARQ8_W(WF, G)                                                                         \
    do { if (!res)            R4D_ARQ8(WF, false, false, false, G);                               \
         else if (!wp && !sc) R4D_ARQ8(WF, true, false, false, G);                                \
         else if (wp && !sc)  R4D_ARQ8(WF, true, true, false, G);                                 \
         else if (wp && sc)   R4D_ARQ8(WF, true, true, true, G);                                  \
         else                 R4D_ARQ8(WF, true, false, true, G); } while (0)
    if (grp) {
        if (w_f32) R4D_ARQ8_W(true, true);
        else       R4D_ARQ8_W(false, true);
    } else {
        if (w_f32) R4D_ARQ8_W(true, false);
        else       R4D_ARQ8_W(false, false);
    }
#undef R4D_ARQ8_W
#undef R4D_ARQ8
#undef R4D_ARQ8_T
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// The four-rank form's launch. The IPC arguments are r4d_ar_oneshot_4rank_exact's -- each peer's
// scratch and flags in ascending peer rank, this rank's own, the device-resident sequence counters
// and the per-source slot stride -- and the rest is the junction's. The checks are the wide
// launchers' (rank range, the message inside one slot, the grid inside the flag array, every
// allocation's extent), returned as codes rather than thrown, for the reason this file gives.
// The all-reduce is the one-shot. With `tail_nbt` blocks (positive) it is instead the standalone two-shot over words on that many blocks with the junction as its tail,
// a block a row (ar_rmsnorm_quant_i8_tail_kernel), counting on the two words at `tail_ctr`.
extern "C" int r4d_ar_rmsnorm_quant_i8_4rank(long peer_scratch0, long peer_scratch1,
                                             long peer_scratch2, long peer_flags0,
                                             long peer_flags1, long peer_flags2, long my_scratch,
                                             long my_flags, long seq_ctrs, long slot_stride16,
                                             long x, long res, long w, long wp, long sc,
                                             int sc_f32, long y, long q, long s, int M, int n,
                                             float eps, float wadd, int w_f32, int r_ld, int y_ld,
                                             int rank, long stream, int nblocks, int drain, int acq,
                                             int pub, long gate, long gate_ctr,
                                             long tail_ctr, int tail_nbt, int chain, int sp) {
    constexpr int WS = 4, NPEERS = WS - 1;
    const long ps[NPEERS] = { peer_scratch0, peer_scratch1, peer_scratch2 };
    const long pf[NPEERS] = { peer_flags0, peer_flags1, peer_flags2 };
    if (!x || !w || !q || !s) return -1;
    if (!my_scratch || !my_flags || !seq_ctrs) return -1;
    if (gate && !gate_ctr) return -1;   /* the release counts blocks in the instance's counter */
    if (rank < 0 || rank >= WS) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 16) return -3;
    if ((wp || sc) && !res) return -5;
    if ((size_t)n * sizeof(uint16_t) > 60 * 1024) return -6;
    if ((long)M * n / 8 > slot_stride16) return -7;   /* the message inside one source slot */
    if (!r_ld) r_ld = n;
    if (!y_ld) y_ld = n;

    R4DArWidePeers<int4, NPEERS> peers{};
    for (int i = 0; i < NPEERS; ++i) {
        if (!ps[i] || !pf[i]) return -1;
        peers.scratch[i] = (int4*)ps[i];
        peers.flags[i] = (unsigned int*)pf[i];
        peers.rank[i] = i < rank ? i : i + 1;   /* ascending global rank, this one removed */
    }
    int nb = nblocks < 1 ? 1 : nblocks;
    if (nb > M) nb = M;                          /* a block owns whole rows */
    if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
    try {
        const long slot_bytes = 2L * WS * slot_stride16 * 16;
        const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
        r4d_ar_check_extent(my_scratch, slot_bytes, "ar_rmsnorm_quant_i8", "my_scratch");
        r4d_ar_check_extent(my_flags, flag_bytes, "ar_rmsnorm_quant_i8", "my_flags");
        r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                            "ar_rmsnorm_quant_i8", "seq_ctrs");
        for (int i = 0; i < NPEERS; ++i) {
            r4d_ar_check_extent(ps[i], slot_bytes, "ar_rmsnorm_quant_i8", "peer scratch");
            r4d_ar_check_extent(pf[i], flag_bytes, "ar_rmsnorm_quant_i8", "peer flags");
        }
    } catch (...) {
        return -8;
    }

    const int thr = r4d_act_threads(M, n);
    const size_t lds = (size_t)n * sizeof(uint16_t);

    /* The all-reduce split around the junction (ar_rmsnorm_quant_i8_sp_kernel): `sp` 1, or 2 with
     * the residual gathered too. A block a local row, ceil(M / 4) of them; its two hops in the slot
     * the message would fill. Where it cannot run (no residual or post-norm, a width without a
     * register form, rows past the flag array or the slot, a residual pitch off 16 bytes) the
     * forms below run instead, which leave every byte this one does and more. It leaves no bf16
     * norm output, so one asked for (`y`) takes those forms too. */
    if (sp > 0 && res && wp && !y) {
        const int ep = thr == 1024 ? (n <= 4096 ? 4 : n <= 6144 ? 6 : n <= 8192 ? 8 : 0)
                     : thr == 512  ? (n <= 4096 ? 8 : n <= 6144 ? 12 : n <= 8192 ? 16 : 0) : 0;
        const int nbs = (M + WS - 1) / WS;
        const long need = (long)nbs * (n / 8) + (long)nbs * (n / 16 + 1 + (sp == 2 ? n / 8 : 0));
        if (ep > 0 && nbs <= R4D_AR_WIDE_MAX_BLOCKS && need <= slot_stride16 &&
            (sp != 2 || r_ld % 8 == 0)) {
            try {
                const long fb = (long)nbs * WS * (long)sizeof(unsigned int);
                r4d_ar_check_extent(my_flags, fb, "ar_rmsnorm_quant_i8", "my_flags");
                for (int i = 0; i < NPEERS; ++i)
                    r4d_ar_check_extent(pf[i], fb, "ar_rmsnorm_quant_i8", "peer flags");
            } catch (...) {
                return -8;
            }
            const size_t lds_sp = (size_t)n * (sizeof(uint16_t) + 1);
#define R4D_ARQ8S_K(E, T)                                                                         \
            hipLaunchKernelGGL((chain ? r4dchain_ar_rmsnorm_quant_i8_sp<E, T, WS>                 \
                                      : ar_rmsnorm_quant_i8_sp_kernel<E, T, WS>),                \
                               dim3((uint32_t)nbs), dim3(T), lds_sp, (hipStream_t)stream,         \
                               (const uint16_t*)x, (uint16_t*)res, (const void*)w,                \
                               (const void*)wp, (const void*)sc, (uint32_t)sc_f32,                \
                               (uint32_t)w_f32, (uint32_t)(sp == 2), (signed char*)q, (float*)s,  \
                               peers, (const int4*)my_scratch, (unsigned int*)my_flags,           \
                               (unsigned int*)seq_ctrs, M, (uint32_t)n, eps, wadd,                \
                               (uint32_t)r_ld, (int)slot_stride16, rank, drain, acq, pub,         \
                               (long long*)gate, (unsigned int*)gate_ctr)
            if (thr == 1024 && ep == 4)      R4D_ARQ8S_K(4, 1024);
            else if (thr == 1024 && ep == 6) R4D_ARQ8S_K(6, 1024);
            else if (thr == 1024)            R4D_ARQ8S_K(8, 1024);
            else if (ep == 8)                R4D_ARQ8S_K(8, 512);
            else if (ep == 12)               R4D_ARQ8S_K(12, 512);
            else                             R4D_ARQ8S_K(16, 512);
#undef R4D_ARQ8S_K
            return hipGetLastError() == hipSuccess ? 0 : -4;
        }
    }

    /* The junction as the standalone two-shot's tail (ar_rmsnorm_quant_i8_tail_kernel): at the
     * junction's register width (1024 threads), up to R4D_ARQ8_TAIL_MAX_ROWS rows. */
    const int ept = n <= 4096 ? 4 : n <= 6144 ? 6 : n <= 8192 ? 8 : 0;
    if (tail_nbt > 0 && tail_ctr && thr == 1024 && ept > 0 && M <= R4D_ARQ8_TAIL_MAX_ROWS &&
        tail_nbt <= R4D_AR_WIDE_MAX_BLOCKS) {
        try {
            r4d_ar_check_extent(my_flags, (long)tail_nbt * WS * (long)sizeof(unsigned int),
                                "ar_rmsnorm_quant_i8", "my_flags");
            r4d_ar_check_extent(tail_ctr, 2L * (long)sizeof(unsigned int), "ar_rmsnorm_quant_i8",
                                "tail counters");
            for (int i = 0; i < NPEERS; ++i)
                r4d_ar_check_extent(pf[i], (long)tail_nbt * WS * (long)sizeof(unsigned int),
                                    "ar_rmsnorm_quant_i8", "peer flags");
        } catch (...) {
            return -8;
        }
#define R4D_ARQ8T_K(WF, RS, PO, SC, E)                                                            \
        hipLaunchKernelGGL((chain ? r4dchain_ar_rmsnorm_quant_i8_tail<WF, RS, PO, SC, E, WS>       \
                                  : ar_rmsnorm_quant_i8_tail_kernel<WF, RS, PO, SC, E, WS>),      \
                           dim3((uint32_t)(tail_nbt + M)), dim3(1024), lds, (hipStream_t)stream,  \
                           (uint16_t*)x, (uint16_t*)res, (const void*)w, (const void*)wp,         \
                           (const void*)sc, (uint32_t)sc_f32, (signed char*)q, (float*)s,         \
                           (uint16_t*)y, peers, (const int4*)my_scratch, (unsigned int*)my_flags, \
                           (unsigned int*)seq_ctrs, (unsigned int*)tail_ctr, M, (uint32_t)n, eps,  \
                           wadd, (uint32_t)r_ld, (uint32_t)y_ld, (int)slot_stride16, rank,         \
                           tail_nbt, drain, acq, pub, (long long*)gate, (unsigned int*)gate_ctr)
#define R4D_ARQ8T_E(WF, RS, PO, SC)                                                               \
        do { if (ept == 4)      R4D_ARQ8T_K(WF, RS, PO, SC, 4);                                   \
             else if (ept == 6) R4D_ARQ8T_K(WF, RS, PO, SC, 6);                                   \
             else               R4D_ARQ8T_K(WF, RS, PO, SC, 8); } while (0)
#define R4D_ARQ8T_W(WF)                                                                           \
        do { if (!res)            R4D_ARQ8T_E(WF, false, false, false);                           \
             else if (!wp && !sc) R4D_ARQ8T_E(WF, true, false, false);                            \
             else if (wp && !sc)  R4D_ARQ8T_E(WF, true, true, false);                             \
             else if (wp && sc)   R4D_ARQ8T_E(WF, true, true, true);                              \
             else                 R4D_ARQ8T_E(WF, true, false, true); } while (0)
        if (w_f32) R4D_ARQ8T_W(true);
        else       R4D_ARQ8T_W(false);
#undef R4D_ARQ8T_W
#undef R4D_ARQ8T_E
#undef R4D_ARQ8T_K
        return hipGetLastError() == hipSuccess ? 0 : -4;
    }

#define R4D_ARQ8W_T(WF, RS, PO, SC, T)                                                            \
    hipLaunchKernelGGL((ar_rmsnorm_quant_i8_wide_kernel<WF, RS, PO, SC, T, WS>), dim3((uint32_t)nb), \
                       dim3(T), lds, (hipStream_t)stream, (uint16_t*)x, (uint16_t*)res,          \
                       (const void*)w, (const void*)wp, (const void*)sc, (uint32_t)sc_f32,       \
                       (signed char*)q, (float*)s, (uint16_t*)y, peers, (const int4*)my_scratch, \
                       (unsigned int*)my_flags, (unsigned int*)seq_ctrs, M, (uint32_t)n, eps,    \
                       wadd, (uint32_t)r_ld, (uint32_t)y_ld, (int)slot_stride16, rank, drain, acq, \
                       pub, (long long*)gate, (unsigned int*)gate_ctr)
#define R4D_ARQ8W(WF, RS, PO, SC)                                                                 \
    do { if (thr == 1024)     R4D_ARQ8W_T(WF, RS, PO, SC, 1024);                                  \
         else if (thr == 512) R4D_ARQ8W_T(WF, RS, PO, SC, 512);                                   \
         else                 R4D_ARQ8W_T(WF, RS, PO, SC, 256); } while (0)
#define R4D_ARQ8W_W(WF)                                                                           \
    do { if (!res)            R4D_ARQ8W(WF, false, false, false);                                 \
         else if (!wp && !sc) R4D_ARQ8W(WF, true, false, false);                                  \
         else if (wp && !sc)  R4D_ARQ8W(WF, true, true, false);                                   \
         else if (wp && sc)   R4D_ARQ8W(WF, true, true, true);                                    \
         else                 R4D_ARQ8W(WF, true, false, true); } while (0)
    if (w_f32) R4D_ARQ8W_W(true);
    else       R4D_ARQ8W_W(false);
#undef R4D_ARQ8W_W
#undef R4D_ARQ8W
#undef R4D_ARQ8W_T
    return hipGetLastError() == hipSuccess ? 0 : -4;
}
