// r4d_hc_bf16.hip -- the gated residual (hyper-connections): hc_enter, hc_read,
// hc_write and hc_project.
//
// Qwen4-Exp carries a residual stream `hc` times the model width, with no pre-norm and no residual
// add: every block reads a data-dependent mix of the hc sub-streams and writes its output back into
// all of them with hc data-dependent gains. Schema: docs/OPS.md; composition:
// arch/common/rad_block_hc.h.
//
// ============================== hc_read is two kernels ==============================
//
// The read is
//
//     N   = grouped_rmsnorm(h)                     [HN]   HN = hc * n
//     d   = silu(W_down @ N / hc)                  [LR]   LR = lowrank
//     x   = mean_c( sigmoid(W_up @ d)[c] * N[c] )  [n]
//     inj = 2 * sigmoid(W_inj @ N / hc)            [hc]
//
// `x` depends on all of `d`, so no workgroup can produce a channel of `x` before every column of
// `d` exists. The kernel boundary serves as the grid-wide barrier, without cooperative launch.
// For a small decode step (E4M3, hc 4, n 2560, up to eight rows), hc_read_fused_kernel runs A and
// B as one grid instead, its B blocks waiting on a ticket-ordered counter (see that kernel's note).
//
// Weight traffic, not arithmetic, decides the decomposition. At Qwen3.8-Flash-Next's shape
// (n 2560, hc 4, LR 320) each mixing matrix is 6.55 MB and there are 96 connections: 1.27 GB per
// step, a 2.1 ms floor at ~600 GB/s, against a negligible 6.6M MACs per token. So each weight byte
// must be read once per step:
//
//   kernel A   grid (ceil(LR/64), T). A workgroup owns 64 columns of W_down and reads all of h
//              for the norm, so W_down is read once and h ceil(LR/64) times per row (100 KB
//              against 1.3 MB of W_down per workgroup).
//   kernel B   grid (n/128, T). A workgroup owns 128 channels (hc*128 rows of W_up), so W_up is
//              read once, and `u` is never materialised because the workgroup holds all hc rows of
//              a channel and does the mean itself.
//
// 128 channels are exactly one fp8 scale group, so kernel B also quantises `x` to fp8 with no
// extra pass or launch (a separate quant_act_fp8 at 96 connections would cost ~0.43 ms of launch
// overhead).
//
// Tiled path: above ~8 rows the per-row kernels would re-read both matrices per row, so the
// products run tiled. A tile over the flattened channel index (64 consecutive `j`) covers one
// sub-stream, putting a channel's hc terms in different workgroups; `u` must then go through
// `ubuf` and a third kernel (`hc_mix`), costing a launch and a 1.28 MB round trip per connection.
//
// `hc_up_mix_tiled_kernel` instead tiles over (sub-stream, channel) pairs: the same W_up bytes in
// hc runs of NT/hc rows, with all hc terms of a channel in one workgroup. A workgroup then owns
// NT/hc = 16 channels of a 128-wide scale group, so an arrival counter lets the last of the eight
// workgroups covering a group quantise it (1-2% of the step at multi-sequence decode, same
// output). At the 128-row prefill tile the same pairing is `hc_up_mix_wide_kernel`, which keeps a
// channel's sub-streams in one lane's accumulators instead of LDS.
//
// ============================== matching the reference ==============================
//
// libref/ref_hc.cpp is the oracle. torch's grouped norm returns `.type_as(x)`, so the reference
// model reads a bf16 N in both matrix products and the final multiply. This kernel narrows N, `d`
// and sigmoid(u) in the same places. Sums of squares are f32 here and f64 there, within bf16
// tolerance.
#include <hip/hip_runtime.h>

#include <atomic>
#include <type_traits>
#include <cstdlib>
#include <mutex>
#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_gdn_args.h"
#include "r4d_common.h"
#include "r4d_chain.h"
#include "r4d_gdn_wmma.h"
#include "r4d_fwht.h"
#include "r4d_hc_fp8.h"
#include "r4d_q8.h"

namespace {

constexpr int kWave    = 32;
constexpr int kThreads = 256;
constexpr int kWaves   = kThreads / kWave;
constexpr int kHcMax   = 8;      // streams; the reference refuses hc <= 1 and nothing needs > 8
constexpr int kDownTAMax = 64;   // columns of W_down a workgroup owns, at most
constexpr int kUpRows  = 64;     // rows of W_up a workgroup owns
constexpr int kWantWG  = 64;     // workgroups worth launching, when the CU count cannot be read

/* Returns WGPs, not CUs: `hipDeviceProp_t::multiProcessorCount` reports WGPs on RDNA, two CUs
 * each (the R9700 has 64 CUs in 32 WGPs). Left as is on purpose: it is a floor on workgroups worth
 * launching, and here grids of 40, 80 and 160 perform the same. Halving `ta` only doubles redundant
 * work, since each workgroup re-reads all of `h` and the norm weight (60 KiB here) before its own
 * W_down columns (4.8 MB of re-read at ta=4, 2.4 MB at ta=8). A kernel short of blocks should
 * double the number at its own call site. */
static inline int want_wg() {
    static const int v = [] {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return kWantWG;
        hipDeviceProp_t pr{};
        if (hipGetDeviceProperties(&pr, dev) != hipSuccess) return kWantWG;
        return pr.multiProcessorCount > 0 ? pr.multiProcessorCount : kWantWG;
    }();
    return v;
}

constexpr int kUpTB    = 128;    // CHANNELS a workgroup owns -- and one fp8 scale group
constexpr int kFp8Grp  = 128;
constexpr int kDotU    = 4;      // 512-byte loads a wave keeps in flight over a long weight row

__device__ __forceinline__ float sigmoidf_(float v) { return 1.0f / (1.0f + __expf(-v)); }
__device__ __forceinline__ float siluf_(float v)    { return v * sigmoidf_(v); }
__device__ __forceinline__ float nbf(float v)       { return bf2f(f2bf(v)); }

/* bf16 in whole registers. gfx12's compiler holds a 16-bit value in a register half: shorts written
 * through a union cost a half move each before the pack, and a lone d16 load leaves the other half
 * live, waiting on that register's last write. These helpers keep values 32 bits wide (pairs packed
 * with f2bf2, same rounding as f2bf; halves read by shift and mask; lone bf16 loaded zero-extended)
 * and give the same bits. */
__device__ __forceinline__ uint4 f2bf8(const float (&f)[8]) {
    return make_uint4(f2bf2(f[0], f[1]), f2bf2(f[2], f[3]), f2bf2(f[4], f[5]), f2bf2(f[6], f[7]));
}
__device__ __forceinline__ void bf8_f(uint4 v, float (&f)[8]) {
    f[0] = bf16w_lo(v.x); f[1] = bf16w_hi(v.x); f[2] = bf16w_lo(v.y); f[3] = bf16w_hi(v.y);
    f[4] = bf16w_lo(v.z); f[5] = bf16w_hi(v.z); f[6] = bf16w_lo(v.w); f[7] = bf16w_hi(v.w);
}
__device__ __forceinline__ void bf8_f(r4d_u32x4 v, float (&f)[8]) {
    f[0] = bf16w_lo(v.x); f[1] = bf16w_hi(v.x); f[2] = bf16w_lo(v.y); f[3] = bf16w_hi(v.y);
    f[4] = bf16w_lo(v.z); f[5] = bf16w_hi(v.z); f[6] = bf16w_lo(v.w); f[7] = bf16w_hi(v.w);
}
/* Read through a global pointer; through a generic one the load is still flat_load_d16_b16. */
typedef __attribute__((address_space(1))) const unsigned short* hc_gu16;
__device__ __forceinline__ uint32_t bfw(const unsigned short* p) { return *(hc_gu16)p; }
__device__ __forceinline__ float    bfw_f(uint32_t w)            { return __uint_as_float(w << 16); }

/* ---- E4M3 mixing matrices (r4d_hc_fp8.h) ------------------------------------------------
 *
 * Every kernel reading mix_down or mix_up has an F8 form, selected by the stored dtype: a row is
 * its codes then its f32 group scales, `ld` bytes apart. Per-row kernels multiply codes in f32 and
 * scale each group's partial sum; tiled kernels decode into their bf16 staging tile with the scale
 * applied and keep the bf16 WMMA. The bf16 form compiles from the same text with F8 false. */

/* Eight E4M3 codes, two dwords of them, to f32 in order: v_cvt_pk_f32_fp8, two a time. */
__device__ __forceinline__ void e4m3x8(uint32_t lo, uint32_t hi, float* f) {
    const v2f a = __builtin_amdgcn_cvt_pk_f32_fp8((int)lo, false);
    const v2f b = __builtin_amdgcn_cvt_pk_f32_fp8((int)lo, true);
    const v2f c = __builtin_amdgcn_cvt_pk_f32_fp8((int)hi, false);
    const v2f d = __builtin_amdgcn_cvt_pk_f32_fp8((int)hi, true);
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
    f[4] = c.x; f[5] = c.y; f[6] = d.x; f[7] = d.y;
}

/* Eight bf16 of a uint4 to f32, in order. */
__device__ __forceinline__ void bf16x8(const uint4& v, float* f) {
    const uint32_t w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i]     = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

/* Eight codes of an E4M3 row, decoded and scaled into the bf16 a tiled kernel stages. */
__device__ __forceinline__ uint4 e4m3x8_bf16(const unsigned char* __restrict__ codes, float sc) {
    const uint2 c = *(const uint2*)codes;
    float f[8];
    e4m3x8(c.x, c.y, f);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] *= sc;
    return f2bf8(f);
}

/* A wave's dot of the natural-order LDS vector with one E4M3 row of mix_down. A lane takes sixteen
 * consecutive codes of every 512-element step (one 16-byte load, so a wave requests 512 bytes like
 * the bf16 dots). The sixteen share one 128-group, whose scale multiplies the lane's partial sum
 * for the step. kU steps are loaded before the first is summed. K is a multiple of 512. */
constexpr long kHc8Step = kWave * 16;

__device__ __forceinline__ float hc8_dot16(const unsigned short* __restrict__ sv, uint4 c) {
    float wf[16], nf[16];
    e4m3x8(c.x, c.y, wf);
    e4m3x8(c.z, c.w, wf + 8);
    bf16x8(*(const uint4*)sv, nf);
    bf16x8(*(const uint4*)(sv + 8), nf + 8);
    float p = 0.0f;
#pragma unroll
    for (int t = 0; t < 16; ++t) p = fmaf(wf[t], nf[t], p);
    return p;
}

template <int kU>
__device__ __forceinline__ float hc8_dot_down(const unsigned short* __restrict__ sv,
                                              const unsigned char* __restrict__ wr, long K,
                                              int lane) {
    const float* __restrict__ sr = (const float*)(wr + K);
    float acc = 0.0f;
    long  j = 0;
    for (; j + kHc8Step * kU <= K; j += kHc8Step * kU) {
        uint4 c[kU];
        float s[kU];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const long e = j + (long)u * kHc8Step + lane * 16;
            c[u] = *(const uint4*)(wr + e);
            s[u] = sr[e / r4d_hc8::kDownGroup];
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const long e = j + (long)u * kHc8Step + lane * 16;
            acc = fmaf(hc8_dot16(sv + e, c[u]), s[u], acc);
        }
    }
    for (; j < K; j += kHc8Step) {
        const long e = j + lane * 16;
        acc = fmaf(hc8_dot16(sv + e, *(const uint4*)(wr + e)), sr[e / r4d_hc8::kDownGroup], acc);
    }
    return acc;
}

/* hc8_dot_down for two rows of W_down at once, one read of N from LDS feeding both, so a wave
 * owning two columns (four rows of a decode step) reads the 20 KB vector once rather than once per
 * column; kernel A's dots are LDS-bound. Each row's sum is the same fmaf sequence as
 * hc8_dot_down's. */
__device__ __forceinline__ float hc8_dot16n(const float* nf, uint4 c) {
    float wf[16];
    e4m3x8(c.x, c.y, wf);
    e4m3x8(c.z, c.w, wf + 8);
    float p = 0.0f;
#pragma unroll
    for (int t = 0; t < 16; ++t) p = fmaf(wf[t], nf[t], p);
    return p;
}

template <int kU>
__device__ __forceinline__ void hc8_dot_down2(const unsigned short* __restrict__ sv,
                                              const unsigned char* __restrict__ wr0,
                                              const unsigned char* __restrict__ wr1, long K,
                                              int lane, float& out0, float& out1) {
    const float* __restrict__ sr0 = (const float*)(wr0 + K);
    const float* __restrict__ sr1 = (const float*)(wr1 + K);
    float acc0 = 0.0f, acc1 = 0.0f;
    long  j = 0;
    for (; j + kHc8Step * kU <= K; j += kHc8Step * kU) {
        uint4 c0[kU], c1[kU];
        float s0[kU], s1[kU];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const long e = j + (long)u * kHc8Step + lane * 16;
            c0[u] = *(const uint4*)(wr0 + e);
            c1[u] = *(const uint4*)(wr1 + e);
            s0[u] = sr0[e / r4d_hc8::kDownGroup];
            s1[u] = sr1[e / r4d_hc8::kDownGroup];
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const long e = j + (long)u * kHc8Step + lane * 16;
            float nf[16];
            bf16x8(*(const uint4*)(sv + e), nf);
            bf16x8(*(const uint4*)(sv + e + 8), nf + 8);
            acc0 = fmaf(hc8_dot16n(nf, c0[u]), s0[u], acc0);
            acc1 = fmaf(hc8_dot16n(nf, c1[u]), s1[u], acc1);
        }
    }
    for (; j < K; j += kHc8Step) {
        const long e = j + lane * 16;
        float nf[16];
        bf16x8(*(const uint4*)(sv + e), nf);
        bf16x8(*(const uint4*)(sv + e + 8), nf + 8);
        acc0 = fmaf(hc8_dot16n(nf, *(const uint4*)(wr0 + e)), sr0[e / r4d_hc8::kDownGroup], acc0);
        acc1 = fmaf(hc8_dot16n(nf, *(const uint4*)(wr1 + e)), sr1[e / r4d_hc8::kDownGroup], acc1);
    }
    out0 = acc0;
    out1 = acc1;
}

/* One thread's quarter of an E4M3 row of mix_up against RG rows of `d` in LDS. The quarter is the
 * row's scale group (r4d_hc8::up_group), so its partial sums are complete before scaling, and four
 * threads' sums make the row. All of a quarter's sixteen-code loads issue before the first is used
 * (five at lowrank 320); kUpMaxQ bounds lowrank at 512. */
constexpr int kUpMaxQ = 8;
template <int RG>
__device__ __forceinline__ void hc8_dot_up(const float* __restrict__ sd, long lr,
                                           const unsigned char* __restrict__ wr, int q4,
                                           float* acc) {
    const long G = lr / 4, k0 = (long)q4 * G;
    const int  nq = (int)(G / 16);
    uint4 c[kUpMaxQ];
#pragma unroll
    for (int i = 0; i < kUpMaxQ; ++i)
        if (i < nq) c[i] = *(const uint4*)(wr + k0 + 16 * i);
#pragma unroll
    for (int r = 0; r < RG; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < kUpMaxQ; ++i) {
        if (i >= nq) break;
        float wf[16];
        e4m3x8(c[i].x, c[i].y, wf);
        e4m3x8(c[i].z, c[i].w, wf + 8);
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            const float* dr = sd + (long)r * lr + k0 + 16 * i;
#pragma unroll
            for (int t = 0; t < 16; ++t) acc[r] = fmaf(wf[t], dr[t], acc[r]);
        }
    }
    const float sc = ((const float*)(wr + lr))[q4];
#pragma unroll
    for (int r = 0; r < RG; ++r) {
        float a = acc[r] * sc;
        a += __shfl_xor(a, 1);
        a += __shfl_xor(a, 2);
        acc[r] = a;
    }
}

/* hc8_dot_up split at its loads, so a kernel can issue them before `d` exists: a
 * quarter's codes and its group scale, then the same arithmetic in the same order. */
struct Hc8UpQuarter {
    uint4 c[kUpMaxQ];
    float sc;
};
__device__ __forceinline__ void hc8_up_load(const unsigned char* __restrict__ wr, long lr,
                                            int q4, Hc8UpQuarter& o) {
    const long G = lr / 4, k0 = (long)q4 * G;
    const int  nq = (int)(G / 16);
#pragma unroll
    for (int i = 0; i < kUpMaxQ; ++i)
        if (i < nq) o.c[i] = *(const uint4*)(wr + k0 + 16 * i);
    o.sc = ((const float*)(wr + lr))[q4];
}
/* RPT rows of W_up with the same quarter against RG rows of `d`: each value of `d` read from LDS
 * feeds RPT products, and `d` sits in LDS as the bf16 it is, sixteen values in two 16-byte reads
 * (the quarter, its group and lr are multiples of 16 by r4d_hc8::group_ok, and the base is aligned).
 * Read as floats a value at a time, the dot would be a chain of LDS round trips. Each (row, r)
 * accumulates the same fmaf sequence on the same values as hc8_dot_up. */
template <int RG, int RPT>
__device__ __forceinline__ void hc8_up_dot_rows(const unsigned short* __restrict__ sdh, long lr,
                                                const Hc8UpQuarter* o, int q4,
                                                float (*acc)[RG]) {
    const long G = lr / 4, k0 = (long)q4 * G;
    const int  nq = (int)(G / 16);
#pragma unroll
    for (int m = 0; m < RPT; ++m)
#pragma unroll
        for (int r = 0; r < RG; ++r) acc[m][r] = 0.0f;
#pragma unroll
    for (int i = 0; i < kUpMaxQ; ++i) {
        if (i >= nq) break;
        float wf[RPT][16];
#pragma unroll
        for (int m = 0; m < RPT; ++m) {
            e4m3x8(o[m].c[i].x, o[m].c[i].y, wf[m]);
            e4m3x8(o[m].c[i].z, o[m].c[i].w, wf[m] + 8);
        }
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            const unsigned short* dr = sdh + (long)r * lr + k0 + 16 * i;
            float dv[16];
            bf8_f(*(const uint4*)dr, *(float(*)[8])dv);
            bf8_f(*(const uint4*)(dr + 8), *(float(*)[8])(dv + 8));
#pragma unroll
            for (int t = 0; t < 16; ++t) {
#pragma unroll
                for (int m = 0; m < RPT; ++m) acc[m][r] = fmaf(wf[m][t], dv[t], acc[m][r]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < RPT; ++m)
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            float a = acc[m][r] * o[m].sc;
            a += __shfl_xor(a, 1);
            a += __shfl_xor(a, 2);
            acc[m][r] = a;
        }
}

/* On the VALU, not the LDS crossbar (r4d_common.h). The norm prologue reduces `hc + 1`
 * accumulators back to back with a runtime `hc`, so the exchanges form one long dependency chain.
 * Bit-identical: same lanes, same order. */
__device__ __forceinline__ float wave_sum(float v) { return r4d_wave_sum(v); }

/* A barrier that orders LDS only. __syncthreads() fences at workgroup scope, which in
 * WGP mode on gfx12 waits for every outstanding global load: a load issued for the next pass of a
 * K loop cannot stay in flight across it, so a software-pipelined loop runs as serially as the
 * plain one. Where a barrier guards only LDS tiles, this one leaves global loads in flight. */
__device__ __forceinline__ void hc_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

/* ============================================================== hc_enter
 *
 * h[c][i] = x[i] for every stream. One launch per step; its own op because the embedding lookup's
 * weight is [vocab, n] and a wide output would need a different weight shape. */
#define HC_ENTER_PARAMS                                                                         \
    const unsigned short* __restrict__ x, unsigned short* __restrict__ h, long rows, long n,    \
    int hc, long xrow, long hrow
__device__ __forceinline__ void hc_enter_body(HC_ENTER_PARAMS) {
    const long r = blockIdx.y;
    if (r >= rows) return;
    const unsigned short* xp = x + r * xrow;
    unsigned short*       hp = h + r * hrow;
    for (long i = (long)blockIdx.x * kThreads + threadIdx.x; i < n; i += (long)gridDim.x * kThreads) {
        const unsigned short v = xp[i];
        for (int c = 0; c < hc; ++c) hp[(long)c * n + i] = v;
    }
}

/* The kernel around the body, whose blocks return where their work ends: CH `r4dchain` is the
 * chained form (r4d_chain.h). */
template <class CH>
__global__ __launch_bounds__(kThreads)
void hc_enter_kernel(HC_ENTER_PARAMS) {
    if (CH::on) r4d_chain_wait();
    hc_enter_body(x, h, rows, n, hc, xrow, hrow);
    if (CH::on) r4d_chain_done();
}
#undef HC_ENTER_PARAMS

/* ============================================================== hc_write
 *
 * h[c][i] += inj[c] * y[i], in place on the unnormed stream: the read's normalisation feeds the
 * block input and gates only, never the residual. */
__global__ __launch_bounds__(kThreads)
void hc_write_kernel(const unsigned short* __restrict__ y, const unsigned short* __restrict__ inj,
                     unsigned short* __restrict__ h,
                     long rows, long n, int hc, long yrow, long irow, long hrow) {
    const long r = blockIdx.y;
    if (r >= rows) return;
    /* Gains go through LDS, not a register array: with a runtime `hc`, `g[c]` would be a dynamic
     * private index, which the compiler spills to scratch. */
    __shared__ float s_g[kHcMax];
    if ((int)threadIdx.x < hc) s_g[threadIdx.x] = bf2f(inj[r * irow + threadIdx.x]);
    __syncthreads();
    const unsigned short* yp = y + r * yrow;
    unsigned short*       hp = h + r * hrow;
    for (long i = (long)blockIdx.x * kThreads + threadIdx.x; i < n; i += (long)gridDim.x * kThreads) {
        const float v = bf2f(yp[i]);
        for (int c = 0; c < hc; ++c) {
            unsigned short* p = hp + (long)c * n + i;
            *p = f2bf(bf2f(*p) + s_g[c] * v);
        }
    }
}

/* ============================================================== hc_project
 *
 * The control vector (core/cvec.h): h[s] -= scale * (h[s] . v) * v for each of the hc streams of a
 * row whose `on` flag is 1 -- llama.cpp's `--cvec-mode project`, issued after a layer's write into
 * the stream. ONE direction for every stream: the dot is per stream, against the same v. One
 * workgroup per (row, stream), and a zero flag is not read and not written, so an unprojected
 * request's stream is the stock model's. */
__global__ __launch_bounds__(kThreads)
void hc_project_kernel(unsigned short* __restrict__ h, const float* __restrict__ v,
                       const int32_t* __restrict__ on, long rows, long n, long hrow, float scale) {
    const long r = blockIdx.y;
    if (r >= rows || on[r] == 0) return;   /* a row's switch: the whole workgroup takes it */
    unsigned short* hp = h + r * hrow + (long)blockIdx.x * n;

    float acc = 0.0f;
    for (long i = threadIdx.x; i < n; i += kThreads) acc += bf2f(hp[i]) * v[i];
    /* A wave sums itself and the waves meet in LDS, at kWave 32: the shuttle is the wave's own
     * width here, not the 64 a 64-wide __shfl_down would assume but gfx12 does not run. */
    __shared__ float s_g[kWaves];
    const float w = r4d_wave_sum(acc);
    if ((threadIdx.x % kWave) == 0) s_g[threadIdx.x / kWave] = w;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = 0.0f;
        for (int k = 0; k < kWaves; ++k) g += s_g[k];
        s_g[0] = g * scale;
    }
    __syncthreads();

    const float g = s_g[0];
    for (long i = threadIdx.x; i < n; i += kThreads) hp[i] = f2bf(bf2f(hp[i]) - g * v[i]);
}

/* A wave's dot product of the LDS vector with a contiguous bf16 row, using 512-byte loads but
 * summing in the scalar form's order.
 *
 * A strided `j = lane` loop issues 64-byte loads per wave instruction (219 GB/s here, against 631
 * for the lm head). Eight contiguous bf16 per lane (512 bytes) reaches 554 but changes which j a
 * lane owns, and so the last bit of every output. So the wide load is staged through LDS and each
 * lane reads back its strided subset, ascending (lane L sums j = L, L+32, ...). This is also
 * faster than reordering j (8.9 us vs 9.0, scalar 29.9), whose lanes conflict on `sN`'s banks.
 * `stage` is this wave's own 256-entry slot, so no barrier is needed: a wave's LDS operations
 * complete in order.
 *
 * An unaligned row (base or `HN % 8`) falls back to the scalar loop. */
__device__ __forceinline__ float sv_at(const unsigned short* p, long i) { return bf2f(p[i]); }
__device__ __forceinline__ float sv_at(const float* p, long i) { return p[i]; }

/* `kU` is how many loads a wave keeps in flight, which sets kernel A's speed. With one load per
 * iteration the global latency is fully exposed: 40 workgroups of 8 waves keep only 320 x 512 B in
 * flight, 258 GB/s, while kernel B reaches 504 GB/s on the same bytes because each wave owns eight
 * rows. Block count (40 to 160) makes no difference; depth does. So kU loads are issued into
 * registers first (back-to-back global_load_dwordx4), then walked through LDS one at a time.
 *
 * Bit-identical: the summation order is unchanged, only arrival time moves. The staging slot is
 * reused kU times (kU ordered LDS round trips, ~100 cycles each) rather than replicated, which
 * would raise kernel A's LDS from 25 to 37 KiB and halve its occupancy. */
template <int kU, typename SV>
__device__ __forceinline__ float wave_dot_lds(const SV* __restrict__ sv,
                                              const unsigned short* __restrict__ wr, long HN,
                                              int lane, unsigned short* stage) {
    constexpr long kPer = kWave * 8;
    float acc = 0.0f;
    long  j = 0;
    if (((uintptr_t)wr & 15u) == 0u) {
        if (kU > 1) {
            for (; j + kPer * kU <= HN; j += kPer * kU) {
                uint4 v[kU > 1 ? kU : 1];
#pragma unroll
                for (int u = 0; u < kU; ++u)
                    v[u] = *(const uint4*)(wr + j + (long)u * kPer + lane * 8);
#pragma unroll
                for (int u = 0; u < kU; ++u) {
                    *(uint4*)(stage + lane * 8) = v[u];
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        const int jj = lane + t * kWave;
                        acc += sv_at(sv, j + (long)u * kPer + jj) * bf2f(stage[jj]);
                    }
                }
            }
        }
        for (; j + kPer <= HN; j += kPer) {
            *(uint4*)(stage + lane * 8) = *(const uint4*)(wr + j + lane * 8);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int jj = lane + t * kWave;
                acc += sv_at(sv, j + jj) * bf2f(stage[jj]);
            }
        }
    }
    for (long i = j + lane; i < HN; i += kWave) acc += sv_at(sv, i) * bf2f(wr[i]);
    return acc;
}

/* Swizzled form: N is held in LDS transposed within each 256-element chunk, and the weight is
 * transposed by the load.
 *
 * wave_dot_lds reads N at strided addresses (lane L takes L, L+32, ...). A strided 16-bit LDS read
 * puts 32 lanes on 16 banks, two cycles for half a cycle's bytes; with the weight's matching reads
 * the inner loop is 228 instructions for 32 lane-FMAs, and the kernel is issue-bound (M=1 at
 * 6.55 MB and M=4 at 26.2 MB both take 18.7 us).
 *
 * N layout: within a 256-element chunk, logical p is stored at physical (p % 32) * 8 + (p / 32),
 * so lane L's eight elements are one conflict-free ds_load_b128. Requires n % 256 == 0 so a chunk
 * never straddles a sub-stream; otherwise the plain layout and loop are used (the `swz` flag).
 * Whoever writes N pays: a plain staged copy transposed in LDS would be an 8-way conflict, so
 * staging and rescale are one pass in which a thread reads its eight `h` and `w` from global and
 * writes one ds_store_b128. The sum of squares then reads `h` from global (20 KiB a block against
 * 320 KiB of W_down). Its order is unchanged: thread tid accumulates i = tid, tid + 256, ...
 *
 * Weight: `global_load_tr_b128` (r4d_common.h) gives lane j of each 8-lane group element j from
 * all eight lanes' addresses. With L = 8g + i naming `wr + j + 8g + 32i`, lane L receives elements
 * L, L+32, ... L+224 in ascending order, all 16-byte aligned (`swz` requires `down` aligned). This
 * removes the staging store and eight conflicting ds_load_u16 per chunk; no `stage` slot is used.
 *
 * Bit-identical: lane L multiplies the same elements in the same order; only addresses move. */
template <int kU>
__device__ __forceinline__ float wave_dot_swz(const unsigned short* __restrict__ sv,
                                              const unsigned short* __restrict__ wr, long HN,
                                              int lane) {
    constexpr long kPer = kWave * 8;
    const long woff = 8 * (long)(lane / 8) + 32 * (long)(lane % 8);
    float acc = 0.0f;
    long  j = 0;
    for (; j + kPer * kU <= HN; j += kPer * kU) {
        uint4 v[kU];
#pragma unroll
        for (int u = 0; u < kU; ++u)
            v[u] = __builtin_bit_cast(uint4, load_tr_b128(wr + j + (long)u * kPer + woff));
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const uint4 g = *(const uint4*)(sv + j + (long)u * kPer + lane * 8);
            const unsigned short* gp = (const unsigned short*)&g;
            const unsigned short* wp = (const unsigned short*)&v[u];
#pragma unroll
            for (int t = 0; t < 8; ++t) acc += bf2f(gp[t]) * bf2f(wp[t]);
        }
    }
    for (; j + kPer <= HN; j += kPer) {
        const uint4 g = *(const uint4*)(sv + j + lane * 8);
        const unsigned short* gp = (const unsigned short*)&g;
        const uint4 w = __builtin_bit_cast(uint4, load_tr_b128(wr + j + woff));
        const unsigned short* wp = (const unsigned short*)&w;
#pragma unroll
        for (int t = 0; t < 8; ++t) acc += bf2f(gp[t]) * bf2f(wp[t]);
    }
    return acc;
}

/* ============================================================== mtp_enter
 *
 * The multi-token-prediction head's entry. It lives here because it has hc_read's shape: a wide
 * stream, a grouped norm over it, and a matvec whose weight dominates traffic.
 *
 *     ne[j]    = e[j] * rsqrt(mean(e^2) + eps) * (w_e[j] + wadd)
 *     fe[i]    = fc_e[i] . ne
 *     nh[k][j] = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)          rs[k] over sub-stream k
 *     x[k*n+i] = fc_h[i] . nh[k] + fe[i]                         for each of the hc sub-streams
 *
 * fc_h[i] is read once and used hc times. The two square matrices are 13.1 MB here and the
 * arithmetic is negligible, so a workgroup owns kMtpOutTB output channels, stages `nh` and `ne` in
 * LDS, and walks each weight row once with hc accumulators live (hc separate dots would read fc_h
 * hc times).
 *
 * Every workgroup recomputes the norms: 25 KB of `h` and `e` against its 327 KB of weights,
 * cheaper than a second launch and a scratch buffer for five scalars. */
constexpr int kMtpOutTB = 32;    // output channels a workgroup owns: 80 blocks at n = 2560
/* From this many rows the tiled form is used: at one row the per-row kernel reads the weight once
 * and is the floor; from two rows the tiled form is never worse per step and up to 2.7x faster per
 * call. */
constexpr long kMtpTileMinRows = 2;
/* `nh` + `ne` in LDS, in elements: 64 KiB less the staging slots and reduction planes. */
constexpr long kMtpLdsMaxN = 28672;

/* One pass over a weight row with hc accumulators live, staged through LDS as in wave_dot_lds to
 * keep the summation order (lane L sums j = L, L+32, ...). */
template <int kU>
__device__ __forceinline__ void wave_dot_hc(const unsigned short* __restrict__ sv, long svstride,
                                            int hc, const unsigned short* __restrict__ wr, long n,
                                            int lane, unsigned short* stage, float* acc) {
    constexpr long kPer = kWave * 8;
    long j = 0;
    if (((uintptr_t)wr & 15u) == 0u && (n % 8) == 0) {
        if (kU > 1) {
            for (; j + kPer * kU <= n; j += kPer * kU) {
                uint4 v[kU > 1 ? kU : 1];
#pragma unroll
                for (int u = 0; u < kU; ++u)
                    v[u] = *(const uint4*)(wr + j + (long)u * kPer + lane * 8);
#pragma unroll
                for (int u = 0; u < kU; ++u) {
                    *(uint4*)(stage + lane * 8) = v[u];
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        const int  jj = lane + t * kWave;
                        const long jg = j + (long)u * kPer + jj;
                        const float wv = bf2f(stage[jj]);
                        for (int c = 0; c < hc; ++c)
                            acc[c] += bf2f(sv[(long)c * svstride + jg]) * wv;
                    }
                }
            }
        }
        for (; j + kPer <= n; j += kPer) {
            *(uint4*)(stage + lane * 8) = *(const uint4*)(wr + j + lane * 8);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int  jj = lane + t * kWave;
                const long jg = j + jj;
                const float wv = bf2f(stage[jj]);
                for (int c = 0; c < hc; ++c)
                    acc[c] += bf2f(sv[(long)c * svstride + jg]) * wv;
            }
        }
    }
    for (long i = j + lane; i < n; i += kWave) {
        const float wv = bf2f(wr[i]);
        for (int c = 0; c < hc; ++c) acc[c] += bf2f(sv[(long)c * svstride + i]) * wv;
    }
}

/* Both weight rows of one output channel in one pass, over `nh` and `ne` held transposed per
 * 256-element chunk as in wave_dot_swz: a lane's eight elements are one ds_load_b128 per stream,
 * and each weight row arrives transposed by global_load_tr_b128. The fc_h row feeds the hc stream
 * accumulators, the fc_e row feeds `ae`; all 2*kU loads of an iteration issue before the first is
 * read.
 *
 * The plain-layout pair (wave_dot_hc, wave_dot_lds) uses conflicting strided 16-bit LDS reads and
 * an accumulator array indexed by runtime `hc` (private memory). Here a chunk is hc + 1
 * conflict-free loads and the accumulators are unrolled over kHcMax with a guard, so they stay in
 * registers. Bit-identical to that pair: each accumulator's summation order is unchanged. */
template <int kU>
__device__ __forceinline__ void wave_dot_mtp_swz(const unsigned short* __restrict__ snh,
                                                 const unsigned short* __restrict__ sne,
                                                 const unsigned short* __restrict__ wh_row,
                                                 const unsigned short* __restrict__ we_row,
                                                 long n, int hc, int lane, float* acc, float& ae) {
    constexpr long kPer = kWave * 8;
    const long woff = 8 * (long)(lane / 8) + 32 * (long)(lane % 8);
    auto chunk = [&](long jc, const uint4& vh, const uint4& ve) {
        const unsigned short* wp = (const unsigned short*)&vh;
#pragma unroll
        for (int c = 0; c < kHcMax; ++c) {
            if (c < hc) {
                const uint4 g = *(const uint4*)(snh + (long)c * n + jc);
                const unsigned short* gp = (const unsigned short*)&g;
#pragma unroll
                for (int t = 0; t < 8; ++t) acc[c] += bf2f(gp[t]) * bf2f(wp[t]);
            }
        }
        const uint4 g = *(const uint4*)(sne + jc);
        const unsigned short* gp = (const unsigned short*)&g;
        const unsigned short* ep = (const unsigned short*)&ve;
#pragma unroll
        for (int t = 0; t < 8; ++t) ae += bf2f(gp[t]) * bf2f(ep[t]);
    };
    long j = 0;
    for (; j + kPer * kU <= n; j += kPer * kU) {
        uint4 vh[kU], ve[kU];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            vh[u] = __builtin_bit_cast(uint4, load_tr_b128(wh_row + j + (long)u * kPer + woff));
            ve[u] = __builtin_bit_cast(uint4, load_tr_b128(we_row + j + (long)u * kPer + woff));
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) chunk(j + (long)u * kPer + lane * 8, vh[u], ve[u]);
    }
    for (; j + kPer <= n; j += kPer) {
        const uint4 vh = __builtin_bit_cast(uint4, load_tr_b128(wh_row + j + woff));
        const uint4 ve = __builtin_bit_cast(uint4, load_tr_b128(we_row + j + woff));
        chunk(j + lane * 8, vh, ve);
    }
}

__global__ __launch_bounds__(kThreads)
void mtp_enter_kernel(const unsigned short* __restrict__ h, const unsigned short* __restrict__ e,
                      const float* __restrict__ wh, const float* __restrict__ we,
                      const unsigned short* __restrict__ fch,
                      const unsigned short* __restrict__ fce,
                      unsigned short* __restrict__ x,
                      long rows, long n, int hc, int flat, float eps, float wadd,
                      long hrow, long erow, long xrow) {
    extern __shared__ unsigned short sM[];        /* [hc * n] nh, then [n] ne */
    __shared__ float s_part[kWaves][kHcMax];
    __shared__ float s_pe[kWaves];
    __shared__ float s_scale[kHcMax];
    __shared__ float s_rse;
    __shared__ unsigned short s_stage[kWaves][kWave * 8];

    const long r = blockIdx.y;
    if (r >= rows) return;
    const long HN = (long)hc * n;
    unsigned short* s_nh = sM;
    unsigned short* s_ne = sM + HN;

    const unsigned short* hp = h + r * hrow;
    const unsigned short* ep = e + r * erow;
    unsigned short*       xp = x + r * xrow;

    const int tid  = (int)threadIdx.x;
    const int wave = tid / kWave;
    const int lane = tid % kWave;

    /* ---- sums of squares: one partial per sub-stream plus the embedding's, in one pass. The
     * `flat` norm adds the same partials together. */
    float part[kHcMax];
#pragma unroll
    for (int c = 0; c < kHcMax; ++c) part[c] = 0.0f;
    float pe = 0.0f;
    /* kSsIter iterations' loads issue before the first is summed, so a thread's ten iterations
     * wait out two load latencies instead of ten. Same elements, same order. */
    constexpr int kSsIter = 5;
    long i = tid;
    for (; i + (long)(kSsIter - 1) * kThreads < n; i += (long)kSsIter * kThreads) {
        uint32_t ev[kSsIter], hv[kSsIter][kHcMax];
#pragma unroll
        for (int q = 0; q < kSsIter; ++q) {
            const long iq = i + (long)q * kThreads;
            ev[q] = bfw(ep + iq);
#pragma unroll
            for (int c = 0; c < kHcMax; ++c)
                if (c < hc) hv[q][c] = bfw(hp + (long)c * n + iq);
        }
#pragma unroll
        for (int q = 0; q < kSsIter; ++q) {
            const float evf = bfw_f(ev[q]);
            pe += evf * evf;
#pragma unroll
            for (int c = 0; c < kHcMax; ++c)
                if (c < hc) { const float v = bfw_f(hv[q][c]); part[c] += v * v; }
        }
    }
    for (; i < n; i += kThreads) {
        const float ev = bf2f(ep[i]);
        pe += ev * ev;
        for (int c = 0; c < hc; ++c) {
            const float v = bf2f(hp[(long)c * n + i]);
            part[c] += v * v;
        }
    }
    for (int c = 0; c < hc; ++c) part[c] = wave_sum(part[c]);
    pe = wave_sum(pe);
    if (lane == 0) {
        for (int c = 0; c < hc; ++c) s_part[wave][c] = part[c];
        s_pe[wave] = pe;
    }
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int c = 0; c < hc; ++c) {
            float v = 0.0f;
            for (int w = 0; w < kWaves; ++w) v += s_part[w][c];
            s_scale[c] = v;
            tot += v;
        }
        if (flat) {
            const float sc = 1.0f / sqrtf(tot / (float)HN + eps);
            for (int c = 0; c < hc; ++c) s_scale[c] = sc;
        } else {
            for (int c = 0; c < hc; ++c)
                s_scale[c] = 1.0f / sqrtf(s_scale[c] / (float)n + eps);
        }
        float ve = 0.0f;
        for (int w = 0; w < kWaves; ++w) ve += s_pe[w];
        s_rse = 1.0f / sqrtf(ve / (float)n + eps);
    }
    __syncthreads();

    /* Transposed layout when the shape allows (as in hc_read_a): every 256-element chunk inside
     * one stream and every weight row 16-byte aligned. Otherwise the plain layout; both write the
     * same values. */
    constexpr long kPer = kWave * 8;
    const bool swz = (n % kPer == 0) && (((uintptr_t)fch & 15u) == 0u)
                  && (((uintptr_t)fce & 15u) == 0u);

    /* ---- the normed vectors, staged bf16, so one pass over each weight row suffices. */
    {
        const float rse = s_rse;
        if (swz) {
            /* A thread owns one (chunk, lane) group: logical elements lane + 32t of its chunk,
             * written as one 16-byte store at physical lane * 8 (see wave_dot_swz). */
            const int lane8 = tid & (kWave - 1);
            for (int c = -1; c < hc; ++c) {
                const unsigned short* src = c < 0 ? ep : hp + (long)c * n;
                const float* wv = c < 0 ? we : wh + (long)c * n;
                const float sc  = c < 0 ? rse : s_scale[c];
                unsigned short* dst = c < 0 ? s_ne : s_nh + (long)c * n;
                for (long base = 0; base < n; base += (long)kThreads * 8) {
                    const long q = base + (long)tid * 8;
                    if (q >= n) break;
                    const long ch = q & ~(kPer - 1);
                    float out[8];
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        const long li = ch + lane8 + (long)t * kWave;
                        out[t] = bfw_f(bfw(src + li)) * sc * (wv[li] + wadd);
                    }
                    *(uint4*)(dst + q) = f2bf8(out);
                }
            }
        } else {
            for (long i = tid; i < n; i += kThreads)
                s_ne[i] = f2bf(bf2f(ep[i]) * rse * (we[i] + wadd));
            for (int c = 0; c < hc; ++c) {
                const float sc  = s_scale[c];
                const long  base = (long)c * n;
                for (long i = tid; i < n; i += kThreads)
                    s_nh[base + i] = f2bf(bf2f(hp[base + i]) * sc * (wh[base + i] + wadd));
            }
        }
    }
    __syncthreads();

    /* ---- this workgroup's output channels: one weight row read, hc + 1 dots from it. */
    const long i0 = (long)blockIdx.x * kMtpOutTB;
    for (int t = wave; t < kMtpOutTB; t += kWaves) {
        const long i = i0 + t;
        if (i >= n) break;
        float acc[kHcMax];
#pragma unroll
        for (int c = 0; c < kHcMax; ++c) acc[c] = 0.0f;
        float ae = 0.0f;
        if (swz) {
            wave_dot_mtp_swz<kDotU>(s_nh, s_ne, fch + i * n, fce + i * n, n, hc, lane, acc, ae);
        } else {
            wave_dot_hc<kDotU>(s_nh, n, hc, fch + i * n, n, lane, s_stage[wave], acc);
            ae = wave_dot_lds<kDotU>(s_ne, fce + i * n, n, lane, s_stage[wave]);
        }
        ae = wave_sum(ae);
        for (int c = 0; c < hc; ++c) {
            const float v = wave_sum(acc[c]);
            if (lane == 0) xp[(long)c * n + i] = f2bf(v + ae);
        }
    }
}

/* Kernel A's static LDS beside N (reduction planes and staging slots). N's dynamic allocation must
 * fit in the rest of 64 KiB, so the launcher bounds hc * n with this; the kernel asserts it. */
constexpr long kHcReadAStaticLds = (long)(kWaves * kHcMax + kHcMax) * (long)sizeof(float)
                                 + (long)kWaves * kWave * 8 * (long)sizeof(unsigned short);

/* ============================================================== hc_read, kernel A
 *
 * The grouped norm, then `d = silu(W_down @ N / hc)` for this workgroup's `ta` columns. One
 * workgroup also produces the hc inject gains and publishes the per-stream norm scales for kernel
 * B, which reads only its own 128 channels of `h` and so cannot recompute them.
 *
 * LDS holds N as bf16 (HN * 2 bytes, 20 KiB here), so the columns share one pass over `h` and one
 * grouped reduction. It is transposed per 256-element chunk; see wave_dot_swz. */
/* Waves per SIMD: nine for the E4M3 dot, as tuned. Unconstrained, LLVM 24 builds it at 120 VGPRs
 * for twelve waves, 11-20% slower at the 4-7 rows of a speculative decode step. The bf16 form keeps
 * the full range; pinning it would change its code.
 *
 * HCP and PER, when nonzero, are hc and n / kThreads known at compile time (4 and 10 at
 * Flash-Next's shape). Thread t's elements of `h` are then t + e * kThreads for e < HCP * PER, the
 * stream being e / PER, so every one of them and of the norm weight can be loaded before the first
 * is used: one round trip from the cache instead of one per stream for the sums of squares and
 * again for N. Those ten-odd dependent latencies, not bytes, set the kernel's time at decode. The
 * sums and N take the same values in the same order, so the output is bit-identical to the loops
 * below. */
template <bool F8, int HCP = 0, int PER = 0>
__device__ __forceinline__ void hc_read_a_body(const unsigned short* __restrict__ h, const float* __restrict__ w,
                      const void* __restrict__ downv, long dld,
                      const unsigned short* __restrict__ injw,
                      unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
                      float* __restrict__ scaleout,
                      long rows, long n, int hc, long lr, long ta, long hrow, long irow,
                      float eps, float wadd, unsigned bx, unsigned by,
                      unsigned xpub) {
    /* 16-byte aligned, for the 16-byte LDS reads of the dots. The dynamic region starts
     * after the kernel's static LDS, which in hc_read_fused_kernel is not a multiple of 16; a
     * misaligned base makes the fused form's A blocks three times slower at four rows. */
    extern __shared__ __attribute__((aligned(16))) unsigned short sN[];   /* [hc * n], transposed per 256-element chunk */
    __shared__ float s_part[kWaves][kHcMax];
    __shared__ float s_scale[kHcMax];
    /* One 256-entry staging slot per wave (4 KiB); no barrier, a wave uses only its own. */
    __shared__ unsigned short s_stage[kWaves][kWave * 8];
    static_assert(sizeof(s_part) + sizeof(s_scale) + sizeof(s_stage) == kHcReadAStaticLds,
                  "the launcher sizes N's allocation against this kernel's static LDS");

    constexpr long kPer = kWave * 8;

    const long r  = by;
    if (r >= rows) return;
    const long HN = (long)hc * n;
    /* The inject gains get a block of their own for load balance. With ta = kWaves every wave does
     * one dot over HN; adding the gains to a column-owning block would give its waves a second
     * one, and that slowest block would set the kernel's duration. So the grid is one wider and
     * the last block owns no columns. */
    const bool pub = (bx == xpub);
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const unsigned short* hp = h + r * hrow;
    const unsigned short* __restrict__ down = (const unsigned short*)downv;

    /* The transposed layout follows from the shape: the chunk must sit inside one sub-stream
     * (n % 256) and every 16-byte group be aligned in LDS and in `h`; otherwise the plain layout,
     * with the same N. The E4M3 dot reads N in natural order, so only the bf16 form transposes. */
    const bool swz = !F8 && (n % kPer == 0) && (((uintptr_t)hp   & 15u) == 0u)
                  && (((uintptr_t)down & 15u) == 0u)
                  && (!injw || ((uintptr_t)injw & 15u) == 0u);

    /* ---- sums of squares, one stream at a time, straight from `h`. One stream at a time keeps
     * the accumulator a scalar; an array indexed by runtime `hc` would live in scratch.
     *
     * This prologue is latency-bound, not traffic-bound: 20 KiB against the block's 320 KiB of
     * W_down, yet 6.7 of the kernel's 18.7 us at the decode grid, because ten dependent iterations
     * wait on loads with nothing else resident. Issuing kPreSs iterations' loads first turns eight
     * serial latencies into one, bit-identically. Wider loads (these bytes are served from the
     * 64 MB last-level cache, not DRAM), reading `h` once instead of twice, and deeper prefetch
     * were all measured as no gain. */
    if constexpr (HCP > 0) {
        /* The preloaded form (see the kernel's note): N stays the natural-order E4M3 layout. */
        static_assert(F8, "only the E4M3 dot reads N in natural order");
        constexpr int kE = HCP * PER;
        uint32_t hv[kE];
        float    wv[kE];
#pragma unroll
        for (int e = 0; e < kE; ++e) {
            hv[e] = bfw(hp + tid + (long)e * kThreads);
            wv[e] = w[tid + (long)e * kThreads];
        }
#pragma unroll
        for (int c = 0; c < HCP; ++c) {
            float acc = 0.0f;
#pragma unroll
            for (int p = 0; p < PER; ++p) { const float v = bfw_f(hv[c * PER + p]); acc += v * v; }
            acc = wave_sum(acc);
            if (lane == 0) s_part[wave][c] = acc;
        }
        __syncthreads();
        if (tid < HCP) {
            float ss = 0.0f;
            for (int k = 0; k < kWaves; ++k) ss += s_part[k][tid];
            s_scale[tid] = __frsqrt_rn(ss / (float)n + eps);
            if (pub && scaleout) scaleout[r * kHcMax + tid] = s_scale[tid];
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < HCP; ++c) {
            const float sc = s_scale[c];
#pragma unroll
            for (int p = 0; p < PER; ++p) {
                const int e = c * PER + p;
                sN[tid + (long)e * kThreads] = f2bf(bfw_f(hv[e]) * sc * (wv[e] + wadd));
            }
        }
        __syncthreads();
    } else {
    constexpr int kPreSs = 8;      // n / kThreads is 10 a stream at this shape, and 12 spills
    for (int c = 0; c < hc; ++c) {
        const long cb = (long)c * n;
        float acc = 0.0f;
        long  i = tid;
        for (; i + (long)(kPreSs - 1) * kThreads < n; i += (long)kPreSs * kThreads) {
            uint32_t hv[kPreSs];
#pragma unroll
            for (int p = 0; p < kPreSs; ++p) hv[p] = bfw(hp + cb + i + (long)p * kThreads);
#pragma unroll
            for (int p = 0; p < kPreSs; ++p) { const float v = bfw_f(hv[p]); acc += v * v; }
        }
        for (; i < n; i += kThreads) { const float v = bf2f(hp[cb + i]); acc += v * v; }
        acc = wave_sum(acc);
        if (lane == 0) s_part[wave][c] = acc;
    }
    __syncthreads();
    if (tid < hc) {
        float ss = 0.0f;
        for (int k = 0; k < kWaves; ++k) ss += s_part[k][tid];
        s_scale[tid] = __frsqrt_rn(ss / (float)n + eps);
        /* One workgroup publishes; the others compute the identical value. */
        if (pub && scaleout) scaleout[r * kHcMax + tid] = s_scale[tid];
    }
    __syncthreads();

    /* ---- N into LDS, narrowed to bf16 where the reference narrows it. Looped per stream rather
     * than computing `j / n`, a 64-bit integer division per element (a software sequence here). */
    for (int c = 0; c < hc; ++c) {
        const long  cb = (long)c * n;
        const float sc = s_scale[c];
        if (swz) {
            /* A thread owns one (chunk, lane) group: logical elements `lane + 32t`, one 16-byte
             * store at physical `q`. A wave covers one chunk, so `ch` is wave-uniform and the
             * `h` and `w` reads stay coalesced. */
            for (long base = 0; base < n; base += (long)kThreads * 8) {
                const long q = base + (long)tid * 8;
                if (q >= n) break;
                const long ch = cb + (q & ~(kPer - 1));
                float out[8];
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const long li = ch + lane + (long)t * kWave;
                    out[t] = bfw_f(bfw(hp + li)) * sc * (w[li] + wadd);
                }
                *(uint4*)(sN + cb + q) = f2bf8(out);
            }
        } else {
            constexpr int kPreScale = 8;
            long i = tid;
            for (; i + (long)(kPreScale - 1) * kThreads < n; i += (long)kPreScale * kThreads) {
                float          wv[kPreScale];
                uint32_t       hv[kPreScale];
#pragma unroll
                for (int p = 0; p < kPreScale; ++p) {
                    wv[p] = w[cb + i + (long)p * kThreads];
                    hv[p] = bfw(hp + cb + i + (long)p * kThreads);
                }
#pragma unroll
                for (int p = 0; p < kPreScale; ++p)
                    sN[cb + i + (long)p * kThreads] = f2bf(bfw_f(hv[p]) * sc * (wv[p] + wadd));
            }
            for (; i < n; i += kThreads)
                sN[cb + i] = f2bf(bf2f(hp[cb + i]) * sc * (w[cb + i] + wadd));
        }
    }
    __syncthreads();
    }

    /* ---- this workgroup's columns of W_down, one wave per column. */
    const float inv_hc = 1.0f / (float)hc;
    const long  k0 = (long)bx * ta;
    for (long k = k0 + wave; k < k0 + ta && k < lr; k += kWaves) {
        float acc;
        if constexpr (F8) {
            /* This wave's next column too, when it has one (hc8_dot_down2). */
            const long k2 = k + kWaves;
            if (k2 < k0 + ta && k2 < lr) {
                float acc2;
                hc8_dot_down2<kDotU>(sN, (const unsigned char*)downv + k * dld,
                                     (const unsigned char*)downv + k2 * dld, HN, lane, acc, acc2);
                acc  = wave_sum(acc);
                acc2 = wave_sum(acc2);
                if (lane == 0) {
                    dout[r * lr + k]  = f2bf(siluf_(acc * inv_hc));
                    dout[r * lr + k2] = f2bf(siluf_(acc2 * inv_hc));
                }
                k = k2;
                continue;
            }
            acc = hc8_dot_down<kDotU>(sN, (const unsigned char*)downv + k * dld, HN, lane);
        } else {
            const unsigned short* wr = down + k * HN;
            acc = swz ? wave_dot_swz<kDotU>(sN, wr, HN, lane)
                      : wave_dot_lds<kDotU>(sN, wr, HN, lane, s_stage[wave]);
        }
        acc = wave_sum(acc);
        if (lane == 0) dout[r * lr + k] = f2bf(siluf_(acc * inv_hc));
    }

    /* ---- the inject gains, on the block that owns no columns and already holds N. */
    if (pub && injw && inj) {
        for (int c = wave; c < hc; c += kWaves) {
            const unsigned short* wr = injw + (long)c * HN;
            float acc = swz ? wave_dot_swz<kDotU>(sN, wr, HN, lane)
                            : wave_dot_lds<kDotU>(sN, wr, HN, lane, s_stage[wave]);
            acc = wave_sum(acc);
            if (lane == 0) inj[r * irow + c] = f2bf(2.0f * sigmoidf_(acc * inv_hc));
        }
    }
}

template <bool F8, int HCP = 0, int PER = 0>
__global__ __launch_bounds__(kThreads)
__attribute__((amdgpu_waves_per_eu(F8 ? 9 : 1, F8 ? 9 : 16)))
void hc_read_a_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                      const void* __restrict__ downv, long dld,
                      const unsigned short* __restrict__ injw,
                      unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
                      float* __restrict__ scaleout,
                      long rows, long n, int hc, long lr, long ta, long hrow, long irow,
                      float eps, float wadd) {
    hc_read_a_body<F8, HCP, PER>(h, w, downv, dld, injw, dout, inj, scaleout, rows, n, hc, lr, ta,
                                 hrow, irow, eps, wadd, blockIdx.x, blockIdx.y, gridDim.x - 1u);
}

/* ============================================================== hc_read, kernel B1
 *
 * `u = sigmoid(W_up @ d)`, one wave per row of W_up. Separate from the mix for occupancy: the mix
 * owns 128 channels (hc*128 rows of W_up), only twenty workgroups at width 2560 on 64 CUs, while
 * this kernel runs HN/64 = 160 workgroups on the same bytes. `u` costs a 20 KiB-per-row round trip
 * through scratch against 6.55 MiB of weight.
 *
 * RG rows share one block and one pass over the weight. With the row on blockIdx.y, a depth-3 MTP
 * verify (four rows) streams W_up four times; the last-level cache absorbs most of it (1.4x, not
 * 4x), but needlessly. `d` is 320 floats per row, so RG rows fit in LDS and a wave takes RG dots
 * from each weight row it loads (~3% of a multi-row step, same output). RG=1 is the one-row case.
 *
 * At the decode grid this moves 6.55 MB in 14.9 us (440 GB/s; the part reads 621 cold). Measured
 * slower: four rows in flight per wave (+16%); kUpRows 16 or 128; staging the 64-element tail as
 * one wide load (+4.6%; both narrow loads hit one 128-byte line); eight consecutive rows per wave
 * instead of strided (+9.9%; strided rows make a block's eight waves read one contiguous burst);
 * and register-staging the tail (+13.1%; the compiler hoists the wait in front of the chunk load).
 * A block-cooperative stage of the span would take 40 KiB of LDS (1 block per CU instead of 7).
 *
 * Bit-identical to wave_dot_lds: only the loop nest changes (rows move innermost), and `acc[]` is
 * indexed by an unrolled constant so it stays in registers; a runtime `rows` bound would put it in
 * scratch. */

/* RG rows of the LDS vector against one contiguous bf16 weight row, read once. `sv_ld` is the row
 * pitch of `sv`. */
template <int RG>
__device__ __forceinline__ void wave_dot_lds_rows(const float* __restrict__ sv, long sv_ld,
                                                  const unsigned short* __restrict__ wr, long HN,
                                                  int lane, unsigned short* stage,
                                                  float* __restrict__ acc) {
    constexpr long kPer = kWave * 8;
#pragma unroll
    for (int r = 0; r < RG; ++r) acc[r] = 0.0f;
    long j = 0;
    if (((uintptr_t)wr & 15u) == 0u) {
        for (; j + kPer <= HN; j += kPer) {
            *(uint4*)(stage + lane * 8) = *(const uint4*)(wr + j + lane * 8);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int   jj = lane + t * kWave;
                const float wv = bf2f(stage[jj]);
#pragma unroll
                for (int r = 0; r < RG; ++r) acc[r] += sv_at(sv, (long)r * sv_ld + j + jj) * wv;
            }
        }
    }
    for (long i = j + lane; i < HN; i += kWave) {
        const float wv = bf2f(wr[i]);
#pragma unroll
        for (int r = 0; r < RG; ++r) acc[r] += sv_at(sv, (long)r * sv_ld + i) * wv;
    }
}

template <int RG, bool F8>
__global__ __launch_bounds__(kThreads)
void hc_up_kernel(const void* __restrict__ upv, long uld, const unsigned short* __restrict__ dbuf,
                  unsigned short* __restrict__ ubuf, long rows, long hn, long lr) {
    const unsigned short* __restrict__ up = (const unsigned short*)upv;
    extern __shared__ __attribute__((aligned(16))) float sd[];   /* [RG * lr] */
    __shared__ unsigned short s_stage[kWaves][kWave * 8];   /* wave_dot_lds, one slot a wave */
    const long r0 = (long)blockIdx.y * RG;
    if (r0 >= rows) return;
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;

    /* A dead row of the tail group is zero-filled, so its dot is defined; its store is dropped. */
    for (long k = tid; k < (long)RG * lr; k += kThreads) {
        const long r = k / lr;
        sd[k] = (r0 + r < rows) ? bf2f(dbuf[(r0 + r) * lr + (k - r * lr)]) : 0.0f;
    }
    __syncthreads();

    const long j0 = (long)blockIdx.x * kUpRows;
    /* The E4M3 form takes a row four threads at a time, a quarter (one scale group) each, so 256
     * threads cover the block's 64 rows in one pass. */
    if constexpr (F8) {
        static_assert(kUpRows * 4 == kThreads, "four threads a row of W_up");
        const int  tr = tid >> 2, q4 = tid & 3;
        const long j  = j0 + tr;
        float acc[RG];
        hc8_dot_up<RG>(sd, lr, (const unsigned char*)upv + (j < hn ? j : hn - 1) * uld, q4, acc);
#pragma unroll
        for (int r = 0; r < RG; ++r)
            if (q4 == 0 && j < hn && r0 + r < rows) ubuf[(r0 + r) * hn + j] = f2bf(sigmoidf_(acc[r]));
        return;
    }
    for (long t = wave; t < kUpRows; t += kWaves) {
        const long j = j0 + t;
        if (j >= hn) break;
        float acc[RG];
        wave_dot_lds_rows<RG>(sd, lr, up + j * lr, lr, lane, s_stage[wave], acc);
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            const float a = wave_sum(acc[r]);
            /* Narrowed as in the reference, where the sigmoid output is bf16 before the multiply. */
            if (lane == 0 && r0 + r < rows) ubuf[(r0 + r) * hn + j] = f2bf(sigmoidf_(a));
        }
    }
}

/* Rotated codes of one 128-channel group, for a consumer whose weight carries a 128-wide Hadamard
 * (the rotated expert planes). A wave holds the group four consecutive values per lane (the
 * r4d_fwht_wave<128> layout), loaded by the caller from the stored bf16, and quantises exactly as
 * the rotated quant_act_fp8 kernel does (transform, absmax, scale, clamp, packing), so no separate
 * launch over `x` is needed. I8: had_quant_act_i8g's codes instead (scale amax / 127, round to
 * nearest-even into -127..127), which a read whose plain codes are int8 writes: the rotated pair
 * takes the plain pair's dtype. */
template <bool I8>
__device__ __forceinline__ void hc_rot_group(float* v, int lane, unsigned char* qg, float* sg) {
    r4d_fwht_wave<kUpTB>(v, lane);
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
    m = r4d_wave_max(m);
    constexpr float kTop = I8 ? 127.0f : 448.0f;
    const float sc = m > 0.0f ? m * (1.0f / kTop) : 1.0f;
    if (lane == 0) *sg = sc;
    const float inv = 1.0f / sc;
    if constexpr (I8) {
        unsigned b = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            b |= ((unsigned)(int)rintf(fminf(fmaxf(v[j] * inv, -127.0f), 127.0f)) & 0xFFu) << (8 * j);
        *(unsigned*)(qg + lane * 4) = b;
        return;
    }
    int pk = 0;
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[0] * inv, -448.0f), 448.0f),
                                         fminf(fmaxf(v[1] * inv, -448.0f), 448.0f), pk, false);
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[2] * inv, -448.0f), 448.0f),
                                         fminf(fmaxf(v[3] * inv, -448.0f), 448.0f), pk, true);
    *(unsigned*)(qg + lane * 4) = (unsigned)pk;
}

/* The fused form's wait (hc_read_fused_kernel, below): its B blocks hold here until every A
 * block has published `d` and the scales. */
constexpr unsigned kHcSyncCtr = 3;                  /* tickets, A blocks done, B blocks past the wait */

__device__ __forceinline__ void hc_role_wait(unsigned int* __restrict__ sync, unsigned na,
                                             unsigned nb_all) {
    if (threadIdx.x == 0) {
        const unsigned long t0 = wall_clock64();
        /* Relaxed while polling, acquired once after: an acquire load also invalidates the
         * vector caches, and a poll of them would keep emptying them under the A blocks. */
        while (__hip_atomic_load(&sync[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < na) {
            __builtin_amdgcn_s_sleep(1);
            /* 0.5 s at the 100 MHz wall clock. A blocks that never finish are a counter fault:
             * trap rather than read a stale `d`, as the all-reduce handshakes do. */
            if (wall_clock64() - t0 > 50000000ull) __builtin_trap();
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        if (__hip_atomic_fetch_add(&sync[2], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u
            == nb_all) {
            __hip_atomic_store(&sync[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&sync[1], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&sync[2], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
    __syncthreads();
    /* Every wave, not only thread 0's: a workgroup's waves can sit on both CUs of a WGP, each
     * with its own vector cache, and `d` must be read past it. */
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

/* ============================================================== hc_read, kernels B and B2 as one
 *
 * hc_up tiles the flattened index, so a channel's hc terms land in different workgroups and `u`
 * round-trips through `ubuf` into hc_mix, a dependent launch that costs more than either kernel's
 * work at one row.
 *
 * This workgroup instead owns 64 / hc channels in all hc sub-streams (the same 64 rows of W_up and
 * grid, as hc runs of 64 / hc rows), so `u` stays in LDS. An arrival counter replaces the whole
 * 128-channel scale group, as in hc_up_mix_tiled: the last workgroup covering a group of a row
 * group quantises it from the bf16 `x` the others wrote.
 *
 * Bit-identical to the pair: the same per-row dots and narrowing as hc_up, the mean over c
 * ascending as in hc_mix, and codes from the same bf16 values and amax. */
/* RPT > 1 (E4M3 only) gives the workgroup RPT * 64 rows of W_up, RPT a thread with one
 * quarter each (hc8_up_dot_rows), and 1 / RPT the grid (kUpRpt). WAIT is the fused form's
 * (hc_read_fused_kernel): `d` and kernel A's scales are read only after hc_role_wait. */
template <int RG, bool F8, bool I8 = false, int RPT = 1, bool WAIT = false>
__device__ __forceinline__ void hc_up_mix_body(const void* __restrict__ upv, long uld,
                      const unsigned short* __restrict__ dbuf,
                      const unsigned short* __restrict__ h, const float* __restrict__ w,
                      const float* __restrict__ scalein,
                      unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                      float* __restrict__ qs, unsigned char* __restrict__ rq,
                      float* __restrict__ rqs, unsigned int* __restrict__ ctr,
                      long rows, long n, int hc, long lr, long hrow, long xrow,
                      long qrow, long srow, long rqrow, long rsrow, float wadd, unsigned bx, unsigned by,
                      unsigned int* __restrict__ sync = nullptr, unsigned na = 0, unsigned nb_all = 0) {
    static_assert(RPT == 1 || F8, "several rows a thread is the E4M3 form's");
    constexpr int kBR = kUpRows * RPT;              /* rows of W_up this workgroup owns */
    extern __shared__ __attribute__((aligned(16))) float sd[];   /* [RG * lr] */
    __shared__ unsigned short s_stage[kWaves][kWave * 8];   /* wave_dot_lds, one slot a wave */
    __shared__ unsigned short s_u[RG * kBR];
    const unsigned short* __restrict__ up = (const unsigned short*)upv;
    const long r0 = (long)by * RG;
    if (r0 >= rows) return;
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const int  ipb = kBR / hc;                      /* channels this workgroup owns */
    const long i0  = (long)bx * ipb;

    /* Every load this kernel makes before its first barrier, issued together: at decode the
     * kernel is a chain of dependent latencies, not a stream of bytes. `d` goes first, all of a
     * thread's share at once (lr is at most two thread strides here), since loads return in order
     * and `d` is what the barrier waits on; then this thread's quarter row of W_up, which does not
     * depend on `d`; then, when one mix item a thread covers the block's (RG * ipb <= the thread
     * count, always at hc 4), the hc values N of its channel from `h`, the norm gain and kernel
     * A's scales. Only the issue order moves; the arithmetic sees the same values in the same
     * order, so the result is bit-identical. */
    constexpr int kDStage = 2;
    const bool dpre = lr <= (long)kDStage * kThreads;
    unsigned short dv[RG][kDStage];                 /* bf16 bits; the E4M3 dot keeps them as is */
    if (dpre && !WAIT) {
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            const bool live = r0 + r < rows;
#pragma unroll
            for (int s = 0; s < kDStage; ++s) {
                const long k = tid + (long)s * kThreads;
                dv[r][s] = (live && k < lr) ? dbuf[(r0 + r) * lr + k] : (unsigned short)0;
            }
        }
    }
    const int  t = tid >> 2, q4 = tid & 3;
    Hc8UpQuarter wq[RPT];
    if constexpr (F8) {
        static_assert(kUpRows * 4 == kThreads, "four threads a row of W_up");
#pragma unroll
        for (int m = 0; m < RPT; ++m) {
            const int  tt = t + kUpRows * m;
            const long j = (long)(tt / ipb) * n + i0 + tt % ipb;
            hc8_up_load((const unsigned char*)upv + j * uld, lr, q4, wq[m]);
        }
    }
    const bool one = RG * ipb <= kThreads;
    float Npre[kHcMax];
    if constexpr (WAIT) {
        /* The fused form (hc_read_fused_kernel): `h` and the gain here, kernel A's half after it. */
        float hv[kHcMax], wv[kHcMax];
        const int  rl = one && tid < RG * ipb ? tid / ipb : 0, ii = tid - rl * ipb;
        const long r  = r0 + rl < rows ? r0 + rl : rows - 1;
        if (one && tid < RG * ipb) {
#pragma unroll
            for (int c = 0; c < kHcMax; ++c) {
                if (c >= hc) break;
                const long j = (long)c * n + i0 + ii;
                hv[c] = bf2f(h[r * hrow + j]);
                wv[c] = w[j];
            }
        }
        hc_role_wait(sync, na, nb_all);
        if (dpre) {
#pragma unroll
            for (int rr = 0; rr < RG; ++rr) {
                const bool live = r0 + rr < rows;
#pragma unroll
                for (int s2 = 0; s2 < kDStage; ++s2) {
                    const long k = tid + (long)s2 * kThreads;
                    dv[rr][s2] = (live && k < lr) ? dbuf[(r0 + rr) * lr + k] : (unsigned short)0;
                }
            }
        }
        if (one && tid < RG * ipb) {
#pragma unroll
            for (int c = 0; c < kHcMax; ++c) {
                if (c >= hc) break;
                Npre[c] = nbf(hv[c] * scalein[r * kHcMax + c] * (wv[c] + wadd));
            }
        }
    } else if (one && tid < RG * ipb) {
        const int  rl = tid / ipb, ii = tid - rl * ipb;
        const long r  = r0 + rl < rows ? r0 + rl : rows - 1;
        const long i  = i0 + ii;
#pragma unroll
        for (int c = 0; c < kHcMax; ++c) {
            if (c >= hc) break;
            const long j = (long)c * n + i;
            Npre[c] = nbf(bf2f(h[r * hrow + j]) * scalein[r * kHcMax + c] * (w[j] + wadd));
        }
    }

    /* `d` as bf16 for the E4M3 dot (hc8_up_dot_rows), as floats for the bf16 one. */
    unsigned short* const sdh = (unsigned short*)sd;
    if (dpre) {
#pragma unroll
        for (int r = 0; r < RG; ++r)
#pragma unroll
            for (int s = 0; s < kDStage; ++s) {
                const long k = tid + (long)s * kThreads;
                if (k >= lr) continue;
                if constexpr (F8) sdh[(long)r * lr + k] = dv[r][s];
                else              sd[(long)r * lr + k] = bf2f(dv[r][s]);
            }
    } else {
        /* Row by row rather than `k / lr`, a 64-bit division per element. */
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            const bool live = r0 + r < rows;
            for (long k = tid; k < lr; k += kThreads) {
                const unsigned short v = live ? dbuf[(r0 + r) * lr + k] : (unsigned short)0;
                if constexpr (F8) sdh[(long)r * lr + k] = v;
                else              sd[(long)r * lr + k] = bf2f(v);
            }
        }
    }
    __syncthreads();

    /* Row t of this workgroup's slice of W_up is sub-stream t / ipb of channel i0 + t % ipb. The
     * E4M3 form takes each row four threads at a time, as in hc_up_kernel. */
    if constexpr (F8) {
        float acc[RPT][RG];
        hc8_up_dot_rows<RG, RPT>(sdh, lr, wq, q4, acc);
#pragma unroll
        for (int m = 0; m < RPT; ++m)
#pragma unroll
            for (int r = 0; r < RG; ++r)
                if (q4 == 0) s_u[r * kBR + t + kUpRows * m] = f2bf(sigmoidf_(acc[m][r]));
    } else {
        for (int t = wave; t < kUpRows; t += kWaves) {
            const long j = (long)(t / ipb) * n + i0 + t % ipb;
            float acc[RG];
            wave_dot_lds_rows<RG>(sd, lr, up + j * lr, lr, lane, s_stage[wave], acc);
#pragma unroll
            for (int r = 0; r < RG; ++r) {
                const float a = wave_sum(acc[r]);
                if (lane == 0) s_u[r * kBR + t] = f2bf(sigmoidf_(a));
            }
        }
    }
    __syncthreads();

    const float inv_hc = 1.0f / (float)hc;
    if (one) {
        if (tid < RG * ipb) {
            const int  rl = tid / ipb, ii = tid - rl * ipb;
            const long r  = r0 + rl;
            if (r < rows) {
                float acc = 0.0f;
#pragma unroll
                for (int c = 0; c < kHcMax; ++c) {
                    if (c >= hc) break;
                    acc += bf2f(s_u[rl * kBR + c * ipb + ii]) * Npre[c];
                }
                acc *= inv_hc;
                x[r * xrow + i0 + ii] = f2bf(acc);
            }
        }
    } else
    for (int idx = tid; idx < RG * ipb; idx += kThreads) {
        const int  rl = idx / ipb, ii = idx - rl * ipb;
        const long r  = r0 + rl;
        if (r >= rows) continue;
        const long i = i0 + ii;
        float acc = 0.0f;
        for (int c = 0; c < hc; ++c) {
            const long j = (long)c * n + i;
            const float Nv = nbf(bf2f(h[r * hrow + j]) * scalein[r * kHcMax + c]
                                 * (w[j] + wadd));
            acc += bf2f(s_u[rl * kBR + c * ipb + ii]) * Nv;
        }
        acc *= inv_hc;
        x[r * xrow + i] = f2bf(acc);
    }
    if (!q || !ctr) return;

    /* The group's last workgroup quantises it: kUpTB / ipb of them cover one group of a row group. */
    const int nb = kUpTB / ipb;
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_arrived;
    const unsigned slot = by * (unsigned)(n / kUpTB) + (unsigned)(i0 / kUpTB);
    if (tid == 0)
        s_arrived = __hip_atomic_fetch_add(&ctr[slot], 1u, __ATOMIC_ACQ_REL,
                                           __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived + 1u != (unsigned)nb) return;
    if (tid == 0)
        __hip_atomic_store(&ctr[slot], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();

    /* One wave per row (RG <= wave count); a lane takes four of the group's 128 channels, one
     * packed dword of codes. */
    if (wave >= RG) return;
    const long r = r0 + wave;
    if (r >= rows) return;
    const long base = (i0 / kUpTB) * kUpTB;
    const unsigned short* const xr = x + r * xrow + base;
    /* A lane's four channels, read once for the amax, the codes and the rotation; separate reads
     * would each be a dependent round trip at the end of the step's chain. The amax is over the
     * same 128 values and max is exact, so the result is bit-identical. */
    static_assert(kUpTB == 4 * kWave, "a lane holds four channels of the group");
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = bf2f(xr[lane * 4 + j]);
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    m = r4d_wave_max(m);
    const float sc  = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) qs[r * srow + base / kUpTB] = sc;
    const uint32_t pk = q8_pack4<I8>(v[0], v[1], v[2], v[3], inv);
    *(unsigned*)(q + r * qrow + base + lane * 4) = (unsigned)pk;
    if (rq) hc_rot_group<I8>(v, lane, rq + r * rqrow + base, rqs + r * rsrow + base / kUpTB);
}

template <int RG, bool F8, bool I8 = false, int RPT = 1>
__global__ __launch_bounds__(kThreads)
void hc_up_mix_kernel(const void* __restrict__ upv, long uld,
                      const unsigned short* __restrict__ dbuf,
                      const unsigned short* __restrict__ h, const float* __restrict__ w,
                      const float* __restrict__ scalein,
                      unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                      float* __restrict__ qs, unsigned char* __restrict__ rq,
                      float* __restrict__ rqs, unsigned int* __restrict__ ctr,
                      long rows, long n, int hc, long lr, long hrow, long xrow,
                      long qrow, long srow, long rqrow, long rsrow, float wadd) {
    hc_up_mix_body<RG, F8, I8, RPT>(upv, uld, dbuf, h, w, scalein, x, q, qs, rq, rqs, ctr, rows, n,
                                    hc, lr, hrow, xrow, qrow, srow, rqrow, rsrow, wadd, blockIdx.x,
                                    blockIdx.y);
}

/* ============================================================== hc_read, A and B in one grid
 *
 * At decode, kernel A is mostly latency (its norm prologue and the L2 round trips around
 * its dots) with memory idle meanwhile, and kernel B cannot ask for W_up until A has ended. Run
 * together, B's 3.4 MB of W_up is read while A's prologue waits, and B's work after it is short.
 *
 * Roles are handed out by ticket, not block index: a block's first act is an atomic increment of
 * sync[0], and tickets below `na` run kernel A's body (then add one to sync[1]), the rest kernel
 * B's. A B block loads its W_up rows, then waits for sync[1] to reach `na`. Every block holding a
 * smaller ticket has already started, so the blocks it waits on are resident and finish whatever
 * the waiters do: no co-residency is assumed and none can deadlock. The last B block past the
 * wait clears the three counters for the next launch on this stream (stream order publishes it).
 * A wait that runs past half a second traps rather than hang the card or read a stale `d`; it
 * cannot happen with the counters in order.
 *
 * Bit-identical to the two launches: the same bodies, the same per-block work. */
template <int RG, bool I8, int RPT, int HCP, int PER, class CH>
__global__ __launch_bounds__(kThreads)
__attribute__((amdgpu_waves_per_eu(9, 9)))
void hc_read_fused_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                          const void* __restrict__ downv, long dld,
                          const unsigned short* __restrict__ injw,
                          unsigned short* __restrict__ dbuf, unsigned short* __restrict__ inj,
                          float* __restrict__ scale, long ta, long irow, float eps,
                          const void* __restrict__ upv, long uld,
                          unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                          float* __restrict__ qs, unsigned char* __restrict__ rq,
                          float* __restrict__ rqs, unsigned int* __restrict__ ctr,
                          long rows, long n, int hc, long lr, long hrow, long xrow,
                          long qrow, long srow, long rqrow, long rsrow, float wadd,
                          unsigned int* __restrict__ sync,
                          unsigned na, unsigned gxa, unsigned gxb, unsigned nb_all) {
    if (CH::on) r4d_chain_wait();
    __shared__ unsigned s_ticket;
    if (threadIdx.x == 0)
        s_ticket = __hip_atomic_fetch_add(&sync[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    const unsigned tk = s_ticket;
    if (tk < na) {
        hc_read_a_body<true, HCP, PER>(h, w, downv, dld, injw, dbuf, inj, scale, rows, n, hc, lr,
                                       ta, hrow, irow, eps, wadd, tk % gxa, tk / gxa, gxa - 1u);
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
            __hip_atomic_fetch_add(&sync[1], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    } else {
        const unsigned b = tk - na;
        hc_up_mix_body<RG, true, I8, RPT, true>(upv, uld, dbuf, h, w, scale, x, q, qs, rq, rqs,
                                                ctr, rows, n, hc, lr, hrow, xrow, qrow, srow, rqrow,
                                                rsrow, wadd, b % gxb, b / gxb, sync, na, nb_all);
    }
    if (CH::on) r4d_chain_done();
}

/* Row group size: the row count, up to eight. RG is a template parameter so `acc[RG]` stays in
 * registers. Every size from 1 to 8 is instantiated, so up to eight rows read the weight once
 * (a five-row verify split 4 + 1 would read W_up twice). Output bytes do not depend on RG.
 *
 * LDS bounds it: RG rows of `d` are RG*lr floats beside 4 KiB of staging, so a wide low rank gets
 * the largest group that fits rather than a launch failure. */
static inline long hc_up_rg(long long rows, long long lr) {
    constexpr long kLdsCap = 60 * 1024;          /* leave the 4 KiB of staging slots room */
    long rg = rows < 8 ? (rows > 1 ? (long)rows : 1) : 8;
    while (rg > 1 && (long long)rg * lr * (long long)sizeof(float) > kLdsCap) --rg;
    return rg;
}

static inline int hc_up_launch(const RadTensor* up, long uld, bool f8, const unsigned short* dbuf,
                               unsigned short* ubuf, long long rows, long long HN, long long lr,
                               RadStream stream) {
    int emit_rc = RAD_OK;
    const long rg = hc_up_rg(rows, lr);
    const dim3 grid((unsigned)((HN + kUpRows - 1) / kUpRows),
                    (unsigned)((rows + rg - 1) / rg));
    const unsigned lds = (unsigned)(rg * lr * sizeof(float));
    const void* upp = up->data;
#define HC_UP1_CASE(RG_, F8_)                                                                    \
    emit_rc = R4D_EMIT(stream, (hc_up_kernel<RG_, F8_>), grid, dim3(kThreads), lds, upp, uld,    \
                       dbuf, ubuf, (long)rows, (long)HN, (long)lr)
    switch (rg) {
        case 8:  if (f8) HC_UP1_CASE(8, true); else HC_UP1_CASE(8, false); break;
        case 7:  if (f8) HC_UP1_CASE(7, true); else HC_UP1_CASE(7, false); break;
        case 6:  if (f8) HC_UP1_CASE(6, true); else HC_UP1_CASE(6, false); break;
        case 5:  if (f8) HC_UP1_CASE(5, true); else HC_UP1_CASE(5, false); break;
        case 4:  if (f8) HC_UP1_CASE(4, true); else HC_UP1_CASE(4, false); break;
        case 3:  if (f8) HC_UP1_CASE(3, true); else HC_UP1_CASE(3, false); break;
        case 2:  if (f8) HC_UP1_CASE(2, true); else HC_UP1_CASE(2, false); break;
        default: if (f8) HC_UP1_CASE(1, true); else HC_UP1_CASE(1, false); break;
    }
#undef HC_UP1_CASE
    return emit_rc;
}

/* Rows of W_up a thread (hc_up_mix_kernel's RPT), where the wider workgroup's channels
 * still divide a scale group and the width. Two: at Flash-Next's shape (cold caches, the fused
 * form at one, two and four rows) two beats one at every row count, and four runs past the
 * register budget and loses to two. */
constexpr int kUpRpt = 2;
static inline int hc_up_rpt(bool f8, int hc, long n) {
    int rpt = f8 ? kUpRpt : 1;
    while (rpt > 1 && ((kUpRows * rpt) % hc || kUpTB % ((kUpRows * rpt) / hc) ||
                       n % ((kUpRows * rpt) / hc)))
        rpt >>= 1;
    return rpt;
}

/* ============================================================== the two-grid decode kernels
 *
 * hc_read's decode path as two launches: kernel A, then the up-mix kernel in a second launch, and
 * the tiled up-mix with its K passes unpipelined. r4d_hc_read_pin_decode(1) selects them, so
 * r4d_selftest can compare the served forms against them byte for byte and --perfhc can time both
 * from one build. They are the reference for that comparison; keep them as they are. */
static int g_hc_decode_form = 0;

template <bool F8>
__global__ __launch_bounds__(kThreads)
__attribute__((amdgpu_waves_per_eu(F8 ? 9 : 1, F8 ? 9 : 16)))
void hc_read_a_2grid_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                      const void* __restrict__ downv, long dld,
                      const unsigned short* __restrict__ injw,
                      unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
                      float* __restrict__ scaleout,
                      long rows, long n, int hc, long lr, long ta, long hrow, long irow,
                      float eps, float wadd) {
    extern __shared__ unsigned short sN[];          /* [hc * n], transposed per 256-element chunk */
    __shared__ float s_part[kWaves][kHcMax];
    __shared__ float s_scale[kHcMax];
    /* One 256-entry staging slot per wave (4 KiB); no barrier, a wave uses only its own. */
    __shared__ unsigned short s_stage[kWaves][kWave * 8];
    static_assert(sizeof(s_part) + sizeof(s_scale) + sizeof(s_stage) == kHcReadAStaticLds,
                  "the launcher sizes N's allocation against this kernel's static LDS");

    constexpr long kPer = kWave * 8;

    const long r  = blockIdx.y;
    if (r >= rows) return;
    const long HN = (long)hc * n;
    /* The inject gains get a block of their own for load balance. With ta = kWaves every wave does
     * one dot over HN; adding the gains to a column-owning block would give its waves a second
     * one, and that slowest block would set the kernel's duration. So the grid is one wider and
     * the last block owns no columns. */
    const bool pub = (blockIdx.x == gridDim.x - 1);
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const unsigned short* hp = h + r * hrow;
    const unsigned short* __restrict__ down = (const unsigned short*)downv;

    /* The transposed layout follows from the shape: the chunk must sit inside one sub-stream
     * (n % 256) and every 16-byte group be aligned in LDS and in `h`; otherwise the plain layout,
     * with the same N. The E4M3 dot reads N in natural order, so only the bf16 form transposes. */
    const bool swz = !F8 && (n % kPer == 0) && (((uintptr_t)hp   & 15u) == 0u)
                  && (((uintptr_t)down & 15u) == 0u)
                  && (!injw || ((uintptr_t)injw & 15u) == 0u);

    /* ---- sums of squares, one stream at a time, straight from `h`. One stream at a time keeps
     * the accumulator a scalar; an array indexed by runtime `hc` would live in scratch.
     *
     * This prologue is latency-bound, not traffic-bound: 20 KiB against the block's 320 KiB of
     * W_down, yet 6.7 of the kernel's 18.7 us at the decode grid, because ten dependent iterations
     * wait on loads with nothing else resident. Issuing kPreSs iterations' loads first turns eight
     * serial latencies into one, bit-identically. Wider loads (these bytes are served from the
     * 64 MB last-level cache, not DRAM), reading `h` once instead of twice, and deeper prefetch
     * were all measured as no gain. */
    constexpr int kPreSs = 8;      // n / kThreads is 10 a stream at this shape, and 12 spills
    for (int c = 0; c < hc; ++c) {
        const long cb = (long)c * n;
        float acc = 0.0f;
        long  i = tid;
        for (; i + (long)(kPreSs - 1) * kThreads < n; i += (long)kPreSs * kThreads) {
            uint32_t hv[kPreSs];
#pragma unroll
            for (int p = 0; p < kPreSs; ++p) hv[p] = bfw(hp + cb + i + (long)p * kThreads);
#pragma unroll
            for (int p = 0; p < kPreSs; ++p) { const float v = bfw_f(hv[p]); acc += v * v; }
        }
        for (; i < n; i += kThreads) { const float v = bf2f(hp[cb + i]); acc += v * v; }
        acc = wave_sum(acc);
        if (lane == 0) s_part[wave][c] = acc;
    }
    __syncthreads();
    if (tid < hc) {
        float ss = 0.0f;
        for (int k = 0; k < kWaves; ++k) ss += s_part[k][tid];
        s_scale[tid] = __frsqrt_rn(ss / (float)n + eps);
        /* One workgroup publishes; the others compute the identical value. */
        if (pub && scaleout) scaleout[r * kHcMax + tid] = s_scale[tid];
    }
    __syncthreads();

    /* ---- N into LDS, narrowed to bf16 where the reference narrows it. Looped per stream rather
     * than computing `j / n`, a 64-bit integer division per element (a software sequence here). */
    for (int c = 0; c < hc; ++c) {
        const long  cb = (long)c * n;
        const float sc = s_scale[c];
        if (swz) {
            /* A thread owns one (chunk, lane) group: logical elements `lane + 32t`, one 16-byte
             * store at physical `q`. A wave covers one chunk, so `ch` is wave-uniform and the
             * `h` and `w` reads stay coalesced. */
            for (long base = 0; base < n; base += (long)kThreads * 8) {
                const long q = base + (long)tid * 8;
                if (q >= n) break;
                const long ch = cb + (q & ~(kPer - 1));
                float out[8];
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const long li = ch + lane + (long)t * kWave;
                    out[t] = bfw_f(bfw(hp + li)) * sc * (w[li] + wadd);
                }
                *(uint4*)(sN + cb + q) = f2bf8(out);
            }
        } else {
            constexpr int kPreScale = 8;
            long i = tid;
            for (; i + (long)(kPreScale - 1) * kThreads < n; i += (long)kPreScale * kThreads) {
                float          wv[kPreScale];
                uint32_t       hv[kPreScale];
#pragma unroll
                for (int p = 0; p < kPreScale; ++p) {
                    wv[p] = w[cb + i + (long)p * kThreads];
                    hv[p] = bfw(hp + cb + i + (long)p * kThreads);
                }
#pragma unroll
                for (int p = 0; p < kPreScale; ++p)
                    sN[cb + i + (long)p * kThreads] = f2bf(bfw_f(hv[p]) * sc * (wv[p] + wadd));
            }
            for (; i < n; i += kThreads)
                sN[cb + i] = f2bf(bf2f(hp[cb + i]) * sc * (w[cb + i] + wadd));
        }
    }
    __syncthreads();

    /* ---- this workgroup's columns of W_down, one wave per column. */
    const float inv_hc = 1.0f / (float)hc;
    const long  k0 = (long)blockIdx.x * ta;
    for (long k = k0 + wave; k < k0 + ta && k < lr; k += kWaves) {
        float acc;
        if constexpr (F8) {
            acc = hc8_dot_down<kDotU>(sN, (const unsigned char*)downv + k * dld, HN, lane);
        } else {
            const unsigned short* wr = down + k * HN;
            acc = swz ? wave_dot_swz<kDotU>(sN, wr, HN, lane)
                      : wave_dot_lds<kDotU>(sN, wr, HN, lane, s_stage[wave]);
        }
        acc = wave_sum(acc);
        if (lane == 0) dout[r * lr + k] = f2bf(siluf_(acc * inv_hc));
    }

    /* ---- the inject gains, on the block that owns no columns and already holds N. */
    if (pub && injw && inj) {
        for (int c = wave; c < hc; c += kWaves) {
            const unsigned short* wr = injw + (long)c * HN;
            float acc = swz ? wave_dot_swz<kDotU>(sN, wr, HN, lane)
                            : wave_dot_lds<kDotU>(sN, wr, HN, lane, s_stage[wave]);
            acc = wave_sum(acc);
            if (lane == 0) inj[r * irow + c] = f2bf(2.0f * sigmoidf_(acc * inv_hc));
        }
    }
}

template <int RG, bool F8, bool I8 = false>
__global__ __launch_bounds__(kThreads)
void hc_up_mix_2grid_kernel(const void* __restrict__ upv, long uld,
                      const unsigned short* __restrict__ dbuf,
                      const unsigned short* __restrict__ h, const float* __restrict__ w,
                      const float* __restrict__ scalein,
                      unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                      float* __restrict__ qs, unsigned char* __restrict__ rq,
                      float* __restrict__ rqs, unsigned int* __restrict__ ctr,
                      long rows, long n, int hc, long lr, long hrow, long xrow,
                      long qrow, long srow, long rqrow, long rsrow, float wadd) {
    extern __shared__ float sd[];                   /* [RG * lr] */
    __shared__ unsigned short s_stage[kWaves][kWave * 8];   /* wave_dot_lds, one slot a wave */
    __shared__ unsigned short s_u[RG * kUpRows];
    const unsigned short* __restrict__ up = (const unsigned short*)upv;
    const long r0 = (long)blockIdx.y * RG;
    if (r0 >= rows) return;
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const int  ipb = kUpRows / hc;                  /* channels this workgroup owns */
    const long i0  = (long)blockIdx.x * ipb;

    for (long k = tid; k < (long)RG * lr; k += kThreads) {
        const long r = k / lr;
        sd[k] = (r0 + r < rows) ? bf2f(dbuf[(r0 + r) * lr + (k - r * lr)]) : 0.0f;
    }
    __syncthreads();

    /* Row t of this workgroup's slice of W_up is sub-stream t / ipb of channel i0 + t % ipb. The
     * E4M3 form takes each row four threads at a time, as in hc_up_kernel. */
    if constexpr (F8) {
        static_assert(kUpRows * 4 == kThreads, "four threads a row of W_up");
        const int  t = tid >> 2, q4 = tid & 3;
        const long j = (long)(t / ipb) * n + i0 + t % ipb;
        float acc[RG];
        hc8_dot_up<RG>(sd, lr, (const unsigned char*)upv + j * uld, q4, acc);
#pragma unroll
        for (int r = 0; r < RG; ++r)
            if (q4 == 0) s_u[r * kUpRows + t] = f2bf(sigmoidf_(acc[r]));
    } else {
        for (int t = wave; t < kUpRows; t += kWaves) {
            const long j = (long)(t / ipb) * n + i0 + t % ipb;
            float acc[RG];
            wave_dot_lds_rows<RG>(sd, lr, up + j * lr, lr, lane, s_stage[wave], acc);
#pragma unroll
            for (int r = 0; r < RG; ++r) {
                const float a = wave_sum(acc[r]);
                if (lane == 0) s_u[r * kUpRows + t] = f2bf(sigmoidf_(a));
            }
        }
    }
    __syncthreads();

    const float inv_hc = 1.0f / (float)hc;
    for (int idx = tid; idx < RG * ipb; idx += kThreads) {
        const int  rl = idx / ipb, ii = idx - rl * ipb;
        const long r  = r0 + rl;
        if (r >= rows) continue;
        const long i = i0 + ii;
        float acc = 0.0f;
        for (int c = 0; c < hc; ++c) {
            const long j = (long)c * n + i;
            const float Nv = nbf(bf2f(h[r * hrow + j]) * scalein[r * kHcMax + c]
                                 * (w[j] + wadd));
            acc += bf2f(s_u[rl * kUpRows + c * ipb + ii]) * Nv;
        }
        acc *= inv_hc;
        x[r * xrow + i] = f2bf(acc);
    }
    if (!q || !ctr) return;

    /* The group's last workgroup quantises it: kUpTB / ipb of them cover one group of a row group. */
    const int nb = kUpTB / ipb;
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_arrived;
    const unsigned slot = blockIdx.y * (unsigned)(n / kUpTB) + (unsigned)(i0 / kUpTB);
    if (tid == 0)
        s_arrived = __hip_atomic_fetch_add(&ctr[slot], 1u, __ATOMIC_ACQ_REL,
                                           __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived + 1u != (unsigned)nb) return;
    if (tid == 0)
        __hip_atomic_store(&ctr[slot], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();

    /* One wave per row (RG <= wave count); a lane takes four of the group's 128 channels, one
     * packed dword of codes. */
    if (wave >= RG) return;
    const long r = r0 + wave;
    if (r >= rows) return;
    const long base = (i0 / kUpTB) * kUpTB;
    const unsigned short* const xr = x + r * xrow + base;
    float m = 0.0f;
    for (int t = lane; t < kUpTB; t += kWave) m = fmaxf(m, fabsf(bf2f(xr[t])));
    m = r4d_wave_max(m);
    const float sc  = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) qs[r * srow + base / kUpTB] = sc;
    for (int d = lane; d < kUpTB / 4; d += kWave) {
        const int c4 = d * 4;
        const uint32_t pk = q8_pack4<I8>(bf2f(xr[c4 + 0]), bf2f(xr[c4 + 1]), bf2f(xr[c4 + 2]),
                                         bf2f(xr[c4 + 3]), inv);
        *(unsigned*)(q + r * qrow + base + c4) = (unsigned)pk;
    }
    if (rq) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = bf2f(xr[lane * 4 + j]);
        hc_rot_group<I8>(v, lane, rq + r * rqrow + base, rqs + r * rsrow + base / kUpTB);
    }
}

static inline int hc_up_mix_2grid_launch(const RadTensor* up, long uld, bool f8, bool i8,
                                        const unsigned short* dbuf,
                                        const unsigned short* h, const float* w, const float* scale,
                                        unsigned short* x, unsigned char* q, float* qs,
                                        unsigned char* rq, float* rqs,
                                        unsigned int* ctr, long long rows, long long n, int hc,
                                        long long lr, long long hrow, long long xrow, long long qrow,
                                        long long srow, long long rqrow, long long rsrow, float wadd,
                                        RadStream stream) {
    int emit_rc = RAD_OK;
    const long rg = hc_up_rg(rows, lr);
    const dim3 grid((unsigned)(n / (kUpRows / hc)), (unsigned)((rows + rg - 1) / rg));
    const unsigned lds = (unsigned)(rg * lr * sizeof(float));
    const void* upp = up->data;
#define HC_UPMIX_CASE3(RG_, F8_, I8_)                                                            \
    emit_rc = R4D_EMIT(stream, (hc_up_mix_2grid_kernel<RG_, F8_, I8_>), grid, dim3(kThreads), lds,     \
                       upp, uld, dbuf, h, w, scale, x, q, qs, rq, rqs, ctr, (long)rows, (long)n,  \
                       hc,                                                                        \
                       (long)lr, (long)hrow, (long)xrow, (long)qrow, (long)srow, (long)rqrow,     \
                       (long)rsrow, wadd)
#define HC_UPMIX_CASE(RG_, F8_)                                                                  \
    do { if (i8) HC_UPMIX_CASE3(RG_, F8_, true); else HC_UPMIX_CASE3(RG_, F8_, false); } while (0)
    switch (rg) {
        case 8:  if (f8) HC_UPMIX_CASE(8, true); else HC_UPMIX_CASE(8, false); break;
        case 7:  if (f8) HC_UPMIX_CASE(7, true); else HC_UPMIX_CASE(7, false); break;
        case 6:  if (f8) HC_UPMIX_CASE(6, true); else HC_UPMIX_CASE(6, false); break;
        case 5:  if (f8) HC_UPMIX_CASE(5, true); else HC_UPMIX_CASE(5, false); break;
        case 4:  if (f8) HC_UPMIX_CASE(4, true); else HC_UPMIX_CASE(4, false); break;
        case 3:  if (f8) HC_UPMIX_CASE(3, true); else HC_UPMIX_CASE(3, false); break;
        case 2:  if (f8) HC_UPMIX_CASE(2, true); else HC_UPMIX_CASE(2, false); break;
        default: if (f8) HC_UPMIX_CASE(1, true); else HC_UPMIX_CASE(1, false); break;
    }
#undef HC_UPMIX_CASE
#undef HC_UPMIX_CASE3
    return emit_rc;
}

/* hc_up_mix_kernel's launch: hc_up's row group; counter slots are (rows / rg) row groups of
 * n / kUpTB scale groups. */
static inline int hc_up_mix_launch(const RadTensor* up, long uld, bool f8, bool i8,
                                   const unsigned short* dbuf,
                                   const unsigned short* h, const float* w, const float* scale,
                                   unsigned short* x, unsigned char* q, float* qs,
                                   unsigned char* rq, float* rqs,
                                   unsigned int* ctr, long long rows, long long n, int hc,
                                   long long lr, long long hrow, long long xrow, long long qrow,
                                   long long srow, long long rqrow, long long rsrow, float wadd,
                                   RadStream stream) {
    if (g_hc_decode_form == 1)
        return hc_up_mix_2grid_launch(up, uld, f8, i8, dbuf, h, w, scale, x, q, qs, rq, rqs, ctr,
                                     rows, n, hc, lr, hrow, xrow, qrow, srow, rqrow, rsrow, wadd,
                                     stream);
    int emit_rc = RAD_OK;
    const long rg = hc_up_rg(rows, lr);
    const int rpt = hc_up_rpt(f8, hc, n);
    const dim3 grid((unsigned)(n / (kUpRows * rpt / hc)), (unsigned)((rows + rg - 1) / rg));
    const unsigned lds = (unsigned)(rg * lr * sizeof(float));
    const void* upp = up->data;
#define HC_UPMIX_K(RG_, F8_, I8_, RPT_)                                                          \
    emit_rc = R4D_EMIT(stream, (hc_up_mix_kernel<RG_, F8_, I8_, RPT_>), grid, dim3(kThreads), lds, \
                       upp, uld, dbuf, h, w, scale, x, q, qs, rq, rqs, ctr, (long)rows, (long)n,  \
                       hc,                                                                        \
                       (long)lr, (long)hrow, (long)xrow, (long)qrow, (long)srow, (long)rqrow,     \
                       (long)rsrow, wadd)
#define HC_UPMIX_F8(RG_, I8_)                                                                    \
    do { if (rpt == 2) HC_UPMIX_K(RG_, true, I8_, 2); else HC_UPMIX_K(RG_, true, I8_, 1); } while (0)
#define HC_UPMIX_CASE(RG_)                                                                       \
    do { if (f8) { if (i8) HC_UPMIX_F8(RG_, true); else HC_UPMIX_F8(RG_, false); }               \
         else if (i8) HC_UPMIX_K(RG_, false, true, 1); else HC_UPMIX_K(RG_, false, false, 1); } while (0)
    switch (rg) {
        case 8:  HC_UPMIX_CASE(8); break;
        case 7:  HC_UPMIX_CASE(7); break;
        case 6:  HC_UPMIX_CASE(6); break;
        case 5:  HC_UPMIX_CASE(5); break;
        case 4:  HC_UPMIX_CASE(4); break;
        case 3:  HC_UPMIX_CASE(3); break;
        case 2:  HC_UPMIX_CASE(2); break;
        default: HC_UPMIX_CASE(1); break;
    }
#undef HC_UPMIX_CASE
#undef HC_UPMIX_F8
#undef HC_UPMIX_K
    return emit_rc;
}

/* ============================================================== hc_read, kernel B2
 *
 * `x = mean_c( u[c] * N[c] )` for this workgroup's 128 channels, plus their fp8 codes: 128 is one
 * scale group, so the amax needs no second pass.
 *
 * N is rebuilt from `h` and kernel A's published scales (512 elements here, against 20 KiB to
 * carry the whole row). */
template <bool I8 = false>
__global__ __launch_bounds__(kThreads)
void hc_mix_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                   const unsigned short* __restrict__ ubuf, const float* __restrict__ scalein,
                   unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                   float* __restrict__ qs, unsigned char* __restrict__ rq,
                   float* __restrict__ rqs,
                   long rows, long n, int hc, long hrow, long xrow,
                   long qrow, long srow, long rqrow, long rsrow, float wadd) {
    __shared__ float s_x[kUpTB];
    __shared__ float s_amax[kWaves];

    const long r = blockIdx.y;
    if (r >= rows) return;
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long i0  = (long)blockIdx.x * kUpTB;
    const long hn  = (long)hc * n;

    const float inv_hc = 1.0f / (float)hc;
    if (tid < kUpTB) {
        const long i = i0 + tid;
        float acc = 0.0f;
        if (i < n) {
            for (int c = 0; c < hc; ++c) {
                const long j = (long)c * n + i;
                const float Nv = nbf(bf2f(h[r * hrow + j]) * scalein[r * kHcMax + c]
                                     * (w[j] + wadd));
                acc += bf2f(ubuf[r * hn + j]) * Nv;
            }
            acc *= inv_hc;
            x[r * xrow + i] = f2bf(acc);
        }
        s_x[tid] = acc;
    }
    if (!q) return;
    __syncthreads();

    /* Quantise what was written: `s_x` is narrowed to bf16 first, as the reference reads it back. */
    float m = 0.0f;
    for (int t = tid; t < kUpTB; t += kThreads) m = fmaxf(m, fabsf(nbf(s_x[t])));
    m = r4d_wave_max(m);
    if (lane == 0) s_amax[wave] = m;
    __syncthreads();
    float amax = 0.0f;
    for (int k = 0; k < kWaves; ++k) amax = fmaxf(amax, s_amax[k]);
    const float sc  = amax > 0.0f ? amax * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (tid == 0) qs[r * srow + blockIdx.x] = sc;
    /* Four codes per lane packed into one dword, as in r4d_quant_act_fp8. */
    if (tid < kUpTB / 4) {
        const long c0 = i0 + (long)tid * 4;
        float v[4];
        for (int j = 0; j < 4; ++j) v[j] = (c0 + j) < n ? nbf(s_x[tid * 4 + j]) : 0.0f;
        const uint32_t pk = q8_pack4<I8>(v[0], v[1], v[2], v[3], inv);
        if (c0 < n) *(unsigned*)(q + r * qrow + c0) = (unsigned)pk;
    }
    /* The rotated codes from the same narrowed values; the launch requires whole groups. */
    if (rq && wave == 0) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = nbf(s_x[lane * 4 + j]);
        hc_rot_group<I8>(v, lane, rq + r * rqrow + i0, rqs + r * rsrow + blockIdx.x);
    }
}

/* ==================================================== hc_read, the tiled path
 *
 * The kernels above put rows on the grid, so each token reads its whole slice of the mixing
 * matrices. Right at decode; at a 2048-token prefill chunk it is 2.5 TB of weight traffic per step
 * (56% of a Flash-Next prefill step).
 *
 * Above a row threshold the two products run as tiled GEMMs: a workgroup owns 128 rows and 64
 * output columns and reuses each weight tile across its rows, so the weight is read
 * ceil(rows/128) times. Only the summation order changes (WMMA fragments instead of lane-strided
 * partials), the same kind of reordering the per-row form already has against the reference.
 *
 * The norm moves out: N for 128 rows is 2.5 MB, too big for LDS. The scales get their own kernel
 * (one pass over `h`) and the GEMM rebuilds N while staging each A tile, which it must stage
 * anyway.
 *
 * Two GEMMs, not one: a workgroup owning a whole 128-channel scale group of the mix would own
 * hc*128 = 512 rows of W_up but only 16 rows of accumulator, re-reading the weight rows/16 times.
 * Instead a workgroup owns sixteen channels of every sub-stream with an arrival counter for the
 * scale group (hc_up_mix_tiled_kernel at decode tiles, hc_up_mix_wide_kernel at prefill).
 */
constexpr int kTNT = 64;            // output columns a workgroup owns: four n tiles
/* kTNT is fitted to decode shapes. The tiled down product's grid is (ncol/kTNT, rows/rt, HN/kc), so
 * `h` is re-read per column tile and the weight per row tile; at decode h is small, but at a
 * 2048-row chunk six reads of `h` (251 MB) outweigh the weight (105 MB). So the prefill tile uses
 * hc_down_wide_kernel with kWideNT columns per workgroup, keeping hc_down_kchunk's split counted at
 * kTNT columns (at 2048 rows, splits of 1, 3, 4, 5 and 8 are all slower than its 2; no split
 * leaves 48 workgroups on 64 CUs). */
constexpr int kTKT = 64;            // K staged per pass
constexpr int kTAP = kTKT + 4;      // LDS row pitch in shorts (136 B): 8-byte aligned, and a
                                    // fragment's 16 lanes hit 16 distinct bank pairs (as in the
                                    // kkt solve).
constexpr long kTileMinRows = 64;   // from here the 128-row tile fills and the down product needs
                                    // no K-split: the prefill form.
constexpr long kNdFuseRows  = 64;   // to here the tiled read is two launches (hc_norm_down_tiled_kernel)
constexpr long kDecMinRows  = 8;    // below this the per-row kernels win; the two forms are level
                                    // at eight rows, and the tiled form gains ~25% of the step by
                                    // eight concurrent sequences.
/* The fused kernel (hc_read_fused_kernel, the E4M3 form at hc 4 and n 2560) takes eight rows too.
 * Alone the two forms are level there, but the tiled pair is two launches and in a step the fused
 * grid is ahead: two sequences at depth 3 (eight verify rows) 17.77 -> 17.02 ms a step. Past eight
 * rows the per-row dots lose (ten rows level, twelve +0.3 ms, sixteen +1.6 ms a step). The two
 * forms round differently (fmaf dots against WMMA tiles), each within the reference's tolerance. */
constexpr long kFusedMaxRows = 8;

/* ==================================================== the norm scales, alone
 *
 * The inject gains ride along as extra columns of the down product (hc_down_tiled_kernel), so
 * streams are independent and the grid is (hc, rows): one rsqrt per (row, stream), no LDS copy of
 * N. Bit-identical to the fused form: same thread stride, wave_sum and ascending sum over the wave
 * partials. Removing the launch itself would need `h`'s producer (hc_write, another op with its
 * own arena) to compute the scales. */
__device__ __forceinline__ void hc_norm_scale_body(const unsigned short* __restrict__ h,
                                                   float* scaleout, long rows, long n, long hrow,
                                                   float eps, long r, long c) {
    __shared__ float s_part[kWaves];
    if (r >= rows) return;
    const int tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const unsigned short* hp = h + r * hrow + c * n;
    /* kNormPre of a thread's elements are loaded before the first is used: the loop is otherwise
     * n / kThreads dependent round trips with nothing else resident (hc_read_a_body's prologue has
     * the same note). A load past the stream's end repeats the batch's first and is not added, so
     * the sum takes the same values in the same order. */
    constexpr int kNormPre = 16;
    float acc = 0.0f;
    for (long i = tid; i < n; i += (long)kNormPre * kThreads) {
        unsigned short hv[kNormPre];
#pragma unroll
        for (int p = 0; p < kNormPre; ++p) {
            const long j = i + (long)p * kThreads;
            hv[p] = hp[j < n ? j : i];
        }
#pragma unroll
        for (int p = 0; p < kNormPre; ++p) {
            if (i + (long)p * kThreads >= n) continue;
            const float v = bf2f(hv[p]);
            acc += v * v;
        }
    }
    acc = wave_sum(acc);
    if (lane == 0) s_part[wave] = acc;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (int k = 0; k < kWaves; ++k) ss += s_part[k];
        scaleout[r * kHcMax + c] = __frsqrt_rn(ss / (float)n + eps);
    }
}
__global__ __launch_bounds__(kThreads)
void hc_norm_scale_kernel(const unsigned short* __restrict__ h, float* __restrict__ scaleout,
                          long rows, long n, long hrow, float eps) {
    hc_norm_scale_body(h, scaleout, rows, n, hrow, eps, (long)blockIdx.y, (long)blockIdx.x);
}

/* ==================================================== tile shape follows the row count
 *
 * RT varies with rows: 128 suits thousands of rows, but at 32 it would stage 128 rows to use 32
 * and leave the down product five workgroups on 64 CUs (the 128-row tile at decode roughly doubles
 * the step at one sequence; kTileMinRows guards it).
 *
 * Wave mapping follows RT: WM = min(MT, waves) waves across the m-tiles and WN = waves/WM across
 * the n-tiles. RT 128 gives one m-tile per wave (WM 8, WN 1).
 *
 * Only the down product splits K: its output is [rows, 320], five 64-column tiles, while the up
 * product's [rows, 10240] already fills the part. Split blocks write f32 partials over blockIdx.z
 * (ksplit * rows * lr floats, 655 KB at the decode band) and a reduce pass sums them in ascending z
 * and applies the silu.
 */
constexpr int kTKSPLIT_WANT = 128;   // workgroups the down product aims to cover the part with

/* ==================================================== d = silu(W_down @ N / hc), tiled
 *
 * C[RT rows, NT cols] over K = hc*n. The A tile is N, rebuilt from `h` while staged; the launch
 * requires n % kTKT == 0, so a K slice never straddles a stream and the scale index is one divide
 * per tile.
 *
 * `part` non-null is the split-K form: the block owns K in [z*kc, min(HN, (z+1)*kc)) and writes its
 * raw f32 accumulator; silu and /hc move to the reduce. With `part` null, z is 0 and the epilogue
 * applies them.
 *
 * The inject gains ride along as hc more columns: `inj[c] = 2 sigmoid(W_inj[c] . N / hc)` uses the
 * same staged A, and W_inj [hc, HN] is just more weight rows (a separate kernel would cost ~18 us
 * at eight sequences). `ncol` is lr + hc with gains, lr without; columns at or past lr read injw
 * and take the sigmoid epilogue. */
__device__ __forceinline__ uint4 e4m3x8_bf16v(uint2 c, float sc);
template <int RT, int NT, bool F8>
__device__ __forceinline__ void hc_down_tiled_body(
    const unsigned short* __restrict__ h, const float* __restrict__ w,
    const void* __restrict__ downv, long dld, const float* scalein,
    const unsigned short* __restrict__ injw, unsigned short* __restrict__ dout,
    unsigned short* __restrict__ inj, float* __restrict__ part,
    long rows, long n, int hc, long lr, long ncol, long hrow, long irow, long kc, float wadd,
    unsigned int* __restrict__ ctr, unsigned bx, unsigned by, unsigned bz, unsigned gdx,
    unsigned gdz) {
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];

    constexpr int MT = RT / 16, NT16 = NT / 16;
    constexpr int WM = (MT < kWaves) ? MT : kWaves, WN = kWaves / WM;
    constexpr int AM = (MT + WM - 1) / WM, AN = (NT16 + WN - 1) / WN;

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long HN = (long)hc * n;
    const long r0 = (long)by * RT;
    const long c0 = (long)bx * NT;
    const long ka = (long)bz * kc;
    const long kb = (ka + kc < HN) ? ka + kc : HN;
    if (r0 >= rows || c0 >= ncol || ka >= HN) return;

    v8f acc[AM][AN];
#pragma unroll
    for (int im = 0; im < AM; ++im)
#pragma unroll
        for (int in = 0; in < AN; ++in) acc[im][in] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    /* The WMMA of one staged pass. */
    auto mma = [&]() {
#pragma unroll
        for (int im = 0; im < AM; ++im) {
            const int mt = wave / WN + im * WM;
            if (mt >= MT) continue;
#pragma unroll
            for (int in = 0; in < AN; ++in) {
                const int nt = wave % WN + in * WN;
                if (nt >= NT16) continue;
                acc[im][in] = gemm<kTKT / 16>(sA, kTAP, mt, sB, kTAP, nt, lane, acc[im][in]);
            }
        }
    };

    /* On the decode tiles the K loop is software-pipelined, as hc_up_mix_tiled_body's is: a pass
     * issues the next pass's loads before it stages its own from registers, and the barriers order
     * LDS only (hc_lds_barrier), so those loads stay in flight across the staging and the WMMA.
     * A split block's seven passes are otherwise seven round trips to the weight with nothing else
     * resident. The staged values are the plain loop's, in its order: bit-identical. With the
     * norm's preload (hc_norm_scale_body), a cold read alone at 12 / 16 / 32 / 64 rows takes
     * 42.0 / 42.6 / 46.4 / 74.9 us against 43.8 / 44.2 / 47.9 / 77.9. */
    constexpr int  kAPer = RT * (kTKT / 8) / kThreads, kBPer = NT * (kTKT / 8) / kThreads;
    constexpr bool kPipe = RT <= 64 && kAPer >= 1 && kBPer >= 1
                           && kAPer * kThreads == RT * (kTKT / 8)
                           && kBPer * kThreads == NT * (kTKT / 8);
    if constexpr (kPipe) {
        int  rrA[kAPer], k8A[kAPer], ttB[kBPer], k8B[kBPer];
        long rA[kAPer];
        bool f8B[kBPer];
        const unsigned char* bp[kBPer];
#pragma unroll
        for (int a = 0; a < kAPer; ++a) {
            const int idx = tid + a * kThreads;
            rrA[a] = idx / (kTKT / 8);
            k8A[a] = (idx % (kTKT / 8)) * 8;
            long r = r0 + rrA[a]; if (r >= rows) r = rows - 1;
            rA[a] = r;
        }
#pragma unroll
        for (int b = 0; b < kBPer; ++b) {
            const int idx = tid + b * kThreads;
            ttB[b] = idx / (kTKT / 8);
            k8B[b] = (idx % (kTKT / 8)) * 8;
            long col = c0 + ttB[b]; if (col >= ncol) col = ncol - 1;
            f8B[b] = F8 && col < lr;
            bp[b]  = f8B[b] ? (const unsigned char*)downv + col * dld
                            : (const unsigned char*)((col < lr) ? (const unsigned short*)downv + col * HN
                                                                : injw + (col - lr) * HN);
        }
        struct Pass {
            uint4 ah[kAPer];
            float aw[kAPer][8];
            float as[kAPer];
            uint4 bv[kBPer];
            float bs[kBPer];
        };
        auto load = [&](long k0, Pass& P) {
            const int c = (int)(k0 / n);
#pragma unroll
            for (int a = 0; a < kAPer; ++a) {
                P.ah[a] = *(const uint4*)(h + rA[a] * hrow + k0 + k8A[a]);
                const float* wp = w + k0 + k8A[a];
#pragma unroll
                for (int e = 0; e < 8; ++e) P.aw[a][e] = wp[e];
                P.as[a] = scalein[rA[a] * kHcMax + c];
            }
#pragma unroll
            for (int b = 0; b < kBPer; ++b) {
                const long k = k0 + k8B[b];
                if (f8B[b]) {
                    const uint2 cc = *(const uint2*)(bp[b] + k);
                    P.bv[b] = make_uint4(cc.x, cc.y, 0u, 0u);
                    P.bs[b] = ((const float*)(bp[b] + HN))[k / r4d_hc8::kDownGroup];
                } else {
                    P.bv[b] = *(const uint4*)((const unsigned short*)bp[b] + k);
                    P.bs[b] = 0.0f;
                }
            }
        };
        Pass cur, nxt;
        load(ka, cur);
        for (long k0 = ka; k0 < kb; k0 += kTKT) {
            /* The next pass's operands; after the last pass the first's again, in bounds. */
            load(k0 + kTKT < kb ? k0 + kTKT : ka, nxt);
            hc_lds_barrier();
#pragma unroll
            for (int a = 0; a < kAPer; ++a) {
                float f8v[8];
                bf8_f(cur.ah[a], f8v);
#pragma unroll
                for (int e = 0; e < 8; ++e) f8v[e] = f8v[e] * cur.as[a] * (cur.aw[a][e] + wadd);
                *(uint4*)(sA + rrA[a] * kTAP + k8A[a]) = f2bf8(f8v);
            }
#pragma unroll
            for (int b = 0; b < kBPer; ++b)
                *(uint4*)(sB + ttB[b] * kTAP + k8B[b]) =
                    f8B[b] ? e4m3x8_bf16v(make_uint2(cur.bv[b].x, cur.bv[b].y), cur.bs[b]) : cur.bv[b];
            hc_lds_barrier();
            mma();
            cur = nxt;
        }
    } else
    for (long k0 = ka; k0 < kb; k0 += kTKT) {
        const int c = (int)(k0 / n);
        __syncthreads();
        for (int idx = tid; idx < RT * (kTKT / 8); idx += kThreads) {
            const int rr = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long r = r0 + rr; if (r >= rows) r = rows - 1;
            const float sc = scalein[r * kHcMax + c];
            const float* wp = w + k0 + k8;
            float f8v[8];
            bf8_f(*(const uint4*)(h + r * hrow + k0 + k8), f8v);
#pragma unroll
            for (int e = 0; e < 8; ++e) f8v[e] = f8v[e] * sc * (wp[e] + wadd);
            *(uint4*)(sA + rr * kTAP + k8) = f2bf8(f8v);
        }
        for (int idx = tid; idx < NT * (kTKT / 8); idx += kThreads) {
            const int tt = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long col = c0 + tt; if (col >= ncol) col = ncol - 1;
            if (F8 && col < lr) {
                const unsigned char* row = (const unsigned char*)downv + col * dld;
                const long k = k0 + k8;
                *(uint4*)(sB + tt * kTAP + k8) =
                    e4m3x8_bf16(row + k, ((const float*)(row + HN))[k / r4d_hc8::kDownGroup]);
            } else {
                const unsigned short* src = (col < lr) ? (const unsigned short*)downv + col * HN
                                                       : injw + (col - lr) * HN;
                *(uint4*)(sB + tt * kTAP + k8) = *(const uint4*)(src + k0 + k8);
            }
        }
        __syncthreads();
        mma();
    }

    const float inv_hc = 1.0f / (float)hc;
    const int   e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int im = 0; im < AM; ++im) {
        const int mt = wave / WN + im * WM;
        if (mt >= MT) continue;
#pragma unroll
        for (int in = 0; in < AN; ++in) {
            const int nt = wave % WN + in * WN;
            if (nt >= NT16) continue;
            const long col = c0 + nt * 16 + nn;
            if (col >= ncol) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const long r = r0 + mt * 16 + e0 + e;
                if (r >= rows) continue;
                const float v = acc[im][in][e];
                if (part)          part[((long)bz * rows + r) * ncol + col] = v;
                else if (col < lr) dout[r * lr + col] = f2bf(siluf_(v * inv_hc));
                else               inj[r * irow + (col - lr)] = f2bf(2.0f * sigmoidf_(v * inv_hc));
            }
        }
    }

    /* ---- the split's finisher. With `ctr` the last K slice of a (row tile, column tile) to arrive
     * reduces it, as gemm_fp8a8's split finisher does, instead of a separate hc_down_reduce launch
     * (a dependent dispatch and its ~3 us of ordering a connection). The counter resets itself,
     * sound while two launches on its block never overlap (hc_mix_ctr). Each element is summed
     * from zero in ascending z with hc_down_reduce_kernel's loads and adds, so the bytes are that
     * kernel's. */
    if (part && ctr) {
        /* every thread's partials visible device-wide before the arrival is published */
        __threadfence();
        __syncthreads();
        __shared__ unsigned int s_arrived;
        const unsigned ci = by * gdx + bx;
        if (tid == 0)
            s_arrived = __hip_atomic_fetch_add(&ctr[ci], 1u, __ATOMIC_ACQ_REL,
                                               __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
        if (s_arrived + 1u == gdz) {
            if (tid == 0)
                __hip_atomic_store(&ctr[ci], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __threadfence();                       /* acquire the other slices' partials */
            const long nr  = (r0 + RT < rows) ? RT : rows - r0;
            const long nc  = (c0 + NT < ncol) ? NT : ncol - c0;
            const long tot = rows * ncol;
            const int  ksplit = (int)gdz;
            /* kFinE elements a thread at a time, four slices of each in flight: one block reduces
             * the whole tile, so a load a round (hc_down_reduce_kernel's pace, which a grid of
             * blocks shares) would hold the launch for a slice count of DRAM round trips. Each
             * element is still summed from zero in ascending z. */
            constexpr int kFinE = 8;
            for (long base = tid; base < nr * nc; base += (long)kThreads * kFinE) {
                long  i[kFinE];
                float sum[kFinE];
#pragma unroll
                for (int e = 0; e < kFinE; ++e) {
                    const long idx = base + (long)e * kThreads;
                    const long ix = idx < nr * nc ? idx : base;
                    i[e] = (r0 + ix / nc) * ncol + c0 + ix % nc;
                    sum[e] = 0.0f;
                }
                int z = 0;
                for (; z + 4 <= ksplit; z += 4) {
                    float v[kFinE][4];
#pragma unroll
                    for (int e = 0; e < kFinE; ++e)
#pragma unroll
                        for (int t = 0; t < 4; ++t) v[e][t] = part[(long)(z + t) * tot + i[e]];
#pragma unroll
                    for (int e = 0; e < kFinE; ++e)
#pragma unroll
                        for (int t = 0; t < 4; ++t) sum[e] += v[e][t];
                }
                for (; z < ksplit; ++z) {
#pragma unroll
                    for (int e = 0; e < kFinE; ++e) sum[e] += part[(long)z * tot + i[e]];
                }
#pragma unroll
                for (int e = 0; e < kFinE; ++e) {
                    if (base + (long)e * kThreads >= nr * nc) continue;
                    const long r = i[e] / ncol, col = i[e] - r * ncol;
                    if (col < lr) dout[r * lr + col] = f2bf(siluf_(sum[e] * inv_hc));
                    else          inj[r * irow + (col - lr)] = f2bf(2.0f * sigmoidf_(sum[e] * inv_hc));
                }
            }
        }
    }
}
template <int RT, int NT, bool F8>
__global__ __launch_bounds__(kThreads)
void hc_down_tiled_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                          const void* __restrict__ downv, long dld,
                          const float* __restrict__ scalein,
                          const unsigned short* __restrict__ injw,
                          unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
                          float* __restrict__ part,
                          long rows, long n, int hc, long lr, long ncol, long hrow, long irow,
                          long kc, float wadd, unsigned int* __restrict__ ctr) {
    hc_down_tiled_body<RT, NT, F8>(h, w, downv, dld, scalein, injw, dout, inj, part, rows, n, hc,
                                   lr, ncol, hrow, irow, kc, wadd, ctr, blockIdx.x, blockIdx.y,
                                   blockIdx.z, gridDim.x, gridDim.z);
}

/* The norm scales and the tiled down product as one grid: the down blocks need every (row, stream)
 * scale, so the two were two launches, the second a dependent dispatch. Roles go by ticket as in
 * hc_read_fused_kernel: tickets below `nn` take one (row, stream) scale each and count themselves
 * done in sync[1]; the rest are the down product's blocks (x fastest, then y, then z) and hold in
 * hc_role_wait until sync[1] reaches `nn`. Every block with a smaller ticket has started, so the
 * waiters never wait on a block that is not resident. Same bodies, same bytes. */
template <int RT, int NT, bool F8, class CH>
__global__ __launch_bounds__(kThreads)
void hc_norm_down_tiled_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                               const void* __restrict__ downv, long dld, float* scale,
                               const unsigned short* __restrict__ injw,
                               unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
                               float* __restrict__ part,
                               long rows, long n, int hc, long lr, long ncol, long hrow, long irow,
                               long kc, float wadd, unsigned int* __restrict__ ctr,
                               unsigned int* __restrict__ sync, float eps,
                               unsigned nn, unsigned gdx, unsigned gdy, unsigned gdz) {
    if (CH::on) r4d_chain_wait();
    __shared__ unsigned s_ticket;
    if (threadIdx.x == 0)
        s_ticket = __hip_atomic_fetch_add(&sync[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    const unsigned tk = s_ticket;
    if (tk < nn) {
        hc_norm_scale_body(h, scale, rows, n, hrow, eps, (long)(tk / (unsigned)hc),
                           (long)(tk % (unsigned)hc));
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
            __hip_atomic_fetch_add(&sync[1], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    } else {
        const unsigned b = tk - nn;
        hc_role_wait(sync, nn, gdx * gdy * gdz);
        hc_down_tiled_body<RT, NT, F8>(h, w, downv, dld, scale, injw, dout, inj, part, rows, n, hc,
                                       lr, ncol, hrow, irow, kc, wadd, ctr, b % gdx,
                                       (b / gdx) % gdy, b / (gdx * gdy), gdx, gdz);
    }
    if (CH::on) r4d_chain_done();
}

/* The split-K reduce: partials summed in ascending z (reproducible), then the epilogue: silu for a
 * W_down column, doubled sigmoid for an inject column. */
__global__ __launch_bounds__(kThreads)
void hc_down_reduce_kernel(const float* __restrict__ part, unsigned short* __restrict__ dout,
                           unsigned short* __restrict__ inj, long rows, long lr, long ncol,
                           long irow, int ksplit, float inv_hc) {
    const long i   = (long)blockIdx.x * kThreads + threadIdx.x;
    const long tot = rows * ncol;
    if (i >= tot) return;
    /* Eight loads before the adds: with one `s +=` per iteration a deep split becomes a dependent
     * chain of DRAM latency. The add order is unchanged (as in gemm_fp8a8's splitk_reduce_kernel). */
    float s = 0.0f;
    int   z = 0;
    for (; z + 8 <= ksplit; z += 8) {
        float v[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) v[t] = part[(long)(z + t) * tot + i];
#pragma unroll
        for (int t = 0; t < 8; ++t) s += v[t];
    }
    for (; z < ksplit; ++z) s += part[(long)z * tot + i];
    const long r = i / ncol, col = i - r * ncol;
    if (col < lr) dout[r * lr + col] = f2bf(siluf_(s * inv_hc));
    else          inj[r * irow + (col - lr)] = f2bf(2.0f * sigmoidf_(s * inv_hc));
}

/* The prefill tile: rows, down-product columns per workgroup and up-product channels per
 * workgroup. 112 columns is three tiles over 320 + 4, the fewest that fit the 128-row A tile and
 * the B tile in LDS twice per CU; faster than 64 and 128 at 129 to 4096 rows. */
constexpr int kWideRT = 128, kWideNT = 112, kWideCH = 16;

/* The prefill form, pinnable for tests: 0 the launcher's rule, 1 the decode tiles' kernels at
 * RT 128 (hc_down_tiled, then hc_up_tiled into `ubuf` and hc_mix), 2 the wide kernels (refusing
 * shapes they are not built for), 3 the PF2 form (hc_pf2_launch; refused likewise, and below
 * kPf2MinRows, where `u` is not sized for its weight planes). */
static int g_hc_prefill_form = 0;

/* ==================================================== d = silu(W_down @ N / hc) at the PREFILL tile
 *
 * hc_down_tiled_kernel at RT 128 has three costs at a 2048-row chunk: a sixth 64-column tile
 * rebuilds the whole A tile and runs every WMMA for four inject columns; every column tile rebuilds
 * the same N; and slice loads never overlap the WMMAs.
 *
 * This kernel takes NT columns per workgroup, skips n-tiles past the last real column, and issues
 * the next slice's global loads into registers before this slice's WMMAs. A thread stages one
 * fixed 8-wide column of four A rows, so the norm weight is loaded once per slice.
 *
 * Arithmetic and bytes match hc_down_tiled_kernel (same N, W_down decode, 16x16 tiles, K order and
 * hc_down_kchunk split; r4d_selftest checks). With hc_up_mix_wide_kernel, hc_read at
 * Qwen3.8-Flash-Next's shape takes 767 us against 1134 at 2048 rows. */
template <int NT, bool F8>
__global__ __launch_bounds__(kThreads)
void hc_down_wide_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                         const void* __restrict__ downv, long dld,
                         const float* __restrict__ scalein,
                         const unsigned short* __restrict__ injw,
                         unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
                         float* __restrict__ part,
                         long rows, long n, int hc, long lr, long ncol, long hrow, long irow,
                         long kc, float wadd) {
    constexpr int RT = kWideRT, NT16 = NT / 16;
    constexpr int kKV = kTKT / 8;                              /* 16-byte vectors a staged row */
    constexpr int kRS = kThreads / kKV;                        /* rows between a thread's A rows */
    constexpr int kAV = RT / kRS;
    constexpr int kBV = (NT * kKV + kThreads - 1) / kThreads;
    /* Waves tile the block 4 (rows) x 2 (columns), each holding two m-tiles and up to four
     * n-tiles: six LDS fragment reads per eight WMMA instead of eight per seven, so the block's
     * reads per k step fall from 64 fragments to 44. A wave's column half is `wave / kDWM`, so
     * waves w and w + 4 (one of each half) share a SIMD. */
    constexpr int kDWM = 4, kDTM = RT / 16 / kDWM, kDTN = (NT16 + 1) / 2;
    static_assert(kDWM * 2 == kWaves && kDTM * 16 * kDWM == RT && RT % kRS == 0,
                  "the waves cover the tile, whole A rows a pass");
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long HN = (long)hc * n;
    const long r0 = (long)blockIdx.y * RT;
    const long c0 = (long)blockIdx.x * NT;
    const long ka = (long)blockIdx.z * kc;
    const long kb = (ka + kc < HN) ? ka + kc : HN;
    if (r0 >= rows || c0 >= ncol || ka >= HN) return;
    const int nlive = (ncol - c0 >= NT) ? NT16 : (int)((ncol - c0 + 15) / 16);

    const int k8a = (tid % kKV) * 8, ar = tid / kKV;
    long arow[kAV];
#pragma unroll
    for (int i = 0; i < kAV; ++i) {
        const long r = r0 + ar + i * kRS;
        arow[i] = r < rows ? r : rows - 1;
    }

    /* The bf16-weight form stages through a native vector: in uint4 LLVM 24 keeps these arrays in
     * scratch. The fp8 form is promoted either way and keeps uint4, at its tuned register count. */
    using Stg = std::conditional_t<F8, uint4, r4d_u32x4>;
    Stg pa[kAV];
    float psc[kAV], pw[8];
    Stg pb[kBV];
    float pbs[kBV];
    auto fetch = [&](long k0) {
        const int c = (int)(k0 / n);
#pragma unroll
        for (int i = 0; i < kAV; ++i) {
            pa[i]  = *(const Stg*)(h + arow[i] * hrow + k0 + k8a);
            psc[i] = scalein[arow[i] * kHcMax + c];
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) pw[e] = w[k0 + k8a + e];
#pragma unroll
        for (int b = 0; b < kBV; ++b) {
            const int idx = tid + b * kThreads, tt = idx / kKV, k8 = (idx % kKV) * 8;
            if (idx >= NT * kKV || tt >= nlive * 16) continue;
            long col = c0 + tt; if (col >= ncol) col = ncol - 1;
            const long k = k0 + k8;
            if (F8 && col < lr) {
                const unsigned char* row = (const unsigned char*)downv + col * dld;
                const uint2 cc = *(const uint2*)(row + k);
                pb[b]  = Stg{cc.x, cc.y, 0u, 0u};
                pbs[b] = ((const float*)(row + HN))[k / r4d_hc8::kDownGroup];
            } else {
                const unsigned short* src = (col < lr) ? (const unsigned short*)downv + col * HN
                                                       : injw + (col - lr) * HN;
                pb[b] = *(const Stg*)(src + k);
            }
        }
    };
    auto stage = [&]() {
#pragma unroll
        for (int i = 0; i < kAV; ++i) {
            float f8v[8];
            bf8_f(pa[i], f8v);
#pragma unroll
            for (int e = 0; e < 8; ++e) f8v[e] = f8v[e] * psc[i] * (pw[e] + wadd);
            *(uint4*)(sA + (ar + i * kRS) * kTAP + k8a) = f2bf8(f8v);
        }
#pragma unroll
        for (int b = 0; b < kBV; ++b) {
            const int idx = tid + b * kThreads, tt = idx / kKV, k8 = (idx % kKV) * 8;
            if (idx >= NT * kKV || tt >= nlive * 16) continue;
            long col = c0 + tt; if (col >= ncol) col = ncol - 1;
            Stg v = pb[b];
            if constexpr (F8) {
                if (col < lr) {
                    float f[8];
                    e4m3x8(v.x, v.y, f);
#pragma unroll
                    for (int e = 0; e < 8; ++e) f[e] = f[e] * pbs[b];
                    v = f2bf8(f);
                }
            }
            *(Stg*)(sB + tt * kTAP + k8) = v;
        }
    };

    const int wm = wave % kDWM, nb = (wave / kDWM) * kDTN;     /* first m-tile / n-tile */
    v8f acc[kDTM][kDTN];
#pragma unroll
    for (int i = 0; i < kDTM; ++i)
#pragma unroll
        for (int j = 0; j < kDTN; ++j) acc[i][j] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    fetch(ka);
    for (long k0 = ka; k0 < kb; k0 += kTKT) {
        __syncthreads();
        stage();
        __syncthreads();
        if (k0 + kTKT < kb) fetch(k0 + kTKT);
        /* Each fragment is read once a k step and taken to every tile of the wave that uses it;
         * each accumulator still sees its k steps in ascending order. */
#pragma unroll
        for (int ks = 0; ks < kTKT / 16; ++ks) {
            v8bf af[kDTM];
#pragma unroll
            for (int i = 0; i < kDTM; ++i)
                af[i] = frag(sA, kTAP, (wm * kDTM + i) * 16, ks * 16, lane);
#pragma unroll
            for (int j = 0; j < kDTN; ++j) {
                if (nb + j >= nlive) continue;
                const v8bf bf = frag(sB, kTAP, (nb + j) * 16, ks * 16, lane);
#pragma unroll
                for (int i = 0; i < kDTM; ++i) acc[i][j] = mma(af[i], bf, acc[i][j]);
            }
        }
    }

    const float inv_hc = 1.0f / (float)hc;
    const int   e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int j = 0; j < kDTN; ++j) {
        if (nb + j >= nlive) continue;
        const long col = c0 + (nb + j) * 16 + nn;
        if (col >= ncol) continue;
#pragma unroll
        for (int i = 0; i < kDTM; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const long r = r0 + (wm * kDTM + i) * 16 + e0 + e;
                if (r >= rows) continue;
                const float v = acc[i][j][e];
                if (part)          part[((long)blockIdx.z * rows + r) * ncol + col] = v;
                else if (col < lr) dout[r * lr + col] = f2bf(siluf_(v * inv_hc));
                else               inj[r * irow + (col - lr)] = f2bf(2.0f * sigmoidf_(v * inv_hc));
            }
    }
}

/* Eight elements of row j of W_up at k as staged bf16: the stored bf16, or eight E4M3 codes decoded
 * with their group's scale. A group is a multiple of sixteen codes, so eight aligned codes never
 * straddle two groups. */
/* e4m3x8_bf16 on codes already in registers. */
__device__ __forceinline__ uint4 e4m3x8_bf16v(uint2 c, float sc) {
    float f[8];
    e4m3x8(c.x, c.y, f);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] *= sc;
    return f2bf8(f);
}

template <bool F8>
__device__ __forceinline__ uint4 hc_up_stage(const void* __restrict__ upv, long uld, long j, long k,
                                            long lr) {
    if constexpr (F8) {
        const unsigned char* row = (const unsigned char*)upv + j * uld;
        return e4m3x8_bf16(row + k, ((const float*)(row + lr))[k / r4d_hc8::up_group(lr)]);
    } else {
        return *(const uint4*)((const unsigned short*)upv + j * lr + k);
    }
}

/* ==================================================== u = sigmoid(W_up @ d), tiled
 *
 * C[RT rows, NT channels] over K = lowrank. Same tiling; the A tile is `d` straight from scratch
 * and the K loop is five passes. No K-split: hn/NT is already 160 workgroups. */
template <int RT, int NT, bool F8>
__global__ __launch_bounds__(kThreads)
void hc_up_tiled_kernel(const void* __restrict__ upv, long uld,
                        const unsigned short* __restrict__ dbuf,
                        unsigned short* __restrict__ ubuf, long rows, long hn, long lr) {
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];

    constexpr int MT = RT / 16, NT16 = NT / 16;
    constexpr int WM = (MT < kWaves) ? MT : kWaves, WN = kWaves / WM;
    constexpr int AM = (MT + WM - 1) / WM, AN = (NT16 + WN - 1) / WN;

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long r0 = (long)blockIdx.y * RT;
    const long j0 = (long)blockIdx.x * NT;
    if (r0 >= rows || j0 >= hn) return;

    v8f acc[AM][AN];
#pragma unroll
    for (int im = 0; im < AM; ++im)
#pragma unroll
        for (int in = 0; in < AN; ++in) acc[im][in] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (long k0 = 0; k0 < lr; k0 += kTKT) {
        __syncthreads();
        for (int idx = tid; idx < RT * (kTKT / 8); idx += kThreads) {
            const int rr = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long r = r0 + rr; if (r >= rows) r = rows - 1;
            const long k = k0 + k8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (k + 8 <= lr) v = *(const uint4*)(dbuf + r * lr + k);
            *(uint4*)(sA + rr * kTAP + k8) = v;
        }
        for (int idx = tid; idx < NT * (kTKT / 8); idx += kThreads) {
            const int tt = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long j = j0 + tt; if (j >= hn) j = hn - 1;
            const long k = k0 + k8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (k + 8 <= lr) v = hc_up_stage<F8>(upv, uld, j, k, lr);
            *(uint4*)(sB + tt * kTAP + k8) = v;
        }
        __syncthreads();
#pragma unroll
        for (int im = 0; im < AM; ++im) {
            const int mt = wave / WN + im * WM;
            if (mt >= MT) continue;
#pragma unroll
            for (int in = 0; in < AN; ++in) {
                const int nt = wave % WN + in * WN;
                if (nt >= NT16) continue;
                acc[im][in] = gemm<kTKT / 16>(sA, kTAP, mt, sB, kTAP, nt, lane, acc[im][in]);
            }
        }
    }

    const int e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int im = 0; im < AM; ++im) {
        const int mt = wave / WN + im * WM;
        if (mt >= MT) continue;
#pragma unroll
        for (int in = 0; in < AN; ++in) {
            const int nt = wave % WN + in * WN;
            if (nt >= NT16) continue;
            const long j = j0 + nt * 16 + nn;
            if (j >= hn) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const long r = r0 + mt * 16 + e0 + e;
                if (r < rows) ubuf[r * hn + j] = f2bf(sigmoidf_(acc[im][in][e]));
            }
        }
    }
}

/* ==================================== hc_read, kernels B and B2 in one launch, tiled
 *
 * `x = mean_c( u[c] * N[c] )` without materialising `u`. The tile is over (stream, channel) pairs,
 * so all hc terms of a channel's mean are in one workgroup (see the file header).
 *
 * The B tile becomes hc runs of NT/hc rows of W_up (same bytes), and a workgroup owns only
 * NT/hc = 16 channels of a 128-wide scale group, so an arrival counter lets the last of the eight
 * covering workgroups quantise the group from the bf16 the others wrote. That tail runs on the
 * same 20 workgroups hc_mix would use, so it costs less than the launch it saves.
 *
 * Bit-identical to the unfused pair: `u` and `N` are narrowed to bf16 where hc_mix would read them
 * back, both pass through LDS as bf16, and the sum over c ascends. */
#define HC_UMT_PARAMS                                                                            \
    const void* __restrict__ upv, long uld, const unsigned short* __restrict__ dbuf,             \
    const unsigned short* __restrict__ h, const float* __restrict__ w,                           \
    const float* __restrict__ scalein, unsigned short* __restrict__ x,                           \
    unsigned char* __restrict__ q, float* __restrict__ qs, unsigned char* __restrict__ rq,       \
    float* __restrict__ rqs, unsigned int* __restrict__ ctr, long rows, long n, int hc, long lr, \
    long hrow, long xrow, long qrow, long srow, long rqrow, long rsrow, float wadd
template <int RT, int NT, bool F8, bool I8 = false>
__device__ __forceinline__ void hc_up_mix_tiled_body(HC_UMT_PARAMS) {
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];
    __shared__ unsigned short s_u[RT * NT];
    __shared__ unsigned short s_N[RT * NT];
    __shared__ long           s_j[NT];

    constexpr int MT = RT / 16, NT16 = NT / 16;
    constexpr int WM = (MT < kWaves) ? MT : kWaves, WN = kWaves / WM;
    constexpr int AM = (MT + WM - 1) / WM, AN = (NT16 + WN - 1) / WN;

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const int  ipb = NT / hc;                       /* channels this workgroup owns */
    const long r0  = (long)blockIdx.y * RT;
    const long i0  = (long)blockIdx.x * ipb;
    if (r0 >= rows) return;

    /* The column map, once: tile column t is sub-stream t % hc of channel i0 + t / hc. */
    if (tid < NT) s_j[tid] = (long)(tid % hc) * n + i0 + tid / hc;
    __syncthreads();

    v8f acc[AM][AN];
#pragma unroll
    for (int im = 0; im < AM; ++im)
#pragma unroll
        for (int in = 0; in < AN; ++in) acc[im][in] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    /* The WMMA of one staged pass. */
    auto mma = [&]() {
#pragma unroll
        for (int im = 0; im < AM; ++im) {
            const int mt = wave / WN + im * WM;
            if (mt >= MT) continue;
#pragma unroll
            for (int in = 0; in < AN; ++in) {
                const int nt = wave % WN + in * WN;
                if (nt >= NT16) continue;
                acc[im][in] = gemm<kTKT / 16>(sA, kTAP, mt, sB, kTAP, nt, lane, acc[im][in]);
            }
        }
    };

    /* At the decode tile (RT 32, E4M3, lr a whole number of passes) the K loop is
     * software-pipelined: a pass issues the next pass's loads before staging its own from
     * registers, and the mix's operands are loaded with the first pass. Addresses are computed
     * once, nothing branches and the barriers order LDS only (hc_lds_barrier), so the waits stay
     * partial; the kernel is bound less by memory than by its staging, barriers and WMMA. The
     * staged values are the loop's own, in the same order: bit-identical. */
    constexpr int  kAPer = RT * (kTKT / 8) / kThreads, kBPer = NT * (kTKT / 8) / kThreads;
    constexpr int  kNPer = RT * NT / kThreads;
    constexpr bool kPipe = F8 && RT == 32 && kAPer == 1 && kBPer * kThreads == NT * (kTKT / 8)
                           && kNPer * kThreads == RT * NT;
    const bool pipe = kPipe && lr % kTKT == 0;
    float nh[kPipe ? kNPer : 1], nw[kPipe ? kNPer : 1], nsc[kPipe ? kNPer : 1];
    if constexpr (kPipe) {
        if (pipe) {
            const int nit = (int)(lr / kTKT), G = (int)r4d_hc8::up_group(lr);
            const int rrA = tid / (kTKT / 8), k8A = (tid % (kTKT / 8)) * 8;
            long rA = r0 + rrA;
            if (rA >= rows) rA = rows - 1;
            const unsigned short* const ap = dbuf + rA * lr + k8A;
            const unsigned char* bp[kBPer];
            int k8B[kBPer], ttB[kBPer];
#pragma unroll
            for (int b = 0; b < kBPer; ++b) {
                const int idx = tid + b * kThreads;
                ttB[b] = idx / (kTKT / 8);
                k8B[b] = (idx % (kTKT / 8)) * 8;
                bp[b]  = (const unsigned char*)upv + s_j[ttB[b]] * uld;
            }
            uint4 ca = *(const uint4*)ap;
            uint2 cb[kBPer];
            float cs[kBPer];
#pragma unroll
            for (int b = 0; b < kBPer; ++b) {
                cb[b] = *(const uint2*)(bp[b] + k8B[b]);
                cs[b] = ((const float*)(bp[b] + lr))[k8B[b] / G];
            }
#pragma unroll
            for (int q2 = 0; q2 < kNPer; ++q2) {
                const int  idx = tid + q2 * kThreads;
                const int  rl = idx / NT, rem = idx - rl * NT;
                const int  c  = rem / ipb, ii = rem - c * ipb;
                long r = r0 + rl;
                if (r >= rows) r = rows - 1;
                const long j = (long)c * n + i0 + ii;
                nh[q2]  = bf2f(h[r * hrow + j]);
                nsc[q2] = scalein[r * kHcMax + c];
                nw[q2]  = w[j];
            }
            for (int it = 0; it < nit; ++it) {
                /* The next pass's operands; after the last pass the first's again, in bounds. */
                const int kn = it + 1 < nit ? (it + 1) * kTKT : 0;
                const uint4 na = *(const uint4*)(ap + kn);
                uint2 nb[kBPer];
                float ns[kBPer];
#pragma unroll
                for (int b = 0; b < kBPer; ++b) {
                    nb[b] = *(const uint2*)(bp[b] + kn + k8B[b]);
                    ns[b] = ((const float*)(bp[b] + lr))[(kn + k8B[b]) / G];
                }
                hc_lds_barrier();
                *(uint4*)(sA + rrA * kTAP + k8A) = ca;
#pragma unroll
                for (int b = 0; b < kBPer; ++b)
                    *(uint4*)(sB + ttB[b] * kTAP + k8B[b]) = e4m3x8_bf16v(cb[b], cs[b]);
                hc_lds_barrier();
                mma();
                ca = na;
#pragma unroll
                for (int b = 0; b < kBPer; ++b) {
                    cb[b] = nb[b];
                    cs[b] = ns[b];
                }
            }
        }
    }

    if (!pipe)
    for (long k0 = 0; k0 < lr; k0 += kTKT) {
        __syncthreads();
        for (int idx = tid; idx < RT * (kTKT / 8); idx += kThreads) {
            const int rr = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long r = r0 + rr; if (r >= rows) r = rows - 1;
            const long ka = k0 + k8;
            uint4 va = make_uint4(0u, 0u, 0u, 0u);
            if (ka + 8 <= lr) va = *(const uint4*)(dbuf + r * lr + ka);
            *(uint4*)(sA + rr * kTAP + k8) = va;
        }
        for (int idx = tid; idx < NT * (kTKT / 8); idx += kThreads) {
            const int tt = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            const long k = k0 + k8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (k + 8 <= lr) v = hc_up_stage<F8>(upv, uld, s_j[tt], k, lr);
            *(uint4*)(sB + tt * kTAP + k8) = v;
        }
        __syncthreads();
        mma();
    }
    /* N in a separate coalesced pass: in the WMMA epilogue consecutive lanes hold consecutive
     * sub-streams, so reading `h` there would stride n elements per lane. Here consecutive threads
     * take consecutive channels of one sub-stream, as in hc_mix. */
    if (pipe) {
#pragma unroll
        for (int q2 = 0; q2 < (kPipe ? kNPer : 1); ++q2) {
            const int  idx = tid + q2 * kThreads;
            const int  rl = idx / NT, rem = idx - rl * NT;
            const int  c  = rem / ipb, ii = rem - c * ipb;
            if (r0 + rl < rows) s_N[rl * NT + ii * hc + c] = f2bf(nh[q2] * nsc[q2] * (nw[q2] + wadd));
        }
    } else
    for (int idx = tid; idx < RT * NT; idx += kThreads) {
        const int  rl = idx / NT, rem = idx - rl * NT;
        const int  c  = rem / ipb, ii = rem - c * ipb;
        const long r  = r0 + rl;
        if (r >= rows) continue;
        const long j = (long)c * n + i0 + ii;
        s_N[rl * NT + ii * hc + c] =
            f2bf(bf2f(h[r * hrow + j]) * scalein[r * kHcMax + c] * (w[j] + wadd));
    }

    /* u from the accumulators. */
    const int e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int im = 0; im < AM; ++im) {
        const int mt = wave / WN + im * WM;
        if (mt >= MT) continue;
#pragma unroll
        for (int in = 0; in < AN; ++in) {
            const int nt = wave % WN + in * WN;
            if (nt >= NT16) continue;
            const int  t = nt * 16 + nn;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const long r = r0 + mt * 16 + e0 + e;
                if (r >= rows) continue;
                s_u[(r - r0) * NT + t] = f2bf(sigmoidf_(acc[im][in][e]));
            }
        }
    }
    __syncthreads();

    const float inv_hc = 1.0f / (float)hc;
    for (int idx = tid; idx < RT * ipb; idx += kThreads) {
        const int  rl = idx / ipb, ii = idx - rl * ipb;
        const long r  = r0 + rl;
        if (r >= rows) continue;
        const unsigned short* const up_ = s_u + rl * NT + ii * hc;
        const unsigned short* const np_ = s_N + rl * NT + ii * hc;
        float a = 0.0f;
        for (int c = 0; c < hc; ++c) a += bf2f(up_[c]) * bf2f(np_[c]);
        x[r * xrow + i0 + ii] = f2bf(a * inv_hc);
    }
    if (!q || !ctr) return;

    /* The group's last workgroup quantises it; `nb` workgroups cover one 128-channel group. */
    const int nb = kUpTB / ipb;
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_arrived;
    const unsigned slot = blockIdx.y * (gridDim.x / nb) + (unsigned)(i0 / kUpTB);
    if (tid == 0)
        s_arrived = __hip_atomic_fetch_add(&ctr[slot], 1u, __ATOMIC_ACQ_REL,
                                           __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived + 1u != (unsigned)nb) return;
    if (tid == 0)
        __hip_atomic_store(&ctr[slot], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();

    /* The rotated codes need a whole wave per row, while the codes below use kThreads / RT threads
     * per row, so the rotation walks the tile's rows a wave at a time first. */
    /* Both finishers below load what they read before using any of it (the rotation every row
     * of the wave's, the codes a thread's four-channel runs) rather than one dependent load at a
     * time, since the tail of this kernel is one block's latency chain. The codes' amax is over
     * the same 128 values, so the result is bit-identical. */
    if (rq) {
        const long gb = (i0 / kUpTB) * kUpTB;
        constexpr int kRPW = (RT + kWaves - 1) / kWaves;   /* rows a wave */
        float v[kRPW][4];
#pragma unroll
        for (int i = 0; i < kRPW; ++i) {
            const long r = r0 + wave + i * kWaves;
            const unsigned short* const xr = x + (r < rows ? r : rows - 1) * xrow + gb;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[i][j] = bf2f(xr[lane * 4 + j]);
        }
#pragma unroll
        for (int i = 0; i < kRPW; ++i) {
            const long r = r0 + wave + i * kWaves;
            if (wave + i * kWaves >= RT || r >= rows) break;
            hc_rot_group<I8>(v[i], lane, rq + r * rqrow + gb, rqs + r * rsrow + gb / kUpTB);
        }
    }

    constexpr int kTPR = kThreads / RT;             /* threads a row; RT divides kThreads */
    constexpr int kRuns = (kUpTB / 4) / kTPR;       /* a thread's four-channel runs */
    const int  rl = tid / kTPR, lr_ = tid % kTPR;
    const long r  = r0 + rl;
    if (r >= rows) return;
    const long base = (i0 / kUpTB) * kUpTB;
    const unsigned short* const xr = x + r * xrow + base;
    float xv[kRuns][4];
#pragma unroll
    for (int u = 0; u < kRuns; ++u)
#pragma unroll
        for (int j = 0; j < 4; ++j) xv[u][j] = bf2f(xr[(lr_ + u * kTPR) * 4 + j]);
    float m = 0.0f;
#pragma unroll
    for (int u = 0; u < kRuns; ++u)
#pragma unroll
        for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(xv[u][j]));
    m = r4d_xor_max<kTPR>(m);
    const float sc  = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lr_ == 0) qs[r * srow + base / kUpTB] = sc;
#pragma unroll
    for (int u = 0; u < kRuns; ++u) {
        const int c4 = (lr_ + u * kTPR) * 4;
        const uint32_t pk = q8_pack4<I8>(xv[u][0], xv[u][1], xv[u][2], xv[u][3], inv);
        *(unsigned*)(q + r * qrow + base + c4) = (unsigned)pk;
    }
}

/* The kernel around the body, whose workgroups and threads return where their work ends: CH
 * `r4dchain` is the chained form (r4d_chain.h). */
template <int RT, int NT, bool F8, bool I8, class CH>
__global__ __launch_bounds__(kThreads)
void hc_up_mix_tiled_kernel(HC_UMT_PARAMS) {
    if (CH::on) r4d_chain_wait();
    hc_up_mix_tiled_body<RT, NT, F8, I8>(upv, uld, dbuf, h, w, scalein, x, q, qs, rq, rqs, ctr,
                                         rows, n, hc, lr, hrow, xrow, qrow, srow, rqrow, rsrow,
                                         wadd);
    if (CH::on) r4d_chain_done();
}
#undef HC_UMT_PARAMS

/* The tiled up-mix with its K passes unpipelined, for r4d_hc_read_pin_decode(1) (see the
 * two-grid decode kernels above). */
template <int RT, int NT, bool F8, bool I8 = false>
__global__ __launch_bounds__(kThreads)
void hc_up_mix_tiled_2grid_kernel(const void* __restrict__ upv, long uld,
                            const unsigned short* __restrict__ dbuf,
                            const unsigned short* __restrict__ h, const float* __restrict__ w,
                            const float* __restrict__ scalein,
                            unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                            float* __restrict__ qs, unsigned char* __restrict__ rq,
                            float* __restrict__ rqs, unsigned int* __restrict__ ctr,
                            long rows, long n, int hc, long lr, long hrow, long xrow,
                            long qrow, long srow, long rqrow, long rsrow, float wadd) {
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];
    __shared__ unsigned short s_u[RT * NT];
    __shared__ unsigned short s_N[RT * NT];
    __shared__ long           s_j[NT];

    constexpr int MT = RT / 16, NT16 = NT / 16;
    constexpr int WM = (MT < kWaves) ? MT : kWaves, WN = kWaves / WM;
    constexpr int AM = (MT + WM - 1) / WM, AN = (NT16 + WN - 1) / WN;

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const int  ipb = NT / hc;                       /* channels this workgroup owns */
    const long r0  = (long)blockIdx.y * RT;
    const long i0  = (long)blockIdx.x * ipb;
    if (r0 >= rows) return;

    /* The column map, once: tile column t is sub-stream t % hc of channel i0 + t / hc. */
    if (tid < NT) s_j[tid] = (long)(tid % hc) * n + i0 + tid / hc;
    __syncthreads();

    v8f acc[AM][AN];
#pragma unroll
    for (int im = 0; im < AM; ++im)
#pragma unroll
        for (int in = 0; in < AN; ++in) acc[im][in] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (long k0 = 0; k0 < lr; k0 += kTKT) {
        __syncthreads();
        for (int idx = tid; idx < RT * (kTKT / 8); idx += kThreads) {
            const int rr = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long r = r0 + rr; if (r >= rows) r = rows - 1;
            const long ka = k0 + k8;
            uint4 va = make_uint4(0u, 0u, 0u, 0u);
            if (ka + 8 <= lr) va = *(const uint4*)(dbuf + r * lr + ka);
            *(uint4*)(sA + rr * kTAP + k8) = va;
        }
        for (int idx = tid; idx < NT * (kTKT / 8); idx += kThreads) {
            const int tt = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            const long k = k0 + k8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (k + 8 <= lr) v = hc_up_stage<F8>(upv, uld, s_j[tt], k, lr);
            *(uint4*)(sB + tt * kTAP + k8) = v;
        }
        __syncthreads();
#pragma unroll
        for (int im = 0; im < AM; ++im) {
            const int mt = wave / WN + im * WM;
            if (mt >= MT) continue;
#pragma unroll
            for (int in = 0; in < AN; ++in) {
                const int nt = wave % WN + in * WN;
                if (nt >= NT16) continue;
                acc[im][in] = gemm<kTKT / 16>(sA, kTAP, mt, sB, kTAP, nt, lane, acc[im][in]);
            }
        }
    }
    /* N in a separate coalesced pass: in the WMMA epilogue consecutive lanes hold consecutive
     * sub-streams, so reading `h` there would stride n elements per lane. Here consecutive threads
     * take consecutive channels of one sub-stream, as in hc_mix. */
    for (int idx = tid; idx < RT * NT; idx += kThreads) {
        const int  rl = idx / NT, rem = idx - rl * NT;
        const int  c  = rem / ipb, ii = rem - c * ipb;
        const long r  = r0 + rl;
        if (r >= rows) continue;
        const long j = (long)c * n + i0 + ii;
        s_N[rl * NT + ii * hc + c] =
            f2bf(bf2f(h[r * hrow + j]) * scalein[r * kHcMax + c] * (w[j] + wadd));
    }

    /* u from the accumulators. */
    const int e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int im = 0; im < AM; ++im) {
        const int mt = wave / WN + im * WM;
        if (mt >= MT) continue;
#pragma unroll
        for (int in = 0; in < AN; ++in) {
            const int nt = wave % WN + in * WN;
            if (nt >= NT16) continue;
            const int  t = nt * 16 + nn;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const long r = r0 + mt * 16 + e0 + e;
                if (r >= rows) continue;
                s_u[(r - r0) * NT + t] = f2bf(sigmoidf_(acc[im][in][e]));
            }
        }
    }
    __syncthreads();

    const float inv_hc = 1.0f / (float)hc;
    for (int idx = tid; idx < RT * ipb; idx += kThreads) {
        const int  rl = idx / ipb, ii = idx - rl * ipb;
        const long r  = r0 + rl;
        if (r >= rows) continue;
        const unsigned short* const up_ = s_u + rl * NT + ii * hc;
        const unsigned short* const np_ = s_N + rl * NT + ii * hc;
        float a = 0.0f;
        for (int c = 0; c < hc; ++c) a += bf2f(up_[c]) * bf2f(np_[c]);
        x[r * xrow + i0 + ii] = f2bf(a * inv_hc);
    }
    if (!q || !ctr) return;

    /* The group's last workgroup quantises it; `nb` workgroups cover one 128-channel group. */
    const int nb = kUpTB / ipb;
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_arrived;
    const unsigned slot = blockIdx.y * (gridDim.x / nb) + (unsigned)(i0 / kUpTB);
    if (tid == 0)
        s_arrived = __hip_atomic_fetch_add(&ctr[slot], 1u, __ATOMIC_ACQ_REL,
                                           __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived + 1u != (unsigned)nb) return;
    if (tid == 0)
        __hip_atomic_store(&ctr[slot], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();

    /* The rotated codes need a whole wave per row, while the codes below use kThreads / RT threads
     * per row, so the rotation walks the tile's rows a wave at a time first. */
    if (rq) {
        const long gb = (i0 / kUpTB) * kUpTB;
        for (int rl = wave; rl < RT; rl += kWaves) {
            const long r = r0 + rl;
            if (r >= rows) break;
            const unsigned short* const xr = x + r * xrow + gb;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = bf2f(xr[lane * 4 + j]);
            hc_rot_group<I8>(v, lane, rq + r * rqrow + gb, rqs + r * rsrow + gb / kUpTB);
        }
    }

    constexpr int kTPR = kThreads / RT;             /* threads a row; RT divides kThreads */
    const int  rl = tid / kTPR, lr_ = tid % kTPR;
    const long r  = r0 + rl;
    if (r >= rows) return;
    const long base = (i0 / kUpTB) * kUpTB;
    const unsigned short* const xr = x + r * xrow + base;
    float m = 0.0f;
    for (int t = lr_; t < kUpTB; t += kTPR) m = fmaxf(m, fabsf(bf2f(xr[t])));
    m = r4d_xor_max<kTPR>(m);
    const float sc  = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lr_ == 0) qs[r * srow + base / kUpTB] = sc;
    for (int d = lr_; d < kUpTB / 4; d += kTPR) {
        const int c4 = d * 4;
        const uint32_t pk = q8_pack4<I8>(bf2f(xr[c4 + 0]), bf2f(xr[c4 + 1]), bf2f(xr[c4 + 2]),
                                         bf2f(xr[c4 + 3]), inv);
        *(unsigned*)(q + r * qrow + base + c4) = (unsigned)pk;
    }
}

/* ==================================== hc_read, kernels B and B2 in one launch at the prefill tile
 *
 * At RT 128 hc_up_mix_tiled_kernel's two staging planes plus the A tile take 58 KiB of LDS (one
 * workgroup per CU), and the unfused pair moves 42 MB through `ubuf` each way per call at a
 * 2048-row chunk, plus a third launch.
 *
 * Neither plane is needed when a lane holds every sub-stream of a channel: a WMMA accumulator gives
 * lane `l` column `l & 15` of each 16-column n-tile, so with n-tile `c` holding sub-stream c of the
 * workgroup's sixteen channels, a channel's hc terms are one lane's `acc[c]`. N is staged once,
 * coalesced, into the A tile's LDS after the K loop. LDS use equals the unfused up product's.
 *
 * Bit-identical to the unfused pair: `u` and N are narrowed to bf16 as hc_mix reads them, and the
 * sum over c ascends in the same `a += u * N` form. fp8 codes come via the arrival counter, as in
 * hc_up_mix_tiled_kernel.
 *
 * MW = RT / 128 m-tiles per wave: 128 here (one), 256 in the PF2 form over its decoded mix_up (two,
 * so each B fragment read feeds two WMMAs). The accumulators are one flat array, acc[mi * HC + c],
 * because an [MW][HC] array changes gfx1201's register allocation for the 128-row form. Rows keep
 * their place in a 16-row tile at either RT, so both produce the same bytes. */
template <int HC, bool F8, int RT = kWideRT>
__global__ __launch_bounds__(kThreads)
void hc_up_mix_wide_kernel(const void* __restrict__ upv, long uld,
                           const unsigned short* __restrict__ dbuf,
                           const unsigned short* __restrict__ h, const float* __restrict__ w,
                           const float* __restrict__ scalein,
                           unsigned short* __restrict__ x,
                           long rows, long n, long lr, long hrow, long xrow, float wadd) {
    constexpr int CH = kWideCH, NT = HC * CH;
    constexpr int MW = RT / 16 / kWaves;                        /* m-tiles a wave */
    static_assert(MW >= 1 && MW * 16 * kWaves == RT, "the waves cover the rows");
    static_assert(RT * NT <= RT * kTAP, "N must fit in the A tile's LDS");
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long r0 = (long)blockIdx.y * RT;
    const long i0 = (long)blockIdx.x * CH;
    if (r0 >= rows) return;

    /* Every global load is issued a step early. The mix's `h` (42 MB per call at a 2048-row chunk,
     * needed only after the product) is loaded into registers before the K loop so its latency
     * hides under the matrix work, and the loop holds the next slice's A rows and weight codes in
     * registers during this slice's WMMA, as in hc_down_wide_kernel. Output is unchanged.
     *
     * A thread's slots are fixed: A and B rows `tid / 8 + 32 i` at column `(tid % 8) * 8` of a
     * slice, and for N the (row, sub-stream, eight channels) that idx `tid + 256 i` names. */
    constexpr int kKV = kTKT / 8;                               /* 16-byte vectors a staged row */
    constexpr int kRS = kThreads / kKV;                         /* rows between a thread's rows */
    constexpr int kAV = RT / kRS, kBV = NT / kRS;
    constexpr int kNV = RT * HC * (CH / 8) / kThreads;
    static_assert(RT % kRS == 0 && NT % kRS == 0 && (RT * HC * (CH / 8)) % kThreads == 0,
                  "every thread stages whole vectors");
    const int k8a = (tid % kKV) * 8, tr = tid / kKV;

    uint4 pn[kNV];
    float pns[kNV];
#pragma unroll
    for (int i = 0; i < kNV; ++i) {
        const int  idx = tid + i * kThreads;
        const int  rl = idx / (HC * CH / 8), rem = idx % (HC * CH / 8);
        const int  c  = rem / (CH / 8), i8 = (rem % (CH / 8)) * 8;
        const long r  = r0 + rl;
        pn[i] = make_uint4(0u, 0u, 0u, 0u);
        pns[i] = 0.0f;
        if (r < rows) {
            pn[i]  = *(const uint4*)(h + r * hrow + (long)c * n + i0 + i8);
            pns[i] = scalein[r * kHcMax + c];
        }
    }

    uint4 pa[kAV], pb[kBV];
    float pbs[kBV];
    auto fetch = [&](long k0) {
        const long k = k0 + k8a;
        const bool in = k + 8 <= lr;
#pragma unroll
        for (int i = 0; i < kAV; ++i) {
            long r = r0 + tr + i * kRS; if (r >= rows) r = rows - 1;
            pa[i] = in ? *(const uint4*)(dbuf + r * lr + k) : make_uint4(0u, 0u, 0u, 0u);
        }
        /* Tile row tt is sub-stream tt / CH of channel i0 + tt % CH. */
#pragma unroll
        for (int i = 0; i < kBV; ++i) {
            const int  tt = tr + i * kRS;
            const long j  = (long)(tt / CH) * n + i0 + tt % CH;
            pb[i] = make_uint4(0u, 0u, 0u, 0u);
            pbs[i] = 0.0f;
            if (!in) continue;
            if constexpr (F8) {
                const unsigned char* row = (const unsigned char*)upv + j * uld;
                const uint2 cc = *(const uint2*)(row + k);
                pb[i]  = make_uint4(cc.x, cc.y, 0u, 0u);
                pbs[i] = ((const float*)(row + lr))[k / r4d_hc8::up_group(lr)];
            } else {
                pb[i] = *(const uint4*)((const unsigned short*)upv + j * lr + k);
            }
        }
    };

    v8f acc[MW * HC];
#pragma unroll
    for (int c = 0; c < MW * HC; ++c) acc[c] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    fetch(0);
    for (long k0 = 0; k0 < lr; k0 += kTKT) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kAV; ++i) *(uint4*)(sA + (tr + i * kRS) * kTAP + k8a) = pa[i];
#pragma unroll
        for (int i = 0; i < kBV; ++i) {
            uint4 v = pb[i];
            if constexpr (F8) {
                float f[8];
                e4m3x8(v.x, v.y, f);
#pragma unroll
                for (int e = 0; e < 8; ++e) f[e] = f[e] * pbs[i];
                v = k0 + k8a + 8 <= lr ? f2bf8(f) : make_uint4(0u, 0u, 0u, 0u);
            }
            *(uint4*)(sB + (tr + i * kRS) * kTAP + k8a) = v;
        }
        __syncthreads();
        if (k0 + kTKT < lr) fetch(k0 + kTKT);
        /* A's fragments are read once per k step and used for every n-tile, B's for every m-tile.
         * Each accumulator still sees its k steps in ascending order. */
#pragma unroll
        for (int ks = 0; ks < kTKT / 16; ++ks) {
            v8bf af[MW];
#pragma unroll
            for (int mi = 0; mi < MW; ++mi)
                af[mi] = frag(sA, kTAP, (wave * MW + mi) * 16, ks * 16, lane);
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const v8bf bf = frag(sB, kTAP, c * 16, ks * 16, lane);
#pragma unroll
                for (int mi = 0; mi < MW; ++mi)
                    acc[mi * HC + c] = mma(af[mi], bf, acc[mi * HC + c]);
            }
        }
    }
    __syncthreads();

    /* N for the tile, [RT][HC][CH] bf16 in the A tile's LDS, from the `h` loaded before the loop. */
    unsigned short* const sN = sA;
#pragma unroll
    for (int i = 0; i < kNV; ++i) {
        const int  idx = tid + i * kThreads;
        const int  rl = idx / (HC * CH / 8), rem = idx % (HC * CH / 8);
        const int  c  = rem / (CH / 8), i8 = (rem % (CH / 8)) * 8;
        if (r0 + rl >= rows) continue;
        const long j  = (long)c * n + i0 + i8;
        float f8v[8];
        bf8_f(pn[i], f8v);
#pragma unroll
        for (int e = 0; e < 8; ++e) f8v[e] = f8v[e] * pns[i] * (w[j + e] + wadd);
        *(uint4*)(sN + (rl * HC + c) * CH + i8) = f2bf8(f8v);
    }
    __syncthreads();

    const float inv_hc = 1.0f / (float)HC;
    const int   e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int mi = 0; mi < MW; ++mi)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int  rl = (wave * MW + mi) * 16 + e0 + e;
            const long r  = r0 + rl;
            float a = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c)
                a += bf2f(f2bf(sigmoidf_(acc[mi * HC + c][e]))) * bf2f(sN[(rl * HC + c) * CH + nn]);
            if (r < rows) x[r * xrow + i0 + nn] = f2bf(a * inv_hc);
        }
}

/* fp8 codes of `x` at the prefill tile, plus the rotated codes: one wave per (row, 128-channel
 * scale group), after hc_up_mix_wide_kernel. Quantising inside that kernel would leave the last of
 * eight workgroups walking the group's 128 rows alone behind a counter and fences (82 of its
 * 356 us at a 2048-row chunk); this launch streams over `x` at memory speed. Same bytes: the amax
 * is exact in any order and each code converts the same bf16 by the same scale. */
template <bool I8 = false>
__global__ __launch_bounds__(kThreads)
void hc_quant_rows_kernel(const unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                          float* __restrict__ qs, unsigned char* __restrict__ rq,
                          float* __restrict__ rqs, long rows, long ngrp, long xrow, long qrow,
                          long srow, long rqrow, long rsrow) {
    const int  lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const long item = (long)blockIdx.x * kWaves + wave;
    if (item >= rows * ngrp) return;
    const long r = item / ngrp, g = item % ngrp, base = g * kUpTB;
    const uint2 raw = *(const uint2*)(x + r * xrow + base + lane * 4);
    const unsigned short* xs = reinterpret_cast<const unsigned short*>(&raw);
    float v[4], t[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] = bf2f(xs[j]); t[j] = v[j]; }

    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
    m = r4d_wave_max(m);
    const float sc  = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) qs[r * srow + g] = sc;
    const uint32_t pk = q8_pack4<I8>(v[0], v[1], v[2], v[3], inv);
    *(unsigned*)(q + r * qrow + base + lane * 4) = (unsigned)pk;

    if (rq) hc_rot_group<I8>(t, lane, rq + r * rqrow + base, rqs + r * rsrow + g);
}

/* The mix finisher's arrival counters, one block per stream (as gemm_fp8a8's split_ctr): a
 * self-resetting counter is sound only while two launches of its kernel cannot overlap, which one
 * stream guarantees. Allocation failure is not an error; the caller runs the unfused pair. */
constexpr unsigned kMaxMixCtr = 2048;

unsigned int* hc_mix_ctr(RadStream s) {
    /* A lock a device: zeroing a block waits for the card, and under tensor parallelism the card
     * can be inside an all-reduce whose other half the peer rank has yet to issue, so a peer's
     * first call must not wait behind this card's (see split_ctr). */
    static std::mutex m[16];
    struct Slot { std::atomic<RadStream> key{nullptr}; std::atomic<unsigned int*> ctr{nullptr}; };
    static Slot slot[16][4];
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16) return nullptr;
    for (int i = 0; i < 4; ++i)
        if (slot[dev][i].key.load(std::memory_order_acquire) == s)
            return slot[dev][i].ctr.load(std::memory_order_acquire);
    std::lock_guard<std::mutex> lk(m[dev]);
    for (int i = 0; i < 4; ++i)
        if (slot[dev][i].key.load(std::memory_order_relaxed) == s)
            return slot[dev][i].ctr.load(std::memory_order_relaxed);
    int fi = -1;
    for (int i = 0; i < 4; ++i)
        if (slot[dev][i].ctr.load(std::memory_order_relaxed) == nullptr) { fi = i; break; }
    if (fi < 0) return nullptr;
    void* p = nullptr;
    if (hipMalloc(&p, (size_t)(kMaxMixCtr + kHcSyncCtr) * sizeof(unsigned int)) != hipSuccess)
        return nullptr;
    /* Zero before any launch reads it. In the engine hipMemset is the AQL backend's and has
     * finished when it returns (core/device/aql.cpp); under HIP's own runtime (r4d_selftest) it
     * runs on the null stream, which a non-blocking stream does not wait for, so wait here. */
    if (hipMemset(p, 0, (size_t)(kMaxMixCtr + kHcSyncCtr) * sizeof(unsigned int)) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) {
        (void)hipFree(p);
        return nullptr;
    }
    slot[dev][fi].ctr.store((unsigned int*)p, std::memory_order_relaxed);
    slot[dev][fi].key.store(s, std::memory_order_release);
    return (unsigned int*)p;
}

static inline void hc_up_mix_tiled_launch(int rt, bool f8, bool i8, const void* up, long uld,
                                          const unsigned short* dbuf, const unsigned short* h,
                                          const float* w, const float* scale, unsigned short* x,
                                          unsigned char* q, float* qs, unsigned char* rq,
                                          float* rqs, unsigned int* ctr,
                                          long rows, long n, int hc, long lr, long hrow,
                                          long xrow, long qrow, long srow, long rqrow, long rsrow,
                                          float wadd, RadStream s, bool chain) {
    const dim3 gm((unsigned)(n / (kTNT / hc)), (unsigned)((rows + rt - 1) / rt));
    /* Through r4d_emit, so r4d_record_dispatches sees which form ran. */
#define HC_UM_K(K_)                                                                               \
    (void)R4D_EMIT(s, K_, gm, dim3(kThreads), 0, up, uld, dbuf, h, w, scale, x, q, qs, rq, rqs, ctr, \
                   rows, n, hc, lr, hrow, xrow, qrow, srow, rqrow, rsrow, wadd)
#define HC_UM_CASE3(RT_, F8_, I8_)                                                               \
    do {                                                                                         \
        if (g_hc_decode_form == 1) HC_UM_K((hc_up_mix_tiled_2grid_kernel<RT_, kTNT, F8_, I8_>));   \
        else if (chain) HC_UM_K((hc_up_mix_tiled_kernel<RT_, kTNT, F8_, I8_, r4dchain>));        \
        else            HC_UM_K((hc_up_mix_tiled_kernel<RT_, kTNT, F8_, I8_, R4dUnchained>));    \
    } while (0)
#define HC_UM_CASE(RT_, F8_)                                                                     \
    do { if (i8) HC_UM_CASE3(RT_, F8_, true); else HC_UM_CASE3(RT_, F8_, false); } while (0)
    if (f8) { if (rt == 32) HC_UM_CASE(32, true);  else HC_UM_CASE(64, true); }
    else    { if (rt == 32) HC_UM_CASE(32, false); else HC_UM_CASE(64, false); }
#undef HC_UM_CASE
#undef HC_UM_CASE3
#undef HC_UM_K
}

/* The codes of `x` and the rotated codes, after a mix that writes only `x`: hc_quant_rows_kernel
 * over every (row, scale group). No-op without codes. */
static inline void hc_quant_rows_launch(bool i8, const unsigned short* x, unsigned char* q,
                                        float* qs, unsigned char* rq, float* rqs, long rows,
                                        long n, long xrow, long qrow, long srow, long rqrow,
                                        long rsrow, RadStream s) {
    if (!q) return;
    const long items = rows * (n / kUpTB);
    if (i8)
        hipLaunchKernelGGL(hc_quant_rows_kernel<true>, dim3((unsigned)((items + kWaves - 1) / kWaves)),
                           dim3(kThreads), 0, (hipStream_t)s, x, q, qs, rq, rqs, rows, n / kUpTB,
                           xrow, qrow, srow, rqrow, rsrow);
    else
        hipLaunchKernelGGL(hc_quant_rows_kernel<false>, dim3((unsigned)((items + kWaves - 1) / kWaves)),
                           dim3(kThreads), 0, (hipStream_t)s, x, q, qs, rq, rqs, rows, n / kUpTB,
                           xrow, qrow, srow, rqrow, rsrow);
}

static inline void hc_up_mix_wide_launch(bool f8, bool i8, const void* up, long uld,
                                         const unsigned short* dbuf, const unsigned short* h,
                                         const float* w, const float* scale, unsigned short* x,
                                         unsigned char* q, float* qs, unsigned char* rq,
                                         float* rqs,
                                         long rows, long n, long lr, long hrow, long xrow,
                                         long qrow, long srow, long rqrow, long rsrow, float wadd,
                                         RadStream s) {
    const dim3 gm((unsigned)(n / kWideCH), (unsigned)((rows + kWideRT - 1) / kWideRT));
#define HC_UMW_CASE(F8_)                                                                         \
    hipLaunchKernelGGL((hc_up_mix_wide_kernel<4, F8_>), gm, dim3(kThreads), 0, (hipStream_t)s,   \
                       up, uld, dbuf, h, w, scale, x, rows, n, lr, hrow, xrow, wadd)
    if (f8) HC_UMW_CASE(true); else HC_UMW_CASE(false);
#undef HC_UMW_CASE
    hc_quant_rows_launch(i8, x, q, qs, rq, rqs, rows, n, xrow, qrow, srow, rqrow, rsrow, s);
}

/* ==================================================== hc_read, second prefill form (PF2)
 *
 * Used for prefill chunks of at least kPf2MinRows rows whose shape fits (hc 4, n a multiple of
 * 256, at most kPf2MaxNT column n-tiles), with either mixing-matrix dtype; pinned form 3 forces it
 * at any 128-row tile of such a band. Other prefill calls use the wide form. Same steps and bytes
 * as the wide form, with the work redistributed (711 -> 611 us per call at a 2048-row chunk on
 * gfx1201, E4M3 matrices):
 *
 *   N is built once: a PF2 workgroup owns 64 rows and every column, where hc_down_wide_kernel's
 *   three column tiles each rebuild N from `h`.
 *   Weights are decoded once, in the prologue, into the `u` scratch (no fused form uses it),
 *   instead of by every row tile. At bf16 the prologue copies mix_down and mix_up is read in place.
 *   The down product's B skips LDS: each 16-column n-tile belongs to one wave, so the prologue
 *   writes the decoded matrix in WMMA fragment order and a fragment is one 512-byte load. Only N
 *   goes through LDS, double-buffered, one barrier per slice.
 *   The up product uses a 256-row tile (hc_up_mix_wide_kernel at RT 256, two m-tiles per wave),
 *   so each B fragment read feeds two WMMAs.
 *
 * Same bytes: every output element is the same chain of 16x16x16 WMMAs over the same fragments in
 * the same order (layout from r4d_gdn_wmma.h), the down product keeps hc_down_kchunk's K-split at
 * the 128-row tile with the same reduce, and the norm scales match hc_norm_scale_kernel's order.
 * r4d_selftest compares all three forms byte for byte at both dtypes.
 *
 * r4d_rad_hc_read_scratch sizes `u` for both planes at every band reaching kPf2MinRows
 * (hc_u_rows). */
constexpr int  kPf2RT      = 64;                 // down product: rows a workgroup, four m-tiles
constexpr int  kPf2NW      = 3;                  // down product: n-tiles a wave, at most
constexpr int  kPf2MaxNT   = kWaves * kPf2NW;    // ...so ceil((lr + hc) / 16) <= 24
constexpr int  kPf2UpRT    = 256;                // up product: rows a workgroup, two m-tiles a wave
constexpr long kPf2MinRows = 512;                // the rule's floor, and where `u` holds the planes

/* The prologue: per-stream norm scales and the two weight planes in one launch, since the down
 * product needs all three and none depends on another.
 *
 * Blocks [0, nnb) compute the scales, one wave per (row, stream), matching hc_norm_scale_kernel
 * (a workgroup per (row, stream): thread t sums elements t, t + 256, ... ascending, waves fold with
 * r4d_wave_sum, thread 0 adds the eight wave totals). Here lane L loads elements 8L .. 8L + 7 of
 * each 256-element chunk, holding threads 8L .. 8L + 7's running sums, and a wave-private LDS slot
 * puts them back in thread order so lane L of the k-th r4d_wave_sum holds thread 32k + L's sum.
 * Same sums, same scale. No barrier: a wave's LDS operations complete in order.
 *
 * Blocks [nnb, ...) write the planes, a 16-byte vector per item:
 *
 *   `plane`, the down product's B in fragment order: [ntt][HN/16][32 lanes][8 bf16]. Lane l of the
 *   fragment for n-tile nt and k-step g holds column nt*16 + (l & 15) at k = 16g + 4(l >> 4)
 *   + {0..3, 8..11}, as frag() reads an LDS tile. Columns below lr are mix_down (decoded, or
 *   copied at bf16), the next hc the inject weights, and the pad clamps to the last real column,
 *   as the wide form's staging does.
 *   `udec`, mix_up decoded to plain [HN][lr] bf16 (E4M3 only; a bf16 mix_up is read in place).
 *
 * Every decoded element is f2bf(code * scale) as in the wide form's staging. */
__global__ __launch_bounds__(kThreads)
void hc_pf2_prep_kernel(const unsigned short* __restrict__ h, float* __restrict__ scaleout,
                        long rows, long n, int hc, long hrow, float eps, long nnb,
                        const void* __restrict__ downv, long dld, bool f8,
                        const unsigned short* __restrict__ injw,
                        const unsigned char* __restrict__ up8, long uld,
                        uint4* __restrict__ plane, unsigned short* __restrict__ udec,
                        long lr, long ncol, long ntt) {
    __shared__ float s_t[kWaves][kThreads];
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long HN = (long)hc * n;
    if ((long)blockIdx.x < nnb) {
        const long p = (long)blockIdx.x * kWaves + wave;
        if (p >= rows * hc) return;
        const long r = p / hc;
        const int  c = (int)(p - r * hc);
        const unsigned short* hp = h + r * hrow + (long)c * n;
        /* Five chunks' loads in flight before the first is summed; summation order is unchanged. */
        constexpr int kU = 5;
        float acc[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
        long j0 = 0;
        for (; j0 + kU * kThreads <= n; j0 += kU * kThreads) {
            uint4 v[kU];
#pragma unroll
            for (int u = 0; u < kU; ++u) v[u] = *(const uint4*)(hp + j0 + u * kThreads + lane * 8);
#pragma unroll
            for (int u = 0; u < kU; ++u) {
                float f[8];
                bf16x8(v[u], f);
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[e] += f[e] * f[e];
            }
        }
        for (; j0 < n; j0 += kThreads) {
            float f[8];
            bf16x8(*(const uint4*)(hp + j0 + lane * 8), f);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += f[e] * f[e];
        }
        float* const st = s_t[wave];
#pragma unroll
        for (int e = 0; e < 8; ++e) st[lane * 8 + e] = acc[e];
        float ss = 0.0f;
#pragma unroll
        for (int k = 0; k < kWaves; ++k) ss += wave_sum(st[k * kWave + lane]);
        if (lane == 0) scaleout[r * kHcMax + c] = __frsqrt_rn(ss / (float)n + eps);
        return;
    }

    const long KS   = HN / 16;
    const long nswz = ntt * KS * kWave;
    const long nup  = udec ? HN * (lr / 8) : 0;
    const long step = ((long)gridDim.x - nnb) * kThreads;
    for (long v = ((long)blockIdx.x - nnb) * kThreads + tid; v < nswz + nup; v += step) {
        if (v < nswz) {
            const long f   = v / kWave;
            const int  l   = (int)(v - f * kWave);
            const long nt  = f / KS, g = f - nt * KS;
            long col = nt * 16 + (l & 15);
            if (col >= ncol) col = ncol - 1;
            const long k = g * 16 + 4 * (l >> 4);
            uint4 o;
            if (col < lr && f8) {
                const unsigned char* row = (const unsigned char*)downv + col * dld;
                const float sc = ((const float*)(row + HN))[k / r4d_hc8::kDownGroup];
                float fv[8];
                e4m3x8(*(const uint32_t*)(row + k), *(const uint32_t*)(row + k + 8), fv);
#pragma unroll
                for (int e = 0; e < 8; ++e) fv[e] = fv[e] * sc;
                o = f2bf8(fv);
            } else {
                const unsigned short* src = (col < lr) ? (const unsigned short*)downv + col * HN
                                                       : injw + (col - lr) * HN;
                const uint2 lo = *(const uint2*)(src + k), hi = *(const uint2*)(src + k + 8);
                o = make_uint4(lo.x, lo.y, hi.x, hi.y);
            }
            plane[v] = o;
        } else {
            const long u  = v - nswz;
            const long j  = u / (lr / 8), k8 = (u - j * (lr / 8)) * 8;
            const unsigned char* row = up8 + j * uld;
            *(uint4*)(udec + j * lr + k8) =
                e4m3x8_bf16(row + k8, ((const float*)(row + lr))[k8 / r4d_hc8::up_group(lr)]);
        }
    }
}

/* d = silu(W_down @ N / hc) and the inject gains, PF2: a workgroup owns 64 rows, every column and
 * one K chunk (hc_down_kchunk's at the 128-row tile, so partials match the wide form's).
 *
 * Wave w owns n-tiles w, w + 8 and w + 16 (those below ntt) across all four m-tiles: twelve
 * accumulators, each B fragment feeding four WMMAs and each A fragment three. B comes straight
 * from `plane` a slice ahead: the next slice's fragment for a k-step is loaded as soon as this
 * slice's has been consumed. N is built from `h` into one of two LDS tiles while the other is
 * read, one barrier per slice.
 *
 * Each accumulator sees its chunk's K steps in ascending order from zero with the wide form's
 * fragments, so partials (or the one-chunk epilogue) match. */
__global__ __launch_bounds__(kThreads)
void hc_down_pf2_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                        const uint4* __restrict__ plane, const float* __restrict__ scalein,
                        unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
                        float* __restrict__ part,
                        long rows, long n, int hc, long lr, long ncol, long ntt, long hrow,
                        long irow, long kc, float wadd) {
    constexpr int RT = kPf2RT, MT = RT / 16, NW = kPf2NW;
    constexpr int kKV = kTKT / 8;                    /* 16-byte vectors a staged row */
    constexpr int kRS = kThreads / kKV;              /* rows between a thread's A rows */
    constexpr int kAV = RT / kRS;
    constexpr int KST = kTKT / 16;                   /* k-steps a slice */
    static_assert(RT % kRS == 0 && MT * 16 == RT, "whole A rows a pass");
    __shared__ unsigned short sA[2 * RT * kTAP];

    /* `wave` via readfirstlane, so skipping an n-tile is a scalar branch rather than an EXEC mask. */
    const int  tid = threadIdx.x, lane = tid & (kWave - 1);
    const int  wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const long HN = (long)hc * n, KS = HN / 16;
    const long r0 = (long)blockIdx.x * RT;
    const long ka = (long)blockIdx.y * kc;
    const long kb = (ka + kc < HN) ? ka + kc : HN;
    if (r0 >= rows || ka >= HN) return;
    const long nsl = (kb - ka) / kTKT;

    const int k8a = (tid % kKV) * 8, ar = tid / kKV;
    long arow[kAV];
#pragma unroll
    for (int i = 0; i < kAV; ++i) {
        const long r = r0 + ar + i * kRS;
        arow[i] = r < rows ? r : rows - 1;
    }
    uint4 pa[kAV];
    float psc[kAV], pw[8];
    auto fetchA = [&](long k0) {
        const int c = (int)(k0 / n);
#pragma unroll
        for (int i = 0; i < kAV; ++i) {
            pa[i]  = *(const uint4*)(h + arow[i] * hrow + k0 + k8a);
            psc[i] = scalein[arow[i] * kHcMax + c];
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) pw[e] = w[k0 + k8a + e];
    };
    auto stageA = [&](unsigned short* dst) {
#pragma unroll
        for (int i = 0; i < kAV; ++i) {
            float f8v[8];
            bf8_f(pa[i], f8v);
#pragma unroll
            for (int e = 0; e < 8; ++e) f8v[e] = f8v[e] * psc[i] * (pw[e] + wadd);
            *(uint4*)(dst + (ar + i * kRS) * kTAP + k8a) = f2bf8(f8v);
        }
    };

    /* This lane's fragment stream for each of the wave's n-tiles, from the chunk's first k-step. */
    bool live[NW];
    const uint4* bp[NW];
#pragma unroll
    for (int i = 0; i < NW; ++i) {
        const long nt = wave + i * kWaves;
        live[i] = nt < ntt;
        bp[i] = plane + ((live[i] ? nt : 0) * KS + ka / 16) * kWave + lane;
    }
    uint4 rb[KST][NW];
#pragma unroll
    for (int s = 0; s < KST; ++s)
#pragma unroll
        for (int i = 0; i < NW; ++i)
            rb[s][i] = live[i] ? bp[i][(long)s * kWave] : make_uint4(0u, 0u, 0u, 0u);

    v8f acc[MT][NW];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < NW; ++i) acc[m][i] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    /* No branch on the slice in the loop body: the last slice prefetches itself again into the
     * unread tile, costing one redundant slice of loads per call but keeping every wait count
     * straight-line. */
    fetchA(ka);
    stageA(sA);
    __syncthreads();
    for (long t = 0; t < nsl; ++t) {
        const long tn = (t + 1 < nsl) ? t + 1 : t;
        const unsigned short* const cur = sA + (t & 1) * (RT * kTAP);
        fetchA(ka + tn * kTKT);
#pragma unroll
        for (int s = 0; s < KST; ++s) {
            v8bf af[MT];
#pragma unroll
            for (int m = 0; m < MT; ++m) af[m] = frag(cur, kTAP, m * 16, s * 16, lane);
#pragma unroll
            for (int i = 0; i < NW; ++i) {
                if (!live[i]) continue;
                union { uint4 q; v8bf f; } bu;
                bu.q = rb[s][i];
#pragma unroll
                for (int m = 0; m < MT; ++m) acc[m][i] = mma(af[m], bu.f, acc[m][i]);
                rb[s][i] = bp[i][(tn * KST + s) * kWave];
            }
        }
        stageA(sA + ((t + 1) & 1) * (RT * kTAP));
        __syncthreads();
    }

    const float inv_hc = 1.0f / (float)hc;
    const int   e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int i = 0; i < NW; ++i) {
        if (!live[i]) continue;
        const long col = (long)(wave + i * kWaves) * 16 + nn;
        if (col >= ncol) continue;
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const long r = r0 + m * 16 + e0 + e;
                if (r >= rows) continue;
                const float v = acc[m][i][e];
                if (part)          part[((long)blockIdx.y * rows + r) * ncol + col] = v;
                else if (col < lr) dout[r * lr + col] = f2bf(siluf_(v * inv_hc));
                else               inj[r * irow + (col - lr)] = f2bf(2.0f * sigmoidf_(v * inv_hc));
            }
    }
}

/* K per down-product workgroup: enough z-blocks that lr/kTNT * z covers the part, rounded to whole
 * kTKT slices so a slice never straddles a sub-stream. Returns HN when one block covers K (the
 * prefill form, un-split epilogue). */

/* ==================================================== mtp_enter, tiled
 *
 * The per-row form reads both square matrices (26.2 MB) once per row, 210 MB per call at eight
 * concurrent sequences. As for the gated residual, the fix is a tiled GEMM.
 *
 * It is two products sharing one accumulator. With output rows (r, k), one per sub-stream of each
 * token,
 *
 *     x[(r,k)][i] = nh[(r,k)] . fc_h[i]  +  ne[r] . fc_e[i]
 *
 * is C[R, n] over K = n with R = rows*hc; the second term's A operand `ne[r]` repeats for a token's
 * hc rows. The fc_e product is computed hc times, using otherwise idle WMMA slots, so each weight
 * is read once.
 *
 * Norm scales are recomputed per block, since the op has no scratch hook: `h` and `e` (25 KiB per
 * row) stay in cache against 655 KiB of weight per block. The reduction order differs from the
 * per-row kernel's, so scales can differ in the last bit; acceptable for a draft head. */
/* The column tile follows the row tile. There is no K-split (the two products share an
 * accumulator, and a split would need a reduce launch and scratch), so the grid is n / NT
 * workgroups per row tile, each summing squares for every row of its tile first. Narrow tiles add
 * workgroups for the weight stream but repeat that prologue. At n = 2560, us per call:
 *
 *     columns   M=2   M=4   M=8   M=16   M=32
 *       16       68    70    72    124    280
 *       32       48    50    55     92    303
 *       64       60    63    69    103    189
 *
 * so 32 up to the 64-row tile and 64 at the 128-row one. The choice does not change any output. */
constexpr int kMtpTNTNarrow = 32, kMtpTNTWide = 64;
/* At a draft round's heights the E4M3 form is bound by its chain of K passes, not by bytes or
 * arithmetic (13.1 MB in ~50 us with most of the card idle): a workgroup walks K = n in passes of
 * kTKT, two barriers each, and a 32-row tile stages and multiplies 32 rows for the four to
 * sixteen a draft round has. So up to 16 output rows the tile is 16 rows, and up to 32 rows each
 * pass takes kMtpKTShort of K over kMtpTNTShort columns: half the passes on half the workgroups.
 * Cold, us a call at M = 1 / 4 / 6 / 8 tokens: 51.2 / 51.7 / 52.8 / 54.3 -> 40.0 / 40.6 / 48.0 /
 * 48.3. 256 of K a pass is slower (52.8 at M = 1: its 50 KiB of LDS leaves a workgroup a CU), and
 * the 64-row tile keeps kTKT (128 there: 85 against 70). The two products share an accumulator
 * and alternate by pass, so a longer pass changes where their partial sums meet: an output can
 * differ in its last bit from the kTKT form's, as the forms already differ from each other. */
constexpr int kMtpTNTShort = 64, kMtpKTShort = 128;

#define MTP_ENTER_TILED_PARAMS                                                                  \
    const unsigned short* __restrict__ h, const unsigned short* __restrict__ e,                 \
    const float* __restrict__ wh, const float* __restrict__ we,                                 \
    const void* __restrict__ fchv, const void* __restrict__ fcev, long fld,                     \
    unsigned short* __restrict__ x, long rows, long n, int hc, int flat, float eps,             \
    float wadd, long hrow, long erow, long xrow
template <int RT, int NT, bool F8, int KT = kTKT>
__device__ __forceinline__ void mtp_enter_tiled_body(MTP_ENTER_TILED_PARAMS) {
    constexpr int AP = KT + 4;      /* LDS row pitch in shorts, as kTAP */
    __shared__ unsigned short sAh[RT * AP];
    __shared__ unsigned short sAe[RT * AP];
    __shared__ unsigned short sBh[NT * AP];
    __shared__ unsigned short sBe[NT * AP];
    __shared__ float s_rs[RT];                 /* one per OUTPUT row (r,k) */
    __shared__ float s_re[RT];                 /* ne's scale, replicated per output row */
    __shared__ float s_et[RT + 1];             /* ne's scale, one per token of the tile */

    constexpr int MT = RT / 16, NT16 = NT / 16;
    constexpr int WM = (MT < kWaves) ? MT : kWaves, WN = kWaves / WM;
    constexpr int AM = (MT + WM - 1) / WM, AN = (NT16 + WN - 1) / WN;

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long R    = rows * hc;
    const long r0   = (long)blockIdx.y * RT;
    const long j0   = (long)blockIdx.x * NT;
    if (r0 >= R || j0 >= n) return;

    /* ---- the scales, one output row per wave. `flat` sums the whole wide row (the same partials
     * added together).
     *
     * The sums of squares are latency chains (80 dependent loads per lane per row) and unhidden
     * they dominated (~170 us per call against ~50 for the weight stream). So a wave keeps loads in
     * flight: kSsRows rows at a time, kSsPre of each row's loads issued before the first is summed.
     * The embedding's scale is per token, since a token's hc output rows share `e`.
     *
     * Bit-identical: each row is summed by one wave, lane j = lane, lane + 32, ... ascending. */
    constexpr int kSsPre = 8, kSsRows = 4;
    auto sumsq = [&](const unsigned short* const* p, int nr, long len, float* out) {
        float acc[kSsRows];
#pragma unroll
        for (int k = 0; k < kSsRows; ++k) acc[k] = 0.0f;
        long j = lane;
        for (; j + (long)(kSsPre - 1) * kWave < len; j += (long)kSsPre * kWave) {
            uint32_t v[kSsRows][kSsPre];
#pragma unroll
            for (int k = 0; k < kSsRows; ++k)
                if (k < nr) {
#pragma unroll
                    for (int q = 0; q < kSsPre; ++q) v[k][q] = bfw(p[k] + j + (long)q * kWave);
                }
#pragma unroll
            for (int k = 0; k < kSsRows; ++k)
                if (k < nr) {
#pragma unroll
                    for (int q = 0; q < kSsPre; ++q) { const float f = bfw_f(v[k][q]); acc[k] += f * f; }
                }
        }
        for (; j < len; j += kWave)
#pragma unroll
            for (int k = 0; k < kSsRows; ++k)
                if (k < nr) { const float f = bf2f(p[k][j]); acc[k] += f * f; }
#pragma unroll
        for (int k = 0; k < kSsRows; ++k)
            if (k < nr) out[k] = wave_sum(acc[k]);
    };
    const long hlen = flat ? (long)hc * n : n;
    for (int rb = wave; rb < RT; rb += kWaves * kSsRows) {
        const unsigned short* p[kSsRows];
        int nr = 0;
        for (int rr = rb; rr < RT && nr < kSsRows; rr += kWaves, ++nr) {
            const long Ri = r0 + rr;
            const long rt = (Ri < R) ? Ri / hc : rows - 1, kt = (Ri < R) ? Ri % hc : 0;
            p[nr] = h + rt * hrow + (flat ? 0 : kt * n);
        }
        float ss[kSsRows];
        sumsq(p, nr, hlen, ss);
        if (lane == 0)
            for (int k = 0; k < nr; ++k)
                s_rs[rb + k * kWaves] = __frsqrt_rn(ss[k] / (float)hlen + eps);
    }
    /* The tile's tokens [t0, t1]: those of its real rows; padding rows past R use `rows - 1`, the
     * last real row's token. */
    const long t0 = r0 / hc;
    const long t1 = ((r0 + RT < R) ? (r0 + RT - 1) : (R - 1)) / hc;
    for (long tb = t0 + wave; tb <= t1; tb += kWaves * kSsRows) {
        const unsigned short* p[kSsRows];
        int nr = 0;
        for (long tt = tb; tt <= t1 && nr < kSsRows; tt += kWaves, ++nr) p[nr] = e + tt * erow;
        float ss[kSsRows];
        sumsq(p, nr, n, ss);
        if (lane == 0)
            for (int k = 0; k < nr; ++k)
                s_et[tb - t0 + k * kWaves] = __frsqrt_rn(ss[k] / (float)n + eps);
    }
    __syncthreads();
    for (int rr = tid; rr < RT; rr += kThreads) {
        const long Ri = r0 + rr;
        s_re[rr] = s_et[((Ri < R) ? Ri / hc : rows - 1) - t0];
    }
    __syncthreads();

    v8f acc[AM][AN];
#pragma unroll
    for (int im = 0; im < AM; ++im)
#pragma unroll
        for (int in = 0; in < AN; ++in) acc[im][in] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    /* The next K slice is loaded into registers while this one computes; otherwise each of the
     * n / KT slices exposes a full global latency, with few workgroups per CU at decode. A
     * thread's share of slice k + 1 (raw `h`, `e` and norm-weight values for its A items, weight
     * rows for its B items) is issued after slice k is written to LDS and awaited at the next
     * write; the loop's barriers order LDS only (hc_lds_barrier), so those loads stay in flight
     * across them. Values are unchanged. */
    constexpr int kAI = (RT * (KT / 8) + kThreads - 1) / kThreads;
    constexpr int kBI = (NT * (KT / 8) + kThreads - 1) / kThreads;
    struct APre { uint4 h, e; float4 wh0, wh1, we0, we1; };
    APre ap[kAI];
    /* A weight item is eight bf16 of a row, or at E4M3 its eight codes and their group's scale. */
    struct BPre16 { uint4 v; };
    struct BPre8  { uint2 c; float s; };
    using BPre = typename std::conditional<F8, BPre8, BPre16>::type;
    BPre bph[kBI], bpe[kBI];
    auto bload = [&](const void* base, long j, long k, BPre& o) {
        if constexpr (F8) {
            const unsigned char* row = (const unsigned char*)base + j * fld;
            o.c = *(const uint2*)(row + k);
            o.s = ((const float*)(row + n))[k / r4d_hc8::kDownGroup];
        } else {
            o.v = *(const uint4*)((const unsigned short*)base + j * n + k);
        }
    };
    auto bval = [&](const BPre& o) -> uint4 {
        if constexpr (F8) {
            float f[8];
            e4m3x8(o.c.x, o.c.y, f);
#pragma unroll
            for (int q = 0; q < 8; ++q) f[q] = f[q] * o.s;
            return f2bf8(f);
        } else {
            return o.v;
        }
    };
    /* An A item's rows are the same every K step: its token and sub-stream (a 64-bit division by
     * `hc`) are computed once here as element offsets. Inside the step, gfx12's compiler would
     * redo the division every slice. */
    uint32_t aoh[kAI], aoe[kAI], aow[kAI], ak8[kAI];
#pragma unroll
    for (int a = 0; a < kAI; ++a) {
        const int idx = tid + a * kThreads;
        const int rr = idx / (KT / 8), k8 = (idx % (KT / 8)) * 8;
        const long Ri = r0 + rr;
        const long rt = (Ri < R) ? Ri / hc : rows - 1, kt = (Ri < R) ? Ri % hc : 0;
        aoh[a] = (uint32_t)(rt * hrow + kt * n + k8);
        aoe[a] = (uint32_t)(rt * erow + k8);
        aow[a] = (uint32_t)(kt * n + k8);
        ak8[a] = (uint32_t)k8;
    }
    auto issue = [&](long k0) {
#pragma unroll
        for (int a = 0; a < kAI; ++a) {
            const int idx = tid + a * kThreads;
            if (idx >= RT * (KT / 8)) break;
            ap[a].h   = *(const uint4*)(h + aoh[a] + k0);
            ap[a].e   = *(const uint4*)(e + aoe[a] + k0);
            ap[a].wh0 = *(const float4*)(wh + aow[a] + k0);
            ap[a].wh1 = *(const float4*)(wh + aow[a] + k0 + 4);
            ap[a].we0 = *(const float4*)(we + ak8[a] + k0);
            ap[a].we1 = *(const float4*)(we + ak8[a] + k0 + 4);
        }
#pragma unroll
        for (int b = 0; b < kBI; ++b) {
            const int idx = tid + b * kThreads;
            if (idx >= NT * (KT / 8)) break;
            const int tt = idx / (KT / 8), k8 = (idx % (KT / 8)) * 8;
            long j = j0 + tt; if (j >= n) j = n - 1;
            bload(fchv, j, k0 + k8, bph[b]);
            bload(fcev, j, k0 + k8, bpe[b]);
        }
    };
    issue(0);

    for (long k0 = 0; k0 < n; k0 += KT) {
        hc_lds_barrier();
#pragma unroll
        for (int a = 0; a < kAI; ++a) {
            const int idx = tid + a * kThreads;
            if (idx >= RT * (KT / 8)) break;
            const int rr = idx / (KT / 8), k8 = (idx % (KT / 8)) * 8;
            const float sc = s_rs[rr], se = s_re[rr];
            const float wp[8] = { ap[a].wh0.x, ap[a].wh0.y, ap[a].wh0.z, ap[a].wh0.w,
                                  ap[a].wh1.x, ap[a].wh1.y, ap[a].wh1.z, ap[a].wh1.w };
            const float wq[8] = { ap[a].we0.x, ap[a].we0.y, ap[a].we0.z, ap[a].we0.w,
                                  ap[a].we1.x, ap[a].we1.y, ap[a].we1.z, ap[a].we1.w };
            float f8v[8];
            bf8_f(ap[a].h, f8v);
#pragma unroll
            for (int q = 0; q < 8; ++q) f8v[q] = f8v[q] * sc * (wp[q] + wadd);
            *(uint4*)(sAh + rr * AP + k8) = f2bf8(f8v);
            bf8_f(ap[a].e, f8v);
#pragma unroll
            for (int q = 0; q < 8; ++q) f8v[q] = f8v[q] * se * (wq[q] + wadd);
            *(uint4*)(sAe + rr * AP + k8) = f2bf8(f8v);
        }
#pragma unroll
        for (int b = 0; b < kBI; ++b) {
            const int idx = tid + b * kThreads;
            if (idx >= NT * (KT / 8)) break;
            const int tt = idx / (KT / 8), k8 = (idx % (KT / 8)) * 8;
            *(uint4*)(sBh + tt * AP + k8) = bval(bph[b]);
            *(uint4*)(sBe + tt * AP + k8) = bval(bpe[b]);
        }
        if (k0 + KT < n) issue(k0 + KT);
        hc_lds_barrier();
#pragma unroll
        for (int im = 0; im < AM; ++im) {
            const int mt = wave / WN + im * WM;
            if (mt >= MT) continue;
#pragma unroll
            for (int in = 0; in < AN; ++in) {
                const int nt = wave % WN + in * WN;
                if (nt >= NT16) continue;
                acc[im][in] = gemm<KT / 16>(sAh, AP, mt, sBh, AP, nt, lane, acc[im][in]);
                acc[im][in] = gemm<KT / 16>(sAe, AP, mt, sBe, AP, nt, lane, acc[im][in]);
            }
        }
    }

    const int e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int im = 0; im < AM; ++im) {
        const int mt = wave / WN + im * WM;
        if (mt >= MT) continue;
#pragma unroll
        for (int in = 0; in < AN; ++in) {
            const int nt = wave % WN + in * WN;
            if (nt >= NT16) continue;
            const long j = j0 + nt * 16 + nn;
            if (j >= n) continue;
            /* The eight rows are consecutive output rows: one division for the first, then the
             * (token, sub-stream) pair is stepped. */
            const long Rq = r0 + mt * 16 + e0;
            long tq = Rq / hc, kq = Rq % hc;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                if (Rq + q < R) x[tq * xrow + kq * n + j] = f2bf(acc[im][in][q]);
                if (++kq == hc) { kq = 0; ++tq; }
            }
        }
    }
}

/* The kernel around the body, whose blocks return where their work ends: CH `r4dchain` is the
 * chained form (r4d_chain.h), taken at decode heights. */
template <int RT, int NT, bool F8, class CH, int KT = kTKT>
__global__ __launch_bounds__(kThreads)
void mtp_enter_tiled_kernel(MTP_ENTER_TILED_PARAMS) {
    if (CH::on) r4d_chain_wait();
    mtp_enter_tiled_body<RT, NT, F8, KT>(h, e, wh, we, fchv, fcev, fld, x, rows, n, hc, flat, eps, wadd, hrow, erow, xrow);
    if (CH::on) r4d_chain_done();
}
#undef MTP_ENTER_TILED_PARAMS
/* ==================================================== mtp_enter from 32 tokens up
 *
 * The tiled kernel above suits a draft round's few rows; beyond that its costs grow with rows: it
 * computes the embedding term `ne[r] . fc_e[i]` once per output row, and each of its 40 column
 * blocks re-sums squares and re-stages the same A rows (6.2 ms per 2048-token chunk).
 *
 * Here the two products are separate GEMMs. Scales take one pass, a wave per row
 * (mtp_prefill_scale_kernel). The embedding term is a [rows, n] GEMM kept in f32 scratch; the
 * stream term is the [rows * hc, n] GEMM whose epilogue adds the token's embedding term before the
 * single bf16 rounding. Both use mtp_prefill_gemm_kernel: 128 x 128 tiles, eight waves of 64 x 32,
 * the next K tile in registers during compute, A normalised and weights decoded on the way to LDS.
 *
 * Bytes differ from the tiled kernel's (which accumulates both products together per K step);
 * acceptable for a draft head the trunk verifies. */
/* The form switches at 32 tokens: below, the grid is n/128 = 20 blocks and the three launches cost
 * ~100 us regardless of rows. us per call (tiled / this form), Flash-Next:
 *
 *      tokens    8        16       32        64        128       512        2048
 *              52 / 98  88 / 105  182 / 117  267 / 158  433 / 213  1540 / 627  6180 / 2430 */
constexpr long kMtpPrefillRows = 32;

/* One wave per row: rows [0, R) are the output rows (t, k) over `h` (sub-stream k of token t, or
 * the whole wide row when `flat`), rows [R, R + rows) the tokens over `e`. A lane sums j = lane,
 * lane + 32, ... in order, eight loads ahead of the adds. */
__global__ __launch_bounds__(kThreads)
void mtp_prefill_scale_kernel(const unsigned short* __restrict__ h,
                              const unsigned short* __restrict__ e,
                              float* __restrict__ rs, float* __restrict__ re,
                              long rows, long n, int hc, int flat, float eps, long hrow,
                              long erow) {
    const int  lane = threadIdx.x & (kWave - 1);
    const long row  = (long)blockIdx.x * kWaves + threadIdx.x / kWave;
    const long R    = rows * hc;
    if (row >= R + rows) return;
    const bool tok = row >= R;
    const long len = tok ? n : (flat ? (long)hc * n : n);
    const unsigned short* p = tok ? e + (row - R) * erow
                                  : h + (row / hc) * hrow + (flat ? 0 : (row % hc) * n);
    constexpr int kPre = 8;
    float acc = 0.0f;
    long j = lane;
    for (; j + (long)(kPre - 1) * kWave < len; j += (long)kPre * kWave) {
        uint32_t v[kPre];
#pragma unroll
        for (int q = 0; q < kPre; ++q) v[q] = bfw(p + j + (long)q * kWave);
#pragma unroll
        for (int q = 0; q < kPre; ++q) { const float f = bfw_f(v[q]); acc += f * f; }
    }
    for (; j < len; j += kWave) { const float f = bf2f(p[j]); acc += f * f; }
    acc = wave_sum(acc);
    if (lane == 0) {
        const float s = __frsqrt_rn(acc / (float)len + eps);
        if (tok) re[row - R] = s; else rs[row] = s;
    }
}

#define MP_BK    64                  /* K a staged tile: four WMMA k steps */
#define MP_PITCH (MP_BK + 8)         /* 144-byte LDS rows, as r4d_gemm_bf16_nt_tiled's */
#define MP_TM    4                   /* WMMA tiles a wave over M ... */
#define MP_TN    2                   /* ... and over N: 64 x 32 a wave, 128 x 128 a block */

/* C = normalise(A) @ fc^T for one of the two products. EPASS: A is `e`, one row per token, and the
 * f32 result goes to `eterm`. Otherwise A is `h` read as rows (t, k), and the result plus the
 * token's `eterm` row is rounded once into `x`. Weights are bf16 rows or, with F8, E4M3 rows with
 * f32 128-column group scales in each row's tail (r4d_hc_fp8.h), `fld` bytes apart. */
template <bool F8, bool EPASS>
__global__ __launch_bounds__(kThreads)
void mtp_prefill_gemm_kernel(const unsigned short* __restrict__ src, long srow,
                             const float* __restrict__ gain, const float* __restrict__ rscale,
                             const void* __restrict__ fcv, long fld,
                             float* __restrict__ eterm, unsigned short* __restrict__ x,
                             long rows, long n, int hc, float wadd, long xrow) {
    constexpr int BM = 2 * MP_TM * 16, BN = 4 * MP_TN * 16;
    constexpr int CPR = MP_BK / 8;                      /* 16-byte chunks a staged row */
    constexpr int AL = BM * CPR / kThreads, WL = BN * CPR / kThreads;
    static_assert(BM * CPR % kThreads == 0 && BN * CPR % kThreads == 0, "whole chunks");
    __shared__ __attribute__((aligned(16))) unsigned short sA[BM * MP_PITCH];
    __shared__ __attribute__((aligned(16))) unsigned short sW[BN * MP_PITCH];

    const int  tid = threadIdx.x, wave = tid / kWave, lane = tid & (kWave - 1);
    const int  wm = wave / 4, wn = wave % 4;
    const long Mr = EPASS ? rows : rows * (long)hc;     /* A rows of this product */
    const long m0 = (long)blockIdx.y * BM, n0 = (long)blockIdx.x * BN;

    /* A thread's staging slots: chunk c = tid + 256 i is tile row c / CPR, K offset (c % CPR) * 8.
     * Each slot's row source, gain offset and scale are fixed for the whole K loop. */
    const unsigned short* asrc[AL];
    const float* again[AL];
    float asc[AL];
    unsigned short* adst[AL];
#pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int  c = tid + kThreads * i, r = c / CPR, k8 = (c % CPR) * 8;
        const long gr = m0 + r < Mr ? m0 + r : Mr - 1;
        const long t = EPASS ? gr : gr / hc, kk = EPASS ? 0 : gr % hc;
        asrc[i]  = src + t * srow + kk * n + k8;
        again[i] = gain + kk * n + k8;
        asc[i]   = rscale[gr];
        adst[i]  = sA + r * MP_PITCH + k8;
    }
    const unsigned char* wrow[WL];
    const unsigned short* wsrc[WL];
    unsigned short* wdst[WL];
    int wk8[WL];
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int  c = tid + kThreads * i, r = c / CPR, k8 = (c % CPR) * 8;
        const long gj = n0 + r < n ? n0 + r : n - 1;
        wrow[i] = (const unsigned char*)fcv + gj * fld;
        wsrc[i] = (const unsigned short*)fcv + gj * n + k8;
        wk8[i]  = k8;
        wdst[i] = sW + r * MP_PITCH + k8;
    }

    const unsigned short* fa = sA + (wm * MP_TM * 16 + (lane & 15)) * MP_PITCH + 4 * (lane >> 4);
    const unsigned short* fw = sW + (wn * MP_TN * 16 + (lane & 15)) * MP_PITCH + 4 * (lane >> 4);

    v8f acc[MP_TM][MP_TN];
#pragma unroll
    for (int i = 0; i < MP_TM; ++i)
#pragma unroll
        for (int j = 0; j < MP_TN; ++j) acc[i][j] = (v8f)(0.f);

    /* The next K tile's raw bytes, in registers while this one computes. */
    uint4  ra[AL];
    float4 rg0[AL], rg1[AL];
    uint2  rc[WL];
    float  rsc[WL];
    r4d_u32x4 rw[WL];                               /* registers; uint4 would sit in scratch */
    auto fetch = [&](long k0) {
#pragma unroll
        for (int i = 0; i < AL; ++i) {
            ra[i]  = *(const uint4*)(asrc[i] + k0);
            rg0[i] = *(const float4*)(again[i] + k0);
            rg1[i] = *(const float4*)(again[i] + k0 + 4);
        }
#pragma unroll
        for (int i = 0; i < WL; ++i) {
            if constexpr (F8) {
                const long k = k0 + wk8[i];
                rc[i]  = *(const uint2*)(wrow[i] + k);
                rsc[i] = ((const float*)(wrow[i] + n))[k / r4d_hc8::kDownGroup];
            } else {
                rw[i] = *(const r4d_u32x4*)(wsrc[i] + k0);
            }
        }
    };
    fetch(0);

    const long nk = n / MP_BK;
    for (long kt = 0; kt < nk; ++kt) {
        __syncthreads();                                /* the previous tile's fragments are read */
#pragma unroll
        for (int i = 0; i < AL; ++i) {
            float f8v[8];
            bf8_f(ra[i], f8v);
            const float g[8] = { rg0[i].x, rg0[i].y, rg0[i].z, rg0[i].w,
                                 rg1[i].x, rg1[i].y, rg1[i].z, rg1[i].w };
#pragma unroll
            for (int q = 0; q < 8; ++q) f8v[q] = f8v[q] * asc[i] * (g[q] + wadd);
            *(uint4*)adst[i] = f2bf8(f8v);
        }
#pragma unroll
        for (int i = 0; i < WL; ++i) {
            if constexpr (F8) {
                float f[8];
                e4m3x8(rc[i].x, rc[i].y, f);
#pragma unroll
                for (int q = 0; q < 8; ++q) f[q] = f[q] * rsc[i];
                *(uint4*)wdst[i] = f2bf8(f);
            } else {
                *(r4d_u32x4*)wdst[i] = rw[i];
            }
        }
        __syncthreads();
        if (kt + 1 < nk) fetch((kt + 1) * MP_BK);
#pragma unroll
        for (int s = 0; s < MP_BK / 16; ++s) {
            v8bf af[MP_TM], bf[MP_TN];
#pragma unroll
            for (int i = 0; i < MP_TM; ++i) af[i] = fragP(fa + i * 16 * MP_PITCH + s * 16);
#pragma unroll
            for (int j = 0; j < MP_TN; ++j) bf[j] = fragP(fw + j * 16 * MP_PITCH + s * 16);
#pragma unroll
            for (int i = 0; i < MP_TM; ++i)
#pragma unroll
                for (int j = 0; j < MP_TN; ++j) acc[i][j] = mma(af[i], bf[j], acc[i][j]);
        }
    }

    /* Lane l holds column l & 15 of a tile and rows 8 (l >> 4) + e. */
#pragma unroll
    for (int i = 0; i < MP_TM; ++i)
#pragma unroll
        for (int j = 0; j < MP_TN; ++j) {
            const long col = n0 + (wn * MP_TN + j) * 16 + (lane & 15);
            const long mb  = m0 + (wm * MP_TM + i) * 16 + 8 * (lane >> 4);
            if (col >= n) continue;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const long m = mb + q;
                if (m >= Mr) continue;
                if constexpr (EPASS) {
                    eterm[m * n + col] = acc[i][j][q];
                } else {
                    const long t = m / hc, kk = m % hc;
                    x[t * xrow + kk * n + col] = f2bf(acc[i][j][q] + eterm[t * n + col]);
                }
            }
        }
}
#undef MP_BK
#undef MP_PITCH
#undef MP_TM
#undef MP_TN

/* The prefill form's scratch: the stream rows' scales, the tokens' scales, and the embedding term
 * in f32, each region 256-byte aligned. */
static inline long long mtp_prefill_bytes(long long rows, long long n, long long hc) {
    auto al = [](long long b) { return (b + 255) & ~255LL; };
    return al(rows * hc * 4) + al(rows * 4) + al(rows * n * 4);
}

/* ==================================================== down-product workgroup count
 *
 * The split counts row tiles (`nrt`) as well as column tiles. Counting only ceil(ncol/kTNT) (five
 * or six workgroups), or disabling the split at rt = 128, leaves a large decode step (e.g. sixteen
 * sequences at depth 3) with six workgroups on 64 CUs, a ~25% throughput cliff. As a function of
 * the whole grid, the split turns itself off once the grid is wide enough (about 2800 rows at this
 * model's ncol). */
static inline long hc_down_kchunk(long long lr, long long HN, long long nrt) {
    const long long nb     = ((lr + kTNT - 1) / kTNT) * (nrt > 0 ? nrt : 1);
    const long long chunks = (HN + kTKT - 1) / kTKT;
    long long want = (kTKSPLIT_WANT + nb - 1) / nb;
    if (want < 1) want = 1;
    if (want > chunks) want = chunks;
    return (long)(((chunks + want - 1) / want) * kTKT);
}
/* The row tile for a row count, defined once so the planner and the call site agree.
 *
 * Strictly greater than 64: at exactly 64 rows a 128-row tile re-reads the weight just as often
 * (once) but wastes half its LDS traffic and WMMA slots on clamped duplicate rows, and its 26112 B
 * of LDS fits two workgroups per WGP instead of three. Do not change to `>=`. */
static inline int hc_down_rt(long long rows) {
    return (rows > kTileMinRows) ? 128 : (rows > 32 ? 64 : 32);
}

/* The partial buffer for any row count up to the band's M, in bytes.
 *
 * `ks * rows` is not monotone in rows (fewer rows take a coarser tile and a deeper split), so the
 * band is walked rather than its endpoints checked. Once row tiles alone cover the part the split
 * is off and the buffer grows linearly, so the walk stops there and takes M as the last candidate.
 *
 * Invariant: derive the split from `lr` on both sides and size for the widest `ncol` a gain fold
 * can produce, or the partial buffer overruns. */
static inline long long hc_part_bytes(long long M, long long lr, long long HN) {
    const long long ncol = lr + (long long)kHcMax;
    const long long hi   = (M < 4096) ? M : 4096;
    long long worst = 0;
    for (long long r = 1; r <= hi; ++r) {
        const long long nrt = (r + hc_down_rt(r) - 1) / hc_down_rt(r);
        const long long ks  = (HN + hc_down_kchunk(lr, HN, nrt) - 1) / hc_down_kchunk(lr, HN, nrt);
        const long long b   = ks * r * ncol * 4;
        if (b > worst) worst = b;
    }
    if (M > hi) {
        const long long nrt = (M + hc_down_rt(M) - 1) / hc_down_rt(M);
        const long long ks  = (HN + hc_down_kchunk(lr, HN, nrt) - 1) / hc_down_kchunk(lr, HN, nrt);
        const long long b   = ks * M * ncol * 4;
        if (b > worst) worst = b;
    }
    return worst;
}

static inline void hc_down_tiled_launch(int rt, bool f8, const unsigned short* h, const float* w,
                                        const void* down, long dld, const float* scale,
                                        const unsigned short* injw,
                                        unsigned short* dbuf, unsigned short* inj, float* part,
                                        long rows, long n, int hc, long lr, long ncol, long hrow,
                                        long irow, long kc, float wadd, RadStream s,
                                        unsigned int* ctr = nullptr) {
    const long HN = (long)hc * n;
    const dim3 gd((unsigned)((ncol + kTNT - 1) / kTNT), (unsigned)((rows + rt - 1) / rt),
                  (unsigned)((HN + kc - 1) / kc));
#define HC_DOWN_CASE(RT_, F8_)                                                                   \
    hipLaunchKernelGGL((hc_down_tiled_kernel<RT_, kTNT, F8_>), gd, dim3(kThreads), 0,             \
                       (hipStream_t)s, h, w, down, dld, scale, injw, dbuf, inj, part, rows, n,  \
                       hc, lr, ncol, hrow, irow, kc, wadd, ctr)
    switch (rt) {
        case 32:  if (f8) HC_DOWN_CASE(32, true);  else HC_DOWN_CASE(32, false);  break;
        case 64:  if (f8) HC_DOWN_CASE(64, true);  else HC_DOWN_CASE(64, false);  break;
        default:  if (f8) HC_DOWN_CASE(128, true); else HC_DOWN_CASE(128, false); break;
    }
#undef HC_DOWN_CASE
}

/* hc_norm_down_tiled_kernel: `sync` its three ticket counters (hc_mix_ctr's, past the finishers'). */
static inline void hc_norm_down_tiled_launch(int rt, bool f8, const unsigned short* h,
                                             const float* w, const void* down, long dld,
                                             float* scale, const unsigned short* injw,
                                             unsigned short* dbuf, unsigned short* inj,
                                             float* part, long rows, long n, int hc, long lr,
                                             long ncol, long hrow, long irow, long kc, float wadd,
                                             float eps, RadStream s, unsigned int* ctr,
                                             unsigned int* sync, bool chain) {
    const long HN = (long)hc * n;
    const unsigned gdx = (unsigned)((ncol + kTNT - 1) / kTNT), gdy = (unsigned)((rows + rt - 1) / rt);
    const unsigned gdz = (unsigned)((HN + kc - 1) / kc), nn = (unsigned)hc * (unsigned)rows;
#define HC_ND_E(RT_, F8_, CH_)                                                                   \
    hipLaunchKernelGGL((hc_norm_down_tiled_kernel<RT_, kTNT, F8_, CH_>),                         \
                       dim3(nn + gdx * gdy * gdz),                                               \
                       dim3(kThreads), 0, (hipStream_t)s, h, w, down, dld, scale, injw, dbuf,   \
                       inj, part, rows, n, hc, lr, ncol, hrow, irow, kc, wadd, ctr, sync, eps,  \
                       nn, gdx, gdy, gdz)
#define HC_ND_CASE(RT_, F8_)                                                                     \
    do { if (chain) HC_ND_E(RT_, F8_, r4dchain); else HC_ND_E(RT_, F8_, R4dUnchained); } while (0)
    switch (rt) {
        case 32:  if (f8) HC_ND_CASE(32, true);  else HC_ND_CASE(32, false);  break;
        default:  if (f8) HC_ND_CASE(64, true);  else HC_ND_CASE(64, false);  break;
    }
#undef HC_ND_CASE
#undef HC_ND_E
}

static inline void hc_down_wide_launch(bool f8, const unsigned short* h, const float* w,
                                       const void* down, long dld, const float* scale,
                                       const unsigned short* injw,
                                       unsigned short* dbuf, unsigned short* inj, float* part,
                                       long rows, long n, int hc, long lr, long ncol, long hrow,
                                       long irow, long kc, float wadd, RadStream s) {
    const long HN = (long)hc * n;
    const dim3 gd((unsigned)((ncol + kWideNT - 1) / kWideNT),
                  (unsigned)((rows + kWideRT - 1) / kWideRT), (unsigned)((HN + kc - 1) / kc));
#define HC_DW_CASE(F8_)                                                                          \
    hipLaunchKernelGGL((hc_down_wide_kernel<kWideNT, F8_>), gd, dim3(kThreads), 0,               \
                       (hipStream_t)s, h, w, down, dld, scale, injw, dbuf, inj, part, rows, n,  \
                       hc, lr, ncol, hrow, irow, kc, wadd)
    if (f8) HC_DW_CASE(true); else HC_DW_CASE(false);
#undef HC_DW_CASE
}

static inline void hc_up_tiled_launch(int rt, bool f8, const void* up, long uld,
                                      const unsigned short* dbuf, unsigned short* ubuf, long rows,
                                      long hn, long lr, RadStream s) {
    const dim3 gu((unsigned)((hn + kTNT - 1) / kTNT), (unsigned)((rows + rt - 1) / rt));
#define HC_UP_CASE(RT_, F8_)                                                                     \
    hipLaunchKernelGGL((hc_up_tiled_kernel<RT_, kTNT, F8_>), gu, dim3(kThreads), 0,               \
                       (hipStream_t)s, up, uld, dbuf, ubuf, rows, hn, lr)
    switch (rt) {
        case 32:  if (f8) HC_UP_CASE(32, true);  else HC_UP_CASE(32, false);  break;
        case 64:  if (f8) HC_UP_CASE(64, true);  else HC_UP_CASE(64, false);  break;
        default:  if (f8) HC_UP_CASE(128, true); else HC_UP_CASE(128, false); break;
    }
#undef HC_UP_CASE
}

/* The PF2 form's weight planes in the `u` scratch, in bytes: mix_down (and the inject weights) in
 * fragment order over ntt n-tiles, then (E4M3 only) mix_up decoded to [HN][lr] bf16. */
static inline long long pf2_plane_bytes(long long HN, long long lr, long long ntt, bool f8) {
    return ntt * 16 * HN * 2 + (f8 ? HN * lr * 2 : 0);
}

/* Rows of `u` in the scratch, [rows][HN] bf16: the band's M and, at a band reaching the PF2 rule,
 * at least the rows its planes need at the widest fold (ncol = lr + hc) and either dtype,
 * ceil((lr + hc) / 16) * 16 + lr. The scratch hook and the launcher both size `u` from this. */
static inline long long hc_u_rows(long long M, long long lr, long long hc) {
    const long long pf2 = ((lr + hc + 15) / 16) * 16 + lr;
    return (M >= kPf2MinRows && pf2 > M) ? pf2 : M;
}

/* The PF2 form: prologue, down product, reduce, up product and mix, codes. `wsp` is the plane
 * space (the `u` buffer); the caller has checked it holds pf2_plane_bytes() and that hc is 4.
 * Returns a RAD code. */
static int hc_pf2_launch(bool f8, bool i8, const unsigned short* h, const float* w,
                         const void* down, long dld, const void* up, long uld,
                         const unsigned short* injw, unsigned short* dbuf, float* scale,
                         unsigned char* wsp, float* part, unsigned short* x,
                         unsigned short* inj, unsigned char* q, float* qs, unsigned char* rq,
                         float* rqs, long rows, long n, int hc, long lr, long ncol, long hrow,
                         long irow, long xrow, long qrow, long srow, long rqrow, long rsrow,
                         long kc, int ks, float eps, float wadd, RadStream s) {
    const long HN  = (long)hc * n;
    const long ntt = (ncol + 15) / 16;                        /* plane n-tiles */
    uint4* const          plane = (uint4*)wsp;
    unsigned short* const udec  = f8 ? (unsigned short*)(wsp + ntt * 16 * HN * 2) : nullptr;
    const unsigned short* ubf   = f8 ? udec : (const unsigned short*)up;

    const long nnb  = (rows * hc + kWaves - 1) / kWaves;
    const long work = ntt * (HN / 16) * kWave + (udec ? HN * (lr / 8) : 0);
    long ndec = (work + kThreads - 1) / kThreads;
    if (ndec > 512) ndec = 512;
    hipLaunchKernelGGL(hc_pf2_prep_kernel, dim3((unsigned)(nnb + ndec)), dim3(kThreads), 0,
                       (hipStream_t)s, h, scale, rows, n, hc, hrow, eps, nnb, down, dld, f8, injw,
                       (const unsigned char*)up, uld, plane, udec, lr, ncol, ntt);
    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;

    float* const kpart = ks > 1 ? part : nullptr;
    hipLaunchKernelGGL(hc_down_pf2_kernel,
                       dim3((unsigned)((rows + kPf2RT - 1) / kPf2RT), (unsigned)ks),
                       dim3(kThreads), 0, (hipStream_t)s, h, w, (const uint4*)plane,
                       (const float*)scale, dbuf, inj, kpart, rows, n, hc, lr, ncol, ntt, hrow,
                       irow, kc, wadd);
    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;

    if (ks > 1) {
        const long tot = rows * ncol;
        hipLaunchKernelGGL(hc_down_reduce_kernel,
                           dim3((unsigned)((tot + kThreads - 1) / kThreads)), dim3(kThreads), 0,
                           (hipStream_t)s, (const float*)part, dbuf, inj, rows, lr, ncol, irow, ks,
                           1.0f / (float)hc);
        if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
    }

    /* The up product and mix: the wide kernel at 256 rows over a bf16 mix_up with row pitch lr
     * (the decoded plane, or the stored matrix at bf16). */
    hipLaunchKernelGGL((hc_up_mix_wide_kernel<4, false, kPf2UpRT>),
                       dim3((unsigned)(n / kWideCH), (unsigned)((rows + kPf2UpRT - 1) / kPf2UpRT)),
                       dim3(kThreads), 0, (hipStream_t)s, (const void*)ubf, lr,
                       (const unsigned short*)dbuf, h, w, (const float*)scale, x, rows, n, lr, hrow,
                       xrow, wadd);
    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
    hc_quant_rows_launch(i8, x, q, qs, rq, rqs, rows, n, xrow, qrow, srow, rqrow, rsrow, s);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  // namespace

// ============================================================== the plugin ABI entry points

enum { HCE_X = 0, HCE_H };
enum { HCW_Y = 0, HCW_INJ, HCW_H };
enum { HCR_H = 0, HCR_W, HCR_DOWN, HCR_UP, HCR_INJW, HCR_X, HCR_INJ, HCR_Q, HCR_S, HCR_RQ, HCR_RS };

/* LDS for kernel A's bf16 N beside its static planes: gfx1201 gives a workgroup 64 KiB, so N gets
 * 64 KiB less kHcReadAStaticLds (30576 elements). The row table claims `n <= 7552` and `hc <= 4`
 * (30208); the shim refuses larger shapes instead of failing the launch. */
static const long kHcLdsMaxHN = (65536 - kHcReadAStaticLds) / (long)sizeof(unsigned short);

static int hc_common(const RadArgs* a, const RadTensor* h, long long* rows, long long* n,
                     long long* hc, long long* hrow) {
    *n  = r4d_geti_or(a, "n", 0);
    *hc = r4d_geti_or(a, "hc", 0);
    if (*n <= 0 || *hc <= 1 || *hc > kHcMax) return R4D_NO(RAD_E_SHAPE);
    const long long HN = *n * *hc;
    *hrow = r4d_rows_pitch(h, HN, rows);
    if (*hrow < 0 || *rows <= 0) return R4D_NO(RAD_E_STRIDE);
    return RAD_OK;
}

extern "C" int r4d_rad_hc_enter(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, HCE_X);
    const RadTensor* h = r4d_opt(a, HCE_H);
    if (!x || !h) return R4D_NO(RAD_E_INVAL);
    R4D_DT(x, RAD_BF16); R4D_DT(h, RAD_BF16);
    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    long long xrows = 0;
    const long long xrow = r4d_rows_pitch(x, n, &xrows);
    if (xrow < 0 || xrows < rows) return R4D_NO(RAD_E_STRIDE);

    const unsigned blocks = (unsigned)((n + kThreads - 1) / kThreads);
#define R4D_HCE(CH_)                                                                             \
    hipLaunchKernelGGL(hc_enter_kernel<CH_>, dim3(blocks, (unsigned)rows), dim3(kThreads), 0,    \
                       (hipStream_t)s, (const unsigned short*)x->data, (unsigned short*)h->data, \
                       (long)rows, (long)n, (int)hc, (long)xrow, (long)hrow)
    if (r4d_geti_or(a, "chain", 0) != 0 && rows <= 64) R4D_HCE(r4dchain);
    else                                              R4D_HCE(R4dUnchained);
#undef R4D_HCE
    return (int)hipGetLastError();
}

extern "C" int r4d_rad_hc_write(const RadArgs* a, RadStream stream) {
    const RadTensor* y   = r4d_opt(a, HCW_Y);
    const RadTensor* inj = r4d_opt(a, HCW_INJ);
    const RadTensor* h   = r4d_opt(a, HCW_H);
    if (!y || !inj || !h) return R4D_NO(RAD_E_INVAL);
    R4D_DT(y, RAD_BF16); R4D_DT(inj, RAD_BF16); R4D_DT(h, RAD_BF16);
    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    long long yrows = 0, irows = 0;
    const long long yrow = r4d_rows_pitch(y, n, &yrows);
    const long long irow = r4d_rows_pitch(inj, hc, &irows);
    if (yrow < 0 || irow < 0 || yrows < rows || irows < rows) return R4D_NO(RAD_E_STRIDE);

    const unsigned blocks = (unsigned)((n + kThreads - 1) / kThreads);
    return R4D_EMIT(stream, hc_write_kernel, dim3(blocks, (unsigned)rows), dim3(kThreads), 0,
                    (const unsigned short*)y->data,
                    (const unsigned short*)inj->data, (unsigned short*)h->data,
                    (long)rows, (long)n, (int)hc, (long)yrow, (long)irow, (long)hrow);
}

enum { HCP_H = 0, HCP_V, HCP_ON };

/* The control vector's projection. `v` is one layer's unit direction, [n], used for EVERY stream
 * (the dot is per stream against the same direction): a [hc, n] table is refused here rather than
 * read as one direction a stream, which is what a per-stream offset would silently do. */
extern "C" int r4d_rad_hc_project(const RadArgs* a, RadStream stream) {
    const RadTensor* h  = r4d_opt(a, HCP_H);
    const RadTensor* v  = r4d_opt(a, HCP_V);
    const RadTensor* on = r4d_opt(a, HCP_ON);
    if (!h || !v || !on) return R4D_NO(RAD_E_INVAL);
    R4D_DT(h, RAD_BF16); R4D_DT(v, RAD_F32); R4D_DT(on, RAD_I32);
    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    long long vrows = 0, onrows = 0;
    if (r4d_rows_pitch(v, n, &vrows) < 0 || vrows != 1) return R4D_NO(RAD_E_STRIDE);
    if (r4d_rows_pitch(on, 1, &onrows) < 0 || onrows < rows) return R4D_NO(RAD_E_STRIDE);

    return R4D_EMIT(stream, hc_project_kernel, dim3((unsigned)hc, (unsigned)rows), dim3(kThreads),
                    0, (unsigned short*)h->data, (const float*)v->data,
                    (const int32_t*)on->data, (long)rows, (long)n, (long)hrow,
                    (float)r4d_getf(a, "scale", 1.0));
}

enum { MTE_H = 0, MTE_E, MTE_WH, MTE_WE, MTE_FCH, MTE_FCE, MTE_X };

extern "C" int r4d_rad_mtp_enter(const RadArgs* a, RadStream stream) {
    int emit_rc = RAD_OK;
    const RadTensor* h   = r4d_opt(a, MTE_H);
    const RadTensor* e   = r4d_opt(a, MTE_E);
    const RadTensor* wh  = r4d_opt(a, MTE_WH);
    const RadTensor* we  = r4d_opt(a, MTE_WE);
    const RadTensor* fch = r4d_opt(a, MTE_FCH);
    const RadTensor* fce = r4d_opt(a, MTE_FCE);
    const RadTensor* x   = r4d_opt(a, MTE_X);
    if (!h || !e || !wh || !we || !fch || !fce || !x) return R4D_NO(RAD_E_INVAL);
    R4D_DT(h, RAD_BF16); R4D_DT(e, RAD_BF16); R4D_DT(x, RAD_BF16);
    R4D_DT(wh, RAD_F32); R4D_DT(we, RAD_F32);
    /* Both fc matrices bf16, or both E4M3 rows with f32 128-column group scales in each row's tail
     * (r4d_hc_fp8.h at K = n), per the stored dtype. */
    const bool f8 = fch->dtype == RAD_F8E4M3;
    if (f8) { R4D_DT(fce, RAD_F8E4M3); } else { R4D_DT(fch, RAD_BF16); R4D_DT(fce, RAD_BF16); }

    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    const long long HN = n * hc;
    /* `nh` and `ne` stay in LDS for the second phase, so the wide stream plus one row must fit
     * beside the staging slots and reduction planes in 64 KiB. */
    if ((HN + n) > kMtpLdsMaxN) return R4D_NO(RAD_E_SHAPE);

    long long erows = 0, xrows = 0;
    const long long erow = r4d_rows_pitch(e, n, &erows);
    const long long xrow = r4d_rows_pitch(x, HN, &xrows);
    if (erow < 0 || xrow < 0 || erows < rows || xrows < rows) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(wh) < HN || r4d_numel(we) < n) return R4D_NO(RAD_E_SHAPE);
    /* A stored row's pitch in the stored dtype's elements. At E4M3 the declared shape ([n, n])
     * arrives, so the pitch comes from the layout rule, as in hc_read. */
    long long fld = n;
    if (f8) {
        if (!r4d_hc8::group_ok(n, r4d_hc8::kDownGroup)) return R4D_NO(RAD_E_SHAPE);
        if (fch->rank != 2 || fch->shape[0] != n || fch->shape[1] != n ||
            fce->rank != 2 || fce->shape[0] != n || fce->shape[1] != n)
            return R4D_NO(RAD_E_SHAPE);
        fld = r4d_hc8::row_bytes(n, r4d_hc8::kDownGroup);
    } else if (r4d_numel(fch) < n * n || r4d_numel(fce) < n * n) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (!r4d_contig(wh) || !r4d_contig(we) || !r4d_contig(fch) || !r4d_contig(fce))
        return R4D_NO(RAD_E_STRIDE);

    /* `ngroup` == 1 is the flat norm: one rsqrt over the whole wide row instead of one per
     * sub-stream. The checkpoint does not determine which, so it is a parameter; the default is
     * grouped, as in every gated residual of this architecture. */
    const long long ngroup = r4d_geti_or(a, "ngroup", hc);
    if (ngroup != 1 && ngroup != hc) return R4D_NO(RAD_E_SHAPE);
    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);

    const unsigned blocks = (unsigned)((n + kMtpOutTB - 1) / kMtpOutTB);
    const size_t   lds    = (size_t)(HN + n) * sizeof(unsigned short);

    /* The row count picks the form, as for the gated residual: the per-row kernel is the floor at
     * one row but re-reads the weight per row. */
    const long R = rows * hc;
    const bool tileable = (n % kTKT) == 0 && (n % 8) == 0 && (hrow % 8) == 0
        && (((uintptr_t)h->data & 15u) == 0u) && (((uintptr_t)e->data & 15u) == 0u)
        && (((uintptr_t)fch->data & 15u) == 0u) && (((uintptr_t)fce->data & 15u) == 0u)
        && (((uintptr_t)wh->data & 15u) == 0u) && (((uintptr_t)we->data & 15u) == 0u)
        && (erow % 8) == 0;
    /* E4M3 fc rows always use the tiled kernel, even at one row (46 us, against 57 for a per-row
     * E4M3 dot): the per-row kernel's advantage is its transposed LDS layout, which codes cannot
     * use. A shape the tile cannot take is refused. */
    if (f8 && !tileable) return R4D_NO(RAD_E_SHAPE);
    /* The prefill form from kMtpPrefillRows tokens, on its scratch; it accepts every shape the
     * tile does. */
    if (rows >= kMtpPrefillRows && tileable) {
        if (!a->scratch || a->scratch_bytes < mtp_prefill_bytes(rows, n, hc))
            return R4D_NO(RAD_E_SCRATCH);
        auto al = [](long long b) { return (b + 255) & ~255LL; };
        float* rs = (float*)a->scratch;
        float* re = (float*)((unsigned char*)a->scratch + al(R * 4));
        float* et = (float*)((unsigned char*)a->scratch + al(R * 4) + al(rows * 4));
        const unsigned sb = (unsigned)((R + rows + kWaves - 1) / kWaves);
        hipLaunchKernelGGL(mtp_prefill_scale_kernel, dim3(sb), dim3(kThreads), 0,
                           (hipStream_t)stream,
                           (const unsigned short*)h->data, (const unsigned short*)e->data, rs, re,
                           (long)rows, (long)n, (int)hc, ngroup == 1 ? 1 : 0, eps, (long)hrow,
                           (long)erow);
        const dim3 ge((unsigned)((n + 127) / 128), (unsigned)((rows + 127) / 128));
        const dim3 gh((unsigned)((n + 127) / 128), (unsigned)((R + 127) / 128));
#define MTP_PF(F8_, EP_, G_, SRC_, SROW_, GAIN_, RSC_, FC_)                                      \
        hipLaunchKernelGGL((mtp_prefill_gemm_kernel<F8_, EP_>), G_, dim3(kThreads), 0,           \
                           (hipStream_t)stream, (const unsigned short*)SRC_, (long)SROW_,       \
                           (const float*)GAIN_, RSC_, (const void*)FC_, (long)fld, et,           \
                           (unsigned short*)x->data, (long)rows, (long)n, (int)hc, wadd,         \
                           (long)xrow)
        if (f8) {
            MTP_PF(true,  true,  ge, e->data, erow, we->data, re, fce->data);
            MTP_PF(true,  false, gh, h->data, hrow, wh->data, rs, fch->data);
        } else {
            MTP_PF(false, true,  ge, e->data, erow, we->data, re, fce->data);
            MTP_PF(false, false, gh, h->data, hrow, wh->data, rs, fch->data);
        }
#undef MTP_PF
        return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
    }
    if ((f8 || rows >= kMtpTileMinRows) && tileable) {
        /* The tile for a row count; the E4M3 form's short tiles as kMtpTNTShort notes, where K
         * is a whole number of their passes. */
        const bool shortk = f8 && R <= 32 && (n % kMtpKTShort) == 0;
        const int rt = (R > 64) ? 128 : (R > 32) ? 64 : (shortk && R <= 16) ? 16 : 32;
        const int nt = shortk ? kMtpTNTShort : rt >= 128 ? kMtpTNTWide : kMtpTNTNarrow;
        const dim3 gm((unsigned)((n + nt - 1) / nt), (unsigned)((R + rt - 1) / rt));
        const bool chain = r4d_geti_or(a, "chain", 0) != 0 && R <= 64;
#define MTP_TILE_CASE(RT_, NT_, F8_, KT_)                                                        \
        do { if (chain) MTP_TILE_CH(RT_, NT_, F8_, r4dchain, KT_);                               \
             else       MTP_TILE_CH(RT_, NT_, F8_, R4dUnchained, KT_); } while (0)
#define MTP_TILE_CH(RT_, NT_, F8_, CH_, KT_)                                                     \
        emit_rc = R4D_EMIT(stream, (mtp_enter_tiled_kernel<RT_, NT_, F8_, CH_, KT_>), gm,        \
                           dim3(kThreads), 0,                                                    \
                           (const unsigned short*)h->data,                                       \
                           (const unsigned short*)e->data, (const float*)wh->data,               \
                           (const float*)we->data, (const void*)fch->data,                       \
                           (const void*)fce->data, (long)fld, (unsigned short*)x->data,          \
                           (long)rows, (long)n, (int)hc, ngroup == 1 ? 1 : 0, eps, wadd,         \
                           (long)hrow, (long)erow, (long)xrow)
        switch (rt) {
            case 16:  MTP_TILE_CASE(16, kMtpTNTShort, true, kMtpKTShort);  break;
            case 32:  if (shortk)  MTP_TILE_CASE(32, kMtpTNTShort, true, kMtpKTShort);
                      else if (f8) MTP_TILE_CASE(32, kMtpTNTNarrow, true, kTKT);
                      else         MTP_TILE_CASE(32, kMtpTNTNarrow, false, kTKT);  break;
            case 64:  if (f8) MTP_TILE_CASE(64, kMtpTNTNarrow, true, kTKT);
                      else    MTP_TILE_CASE(64, kMtpTNTNarrow, false, kTKT);  break;
            default:  if (f8) MTP_TILE_CASE(128, kMtpTNTWide, true, kTKT);
                      else    MTP_TILE_CASE(128, kMtpTNTWide, false, kTKT);   break;
        }
#undef MTP_TILE_CASE
#undef MTP_TILE_CH
        if (emit_rc != RAD_OK) return emit_rc;
        return (int)hipGetLastError();
    }

    emit_rc = R4D_EMIT(stream, mtp_enter_kernel, dim3(blocks, (unsigned)rows), dim3(kThreads), lds,
                       (const unsigned short*)h->data,
                       (const unsigned short*)e->data, (const float*)wh->data,
                       (const float*)we->data, (const unsigned short*)fch->data,
                       (const unsigned short*)fce->data, (unsigned short*)x->data,
                       (long)rows, (long)n, (int)hc, ngroup == 1 ? 1 : 0, eps, wadd,
                       (long)hrow, (long)erow, (long)xrow);
    if (emit_rc != RAD_OK) return emit_rc;
    return (int)hipGetLastError();
}

/* The prefill form's scratch, for a band that reaches it. The form is chosen per call, so a band
 * reaching kMtpPrefillRows is sized at its ceiling `M`. */
extern "C" int64_t r4d_rad_mtp_enter_scratch(const RadArgs* a) {
    const long long M  = r4d_geti_or(a, "M", 0);
    const long long n  = r4d_geti_or(a, "n", 0);
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (M <= 0 || n <= 0 || hc <= 0) return R4D_NO(RAD_E_SHAPE);
    return M >= kMtpPrefillRows ? mtp_prefill_bytes(M, n, hc) : 0;
}
/* `d` between the two kernels, with the per-stream norm scales beside it. Sized from the band's `M`,
 * not the operand: the arena is sized once per band and the launch checks the same number. */
extern "C" int64_t r4d_rad_hc_read_scratch(const RadArgs* a) {
    const long long M  = r4d_geti_or(a, "M", 0);
    const long long lr = r4d_geti_or(a, "lowrank", 0);
    const long long n  = r4d_geti_or(a, "n", 0);
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (M <= 0 || lr <= 0 || n <= 0 || hc <= 0) return R4D_NO(RAD_E_SHAPE);
    /* `d` [M, lowrank] bf16, the per-stream norm scales [M, kHcMax] f32, and `u` [M, hc*n] bf16
     * (20 KiB per row here; it gives the up matvec its own grid). At a band reaching the PF2 rule
     * `u` also holds the weight planes (hc_u_rows).
     *
     * Plus the split-K partials: every row count below a few thousand splits, so this is not
     * bounded by kTileMinRows; hc_part_bytes walks the band. */
    const long long HN   = hc * n;
    /* The split must come from the same number here and at the call site. Only the call site knows
     * whether there are inject weights, so both derive `kc` from `lr` and the buffer is sized for
     * the widest ncol; mismatched derivations overrun the partial buffer. */
    return M * ((long long)lr * 2 + (long long)kHcMax * 4) + hc_u_rows(M, lr, hc) * HN * 2
           + hc_part_bytes(M, lr, HN);
}

extern "C" int r4d_rad_hc_read(const RadArgs* a, RadStream s) {
    int emit_rc = RAD_OK;
    /* The chained forms (r4d_chain.h): the fused grid of a small decode step, and the tiled
     * read's two launches to kNdFuseRows rows. */
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    const RadTensor* h    = r4d_opt(a, HCR_H);
    const RadTensor* w    = r4d_opt(a, HCR_W);
    const RadTensor* down = r4d_opt(a, HCR_DOWN);
    const RadTensor* up   = r4d_opt(a, HCR_UP);
    const RadTensor* injw = r4d_opt(a, HCR_INJW);
    const RadTensor* x    = r4d_opt(a, HCR_X);
    const RadTensor* inj  = r4d_opt(a, HCR_INJ);
    const RadTensor* q    = r4d_opt(a, HCR_Q);
    const RadTensor* qs   = r4d_opt(a, HCR_S);
    const RadTensor* rq   = r4d_opt(a, HCR_RQ);
    const RadTensor* rqs  = r4d_opt(a, HCR_RS);
    if (!h || !w || !down || !up || !x) return R4D_NO(RAD_E_INVAL);
    R4D_DT(h, RAD_BF16); R4D_DT(w, RAD_F32);
    /* The mixing matrices are both bf16 or both E4M3 rows (r4d_hc_fp8.h), per the stored dtype
     * written by this row's layout hook. */
    const bool f8 = down->dtype == RAD_F8E4M3;
    if (f8) { R4D_DT(up, RAD_F8E4M3); } else { R4D_DT(down, RAD_BF16); R4D_DT(up, RAD_BF16); }
    R4D_DT(x, RAD_BF16);
    /* An inject weight without an `inj` output (or vice versa) is inconsistent and refused. */
    if ((injw != nullptr) != (inj != nullptr)) return R4D_NO(RAD_E_INVAL);
    if (injw) { R4D_DT(injw, RAD_BF16); R4D_DT(inj, RAD_BF16); }
    if ((q != nullptr) != (qs != nullptr)) return R4D_NO(RAD_E_INVAL);
    /* The codes' dtype is the format: E4M3 for an fp8 trunk, int8 for an int8 one (gemm_i8a8_*),
     * on the same scale grid (r4d_q8.h). The rotated codes take it too: E4M3 as had_quant_act_fp8
     * writes them, or beside int8 codes had_quant_act_i8g's, the routed experts' operand either
     * way. */
    if (q && q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    if (q) R4D_DT(qs, RAD_F32);
    const bool i8 = q && q->dtype == RAD_I8;
    /* The rotated codes are produced by the plain codes' finisher, so they require them. */
    if ((rq != nullptr) != (rqs != nullptr) || (rq && !q)) return R4D_NO(RAD_E_INVAL);
    if (rq) {
        if (rq->dtype != (i8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3)) return R4D_NO(RAD_E_DTYPE);
        R4D_DT(rqs, RAD_F32);
    }

    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    const long long HN = n * hc;
    if (HN > kHcLdsMaxHN) return R4D_NO(RAD_E_SHAPE);
    /* A workgroup owns 128 channels and their scale group, so n must be whole groups. */
    if (n % kUpTB) return R4D_NO(RAD_E_SHAPE);

    const long long lr = r4d_geti_or(a, "lowrank", 0);
    if (lr <= 0) return R4D_NO(RAD_E_SHAPE);
    /* Row pitch in elements of the stored dtype: HN or lr for bf16, the stored row's bytes for
     * E4M3, whose dot needs K a multiple of its 256-element step and whose up rows split into four
     * sixteen-code quarters of at most kUpMaxQ loads. */
    long long dld = HN, uld = lr;
    if (f8) {
        const long long gd = r4d_hc8::down_group(HN), gu = r4d_hc8::up_group(lr);
        if (!r4d_hc8::group_ok(HN, gd) || !r4d_hc8::group_ok(lr, gu) || HN % kHc8Step ||
            gu / 16 > kUpMaxQ)
            return R4D_NO(RAD_E_SHAPE);
        /* A weight's RadTensor carries the declared shape with the stored dtype (as for the fp8
         * lm_head), so the row pitch is derived by the layout hook's rule; the engine's layout-tag
         * check guarantees the plane matches. */
        if (down->rank != 2 || down->shape[0] != lr || down->shape[1] != HN ||
            up->rank != 2 || up->shape[0] != HN || up->shape[1] != lr)
            return R4D_NO(RAD_E_SHAPE);
        dld = r4d_hc8::row_bytes(HN, gd);
        uld = r4d_hc8::row_bytes(lr, gu);
    } else if (r4d_numel(down) < lr * HN || r4d_numel(up) < HN * lr) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (r4d_numel(w) < HN) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(w) || !r4d_contig(down) || !r4d_contig(up)) return R4D_NO(RAD_E_STRIDE);
    if (injw && (!r4d_contig(injw) || r4d_numel(injw) < hc * HN)) return R4D_NO(RAD_E_STRIDE);

    long long xrows = 0, irows = 0;
    const long long xrow = r4d_rows_pitch(x, n, &xrows);
    if (xrow < 0 || xrows < rows) return R4D_NO(RAD_E_STRIDE);
    long long irow = hc;
    if (inj) {
        irow = r4d_rows_pitch(inj, hc, &irows);
        if (irow < 0 || irows < rows) return R4D_NO(RAD_E_STRIDE);
    }

    long long qrow = 0, srow = 0;
    if (q) {
        const long long group = r4d_geti_or(a, "group", 0);
        if (group != kFp8Grp) return R4D_NO(RAD_E_SHAPE);
        long long qrows = 0, srows = 0;
        qrow = r4d_rows_pitch(q, n, &qrows);
        srow = r4d_rows_pitch(qs, n / kFp8Grp, &srows);
        if (qrow < 0 || srow < 0 || qrows < rows || srows < rows) return R4D_NO(RAD_E_STRIDE);
    }
    long long rqrow = 0, rsrow = 0;
    if (rq) {
        long long qrows = 0, srows = 0;
        rqrow = r4d_rows_pitch(rq, n, &qrows);
        rsrow = r4d_rows_pitch(rqs, n / kFp8Grp, &srows);
        if (rqrow < 0 || rsrow < 0 || qrows < rows || srows < rows) return R4D_NO(RAD_E_STRIDE);
        /* A lane stores its four codes as one dword. */
        if ((((uintptr_t)rq->data | (uintptr_t)rqrow) & 3u) != 0u) return R4D_NO(RAD_E_STRIDE);
    }

    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);

    /* Check what this launch touches, not the scratch hook's total: the hook walks the band, which
     * is too slow per launch. `d`, the scales and `u` sit at offsets fixed by the band's M; the
     * split-K partials are checked in the tiled branch, where the split is known. */
    const long long Mband = r4d_geti_or(a, "M", 0);
    if (Mband <= 0) return R4D_NO(RAD_E_SHAPE);
    /* `d`, the scales and `u` are sized from the band; a wider step would overrun them. */
    if (rows > Mband) return R4D_NO(RAD_E_SHAPE);
    const long long ubytes = hc_u_rows(Mband, lr, hc) * HN * 2;
    const long long fixed  = Mband * (lr * 2 + (long long)kHcMax * 4) + ubytes;
    if (!a->scratch || a->scratch_bytes < fixed) return R4D_NO(RAD_E_SCRATCH);
    unsigned short* dbuf  = (unsigned short*)a->scratch;
    float*          scale = (float*)((unsigned char*)a->scratch + Mband * lr * 2);
    unsigned short* ubuf  = (unsigned short*)((unsigned char*)a->scratch
                                              + Mband * (lr * 2 + kHcMax * 4));

    /* The column tile is chosen to fill the grid: 64 of 320 columns per workgroup would be five
     * workgroups on 64 CUs. Narrowing costs one extra pass over `h` per workgroup (20 KiB against
     * up to 1.3 MiB of weight). It only narrows; rows past kDecMinRows take the tiled path, so
     * `rows` here is a one-to-seven-row speculative window. */
    const bool can_tile = (n % kTKT) == 0 && (HN % 8) == 0 && (lr % 8) == 0 && (hrow % 8) == 0
                          && (((uintptr_t)h->data & 15u) == 0u)
                          && (((uintptr_t)down->data & 15u) == 0u)
                          && (((uintptr_t)up->data & 15u) == 0u);
    /* The row count picks the form. The per-row kernels read both mixing matrices once per row:
     * four rows is hidden by an issue-bound kernel, but 32 (eight sequences at depth 3) is 210 MB
     * per connection. So from kDecMinRows the products run tiled, 128 rows when there are enough
     * (prefill), else 32 or 64 with a K-split down product. Below kDecMinRows the per-row kernels
     * win. */
    bool fused = false;
    /* The shape the fused kernel is built for keeps its rows per-row to kFusedMaxRows, under the
     * two-grid pin too: the pinned pair is then the same bodies as the fused grid. */
    const bool fuses  = f8 && hc == 4 && n == 10LL * kThreads;
    const bool tiled  = can_tile && rows >= (fuses ? kFusedMaxRows + 1 : kDecMinRows);
    if (tiled) {
        const int  rt   = hc_down_rt(rows);
        /* The inject gains, if any, are hc more columns of the same GEMM. */
        const bool fold = injw != nullptr && inj != nullptr;
        const long ncol = fold ? lr + hc : lr;
        const long nrt  = (rows + rt - 1) / rt;
        const bool wdown = rt == kWideRT && g_hc_prefill_form != 1;
        const long kc   = hc_down_kchunk(lr, HN, nrt);
        const int  ks   = (int)((HN + kc - 1) / kc);
        if (ks > 1 && a->scratch_bytes < fixed + (long long)ks * rows * ncol * 4)
            return R4D_NO(RAD_E_SCRATCH);
        float*     part = (ks > 1) ? (float*)((unsigned char*)a->scratch + fixed) : nullptr;

        /* PF2 (hc_pf2_launch) at the 128-row tile, same bytes: used from kPf2MinRows rows when
         * the shape fits (hc 4, n a multiple of 256, at most kPf2MaxNT column n-tiles), else the
         * wide form. Its planes live in `u`, sized by the scratch hook at every band the rule
         * reaches, so the room check can only refuse a pin below kPf2MinRows. A pin also refuses
         * an unsupported shape, so a test cannot pass on a form it did not ask for. */
        const long ntt       = (ncol + 15) / 16;
        const bool pf2_shape = rt == kWideRT && hc == 4 && (n % kThreads) == 0
                               && ntt <= (long)kPf2MaxNT;
        const bool pf2       = g_hc_prefill_form == 3
                               || (g_hc_prefill_form == 0 && rows >= kPf2MinRows && pf2_shape);
        if (pf2) {
            if (!pf2_shape) return R4D_NO(RAD_E_SHAPE);
            if (ubytes < pf2_plane_bytes(HN, lr, ntt, f8)) return R4D_NO(RAD_E_SCRATCH);
            const int prc = hc_pf2_launch(
                f8, i8, (const unsigned short*)h->data, (const float*)w->data, down->data,
                (long)dld, up->data, (long)uld, fold ? (const unsigned short*)injw->data : nullptr,
                dbuf, scale, (unsigned char*)ubuf, part, (unsigned short*)x->data,
                fold ? (unsigned short*)inj->data : nullptr,
                q ? (unsigned char*)q->data : nullptr, q ? (float*)qs->data : nullptr,
                rq ? (unsigned char*)rq->data : nullptr, rq ? (float*)rqs->data : nullptr,
                (long)rows, (long)n, (int)hc, (long)lr, ncol, (long)hrow, (long)irow, (long)xrow,
                (long)qrow, (long)srow, (long)rqrow, (long)rsrow, kc, ks, eps, wadd, s);
            return prc == RAD_OK ? RAD_OK : R4D_NO(prc);
        }

        /* On the decode tiles the norm scales share the down product's grid and its split reduces
         * itself (hc_norm_down_tiled_kernel): two launches a read where there were four. The
         * counters are the mix finisher's block, one per (column tile, row tile), which the up-mix
         * launch behind this one finds zero again. Alone the two forms are level (us a read, four
         * launches -> two: 8 rows 31.8 -> 31.3, 16 rows 34.3 -> 33.8, 32 rows 37.0 -> 37.9,
         * 48 rows 57.6 -> 62.2, 64 rows 68.0 -> 68.0); in a step each launch saved is about 3 us of
         * dispatch ordering besides. */
        unsigned int* dctr = nullptr;
        if (rows <= kNdFuseRows && ks > 1 && !wdown && (rt == 32 || rt == 64) &&
            ((ncol + kTNT - 1) / kTNT) * nrt <= (long)kMaxMixCtr)
            dctr = hc_mix_ctr(s);
        const bool ndfuse = dctr != nullptr;

        /* The per-s norm scales: one workgroup per (row, s). */
        if (!ndfuse)
            hipLaunchKernelGGL(hc_norm_scale_kernel,
                               dim3((unsigned)hc, (unsigned)rows), dim3(kThreads), 0,
                               (hipStream_t)s, (const unsigned short*)h->data, scale,
                               (long)rows, (long)n, (long)hrow, eps);
        if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);

        if (ndfuse)
            hc_norm_down_tiled_launch(rt, f8, (const unsigned short*)h->data,
                                      (const float*)w->data, down->data, (long)dld, scale,
                                      fold ? (const unsigned short*)injw->data : nullptr,
                                      dbuf, fold ? (unsigned short*)inj->data : nullptr, part,
                                      (long)rows, (long)n, (int)hc, (long)lr, ncol, (long)hrow,
                                      (long)irow, kc, wadd, eps, s, dctr, dctr + kMaxMixCtr, chain);
        else if (wdown)
            hc_down_wide_launch(f8, (const unsigned short*)h->data, (const float*)w->data,
                                down->data, (long)dld, scale,
                                fold ? (const unsigned short*)injw->data : nullptr,
                                dbuf, fold ? (unsigned short*)inj->data : nullptr, part,
                                (long)rows, (long)n, (int)hc, (long)lr, ncol, (long)hrow,
                                (long)irow, kc, wadd, s);
        else
            hc_down_tiled_launch(rt, f8, (const unsigned short*)h->data, (const float*)w->data,
                                 down->data, (long)dld, scale,
                                 fold ? (const unsigned short*)injw->data : nullptr,
                                 dbuf, fold ? (unsigned short*)inj->data : nullptr, part,
                                 (long)rows, (long)n, (int)hc, (long)lr, ncol, (long)hrow,
                                 (long)irow, kc, wadd, s, dctr);
        if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);

        if (ks > 1 && !dctr) {
            const long tot = (long)rows * ncol;
            hipLaunchKernelGGL(hc_down_reduce_kernel,
                               dim3((unsigned)((tot + kThreads - 1) / kThreads)), dim3(kThreads),
                               0, (hipStream_t)s,
                               part, dbuf, fold ? (unsigned short*)inj->data : nullptr,
                               (long)rows, (long)lr, ncol, (long)irow, ks, 1.0f / (float)hc);
            if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);
        }

        /* Up product and mix in one launch when the tile divides into (s, channel) pairs and
         * the finisher's counters fit: hc_up_mix_tiled_kernel at 32- and 64-row tiles,
         * hc_up_mix_wide_kernel at the prefill tile (registers instead of the 58 KiB of LDS staging
         * planes). */
        const int ipb = (hc > 0 && (kTNT % hc) == 0) ? kTNT / hc : 0;
        const long ntile = (long)(n / kUpTB) * ((rows + rt - 1) / rt);
        unsigned int* mctr = nullptr;
        const bool wide = rt == kWideRT && hc == 4 && g_hc_prefill_form != 1;
        if (g_hc_prefill_form == 2 && !wide) return R4D_NO(RAD_E_SHAPE);
        fused = wide || (rt <= 64 && ipb > 0 && (kUpTB % ipb) == 0 && (n % ipb) == 0
                         && ntile <= (long)kMaxMixCtr);
        if (fused && !wide && q) { mctr = hc_mix_ctr(s); fused = (mctr != nullptr); }
        if (fused && wide)
            hc_up_mix_wide_launch(f8, i8, up->data, (long)uld, dbuf, (const unsigned short*)h->data,
                                  (const float*)w->data, scale, (unsigned short*)x->data,
                                  q ? (unsigned char*)q->data : nullptr,
                                  q ? (float*)qs->data : nullptr,
                                  rq ? (unsigned char*)rq->data : nullptr,
                                  rq ? (float*)rqs->data : nullptr,
                                  (long)rows, (long)n, (long)lr, (long)hrow, (long)xrow,
                                  (long)qrow, (long)srow, (long)rqrow, (long)rsrow, wadd, s);
        else if (fused)
            hc_up_mix_tiled_launch(rt, f8, i8, up->data, (long)uld, dbuf,
                                   (const unsigned short*)h->data, (const float*)w->data, scale,
                                   (unsigned short*)x->data,
                                   q ? (unsigned char*)q->data : nullptr,
                                   q ? (float*)qs->data : nullptr,
                                   rq ? (unsigned char*)rq->data : nullptr,
                                   rq ? (float*)rqs->data : nullptr, mctr,
                                   (long)rows, (long)n, (int)hc, (long)lr, (long)hrow,
                                   (long)xrow, (long)qrow, (long)srow, (long)rqrow, (long)rsrow,
                                   wadd, s, chain);
        else
            hc_up_tiled_launch(rt, f8, up->data, (long)uld, dbuf, ubuf,
                               (long)rows, (long)HN, (long)lr, s);
        if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    } else {
    long long ta = kDownTAMax;
    /* Target two workgroups per WGP, i.e. one per CU, since want_wg() returns WGPs (see its note).
     * A one-WGP target would accept 44 blocks at a depth-3 verify (four rows, ta = 32); the one-CU
     * target is worth ~1.3% of the step there. Below about 64 blocks the step degrades sharply.
     *
     * The ladder halves ta, landing on 44 / 84 / 164 blocks. At one row ta's floor of 4 (81 blocks,
     * 18.6 us) is a sharp optimum: ta 5 (65 blocks) is +50%, ta 3 (108) +76%, while ta 11-24 sit
     * within 1.7 us of each other. Performance is not smooth in block count, so leave ta at 4. */
    while (ta > 4 && ((lr + ta - 1) / ta) * rows < 2 * want_wg()) ta >>= 1;

    /* The preloaded prologue at the shape it is instantiated for (hc 4, n 2560), and
     * there kernels A and B as one grid (hc_read_fused_kernel) when the row group is the whole
     * step, the up-mix fuses (q's counters or none) and the per-stream counters exist. */
    const bool pre = f8 && hc == 4 && n == 10LL * kThreads && g_hc_decode_form != 1;
    if (pre) {
        const long rgf = hc_up_rg(rows, lr);
        const int       rpt = hc_up_rpt(true, (int)hc, n);
        const long ipf = (long)kUpRows * rpt / hc;
        const long lds = HN * 2 > rgf * lr * 4 ? HN * 2 : rgf * lr * 4;
        unsigned int* fctr = (rgf >= rows && rows <= kFusedMaxRows && kUpTB % ipf == 0 && n % ipf == 0
                              && lds + kHcReadAStaticLds + 8 * kUpRows * 4 * 2 + 256 <= 65536)
                                 ? hc_mix_ctr(s) : nullptr;
        if (fctr) {
            const unsigned gxa = (unsigned)((lr + ta - 1) / ta) + 1u;
            const unsigned na  = gxa * (unsigned)rows;
            const unsigned gxb = (unsigned)(n / ipf);
            const unsigned nbf_ = gxb;          /* one row group */
#define HC_FUSED_E(RG_, I8_, RPT_, CH_)                                                          \
            emit_rc = R4D_EMIT(s, (hc_read_fused_kernel<RG_, I8_, RPT_, 4, 10, CH_>),            \
                dim3(na + nbf_), dim3(kThreads), (unsigned)lds,                                  \
                (const unsigned short*)h->data, (const float*)w->data, (const void*)down->data,   \
                (long)dld, injw ? (const unsigned short*)injw->data : nullptr, dbuf,             \
                inj ? (unsigned short*)inj->data : nullptr, scale, (long)ta, (long)irow, eps, \
                (const void*)up->data, (long)uld, (unsigned short*)x->data,                  \
                q ? (unsigned char*)q->data : nullptr, q ? (float*)qs->data : nullptr,            \
                rq ? (unsigned char*)rq->data : nullptr, rq ? (float*)rqs->data : nullptr,        \
                q ? fctr : nullptr, (long)rows, (long)n, (int)hc, (long)lr,        \
                (long)hrow, (long)xrow, (long)qrow, (long)srow, (long)rqrow, \
                (long)rsrow, wadd, fctr + kMaxMixCtr, na, gxa, gxb, nbf_)
#define HC_FUSED_K(RG_, I8_, RPT_)                                                               \
            do { if (chain) HC_FUSED_E(RG_, I8_, RPT_, r4dchain);                                \
                 else       HC_FUSED_E(RG_, I8_, RPT_, R4dUnchained); } while (0)
#define HC_FUSED_R(RG_, I8_)                                                                     \
            do { if (rpt == 2) HC_FUSED_K(RG_, I8_, 2); else HC_FUSED_K(RG_, I8_, 1); } while (0)
#define HC_FUSED_CASE(RG_)                                                                       \
            do { if (i8) HC_FUSED_R(RG_, true); else HC_FUSED_R(RG_, false); } while (0)
            switch (rgf) {
                case 8:  HC_FUSED_CASE(8); break;
                case 7:  HC_FUSED_CASE(7); break;
                case 6:  HC_FUSED_CASE(6); break;
                case 5:  HC_FUSED_CASE(5); break;
                case 4:  HC_FUSED_CASE(4); break;
                case 3:  HC_FUSED_CASE(3); break;
                case 2:  HC_FUSED_CASE(2); break;
                default: HC_FUSED_CASE(1); break;
            }
#undef HC_FUSED_CASE
#undef HC_FUSED_R
#undef HC_FUSED_K
#undef HC_FUSED_E
            if (emit_rc != RAD_OK) return R4D_NO(RAD_E_DEVICE);
            return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
        }
    }

#define HC_READ_A(KERNEL_)                                                                       \
    R4D_EMIT(s, KERNEL_, dim3((unsigned)((lr + ta - 1) / ta) + 1u, (unsigned)rows),              \
             dim3(kThreads), (unsigned)(HN * sizeof(unsigned short)),                             \
             (const unsigned short*)h->data, (const float*)w->data, (const void*)down->data,       \
             (long)dld, injw ? (const unsigned short*)injw->data : nullptr, dbuf,                 \
             inj ? (unsigned short*)inj->data : nullptr, scale, (long)rows, (long)n, (int)hc,     \
             (long)lr, (long)ta, (long)hrow, (long)irow, eps, wadd)
    if (g_hc_decode_form == 1)
        emit_rc = f8 ? HC_READ_A(hc_read_a_2grid_kernel<true>) : HC_READ_A(hc_read_a_2grid_kernel<false>);
    else if (pre) emit_rc = HC_READ_A((hc_read_a_kernel<true, 4, 10>));
    else          emit_rc = f8 ? HC_READ_A(hc_read_a_kernel<true>) : HC_READ_A(hc_read_a_kernel<false>);
#undef HC_READ_A
    if (emit_rc != RAD_OK) return R4D_NO(RAD_E_DEVICE);
    if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);

    /* Up product and mix in one launch whenever hc divides a workgroup's 64 rows and the
     * finisher's counters fit (always, at this model's hc 4). */
    const long long ipb   = (hc > 0 && (kUpRows % hc) == 0) ? kUpRows / hc : 0;
    const long long slots = ((rows + hc_up_rg(rows, lr) - 1) / hc_up_rg(rows, lr)) * (n / kUpTB);
    unsigned int* mctr = nullptr;
    fused = ipb > 0 && (kUpTB % ipb) == 0 && (n % ipb) == 0 && slots <= (long long)kMaxMixCtr;
    if (fused && q) {
        mctr = hc_mix_ctr(s);
        fused = (mctr != nullptr);
    }
    if (fused)
        emit_rc = hc_up_mix_launch(up, (long)uld, f8, i8, dbuf, (const unsigned short*)h->data,
                                   (const float*)w->data,
                                   scale, (unsigned short*)x->data,
                                   q ? (unsigned char*)q->data : nullptr,
                                   q ? (float*)qs->data : nullptr,
                                   rq ? (unsigned char*)rq->data : nullptr,
                                   rq ? (float*)rqs->data : nullptr, mctr, rows, n, (int)hc, lr,
                                   hrow, xrow, qrow, srow, rqrow, rsrow, wadd, s);
    else
        emit_rc = hc_up_launch(up, (long)uld, f8, dbuf, ubuf, rows, HN, lr, s);
    if (emit_rc != RAD_OK) return R4D_NO(RAD_E_DEVICE);
    if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    }

    if (!fused) {
#define HC_MIX_EMIT(I8_)                                                                         \
        R4D_EMIT(s, hc_mix_kernel<I8_>,                                                          \
                 dim3((unsigned)(n / kUpTB), (unsigned)rows),                                    \
                 dim3(kThreads), 0,                                                              \
                 (const unsigned short*)h->data, (const float*)w->data, ubuf, scale,             \
                 (unsigned short*)x->data, q ? (unsigned char*)q->data : nullptr,                \
                 q ? (float*)qs->data : nullptr,                                                 \
                 rq ? (unsigned char*)rq->data : nullptr,                                        \
                 rq ? (float*)rqs->data : nullptr,                                               \
                 (long)rows, (long)n, (int)hc, (long)hrow, (long)xrow,                           \
                 (long)qrow, (long)srow, (long)rqrow, (long)rsrow, wadd)
        emit_rc = i8 ? HC_MIX_EMIT(true) : HC_MIX_EMIT(false);
#undef HC_MIX_EMIT
    }

    if (emit_rc != RAD_OK) return emit_rc;
    return (int)hipGetLastError();
}

extern "C" void r4d_hc_read_pin_prefill(int form) { g_hc_prefill_form = form; }
extern "C" void r4d_hc_read_pin_decode(int form) { g_hc_decode_form = form; }
