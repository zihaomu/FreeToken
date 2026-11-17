// Small encoder ops between GEMMs and attention: gated activation over two operands, QK(V) norm +
// rotary over a fused projection, mean pooling over cu_seqlens segments, bf16 -> f32 widening,
// and the audio tower's ops. As in r4d_model_bf16.hip, each kernel equals a libref function:
// reductions accumulate in f32 (seg_mean in f64, as ref does), only the store narrows, and
// transcendentals use the accurate libm forms.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t kThreads = 256;

__device__ __forceinline__ float bf2f(uint16_t h) {
    return __builtin_bit_cast(float, (uint32_t)h << 16);
}

// ---------------------------------------------------------------------------- act_mul
// ref_act_mul: y = round(act(a)) * b. The activation is rounded to bf16 before the product
// because the unfused pair stores it.
template <int ACT>
__device__ __forceinline__ float act_of(float g) {
    if (ACT == 0) return g / (1.0f + expf(-g));
    if (ACT == 1) return 0.5f * g * (1.0f + erff(g * 0.70710678118654752f));
    if (ACT == 3) return 1.0f / (1.0f + expf(-g));
    if (ACT == 5) return fmaxf(g, 0.0f);
    if (ACT == 4)   // quick_gelu: sigmoid of the bf16-rounded 1.702*g, as the unfused mul stores it
        return 1.0f / (1.0f + expf(-bf2f(f32_to_bf16(g * bf2f(f32_to_bf16(1.702f))))));
    return 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
}

// The row axis is grid-strided: grid y is capped at 65535, and the audio subsampler's
// (time, frequency) rows reach 96000. Row kernels launch min(M, kGridYMax) and loop over the rest.
constexpr uint32_t kGridYMax = 65535;
static inline uint32_t grid_rows(int64_t M) {
    return (uint32_t)(M < (int64_t)kGridYMax ? M : (int64_t)kGridYMax);
}

template <int ACT>
__global__ __launch_bounds__(kThreads) void act_mul_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, uint16_t* __restrict__ y,
    uint32_t n, uint32_t a_ld, uint32_t b_ld, uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y) {
        const float g = bf2f(f32_to_bf16(act_of<ACT>(bf2f(a[(size_t)m * a_ld + i]))));
        y[(size_t)m * y_ld + i] = f32_to_bf16(g * bf2f(b[(size_t)m * b_ld + i]));
    }
}

// ---------------------------------------------------------------------------- seg_mean
// ref_seg_mean: row s of y is the mean of x's rows [cu[s], cu[s+1]), summed in f64 and rounded
// once. A workgroup is 32 columns by 8 row lanes; each lane sums every eighth row of its column
// and the eight partial sums meet in LDS. An empty segment writes zeros.
__global__ __launch_bounds__(kThreads) void seg_mean_kernel(
    const uint16_t* __restrict__ x, const int* __restrict__ cu, uint16_t* __restrict__ y,
    uint32_t n, uint32_t x_ld, uint32_t y_ld) {
    __shared__ double part[8][33];
    const uint32_t s = blockIdx.y;
    const uint32_t col = threadIdx.x & 31, rl = threadIdx.x >> 5;
    const uint32_t i = blockIdx.x * 32 + col;
    const int b0 = cu[s], b1 = cu[s + 1];
    double acc = 0.0;
    if (i < n)
        for (int r = b0 + (int)rl; r < b1; r += 8) acc += (double)bf2f(x[(size_t)r * x_ld + i]);
    part[rl][col] = acc;
    __syncthreads();
    if (rl == 0 && i < n) {
        double t = 0.0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) t += part[k][col];
        const int len = b1 - b0;
        y[(size_t)s * y_ld + i] = f32_to_bf16(len > 0 ? (float)(t / (double)len) : 0.0f);
    }
}

// ---------------------------------------------------------------------------- qk_norm_rope
// ref_qk_norm_rope over a fused [q heads | k heads | v heads] row, in place: each q and k head is
// RMS-normed against its gain and rounded to bf16 (the unfused pair stores the norm), then
// rotated; with v_norm each v head is RMS-normed with a unit gain and not rotated.
//
// One workgroup per token, one wave per head, the head in registers. Lane l holds elements
// l, l+32, l+64, ... (neox) or pairs 2l, 2l+1, 2l+64, 2l+65, ... (gptj), so both halves of every
// rotary pair are in one lane: the norm is a wave sum and the rotation needs no exchange. The
// token's sin/cos are computed once into LDS (4 KiB, not an occupancy limit).
//
// Requires head_dim a multiple of 32 and, for neox, rotary_dim / 2 a multiple of 32 (gptj:
// head_dim a multiple of 64); the launcher refuses anything else.
constexpr uint32_t kQkThreads = 256;
constexpr uint32_t kQkMaxHalf = 512;    // rotary pairs a head
constexpr int      kQkMaxE    = 32;     // elements a lane: head_dim up to 1024

/* `blocks` mode sections: pairs per section and the position component each reads. */
struct QkBlocks {
    int n = 0;
    int sec[8] = {};
    int comp[8] = {};
};

// A norm gain element, the gains bf16 or (w_f32) f32; an absent gain is one, without `wadd`.
__device__ __forceinline__ float qk_gain(const void* w, uint32_t e, int w_f32, float wadd) {
    if (!w) return 1.0f;
    return (w_f32 ? ((const float*)w)[e] : bf2f(((const uint16_t*)w)[e])) + wadd;
}

// Element `i` of the positions: an i32 value, or with `pos_f32` an f32 one read as the value it
// holds (a fractional coordinate, such as a video decoder's normalised grid times 2 pi).
__device__ __forceinline__ float qk_pos(const void* pos, int pos_f32, size_t i) {
    return pos_f32 ? ((const float*)pos)[i] : (float)((const int*)pos)[i];
}

// MODE: 0 neox, 1 gptj, 2 blocks. Positions are [M] for the first two and [C, M] (plane stride
// `pplane`) for blocks.
template <int MODE>
__global__ __launch_bounds__(kQkThreads) void qk_norm_rope_kernel(
    uint16_t* __restrict__ qkv, const void* __restrict__ pos, const void* __restrict__ qw,
    const void* __restrict__ kw, uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
    uint32_t rot, int v_norm, float theta, float eps, float rscale, float wadd, uint32_t ld,
    QkBlocks blk, uint32_t pplane, int w_f32, int pos_f32) {
    __shared__ float cs_s[kQkMaxHalf], sn_s[kQkMaxHalf];
    const uint32_t m = blockIdx.x;
    const uint32_t n_rope = n_head + n_head_kv;
    const uint32_t n_norm = n_rope + (v_norm ? n_head_kv : 0);
    const uint32_t half = rot / 2;
    uint16_t* row = qkv + (size_t)m * ld;
    const float div = rscale != 0.0f ? rscale : 1.0f;

    if (MODE == 2) {
        // Pair i of section c is the (i - start_c)-th of its block, at theta^(-j / sec[c]).
        for (uint32_t i = threadIdx.x; i < half; i += kQkThreads) {
            int c = 0, s0 = 0;
            while (c + 1 < blk.n && (int)i >= s0 + blk.sec[c]) { s0 += blk.sec[c]; ++c; }
            const float p = qk_pos(pos, pos_f32, (size_t)blk.comp[c] * pplane + m) / div;
            const float inv = 1.0f / powf(theta, (float)((int)i - s0) / (float)blk.sec[c]);
            sincosf(p * inv, &sn_s[i], &cs_s[i]);
        }
    } else {
        const float p = qk_pos(pos, pos_f32, m) / div;
        for (uint32_t i = threadIdx.x; i < half; i += kQkThreads) {
            const float inv = 1.0f / powf(theta, (float)(2 * i) / (float)rot);
            sincosf(p * inv, &sn_s[i], &cs_s[i]);
        }
    }
    __syncthreads();

    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int E = (int)(head_dim / 32);
    auto at = [&](int j) -> uint32_t {
        return MODE == 1 ? 64u * (uint32_t)(j >> 1) + 2u * lane + (uint32_t)(j & 1)
                         : lane + 32u * (uint32_t)j;
    };
    for (uint32_t hh = wave; hh < n_norm; hh += kQkThreads / 32) {
        uint16_t* h = row + hh * head_dim;
        float x[kQkMaxE];
        float ss = 0.0f;
        #pragma unroll
        for (int j = 0; j < kQkMaxE; ++j)
            if (j < E) { x[j] = bf2f(h[at(j)]); ss += x[j] * x[j]; }
        ss = r4d_wave_sum(ss);
        const float sc = 1.0f / sqrtf(ss / (float)head_dim + eps);
        const bool is_v = hh >= n_rope;
        const void* w = hh < n_head ? qw : kw;
        #pragma unroll
        for (int j = 0; j < kQkMaxE; ++j)
            if (j < E) {
                const float g = is_v ? 1.0f : qk_gain(w, at(j), w_f32, wadd);
                x[j] = bf2f(f32_to_bf16(x[j] * sc * g));
            }
        if (!is_v) {
            if (MODE == 1) {
                #pragma unroll
                for (int j = 0; j < kQkMaxE; j += 2) {
                    const uint32_t i = at(j) / 2;
                    if (j < E && i < half) {
                        const float x0 = x[j], x1 = x[j + 1];
                        x[j]     = x0 * cs_s[i] - x1 * sn_s[i];
                        x[j + 1] = x0 * sn_s[i] + x1 * cs_s[i];
                    }
                }
            } else if (MODE == 0) {
                const int hj = (int)(half / 32);
                #pragma unroll
                for (int j = 0; j < kQkMaxE / 2; ++j)
                    if (j < hj) {
                        const uint32_t i = lane + 32u * (uint32_t)j;
                        const float x0 = x[j], x1 = x[j + hj];
                        x[j]      = x0 * cs_s[i] - x1 * sn_s[i];
                        x[j + hj] = x0 * sn_s[i] + x1 * cs_s[i];
                    }
            } else {
                // Each section's block starts on a 32-channel slice. A section of 16 pairs is
                // one slice, its partners in the other half-wave; a multiple of 32 pairs keeps
                // both halves of a pair in the lane. (The launcher refuses anything else.)
                int s0 = 0;
                for (int c = 0; c < blk.n; ++c) {
                    const int sec = blk.sec[c], j0 = s0 / 16;
                    if (sec == 16) {
                        #pragma unroll
                        for (int j = 0; j < kQkMaxE; ++j)
                            if (j == j0) {
                                const float mine = x[j], other = swap16(x[j]);
                                const uint32_t i = (uint32_t)s0 + (lane & 15u);
                                x[j] = lane < 16 ? mine * cs_s[i] - other * sn_s[i]
                                                 : other * sn_s[i] + mine * cs_s[i];
                            }
                    } else {
                        const int hj = sec / 32;
                        #pragma unroll
                        for (int j = 0; j < kQkMaxE; ++j)
                            if (j >= j0 && j < j0 + hj) {
                                const uint32_t i = (uint32_t)s0 + lane + 32u * (uint32_t)(j - j0);
                                const float x0 = x[j], x1 = x[j + hj];
                                x[j]      = x0 * cs_s[i] - x1 * sn_s[i];
                                x[j + hj] = x0 * sn_s[i] + x1 * cs_s[i];
                            }
                    }
                    s0 += sec;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < kQkMaxE; ++j)
            if (j < E) h[at(j)] = f32_to_bf16(x[j]);
    }
}

// Section modes at any head_dim and section size, which the register layout above cannot hold
// (e.g. Gemma 4's vision tower: 72-wide heads, two sections of 18 pairs). One wave per head, one
// lane per rotated pair: sum of squares first, then each lane re-reads its pair, norms, applies
// the gain, rounds (before rotation, as above) and rotates. Channels past `rot` are normed only;
// a v head (v_norm) is normed at unit gain.
//
// `pm` selects the pairing:
//   0  blocks       a section's pairs are (start + j, start + sec + j), its own ladder
//   1  gptj_blocks  a section's pairs are ADJACENT, (start + 2j, start + 2j + 1), its own ladder:
//                   the complex-number form of a 3-axis image rotary (Qwen-Image)
//   2  imrope       NeoX pairs (i, i + rot/2) on one ladder theta^(-2i/rot), the position
//                   component interleaved: 1 where i % 3 == 1 and i < 3 * sec[1], 2 where
//                   i % 3 == 2 and i < 3 * sec[2], 0 elsewhere (Qwen3-VL's text attention)
//   3  axial        NeoX pairs (i, i + rot/2), the rot/2 frequencies in contiguous sections,
//                   section c on its own ladder theta^(-2j/(2 sec[c])) and turned by component c
//                   (a video transformer's or decoder's (t, h, w) rotary). With rank-1 positions
//                   (`pplane` 0) every component is the one position.
//
// `rot` 0 is the norm alone and reads no positions.
__global__ __launch_bounds__(kQkThreads) void qk_norm_rope_any_kernel(
    uint16_t* __restrict__ qkv, const void* __restrict__ pos, const void* __restrict__ qw,
    const void* __restrict__ kw, uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
    uint32_t rot, int v_norm, float theta, float eps, float rscale, float wadd, uint32_t ld,
    QkBlocks blk, uint32_t pplane, int pm, int w_f32, int pos_f32) {
    __shared__ float cs_s[kQkMaxHalf], sn_s[kQkMaxHalf];
    __shared__ int   pa_s[kQkMaxHalf], pb_s[kQkMaxHalf];
    const uint32_t m = blockIdx.x;
    const uint32_t n_rope = n_head + n_head_kv;
    const uint32_t n_norm = n_rope + (v_norm ? n_head_kv : 0);
    const uint32_t half = rot / 2;
    uint16_t* row = qkv + (size_t)m * ld;
    const float div = rscale != 0.0f ? rscale : 1.0f;
    for (uint32_t i = threadIdx.x; i < half; i += kQkThreads) {
        if (pm == 2) {
            int c = 0;
            if (i % 3 == 1 && (int)i < 3 * blk.sec[1]) c = 1;
            else if (i % 3 == 2 && (int)i < 3 * blk.sec[2]) c = 2;
            const float p = qk_pos(pos, pos_f32, (size_t)c * pplane + m) / div;
            const float inv = 1.0f / powf(theta, (float)(2 * i) / (float)rot);
            sincosf(p * inv, &sn_s[i], &cs_s[i]);
            pa_s[i] = (int)i;
            pb_s[i] = (int)(i + half);
            continue;
        }
        int c = 0, s0 = 0;
        while (c + 1 < blk.n && (int)i >= s0 + blk.sec[c]) { s0 += blk.sec[c]; ++c; }
        if (pm == 3) {
            const float p = qk_pos(pos, pos_f32, (size_t)c * pplane + m) / div;
            const float inv =
                1.0f / powf(theta, (float)(2 * ((int)i - s0)) / (float)(2 * blk.sec[c]));
            sincosf(p * inv, &sn_s[i], &cs_s[i]);
            pa_s[i] = (int)i;
            pb_s[i] = (int)(i + half);
            continue;
        }
        const float p = qk_pos(pos, pos_f32, (size_t)blk.comp[c] * pplane + m) / div;
        const float inv = 1.0f / powf(theta, (float)((int)i - s0) / (float)blk.sec[c]);
        sincosf(p * inv, &sn_s[i], &cs_s[i]);
        if (pm == 1) {
            pa_s[i] = 2 * (int)i;
            pb_s[i] = 2 * (int)i + 1;
        } else {
            pa_s[i] = 2 * s0 + ((int)i - s0);
            pb_s[i] = pa_s[i] + blk.sec[c];
        }
    }
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (uint32_t hh = wave; hh < n_norm; hh += kQkThreads / 32) {
        uint16_t* h = row + hh * head_dim;
        float ss = 0.0f;
        for (uint32_t e = lane; e < head_dim; e += 32) { const float v = bf2f(h[e]); ss += v * v; }
        ss = r4d_wave_sum(ss);
        const float sc = 1.0f / sqrtf(ss / (float)head_dim + eps);
        const bool is_v = hh >= n_rope;
        const void* w = hh < n_head ? qw : kw;
        auto nrm = [&](uint32_t e) {
            const float g = is_v ? 1.0f : qk_gain(w, e, w_f32, wadd);
            return bf2f(f32_to_bf16(bf2f(h[e]) * sc * g));
        };
        if (is_v) {
            for (uint32_t e = lane; e < head_dim; e += 32) h[e] = f32_to_bf16(nrm(e));
            continue;
        }
        for (uint32_t i = lane; i < half; i += 32) {
            const uint32_t a = (uint32_t)pa_s[i], b = (uint32_t)pb_s[i];
            const float x0 = nrm(a), x1 = nrm(b);
            h[a] = f32_to_bf16(x0 * cs_s[i] - x1 * sn_s[i]);
            h[b] = f32_to_bf16(x0 * sn_s[i] + x1 * cs_s[i]);
        }
        for (uint32_t e = rot + lane; e < head_dim; e += 32) h[e] = f32_to_bf16(nrm(e));
    }
}

// ---------------------------------------------------------------------------- pool_rows
// ref_pool_rows: row r of y is the mean of x's rows [r*k, r*k + k), summed in f32 and rounded to
// bf16, then times `mult` and rounded again. One thread per column.
__global__ __launch_bounds__(kThreads) void pool_rows_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t k, float mult,
    uint32_t x_ld, uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t r = blockIdx.y; r < M; r += gridDim.y) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < k; ++j) acc += bf2f(x[((size_t)r * k + j) * x_ld + i]);
        const float mean = bf2f(f32_to_bf16(acc * (1.0f / (float)k)));
        y[(size_t)r * y_ld + i] = f32_to_bf16(mean * mult);
    }
}

// ---------------------------------------------------------------------------- pool_cols
// ref_pool_cols: column c of row m of y is the mean of x[m]'s columns [c*k, c*k + k), summed in
// f32 and rounded to bf16, then times `mult` and rounded again. A thread an output column.
__global__ __launch_bounds__(kThreads) void pool_cols_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t k, float mult,
    uint32_t x_ld, uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n / k) return;
    for (uint32_t r = blockIdx.y; r < M; r += gridDim.y) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < k; ++j) acc += bf2f(x[(size_t)r * x_ld + (size_t)i * k + j]);
        const float mean = bf2f(f32_to_bf16(acc * (1.0f / (float)k)));
        y[(size_t)r * y_ld + i] = f32_to_bf16(mean * mult);
    }
}

// ---------------------------------------------------------------------------- point_fourier
// ref_point_fourier: row m of y is [x | sin(x_i f_j) | cos(x_i f_j)], f_j = 2^j (times pi), the
// sines then the cosines each i-major then j; f32 in and out. A thread an output column.
__global__ __launch_bounds__(kThreads) void point_fourier_kernel(
    const float* __restrict__ x, float* __restrict__ y, uint32_t n, uint32_t F, uint32_t pi,
    uint32_t x_ld, uint32_t y_ld, uint32_t M) {
    const uint32_t w = n * (2 * F + 1);
    const uint32_t c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= w) return;
    for (uint32_t r = blockIdx.y; r < M; r += gridDim.y) {
        float v;
        if (c < n) {
            v = x[(size_t)r * x_ld + c];
        } else {
            const uint32_t k = c - n, half = n * F;
            const uint32_t i = (k % half) / F, j = (k % half) % F;
            const float f = pi ? (float)ldexp(3.14159265358979323846, (int)j)
                               : ldexpf(1.0f, (int)j);
            const float t = x[(size_t)r * x_ld + i] * f;
            v = k < half ? sinf(t) : cosf(t);
        }
        y[(size_t)r * y_ld + c] = v;
    }
}

// ---------------------------------------------------------------------------- cast bf16 -> f32
__global__ __launch_bounds__(kThreads) void widen_kernel(
    const uint16_t* __restrict__ x, float* __restrict__ y, uint32_t n, uint32_t x_ld,
    uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y)
        y[(size_t)m * y_ld + i] = bf2f(x[(size_t)m * x_ld + i]);
}

// ---------------------------------------------------------------------------- cast bf16 -> f16
// ref_cast: the value rounded to nearest even once, so a bf16 value in f16's normal range is kept
// exactly and one past it becomes an infinity (rad_f32_to_f16).
__global__ __launch_bounds__(kThreads) void narrow_f16_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld,
    uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y)
        y[(size_t)m * y_ld + i] =
            __builtin_bit_cast(uint16_t, (_Float16)bf2f(x[(size_t)m * x_ld + i]));
}

// ---------------------------------------------------------------------------- cast f32 -> bf16
// ref_cast: the value rounded to nearest even once.
__global__ __launch_bounds__(kThreads) void narrow_bf16_kernel(
    const float* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld,
    uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y)
        y[(size_t)m * y_ld + i] = f32_to_bf16(x[(size_t)m * x_ld + i]);
}

// ---------------------------------------------------------------------------- the audio tower
// Gemma 4's USM conformer: elementwise rectifier and clamp, the subsampler's segment-wise
// convolution, the light-conv block's causal depthwise convolution, the relative-position table,
// and local attention with a relative term. An audio pass is at most a few thousand frames
// (750 tokens for 30 s), so each kernel is the plain form (a workgroup per row, a thread per
// element) and matches its libref function: f32 throughout, one rounding.

// ref_relu.
__global__ __launch_bounds__(kThreads) void relu_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld,
    uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y) {
        const float v = bf2f(x[(size_t)m * x_ld + i]);
        y[(size_t)m * y_ld + i] = f32_to_bf16(v > 0.0f ? v : 0.0f);
    }
}

// ref_silu: x * sigmoid(x) with the sigmoid taken first, as libref's act_silu. Used standalone
// between two clipped linears in the audio feed-forward.
__global__ __launch_bounds__(kThreads) void silu_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld,
    uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y) {
        const float v = bf2f(x[(size_t)m * x_ld + i]);
        y[(size_t)m * y_ld + i] = f32_to_bf16(v * (1.0f / (1.0f + expf(-v))));
    }
}

// The f32 form: f32 in, f32 out or bf16 rounded once (a timestep MLP's activation, rounded where
// the bf16 projection after it reads it). Flat over the extent.
__device__ __forceinline__ void st_act(float* p, float v) { *p = v; }
__device__ __forceinline__ void st_act(uint16_t* p, float v) { *p = f32_to_bf16(v); }

template <typename TY>
__global__ __launch_bounds__(kThreads) void silu_f32_kernel(
    const float* __restrict__ x, TY* __restrict__ y, uint32_t n, uint32_t x_ld, uint32_t y_ld,
    uint64_t total) {
    for (uint64_t t = (uint64_t)blockIdx.x * kThreads + threadIdx.x; t < total;
         t += (uint64_t)gridDim.x * kThreads) {
        const uint64_t m = t / n;
        const uint32_t i = (uint32_t)(t - m * n);
        const float v = x[m * x_ld + i];
        st_act(y + m * y_ld + i, v * (1.0f / (1.0f + expf(-v))));
    }
}

// ref_gelu: the exact erf form, 0.5 x (1 + erf(x / sqrt 2)) in libref's order, one rounding (a
// ConvNeXt block's feed-forward activation; gated feed-forwards use the tanh form).
__global__ __launch_bounds__(kThreads) void gelu_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld,
    uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y) {
        const float v = bf2f(x[(size_t)m * x_ld + i]);
        y[(size_t)m * y_ld + i] = f32_to_bf16(0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
    }
}

// ref_clamp: the two bounds are one-element weights, bf16 or f32.
__global__ __launch_bounds__(kThreads) void clamp_kernel(
    const uint16_t* __restrict__ x, const void* __restrict__ lo, const void* __restrict__ hi,
    int b_f32, uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld, uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const float l = b_f32 ? ((const float*)lo)[0] : bf2f(((const uint16_t*)lo)[0]);
    const float h = b_f32 ? ((const float*)hi)[0] : bf2f(((const uint16_t*)hi)[0]);
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y) {
        float v = bf2f(x[(size_t)m * x_ld + i]);
        v = v < l ? l : v;
        v = v > h ? h : v;
        y[(size_t)m * y_ld + i] = f32_to_bf16(v);
    }
}

struct ConvGeom {
    int W, cin, cout, kh, kw, st, pad, cdiv, vst, Wo, S;
    int tl;     /* the weight is [cout, cin, kw, kh]: time last */
    int gelu;   /* the epilogue's activation */
};

// Kernel tap (i over time, j over width) of channel pair (co, ci), in either weight layout; the
// next input channel is kh * kw further on in both.
__device__ __forceinline__ size_t conv_w(const ConvGeom& g, int co, int ci, int i, int j) {
    return g.tl ? (((size_t)co * g.cin + ci) * g.kw + j) * g.kh + i
                : (((size_t)co * g.cin + ci) * g.kh + i) * g.kw + j;
}

// ref_conv2d's epilogue: the sum plus the bias, one rounding; gelu of the rounded value, a second
// rounding.
__device__ __forceinline__ uint16_t conv_out(const ConvGeom& g, float acc,
                                             const uint16_t* __restrict__ bias, int co) {
    if (bias) acc += bf2f(bias[co]);
    const uint16_t r = f32_to_bf16(acc);
    return g.gelu ? f32_to_bf16(act_of<1>(bf2f(r))) : r;
}

// ref_conv2d. One workgroup per output time row: it finds the covering segment (a scan over the
// few cu_seqlens entries), and threads compute the row's (width, channel) outputs. Rows no
// segment covers are left untouched, as in libref.
__global__ __launch_bounds__(kThreads) void conv2d_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const int* __restrict__ cu,
    const int* __restrict__ valid, const uint16_t* __restrict__ bias, uint16_t* __restrict__ y,
    ConvGeom g, uint32_t x_ld, uint32_t y_ld) {
    const int r = (int)blockIdx.x;
    int t0 = -1, T = 0, to = 0;
    for (int s = 0; s < g.S; ++s) {
        const int c0 = cu[s], c1 = cu[s + 1];
        if (c0 % g.cdiv || c1 % g.cdiv || (c0 / g.cdiv) % g.st || c1 < c0) continue;
        const int a = c0 / g.cdiv, Ts = c1 / g.cdiv - a;
        if (Ts <= 0) continue;
        const int To = (Ts + 2 * g.pad - g.kh) / g.st + 1, o0 = a / g.st;
        if (r >= o0 && r < o0 + To) { t0 = a; T = Ts; to = r - o0; break; }
    }
    if (t0 < 0) return;
    const int nel = g.Wo * g.cout;
    for (int e = threadIdx.x; e < nel; e += kThreads) {
        const int wo = e / g.cout, co = e % g.cout;
        float acc = 0.0f;
        for (int i = 0; i < g.kh; ++i) {
            const int ti = to * g.st - g.pad + i;
            if (ti < 0 || ti >= T) continue;
            const int tg = t0 + ti;
            if (valid && valid[(size_t)tg * g.vst] == 0) continue;
            const uint16_t* xr = x + (size_t)tg * x_ld;
            for (int j = 0; j < g.kw; ++j) {
                const int wi = wo * g.st - g.pad + j;
                if (wi < 0 || wi >= g.W) continue;
                const uint16_t* xv = xr + (size_t)wi * g.cin;
                const uint16_t* wv = w + conv_w(g, co, 0, i, j);
                for (int ci = 0; ci < g.cin; ++ci)
                    acc += bf2f(xv[ci]) * bf2f(wv[(size_t)ci * g.kh * g.kw]);
            }
        }
        y[(size_t)r * y_ld + e] = conv_out(g, acc, bias, co);
    }
}

// ref_conv2d_unfold. One workgroup per output time row, found as conv2d_kernel finds it, and one
// thread per (wo, k) element of its Wo unfolded rows: the tap value conv2d would multiply weight
// column k by, or zero in the padding and on a masked row. Columns are in the weight's order, so a
// row's writes are consecutive.
__global__ __launch_bounds__(kThreads) void conv2d_unfold_kernel(
    const uint16_t* __restrict__ x, const int* __restrict__ cu, const int* __restrict__ valid,
    uint16_t* __restrict__ y, ConvGeom g, uint32_t x_ld, uint32_t y_ld) {
    const int r = (int)blockIdx.x;
    int t0 = -1, T = 0, to = 0;
    for (int s = 0; s < g.S; ++s) {
        const int c0 = cu[s], c1 = cu[s + 1];
        if (c0 % g.cdiv || c1 % g.cdiv || (c0 / g.cdiv) % g.st || c1 < c0) continue;
        const int a = c0 / g.cdiv, Ts = c1 / g.cdiv - a;
        if (Ts <= 0) continue;
        const int To = (Ts + 2 * g.pad - g.kh) / g.st + 1, o0 = a / g.st;
        if (r >= o0 && r < o0 + To) { t0 = a; T = Ts; to = r - o0; break; }
    }
    if (t0 < 0) return;
    const int K = g.cin * g.kh * g.kw, nel = g.Wo * K;
    for (int e = threadIdx.x; e < nel; e += kThreads) {
        const int wo = e / K, k = e % K;
        int ci, i, j;
        if (g.tl) { i = k % g.kh; j = (k / g.kh) % g.kw; ci = k / (g.kh * g.kw); }
        else      { j = k % g.kw; i = (k / g.kw) % g.kh; ci = k / (g.kh * g.kw); }
        const int ti = to * g.st - g.pad + i, wi = wo * g.st - g.pad + j;
        uint16_t v = 0;
        if (ti >= 0 && ti < T && wi >= 0 && wi < g.W) {
            const int tg = t0 + ti;
            if (!valid || valid[(size_t)tg * g.vst] != 0)
                v = x[(size_t)tg * x_ld + (size_t)wi * g.cin + ci];
        }
        y[((size_t)r * g.Wo + wo) * y_ld + k] = v;
    }
}

// conv2d_kernel's arithmetic with each kernel row's input row and weights staged in LDS, for the
// audio subsampler's second convolution (64 x 128 channels in, 32 x 32 out per row), where the
// plain kernel re-reads every weight from global per output (6 ms on a 30 s clip). Weights are
// [j][ci][co] so a wave's lanes (consecutive output channels) read consecutive LDS words and each
// input element is a broadcast. A thread owns kConvEpt outputs strided by the workgroup, all of
// one output channel. Summation order (i, j, ci) matches the plain kernel.
constexpr int kConvEpt  = 4;
constexpr int kConvXs   = 8192;     /* W * cin, one input row */
constexpr int kConvWs   = 12288;    /* kw * cin * cout, one kernel row's weights */

__global__ __launch_bounds__(kThreads) void conv2d_lds_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const int* __restrict__ cu,
    const int* __restrict__ valid, const uint16_t* __restrict__ bias, uint16_t* __restrict__ y,
    ConvGeom g, uint32_t x_ld, uint32_t y_ld) {
    __shared__ uint16_t xs[kConvXs];
    __shared__ uint16_t ws[kConvWs];
    const int r = (int)blockIdx.x;
    int t0 = -1, T = 0, to = 0;
    for (int s = 0; s < g.S; ++s) {
        const int c0 = cu[s], c1 = cu[s + 1];
        if (c0 % g.cdiv || c1 % g.cdiv || (c0 / g.cdiv) % g.st || c1 < c0) continue;
        const int a = c0 / g.cdiv, Ts = c1 / g.cdiv - a;
        if (Ts <= 0) continue;
        const int To = (Ts + 2 * g.pad - g.kh) / g.st + 1, o0 = a / g.st;
        if (r >= o0 && r < o0 + To) { t0 = a; T = Ts; to = r - o0; break; }
    }
    if (t0 < 0) return;
    const int nel = g.Wo * g.cout, nx = g.W * g.cin, nw = g.kw * g.cin * g.cout;
    const int tid = (int)threadIdx.x;
    float acc[kConvEpt];
    #pragma unroll
    for (int q = 0; q < kConvEpt; ++q) acc[q] = 0.0f;
    for (int i = 0; i < g.kh; ++i) {
        const int ti = to * g.st - g.pad + i;
        if (ti < 0 || ti >= T) continue;
        const int tg = t0 + ti;
        if (valid && valid[(size_t)tg * g.vst] == 0) continue;
        __syncthreads();
        const uint16_t* xr = x + (size_t)tg * x_ld;
        for (int e = tid; e < nx; e += kThreads) xs[e] = xr[e];
        for (int e = tid; e < nw; e += kThreads) {
            const int co = e % g.cout, rest = e / g.cout, ci = rest % g.cin, j = rest / g.cin;
            ws[e] = w[conv_w(g, co, ci, i, j)];
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < kConvEpt; ++q) {
            const int e = tid + q * kThreads;
            if (e >= nel) break;
            const int wo = e / g.cout, co = e % g.cout;
            float a = acc[q];
            for (int j = 0; j < g.kw; ++j) {
                const int wi = wo * g.st - g.pad + j;
                if (wi < 0 || wi >= g.W) continue;
                const uint16_t* xv = xs + wi * g.cin;
                const uint16_t* wv = ws + (size_t)j * g.cin * g.cout + co;
                for (int ci = 0; ci < g.cin; ++ci) a += bf2f(xv[ci]) * bf2f(wv[(size_t)ci * g.cout]);
            }
            acc[q] = a;
        }
    }
    #pragma unroll
    for (int q = 0; q < kConvEpt; ++q) {
        const int e = tid + q * kThreads;
        if (e < nel) y[(size_t)r * y_ld + e] = conv_out(g, acc[q], bias, e % g.cout);
    }
}

// ref_conv1d_dw. One thread per (row, channel); the segment is found by a scan over cu_seqlens.
// `b`, when bound, is a bias per channel (bf16, or f32 with b_f32), added before the rounding.
__global__ __launch_bounds__(kThreads) void conv1d_dw_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const int* __restrict__ cu,
    uint16_t* __restrict__ y, int C, int K, int cdiv, int S, uint32_t x_ld, uint32_t y_ld,
    int rows, const void* __restrict__ b, int b_f32) {
    const int c = (int)(blockIdx.x * kThreads + threadIdx.x);
    if (c >= C) return;
    for (int t = (int)blockIdx.y; t < rows; t += (int)gridDim.y) {
        int t0 = -1;
        for (int s = 0; s < S; ++s) {
            const int c0 = cu[s], c1 = cu[s + 1];
            if (c0 % cdiv || c1 % cdiv || c1 < c0) continue;
            if (t >= c0 / cdiv && t < c1 / cdiv) { t0 = c0 / cdiv; break; }
        }
        if (t0 < 0) continue;
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
            const int tt = t - (K - 1) + j;
            if (tt < t0) continue;
            acc += bf2f(x[(size_t)tt * x_ld + c]) * bf2f(w[(size_t)c * K + j]);
        }
        if (b) acc += b_f32 ? ((const float*)b)[c] : bf2f(((const uint16_t*)b)[c]);
        y[(size_t)t * y_ld + c] = f32_to_bf16(acc);
    }
}

// ref_rel_sinusoid, the same f32 steps with the accurate libm forms.
__global__ __launch_bounds__(kThreads) void rel_sinusoid_kernel(
    uint16_t* __restrict__ y, uint32_t h, float inc, uint32_t y_ld) {
    const uint32_t d = blockIdx.x;
    for (uint32_t i = threadIdx.x; i < h; i += kThreads) {
        const float inv = expf((float)i * -inc);
        const float ang = (float)d * inv;
        y[(size_t)d * y_ld + i] = f32_to_bf16(sinf(ang));
        y[(size_t)d * y_ld + h + i] = f32_to_bf16(cosf(ang));
    }
}

// ref_seg_sinusoid: rel_sinusoid's table at position (r - segment start) mod period. One workgroup
// per row; its segment is found by a scan over the few cu_seqlens entries, and rows no segment
// covers are left untouched, as in libref.
__global__ __launch_bounds__(kThreads) void seg_sinusoid_kernel(
    const int* __restrict__ cu, uint16_t* __restrict__ y, uint32_t h, float inc, uint32_t y_ld,
    int S, int cdiv, int period) {
    const int r = (int)blockIdx.x;
    int p = -1;
    for (int s = 0; s < S; ++s) {
        const int r0 = cu[s] / cdiv, r1 = cu[s + 1] / cdiv;
        if (r >= r0 && r < r1) { p = (r - r0) % period; break; }
    }
    if (p < 0) return;
    for (uint32_t i = threadIdx.x; i < h; i += kThreads) {
        const float inv = expf((float)i * -inc);
        const float ang = (float)p * inv;
        y[(size_t)r * y_ld + i] = f32_to_bf16(sinf(ang));
        y[(size_t)r * y_ld + h + i] = f32_to_bf16(cosf(ang));
    }
}

// ref_attn_local_rel. One workgroup per (query row, head), one thread per head dimension. Each of
// the at most left + 1 logits is a workgroup sum, followed by the softmax and the weighted sum of v.
constexpr uint32_t kLrThreads = 128;
constexpr int      kLrMaxKeys = 64;

struct LocalRelGeom {
    int D, H, L, cdiv, S, pds_f32;
    float cap, qs, ks;
    uint32_t q_ld, k_ld, v_ld, r_ld, o_ld;
};

__global__ __launch_bounds__(kLrThreads) void attn_local_rel_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v, const uint16_t* __restrict__ rk, const void* __restrict__ pds,
    const int* __restrict__ cu, uint16_t* __restrict__ o, LocalRelGeom g) {
    __shared__ float red[kLrThreads / 32];
    __shared__ float lg[kLrMaxKeys];
    const int i = (int)blockIdx.x, h = (int)blockIdx.y;
    int t0 = -1;
    for (int s = 0; s < g.S; ++s) {
        const int c0 = cu[s], c1 = cu[s + 1];
        if (c0 % g.cdiv || c1 % g.cdiv || c1 < c0) continue;
        if (i >= c0 / g.cdiv && i < c1 / g.cdiv) { t0 = c0 / g.cdiv; break; }
    }
    if (t0 < 0) return;
    const int j0 = i - g.L > t0 ? i - g.L : t0, nk = i - j0 + 1;
    const int tid = (int)threadIdx.x;
    for (int jj = 0; jj < nk; ++jj) {
        const int j = j0 + jj;
        float part = 0.0f;
        for (int d = tid; d < g.D; d += kLrThreads) {
            const float x = g.pds_f32 ? ((const float*)pds)[d] : bf2f(((const uint16_t*)pds)[d]);
            float sp = x > 20.0f ? x : (x > 0.0f ? x + log1pf(expf(-x)) : log1pf(expf(x)));
            if (!g.pds_f32) sp = bf2f(f32_to_bf16(sp));
            const float qf = bf2f(q[(size_t)i * g.q_ld + h * g.D + d]) * g.qs * sp;
            part += qf * (bf2f(k[(size_t)j * g.k_ld + h * g.D + d]) * g.ks);
            part += qf * bf2f(rk[(size_t)(i - j) * g.r_ld + h * g.D + d]);
        }
        part = r4d_wave_sum(part);
        if ((tid & 31) == 0) red[tid >> 5] = part;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            for (uint32_t w = 0; w < kLrThreads / 32; ++w) s += red[w];
            lg[jj] = g.cap * tanhf(s / g.cap);
        }
        __syncthreads();
    }
    float mx = -INFINITY;
    for (int jj = 0; jj < nk; ++jj) mx = lg[jj] > mx ? lg[jj] : mx;
    float sum = 0.0f;
    for (int jj = 0; jj < nk; ++jj) sum += expf(lg[jj] - mx);
    for (int d = tid; d < g.D; d += kLrThreads) {
        float acc = 0.0f;
        for (int jj = 0; jj < nk; ++jj)
            acc += expf(lg[jj] - mx) / sum * bf2f(v[(size_t)(j0 + jj) * g.v_ld + h * g.D + d]);
        o[(size_t)i * g.o_ld + h * g.D + d] = f32_to_bf16(acc);
    }
}

}  // namespace

// ============================================================== launchers

// act: 0 silu, 1 gelu (erf), 2 gelu_tanh, 3 sigmoid, 4 quick_gelu (sigmoid(bf16(1.702*a))), 5 relu.
extern "C" int r4d_act_mul_bf16(long a, long b, long y, int M, int n, int act, int a_ld, int b_ld,
                                int y_ld, long stream) {
    if (!a || !b || !y) return -1;
    if (M <= 0 || n <= 0 || act < 0 || act > 5) return -2;
    const dim3 grid((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M));
#define R4D_AM(A)                                                                               \
    hipLaunchKernelGGL(act_mul_kernel<A>, grid, dim3(kThreads), 0, (hipStream_t)stream,         \
                       (const uint16_t*)a, (const uint16_t*)b, (uint16_t*)y, (uint32_t)n,      \
                       (uint32_t)a_ld, (uint32_t)b_ld, (uint32_t)y_ld, (uint32_t)M)
    if (act == 0)      R4D_AM(0);
    else if (act == 1) R4D_AM(1);
    else if (act == 2) R4D_AM(2);
    else if (act == 3) R4D_AM(3);
    else if (act == 4) R4D_AM(4);
    else               R4D_AM(5);
#undef R4D_AM
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_seg_mean_bf16(long x, long cu, long y, int S, int n, int x_ld, int y_ld,
                                 long stream) {
    if (!x || !cu || !y) return -1;
    if (S <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(seg_mean_kernel, dim3((uint32_t)((n + 31) / 32), (uint32_t)S),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x,
                       (const int*)cu, (uint16_t*)y, (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ---------------------------------------------------------------------------- seg_softmax_pool
// ref_seg_softmax_pool: row s of y is v's rows [cu[s], cu[s+1]) summed with the softmax of their
// scores as weights, the softmax in f32 with the maximum subtracted, the sum in f32, one rounding.
// A workgroup is 256 columns of one segment, and every thread forms the segment's softmax itself:
// a segment is one question's few options (clef's routing), so the repeat is cheaper than a
// reduction. An empty segment writes zeros.
__global__ __launch_bounds__(kThreads) void seg_softmax_pool_kernel(
    const uint16_t* __restrict__ sc, const uint16_t* __restrict__ v, const int* __restrict__ cu,
    uint16_t* __restrict__ y, uint32_t n, uint32_t v_ld, uint32_t y_ld) {
    const uint32_t s = blockIdx.y;
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const int b0 = cu[s], b1 = cu[s + 1];
    float m = -INFINITY;
    for (int r = b0; r < b1; ++r) m = fmaxf(m, bf2f(sc[r]));
    float z = 0.0f;
    for (int r = b0; r < b1; ++r) z += expf(bf2f(sc[r]) - m);
    float acc = 0.0f;
    for (int r = b0; r < b1; ++r)
        acc += expf(bf2f(sc[r]) - m) / z * bf2f(v[(size_t)r * v_ld + i]);
    y[(size_t)s * y_ld + i] = f32_to_bf16(b1 > b0 ? acc : 0.0f);
}

extern "C" int r4d_seg_softmax_pool_bf16(long sc, long v, long cu, long y, int S, int n, int v_ld,
                                         int y_ld, long stream) {
    if (!sc || !v || !cu || !y) return -1;
    if (S <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(seg_softmax_pool_kernel,
                       dim3((uint32_t)((n + kThreads - 1) / kThreads), (uint32_t)S),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)sc,
                       (const uint16_t*)v, (const int*)cu, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)v_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// mode: 0 neox, 1 gptj, 2 blocks, 3 gptj_blocks, 4 imrope, 5 axial (`blk` and the position plane
// stride `pplane` used only by the last four). A null gain is a unit gain; `rot` 0 is the norm
// alone, and then `pos` may be null. `pos_f32` reads f32 positions as the values they hold.
extern "C" int r4d_qk_norm_rope_bf16(long qkv, long pos, long qw, long kw, int M, int head_dim,
                                     int n_head, int n_head_kv, int rot, int mode, int v_norm,
                                     float theta, float eps, float rscale, float wadd, int ld,
                                     const QkBlocks* blk, int pplane, long stream, int w_f32,
                                     int pos_f32) {
    if (!qkv || (rot > 0 && !pos)) return -1;
    if (M <= 0 || head_dim <= 0 || n_head <= 0 || n_head_kv <= 0 || rot < 0 || rot > head_dim ||
        (rot & 1) || rot / 2 > (int)kQkMaxHalf || head_dim > 32 * kQkMaxE || mode < 0 || mode > 5)
        return -2;
    if (rot == 0) {
        /* The norm alone: no pair is rotated, so no position is read. */
        hipLaunchKernelGGL(qk_norm_rope_any_kernel, dim3((uint32_t)M), dim3(kQkThreads), 0,
                           (hipStream_t)stream, (uint16_t*)qkv, (const void*)nullptr,
                           (const void*)qw, (const void*)kw, (uint32_t)head_dim,
                           (uint32_t)n_head, (uint32_t)n_head_kv, 0u, v_norm, theta, eps, rscale,
                           wadd, (uint32_t)ld, QkBlocks{}, 0u, 0, w_f32, 0);
        return hipGetLastError() == hipSuccess ? 0 : -4;
    }
    if (mode >= 3) {
        /* gptj_blocks, imrope and axial: general kernel only (its pair table is built per mode). */
        if (!blk || blk->n <= 0 || blk->n > 8) return -2;
        int total = 0;
        for (int c = 0; c < blk->n; ++c) {
            if (blk->sec[c] <= 0) return -2;
            total += blk->sec[c];
        }
        if ((mode == 3 || mode == 5) && 2 * total != rot) return -2;
        if (mode == 4 && (blk->n < 3 || 2 * total != rot)) return -2;
        hipLaunchKernelGGL(qk_norm_rope_any_kernel, dim3((uint32_t)M), dim3(kQkThreads), 0,
                           (hipStream_t)stream, (uint16_t*)qkv, (const void*)pos,
                           (const void*)qw, (const void*)kw, (uint32_t)head_dim,
                           (uint32_t)n_head, (uint32_t)n_head_kv, (uint32_t)rot, v_norm, theta,
                           eps, rscale, wadd, (uint32_t)ld, *blk, (uint32_t)pplane,
                           mode == 3 ? 1 : mode == 4 ? 2 : 3, w_f32, pos_f32);
        return hipGetLastError() == hipSuccess ? 0 : -4;
    }
    if (mode != 2 && head_dim % 32) return -2;
    if (mode == 1 && head_dim % 64) return -2;
    if (mode == 0 && (rot / 2) % 32) return -2;
    QkBlocks b{};
    bool any = false;   /* blocks mode outside the register layout: use qk_norm_rope_any_kernel */
    if (mode == 2) {
        if (!blk || blk->n <= 0 || blk->n > 8) return -2;
        b = *blk;
        int total = 0;
        for (int c = 0; c < b.n; ++c) {
            if (b.sec[c] <= 0) return -2;
            if ((2 * total) % 32 || (b.sec[c] != 16 && b.sec[c] % 32)) any = true;
            total += b.sec[c];
        }
        if (2 * total != rot) return -2;
        if (head_dim % 32) any = true;
    }
    if (any) {
        hipLaunchKernelGGL(qk_norm_rope_any_kernel, dim3((uint32_t)M), dim3(kQkThreads), 0,
                           (hipStream_t)stream, (uint16_t*)qkv, (const void*)pos,
                           (const void*)qw, (const void*)kw, (uint32_t)head_dim,
                           (uint32_t)n_head, (uint32_t)n_head_kv, (uint32_t)rot, v_norm, theta,
                           eps, rscale, wadd, (uint32_t)ld, b, (uint32_t)pplane, 0, w_f32,
                           pos_f32);
        return hipGetLastError() == hipSuccess ? 0 : -4;
    }
#define R4D_QN(MD)                                                                              \
    hipLaunchKernelGGL(qk_norm_rope_kernel<MD>, dim3((uint32_t)M), dim3(kQkThreads), 0,         \
                       (hipStream_t)stream, (uint16_t*)qkv, (const void*)pos, (const void*)qw,  \
                       (const void*)kw, (uint32_t)head_dim, (uint32_t)n_head,                   \
                       (uint32_t)n_head_kv, (uint32_t)rot, v_norm, theta, eps, rscale, wadd,    \
                       (uint32_t)ld, b, (uint32_t)pplane, w_f32, pos_f32)
    if (mode == 1)      R4D_QN(1);
    else if (mode == 2) R4D_QN(2);
    else                R4D_QN(0);
#undef R4D_QN
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_pool_rows_bf16(long x, long y, int rows, int n, int k, float mult, int x_ld,
                                  int y_ld, long stream) {
    if (!x || !y) return -1;
    if (rows <= 0 || n <= 0 || k <= 0) return -2;
    hipLaunchKernelGGL(pool_rows_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads),
                                              grid_rows(rows)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)y,
                       (uint32_t)n, (uint32_t)k, mult, (uint32_t)x_ld, (uint32_t)y_ld,
                       (uint32_t)rows);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_pool_cols_bf16(long x, long y, int rows, int n, int k, float mult, int x_ld,
                                  int y_ld, long stream) {
    if (!x || !y) return -1;
    if (rows <= 0 || n <= 0 || k <= 0 || n % k) return -2;
    hipLaunchKernelGGL(pool_cols_kernel,
                       dim3((uint32_t)((n / k + kThreads - 1) / kThreads), grid_rows(rows)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)y,
                       (uint32_t)n, (uint32_t)k, mult, (uint32_t)x_ld, (uint32_t)y_ld,
                       (uint32_t)rows);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_point_fourier_f32(long x, long y, int rows, int n, int freqs, int pi,
                                     int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (rows <= 0 || n <= 0 || freqs <= 0 || freqs > 30) return -2;
    const uint32_t w = (uint32_t)(n * (2 * freqs + 1));
    hipLaunchKernelGGL(point_fourier_kernel,
                       dim3((w + kThreads - 1) / kThreads, grid_rows(rows)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const float*)x, (float*)y,
                       (uint32_t)n, (uint32_t)freqs, (uint32_t)(pi != 0), (uint32_t)x_ld,
                       (uint32_t)y_ld, (uint32_t)rows);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_widen_bf16_f32(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(widen_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (float*)y,
                       (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_narrow_bf16_f16(long x, long y, int M, int n, int x_ld, int y_ld,
                                   long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(narrow_f16_kernel,
                       dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)y,
                       (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_narrow_f32_bf16(long x, long y, int M, int n, int x_ld, int y_ld,
                                   long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(narrow_bf16_kernel,
                       dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const float*)x, (uint16_t*)y,
                       (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_relu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(relu_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)y,
                       (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_gelu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(gelu_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)y,
                       (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_silu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(silu_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)y,
                       (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_clamp_bf16(long x, long lo, long hi, int b_f32, long y, int M, int n, int x_ld,
                              int y_ld, long stream) {
    if (!x || !lo || !hi || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(clamp_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x,
                       (const void*)lo, (const void*)hi, b_f32, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* One workgroup per output row (`rows_out`); rows no segment covers return at once. */
extern "C" int r4d_conv2d_bf16(long x, long w, long cu, long valid, long bias, long y,
                               int rows_out, const ConvGeom* g, int x_ld, int y_ld, long stream) {
    if (!x || !w || !cu || !y || !g) return -1;
    if (rows_out <= 0 || g->Wo <= 0 || g->S <= 0) return -2;
    const bool lds = g->W * g->cin <= kConvXs && g->kw * g->cin * g->cout <= kConvWs &&
                     g->Wo * g->cout <= kConvEpt * (int)kThreads;
    if (lds)
        hipLaunchKernelGGL(conv2d_lds_kernel, dim3((uint32_t)rows_out), dim3(kThreads), 0,
                           (hipStream_t)stream, (const uint16_t*)x, (const uint16_t*)w,
                           (const int*)cu, (const int*)valid, (const uint16_t*)bias, (uint16_t*)y,
                           *g, (uint32_t)x_ld, (uint32_t)y_ld);
    else
        hipLaunchKernelGGL(conv2d_kernel, dim3((uint32_t)rows_out), dim3(kThreads), 0,
                           (hipStream_t)stream, (const uint16_t*)x, (const uint16_t*)w,
                           (const int*)cu, (const int*)valid, (const uint16_t*)bias, (uint16_t*)y,
                           *g, (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* One workgroup per output time row (`rows_out`, the operand's rows over Wo). */
extern "C" int r4d_conv2d_unfold_bf16(long x, long cu, long valid, long y, int rows_out,
                                      const ConvGeom* g, int x_ld, int y_ld, long stream) {
    if (!x || !cu || !y || !g) return -1;
    if (rows_out <= 0 || g->Wo <= 0 || g->S <= 0) return -2;
    hipLaunchKernelGGL(conv2d_unfold_kernel, dim3((uint32_t)rows_out), dim3(kThreads), 0,
                       (hipStream_t)stream, (const uint16_t*)x, (const int*)cu,
                       (const int*)valid, (uint16_t*)y, *g, (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_conv1d_dw_bf16(long x, long w, long cu, long y, int rows, int C, int K,
                                  int cdiv, int S, int x_ld, int y_ld, long stream, long b = 0,
                                  int b_f32 = 0) {
    if (!x || !w || !cu || !y) return -1;
    if (rows <= 0 || C <= 0 || K <= 0 || cdiv <= 0 || S <= 0) return -2;
    hipLaunchKernelGGL(conv1d_dw_kernel,
                       dim3((uint32_t)((C + kThreads - 1) / kThreads), grid_rows(rows)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x,
                       (const uint16_t*)w, (const int*)cu, (uint16_t*)y, C, K, cdiv, S,
                       (uint32_t)x_ld, (uint32_t)y_ld, rows, (const void*)b, b_f32);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_rel_sinusoid_bf16(long y, int rows, int n, double max_timescale, int y_ld,
                                     long stream) {
    if (!y) return -1;
    if (rows <= 0 || n <= 0 || (n & 1)) return -2;
    const int h = n / 2;
    const float inc = (float)(std::log(max_timescale / 1.0) / (double)(h - 1 > 1 ? h - 1 : 1));
    hipLaunchKernelGGL(rel_sinusoid_kernel, dim3((uint32_t)rows), dim3(kThreads), 0,
                       (hipStream_t)stream, (uint16_t*)y, (uint32_t)h, inc, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_seg_sinusoid_bf16(long cu, int S, long y, int rows, int n, int period, int cdiv,
                                     double max_timescale, int y_ld, long stream) {
    if (!cu || !y) return -1;
    if (rows <= 0 || n <= 0 || (n & 1) || period <= 0 || cdiv <= 0 || S <= 0) return -2;
    const int h = n / 2;
    const float inc = (float)(std::log(max_timescale / 1.0) / (double)(h - 1 > 1 ? h - 1 : 1));
    hipLaunchKernelGGL(seg_sinusoid_kernel, dim3((uint32_t)rows), dim3(kThreads), 0,
                       (hipStream_t)stream, (const int*)cu, (uint16_t*)y, (uint32_t)h, inc,
                       (uint32_t)y_ld, S, cdiv, period);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_attn_local_rel_bf16(long q, long k, long v, long rk, long pds, long cu, long o,
                                       int rows, const LocalRelGeom* g, long stream) {
    if (!q || !k || !v || !rk || !pds || !cu || !o || !g) return -1;
    if (rows <= 0 || g->D <= 0 || g->H <= 0 || g->L < 0 || g->L + 1 > kLrMaxKeys) return -2;
    hipLaunchKernelGGL(attn_local_rel_kernel, dim3((uint32_t)rows, (uint32_t)g->H),
                       dim3(kLrThreads), 0, (hipStream_t)stream, (const uint16_t*)q,
                       (const uint16_t*)k, (const uint16_t*)v, (const uint16_t*)rk,
                       (const void*)pds, (const int*)cu, (uint16_t*)o, *g);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry points

static inline int dt_bf16(const RadTensor* t) {
    return (t && t->dtype == RAD_BF16) ? RAD_OK : RAD_E_DTYPE;
}
#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)

/* The entry points above return 0, -4 for a failed launch, or another negative code for a shape
 * they do not serve. */
static inline int enc_status(int rc) {
    if (rc == 0) return RAD_OK;
    return R4D_NO(rc == -4 ? RAD_E_DEVICE : RAD_E_SHAPE);
}

/* ------------------------------------------------------------------ act_mul */
extern "C" int r4d_rad_act_mul(const RadArgs* a, RadStream s) {
    const RadTensor* A = r4d_opt(a, 0);
    const RadTensor* B = r4d_opt(a, 1);
    const RadTensor* Y = r4d_opt(a, 2);
    if (!A || !B || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(A));
    R4D_NEED(dt_bf16(B));
    R4D_NEED(dt_bf16(Y));
    const char* act = r4d_gets(a, "act");
    int ai = -1;
    if (act && !std::strcmp(act, "silu")) ai = 0;
    else if (act && !std::strcmp(act, "gelu")) ai = 1;
    else if (act && !std::strcmp(act, "gelu_tanh")) ai = 2;
    else if (act && !std::strcmp(act, "sigmoid")) ai = 3;
    else if (act && !std::strcmp(act, "quick_gelu")) ai = 4;
    else if (act && !std::strcmp(act, "relu")) ai = 5;
    if (ai < 0) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0, Mb = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long a_ld = r4d_rows_pitch(A, n, &Ma);
    const long long b_ld = r4d_rows_pitch(B, n, &Mb);
    if (y_ld < 0 || a_ld < 0 || b_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M || Mb < M) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_act_mul_bf16(r4d_addr(A), r4d_addr(B), r4d_addr(Y), (int)M, (int)n, ai,
                                       (int)a_ld, (int)b_ld, (int)y_ld, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ seg_mean */
extern "C" int r4d_rad_seg_mean(const RadArgs* a, RadStream s) {
    const RadTensor* X  = r4d_opt(a, 0);
    const RadTensor* cu = r4d_opt(a, 1);
    const RadTensor* Y  = r4d_opt(a, 2);
    if (!X || !cu || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long S = r4d_numel(cu) - 1;
    long long Mx = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    const long long y_ld = r4d_rows_pitch(Y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My < S) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_seg_mean_bf16(r4d_addr(X), r4d_addr(cu), r4d_addr(Y), (int)S, (int)n,
                                        (int)x_ld, (int)y_ld, (long)(uintptr_t)s));
}

extern "C" int r4d_rad_seg_softmax_pool(const RadArgs* a, RadStream s) {
    const RadTensor* Sc = r4d_opt(a, 0);
    const RadTensor* V  = r4d_opt(a, 1);
    const RadTensor* cu = r4d_opt(a, 2);
    const RadTensor* Y  = r4d_opt(a, 3);
    if (!Sc || !V || !cu || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(Sc));
    R4D_NEED(dt_bf16(V));
    R4D_NEED(dt_bf16(Y));
    if (!r4d_contig(Sc)) return R4D_NO(RAD_E_STRIDE);
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long S = r4d_numel(cu) - 1;
    long long Mv = 0, My = 0;
    const long long v_ld = r4d_rows_pitch(V, n, &Mv);
    const long long y_ld = r4d_rows_pitch(Y, n, &My);
    if (v_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My < S || r4d_numel(Sc) < Mv) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_seg_softmax_pool_bf16(r4d_addr(Sc), r4d_addr(V), r4d_addr(cu),
                                                r4d_addr(Y), (int)S, (int)n, (int)v_ld,
                                                (int)y_ld, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ qk_norm_rope */
enum { QN_QKV = 0, QN_POS, QN_QW, QN_KW };

static int parse_ints(const char* v, int* out, int max) {
    int n = 0;
    while (v && *v && n < max) {
        char* end = nullptr;
        const long x = std::strtol(v, &end, 10);
        if (end == v) break;
        out[n++] = (int)x;
        v = end;
    }
    return n;
}

extern "C" int r4d_rad_qk_norm_rope(const RadArgs* a, RadStream s) {
    const RadTensor* qkv = r4d_opt(a, QN_QKV);
    const RadTensor* pos = r4d_opt(a, QN_POS);
    const RadTensor* qw  = r4d_opt(a, QN_QW);
    const RadTensor* kw  = r4d_opt(a, QN_KW);
    if (!qkv) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(qkv));
    /* The gains bf16 or f32, both alike (a model may keep its norms f32, libref reads either); an
     * absent gain is a unit gain. */
    const RadTensor* w0 = qw ? qw : kw;
    if (w0 && ((w0->dtype != RAD_BF16 && w0->dtype != RAD_F32) ||
               (qw && kw && kw->dtype != qw->dtype)))
        return R4D_NO(RAD_E_DTYPE);
    if ((qw && !r4d_contig(qw)) || (kw && !r4d_contig(kw))) return R4D_NO(RAD_E_STRIDE);
    const long long hd  = r4d_geti_or(a, "head_dim", 0);
    const long long nh  = r4d_geti_or(a, "n_head", 0);
    const long long nkv = r4d_geti_or(a, "n_head_kv", 0);
    const long long rot = r4d_geti_or(a, "rotary_dim", hd);
    if (hd <= 0 || nh <= 0 || nkv <= 0 || (qw && r4d_numel(qw) < hd) ||
        (kw && r4d_numel(kw) < hd))
        return R4D_NO(RAD_E_SHAPE);
    /* The register layout holds head_dim up to 1024 (32 lanes x kQkMaxE). */
    if (hd > 1024 || rot < 0 || rot > hd || (rot & 1)) return R4D_NO(RAD_E_SHAPE);
    /* A rotation needs positions; the norm alone (rotary_dim 0) may go without. Positions are
     * i32, or f32 read as the values they hold, M-contiguous ([M], or [C, M] at any plane
     * pitch). */
    if (rot > 0 && !pos) return R4D_NO(RAD_E_INVAL);
    if (pos && pos->dtype != RAD_I32 && pos->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (pos && (pos->rank == 2 ? pos->stride[1] != 1 : !r4d_contig(pos)))
        return R4D_NO(RAD_E_STRIDE);
    const int pos_f32 = pos && pos->dtype == RAD_F32;
    const char* mode = r4d_gets(a, "mode");
    int md = 0;
    if (mode && !std::strcmp(mode, "gptj")) md = 1;
    else if (mode && !std::strcmp(mode, "blocks")) md = 2;
    else if (mode && !std::strcmp(mode, "gptj_blocks")) md = 3;
    else if (mode && !std::strcmp(mode, "imrope")) md = 4;
    else if (mode && !std::strcmp(mode, "axial")) md = 5;
    else if (mode && std::strcmp(mode, "neox")) return R4D_NO(RAD_E_UNSUPPORTED);
    /* Any head_dim in the section modes (qk_norm_rope_any_kernel); the others need multiples of 32. */
    if (rot > 0 && md < 2 && hd % 32) return R4D_NO(RAD_E_SHAPE);
    QkBlocks blk{};
    long long M = 0, pplane = 0;
    if (rot == 0) {
        /* Nothing turns: the rows are the positions' when bound, qkv's otherwise. */
        M = !pos ? 0 : pos->rank == 2 ? pos->shape[1] : r4d_numel(pos);
    } else if (md == 5) {
        /* Contiguous sections on their own ladders, one position component each; rank-1
         * positions turn every section by the one position. */
        blk.n = parse_ints(r4d_gets(a, "sections"), blk.sec, 8);
        if (blk.n <= 0) return R4D_NO(RAD_E_SHAPE);
        if (pos->rank == 2) {
            if (pos->shape[0] < blk.n) return R4D_NO(RAD_E_SHAPE);
            M = pos->shape[1];
            pplane = pos->stride[0];
        } else {
            M = r4d_numel(pos);
        }
    } else if (md == 4) {
        /* Interleaved M-RoPE reads three position planes; its sections only bound the
         * interleave. */
        blk.n = parse_ints(r4d_gets(a, "sections"), blk.sec, 8);
        if (blk.n < 3 || pos->rank != 2 || pos->shape[0] < 3) return R4D_NO(RAD_E_SHAPE);
        M = pos->shape[1];
        pplane = pos->stride[0];
    } else if (md >= 2) {
        blk.n = parse_ints(r4d_gets(a, "sections"), blk.sec, 8);
        if (blk.n <= 0) return R4D_NO(RAD_E_SHAPE);
        for (int c = 0; c < blk.n; ++c) blk.comp[c] = c;
        if (const char* cs = r4d_gets(a, "comps"))
            if (parse_ints(cs, blk.comp, 8) != blk.n) return R4D_NO(RAD_E_SHAPE);
        if (pos->rank != 2) return R4D_NO(RAD_E_SHAPE);
        for (int c = 0; c < blk.n; ++c)
            if (blk.comp[c] < 0 || blk.comp[c] >= pos->shape[0]) return R4D_NO(RAD_E_SHAPE);
        M = pos->shape[1];
        pplane = pos->stride[0];
    } else {
        /* neox and gptj turn by one position a row. */
        if (pos->rank >= 2 && pos->shape[0] != 1 && pos->shape[pos->rank - 1] != 1)
            return R4D_NO(RAD_E_SHAPE);
        M = r4d_numel(pos);
    }
    const long long width = (nh + 2 * nkv) * hd;
    long long Mq = 0;
    const long long ld = r4d_rows_pitch(qkv, width, &Mq);
    if (ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (rot == 0 && !pos) M = Mq;
    if (M <= 0 || Mq < M) return R4D_NO(RAD_E_SHAPE);
    const int e = r4d_qk_norm_rope_bf16(r4d_addr(qkv), r4d_addr(pos), r4d_addr(qw), r4d_addr(kw),
                                        (int)M, (int)hd, (int)nh, (int)nkv, (int)rot, md,
                                        (int)r4d_geti_or(a, "v_norm", 0),
                                        (float)r4d_getf(a, "theta", 10000.0),
                                        (float)r4d_getf(a, "eps", 1e-6),
                                        (float)r4d_getf(a, "scale", 1.0),
                                        (float)r4d_getf(a, "wadd", 0.0), (int)ld, &blk,
                                        (int)pplane, (long)(uintptr_t)s,
                                        w0 && w0->dtype == RAD_F32 ? 1 : 0, pos_f32);
    if (e == -2) return R4D_NO(RAD_E_SHAPE);
    return enc_status(e);
}

/* ------------------------------------------------------------------ pool_rows */
extern "C" int r4d_rad_pool_rows(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* Y = r4d_opt(a, 1);
    if (!X || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    const long long n = r4d_geti_or(a, "n", 0), k = r4d_geti_or(a, "k", 0);
    if (n <= 0 || k <= 0) return R4D_NO(RAD_E_SHAPE);
    long long Mx = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    const long long y_ld = r4d_rows_pitch(Y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mx <= 0 || Mx % k || My < Mx / k) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_pool_rows_bf16(r4d_addr(X), r4d_addr(Y), (int)(Mx / k), (int)n, (int)k,
                                         (float)r4d_getf(a, "mult", 1.0), (int)x_ld, (int)y_ld,
                                         (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ pool_cols */
extern "C" int r4d_rad_pool_cols(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* Y = r4d_opt(a, 1);
    if (!X || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    const long long n = r4d_geti_or(a, "n", 0), k = r4d_geti_or(a, "k", 0);
    if (n <= 0 || k <= 0 || n % k) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, n, &M);
    const long long y_ld = r4d_rows_pitch(Y, n / k, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_pool_cols_bf16(r4d_addr(X), r4d_addr(Y), (int)M, (int)n, (int)k,
                                         (float)r4d_getf(a, "mult", 1.0), (int)x_ld, (int)y_ld,
                                         (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ point_fourier */
extern "C" int r4d_rad_point_fourier(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* Y = r4d_opt(a, 1);
    if (!X || !Y) return R4D_NO(RAD_E_INVAL);
    if (X->dtype != RAD_F32 || Y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), F = r4d_geti_or(a, "freqs", 0);
    if (n <= 0 || F <= 0 || F > 30) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, n, &M);
    const long long y_ld = r4d_rows_pitch(Y, n * (2 * F + 1), &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_point_fourier_f32(r4d_addr(X), r4d_addr(Y), (int)M, (int)n, (int)F,
                                           (int)r4d_geti_or(a, "pi", 0), (int)x_ld, (int)y_ld,
                                           (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ cast bf16 -> f32 */
extern "C" int r4d_rad_cast_widen(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    if (y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_widen_bf16_f32(r4d_addr(x), r4d_addr(y), (int)M, (int)n, (int)x_ld,
                                         (int)y_ld, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ cast bf16 -> f16 */
extern "C" int r4d_rad_cast_f16(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    if (y->dtype != RAD_F16) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_narrow_bf16_f16(r4d_addr(x), r4d_addr(y), (int)M, (int)n, (int)x_ld,
                                          (int)y_ld, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ cast f32 -> bf16 */
extern "C" int r4d_rad_cast_bf16(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    R4D_NEED(dt_bf16(y));
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_narrow_f32_bf16(r4d_addr(x), r4d_addr(y), (int)M, (int)n, (int)x_ld,
                                          (int)y_ld, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ the audio tower */

extern "C" int r4d_rad_relu(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* Y = r4d_opt(a, 1);
    if (!X || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_relu_bf16(r4d_addr(X), r4d_addr(Y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                                    (long)(uintptr_t)s));
}

/* Both rows: silu_bf16 (bf16 in and out) and silu_f32 (f32 in, f32 or bf16 out). */
extern "C" int r4d_rad_silu(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* Y = r4d_opt(a, 1);
    if (!X || !Y) return R4D_NO(RAD_E_INVAL);
    const bool x_f32 = X->dtype == RAD_F32;
    if (x_f32 ? (Y->dtype != RAD_F32 && Y->dtype != RAD_BF16)
              : (X->dtype != RAD_BF16 || Y->dtype != RAD_BF16))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    if (x_f32) {
        const uint64_t total = (uint64_t)M * (uint64_t)n;
        const dim3 grid((uint32_t)std::min<uint64_t>((total + kThreads - 1) / kThreads, 65536u));
        if (Y->dtype == RAD_F32)
            hipLaunchKernelGGL(silu_f32_kernel<float>, grid, dim3(kThreads), 0, (hipStream_t)s,
                               (const float*)X->data, (float*)Y->data, (uint32_t)n,
                               (uint32_t)x_ld, (uint32_t)y_ld, total);
        else
            hipLaunchKernelGGL(silu_f32_kernel<uint16_t>, grid, dim3(kThreads), 0, (hipStream_t)s,
                               (const float*)X->data, (uint16_t*)Y->data, (uint32_t)n,
                               (uint32_t)x_ld, (uint32_t)y_ld, total);
        return enc_status(hipGetLastError() == hipSuccess ? 0 : -4);
    }
    return enc_status(r4d_silu_bf16(r4d_addr(X), r4d_addr(Y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                                    (long)(uintptr_t)s));
}

extern "C" int r4d_rad_gelu(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* Y = r4d_opt(a, 1);
    if (!X || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_gelu_bf16(r4d_addr(X), r4d_addr(Y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                                    (long)(uintptr_t)s));
}

extern "C" int r4d_rad_clamp(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* L = r4d_opt(a, 1);
    const RadTensor* H = r4d_opt(a, 2);
    const RadTensor* Y = r4d_opt(a, 3);
    if (!X || !L || !H || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    if (L->dtype != H->dtype || (L->dtype != RAD_BF16 && L->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(L) < 1 || r4d_numel(H) < 1) return R4D_NO(RAD_E_SHAPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_clamp_bf16(r4d_addr(X), r4d_addr(L), r4d_addr(H), L->dtype == RAD_F32,
                                     r4d_addr(Y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                                     (long)(uintptr_t)s));
}

extern "C" int r4d_rad_conv2d(const RadArgs* a, RadStream s) {
    const RadTensor* X  = r4d_opt(a, 0);
    const RadTensor* Wt = r4d_opt(a, 1);
    const RadTensor* cu = r4d_opt(a, 2);
    const RadTensor* V  = r4d_opt(a, 3);
    const RadTensor* Y  = r4d_opt(a, 4);
    const RadTensor* B  = r4d_opt(a, 5);
    if (!X || !Wt || !cu || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Wt));
    R4D_NEED(dt_bf16(Y));
    if (B && (dt_bf16(B) != RAD_OK || !r4d_contig(B))) return R4D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    if (V && (V->dtype != RAD_I32 || !r4d_contig(V))) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(Wt)) return R4D_NO(RAD_E_STRIDE);
    ConvGeom g{};
    g.W = (int)r4d_geti_or(a, "width", 0);
    g.cin = (int)r4d_geti_or(a, "cin", 0);
    g.cout = (int)r4d_geti_or(a, "cout", 0);
    g.kh = (int)r4d_geti_or(a, "kh", 0);
    g.kw = (int)r4d_geti_or(a, "kw", 0);
    g.st = (int)r4d_geti_or(a, "stride", 1);
    g.pad = (int)r4d_geti_or(a, "pad", 0);
    g.cdiv = (int)r4d_geti_or(a, "cu_div", 1);
    g.vst = (int)r4d_geti_or(a, "valid_stride", 1);
    g.tl = r4d_geti_or(a, "t_last", 0) != 0 ? 1 : 0;
    const char* act = r4d_gets(a, "act");
    if (!act) act = "none";
    g.gelu = !std::strcmp(act, "gelu") ? 1 : 0;
    if (!g.gelu && std::strcmp(act, "none")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (g.W <= 0 || g.cin <= 0 || g.cout <= 0 || g.kh <= 0 || g.kw <= 0 || g.st <= 0 || g.pad < 0 ||
        g.cdiv <= 0 || g.vst <= 0)
        return R4D_NO(RAD_E_SHAPE);
    g.Wo = (g.W + 2 * g.pad - g.kw) / g.st + 1;
    g.S = (int)(r4d_numel(cu) - 1);
    if (g.Wo <= 0 || r4d_numel(Wt) < (long long)g.cout * g.cin * g.kh * g.kw)
        return R4D_NO(RAD_E_SHAPE);
    if (B && r4d_numel(B) < g.cout) return R4D_NO(RAD_E_SHAPE);
    long long Mx = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, (long long)g.W * g.cin, &Mx);
    const long long y_ld = r4d_rows_pitch(Y, (long long)g.Wo * g.cout, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mx <= 0 || My <= 0) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_conv2d_bf16(r4d_addr(X), r4d_addr(Wt), r4d_addr(cu), V ? r4d_addr(V) : 0,
                                      B ? r4d_addr(B) : 0, r4d_addr(Y), (int)My, &g, (int)x_ld,
                                      (int)y_ld, (long)(uintptr_t)s));
}

extern "C" int r4d_rad_conv2d_unfold(const RadArgs* a, RadStream s) {
    const RadTensor* X  = r4d_opt(a, 0);
    const RadTensor* cu = r4d_opt(a, 1);
    const RadTensor* V  = r4d_opt(a, 2);
    const RadTensor* Y  = r4d_opt(a, 3);
    if (!X || !cu || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    if (V && (V->dtype != RAD_I32 || !r4d_contig(V))) return R4D_NO(RAD_E_DTYPE);
    ConvGeom g{};
    g.W = (int)r4d_geti_or(a, "width", 0);
    g.cin = (int)r4d_geti_or(a, "cin", 0);
    g.cout = 1;
    g.kh = (int)r4d_geti_or(a, "kh", 0);
    g.kw = (int)r4d_geti_or(a, "kw", 0);
    g.st = (int)r4d_geti_or(a, "stride", 1);
    g.pad = (int)r4d_geti_or(a, "pad", 0);
    g.cdiv = (int)r4d_geti_or(a, "cu_div", 1);
    g.vst = (int)r4d_geti_or(a, "valid_stride", 1);
    g.tl = r4d_geti_or(a, "t_last", 0) != 0 ? 1 : 0;
    if (g.W <= 0 || g.cin <= 0 || g.kh <= 0 || g.kw <= 0 || g.st <= 0 || g.pad < 0 ||
        g.cdiv <= 0 || g.vst <= 0)
        return R4D_NO(RAD_E_SHAPE);
    g.Wo = (g.W + 2 * g.pad - g.kw) / g.st + 1;
    g.S = (int)(r4d_numel(cu) - 1);
    if (g.Wo <= 0) return R4D_NO(RAD_E_SHAPE);
    long long Mx = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, (long long)g.W * g.cin, &Mx);
    const long long y_ld = r4d_rows_pitch(Y, (long long)g.cin * g.kh * g.kw, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mx <= 0 || My <= 0 || My % g.Wo) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_conv2d_unfold_bf16(r4d_addr(X), r4d_addr(cu), V ? r4d_addr(V) : 0,
                                             r4d_addr(Y), (int)(My / g.Wo), &g, (int)x_ld,
                                             (int)y_ld, (long)(uintptr_t)s));
}

extern "C" int r4d_rad_conv1d_dw(const RadArgs* a, RadStream s) {
    const RadTensor* X  = r4d_opt(a, 0);
    const RadTensor* Wt = r4d_opt(a, 1);
    const RadTensor* cu = r4d_opt(a, 2);
    const RadTensor* Y  = r4d_opt(a, 3);
    const RadTensor* B  = r4d_opt(a, 4);
    if (!X || !Wt || !cu || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Wt));
    R4D_NEED(dt_bf16(Y));
    if (B && B->dtype != RAD_BF16 && B->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(Wt) || (B && !r4d_contig(B))) return R4D_NO(RAD_E_STRIDE);
    const long long C = r4d_geti_or(a, "n", 0), K = r4d_geti_or(a, "k", 0);
    const long long cdiv = r4d_geti_or(a, "cu_div", 1);
    if (C <= 0 || K <= 0 || cdiv <= 0 || r4d_numel(Wt) < C * K || (B && r4d_numel(B) < C))
        return R4D_NO(RAD_E_SHAPE);
    long long Mx = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, C, &Mx);
    const long long y_ld = r4d_rows_pitch(Y, C, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My <= 0 || Mx < My) return R4D_NO(RAD_E_SHAPE);
    return enc_status(r4d_conv1d_dw_bf16(r4d_addr(X), r4d_addr(Wt), r4d_addr(cu), r4d_addr(Y),
                                         (int)My, (int)C, (int)K, (int)cdiv,
                                         (int)(r4d_numel(cu) - 1), (int)x_ld, (int)y_ld,
                                         (long)(uintptr_t)s, B ? r4d_addr(B) : 0,
                                         B && B->dtype == RAD_F32 ? 1 : 0));
}

extern "C" int r4d_rad_rel_sinusoid(const RadArgs* a, RadStream s) {
    const RadTensor* Y = r4d_opt(a, 0);
    if (!Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(Y));
    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0 || (n & 1)) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    if (y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    /* Round through float, as libref reads the parameter, so both agree. */
    const double mt = (double)(float)r4d_getf(a, "max_timescale", 10000.0);
    return enc_status(r4d_rel_sinusoid_bf16(r4d_addr(Y), (int)M, (int)n, mt, (int)y_ld,
                                            (long)(uintptr_t)s));
}

extern "C" int r4d_rad_seg_sinusoid(const RadArgs* a, RadStream s) {
    const RadTensor* cu = r4d_opt(a, 0);
    const RadTensor* Y  = r4d_opt(a, 1);
    if (!cu || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(Y));
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    const long long n = r4d_geti_or(a, "n", 0), period = r4d_geti_or(a, "period", 0);
    const long long cdiv = r4d_geti_or(a, "cu_div", 1);
    if (n <= 0 || (n & 1) || period <= 0 || cdiv <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    if (y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const double mt = (double)(float)r4d_getf(a, "max_timescale", 10000.0);
    return enc_status(r4d_seg_sinusoid_bf16(r4d_addr(cu), (int)(r4d_numel(cu) - 1), r4d_addr(Y),
                                            (int)M, (int)n, (int)period, (int)cdiv, mt, (int)y_ld,
                                            (long)(uintptr_t)s));
}

extern "C" int r4d_rad_attn_local_rel(const RadArgs* a, RadStream s) {
    const RadTensor* q   = r4d_opt(a, 0);
    const RadTensor* k   = r4d_opt(a, 1);
    const RadTensor* v   = r4d_opt(a, 2);
    const RadTensor* rk  = r4d_opt(a, 3);
    const RadTensor* pds = r4d_opt(a, 4);
    const RadTensor* cu  = r4d_opt(a, 5);
    const RadTensor* o   = r4d_opt(a, 6);
    if (!q || !k || !v || !rk || !pds || !cu || !o) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(q));
    R4D_NEED(dt_bf16(k));
    R4D_NEED(dt_bf16(v));
    R4D_NEED(dt_bf16(rk));
    R4D_NEED(dt_bf16(o));
    if (pds->dtype != RAD_BF16 && pds->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(pds)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_geti_or(a, "right", 0) != 0) return R4D_NO(RAD_E_UNSUPPORTED);
    LocalRelGeom g{};
    g.D = (int)r4d_geti_or(a, "head_dim", 0);
    g.H = (int)r4d_geti_or(a, "n_head", 0);
    g.L = (int)r4d_geti_or(a, "left", 0);
    g.cdiv = (int)r4d_geti_or(a, "cu_div", 1);
    g.cap = (float)r4d_getf(a, "softcap", 0.0);
    if (g.D <= 0 || g.H <= 0 || g.L < 0 || g.L + 1 > kLrMaxKeys || g.cdiv <= 0 || !(g.cap > 0.0f))
        return R4D_NO(RAD_E_SHAPE);
    const long long Wd = (long long)g.H * g.D;
    if (r4d_numel(pds) < g.D || r4d_numel(rk) < (long long)(g.L + 1) * Wd) return R4D_NO(RAD_E_SHAPE);
    long long Mq = 0, Mk = 0, Mv = 0, Mr = 0, Mo = 0;
    const long long q_ld = r4d_rows_pitch(q, Wd, &Mq), k_ld = r4d_rows_pitch(k, Wd, &Mk);
    const long long v_ld = r4d_rows_pitch(v, Wd, &Mv), r_ld = r4d_rows_pitch(rk, Wd, &Mr);
    const long long o_ld = r4d_rows_pitch(o, Wd, &Mo);
    if (q_ld < 0 || k_ld < 0 || v_ld < 0 || r_ld < 0 || o_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mo <= 0 || Mq < Mo || Mk < Mo || Mv < Mo) return R4D_NO(RAD_E_SHAPE);
    g.S = (int)(r4d_numel(cu) - 1);
    g.pds_f32 = pds->dtype == RAD_F32;
    g.qs = (float)(1.0 / std::sqrt((double)g.D) / std::log(2.0));
    g.ks = (float)(std::log(1.0 + std::exp(1.0)) / std::log(2.0));
    g.q_ld = (uint32_t)q_ld; g.k_ld = (uint32_t)k_ld; g.v_ld = (uint32_t)v_ld;
    g.r_ld = (uint32_t)r_ld; g.o_ld = (uint32_t)o_ld;
    return enc_status(r4d_attn_local_rel_bf16(r4d_addr(q), r4d_addr(k), r4d_addr(v), r4d_addr(rk),
                                              r4d_addr(pds), r4d_addr(cu), r4d_addr(o), (int)Mo, &g,
                                              (long)(uintptr_t)s));
}
