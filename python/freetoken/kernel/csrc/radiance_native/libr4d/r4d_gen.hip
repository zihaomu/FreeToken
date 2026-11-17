// R4D -- the image generator's ops (docs/OPS.md, "Image generation"): a diffusion transformer's
// adaptive norm and gated residual, the timestep embedding, the per-image latent lanes and the
// flow-matching sampler over them, and the image decoder's convolution, channel norm, resampling
// shortcuts and pixel output.
//
// Each kernel rounds where its libref function does (libref/ref_gen.cpp), and libref rounds where
// the reference model materialises a bf16 tensor. Only reduction order differs (a block sum here,
// a sequential one there), so a sum can land one bf16 step apart.
//
// Image geometry comes from the operands: an image operand is a dense [height, width * channels],
// and the shims read both off its shape, because parameters are fixed at declare and image size
// is per request.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_fwht.h"
#include "r4d_gdn_wmma.h"
#include "r4d_q8.h"
#include "rad_philox.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kT = 256;

__device__ __forceinline__ float bfr(float v) { return bf2f(f32_to_bf16(v)); }

// Eight bf16 at a 16-byte-aligned address, widened, and the reverse.
__device__ __forceinline__ void ld8(const uint16_t* p, float* f) {
    const uint4 u = *(const uint4*)p;
    f[0] = bf16w_lo(u.x); f[1] = bf16w_hi(u.x); f[2] = bf16w_lo(u.y); f[3] = bf16w_hi(u.y);
    f[4] = bf16w_lo(u.z); f[5] = bf16w_hi(u.z); f[6] = bf16w_lo(u.w); f[7] = bf16w_hi(u.w);
}
__device__ __forceinline__ void st8(uint16_t* p, const float* f) {
    uint4 u;
    u.x = (uint32_t)f32_to_bf16(f[0]) | ((uint32_t)f32_to_bf16(f[1]) << 16);
    u.y = (uint32_t)f32_to_bf16(f[2]) | ((uint32_t)f32_to_bf16(f[3]) << 16);
    u.z = (uint32_t)f32_to_bf16(f[4]) | ((uint32_t)f32_to_bf16(f[5]) << 16);
    u.w = (uint32_t)f32_to_bf16(f[6]) | ((uint32_t)f32_to_bf16(f[7]) << 16);
    *(uint4*)p = u;
}

// The entry that owns row r of a cu_seqlens partition: linear, because there are a handful.
__device__ __forceinline__ int seq_of(const int* cu, int n_seq, int r) {
    int s = 0;
    while (s + 1 < n_seq && cu[s + 1] <= r) ++s;
    return s;
}

// A sum over the 256 threads, every thread receiving it. `sh` is 8 floats of LDS.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = r4d_wave_sum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    #pragma unroll
    for (int i = 0; i < kT / 32; ++i) t += sh[i];
    return t;
}

// ---------------------------------------------------------------------------- ln_mod
// A workgroup a row; mean, then variance about it, in f32 over the bf16 row.
__global__ __launch_bounds__(kT) void ln_mod_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ s, const int* __restrict__ cu,
    uint16_t* __restrict__ y, int n, int n_seq, float eps, int x_ld, int s_ld, int y_ld) {
    __shared__ float sh[kT / 32];
    const int r = blockIdx.x;
    const uint16_t* xr = x + (size_t)r * x_ld;
    float sum = 0.0f;
    for (int i = threadIdx.x * 8; i < n; i += kT * 8) {
        float f[8];
        ld8(xr + i, f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) sum += f[j];
    }
    const float mean = block_sum(sum, sh) / (float)n;
    float var = 0.0f;
    for (int i = threadIdx.x * 8; i < n; i += kT * 8) {
        float f[8];
        ld8(xr + i, f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) { const float d = f[j] - mean; var += d * d; }
    }
    const float rs = 1.0f / sqrtf(block_sum(var, sh) / (float)n + eps);
    const uint16_t* sr = s + (size_t)seq_of(cu, n_seq, r) * s_ld;
    uint16_t* yr = y + (size_t)r * y_ld;
    for (int i = threadIdx.x * 8; i < n; i += kT * 8) {
        float f[8], g[8];
        ld8(xr + i, f);
        ld8(sr + i, g);
        #pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = bfr((f[j] - mean) * rs) * bfr(1.0f + g[j]);
        st8(yr + i, f);
    }
}

// ---------------------------------------------------------------------------- gate_add
// `act` 0 gates through bf16(tanh(g)), 1 through g as it is (a gate with no tanh).
__global__ __launch_bounds__(kT) void gate_add_kernel(
    const uint16_t* __restrict__ y, const uint16_t* __restrict__ g, const int* __restrict__ cu,
    uint16_t* __restrict__ x, int n, int n_seq, int y_ld, int g_ld, int x_ld, int act) {
    const int r = blockIdx.x;
    const uint16_t* gr = g + (size_t)seq_of(cu, n_seq, r) * g_ld;
    for (int i = threadIdx.x * 8; i < n; i += kT * 8) {
        float a[8], b[8], c[8];
        ld8(y + (size_t)r * y_ld + i, a);
        ld8(gr + i, b);
        ld8(x + (size_t)r * x_ld + i, c);
        #pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = c[j] + bfr((act ? b[j] : bfr(tanhf(b[j]))) * a[j]);
        st8(x + (size_t)r * x_ld + i, c);
    }
}

// ---------------------------------------------------------------------------- table_add
// y[r, l n + c] = x[r, c] + t[l, c], f32 sum rounded once: one workgroup a (row, table row).
__global__ __launch_bounds__(kT) void table_add_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ t, uint16_t* __restrict__ y,
    int n, int L, int x_ld, int y_ld) {
    const int r = blockIdx.x / L, l = blockIdx.x % L;
    for (int i = threadIdx.x * 8; i < n; i += kT * 8) {
        float a[8], b[8];
        ld8(x + (size_t)r * x_ld + i, a);
        ld8(t + (size_t)l * n + i, b);
        #pragma unroll
        for (int j = 0; j < 8; ++j) a[j] += b[j];
        st8(y + (size_t)r * y_ld + (size_t)l * n + i, a);
    }
}

// ---------------------------------------------------------------------------- mix_rows
// y[r, c] = sum_i w[i] x[r k + i, c], f32 in ascending i, rounded once: one workgroup a row.
__global__ __launch_bounds__(kT) void mix_rows_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, uint16_t* __restrict__ y,
    int n, int k, int x_ld, int y_ld) {
    const int r = blockIdx.x;
    for (int i = threadIdx.x * 8; i < n; i += kT * 8) {
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}, v[8];
        for (int q = 0; q < k; ++q) {
            const float wq = bf2f(w[q]);
            ld8(x + (size_t)(r * k + q) * x_ld + i, v);
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[j] += wq * v[j];
        }
        st8(y + (size_t)r * y_ld + i, acc);
    }
}

// ---------------------------------------------------------------------------- chan_stats
// The statistics are rounded on the host side of each thread (cheap: n reads), every step as
// libref rounds it.
__global__ __launch_bounds__(kT) void chan_stats_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ mean,
    const uint16_t* __restrict__ sv, uint16_t* __restrict__ y, int n, int x_ld, int y_ld,
    int var, float eps, int norm) {
    const int r = blockIdx.x;
    for (int i = threadIdx.x * 8; i < n; i += kT * 8) {
        float a[8], mu[8], sd[8];
        ld8(x + (size_t)r * x_ld + i, a);
        ld8(mean + i, mu);
        ld8(sv + i, sd);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float s = var ? bfr(sqrtf(sd[j] + eps)) : sd[j];
            a[j] = norm ? bfr(bfr(a[j] - mu[j]) / s) : bfr(bfr(a[j] * s) + mu[j]);
        }
        st8(y + (size_t)r * y_ld + i, a);
    }
}

// ---------------------------------------------------------------------------- ln_mod_quant_fp8
// gate_add, ln_mod and quant_act_fp8 in one pass over the row: the DiT block's norm in front of
// every quantised linear, with the previous branch's gated residual folded in. Three passes over a
// 16384 x 4096 activation are ~0.9 GB of traffic per norm; this reads the delta and the residual
// and writes the residual and the E4M3 pair. The bf16 norm output is written only when bound.
//
// Bit-identical to the three kernels it replaces: the residual is gate_add's arithmetic rounded to
// bf16 as its store rounds it, mean and variance fold each thread's 8-wide chunks in
// ln_mod_kernel's order through the same block_sum, the norm output is rounded to bf16 before
// quantising, and the codes are r4d_q8.h's. A 128-column group is sixteen consecutive lanes, so its
// amax is a half-wave reduction.
//
// HI8 is ln_mod_had_quant_i8g: had_quant_act_i8g's quantiser in place of quant_act_fp8's. The
// bf16-rounded norm output is rotated by the 128-wide Hadamard where it lies, eight columns a lane
// and sixteen lanes a block (strides 1, 2, 4 in the lane, then 8 to 64 across lanes, the order
// r4d_fwht_wave takes them in, so the sums are the standalone kernel's), then int8 codes under
// one scale per (row, 128 * SUB): SUB 2 is IQ8R's 256-wide group, a whole wave. `z` receives the
// unrotated norm output. Byte-identical to gate_add, ln_mod and had_quant_act_i8g.
constexpr int kLnqChunks = 4;   // 8-wide chunks a thread keeps: n <= kT * 8 * 4 = 8192

template <bool GATED, bool HI8 = false, int SUB = 1>
__global__ __launch_bounds__(kT) void ln_mod_quant_kernel(
    const uint16_t* __restrict__ yd, const uint16_t* __restrict__ g, const uint16_t* __restrict__ s,
    const int* __restrict__ cu, uint16_t* __restrict__ x, uint8_t* __restrict__ q,
    float* __restrict__ qs, uint16_t* __restrict__ z, int n, int n_seq, float eps, int y_ld,
    int g_ld, int s_ld, int x_ld, int q_ld, int qs_ld, int z_ld) {
    __shared__ float sh[kT / 32];
    const int r = blockIdx.x;
    const int e = seq_of(cu, n_seq, r);
    uint16_t* xr = x + (size_t)r * x_ld;
    float v[kLnqChunks][8];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < kLnqChunks; ++k) {
        const int i = ((int)threadIdx.x + k * kT) * 8;
        if (i < n) {
            float f[8];
            ld8(xr + i, f);
            if (GATED) {
                float a[8], b[8];
                ld8(yd + (size_t)r * y_ld + i, a);
                ld8(g + (size_t)e * g_ld + i, b);
                #pragma unroll
                for (int j = 0; j < 8; ++j) f[j] = f[j] + bfr(bfr(tanhf(b[j])) * a[j]);
                st8(xr + i, f);
                #pragma unroll
                for (int j = 0; j < 8; ++j) f[j] = bfr(f[j]);
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) { v[k][j] = f[j]; sum += f[j]; }
        }
    }
    const float mean = block_sum(sum, sh) / (float)n;
    float var = 0.0f;
    #pragma unroll
    for (int k = 0; k < kLnqChunks; ++k)
        if (((int)threadIdx.x + k * kT) * 8 < n) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) { const float d = v[k][j] - mean; var += d * d; }
        }
    const float rs = 1.0f / sqrtf(block_sum(var, sh) / (float)n + eps);
    const uint16_t* sr = s + (size_t)e * s_ld;
    #pragma unroll
    for (int k = 0; k < kLnqChunks; ++k) {
        const int i = ((int)threadIdx.x + k * kT) * 8;
        if (i >= n) break;
        float gs[8], o[8];
        ld8(sr + i, gs);
        #pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = bfr(bfr((v[k][j] - mean) * rs) * bfr(1.0f + gs[j]));
        if (z) st8(z + (size_t)r * z_ld + i, o);
        if constexpr (HI8) {
            #pragma unroll
            for (int h = 1; h < 8; h <<= 1) {
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    if ((j & h) == 0) {
                        const float a = o[j], b = o[j + h];
                        o[j] = a + b;
                        o[j + h] = a - b;
                    }
            }
            const int lane = (int)threadIdx.x & 31;
            r4d_fwht_xstage<8, 1>(o, lane);
            r4d_fwht_xstage<8, 2>(o, lane);
            r4d_fwht_xstage<8, 4>(o, lane);
            r4d_fwht_xstage<8, 8>(o, lane);
        }
        float m = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) m = fmaxf(m, fabsf(o[j]));
        m = fmaxf(m, r4d_xor_lane<8>(m));
        m = fmaxf(m, r4d_xor_lane<4>(m));
        m = fmaxf(m, r4d_xor_lane<2>(m));
        m = fmaxf(m, r4d_xor_lane<1>(m));
        if constexpr (SUB == 2) m = fmaxf(m, r4d_xor_lane<16>(m));
        const float sc = m > 0.0f ? m * (1.0f / q8_top<HI8>()) : 1.0f;
        const float inv = 1.0f / sc;
        if ((threadIdx.x & (16 * SUB - 1)) == 0) qs[(size_t)r * qs_ld + i / (128 * SUB)] = sc;
        *(uint2*)(q + (size_t)r * q_ld + i) =
            make_uint2(q8_pack4<HI8>(o[0], o[1], o[2], o[3], inv),
                       q8_pack4<HI8>(o[4], o[5], o[6], o[7], inv));
    }
}

// ---------------------------------------------------------------------------- timestep_embed
// An output element, bf16 rounded once or f32 as it is.
__device__ __forceinline__ void st1(uint16_t* p, float v) { *p = f32_to_bf16(v); }
__device__ __forceinline__ void st1(float* p, float v) { *p = v; }

template <typename TY>
__global__ __launch_bounds__(kT) void timestep_kernel(
    const float* __restrict__ t, TY* __restrict__ y, int h, float scale, float coef,
    int round, float clamp_min, int y_ld) {
    const int m = blockIdx.x;
    const float tv = fmaxf(t[m], clamp_min);
    const float tt = round == 3 ? bfr(scale * bfr(tv)) : round == 2 ? scale * bfr(tv)
                   : round ? scale * bfr(bfr(tv * scale) / scale) : scale * tv;
    for (int i = threadIdx.x; i < h; i += kT) {
        const float f = expf(coef * (float)i / (float)h);
        const float arg = tt * f;
        st1(y + (size_t)m * y_ld + i, cosf(arg));
        st1(y + (size_t)m * y_ld + h + i, sinf(arg));
    }
}

// The curve form: y[m] is the table [grid, n] f32 at clamp01(max(t[m], clamp_min)) (grid - 1),
// between the two rows around it.
template <typename TY>
__global__ __launch_bounds__(kT) void timestep_curve_kernel(
    const float* __restrict__ t, const float* __restrict__ table, TY* __restrict__ y,
    int grid, int n, float clamp_min, int y_ld) {
    const int m = blockIdx.x;
    const float tv = fmaxf(t[m], clamp_min);
    const float pos = fminf(fmaxf(tv, 0.0f), 1.0f) * (float)(grid - 1);
    float i0f = floorf(pos);
    if (i0f > (float)(grid - 2)) i0f = (float)(grid - 2);   /* as torch clamps it */
    const float fr = pos - i0f;
    const int i0 = (int)i0f;
    for (int i = threadIdx.x; i < n; i += kT) {
        const float a0 = table[(size_t)i0 * n + i];
        const float a1 = table[(size_t)(i0 + 1) * n + i];
        st1(y + (size_t)m * y_ld + i, a0 + fr * (a1 - a0));
    }
}

// ---------------------------------------------------------------------------- row moves
__global__ __launch_bounds__(kT) void gather2_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, const int* __restrict__ idx,
    uint16_t* __restrict__ y, int n, int ra, int rb, int a_ld, int b_ld, int y_ld) {
    const int r = blockIdx.x;
    const int i = idx[r];
    const uint16_t* src = nullptr;
    if (i >= 0 && i < ra) src = a + (size_t)i * a_ld;
    if (i < 0 && -1 - i < rb) src = b + (size_t)(-1 - i) * b_ld;
    for (int c = threadIdx.x * 8; c < n; c += kT * 8)
        *(uint4*)(y + (size_t)r * y_ld + c) = src ? *(const uint4*)(src + c) : make_uint4(0, 0, 0, 0);
}

// The lane copies move 32-bit words: a row is `words` words (64 bf16 latent channels are 32, an
// f32 mask value is one), so one kernel serves every dtype the lanes hold.
__global__ __launch_bounds__(kT) void lane_gather_kernel(
    const uint32_t* __restrict__ src, const int* __restrict__ lane, const int* __restrict__ cu,
    uint32_t* __restrict__ y, int words, int n_seq, int stride, int s_ldw, int y_ldw, int64_t total) {
    const int64_t t = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (t >= total) return;
    const int r = cu[0] + (int)(t / words), w = (int)(t % words);
    if (r >= cu[n_seq]) return;
    const int e = seq_of(cu, n_seq, r);
    const int k = r - cu[e];
    y[(size_t)r * y_ldw + w] = src[((size_t)lane[e] * stride + k) * s_ldw + w];
}

__global__ __launch_bounds__(kT) void lane_store_kernel(
    const uint32_t* __restrict__ src, const int* __restrict__ lane, uint32_t* __restrict__ dst,
    int words, int stride, int s_ldw, int d_ldw, int64_t total) {
    const int64_t t = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (t >= total) return;
    const int r = (int)(t / words), w = (int)(t % words);
    dst[((size_t)lane[blockIdx.y] * stride + r) * d_ldw + w] = src[(size_t)r * s_ldw + w];
}

__global__ __launch_bounds__(kT) void scatter_add_kernel(
    const uint16_t* __restrict__ v, const int* __restrict__ idx, uint16_t* __restrict__ x, int n,
    int v_ld, int x_ld) {
    const int m = blockIdx.x;
    const int r = idx[m];
    if (r < 0) return;
    for (int c = threadIdx.x * 8; c < n; c += kT * 8) {
        float a[8], b[8];
        ld8(v + (size_t)m * v_ld + c, a);
        ld8(x + (size_t)r * x_ld + c, b);
        #pragma unroll
        for (int j = 0; j < 8; ++j) b[j] += a[j];
        st8(x + (size_t)r * x_ld + c, b);
    }
}

// ---------------------------------------------------------------------------- the sampler
// The seed of entry e, low word first.
__device__ __forceinline__ uint64_t seed_of(const uint32_t* seed, int e) {
    return (uint64_t)seed[2 * e] | ((uint64_t)seed[2 * e + 1] << 32);
}

// `TD` is the lane's element: bf16 latents, or f32 ones the sampler keeps in f32.
template <typename TD>
__global__ __launch_bounds__(kT) void randn_lanes_kernel(
    const int* __restrict__ flags, const uint32_t* __restrict__ seed, const int* __restrict__ lane,
    const int* __restrict__ cu, TD* __restrict__ dst, int n, int stride, int d_ld) {
    const int e = blockIdx.y;
    if (!(flags[e] & 1)) return;
    const int64_t i = (int64_t)blockIdx.x * kT + threadIdx.x;
    const int64_t len = cu[e + 1] - cu[e];
    if (i >= len * n) return;
    const int64_t k = i / n, c = i % n;
    st1(dst + ((size_t)lane[e] * stride + k) * d_ld + c,
        rad_philox_normal(seed_of(seed, e), (uint64_t)i));
}

__global__ __launch_bounds__(kT) void flow_euler_kernel(
    const uint16_t* __restrict__ v, const float* __restrict__ sg, const float* __restrict__ sgn,
    const float* __restrict__ cfg, const int* __restrict__ pair, const int* __restrict__ lane,
    const int* __restrict__ flags, const uint32_t* __restrict__ seed, const int* __restrict__ cu,
    const uint16_t* __restrict__ inp, const float* __restrict__ mask, uint16_t* __restrict__ st,
    uint16_t* __restrict__ pv, int n, int stride, int v_ld, int i_ld, int st_ld, int pv_ld) {
    const int e = blockIdx.y;
    const int pr = pair[e];
    if (pr == -2) return;
    const int b0 = cu[e];
    const int64_t len = cu[e + 1] - b0;
    const int64_t i = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (i >= len * n) return;
    const int64_t k = i / n, c = i % n;
    const int L = lane[e], f = flags[e];
    const float s = sg[e], sn = sgn[e];
    float vv = bf2f(v[(size_t)(b0 + k) * v_ld + c]);
    if (pr >= 0) {
        const float vn = bf2f(v[(size_t)(cu[pr] + k) * v_ld + c]);
        vv = bfr(vn + bfr(cfg[e] * bfr(vv - vn)));
    }
    const size_t so = ((size_t)L * stride + k) * st_ld + c;
    const float x = bf2f(st[so]);
    if ((f & 4) && pv) pv[((size_t)L * stride + k) * pv_ld + c] = f32_to_bf16(x - s * vv);
    float xn = bfr(x + bfr(bfr(sn - s) * vv));
    if ((f & 2) && inp && mask) {
        const float m = mask[(size_t)L * stride + k];
        const float src = bf2f(inp[((size_t)L * stride + k) * i_ld + c]);
        const float nz = rad_philox_normal(seed_of(seed, e), (uint64_t)i);
        xn = bfr(m * xn + (1.0f - m) * ((1.0f - sn) * src + sn * nz));
    }
    st[so] = f32_to_bf16(xn);
}

// ---------------------------------------------------------------------------- kd_input, kd_step
// The k-diffusion sampler over a discrete-time denoiser (libref's ref_kd_input / ref_kd_step): the
// lanes hold x = x0 + sigma eps in f32, a thread one element of one entry, every entry's constants
// computed by each of its threads (a handful of f32 operations beside the element's own).
constexpr int kKdEuler = 0, kKdAncestral = 1, kKdDpm2m = 2;

struct KdConst {
    float s, sn, w, cskip, cout, sup, sdown, ratio, em, c1, c2;
    bool second;
};

__device__ __forceinline__ KdConst kd_const(float s, float sn, float sp, float w, int sampler,
                                            bool first) {
    KdConst k{};
    k.s = s;
    k.sn = sn;
    k.w = w;
    const float s2p = s * s + 1.0f;
    k.cskip = 1.0f / s2p;
    k.cout = s / sqrtf(s2p);
    if (sampler == kKdAncestral) {
        const float s2 = s * s, sn2 = sn * sn;
        const float up2 = sn2 * (s2 - sn2) / s2;
        float sup = sqrtf(up2);
        if (!(sup < sn)) sup = sn;
        const float dn2 = sn2 - sup * sup;
        k.sup = sup;
        k.sdown = sqrtf(dn2 > 0.0f ? dn2 : 0.0f);
    } else if (sampler == kKdDpm2m && sn > 0.0f) {
        const float t = -logf(s), tn = -logf(sn);
        const float h = tn - t;
        k.ratio = sn / s;
        k.em = expm1f(-h);
        if (!first) {
            const float r = (t - (-logf(sp))) / h;
            k.c2 = 1.0f / (2.0f * r);
            k.c1 = 1.0f + k.c2;
            k.second = true;
        }
    }
    return k;
}

__device__ __forceinline__ float kd_den(const KdConst& k, bool v_pred, float x, float o) {
    return v_pred ? x * k.cskip - o * k.cout : x - k.s * o;
}

__global__ __launch_bounds__(kT) void kd_input_kernel(
    const float* __restrict__ sg, const int* __restrict__ pair, const int* __restrict__ lane,
    const int* __restrict__ flags, const uint32_t* __restrict__ seed, const int* __restrict__ cu,
    float* __restrict__ st, uint16_t* __restrict__ y, int n, int stride, int st_ld, int y_ld) {
    const int e = blockIdx.y;
    const int pr = pair[e];
    if (pr == -2) return;
    const int b0 = cu[e];
    const int64_t len = cu[e + 1] - b0;
    const int64_t i = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (i >= len * n) return;
    const int64_t k = i / n, c = i % n;
    const float s = sg[e];
    float* xp = st + ((size_t)lane[e] * stride + k) * st_ld + c;
    if (flags[e] & 1) *xp = rad_philox_normal(seed_of(seed, e), (uint64_t)i) * sqrtf(1.0f + s * s);
    const uint16_t v = f32_to_bf16(*xp * (1.0f / sqrtf(s * s + 1.0f)));
    y[(size_t)(b0 + k) * y_ld + c] = v;
    if (pr >= 0) y[(size_t)(cu[pr] + k) * y_ld + c] = v;
}

__global__ __launch_bounds__(kT) void kd_step_kernel(
    const uint16_t* __restrict__ out, const float* __restrict__ sg, const float* __restrict__ sgn,
    const float* __restrict__ sgp, const float* __restrict__ cfg, const int* __restrict__ pair,
    const int* __restrict__ lane, const int* __restrict__ flags, const uint32_t* __restrict__ seed,
    const int* __restrict__ sampler, const int* __restrict__ step, const int* __restrict__ cu,
    float* __restrict__ st, float* __restrict__ hist, float* __restrict__ pv, int n, int stride,
    int o_ld, int st_ld, int h_ld, int pv_ld, int v_pred) {
    const int e = blockIdx.y;
    const int pr = pair[e];
    if (pr == -2) return;
    const int b0 = cu[e];
    const int64_t len = cu[e + 1] - b0;
    const int64_t i = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (i >= len * n) return;
    const int64_t k = i / n, c = i % n;
    const int f = flags[e], sm = sampler[e];
    const KdConst K = kd_const(sg[e], sgn[e], sgp[e], cfg[e], sm, (f & 1) != 0);
    const size_t row = (size_t)lane[e] * stride + k;
    const float x = st[row * st_ld + c];
    float D = kd_den(K, v_pred != 0, x, bf2f(out[(size_t)(b0 + k) * o_ld + c]));
    if (pr >= 0) {
        const float Du = kd_den(K, v_pred != 0, x, bf2f(out[(size_t)(cu[pr] + k) * o_ld + c]));
        D = Du + K.w * (D - Du);
    }
    if ((f & 4) && pv) pv[row * pv_ld + c] = D;
    float xn;
    if (sm == kKdEuler) {
        xn = x + (K.sn - K.s) * ((x - D) / K.s);
    } else if (sm == kKdAncestral) {
        xn = x + ((x - D) / K.s) * (K.sdown - K.s);
        if (K.sn > 0.0f)
            xn += rad_philox_normal_s(seed_of(seed, e), (uint64_t)(uint32_t)(1 + step[e]),
                                      (uint64_t)i) * K.sup;
    } else if (sm != kKdDpm2m) {
        return;
    } else if (K.sn == 0.0f) {
        xn = D;
    } else {
        const float Dd = K.second ? K.c1 * D - K.c2 * hist[row * h_ld + c] : D;
        xn = K.ratio * x - K.em * Dd;
    }
    st[row * st_ld + c] = xn;
    hist[row * h_ld + c] = D;
}

// ---------------------------------------------------------------------------- per-channel ops
// The two lists ride in the kernel arguments, whose segment the backend caps at 768 bytes; two
// lists of 64 channels (a latent's) take 512.
constexpr int kChanMax = 64;
struct ChanLists { float mul[kChanMax], add[kChanMax]; };

// `T` is x's and y's element, bf16 or f32. With `rnd` the inner product or difference rounds to
// y's dtype (and the lists were rounded to bf16 by the shim); without it the arithmetic stays f32
// and rounds once into y.
__device__ __forceinline__ float ca_ld(const uint16_t* p, size_t i) { return bf2f(p[i]); }
__device__ __forceinline__ float ca_ld(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float ca_rd(const uint16_t*, float v) { return bfr(v); }
__device__ __forceinline__ float ca_rd(const float*, float v) { return v; }
__device__ __forceinline__ void ca_st(uint16_t* p, int64_t i, float v) { p[i] = f32_to_bf16(v); }
__device__ __forceinline__ void ca_st(float* p, int64_t i, float v) { p[i] = v; }

// The element-wise form, any of x, the statistics (TW) and y in bf16 or f32: a latent kept in
// f32, or rows whose pitch the 8-wide kernel cannot take (an encoder's mean channels beside its
// log-variance). Each step rounds to y's dtype as the reference stores it; the explicit _rn
// intrinsics keep the compiler from contracting x s + mean into one fused multiply-add.
template <typename TX, typename TW, typename TY>
__global__ __launch_bounds__(kT) void chan_stats_any_kernel(
    const TX* __restrict__ x, const TW* __restrict__ mean, const TW* __restrict__ sv,
    TY* __restrict__ y, int n, int x_ld, int y_ld, int var, float eps, int norm) {
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += kT) {
        const float mu = ca_rd(y, ca_ld(mean, (size_t)i)), v = ca_ld(sv, (size_t)i);
        const float sd = ca_rd(y, var ? __fsqrt_rn(__fadd_rn(v, eps)) : v);
        const float a = ca_ld(x, (size_t)r * x_ld + i);
        ca_st(y, (int64_t)r * y_ld + i,
              norm ? ca_rd(y, __fdiv_rn(ca_rd(y, __fsub_rn(a, mu)), sd))
                   : ca_rd(y, __fadd_rn(ca_rd(y, __fmul_rn(a, sd)), mu)));
    }
}

template <typename T>
__global__ __launch_bounds__(kT) void chan_affine_kernel(
    const T* __restrict__ x, T* __restrict__ y, int64_t total, int n, int x_ld, int inv, int rnd,
    ChanLists l) {
    const int64_t i = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (i >= total) return;
    const int c = (int)(i % n);
    const float v = ca_ld(x, (size_t)(i / n) * x_ld + c);
    const float t = inv ? v - l.add[c] : v * l.mul[c];
    const float r = rnd ? ca_rd(y, t) : t;
    ca_st(y, i, inv ? r / l.mul[c] : r + l.add[c]);
}

// A wave a pixel: the sum of squares over its channels, then the three roundings.
__global__ __launch_bounds__(kT) void l2norm_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ g, uint16_t* __restrict__ y,
    int64_t rows, int n, float sq, int silu) {
    const int64_t r = (int64_t)blockIdx.x * (kT / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const uint16_t* xr = x + (size_t)r * n;
    float ss = 0.0f;
    for (int c = lane * 8; c < n; c += 256) {
        float f[8];
        ld8(xr + c, f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) ss += f[j] * f[j];
    }
    const float nrm = fmaxf(sqrtf(r4d_wave_sum(ss)), 1e-12f);
    uint16_t* yr = y + (size_t)r * n;
    for (int c = lane * 8; c < n; c += 256) {
        float f[8], gg[8];
        ld8(xr + c, f);
        ld8(g + c, gg);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = bfr(bfr(bfr(f[j] / nrm) * sq) * gg[j]);
            if (silu) v = v / (1.0f + expf(-v));
            f[j] = v;
        }
        st8(yr + c, f);
    }
}

// y[2r+i, 2c+j, o] += x[r, c, (o * F + off + 2i + j) / rep]
__global__ __launch_bounds__(kT) void d2s_add_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, int H, int W, int cin, int cout,
    int F, int off, int rep) {
    const int64_t total = (int64_t)4 * H * W * cout;
    const int64_t t = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (t >= total) return;
    const int o = (int)(t % cout);
    const int64_t px = t / cout;
    const int oy = (int)(px / (2 * W)), ox = (int)(px % (2 * W));
    const int ci = (o * F + off + 2 * (oy & 1) + (ox & 1)) / rep;
    const float a = bf2f(x[((size_t)(oy >> 1) * W + (ox >> 1)) * cin + ci]);
    y[t] = f32_to_bf16(bf2f(y[t]) + a);
}

// y[r, c, o] += mean over the group of the dealt 2x2 block (a zero frame counted with ft 2)
__global__ __launch_bounds__(kT) void s2d_avg_add_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, int H, int W, int cin, int cout,
    int F, int gs, int ft) {
    const int64_t total = (int64_t)H * W * cout;
    const int64_t t = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (t >= total) return;
    const int o = (int)(t % cout);
    const int64_t px = t / cout;
    const int oy = (int)(px / W), ox = (int)(px % W);
    float acc = 0.0f;
    for (int gg = 0; gg < gs; ++gg) {
        const int g = o * gs + gg;
        const int c = g / F, rem = g % F, tt = rem / 4, i = (rem % 4) / 2, j = rem % 2;
        if (ft == 2 && tt == 0) continue;
        acc += bf2f(x[((size_t)(2 * oy + i) * (2 * W) + (2 * ox + j)) * cin + c]);
    }
    y[t] = f32_to_bf16(bf2f(y[t]) + bfr(acc / (float)gs));
}

__global__ __launch_bounds__(kT) void to_rgba8_kernel(const uint16_t* __restrict__ x,
                                                      uint8_t* __restrict__ y, int64_t total) {
    const int64_t i = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (i >= total) return;
    float v = fminf(fmaxf(bf2f(x[i]), -1.0f), 1.0f);
    float u = bfr(bfr(v * 0.5f) + 0.5f);
    u = fminf(fmaxf(u, 0.0f), 1.0f);
    y[i] = (uint8_t)rintf(u * 255.0f);
}

// ---------------------------------------------------------------------------- gelu_quick
// CLIP's quick_gelu, libref's three roundings: a workgroup a row, a thread an element.
template <typename T>
__global__ __launch_bounds__(kT) void gelu_quick_kernel(const T* __restrict__ x, T* __restrict__ y,
                                                        int n, int x_ld, int y_ld) {
    const int r = blockIdx.x;
    for (int c = threadIdx.x; c < n; c += kT) {
        const float v = ca_ld(x, (size_t)r * x_ld + c);
        const float a = ca_rd(y, 1.702f * v);
        const float s = ca_rd(y, 1.0f / (1.0f + expf(-a)));
        ca_st(y, (int64_t)r * y_ld + c, v * s);
    }
}

__global__ __launch_bounds__(kT) void gelu_tanh_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, int n, int x_ld, int y_ld) {
    const int r = blockIdx.x;
    for (int c = threadIdx.x * 8; c < n; c += kT * 8) {
        float f[8];
        ld8(x + (size_t)r * x_ld + c, f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float v = f[j];
            f[j] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        }
        st8(y + (size_t)r * y_ld + c, f);
    }
}

// ---------------------------------------------------------------------------- conv2d_nhwc
// An implicit GEMM: y[pixel][cout] = sum over (tap, cin) of x_virtual[pixel's tap][cin] *
// w[cout][tap][cin]. The weight is stored tap-major (r4d_layout.cpp relayouts PyTorch's
// [cout][cin][k][k] at load), so K runs over taps then channels and a 64-wide K tile of the image
// is eight 16-byte reads of NHWC rows; no im2col buffer.
//
// The block is r4d_gemm_bf16_nt_tiled's: eight waves, an activation tile of BP pixels and a weight
// tile of BC output channels staged through LDS 64 K at a time, with the next tile's loads in
// flight under the current WMMA. The weight is the WMMA's A operand and the pixels its B, so a
// lane's accumulator is eight consecutive output channels of one pixel and the store is 16 bytes.
//
// An eight-channel input chunk never straddles a tap because cin is a multiple of eight (the shim
// sends anything else to the direct kernel); a chunk past the image, K or the last pixel is zeros.
//
// Kernel size and upsampling are template parameters (KS, UP; 0 reads the argument): at runtime
// they cost two integer divisions per staged chunk (~500 integer instructions beside 48 WMMA a
// wave), at 3 and 2 a multiply-shift and a shift.
#define CV_BK 64
#define CV_PITCH (CV_BK + 8)

template <int TP, int TC, int KS, int UP>
__global__ __launch_bounds__(256) void conv_igemm_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ bias, uint16_t* __restrict__ y, int Hv, int Wv, int cin, int Wo,
    int cout, int k, int stride, int up, int pt, int pl, int x_ld, int y_ld, int Mpx, int K) {
    constexpr int BP = 2 * TP * 16, BC = 4 * TC * 16;
    constexpr int CPR = CV_BK / 8;
    constexpr int AL = BP * CPR / 256, WL = BC * CPR / 256;
    static_assert(AL >= 1 && WL >= 1, "every thread stages whole chunks");
    __shared__ __attribute__((aligned(16))) uint16_t sA[BP * CV_PITCH];
    __shared__ __attribute__((aligned(16))) uint16_t sW[BC * CV_PITCH];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wp = wave / 4, wc = wave % 4;
    const int p0 = blockIdx.x * BP, c0 = blockIdx.y * BC;
    const int kq = KS ? KS : k, kk2 = kq * kq;

    int a_oy[AL], a_ox[AL], a_t[AL], a_ci[AL];
    bool a_ok[AL];
    uint16_t* adst[AL];
    #pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int m = p0 + r;
        a_ok[i] = m < Mpx;
        const int mm = a_ok[i] ? m : 0;
        a_oy[i] = mm / Wo;
        a_ox[i] = mm % Wo;
        a_t[i] = e / cin;
        a_ci[i] = e % cin;
        adst[i] = sA + r * CV_PITCH + e;
    }
    size_t w_off[WL];
    int w_e[WL];
    bool w_ok[WL];
    uint16_t* wdst[WL];
    #pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int co = c0 + r;
        w_ok[i] = co < cout;
        w_off[i] = (size_t)(w_ok[i] ? co : 0) * K + e;
        w_e[i] = e;
        wdst[i] = sW + r * CV_PITCH + e;
    }
    const r4d_u32x4 zero = {0u, 0u, 0u, 0u};
    auto loadA = [&](int i) -> r4d_u32x4 {
        if (!a_ok[i] || a_t[i] >= kk2) return zero;
        const int kh = a_t[i] / kq, kw = a_t[i] - kh * kq;
        const int iy = a_oy[i] * stride + kh - pt, ix = a_ox[i] * stride + kw - pl;
        if (iy < 0 || iy >= Hv || ix < 0 || ix >= Wv) return zero;
        /* Both are non-negative here, so a halving is a shift. */
        const int sy = UP == 1 ? iy : UP == 2 ? (iy >> 1) : iy / up;
        const int sx = UP == 1 ? ix : UP == 2 ? (ix >> 1) : ix / up;
        return *(const r4d_u32x4*)(x + (size_t)sy * x_ld + (size_t)sx * cin + a_ci[i]);
    };
    auto loadW = [&](int i, int kt) -> r4d_u32x4 {
        if (!w_ok[i] || kt * CV_BK + w_e[i] >= K) return zero;
        return *(const r4d_u32x4*)(w + w_off[i] + (size_t)kt * CV_BK);
    };

    const uint16_t* fa = sA + (wp * TP * 16 + (lane & 15)) * CV_PITCH + 4 * (lane >> 4);
    const uint16_t* fw = sW + (wc * TC * 16 + (lane & 15)) * CV_PITCH + 4 * (lane >> 4);

    v8f acc[TC][TP];
    #pragma unroll
    for (int j = 0; j < TC; ++j)
        #pragma unroll
        for (int i = 0; i < TP; ++i) acc[j][i] = (v8f)(0.f);

    r4d_u32x4 ra[AL], rw[WL];
    #pragma unroll
    for (int i = 0; i < AL; ++i) ra[i] = loadA(i);
    #pragma unroll
    for (int i = 0; i < WL; ++i) rw[i] = loadW(i, 0);

    const int nk = (K + CV_BK - 1) / CV_BK;
    for (int kt = 0; kt < nk; ++kt) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < AL; ++i) *(r4d_u32x4*)adst[i] = ra[i];
        #pragma unroll
        for (int i = 0; i < WL; ++i) *(r4d_u32x4*)wdst[i] = rw[i];
        __syncthreads();
        if (kt + 1 < nk) {
            #pragma unroll
            for (int i = 0; i < AL; ++i) {
                a_ci[i] += CV_BK;
                while (a_ci[i] >= cin) { a_ci[i] -= cin; ++a_t[i]; }
                ra[i] = loadA(i);
            }
            #pragma unroll
            for (int i = 0; i < WL; ++i) rw[i] = loadW(i, kt + 1);
        }
        #pragma unroll
        for (int s = 0; s < CV_BK / 16; ++s) {
            v8bf af[TP], wf[TC];
            #pragma unroll
            for (int i = 0; i < TP; ++i) af[i] = fragP(fa + i * 16 * CV_PITCH + s * 16);
            #pragma unroll
            for (int j = 0; j < TC; ++j) wf[j] = fragP(fw + j * 16 * CV_PITCH + s * 16);
            #pragma unroll
            for (int j = 0; j < TC; ++j)
                #pragma unroll
                for (int i = 0; i < TP; ++i) acc[j][i] = mma(wf[j], af[i], acc[j][i]);
        }
    }

    #pragma unroll
    for (int j = 0; j < TC; ++j)
        #pragma unroll
        for (int i = 0; i < TP; ++i) {
            const int px = p0 + (wp * TP + i) * 16 + (lane & 15);
            const int co = c0 + (wc * TC + j) * 16 + 8 * (lane >> 4);
            if (px >= Mpx || co >= cout) continue;
            const int oy = px / Wo, ox = px % Wo;
            uint16_t* dst = y + (size_t)oy * y_ld + (size_t)ox * cout + co;
            if (co + 8 <= cout && (cout & 7) == 0) {
                float f[8], b[8];
                ld8(bias + co, b);
                #pragma unroll
                for (int e = 0; e < 8; ++e) f[e] = acc[j][i][e] + b[e];
                st8(dst, f);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    if (co + e < cout) dst[e] = f32_to_bf16(acc[j][i][e] + bf2f(bias[co + e]));
            }
        }
}

// The direct form, for an input whose channel count is not a multiple of eight (an RGBA image
// into the first layer): a thread an output element, f32 over (tap, cin), the same tap-major
// weight.
__global__ __launch_bounds__(kT) void conv_direct_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ bias, uint16_t* __restrict__ y, int Hv, int Wv, int cin, int Wo,
    int cout, int k, int stride, int up, int pt, int pl, int x_ld, int y_ld, int64_t total) {
    const int64_t t = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (t >= total) return;
    const int co = (int)(t % cout);
    const int64_t px = t / cout;
    const int oy = (int)(px / Wo), ox = (int)(px % Wo);
    const int K = k * k * cin;
    float acc = 0.0f;
    for (int kh = 0; kh < k; ++kh) {
        const int iy = oy * stride + kh - pt;
        if (iy < 0 || iy >= Hv) continue;
        for (int kw = 0; kw < k; ++kw) {
            const int ix = ox * stride + kw - pl;
            if (ix < 0 || ix >= Wv) continue;
            const uint16_t* xp = x + (size_t)(iy / up) * x_ld + (size_t)(ix / up) * cin;
            const uint16_t* wp = w + (size_t)co * K + (size_t)(kh * k + kw) * cin;
            for (int ci = 0; ci < cin; ++ci) acc += bf2f(wp[ci]) * bf2f(xp[ci]);
        }
    }
    y[(size_t)oy * y_ld + (size_t)ox * cout + co] = f32_to_bf16(acc + bf2f(bias[co]));
}

// ---------------------------------------------------------------------------- im2col_nhwc
// conv2d_nhwc's receptive fields as rows: y[oy Wo + ox][(ci k + i) k + j] = the input at tap
// (i, j) of output pixel (oy, ox), channel ci. A block is a run of IC_TW output pixels of one
// output row and IC_CC input channels: it stages the k input rows those pixels read (through the
// upsample and the zero padding) in LDS channel-major, [cl][i][xw], 16 bytes of NHWC channels a
// load, then writes each pixel's cc k k columns -- one contiguous run of its y row -- VW elements a
// thread, so both the reads and the writes are coalesced. KS is k at compile time (3 and 1), 0 for
// any other. Values are copied bit for bit.
constexpr int IC_TW = 32;
constexpr int IC_CC = 64;

template <int KS, int VW>
__global__ __launch_bounds__(kT) void im2col_nhwc_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, int Hv, int Wv, int cin, int Wo,
    int k_rt, int stride, int up, int pt, int pl, int64_t x_ld, int64_t y_ld, int vec) {
    extern __shared__ uint16_t ic_lds[];
    const int k = KS ? KS : k_rt, kk = k * k;
    const int ox0 = blockIdx.x * IC_TW, oy = blockIdx.y, c0 = blockIdx.z * IC_CC;
    const int cc = min(IC_CC, cin - c0), nq = min(IC_TW, Wo - ox0);
    const int W = (IC_TW - 1) * stride + k;
    const int iy0 = oy * stride - pt, ix0 = ox0 * stride - pl;
    const int sites = k * W;
    const int tid = threadIdx.x;
    if (vec) {
        const int groups = cc >> 3;
        for (int u = tid; u < sites * groups; u += kT) {
            const int g = u % groups, site = u / groups;
            const int i = site / W, xw = site - i * W;
            const int iy = iy0 + i, ix = ix0 + xw;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (iy >= 0 && iy < Hv && ix >= 0 && ix < Wv)
                v = *(const uint4*)(x + (size_t)(iy / up) * x_ld + (size_t)(ix / up) * cin + c0 +
                                    g * 8);
            const uint32_t w4[4] = { v.x, v.y, v.z, v.w };
            #pragma unroll
            for (int e = 0; e < 8; ++e)
                ic_lds[((g * 8 + e) * k + i) * W + xw] =
                    (uint16_t)(e & 1 ? w4[e >> 1] >> 16 : w4[e >> 1] & 0xffffu);
        }
    } else {
        for (int u = tid; u < sites * cc; u += kT) {
            const int cl = u % cc, site = u / cc;
            const int i = site / W, xw = site - i * W;
            const int iy = iy0 + i, ix = ix0 + xw;
            uint16_t v = 0;
            if (iy >= 0 && iy < Hv && ix >= 0 && ix < Wv)
                v = x[(size_t)(iy / up) * x_ld + (size_t)(ix / up) * cin + c0 + cl];
            ic_lds[(cl * k + i) * W + xw] = v;
        }
    }
    __syncthreads();
    /* The block's pixels' runs as one list of VW-element pieces, so every thread stores whatever
     * the run length; a thread's (pixel, piece) advances by the block each pass, without a
     * division. */
    const int span = cc * kk, pieces = span / VW;
    uint16_t* yb = y + (size_t)((int64_t)oy * Wo + ox0) * y_ld + (size_t)c0 * kk;
    int q = tid / pieces, pc = tid - q * pieces;
    for (; q < nq;) {
        const int t = pc * VW;
        uint16_t* yr = yb + (size_t)q * y_ld;
        const int xq = q * stride;
        uint16_t v[VW];
        #pragma unroll
        for (int e = 0; e < VW; ++e) {
            const int te = t + e;
            const int cl = te / kk, r = te - cl * kk, i = r / k, j = r - i * k;
            v[e] = ic_lds[(cl * k + i) * W + xq + j];
        }
        if constexpr (VW == 8)
            *(uint4*)(yr + t) = make_uint4((uint32_t)v[0] | ((uint32_t)v[1] << 16),
                                           (uint32_t)v[2] | ((uint32_t)v[3] << 16),
                                           (uint32_t)v[4] | ((uint32_t)v[5] << 16),
                                           (uint32_t)v[6] | ((uint32_t)v[7] << 16));
        else if constexpr (VW == 4)
            *(uint2*)(yr + t) = make_uint2((uint32_t)v[0] | ((uint32_t)v[1] << 16),
                                           (uint32_t)v[2] | ((uint32_t)v[3] << 16));
        else if constexpr (VW == 2)
            *(uint32_t*)(yr + t) = (uint32_t)v[0] | ((uint32_t)v[1] << 16);
        else
            yr[t] = v[0];
        pc += kT;
        while (pc >= pieces) { pc -= pieces; ++q; }
    }
}

inline int st_of(hipError_t e) { return e == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE); }

inline unsigned blocks_for(int64_t n) { return (unsigned)((n + kT - 1) / kT); }

inline bool is_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }

// A row-major operand's row count and pitch, with the pitch a multiple of eight elements so a row
// is read 16 bytes at a time.
inline bool rows8(const RadTensor* t, long long n, long long* rows, long long* ld) {
    if (!t || n <= 0 || (n & 7)) return false;
    *ld = r4d_rows_pitch(t, n, rows);
    return *ld >= 0 && !(*ld & 7) && !((uintptr_t)t->data & 15);
}

// A dense [H, W * c] image operand.
inline bool image_of(const RadTensor* t, long long c, long long* H, long long* W,
                     bool pitched = false) {
    if (!t || t->rank != 2 || c <= 0 || t->shape[1] % c || t->stride[1] != 1 ||
        (pitched ? t->stride[0] < t->shape[1] : t->stride[0] != t->shape[1]))
        return false;
    *H = t->shape[0];
    *W = t->shape[1] / c;
    return true;
}

// A parameter list of floats, each rounded to bf16 as the reference's bf16 tensor of it, or kept
// in f32 with `bf16` false.
int parse_bf16_list(const char* s, float* out, int max, bool bf16 = true) {
    int n = 0;
    while (s && *s && n < max) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        float f = (float)v;
        if (bf16) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
            std::memcpy(&f, &u, 4);
        }
        out[n++] = f;
        s = end;
    }
    return n;
}

}  // namespace

// ============================================================== the plugin ABI entry points

extern "C" int r4d_rad_ln_mod(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *sc = r4d_opt(a, 1), *cu = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!x || !sc || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(sc) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long rx = 0, xl = 0, rs = 0, sl = 0, ry = 0, yl = 0;
    if (!rows8(x, n, &rx, &xl) || !rows8(y, n, &ry, &yl) || !r4d_contig(cu))
        return R4D_NO(RAD_E_STRIDE);
    sl = r4d_rows_pitch(sc, n, &rs);
    if (sl < 0 || (sl & 7) || ((uintptr_t)sc->data & 15)) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (ry <= 0 || rx < ry || n_seq <= 0 || rs < n_seq) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(ln_mod_kernel, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (const uint16_t*)sc->data, (const int*)cu->data,
                       (uint16_t*)y->data, (int)n, (int)n_seq, (float)r4d_getf(a, "eps", 1e-6),
                       (int)xl, (int)sl, (int)yl);
    return st_of(hipGetLastError());
}

/* ln_mod_quant_fp8 (E4M3 codes, group 128) and with `hi8` ln_mod_had_quant_i8g (rotated int8
 * codes, group 128 or 256). */
static int ln_mod_quant_launch(const RadArgs* a, RadStream s, bool hi8) {
    const RadTensor *y = r4d_opt(a, 0), *g = r4d_opt(a, 1), *sc = r4d_opt(a, 2);
    const RadTensor *cu = r4d_opt(a, 3), *x = r4d_opt(a, 4), *q = r4d_opt(a, 5);
    const RadTensor *qs = r4d_opt(a, 6), *z = r4d_opt(a, 7);
    if (!sc || !cu || !x || !q || !qs) return R4D_NO(RAD_E_INVAL);
    const bool gated = r4d_geti_or(a, "gated", 0) != 0;
    if (gated && (!y || !g)) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(sc) || (gated && (!is_bf16(y) || !is_bf16(g))) ||
        (z && !is_bf16(z)) || q->dtype != (hi8 ? RAD_I8 : RAD_F8E4M3) || qs->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    const long long grp = r4d_geti_or(a, "group", 0);
    if (n <= 0 || (grp != 128 && !(hi8 && grp == 256)) || n % grp ||
        n > (long long)kT * 8 * kLnqChunks)
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, xl = 0, ry = 0, yl = 0, rg = 0, gl = 0, rs = 0, sl = 0, rz = 0, zl = 0;
    long long rq = 0, rqs = 0;
    if (!rows8(x, n, &rx, &xl) || !rows8(sc, n, &rs, &sl) || !r4d_contig(cu) ||
        (gated && (!rows8(y, n, &ry, &yl) || !rows8(g, n, &rg, &gl))) ||
        (z && !rows8(z, n, &rz, &zl)))
        return R4D_NO(RAD_E_STRIDE);
    const long long ql = r4d_rows_pitch(q, n, &rq), qsl = r4d_rows_pitch(qs, n / grp, &rqs);
    if (ql < 0 || (ql & 7) || ((uintptr_t)q->data & 7) || qsl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1, rows = rq;
    if (rows <= 0 || n_seq <= 0 || rx < rows || rqs < rows || rs < n_seq ||
        (gated && (ry < rows || rg < n_seq)) || (z && rz < rows))
        return R4D_NO(RAD_E_SHAPE);
    const float eps = (float)r4d_getf(a, "eps", 1e-6);
#define R4D_LNQ(G, ...)                                                                           \
    hipLaunchKernelGGL((ln_mod_quant_kernel<G, ##__VA_ARGS__>), dim3((unsigned)rows), dim3(kT), 0, \
                       (hipStream_t)s,                                                           \
                       G ? (const uint16_t*)y->data : nullptr,                                   \
                       G ? (const uint16_t*)g->data : nullptr, (const uint16_t*)sc->data,        \
                       (const int*)cu->data, (uint16_t*)x->data, (uint8_t*)q->data,              \
                       (float*)qs->data, z ? (uint16_t*)z->data : nullptr, (int)n, (int)n_seq,   \
                       eps, (int)yl, (int)gl, (int)sl, (int)xl, (int)ql, (int)qsl, (int)zl)
    if (!hi8) {
        if (gated) R4D_LNQ(true);
        else       R4D_LNQ(false);
    } else if (grp == 256) {
        if (gated) R4D_LNQ(true, true, 2);
        else       R4D_LNQ(false, true, 2);
    } else {
        if (gated) R4D_LNQ(true, true, 1);
        else       R4D_LNQ(false, true, 1);
    }
#undef R4D_LNQ
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_ln_mod_had_quant_i8g(const RadArgs* a, RadStream s) {
    return ln_mod_quant_launch(a, s, true);
}

extern "C" int r4d_rad_gate_add(const RadArgs* a, RadStream s) {
    const RadTensor *y = r4d_opt(a, 0), *g = r4d_opt(a, 1), *cu = r4d_opt(a, 2), *x = r4d_opt(a, 3);
    if (!y || !g || !cu || !x) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(g) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long ry = 0, yl = 0, rx = 0, xl = 0, rg = 0;
    if (!rows8(y, n, &ry, &yl) || !rows8(x, n, &rx, &xl) || !r4d_contig(cu))
        return R4D_NO(RAD_E_STRIDE);
    const long long gl = r4d_rows_pitch(g, n, &rg);
    if (gl < 0 || (gl & 7) || ((uintptr_t)g->data & 15)) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (ry <= 0 || rx < ry || n_seq <= 0 || rg < n_seq) return R4D_NO(RAD_E_SHAPE);
    /* The gate's activation: tanh (the default) or none. */
    const char* act = r4d_gets(a, "act");
    const int none = act && !std::strcmp(act, "none");
    if (!none && act && std::strcmp(act, "tanh")) return R4D_NO(RAD_E_INVAL);
    hipLaunchKernelGGL(gate_add_kernel, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)y->data, (const uint16_t*)g->data, (const int*)cu->data,
                       (uint16_t*)x->data, (int)n, (int)n_seq, (int)yl, (int)gl, (int)xl, none);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_table_add(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *t = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !t || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(t) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), L = r4d_geti_or(a, "L", 0);
    long long rx = 0, xl = 0, ry = 0, yl = 0;
    if (L <= 0 || !rows8(x, n, &rx, &xl) || !rows8(y, n * L, &ry, &yl) || !r4d_contig(t) ||
        ((uintptr_t)t->data & 15))
        return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || rx < ry || r4d_numel(t) < L * n) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(table_add_kernel, dim3((unsigned)(ry * L)), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (const uint16_t*)t->data, (uint16_t*)y->data,
                       (int)n, (int)L, (int)xl, (int)yl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_mix_rows(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(w) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), k = r4d_geti_or(a, "k", 0);
    long long rx = 0, xl = 0, ry = 0, yl = 0;
    if (k <= 0 || !rows8(x, n, &rx, &xl) || !rows8(y, n, &ry, &yl) || !r4d_contig(w))
        return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || rx < ry * k || r4d_numel(w) < k) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(mix_rows_kernel, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (const uint16_t*)w->data, (uint16_t*)y->data,
                       (int)n, (int)k, (int)xl, (int)yl);
    return st_of(hipGetLastError());
}

template <typename TX, typename TW, typename TY>
static int chan_stats_any(const RadTensor* x, const RadTensor* mn, const RadTensor* sv,
                          const RadTensor* y, long long rows, long long n, long long xl,
                          long long yl, int var, float eps, int norm, RadStream s) {
    hipLaunchKernelGGL((chan_stats_any_kernel<TX, TW, TY>), dim3((unsigned)rows), dim3(kT), 0,
                       (hipStream_t)s, (const TX*)x->data, (const TW*)mn->data,
                       (const TW*)sv->data, (TY*)y->data, (int)n, (int)xl, (int)yl, var, eps,
                       norm);
    return st_of(hipGetLastError());
}

/* Both rows: bf16 (chan_stats_bf16, eight channels a thread when every row and statistic is
 * 16-byte aligned, element-wise otherwise) and f32 (chan_stats_f32: x and y f32 over bf16 or f32
 * statistics, a latent kept in f32 before a bf16 network reads it). */
extern "C" int r4d_rad_chan_stats(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *mn = r4d_opt(a, 1), *sv = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!x || !mn || !sv || !y) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", 0);
    const char* mode = r4d_gets(a, "mode");
    const int norm = mode && !std::strcmp(mode, "norm");
    if (!mode || (!norm && std::strcmp(mode, "denorm"))) return R4D_NO(RAD_E_INVAL);
    auto bf_f32 = [](const RadTensor* t) { return is_bf16(t) || t->dtype == RAD_F32; };
    if (!bf_f32(x) || !bf_f32(y) || !bf_f32(mn) || mn->dtype != sv->dtype || x->dtype != y->dtype)
        return R4D_NO(RAD_E_DTYPE);
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    const int var = (int)r4d_geti_or(a, "var", 0);
    const float eps = (float)r4d_getf(a, "eps", 0.0);
    long long rx = 0, xl = 0, ry = 0, yl = 0;
    if (is_bf16(x) && is_bf16(mn) && rows8(x, n, &rx, &xl) && rows8(y, n, &ry, &yl) &&
        r4d_contig(mn) && r4d_contig(sv) && !((uintptr_t)mn->data & 15) &&
        !((uintptr_t)sv->data & 15)) {
        if (ry <= 0 || rx < ry || r4d_numel(mn) < n || r4d_numel(sv) < n)
            return R4D_NO(RAD_E_SHAPE);
        hipLaunchKernelGGL(chan_stats_kernel, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
                           (const uint16_t*)x->data, (const uint16_t*)mn->data,
                           (const uint16_t*)sv->data, (uint16_t*)y->data, (int)n, (int)xl, (int)yl,
                           var, eps, norm);
        return st_of(hipGetLastError());
    }
    xl = r4d_rows_pitch(x, n, &rx);
    yl = r4d_rows_pitch(y, n, &ry);
    if (xl < 0 || yl < 0 || !r4d_contig(mn) || !r4d_contig(sv)) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || rx < ry || r4d_numel(mn) < n || r4d_numel(sv) < n) return R4D_NO(RAD_E_SHAPE);
    const bool f = x->dtype == RAD_F32, wf = mn->dtype == RAD_F32;
#define R4D_CS(TX, TW) chan_stats_any<TX, TW, TX>(x, mn, sv, y, ry, n, xl, yl, var, eps, norm, s)
    if (f) return wf ? R4D_CS(float, float) : R4D_CS(float, uint16_t);
    return wf ? R4D_CS(uint16_t, float) : R4D_CS(uint16_t, uint16_t);
#undef R4D_CS
}

/* Both rows: a bf16 embedding (timestep_embed_bf16) or an f32 one (timestep_embed_f32, a timestep
 * MLP kept in f32); with `curve`, the table form. */
extern "C" int r4d_rad_timestep_embed(const RadArgs* a, RadStream s) {
    const RadTensor *t = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!t || !y) return R4D_NO(RAD_E_INVAL);
    if (t->dtype != RAD_F32 || !r4d_contig(t) || (!is_bf16(y) && y->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    const long long grid = r4d_geti_or(a, "curve", 0);
    if (n <= 0 || (!grid && (n & 1))) return R4D_NO(RAD_E_SHAPE);
    long long ry = 0;
    const long long yl = r4d_rows_pitch(y, n, &ry);
    if (yl < 0) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || r4d_numel(t) < ry) return R4D_NO(RAD_E_SHAPE);
    if (grid > 0) {
        const RadTensor* tab = r4d_opt(a, 2);
        if (!tab) return R4D_NO(RAD_E_INVAL);
        if (tab->dtype != RAD_F32 || !r4d_contig(tab) || r4d_numel(tab) < grid * n)
            return R4D_NO(RAD_E_SHAPE);
        const float clamp = (float)r4d_getf(a, "clamp_min", 0.0);
        if (y->dtype == RAD_F32)
            hipLaunchKernelGGL(timestep_curve_kernel<float>, dim3((unsigned)ry), dim3(kT), 0,
                               (hipStream_t)s, (const float*)t->data, (const float*)tab->data,
                               (float*)y->data, (int)grid, (int)n, clamp, (int)yl);
        else
            hipLaunchKernelGGL(timestep_curve_kernel<uint16_t>, dim3((unsigned)ry), dim3(kT), 0,
                               (hipStream_t)s, (const float*)t->data, (const float*)tab->data,
                               (uint16_t*)y->data, (int)grid, (int)n, clamp, (int)yl);
        return st_of(hipGetLastError());
    }
    const float mp = (float)r4d_getf(a, "max_period", 10000.0);
    const float coef = (float)(-std::log((double)mp));
    const float scale = (float)r4d_getf(a, "scale", 1000.0);
    const int round = (int)r4d_geti_or(a, "round", 0);
    const float clamp = (float)r4d_getf(a, "clamp_min", 0.0);
    if (y->dtype == RAD_F32)
        hipLaunchKernelGGL(timestep_kernel<float>, dim3((unsigned)ry), dim3(kT), 0,
                           (hipStream_t)s, (const float*)t->data, (float*)y->data, (int)(n / 2),
                           scale, coef, round, clamp, (int)yl);
    else
        hipLaunchKernelGGL(timestep_kernel<uint16_t>, dim3((unsigned)ry), dim3(kT), 0,
                           (hipStream_t)s, (const float*)t->data, (uint16_t*)y->data, (int)(n / 2),
                           scale, coef, round, clamp, (int)yl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_gather_rows2(const RadArgs* a, RadStream s) {
    const RadTensor *A = r4d_opt(a, 0), *B = r4d_opt(a, 1), *idx = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!A || !B || !idx || !y) return R4D_NO(RAD_E_INVAL);
    if (A->dtype != y->dtype || B->dtype != y->dtype || rad_dtype_bytes(y->dtype, 1) != 2 ||
        idx->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long ra = 0, al = 0, rb = 0, bl = 0, ry = 0, yl = 0;
    if (!rows8(A, n, &ra, &al) || !rows8(B, n, &rb, &bl) || !rows8(y, n, &ry, &yl) ||
        !r4d_contig(idx))
        return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || r4d_numel(idx) < ry) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(gather2_kernel, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)A->data, (const uint16_t*)B->data, (const int*)idx->data,
                       (uint16_t*)y->data, (int)n, (int)ra, (int)rb, (int)al, (int)bl, (int)yl);
    return st_of(hipGetLastError());
}

// A lane operand's row in 32-bit words, and its pitch in words.
static bool lane_rows(const RadTensor* t, long long n, long long* rows, long long* ldw,
                      long long* words) {
    if (!t || n <= 0) return false;
    const long long eb = rad_dtype_bytes(t->dtype, 1);
    if (eb <= 0 || (n * eb) % 4) return false;
    const long long ld = r4d_rows_pitch(t, n, rows);
    if (ld < 0 || (ld * eb) % 4 || ((uintptr_t)t->data & 3)) return false;
    *ldw = ld * eb / 4;
    *words = n * eb / 4;
    return true;
}

extern "C" int r4d_rad_lane_gather(const RadArgs* a, RadStream s) {
    const RadTensor *src = r4d_opt(a, 0), *lane = r4d_opt(a, 1), *cu = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!src || !lane || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (src->dtype != y->dtype) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), stride = r4d_geti_or(a, "stride", 0);
    long long rs = 0, sw = 0, ry = 0, yw = 0, words = 0;
    if (!lane_rows(src, n, &rs, &sw, &words) || !lane_rows(y, n, &ry, &yw, &words) ||
        !r4d_contig(lane) || !r4d_contig(cu))
        return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (stride <= 0 || n_seq <= 0 || r4d_numel(lane) < n_seq || ry <= 0) return R4D_NO(RAD_E_SHAPE);
    /* Every row from cu[0] up is an entry's, and the operand holds them all. */
    const long long total = ry * words;
    hipLaunchKernelGGL(lane_gather_kernel, dim3(blocks_for(total)), dim3(kT), 0, (hipStream_t)s,
                       (const uint32_t*)src->data, (const int*)lane->data, (const int*)cu->data,
                       (uint32_t*)y->data, (int)words, (int)n_seq, (int)stride, (int)sw, (int)yw,
                       (int64_t)total);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_lane_store(const RadArgs* a, RadStream s) {
    const RadTensor *src = r4d_opt(a, 0), *lane = r4d_opt(a, 1), *dst = r4d_opt(a, 2);
    if (!src || !lane || !dst) return R4D_NO(RAD_E_INVAL);
    if (src->dtype != dst->dtype) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), stride = r4d_geti_or(a, "stride", 0);
    long long rs = 0, sw = 0, rd = 0, dw = 0, words = 0;
    if (!lane_rows(src, n, &rs, &sw, &words) || !lane_rows(dst, n, &rd, &dw, &words) ||
        !r4d_contig(lane))
        return R4D_NO(RAD_E_STRIDE);
    const long long nl = r4d_numel(lane);
    if (rs <= 0 || rs > stride || nl <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long total = rs * words;
    hipLaunchKernelGGL(lane_store_kernel, dim3(blocks_for(total), (unsigned)nl), dim3(kT), 0,
                       (hipStream_t)s, (const uint32_t*)src->data, (const int*)lane->data,
                       (uint32_t*)dst->data, (int)words, (int)stride, (int)sw, (int)dw,
                       (int64_t)total);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_randn_lanes(const RadArgs* a, RadStream s) {
    const RadTensor *fl = r4d_opt(a, 0), *seed = r4d_opt(a, 1), *lane = r4d_opt(a, 2);
    const RadTensor *cu = r4d_opt(a, 3), *dst = r4d_opt(a, 4);
    if (!fl || !seed || !lane || !cu || !dst) return R4D_NO(RAD_E_INVAL);
    /* bf16 lanes (randn_lanes_bf16) or f32 ones (randn_lanes_f32). */
    if (!is_bf16(dst) && dst->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), stride = r4d_geti_or(a, "stride", 0);
    long long rd = 0;
    const long long dl = r4d_rows_pitch(dst, n, &rd);
    if (dl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (n <= 0 || stride <= 0 || n_seq <= 0 || rd < stride) return R4D_NO(RAD_E_SHAPE);
    /* One flag and lane an entry, two seed words. A lane's rows (device data) are not checked. */
    if (r4d_numel(fl) < n_seq || r4d_numel(lane) < n_seq || r4d_numel(seed) < 2 * n_seq)
        return R4D_NO(RAD_E_SHAPE);
    const dim3 grid(blocks_for(stride * n), (unsigned)n_seq);
    if (dst->dtype == RAD_F32)
        hipLaunchKernelGGL(randn_lanes_kernel<float>, grid, dim3(kT), 0, (hipStream_t)s,
                           (const int*)fl->data, (const uint32_t*)seed->data,
                           (const int*)lane->data, (const int*)cu->data, (float*)dst->data, (int)n,
                           (int)stride, (int)dl);
    else
        hipLaunchKernelGGL(randn_lanes_kernel<uint16_t>, grid, dim3(kT), 0, (hipStream_t)s,
                           (const int*)fl->data, (const uint32_t*)seed->data,
                           (const int*)lane->data, (const int*)cu->data, (uint16_t*)dst->data,
                           (int)n, (int)stride, (int)dl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_flow_euler(const RadArgs* a, RadStream s) {
    const RadTensor *v = r4d_opt(a, 0), *sg = r4d_opt(a, 1), *sgn = r4d_opt(a, 2), *cfg = r4d_opt(a, 3);
    const RadTensor *pair = r4d_opt(a, 4), *lane = r4d_opt(a, 5), *fl = r4d_opt(a, 6);
    const RadTensor *seed = r4d_opt(a, 7), *cu = r4d_opt(a, 8), *inp = r4d_opt(a, 9);
    const RadTensor *mask = r4d_opt(a, 10), *st = r4d_opt(a, 11), *pv = r4d_opt(a, 12);
    if (!v || !sg || !sgn || !cfg || !pair || !lane || !fl || !seed || !cu || !st)
        return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(v) || !is_bf16(st) || (pv && !is_bf16(pv)) || (inp && !is_bf16(inp)) ||
        (mask && mask->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), stride = r4d_geti_or(a, "stride", 0);
    long long rv = 0, rst = 0, rp = 0, ri = 0;
    const long long vl = r4d_rows_pitch(v, n, &rv), sl = r4d_rows_pitch(st, n, &rst);
    const long long pl = pv ? r4d_rows_pitch(pv, n, &rp) : 0;
    const long long il = inp ? r4d_rows_pitch(inp, n, &ri) : 0;
    if (vl < 0 || sl < 0 || pl < 0 || il < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (n <= 0 || stride <= 0 || n_seq <= 0 || rst < stride) return R4D_NO(RAD_E_SHAPE);
    /* One scalar of each kind an entry, two seed words; inp, mask and preview are addressed by the
     * state's rows. A lane's rows and a pair's entry (device data) are not checked. */
    for (const RadTensor* e : { sg, sgn, cfg, pair, lane, fl })
        if (r4d_numel(e) < n_seq) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(seed) < 2 * n_seq || (pv && rp < rst) || (inp && ri < rst) ||
        (mask && r4d_numel(mask) < rst))
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(flow_euler_kernel, dim3(blocks_for(stride * n), (unsigned)n_seq), dim3(kT), 0,
                       (hipStream_t)s, (const uint16_t*)v->data, (const float*)sg->data,
                       (const float*)sgn->data, (const float*)cfg->data, (const int*)pair->data,
                       (const int*)lane->data, (const int*)fl->data, (const uint32_t*)seed->data,
                       (const int*)cu->data, inp ? (const uint16_t*)inp->data : nullptr,
                       mask ? (const float*)mask->data : nullptr, (uint16_t*)st->data,
                       pv ? (uint16_t*)pv->data : nullptr, (int)n, (int)stride, (int)vl, (int)il,
                       (int)sl, (int)pl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_kd_input(const RadArgs* a, RadStream s) {
    const RadTensor *sg = r4d_opt(a, 0), *pair = r4d_opt(a, 1), *lane = r4d_opt(a, 2);
    const RadTensor *fl = r4d_opt(a, 3), *seed = r4d_opt(a, 4), *cu = r4d_opt(a, 5);
    const RadTensor *st = r4d_opt(a, 6), *y = r4d_opt(a, 7);
    if (!sg || !pair || !lane || !fl || !seed || !cu || !st || !y) return R4D_NO(RAD_E_INVAL);
    if (st->dtype != RAD_F32 || !is_bf16(y) || sg->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), stride = r4d_geti_or(a, "stride", 0);
    long long rs = 0, ry = 0;
    const long long sl = r4d_rows_pitch(st, n, &rs), yl = r4d_rows_pitch(y, n, &ry);
    if (sl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (n <= 0 || stride <= 0 || n_seq <= 0 || rs < stride) return R4D_NO(RAD_E_SHAPE);
    /* One scalar of each kind an entry, two seed words. A lane's rows, a pair's entry and a
     * repaint flag (device data) are not checked. */
    for (const RadTensor* e : { sg, pair, lane, fl })
        if (r4d_numel(e) < n_seq) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(seed) < 2 * n_seq) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(kd_input_kernel, dim3(blocks_for(stride * n), (unsigned)n_seq), dim3(kT), 0,
                       (hipStream_t)s, (const float*)sg->data, (const int*)pair->data,
                       (const int*)lane->data, (const int*)fl->data, (const uint32_t*)seed->data,
                       (const int*)cu->data, (float*)st->data, (uint16_t*)y->data, (int)n,
                       (int)stride, (int)sl, (int)yl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_kd_step(const RadArgs* a, RadStream s) {
    const RadTensor *o = r4d_opt(a, 0), *sg = r4d_opt(a, 1), *sgn = r4d_opt(a, 2);
    const RadTensor *sgp = r4d_opt(a, 3), *cfg = r4d_opt(a, 4), *pair = r4d_opt(a, 5);
    const RadTensor *lane = r4d_opt(a, 6), *fl = r4d_opt(a, 7), *seed = r4d_opt(a, 8);
    const RadTensor *smp = r4d_opt(a, 9), *stp = r4d_opt(a, 10), *cu = r4d_opt(a, 11);
    const RadTensor *st = r4d_opt(a, 12), *hs = r4d_opt(a, 13), *pv = r4d_opt(a, 14);
    if (!o || !sg || !sgn || !sgp || !cfg || !pair || !lane || !fl || !seed || !smp || !stp || !cu ||
        !st || !hs)
        return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(o) || st->dtype != RAD_F32 || hs->dtype != RAD_F32 ||
        (pv && pv->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    const char* pred = r4d_gets(a, "pred");
    const int v_pred = pred && !std::strcmp(pred, "v");
    if (!v_pred && (!pred || std::strcmp(pred, "eps"))) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", 0), stride = r4d_geti_or(a, "stride", 0);
    long long ro = 0, rs = 0, rh = 0, rp = 0;
    const long long ol = r4d_rows_pitch(o, n, &ro), sl = r4d_rows_pitch(st, n, &rs);
    const long long hl = r4d_rows_pitch(hs, n, &rh), pl = pv ? r4d_rows_pitch(pv, n, &rp) : 0;
    if (ol < 0 || sl < 0 || hl < 0 || pl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (n <= 0 || stride <= 0 || n_seq <= 0 || rs < stride || rh < rs || (pv && rp < rs))
        return R4D_NO(RAD_E_SHAPE);
    /* One scalar of each kind an entry, two seed words. A lane's rows, a pair's entry, the
     * sampler ids and the flags (device data) are not checked here: the kernel leaves the lane of
     * an entry whose sampler it does not know, and writes no preview without the operand, where
     * libref refuses both. */
    for (const RadTensor* e : { sg, sgn, sgp, cfg, pair, lane, fl, smp, stp })
        if (r4d_numel(e) < n_seq) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(seed) < 2 * n_seq) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(kd_step_kernel, dim3(blocks_for(stride * n), (unsigned)n_seq), dim3(kT), 0,
                       (hipStream_t)s, (const uint16_t*)o->data, (const float*)sg->data,
                       (const float*)sgn->data, (const float*)sgp->data, (const float*)cfg->data,
                       (const int*)pair->data, (const int*)lane->data, (const int*)fl->data,
                       (const uint32_t*)seed->data, (const int*)smp->data, (const int*)stp->data,
                       (const int*)cu->data, (float*)st->data, (float*)hs->data,
                       pv ? (float*)pv->data : nullptr, (int)n, (int)stride, (int)ol, (int)sl,
                       (int)hl, (int)pl, v_pred);
    return st_of(hipGetLastError());
}

/* Both rows: x and y bf16 (chan_affine_bf16) or both f32 (chan_affine_f32). `round` 0 keeps the
 * lists and the arithmetic in f32. */
extern "C" int r4d_rad_chan_affine(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    const bool f32 = y->dtype == RAD_F32;
    if (x->dtype != y->dtype || (!f32 && !is_bf16(y))) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), rnd = r4d_geti_or(a, "round", 1);
    if (rnd < 0 || rnd > 1) return R4D_NO(RAD_E_INVAL);
    long long rx = 0;
    const long long xl = n > 0 ? r4d_rows_pitch(x, n, &rx) : -1;
    if (xl < 0 || !r4d_contig(y)) return R4D_NO(RAD_E_STRIDE);
    ChanLists l{};
    if (n <= 0 || n > kChanMax ||
        parse_bf16_list(r4d_gets(a, "mul"), l.mul, kChanMax, rnd != 0) != n ||
        parse_bf16_list(r4d_gets(a, "add"), l.add, kChanMax, rnd != 0) != n)
        return R4D_NO(RAD_E_SHAPE);
    const char* mode = r4d_gets(a, "mode");
    const int inv = mode && !std::strcmp(mode, "inv");
    if (!inv && mode && std::strcmp(mode, "fma")) return R4D_NO(RAD_E_INVAL);
    const long long total = r4d_numel(y);
    if (total <= 0 || total % n || rx * n < total) return R4D_NO(RAD_E_SHAPE);
    if (f32)
        hipLaunchKernelGGL(chan_affine_kernel<float>, dim3(blocks_for(total)), dim3(kT), 0,
                           (hipStream_t)s, (const float*)x->data, (float*)y->data, (int64_t)total,
                           (int)n, (int)xl, inv, (int)rnd, l);
    else
        hipLaunchKernelGGL(chan_affine_kernel<uint16_t>, dim3(blocks_for(total)), dim3(kT), 0,
                           (hipStream_t)s, (const uint16_t*)x->data, (uint16_t*)y->data,
                           (int64_t)total, (int)n, (int)xl, inv, (int)rnd, l);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_conv2d_nhwc(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *b = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!x || !w || !b || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(w) || !is_bf16(b) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long cin = r4d_geti_or(a, "cin", 0), cout = r4d_geti_or(a, "cout", 0);
    const long long k = r4d_geti_or(a, "k", 0), stride = r4d_geti_or(a, "stride", 1);
    const long long up = r4d_geti_or(a, "up", 1);
    const char* pad = r4d_gets(a, "pad");
    const bool down = pad && !std::strcmp(pad, "down");
    const bool causal = pad && !std::strcmp(pad, "causal");
    if (pad && !down && !causal && std::strcmp(pad, "same")) return R4D_NO(RAD_E_INVAL);
    if (cin <= 0 || cout <= 0 || k <= 0 || stride <= 0 || up <= 0) return R4D_NO(RAD_E_SHAPE);
    long long Hin = 0, Win = 0, Ho = 0, Wo = 0;
    /* y may be a column slice of wider rows (a channel written into its own columns): both kernels
     * store at its row pitch. */
    if (!image_of(x, cin, &Hin, &Win) || !image_of(y, cout, &Ho, &Wo, true))
        return R4D_NO(RAD_E_SHAPE);
    const long long Hv = Hin * up, Wv = Win * up;
    /* Rows padded above `pt` and below `pb`, columns `pl` left and `pr` right. */
    const long long pt = down ? 0 : causal ? k - 1 : k / 2, pb = down ? 1 : causal ? 0 : k / 2;
    const long long pl = down ? 0 : k / 2, pr = down ? 1 : k / 2;
    if ((Hv + pt + pb - k) / stride + 1 != Ho || (Wv + pl + pr - k) / stride + 1 != Wo)
        return R4D_NO(RAD_E_SHAPE);
    const long long K = k * k * cin;
    if (r4d_numel(w) < cout * K || r4d_numel(b) < cout) return R4D_NO(RAD_E_SHAPE);
    const long long Mpx = Ho * Wo;
    if (Mpx <= 0 || Mpx >= (1ll << 31) || (long long)Ho * Wo * cout >= (1ll << 40))
        return R4D_NO(RAD_E_SHAPE);
    hipStream_t st = (hipStream_t)s;
    if (cin % 8 || ((uintptr_t)x->data & 15) || ((uintptr_t)w->data & 15)) {
        const long long total = Mpx * cout;
        hipLaunchKernelGGL(conv_direct_kernel, dim3(blocks_for(total)), dim3(kT), 0, st,
                           (const uint16_t*)x->data, (const uint16_t*)w->data,
                           (const uint16_t*)b->data, (uint16_t*)y->data, (int)Hv, (int)Wv,
                           (int)cin, (int)Wo, (int)cout, (int)k, (int)stride, (int)up, (int)pt,
                           (int)pl, (int)x->stride[0], (int)y->stride[0], (int64_t)total);
        return st_of(hipGetLastError());
    }
    // 128 pixels a block, and the widest channel tile (192, 128, then 64) that leaves under a third
    // of its last block idle. A wider tile does two or three times the WMMA per K step for the same
    // two barriers, which pays even with the last block partly empty: on Qwen-Image's decoder the
    // 192 tile is 1.3-1.55x faster than the 64 tile at 144, 288 and 576 channels. 96 channels
    // take 128.
    auto fits = [cout](long long bc) {
        const long long padded = (cout + bc - 1) / bc * bc;
        return (padded - cout) * 3 < padded;
    };
#define R4D_CONV(TP, TC, KS, UP)                                                                 \
    hipLaunchKernelGGL((conv_igemm_kernel<TP, TC, KS, UP>),                                      \
                       dim3((unsigned)((Mpx + 2 * TP * 16 - 1) / (2 * TP * 16)),                  \
                            (unsigned)((cout + 4 * TC * 16 - 1) / (4 * TC * 16))),               \
                       dim3(256), 0, st, (const uint16_t*)x->data, (const uint16_t*)w->data,     \
                       (const uint16_t*)b->data, (uint16_t*)y->data, (int)Hv, (int)Wv, (int)cin, \
                       (int)Wo, (int)cout, (int)k, (int)stride, (int)up, (int)pt, (int)pl,       \
                       (int)x->stride[0], (int)y->stride[0], (int)Mpx, (int)K)
    /* The decoder's two kernel sizes at its two sampling rates, and the generic form for the rest. */
#define R4D_CONV_K(TP, TC)                                                                       \
    if (k == 3 && up == 1)      R4D_CONV(TP, TC, 3, 1);                                          \
    else if (k == 3 && up == 2) R4D_CONV(TP, TC, 3, 2);                                          \
    else if (k == 1 && up == 1) R4D_CONV(TP, TC, 1, 1);                                          \
    else                        R4D_CONV(TP, TC, 0, 0)
    if (fits(192))      { R4D_CONV_K(4, 3); }
    else if (fits(128)) { R4D_CONV_K(4, 2); }
    else                { R4D_CONV_K(4, 1); }
#undef R4D_CONV_K
#undef R4D_CONV
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_im2col_nhwc(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long cin = r4d_geti_or(a, "cin", 0), k = r4d_geti_or(a, "k", 0);
    const long long stride = r4d_geti_or(a, "stride", 1), up = r4d_geti_or(a, "up", 1);
    const char* pad = r4d_gets(a, "pad");
    const bool down = pad && !std::strcmp(pad, "down");
    const bool causal = pad && !std::strcmp(pad, "causal");
    if (pad && !down && !causal && std::strcmp(pad, "same")) return R4D_NO(RAD_E_INVAL);
    if (cin <= 0 || k <= 0 || k > 7 || stride <= 0 || stride > 2 || up <= 0 || up > 2)
        return R4D_NO(RAD_E_SHAPE);
    long long Hin = 0, Win = 0;
    if (!image_of(x, cin, &Hin, &Win, true)) return R4D_NO(RAD_E_SHAPE);
    const long long Hv = Hin * up, Wv = Win * up;
    const long long pt = down ? 0 : causal ? k - 1 : k / 2, pb = down ? 1 : causal ? 0 : k / 2;
    const long long pl = down ? 0 : k / 2, pr = down ? 1 : k / 2;
    const long long Ho = (Hv + pt + pb - k) / stride + 1, Wo = (Wv + pl + pr - k) / stride + 1;
    const long long KK = cin * k * k;
    long long rows = 0;
    const long long y_ld = r4d_rows_pitch(y, KK, &rows);
    if (y_ld < 0 || y->stride[y->rank - 1] != 1) return R4D_NO(RAD_E_STRIDE);
    if (Ho <= 0 || Wo <= 0 || rows != Ho * Wo || Ho > 65535 || Ho * Wo * y_ld >= (1ll << 40))
        return R4D_NO(RAD_E_SHAPE);
    /* 16-byte channel loads where every chunk's channels start on one; the widest store every
     * chunk's run, the row pitch and the base allow. */
    const long long tail = cin % IC_CC ? cin % IC_CC : IC_CC;
    const int vec = cin % 8 == 0 && x->stride[0] % 8 == 0 && !((uintptr_t)x->data & 15);
    auto fits = [&](long long vw) {
        return (std::min((long long)IC_CC, cin) * k * k) % vw == 0 && (tail * k * k) % vw == 0 &&
               y_ld % vw == 0 && !((uintptr_t)y->data & (2 * vw - 1));
    };
    const int VW = fits(8) ? 8 : fits(4) ? 4 : fits(2) ? 2 : 1;
    const long long W = (IC_TW - 1) * stride + k;
    const size_t lds = (size_t)(std::min((long long)IC_CC, cin) * k * W) * sizeof(uint16_t);
    const dim3 grid((unsigned)((Wo + IC_TW - 1) / IC_TW), (unsigned)Ho,
                    (unsigned)((cin + IC_CC - 1) / IC_CC));
#define R4D_IC(KS, V)                                                                            \
    hipLaunchKernelGGL((im2col_nhwc_kernel<KS, V>), grid, dim3(kT), lds, (hipStream_t)s,         \
                       (const uint16_t*)x->data, (uint16_t*)y->data, (int)Hv, (int)Wv, (int)cin, \
                       (int)Wo, (int)k, (int)stride, (int)up, (int)pt, (int)pl,                  \
                       (int64_t)x->stride[0], (int64_t)y_ld, vec)
#define R4D_IC_K(V)                                                                              \
    if (k == 3)      R4D_IC(3, V);                                                               \
    else if (k == 1) R4D_IC(1, V);                                                               \
    else             R4D_IC(0, V)
    if (VW == 8)      { R4D_IC_K(8); }
    else if (VW == 4) { R4D_IC_K(4); }
    else if (VW == 2) { R4D_IC_K(2); }
    else              { R4D_IC_K(1); }
#undef R4D_IC_K
#undef R4D_IC
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_l2norm_chan(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *g = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !g || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(g) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    const char* act = r4d_gets(a, "act");
    const int silu = act && !std::strcmp(act, "silu");
    if (!silu && act && std::strcmp(act, "none")) return R4D_NO(RAD_E_INVAL);
    if (n <= 0 || (n & 7) || r4d_numel(g) < n) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(x) || !r4d_contig(y) || ((uintptr_t)x->data & 15) || ((uintptr_t)y->data & 15) ||
        ((uintptr_t)g->data & 15))
        return R4D_NO(RAD_E_STRIDE);
    const long long rows = r4d_numel(y) / n;
    if (rows <= 0 || r4d_numel(x) < rows * n) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(l2norm_kernel, dim3((unsigned)((rows + kT / 32 - 1) / (kT / 32))), dim3(kT),
                       0, (hipStream_t)s, (const uint16_t*)x->data, (const uint16_t*)g->data,
                       (uint16_t*)y->data, (int64_t)rows, (int)n,
                       (float)std::sqrt((double)n), silu);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_d2s_add(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long cin = r4d_geti_or(a, "cin", 0), cout = r4d_geti_or(a, "cout", 0);
    const long long ft = r4d_geti_or(a, "ft", 1);
    long long H = 0, W = 0, H2 = 0, W2 = 0;
    if ((ft != 1 && ft != 2) || !image_of(x, cin, &H, &W) || !image_of(y, cout, &H2, &W2) ||
        H2 != 2 * H || W2 != 2 * W)
        return R4D_NO(RAD_E_SHAPE);
    const long long F = 4 * ft;
    if ((cout * F) % cin) return R4D_NO(RAD_E_SHAPE);
    const long long total = 4 * H * W * cout;
    hipLaunchKernelGGL(d2s_add_kernel, dim3(blocks_for(total)), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (uint16_t*)y->data, (int)H, (int)W, (int)cin,
                       (int)cout, (int)F, ft == 2 ? 4 : 0, (int)(cout * F / cin));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_s2d_avg_add(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long cin = r4d_geti_or(a, "cin", 0), cout = r4d_geti_or(a, "cout", 0);
    const long long ft = r4d_geti_or(a, "ft", 1);
    long long H = 0, W = 0, H2 = 0, W2 = 0;
    if ((ft != 1 && ft != 2) || !image_of(y, cout, &H, &W) || !image_of(x, cin, &H2, &W2) ||
        H2 != 2 * H || W2 != 2 * W)
        return R4D_NO(RAD_E_SHAPE);
    const long long F = 4 * ft;
    if ((cin * F) % cout) return R4D_NO(RAD_E_SHAPE);
    const long long total = H * W * cout;
    hipLaunchKernelGGL(s2d_avg_add_kernel, dim3(blocks_for(total)), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (uint16_t*)y->data, (int)H, (int)W, (int)cin,
                       (int)cout, (int)F, (int)(cin * F / cout), (int)ft);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_to_rgba8(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || y->dtype != RAD_U8) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(x) || !r4d_contig(y)) return R4D_NO(RAD_E_STRIDE);
    const long long total = r4d_numel(y);
    if (total <= 0 || r4d_numel(x) < total) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(to_rgba8_kernel, dim3(blocks_for(total)), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (uint8_t*)y->data, (int64_t)total);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_gelu_quick(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    const bool f32 = x->dtype == RAD_F32;
    if ((!is_bf16(x) && !f32) || y->dtype != x->dtype) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long rx = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, n, &rx), yl = r4d_rows_pitch(y, n, &ry);
    if (n <= 0 || xl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || rx < ry) return R4D_NO(RAD_E_SHAPE);
    if (f32)
        hipLaunchKernelGGL(gelu_quick_kernel<float>, dim3((unsigned)ry), dim3(kT), 0,
                           (hipStream_t)s, (const float*)x->data, (float*)y->data, (int)n, (int)xl,
                           (int)yl);
    else
        hipLaunchKernelGGL(gelu_quick_kernel<uint16_t>, dim3((unsigned)ry), dim3(kT), 0,
                           (hipStream_t)s, (const uint16_t*)x->data, (uint16_t*)y->data, (int)n,
                           (int)xl, (int)yl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_gelu_tanh(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long rx = 0, xl = 0, ry = 0, yl = 0;
    if (!rows8(x, n, &rx, &xl) || !rows8(y, n, &ry, &yl)) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || rx < ry) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(gelu_tanh_kernel, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (uint16_t*)y->data, (int)n, (int)xl, (int)yl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_scatter_add_rows(const RadArgs* a, RadStream s) {
    const RadTensor *v = r4d_opt(a, 0), *idx = r4d_opt(a, 1), *x = r4d_opt(a, 2);
    if (!v || !idx || !x) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(v) || !is_bf16(x) || idx->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long rv = 0, vl = 0, rx = 0, xl = 0;
    if (!rows8(v, n, &rv, &vl) || !rows8(x, n, &rx, &xl) || !r4d_contig(idx))
        return R4D_NO(RAD_E_STRIDE);
    if (rv <= 0 || r4d_numel(idx) < rv) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(scatter_add_kernel, dim3((unsigned)rv), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)v->data, (const int*)idx->data, (uint16_t*)x->data, (int)n,
                       (int)vl, (int)xl);
    return st_of(hipGetLastError());
}
