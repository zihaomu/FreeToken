// R4D -- the audio generator's ops (docs/OPS.md, "Audio generation"): a vocoder's 1-D
// convolutions and snake activation, the guided top-k sampler a frame's codes are drawn with, a
// frame's residual-code embedding and conditioning row, and the window ops of a denoiser that
// runs over overlapping runs of frames.
//
// Each kernel rounds where its libref function does (libref/ref_audio.cpp). Only reduction order
// differs (a WMMA tile here, a sequential sum there), so a sum can land one bf16 step apart.
//
// Sequenced operands follow every sequenced op here: `cu_seqlens` holds row indices of the
// operands themselves, and each sequence is padded on its own.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_f32_tile.h"
#include "r4d_gdn_wmma.h"
#include "rad_philox.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

__global__ void elu_f32_kernel(const float* x, float* y, int rows, int n, int xp, int yp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const float v = x[(size_t)(i / n) * xp + i % n];
    y[(size_t)(i / n) * yp + i % n] = v >= 0 ? v : expm1f(v);
}

/* One workgroup a frame, one thread a candidate at a time. Candidate ties reduce to the
 * lower index. The projected residual is only updated after every candidate has read it. */
__global__ void vq_euclidean_kernel(float* x, const float* cb, int* codes, int n, int bins,
                                   int xp, int cp, int cs) {
    const int r = blockIdx.x, t = threadIdx.x;
    __shared__ float dist[256];
    __shared__ int pick[256];
    float best = INFINITY;
    int id = 0;
    for (int k = t; k < bins; k += 256) {
        float d = 0;
        for (int j = 0; j < n; ++j) {
            const float delta = x[(size_t)r * xp + j] - cb[(size_t)k * cp + j];
            d += delta * delta;
        }
        if (d < best) { best = d; id = k; }
    }
    dist[t] = best; pick[t] = id;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s && (dist[t + s] < dist[t] ||
                      (dist[t + s] == dist[t] && pick[t + s] < pick[t]))) {
            dist[t] = dist[t + s]; pick[t] = pick[t + s];
        }
        __syncthreads();
    }
    if (!t) codes[(size_t)r * cs] = pick[0];
    for (int j = t; j < n; j += 256)
        x[(size_t)r * xp + j] -= cb[(size_t)pick[0] * cp + j];
}

constexpr int kT = 256;

/* Small f32 encoder primitives. The generator remains bf16; these keep the released
 * reference tokenizer in f32 instead of routing its transformer through a host fallback. */
__device__ float encoder_sum(float v, float* buf) {
    const int t = threadIdx.x;
    buf[t] = v;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s) buf[t] += buf[t + s];
        __syncthreads();
    }
    return buf[0];
}

__global__ void encoder_layernorm(const float* x, const float* w, const float* bias,
                                 float* y, int n, int xp, int yp, float eps) {
    __shared__ float buf[256];
    const int r = blockIdx.x, t = threadIdx.x;
    float v = 0;
    for (int j = t; j < n; j += 256) v += x[(size_t)r * xp + j];
    const float mean = encoder_sum(v, buf) / n;
    __syncthreads();
    v = 0;
    for (int j = t; j < n; j += 256) {
        const float d = x[(size_t)r * xp + j] - mean;
        v += d * d;
    }
    const float inv = rsqrtf(encoder_sum(v, buf) / n + eps);
    __syncthreads();
    for (int j = t; j < n; j += 256)
        y[(size_t)r * yp + j] = (x[(size_t)r * xp + j] - mean) * inv * w[j] + (bias ? bias[j] : 0);
}

__global__ void encoder_gelu(const float* x, float* y, int rows, int n, int xp, int yp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const float v = x[(size_t)(i / n) * xp + i % n];
    y[(size_t)(i / n) * yp + i % n] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
}

__global__ void encoder_rope(float* x, const int* positions, int heads, int pitch,
                            float theta, float scale) {
    const int r = blockIdx.x;
    for (int k = threadIdx.x; k < heads * 32; k += 256) {
        const int h = k / 32, j = k % 32;
        float* row = x + (size_t)r * pitch + h * 64;
        const float angle = positions[r] / scale / powf(theta, (2.0f * j) / 64.0f);
        float sn, cs;
        sincosf(angle, &sn, &cs);
        const float a = row[j], b = row[j + 32];
        row[j] = a * cs - b * sn; row[j + 32] = a * sn + b * cs;
    }
}

/* One workgroup a (query, head), <=512 keys. Only the reference encoder uses this f32
 * attention path; its sequence is at most 250 rows, so no quadratic global score plane. */
__global__ void encoder_attn(const float* q, const float* k, const float* v, const int* cu,
                            float* out, int heads, int qp, int kp, int vp, int op,
                            int nseq, int div, int window, float scale) {
    __shared__ float score[512], reduce[256];
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    int seq = 0;
    while (seq + 1 < nseq && cu[seq + 1] <= r * div) ++seq;
    const int first = max(cu[seq] / div, window > 0 ? r - window + 1 : 0);
    const int last = min(cu[seq + 1] / div, r + 1), count = last - first;
    if (count <= 0 || count > 512) return;
    float best = -INFINITY;
    for (int j = t; j < count; j += 256) {
        float dot = 0;
        for (int d = 0; d < 64; ++d)
            dot += q[(size_t)r * qp + h * 64 + d] * k[(size_t)(first + j) * kp + h * 64 + d];
        score[j] = dot * scale;
        best = fmaxf(best, score[j]);
    }
    reduce[t] = best;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s) reduce[t] = fmaxf(reduce[t], reduce[t + s]);
        __syncthreads();
    }
    best = reduce[0];
    __syncthreads();
    float sum = 0;
    for (int j = t; j < count; j += 256) { score[j] = expf(score[j] - best); sum += score[j]; }
    const float denom = encoder_sum(sum, reduce);
    __syncthreads();
    if (t < 64) {
        float result = 0;
        for (int j = 0; j < count; ++j)
            result += score[j] / denom * v[(size_t)(first + j) * vp + h * 64 + t];
        out[(size_t)r * op + h * 64 + t] = result;
    }
}

// A window entry's flags (RadAudioGenBatch, RAD_AGEN_F_*), as libref restates them.
enum { kFirst = 1, kLast = 8, kJoin = 16 };

__device__ __forceinline__ float bfr(float v) { return bf2f(f32_to_bf16(v)); }

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

// The sequence that owns row r: linear, because a vocoder pass has two (a stereo pair) and a
// window pass a handful.
__device__ __forceinline__ int seq_of(const int* cu, int n_seq, int r) {
    int s = 0;
    while (s + 1 < n_seq && cu[s + 1] <= r) ++s;
    return s;
}

__device__ __forceinline__ uint64_t seed_of(const uint32_t* seed, int e) {
    return (uint64_t)seed[2 * e] | ((uint64_t)seed[2 * e + 1] << 32);
}

// ---------------------------------------------------------------------------- conv1d
// An implicit GEMM, r4d_gen.hip's conv_igemm_kernel cut to one dimension: y[row][cout] = sum
// over (tap, cin) of x[row * stride + tap * dil - pad][cin] * w[cout][tap][cin], the weight stored
// tap-major (r4d_layout.cpp relayouts PyTorch's [cout][cin][k] at load), so a 64-wide K tile is
// eight 16-byte reads of input rows. A chunk outside its row's sequence reads zeros, which is the
// per-sequence padding. Eight waves, BP rows x BC output channels a block, the next K tile's
// loads in flight under the current WMMA.
#define C1_BK 64
#define C1_PITCH (C1_BK + 8)

// The output activation: 0 none, 1 tanh, 2 a clamp to [-1, 1] (a vocoder's output clamp), which
// leaves a NaN a NaN as libref's comparisons do (fminf / fmaxf would return the bound).
__device__ __forceinline__ float conv_act(float v, int act) {
    if (act == 1) return tanhf(v);
    if (act == 2) return v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
    return v;
}

/* Output row `row` reads input row row * st + tap * dil - pad, zero outside its sequence. */
template <int TP, int TC>
__global__ __launch_bounds__(256) void conv1d_igemm_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ bias, const uint16_t* __restrict__ res,
    const int* __restrict__ cu, uint16_t* __restrict__ y, int n_seq, int r_beg, int r_end,
    int cin, int cout, int k, int dil, int pad, int stride, int cdiv, int x_ld, int y_ld,
    int res_ld, int act) {
    constexpr int BP = 2 * TP * 16, BC = 4 * TC * 16;
    constexpr int CPR = C1_BK / 8;
    constexpr int AL = BP * CPR / 256, WL = BC * CPR / 256;
    static_assert(AL >= 1 && WL >= 1, "every thread stages whole chunks");
    __shared__ __attribute__((aligned(16))) uint16_t sA[BP * C1_PITCH];
    __shared__ __attribute__((aligned(16))) uint16_t sW[BC * C1_PITCH];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wp = wave / 4, wc = wave % 4;
    const int p0 = r_beg + blockIdx.x * BP, c0 = blockIdx.y * BC;
    const int K = k * cin;

    int a_row[AL], a_t[AL], a_ci[AL], a_s0[AL], a_s1[AL];
    bool a_ok[AL];
    uint16_t* adst[AL];
    #pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int m = p0 + r;
        a_ok[i] = m < r_end;
        a_row[i] = a_ok[i] ? m : r_beg;
        const int sq = seq_of(cu, n_seq, a_row[i] * stride * cdiv);
        a_s0[i] = cu[sq] / cdiv;
        a_s1[i] = cu[sq + 1] / cdiv;
        a_t[i] = e / cin;
        a_ci[i] = e % cin;
        adst[i] = sA + r * C1_PITCH + e;
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
        wdst[i] = sW + r * C1_PITCH + e;
    }
    const r4d_u32x4 zero = {0u, 0u, 0u, 0u};
    auto loadA = [&](int i) -> r4d_u32x4 {
        if (!a_ok[i] || a_t[i] >= k) return zero;
        const int ir = a_row[i] * stride + a_t[i] * dil - pad;
        if (ir < a_s0[i] || ir >= a_s1[i]) return zero;
        return *(const r4d_u32x4*)(x + (size_t)ir * x_ld + a_ci[i]);
    };
    auto loadW = [&](int i, int kt) -> r4d_u32x4 {
        if (!w_ok[i] || kt * C1_BK + w_e[i] >= K) return zero;
        return *(const r4d_u32x4*)(w + w_off[i] + (size_t)kt * C1_BK);
    };

    const uint16_t* fa = sA + (wp * TP * 16 + (lane & 15)) * C1_PITCH + 4 * (lane >> 4);
    const uint16_t* fw = sW + (wc * TC * 16 + (lane & 15)) * C1_PITCH + 4 * (lane >> 4);

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

    const int nk = (K + C1_BK - 1) / C1_BK;
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
                a_ci[i] += C1_BK;
                while (a_ci[i] >= cin) { a_ci[i] -= cin; ++a_t[i]; }
                ra[i] = loadA(i);
            }
            #pragma unroll
            for (int i = 0; i < WL; ++i) rw[i] = loadW(i, kt + 1);
        }
        #pragma unroll
        for (int s = 0; s < C1_BK / 16; ++s) {
            v8bf af[TP], wf[TC];
            #pragma unroll
            for (int i = 0; i < TP; ++i) af[i] = fragP(fa + i * 16 * C1_PITCH + s * 16);
            #pragma unroll
            for (int j = 0; j < TC; ++j) wf[j] = fragP(fw + j * 16 * C1_PITCH + s * 16);
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
            const int row = p0 + (wp * TP + i) * 16 + (lane & 15);
            const int co = c0 + (wc * TC + j) * 16 + 8 * (lane >> 4);
            if (row >= r_end || co >= cout) continue;
            uint16_t* dst = y + (size_t)row * y_ld + co;
            float f[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                float v = acc[j][i][e];
                if (co + e < cout) {
                    if (bias) v += bf2f(bias[co + e]);
                    if (res) v += bf2f(res[(size_t)row * res_ld + co + e]);
                    v = conv_act(v, act);
                }
                f[e] = v;
            }
            if (co + 8 <= cout && (cout & 7) == 0) {
                st8(dst, f);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    if (co + e < cout) dst[e] = f32_to_bf16(f[e]);
            }
        }
}

// The direct form, for a channel count the tiled kernel does not take (a cin not a multiple of
// eight, or a handful of output channels -- a vocoder's last conv has one): a thread an output
// element, f32 over (tap, cin), the same tap-major weight.
__global__ __launch_bounds__(kT) void conv1d_direct_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ bias, const uint16_t* __restrict__ res,
    const int* __restrict__ cu, uint16_t* __restrict__ y, int n_seq, int r_beg, int r_end,
    int cin, int cout, int k, int dil, int pad, int stride, int cdiv, int x_ld, int y_ld,
    int res_ld, int act) {
    const int64_t t = (int64_t)blockIdx.x * kT + threadIdx.x;
    const int64_t total = (int64_t)(r_end - r_beg) * cout;
    if (t >= total) return;
    const int co = (int)(t % cout);
    const int row = r_beg + (int)(t / cout);
    const int sq = seq_of(cu, n_seq, row * stride * cdiv);
    const int s0 = cu[sq] / cdiv, s1 = cu[sq + 1] / cdiv;
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        const int ir = row * stride + kk * dil - pad;
        if (ir < s0 || ir >= s1) continue;
        const uint16_t* xp = x + (size_t)ir * x_ld;
        const uint16_t* wp = w + ((size_t)co * k + kk) * cin;
        for (int ci = 0; ci < cin; ++ci) acc += bf2f(wp[ci]) * bf2f(xp[ci]);
    }
    if (bias) acc += bf2f(bias[co]);
    if (res) acc += bf2f(res[(size_t)row * res_ld + co]);
    acc = conv_act(acc, act);
    y[(size_t)row * y_ld + co] = f32_to_bf16(acc);
}

// ---------------------------------------------------------------------------- conv_t1d
// The transposed convolution of kernel `taps` s (two taps or one), stride s, padding P
// ((k - s) / 2, or 0 for the causal form) as s convolutions of `taps` taps, one a grid slice
// (blockIdx.z is the phase r): output row lo = q s + r - P of a sequence of L input rows reads
// x[q - t] for each tap t (PyTorch's tap r + t s). For a phase below P the q that land inside the
// sequence are 1..L, otherwise 0..L-1, so each phase has L output rows, one per input row: the M
// dimension is the input rows, as for conv1d. The weight is relayouted to [s][cout][taps][cin],
// which does not depend on P.
template <int TP, int TC>
__global__ __launch_bounds__(256) void conv_t1d_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ bias, const int* __restrict__ cu, uint16_t* __restrict__ y,
    int n_seq, int r_beg, int r_end, int cin, int cout, int s, int P, int taps, int x_ld,
    int y_ld) {
    constexpr int BP = 2 * TP * 16, BC = 4 * TC * 16;
    constexpr int CPR = C1_BK / 8;
    constexpr int AL = BP * CPR / 256, WL = BC * CPR / 256;
    static_assert(AL >= 1 && WL >= 1, "every thread stages whole chunks");
    __shared__ __attribute__((aligned(16))) uint16_t sA[BP * C1_PITCH];
    __shared__ __attribute__((aligned(16))) uint16_t sW[BC * C1_PITCH];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wp = wave / 4, wc = wave % 4;
    const int p0 = r_beg + blockIdx.x * BP, c0 = blockIdx.y * BC;
    const int ph = blockIdx.z, qoff = ph < P ? 1 : 0;
    const int K = taps * cin;
    const uint16_t* wph = w + (size_t)ph * cout * K;

    int a_q[AL], a_t[AL], a_ci[AL], a_s0[AL], a_len[AL];
    bool a_ok[AL];
    uint16_t* adst[AL];
    #pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int m = p0 + r;
        a_ok[i] = m < r_end;
        const int mm = a_ok[i] ? m : r_beg;
        const int sq = seq_of(cu, n_seq, mm);
        a_s0[i] = cu[sq];
        a_len[i] = cu[sq + 1] - cu[sq];
        a_q[i] = mm - cu[sq] + qoff;
        a_t[i] = e / cin;
        a_ci[i] = e % cin;
        adst[i] = sA + r * C1_PITCH + e;
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
        wdst[i] = sW + r * C1_PITCH + e;
    }
    const r4d_u32x4 zero = {0u, 0u, 0u, 0u};
    auto loadA = [&](int i) -> r4d_u32x4 {
        if (!a_ok[i] || a_t[i] >= taps) return zero;
        const int qi = a_q[i] - a_t[i];
        if (qi < 0 || qi >= a_len[i]) return zero;
        return *(const r4d_u32x4*)(x + (size_t)(a_s0[i] + qi) * x_ld + a_ci[i]);
    };
    auto loadW = [&](int i, int kt) -> r4d_u32x4 {
        if (!w_ok[i] || kt * C1_BK + w_e[i] >= K) return zero;
        return *(const r4d_u32x4*)(wph + w_off[i] + (size_t)kt * C1_BK);
    };

    const uint16_t* fa = sA + (wp * TP * 16 + (lane & 15)) * C1_PITCH + 4 * (lane >> 4);
    const uint16_t* fw = sW + (wc * TC * 16 + (lane & 15)) * C1_PITCH + 4 * (lane >> 4);

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

    const int nk = (K + C1_BK - 1) / C1_BK;
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
                a_ci[i] += C1_BK;
                while (a_ci[i] >= cin) { a_ci[i] -= cin; ++a_t[i]; }
                ra[i] = loadA(i);
            }
            #pragma unroll
            for (int i = 0; i < WL; ++i) rw[i] = loadW(i, kt + 1);
        }
        #pragma unroll
        for (int ss = 0; ss < C1_BK / 16; ++ss) {
            v8bf af[TP], wf[TC];
            #pragma unroll
            for (int i = 0; i < TP; ++i) af[i] = fragP(fa + i * 16 * C1_PITCH + ss * 16);
            #pragma unroll
            for (int j = 0; j < TC; ++j) wf[j] = fragP(fw + j * 16 * C1_PITCH + ss * 16);
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
            const int m = p0 + (wp * TP + i) * 16 + (lane & 15);
            const int co = c0 + (wc * TC + j) * 16 + 8 * (lane >> 4);
            if (m >= r_end || co >= cout) continue;
            const int sq = seq_of(cu, n_seq, m);
            const int q = m - cu[sq] + qoff;
            const int orow = cu[sq] * s + q * s + ph - P;
            uint16_t* dst = y + (size_t)orow * y_ld + co;
            float f[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) f[e] = acc[j][i][e] + (co + e < cout ? bf2f(bias[co + e]) : 0.f);
            if (co + 8 <= cout && (cout & 7) == 0) {
                st8(dst, f);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    if (co + e < cout) dst[e] = f32_to_bf16(f[e]);
            }
        }
}

// ---------------------------------------------------------------------------- f32 convolutions
// conv1d and conv_t1d over f32 rows, weights and biases (a vocoder kept in f32): the implicit GEMMs
// above on r4d_f32_tile.h's f32 tile, the weights relayouted alike by r4d_layout.cpp's _f32 hooks.
// The tile's rows are the output rows (conv1d) or one phase's input rows (conv_t1d), its columns
// the output channels, its K the (tap, channel) pairs, channel innermost. A thread stages one K
// column a slab and steps it sixteen at a time, so no K index is divided after the first.

// conv1d's A: output row m's tap t reads input row m stride + t dil - pad of m's own sequence,
// zero outside it and past the last tap. As r4d_f32::RowsLoader, every load is issued at an
// address clamped into x's first `r_end` stride rows and the zero selected after it.
template <int CNT>
struct Conv1dRows {
    const float* x;
    const int* cu;
    int n_seq, r_end, cin, k, dil, pad, x_ld, stride, cdiv;
    int row[CNT], s0[CNT], s1[CNT];
    int t = 0, ci = 0;
    __device__ void init(int first_row, int kk) {
#pragma unroll
        for (int j = 0; j < CNT; ++j) {
            const int m = first_row + 16 * j;
            row[j] = m < r_end ? m : -1;
            const int sq = seq_of(cu, n_seq, (m < r_end ? m : r_end - 1) * stride * cdiv);
            s0[j] = cu[sq] / cdiv;
            s1[j] = cu[sq + 1] / cdiv;
        }
        t = kk / cin;
        ci = kk % cin;
    }
    __device__ void fetch(float (&v)[CNT]) const {
#pragma unroll
        for (int j = 0; j < CNT; ++j) {
            const int ir = row[j] * stride + t * dil - pad;
            const bool live = row[j] >= 0 && t < k && ir >= s0[j] && ir < s1[j];
            const int x_rows = r_end * stride;
            const int irc = ir < 0 ? 0 : (ir >= x_rows ? x_rows - 1 : ir);
            const float xv = x[(uint32_t)irc * (uint32_t)x_ld + (uint32_t)ci];
            v[j] = live ? xv : 0.0f;
        }
    }
    __device__ void advance() {
        ci += r4d_f32::kBK;
        while (ci >= cin) { ci -= cin; ++t; }
    }
};

// `res` may be `y` itself (a residual chain updated in place): each element is read by the thread
// that stores it, before the store.
template <int TN>
__global__ __launch_bounds__(r4d_f32::kThr) void conv1d_f32_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias,
    const float* res, const int* __restrict__ cu, float* y, int n_seq, int rows, int cin,
    int cout, int k, int dil, int pad, int stride, int cdiv, int x_ld, int y_ld, int res_ld,
    int act) {
    constexpr int TM = 8, BM = 16 * TM, BN = 16 * TN;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int K = k * cin;
    Conv1dRows<TM> la{ x, cu, n_seq, rows, cin, k, dil, pad, x_ld, stride, cdiv };
    r4d_f32::RowsLoader<TN> lw{ w, K, cout, K };
    la.init(m0 + ty, tx);
    lw.init(n0 + ty, tx);
    auto ep = [&](int r, int c, int cnt, float4 q, int) {
        const int m = m0 + r, co = n0 + c;
        if (m >= rows) return;
        const float v[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            if (e >= cnt || co + e >= cout) continue;
            float a = v[e];
            if (bias) a += bias[co + e];
            if (res) a += res[(size_t)m * res_ld + co + e];
            y[(size_t)m * y_ld + co + e] = conv_act(a, act);
        }
    };
    r4d_f32::tile<TM, TN>(la, lw, K, ep);
}

// The direct form for a handful of output channels (a vocoder's last convolutions have 8 and 1):
// a thread an output element over the same tap-major weight.
__global__ __launch_bounds__(kT) void conv1d_f32_direct_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias,
    const float* res, const int* __restrict__ cu, float* y, int n_seq, int rows, int cin,
    int cout, int k, int dil, int pad, int stride, int cdiv, int x_ld, int y_ld, int res_ld,
    int act) {
    const int64_t t = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (t >= (int64_t)rows * cout) return;
    const int co = (int)(t % cout);
    const int row = (int)(t / cout);
    const int sq = seq_of(cu, n_seq, row * stride * cdiv);
    const int s0 = cu[sq] / cdiv, s1 = cu[sq + 1] / cdiv;
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        const int ir = row * stride + kk * dil - pad;
        if (ir < s0 || ir >= s1) continue;
        const float* xp = x + (size_t)ir * x_ld;
        const float* wp = w + ((size_t)co * k + kk) * cin;
        for (int ci = 0; ci < cin; ++ci) acc = __builtin_fmaf(wp[ci], xp[ci], acc);
    }
    if (bias) acc += bias[co];
    if (res) acc += res[(size_t)row * res_ld + co];
    y[(size_t)row * y_ld + co] = conv_act(acc, act);
}

// conv_t1d's A at phase r: input row m of a sequence at s0 is that phase's q = m - s0 + qoff,
// whose output row is q stride + r - pad, and its tap t reads input row q - t of the sequence,
// zero outside it. `qoff` is the phase's first q with a non-negative output row. Loads are
// issued unguarded, as Conv1dRows'.
template <int CNT>
struct ConvT1dRows {
    const float* x;
    const int* cu;
    int n_seq, r_end, cin, taps, qoff, x_ld;
    int s0[CNT], len[CNT], q[CNT];
    bool ok[CNT];
    int t = 0, ci = 0;
    __device__ void init(int first_row, int kk) {
#pragma unroll
        for (int j = 0; j < CNT; ++j) {
            const int m = first_row + 16 * j;
            ok[j] = m < r_end;
            const int sq = seq_of(cu, n_seq, ok[j] ? m : r_end - 1);
            s0[j] = cu[sq];
            len[j] = cu[sq + 1] - cu[sq];
            q[j] = m - s0[j] + qoff;
        }
        t = kk / cin;
        ci = kk % cin;
    }
    __device__ void fetch(float (&v)[CNT]) const {
#pragma unroll
        for (int j = 0; j < CNT; ++j) {
            const int qi = q[j] - t;
            const bool live = ok[j] && t < taps && qi >= 0 && qi < len[j];
            const int ir = s0[j] + qi;
            const int irc = ir < 0 ? 0 : (ir >= r_end ? r_end - 1 : ir);
            const float xv = x[(uint32_t)irc * (uint32_t)x_ld + (uint32_t)ci];
            v[j] = live ? xv : 0.0f;
        }
    }
    __device__ void advance() {
        ci += r4d_f32::kBK;
        while (ci >= cin) { ci -= cin; ++t; }
    }
};

// One phase a grid slice (blockIdx.z): its weight is [cout][taps cin], taps past k zero.
template <int TN>
__global__ __launch_bounds__(r4d_f32::kThr) void conv_t1d_f32_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias,
    const int* __restrict__ cu, float* __restrict__ y, int n_seq, int rows, int cin, int cout,
    int s, int P, int taps, int x_ld, int y_ld) {
    constexpr int TM = 8, BM = 16 * TM, BN = 16 * TN;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, ph = blockIdx.z;
    const int K = taps * cin;
    const int qoff = P >= ph ? (P - ph + s - 1) / s : 0;
    ConvT1dRows<TM> la{ x, cu, n_seq, rows, cin, taps, qoff, x_ld };
    r4d_f32::RowsLoader<TN> lw{ w + (size_t)ph * cout * K, K, cout, K };
    la.init(m0 + ty, tx);
    lw.init(n0 + ty, tx);
    auto ep = [&](int r, int c, int cnt, float4 qv, int) {
        const int m = m0 + r, co = n0 + c;
        if (m >= rows) return;
        const int sq = seq_of(cu, n_seq, m);
        const int orow = cu[sq] * s + (m - cu[sq] + qoff) * s + ph - P;
        const float v[4] = { qv.x, qv.y, qv.z, qv.w };
#pragma unroll
        for (int e = 0; e < 4; ++e)
            if (e < cnt && co + e < cout) y[(size_t)orow * y_ld + co + e] = v[e] + bias[co + e];
    };
    r4d_f32::tile<TM, TN>(la, lw, K, ep);
}

// ---------------------------------------------------------------------------- elementwise
// Snake, or with `beta` the SnakeBeta form x + (1 / (beta + 1e-9)) sin(alpha x)^2; with `lg` both
// parameters are stored as logarithms and exponentiated here, as the reference does each pass.
__global__ __launch_bounds__(kT) void snake_kernel(
    const uint16_t* __restrict__ x, const float* __restrict__ alpha, const float* __restrict__ beta,
    uint16_t* __restrict__ y, int n, int lg, int x_ld, int y_ld, int64_t total) {
    const int64_t t = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (t >= total) return;
    const int64_t r = t / n;
    const int c = (int)(t % n);
    const float v = bf2f(x[(size_t)r * x_ld + c]);
    const float a = lg ? expf(alpha[c]) : alpha[c];
    const float sn = sinf(v * a);
    if (beta) {
        const float b = lg ? expf(beta[c]) : beta[c];
        y[(size_t)r * y_ld + c] = f32_to_bf16(v + 1.0f / (b + 1e-9f) * (sn * sn));
    } else {
        y[(size_t)r * y_ld + c] = f32_to_bf16(v + sn * sn / (a + 1e-9f));
    }
}

// ---------------------------------------------------------------------------- the sampler
// An order-preserving key: larger floats map to larger unsigned values.
__device__ __forceinline__ uint32_t fkey(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// A block-wide sum of an integer, every thread receiving it. `sh` is 8 ints of LDS.
__device__ __forceinline__ int block_sum_i(int v, int* sh) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    int t = 0;
    #pragma unroll
    for (int i = 0; i < kT / 32; ++i) t += sh[i];
    return t;
}

// The key of the `kth` largest of the values each thread produces through `val(j, &ok)` over its
// slice [j0, j1): an MSB-first radix select, one block-wide count a bit.
template <class F>
__device__ uint32_t kth_key(F val, int j0, int j1, int kth, int* sh) {
    uint32_t prefix = 0, mask = 0;
    int want = kth;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t b = 1u << bit;
        int c = 0;
        for (int j = j0; j < j1; ++j) {
            bool ok;
            const float v = val(j, &ok);
            if (!ok) continue;
            const uint32_t kk = fkey(v);
            if ((kk & mask) == prefix && (kk & b)) ++c;
        }
        const int tot = block_sum_i(c, sh);
        if (tot >= want) prefix |= b;
        else want -= tot;
        mask |= b;
    }
    return prefix;
}

// One workgroup a lane (see libref's ref_cfg_topk_sample for the rule). Thread t owns the
// contiguous slice [t chunk, (t + 1) chunk) of the vocabulary, so a prefix over threads is a
// prefix over indices, which decides the ties at the k-th value exactly as libref does.
//
// `cfgs` (per-lane guidance), `lane` + `hist` (the repetition penalty's history, [n_hist, V], a
// lane whose row is out of range read and written as no history) and the suppressed range
// [sup_lo, sup_hi) are optional. The kept values are divided by the temperature, as libref divides:
// a multiply by its inverse rounds differently and can move a pick across a CDF boundary. A lane
// with no candidate left writes -1.
//
// `top_p` < 1 cuts the kept list where the whole candidate set's cumulative probability passes it
// (Fish Audio's draw). The normaliser is each thread's slice summed in index order, the slices
// then summed by thread 0 in thread order: index order as libref sums, associated differently,
// which in double moves a cut only for a cumulative within ~1e-16 of top_p.
constexpr int kMaxK = 256;

__global__ __launch_bounds__(kT) void cfg_topk_kernel(
    const void* __restrict__ lg, int lg_bf16, int lg_ld, const uint32_t* __restrict__ seed,
    const int* __restrict__ frame, int* __restrict__ ids, int V, int K, int rep, int cond_k,
    float cfg, float temp, uint64_t stream, const float* __restrict__ cfgs,
    const int* __restrict__ lane, int* __restrict__ hist, int n_hist, int sup_lo, int sup_hi,
    float pen, float top_p) {
    __shared__ int sh[8];
    __shared__ int s_pre[kT];
    __shared__ double s_z[kT];
    __shared__ float s_g[kMaxK];
    __shared__ int s_i[kMaxK];
    __shared__ float s_e[kMaxK];
    __shared__ int s_pick;
    const int ln = blockIdx.x, tid = threadIdx.x;
    const int chunk = (V + kT - 1) / kT;
    const int j0 = min(V, tid * chunk), j1 = min(V, j0 + chunk);
    auto ldl = [&](int row, int j) -> float {
        return lg_bf16 ? bf2f(((const uint16_t*)lg)[(size_t)row * lg_ld + j])
                       : ((const float*)lg)[(size_t)row * lg_ld + j];
    };
    const int rc = rep * ln, ru = rep * ln + (rep > 1 ? 1 : 0);
    const float gl = cfgs ? cfgs[ln] : cfg;
    // The lane's history row; at the lane's first frame it holds nothing yet and is cleared below.
    int hb = -1;
    if (hist) {
        const int L = lane[ln];
        if (L >= 0 && L < n_hist) hb = L * V;
    }
    const bool fresh = frame[ln] == 0;
    auto cval = [&](int j) { return ldl(rc, j); };
    auto gval = [&](int j) {
        const float c = ldl(rc, j);
        float g = c;
        if (rep != 1) {
            const float u = ldl(ru, j);
            g = u + gl * (c - u);
        }
        if (hb >= 0 && !fresh && hist[hb + j] != 0) g = g > 0.0f ? g / pen : g * pen;
        return g;
    };
    // The candidates: every entry outside the suppressed range, or of those the ones at or above
    // the conditional's cond_k-th largest (taken over every entry, as libref takes it).
    uint32_t cthr = 0;
    if (cond_k > 0)
        cthr = kth_key([&](int j, bool* ok) { *ok = true; return cval(j); }, j0, j1, cond_k, sh);
    auto cand = [&](int j) {
        return (j < sup_lo || j >= sup_hi) && (cond_k <= 0 || fkey(cval(j)) >= cthr);
    };
    // The k-th largest guided value among the candidates; fewer candidates than k takes them all.
    int nc = 0;
    for (int j = j0; j < j1; ++j) nc += cand(j) ? 1 : 0;
    const int n_cand = block_sum_i(nc, sh);
    const int kk = min(K, n_cand);
    if (kk == 0) {
        if (tid < rep) ids[rep * ln + tid] = -1;
        return;
    }
    const uint32_t gthr = kth_key([&](int j, bool* ok) { *ok = cand(j); return gval(j); }, j0, j1,
                                  kk, sh);
    // Every candidate above it, and of those equal to it the lowest indices up to k.
    int gt = 0, eq = 0;
    for (int j = j0; j < j1; ++j) {
        if (!cand(j)) continue;
        const uint32_t key = fkey(gval(j));
        gt += key > gthr ? 1 : 0;
        eq += key == gthr ? 1 : 0;
    }
    const int n_gt = block_sum_i(gt, sh);
    // An exclusive prefix over threads of (gt, eq): a thread's first slot and how many equal
    // entries precede it.
    __syncthreads();
    s_pre[tid] = (gt << 16) | eq;
    __syncthreads();
    int pg = 0, pe = 0;
    for (int t = 0; t < tid; ++t) { pg += s_pre[t] >> 16; pe += s_pre[t] & 0xffff; }
    int og = pg;
    int oe = n_gt + pe;
    for (int j = j0; j < j1; ++j) {
        if (!cand(j)) continue;
        const float g = gval(j);
        const uint32_t key = fkey(g);
        if (key > gthr) { s_g[og] = g; s_i[og] = j; ++og; }
        else if (key == gthr && oe < kk) { s_g[oe] = g; s_i[oe] = j; ++oe; }
    }
    __syncthreads();
    // The kept entries in descending g, ties to the lower index, each placed by its rank.
    float mg = 0.f;
    int mi = 0;
    if (tid < kk) { mg = s_g[tid]; mi = s_i[tid]; }
    __syncthreads();
    if (tid < kk) {
        int rank = 0;
        for (int t = 0; t < kk; ++t)
            if (s_g[t] > mg || (s_g[t] == mg && s_i[t] < mi)) ++rank;
        s_e[rank] = mg;
        s_pre[rank] = mi;
    }
    __syncthreads();
    if (top_p < 1.f) {
        const float m = s_e[0] / temp;
        double z = 0;
        for (int j = j0; j < j1; ++j)
            if (cand(j)) z += exp((double)(gval(j) / temp - m));
        s_z[tid] = z;
    }
    __syncthreads();
    if (tid == 0) {
        const float m = s_e[0] / temp;
        int kp = kk;
        if (top_p < 1.f) {
            double z = 0;
            for (int t = 0; t < kT; ++t) z += s_z[t];
            double cum = 0;
            kp = 1;
            for (int t = 0; t < kk; ++t) {
                cum += exp((double)(s_e[t] / temp - m)) / z;
                if (t > 0 && cum > (double)top_p) break;
                kp = t + 1;
            }
        }
        const int kk = kp;
        double tot = 0;
        for (int t = 0; t < kk; ++t) tot += exp((double)(s_e[t] / temp - m));
        const double u = (double)rad_philox_uniform_s(seed_of(seed, ln),
                                                      ((uint64_t)1 << 32) + (uint64_t)frame[ln],
                                                      stream) * tot;
        int pick = s_pre[kk - 1];
        double acc = 0;
        for (int t = 0; t < kk; ++t) {
            acc += exp((double)(s_e[t] / temp - m));
            if (u < acc) { pick = s_pre[t]; break; }
        }
        s_pick = pick;
    }
    __syncthreads();
    if (tid < rep) ids[rep * ln + tid] = s_pick;
    // The history: a new piece's row cleared, then the pick recorded. Every read of the row above
    // is done (the barrier after the pick), and a fresh row was not read at all.
    if (hb >= 0) {
        if (fresh)
            for (int j = j0; j < j1; ++j) hist[hb + j] = 0;
        __syncthreads();
        if (tid == 0) hist[hb + s_pick] = 1;
    }
}

// ---------------------------------------------------------------------------- frames
// A table element as f32, the table bf16 or f32.
__device__ __forceinline__ float ldtab(const void* t, int f32, size_t i) {
    return f32 ? ((const float*)t)[i] : bf2f(((const uint16_t*)t)[i]);
}

// Code (k, r) at k * kp + r * rp: the plane's own strides, so a block of a wider plane (a lane's
// codebooks over a run of its frames) is read in place. With `round1` the whole sum is one f32
// accumulation in codebook order, as libref's; otherwise t0 + bf16(sum of the rest), rounded as
// the reference rounds. A first code outside t0 leaves its row unwritten.
__global__ __launch_bounds__(kT) void rvq_embed_kernel(
    const int* __restrict__ codes, int64_t kp, int64_t rp, const void* __restrict__ t0,
    int t0_f32, const void* __restrict__ t1, int t1_f32, void* __restrict__ y, int y_f32, int D,
    int K, int voc, int off0, int r0, int r1, float scale, int y_ld, int round1, int clamp) {
    const int r = blockIdx.x;
    auto cl = [clamp](int c) { return clamp > 0 && c > clamp - 1 ? clamp - 1 : c; };
    const int c0 = cl(codes[(size_t)r * rp]) + off0;
    if (c0 < 0 || c0 >= r0) return;
    for (int d = threadIdx.x; d < D; d += kT) {
        const float e0 = ldtab(t0, t0_f32, (size_t)c0 * D + d);
        float acc = round1 ? e0 : 0.0f;
        for (int k = 1; k < K; ++k) {
            const int ck = cl(codes[(size_t)k * kp + (size_t)r * rp]) + (k - 1) * voc;
            if (ck >= 0 && ck < r1) acc += ldtab(t1, t1_f32, (size_t)ck * D + d);
        }
        const float v = (round1 ? acc : bfr(e0 + bfr(acc))) * scale;
        if (y_f32) ((float*)y)[(size_t)r * y_ld + d] = v;
        else ((uint16_t*)y)[(size_t)r * y_ld + d] = f32_to_bf16(v);
    }
}

// One residual quantiser step (DAC's VectorQuantize), a workgroup a row of x. e = in_w x + in_b,
// each of its `dim` sums split over the workgroup and reduced through LDS; each thread then scores
// a strided run of the codebook against e normalised, every row normalised as it is read
// (F.normalize's division), and the workgroup's argmax keeps the lower index on a tie; last the
// chosen raw row's out-projection leaves x's row, rounded once. libref's arithmetic in another
// summation order.
constexpr int kVqMaxDim = 32;

__global__ __launch_bounds__(kT) void vq_encode_kernel(
    void* __restrict__ x, int x_f32, int x_ld, const void* __restrict__ iw, int iw_f32,
    const void* __restrict__ ib, int ib_f32, const void* __restrict__ cb, int cb_f32,
    const void* __restrict__ ow, int ow_f32, const void* __restrict__ ob, int ob_f32,
    int* __restrict__ codes, int64_t code_st, int n, int dim, int bins) {
    __shared__ float part[kT / 32][kVqMaxDim];
    __shared__ float e[kVqMaxDim], eh[kVqMaxDim];
    __shared__ float wbest[kT / 32];
    __shared__ int wcode[kT / 32];
    const int tid = threadIdx.x, wv = tid >> 5, ln = tid & 31;
    const size_t xb = (size_t)blockIdx.x * x_ld;
    auto ldx = [&](int d) {
        return x_f32 ? ((const float*)x)[xb + d] : bf2f(((const uint16_t*)x)[xb + d]);
    };
    for (int j = 0; j < dim; ++j) {
        float acc = 0.0f;
        for (int d = tid; d < n; d += kT) acc += ldtab(iw, iw_f32, (size_t)j * n + d) * ldx(d);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 32);
        if (ln == 0) part[wv][j] = acc;
    }
    __syncthreads();
    if (tid < dim) {
        float acc = 0.0f;
        for (int w = 0; w < kT / 32; ++w) acc += part[w][tid];
        e[tid] = acc + ldtab(ib, ib_f32, (size_t)tid);
    }
    __syncthreads();
    if (tid < dim) {
        float ss = 0.0f;
        for (int j = 0; j < dim; ++j) ss += e[j] * e[j];
        eh[tid] = e[tid] / fmaxf(sqrtf(ss), 1e-12f);
    }
    __syncthreads();
    // Each thread's best over its rows, ascending, so its first maximum is kept; INT_MAX none.
    float bs = -INFINITY;
    int bc = 0x7fffffff;
    for (int c = tid; c < bins; c += kT) {
        float ss = 0.0f;
        for (int j = 0; j < dim; ++j) {
            const float v = ldtab(cb, cb_f32, (size_t)c * dim + j);
            ss += v * v;
        }
        const float nrm = fmaxf(sqrtf(ss), 1e-12f);
        float dot = 0.0f;
        for (int j = 0; j < dim; ++j) dot += eh[j] * (ldtab(cb, cb_f32, (size_t)c * dim + j) / nrm);
        if (dot > bs) { bs = dot; bc = c; }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float os = __shfl_xor(bs, o, 32);
        const int oc = __shfl_xor(bc, o, 32);
        if (os > bs || (os == bs && oc < bc)) { bs = os; bc = oc; }
    }
    if (ln == 0) { wbest[wv] = bs; wcode[wv] = bc; }
    __syncthreads();
    float fs = wbest[0];
    int fc = wcode[0];
    for (int w = 1; w < kT / 32; ++w)
        if (wbest[w] > fs || (wbest[w] == fs && wcode[w] < fc)) { fs = wbest[w]; fc = wcode[w]; }
    const int code = fc == 0x7fffffff ? 0 : fc;   // no score above -inf: code 0, as libref
    if (tid == 0) codes[(size_t)blockIdx.x * code_st] = code;
    for (int d = tid; d < n; d += kT) {
        float q = 0.0f;
        for (int j = 0; j < dim; ++j)
            q += ldtab(ow, ow_f32, (size_t)d * dim + j) * ldtab(cb, cb_f32, (size_t)code * dim + j);
        const float v = ldx(d) - (q + ldtab(ob, ob_f32, (size_t)d));
        if (x_f32) ((float*)x)[xb + d] = v;
        else ((uint16_t*)x)[xb + d] = f32_to_bf16(v);
    }
}

// dst[(lane n_codes + k) stride + frame - skip] = codes[k, rep i]; a thread a (lane, code). A lane
// outside `status`, or a frame past `stride` or a lane past `dst`, stores nothing (libref refuses
// the call).
__global__ __launch_bounds__(kT) void frame_store_kernel(
    const int* __restrict__ codes, int64_t kp, int64_t rp, const int* __restrict__ lane,
    const int* __restrict__ frame, const int* __restrict__ status, int* __restrict__ dst, int n,
    int K, int stride, int rep, int skip, int n_status, int64_t n_dst) {
    const int t = blockIdx.x * kT + threadIdx.x;
    if (t >= n * K) return;
    const int i = t / K, k = t % K;
    const int L = lane[i], f = frame[i] - skip;
    if (L < 0 || L >= n_status || f < 0 || status[L] < 0) return;
    if (f >= stride || (int64_t)(L + 1) * K * stride > n_dst) return;
    dst[((size_t)L * K + k) * stride + f] = codes[(size_t)k * kp + (size_t)rep * i * rp];
}

__global__ __launch_bounds__(kT) void frame_mix_kernel(
    const uint16_t* __restrict__ hid, int plane, const float* __restrict__ logits,
    const float* __restrict__ scale, const int* __restrict__ lane, const int* __restrict__ frame,
    const int* __restrict__ status, uint16_t* __restrict__ dst, int D, int L, int stride, int rep,
    int h_ld, int d_ld) {
    __shared__ float w[16];
    const int i = blockIdx.x;
    const int ln = lane[i], f = frame[i];
    if (f < 1 || status[ln] < 0) return;
    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int l = 0; l < L; ++l) mx = fmaxf(mx, bfr(logits[l]));
        double tot = 0;
        for (int l = 0; l < L; ++l) tot += exp((double)(bfr(logits[l]) - mx));
        for (int l = 0; l < L; ++l) w[l] = bfr((float)(exp((double)(bfr(logits[l]) - mx)) / tot));
    }
    __syncthreads();
    const float sc = bfr(scale[0]);
    uint16_t* out = dst + ((size_t)ln * stride + f - 1) * d_ld;
    for (int d = threadIdx.x; d < D; d += kT) {
        float acc = 0.0f;
        for (int l = 0; l < L; ++l) acc += w[l] * bf2f(hid[((size_t)l * plane + rep * i) * h_ld + d]);
        out[d] = f32_to_bf16(bfr(acc) * sc);
    }
}

// Fish Audio's repetition-aware choice between two draws, a thread a lane (libref's
// ref_ras_pick): the alternative where the first draw is in [lo, hi) and in the lane's window of
// its last `W` picks, the pick then entering the window. A lane whose window lies outside `hist`
// takes the first draw and keeps no window.
__global__ __launch_bounds__(kT) void ras_pick_kernel(
    const int* __restrict__ first, const int* __restrict__ alt, const int* __restrict__ lane,
    int* __restrict__ hist, int* __restrict__ ids, int n, int W, int lo, int hi, int rep,
    int clear, long long n_hist) {
    const int i = blockIdx.x * kT + threadIdx.x;
    if (i >= n) return;
    const int f = first[rep * i];
    const long long L = lane[i];
    int pick = f;
    if (L >= 0 && (L + 1) * W <= n_hist) {
        int* h = hist + L * W;
        if (clear)
            for (int w = 0; w < W; ++w) h[w] = -1;
        if (f >= lo && f < hi)
            for (int w = 0; w < W; ++w)
                if (h[w] == f) { pick = alt[rep * i]; break; }
        for (int w = 0; w + 1 < W; ++w) h[w] = h[w + 1];
        h[W - 1] = pick;
    }
    for (int r = 0; r < rep; ++r) ids[rep * i + r] = pick;
}

__global__ __launch_bounds__(kT) void end_status_kernel(
    const int* __restrict__ ids, const int* __restrict__ lane, int* __restrict__ status, int n,
    int rep, int end) {
    const int i = blockIdx.x * kT + threadIdx.x;
    if (i >= n) return;
    const int v = ids[rep * i];
    status[lane[i]] = v == end ? -1 : v;
}

// ---------------------------------------------------------------------------- rows
__global__ __launch_bounds__(kT) void resample_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, int n, int in, int out, int x_ld,
    int y_ld) {
    const int j = blockIdx.x;
    const float sc = (float)in / (float)out;
    const int src = min((int)floorf((float)j * sc), in - 1);
    for (int c = threadIdx.x; c < n; c += kT) y[(size_t)j * y_ld + c] = x[(size_t)src * x_ld + c];
}

__global__ __launch_bounds__(kT) void fourier_kernel(
    const float* __restrict__ t, const float* __restrict__ w, uint16_t* __restrict__ y, int h,
    int rnd, int y_ld) {
    const int r = blockIdx.x;
    const float two_pi = 6.28318530717958647692f;
    const float tv = rnd ? bfr(bfr(t[r]) * two_pi) : t[r] * two_pi;
    for (int i = threadIdx.x; i < h; i += kT) {
        float ang = tv * w[i];
        if (rnd) ang = bfr(ang);
        y[(size_t)r * y_ld + i] = f32_to_bf16(cosf(ang));
        y[(size_t)r * y_ld + h + i] = f32_to_bf16(sinf(ang));
    }
}

__global__ __launch_bounds__(kT) void win_input_kernel(
    const uint16_t* __restrict__ lat, const uint16_t* __restrict__ cond,
    const int* __restrict__ lcu, const int* __restrict__ wlane, const int* __restrict__ pair,
    uint16_t* __restrict__ y, int S, int C, int Z, int D, int stride, int l_ld, int c_ld,
    int y_ld) {
    const int row = lcu[0] + blockIdx.x;
    if (row >= lcu[S]) return;
    const int e = seq_of(lcu, S, row);
    const int k = row - lcu[e], L = wlane[e];
    const bool neg = pair[e] == -2;
    uint16_t* yr = y + (size_t)row * y_ld;
    const uint16_t* lr = lat + ((size_t)L * stride + k) * l_ld;
    const uint16_t* cr = cond + ((size_t)L * stride + k) * c_ld;
    for (int c = threadIdx.x; c < C + Z + D; c += kT)
        yr[c] = c < C ? lr[c] : (c < C + Z || neg) ? (uint16_t)0 : cr[c - C - Z];
}

__global__ __launch_bounds__(kT) void win_prep_kernel(
    const int* __restrict__ flags, const float* __restrict__ t, const uint32_t* __restrict__ seed,
    const int* __restrict__ widx, const int* __restrict__ wlane, const int* __restrict__ pair,
    const int* __restrict__ lcu, const uint16_t* __restrict__ prev, uint16_t* __restrict__ noise,
    uint16_t* __restrict__ lat, int n, int stride, int ov, int l_ld, int p_ld, int nz_ld) {
    const int e = blockIdx.y;
    if (pair[e] == -2) return;
    const int f = flags[e];
    if (!(f & (kFirst | kJoin))) return;
    const int64_t i = (int64_t)blockIdx.x * kT + threadIdx.x;
    const int len = lcu[e + 1] - lcu[e];
    if (i >= (int64_t)len * n) return;
    const int k = (int)(i / n), c = (int)(i % n), L = wlane[e];
    uint16_t* lp = lat + ((size_t)L * stride + k) * l_ld + c;
    float nz = 0.0f;
    if (f & kFirst) {
        const float v = bfr(rad_philox_normal_s(seed_of(seed, e), 1 + (uint64_t)widx[e],
                                                (uint64_t)i));
        *lp = f32_to_bf16(v);
        if ((f & kJoin) && k < ov) noise[((size_t)L * ov + k) * nz_ld + c] = f32_to_bf16(v);
        nz = v;
    }
    if ((f & kJoin) && k < ov) {
        if (!(f & kFirst)) nz = bf2f(noise[((size_t)L * ov + k) * nz_ld + c]);
        const float tb = bfr(t[e]);
        const float wn = bfr(1.0f - bfr((1.0f - 1e-6f) * tb));
        const float pv = bf2f(prev[((size_t)L * ov + k) * p_ld + c]);
        *lp = f32_to_bf16(bfr(wn * nz) + bfr(tb * pv));
    }
}

// The overlap copies run in two kernels so a row is not read and written by different threads in
// one: the join (the head from the previous window's prev) first, then the hand-on (the next prev
// from the joined latent's tail). The other order would join the head to the window's own tail.
__global__ __launch_bounds__(kT) void win_handon_kernel(
    const int* __restrict__ flags, const int* __restrict__ pair, const int* __restrict__ wlane,
    const int* __restrict__ lcu, const uint16_t* __restrict__ cond, const uint16_t* __restrict__ lat,
    uint16_t* __restrict__ prev, uint16_t* __restrict__ pcond, int n, int D, int stride, int ov,
    int l_ld, int c_ld, int p_ld, int pc_ld) {
    const int e = blockIdx.y;
    if (pair[e] == -2 || !(flags[e] & kLast)) return;
    const int len = lcu[e + 1] - lcu[e], L = wlane[e];
    const int s0 = max(0, len - 2 * ov), s1 = max(s0, len - ov);
    const int64_t i = (int64_t)blockIdx.x * kT + threadIdx.x;
    const int w = n + D;
    if (i >= (int64_t)(s1 - s0) * w) return;
    const int k = s0 + (int)(i / w), c = (int)(i % w);
    if (c < n)
        prev[((size_t)L * ov + k - s0) * p_ld + c] = lat[((size_t)L * stride + k) * l_ld + c];
    else
        pcond[((size_t)L * ov + k - s0) * pc_ld + c - n] =
            cond[((size_t)L * stride + k) * c_ld + c - n];
}

__global__ __launch_bounds__(kT) void win_join_kernel(
    const int* __restrict__ flags, const int* __restrict__ pair, const int* __restrict__ wlane,
    const int* __restrict__ lcu, const uint16_t* __restrict__ prev, uint16_t* __restrict__ lat,
    int n, int stride, int ov, int l_ld, int p_ld) {
    const int e = blockIdx.y;
    const int f = flags[e];
    if (pair[e] == -2 || !(f & kLast) || !(f & kJoin)) return;
    const int len = lcu[e + 1] - lcu[e], L = wlane[e];
    const int64_t i = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (i >= (int64_t)min(ov, len) * n) return;
    const int k = (int)(i / n), c = (int)(i % n);
    lat[((size_t)L * stride + k) * l_ld + c] = prev[((size_t)L * ov + k) * p_ld + c];
}

__global__ __launch_bounds__(kT) void chan_split_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, int n, int P, int rows, int x_ld,
    int y_ld, int64_t total) {
    const int64_t t = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (t >= total) return;
    const int c = (int)(t % n);
    const int64_t pr = t / n;
    const int p = (int)(pr / rows), r = (int)(pr % rows);
    y[(size_t)pr * y_ld + c] = x[(size_t)r * x_ld + (size_t)p * n + c];
}

// The two streams bf16 (widened exactly) or f32 (copied), the samples f32.
__device__ __forceinline__ float il_ld(const uint16_t* p, int64_t i) { return bf2f(p[i]); }
__device__ __forceinline__ float il_ld(const float* p, int64_t i) { return p[i]; }

template <typename T>
__global__ __launch_bounds__(kT) void interleave2_kernel(
    const T* __restrict__ a, const T* __restrict__ b, float* __restrict__ y, int64_t rows) {
    const int64_t r = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (r >= rows) return;
    y[2 * r] = il_ld(a, r);
    y[2 * r + 1] = il_ld(b, r);
}

// ---------------------------------------------------------------------------- indices
// y[i] = start + i step, or with `period` > 0 start + (i % period) step: positions that restart
// every `period` rows.
__global__ __launch_bounds__(kT) void iota_kernel(int* __restrict__ y, int n, int start, int step,
                                                  int period) {
    const int i = blockIdx.x * kT + threadIdx.x;
    if (i < n) y[i] = start + (period > 0 ? i % period : i) * step;
}

// y[j] = (src[j] < 0 ? 0 : x[src[j]]) * mul[j] + add[j], the lists by value.
constexpr int kAffMax = 16;
struct IntAffine { int src[kAffMax], mul[kAffMax], add[kAffMax]; };

__global__ __launch_bounds__(kAffMax) void int_affine_kernel(const int* __restrict__ x,
                                                             int* __restrict__ y, IntAffine p,
                                                             int n) {
    const int j = threadIdx.x;
    if (j < n) y[j] = (p.src[j] < 0 ? 0 : x[p.src[j]]) * p.mul[j] + p.add[j];
}

__global__ __launch_bounds__(kT) void cu_repeat_kernel(const int* __restrict__ len,
                                                       int* __restrict__ y, int n, int mult) {
    const int j = blockIdx.x * kT + threadIdx.x;
    if (j < n) y[j] = j * len[0] * mult;
}

__global__ __launch_bounds__(kT) void sub_i32_kernel(const int* __restrict__ x,
                                                     const int* __restrict__ s, int* __restrict__ y,
                                                     int n) {
    const int j = blockIdx.x * kT + threadIdx.x;
    if (j < n) y[j] = x[j] - s[0];
}

// ---------------------------------------------------------------------------- attention
// Dense causal attention over short sequences, head_dim 256, one query head a key head: a depth
// decoder runs a frame's handful of positions, the codebooks so far. A workgroup a (sequence,
// head) holds the sequence's keys and values in LDS; each of eight waves takes query rows, a lane
// eight dims. Sequences are at most kSmallSeq rows (the row's predicate).
constexpr int kSmallSeq = 32;

__global__ __launch_bounds__(kT) void attn_small_h256_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int* __restrict__ cu, uint16_t* __restrict__ o, int q_ld, int k_ld, int v_ld, int o_ld,
    float scale, int causal) {
    constexpr int HD = 256;
    __shared__ __attribute__((aligned(16))) uint16_t sk[kSmallSeq * HD];
    __shared__ __attribute__((aligned(16))) uint16_t sv[kSmallSeq * HD];
    const int sq = blockIdx.x, h = blockIdx.y;
    const int r0 = cu[sq], len = min(cu[sq + 1] - r0, kSmallSeq);
    for (int i = threadIdx.x; i < len * (HD / 8); i += kT) {
        const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
        *(uint4*)(sk + r * HD + c) = *(const uint4*)(k + (size_t)(r0 + r) * k_ld + h * HD + c);
        *(uint4*)(sv + r * HD + c) = *(const uint4*)(v + (size_t)(r0 + r) * v_ld + h * HD + c);
    }
    __syncthreads();
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = wave; i < len; i += kT / 32) {
        float qf[8];
        ld8(q + (size_t)(r0 + i) * q_ld + h * HD + lane * 8, qf);
        const int nk = causal ? i + 1 : len;
        float sc[kSmallSeq];
        float mx = -INFINITY;
        for (int j = 0; j < nk; ++j) {
            float kf[8];
            ld8(sk + j * HD + lane * 8, kf);
            float d = 0.f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) d += qf[e] * kf[e];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor(d, off, 32);
            sc[j] = d * scale;
            mx = fmaxf(mx, sc[j]);
        }
        float tot = 0.f, acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int j = 0; j < nk; ++j) {
            const float p = expf(sc[j] - mx);
            tot += p;
            float vf[8];
            ld8(sv + j * HD + lane * 8, vf);
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += p * vf[e];
        }
        #pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] /= tot;
        st8(o + (size_t)(r0 + i) * o_ld + h * HD + lane * 8, acc);
    }
}

// ---------------------------------------------------------------------------- shims
inline int st_of(hipError_t e) { return e == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE); }
inline unsigned blocks_for(int64_t n) { return (unsigned)((n + kT - 1) / kT); }
inline bool is_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }
inline bool is_i32(const RadTensor* t) { return t && t->dtype == RAD_I32; }
inline bool is_f32(const RadTensor* t) { return t && t->dtype == RAD_F32; }

// conv1d's `act` as conv_act's code, -1 for a name it does not know. Absent is none.
inline int conv1d_act(const char* act) {
    if (!act || !std::strcmp(act, "none")) return 0;
    if (!std::strcmp(act, "tanh")) return 1;
    if (!std::strcmp(act, "clamp")) return 2;
    return -1;
}

}  // namespace

// The conv shims grid over the rows of the operand they were handed (the output's for conv1d, the
// input's for conv_t1d), which is the pass's own row count: cu_seqlens lives on the device, so its
// ends are not read here, and its sequences start at row 0 and cover those rows.
extern "C" int r4d_rad_conv1d(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *cu = r4d_opt(a, 2);
    const RadTensor *res = r4d_opt(a, 3), *y = r4d_opt(a, 4), *b = r4d_opt(a, 5);
    if (!x || !w || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(w) || (b && !is_bf16(b)) || !is_bf16(y) || (res && !is_bf16(res)) ||
        !is_i32(cu))
        return R4D_NO(RAD_E_DTYPE);
    const long long cin = r4d_geti_or(a, "cin", 0), cout = r4d_geti_or(a, "cout", 0);
    const long long k = r4d_geti_or(a, "k", 0), dil = r4d_geti_or(a, "dil", 1);
    const long long causal = r4d_geti_or(a, "causal", 0);
    /* Two strided forms: the causal one (its own left pad, `cu_div` bounds) and, with `pad`, a
     * zero-padded one over one sequence of all of x (an audio encoder's downsample). */
    const long long st = r4d_geti_or(a, "stride", 1), pp = r4d_geti_or(a, "pad", -1);
    const long long cd = r4d_geti_or(a, "cu_div", 1);
    const int act = conv1d_act(r4d_gets(a, "act"));
    if (act < 0) return R4D_NO(RAD_E_INVAL);
    const bool padded = pp >= 0 && st > 1;
    if (cin <= 0 || cout <= 0 || k <= 0 || (!causal && st <= 1 && (k & 1) == 0) || dil <= 0 ||
        causal < 0 || causal > 1 || st < 1 || (pp >= 0 && (st <= 1 || causal || cd != 1)) ||
        (st > 1 && pp < 0 && (!causal || dil * (k - 1) + 1 < st)) || cd < 1)
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, ry = 0, rr = 0;
    const long long xl = r4d_rows_pitch(x, cin, &rx), yl = r4d_rows_pitch(y, cout, &ry);
    const long long rl = res ? r4d_rows_pitch(res, cout, &rr) : 0;
    if (xl < 0 || yl < 0 || rl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (n_seq <= 0 || ry <= 0 || (!padded && rx < ry * st) || (res && rr < ry) ||
        r4d_numel(w) < cout * cin * k || (b && r4d_numel(b) < cout))
        return R4D_NO(RAD_E_SHAPE);
    /* The padded form's output rows are PyTorch's (rx and ry are rows, xl and yl pitches); libref
     * checks that the sequence covers x. */
    if (padded) {
        if (n_seq != 1 || rx + 2 * pp < dil * (k - 1) + 1 ||
            ry != (rx + 2 * pp - dil * (k - 1) - 1) / st + 1)
            return R4D_NO(RAD_E_SHAPE);
    }
    const uint16_t* bp = b ? (const uint16_t*)b->data : nullptr;
    /* Both kernels read input row row * stride + tap * dil - pad: half the reach a side, all of
     * it on the left less the stride for the causal form, or the explicit `pad`. */
    const int pad = (int)(padded ? pp : (causal ? dil * (k - 1) + 1 - st : dil * (k - 1) / 2));
    const int cdiv = (int)(padded ? 1 : cd);
    const bool tiled = (cin % 8) == 0 && cout >= 16 && (xl % 8) == 0 &&
                       ((uintptr_t)x->data & 15) == 0 && ((uintptr_t)w->data & 15) == 0;
    const uint16_t* rp = res ? (const uint16_t*)res->data : nullptr;
    if (tiled) {
        // 128 rows a block; 128 output channels where there are that many, else 64.
        if (cout >= 128) {
            constexpr int TP = 4, TC = 2, BP = 128, BC = 128;
            hipLaunchKernelGGL((conv1d_igemm_kernel<TP, TC>),
                               dim3((unsigned)((ry + BP - 1) / BP), (unsigned)((cout + BC - 1) / BC)),
                               dim3(256), 0, (hipStream_t)s, (const uint16_t*)x->data,
                               (const uint16_t*)w->data, bp, rp,
                               (const int*)cu->data, (uint16_t*)y->data, (int)n_seq, 0, (int)ry,
                               (int)cin, (int)cout, (int)k, (int)dil, pad, (int)st, (int)cdiv,
                               (int)xl, (int)yl, (int)rl, act);
        } else {
            constexpr int TP = 4, TC = 1, BP = 128, BC = 64;
            hipLaunchKernelGGL((conv1d_igemm_kernel<TP, TC>),
                               dim3((unsigned)((ry + BP - 1) / BP), (unsigned)((cout + BC - 1) / BC)),
                               dim3(256), 0, (hipStream_t)s, (const uint16_t*)x->data,
                               (const uint16_t*)w->data, bp, rp,
                               (const int*)cu->data, (uint16_t*)y->data, (int)n_seq, 0, (int)ry,
                               (int)cin, (int)cout, (int)k, (int)dil, pad, (int)st, (int)cdiv,
                               (int)xl, (int)yl, (int)rl, act);
        }
    } else {
        hipLaunchKernelGGL(conv1d_direct_kernel, dim3(blocks_for(ry * cout)), dim3(kT), 0,
                           (hipStream_t)s, (const uint16_t*)x->data, (const uint16_t*)w->data,
                           bp, rp, (const int*)cu->data, (uint16_t*)y->data,
                           (int)n_seq, 0, (int)ry, (int)cin, (int)cout, (int)k, (int)dil, pad,
                           (int)st, (int)cdiv, (int)xl, (int)yl, (int)rl, act);
    }
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_conv_t1d(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *b = r4d_opt(a, 2), *cu = r4d_opt(a, 3);
    const RadTensor *y = r4d_opt(a, 4);
    if (!x || !w || !b || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(w) || !is_bf16(b) || !is_bf16(y) || !is_i32(cu))
        return R4D_NO(RAD_E_DTYPE);
    const long long cin = r4d_geti_or(a, "cin", 0), cout = r4d_geti_or(a, "cout", 0);
    const long long st = r4d_geti_or(a, "stride", 0);
    /* Kernel 2 stride (two taps) or stride (one); without `pad`, the symmetric (k - stride) / 2,
     * which needs k - stride even. */
    const long long kw = r4d_geti_or(a, "k", 2 * st), pad = r4d_geti_or(a, "pad", -1);
    const long long P = pad >= 0 ? pad : (kw - st) / 2;
    if (cin <= 0 || cout <= 0 || st <= 0 || (kw != st && kw != 2 * st) ||
        (pad < 0 && ((kw - st) & 1)) || P >= st || (cin % 8))
        return R4D_NO(RAD_E_SHAPE);
    const int taps = (int)(kw / st);
    long long rx = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, cin, &rx), yl = r4d_rows_pitch(y, cout, &ry);
    if (xl < 0 || yl < 0 || (xl % 8) || ((uintptr_t)x->data & 15) || ((uintptr_t)w->data & 15))
        return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (n_seq <= 0 || rx <= 0 || ry < rx * st || r4d_numel(w) < cin * cout * kw ||
        r4d_numel(b) < cout)
        return R4D_NO(RAD_E_SHAPE);
    if (cout >= 128) {
        constexpr int TP = 4, TC = 2, BP = 128, BC = 128;
        hipLaunchKernelGGL((conv_t1d_kernel<TP, TC>),
                           dim3((unsigned)((rx + BP - 1) / BP), (unsigned)((cout + BC - 1) / BC),
                                (unsigned)st),
                           dim3(256), 0, (hipStream_t)s, (const uint16_t*)x->data,
                           (const uint16_t*)w->data, (const uint16_t*)b->data, (const int*)cu->data,
                           (uint16_t*)y->data, (int)n_seq, 0, (int)rx, (int)cin, (int)cout, (int)st,
                           (int)P, taps, (int)xl, (int)yl);
    } else {
        constexpr int TP = 4, TC = 1, BP = 128, BC = 64;
        hipLaunchKernelGGL((conv_t1d_kernel<TP, TC>),
                           dim3((unsigned)((rx + BP - 1) / BP), (unsigned)((cout + BC - 1) / BC),
                                (unsigned)st),
                           dim3(256), 0, (hipStream_t)s, (const uint16_t*)x->data,
                           (const uint16_t*)w->data, (const uint16_t*)b->data, (const int*)cu->data,
                           (uint16_t*)y->data, (int)n_seq, 0, (int)rx, (int)cin, (int)cout, (int)st,
                           (int)P, taps, (int)xl, (int)yl);
    }
    return st_of(hipGetLastError());
}

/* The f32 rows. The weight is the _f32 layout hooks' (tap-major for conv1d, phase-major with
 * ceil(k / stride) taps for conv_t1d). The column tile is the narrowest of 128, 64 and 32 that
 * covers cout in as few workgroups; fewer than 16 channels take the direct conv1d kernel. */
extern "C" int r4d_rad_encoder_layernorm(const RadArgs* a, RadStream s) {
    const auto *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *b = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!x || !w || !y || !is_f32(x) || !is_f32(w) || !is_f32(y) || (b && !is_f32(b)))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n < 1 || n > 4096 || r4d_numel(w) != n || (b && r4d_numel(b) != n)) return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, ry = 0;
    const auto xp = r4d_rows_pitch(x, n, &rx), yp = r4d_rows_pitch(y, n, &ry);
    if (xp < 0 || yp < 0 || !r4d_contig(w) || (b && !r4d_contig(b))) return R4D_NO(RAD_E_STRIDE);
    const float eps = r4d_getf(a, "eps", 1e-5);
    if (ry < 1 || ry > 65535 || rx < ry || xp > 0x7fffffffll || yp > 0x7fffffffll || !(eps > 0))
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(encoder_layernorm, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
        (const float*)x->data, (const float*)w->data, b ? (const float*)b->data : nullptr,
        (float*)y->data, (int)n, (int)xp, (int)yp, eps);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_encoder_gelu(const RadArgs* a, RadStream s) {
    const auto *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y || !is_f32(x) || !is_f32(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n < 1 || n > 0x7fffffffll) return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, ry = 0;
    const auto xp = r4d_rows_pitch(x, n, &rx), yp = r4d_rows_pitch(y, n, &ry);
    if (xp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    if (ry < 1 || rx < ry || ry > 0x7fffffffll / n || xp > 0x7fffffffll || yp > 0x7fffffffll)
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(encoder_gelu, dim3(blocks_for(ry * n)), dim3(kT), 0, (hipStream_t)s,
        (const float*)x->data, (float*)y->data, (int)ry, (int)n, (int)xp, (int)yp);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_encoder_rope(const RadArgs* a, RadStream s) {
    const auto *x = r4d_opt(a, 0), *p = r4d_opt(a, 1);
    if (!x || !p || !is_f32(x) || !is_i32(p)) return R4D_NO(RAD_E_DTYPE);
    const long long nh = r4d_geti_or(a, "n_head", 0), nk = r4d_geti_or(a, "n_head_kv", 0);
    const char* mode = r4d_gets(a, "mode");
    if (r4d_geti_or(a, "head_dim", 0) != 64 || r4d_geti_or(a, "rotary_dim", 64) != 64 ||
        nh < 1 || nh > 64 || nk != nh || !mode || std::strcmp(mode, "neox") || p->rank != 1 ||
        r4d_gets(a, "sections") || r4d_getf(a, "yarn_factor", 0) != 0 ||
        r4d_getf(a, "llama3_factor", 0) != 0) return R4D_NO(RAD_E_UNSUPPORTED);
    long long rx = 0;
    const auto xp = r4d_rows_pitch(x, 3 * nh * 64, &rx), rows = r4d_numel(p);
    if (xp < 0 || !r4d_contig(p)) return R4D_NO(RAD_E_STRIDE);
    const float theta = r4d_getf(a, "theta", 10000), scale = r4d_getf(a, "scale", 1);
    if (rows < 1 || rows > 65535 || rx < rows || xp > 0x7fffffffll || !(theta > 0) || !(scale > 0))
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(encoder_rope, dim3((unsigned)rows), dim3(kT), 0, (hipStream_t)s,
        (float*)x->data, (const int*)p->data, (int)(nh + nk), (int)xp, theta, scale);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_encoder_attn(const RadArgs* a, RadStream s) {
    const auto *q = r4d_opt(a, 0), *k = r4d_opt(a, 1), *v = r4d_opt(a, 2),
               *cu = r4d_opt(a, 3), *y = r4d_opt(a, 4);
    if (!q || !k || !v || !cu || !y || !is_f32(q) || !is_f32(k) || !is_f32(v) ||
        !is_f32(y) || !is_i32(cu)) return R4D_NO(RAD_E_DTYPE);
    if (r4d_geti_or(a, "head_dim", 0) != 64 || r4d_geti_or(a, "gqa", 1) != 1 ||
        r4d_geti_or(a, "causal", 0) != 1) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long n = r4d_dim(q, (int)q->rank - 1), div = r4d_geti_or(a, "cu_div", 1);
    if (n < 64 || n % 64 || n > 4096 || div < 1 || div > 0x7fffffffll) return R4D_NO(RAD_E_SHAPE);
    long long rq = 0, rk = 0, rv = 0, ry = 0;
    const auto qp = r4d_rows_pitch(q, n, &rq), kp = r4d_rows_pitch(k, n, &rk),
               vp = r4d_rows_pitch(v, n, &rv), yp = r4d_rows_pitch(y, n, &ry);
    if (qp < 0 || kp < 0 || vp < 0 || yp < 0 || !r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);
    const auto ns = r4d_numel(cu) - 1, window = r4d_geti_or(a, "window", 0);
    if (ry < 1 || ry > 512 || rq < ry || rk < ry || rv < ry || ns < 1 || ns > 512 ||
        qp > 0x7fffffffll || kp > 0x7fffffffll || vp > 0x7fffffffll || yp > 0x7fffffffll ||
        window < 0 || window > 0x7fffffffll) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(encoder_attn, dim3((unsigned)ry, (unsigned)(n / 64)), dim3(kT), 0,
        (hipStream_t)s, (const float*)q->data, (const float*)k->data, (const float*)v->data,
        (const int*)cu->data, (float*)y->data, (int)(n / 64), (int)qp, (int)kp, (int)vp, (int)yp,
        (int)ns, (int)div, (int)window, (float)r4d_getf(a, "scale", 0.125));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_elu(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y || !is_f32(x) || !is_f32(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long rx = 0, ry = 0;
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_rows_pitch(x, n, &rx), yp = r4d_rows_pitch(y, n, &ry);
    if (xp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || rx < ry || n > 0x7fffffffll || ry > 0x7fffffffll / n ||
        xp > 0x7fffffffll || yp > 0x7fffffffll) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(elu_f32_kernel, dim3(blocks_for(ry * n)), dim3(kT), 0,
                       (hipStream_t)s, (const float*)x->data, (float*)y->data,
                       (int)ry, (int)n, (int)xp, (int)yp);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_vq_euclidean(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *cb = r4d_opt(a, 1), *c = r4d_opt(a, 2);
    if (!x || !cb || !c || !is_f32(x) || !is_f32(cb) || !is_i32(c))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), bins = r4d_geti_or(a, "bins", 0);
    if (n <= 0 || bins <= 0) return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, rc = 0;
    const long long xp = r4d_rows_pitch(x, n, &rx), cp = r4d_rows_pitch(cb, n, &rc);
    if (xp < 0 || cp < 0) return R4D_NO(RAD_E_STRIDE);
    if (rx < 1 || rc != bins || r4d_numel(c) < rx || rx > 65535 || n > 4096 ||
        bins > 65536 || xp > 0x7fffffffll || cp > 0x7fffffffll)
        return R4D_NO(RAD_E_SHAPE);
    const long long cs = c->rank ? c->stride[c->rank - 1] : 1;
    if (cs < 1 || cs > 0x7fffffffll) return R4D_NO(RAD_E_STRIDE);
    hipLaunchKernelGGL(vq_euclidean_kernel, dim3((unsigned)rx), dim3(kT), 0,
                       (hipStream_t)s, (float*)x->data, (const float*)cb->data,
                       (int*)c->data, (int)n, (int)bins, (int)xp, (int)cp, (int)cs);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_conv1d_f32(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *cu = r4d_opt(a, 2);
    const RadTensor *res = r4d_opt(a, 3), *y = r4d_opt(a, 4), *b = r4d_opt(a, 5);
    if (!x || !w || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_f32(x) || !is_f32(w) || (b && !is_f32(b)) || !is_f32(y) || (res && !is_f32(res)) ||
        !is_i32(cu))
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(w) || (b && !r4d_contig(b)) || !r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);
    const long long cin = r4d_geti_or(a, "cin", 0), cout = r4d_geti_or(a, "cout", 0);
    const long long k = r4d_geti_or(a, "k", 0), dil = r4d_geti_or(a, "dil", 1);
    const long long causal = r4d_geti_or(a, "causal", 0), st = r4d_geti_or(a, "stride", 1);
    const long long cd = r4d_geti_or(a, "cu_div", 1);
    const int act = conv1d_act(r4d_gets(a, "act"));
    if (act < 0) return R4D_NO(RAD_E_INVAL);
    /* Explicit symmetric padding is still the bf16-only form. Causal strides share the
     * same loaders as stride 1 and retain each clip's own cu_div bounds. */
    if (r4d_geti_or(a, "pad", -1) >= 0)
        return R4D_NO(RAD_E_DTYPE);
    if (cin <= 0 || cout <= 0 || k <= 0 || (!causal && (k & 1) == 0) || dil <= 0 || causal < 0 ||
        causal > 1 || st < 1 || (st > 1 && (!causal || dil * (k - 1) + 1 < st)) || cd < 1)
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, ry = 0, rr = 0;
    const long long xl = r4d_rows_pitch(x, cin, &rx), yl = r4d_rows_pitch(y, cout, &ry);
    const long long rl = res ? r4d_rows_pitch(res, cout, &rr) : 0;
    if (xl < 0 || yl < 0 || rl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    /* The tile's loaders index x and w with 32-bit element offsets. */
    if (n_seq <= 0 || ry <= 0 || rx < ry * st || (res && rr < ry) ||
        r4d_numel(w) < cout * cin * k || (b && r4d_numel(b) < cout) || ry > 0x7fffffffll ||
        (ry + 127) / 128 > 65535 || ry * st * xl >= (1ll << 32) || cout * cin * k >= (1ll << 32))
        return R4D_NO(RAD_E_SHAPE);
    const int pad = (int)(causal ? dil * (k - 1) + 1 - st : dil * (k - 1) / 2);
    const float* bp = b ? (const float*)b->data : nullptr;
    const float* rp = res ? (const float*)res->data : nullptr;
    if (cout < 16) {
        hipLaunchKernelGGL(conv1d_f32_direct_kernel, dim3(blocks_for(ry * cout)), dim3(kT), 0,
                           (hipStream_t)s, (const float*)x->data, (const float*)w->data, bp, rp,
                           (const int*)cu->data, (float*)y->data, (int)n_seq, (int)ry, (int)cin,
                           (int)cout, (int)k, (int)dil, pad, (int)st, (int)cd, (int)xl, (int)yl,
                           (int)rl, act);
        return st_of(hipGetLastError());
    }
    const int tn = cout > 64 ? 8 : cout > 32 ? 4 : 2;
    const dim3 grid((unsigned)((cout + 16 * tn - 1) / (16 * tn)), (unsigned)((ry + 127) / 128));
#define C1F(TN)                                                                                    \
    hipLaunchKernelGGL((conv1d_f32_kernel<TN>), grid, dim3(r4d_f32::kThr), 0, (hipStream_t)s,      \
                       (const float*)x->data, (const float*)w->data, bp, rp,                       \
                       (const int*)cu->data, (float*)y->data, (int)n_seq, (int)ry, (int)cin,       \
                       (int)cout, (int)k, (int)dil, pad, (int)st, (int)cd, (int)xl, (int)yl,      \
                       (int)rl, act)
    if (tn == 8) C1F(8);
    else if (tn == 4) C1F(4);
    else C1F(2);
#undef C1F
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_conv_t1d_f32(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *b = r4d_opt(a, 2), *cu = r4d_opt(a, 3);
    const RadTensor *y = r4d_opt(a, 4);
    if (!x || !w || !b || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_f32(x) || !is_f32(w) || !is_f32(b) || !is_f32(y) || !is_i32(cu))
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(w) || !r4d_contig(b) || !r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);
    const long long cin = r4d_geti_or(a, "cin", 0), cout = r4d_geti_or(a, "cout", 0);
    const long long st = r4d_geti_or(a, "stride", 0);
    /* Any kernel of at least a stride (two strides by default); without `pad`, the symmetric
     * (k - stride) / 2, which needs k - stride even. A pad of a stride or more is served: the
     * phase's first input row just moves. */
    const long long kw = r4d_geti_or(a, "k", 2 * st), pad = r4d_geti_or(a, "pad", -1);
    const long long P = pad >= 0 ? pad : (kw - st) / 2;
    if (cin <= 0 || cout <= 0 || st <= 0 || kw < st || (pad < 0 && ((kw - st) & 1)) ||
        st > 65535)
        return R4D_NO(RAD_E_SHAPE);
    const int taps = (int)((kw + st - 1) / st);
    long long rx = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, cin, &rx), yl = r4d_rows_pitch(y, cout, &ry);
    if (xl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    /* The weight operand carries its declared [cin, cout, k]; its stored phase-major form (zero
     * taps included) is what the layout hook sized. */
    if (n_seq <= 0 || rx <= 0 || ry < rx * st || r4d_numel(w) < cin * cout * kw ||
        r4d_numel(b) < cout || rx > 0x7fffffffll || ry > 0x7fffffffll ||
        (rx + 127) / 128 > 65535 || rx * xl >= (1ll << 32) || st * cout * taps * cin >= (1ll << 32))
        return R4D_NO(RAD_E_SHAPE);
    const int tn = cout > 64 ? 8 : cout > 32 ? 4 : 2;
    const dim3 grid((unsigned)((cout + 16 * tn - 1) / (16 * tn)), (unsigned)((rx + 127) / 128),
                    (unsigned)st);
#define CTF(TN)                                                                                    \
    hipLaunchKernelGGL((conv_t1d_f32_kernel<TN>), grid, dim3(r4d_f32::kThr), 0, (hipStream_t)s,    \
                       (const float*)x->data, (const float*)w->data, (const float*)b->data,         \
                       (const int*)cu->data, (float*)y->data, (int)n_seq, (int)rx, (int)cin,       \
                       (int)cout, (int)st, (int)P, taps, (int)xl, (int)yl)
    if (tn == 8) CTF(8);
    else if (tn == 4) CTF(4);
    else CTF(2);
#undef CTF
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_snake(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *al = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    const RadTensor* be = r4d_opt(a, 3);
    if (!x || !al || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(y) || al->dtype != RAD_F32 || (be && be->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(al) || (be && !r4d_contig(be))) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_geti_or(a, "n", 0), lg = r4d_geti_or(a, "log", 0);
    long long rx = 0, ry = 0;
    const long long xl = n > 0 ? r4d_rows_pitch(x, n, &rx) : -1;
    const long long yl = n > 0 ? r4d_rows_pitch(y, n, &ry) : -1;
    if (xl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || rx < ry || r4d_numel(al) < n || (be && r4d_numel(be) < n) || lg < 0 || lg > 1)
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(snake_kernel, dim3(blocks_for(ry * n)), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (const float*)al->data,
                       be ? (const float*)be->data : nullptr, (uint16_t*)y->data, (int)n, (int)lg,
                       (int)xl, (int)yl, (int64_t)(ry * n));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_cfg_topk_sample(const RadArgs* a, RadStream s) {
    const RadTensor *lg = r4d_opt(a, 0), *seed = r4d_opt(a, 1), *fr = r4d_opt(a, 2);
    const RadTensor *ids = r4d_opt(a, 3), *cfgs = r4d_opt(a, 4), *lane = r4d_opt(a, 5);
    const RadTensor* hist = r4d_opt(a, 6);
    if (!lg || !seed || !fr || !ids) return R4D_NO(RAD_E_INVAL);
    if ((lg->dtype != RAD_F32 && lg->dtype != RAD_BF16) || !is_i32(ids) || !is_i32(fr) ||
        (cfgs && cfgs->dtype != RAD_F32) || (lane && !is_i32(lane)) || (hist && !is_i32(hist)))
        return R4D_NO(RAD_E_DTYPE);
    if ((cfgs && !r4d_contig(cfgs)) || (lane && !r4d_contig(lane)) || (hist && !r4d_contig(hist)))
        return R4D_NO(RAD_E_STRIDE);
    const long long V = r4d_geti_or(a, "n_vocab", 0), K = r4d_geti_or(a, "k", 0);
    const long long rep = r4d_geti_or(a, "rep", 1), ck = r4d_geti_or(a, "cond_k", 0);
    const long long stream = r4d_geti_or(a, "stream", 0);
    const long long sup_lo = r4d_geti_or(a, "sup_lo", 0), sup_hi = r4d_geti_or(a, "sup_hi", 0);
    const double cfg = r4d_getf(a, "cfg", 1.0), temp = r4d_getf(a, "temp", 1.0);
    const double pen = r4d_getf(a, "penalty", 1.0);
    /* As f32, as libref and libavx read it, so all three cut at the same value. */
    const float top_p = (float)r4d_getf(a, "top_p", 1.0);
    if (V <= 0 || K <= 0 || K > kMaxK || K > V || rep < 1 || rep > 2 || ck < 0 || ck > V ||
        !(temp > 0) || V > (1 << 16) * kT || sup_lo < 0 || sup_hi < sup_lo || sup_hi > V ||
        !(pen > 0) || (!hist != !lane) || !(top_p >= 0.f))
        return R4D_NO(RAD_E_SHAPE);
    long long rl = 0;
    const long long ll = r4d_rows_pitch(lg, V, &rl);
    if (ll < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_numel(ids) / rep;
    if (n <= 0 || rl < rep * n || r4d_numel(fr) < n || r4d_numel(seed) < 2 * n ||
        (cfgs && r4d_numel(cfgs) < n) || (lane && r4d_numel(lane) < n))
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(cfg_topk_kernel, dim3((unsigned)n), dim3(kT), 0, (hipStream_t)s, lg->data,
                       lg->dtype == RAD_BF16 ? 1 : 0, (int)ll, (const uint32_t*)seed->data,
                       (const int*)fr->data, (int*)ids->data, (int)V, (int)K, (int)rep, (int)ck,
                       (float)cfg, (float)temp, (uint64_t)stream,
                       cfgs ? (const float*)cfgs->data : nullptr,
                       lane ? (const int*)lane->data : nullptr, hist ? (int*)hist->data : nullptr,
                       hist ? (int)(r4d_numel(hist) / V) : 0, (int)sup_lo, (int)sup_hi,
                       (float)pen, top_p);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_ras_pick(const RadArgs* a, RadStream s) {
    const RadTensor *first = r4d_opt(a, 0), *alt = r4d_opt(a, 1), *lane = r4d_opt(a, 2);
    const RadTensor *hist = r4d_opt(a, 3), *ids = r4d_opt(a, 4);
    if (!first || !alt || !lane || !hist || !ids) return R4D_NO(RAD_E_INVAL);
    if (!is_i32(first) || !is_i32(alt) || !is_i32(lane) || !is_i32(hist) || !is_i32(ids))
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(first) || !r4d_contig(alt) || !r4d_contig(lane) || !r4d_contig(hist) ||
        !r4d_contig(ids))
        return R4D_NO(RAD_E_STRIDE);
    const long long W = r4d_geti_or(a, "win", 0), lo = r4d_geti_or(a, "lo", 0);
    const long long hi = r4d_geti_or(a, "hi", 0), rep = r4d_geti_or(a, "rep", 1);
    const long long clear = r4d_geti_or(a, "clear", 0);
    const long long n = r4d_numel(lane);
    if (W <= 0 || lo < 0 || hi < lo || rep < 1 || clear < 0 || clear > 1 || n <= 0 ||
        r4d_numel(first) < rep * n || r4d_numel(alt) < rep * n || r4d_numel(ids) < rep * n)
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(ras_pick_kernel, dim3(blocks_for(n)), dim3(kT), 0, (hipStream_t)s,
                       (const int*)first->data, (const int*)alt->data, (const int*)lane->data,
                       (int*)hist->data, (int*)ids->data, (int)n, (int)W, (int)lo, (int)hi,
                       (int)rep, (int)clear, (long long)r4d_numel(hist));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_rvq_embed(const RadArgs* a, RadStream s) {
    const RadTensor *codes = r4d_opt(a, 0), *t0 = r4d_opt(a, 1), *t1 = r4d_opt(a, 2);
    const RadTensor *y = r4d_opt(a, 3);
    if (!codes || !t0 || !t1 || !y) return R4D_NO(RAD_E_INVAL);
    auto bf_or_f32 = [](const RadTensor* t) { return t->dtype == RAD_BF16 || t->dtype == RAD_F32; };
    if (!is_i32(codes) || !bf_or_f32(t0) || !bf_or_f32(t1) || !bf_or_f32(y))
        return R4D_NO(RAD_E_DTYPE);
    const long long D = r4d_geti_or(a, "n_embd", 0), K = r4d_geti_or(a, "n_codes", 0);
    const long long voc = r4d_geti_or(a, "vocab", 0), off0 = r4d_geti_or(a, "off0", 0);
    const long long round1 = r4d_geti_or(a, "round", 0), clamp = r4d_geti_or(a, "clamp", 0);
    const double scale = r4d_getf(a, "scale", 1.0);
    if (D <= 0 || K < 1 || voc <= 0 || round1 < 0 || round1 > 1 || clamp < 0)
        return R4D_NO(RAD_E_SHAPE);
    long long ry = 0;
    const long long yl = r4d_rows_pitch(y, D, &ry);
    if (yl < 0 || !r4d_contig(t0) || !r4d_contig(t1)) return R4D_NO(RAD_E_STRIDE);
    /* Code (k, r) at the plane's own strides (libref reads it so). */
    const long long kp = codes->rank >= 2 ? codes->stride[codes->rank - 2] : ry;
    const long long rp = codes->rank >= 1 ? codes->stride[codes->rank - 1] : 1;
    if (ry <= 0 || r4d_numel(codes) < K * ry) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(rvq_embed_kernel, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
                       (const int*)codes->data, (int64_t)kp, (int64_t)rp, t0->data,
                       t0->dtype == RAD_F32 ? 1 : 0, t1->data, t1->dtype == RAD_F32 ? 1 : 0,
                       y->data, y->dtype == RAD_F32 ? 1 : 0, (int)D, (int)K, (int)voc, (int)off0,
                       (int)(r4d_numel(t0) / D), (int)(r4d_numel(t1) / D), (float)scale, (int)yl,
                       (int)round1, (int)clamp);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_vq_encode(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *iw = r4d_opt(a, 1), *ib = r4d_opt(a, 2);
    const RadTensor *cb = r4d_opt(a, 3), *ow = r4d_opt(a, 4), *ob = r4d_opt(a, 5);
    const RadTensor* codes = r4d_opt(a, 6);
    if (!x || !iw || !ib || !cb || !ow || !ob || !codes) return R4D_NO(RAD_E_INVAL);
    auto bf_or_f32 = [](const RadTensor* t) { return t->dtype == RAD_BF16 || t->dtype == RAD_F32; };
    if (!is_i32(codes) || !bf_or_f32(x) || !bf_or_f32(iw) || !bf_or_f32(ib) || !bf_or_f32(cb) ||
        !bf_or_f32(ow) || !bf_or_f32(ob))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), dim = r4d_geti_or(a, "dim", 0);
    const long long bins = r4d_geti_or(a, "bins", 0);
    if (n <= 0 || dim <= 0 || dim > kVqMaxDim || bins <= 0) return R4D_NO(RAD_E_SHAPE);
    long long rx = 0;
    const long long xl = r4d_rows_pitch(x, n, &rx);
    if (xl < 0 || !r4d_contig(iw) || !r4d_contig(ib) || !r4d_contig(cb) || !r4d_contig(ow) ||
        !r4d_contig(ob))
        return R4D_NO(RAD_E_STRIDE);
    if (rx <= 0 || r4d_numel(codes) < rx || r4d_numel(iw) < dim * n || r4d_numel(ib) < dim ||
        r4d_numel(cb) < bins * dim || r4d_numel(ow) < n * dim || r4d_numel(ob) < n ||
        rx > 0x7fffffffll)
        return R4D_NO(RAD_E_SHAPE);
    /* A code at the operand's own stride: a row of a code-major plane, or a column of one. */
    const long long cs = codes->rank >= 1 ? codes->stride[codes->rank - 1] : 1;
    auto f32 = [](const RadTensor* t) { return t->dtype == RAD_F32 ? 1 : 0; };
    hipLaunchKernelGGL(vq_encode_kernel, dim3((unsigned)rx), dim3(kT), 0, (hipStream_t)s, x->data,
                       f32(x), (int)xl, iw->data, f32(iw), ib->data, f32(ib), cb->data, f32(cb),
                       ow->data, f32(ow), ob->data, f32(ob), (int*)codes->data, (int64_t)cs,
                       (int)n, (int)dim, (int)bins);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_frame_mix_store(const RadArgs* a, RadStream s) {
    const RadTensor *hid = r4d_opt(a, 0), *lw = r4d_opt(a, 1), *sc = r4d_opt(a, 2);
    const RadTensor *lane = r4d_opt(a, 3), *fr = r4d_opt(a, 4), *st = r4d_opt(a, 5);
    const RadTensor *dst = r4d_opt(a, 6);
    if (!hid || !lw || !sc || !lane || !fr || !st || !dst) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(hid) || !is_bf16(dst) || lw->dtype != RAD_F32 || sc->dtype != RAD_F32 ||
        !is_i32(lane) || !is_i32(fr) || !is_i32(st))
        return R4D_NO(RAD_E_DTYPE);
    const long long D = r4d_geti_or(a, "n_embd", 0), L = r4d_geti_or(a, "L", 0);
    const long long stride = r4d_geti_or(a, "stride", 0), rep = r4d_geti_or(a, "rep", 1);
    if (D <= 0 || L <= 0 || L > 16 || stride <= 0 || rep < 1) return R4D_NO(RAD_E_SHAPE);
    long long rh = 0, rd = 0;
    const long long hl = r4d_rows_pitch(hid, D, &rh), dl = r4d_rows_pitch(dst, D, &rd);
    if (hl < 0 || dl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_numel(lane);
    if (n <= 0 || rh / L < rep * n || r4d_numel(fr) < n || r4d_numel(lw) < L)
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(frame_mix_kernel, dim3((unsigned)n), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)hid->data, (int)(rh / L), (const float*)lw->data,
                       (const float*)sc->data, (const int*)lane->data, (const int*)fr->data,
                       (const int*)st->data, (uint16_t*)dst->data, (int)D, (int)L, (int)stride,
                       (int)rep, (int)hl, (int)dl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_end_status(const RadArgs* a, RadStream s) {
    const RadTensor *ids = r4d_opt(a, 0), *lane = r4d_opt(a, 1), *st = r4d_opt(a, 2);
    if (!ids || !lane || !st) return R4D_NO(RAD_E_INVAL);
    if (!is_i32(ids) || !is_i32(lane) || !is_i32(st)) return R4D_NO(RAD_E_DTYPE);
    const long long end = r4d_geti_or(a, "end", -1), rep = r4d_geti_or(a, "rep", 1);
    const long long n = r4d_numel(lane);
    if (n <= 0 || rep < 1 || r4d_numel(ids) < rep * n) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(end_status_kernel, dim3(blocks_for(n)), dim3(kT), 0, (hipStream_t)s,
                       (const int*)ids->data, (const int*)lane->data, (int*)st->data, (int)n,
                       (int)rep, (int)end);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_frame_store(const RadArgs* a, RadStream s) {
    const RadTensor *codes = r4d_opt(a, 0), *lane = r4d_opt(a, 1), *fr = r4d_opt(a, 2);
    const RadTensor *st = r4d_opt(a, 3), *dst = r4d_opt(a, 4);
    if (!codes || !lane || !fr || !st || !dst) return R4D_NO(RAD_E_INVAL);
    if (!is_i32(codes) || !is_i32(lane) || !is_i32(fr) || !is_i32(st) || !is_i32(dst))
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(lane) || !r4d_contig(fr) || !r4d_contig(st) || !r4d_contig(dst))
        return R4D_NO(RAD_E_STRIDE);
    const long long K = r4d_geti_or(a, "n_codes", 0), stride = r4d_geti_or(a, "stride", 0);
    const long long rep = r4d_geti_or(a, "rep", 1), skip = r4d_geti_or(a, "skip", 0);
    if (K <= 0 || stride <= 0 || rep < 1 || skip < 0) return R4D_NO(RAD_E_SHAPE);
    const long long n = r4d_numel(lane);
    if (n <= 0 || r4d_numel(fr) < n || r4d_numel(codes) < K * rep * n) return R4D_NO(RAD_E_SHAPE);
    const long long kp = codes->rank >= 2 ? codes->stride[codes->rank - 2] : rep * n;
    const long long rp = codes->rank >= 1 ? codes->stride[codes->rank - 1] : 1;
    hipLaunchKernelGGL(frame_store_kernel, dim3(blocks_for(n * K)), dim3(kT), 0, (hipStream_t)s,
                       (const int*)codes->data, (int64_t)kp, (int64_t)rp, (const int*)lane->data,
                       (const int*)fr->data, (const int*)st->data, (int*)dst->data, (int)n, (int)K,
                       (int)stride, (int)rep, (int)skip, (int)r4d_numel(st),
                       (int64_t)r4d_numel(dst));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_resample_nearest(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != y->dtype || (x->dtype != RAD_BF16)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long rx = 0, ry = 0;
    const long long xl = n > 0 ? r4d_rows_pitch(x, n, &rx) : -1;
    const long long yl = n > 0 ? r4d_rows_pitch(y, n, &ry) : -1;
    if (xl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    if (rx <= 0 || ry <= 0) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(resample_kernel, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (uint16_t*)y->data, (int)n, (int)rx, (int)ry,
                       (int)xl, (int)yl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_fourier_embed(const RadArgs* a, RadStream s) {
    const RadTensor *t = r4d_opt(a, 0), *w = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!t || !w || !y) return R4D_NO(RAD_E_INVAL);
    if (t->dtype != RAD_F32 || w->dtype != RAD_F32 || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), rnd = r4d_geti_or(a, "round", 0);
    long long ry = 0;
    const long long yl = n > 0 ? r4d_rows_pitch(y, n, &ry) : -1;
    if (yl < 0) return R4D_NO(RAD_E_STRIDE);
    if (n <= 0 || (n & 1) || ry <= 0 || r4d_numel(t) < ry || r4d_numel(w) < n / 2)
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(fourier_kernel, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
                       (const float*)t->data, (const float*)w->data, (uint16_t*)y->data,
                       (int)(n / 2), (int)rnd, (int)yl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_win_input(const RadArgs* a, RadStream s) {
    const RadTensor *lat = r4d_opt(a, 0), *cond = r4d_opt(a, 1), *lcu = r4d_opt(a, 2);
    const RadTensor *wl = r4d_opt(a, 3), *pair = r4d_opt(a, 4), *y = r4d_opt(a, 5);
    if (!lat || !cond || !lcu || !wl || !pair || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(lat) || !is_bf16(cond) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long C = r4d_geti_or(a, "n_lat", 0), Z = r4d_geti_or(a, "n_zero", 0);
    const long long D = r4d_geti_or(a, "n_cond", 0), stride = r4d_geti_or(a, "stride", 0);
    if (C <= 0 || Z < 0 || D <= 0 || stride <= 0) return R4D_NO(RAD_E_SHAPE);
    long long rl = 0, rc = 0, ry = 0;
    const long long ll = r4d_rows_pitch(lat, C, &rl), cl = r4d_rows_pitch(cond, D, &rc);
    const long long yl = r4d_rows_pitch(y, C + Z + D, &ry);
    if (ll < 0 || cl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long S = r4d_numel(lcu) - 1;
    if (S <= 0 || ry <= 0 || r4d_numel(wl) < S || r4d_numel(pair) < S) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(win_input_kernel, dim3((unsigned)ry), dim3(kT), 0, (hipStream_t)s,
                       (const uint16_t*)lat->data, (const uint16_t*)cond->data,
                       (const int*)lcu->data, (const int*)wl->data, (const int*)pair->data,
                       (uint16_t*)y->data, (int)S, (int)C, (int)Z, (int)D, (int)stride, (int)ll,
                       (int)cl, (int)yl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_win_prep(const RadArgs* a, RadStream s) {
    const RadTensor *fl = r4d_opt(a, 0), *t = r4d_opt(a, 1), *seed = r4d_opt(a, 2);
    const RadTensor *widx = r4d_opt(a, 3), *wl = r4d_opt(a, 4), *pair = r4d_opt(a, 5);
    const RadTensor *lcu = r4d_opt(a, 6), *prev = r4d_opt(a, 7), *noise = r4d_opt(a, 8);
    const RadTensor *lat = r4d_opt(a, 9);
    if (!fl || !t || !seed || !widx || !wl || !pair || !lcu || !prev || !noise || !lat)
        return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(prev) || !is_bf16(noise) || !is_bf16(lat) || t->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), stride = r4d_geti_or(a, "stride", 0);
    const long long ov = r4d_geti_or(a, "ov", 0);
    if (n <= 0 || stride <= 0 || ov < 0 || ov > stride) return R4D_NO(RAD_E_SHAPE);
    long long rl = 0, rp = 0, rn = 0;
    const long long ll = r4d_rows_pitch(lat, n, &rl), pl = r4d_rows_pitch(prev, n, &rp);
    const long long nl = r4d_rows_pitch(noise, n, &rn);
    if (ll < 0 || pl < 0 || nl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long S = r4d_numel(lcu) - 1;
    if (S <= 0 || rl < stride) return R4D_NO(RAD_E_SHAPE);
    for (const RadTensor* e : { fl, t, widx, wl, pair })
        if (r4d_numel(e) < S) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(win_prep_kernel, dim3(blocks_for(stride * n), (unsigned)S), dim3(kT), 0,
                       (hipStream_t)s, (const int*)fl->data, (const float*)t->data,
                       (const uint32_t*)seed->data, (const int*)widx->data, (const int*)wl->data,
                       (const int*)pair->data, (const int*)lcu->data, (const uint16_t*)prev->data,
                       (uint16_t*)noise->data, (uint16_t*)lat->data, (int)n, (int)stride, (int)ov,
                       (int)ll, (int)pl, (int)nl);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_win_finish(const RadArgs* a, RadStream s) {
    const RadTensor *fl = r4d_opt(a, 0), *pair = r4d_opt(a, 1), *wl = r4d_opt(a, 2);
    const RadTensor *lcu = r4d_opt(a, 3), *cond = r4d_opt(a, 4), *lat = r4d_opt(a, 5);
    const RadTensor *prev = r4d_opt(a, 6), *pc = r4d_opt(a, 7);
    if (!fl || !pair || !wl || !lcu || !cond || !lat || !prev || !pc) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(cond) || !is_bf16(lat) || !is_bf16(prev) || !is_bf16(pc))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), D = r4d_geti_or(a, "n_cond", 0);
    const long long stride = r4d_geti_or(a, "stride", 0), ov = r4d_geti_or(a, "ov", 0);
    if (n <= 0 || D <= 0 || stride <= 0 || ov <= 0 || ov > stride) return R4D_NO(RAD_E_SHAPE);
    long long rl = 0, rc = 0, rp = 0, rq = 0;
    const long long ll = r4d_rows_pitch(lat, n, &rl), cl = r4d_rows_pitch(cond, D, &rc);
    const long long pl = r4d_rows_pitch(prev, n, &rp), ql = r4d_rows_pitch(pc, D, &rq);
    if (ll < 0 || cl < 0 || pl < 0 || ql < 0) return R4D_NO(RAD_E_STRIDE);
    const long long S = r4d_numel(lcu) - 1;
    if (S <= 0 || r4d_numel(fl) < S || r4d_numel(wl) < S || r4d_numel(pair) < S)
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(win_join_kernel, dim3(blocks_for(ov * n), (unsigned)S), dim3(kT), 0,
                       (hipStream_t)s, (const int*)fl->data, (const int*)pair->data,
                       (const int*)wl->data, (const int*)lcu->data, (const uint16_t*)prev->data,
                       (uint16_t*)lat->data, (int)n, (int)stride, (int)ov, (int)ll, (int)pl);
    int rc2 = st_of(hipGetLastError());
    if (rc2 < 0) return rc2;
    hipLaunchKernelGGL(win_handon_kernel, dim3(blocks_for(ov * (n + D)), (unsigned)S), dim3(kT), 0,
                       (hipStream_t)s, (const int*)fl->data, (const int*)pair->data,
                       (const int*)wl->data, (const int*)lcu->data, (const uint16_t*)cond->data,
                       (const uint16_t*)lat->data, (uint16_t*)prev->data, (uint16_t*)pc->data,
                       (int)n, (int)D, (int)stride, (int)ov, (int)ll, (int)cl, (int)pl, (int)ql);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_chan_split(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), P = r4d_geti_or(a, "parts", 0);
    if (n <= 0 || P <= 0) return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, P * n, &rx), yl = r4d_rows_pitch(y, n, &ry);
    if (xl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long rows = ry / P;
    if (rows <= 0 || rx < rows) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(chan_split_kernel, dim3(blocks_for(P * rows * n)), dim3(kT), 0,
                       (hipStream_t)s, (const uint16_t*)x->data, (uint16_t*)y->data, (int)n, (int)P,
                       (int)rows, (int)xl, (int)yl, (int64_t)(P * rows * n));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_interleave2(const RadArgs* a, RadStream s) {
    const RadTensor *x0 = r4d_opt(a, 0), *x1 = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x0 || !x1 || !y) return R4D_NO(RAD_E_INVAL);
    /* Both streams bf16 or both f32; the samples f32. */
    if ((!is_bf16(x0) && x0->dtype != RAD_F32) || x1->dtype != x0->dtype || y->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(x0) || !r4d_contig(x1) || !r4d_contig(y)) return R4D_NO(RAD_E_STRIDE);
    const long long rows = r4d_numel(y) / 2;
    if (rows <= 0 || r4d_numel(x0) < rows || r4d_numel(x1) < rows) return R4D_NO(RAD_E_SHAPE);
    if (x0->dtype == RAD_F32)
        hipLaunchKernelGGL(interleave2_kernel<float>, dim3(blocks_for(rows)), dim3(kT), 0,
                           (hipStream_t)s, (const float*)x0->data, (const float*)x1->data,
                           (float*)y->data, (int64_t)rows);
    else
        hipLaunchKernelGGL(interleave2_kernel<uint16_t>, dim3(blocks_for(rows)), dim3(kT), 0,
                           (hipStream_t)s, (const uint16_t*)x0->data, (const uint16_t*)x1->data,
                           (float*)y->data, (int64_t)rows);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_attn_dense_small_h256(const RadArgs* a, RadStream s) {
    const RadTensor *q = r4d_opt(a, 0), *k = r4d_opt(a, 1), *v = r4d_opt(a, 2), *cu = r4d_opt(a, 3);
    const RadTensor *o = r4d_opt(a, 4);
    if (!q || !k || !v || !cu || !o) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(q) || !is_bf16(k) || !is_bf16(v) || !is_bf16(o) || !is_i32(cu))
        return R4D_NO(RAD_E_DTYPE);
    const long long hd = r4d_geti_or(a, "head_dim", 0), gqa = r4d_geti_or(a, "gqa", 1);
    const long long causal = r4d_geti_or(a, "causal", 0);
    if (hd != 256 || gqa != 1) return R4D_NO(RAD_E_SHAPE);
    /* Every key of a sequence, causally or not: no window, blocks or cu_div. */
    if (r4d_geti_or(a, "window", 0) != 0 || r4d_geti_or(a, "window_right", 0) != 0 ||
        r4d_geti_or(a, "block", 0) != 0 || r4d_geti_or(a, "cu_div", 1) != 1)
        return R4D_NO(RAD_E_UNSUPPORTED);
    const long long width = q->rank >= 2 ? q->shape[q->rank - 1] : 0;
    if (width <= 0 || width % 256) return R4D_NO(RAD_E_SHAPE);
    const long long n_head = width / 256;
    long long rq = 0, rk = 0, rv = 0, ro = 0;
    const long long ql = r4d_rows_pitch(q, width, &rq), kl = r4d_rows_pitch(k, width, &rk);
    const long long vl = r4d_rows_pitch(v, width, &rv), ol = r4d_rows_pitch(o, width, &ro);
    if (ql < 0 || kl < 0 || vl < 0 || ol < 0 || (ql % 8) || (kl % 8) || (vl % 8) || (ol % 8))
        return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (n_seq <= 0) return R4D_NO(RAD_E_SHAPE);
    const double sc = r4d_getf(a, "scale", 1.0 / std::sqrt(256.0));
    hipLaunchKernelGGL(attn_small_h256_kernel, dim3((unsigned)n_seq, (unsigned)n_head), dim3(kT), 0,
                       (hipStream_t)s, (const uint16_t*)q->data, (const uint16_t*)k->data,
                       (const uint16_t*)v->data, (const int*)cu->data, (uint16_t*)o->data, (int)ql,
                       (int)kl, (int)vl, (int)ol, (float)sc, (int)causal);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_iota(const RadArgs* a, RadStream s) {
    const RadTensor* y = r4d_opt(a, 0);
    if (!y) return R4D_NO(RAD_E_INVAL);
    if (!is_i32(y)) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(y)) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_numel(y), period = r4d_geti_or(a, "period", 0);
    if (n <= 0 || period < 0) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(iota_kernel, dim3(blocks_for(n)), dim3(kT), 0, (hipStream_t)s, (int*)y->data,
                       (int)n, (int)r4d_geti_or(a, "start", 0), (int)r4d_geti_or(a, "step", 1),
                       (int)period);
    return st_of(hipGetLastError());
}

namespace {
int parse_ints(const char* s, int* v, int cap) {
    int n = 0;
    while (s && *s) {
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        if (n == cap) return -1;
        v[n++] = (int)x;
        s = e;
    }
    return n;
}
}  // namespace

extern "C" int r4d_rad_int_affine(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_i32(x) || !is_i32(y)) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(x) || !r4d_contig(y)) return R4D_NO(RAD_E_STRIDE);
    IntAffine p{};
    const int n = parse_ints(r4d_gets(a, "src"), p.src, kAffMax);
    if (n <= 0 || parse_ints(r4d_gets(a, "mul"), p.mul, kAffMax) != n ||
        parse_ints(r4d_gets(a, "add"), p.add, kAffMax) != n || r4d_numel(y) < n)
        return R4D_NO(RAD_E_SHAPE);
    for (int j = 0; j < n; ++j)
        if (p.src[j] >= r4d_numel(x)) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(int_affine_kernel, dim3(1), dim3(kAffMax), 0, (hipStream_t)s,
                       (const int*)x->data, (int*)y->data, p, n);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_sub_i32(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *sc = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !sc || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_i32(x) || !is_i32(sc) || !is_i32(y)) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(x) || !r4d_contig(y)) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_numel(y);
    if (n <= 0 || r4d_numel(x) < n || r4d_numel(sc) < 1) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(sub_i32_kernel, dim3(blocks_for(n)), dim3(kT), 0, (hipStream_t)s,
                       (const int*)x->data, (const int*)sc->data, (int*)y->data, (int)n);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_cu_repeat(const RadArgs* a, RadStream s) {
    const RadTensor *len = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!len || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_i32(len) || !is_i32(y)) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(y)) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_numel(y), mult = r4d_geti_or(a, "mult", 1);
    if (n <= 0 || mult <= 0 || r4d_numel(len) < 1) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(cu_repeat_kernel, dim3(blocks_for(n)), dim3(kT), 0, (hipStream_t)s,
                       (const int*)len->data, (int*)y->data, (int)n, (int)mult);
    return st_of(hipGetLastError());
}
