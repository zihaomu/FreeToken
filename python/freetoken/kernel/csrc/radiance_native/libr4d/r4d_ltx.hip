// R4D -- LTX-2's ops (docs/OPS.md, "LTX-2"): its rotary over fractional coordinates, the per-head
// attention gate and the guided sampler step.
//
// Each kernel rounds where its libref function does (libref/ref_ltx.cpp). The rotary's angles take
// the device's sincosf, within two units in the last place of libref's correctly rounded cos and
// sin; the guided step's deviations are f64 block sums in a fixed order, so it is deterministic.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kT = 256;

__device__ __forceinline__ float bfr(float v) { return bf2f(f32_to_bf16(v)); }

// A value stored to an operand of dtype `bf`: bf16 when set, else f32.
__device__ __forceinline__ float ldx(const void* p, size_t i, bool bf) {
    return bf ? bf2f(((const uint16_t*)p)[i]) : ((const float*)p)[i];
}
__device__ __forceinline__ void stx(void* p, size_t i, float v, bool bf) {
    if (bf) ((uint16_t*)p)[i] = f32_to_bf16(v);
    else    ((float*)p)[i] = v;
}

// ---------------------------------------------------------------------------- rope_split
// One workgroup a (head, run of kRopeRows rows): the head's half-width pairs across the threads,
// their frequency and axis worked out once into LDS (f64 pow, as libref), then the rows.
constexpr int kRopeRows = 64;
constexpr int kMaxDims = 4;

__global__ __launch_bounds__(kT) void rope_split_kernel(
    void* __restrict__ x, const void* __restrict__ pos, int rows, int nh, int hd, int head0,
    int nd, int steps, int pad, double theta, float m0, float m1, float m2, float m3, long long x_ld,
    long long p_ld, long long p_cs, bool xbf, bool pint) {
    __shared__ float freq[256];
    __shared__ int axis[256];
    const int half = hd / 2, h = blockIdx.y, r0 = blockIdx.x * kRopeRows;
    if ((int)threadIdx.x < half) {
        const int p = (head0 + h) * half + (int)threadIdx.x;
        if (p < pad) {
            axis[threadIdx.x] = -1;
            freq[threadIdx.x] = 0.0f;
        } else {
            const int j = p - pad, k = j / nd;
            axis[threadIdx.x] = j % nd;
            double l = 0.0;
            if (steps > 1) {
                const double st = 1.0 / (double)(steps - 1);
                l = k < steps / 2 ? st * (double)k : 1.0 - st * (double)(steps - 1 - k);
            }
            freq[threadIdx.x] = (float)(pow(theta, l) * M_PI / 2.0);
        }
    }
    __syncthreads();
    const float mx[kMaxDims] = { m0, m1, m2, m3 };
    const int per = kT / half;   // rows a pass of the workgroup covers
    const int i = (int)threadIdx.x % half, rr = (int)threadIdx.x / half;
    if (rr >= per) return;
    const int d = axis[i];
    if (d < 0) return;   // cos 1, sin 0: the value as it is
    const float f = freq[i];
    for (int r = r0 + rr; r < min(r0 + kRopeRows, rows); r += per) {
        const long long pi = (long long)d * p_cs + (long long)r * p_ld;
        const float c0 = pint ? (float)((const int*)pos)[pi] : ((const float*)pos)[pi];
        const float g = c0 / mx[d];
        const float ang = (2.0f * g - 1.0f) * f;
        float s, c;
        sincosf(ang, &s, &c);
        const size_t o1 = (size_t)r * x_ld + (size_t)h * hd + i, o2 = o1 + half;
        const float x1 = ldx(x, o1, xbf), x2 = ldx(x, o2, xbf);
        const float a1 = x1 * c, b1 = x2 * s, a2 = x2 * c, b2 = x1 * s;
        stx(x, o1, a1 - b1, xbf);
        stx(x, o2, a2 + b2, xbf);
    }
}

// ---------------------------------------------------------------------------- head_gate
// x = rd(x rd(2 rd(sigmoid(g)))): a thread an element.
__global__ __launch_bounds__(kT) void head_gate_kernel(
    void* __restrict__ x, const uint16_t* __restrict__ g, long long total, int width, int hd,
    long long x_ld, long long g_ld, bool xbf) {
    const long long e = (long long)blockIdx.x * kT + threadIdx.x;
    if (e >= total) return;
    const long long r = e / width;
    const int col = (int)(e % width), h = col / hd;
    const float sg = bfr(1.0f / (1.0f + expf(-bf2f(g[r * g_ld + h]))));
    const float gate = xbf ? bfr(2.0f * sg) : 2.0f * sg;
    const size_t o = (size_t)r * x_ld + col;
    stx(x, o, ldx(x, o, xbf) * gate, xbf);
}

// ---------------------------------------------------------------------------- ltx_guide
// Two launches. The first, when the rescale is on (read off `scales` on the device), sums x0_c,
// x0_c^2, g and g^2 in f64 a workgroup a part into `stats`; the second reduces the parts (each
// workgroup the same sums in the same order), then recomputes g from the passes, rescales it, and
// takes the Euler step in place.
constexpr int kGuideParts = 1024;

struct GuideIn {
    const void* vc; const void* vu; const void* vs; const void* vm;
    long long lc, lu, ls, lm;   // row pitches
    bool bc, bu, bs, bm;        // bf16
};

__device__ __forceinline__ float guide_g(const GuideIn& v, long long r, int c, float x, float sig,
                                         float cu, float stg, float cm, bool uu, bool us, bool um,
                                         float* x0c) {
    const float xc = x - ldx(v.vc, (size_t)(r * v.lc + c), v.bc) * sig;
    float g = xc;
    if (uu) g = g + cu * (xc - (x - ldx(v.vu, (size_t)(r * v.lu + c), v.bu) * sig));
    if (us) g = g + stg * (xc - (x - ldx(v.vs, (size_t)(r * v.ls + c), v.bs) * sig));
    if (um) g = g + cm * (xc - (x - ldx(v.vm, (size_t)(r * v.lm + c), v.bm) * sig));
    *x0c = xc;
    return g;
}

__global__ __launch_bounds__(kT) void ltx_guide_sum_kernel(
    GuideIn v, const float* __restrict__ x, const float* __restrict__ sc,
    const float* __restrict__ sh, double* __restrict__ parts, long long rows, int n, long long x_ld) {
    __shared__ double red[4][kT];
    const float gs = sc[0], stg = sc[1], ms = sc[2], rr = sc[3], sig = sh[0];
    const bool uu = v.vu && gs != 1.0f, us = v.vs && stg != 0.0f, um = v.vm && ms != 1.0f;
    double a[4] = { 0, 0, 0, 0 };
    if (rr > 0.0f) {
        const long long total = rows * n;
        for (long long e = (long long)blockIdx.x * kT + threadIdx.x; e < total;
             e += (long long)gridDim.x * kT) {
            const long long r = e / n;
            const int c = (int)(e % n);
            float xc;
            const float g = guide_g(v, r, c, x[r * x_ld + c], sig, gs - 1.0f, stg, ms - 1.0f, uu, us,
                                    um, &xc);
            a[0] += xc;
            a[1] += (double)xc * xc;
            a[2] += g;
            a[3] += (double)g * g;
        }
    }
    for (int k = 0; k < 4; ++k) red[k][threadIdx.x] = a[k];
    __syncthreads();
    for (int s = kT / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            for (int k = 0; k < 4; ++k) red[k][threadIdx.x] += red[k][threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x < 4) parts[(size_t)blockIdx.x * 4 + threadIdx.x] = red[threadIdx.x][0];
}

__global__ __launch_bounds__(kT) void ltx_guide_step_kernel(
    GuideIn v, float* __restrict__ x, const float* __restrict__ sc, const float* __restrict__ sh,
    const double* __restrict__ parts, int n_parts, long long rows, int n, long long x_ld) {
    __shared__ double red[4][kT];
    __shared__ float fac_s;
    const float gs = sc[0], stg = sc[1], ms = sc[2], rr = sc[3], sig = sh[0], sn = sh[1];
    const bool uu = v.vu && gs != 1.0f, us = v.vs && stg != 0.0f, um = v.vm && ms != 1.0f;
    float fac = 1.0f;
    if (rr > 0.0f) {
        double a[4] = { 0, 0, 0, 0 };
        for (int p = threadIdx.x; p < n_parts; p += kT)
            for (int k = 0; k < 4; ++k) a[k] += parts[(size_t)p * 4 + k];
        for (int k = 0; k < 4; ++k) red[k][threadIdx.x] = a[k];
        __syncthreads();
        for (int s = kT / 2; s > 0; s >>= 1) {
            if ((int)threadIdx.x < s)
                for (int k = 0; k < 4; ++k) red[k][threadIdx.x] += red[k][threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const double N = (double)(rows * n), dof = N > 1.0 ? N - 1.0 : 1.0;
            const double vc = (red[1][0] - red[0][0] * red[0][0] / N) / dof;
            const double vg = (red[3][0] - red[2][0] * red[2][0] / N) / dof;
            fac_s = (float)sqrt(vc > 0.0 ? vc : 0.0) / (float)sqrt(vg > 0.0 ? vg : 0.0);
        }
        __syncthreads();
        fac = fac_s;
    }
    const float kept = (float)(1.0 - (double)rr), dt = sn - sig;
    const long long total = rows * n;
    for (long long e = (long long)blockIdx.x * kT + threadIdx.x; e < total;
         e += (long long)gridDim.x * kT) {
        const long long r = e / n;
        const int c = (int)(e % n);
        float* xp = x + r * x_ld + c;
        const float xv = *xp;
        float xc;
        float g = guide_g(v, r, c, xv, sig, gs - 1.0f, stg, ms - 1.0f, uu, us, um, &xc);
        if (rr > 0.0f) g = rr * (g * fac) + kept * g;
        const float vel = (xv - g) / sig;
        *xp = xv + dt * vel;
    }
}

// ---------------------------------------------------------------------------- pixel_norm
// A wave a row, eight rows a workgroup: the lanes' strided sums of rd(x^2) reduced by the wave
// (a different f32 order from libref's ascending one), then the row scaled. `ileave` L sends
// row t L + l to columns c L + l of y's row t.
__device__ __forceinline__ float rdx(float v, bool bf) { return bf ? bfr(v) : v; }

__global__ __launch_bounds__(kT) void pixel_norm_kernel(
    const void* __restrict__ x, void* __restrict__ y, long long rows, int n, int L, float eps,
    float mult, long long x_ld, long long y_ld, bool xbf, bool ybf, bool divf, bool silu) {
    const long long r = (long long)blockIdx.x * (kT / 32) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    const size_t xb = (size_t)r * x_ld;
    float ss = 0.0f;
    for (int i = lane; i < n; i += 32) {
        const float v = ldx(x, xb + i, xbf);
        ss += rdx(v * v, ybf);
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    const float v = rdx(rdx(ss / (float)n, ybf) + eps, ybf);
    const float s = divf ? rdx(sqrtf(v), ybf) : rdx(1.0f / sqrtf(v), ybf);
    const size_t yb = (size_t)(r / L) * y_ld + (size_t)(r % L);
    for (int i = lane; i < n; i += 32) {
        const float xv = ldx(x, xb + i, xbf);
        float z = rdx(divf ? xv / s : xv * s, ybf);
        if (silu) z = rdx(z / (1.0f + expf(-z)), ybf);
        if (mult != 1.0f) z = rdx(z * mult, ybf);
        stx(y, yb + (size_t)i * L, z, ybf);
    }
}

// ---------------------------------------------------------------------------- d2s3d / s2d3d
// Copies over voxel rows, a thread an output value; the volume is read off `geom` on the device,
// and rows past it (or past the `rows` the output holds) are left alone.
__global__ __launch_bounds__(kT) void d2s3d_kernel(
    const void* __restrict__ x, void* __restrict__ y, const int* __restrict__ geom, long long rows,
    int C, int st, int sh, int sw, int drop, long long x_ld, long long y_ld, bool xbf, bool ybf) {
    const int T = geom[0], H = geom[1], W = geom[2];
    if (T <= 0 || H <= 0 || W <= 0) return;
    const long long Ho = (long long)H * sh, Wo = (long long)W * sw;
    const long long To = (long long)T * st - drop;
    const long long n = min(rows, To * Ho * Wo) * C;
    for (long long e = (long long)blockIdx.x * kT + threadIdx.x; e < n;
         e += (long long)gridDim.x * kT) {
        const long long r = e / C;
        const int c = (int)(e % C);
        const long long tt = r / (Ho * Wo) + drop, ii = (r / Wo) % Ho, jj = r % Wo;
        const long long t = tt / st, i = ii / sh, j = jj / sw;
        const int aa = (int)(tt % st), b = (int)(ii % sh), d = (int)(jj % sw);
        const long long xr = (t * H + i) * W + j;
        stx(y, (size_t)(r * y_ld + c),
            ldx(x, (size_t)(xr * x_ld + ((c * st + aa) * sh + b) * sw + d), xbf), ybf);
    }
}

__global__ __launch_bounds__(kT) void s2d3d_kernel(
    const void* __restrict__ x, void* __restrict__ y, const int* __restrict__ geom, long long rows,
    int C, int st, int sh, int sw, int tpre, int grp, int Co, long long x_ld, long long y_ld,
    bool xbf, bool ybf, bool mean, bool add) {
    const int T = geom[0], H = geom[1], W = geom[2];
    if (T <= 0 || H <= 0 || W <= 0 || (T + tpre) % st || H % sh || W % sw) return;
    const long long Ho = H / sh, Wo = W / sw, To = (T + tpre) / st;
    const long long n = min(rows, To * Ho * Wo) * Co;
    for (long long e = (long long)blockIdx.x * kT + threadIdx.x; e < n;
         e += (long long)gridDim.x * kT) {
        const long long r = e / Co;
        const int o = (int)(e % Co);
        const long long t = r / (Ho * Wo), i = (r / Wo) % Ho, j = r % Wo;
        auto val = [&](int q) {
            const int d = q % sw, b = (q / sw) % sh, aa = (q / (sw * sh)) % st;
            const int c = q / (sw * sh * st);
            const long long v = t * st + aa, tr = v < tpre ? v : v - tpre;
            return ldx(x, (size_t)(((tr * H + i * sh + b) * W + j * sw + d) * x_ld + c), xbf);
        };
        float v;
        if (!mean) {
            v = val(o);
        } else {
            float s = 0.0f;
            for (int q = o * grp; q < (o + 1) * grp; ++q) s += val(q);
            v = rdx(s / (float)grp, ybf);
        }
        const size_t oy = (size_t)(r * y_ld + o);
        stx(y, oy, add ? ldx(y, oy, ybf) + rdx(v, ybf) : v, ybf);
    }
}

// ---------------------------------------------------------------------------- pix_patch
__device__ __forceinline__ int patch_col(int c, int ph, int pw, int p, bool wh) {
    return c * p * p + (wh ? pw * p + ph : ph * p + pw);
}

// A thread an image value.
__global__ __launch_bounds__(kT) void pix_patch_kernel(
    const void* __restrict__ x, void* __restrict__ y, int H, int W, int C, int p, bool wh,
    long long x_ld, long long y_ld, bool xbf, bool ybf) {
    const long long n = (long long)H * W * C;
    for (long long e = (long long)blockIdx.x * kT + threadIdx.x; e < n;
         e += (long long)gridDim.x * kT) {
        const int c = (int)(e % C);
        const long long px = e / C;
        const int yy = (int)(px / W), xx = (int)(px % W);
        stx(y, (size_t)((yy / p) * y_ld + (long long)(xx / p) * C * p * p +
                        patch_col(c, yy % p, xx % p, p, wh)),
            ldx(x, (size_t)(yy * x_ld + (long long)xx * C + c), xbf), ybf);
    }
}

// ---------------------------------------------------------------------------- attn_na3d
// 3-D neighborhood attention (libref/ref_attn.cpp states it): a thread a (query, head), its query
// in registers, the window's keys streamed from memory -- neighbouring queries share all but a
// slice of their windows, so the caches serve most reads -- in libref's two passes, the maximum
// and then the exponentials' sum and the weighted values.
template <int HD>
__global__ __launch_bounds__(kT) void attn_na3d_kernel(
    const void* __restrict__ q, const void* __restrict__ k, const void* __restrict__ v,
    const int* __restrict__ geom, void* __restrict__ o, long long rows, int kt, int kh, int kw,
    float scale, long long q_ld, long long k_ld, long long v_ld, long long o_ld, bool qbf, bool kbf,
    bool vbf, bool obf) {
    const int h = blockIdx.y;
    const long long r = (long long)blockIdx.x * kT + threadIdx.x;
    const int T = geom[0], H = geom[1], W = geom[2];
    if (T <= 0 || H <= 0 || W <= 0 || r >= min(rows, (long long)T * H * W)) return;
    const int nt = min(kt, T), nh = min(kh, H), nw = min(kw, W);
    const int t = (int)(r / ((long long)H * W)), i = (int)((r / W) % H), j = (int)(r % W);
    const int t0 = min(max(t - nt / 2, 0), T - nt), i0 = min(max(i - nh / 2, 0), H - nh);
    const int j0 = min(max(j - nw / 2, 0), W - nw);
    float qv[HD];
    for (int d = 0; d < HD; ++d) qv[d] = ldx(q, (size_t)(r * q_ld + h * HD + d), qbf);
    auto score = [&](long long key) {
        float dot = 0.0f;
        const size_t kb = (size_t)(key * k_ld + h * HD);
        for (int d = 0; d < HD; ++d) dot = __fadd_rn(dot, __fmul_rn(qv[d], ldx(k, kb + d, kbf)));
        return dot * scale;
    };
    float mx = -INFINITY;
    for (int a1 = t0; a1 < t0 + nt; ++a1)
        for (int b1 = i0; b1 < i0 + nh; ++b1) {
            const long long base = ((long long)a1 * H + b1) * W;
            for (int c1 = j0; c1 < j0 + nw; ++c1) mx = fmaxf(mx, score(base + c1));
        }
    float acc[HD];
    for (int d = 0; d < HD; ++d) acc[d] = 0.0f;
    float sum = 0.0f;
    for (int a1 = t0; a1 < t0 + nt; ++a1)
        for (int b1 = i0; b1 < i0 + nh; ++b1) {
            const long long base = ((long long)a1 * H + b1) * W;
            for (int c1 = j0; c1 < j0 + nw; ++c1) {
                const float w = expf(score(base + c1) - mx);
                sum += w;
                const size_t vb = (size_t)((base + c1) * v_ld + h * HD);
                for (int d = 0; d < HD; ++d) acc[d] = fmaf(w, ldx(v, vb + d, vbf), acc[d]);
            }
        }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int d = 0; d < HD; ++d) stx(o, (size_t)(r * o_ld + h * HD + d), acc[d] * inv, obf);
}

// The same over bf16 operands, each row read 8 values at a time (16-byte loads: the rows sit at
// multiples of 8 values and the heads at multiples of 64).
__device__ __forceinline__ void bf16x8(const uint16_t* __restrict__ p, float* f) {
    const uint4 u = *(const uint4*)p;
    const uint32_t w[4] = { u.x, u.y, u.z, u.w };
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

template <int HD>
__global__ __launch_bounds__(kT) void attn_na3d_bf16_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int* __restrict__ geom, uint16_t* __restrict__ o, long long rows, int kt, int kh, int kw,
    float scale, long long q_ld, long long k_ld, long long v_ld, long long o_ld) {
    const int h = blockIdx.y;
    const long long r = (long long)blockIdx.x * kT + threadIdx.x;
    const int T = geom[0], H = geom[1], W = geom[2];
    if (T <= 0 || H <= 0 || W <= 0 || r >= min(rows, (long long)T * H * W)) return;
    const int nt = min(kt, T), nh = min(kh, H), nw = min(kw, W);
    const int t = (int)(r / ((long long)H * W)), i = (int)((r / W) % H), j = (int)(r % W);
    const int t0 = min(max(t - nt / 2, 0), T - nt), i0 = min(max(i - nh / 2, 0), H - nh);
    const int j0 = min(max(j - nw / 2, 0), W - nw);
    float qv[HD];
    for (int d = 0; d < HD; d += 8) bf16x8(q + r * q_ld + h * HD + d, qv + d);
    auto score = [&](long long key) {
        const uint16_t* kr = k + key * k_ld + h * HD;
        float dot = 0.0f;
        for (int d = 0; d < HD; d += 8) {
            float kk[8];
            bf16x8(kr + d, kk);
            for (int e = 0; e < 8; ++e) dot = fmaf(qv[d + e], kk[e], dot);
        }
        return dot * scale;
    };
    float mx = -INFINITY;
    for (int a1 = t0; a1 < t0 + nt; ++a1)
        for (int b1 = i0; b1 < i0 + nh; ++b1) {
            const long long base = ((long long)a1 * H + b1) * W;
            for (int c1 = j0; c1 < j0 + nw; ++c1) mx = fmaxf(mx, score(base + c1));
        }
    float acc[HD];
    for (int d = 0; d < HD; ++d) acc[d] = 0.0f;
    float sum = 0.0f;
    for (int a1 = t0; a1 < t0 + nt; ++a1)
        for (int b1 = i0; b1 < i0 + nh; ++b1) {
            const long long base = ((long long)a1 * H + b1) * W;
            for (int c1 = j0; c1 < j0 + nw; ++c1) {
                const float w = expf(score(base + c1) - mx);
                sum += w;
                const uint16_t* vr = v + (base + c1) * v_ld + h * HD;
                for (int d = 0; d < HD; d += 8) {
                    float vv[8];
                    bf16x8(vr + d, vv);
                    for (int e = 0; e < 8; ++e) acc[d + e] = fmaf(w, vv[e], acc[d + e]);
                }
            }
        }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int d = 0; d < HD; ++d) o[r * o_ld + h * HD + d] = f32_to_bf16(acc[d] * inv);
}

// ---------------------------------------------------------------------------- vol_index
// A thread a voxel's three coordinates.
__global__ __launch_bounds__(kT) void vol_index_kernel(const int* __restrict__ geom,
                                                       int* __restrict__ y, long long n,
                                                       long long y_ld) {
    const int T = geom[0], H = geom[1], W = geom[2];
    if (T <= 0 || H <= 0 || W <= 0) return;
    const long long m1 = min(n, (long long)T * H * W);
    for (long long m = (long long)blockIdx.x * kT + threadIdx.x; m < m1;
         m += (long long)gridDim.x * kT) {
        y[m] = (int)(m / ((long long)H * W));
        y[y_ld + m] = (int)((m / W) % H);
        y[2 * y_ld + m] = (int)(m % W);
    }
}

// ---------------------------------------------------------------------------- tile_grid / tile_crop / tile_stitch
// diffusers' tiled decodes (libref/ref_ltx.cpp states the cuts and the blends): the boxes, a
// tile's voxels out of a volume, the tiles blended to bytes. The volumes are read off the device.
struct TileCutD {
    int tile, stride, least;
    bool intervals;
    __device__ int count(int L) const {
        if (intervals && L <= tile) return 1;
        int n = (L + stride - 1) / stride;
        while (intervals && n > 1 && L - (n - 1) * stride < least) --n;
        return n;
    }
    __device__ int len(int L, int k) const {
        return k == count(L) - 1 ? L - k * stride : min(tile, L - k * stride);
    }
};

// A thread a row of the box table.
__global__ __launch_bounds__(kT) void tile_grid_kernel(const int* __restrict__ geom,
                                                       int* __restrict__ y, int n, TileCutD cs,
                                                       TileCutD ct, int tail) {
    const int k = blockIdx.x * kT + threadIdx.x;
    if (k >= n) return;
    const int T = geom[0], H = geom[1], W = geom[2];
    int row[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (T > 0 && H > 0 && W > 0) {
        const int nt = ct.count(T), ni = cs.count(H), nj = cs.count(W), count = nt * ni * nj;
        if (k < count) {
            const int kt = k / (ni * nj), i = (k / nj) % ni, j = k % nj;
            row[0] = ct.len(T, kt) + (kt == nt - 1 ? tail : 0);
            row[1] = cs.len(H, i);
            row[2] = cs.len(W, j);
            row[3] = kt * ct.stride;
            row[4] = i * cs.stride;
            row[5] = j * cs.stride;
            row[6] = k;
            row[7] = count;
        }
    }
    for (int e = 0; e < 8; ++e) y[(long long)k * 8 + e] = row[e];
}

// A thread an output value; rows past the box's voxels (or the `rows` y holds) are left alone.
__global__ __launch_bounds__(kT) void tile_crop_kernel(
    const void* __restrict__ x, void* __restrict__ y, const int* __restrict__ geom,
    const int* __restrict__ box, long long rows, int C, long long x_ld, long long y_ld, bool xbf,
    bool ybf) {
    const int T = geom[0], H = geom[1], W = geom[2];
    const int t = box[0], h = box[1], w = box[2], t0 = box[3], y0 = box[4], x0 = box[5];
    if (t <= 0 || h <= 0 || w <= 0 || t0 < 0 || y0 < 0 || x0 < 0 || t0 + t > T || y0 + h > H ||
        x0 + w > W)
        return;
    const long long n = min(rows, (long long)t * h * w) * C;
    for (long long e = (long long)blockIdx.x * kT + threadIdx.x; e < n;
         e += (long long)gridDim.x * kT) {
        const long long r = e / C;
        const int c = (int)(e % C);
        const long long f = r / ((long long)h * w), i = (r / w) % h, j = r % w;
        const long long xr = ((t0 + f) * H + y0 + i) * W + x0 + j;
        stx(y, (size_t)(r * y_ld + c), ldx(x, (size_t)(xr * x_ld + c), xbf), ybf);
    }
}

// The tile store's layout: time bands of every tile back to back.
struct TileStoreD {
    TileCutD cs, ct;
    int T, H, W, sc, st;
    __device__ int frames(int kt) const { return st * ct.len(T, kt) - (kt == 0 ? st - 1 : 0); }
    __device__ long long offset(int kt, int i, int j) const {
        const int nj = cs.count(W), ni = cs.count(H);
        long long hsum = 0, wsum = 0, left = 0, above = 0, before = 0;
        for (int q = 0; q < nj; ++q) {
            const int b = cs.len(W, q);
            wsum += b;
            if (q < j) left += b;
        }
        for (int q = 0; q < ni; ++q) {
            const int b = cs.len(H, q);
            hsum += b;
            if (q < i) above += b;
        }
        for (int q = 0; q < kt; ++q) before += frames(q);
        return (long long)sc * sc *
               (before * hsum * wsum + (long long)frames(kt) * (above * wsum + (long long)cs.len(H, i) * left));
    }
};

// A thread an output byte: up to eight tiles' values, the blends in libref's order and roundings.
__global__ __launch_bounds__(kT) void tile_stitch_kernel(
    const void* __restrict__ x, uint8_t* __restrict__ y, const int* __restrict__ geom,
    const int* __restrict__ first, long long y_rows, long long y_cols, TileCutD cs, TileCutD ct,
    int sc, int st, int p, int C, bool wh, long long x_ld, long long y_ld, bool xbf) {
    const int T = geom[0], H = geom[1], W = geom[2], f0 = first[0];
    const long long Hpx = (long long)H * sc * p, Wpx = (long long)W * sc * p;
    if (T <= 0 || H <= 0 || W <= 0 || y_cols != Wpx * C || y_rows % Hpx) return;
    const long long nf = y_rows / Hpx;
    if (f0 < 0 || f0 + nf > (long long)st * T - (st - 1)) return;
    const TileStoreD ts{ cs, ct, T, H, W, sc, st };
    const int nt = ct.count(T), ni = cs.count(H), nj = cs.count(W);
    const long long s = (long long)cs.stride * sc * p, E = (long long)(cs.tile - cs.stride) * sc * p;
    const long long Et = (long long)(ct.tile - ct.stride) * st, band = (long long)ct.stride * st;
    const long long n = y_rows * y_cols;
    for (long long e = (long long)blockIdx.x * kT + threadIdx.x; e < n;
         e += (long long)gridDim.x * kT) {
        const long long r = e / y_cols;
        const int rem = (int)(e % y_cols), Xp = rem / C, c = rem % C;
        const long long nfr = f0 + r / Hpx, Yp = r % Hpx;
        const int kt = (int)min((nfr + st - 1) / band, (long long)nt - 1);
        const int f = (int)(nfr - (kt == 0 ? 0 : kt * band - (st - 1)));
        auto R = [&](int bk, int ti, int tj, int fr, int yy, int xx) {
            const long long wp = (long long)cs.len(W, tj) * sc, hp = (long long)cs.len(H, ti) * sc;
            const long long row = ts.offset(bk, ti, tj) + ((long long)fr * hp + yy / p) * wp + xx / p;
            return ldx(x, (size_t)(row * x_ld + patch_col(c, yy % p, xx % p, p, wh)), xbf);
        };
        auto mix = [&](float va, float vb, int k, int ext) {
            const float wa = (float)(1.0 - (double)k / (double)ext);
            const float wb = (float)((double)k / (double)ext);
            return rdx(rdx(__fmul_rn(va, wa), xbf) + rdx(__fmul_rn(vb, wb), xbf), xbf);
        };
        const int i = (int)min(Yp / s, (long long)ni - 1), j = (int)min((long long)Xp / s, (long long)nj - 1);
        const int ly = (int)(Yp - i * s), lx = (int)(Xp - j * s);
        const int ha = i > 0 ? cs.len(H, i - 1) * sc * p : 0;
        const int ev = i > 0 ? (int)min(min((long long)ha, (long long)cs.len(H, i) * sc * p), E) : 0;
        const int wl = j > 0 ? cs.len(W, j - 1) * sc * p : 0;
        const int eh = j > 0 ? (int)min(min((long long)wl, (long long)cs.len(W, j) * sc * p), E) : 0;
        const bool vb = i > 0 && ly < ev, hb = j > 0 && lx < eh;
        auto S = [&](int bk, int fr) {
            float V = R(bk, i, j, fr, ly, lx);
            if (vb) {
                const int ya = ha - ev + ly;
                float A = R(bk, i - 1, j, fr, ya, lx);
                if (hb) A = mix(R(bk, i - 1, j - 1, fr, ya, wl - eh + lx), A, lx, eh);
                V = mix(A, V, ly, ev);
            }
            if (hb) {
                float L = R(bk, i, j - 1, fr, ly, wl - eh + lx);
                if (vb) L = mix(R(bk, i - 1, j - 1, fr, ha - ev + ly, wl - eh + lx), L, ly, ev);
                V = mix(L, V, lx, eh);
            }
            return V;
        };
        float v = S(kt, f);
        if (kt > 0) {
            const int et = (int)min(min((long long)ts.frames(kt - 1), (long long)ts.frames(kt)), Et);
            if (f < et) v = mix(S(kt - 1, ts.frames(kt - 1) - et + f), v, f, et);
        }
        const float cl = fminf(fmaxf(v, -1.0f), 1.0f);
        const float u = fminf(fmaxf(rdx(rdx(cl / 2.0f, xbf) + 0.5f, xbf), 0.0f), 1.0f);
        y[r * y_ld + (long long)Xp * C + c] = (uint8_t)rintf(u * 255.0f);
    }
}

// ---------------------------------------------------------------------------- stft_logmel
// A workgroup a (frame, channel): the frame's samples into LDS, a thread a frequency bin's real
// and imaginary sums (ascending, as libref), the magnitudes into LDS, then a thread a mel bin.
constexpr int kStftMaxN = 4096;

__global__ __launch_bounds__(kT) void stft_logmel_kernel(
    const void* __restrict__ x, const void* __restrict__ basis, const void* __restrict__ melb,
    void* __restrict__ y, int frames, int ch, int hop, int nfft, int nm, long long x_ld,
    long long y_ld, bool xbf, bool bbf, bool mbf, bool ybf) {
    __shared__ float sx[kStftMaxN];
    __shared__ float mag[kStftMaxN / 2 + 1];
    const int t = blockIdx.x, h = blockIdx.y, F = nfft / 2 + 1, pad = nfft - hop;
    for (int j = threadIdx.x; j < nfft; j += kT) {
        const long long s = (long long)t * hop + j - pad;
        sx[j] = s < 0 ? 0.0f : ldx(x, (size_t)(s * x_ld + h), xbf);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < F; k += kT) {
        float re = 0.0f, im = 0.0f;
        const size_t br = (size_t)k * nfft, bi = (size_t)(F + k) * nfft;
        for (int j = 0; j < nfft; ++j) {
            re += ldx(basis, br + j, bbf) * sx[j];
            im += ldx(basis, bi + j, bbf) * sx[j];
        }
        re = rdx(re, ybf);
        im = rdx(im, ybf);
        mag[k] = rdx(sqrtf(rdx(rdx(re * re, ybf) + rdx(im * im, ybf), ybf)), ybf);
    }
    __syncthreads();
    const float fl = rdx(1e-5f, ybf);
    for (int m = threadIdx.x; m < nm; m += kT) {
        float s = 0.0f;
        for (int k = 0; k < F; ++k) s += ldx(melb, (size_t)m * F + k, mbf) * mag[k];
        stx(y, (size_t)((long long)t * y_ld + (long long)h * nm + m), logf(fmaxf(rdx(s, ybf), fl)),
            ybf);
    }
}

// ---------------------------------------------------------------------------- bwe_up
// A thread an output value; the resampler's taps are worked out on the host as libref does and
// passed by value.
// Taps for ratios up to 4 (2 w r + 1 = 57): the filter is passed by value, and the backend's
// kernel-argument slot is 768 bytes.
constexpr int kBweTaps = 64;
struct BweTaps { float f[kBweTaps]; };

__global__ __launch_bounds__(kT) void bwe_up_kernel(
    const void* __restrict__ x, const void* __restrict__ res, void* __restrict__ y, BweTaps f,
    long long S, long long O, int ch, int r, int w, int K, long long x_ld, long long r_ld,
    long long y_ld, bool xbf, bool rbf, bool ybf) {
    const long long n = O * ch;
    for (long long e = (long long)blockIdx.x * kT + threadIdx.x; e < n;
         e += (long long)gridDim.x * kT) {
        const long long o = e / ch;
        const int h = (int)(e % ch);
        const long long u = o + 2ll * w * r;
        float acc = 0.0f;
        for (int k = (int)(u % r); k < K; k += r) {
            const long long i = (u - k) / r;
            const long long s = min(max(i - w, 0ll), S - 1);
            acc += f.f[k] * ldx(x, (size_t)(s * x_ld + h), xbf);
        }
        const float up = rdx((float)r * rdx(acc, xbf), xbf);
        const float v = rdx(ldx(res, (size_t)(o * r_ld + h), rbf) + up, xbf);
        stx(y, (size_t)(o * y_ld + h), fminf(fmaxf(v, -1.0f), 1.0f), ybf);
    }
}

inline int st_of(hipError_t e) { return e == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE); }
inline bool is_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }
inline bool is_f32(const RadTensor* t) { return t && t->dtype == RAD_F32; }
inline bool is_bf_or_f32(const RadTensor* t) { return is_bf16(t) || is_f32(t); }

int parse_floats(const char* s, float* out, int max) {
    int n = 0;
    while (s && *s) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        if (n == max) return max + 1;
        out[n++] = (float)v;
        s = end;
    }
    return n;
}

}  // namespace

// ============================================================== the plugin ABI entry points

extern "C" int r4d_rad_rope_split(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *p = r4d_opt(a, 1);
    if (!x || !p) return R4D_NO(RAD_E_INVAL);
    if (!is_bf_or_f32(x) || !(p->dtype == RAD_F32 || p->dtype == RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    const long long nh = r4d_geti_or(a, "n_head", 0), hd = r4d_geti_or(a, "head_dim", 0);
    const long long dim = r4d_geti_or(a, "dim", 0), nd = r4d_geti_or(a, "n_dims", 0);
    const long long h0 = r4d_geti_or(a, "head0", 0);
    float mx[kMaxDims] = { 1, 1, 1, 1 };
    if (nh <= 0 || hd <= 0 || (hd & 1) || hd / 2 > 256 || (kT % (hd / 2)) || nd <= 0 ||
        nd > kMaxDims || dim < (h0 + nh) * hd ||
        parse_floats(r4d_gets(a, "max"), mx, kMaxDims) != (int)nd)
        return R4D_NO(RAD_E_SHAPE);
    long long rows = 0;
    const long long xl = r4d_rows_pitch(x, nh * hd, &rows);
    if (xl < 0) return R4D_NO(RAD_E_STRIDE);
    if (rows <= 0) return R4D_NO(RAD_E_SHAPE);
    // positions [n_dims, M] at its strides, or [M].
    long long pl = 1, pc = 0;
    if (p->rank >= 2) {
        pc = p->stride[0];
        pl = p->stride[1];
        if (p->shape[0] < nd || p->shape[1] < rows) return R4D_NO(RAD_E_SHAPE);
    } else {
        pl = p->rank ? p->stride[0] : 1;
        if (nd != 1 || r4d_numel(p) < rows) return R4D_NO(RAD_E_SHAPE);
    }
    const long long P2 = dim / 2, steps = P2 / nd, pad = P2 - steps * nd;
    dim3 grid((unsigned)((rows + kRopeRows - 1) / kRopeRows), (unsigned)nh);
    hipLaunchKernelGGL(rope_split_kernel, grid, dim3(kT), 0, (hipStream_t)s, x->data, p->data,
                       (int)rows, (int)nh, (int)hd, (int)h0, (int)nd, (int)steps, (int)pad,
                       r4d_getf(a, "theta", 10000.0), mx[0], mx[1], mx[2], mx[3], xl, pl, pc,
                       is_bf16(x), p->dtype == RAD_I32);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_head_gate(const RadArgs* a, RadStream s) {
    const RadTensor *g = r4d_opt(a, 0), *x = r4d_opt(a, 1);
    if (!g || !x) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(g) || !is_bf_or_f32(x)) return R4D_NO(RAD_E_DTYPE);
    const long long nh = r4d_geti_or(a, "n_head", 0), hd = r4d_geti_or(a, "head_dim", 0);
    if (nh <= 0 || hd <= 0) return R4D_NO(RAD_E_SHAPE);
    long long rows = 0, rg = 0;
    const long long xl = r4d_rows_pitch(x, nh * hd, &rows), gl = r4d_rows_pitch(g, nh, &rg);
    if (xl < 0 || gl < 0) return R4D_NO(RAD_E_STRIDE);
    if (rows <= 0 || rg < rows) return R4D_NO(RAD_E_SHAPE);
    const long long total = rows * nh * hd;
    hipLaunchKernelGGL(head_gate_kernel, dim3((unsigned)((total + kT - 1) / kT)), dim3(kT), 0,
                       (hipStream_t)s, x->data, (const uint16_t*)g->data, total, (int)(nh * hd),
                       (int)hd, xl, gl, is_bf16(x));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_ltx_guide(const RadArgs* a, RadStream s) {
    const RadTensor *vc = r4d_opt(a, 0), *vu = r4d_opt(a, 1), *vs = r4d_opt(a, 2);
    const RadTensor *vm = r4d_opt(a, 3), *sc = r4d_opt(a, 4), *sh = r4d_opt(a, 5);
    const RadTensor *st = r4d_opt(a, 6), *x = r4d_opt(a, 7);
    if (!vc || !sc || !sh || !st || !x) return R4D_NO(RAD_E_INVAL);
    if (!is_f32(x) || !is_f32(sc) || !is_f32(sh) || !is_f32(st)) return R4D_NO(RAD_E_DTYPE);
    for (const RadTensor* v : { vc, vu, vs, vm })
        if (v && !is_bf_or_f32(v)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long rows = 0;
    const long long xl = r4d_rows_pitch(x, n, &rows);
    if (xl < 0) return R4D_NO(RAD_E_STRIDE);
    if (n <= 0 || rows <= 0 || r4d_numel(sc) < 4 || r4d_numel(sh) < 2 ||
        r4d_numel(st) < 2 * 4 * kGuideParts || !r4d_contig(sc) || !r4d_contig(sh) ||
        ((uintptr_t)st->data & 7))
        return R4D_NO(RAD_E_SHAPE);
    GuideIn in{};
    const RadTensor* vv[4] = { vc, vu, vs, vm };
    const void** dst[4] = { &in.vc, &in.vu, &in.vs, &in.vm };
    long long* ld[4] = { &in.lc, &in.lu, &in.ls, &in.lm };
    bool* bf[4] = { &in.bc, &in.bu, &in.bs, &in.bm };
    for (int i = 0; i < 4; ++i) {
        *dst[i] = nullptr;
        if (!vv[i]) continue;
        long long rv = 0;
        const long long l = r4d_rows_pitch(vv[i], n, &rv);
        if (l < 0) return R4D_NO(RAD_E_STRIDE);
        if (rv < rows) return R4D_NO(RAD_E_SHAPE);
        *dst[i] = vv[i]->data;
        *ld[i] = l;
        *bf[i] = is_bf16(vv[i]);
    }
    const long long total = rows * n;
    const int parts = (int)std::min<long long>(kGuideParts, (total + kT - 1) / kT);
    double* pd = (double*)st->data;
    hipLaunchKernelGGL(ltx_guide_sum_kernel, dim3((unsigned)parts), dim3(kT), 0, (hipStream_t)s,
                       in, (const float*)x->data, (const float*)sc->data, (const float*)sh->data,
                       pd, rows, (int)n, xl);
    hipLaunchKernelGGL(ltx_guide_step_kernel, dim3((unsigned)parts), dim3(kT), 0, (hipStream_t)s,
                       in, (float*)x->data, (const float*)sc->data, (const float*)sh->data, pd,
                       parts, rows, (int)n, xl);
    return st_of(hipGetLastError());
}

namespace {

// Workgroups for a grid-stride loop over n values.
inline unsigned grid_for(long long n) {
    return (unsigned)std::max(1ll, std::min((n + kT - 1) / kT, 1ll << 20));
}

// A dense 2-D operand: its rows, columns and row pitch, or false.
inline bool dense2(const RadTensor* t, long long* rows, long long* cols, long long* ld) {
    if (!t || t->rank != 2 || t->stride[1] != 1 || t->shape[0] <= 0 || t->shape[1] <= 0)
        return false;
    *rows = t->shape[0];
    *cols = t->shape[1];
    *ld = t->stride[0];
    return true;
}

// `order`: wh (the default, LTX-2's) or hw; -1 for anything else.
inline int patch_order(const RadArgs* a) {
    const char* o = r4d_gets(a, "order");
    if (!o || !*o || !std::strcmp(o, "wh")) return 1;
    return std::strcmp(o, "hw") ? -1 : 0;
}

}  // namespace

extern "C" int r4d_rad_pixel_norm(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf_or_f32(x) || !is_bf_or_f32(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), L = r4d_geti_or(a, "ileave", 1);
    const char *form = r4d_gets(a, "form"), *act = r4d_gets(a, "act");
    const bool divf = form && !std::strcmp(form, "div");
    const bool silu = act && !std::strcmp(act, "silu");
    if ((form && *form && !divf && std::strcmp(form, "rsqrt")) ||
        (act && *act && !silu && std::strcmp(act, "none")))
        return R4D_NO(RAD_E_INVAL);
    if (n <= 0 || L <= 0) return R4D_NO(RAD_E_SHAPE);
    long long ry = 0, rx = 0;
    const long long yl = r4d_rows_pitch(y, n * L, &ry), xl = r4d_rows_pitch(x, n, &rx);
    if (yl < 0 || xl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long rows = ry * L;
    if (rows <= 0 || rx < rows) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(pixel_norm_kernel, dim3((unsigned)((rows + kT / 32 - 1) / (kT / 32))),
                       dim3(kT), 0, (hipStream_t)s, x->data, y->data, rows, (int)n, (int)L,
                       (float)r4d_getf(a, "eps", 1e-6), (float)r4d_getf(a, "mult", 1.0), xl, yl,
                       is_bf16(x), is_bf16(y), divf, silu);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_d2s3d(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *g = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !g || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf_or_f32(x) || !is_bf_or_f32(y) || g->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long long C = r4d_geti_or(a, "C", 0), st = r4d_geti_or(a, "st", 1);
    const long long sh = r4d_geti_or(a, "sh", 1), sw = r4d_geti_or(a, "sw", 1);
    const long long drop = r4d_geti_or(a, "drop", st - 1);
    if (C <= 0 || st <= 0 || sh <= 0 || sw <= 0 || drop < 0 || drop >= st || r4d_numel(g) < 3 ||
        !r4d_contig(g))
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, C * st * sh * sw, &rx), yl = r4d_rows_pitch(y, C, &ry);
    if (xl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    if (rx <= 0 || ry <= 0) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(d2s3d_kernel, dim3(grid_for(ry * C)), dim3(kT), 0, (hipStream_t)s, x->data,
                       y->data, (const int*)g->data, ry, (int)C, (int)st, (int)sh, (int)sw, (int)drop,
                       xl, yl, is_bf16(x), is_bf16(y));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_s2d3d(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *g = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !g || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf_or_f32(x) || !is_bf_or_f32(y) || g->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long long C = r4d_geti_or(a, "C", 0), st = r4d_geti_or(a, "st", 1);
    const long long sh = r4d_geti_or(a, "sh", 1), sw = r4d_geti_or(a, "sw", 1);
    const long long tpre = r4d_geti_or(a, "tpre", 0), grp = r4d_geti_or(a, "group", 1);
    const char* mode = r4d_gets(a, "mode");
    const bool mean = mode && !std::strcmp(mode, "mean");
    if (mode && *mode && !mean && std::strcmp(mode, "copy")) return R4D_NO(RAD_E_INVAL);
    const long long P = C * st * sh * sw;
    if (C <= 0 || st <= 0 || sh <= 0 || sw <= 0 || tpre < 0 || grp <= 0 || (mean && P % grp) ||
        r4d_numel(g) < 3 || !r4d_contig(g))
        return R4D_NO(RAD_E_SHAPE);
    const long long Co = mean ? P / grp : P;
    long long rx = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, C, &rx), yl = r4d_rows_pitch(y, Co, &ry);
    if (xl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    if (rx <= 0 || ry <= 0) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(s2d3d_kernel, dim3(grid_for(ry * Co)), dim3(kT), 0, (hipStream_t)s, x->data,
                       y->data, (const int*)g->data, ry, (int)C, (int)st, (int)sh, (int)sw,
                       (int)tpre, (int)grp, (int)Co, xl, yl, is_bf16(x), is_bf16(y), mean,
                       r4d_geti_or(a, "add", 0) != 0);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_pix_patch(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf_or_f32(x) || !is_bf_or_f32(y)) return R4D_NO(RAD_E_DTYPE);
    const long long p = r4d_geti_or(a, "p", 0), C = r4d_geti_or(a, "C", 0);
    const int wh = patch_order(a);
    if (wh < 0) return R4D_NO(RAD_E_INVAL);
    long long H = 0, WC = 0, xl = 0, Ry = 0, Cy = 0, yl = 0;
    if (!dense2(x, &H, &WC, &xl) || !dense2(y, &Ry, &Cy, &yl)) return R4D_NO(RAD_E_SHAPE);
    if (p <= 0 || C <= 0 || WC % C) return R4D_NO(RAD_E_SHAPE);
    const long long W = WC / C;
    if (H % p || W % p || Ry != H / p || Cy != (W / p) * C * p * p) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(pix_patch_kernel, dim3(grid_for(H * W * C)), dim3(kT), 0, (hipStream_t)s,
                       x->data, y->data, (int)H, (int)W, (int)C, (int)p, wh == 1, xl, yl,
                       is_bf16(x), is_bf16(y));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_attn_na3d(const RadArgs* a, RadStream s) {
    const RadTensor *q = r4d_opt(a, 0), *k = r4d_opt(a, 1), *v = r4d_opt(a, 2), *g = r4d_opt(a, 3);
    const RadTensor* o = r4d_opt(a, 4);
    if (!q || !k || !v || !g || !o) return R4D_NO(RAD_E_INVAL);
    if (!is_bf_or_f32(q) || !is_bf_or_f32(k) || !is_bf_or_f32(v) || !is_bf_or_f32(o) ||
        g->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    const long long hd = r4d_geti_or(a, "head_dim", 0), kt = r4d_geti_or(a, "kt", 0);
    const long long kh = r4d_geti_or(a, "kh", 0), kw = r4d_geti_or(a, "kw", 0);
    if (hd != 64 || kt <= 0 || kh <= 0 || kw <= 0 || r4d_numel(g) < 3 || !r4d_contig(g) ||
        q->rank != 2 || q->shape[1] % hd)
        return R4D_NO(RAD_E_SHAPE);
    const long long heads = q->shape[1] / hd, D = heads * hd;
    long long rq = 0, rk = 0, rv = 0, ro = 0;
    const long long ql = r4d_rows_pitch(q, D, &rq), kl = r4d_rows_pitch(k, D, &rk);
    const long long vl = r4d_rows_pitch(v, D, &rv), ol = r4d_rows_pitch(o, D, &ro);
    if (ql < 0 || kl < 0 || vl < 0 || ol < 0) return R4D_NO(RAD_E_STRIDE);
    if (rq <= 0 || rk < rq || rv < rq || ro < rq) return R4D_NO(RAD_E_SHAPE);
    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));
    const dim3 grid((unsigned)((rq + kT - 1) / kT), (unsigned)heads);
    if (is_bf16(q) && is_bf16(k) && is_bf16(v) && is_bf16(o) && ql % 8 == 0 && kl % 8 == 0 &&
        vl % 8 == 0 && ((uintptr_t)q->data | (uintptr_t)k->data | (uintptr_t)v->data) % 16 == 0) {
        hipLaunchKernelGGL(attn_na3d_bf16_kernel<64>, grid, dim3(kT), 0, (hipStream_t)s,
                           (const uint16_t*)q->data, (const uint16_t*)k->data,
                           (const uint16_t*)v->data, (const int*)g->data, (uint16_t*)o->data, rq,
                           (int)kt, (int)kh, (int)kw, scale, ql, kl, vl, ol);
        return st_of(hipGetLastError());
    }
    hipLaunchKernelGGL(attn_na3d_kernel<64>, grid, dim3(kT), 0, (hipStream_t)s, q->data, k->data,
                       v->data, (const int*)g->data, o->data, rq, (int)kt, (int)kh, (int)kw, scale,
                       ql, kl, vl, ol, is_bf16(q), is_bf16(k), is_bf16(v), is_bf16(o));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_vol_index(const RadArgs* a, RadStream s) {
    const RadTensor *g = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!g || !y) return R4D_NO(RAD_E_INVAL);
    if (g->dtype != RAD_I32 || y->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(g) < 3 || !r4d_contig(g) || y->rank != 2 || y->shape[0] != 3 ||
        y->shape[1] <= 0 || y->stride[1] != 1)
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(vol_index_kernel, dim3(grid_for(y->shape[1])), dim3(kT), 0, (hipStream_t)s,
                       (const int*)g->data, (int*)y->data, (long long)y->shape[1],
                       (long long)y->stride[0]);
    return st_of(hipGetLastError());
}

// The cuts a tile op's parameters name (libref's tile_cuts); false when one is out of range.
static bool tile_cuts_d(const RadArgs* a, TileCutD* s, TileCutD* t) {
    const char* rs = r4d_gets(a, "rule");
    const char* rule = rs ? rs : "stride";
    const bool iv = std::strcmp(rule, "intervals") == 0;
    if (!iv && std::strcmp(rule, "stride") != 0) return false;
    *s = TileCutD{ (int)r4d_geti_or(a, "tile", 0), (int)r4d_geti_or(a, "stride", 0),
                   (int)r4d_geti_or(a, "min", 1), iv };
    *t = TileCutD{ (int)r4d_geti_or(a, "tile_t", 1 << 30), (int)r4d_geti_or(a, "stride_t", 1 << 30),
                   (int)r4d_geti_or(a, "min_t", 1), iv };
    return s->tile > 0 && s->stride > 0 && s->stride <= s->tile && s->least > 0 && t->tile > 0 &&
           t->stride > 0 && t->stride <= t->tile && t->least > 0;
}

extern "C" int r4d_rad_tile_grid(const RadArgs* a, RadStream s) {
    const RadTensor *g = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!g || !y) return R4D_NO(RAD_E_INVAL);
    if (g->dtype != RAD_I32 || y->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    TileCutD cs, ct;
    const long long n = r4d_numel(y) / 8, tail = r4d_geti_or(a, "tail", 0);
    if (!tile_cuts_d(a, &cs, &ct) || tail < 0 || n <= 0 || r4d_numel(g) < 3 || !r4d_contig(g) ||
        !r4d_contig(y))
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(tile_grid_kernel, dim3((unsigned)((n + kT - 1) / kT)), dim3(kT), 0,
                       (hipStream_t)s, (const int*)g->data, (int*)y->data, (int)n, cs, ct, (int)tail);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_tile_crop(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *g = r4d_opt(a, 1), *b = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!x || !g || !b || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf_or_f32(x) || !is_bf_or_f32(y) || g->dtype != RAD_I32 || b->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    const long long C = r4d_geti_or(a, "C", 0);
    if (C <= 0 || r4d_numel(g) < 3 || r4d_numel(b) < 6 || !r4d_contig(g) || !r4d_contig(b))
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, C, &rx), yl = r4d_rows_pitch(y, C, &ry);
    if (xl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    if (rx <= 0 || ry <= 0) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(tile_crop_kernel, dim3(grid_for(ry * C)), dim3(kT), 0, (hipStream_t)s,
                       x->data, y->data, (const int*)g->data, (const int*)b->data, ry, (int)C, xl,
                       yl, is_bf16(x), is_bf16(y));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_tile_stitch(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *g = r4d_opt(a, 1), *f = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!x || !g || !f || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf_or_f32(x) || g->dtype != RAD_I32 || f->dtype != RAD_I32 || y->dtype != RAD_U8)
        return R4D_NO(RAD_E_DTYPE);
    TileCutD cs, ct;
    const long long sc = r4d_geti_or(a, "scale", 0), st = r4d_geti_or(a, "scale_t", 1);
    const long long p = r4d_geti_or(a, "p", 0), C = r4d_geti_or(a, "C", 0);
    const int wh = patch_order(a);
    if (wh < 0) return R4D_NO(RAD_E_INVAL);
    if (!tile_cuts_d(a, &cs, &ct) || sc <= 0 || st <= 0 || p <= 0 || C <= 0 ||
        r4d_numel(g) < 3 || r4d_numel(f) < 1 || !r4d_contig(g) || !r4d_contig(f))
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, Cx = 0, xl = 0, ry = 0, Cy = 0, yl = 0;
    if (!dense2(x, &rx, &Cx, &xl) || !dense2(y, &ry, &Cy, &yl)) return R4D_NO(RAD_E_SHAPE);
    if (Cx != C * p * p || ry <= 0 || Cy <= 0) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(tile_stitch_kernel, dim3(grid_for(ry * Cy)), dim3(kT), 0, (hipStream_t)s,
                       x->data, (uint8_t*)y->data, (const int*)g->data, (const int*)f->data, ry, Cy,
                       cs, ct, (int)sc, (int)st, (int)p, (int)C, wh == 1, xl, yl, is_bf16(x));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_stft_logmel(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *b = r4d_opt(a, 1), *m = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!x || !b || !m || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf_or_f32(x) || !is_bf_or_f32(b) || !is_bf_or_f32(m) || !is_bf_or_f32(y))
        return R4D_NO(RAD_E_DTYPE);
    const long long hop = r4d_geti_or(a, "hop", 0), nfft = r4d_geti_or(a, "nfft", 0);
    const long long nm = r4d_geti_or(a, "n_mel", 0), ch = r4d_geti_or(a, "ch", 0);
    if (hop <= 0 || nfft < hop || nfft > kStftMaxN || nm <= 0 || ch <= 0)
        return R4D_NO(RAD_E_SHAPE);
    const long long F = nfft / 2 + 1;
    long long S = 0, frames = 0;
    const long long xl = r4d_rows_pitch(x, ch, &S), yl = r4d_rows_pitch(y, ch * nm, &frames);
    if (xl < 0 || yl < 0 || !r4d_contig(b) || !r4d_contig(m)) return R4D_NO(RAD_E_STRIDE);
    if (S <= 0 || frames <= 0 || frames * hop > S || frames > INT32_MAX ||
        r4d_numel(b) < 2 * F * nfft || r4d_numel(m) < nm * F)
        return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(stft_logmel_kernel, dim3((unsigned)frames, (unsigned)ch), dim3(kT), 0,
                       (hipStream_t)s, x->data, b->data, m->data, y->data, (int)frames, (int)ch,
                       (int)hop, (int)nfft, (int)nm, xl, yl, is_bf16(x), is_bf16(b), is_bf16(m),
                       is_bf16(y));
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_bwe_up(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *rz = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !rz || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf_or_f32(x) || !is_bf_or_f32(rz) || !is_bf_or_f32(y)) return R4D_NO(RAD_E_DTYPE);
    const long long r = r4d_geti_or(a, "ratio", 0), ch = r4d_geti_or(a, "ch", 0);
    const int w = (int)std::ceil(6.0 / 0.99);
    const long long K = 2ll * w * r + 1;
    if (r <= 0 || ch <= 0 || K > kBweTaps) return R4D_NO(RAD_E_SHAPE);
    long long S = 0, O = 0, Rr = 0;
    const long long xl = r4d_rows_pitch(x, ch, &S), yl = r4d_rows_pitch(y, ch, &O);
    const long long rl = r4d_rows_pitch(rz, ch, &Rr);
    if (xl < 0 || yl < 0 || rl < 0) return R4D_NO(RAD_E_STRIDE);
    if (S <= 0 || O <= 0 || O > S * r || Rr < O) return R4D_NO(RAD_E_SHAPE);
    // libref's taps: f32 arithmetic as torch builds the filter, rounded to x's dtype.
    BweTaps f{};
    for (long long n = 0; n < K; ++n) {
        const float u = ((float)n / (float)r - (float)w) * 0.99f;
        const float uc = std::min(std::max(u, -6.0f), 6.0f);
        const float cw = std::cos(uc * (float)M_PI / 6.0f / 2.0f);
        const float pu = (float)M_PI * u;
        const float sinc = u == 0.0f ? 1.0f : std::sin(pu) / pu;
        const float v = sinc * (cw * cw) * 0.99f / (float)r;
        f.f[n] = is_bf16(x) ? rad_bf16_to_f32(rad_f32_to_bf16(v)) : v;
    }
    hipLaunchKernelGGL(bwe_up_kernel, dim3(grid_for(O * ch)), dim3(kT), 0, (hipStream_t)s,
                       x->data, rz->data, y->data, f, S, O, (int)ch, (int)r, w, (int)K, xl, rl, yl,
                       is_bf16(x), is_bf16(rz), is_bf16(y));
    return st_of(hipGetLastError());
}
