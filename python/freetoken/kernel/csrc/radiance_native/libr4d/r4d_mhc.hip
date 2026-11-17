/* GLM's mHC, kept separate from the Qwen gated-residual kernels. The GLM shape
 * projects in parallel before one workgroup per token forms the Sinkhorn matrix.
 * Mixing weights are declared f32; activations are bf16, coefficients stay f32. */
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {
__device__ float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down(v, d, 32);
    return v;
}

// The final collapsed output is large enough to hold the projection partials
// until the finishing kernel has read them. Align within each pitched row.
__device__ float* partials(uint16_t* x, long row, long pitch) {
    return reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(x + row * pitch) + 3) &
                                    ~uintptr_t(3));
}
// Where projection partial (row, coefficient j, chunk) lives: inside the collapsed output's rows
// (mhc_pre), or a separate [rows][chunk][stride] matrix (mhc_pre_partial writes this rank's chunks
// from `first`, mhc_pre_finish reads every rank's after the gather). Kernels differ in address only.
// `pad` clears a Dense chunk's slots past the last partial, so the gathered payload is defined.
struct InX {
    uint16_t* x;
    long xp;
    int chunks;
    __device__ float* at(long row, int j, int chunk) const {
        return partials(x, row, xp) + j * chunks + chunk;
    }
    __device__ void pad(long, int, int) const {}
};
struct Dense {
    float* p;
    long pp;
    int stride, first;
    __device__ float* at(long row, int j, int chunk) const {
        return p + row * pp + (long)(chunk - first) * stride + j;
    }
    __device__ void pad(long row, int last, int chunk) const {
        for (int j = last + 1; j < stride; ++j) *at(row, j, chunk) = 0;
    }
};
template <class Out>
__global__ void mhc_partial_kernel(const uint16_t* h, const float* fn, Out out, int width, int mix,
                                   int chunk0, long hp, long fp) {
    __shared__ float sums[4];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int j = blockIdx.y, chunk = chunk0 + blockIdx.x;
    const long row = blockIdx.z;
    float sum = 0;
    for (int i = chunk * 2048 + tid; i < min(width, (chunk + 1) * 2048); i += 128) {
        const float v = bf16_to_f32(h[row * hp + i]);
        sum += v * (j == mix ? v : fn[(long)j * fp + i]);
    }
    sum = warp_sum(sum);
    if (!lane) sums[wave] = sum;
    __syncthreads();
    if (!tid) {
        *out.at(row, j, chunk) = sums[0] + sums[1] + sums[2] + sums[3];
        if (j == mix) out.pad(row, mix, chunk);
    }
}

// Prefill reuses each input across four coefficients and each F32 mixing weight across four BF16
// token rows. Every output keeps mhc_partial_kernel's accumulation and reduction order.
template <class Out>
__global__ void mhc_partial_rows4_coeff4_kernel(const uint16_t* h, const float* fn, Out out,
                                                int width, int mix, int chunk0, long hp, long fp,
                                                long rows) {
    __shared__ float sums[4][4][4];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int first_j = blockIdx.y * 4, chunk = chunk0 + blockIdx.x;
    const long first = (long)blockIdx.z * 4;
    float sum[4][4]{};
    for (int i = chunk * 2048 + tid; i < min(width, (chunk + 1) * 2048); i += 128) {
        float weight[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
            weight[q] = first_j + q < mix ? fn[(long)(first_j + q) * fp + i] : 0;
#pragma unroll
        for (int r = 0; r < 4; ++r)
            if (first + r < rows) {
                const float v = bf16_to_f32(h[(first + r) * hp + i]);
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    if (first_j + q <= mix) sum[q][r] += v * (first_j + q == mix ? v : weight[q]);
            }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q)
        if (first_j + q <= mix) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                sum[q][r] = warp_sum(sum[q][r]);
                if (!lane) sums[q][r][wave] = sum[q][r];
            }
        }
    __syncthreads();
    if (tid < 16) {
        const int q = tid / 4, r = tid % 4;
        if (first_j + q <= mix && first + r < rows) {
            *out.at(first + r, first_j + q, chunk) =
                sums[q][r][0] + sums[q][r][1] + sums[q][r][2] + sums[q][r][3];
            if (first_j + q == mix) out.pad(first + r, mix, chunk);
        }
    }
}

// The collapsed read x[i] = sum_j pre[j] * h[j*n + i] eight columns a thread, in 16-byte loads
// and stores. Each element keeps the scalar loop's order (from zero, j ascending), so its bits
// are the same; false when the rows are not 16-byte aligned, and the caller runs its own loop.
__device__ __forceinline__ bool mix_rows_vec(const uint16_t* hr, uint16_t* xr, const float* pre,
                                             int n, int hc, int tid, int nt) {
    if ((((uintptr_t)hr | (uintptr_t)xr) & 15) || (n & 7)) return false;
    for (int i = tid * 8; i < n; i += nt * 8) {
        float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int j = 0; j < hc; ++j) {
            const uint4 q = *reinterpret_cast<const uint4*>(hr + (long)j * n + i);
            const uint32_t w[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int e = 0; e < 8; ++e)
                v[e] += pre[j] * bf16_to_f32((uint16_t)(w[e >> 1] >> (16 * (e & 1))));
        }
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
            o[e] = (uint32_t)f32_to_bf16(v[2 * e]) | ((uint32_t)f32_to_bf16(v[2 * e + 1]) << 16);
        *reinterpret_cast<uint4*>(xr + i) = uint4{ o[0], o[1], o[2], o[3] };
    }
    return true;
}

// The projection partials four slots a lane. The kernels above give each of 128 threads one slot
// s, the sum over the chunk's columns s, s+128, ... in order; here lane L holds slots 4L..4L+3,
// so a row is read 8 bytes and a weight 16 bytes a load, and one wave does a whole block's work.
// Their reduction is replayed exactly: warp_sum pairs slot l with l+16, l+8, l+4, l+2 and l+1,
// here lanes +4, +2, +1 and then the lane's own slots, and the four waves' sums, at lanes 0, 8,
// 16 and 24, are added in order. R rows and four coefficients a wave, four independent waves a
// block, no barrier.
template <class Out, int R>
__global__ __launch_bounds__(128) void mhc_partial_vec_kernel(const uint16_t* h, const float* fn,
                                                              Out out, int mix, int chunk0,
                                                              int count, long hp, long fp,
                                                              long rows, long items) {
    const int lane = threadIdx.x & 31;
    const long item = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (item >= items) return;
    const int groups = (mix + 4) / 4;
    const int chunk = chunk0 + (int)(item % count);
    const int first_j = (int)((item / count) % groups) * 4;
    const long first = item / count / groups * R;
    float acc[4][R][4]{};
    for (int t = 0; t < 16; ++t) {
        const int col = chunk * 2048 + t * 128 + lane * 4;
        float w[4][4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 v =
                first_j + q < mix
                    ? *reinterpret_cast<const float4*>(fn + (long)(first_j + q) * fp + col)
                    : float4{ 0, 0, 0, 0 };
            w[q][0] = v.x;
            w[q][1] = v.y;
            w[q][2] = v.z;
            w[q][3] = v.w;
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (first + r < rows) {
                const uint2 hv = *reinterpret_cast<const uint2*>(h + (first + r) * hp + col);
                const float v[4] = { bf16_to_f32((uint16_t)hv.x),
                                     bf16_to_f32((uint16_t)(hv.x >> 16)),
                                     bf16_to_f32((uint16_t)hv.y),
                                     bf16_to_f32((uint16_t)(hv.y >> 16)) };
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    if (first_j + q <= mix) {
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            acc[q][r][u] += v[u] * (first_j + q == mix ? v[u] : w[q][u]);
                    }
            }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q)
        if (first_j + q <= mix) {
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (first + r < rows) {
                    float c[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        float a = acc[q][r][u];
                        a += __shfl_down(a, 4, 32);
                        a += __shfl_down(a, 2, 32);
                        a += __shfl_down(a, 1, 32);
                        c[u] = a;
                    }
                    const float e0 = c[0] + c[2], e1 = c[1] + c[3], f = e0 + e1;
                    const float s1 = __shfl(f, 8, 32), s2 = __shfl(f, 16, 32),
                                s3 = __shfl(f, 24, 32);
                    if (!lane) {
                        *out.at(first + r, first_j + q, chunk) = f + s1 + s2 + s3;
                        if (first_j + q == mix) out.pad(first + r, mix, chunk);
                    }
                }
        }
}

// The geometries without the chunk split: one workgroup a row projects the residual itself from
// `fn`, then mixes and collapses it.
__global__ void mhc_pre_kernel(const uint16_t* h, const float* fn, const float* scale,
                               const float* base, uint16_t* x, float* post, float* comb, int n,
                               int hc, int iterations, float eps, float heps, long hp, long fp,
                               long xp, long pp, long cp) {
    __shared__ float sums[8], inverse, coefficients[80], pre[8], c[64];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const long r = blockIdx.x;
    const int width = hc * n, mix = hc * (hc + 2);
    float ss = 0;
    for (int i = tid; i < width; i += 256) {
        const float v = bf16_to_f32(h[r * hp + i]);
        ss += v * v;
    }
    ss = warp_sum(ss);
    if (!lane) sums[wave] = ss;
    __syncthreads();
    if (!tid) {
        float s = 0;
        for (int i = 0; i < 8; ++i) s += sums[i];
        inverse = rsqrtf(s / (float)width + eps);
    }
    __syncthreads();
    for (int j = wave; j < mix; j += 8) {
        float dot = 0;
        for (int i = lane; i < width; i += 32)
            dot += bf16_to_f32(h[r * hp + i]) * fn[(long)j * fp + i];
        dot = warp_sum(dot);
        if (!lane) coefficients[j] = dot * inverse;
    }
    __syncthreads();
    if (!tid) {
        for (int j = 0; j < hc; ++j) {
            pre[j] = 1.0f / (1.0f + expf(-(coefficients[j] * scale[0] + base[j]))) + heps;
            post[r * pp + j] =
                2.0f / (1.0f + expf(-(coefficients[hc + j] * scale[1] + base[hc + j])));
        }
        for (int j = 0; j < hc; ++j) {
            float mx = -INFINITY;
            for (int k = 0; k < hc; ++k) {
                const int q = 2 * hc + j * hc + k;
                c[j * hc + k] = coefficients[q] * scale[2] + base[q];
                mx = fmaxf(mx, c[j * hc + k]);
            }
            float sum = 0;
            for (int k = 0; k < hc; ++k) {
                c[j * hc + k] = expf(c[j * hc + k] - mx);
                sum += c[j * hc + k];
            }
            for (int k = 0; k < hc; ++k) c[j * hc + k] = c[j * hc + k] / sum + heps;
        }
        for (int it = 0; it < iterations; ++it) {
            if (it)
                for (int j = 0; j < hc; ++j) {
                    float sum = 0;
                    for (int k = 0; k < hc; ++k) sum += c[j * hc + k];
                    for (int k = 0; k < hc; ++k) c[j * hc + k] /= sum + heps;
                }
            for (int k = 0; k < hc; ++k) {
                float sum = 0;
                for (int j = 0; j < hc; ++j) sum += c[j * hc + k];
                for (int j = 0; j < hc; ++j) c[j * hc + k] /= sum + heps;
            }
        }
        for (int j = 0; j < hc * hc; ++j) comb[r * cp + j] = c[j];
    }
    __syncthreads();
    if (mix_rows_vec(h + r * hp, x + r * xp, pre, n, hc, tid, 256)) return;
    for (int i = tid; i < n; i += 256) {
        float v = 0;
        for (int j = 0; j < hc; ++j) v += pre[j] * bf16_to_f32(h[r * hp + j * n + i]);
        x[r * xp + i] = f32_to_bf16(v);
    }
}

// One wave handles GLM's 4x4 Sinkhorn matrix. Keep row/column sums in the
// scalar reference order; the other rows never participate in these shuffles.
// Writes post and comb; returns pre[lane] in lanes 0..3.
template <class In>
__device__ __forceinline__ float glm_warp_mix(int lane, long row, const float* scale,
                                              const float* base, float* post, float* comb,
                                              int iterations, float eps, float heps, long pp,
                                              long cp, const In& in) {
    float inverse = 0;
    if (!lane) {
        float ss = 0;
        for (int i = 0; i < 8; ++i) ss += *in.at(row, 24, i);
        inverse = rsqrtf(ss / 16384.0f + eps);
    }
    inverse = __shfl(inverse, 0, 32);
    float dot = 0;
    if (lane < 24) {
        float v[8];
        for (int i = 0; i < 8; ++i) v[i] = *in.at(row, lane, i);
        // The existing eight-chunk warp reduction pairs 0/4, 2/6, 1/5, 3/7.
        dot = ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
    }
    const float coefficient = dot * inverse;
    const float post_coefficient = __shfl(coefficient, (lane + 4) & 31, 32);
    const float matrix_coefficient = __shfl(coefficient, (lane + 8) & 31, 32);
    float pre = 0;
    if (lane < 4) {
        pre = 1.0f / (1.0f + expf(-(coefficient * scale[0] + base[lane]))) + heps;
        post[row * pp + lane] =
            2.0f / (1.0f + expf(-(post_coefficient * scale[1] + base[4 + lane])));
    }
    float value = lane < 16 ? matrix_coefficient * scale[2] + base[8 + lane] : 0;
    const int row_base = lane & ~3, col = lane & 3;
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j) mx = fmaxf(mx, __shfl(value, row_base + j, 32));
    value = expf(value - mx);
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) sum += __shfl(value, row_base + j, 32);
    value = value / sum + heps;
    for (int it = 0; it < iterations; ++it) {
        if (it) {
            sum = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) sum += __shfl(value, row_base + j, 32);
            value /= sum + heps;
        }
        sum = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sum += __shfl(value, col + 4 * j, 32);
        value /= sum + heps;
    }
    if (lane < 16) comb[row * cp + lane] = value;
    return pre;
}
// glm_warp_mix on wave 0, then the collapsed read over the whole block: one wave writing 4096
// columns a row would be most of a decode step's mhc_pre.
template <class In>
__global__ __launch_bounds__(256) void mhc_pre_glm_wide_kernel(
    const uint16_t* h, const float* scale, const float* base, uint16_t* x, float* post, float* comb,
    int iterations, float eps, float heps, long hp, long xp, long pp, long cp, In in) {
    __shared__ float spre[4];
    const int tid = threadIdx.x;
    const long row = blockIdx.x;
    if (tid < 32) {
        const float pre =
            glm_warp_mix(tid, row, scale, base, post, comb, iterations, eps, heps, pp, cp, in);
        if (tid < 4) spre[tid] = pre;
    }
    __syncthreads();
    const float pre[4] = { spre[0], spre[1], spre[2], spre[3] };
    if (mix_rows_vec(h + row * hp, x + row * xp, pre, 4096, 4, tid, 256)) return;
    for (int i = tid; i < 4096; i += 256) {
        float v = 0;
        v += pre[0] * bf16_to_f32(h[row * hp + i]);
        v += pre[1] * bf16_to_f32(h[row * hp + 4096 + i]);
        v += pre[2] * bf16_to_f32(h[row * hp + 8192 + i]);
        v += pre[3] * bf16_to_f32(h[row * hp + 12288 + i]);
        x[row * xp + i] = f32_to_bf16(v);
    }
}

__global__ void mhc_post_kernel(const uint16_t* x, const uint16_t* h, const float* post,
                                const float* comb, uint16_t* out, int n, int hc, long xp, long hp,
                                long pp, long cp, long op) {
    const long r = blockIdx.y;
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n * hc) return;
    const int k = q / n, i = q % n;
    float v = 0;
    for (int j = 0; j < hc; ++j)
        v += comb[r * cp + j * hc + k] * bf16_to_f32(h[r * hp + j * n + i]);
    v += post[r * pp + k] * bf16_to_f32(x[r * xp + i]);
    out[r * op + q] = f32_to_bf16(v);
}

// mhc_post at four streams, eight columns of every output stream a thread: each input word is
// loaded once instead of once per output stream. Every output keeps mhc_post_kernel's order.
__global__ void mhc_post_vec_kernel(const uint16_t* x, const uint16_t* h, const float* post,
                                    const float* comb, uint16_t* out, int n, long xp, long hp,
                                    long pp, long cp, long op) {
    const long r = blockIdx.y;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= n) return;
    float hv[4][8], xv[8];
    auto unpack = [](const uint16_t* p, float* v) {
        const uint4 q = *reinterpret_cast<const uint4*>(p);
        const uint32_t w[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = bf16_to_f32((uint16_t)(w[e >> 1] >> (16 * (e & 1))));
    };
#pragma unroll
    for (int j = 0; j < 4; ++j) unpack(h + r * hp + (long)j * n + i, hv[j]);
    unpack(x + r * xp + i, xv);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float c[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) c[j] = comb[r * cp + j * 4 + k];
        const float pk = post[r * pp + k];
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
            float v0 = 0, v1 = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v0 += c[j] * hv[j][e];
                v1 += c[j] * hv[j][e + 1];
            }
            v0 += pk * xv[e];
            v1 += pk * xv[e + 1];
            o[e >> 1] = (uint32_t)f32_to_bf16(v0) | ((uint32_t)f32_to_bf16(v1) << 16);
        }
        *reinterpret_cast<uint4*>(out + r * op + (long)k * n + i) = uint4{ o[0], o[1], o[2], o[3] };
    }
}

__global__ void mhc_contract_kernel(const uint16_t* h, uint16_t* x, int n, int hc, long hp,
                                    long xp) {
    const long r = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0;
    for (int j = 0; j < hc; ++j) v += bf16_to_f32(h[r * hp + j * n + i]);
    x[r * xp + i] = f32_to_bf16(v / (float)hc);
}

bool matrix(const RadTensor& t, uint32_t dt, long rows, long cols) {
    return t.data && t.dtype == dt && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols &&
           t.stride[1] == 1 && t.stride[0] >= cols;
}
bool vector(const RadTensor& t, long n) {
    return t.data && t.dtype == RAD_F32 && t.rank == 1 && t.shape[0] == n && t.stride[0] == 1;
}
bool geometry(const RadArgs* a, int* n, int* hc) {
    const long nn = r4d_geti_or(a, "n", 0), hh = r4d_geti_or(a, "hc", 0);
    if (nn < 1 || nn > 16384 || hh < 2 || hh > 8) return false;
    *n = (int)nn;
    *hc = (int)hh;
    return true;
}
// GLM's split projection: eight 2048-column chunks of the 16384-wide residual, 24 coefficients
// and the sum of squares each. A Dense chunk holds those 25 partials in kPartStride slots.
constexpr int kChunkCols = 2048, kPartStride = 28;
template <class Out>
int emit_partials(RadStream stream, const RadTensor& h, const RadTensor& fn, Out out, int n, int hc,
                  int chunk0, int count, long rows) {
    /* From 2 rows the four-slot form, on 16-byte weight and 8-byte residual rows; GLM's
     * 2048-column chunks only. Measured cold on gfx1201 at GLM's shape: one row a wave leads up
     * to 16 rows (13 us against 20 to 36 for the one-slot kernels), two rows from 48 (20 against
     * 22 us, at 128 rows 34 against 39, at 2048 391 against 500); four rows a wave trail two at
     * every count. */
    const bool vec_aligned = ((uintptr_t)fn.data & 15) == 0 && fn.stride[0] % 4 == 0 &&
                             ((uintptr_t)h.data & 7) == 0 && h.stride[0] % 4 == 0 &&
                             hc * n % kChunkCols == 0;
    if (vec_aligned && rows >= 2) {
        const int mix = hc * (hc + 2), groups = (mix + 4) / 4;
#define MHC_VEC(RR)                                                                                \
    do {                                                                                           \
        const long items = (long)count * groups * ((rows + RR - 1) / RR);                          \
        return R4D_EMIT(stream, (mhc_partial_vec_kernel<Out, RR>),                                 \
                        dim3((unsigned)((items + 3) / 4)), dim3(128), 0, (const uint16_t*)h.data,  \
                        (const float*)fn.data, out, mix, chunk0, count, (long)h.stride[0],         \
                        (long)fn.stride[0], rows, items);                                          \
    } while (0)
        if (rows >= 32) MHC_VEC(2);
        MHC_VEC(1);
#undef MHC_VEC
    }
    /* Otherwise four rows and four coefficients a block from 32 rows, one row and one coefficient
     * below. */
    constexpr long kRows4Min = 32;
    if (rows >= kRows4Min)
        return R4D_EMIT(stream, (mhc_partial_rows4_coeff4_kernel<Out>),
                        dim3(count, (hc * (hc + 2) + 4) / 4, (unsigned)((rows + 3) / 4)), dim3(128),
                        0, (const uint16_t*)h.data, (const float*)fn.data, out, hc * n,
                        hc * (hc + 2), chunk0, (long)h.stride[0], (long)fn.stride[0], rows);
    return R4D_EMIT(stream, (mhc_partial_kernel<Out>),
                    dim3(count, hc * (hc + 2) + 1, (unsigned)rows), dim3(128), 0,
                    (const uint16_t*)h.data, (const float*)fn.data, out, hc * n, hc * (hc + 2),
                    chunk0, (long)h.stride[0], (long)fn.stride[0]);
}
// The finishing dispatch. `split` says the partials exist (in `in`); without them
// mhc_pre_kernel projects the residual itself from `fn`.
template <class In>
int emit_finish(RadStream stream, const RadTensor& h, const float* fn, long fp, const RadTensor& s,
                const RadTensor& b, const RadTensor& x, const RadTensor& p, const RadTensor& c,
                int n, int hc, int iters, float eps, float heps, long rows, bool split, In in) {
    /* The warp mix at every row count: byte-identical to the scalar one and a quarter of its time
     * at one to six rows. */
    if (split)
        return R4D_EMIT(stream, (mhc_pre_glm_wide_kernel<In>), dim3((unsigned)rows), dim3(256), 0,
                        (const uint16_t*)h.data, (const float*)s.data, (const float*)b.data,
                        (uint16_t*)x.data, (float*)p.data, (float*)c.data, iters, eps, heps,
                        (long)h.stride[0], (long)x.stride[0], (long)p.stride[0], (long)c.stride[0],
                        in);
    return R4D_EMIT(stream, mhc_pre_kernel, dim3((unsigned)rows), dim3(256), 0,
                    (const uint16_t*)h.data, fn, (const float*)s.data, (const float*)b.data,
                    (uint16_t*)x.data, (float*)p.data, (float*)c.data, n, hc, iters, eps, heps,
                    (long)h.stride[0], fp, (long)x.stride[0], (long)p.stride[0], (long)c.stride[0]);
}
bool coefficients(const RadArgs* a, long* iters, float* eps, float* heps) {
    *iters = r4d_geti_or(a, "iterations", 0);
    *eps = (float)r4d_getf(a, "eps", 0);
    *heps = (float)r4d_getf(a, "hc_eps", 0);
    return *iters >= 1 && *iters <= 256 && std::isfinite(*eps) && std::isfinite(*heps) &&
           *eps > 0 && *heps >= 0;
}
// mhc_pre_partial and mhc_pre_finish serve GLM's split geometry only.
bool glm_chunks(const RadArgs* a, int n, int hc) {
    return n == 4096 && hc == 4 && r4d_geti_or(a, "chunk", 0) == kChunkCols;
}
int pre_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    int n, hc;
    if (!geometry(a, &n, &hc)) return RAD_E_SHAPE;
    long iters;
    float eps, heps;
    if (!coefficients(a, &iters, &eps, &heps)) return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &s = a->t[2], &b = a->t[3];
    const auto &x = a->t[4], &p = a->t[5], &c = a->t[6];
    if (h.rank != 2 || h.shape[0] < 1 || h.shape[0] > INT_MAX) return RAD_E_SHAPE;
    const long rows = h.shape[0];
    if (!matrix(h, RAD_BF16, rows, hc * n) || !matrix(fn, RAD_F32, hc * (hc + 2), hc * n) ||
        !vector(s, 3) || !vector(b, hc * (hc + 2)) || !matrix(x, RAD_BF16, rows, n) ||
        !matrix(p, RAD_F32, rows, hc) || !matrix(c, RAD_F32, rows, hc * hc))
        return RAD_E_SHAPE;
    const int chunks = (hc * n + kChunkCols - 1) / kChunkCols;
    const bool split =
        n == 4096 && hc == 4 && rows <= 65535 && (hc * (hc + 2) + 1) * chunks * 4 + 2 <= n * 2;
    const InX in{ (uint16_t*)x.data, (long)x.stride[0], chunks };
    if (split) {
        const int rc = emit_partials(stream, h, fn, in, n, hc, 0, chunks, rows);
        if (rc < 0) return rc;
    }
    return emit_finish(stream, h, (const float*)fn.data, (long)fn.stride[0], s, b, x, p, c, n, hc,
                       (int)iters, eps, heps, rows, split, in);
}
int partial_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    int n, hc;
    if (!geometry(a, &n, &hc) || !glm_chunks(a, n, hc)) return RAD_E_SHAPE;
    const int total = hc * n / kChunkCols;
    const long first = r4d_geti_or(a, "first", -1), count = r4d_geti_or(a, "chunks", 0);
    if (first < 0 || count < 1 || first + count > total) return RAD_E_SHAPE;
    const auto &h = a->t[0], &fn = a->t[1], &part = a->t[2];
    if (h.rank != 2 || h.shape[0] < 1 || h.shape[0] > 65535) return RAD_E_SHAPE;
    const long rows = h.shape[0];
    if (!matrix(h, RAD_BF16, rows, hc * n) || !matrix(fn, RAD_F32, hc * (hc + 2), hc * n) ||
        !matrix(part, RAD_F32, rows, count * kPartStride))
        return RAD_E_SHAPE;
    const Dense out{ (float*)part.data, (long)part.stride[0], kPartStride, (int)first };
    return emit_partials(stream, h, fn, out, n, hc, (int)first, (int)count, rows);
}
int finish_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    int n, hc;
    if (!geometry(a, &n, &hc) || !glm_chunks(a, n, hc)) return RAD_E_SHAPE;
    long iters;
    float eps, heps;
    if (!coefficients(a, &iters, &eps, &heps)) return RAD_E_SHAPE;
    const int chunks = hc * n / kChunkCols;
    const auto &part = a->t[0], &h = a->t[1], &s = a->t[2], &b = a->t[3];
    const auto &x = a->t[4], &p = a->t[5], &c = a->t[6];
    if (h.rank != 2 || h.shape[0] < 1 || h.shape[0] > 65535) return RAD_E_SHAPE;
    const long rows = h.shape[0];
    if (!matrix(part, RAD_F32, rows, chunks * kPartStride) || !matrix(h, RAD_BF16, rows, hc * n) ||
        !vector(s, 3) || !vector(b, hc * (hc + 2)) || !matrix(x, RAD_BF16, rows, n) ||
        !matrix(p, RAD_F32, rows, hc) || !matrix(c, RAD_F32, rows, hc * hc))
        return RAD_E_SHAPE;
    const Dense in{ (float*)part.data, (long)part.stride[0], kPartStride, 0 };
    return emit_finish(stream, h, nullptr, 0, s, b, x, p, c, n, hc, (int)iters, eps, heps, rows,
                       true, in);
}
int post_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    int n, hc;
    if (!geometry(a, &n, &hc)) return RAD_E_SHAPE;
    const auto &x = a->t[0], &h = a->t[1], &p = a->t[2], &c = a->t[3], &o = a->t[4];
    if (x.rank != 2 || x.shape[0] < 1 || x.shape[0] > 65535) return RAD_E_SHAPE;
    const long rows = x.shape[0];
    if (!matrix(x, RAD_BF16, rows, n) || !matrix(h, RAD_BF16, rows, hc * n) ||
        !matrix(p, RAD_F32, rows, hc) || !matrix(c, RAD_F32, rows, hc * hc) ||
        !matrix(o, RAD_BF16, rows, hc * n) || o.data == h.data)
        return RAD_E_SHAPE;
    const auto aligned = [](const RadTensor& t) {
        return ((uintptr_t)t.data & 15) == 0 && t.stride[0] % 8 == 0;
    };
    if (hc == 4 && n % 8 == 0 && aligned(x) && aligned(h) && aligned(o))
        return R4D_EMIT(stream, mhc_post_vec_kernel, dim3((n / 8 + 255) / 256, (unsigned)rows),
                        dim3(256), 0, (const uint16_t*)x.data, (const uint16_t*)h.data,
                        (const float*)p.data, (const float*)c.data, (uint16_t*)o.data, n,
                        (long)x.stride[0], (long)h.stride[0], (long)p.stride[0], (long)c.stride[0],
                        (long)o.stride[0]);
    return R4D_EMIT(stream, mhc_post_kernel, dim3((n * hc + 255) / 256, (unsigned)rows), dim3(256),
                    0, (const uint16_t*)x.data, (const uint16_t*)h.data, (const float*)p.data,
                    (const float*)c.data, (uint16_t*)o.data, n, hc, (long)x.stride[0],
                    (long)h.stride[0], (long)p.stride[0], (long)c.stride[0], (long)o.stride[0]);
}
int contract_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 2) return RAD_E_INVAL;
    int n, hc;
    if (!geometry(a, &n, &hc)) return RAD_E_SHAPE;
    const auto &h = a->t[0], &x = a->t[1];
    if (h.rank != 2 || h.shape[0] < 1 || h.shape[0] > 65535) return RAD_E_SHAPE;
    const long rows = h.shape[0];
    if (!matrix(h, RAD_BF16, rows, hc * n) || !matrix(x, RAD_BF16, rows, n)) return RAD_E_SHAPE;
    return R4D_EMIT(stream, mhc_contract_kernel, dim3((n + 255) / 256, (unsigned)rows), dim3(256),
                    0, (const uint16_t*)h.data, (uint16_t*)x.data, n, hc, (long)h.stride[0],
                    (long)x.stride[0]);
}
}
extern "C" int r4d_rad_mhc_pre(const RadArgs* a, RadStream stream) { return pre_body(a, stream); }
extern "C" int r4d_rad_mhc_pre_partial(const RadArgs* a, RadStream stream) {
    return partial_body(a, stream);
}
extern "C" int r4d_rad_mhc_pre_finish(const RadArgs* a, RadStream stream) {
    return finish_body(a, stream);
}
extern "C" int r4d_rad_mhc_post(const RadArgs* a, RadStream stream) { return post_body(a, stream); }
extern "C" int r4d_rad_mhc_contract(const RadArgs* a, RadStream stream) {
    return contract_body(a, stream);
}
