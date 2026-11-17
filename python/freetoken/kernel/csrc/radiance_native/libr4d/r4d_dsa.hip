#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_fwht.h"
#include "r4d_q8.h"
#include "r4d_fp8_wmma.h"
#include <climits>

namespace {
__global__ void dsa_query_kernel(const uint16_t* x, uint8_t* q, float* scale, long xp, long qp,
                                 long sp, int rows) {
    const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= rows) return;
    float v[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) v[d] = bf16_to_f32(x[(long)r * xp + lane * 4 + d]);
    r4d_fwht_wave<128>(v, lane);
    float am = 1e-4f;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        v[d] = bf16_to_f32(f32_to_bf16(v[d] * 0.08838834764831845f));
        am = fmaxf(am, fabsf(v[d]));
    }
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) am = fmaxf(am, __shfl_xor(am, mask, 32));
    const uint32_t bits = __float_as_uint(am * (1.0f / 448.0f));
    const float sc = __uint_as_float((bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0));
    if (!lane) scale[(long)r * sp] = sc;
    const uint32_t packed = q8_pack4<false>(v[0], v[1], v[2], v[3], 1.0f / sc);
    uint8_t* dst = q + (long)r * qp + lane * 4;
    if (((uintptr_t)dst & 3) == 0)
        *(uint32_t*)dst = packed;
    else {
#pragma unroll
        for (int d = 0; d < 4; ++d) dst[d] = (uint8_t)(packed >> (8 * d));
    }
}
int query_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const auto &x = a->t[0], &q = a->t[1], &s = a->t[2];
    if (x.dtype != RAD_BF16 || q.dtype != RAD_F8E4M3 || s.dtype != RAD_F32) return RAD_E_DTYPE;
    if (!x.data || !q.data || !s.data) return RAD_E_INVAL;
    if (r4d_geti_or(a, "n", 0) != 128 || x.rank != 2 || q.rank != 2 || s.rank != 2 ||
        x.shape[0] < 1 || x.shape[0] > INT_MAX - 3 || x.shape[1] != 128 ||
        q.shape[0] != x.shape[0] || q.shape[1] != 128 || s.shape[0] != x.shape[0] ||
        s.shape[1] != 1 || x.stride[1] != 1 || q.stride[1] != 1 || s.stride[1] != 1 ||
        x.stride[0] < 128 || q.stride[0] < 128 || s.stride[0] < 1)
        return RAD_E_SHAPE;
    return R4D_EMIT(stream, dsa_query_kernel, dim3((x.shape[0] + 3) / 4), dim3(128), 0,
                    (const uint16_t*)x.data, (uint8_t*)q.data, (float*)s.data, x.stride[0],
                    q.stride[0], s.stride[0], (int)x.shape[0]);
}
struct ScoreMeta {
    long qr, qh, qsr, qsc, gr, gc, kr, ksr, btr, btc, cur, ncr, op;
    int rows, blocks, seqs, pages, row0;
};
__device__ uint32_t bytes4(const uint8_t* p) {
    if (!((uintptr_t)p & 3)) return *(const uint32_t*)p;
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
/* Two waves cover all 32 heads and reuse the query across `kScoreTiles` 16-key tiles, one up to 32
 * blocks and two past them. FP8 dot products stay on WMMA; scales and the signed per-head gates
 * fold only after the per-head ReLU. The query operands start at a part's first row, so only the
 * sequence lookup sees the step row (`row0 + r`). */
template <int kScoreTiles>
__global__ void dsa_score_kernel(const uint8_t* q, const float* qs, const float* gate,
                                 const uint8_t* k, const float* ks, const int* bt, const int* cu,
                                 const int* nc, float* out, ScoreMeta m) {
    constexpr int kScoreKeys = 16 * kScoreTiles;
    __shared__ __align__(16) uint8_t qa[32 * 136], kb[kScoreKeys * 136];
    __shared__ int seq, pages[kScoreKeys];
    __shared__ float scales[kScoreKeys], coefficients[32], partial[2][kScoreKeys];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int r = blockIdx.y, b0 = blockIdx.x * kScoreKeys;
    if (!tid) {
        seq = -1;
        for (int s = 0; s < m.seqs; ++s)
            if (m.row0 + r >= cu[(long)s * m.cur] && m.row0 + r < cu[(long)(s + 1) * m.cur]) {
                seq = s;
                break;
            }
    }
    __syncthreads();
    const int live = seq >= 0 ? min(nc[(long)seq * m.ncr], m.blocks) : 0;
    if (b0 >= live) {
        for (int i = tid; i < kScoreKeys; i += 64)
            if (b0 + i < m.blocks) out[(long)r * m.op + b0 + i] = -INFINITY;
        return;
    }
    for (int i = tid; i < kScoreKeys; i += 64) {
        const int b = b0 + i;
        int p = seq >= 0 && b < m.blocks && b < nc[(long)seq * m.ncr]
                    ? bt[(long)seq * m.btr + b * m.btc]
                    : -1;
        if (p >= m.pages) p = -1;
        pages[i] = p;
        scales[i] = p >= 0 ? ks[(long)p * m.ksr] : 0;
    }
    if (tid < 32)
        coefficients[tid] = qs[(long)r * m.qsr + tid * m.qsc] * gate[(long)r * m.gr + tid * m.gc];
    for (int j = tid; j < 32 * 32; j += 64) {
        const int h = j / 32, d = (j % 32) * 4;
        *(uint32_t*)(qa + h * 136 + d) = bytes4(q + (long)r * m.qr + (long)h * m.qh + d);
    }
    __syncthreads();
    for (int j = tid; j < kScoreKeys * 32; j += 64) {
        const int b = j / 32, d = (j % 32) * 4, p = pages[b];
        *(uint32_t*)(kb + b * 136 + d) = p >= 0 ? bytes4(k + (long)p * m.kr + d) : 0;
    }
    __syncthreads();
    r4d_fp8::f8v acc[kScoreTiles];
#pragma unroll
    for (int tile = 0; tile < kScoreTiles; ++tile) acc[tile] = r4d_fp8::acc_zero();
#pragma unroll
    for (int d = 0; d < 128; d += 16) {
        const auto a = r4d_fp8::frag<128>(qa, wave * 16, d, lane);
#pragma unroll
        for (int tile = 0; tile < kScoreTiles; ++tile)
            acc[tile] = r4d_fp8::mma(a, r4d_fp8::frag<128>(kb, tile * 16, d, lane), acc[tile]);
    }
#pragma unroll
    for (int tile = 0; tile < kScoreTiles; ++tile) {
        float value = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            value += fmaxf(acc[tile][j], 0.0f) * coefficients[wave * 16 + (lane >> 4) * 8 + j];
        value += __shfl_xor(value, 16, 32);
        if (lane < 16) partial[wave][tile * 16 + lane] = value;
    }
    __syncthreads();
    for (int i = tid; i < kScoreKeys; i += 64)
        if (b0 + i < m.blocks)
            out[(long)r * m.op + b0 + i] =
                pages[i] >= 0 ? (partial[0][i] + partial[1][i]) * scales[i] * 0.015625f : -INFINITY;
}
/* The step's cu rebased to a part's first row: an op that reads cu over the part's own rows
 * (qsa_select) then finds each row's sequence unchanged. */
__global__ void dsa_rebase_kernel(const int* cu, int* out, long cs, long os, int n, int row0) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[(long)i * os] = cu[(long)i * cs] - row0;
}
int score_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 9) return RAD_E_INVAL;
    RadTensor q = a->t[0], qs = a->t[1], g = a->t[2];
    RadTensor k = a->t[3], ks = a->t[4];
    const bool packed = k.dtype == RAD_U8;
    if (packed) {
        if (ks.data || k.rank != 4 || k.shape[1] != 1 || k.shape[2] != 4 || k.shape[3] != 34 ||
            k.stride[3] != 1 || k.stride[2] != 34 || k.stride[0] < 136 || k.stride[0] % 4 ||
            ((uintptr_t)k.data & 3))
            return RAD_E_SHAPE;
        ks = {};
        ks.data = k.data ? (uint8_t*)k.data + 128 : nullptr;
        ks.dtype = RAD_F32;
        ks.rank = 1;
        ks.shape[0] = k.shape[0];
        ks.stride[0] = k.stride[0] / 4;
        k.dtype = RAD_F8E4M3;
        k.rank = 2;
        k.shape[1] = 128;
        k.stride[1] = 1;
    }
    const auto &bt = a->t[5], &cu = a->t[6], &nc = a->t[7], &o = a->t[8];
    const RadTensor* cu_part = rad_arg_in(a, 9);
    const long long part = r4d_geti_or(a, "row_part", -1);
    if (q.rank == 2 && qs.rank == 2 && qs.shape[1] == 1 && g.rank == 2 &&
        q.shape[0] == g.shape[0] * 32 && qs.shape[0] == q.shape[0]) {
        q.rank = 3;
        q.shape[0] = g.shape[0];
        q.shape[2] = q.shape[1];
        q.shape[1] = 32;
        q.stride[2] = q.stride[1];
        q.stride[1] = q.stride[0];
        q.stride[0] *= 32;
        qs.shape[0] = g.shape[0];
        qs.shape[1] = 32;
        qs.stride[1] = qs.stride[0];
        qs.stride[0] *= 32;
    }
    /* A part scores o's rows from step row part * rows; the query operands span the step. */
    const long long rows = o.rank == 2 ? o.shape[0] : 0, row0 = part > 0 ? part * rows : 0;
    if (part >= 0) {
        if (q.rank != 3 || qs.rank != 2 || g.rank != 2 || rows < 1 || part > INT_MAX / rows ||
            q.shape[0] < row0 + rows || qs.shape[0] != q.shape[0] || g.shape[0] != q.shape[0])
            return RAD_E_SHAPE;
        if (q.data && qs.data && g.data) {
            q.data = (uint8_t*)q.data + row0 * q.stride[0];
            qs.data = (float*)qs.data + row0 * qs.stride[0];
            g.data = (float*)g.data + row0 * g.stride[0];
        }
        q.shape[0] = qs.shape[0] = g.shape[0] = rows;
    }
    if (cu_part) {
        if (part < 0) return RAD_E_INVAL;
        if (cu_part->dtype != RAD_I32) return RAD_E_DTYPE;
        if (cu_part->rank != 1 || cu_part->shape[0] != cu.shape[0] || cu_part->stride[0] < 1)
            return RAD_E_SHAPE;
    }
    if (q.dtype != RAD_F8E4M3 || k.dtype != RAD_F8E4M3 || qs.dtype != RAD_F32 ||
        g.dtype != RAD_F32 || ks.dtype != RAD_F32 || o.dtype != RAD_F32 || bt.dtype != RAD_I32 ||
        cu.dtype != RAD_I32 || nc.dtype != RAD_I32)
        return RAD_E_DTYPE;
    for (int i = 0; i < 9; ++i)
        if (!(packed && i == 4) && !a->t[i].data) return RAD_E_INVAL;
    if (r4d_geti_or(a, "n", 0) != 128 || r4d_geti_or(a, "heads", 0) != 32 || q.rank != 3 ||
        q.shape[0] < 1 || q.shape[0] > 65535 || q.shape[1] != 32 || q.shape[2] != 128 ||
        q.stride[2] != 1 || q.stride[1] < 128 || q.stride[0] < 32 * q.stride[1] || qs.rank != 2 ||
        g.rank != 2 || qs.shape[0] != q.shape[0] || g.shape[0] != q.shape[0] || qs.shape[1] != 32 ||
        g.shape[1] != 32 || qs.stride[1] < 1 || g.stride[1] < 1 ||
        qs.stride[0] < 32 * qs.stride[1] || g.stride[0] < 32 * g.stride[1] || k.rank != 2 ||
        k.shape[0] < 1 || k.shape[0] > INT_MAX || k.shape[1] != 128 || k.stride[1] != 1 ||
        k.stride[0] < 128 || ks.rank != 1 || ks.shape[0] != k.shape[0] || ks.stride[0] < 1 ||
        bt.rank != 2 || bt.shape[0] < 1 || bt.shape[0] > INT_MAX || bt.stride[1] < 1 ||
        cu.rank != 1 || cu.shape[0] != bt.shape[0] + 1 || nc.rank != 1 ||
        nc.shape[0] != bt.shape[0] || cu.stride[0] < 1 || nc.stride[0] < 1 || o.rank != 2 ||
        o.shape[0] != q.shape[0] || o.shape[1] < 1 || o.shape[1] > INT_MAX - 15 ||
        o.shape[1] > bt.shape[1] || o.stride[1] != 1 || o.stride[0] < o.shape[1] ||
        bt.stride[0] < bt.shape[1] * bt.stride[1])
        return RAD_E_SHAPE;
    ScoreMeta m{ q.stride[0],      q.stride[1],    qs.stride[0], qs.stride[1],    g.stride[0],
                 g.stride[1],      k.stride[0],    ks.stride[0], bt.stride[0],    bt.stride[1],
                 cu.stride[0],     nc.stride[0],   o.stride[0],  (int)q.shape[0], (int)o.shape[1],
                 (int)bt.shape[0], (int)k.shape[0], (int)row0 };
    if (cu_part) {
        const int rc = R4D_EMIT(stream, dsa_rebase_kernel, dim3(1), dim3(64), 0,
                                (const int*)cu.data, (int*)cu_part->data, cu.stride[0],
                                cu_part->stride[0], (int)cu.shape[0], (int)row0);
        if (rc != RAD_OK) return rc;
    }
#define SCORE(NT)                                                                                  \
    R4D_EMIT(stream, (dsa_score_kernel<NT>), dim3((m.blocks + 16 * NT - 1) / (16 * NT), m.rows),   \
             dim3(64), 0, (const uint8_t*)q.data, (const float*)qs.data, (const float*)g.data,     \
             (const uint8_t*)k.data, (const float*)ks.data, (const int*)bt.data,                   \
             (const int*)cu.data, (const int*)nc.data, (float*)o.data, m)
    if (m.blocks <= 32) return SCORE(1);
    return SCORE(2);
#undef SCORE
}
/* One wave per pool: all softmax channels and the full Hadamard stay in registers. */
__global__ void dsa_pool_kernel(const uint16_t* kg, const float* ape, uint8_t* q, float* scale,
                                const int* page, uint8_t* cache, long kp, long ap, long qp, long sp,
                                long pp, long cp, int rows, int pages) {
    const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= rows) return;
    float v[4];
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const int col = lane * 4 + d;
        float g[4], mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            g[j] = bf16_to_f32(kg[((long)r * 4 + j) * kp + 128 + col]) + ape[(long)j * ap + col];
            mx = fmaxf(mx, g[j]);
        }
        float num = 0, den = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float p = expf(g[j] - mx);
            num += p * bf16_to_f32(kg[((long)r * 4 + j) * kp + col]);
            den += p;
        }
        v[d] = bf16_to_f32(f32_to_bf16(num / den));
    }
    r4d_fwht_wave<128>(v, lane);
    float am = 1e-4f;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        v[d] = bf16_to_f32(f32_to_bf16(v[d] * 0.08838834764831845f));
        am = fmaxf(am, fabsf(v[d]));
    }
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) am = fmaxf(am, __shfl_xor(am, mask, 32));
    const uint32_t bits = __float_as_uint(am * (1.0f / 448.0f));
    const float sc = __uint_as_float((bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0));
    if (!lane) scale[(long)r * sp] = sc;
    const uint32_t packed = q8_pack4<false>(v[0], v[1], v[2], v[3], 1.0f / sc);
    uint8_t* dst = q + (long)r * qp + lane * 4;
    if (((uintptr_t)dst & 3) == 0)
        *(uint32_t*)dst = packed;
    else {
#pragma unroll
        for (int d = 0; d < 4; ++d) dst[d] = (uint8_t)(packed >> (8 * d));
    }
    if (cache) {
        const int p = page[(long)r * pp];
        if (p >= 0 && p < pages) {
            uint8_t* out = cache + (long)p * cp;
            *(uint32_t*)(out + lane * 4) = packed;
            if (!lane) *(float*)(out + 128) = sc;
        }
    }
}
int pool_body(const RadArgs* a, RadStream stream) {
    if (r4d_geti_or(a, "block_size", 4) != 4) return RAD_E_SHAPE;
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    const auto &kg = a->t[0], &ape = a->t[1], &q = a->t[2], &s = a->t[3];
    const auto *page = rad_arg_in(a, 4), *cache = rad_arg_in(a, 5);
    if ((page != nullptr) != (cache != nullptr)) return RAD_E_INVAL;
    if (cache &&
        (page->dtype != RAD_I32 || page->rank != 1 || page->shape[0] != q.shape[0] ||
         page->stride[0] < 1 || cache->dtype != RAD_U8 || cache->rank != 4 || cache->shape[0] < 1 ||
         cache->shape[0] > INT_MAX || cache->shape[1] != 1 || cache->shape[2] != 4 ||
         cache->shape[3] != 34 || cache->stride[3] != 1 || cache->stride[2] != 34 ||
         cache->stride[0] < 136 || cache->stride[0] % 4 || ((uintptr_t)cache->data & 3)))
        return RAD_E_SHAPE;
    if (kg.dtype != RAD_BF16 || ape.dtype != RAD_F32 || q.dtype != RAD_F8E4M3 || s.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (r4d_geti_or(a, "n", 0) != 128 || r4d_geti_or(a, "ratio", 0) != 4 || kg.rank != 2 ||
        ape.rank != 2 || q.rank != 2 || s.rank != 2 || !kg.data || !ape.data || !q.data ||
        !s.data || q.shape[0] < 1 || q.shape[0] > INT_MAX / 4 || kg.shape[0] != q.shape[0] * 4 ||
        kg.shape[1] != 256 || ape.shape[0] != 4 || ape.shape[1] != 128 || q.shape[1] != 128 ||
        s.shape[0] != q.shape[0] || s.shape[1] != 1 || kg.stride[1] != 1 || ape.stride[1] != 1 ||
        q.stride[1] != 1 || s.stride[1] != 1 || kg.stride[0] < 256 || ape.stride[0] < 128 ||
        q.stride[0] < 128 || s.stride[0] < 1)
        return RAD_E_SHAPE;
    return R4D_EMIT(
        stream, dsa_pool_kernel, dim3((q.shape[0] + 3) / 4), dim3(128), 0, (const uint16_t*)kg.data,
        (const float*)ape.data, (uint8_t*)q.data, (float*)s.data,
        page ? (const int*)page->data : nullptr, cache ? (uint8_t*)cache->data : nullptr,
        kg.stride[0], ape.stride[0], q.stride[0], s.stride[0], page ? page->stride[0] : 0,
        cache ? cache->stride[0] : 0, (int)q.shape[0], cache ? (int)cache->shape[0] : 0);
}
}
extern "C" int r4d_rad_dsa_pool(const RadArgs* a, RadStream stream) { return pool_body(a, stream); }
extern "C" int r4d_rad_dsa_query(const RadArgs* a, RadStream stream) {
    return query_body(a, stream);
}
extern "C" int r4d_rad_dsa_score(const RadArgs* a, RadStream stream) {
    return score_body(a, stream);
}
