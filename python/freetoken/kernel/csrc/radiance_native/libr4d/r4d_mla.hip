/* Absorbed MLA keeps a single 512-wide latent per token. Decode splits the
 * selected history across waves, then merges FP32 softmax statistics. It never
 * materializes per-head K/V in the growing cache. */
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_fp8_wmma.h"
#include "r4d_gdn_wmma.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>

namespace {
struct ProjectMeta {
    long xr, xh, wp, yr, yh;
    int rows, heads;
};
/* Absorption streams adjacent latent columns; expansion reduces a contiguous weight row. R rows
 * a thread (absorb) or a wave (expand): each weight element is loaded once for R rows. Every row's
 * sum runs over j in the same order whatever R is (expand: each lane's share in ascending j, then
 * r4d_xor_sum<32>), so a row's bytes do not depend on the batch it is in. A tail group past the
 * last row reads that row again and stores nothing for it. */
template <int R, int HD>
__global__ void project_absorb_rows(const uint16_t* x, const uint16_t* w, uint16_t* y,
                                    ProjectMeta m) {
    const int col = blockIdx.x * 128 + threadIdx.x, head = blockIdx.y, r0 = blockIdx.z * R;
    if (col >= 512) return;
    const uint16_t* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = x + (long)min(r0 + r, m.rows - 1) * m.xr + head * m.xh;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0;
    for (int j = 0; j < HD; ++j) {
        const float wv = bf16_to_f32(w[((long)head * 2 * HD + j) * m.wp + col]);
#pragma unroll
        for (int r = 0; r < R; ++r) sum[r] = fmaf(bf16_to_f32(xr[r][j]), wv, sum[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (r0 + r < m.rows) y[(long)(r0 + r) * m.yr + head * m.yh + col] = f32_to_bf16(sum[r]);
}
template <int R, int HD>
__global__ void project_expand_rows(const uint16_t* x, const uint16_t* w, uint16_t* y,
                                    ProjectMeta m) {
    const int lane = threadIdx.x & 31, col = blockIdx.x * 4 + (threadIdx.x >> 5), head = blockIdx.y,
              r0 = blockIdx.z * R;
    const uint16_t* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = x + (long)min(r0 + r, m.rows - 1) * m.xr + head * m.xh;
    const uint16_t* const wr = w + ((long)head * 2 * HD + HD + col) * m.wp;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0;
    for (int j = lane; j < 512; j += 32) {
        const float wv = bf16_to_f32(wr[j]);
#pragma unroll
        for (int r = 0; r < R; ++r) sum[r] = fmaf(bf16_to_f32(xr[r][j]), wv, sum[r]);
    }
    float mine = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        sum[r] = r4d_xor_sum<32>(sum[r]);
        if (lane == r) mine = sum[r];
    }
    if (lane < R && r0 + lane < m.rows)
        y[(long)(r0 + lane) * m.yr + head * m.yh + col] = f32_to_bf16(mine);
}
int project_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    RadTensor x = a->t[0], y = a->t[2];
    const auto& w = a->t[1];
    const char* mode = r4d_gets(a, "mode");
    const bool absorb = mode && !std::strcmp(mode, "absorb"),
               expand = mode && !std::strcmp(mode, "expand");
    const long h = r4d_geti_or(a, "heads", 0), hd = r4d_geti_or(a, "head_dim", 0);
    auto heads_view = [&](RadTensor& t, long width) {
        if (h > 0 && t.rank == 2 && t.shape[1] == h * width) {
            t.rank = 3;
            t.shape[1] = h;
            t.shape[2] = width;
            t.stride[2] = t.stride[1];
            t.stride[1] *= width;
        }
    };
    heads_view(x, absorb ? hd : 512);
    heads_view(y, absorb ? 512 : hd);
    if (x.dtype != RAD_BF16 || w.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
    if ((!absorb && !expand) || h < 1 || h > 64 || (hd != 128 && hd != 256) ||
        r4d_geti_or(a, "latent_dim", 0) != 512 || x.rank != 3 || w.rank != 2 || y.rank != 3 ||
        x.shape[0] < 1 || x.shape[0] > 65535 || x.shape[1] != h ||
        x.shape[2] != (absorb ? hd : 512) || w.shape[0] != h * 2 * hd || w.shape[1] != 512 ||
        y.shape[0] != x.shape[0] || y.shape[1] != h || y.shape[2] != (absorb ? 512 : hd) ||
        x.stride[2] != 1 || w.stride[1] != 1 || y.stride[2] != 1 || x.stride[1] < x.shape[2] ||
        x.stride[0] < h * x.stride[1] || w.stride[0] < 512 || y.stride[1] < y.shape[2] ||
        y.stride[0] < h * y.stride[1])
        return RAD_E_SHAPE;
    const ProjectMeta m{ x.stride[0], x.stride[1],     w.stride[0], y.stride[0],
                         y.stride[1], (int)x.shape[0], (int)h };
    const auto* xp = (const uint16_t*)x.data;
    const auto* wp = (const uint16_t*)w.data;
    auto* yp = (uint16_t*)y.data;
    /* The largest of 8, 4, 2 and 1 rows a group that the batch fills. */
    const int R = m.rows >= 8 ? 8 : m.rows >= 4 ? 4 : m.rows >= 2 ? 2 : 1;
#define MLA_PROJECT_HD(RR, HD)                                                                     \
    do {                                                                                           \
        const unsigned groups = (unsigned)((m.rows + RR - 1) / RR);                                \
        if (absorb)                                                                                \
            return R4D_EMIT(stream, (project_absorb_rows<RR, HD>), dim3(4, h, groups), dim3(128),  \
                            0, xp, wp, yp, m);                                                     \
        return R4D_EMIT(stream, (project_expand_rows<RR, HD>), dim3(HD / 4, h, groups), dim3(128), \
                        0, xp, wp, yp, m);                                                         \
    } while (0)
#define MLA_PROJECT_ROWS(RR)                                                                       \
    do {                                                                                           \
        if (hd == 128) MLA_PROJECT_HD(RR, 128);                                                    \
        MLA_PROJECT_HD(RR, 256);                                                                   \
    } while (0)
    if (R == 8) MLA_PROJECT_ROWS(8);
    if (R == 4) MLA_PROJECT_ROWS(4);
    if (R == 2) MLA_PROJECT_ROWS(2);
    MLA_PROJECT_ROWS(1);
#undef MLA_PROJECT_ROWS
#undef MLA_PROJECT_HD
}
constexpr int D = 512, PART = D + 2;
constexpr int64_t scratch_bytes = 512ll * PART * 4;
constexpr int64_t matrix_scratch_bytes = 2048ll * PART * 4;
/* r4d_mla_matrix_pin_form: 0 is the launcher's rule (vector key loads where the cache is
 * aligned), 1 the element-wise key load. */
int g_mla_matrix_form = 0;
struct Meta {
    long qr, qh, kp, kt, bp, bc, lp, orr, oh;
    int rows, heads, blocks, bs, pages, splits;
    float scale;
};
/* RADIANCE_MLA_CONFLICT_STATS: per device, the 16-token passes the matrix row ran, how many of them
 * were a pair's second pass, and the chunks of the max(length) raster a pass-free pairing would
 * run. Read back by r4d_mla_matrix_stats, and printed every 1024th counted launch. */
__device__ unsigned long long g_mla_stats[3];
__device__ void load_latent(const uint16_t* p, float (&v)[16]) {
    if (!((uintptr_t)p & 15)) {
        const uint4 a = *(const uint4*)p, b = *(const uint4*)(p + 8);
        const uint32_t packed[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
#pragma unroll
        for (int j = 0; j < 16; ++j)
            v[j] = bf16_to_f32((uint16_t)(packed[j / 2] >> (16 * (j & 1))));
    } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = bf16_to_f32(p[j]);
    }
}
/* The latent cache is bf16, or E4M3 as kv_store_fp8 writes it at its default descale of 1, so a
 * code is the value itself. Every E4M3 value is a bf16 (three mantissa bits against seven, an
 * exponent range bf16 contains), so an fp8 key decodes exactly to the bf16 the tiles hold and every
 * kernel's arithmetic past the load is its bf16 instance's. The loads below are overloaded on the
 * cache's element type; at uint16_t each is the plain load the bf16 kernels always made. */
template <bool HI>
__device__ __forceinline__ uint32_t e4m3x2_bf16(uint32_t w) {
    const v2f f = r4d_fp8::e4m3_pk<HI>(w);
    return (__float_as_uint(f.x) >> 16) | (__float_as_uint(f.y) & 0xffff0000u);
}
__device__ __forceinline__ void load_latent(const uint8_t* p, float (&v)[16]) {
    if (!((uintptr_t)p & 15)) {
        const uint4 a = *(const uint4*)p;
        const uint32_t packed[4] = { a.x, a.y, a.z, a.w };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const v2f lo = r4d_fp8::e4m3_pk<false>(packed[j]),
                      hi = r4d_fp8::e4m3_pk<true>(packed[j]);
            v[4 * j] = lo.x;
            v[4 * j + 1] = lo.y;
            v[4 * j + 2] = hi.x;
            v[4 * j + 3] = hi.y;
        }
    } else {
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = r4d_fp8::e4m3(p[j]);
    }
}
/* Eight consecutive cache elements as four bf16 pairs: one 16-byte load at bf16, one 8-byte load
 * at E4M3. The caller keeps the 16-byte alignment the bf16 load needs, which covers the E4M3 one. */
__device__ __forceinline__ uint4 load8(const uint16_t* p) { return *(const uint4*)p; }
__device__ __forceinline__ uint4 load8(const uint8_t* p) {
    const uint2 raw = *(const uint2*)p;
    return make_uint4(e4m3x2_bf16<false>(raw.x), e4m3x2_bf16<true>(raw.x),
                      e4m3x2_bf16<false>(raw.y), e4m3x2_bf16<true>(raw.y));
}
/* One cache element as bf16 bits. */
__device__ __forceinline__ uint16_t elem(const uint16_t* p) { return *p; }
__device__ __forceinline__ uint16_t elem(const uint8_t* p) {
    return (uint16_t)(__float_as_uint(r4d_fp8::e4m3(*p)) >> 16);
}
template <typename K>
__global__ __launch_bounds__(128) void attend(const uint16_t* q, const K* kv, const int* bt,
                                              const int* len, uint16_t* out, float* scratch,
                                              Meta m) {
    const int lane = threadIdx.x & 31, head = blockIdx.y * 4 + (threadIdx.x >> 5), r = blockIdx.x,
              split = blockIdx.z;
    if (head >= m.heads) return;
    const int live = min(max(len[(long)r * m.lp], 0), m.blocks * m.bs);
    const int begin = (int)((long)live * split / m.splits),
              end = (int)((long)live * (split + 1) / m.splits);
    float query[16], value[16] = {};
    load_latent(q + (long)r * m.qr + (long)head * m.qh + lane * 16, query);
    float max_score = -INFINITY, den = 0;
    for (int token = begin; token < end; ++token) {
        const int page = bt[(long)r * m.bp + (token / m.bs) * m.bc];
        if (page < 0 || page >= m.pages) continue;
        float key[16];
        load_latent(kv + (long)page * m.kp + (token % m.bs) * m.kt + lane * 16, key);
        float score = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) score += query[j] * key[j];
        score = r4d_wave_sum(score) * m.scale;
        const bool higher = score > max_score;
        const float weight = expf(higher ? max_score - score : score - max_score);
        if (higher) {
#pragma unroll
            for (int j = 0; j < 16; ++j) value[j] = value[j] * weight + key[j];
            den = den * weight + 1;
            max_score = score;
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) value[j] += key[j] * weight;
            den += weight;
        }
    }
    if (m.splits == 1) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
            out[(long)r * m.orr + (long)head * m.oh + lane * 16 + j] =
                f32_to_bf16(den > 0 ? value[j] / den : 0);
    } else {
        float* dst = scratch + (((long)r * m.heads + head) * m.splits + split) * PART;
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[lane * 16 + j] = value[j];
        if (!lane) {
            dst[D] = max_score;
            dst[D + 1] = den;
        }
    }
}
__global__ __launch_bounds__(128) void merge(const float* scratch, uint16_t* out, Meta m) {
    const int lane = threadIdx.x & 31, head = blockIdx.y * 4 + (threadIdx.x >> 5), r = blockIdx.x;
    if (head >= m.heads) return;
    const float* src = scratch + ((long)r * m.heads + head) * m.splits * PART;
    float max_score = -INFINITY;
    for (int s = 0; s < m.splits; ++s)
        if (src[s * PART + D + 1] > 0) max_score = fmaxf(max_score, src[s * PART + D]);
    float value[16] = {}, den = 0;
    for (int s = 0; s < m.splits; ++s) {
        const float part_den = src[s * PART + D + 1];
        if (!(part_den > 0)) continue;
        const float w = expf(src[s * PART + D] - max_score);
        den += part_den * w;
#pragma unroll
        for (int j = 0; j < 16; ++j) value[j] += src[s * PART + lane * 16 + j] * w;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j)
        out[(long)r * m.orr + (long)head * m.oh + lane * 16 + j] =
            f32_to_bf16(den > 0 ? value[j] / den : 0);
}
/* mla_attn_paged: dense causal latent attention over a cache row of [latent 512 | rope 64], one
 * page table a sequence (Ling-3.0's MLA). A wave a (query row, head) as `attend`: lane l holds
 * latent columns 16l..16l+15 and rotary columns 2l, 2l+1, so a score is one wave sum. A row's
 * sequence, position and causal bound come from cu_seqlens and seqused as attn_paged reads them.
 * A short batch splits its history over waves and `merge_paged` joins the f32 partials; the
 * optional head gate multiplies the rounded output by the rounded sigmoid of the head's logit. */
constexpr int RD = 64;
struct PagedMeta {
    long qr, qh, rr, rh, kp, kt, bp, bc, orr, oh, gr;
    int rows, heads, seqs, blocks, bs, pages, splits;
    float scale;
};
__device__ __forceinline__ int paged_seq(const int* cu, int seqs, int r) {
    int s = 0;
    while (s + 1 < seqs && cu[s + 1] <= r) ++s;
    return s;
}
__device__ __forceinline__ void paged_store(uint16_t* out, const uint16_t* gate, const PagedMeta& m,
                                            int r, int head, int lane, const float (&value)[16],
                                            float den) {
    float g = 1.0f;
    if (gate) g = bf16_to_f32(f32_to_bf16(1.0f / (1.0f + expf(-bf16_to_f32(gate[(long)r * m.gr + head])))));
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        float v = den > 0 ? value[j] / den : 0;
        if (gate) v = bf16_to_f32(f32_to_bf16(v)) * g;
        out[(long)r * m.orr + (long)head * m.oh + lane * 16 + j] = f32_to_bf16(v);
    }
}
__global__ __launch_bounds__(128) void attend_paged(const uint16_t* q, const uint16_t* qrot,
                                                    const uint16_t* kv, const int* bt,
                                                    const int* used, const int* cu,
                                                    const uint16_t* gate, uint16_t* out,
                                                    float* scratch, PagedMeta m) {
    const int lane = threadIdx.x & 31, head = blockIdx.y * 4 + (threadIdx.x >> 5), r = blockIdx.x,
              split = blockIdx.z;
    if (head >= m.heads) return;
    const int s = paged_seq(cu, m.seqs, r);
    const int qlen = cu[s + 1] - cu[s], i = r - cu[s];
    const int live = min(max(used[s] - qlen + i + 1, 0), m.blocks * m.bs);
    const int begin = (int)((long)live * split / m.splits),
              end = (int)((long)live * (split + 1) / m.splits);
    float query[16], value[16] = {};
    load_latent(q + (long)r * m.qr + (long)head * m.qh + lane * 16, query);
    const uint16_t* const qp = qrot + (long)r * m.rr + (long)head * m.rh + lane * 2;
    const float rq0 = bf16_to_f32(qp[0]), rq1 = bf16_to_f32(qp[1]);
    float max_score = -INFINITY, den = 0;
    for (int token = begin; token < end; ++token) {
        const int page = bt[(long)s * m.bp + (token / m.bs) * m.bc];
        if (page < 0 || page >= m.pages) continue;
        const uint16_t* const row = kv + (long)page * m.kp + (token % m.bs) * m.kt;
        float key[16];
        load_latent(row + lane * 16, key);
        float score = rq0 * bf16_to_f32(row[D + lane * 2]) + rq1 * bf16_to_f32(row[D + lane * 2 + 1]);
#pragma unroll
        for (int j = 0; j < 16; ++j) score += query[j] * key[j];
        score = r4d_wave_sum(score) * m.scale;
        const bool higher = score > max_score;
        const float weight = expf(higher ? max_score - score : score - max_score);
        if (higher) {
#pragma unroll
            for (int j = 0; j < 16; ++j) value[j] = value[j] * weight + key[j];
            den = den * weight + 1;
            max_score = score;
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) value[j] += key[j] * weight;
            den += weight;
        }
    }
    if (m.splits == 1) {
        paged_store(out, gate, m, r, head, lane, value, den);
    } else {
        float* dst = scratch + (((long)r * m.heads + head) * m.splits + split) * PART;
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[lane * 16 + j] = value[j];
        if (!lane) {
            dst[D] = max_score;
            dst[D + 1] = den;
        }
    }
}
__global__ __launch_bounds__(128) void merge_paged(const float* scratch, const uint16_t* gate,
                                                   uint16_t* out, PagedMeta m) {
    const int lane = threadIdx.x & 31, head = blockIdx.y * 4 + (threadIdx.x >> 5), r = blockIdx.x;
    if (head >= m.heads) return;
    const float* src = scratch + ((long)r * m.heads + head) * m.splits * PART;
    float max_score = -INFINITY;
    for (int s = 0; s < m.splits; ++s)
        if (src[s * PART + D + 1] > 0) max_score = fmaxf(max_score, src[s * PART + D]);
    float value[16] = {}, den = 0;
    for (int s = 0; s < m.splits; ++s) {
        const float part_den = src[s * PART + D + 1];
        if (!(part_den > 0)) continue;
        const float w = expf(src[s * PART + D] - max_score);
        den += part_den * w;
#pragma unroll
        for (int j = 0; j < 16; ++j) value[j] += src[s * PART + lane * 16 + j] * w;
    }
    paged_store(out, gate, m, r, head, lane, value, den);
}
/* Splits of a (row, head) job's history. A row's bytes must not depend on the batch it is in: a
 * speculative verify scores the k + 1 rows of a sequence in one launch, and greedy output equals
 * the run without speculation only if each row gives the single-row decode at its position bit for
 * bit. So the count is a function of the head count alone, not of the launch's rows or its longest
 * table. A history shorter than the count leaves splits empty; they exit at once and merge_paged
 * skips them. */
/* The rows the scratch serves at full split: a verify at depth 31. */
constexpr long kPagedRows = 32;
int paged_splits(long heads) { return (int)std::min(16l, (256 + heads - 1) / heads); }
int64_t paged_scratch_bytes(long rows, long heads) {
    return std::min(rows, kPagedRows) * heads * paged_splits(heads) * PART * 4;
}
/* Operands: q [M, heads, 512], q_rope [M, heads, 64], kv [pages, 1, bs, >= 576], block_table
 * [seqs, blocks], seqused [seqs], cu_seqlens [seqs + 1], gate [M, >= heads] (optional), out
 * [M, heads, 512]. */
int paged_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 8) return RAD_E_INVAL;
    RadTensor q = a->t[0], qr = a->t[1], o = a->t[7];
    const auto &kv = a->t[2], &bt = a->t[3], &su = a->t[4], &cu = a->t[5];
    const RadTensor* gate = r4d_opt(a, 6);
    const long heads = r4d_geti_or(a, "heads", 0), bs = r4d_geti_or(a, "block_size", 0);
    const float scale = (float)r4d_getf(a, "scale", 0);
    auto heads_view = [&](RadTensor& t, long width) {
        if (heads > 0 && t.rank == 2 && t.shape[1] == heads * width) {
            t.rank = 3;
            t.shape[1] = heads;
            t.shape[2] = width;
            t.stride[2] = t.stride[1];
            t.stride[1] *= width;
        }
    };
    heads_view(q, D);
    heads_view(qr, RD);
    heads_view(o, D);
    if (q.dtype != RAD_BF16 || qr.dtype != RAD_BF16 || kv.dtype != RAD_BF16 ||
        o.dtype != RAD_BF16 || bt.dtype != RAD_I32 || su.dtype != RAD_I32 || cu.dtype != RAD_I32 ||
        (gate && gate->dtype != RAD_BF16))
        return RAD_E_DTYPE;
    if (!q.data || !qr.data || !kv.data || !bt.data || !su.data || !cu.data || !o.data)
        return RAD_E_INVAL;
    if (r4d_geti_or(a, "latent_dim", 0) != D || r4d_geti_or(a, "rope_dim", 0) != RD || heads < 1 ||
        heads > 64 || bs < 1 || bs > 256 || !std::isfinite(scale) || scale <= 0 || q.rank != 3 ||
        q.shape[0] < 1 || q.shape[0] > 65535 || q.shape[1] != heads || q.shape[2] != D ||
        q.stride[2] != 1 || q.stride[1] < D || q.stride[0] < heads * q.stride[1] || qr.rank != 3 ||
        qr.shape[0] != q.shape[0] || qr.shape[1] != heads || qr.shape[2] != RD ||
        qr.stride[2] != 1 || qr.stride[1] < RD || qr.stride[0] < heads * qr.stride[1] ||
        kv.rank != 4 || kv.shape[0] < 1 || kv.shape[0] > INT_MAX || kv.shape[1] != 1 ||
        kv.shape[2] != bs || kv.shape[3] < D + RD || kv.stride[3] != 1 || kv.stride[2] < D + RD ||
        kv.stride[0] < bs * kv.stride[2] || bt.rank != 2 || bt.shape[1] < 1 ||
        bt.shape[1] > INT_MAX / bs || bt.stride[1] < 1 || bt.stride[0] < bt.shape[1] * bt.stride[1] ||
        su.rank != 1 || su.shape[0] < 1 || bt.shape[0] < su.shape[0] || !r4d_contig(&su) ||
        !r4d_contig(&cu) || r4d_numel(&cu) != su.shape[0] + 1 || o.rank != 3 ||
        o.shape[0] != q.shape[0] || o.shape[1] != heads || o.shape[2] != D || o.stride[2] != 1 ||
        o.stride[1] < D || o.stride[0] < heads * o.stride[1] ||
        (gate && (gate->rank != 2 || gate->shape[0] != q.shape[0] || gate->shape[1] < heads ||
                  gate->stride[1] != 1)))
        return RAD_E_SHAPE;
    /* The split comes from the head count alone (paged_splits). Only a batch whose partials the
     * scratch cannot hold takes fewer, as many as fit. */
    const long rows = q.shape[0], slots = a->scratch ? a->scratch_bytes / (PART * 4) : 0;
    int splits = paged_splits(heads);
    if (rows * heads * splits > slots) splits = (int)std::max(1l, slots / (rows * heads));
    const PagedMeta m{ q.stride[0],  q.stride[1],     qr.stride[0],       qr.stride[1],
                       kv.stride[0], kv.stride[2],    bt.stride[0],       bt.stride[1],
                       o.stride[0],  o.stride[1],     gate ? (long)gate->stride[0] : 0L,
                       (int)q.shape[0], (int)heads,  (int)su.shape[0],   (int)bt.shape[1],
                       (int)bs,      (int)kv.shape[0], splits,           scale };
    const uint16_t* gp = gate ? (const uint16_t*)gate->data : nullptr;
    int rc = R4D_EMIT(stream, attend_paged, dim3(m.rows, (m.heads + 3) / 4, m.splits), dim3(128), 0,
                      (const uint16_t*)q.data, (const uint16_t*)qr.data, (const uint16_t*)kv.data,
                      (const int*)bt.data, (const int*)su.data, (const int*)cu.data, gp,
                      (uint16_t*)o.data, (float*)a->scratch, m);
    if (rc == RAD_OK && splits > 1)
        rc = R4D_EMIT(stream, merge_paged, dim3(m.rows, (m.heads + 3) / 4), dim3(128), 0,
                      (const float*)a->scratch, gp, (uint16_t*)o.data, m);
    return rc;
}
/* Heads are the matrix rows, so all sixteen share each latent load even when
 * each query selected different pages. Splitting P into BF16 high/low parts
 * preserves FP32 softmax precision without a materialized per-head KV cache.
 *
 * PAIR fills the sixteen rows with eight heads of two neighbouring requests
 * (rows 0-7 request r, rows 8-15 request r+1). A key is loaded once when both
 * requests address the same cache row at that table position, as consecutive
 * queries of one sequence do; each half masks columns that are not its own,
 * which carries the causal edge. Where the two tables name different rows the
 * chunk runs a second pass with the second request's keys.
 *
 * UNION pairs by page instead of by position: past the DSA budget each request
 * selects its own blocks, neighbours share most of them at shifted positions,
 * and the positional pairing would run nearly every chunk twice. The walk is
 * request r's table in its own order (so r sums exactly as it does unpaired at
 * one split), each entry carrying where r+1's table names the same page, then
 * the pages only r+1 names, in r+1's order. Every chunk runs once. */
constexpr int kUnionBlocks = 520, kUnionSlots = 1024;
__device__ __forceinline__ unsigned union_slot(int page) {
    return ((unsigned)page * 2654435761u) >> 22;
}
/* STATS is its own instance: counting inside the loop costs the paired tile spilled VGPRs. K is
 * the cache's element (uint16_t bf16, uint8_t E4M3). */
template <typename K, bool PAIR, bool WIDE, bool UNION, bool STATS>
__global__ __launch_bounds__(128) void attend_matrix(const uint16_t* q, const K* kv,
                                                     const int* bt, const int* len, uint16_t* out,
                                                     float* scratch, Meta m) {
    static_assert(PAIR || !UNION, "the union pairs two requests");
    static_assert(16 * 520 * sizeof(uint16_t) >= (2 * kUnionSlots + 2 * kUnionBlocks) * sizeof(int),
                  "the union's hash lives in the key tile");
    __shared__ __align__(16) uint16_t query[16 * 520], keys[16 * 520], prob_hi[16 * 24],
        prob_lo[16 * 24];
    __shared__ float score[16 * 16], maximum[16], denominator[16], alpha[16];
    __shared__ long source[16];
    __shared__ long own[PAIR ? 32 : 1];
    __shared__ short pair_of[UNION ? kUnionBlocks : 1], only_b[UNION ? kUnionBlocks : 1];
    __shared__ int union_count[UNION ? 4 : 1], union_end;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int r = PAIR ? blockIdx.x * 2 : blockIdx.x, h0 = blockIdx.y * (PAIR ? 8 : 16),
              split = blockIdx.z;
    int live = min(max(len[(long)r * m.lp], 0), m.blocks * m.bs), live_pair[2] = { live, 0 };
    if constexpr (PAIR) {
        if (r + 1 < m.rows) live_pair[1] = min(max(len[(long)(r + 1) * m.lp], 0), m.blocks * m.bs);
        live = max(live_pair[0], live_pair[1]);
    }
    for (int j = tid; j < 16 * 512; j += 128) {
        if constexpr (PAIR) {
            const int row = j / 512, request = r + row / 8;
            query[row * 520 + j % 512] =
                request < m.rows ? q[(long)request * m.qr + (long)(h0 + row % 8) * m.qh + j % 512]
                                 : 0;
        } else
            query[(j / 512) * 520 + j % 512] =
                q[(long)r * m.qr + (long)(h0 + j / 512) * m.qh + j % 512];
    }
    if (tid < 16) {
        maximum[tid] = -INFINITY;
        denominator[tid] = 0;
    }
    v8f value[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) value[nt] = (v8f){ 0 };
    const int* const bt_a = bt + (long)r * m.bp;
    const int* const bt_b = bt + (long)(r + 1) * m.bp;
    int blocks_a = 0;
    if constexpr (UNION) {
        /* The key tile is free until the first chunk: an open-addressed page -> lowest r entry
         * hash, the lowest r+1 entry naming each r entry's page, and each r+1 entry's match. */
        int* const slot_page = (int*)keys;
        int* const slot_first = slot_page + kUnionSlots;
        int* const first_b = slot_first + kUnionSlots;
        int* const match = first_b + kUnionBlocks;
        blocks_a = (live_pair[0] + m.bs - 1) / m.bs;
        const int blocks_b = (live_pair[1] + m.bs - 1) / m.bs;
        for (int i = tid; i < kUnionSlots; i += 128) {
            slot_page[i] = -1;
            slot_first[i] = INT_MAX;
        }
        for (int i = tid; i < blocks_a; i += 128) first_b[i] = INT_MAX;
        if (!tid) union_end = live_pair[0];
        __syncthreads();
        for (int i = tid; i < blocks_a; i += 128) {
            const int page = bt_a[(long)i * m.bc];
            if (page < 0 || page >= m.pages) continue;
            for (unsigned at = union_slot(page);; at = (at + 1) & (kUnionSlots - 1)) {
                const int seen = atomicCAS(&slot_page[at], -1, page);
                if (seen == -1 || seen == page) {
                    atomicMin(&slot_first[at], i);
                    break;
                }
            }
        }
        __syncthreads();
        for (int j = tid; j < blocks_b; j += 128) {
            const int page = bt_b[(long)j * m.bc];
            int found = -1;
            if (page >= 0 && page < m.pages)
                for (unsigned at = union_slot(page); slot_page[at] != -1;
                     at = (at + 1) & (kUnionSlots - 1))
                    if (slot_page[at] == page) {
                        found = slot_first[at];
                        break;
                    }
            match[j] = found;
            if (found >= 0) atomicMin(&first_b[found], j);
        }
        __syncthreads();
        for (int i = tid; i < blocks_a; i += 128)
            pair_of[i] = (short)(first_b[i] == INT_MAX ? -1 : first_b[i]);
        /* r+1's own pages in its table order: slots from a ballot prefix, not an atomic, so the
         * walk (and the summation order) is the same on every run. */
        int appended = 0;
        for (int j0 = 0; j0 < blocks_b; j0 += 128) {
            const int j = j0 + tid;
            bool only = false;
            if (j < blocks_b) {
                const int page = bt_b[(long)j * m.bc];
                only = page >= 0 && page < m.pages && (match[j] < 0 || first_b[match[j]] != j);
            }
            const unsigned long long ballot = __ballot(only);
            if (!lane) union_count[wave] = (int)__popcll(ballot);
            __syncthreads();
            int before = appended, total = appended;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                before += w < wave ? union_count[w] : 0;
                total += union_count[w];
            }
            if (only) {
                const int at = before + (int)__popcll(ballot & ((1ull << lane) - 1ull));
                only_b[at] = (short)j;
                const int seen = min(m.bs, live_pair[1] - j * m.bs);
                atomicMax(&union_end, (blocks_a + at) * m.bs + seen);
            } else if (j < blocks_b && match[j] >= 0 && first_b[match[j]] == j)
                atomicMax(&union_end, match[j] * m.bs + min(m.bs, live_pair[1] - j * m.bs));
            appended = total;
            __syncthreads();
        }
        live = union_end;
    }
    const int begin = (int)((long)live * split / m.splits),
              end = (int)((long)live * (split + 1) / m.splits);
    __syncthreads();
    for (int start = begin; start < end; start += 16) {
        bool second = false;
        if constexpr (UNION) {
            if (tid < 16) {
                const int token = start + tid, entry = token / m.bs, offset = token % m.bs;
                long row[2] = { -1, -1 };
                if (token < end) {
                    const int at_a = entry < blocks_a ? entry : -1;
                    const int at_b = entry < blocks_a ? pair_of[entry] : only_b[entry - blocks_a];
                    const int page = at_a >= 0 ? bt_a[(long)at_a * m.bc] : bt_b[(long)at_b * m.bc];
                    if (page >= 0 && page < m.pages) {
                        const long at = (long)page * m.kp + offset * m.kt;
                        if (at_a >= 0 && at_a * m.bs + offset < live_pair[0]) row[0] = at;
                        if (at_b >= 0 && at_b * m.bs + offset < live_pair[1]) row[1] = at;
                    }
                }
                own[tid] = row[0];
                own[16 + tid] = row[1];
                source[tid] = row[0] >= 0 ? row[0] : row[1];
            }
            __syncthreads();
        } else if constexpr (PAIR) {
            bool conflict = false;
            if (tid < 16) {
                const int token = start + tid;
                long row[2];
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int page = token < end && token < live_pair[k]
                                         ? bt[(long)(r + k) * m.bp + (token / m.bs) * m.bc]
                                         : -1;
                    row[k] = page >= 0 && page < m.pages ? (long)page * m.kp + (token % m.bs) * m.kt
                                                         : -1;
                    own[k * 16 + tid] = row[k];
                }
                source[tid] = row[0] >= 0 ? row[0] : row[1];
                conflict = row[0] >= 0 && row[1] >= 0 && row[0] != row[1];
            }
            second = __syncthreads_or(conflict);
            if (STATS && second && !tid) {
                atomicAdd(&g_mla_stats[0], 1ull);
                atomicAdd(&g_mla_stats[1], 1ull);
            }
        } else {
            if (tid < 16) {
                const int token = start + tid;
                const int page = token < end ? bt[(long)r * m.bp + (token / m.bs) * m.bc] : -1;
                source[tid] =
                    page >= 0 && page < m.pages ? (long)page * m.kp + (token % m.bs) * m.kt : -1;
            }
            __syncthreads();
        }
        for (int pass = 0; pass < (second ? 2 : 1); ++pass) {
            if constexpr (PAIR)
                if (pass) {
                    if (tid < 16) {
                        const long first = own[tid], other = own[16 + tid];
                        source[tid] = first >= 0 && other >= 0 && first != other ? other : -1;
                    }
                    __syncthreads();
                }
            if constexpr (WIDE) {
                /* Unconditional loads (an absent row reads row 0 and is zeroed after) so all
                 * eight are in flight before the first wait. */
                uint4 part[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int flat = tid + j * 128;
                    const long at = source[flat >> 6];
                    part[j] = load8(kv + (at >= 0 ? at : 0) + (flat & 63) * 8);
                    if (at < 0) part[j] = make_uint4(0, 0, 0, 0);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int flat = tid + j * 128;
                    *(uint4*)(keys + (flat >> 6) * 520 + (flat & 63) * 8) = part[j];
                }
            } else {
                for (int j = tid; j < 16 * 512; j += 128) {
                    const int token = j / 512, col = j % 512;
                    keys[token * 520 + col] =
                        source[token] >= 0 ? elem(kv + source[token] + col) : 0;
                }
            }
            __syncthreads();
            if (!wave) {
                v8f dot = { 0 };
#pragma unroll
                for (int d = 0; d < 512; d += 16)
                    dot = mma(frag(query, 520, 0, d, lane), frag(keys, 520, 0, d, lane), dot);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int head = (lane >> 4) * 8 + j, col = lane & 15;
                    bool visible = source[col] >= 0;
                    if constexpr (PAIR)
                        visible = visible && own[(lane >> 4) * 16 + col] == source[col];
                    score[head * 16 + col] = visible ? dot[j] * m.scale : -INFINITY;
                }
            }
            __syncthreads();
            if (tid < 16) {
                float mx = maximum[tid];
                for (int j = 0; j < 16; ++j) mx = fmaxf(mx, score[tid * 16 + j]);
                const float a = isfinite(mx) ? expf(maximum[tid] - mx) : 1;
                float den = denominator[tid] * a;
                for (int j = 0; j < 16; ++j) {
                    const float p =
                        isfinite(score[tid * 16 + j]) ? expf(score[tid * 16 + j] - mx) : 0;
                    const uint16_t hi = f32_to_bf16(p);
                    prob_hi[tid * 24 + j] = hi;
                    prob_lo[tid * 24 + j] = f32_to_bf16(p - bf16_to_f32(hi));
                    den += p;
                }
                maximum[tid] = mx;
                denominator[tid] = den;
                alpha[tid] = a;
            }
            // Read before the in-place transpose overwrites keys. Query remains resident.
            uint16_t held[64];
#pragma unroll
            for (int j = 0; j < 64; ++j) {
                const int flat = tid + j * 128;
                held[j] = keys[(flat / 512) * 520 + flat % 512];
            }
            __syncthreads();
#pragma unroll
            for (int j = 0; j < 64; ++j) {
                const int flat = tid + j * 128;
                keys[(flat % 512) * 16 + flat / 512] = held[j];
            }
            __syncthreads();
            const auto hi = frag(prob_hi, 24, 0, 0, lane), lo = frag(prob_lo, 24, 0, 0, lane);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                for (int j = 0; j < 8; ++j) value[nt][j] *= alpha[(lane >> 4) * 8 + j];
                const auto v = frag(keys, 16, wave * 128 + nt * 16, 0, lane);
                value[nt] = mma(hi, v, value[nt]);
                value[nt] = mma(lo, v, value[nt]);
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int col = wave * 128 + nt * 16 + (lane & 15);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = (lane >> 4) * 8 + j, request = PAIR ? r + (lane >> 4) : r,
                      head = PAIR ? h0 + j : h0 + row;
            if (PAIR && request >= m.rows) continue;
            if (m.splits == 1) {
                const float den = denominator[row];
                out[(long)request * m.orr + (long)head * m.oh + col] =
                    f32_to_bf16(den > 0 ? value[nt][j] / den : 0);
            } else
                scratch[(((long)request * m.heads + head) * m.splits + split) * PART + col] =
                    value[nt][j];
        }
    }
    if (m.splits > 1 && tid < 16) {
        const int request = PAIR ? r + tid / 8 : r, head = PAIR ? h0 + tid % 8 : h0 + tid;
        if (!PAIR || request < m.rows) {
            float* dst = scratch + (((long)request * m.heads + head) * m.splits + split) * PART;
            dst[D] = maximum[tid];
            dst[D + 1] = denominator[tid];
        }
    }
    if (STATS && !tid) {
        const int whole = max(live_pair[0], live_pair[1]);
        const int raster =
            (int)((long)whole * (split + 1) / m.splits) - (int)((long)whole * split / m.splits);
        const unsigned long long chunks = (unsigned long long)(end - begin + 15) / 16;
        atomicAdd(&g_mla_stats[0], chunks);
        atomicAdd(&g_mla_stats[2], (unsigned long long)(raster + 15) / 16);
    }
}
/* ROWS gives a workgroup one query row and HG groups of sixteen heads, four waves a group, so a
 * key chunk is loaded once for 16*HG heads instead of the head-tiled row's sixteen. GLM's prefill
 * splits attention by rows, one rank holding all 64 heads of its rows; that is what this serves.
 *
 * Scores are S^T = K Q^T: tokens are the matrix rows and heads the columns, so a head is a lane
 * and its running maximum and denominator stay in registers. Each wave of a group takes a quarter
 * of the latent dimension; the four partial dot products meet in LDS and every wave of the group
 * adds them in the same order, so all four hold identical scores and statistics. The
 * probabilities become the second product's B operand with one permlanex16 swap, and each wave
 * accumulates O^T for its quarter of the latent columns, the keys read transposed from the tile.
 * The next chunk's keys are fetched into registers while the current one is multiplied, so a
 * chunk costs two barriers. */
constexpr int kRowsPitch = 520;
template <typename K, int HG, bool WIDE>
__global__ __launch_bounds__(HG * 128) void attend_rows(const uint16_t* q, const K* kv,
                                                        const int* bt, const int* len,
                                                        uint16_t* out, float* scratch, Meta m) {
    constexpr int NT = HG * 128, LOADS = 16 * 64 / NT;
    static_assert(LOADS * NT == 16 * 64 && LOADS <= 32, "a chunk's 16-byte pieces over the group");
    __shared__ __align__(16) uint16_t keys[2][16 * kRowsPitch];
    __shared__ __align__(32) float partial[HG * 4][32 * 8];
    __shared__ int visible[2][16];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, quarter = wave & 3,
              group = wave >> 2, r = blockIdx.x, split = blockIdx.z;
    const int head = blockIdx.y * 16 * HG + group * 16 + (lane & 15);
    const int live = min(max(len[(long)r * m.lp], 0), m.blocks * m.bs);
    const int begin = (int)((long)live * split / m.splits),
              end = (int)((long)live * (split + 1) / m.splits);
    v8bf query[8];
    {
        const uint16_t* qp =
            q + (long)r * m.qr + (long)head * m.qh + quarter * 128 + 4 * (lane >> 4);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            union {
                uint16_t h[8];
                v8bf f;
            } u;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                u.h[e] = qp[s * 16 + e];
                u.h[4 + e] = qp[s * 16 + 8 + e];
            }
            query[s] = u.f;
        }
    }
    uint4 staged[LOADS];
    unsigned staged_ok = 0;
    auto fetch = [&](int start) {
#pragma unroll
        for (int j = 0; j < LOADS; ++j) {
            const int flat = tid + j * NT, token = start + (flat >> 6), piece = flat & 63;
            const int page = token < end ? bt[(long)r * m.bp + (token / m.bs) * m.bc] : -1;
            const bool ok = page >= 0 && page < m.pages;
            const long at = ok ? (long)page * m.kp + (token % m.bs) * m.kt + piece * 8 : 0;
            if constexpr (WIDE) {
                staged[j] = load8(kv + at);
                if (!ok) staged[j] = make_uint4(0, 0, 0, 0);
            } else {
                uint32_t w[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    w[i] = ok ? (uint32_t)elem(kv + at + 2 * i) |
                                    (uint32_t)elem(kv + at + 2 * i + 1) << 16
                              : 0u;
                staged[j] = make_uint4(w[0], w[1], w[2], w[3]);
            }
            staged_ok = ok ? staged_ok | 1u << j : staged_ok & ~(1u << j);
        }
    };
    auto stash = [&](int b) {
#pragma unroll
        for (int j = 0; j < LOADS; ++j) {
            const int flat = tid + j * NT, token = flat >> 6, piece = flat & 63;
            *(uint4*)(keys[b] + token * kRowsPitch + piece * 8) = staged[j];
            if (!piece) visible[b][token] = (int)(staged_ok >> j & 1u);
        }
    };
    v8f value[8];
#pragma unroll
    for (int mt = 0; mt < 8; ++mt) value[mt] = (v8f){ 0 };
    float run_max = -INFINITY, run_den = 0;
    fetch(begin);
    stash(0);
    __syncthreads();
    int buf = 0;
    for (int start = begin; start < end; start += 16) {
        const bool more = start + 16 < end;
        if (more) fetch(start + 16);
        const uint16_t* tile = keys[buf];
        v8f dot = { 0 };
#pragma unroll
        for (int s = 0; s < 8; ++s)
            dot = mma(frag(tile, kRowsPitch, 0, quarter * 128 + s * 16, lane), query[s], dot);
        *(v8f*)&partial[wave][lane * 8] = dot;
        __syncthreads();
        v8f score = *(const v8f*)&partial[group * 4][lane * 8];
#pragma unroll
        for (int k = 1; k < 4; ++k) score += *(const v8f*)&partial[group * 4 + k][lane * 8];
        float chunk_max = -INFINITY;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            score[e] = visible[buf][8 * (lane >> 4) + e] ? score[e] * m.scale : -INFINITY;
            chunk_max = fmaxf(chunk_max, score[e]);
        }
        chunk_max = fmaxf(chunk_max, swap16(chunk_max));
        const float next_max = fmaxf(run_max, chunk_max);
        const float alpha = isfinite(next_max) ? expf(run_max - next_max) : 1;
        float p[8], sum = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            p[e] = isfinite(score[e]) ? expf(score[e] - next_max) : 0;
            sum += p[e];
        }
        /* Both halves add tokens 0-7 before 8-15, so the sixteen lanes of a head agree. */
        const float other = swap16(sum);
        run_den = run_den * alpha + (lane < 16 ? sum + other : other + sum);
        run_max = next_max;
        /* A B fragment's lane holds its head's tokens 0-3 and 8-11 (lanes 0-15) or 4-7 and 12-15
         * (16-31); a score lane holds 0-7 or 8-15. Each half sends the other the four it needs. */
        uint32_t hi[4], lo[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint16_t h0 = f32_to_bf16(p[2 * i]), h1 = f32_to_bf16(p[2 * i + 1]);
            hi[i] = (uint32_t)h0 | (uint32_t)h1 << 16;
            lo[i] = (uint32_t)f32_to_bf16(p[2 * i] - bf16_to_f32(h0)) |
                    (uint32_t)f32_to_bf16(p[2 * i + 1] - bf16_to_f32(h1)) << 16;
        }
        const bool low = lane < 16;
        const uint32_t hx0 = r4d_xor_lane_u32<16>(low ? hi[2] : hi[0]),
                       hx1 = r4d_xor_lane_u32<16>(low ? hi[3] : hi[1]),
                       lx0 = r4d_xor_lane_u32<16>(low ? lo[2] : lo[0]),
                       lx1 = r4d_xor_lane_u32<16>(low ? lo[3] : lo[1]);
        union {
            uint32_t w[4];
            v8bf f;
        } ph, pl;
        ph.w[0] = low ? hi[0] : hx0;
        ph.w[1] = low ? hi[1] : hx1;
        ph.w[2] = low ? hx0 : hi[2];
        ph.w[3] = low ? hx1 : hi[3];
        pl.w[0] = low ? lo[0] : lx0;
        pl.w[1] = low ? lo[1] : lx1;
        pl.w[2] = low ? lx0 : lo[2];
        pl.w[3] = low ? lx1 : lo[3];
#pragma unroll
        for (int mt = 0; mt < 8; ++mt) {
            const auto v = fragT(tile, kRowsPitch, quarter * 128 + mt * 16, 0, lane);
            value[mt] *= alpha;
            value[mt] = mma(v, ph.f, value[mt]);
            value[mt] = mma(v, pl.f, value[mt]);
        }
        if (more) stash(buf ^ 1);
        __syncthreads();
        buf ^= 1;
    }
#pragma unroll
    for (int mt = 0; mt < 8; ++mt)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int col = quarter * 128 + mt * 16 + 8 * (lane >> 4) + e;
            if (m.splits == 1)
                out[(long)r * m.orr + (long)head * m.oh + col] =
                    f32_to_bf16(run_den > 0 ? value[mt][e] / run_den : 0);
            else
                scratch[(((long)r * m.heads + head) * m.splits + split) * PART + col] =
                    value[mt][e];
        }
    if (m.splits > 1 && !quarter && lane < 16) {
        float* dst = scratch + (((long)r * m.heads + head) * m.splits + split) * PART;
        dst[D] = run_max;
        dst[D + 1] = run_den;
    }
}
/* Read once: the launch path asks on every matrix launch. */
bool stats_on() {
    static const bool on = [] {
        const char* e = std::getenv("RADIANCE_MLA_CONFLICT_STATS");
        return e && *e && *e != '0';
    }();
    return on;
}
void read_stats(unsigned long long (&v)[3], bool reset) {
    v[0] = v[1] = v[2] = 0;
    if (hipMemcpyFromSymbol(v, HIP_SYMBOL(g_mla_stats), sizeof v, 0, hipMemcpyDeviceToHost) !=
        hipSuccess)
        return;
    const unsigned long long zero[3] = {};
    if (reset)
        (void)hipMemcpyToSymbol(HIP_SYMBOL(g_mla_stats), zero, sizeof zero, 0,
                                hipMemcpyHostToDevice);
}
/* A server has no point at which to read the counters: running totals every 1024th counted launch.
 * Not at exit, where a HIP call meets a runtime that is tearing down; not inside a graph capture,
 * which a blocking copy would break. */
void report_stats(RadStream stream) {
    static std::atomic<unsigned> launches{ 0 };
    if (++launches % 1024) return;
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing((hipStream_t)stream, &capture) != hipSuccess ||
        capture != hipStreamCaptureStatusNone)
        return;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return;
    unsigned long long v[3];
    read_stats(v, false);
    if (!v[2]) return;
    std::fprintf(stderr,
                 "r4d mla_attn_matrix device %d: %llu passes over %llu raster chunks, "
                 "%llu second passes (%.3fx)\n",
                 device, v[0], v[2], v[1], (double)v[0] / (double)v[2]);
}
/* K is the cache's element: uint16_t for a bf16 cache, uint8_t for an E4M3 one. */
template <bool MATRIX, typename K = uint16_t>
int body(const RadArgs* a, RadStream stream) {
    constexpr uint32_t KV = sizeof(K) == 1 ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16;
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const auto &q = a->t[0], &kv = a->t[1], &bt = a->t[2], &len = a->t[3], &o = a->t[4];
    if (q.dtype != RAD_BF16 || kv.dtype != KV || o.dtype != RAD_BF16 || bt.dtype != RAD_I32 ||
        len.dtype != RAD_I32)
        return RAD_E_DTYPE;
    for (int i = 0; i < 5; ++i)
        if (!a->t[i].data) return RAD_E_INVAL;
    const long heads = r4d_geti_or(a, "heads", 0), bs = r4d_geti_or(a, "block_size", 0);
    const float scale = (float)r4d_getf(a, "scale", 0);
    if (r4d_geti_or(a, "latent_dim", 0) != D || heads < 1 || heads > 64 || bs < 1 || bs > 256 ||
        !std::isfinite(scale) || scale <= 0 || q.rank != 3 || q.shape[0] < 1 ||
        q.shape[0] > 65535 || q.shape[1] != heads || q.shape[2] != D || q.stride[2] != 1 ||
        q.stride[1] < D || q.stride[0] < heads * q.stride[1] || kv.rank != 4 || kv.shape[0] < 1 ||
        kv.shape[0] > INT_MAX || kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] != D ||
        kv.stride[3] != 1 || kv.stride[2] < D || kv.stride[0] < bs * kv.stride[2] || bt.rank != 2 ||
        bt.shape[0] != q.shape[0] || bt.shape[1] < 1 || bt.shape[1] > INT_MAX / bs ||
        bt.stride[1] < 1 || bt.stride[0] < bt.shape[1] * bt.stride[1] || len.rank != 1 ||
        len.shape[0] != q.shape[0] || len.stride[0] < 1 || o.rank != 3 ||
        o.shape[0] != q.shape[0] || o.shape[1] != heads || o.shape[2] != D || o.stride[2] != 1 ||
        o.stride[1] < D || o.stride[0] < heads * o.stride[1])
        return RAD_E_SHAPE;
    if (MATRIX && heads % 8) return RAD_E_UNSUPPORTED;
    const bool pair = MATRIX && heads % 16;
    const long jobs = MATRIX ? (pair ? (q.shape[0] + 1) / 2 * (heads / 8) : q.shape[0] * heads / 16)
                             : q.shape[0] * heads;
    int splits = (int)std::min(MATRIX ? 64l : 16l, ((MATRIX ? 64 : 256) + jobs - 1) / jobs);
    const long minimum = MATRIX ? 16 : 64;
    splits = (int)std::max(1l, std::min((long)splits, (bt.shape[1] * bs + minimum - 1) / minimum));
    if (!a->scratch || a->scratch_bytes < (MATRIX ? matrix_scratch_bytes : scratch_bytes))
        splits = 1;
    // Vectors need every cache row on a sixteen-byte boundary.
    const bool wide = g_mla_matrix_form != 1 && !((uintptr_t)kv.data & 15) && !(kv.stride[0] % 8) &&
                      !(kv.stride[2] % 8);
    /* The union's LDS lists hold a table of at most kUnionBlocks entries; a wider table takes the
     * positional pairing. */
    const bool by_page = bt.shape[1] <= kUnionBlocks;
    const bool stats = MATRIX && stats_on();
    Meta m{ q.stride[0],  q.stride[1],      kv.stride[0],  kv.stride[2],
            bt.stride[0], bt.stride[1],     len.stride[0], o.stride[0],
            o.stride[1],  (int)q.shape[0],  (int)heads,    (int)bt.shape[1],
            (int)bs,      (int)kv.shape[0], splits,        scale };
    int rc;
    if constexpr (MATRIX) {
#define MATRIX_EMIT(P, W, U, S, X, Y)                                                              \
    R4D_EMIT(stream, (attend_matrix<K, P, W, U, S>), dim3(X, Y, m.splits), dim3(128), 0,           \
             (const uint16_t*)q.data, (const K*)kv.data, (const int*)bt.data,                      \
             (const int*)len.data, (uint16_t*)o.data, (float*)a->scratch, m)
#define MATRIX_LAUNCH(P, W, U, X, Y)                                                               \
    (stats ? MATRIX_EMIT(P, W, U, true, X, Y) : MATRIX_EMIT(P, W, U, false, X, Y))
        if (pair && by_page)
            rc = wide ? MATRIX_LAUNCH(true, true, true, (m.rows + 1) / 2, m.heads / 8)
                      : MATRIX_LAUNCH(true, false, true, (m.rows + 1) / 2, m.heads / 8);
        else if (pair)
            rc = wide ? MATRIX_LAUNCH(true, true, false, (m.rows + 1) / 2, m.heads / 8)
                      : MATRIX_LAUNCH(true, false, false, (m.rows + 1) / 2, m.heads / 8);
        else
            rc = wide ? MATRIX_LAUNCH(false, true, false, m.rows, m.heads / 16)
                      : MATRIX_LAUNCH(false, false, false, m.rows, m.heads / 16);
#undef MATRIX_LAUNCH
#undef MATRIX_EMIT
    } else
        rc = R4D_EMIT(stream, attend<K>, dim3(m.rows, (m.heads + 3) / 4, m.splits), dim3(128), 0,
                      (const uint16_t*)q.data, (const K*)kv.data, (const int*)bt.data,
                      (const int*)len.data, (uint16_t*)o.data, (float*)a->scratch, m);
    if (rc == RAD_OK && splits > 1)
        rc = R4D_EMIT(stream, merge, dim3(m.rows, (m.heads + 3) / 4), dim3(128), 0,
                      (const float*)a->scratch, (uint16_t*)o.data, m);
    if (rc == RAD_OK && stats) report_stats(stream);
    return rc;
}
/* r4d_mla_rows_pin_form: 0 is the launcher's rule (64 heads a workgroup, vector key loads where
 * the cache is aligned, the matrix row's kernel under kRowsMin rows), 1 the element-wise key load,
 * 2 thirty-two heads a workgroup, 3 the rule's rows kernel at any row count, 4 the matrix row's
 * kernel at every row count. The pin is a test hook (r4d_selftest, r4d_emu); nothing else sets
 * it. */
int g_mla_rows_form = 0;
/* Under it, a decode step's rows at one rank, a workgroup a row leaves the card mostly idle, and
 * the matrix row's four tiles a row and its splits are faster (r4d_selftest --case
 * mla_attn_rows_bf16, M 7: 0.5x). */
constexpr long kRowsMin = 32;
template <typename K = uint16_t>
int rows_body(const RadArgs* a, RadStream stream) {
    constexpr uint32_t KV = sizeof(K) == 1 ? (uint32_t)RAD_F8E4M3 : (uint32_t)RAD_BF16;
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const auto &q = a->t[0], &kv = a->t[1], &bt = a->t[2], &len = a->t[3], &o = a->t[4];
    if (q.dtype != RAD_BF16 || kv.dtype != KV || o.dtype != RAD_BF16 || bt.dtype != RAD_I32 ||
        len.dtype != RAD_I32)
        return RAD_E_DTYPE;
    for (int i = 0; i < 5; ++i)
        if (!a->t[i].data) return RAD_E_INVAL;
    const long heads = r4d_geti_or(a, "heads", 0), bs = r4d_geti_or(a, "block_size", 0);
    const float scale = (float)r4d_getf(a, "scale", 0);
    if (r4d_geti_or(a, "latent_dim", 0) != D || heads < 1 || heads > 64 || bs < 1 || bs > 256 ||
        !std::isfinite(scale) || scale <= 0 || q.rank != 3 || q.shape[0] < 1 ||
        q.shape[0] > 65535 || q.shape[1] != heads || q.shape[2] != D || q.stride[2] != 1 ||
        q.stride[1] < D || q.stride[0] < heads * q.stride[1] || kv.rank != 4 || kv.shape[0] < 1 ||
        kv.shape[0] > INT_MAX || kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] != D ||
        kv.stride[3] != 1 || kv.stride[2] < D || kv.stride[0] < bs * kv.stride[2] || bt.rank != 2 ||
        bt.shape[0] != q.shape[0] || bt.shape[1] < 1 || bt.shape[1] > INT_MAX / bs ||
        bt.stride[1] < 1 || bt.stride[0] < bt.shape[1] * bt.stride[1] || len.rank != 1 ||
        len.shape[0] != q.shape[0] || len.stride[0] < 1 || o.rank != 3 ||
        o.shape[0] != q.shape[0] || o.shape[1] != heads || o.shape[2] != D || o.stride[2] != 1 ||
        o.stride[1] < D || o.stride[0] < heads * o.stride[1])
        return RAD_E_SHAPE;
    if (heads % 32) return RAD_E_UNSUPPORTED;
    const int form = g_mla_rows_form;
    if (form == 4 || (!form && q.shape[0] < kRowsMin)) return body<true, K>(a, stream);
    const int group_heads = heads % 64 || form == 2 ? 32 : 64;
    const long rows = q.shape[0], groups = heads / group_heads, jobs = rows * groups;
    /* Short batches split the history until 64 workgroups run, two to each of the R9700's 32
     * WGPs, as far as 16-token chunks and the scratch's 2048 partials allow. */
    int splits = (int)std::min(64l, (64 + jobs - 1) / jobs);
    splits = (int)std::max(1l, std::min((long)splits, (bt.shape[1] * bs + 15) / 16));
    splits = (int)std::max(1l, std::min((long)splits, 2048 / (rows * heads)));
    if (!a->scratch || a->scratch_bytes < matrix_scratch_bytes) splits = 1;
    const bool wide = form != 1 && !((uintptr_t)kv.data & 15) && !(kv.stride[0] % 8) &&
                      !(kv.stride[2] % 8);
    Meta m{ q.stride[0],  q.stride[1],      kv.stride[0],  kv.stride[2],
            bt.stride[0], bt.stride[1],     len.stride[0], o.stride[0],
            o.stride[1],  (int)q.shape[0],  (int)heads,    (int)bt.shape[1],
            (int)bs,      (int)kv.shape[0], splits,        scale };
#define ROWS_EMIT(HG, W)                                                                           \
    R4D_EMIT(stream, (attend_rows<K, HG, W>), dim3(m.rows, (unsigned)groups, m.splits),          \
             dim3(HG * 128), 0, (const uint16_t*)q.data, (const K*)kv.data,                      \
             (const int*)bt.data, (const int*)len.data, (uint16_t*)o.data, (float*)a->scratch, m)
    int rc;
    if (group_heads == 64)
        rc = wide ? ROWS_EMIT(4, true) : ROWS_EMIT(4, false);
    else
        rc = wide ? ROWS_EMIT(2, true) : ROWS_EMIT(2, false);
#undef ROWS_EMIT
    if (rc == RAD_OK && splits > 1)
        rc = R4D_EMIT(stream, merge, dim3(m.rows, (m.heads + 3) / 4), dim3(128), 0,
                      (const float*)a->scratch, (uint16_t*)o.data, m);
    return rc;
}
}
extern "C" int r4d_rad_mla_attn(const RadArgs* a, RadStream stream) {
    return body<false>(a, stream);
}
extern "C" int r4d_rad_mla_attn_matrix(const RadArgs* a, RadStream stream) {
    return body<true>(a, stream);
}
extern "C" int r4d_rad_mla_attn_rows(const RadArgs* a, RadStream stream) {
    return rows_body(a, stream);
}
/* The same three rows over an E4M3 latent cache (kv_dtype fp8_e4m3). */
extern "C" int r4d_rad_mla_attn_fp8kv(const RadArgs* a, RadStream stream) {
    return body<false, uint8_t>(a, stream);
}
extern "C" int r4d_rad_mla_attn_matrix_fp8kv(const RadArgs* a, RadStream stream) {
    return body<true, uint8_t>(a, stream);
}
extern "C" int r4d_rad_mla_attn_rows_fp8kv(const RadArgs* a, RadStream stream) {
    return rows_body<uint8_t>(a, stream);
}
extern "C" int r4d_rad_mla_project(const RadArgs* a, RadStream stream) {
    return project_body(a, stream);
}
extern "C" int r4d_rad_mla_attn_paged(const RadArgs* a, RadStream stream) {
    return paged_body(a, stream);
}
extern "C" int64_t r4d_rad_mla_attn_scratch(const RadArgs*) { return scratch_bytes; }
/* Sized at the band's top row count, at most kPagedRows (paged_splits). */
extern "C" int64_t r4d_rad_mla_attn_paged_scratch(const RadArgs* a) {
    const long heads = r4d_geti_or(a, "heads", 0), rows = r4d_geti_or(a, "M", kPagedRows);
    if (heads < 1 || heads > 64 || rows < 1) return RAD_E_SHAPE;
    return paged_scratch_bytes(rows, heads);
}
extern "C" int64_t r4d_rad_mla_attn_matrix_scratch(const RadArgs*) { return matrix_scratch_bytes; }
extern "C" void r4d_mla_matrix_pin_form(int form) { g_mla_matrix_form = form; }
extern "C" void r4d_mla_rows_pin_form(int form) { g_mla_rows_form = form; }
/* Test hook: the current device's RADIANCE_MLA_CONFLICT_STATS counters since the last call (passes,
 * second passes, raster chunks), then zeroed. */
extern "C" void r4d_mla_matrix_stats(unsigned long long* out) {
    unsigned long long v[3];
    (void)hipDeviceSynchronize();
    read_stats(v, true);
    if (out) std::memcpy(out, v, sizeof v);
}
