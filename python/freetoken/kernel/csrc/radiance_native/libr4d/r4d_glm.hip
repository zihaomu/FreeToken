#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_moe_sort.h"
#include <climits>
#include <cmath>

namespace {
// One wave computes R rows by C columns, so each loaded weight serves R rows and each input
// C columns. Every output has the same arithmetic: lane l accumulates k = l, l+32, ... in
// ascending order with fmaf, then the xor butterfly over 16, 8, 4, 2, 1. A lane loads U steps of
// k before their FMAs: at a decode pass's few rows a wave streams its weight row with no other
// work to hide the load behind, and one step in flight left the router at 150 GB/s. Tail rows and
// columns read a clamped valid address and are never stored.
template <int R, int C, int U>
__global__ void __launch_bounds__(128)
    gemm_nt_f32_tile_kernel(const uint16_t* x, const float* w, float* y, long xp, long wp, long yp,
                            int m, int n, int k) {
    const int lane = threadIdx.x & 31, c0 = (blockIdx.x * 4 + (threadIdx.x >> 5)) * C,
              r0 = blockIdx.y * R;
    if (c0 >= n) return;
    const uint16_t* xr[R];
    const float* wc[C];
#pragma unroll
    for (int q = 0; q < R; ++q) xr[q] = x + (long)(r0 + q < m ? r0 + q : m - 1) * xp;
#pragma unroll
    for (int c = 0; c < C; ++c) wc[c] = w + (long)(c0 + c < n ? c0 + c : n - 1) * wp;
    float sum[R][C];
#pragma unroll
    for (int q = 0; q < R; ++q)
#pragma unroll
        for (int c = 0; c < C; ++c) sum[q][c] = 0;
    int j = lane;
    for (; j + 32 * (U - 1) < k; j += 32 * U) {
        float weight[U][C];
        uint16_t in[U][R];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int c = 0; c < C; ++c) weight[u][c] = wc[c][j + 32 * u];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < R; ++q) in[u][q] = xr[q][j + 32 * u];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < R; ++q) {
                const float v = bf16_to_f32(in[u][q]);
#pragma unroll
                for (int c = 0; c < C; ++c) sum[q][c] = fmaf(v, weight[u][c], sum[q][c]);
            }
    }
    for (; j < k; j += 32) {
#pragma unroll
        for (int q = 0; q < R; ++q) {
            const float v = bf16_to_f32(xr[q][j]);
#pragma unroll
            for (int c = 0; c < C; ++c) sum[q][c] = fmaf(v, wc[c][j], sum[q][c]);
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q)
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) sum[q][c] += __shfl_xor(sum[q][c], mask, 32);
        }
    if (lane) return;
#pragma unroll
    for (int q = 0; q < R; ++q)
        if (r0 + q < m)
#pragma unroll
            for (int c = 0; c < C; ++c)
                if (c0 + c < n) y[(long)(r0 + q) * yp + c0 + c] = sum[q][c];
}

template <int R, int C, int U = 1>
int gemm_f32_tile(RadStream stream, const RadTensor& x, const RadTensor& w, const RadTensor& y,
                  long n, long k) {
    const long m = x.shape[0];
    return R4D_EMIT(stream, (gemm_nt_f32_tile_kernel<R, C, U>),
                    dim3((unsigned)((n + 4 * C - 1) / (4 * C)), (unsigned)((m + R - 1) / R)),
                    dim3(128), 0, (const uint16_t*)x.data, (const float*)w.data, (float*)y.data,
                    x.stride[0], w.stride[0], y.stride[0], (int)m, (int)n, (int)k);
}
int gemm_f32_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const auto &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const long n = r4d_geti_or(a, "N", 0), k = r4d_geti_or(a, "K", 0);
    if (x.dtype != RAD_BF16 || w.dtype != RAD_F32 || y.dtype != RAD_F32) return RAD_E_DTYPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
    if (n < 1 || n > 512 || k < 1 || k > 16384 || x.rank != 2 || w.rank != 2 || y.rank != 2 ||
        x.shape[0] < 1 || x.shape[0] > 65535 || x.shape[1] != k || w.shape[0] != n ||
        w.shape[1] != k || y.shape[0] != x.shape[0] || y.shape[1] != n || x.stride[1] != 1 ||
        w.stride[1] != 1 || y.stride[1] != 1 || x.stride[0] < k || w.stride[0] < k ||
        y.stride[0] < n)
        return RAD_E_SHAPE;
    /* The tile form, rows x columns per wave and steps of k in flight, by M; all forms are
     * byte-identical, so the choice is speed only. Measured on gfx1201 (r4d_selftest --perff32,
     * cold) at N 4/32/36/288, K 4096: 16x4 gains 3-5 % at N 288/32 above 64 rows but loses 16 % at
     * N 36; 8x4 holds at every width. Below 64 rows the steps of k in flight decide: 2x2 with 16
     * steps is 2 to 2.5 times as fast as one step at N 288 up to 16 rows, and about 3 times at N 32
     * at every count below 64; at N 288 from 17 rows 8x1 with 8 steps is faster.
     * A projection of a few columns has too few waves for wide tiles at any M. */
    const long rows = x.shape[0];
    if (n <= 8) return gemm_f32_tile<4, 1>(stream, x, w, y, n, k);
    if (rows <= 16 || (rows < 64 && n < 128)) return gemm_f32_tile<2, 2, 16>(stream, x, w, y, n, k);
    if (rows < 64) return gemm_f32_tile<8, 1, 8>(stream, x, w, y, n, k);
    return gemm_f32_tile<8, 4>(stream, x, w, y, n, k);
}
// router_sigmoid_topk in one wave per row: p = sigmoid(logit), the top_k largest ranks p + bias
// in descending order, equal ranks to the lower expert, as a serial scan with a strict `>` takes
// them. Lane l holds experts l, l+32, ... as 64-bit keys: the rank's order-preserving bits (-0
// folded to +0) over ~e, so equal ranks order to the lower expert. Round j takes the largest key
// below round j-1's winner and parks it on lane j (top_k <= 32). `total` sums the selected p in
// round order, mult = norm ? factor / (total + 1e-20) : factor, and a weight is p * mult. A NaN
// rank is never selected.
// OWN is a template bound because the keys are registers; a lane-derived trip count would index
// them divergently, which gfx12 lowers to a v_movrels loop.
__device__ __forceinline__ unsigned long long router_rank_key(float rank, int e) {
    uint32_t u = __float_as_uint(rank);
    if (u == 0x80000000u) u = 0;
    const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return rank == rank ? ((unsigned long long)k << 32) | (uint32_t)~e : 0ull;
}
__device__ __forceinline__ unsigned long long router_wave_max(unsigned long long b) {
    unsigned long long o;
    o = r4d_xor_lane_u64<16>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<8>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<4>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<2>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<1>(b);
    b = b > o ? b : o;
    return b;
}
// One row in one wave, from any grid: the standalone select and the fused one below run this. A
// lane below `k` returns its slot's expert as written to `ids`.
template <int OWN>
__device__ __forceinline__ int router_sigmoid_row(const float* logits, const float* bias, int* ids,
                                                  float* weights, long lp, long bp, long ip,
                                                  long wp, int r, int ne, int k, int norm,
                                                  float factor, int lane) {
    float p[OWN];
    unsigned long long key[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i, s = e < ne ? e : 0;
        p[i] = 1.0f / (1.0f + expf(-logits[(long)r * lp + s]));
        key[i] = e < ne ? router_rank_key(p[i] + bias[(long)s * bp], e) : 0ull;
    }
    unsigned long long below = ~0ull, mine = 0;
    for (int j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = router_wave_max(best);
        if (lane == j) mine = best;
        below = best;
    }
    const int e = mine ? (int)~(uint32_t)mine : 0;
    float chosen = 0;
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const float v = __shfl(p[i], e & 31, 32);
        if ((e >> 5) == i) chosen = v;
    }
    float total = 0;
    for (int j = 0; j < k; ++j)
        total += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen), j));
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[(long)r * ip + lane] = e;
        weights[(long)r * wp + lane] = chosen * mult;
    }
    return e;
}
// router_sigmoid_group_topk in one wave per row: router_sigmoid_row's keys, but only the experts of
// the `kg` best of `ng` equal groups are selectable. A group ranks by the sum of its two highest
// ranks; lane g holds group g's sum (ng <= 32), and the groups are taken as the experts are, the
// largest first, equal sums to the lower group, a group with no finite pair never. Then the
// selection, the weights and the stores are router_sigmoid_row's.
template <int OWN>
__global__ void __launch_bounds__(128)
    router_group_wave_kernel(const float* logits, const float* bias, int* ids, float* weights,
                             long lp, long bp, long ip, long wp, int m, int ne, int k, int ng,
                             int kg, int norm, float factor) {
    const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= m) return;
    const int gs = ne / ng;
    float p[OWN], rank[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i, s = e < ne ? e : 0;
        p[i] = 1.0f / (1.0f + expf(-logits[(long)r * lp + s]));
        rank[i] = e < ne ? p[i] + bias[(long)s * bp] : -INFINITY;
    }
    float group = -INFINITY;
    for (int g = 0; g < ng; ++g) {
        float t0 = -INFINITY, t1 = -INFINITY;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const int e = lane + 32 * i;
            if (e < ne && e / gs == g) {
                const float v = rank[i];
                if (v > t0) {
                    t1 = t0;
                    t0 = v;
                } else if (v > t1)
                    t1 = v;
            }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float o0 = __shfl_xor(t0, mask, 32), o1 = __shfl_xor(t1, mask, 32);
            const float hi = fmaxf(t0, o0), lo = fmaxf(fminf(t0, o0), fmaxf(t1, o1));
            t0 = hi;
            t1 = lo;
        }
        if (lane == g) group = t0 + t1;
    }
    unsigned long long gkey = lane < ng && group > -INFINITY ? router_rank_key(group, lane) : 0ull;
    unsigned kept = 0;   // bit g: group g is kept
    unsigned long long below = ~0ull;
    for (int j = 0; j < kg; ++j) {
        const unsigned long long best = router_wave_max(gkey < below ? gkey : 0ull);
        if (best) kept |= 1u << (int)~(uint32_t)best;
        below = best;
    }
    unsigned long long key[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i;
        key[i] = e < ne && ((kept >> (e / gs)) & 1u) ? router_rank_key(rank[i], e) : 0ull;
    }
    unsigned long long mine = 0;
    below = ~0ull;
    for (int j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = router_wave_max(best);
        if (lane == j) mine = best;
        below = best;
    }
    const int e = mine ? (int)~(uint32_t)mine : 0;
    float chosen = 0;
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const float v = __shfl(p[i], e & 31, 32);
        if ((e >> 5) == i) chosen = v;
    }
    float total = 0;
    for (int j = 0; j < k; ++j)
        total += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen), j));
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[(long)r * ip + lane] = e;
        weights[(long)r * wp + lane] = chosen * mult;
    }
}
template <int OWN>
__global__ void __launch_bounds__(128)
    router_sigmoid_wave_kernel(const float* logits, const float* bias, int* ids, float* weights,
                               long lp, long bp, long ip, long wp, int m, int ne, int k, int norm,
                               float factor) {
    const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= m) return;
    (void)router_sigmoid_row<OWN>(logits, bias, ids, weights, lp, bp, ip, wp, r, ne, k, norm,
                                  factor, lane);
}
// router_sigmoid_topk, then moe_scatter over its ids, then the `fill` of an ungated shared arm's
// gate with 1, in one workgroup. Rows wrap over the NW waves; each leaves its slots in the sort's
// LDS list instead of re-reading `ids`. A slot carries the expert its row wrote, including the
// expert 0 of a row with fewer finite ranks than top_k, so every output equals the three launches'.
// INV also writes the inverse, as moe_scatter_inv does.
template <int OWN, uint32_t NW, bool INV>
__global__ void __launch_bounds__(NW * 32)
    router_sigmoid_scatter_kernel(const float* logits, const float* bias, int* ids, float* weights,
                                  int32_t* sorted, int32_t* off, int32_t* cnt, uint16_t* gate,
                                  int32_t* inv, long lp, long bp, long ip, long wp, long gp, int m,
                                  int ne, int n_local, int k, int norm, float factor, int base,
                                  r4d_moesort::Protect prot) {
    __shared__ r4d_moesort::ScatterLds<NW> sl;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int e = threadIdx.x; e < n_local; e += (int)(NW * 32)) sl.cnt[e] = 0;
    for (int r = wave; r < m; r += (int)NW) {
        const int e = router_sigmoid_row<OWN>(logits, bias, ids, weights, lp, bp, ip, wp, r, ne, k,
                                              norm, factor, lane) -
                      base;
        if (lane < k)
            sl.e[r * k + lane] =
                e >= 0 && e < n_local ? r4d_moesort::sort_place(e, (uint32_t)n_local, prot) : -1;
        if (gate && !lane) gate[(long)r * gp] = f32_to_bf16(1.0f);
    }
    lds_barrier();
    r4d_moesort::scatter_small_place<NW, INV>(sorted, off, cnt, (uint32_t)(m * k),
                                              (uint32_t)n_local, sl, inv);
}
int router_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    const auto &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const long ne = r4d_geti_or(a, "n_expert", 0), k = r4d_geti_or(a, "top_k", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || w.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || lg.rank != 2 || lg.shape[0] < 1 ||
        lg.shape[0] > 65535 || lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne ||
        ids.rank != 2 || w.rank != 2 || ids.shape[0] != lg.shape[0] || w.shape[0] != lg.shape[0] ||
        ids.shape[1] != k || w.shape[1] != k || lg.stride[1] != 1 || ids.stride[1] != 1 ||
        w.stride[1] != 1 || lg.stride[0] < ne || ids.stride[0] < k || w.stride[0] < k ||
        bias.stride[0] < 1)
        return RAD_E_SHAPE;
    const int norm = (int)(r4d_geti_or(a, "norm", 0) != 0);
    const dim3 grid((unsigned)((lg.shape[0] + 3) / 4));
    if (ne <= 9 * 32)
        return R4D_EMIT(stream, router_sigmoid_wave_kernel<9>, grid, dim3(128), 0,
                        (const float*)lg.data, (const float*)bias.data, (int*)ids.data,
                        (float*)w.data, lg.stride[0], bias.stride[0], ids.stride[0], w.stride[0],
                        (int)lg.shape[0], (int)ne, (int)k, norm, factor);
    return R4D_EMIT(stream, router_sigmoid_wave_kernel<16>, grid, dim3(128), 0,
                    (const float*)lg.data, (const float*)bias.data, (int*)ids.data, (float*)w.data,
                    lg.stride[0], bias.stride[0], ids.stride[0], w.stride[0], (int)lg.shape[0],
                    (int)ne, (int)k, norm, factor);
}
int router_group_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    const auto &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const long ne = r4d_geti_or(a, "n_expert", 0), k = r4d_geti_or(a, "top_k", 0),
               ng = r4d_geti_or(a, "n_group", 0), kg = r4d_geti_or(a, "topk_group", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || w.dtype != RAD_F32)
        return RAD_E_DTYPE;
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || ng < 1 || ng > 32 || ne % ng || ne / ng < 2 ||
        kg < 1 || kg > ng || k > kg * (ne / ng) || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || lg.rank != 2 || lg.shape[0] < 1 ||
        lg.shape[0] > 65535 || lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne ||
        ids.rank != 2 || w.rank != 2 || ids.shape[0] != lg.shape[0] || w.shape[0] != lg.shape[0] ||
        ids.shape[1] != k || w.shape[1] != k || lg.stride[1] != 1 || ids.stride[1] != 1 ||
        w.stride[1] != 1 || lg.stride[0] < ne || ids.stride[0] < k || w.stride[0] < k ||
        bias.stride[0] < 1)
        return RAD_E_SHAPE;
    const int norm = (int)(r4d_geti_or(a, "norm", 0) != 0);
    const dim3 grid((unsigned)((lg.shape[0] + 3) / 4));
    if (ne <= 9 * 32)
        return R4D_EMIT(stream, router_group_wave_kernel<9>, grid, dim3(128), 0,
                        (const float*)lg.data, (const float*)bias.data, (int*)ids.data,
                        (float*)w.data, lg.stride[0], bias.stride[0], ids.stride[0], w.stride[0],
                        (int)lg.shape[0], (int)ne, (int)k, (int)ng, (int)kg, norm, factor);
    return R4D_EMIT(stream, router_group_wave_kernel<16>, grid, dim3(128), 0,
                    (const float*)lg.data, (const float*)bias.data, (int*)ids.data, (float*)w.data,
                    lg.stride[0], bias.stride[0], ids.stride[0], w.stride[0], (int)lg.shape[0],
                    (int)ne, (int)k, (int)ng, (int)kg, norm, factor);
}
/* Operands: router_sigmoid_topk's four, moe_scatter's three outputs, then the optional gate and the
 * optional inverse. The sort is the single-workgroup form, so M * top_k must fit it; a larger pass
 * is refused rather than served at another cost. */
int router_scatter_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    const auto &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const RadTensor *sorted = &a->t[4], *off = &a->t[5], *cnt = &a->t[6];
    const RadTensor* gate = a->n_t > 7 ? r4d_opt(a, 7) : nullptr;
    const RadTensor* inv = a->n_t > 8 ? r4d_opt(a, 8) : nullptr;
    const long ne = r4d_geti_or(a, "n_expert", 0), k = r4d_geti_or(a, "top_k", 0);
    const long nl = r4d_geti_or(a, "n_local", ne), base = r4d_geti_or(a, "expert_base", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 ||
        w.dtype != RAD_F32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32 || (gate && gate->dtype != RAD_BF16) ||
        (inv && inv->dtype != RAD_I32))
        return RAD_E_DTYPE;
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || !sorted->data || !off->data ||
        !cnt->data || lg.rank != 2 || lg.shape[0] < 1 || lg.shape[1] != ne || bias.rank != 1 ||
        bias.shape[0] != ne || ids.rank != 2 || w.rank != 2 || ids.shape[0] != lg.shape[0] ||
        w.shape[0] != lg.shape[0] || ids.shape[1] != k || w.shape[1] != k || lg.stride[1] != 1 ||
        ids.stride[1] != 1 || w.stride[1] != 1 || lg.stride[0] < ne || ids.stride[0] < k ||
        w.stride[0] < k || bias.stride[0] < 1)
        return RAD_E_SHAPE;
    const long m = lg.shape[0], T = m * k;
    if (T > (long)r4d_moesort::kScatterSmall || nl < 1 || nl > ne || base < 0 || base + nl > ne ||
        !r4d_contig(sorted) || !r4d_contig(off) || !r4d_contig(cnt) || r4d_numel(sorted) < T ||
        r4d_numel(off) < nl + 1 || r4d_numel(cnt) < nl ||
        (inv && (!r4d_contig(inv) || r4d_numel(inv) < T)))
        return RAD_E_SHAPE;
    long long grows = 0;
    const long long gp = gate ? r4d_rows_pitch(gate, 1, &grows) : 0;
    if (gate && (gp < 1 || grows < m)) return RAD_E_SHAPE;
    r4d_moesort::Protect prot;
    const int prc = r4d_moesort::protect_of(a, nl, &prot);
    if (prc != RAD_OK) return prc;
    const int norm = (int)(r4d_geti_or(a, "norm", 0) != 0);
    // A wave a row up to 32 rows, as router_topk_scatter: the rows a wave serves run back to back.
    const int nw = m <= 8 ? 8 : m <= 16 ? 16 : 32;
#define R4D_RSS_I(OWN, NW, INV)                                                                    \
    return R4D_EMIT(stream, (router_sigmoid_scatter_kernel<OWN, NW, INV>), dim3(1),                \
                    dim3((NW) * 32), 0, (const float*)lg.data, (const float*)bias.data,            \
                    (int*)ids.data, (float*)w.data, (int32_t*)sorted->data, (int32_t*)off->data,   \
                    (int32_t*)cnt->data, gate ? (uint16_t*)gate->data : (uint16_t*)nullptr,        \
                    inv ? (int32_t*)inv->data : (int32_t*)nullptr, lg.stride[0], bias.stride[0],   \
                    ids.stride[0], w.stride[0], (long)gp, (int)m, (int)ne, (int)nl, (int)k, norm,  \
                    factor, (int)base, prot)
#define R4D_RSS(OWN, NW)                                                                           \
    do {                                                                                           \
        if (inv) R4D_RSS_I(OWN, NW, true);                                                         \
        R4D_RSS_I(OWN, NW, false);                                                                 \
    } while (0)
    if (ne <= 9 * 32) {
        if (nw == 8) R4D_RSS(9, 8u);
        if (nw == 16) R4D_RSS(9, 16u);
        R4D_RSS(9, 32u);
    }
    if (nw == 8) R4D_RSS(16, 8u);
    if (nw == 16) R4D_RSS(16, 16u);
    R4D_RSS(16, 32u);
#undef R4D_RSS
#undef R4D_RSS_I
}
__global__ void swiglu_clamped_kernel(const uint16_t* gu, uint16_t* y, long gp, long yp, int n,
                                      float limit) {
    const long r = blockIdx.x;
    const int i = blockIdx.y * 256 + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(bf16_to_f32(gu[r * gp + i]), limit);
    const float u = fminf(fmaxf(bf16_to_f32(gu[r * gp + n + i]), -limit), limit);
    y[r * yp + i] = f32_to_bf16((g / (1.0f + expf(-g))) * u);
}
// swiglu_step: the SiLU first, its value clipped above (Ling-3.0; vLLM's SwigluStepAndMul).
__global__ void swiglu_step_kernel(const uint16_t* gu, uint16_t* y, long gp, long yp, int n,
                                   float limit) {
    const long r = blockIdx.x;
    const int i = blockIdx.y * 256 + threadIdx.x;
    if (i >= n) return;
    const float x = bf16_to_f32(gu[r * gp + i]);
    const float g = fminf(x / (1.0f + expf(-x)), limit);
    const float u = fminf(fmaxf(bf16_to_f32(gu[r * gp + n + i]), -limit), limit);
    y[r * yp + i] = f32_to_bf16(g * u);
}
int clamped_body(const RadArgs* a, RadStream stream, bool step = false) {
    if (!a || a->n_t < 2) return RAD_E_INVAL;
    const auto &gu = a->t[0], &y = a->t[1];
    const long n = r4d_geti_or(a, "n", 0);
    const float limit = (float)r4d_getf(a, "limit", 0);
    if (gu.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_DTYPE;
    if (n < 1 || n > INT_MAX / 2 || !std::isfinite(limit) || limit <= 0 || !gu.data || !y.data ||
        gu.rank != 2 || y.rank != 2 || y.shape[0] < 1 || y.shape[0] > 65535 ||
        gu.shape[0] != y.shape[0] || gu.shape[1] != 2 * n || y.shape[1] != n || gu.stride[1] != 1 ||
        y.stride[1] != 1 || gu.stride[0] < 2 * n || y.stride[0] < n)
        return RAD_E_SHAPE;
    if (step)
        return R4D_EMIT(stream, swiglu_step_kernel, dim3(y.shape[0], (n + 255) / 256), dim3(256), 0,
                        (const uint16_t*)gu.data, (uint16_t*)y.data, gu.stride[0], y.stride[0],
                        (int)n, limit);
    return R4D_EMIT(stream, swiglu_clamped_kernel, dim3(y.shape[0], (n + 255) / 256), dim3(256), 0,
                    (const uint16_t*)gu.data, (uint16_t*)y.data, gu.stride[0], y.stride[0], (int)n,
                    limit);
}
// swiglu_clamped, then quant_act_fp8 over its product, in one pass: a wave per (row, 128 columns),
// four columns a lane, as the quantiser maps them. Each element is swiglu_clamped_kernel's
// expression and each group quant_act_fp8_kernel's. The product is rounded to bf16 and read back
// before the absmax because the separate quantiser reads the stored bf16 bytes.
__global__ void __launch_bounds__(256)
    swiglu_clamped_quant_kernel(const uint16_t* gu, uint16_t* y, uint8_t* q, float* s, long gp,
                                long yp, long qp, long sp, int n, int nkb, float limit) {
    const long r = blockIdx.x;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int kb = wave; kb < nkb; kb += (int)(blockDim.x >> 5)) {
        const int c0 = kb * 128 + lane * 4;
        float v[4], m = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + j;
            v[j] = 0.0f;
            if (c < n) {
                const float g = fminf(bf16_to_f32(gu[r * gp + c]), limit);
                const float u = fminf(fmaxf(bf16_to_f32(gu[r * gp + n + c]), -limit), limit);
                const uint16_t h = f32_to_bf16((g / (1.0f + expf(-g))) * u);
                if (y) y[r * yp + c] = h;
                v[j] = bf16_to_f32(h);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
        m = r4d_wave_max(m);
        const float sc = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f;
        if (!lane) s[r * sp + kb] = sc;
        const float inv = 1.0f / sc;
        int pk = 0;
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[0] * inv, -448.0f), 448.0f),
                                             fminf(fmaxf(v[1] * inv, -448.0f), 448.0f), pk, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[2] * inv, -448.0f), 448.0f),
                                             fminf(fmaxf(v[3] * inv, -448.0f), 448.0f), pk, true);
        if (c0 < n) *(uint32_t*)(q + r * qp + c0) = (uint32_t)pk;
    }
}
/* Operands: gate_up [M, 2n], q [M, n] E4M3, scale [M, ceil(n/128)] f32, out_bf16 [M, n] optional.
 * n a multiple of 4 and group 128, as quant_act_fp8 requires. */
int clamped_quant_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const auto &gu = a->t[0], &q = a->t[1], &sc = a->t[2];
    const RadTensor* y = a->n_t > 3 ? r4d_opt(a, 3) : nullptr;
    const long n = r4d_geti_or(a, "n", 0), nkb = (n + 127) / 128;
    const float limit = (float)r4d_getf(a, "limit", 0);
    if (gu.dtype != RAD_BF16 || q.dtype != RAD_F8E4M3 || sc.dtype != RAD_F32 ||
        (y && y->dtype != RAD_BF16))
        return RAD_E_DTYPE;
    if (r4d_geti_or(a, "group", 128) != 128 || n < 4 || n % 4 || n > INT_MAX / 2 ||
        !std::isfinite(limit) || limit <= 0 || !gu.data || !q.data || !sc.data || gu.rank != 2 ||
        q.rank != 2 || sc.rank != 2 || gu.shape[0] < 1 || gu.shape[0] > 65535 ||
        gu.shape[1] != 2 * n || q.shape[0] != gu.shape[0] || q.shape[1] != n ||
        sc.shape[0] != gu.shape[0] || sc.shape[1] != nkb || gu.stride[1] != 1 || q.stride[1] != 1 ||
        sc.stride[1] != 1 || gu.stride[0] < 2 * n || q.stride[0] < n || sc.stride[0] < nkb ||
        (y && (y->rank != 2 || y->shape[0] != gu.shape[0] || y->shape[1] != n ||
               y->stride[1] != 1 || y->stride[0] < n)))
        return RAD_E_SHAPE;
    // A wave a group, at most eight: a row of GLM's TP8 expert width is two groups.
    const unsigned thr = (unsigned)(nkb < 8 ? nkb : 8) * 32u;
    return R4D_EMIT(stream, swiglu_clamped_quant_kernel, dim3((unsigned)gu.shape[0]), dim3(thr), 0,
                    (const uint16_t*)gu.data, y ? (uint16_t*)y->data : (uint16_t*)nullptr,
                    (uint8_t*)q.data, (float*)sc.data, gu.stride[0], y ? (long)y->stride[0] : 0L,
                    q.stride[0], sc.stride[0], (int)n, (int)nkb, limit);
}
}
extern "C" int r4d_rad_swiglu_clamped(const RadArgs* a, RadStream stream) {
    return clamped_body(a, stream);
}
extern "C" int r4d_rad_swiglu_step(const RadArgs* a, RadStream stream) {
    return clamped_body(a, stream, true);
}
extern "C" int r4d_rad_router_sigmoid_group_topk(const RadArgs* a, RadStream stream) {
    return router_group_body(a, stream);
}
extern "C" int r4d_rad_gemm_nt_f32(const RadArgs* a, RadStream stream) {
    return gemm_f32_body(a, stream);
}
extern "C" int r4d_rad_router_sigmoid_topk(const RadArgs* a, RadStream stream) {
    return router_body(a, stream);
}
extern "C" int r4d_rad_router_sigmoid_topk_scatter(const RadArgs* a, RadStream stream) {
    return router_scatter_body(a, stream);
}
