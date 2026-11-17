// R4D video sparse attention (docs/OPS.md, "Video generation"): `vsa_select`, the per-head tile
// selection of a distilled video transformer, and `attn_vsa`, the block-sparse attention that reads
// it, both at head_dim 128 over 4 x 4 x 4 tiles.
//
// The packed sequence is dense segments (text, a soundtrack's channels) and one video grid of
// F x Hp x Wp rows. Each dense segment is cut on its own into chunks of 64 rows, the prefix tiles;
// the grid into 4 x 4 x 4 boxes, the video tiles. A tile's rows are found from its number in
// closed form (vsa_tile, vsa_row), so neither op permutes rows. The geometry is a device operand,
// read by every workgroup: a video grid that does not fit the operand's rows has no tiles, and the
// tile count is cut to the selection's capacity, so no value of `segs` or `geom` reaches outside
// an operand.
//
// vsa_select, two steps a group of heads (the scratch holds the means of as many heads as fit):
//   pool    the f32 means of q, k (and v) over each tile, summed in the tile's (t, h, w) raster;
//   select  a workgroup a few query tiles: their q means, the f32 scores against every tile's k
//           mean (each a sequential 128-term dot product, so a score is the reference's to the
//           bit), the top k video tiles by a bisection on the scores' 32 bits with ties to the
//           lower tile, written ascending, and with `coarse` the softmax over every tile times
//           the v means, rounded once.
//
// attn_vsa: r4d_attn_gen.hip's S^T = K.Q^T flash attention (a lane owns a query row; P stays in
// registers, truncated to bf16 with the row sum taken from the truncated values), a workgroup two
// query tiles of one head walking the merge of their visible key tiles: every tile for a prefix
// query tile, the prefix tiles and its selected tiles for a video one. With `coarse` the output
// is bf16(bf16(o) + coarse * gate).
//
// attn_vsa's int8 Q.K form (`qk` "i8", operands `k_i8` and `k_scale`), attn_prefix_self's: the
// keys as quant_act_i8g codes and scales at group head_dim (per (row, head) scale amax / 127, 1
// for an all-zero row, code rint(x / scale) clamped to +-127), the query quantised the same way
// here, a score the exact integer dot product times both scales on v_wmma_i32_16x16x16_iu8 (half
// the cycles of bf16), P.V in bf16 as above. The K stream halves (8 KiB a tile instead of 16).
//
// The fp8 P.V form (`pv` "fp8"): `v` as E4M3 codes with a scale per column (quant_cols_fp8's),
// P.V on v_wmma_f32_16x16x16_fp8_fp8 with P as E4M3 codes of 448 p. The softmax keeps an exact
// running max (every p at most 1, E4M3 ending at 448), the row sum takes the unrounded 448 p,
// and the epilogue multiplies by the column scale. The V stream halves too.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_dt16.h"

#include <cmath>
#include <cstring>
#include <type_traits>

namespace {

constexpr int kHd   = 128;   // head_dim
constexpr int kTile = 64;    // rows a tile at most: 4 x 4 x 4, and a prefix chunk

__device__ __forceinline__ int vsa_clamp(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// The tiling of one pass, the same in every workgroup.
struct VsaGeom {
    int nd;        // dense segments
    int P, V, N;   // prefix, video and all tiles
    int F, Hp, Wp; // the video grid
    int nb, nc;    // boxes along h and w
    unsigned mb, mc; // ceil(2^32 / nb), ceil(2^32 / nc): x / n = mulhi(x, m) for x < 2^16
    int vbase;     // the grid's first row
    int k;         // video tiles a video query tile selects
};

__device__ __forceinline__ VsaGeom vsa_geom(const int* __restrict__ segs, int n_seg,
                                            const int* __restrict__ geom, int patch, int R,
                                            int tiles_cap, double sparsity) {
    VsaGeom g;
    g.nd = n_seg - 1;
    int P = 0;
    for (int s = 0; s < g.nd; ++s) {
        const int a = vsa_clamp(segs[s], 0, R), b = vsa_clamp(segs[s + 1], a, R);
        P += (b - a + kTile - 1) / kTile;
    }
    g.vbase = vsa_clamp(segs[g.nd], 0, R);
    int F = geom[0], Hp = geom[1] / patch, Wp = geom[2] / patch;
    // A grid that does not fit the rows after its first is no grid: every row index below stays
    // under R in 32 bits.
    bool fits = F > 0 && Hp > 0 && Wp > 0 && F <= R && Hp <= R && Wp <= R;
    if (fits) {
        const long long fh = (long long)F * Hp;
        fits = fh <= R && fh * Wp <= (long long)(R - g.vbase);
    }
    if (!fits) F = Hp = Wp = 0;
    g.F = F; g.Hp = Hp; g.Wp = Wp;
    g.nb = (Hp + 3) >> 2;
    g.nc = (Wp + 3) >> 2;
    g.mb = g.nb > 1 ? 0xffffffffu / (unsigned)g.nb + 1u : 0u;
    g.mc = g.nc > 1 ? 0xffffffffu / (unsigned)g.nc + 1u : 0u;
    int V = ((F + 3) >> 2) * g.nb * g.nc;
    P = min(P, tiles_cap);
    V = min(V, tiles_cap - P);
    g.P = P; g.V = V; g.N = P + V;
    g.k = 0;
    if (V > 0) {
        const double keep = (1.0 - sparsity) * (double)V;
        g.k = max(1, min((int)ceil(keep), V));
    }
    return g;
}

// A tile's rows: local index kk (its raster, w fastest) is row base + dt sT + dh sH + dw, with
// (dt, dh, dw) from kk by two divisions by extents of at most 4, done as (x * m) >> 8 with
// m = ceil(256 / n), exact for x < 64. A prefix tile is one run (nw 64, so dt = dh = 0).
struct VsaTile {
    int base, sT, sH, nw, nh, mw, mh, size;
};

__device__ __forceinline__ int vsa_magic(int n) { return (256 + n - 1) / n; }

__device__ __forceinline__ VsaTile vsa_tile(const VsaGeom& g, const int* __restrict__ segs,
                                            int R, int j) {
    VsaTile d;
    if (j < g.P) {
        d.base = 0;
        d.size = 1;
        for (int s = 0, jj = j; s < g.nd; ++s) {
            const int a = vsa_clamp(segs[s], 0, R), b = vsa_clamp(segs[s + 1], a, R);
            const int nt = (b - a + kTile - 1) / kTile;
            if (jj < nt) {
                d.base = a + jj * kTile;
                d.size = min(kTile, b - d.base);
                break;
            }
            jj -= nt;
        }
        d.sT = 0; d.sH = 0; d.nw = kTile; d.nh = 1; d.mw = 4; d.mh = 256;
        return d;
    }
    // Tile numbers are below 2^16 (the capacity is), where the multiply-high is the quotient; a
    // divisor of 1 has no 32-bit reciprocal and is the identity.
    const int v = j - g.P;
    const int ab = g.nc > 1 ? (int)__umulhi((unsigned)v, g.mc) : v, c = v - ab * g.nc;
    const int a = g.nb > 1 ? (int)__umulhi((unsigned)ab, g.mb) : ab, b = ab - a * g.nb;
    const int t0 = 4 * a, h0 = 4 * b, w0 = 4 * c;
    const int nt = min(4, g.F - t0), nh = min(4, g.Hp - h0), nw = min(4, g.Wp - w0);
    d.base = g.vbase + (t0 * g.Hp + h0) * g.Wp + w0;
    d.sT = g.Hp * g.Wp;
    d.sH = g.Wp;
    d.nw = nw; d.nh = nh;
    d.mw = vsa_magic(nw); d.mh = vsa_magic(nh);
    d.size = nt * nh * nw;
    return d;
}

__device__ __forceinline__ int vsa_row(const VsaTile& d, int kk) {
    const int q1 = (kk * d.mw) >> 8, dw = kk - q1 * d.nw;
    const int dt = (q1 * d.mh) >> 8, dh = q1 - dt * d.nh;
    return d.base + dt * d.sT + dh * d.sH + dw;
}

// Four bf16 (one uint2) added in order to four f32 sums.
__device__ __forceinline__ void vsa_acc4(float4& s, uint2 u) {
    s.x = s.x + bf16w_lo(u.x); s.y = s.y + bf16w_hi(u.x);
    s.z = s.z + bf16w_lo(u.y); s.w = s.w + bf16w_hi(u.y);
}

// The sums of q, k and (when `xv` is given) v over a tile's rows, one head's four dims each, in the
// tile's raster: the three share each row's address, and a batch of four rows' loads is issued
// before its adds, which stay in row order.
__device__ __forceinline__ void vsa_tile_sum3(const uint16_t* __restrict__ xq, int q_ld,
                                              const uint16_t* __restrict__ xk, int k_ld,
                                              const uint16_t* __restrict__ xv, int v_ld,
                                              const VsaTile& d, float4& sq, float4& sk,
                                              float4& sv) {
    sq = sk = sv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int kk = 0;
    for (; kk + 4 <= d.size; kk += 4) {
        uint2 uq[4], uk[4], uv[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const long row = vsa_row(d, kk + r);
            uq[r] = *(const uint2*)(xq + row * q_ld);
            uk[r] = *(const uint2*)(xk + row * k_ld);
            if (xv) uv[r] = *(const uint2*)(xv + row * v_ld);
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            vsa_acc4(sq, uq[r]);
            vsa_acc4(sk, uk[r]);
            if (xv) vsa_acc4(sv, uv[r]);
        }
    }
    for (; kk < d.size; ++kk) {
        const long row = vsa_row(d, kk);
        vsa_acc4(sq, *(const uint2*)(xq + row * q_ld));
        vsa_acc4(sk, *(const uint2*)(xk + row * k_ld));
        if (xv) vsa_acc4(sv, *(const uint2*)(xv + row * v_ld));
    }
}

// A score as an unsigned key that orders like the float, -0 folded onto +0 first so that the two
// zeros tie as they do as floats.
__device__ __forceinline__ uint32_t vsa_key(float s) {
    const uint32_t u = __float_as_uint(s + 0.0f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float vsa_unkey(uint32_t k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

__device__ __forceinline__ int vsa_wave_count(int c) { return (int)r4d_wave_sum((float)c); }

// ---------------------------------------------------------------------------- vsa_select: pool
// The q, k (and v) means of every tile of G heads into the scratch: k as [G][32][tiles_cap]
// float4, the dims in groups of four outermost, so the score loop's lanes (consecutive tiles) read
// 512 contiguous bytes; q and v as [G][tiles_cap][128] f32 rows, read by a wave a tile and by the
// coarse product's lanes (consecutive dims). A workgroup a tile and four heads, a lane four dims
// of one head. The sums are a dependent chain a lane, so they run here, at this kernel's
// occupancy, rather than in the select kernel's.
__global__ __launch_bounds__(128) void vsa_pool_kernel(
    const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
    const uint16_t* __restrict__ gv, int q_ld, int k_ld, int v_ld,
    const int* __restrict__ segs, int n_seg, const int* __restrict__ geom, int patch, int R,
    int tiles_cap, int h0, int G, float4* __restrict__ kpT, float4* __restrict__ qp,
    float4* __restrict__ vp) {
    const VsaGeom g = vsa_geom(segs, n_seg, geom, patch, R, tiles_cap, 0.0);
    const int j = blockIdx.x;
    const int gh = blockIdx.y * 4 + (threadIdx.x >> 5), d4 = threadIdx.x & 31;
    if (j >= g.N || gh >= G) return;
    const VsaTile d = vsa_tile(g, segs, R, j);
    const long col = (long)(h0 + gh) * kHd + 4 * d4;
    const float n = (float)d.size;
    float4 sq, sk, sv;
    vsa_tile_sum3(gq + col, q_ld, gk + col, k_ld, vp ? gv + col : nullptr, v_ld, d, sq, sk, sv);
    const long rw = ((long)gh * tiles_cap + j) * 32 + d4;
    kpT[((long)gh * 32 + d4) * tiles_cap + j] = make_float4(sk.x / n, sk.y / n, sk.z / n, sk.w / n);
    qp[rw] = make_float4(sq.x / n, sq.y / n, sq.z / n, sq.w / n);
    if (vp) vp[rw] = make_float4(sv.x / n, sv.y / n, sv.z / n, sv.w / n);
}

// ---------------------------------------------------------------------------- vsa_select: select
// QT query tiles of one head a workgroup of eight waves; NMAX bounds the tile count (the scores
// of every tile for each query tile are kept in LDS, as order keys, then as softmax weights).
//
//   1. q means of the QT tiles from the scratch into LDS.
//   2. Scores: a thread a key tile and all QT query tiles, the k mean read four dims at a time
//      from the transposed scratch (eight loads in flight), each dot product summed over the
//      dims in order.
//   3. Selection, a wave a query tile: the k-th largest key by bisection on its 32 bits (T is the
//      largest value with at least k keys >= T), then one ascending pass taking the keys above T
//      and the first (k - #above) keys equal to it, compacted by ballots.
//   4. Coarse: the wave's softmax over every tile (max-subtracted, f32) into the same LDS row,
//      then p times the v means summed over the tiles, a thread a dim and every query tile of
//      the group over half of them, eight tiles' loads in flight.
template <int QT, int NMAX>
__global__ __launch_bounds__(256) void vsa_select_kernel(
    const int* __restrict__ segs, int n_seg, const int* __restrict__ geom, int patch, int R,
    int tiles_cap, double sparsity, float scale, int h0, const float4* __restrict__ kpT,
    const float4* __restrict__ qp, const float* __restrict__ vp, int* __restrict__ sel,
    long long sel_hs, long long sel_ts, int k_cap, uint16_t* __restrict__ coarse,
    long long c_hs, long long c_ts) {
    static_assert(QT <= 8, "a wave a query tile");
    __shared__ uint32_t sS[QT * NMAX];
    __shared__ float4 sQ[QT * 32];

    const VsaGeom g = vsa_geom(segs, n_seg, geom, patch, R, tiles_cap, sparsity);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int i0 = blockIdx.x * QT, gh = blockIdx.y, hq = h0 + gh;
    if (i0 >= g.N) return;

    // 1. the q means
    if (wave < QT) {
        const int i = i0 + wave;
        sQ[wave * 32 + lane] = i < g.N ? qp[((long)gh * tiles_cap + i) * 32 + lane]
                                       : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    // 2. the scores
    const float4* kb = kpT + (long)gh * 32 * tiles_cap;
    for (int j0 = 0; j0 < g.N; j0 += 256) {
        const int j = j0 + tid, jc = min(j, g.N - 1);
        float acc[QT];
        #pragma unroll
        for (int q = 0; q < QT; ++q) acc[q] = 0.0f;
        #pragma unroll 1
        for (int d0 = 0; d0 < 32; d0 += 8) {
            float4 kv[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) kv[u] = kb[(long)(d0 + u) * tiles_cap + jc];
            // A fence a step: the eight loads above are what is hoisted, not the LDS reads of
            // every step (eight query tiles' worth each).
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                #pragma unroll
                for (int q = 0; q < QT; ++q) {
                    const float4 qv = sQ[q * 32 + d0 + u];
                    acc[q] = acc[q] + qv.x * kv[u].x;
                    acc[q] = acc[q] + qv.y * kv[u].y;
                    acc[q] = acc[q] + qv.z * kv[u].z;
                    acc[q] = acc[q] + qv.w * kv[u].w;
                }
                sched_barrier();
            }
        }
        if (j < g.N) {
            #pragma unroll
            for (int q = 0; q < QT; ++q) sS[q * NMAX + j] = vsa_key(acc[q] * scale);
        }
    }
    __syncthreads();

    // 3. the selection
    const int i = i0 + wave;
    uint32_t* row = sS + wave * NMAX;
    if (wave < QT && i < g.N && i >= g.P && g.V > 0) {
        const int k = g.k;
        uint32_t T = 0;
        for (int b = 31; b >= 0; --b) {
            const uint32_t cand = T | (1u << b);
            int c = 0;
            for (int j = g.P + lane; j < g.N; j += 32) c += row[j] >= cand ? 1 : 0;
            if (vsa_wave_count(c) >= k) T = cand;
        }
        int c = 0;
        for (int j = g.P + lane; j < g.N; j += 32) c += row[j] > T ? 1 : 0;
        const int need = k - vsa_wave_count(c);
        int* out = sel + (long long)hq * sel_hs + (long long)i * sel_ts;
        const int kw = min(k, k_cap);
        const uint32_t below = (1u << lane) - 1u;
        int eq_seen = 0, written = 0;
        for (int j0 = g.P; j0 < g.N && written < kw; j0 += 32) {
            const int j = j0 + lane;
            const uint32_t key = j < g.N ? row[j] : 0u;
            const bool gt = j < g.N && key > T, eq = j < g.N && key == T;
            const uint32_t em = (uint32_t)__builtin_amdgcn_ballot_w32(eq);
            const bool take = gt || (eq && eq_seen + (int)__popc(em & below) < need);
            const uint32_t tm = (uint32_t)__builtin_amdgcn_ballot_w32(take);
            const int pos = written + (int)__popc(tm & below);
            if (take && pos < kw) out[pos] = j;
            written += (int)__popc(tm);
            eq_seen += (int)__popc(em);
        }
    }
    if (!coarse) return;

    // 4. the coarse branch
    if (wave < QT && i < g.N) {
        float mx = -INFINITY;
        for (int j = lane; j < g.N; j += 32) mx = fmaxf(mx, vsa_unkey(row[j]));
        mx = r4d_wave_max(mx);
        float sum = 0.0f;
        for (int j = lane; j < g.N; j += 32) {
            const float e = expf(vsa_unkey(row[j]) - mx);
            row[j] = __float_as_uint(e);
            sum += e;
        }
        sum = r4d_wave_sum(sum);
        for (int j = lane; j < g.N; j += 32)
            row[j] = __float_as_uint(__uint_as_float(row[j]) / sum);
    }
    __syncthreads();
    // A thread a dim and every query tile of the group over half the tiles, so each v load feeds
    // QT products; the second half's sums meet the first's through LDS (sQ is free once the
    // scores are done).
    const int dim = tid & (kHd - 1), half = tid >> 7;
    const int jm = (g.N + 1) / 2, j1 = half ? g.N : jm;
    float acc[QT];
    #pragma unroll
    for (int q = 0; q < QT; ++q) acc[q] = 0.0f;
    const float* vb = vp + (long)gh * tiles_cap * kHd + dim;
    int j = half ? jm : 0;
    for (; j + 8 <= j1; j += 8) {
        float vv[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) vv[u] = vb[(long)(j + u) * kHd];
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            #pragma unroll
            for (int q = 0; q < QT; ++q)
                acc[q] = acc[q] + __uint_as_float(sS[q * NMAX + j + u]) * vv[u];
            sched_barrier();
        }
    }
    for (; j < j1; ++j) {
        const float vv = vb[(long)j * kHd];
        #pragma unroll
        for (int q = 0; q < QT; ++q) acc[q] = acc[q] + __uint_as_float(sS[q * NMAX + j]) * vv;
    }
    float* part = (float*)sQ;
    if (half) {
        #pragma unroll
        for (int q = 0; q < QT; ++q) part[q * kHd + dim] = acc[q];
    }
    __syncthreads();
    if (half) return;
    #pragma unroll
    for (int q = 0; q < QT; ++q) {
        const int iq = i0 + q;
        if (iq < g.N)
            coarse[(long long)hq * c_hs + (long long)iq * c_ts + dim] =
                f32_to_bf16(acc[q] + part[q * kHd + dim]);
    }
}

// ---------------------------------------------------------------------------- attn_vsa
// A workgroup NQT query tiles of one head, four waves of 16 query rows each (a tile's raster).
// Each query tile's visible key tiles are ascending (the prefix tiles, then its selection, which
// vsa_select writes ascending; every tile for a prefix query tile), so the workgroup walks their
// merge: a key tile is staged once for every query tile of the group that sees it, and the waves
// of the others skip its arithmetic.
//
// The kernel is bound by the stream of staged K/V tiles, which a random selection draws from the
// whole head (19 MB at 1344 x 768, past the 8 MB L2): two query tiles a workgroup stage 17%
// fewer tiles than one at random selections and a third fewer where neighbouring tiles select
// alike, and four lose more to idle waves than they save. Staging is r4d_attn_ps_kernel's (K
// rows into LDS, V through the transposing load into V^T), one tile ahead: the next tile's loads
// are issued into registers before the current tile's arithmetic and land in LDS after it. A key
// past its tile's size reads the tile's last row and is masked; 16-key blocks wholly past it are
// skipped, and full tiles take the unmasked arithmetic. LDS rows are not padded: 16-byte chunk c
// of row r is stored at c ^ (r & 7), so the eight lanes of a 16-byte access reach eight
// different bank groups, and the two tiles take 32 KiB (four workgroups a WGP).
constexpr int kNone = 0x7fffffff;

// log2(448) less 2^-16: exp2 of it is 448 (1 - 1.1e-5), under 448 with room for the hardware
// exp2's error, and still the E4M3 code of 448.
constexpr float kLog448 = 8.8073397f;

typedef int v2i_vsa __attribute__((ext_vector_type(2)));
typedef int v8i_vsa __attribute__((ext_vector_type(8)));

// quant_act_i8g codes for four values against `inv` = 1 / scale, packed into one dword.
__device__ __forceinline__ uint32_t vsa_q8x4(float a, float b, float c, float d, float inv) {
    auto q = [inv](float x) {
        return (uint32_t)((int)rintf(fminf(fmaxf(x * inv, -127.0f), 127.0f)) & 0xff);
    };
    return q(a) | (q(b) << 8) | (q(c) << 16) | (q(d) << 24);
}
__device__ __forceinline__ float vsa_amax8(uint4 u) {
    return fmaxf(fmaxf(fmaxf(fabsf(bf16w_lo(u.x)), fabsf(bf16w_hi(u.x))),
                       fmaxf(fabsf(bf16w_lo(u.y)), fabsf(bf16w_hi(u.y)))),
                 fmaxf(fmaxf(fabsf(bf16w_lo(u.z)), fabsf(bf16w_hi(u.z))),
                       fmaxf(fabsf(bf16w_lo(u.w)), fabsf(bf16w_hi(u.w)))));
}
__device__ __forceinline__ v2i_vsa vsa_q8x8(uint4 u, float inv) {
    const uint32_t a = vsa_q8x4(bf16w_lo(u.x), bf16w_hi(u.x), bf16w_lo(u.y), bf16w_hi(u.y), inv);
    const uint32_t b = vsa_q8x4(bf16w_lo(u.z), bf16w_hi(u.z), bf16w_lo(u.w), bf16w_hi(u.w), inv);
    return (v2i_vsa){ (int)a, (int)b };
}

__device__ __forceinline__ int vsa_sw(int r, int c) { return c ^ (r & 7); }

// The int8 + fp8 form is held to 192 VGPRs (eight waves a SIMD, four workgroups a WGP): it lands
// a few over without the target, which costs a workgroup a WGP. The bf16 + fp8 form spills
// under it and stays without.
template <int NQT, bool I8, bool PV8>
__global__ __launch_bounds__(NQT * 128) __attribute__((amdgpu_waves_per_eu(I8 && PV8 ? 8 : 1)))
void vsa_attn_kernel(
    const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
    const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
    const uint16_t* __restrict__ gc, const uint16_t* __restrict__ gg,
    const int* __restrict__ segs, int n_seg, const int* __restrict__ geom, int patch, int R,
    int tiles_cap, double sparsity, const int* __restrict__ sel, long long sel_hs,
    long long sel_ts, int k_cap, long long c_hs, long long c_ts, float scale, int q_ld, int k_ld,
    int v_ld, int g_ld, int o_ld, const int8_t* __restrict__ gk8, const float* __restrict__ gks,
    int k8_ld, int ks_ld, const float* __restrict__ gvs) {
    typedef DT16<0> D;
    typedef D::frag frag16;
    typedef typename std::conditional<I8, v2i_vsa, frag16>::type qfrag;

    constexpr int HD     = kHd;
    constexpr int TILE   = kTile;
    constexpr int NTHR   = NQT * 128;
    constexpr int NKS    = HD / 16;
    constexpr int DG     = HD / 8;              // 16-byte chunks of a K row
    constexpr int KP     = TILE * DG / NTHR;    // 16-byte K units a thread, and V units
    constexpr int VP     = KP;
    constexpr int VP8    = VP / 2;              // the same at E4M3: 16 dims of 8 keys a unit
    constexpr int MT     = TILE / 16;
    constexpr float GROW = D::GROW;

    // The int8 key tile is r4d_attn_ps_kernel's: HD bytes a row padded by 8, so a fragment read's
    // sixteen rows hit distinct bank pairs, written as 8-byte halves (odd rows are 8-byte
    // aligned), and a scale a key.
    constexpr int KSTR8  = HD + 8;
    // The E4M3 V^T tile: HD rows of TILE key bytes padded by 8, so the sixteen rows of a fragment
    // read (8 bytes a lane) fall on distinct bank pairs.
    constexpr int VSTR8  = TILE + 8;
    __shared__ __align__(16) uint16_t sK[I8 ? 8 : TILE * HD];
    __shared__ __align__(16) int8_t   sK8[I8 ? TILE * KSTR8 : 16];
    __shared__ __align__(16) float    sKs[I8 ? TILE : 4];
    __shared__ __align__(16) uint16_t sV[PV8 ? 8 : HD * TILE];
    __shared__ __align__(16) uint8_t  sV8[PV8 ? HD * VSTR8 : 16];

    const VsaGeom g = vsa_geom(segs, n_seg, geom, patch, R, tiles_cap, sparsity);
    const int i0 = blockIdx.x * NQT, hq = blockIdx.y;
    if (i0 >= g.N) return;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c = lane & 15, h = lane >> 4, swz = c & 7;
    const int qt = warp >> 2, wq = warp & 3;
    const int i = i0 + qt;
    const bool tval = i < g.N;
    const VsaTile qd = vsa_tile(g, segs, R, tval ? i : i0);
    const int qpos = wq * 16 + c;
    const bool live = tval && qpos < qd.size;
    const bool wlive = tval && wq * 16 < qd.size;
    const int qrow = vsa_row(qd, qpos < qd.size ? qpos : qd.size - 1);

    // The query: bf16 fragments pre-multiplied by the softmax scale, or int8 codes against the
    // row's absmax (a lane holds half the head's dims, lane + 16 the other half) with the scale
    // in `qs`, applied to the integer score.
    const float mul = scale * 1.44269504089f;
    qfrag qf[NKS];
    float qs = mul;
    {
        const uint16_t* qp = gq + (long)qrow * q_ld + (long)hq * HD;
        if constexpr (I8) {
            uint4 qu[NKS];
            float am = 0.0f;
            #pragma unroll
            for (int t = 0; t < NKS; ++t) {
                qu[t] = *(const uint4*)(qp + 16 * t + 8 * h);
                am = fmaxf(am, vsa_amax8(qu[t]));
            }
            am = fmaxf(am, swap16(am));
            const float sq = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
            const float inv = 1.0f / sq;
            qs = sq * mul;
            #pragma unroll
            for (int t = 0; t < NKS; ++t) qf[t] = vsa_q8x8(qu[t], inv);
        } else {
            #pragma unroll
            for (int t = 0; t < NKS; ++t) {
                const uint4 u = *(const uint4*)(qp + 16 * t + 8 * h);
                auto cv = [&](uint32_t w) {
                    return D::pk(__builtin_bit_cast(float, w << 16) * mul,
                                 __builtin_bit_cast(float, w & 0xffff0000u) * mul);
                };
                qf[t] = D::mk(make_uint2(cv(u.x), cv(u.y)), make_uint2(cv(u.z), cv(u.w)));
            }
        }
    }

    float m_ref = -3.0e38f, soff = 3.0e38f, l_i = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

    // The key-tile lists: query tile q of the group sees its entries e < len[q], entry e being e
    // itself while e < P or the query tile is a prefix tile, else its selection's slot e - P.
    const int ks = min(g.k, k_cap);
    int len[NQT], pos[NQT], cur[NQT];
    const int* srow[NQT];
    #pragma unroll
    for (int q = 0; q < NQT; ++q) {
        const int iq = i0 + q;
        len[q] = iq >= g.N ? 0 : iq < g.P ? g.N : g.P + ks;
        srow[q] = sel + (long long)hq * sel_hs + (long long)(iq < g.N ? iq : 0) * sel_ts;
        pos[q] = 0;
    }
    auto entry = [&](int q, int e) -> int {
        if (e >= len[q]) return kNone;
        if (e < g.P || i0 + q < g.P) return e;
        return vsa_clamp(srow[q][e - g.P], 0, g.N - 1);
    };
    #pragma unroll
    for (int q = 0; q < NQT; ++q) cur[q] = entry(q, 0);
    // The next tile of the merge and the query tiles that see it, as a bit mask.
    auto advance = [&](int& j, int& m) {
        j = kNone;
        #pragma unroll
        for (int q = 0; q < NQT; ++q) j = min(j, cur[q]);
        m = 0;
        if (j == kNone) return;
        #pragma unroll
        for (int q = 0; q < NQT; ++q)
            if (cur[q] == j) { m |= 1 << q; cur[q] = entry(q, ++pos[q]); }
    };

    // Staging: wave w stages keys KW w .. KW w + KW - 1, their K rows and V rows. Lane l decodes
    // the row of key KW w + l % KW once. Its V units read that key: eight-lane group g of the
    // wave takes key group g % (KW / 8) and dim groups g / (KW / 8) + (32 / KW) r, the eight
    // lanes of a group the group's eight keys, as the transposing load wants. Its K units take
    // keys KW w + l / 16 + 2 p at dims 8 (l % 16), the row from lane l / 16 + 2 p by a permute.
    // So a wave reads whole rows, and a row is decoded once a tile (decoding it per unit cost a
    // tenth of the kernel). An int8 K row is eight 16-byte units: keys KW w + l / 8 + 4 p, the
    // row from lane l / 8 + 4 p, and lane l < KW reads its key's scale. An E4M3 V unit is the same
    // transposing load over pairs of codes (16 dims of the group's 8 keys): lane j receives dims
    // 2j and 2j + 1 interleaved, and four byte permutes split them into two V^T rows of 8 keys.
    constexpr int NW = NTHR / 32;
    constexpr int KW = TILE / NW;
    constexpr int K8P = KW / 4;
    static_assert(KW == 2 * KP && VP == KW / 2 && (KW == 8 || KW == 16), "a wave's keys");
    const int kw0 = KW * warp, lg = lane >> 3, vj = lane & 7;
    const int vkg = kw0 / 8 + lg % (KW / 8), vdg = lg / (KW / 8);
    const uint16_t* kbase = gk + (long)hq * HD + (lane & 15) * 8;
    const int8_t* k8base = I8 ? gk8 + (long)hq * HD + (lane & 7) * 16 : nullptr;
    const uint16_t* vbase = gv + (long)hq * HD;
    const uint8_t* vbase8 = (const uint8_t*)gv + (long)hq * HD;
    // ext_vector_type staging registers: an array of HIP's uint4 carried across the loop is kept
    // in scratch (r4d_common.h).
    r4d_u32x4 kr[I8 ? K8P : KP], vr[PV8 ? VP8 : VP];
    float ksr = 0.0f;
    auto load = [&](const VsaTile& kd) {
        const int kk = kw0 + lane % KW;
        const int row = vsa_row(kd, kk < kd.size ? kk : kd.size - 1);
        if constexpr (PV8) {
            const uint8_t* v0 = vbase8 + (long)row * v_ld;
            #pragma unroll
            for (int r = 0; r < VP8; ++r)
                vr[r] = __builtin_bit_cast(r4d_u32x4,
                                           load_tr_b128(v0 + (vdg + (32 / KW) * r) * 16));
        } else {
            const uint16_t* v0 = vbase + (long)row * v_ld;
            #pragma unroll
            for (int r = 0; r < VP; ++r)
                vr[r] = __builtin_bit_cast(r4d_u32x4, load_tr_b128(v0 + (vdg + (32 / KW) * r) * 8));
        }
        if constexpr (I8) {
            #pragma unroll
            for (int p = 0; p < K8P; ++p) {
                const int krow = __shfl(row, (lane >> 3) + 4 * p);
                kr[p] = *(const r4d_u32x4*)(k8base + (long)krow * k8_ld);
            }
            if (lane < KW) ksr = gks[(long)row * ks_ld + hq];
        } else {
            #pragma unroll
            for (int p = 0; p < KP; ++p) {
                const int krow = __shfl(row, (lane >> 4) + 2 * p);
                kr[p] = *(const r4d_u32x4*)(kbase + (long)krow * k_ld);
            }
        }
    };
    auto store = [&]() {
        if constexpr (PV8) {
            #pragma unroll
            for (int r = 0; r < VP8; ++r) {
                const int row = (vdg + (32 / KW) * r) * 16 + 2 * vj;
                const r4d_u32x4 w = vr[r];
                *(uint2*)&sV8[row * VSTR8 + vkg * 8] =
                    make_uint2(__builtin_amdgcn_perm(w.y, w.x, 0x06040200u),
                               __builtin_amdgcn_perm(w.w, w.z, 0x06040200u));
                *(uint2*)&sV8[(row + 1) * VSTR8 + vkg * 8] =
                    make_uint2(__builtin_amdgcn_perm(w.y, w.x, 0x07050301u),
                               __builtin_amdgcn_perm(w.w, w.z, 0x07050301u));
            }
        } else {
            #pragma unroll
            for (int r = 0; r < VP; ++r) {
                const int row = (vdg + (32 / KW) * r) * 8 + vj;
                *(r4d_u32x4*)&sV[row * TILE + vsa_sw(row, vkg) * 8] = vr[r];
            }
        }
        if constexpr (I8) {
            #pragma unroll
            for (int p = 0; p < K8P; ++p) {
                int8_t* d = &sK8[(kw0 + (lane >> 3) + 4 * p) * KSTR8 + (lane & 7) * 16];
                *(uint2*)d       = make_uint2(kr[p].x, kr[p].y);
                *(uint2*)(d + 8) = make_uint2(kr[p].z, kr[p].w);
            }
            if (lane < KW) sKs[kw0 + lane] = ksr;
        } else {
            #pragma unroll
            for (int p = 0; p < KP; ++p) {
                const int key = kw0 + (lane >> 4) + 2 * p;
                *(r4d_u32x4*)&sK[key * HD + vsa_sw(key, lane & 15) * 8] = kr[p];
            }
        }
    };

    // A lane reads K row mt * 16 + c and V^T rows t * 16 + c, both at swizzle c & 7: the K
    // chunk of dims 16 t + 8 h is (2 t + h) ^ swz = 2 t ^ (h ^ swz).
    int koff[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) koff[t] = c * HD + ((2 * t) ^ (h ^ swz)) * 8;

    // Scores of keys mt * 16 + 8 h .. + 8 for the lane's query, in log2 units once multiplied by
    // `qs` (int8 only; the bf16 query already carries the scale).
    auto score = [&](int mt, float (&sc)[8]) {
            if constexpr (I8) {
                v8i_vsa si = (v8i_vsa){0, 0, 0, 0, 0, 0, 0, 0};
                const int8_t* kr2 = &sK8[(mt * 16 + c) * KSTR8] + 8 * h;
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    v2i_vsa kf[4];
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint2 u = *(const uint2*)(kr2 + 16 * (t0 + jj));
                        kf[jj] = (v2i_vsa){ (int)u.x, (int)u.y };
                    }
                    sched_barrier();
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj)
                        si = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                            true, kf[jj], true, qf[t0 + jj], si, false);
                }
                const float4 ka = *(const float4*)&sKs[mt * 16 + 8 * h];
                const float4 kb = *(const float4*)&sKs[mt * 16 + 8 * h + 4];
                const float kss[8] = { ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w };
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = (float)si[e] * kss[e];
            } else {
                v8f s = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
                const uint16_t* kr2 = &sK[mt * 16 * HD];
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    frag16 kf[4];
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint4 u = *(const uint4*)(kr2 + koff[t0 + jj]);
                        kf[jj] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                    }
                    sched_barrier();
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) s = D::wmma(kf[jj], qf[t0 + jj], s);
                }
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = s[e];
            }
    };

    // MASK is the edge tiles' form; a full tile's keys are all valid.
    auto compute = [&](int klim, auto maskc) {
        constexpr bool mask = decltype(maskc)::value;
        const int mte = mask ? (klim + 15) >> 4 : TILE / 16;
        #pragma unroll 1
        for (int mt = 0; mt < mte; ++mt) {
            float sc[8];
            score(mt, sc);
            if constexpr (mask) {
                const int kq0 = mt * 16 + 8 * h;
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = kq0 + e < klim ? sc[e] : -INFINITY;
            }

            // qs > 0, so the max of the scaled scores is the scaled max.
            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
            if (I8) smax *= qs;
            if (wave_any(smax > m_ref + GROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
                const float alpha = ex2(m_ref - mnew);
                #pragma unroll
                for (int t = 0; t < NKS; ++t)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[t][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
                soff  = -mnew;
            }

            // P to bf16 by truncation, the row sum from the same truncated values (see
            // r4d_attn_ps_kernel).
            uint32_t pw[4];
            #pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const float a0 = I8 ? __builtin_fmaf(sc[e], qs, soff) : sc[e] + soff;
                const float a1 = I8 ? __builtin_fmaf(sc[e + 1], qs, soff) : sc[e + 1] + soff;
                const uint32_t lo = __builtin_bit_cast(uint32_t, ex2(a0));
                const uint32_t hi = __builtin_bit_cast(uint32_t, ex2(a1));
                pw[e >> 1] = __builtin_amdgcn_perm(hi, lo, 0x07060302u);
            }
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                l_i += __builtin_bit_cast(float, pw[e] << 16) +
                       __builtin_bit_cast(float, pw[e] & 0xffff0000u);
            const frag16 p16 = D::mk(make_uint2(pw[0], pw[1]), make_uint2(pw[2], pw[3]));

            const uint16_t* vr2 = &sV[c * TILE + (((2 * mt) ^ (h ^ swz)) * 8)];
            #pragma unroll
            for (int t0 = 0; t0 < NKS; t0 += 4) {
                frag16 vf[4];
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const uint4 u = *(const uint4*)(vr2 + (t0 + jj) * 16 * TILE);
                    vf[jj] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                }
                sched_barrier();
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) acc[t0 + jj] = D::wmma(vf[jj], p16, acc[t0 + jj]);
            }
        }
    };

    // The fp8 P.V form, a tile at a time: the scores of all its 16-key blocks, one exact running
    // max over them (one rescale test a tile rather than a block), P as E4M3 codes of 448 p, then
    // the blocks' fp8 WMMAs. The max is kept in the scores' own units (the int8 form's are
    // unscaled, `qs` being one positive factor a row), so s - max is at most 0 exactly, and
    // x = exp2((s - max) qs + kLog448) cannot pass exp2(kLog448) < 448, where v_cvt_pk_fp8_f32
    // would give NaN. x is 448 p up to a factor common to every key, which the row sum shares.
    auto compute8 = [&](int klim, auto maskc) {
        constexpr bool mask = decltype(maskc)::value;
        const int mte = mask ? (klim + 15) >> 4 : MT;
        float sc[MT][8];
        auto score_all = [&]() {
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                if (mask && mt >= mte) {
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) sc[mt][e] = -INFINITY;
                    continue;
                }
                score(mt, sc[mt]);
                if constexpr (mask) {
                    const int kq0 = mt * 16 + 8 * h;
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) sc[mt][e] = kq0 + e < klim ? sc[mt][e] : -INFINITY;
                }
            }
        };
        bool scored = false;
        if constexpr (I8) {
            if constexpr (!mask) {
                // Two blocks' int8 chains interleaved: each WMMA waits on the one two back.
                #pragma unroll
                for (int mt = 0; mt < MT; mt += 2) {
                    v8i_vsa s0 = (v8i_vsa){0, 0, 0, 0, 0, 0, 0, 0}, s1 = s0;
                    const int8_t* r0 = &sK8[(mt * 16 + c) * KSTR8] + 8 * h;
                    const int8_t* r1 = r0 + 16 * KSTR8;
                    #pragma unroll
                    for (int t0 = 0; t0 < NKS; t0 += 4) {
                        v2i_vsa k0[4], k1[4];
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            const uint2 u0 = *(const uint2*)(r0 + 16 * (t0 + jj));
                            const uint2 u1 = *(const uint2*)(r1 + 16 * (t0 + jj));
                            k0[jj] = (v2i_vsa){ (int)u0.x, (int)u0.y };
                            k1[jj] = (v2i_vsa){ (int)u1.x, (int)u1.y };
                        }
                        sched_barrier();
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            s0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                true, k0[jj], true, qf[t0 + jj], s0, false);
                            s1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                true, k1[jj], true, qf[t0 + jj], s1, false);
                        }
                    }
                    #pragma unroll
                    for (int b = 0; b < 2; ++b) {
                        const float4 ka = *(const float4*)&sKs[(mt + b) * 16 + 8 * h];
                        const float4 kb = *(const float4*)&sKs[(mt + b) * 16 + 8 * h + 4];
                        const float kss[8] = { ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w };
                        #pragma unroll
                        for (int e = 0; e < 8; ++e)
                            sc[mt + b][e] = (float)(b ? s1[e] : s0[e]) * kss[e];
                    }
                }
                scored = true;
            }
        }
        if (!scored) score_all();
        float smax = sc[0][0];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int e = 0; e < 8; ++e) smax = fmaxf(smax, sc[mt][e]);
        if (wave_any(smax > m_ref)) {
            const float rmax  = fmaxf(smax, swap16(smax));
            const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
            const float alpha = ex2(I8 ? (m_ref - mnew) * qs : m_ref - mnew);
            #pragma unroll
            for (int t = 0; t < NKS; ++t)
                #pragma unroll
                for (int e = 0; e < 8; ++e) acc[t][e] *= alpha;
            l_i *= alpha;
            m_ref = mnew;
        }
        v2i_vsa p8[MT];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            float x[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float d = sc[mt][e] - m_ref;
                x[e] = ex2(I8 ? __builtin_fmaf(d, qs, kLog448) : d + kLog448);
            }
            l_i += ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
            // The first conversion's high half is dead (the second writes it), so any live
            // register serves as its old value: no zeroing moves.
            int w0 = __builtin_amdgcn_cvt_pk_fp8_f32(x[0], x[1], __builtin_bit_cast(int, x[0]),
                                                     false);
            w0 = __builtin_amdgcn_cvt_pk_fp8_f32(x[2], x[3], w0, true);
            int w1 = __builtin_amdgcn_cvt_pk_fp8_f32(x[4], x[5], __builtin_bit_cast(int, x[4]),
                                                     false);
            w1 = __builtin_amdgcn_cvt_pk_fp8_f32(x[6], x[7], w1, true);
            p8[mt] = (v2i_vsa){ w0, w1 };
        }
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (mask && mt >= mte) break;
            const uint8_t* vr2 = &sV8[c * VSTR8 + mt * 16 + 8 * h];
            #pragma unroll
            for (int t0 = 0; t0 < NKS; t0 += 4) {
                v2i_vsa vf[4];
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const uint2 u = *(const uint2*)(vr2 + (t0 + jj) * 16 * VSTR8);
                    vf[jj] = (v2i_vsa){ (int)u.x, (int)u.y };
                }
                sched_barrier();
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj)
                    acc[t0 + jj] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        vf[jj], p8[mt], acc[t0 + jj]);
            }
        }
    };

    // The walk: the staged tile (jc, the query tiles mc that see it) goes to LDS, the next one's
    // loads are issued, then the staged tile's arithmetic.
    int jc, mc;
    advance(jc, mc);
    VsaTile kdc = qd;
    if (jc != kNone) {
        kdc = vsa_tile(g, segs, R, jc);
        load(kdc);
    }
    while (jc != kNone) {
        int jn, mn;
        advance(jn, mn);
        lds_barrier();
        store();
        lds_barrier();
        VsaTile kdn = kdc;
        if (jn != kNone) {
            kdn = vsa_tile(g, segs, R, jn);
            load(kdn);
        }
        if (wlive && ((mc >> qt) & 1)) {
            if constexpr (PV8) {
                if (kdc.size == TILE) compute8(kdc.size, std::false_type{});
                else                  compute8(kdc.size, std::true_type{});
            } else {
                if (kdc.size == TILE) compute(kdc.size, std::false_type{});
                else                  compute(kdc.size, std::true_type{});
            }
        }
        jc = jn;
        mc = mn;
        kdc = kdn;
    }

    if (!live) return;
    const float ls = l_i + swap16(l_i);
    const float inv = (ls > 0.0f) ? (1.0f / ls) : 0.0f;
    uint16_t* op = go + (long)qrow * o_ld + (long)hq * HD;
    const uint16_t* cp = gc ? gc + (long long)hq * c_hs + (long long)i * c_ts : nullptr;
    const uint16_t* gp = gg ? gg + (long)qrow * g_ld + (long)hq * HD : nullptr;
    #pragma unroll
    for (int t = 0; t < NKS; ++t) {
        uint32_t tw[4];
        if constexpr (PV8) {
            // The column scales of the lane's eight dims.
            const float4 sa = *(const float4*)(gvs + (long)hq * HD + t * 16 + 8 * h);
            const float4 sb = *(const float4*)(gvs + (long)hq * HD + t * 16 + 8 * h + 4);
            const float vs[8] = { sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w };
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[t][e] *= vs[e];
        }
        if (cp) {
            // The epilogue's addcmul: o rounded to bf16, plus coarse * gate (exact in f32), one
            // rounding.
            const uint4 cu = *(const uint4*)(cp + t * 16 + 8 * h);
            const uint4 gu = *(const uint4*)(gp + t * 16 + 8 * h);
            const uint32_t cw[4] = { cu.x, cu.y, cu.z, cu.w }, gw[4] = { gu.x, gu.y, gu.z, gu.w };
            #pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const float o0 = bf16_to_f32(f32_to_bf16(acc[t][e] * inv));
                const float o1 = bf16_to_f32(f32_to_bf16(acc[t][e + 1] * inv));
                const float r0 = o0 + bf16w_lo(cw[e >> 1]) * bf16w_lo(gw[e >> 1]);
                const float r1 = o1 + bf16w_hi(cw[e >> 1]) * bf16w_hi(gw[e >> 1]);
                tw[e >> 1] = (uint32_t)f32_to_bf16(r0) | ((uint32_t)f32_to_bf16(r1) << 16);
            }
        } else {
            #pragma unroll
            for (int e = 0; e < 8; e += 2)
                tw[e >> 1] = (uint32_t)f32_to_bf16(acc[t][e] * inv)
                           | ((uint32_t)f32_to_bf16(acc[t][e + 1] * inv) << 16);
        }
        *(uint4*)(op + t * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
}

// ---------------------------------------------------------------------------- shared checks
inline bool is_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }
inline bool is_i32(const RadTensor* t) { return t && t->dtype == RAD_I32; }
inline int st_of(hipError_t e) { return e == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE); }

// What both ops read alike: the geometry parameters and the device geometry operands.
struct VsaCommon {
    long long R, nh, q_ld, k_ld;
    int n_seg, patch;
    double sparsity;
    float scale;
};

int vsa_common(const RadArgs* a, const RadTensor* q, const RadTensor* k, const RadTensor* segs,
               const RadTensor* geom, VsaCommon* c) {
    if (!is_bf16(q) || !is_bf16(k) || !is_i32(segs) || !is_i32(geom)) return R4D_NO(RAD_E_DTYPE);
    if (r4d_geti_or(a, "head_dim", 0) != kHd || r4d_geti_or(a, "tile_t", 0) != 4 ||
        r4d_geti_or(a, "tile_h", 0) != 4 || r4d_geti_or(a, "tile_w", 0) != 4)
        return R4D_NO(RAD_E_SHAPE);
    const long long patch = r4d_geti_or(a, "patch", 0);
    const double sp = r4d_getf(a, "sparsity", -1.0);
    if (patch < 1 || !(sp >= 0.0 && sp < 1.0)) return R4D_NO(RAD_E_INVAL);
    long long tk = 0, hk = 0;
    if (!r4d_head_rows(q, kHd, &c->R, &c->nh, &c->q_ld) ||
        !r4d_head_rows(k, kHd, &tk, &hk, &c->k_ld) ||
        ((c->q_ld | c->k_ld) & 7) || ((uintptr_t)q->data & 15) || ((uintptr_t)k->data & 15))
        return R4D_NO(RAD_E_STRIDE);
    if (hk != c->nh || tk < c->R || c->R <= 0 || c->R > 0x7fffffffLL) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(segs) || !r4d_contig(geom)) return R4D_NO(RAD_E_STRIDE);
    const long long ns = r4d_numel(segs) - 1;
    if (ns < 1 || ns > 4096 || r4d_numel(geom) < 3) return R4D_NO(RAD_E_SHAPE);
    c->n_seg = (int)ns;
    c->patch = (int)patch;
    c->sparsity = sp;
    c->scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)kHd));
    return RAD_OK;
}

// An operand of R rows of nh heads at its own pitch, 16-byte aligned.
bool vsa_rows(const RadTensor* t, long long R, long long nh, long long* ld) {
    long long rows = 0, heads = 0;
    return is_bf16(t) && r4d_head_rows(t, kHd, &rows, &heads, ld) && !(*ld & 7) &&
           !((uintptr_t)t->data & 15) && heads == nh && rows >= R;
}

// sel [>= nh, tiles_cap, k_cap] i32 with unit stride on the last axis, and coarse
// [>= nh, >= tiles_cap, 128] bf16 likewise.
int vsa_sel(const RadTensor* sel, const RadTensor* coarse, long long nh, long long* tiles_cap,
            long long* k_cap) {
    if (!is_i32(sel) || (coarse && !is_bf16(coarse))) return R4D_NO(RAD_E_DTYPE);
    if (sel->rank != 3 || sel->stride[2] != 1 ||
        (coarse && (coarse->rank != 3 || coarse->stride[2] != 1)))
        return R4D_NO(RAD_E_STRIDE);
    *tiles_cap = sel->shape[1];
    *k_cap = sel->shape[2];
    /* Tile numbers stay below 2^16 (the tile decode's multiply-high division). */
    if (sel->shape[0] < nh || *tiles_cap < 1 || *k_cap < 1 || *tiles_cap > 65536 ||
        *k_cap > 0x7fffffffLL)
        return R4D_NO(RAD_E_SHAPE);
    if (coarse && (coarse->shape[0] < nh || coarse->shape[1] < *tiles_cap ||
                   coarse->shape[2] != kHd || ((uintptr_t)coarse->data & 15) ||
                   (coarse->stride[1] & 7) || (coarse->stride[0] & 7)))
        return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

}  // namespace

// ============================================================== the plugin ABI entry points
enum { VS_Q = 0, VS_K, VS_V, VS_SEGS, VS_GEOM, VS_SEL, VS_COARSE };

/* The scratch holds the q, k and v means of a group of heads, 1.5 KiB a (tile, head) at the
 * layouts the select kernel reads. The hook sees neither the head count nor the selection's tile
 * capacity, so it sizes for M / 4 (tile, head) slots; the entry point takes as many heads a group
 * as fit (tiles_cap slots each) and refuses only when one does not. At 1344 x 768 (M 37.7K,
 * tiles_cap up to ~1500) that is six heads a group. */
extern "C" int64_t r4d_rad_vsa_select_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    return M * 384;
}

extern "C" int r4d_rad_vsa_select(const RadArgs* a, RadStream s) {
    const RadTensor *q = r4d_opt(a, VS_Q), *k = r4d_opt(a, VS_K), *v = r4d_opt(a, VS_V);
    const RadTensor *segs = r4d_opt(a, VS_SEGS), *geom = r4d_opt(a, VS_GEOM);
    const RadTensor *sel = r4d_opt(a, VS_SEL), *coarse = r4d_opt(a, VS_COARSE);
    if (!q || !k || !segs || !geom || !sel) return R4D_NO(RAD_E_INVAL);
    /* v is read only for the coarse branch: both or neither. */
    if ((v != nullptr) != (coarse != nullptr)) return R4D_NO(RAD_E_INVAL);
    VsaCommon c;
    int rc = vsa_common(a, q, k, segs, geom, &c);
    if (rc != RAD_OK) return rc;
    long long v_ld = 0, tiles_cap = 0, k_cap = 0;
    if (v && !vsa_rows(v, c.R, c.nh, &v_ld)) return R4D_NO(RAD_E_STRIDE);
    rc = vsa_sel(sel, coarse, c.nh, &tiles_cap, &k_cap);
    if (rc != RAD_OK) return rc;
    /* Two LDS budgets: eight query tiles a workgroup up to 1536 tiles, four up to 3072. */
    if (tiles_cap > 3072) return R4D_NO(RAD_E_SHAPE);
    const long long per = tiles_cap * (coarse ? 1536 : 1024);
    long long G = a->scratch && !((uintptr_t)a->scratch & 15)
                      ? std::min<long long>(c.nh, a->scratch_bytes / per) : 0;
    if (G < 1) return R4D_NO(RAD_E_SCRATCH);
    /* As many groups as the scratch needs, the heads spread evenly over them: a last group of a
     * head or two would leave most of the card idle for its two launches. */
    const long long rounds = (c.nh + G - 1) / G;
    G = (c.nh + rounds - 1) / rounds;
    float4* kpT = (float4*)a->scratch;
    float4* qp4 = (float4*)((char*)a->scratch + G * tiles_cap * 512);
    float4* vp4 = coarse ? (float4*)((char*)a->scratch + G * tiles_cap * 1024) : nullptr;
    const hipStream_t st = (hipStream_t)s;
    for (long long h0 = 0; h0 < c.nh; h0 += G) {
        const int g = (int)std::min<long long>(G, c.nh - h0);
        hipLaunchKernelGGL(vsa_pool_kernel, dim3((unsigned)tiles_cap, (unsigned)((g + 3) / 4)),
                           dim3(128), 0, st, (const uint16_t*)q->data, (const uint16_t*)k->data,
                           v ? (const uint16_t*)v->data : nullptr, (int)c.q_ld, (int)c.k_ld,
                           (int)v_ld, (const int*)segs->data, c.n_seg, (const int*)geom->data,
                           c.patch, (int)c.R, (int)tiles_cap, (int)h0, g, kpT, qp4, vp4);
#define R4D_VSEL(QT, NMAX)                                                                        \
        hipLaunchKernelGGL((vsa_select_kernel<QT, NMAX>),                                         \
                           dim3((unsigned)((tiles_cap + QT - 1) / QT), (unsigned)g), dim3(256), 0, \
                           st, (const int*)segs->data, c.n_seg, (const int*)geom->data,           \
                           c.patch, (int)c.R, (int)tiles_cap, c.sparsity, c.scale, (int)h0,       \
                           (const float4*)kpT, (const float4*)qp4, (const float*)vp4,             \
                           (int*)sel->data, (long long)sel->stride[0],                            \
                           (long long)sel->stride[1], (int)k_cap,                                 \
                           coarse ? (uint16_t*)coarse->data : nullptr,                            \
                           coarse ? (long long)coarse->stride[0] : 0LL,                           \
                           coarse ? (long long)coarse->stride[1] : 0LL)
        if (tiles_cap <= 1536) R4D_VSEL(8, 1536);
        else                   R4D_VSEL(4, 3072);
#undef R4D_VSEL
        const hipError_t e = hipGetLastError();
        if (e != hipSuccess) return st_of(e);
    }
    return RAD_OK;
}

enum { VA_Q = 0, VA_K, VA_V, VA_SEGS, VA_GEOM, VA_SEL, VA_COARSE, VA_GATE, VA_OUT, VA_K8, VA_KS,
       VA_VS };

extern "C" int r4d_rad_attn_vsa(const RadArgs* a, RadStream s) {
    const RadTensor *q = r4d_opt(a, VA_Q), *k = r4d_opt(a, VA_K), *v = r4d_opt(a, VA_V);
    const RadTensor *segs = r4d_opt(a, VA_SEGS), *geom = r4d_opt(a, VA_GEOM);
    const RadTensor *sel = r4d_opt(a, VA_SEL), *coarse = r4d_opt(a, VA_COARSE);
    const RadTensor *gate = r4d_opt(a, VA_GATE), *o = r4d_opt(a, VA_OUT);
    if (!q || !k || !v || !segs || !geom || !sel || !o) return R4D_NO(RAD_E_INVAL);
    /* The gated coarse term takes both. */
    if ((coarse != nullptr) != (gate != nullptr)) return R4D_NO(RAD_E_INVAL);
    VsaCommon c;
    int rc = vsa_common(a, q, k, segs, geom, &c);
    if (rc != RAD_OK) return rc;
    /* The fp8 P.V form (`pv` "fp8"): `v` as E4M3 codes [M, n_head * head_dim] and their column
     * scales `v_scale` [n_head * head_dim], as quant_cols_fp8 writes them. */
    const RadTensor* vs = r4d_opt(a, VA_VS);
    const char* pvs = r4d_gets(a, "pv");
    const bool pv8 = pvs && !std::strcmp(pvs, "fp8");
    if (pvs && !pv8 && std::strcmp(pvs, "bf16")) return R4D_NO(RAD_E_INVAL);
    if (pv8 != (vs != nullptr)) return R4D_NO(RAD_E_INVAL);
    long long v_ld = 0, o_ld = 0, g_ld = 0, tiles_cap = 0, k_cap = 0;
    if (pv8) {
        long long tv = 0, hv = 0;
        if (v->dtype != RAD_F8E4M3 || vs->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
        if (!r4d_head_rows(v, kHd, &tv, &hv, &v_ld) || (v_ld & 15) || ((uintptr_t)v->data & 15) ||
            !r4d_contig(vs) || ((uintptr_t)vs->data & 15))
            return R4D_NO(RAD_E_STRIDE);
        if (hv != c.nh || tv < c.R || r4d_numel(vs) < c.nh * kHd) return R4D_NO(RAD_E_SHAPE);
    } else if (!vsa_rows(v, c.R, c.nh, &v_ld)) {
        return R4D_NO(RAD_E_STRIDE);
    }
    if (!vsa_rows(o, c.R, c.nh, &o_ld) || (gate && !vsa_rows(gate, c.R, c.nh, &g_ld)))
        return R4D_NO(RAD_E_STRIDE);
    rc = vsa_sel(sel, coarse, c.nh, &tiles_cap, &k_cap);
    if (rc != RAD_OK) return rc;
    /* The int8 Q.K form (`qk` "i8"): the key codes [R, n_head * head_dim] and their scales
     * [R, n_head], as quant_act_i8g writes them at group head_dim. */
    const RadTensor *k8 = r4d_opt(a, VA_K8), *ks = r4d_opt(a, VA_KS);
    const char* qk = r4d_gets(a, "qk");
    const bool i8 = qk && !std::strcmp(qk, "i8");
    if (qk && !i8 && std::strcmp(qk, "bf16")) return R4D_NO(RAD_E_INVAL);
    if (i8 != (k8 != nullptr) || i8 != (ks != nullptr)) return R4D_NO(RAD_E_INVAL);
    long long k8_ld = 0, ks_ld = 0;
    if (i8) {
        if (k8->dtype != RAD_I8 || ks->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
        long long t8 = 0, h8 = 0, ts = 0;
        if (!r4d_head_rows(k8, kHd, &t8, &h8, &k8_ld) || (k8_ld & 15) || ((uintptr_t)k8->data & 15))
            return R4D_NO(RAD_E_STRIDE);
        ks_ld = r4d_rows_pitch(ks, c.nh, &ts);
        if (ks_ld < 0) return R4D_NO(RAD_E_STRIDE);
        if (h8 != c.nh || t8 < c.R || ts < c.R) return R4D_NO(RAD_E_SHAPE);
    }
    /* Two query tiles a workgroup (see vsa_attn_kernel). */
    constexpr int NQT = 2;
#define R4D_VATT(I8, PV8)                                                                         \
    hipLaunchKernelGGL((vsa_attn_kernel<NQT, I8, PV8>),                                           \
                       dim3((unsigned)((tiles_cap + NQT - 1) / NQT), (unsigned)c.nh),             \
                       dim3(NQT * 128), 0, (hipStream_t)s, (const uint16_t*)q->data,              \
                       (const uint16_t*)k->data, (const uint16_t*)v->data, (uint16_t*)o->data,    \
                       coarse ? (const uint16_t*)coarse->data : nullptr,                          \
                       gate ? (const uint16_t*)gate->data : nullptr, (const int*)segs->data,      \
                       c.n_seg, (const int*)geom->data, c.patch, (int)c.R, (int)tiles_cap,        \
                       c.sparsity, (const int*)sel->data, (long long)sel->stride[0],              \
                       (long long)sel->stride[1], (int)k_cap,                                     \
                       coarse ? (long long)coarse->stride[0] : 0LL,                               \
                       coarse ? (long long)coarse->stride[1] : 0LL, c.scale, (int)c.q_ld,         \
                       (int)c.k_ld, (int)v_ld, (int)g_ld, (int)o_ld,                              \
                       i8 ? (const int8_t*)k8->data : nullptr,                                    \
                       i8 ? (const float*)ks->data : nullptr, (int)k8_ld, (int)ks_ld,             \
                       pv8 ? (const float*)vs->data : nullptr)
    if (i8) {
        if (pv8) R4D_VATT(true, true);
        else     R4D_VATT(true, false);
    } else {
        if (pv8) R4D_VATT(false, true);
        else     R4D_VATT(false, false);
    }
#undef R4D_VATT
    return st_of(hipGetLastError());
}
