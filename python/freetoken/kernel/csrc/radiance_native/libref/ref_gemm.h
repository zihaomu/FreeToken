/* ref_gemm.h -- how libref reads a matrix operand and its scale stream.
 *
 * Shared by the dense and mixture-of-experts GEMMs, which need the same answers: where row `r` of
 * an operand begins, the scale of a (row, k) pair, and how a block-scaled grid maps onto its
 * operand. Keeping one copy keeps one rblk rule, which block-scaled fp8 checkpoints depend on.
 *
 * Pure reads of a RadTensor: no op, no dispatch, no allocation.
 */
#ifndef REF_GEMM_H
#define REF_GEMM_H

#include "ref_common.h"

namespace ref {

/* Row stride of a tensor read as [rows, cols], which is not the same as dim 0 by dim 1.
 *
 * `stride[rank - 2]` is the row stride only when the last dimension is already `cols`. A rank-3
 * activation [tokens, heads, width] (the gated delta net's out-projection input, rank 3 because the
 * scan kernels need it) is read as [tokens, heads * width], whose row stride spans the trailing
 * dimensions. Using the head stride gets row 0 right and every later row wrong.
 *
 * Outputs follow the same rule. A gated attention q|gate projection is [tokens, heads,
 * 2 * head_dim]; writing it at the head stride leaves most of the buffer unwritten, so later rows
 * end in exact zeros. */
static inline int64_t row_stride(const RadTensor* t, int64_t cols) {
    if (!t || t->rank < 2) return cols;
    /* Fold trailing dimensions until they span `cols`; the next one out carries the row. */
    int64_t span = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        span *= t->shape[i];
        if (span == cols) return (i > 0) ? t->stride[i - 1] : cols;
        if (span > cols) break;
    }
    return t->stride[t->rank - 2];      /* the shapes disagree; fall back to the row axis */
}

/* Row and column strides of a [rows, cols] operand. Weights arrive contiguous; activations may be
 * a slice of a wider arena buffer, so both are read through their strides. */
struct Mat {
    const RadTensor* t;
    int64_t rs;      /* elements between rows */
    int64_t cs;      /* elements between columns */
};

static inline Mat as_mat(const RadTensor* t, int64_t cols) {
    Mat m{ t, cols, 1 };
    if (t->rank < 2) return m;
    m.cs = t->stride[t->rank - 1];
    m.rs = row_stride(t, cols);
    return m;
}

/* ------------------------------------------------------------------ scale streams
 * A quantised operand's scales are a separate SoA stream (spec §4.1), [rows, groups] with the
 * group index fastest, or [rows] for one scale per row. The group size is derived from the
 * stream's extent rather than the `group` parameter, so the two cannot disagree.
 *
 * A block-scaled grid is the same stream with fewer rows than the operand. DeepSeek-style fp8
 * (Qwen3.8-27B-FP8 and any `weight_block_size: [128, 128]` checkpoint) stores one scale per
 * 128x128 tile, so `weight_scale_inv` is [ceil(N/128)][ceil(K/128)] against an [N][K] weight.
 * `rblk` is how many operand rows share a scale row; 1 for every per-row grid. Without it a
 * 17408-row weight's 136-row scale plane would be read per row and give plausible wrong numbers. */
struct Scales {
    const void* p;
    uint32_t dt;
    int64_t   rs;        /* elements between scale rows */
    int64_t   cs;        /* elements between groups */
    int64_t   group;     /* K elements per scale */
    int64_t   rblk;      /* operand rows per scale row; 1 unless the grid is block-scaled */
    bool      e8m0;      /* the scale stream is OCP e8m0 exponent bytes, not floats */

    /* The scale of group `g` of `row`. */
    inline float grp(int64_t row, int64_t g) const {
        int64_t i = (rblk > 1 ? row / rblk : row) * rs + g * cs;
        return e8m0 ? e8m0_to_f32(((const uint8_t*)p)[i]) : ld_dt(dt, p, i);
    }
    /* The scale of element `k` of `row`. */
    inline float at(int64_t row, int64_t k) const { return grp(row, group > 0 ? k / group : 0); }
};

/* The block size a grid of `blocks` pieces implies over `ext` elements; not ext / blocks, since
 * the last block may be partial (300 over 128 is 3 scales covering 128, 128 and 44, but 300 / 3 =
 * 100 would misassign 100..127). It is the power of two whose ceiling division gives `blocks`,
 * unique for more than one block; every scale grid here is one (32, 64, 128). Otherwise the
 * smallest block giving `blocks` pieces. Either way ceil(ext / block) <= `blocks`, so derived
 * indices stay in the grid. */
static inline int64_t grid_block(int64_t ext, int64_t blocks) {
    if (ext <= 0) return 1;
    if (blocks <= 1) return ext;
    const int64_t lo = (ext + blocks - 1) / blocks;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (ext + b - 1) / b == blocks ? b : lo;
}

static inline Scales make_scales(const RadTensor* s, int64_t rows, int64_t K, bool mxfp4_weight) {
    Scales q{};
    q.rblk = 1;
    if (!s || !s->data || rows <= 0 || numel(s) <= 0) {
        q.p = nullptr; q.group = K; q.dt = RAD_F32; return q;
    }
    q.p     = s->data;
    q.dt    = s->dtype;
    /* A rank-2 stream states its own grid, the only way to tell per-row groups from per-row-block
     * scales. Fewer scale rows than operand rows is a block grid whatever the remainder
     * ([ceil(N/128)] rows against N = 129 is two blocks, the second one row); otherwise one scale
     * row per operand row, through the stream's row stride.
     *
     * A rank-1 or flattened stream states no grid. With a scale per row it is nel / rows groups per
     * row; with fewer it is one scale per block of rows (reading it per row would run off the end). */
    int64_t groups = 0;
    if (s->rank >= 2) {
        const int64_t srows = s->shape[s->rank - 2];
        q.rblk = srows >= rows ? 1 : grid_block(rows, srows);
        groups = s->shape[s->rank - 1];
        q.rs   = s->stride[s->rank - 2];
        q.cs   = s->stride[s->rank - 1];
    } else {
        const int64_t nel = numel(s);
        if (nel >= rows) {
            groups = nel / rows;
            q.rs   = groups;
        } else {
            groups = 1;
            q.rblk = grid_block(rows, nel);
            q.rs   = 1;
        }
        q.cs = 1;
    }
    q.group = grid_block(K, groups);
    /* mxfp4's block scale is an e8m0 exponent byte: a u8 stream beside an mxfp4 weight is read as
     * e8m0, while a float stream is read as floats, so plainly stored scales still work. */
    q.e8m0  = mxfp4_weight && (s->dtype == RAD_U8);
    return q;
}

/* ------------------------------------------------------------------ one quantised product
 *
 * One (m, n) dot of a quantised GEMM, and the single definition of how scales fold into it.
 *
 * Each group's sub-sum is accumulated from the raw codes and scaled once at the group's end, as an
 * int8 or fp8 WMMA kernel does when a tile closes. Scaling every product is equal in exact
 * arithmetic but differs in the last bits.
 *
 * `sub * sa * sb` is an order: (sub*sa)*sb and (sub*sb)*sa can differ by an ulp, enough to move an
 * fp8 code, and fused ops must be byte-identical to the pair they replace. Every caller uses this
 * function so there is one order.
 *
 * LA and LB load A's and B's elements: `Dyn` switches on the dtype per element and serves every
 * pair; `Fix<DT>` resolves it at compile time for the GEMM's hot pairs. Both return the same
 * floats, so the result does not depend on which was used.
 */
struct Dyn {
    static inline float ld(const RadTensor* t, int64_t i) { return ld_dt(t->dtype, t->data, i); }
};
template <uint32_t DT> struct Fix {
    static inline float ld(const RadTensor* t, int64_t i) { return Cvt<DT>::ld(t->data, i); }
};


template <class LA = Dyn, class LB = Dyn>
static inline float quant_dot(const Mat& A, const Mat& B, const Scales& SA, const Scales& SB,
                              int64_t m, int64_t n, int64_t K) {
    const int64_t g = SB.group > 0 ? SB.group : K;
    const int64_t ngrp = (K + g - 1) / g;
    /* No format here gives the activation a finer group than the weight; if one does, its scale
     * cannot be hoisted and is applied per element. */
    const bool coarse_a = !SA.p || SA.group >= g;
    const int64_t a0 = m * A.rs, b0 = n * B.rs;
    float acc = 0.0f;
    for (int64_t gi = 0; gi < ngrp; ++gi) {
        const int64_t k0 = gi * g, k1 = (k0 + g < K) ? k0 + g : K;
        const float sb = SB.p ? SB.grp(n, gi) : 1.0f;
        float sub = 0.0f;
        if (coarse_a) {
            for (int64_t k = k0; k < k1; ++k)
                sub += LA::ld(A.t, a0 + k * A.cs) * LB::ld(B.t, b0 + k * B.cs);
            acc += sub * (SA.p ? SA.at(m, k0) : 1.0f) * sb;
        } else {
            for (int64_t k = k0; k < k1; ++k)
                sub += LA::ld(A.t, a0 + k * A.cs) * SA.at(m, k) * LB::ld(B.t, b0 + k * B.cs);
            acc += sub * sb;
        }
    }
    return acc;
}

/* ------------------------------------------------------------------ the blocked product
 *
 * quant_dot over a tile of outputs: A rows `arow[0..R)` against B rows [n0, n0 + nb). Every
 * output is the chain quant_dot forms, per group a sub-sum from zero over ascending k and then
 * `acc + (sub * sa) * sb`, so the result is bit-identical to calling it per element. What changes
 * is that REF_NB outputs advance together: the chains are independent, so they run in vector lanes
 * and their add latencies overlap, and B's codes are widened into an f32 panel once per K chunk
 * and reused by every row of the tile (exact conversions, so the panel holds the same floats the
 * loaders return).
 *
 * Only the coarse activation grid (no finer scale on A than on B) is tiled; quant_dot keeps the
 * other, which no format uses. */
static constexpr int64_t REF_NB = 16, REF_KC = 256, REF_MB = 2;

typedef float RefV4 __attribute__((vector_size(16)));

/* Floats a tile of R rows needs: the panel, two A rows, and the sub-sums and accumulators of R
 * rows rounded up to whole row pairs. One ws_get() carves all of it. */
static inline size_t tile_ws_floats(int64_t R) {
    const int64_t rp = (R + REF_MB - 1) / REF_MB * REF_MB;
    return (size_t)(REF_KC * REF_NB + REF_MB * REF_KC + 2 * rp * REF_NB);
}

/* sub[2][REF_NB] += abuf[i][k] * panel[k][*] over k < kl, one chain per output. */
static inline void tile_kernel(const float* abuf, const float* panel, float* sub, int64_t kl) {
    static_assert(REF_NB == 16 && REF_MB == 2, "the kernel holds two rows of four vectors");
    auto ld = [](const float* f) { RefV4 v; std::memcpy(&v, f, sizeof v); return v; };
    auto st = [](float* f, RefV4 v) { std::memcpy(f, &v, sizeof v); };
    RefV4 c00 = ld(sub),      c01 = ld(sub + 4),  c02 = ld(sub + 8),  c03 = ld(sub + 12);
    RefV4 c10 = ld(sub + 16), c11 = ld(sub + 20), c12 = ld(sub + 24), c13 = ld(sub + 28);
    for (int64_t k = 0; k < kl; ++k) {
        const float a0 = abuf[k], a1 = abuf[REF_KC + k];
        const float* p = panel + k * REF_NB;
        const RefV4 p0 = ld(p), p1 = ld(p + 4), p2 = ld(p + 8), p3 = ld(p + 12);
        c00 += a0 * p0; c01 += a0 * p1; c02 += a0 * p2; c03 += a0 * p3;
        c10 += a1 * p0; c11 += a1 * p1; c12 += a1 * p2; c13 += a1 * p3;
    }
    st(sub, c00);      st(sub + 4, c01);  st(sub + 8, c02);  st(sub + 12, c03);
    st(sub + 16, c10); st(sub + 20, c11); st(sub + 24, c12); st(sub + 28, c13);
}

/* The tile's results land in `ws`'s accumulator block, [R][REF_NB] row-major, which this returns;
 * columns at and past `nb` are padding. Requires the coarse activation grid (see above). */
template <class LA, class LB>
static float* quant_tile(const Mat& A, const Mat& B, const Scales& SA, const Scales& SB,
                         const int64_t* arow, int64_t R, int64_t n0, int64_t nb, int64_t K,
                         float* ws) {
    const int64_t rp = (R + REF_MB - 1) / REF_MB * REF_MB;
    float* const panel = ws;
    float* const abuf  = panel + REF_KC * REF_NB;
    float* const sub   = abuf + REF_MB * REF_KC;
    float* const acc   = sub + rp * REF_NB;
    const int64_t g = SB.group > 0 ? SB.group : K;
    const int64_t ngrp = (K + g - 1) / g;
    /* No scale on either side: quant_dot's acc is 0 + (sub * 1) * 1, which is sub exactly (a
     * sub-sum started at +0 is never -0). */
    const bool unscaled = !SA.p && !SB.p && ngrp == 1;
    for (int64_t i = 0; i < rp * REF_NB; ++i) acc[i] = 0.0f;
    float sb[REF_NB];
    for (int64_t gi = 0; gi < ngrp; ++gi) {
        const int64_t k0 = gi * g, k1 = (k0 + g < K) ? k0 + g : K;
        for (int64_t j = 0; j < REF_NB; ++j) sb[j] = (j < nb && SB.p) ? SB.grp(n0 + j, gi) : 1.0f;
        for (int64_t i = 0; i < rp * REF_NB; ++i) sub[i] = 0.0f;
        for (int64_t kc0 = k0; kc0 < k1; kc0 += REF_KC) {
            const int64_t kl = (kc0 + REF_KC < k1) ? REF_KC : k1 - kc0;
            for (int64_t j = 0; j < REF_NB; ++j) {
                if (j < nb) {
                    const int64_t b0 = (n0 + j) * B.rs + kc0 * B.cs;
                    for (int64_t k = 0; k < kl; ++k)
                        panel[k * REF_NB + j] = LB::ld(B.t, b0 + k * B.cs);
                } else {
                    for (int64_t k = 0; k < kl; ++k) panel[k * REF_NB + j] = 0.0f;
                }
            }
            for (int64_t r0 = 0; r0 < R; r0 += REF_MB) {
                for (int64_t i = 0; i < REF_MB; ++i) {
                    float* const ar = abuf + i * REF_KC;
                    if (r0 + i < R) {
                        const int64_t a0 = arow[r0 + i] * A.rs + kc0 * A.cs;
                        for (int64_t k = 0; k < kl; ++k) ar[k] = LA::ld(A.t, a0 + k * A.cs);
                    } else {
                        for (int64_t k = 0; k < kl; ++k) ar[k] = 0.0f;
                    }
                }
                tile_kernel(abuf, panel, sub + r0 * REF_NB, kl);
            }
        }
        if (unscaled) {
            for (int64_t i = 0; i < rp * REF_NB; ++i) acc[i] = sub[i];
            continue;
        }
        for (int64_t r = 0; r < R; ++r) {
            const float sa = SA.p ? SA.at(arow[r], k0) : 1.0f;
            for (int64_t j = 0; j < REF_NB; ++j)
                acc[r * REF_NB + j] += sub[r * REF_NB + j] * sa * sb[j];
        }
    }
    return acc;
}

/* How a grid of R rows by N outputs is cut into tiles for one parallel loop: `nblk` column blocks
 * of REF_NB, each split into `rgc` row groups of `rg` rows. Rows are split only when there are too
 * few column blocks to occupy the threads (a narrow projection at a prefill M); a row group then
 * pays its own panel widening, which is cheap against rg rows of products. */
struct TileGrid {
    int64_t nblk, rg, rgc;
    int64_t items() const { return nblk * rgc; }
};
static inline TileGrid tile_grid(int64_t R, int64_t N) {
    TileGrid t;
    t.nblk = (N + REF_NB - 1) / REF_NB;
    int64_t rg = R;
    if (t.nblk < 128 && R > 8 * REF_MB) {
        rg = (R * t.nblk + 127) / 128;
        if (rg < 8 * REF_MB) rg = 8 * REF_MB;
        rg = (rg + REF_MB - 1) / REF_MB * REF_MB;
    }
    t.rg  = rg > 0 ? rg : 1;
    t.rgc = R > 0 ? (R + t.rg - 1) / t.rg : 0;
    return t;
}

/* ------------------------------------------------------------------ weight tables
 *
 * A RAD_OPD_WTAB operand: `data` is an array of `shape[0]` pointers, one per expert, marked by
 * `stride[0]` == 0 (RAD_WTAB in abi/rad_runtime.h). The stacked form, a plain rank-3
 * [n_expert, N, K] tensor, is also accepted; callers outside the engine such as rad-kbench pass
 * that, having nowhere to put a pointer array.
 *
 * Returns entry `e` as a rank-(rank-1) tensor of the same dtype, or a null-data tensor when the
 * operand cannot be read that way. */
static inline RadTensor expert_entry(const RadTensor* w, int64_t e) {
    RadTensor t{};
    if (!w || !w->data || w->rank < 2 || e < 0 || e >= w->shape[0]) return t;
    t.dtype = w->dtype;
    t.rank  = w->rank - 1;
    for (uint32_t d = 0; d + 1 < w->rank; ++d) {
        t.shape[d]  = w->shape[d + 1];
        t.stride[d] = w->stride[d + 1];
    }
    if (w->stride[0] == 0) {
        void* const* tab = (void* const*)w->data;
        t.data = tab[e];
    } else {
        /* A packed-dtype element offset can be mid-byte; stacked planes are whole bytes, so
         * compute in bits off the leading stride and refuse a fractional byte. */
        const int64_t bits = rad_dtype_bits(w->dtype);
        const int64_t nbits = (int64_t)bits * w->stride[0] * e;
        if (bits <= 0 || (nbits & 7)) return t;          /* data stays null: unreadable */
        t.data = (void*)((const char*)w->data + (nbits >> 3));
    }
    return t;
}

}  /* namespace ref */

#endif /* REF_GEMM_H */
