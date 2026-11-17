/* ref_vsa.cpp -- video sparse attention (docs/OPS.md, "Video sparse attention"): the tile
 * selection and the block-sparse attention a step-distilled MiniMax-H3 was trained with.
 *
 * A pass's rows are one packed sequence cut by `segs` (cumulative row offsets from 0) into
 * segments: every segment but the last is dense (the text, the soundtrack), the last is the video
 * grid of F x Hp x Wp patch rows, (t, h, w) row-major, its extents read from the `geom` operand the
 * run phase writes on the device (RAD_VGEOM_*), never from the parameters, which are fixed at
 * declare. The tiles, at most TT = tile_t tile_h tile_w rows each:
 *
 *   prefix   each dense segment cut on its own into chunks of TT rows from its start, the last
 *            chunk shorter, an empty segment none; tiles 0 .. P-1 in segment order
 *   video    boxes of tile_t x tile_h x tile_w over the grid, edge boxes smaller; box (a, b, c)
 *            is tile P + (a nb + b) nc + c, nb and nc the boxes along h and w; V of them
 *
 * A tile's rows are taken in its (t, h, w) raster order, which fixes the order of every f32 sum
 * over a tile. Video query tiles keep the k = max(1, min(ceil((1 - sparsity) V), V)) video tiles
 * of largest pooled score, k computed in f64 as the trained schedule computes it, and see the
 * prefix tiles besides; prefix query tiles see every tile.
 */
#include "ref_common.h"
#include "ref_gemm.h"   /* row_stride */
#include "ref_ops.h"
#include "rad_runtime.h"   /* RAD_VGEOM_* */

#include <algorithm>
#include <vector>

using namespace ref;

namespace {

/* `sparsity` at full precision, since k is computed from it in f64; NaN when absent, which every
 * caller refuses. As p_f32, an integer or string spelling is read for a direct caller. */
double p_f64(const RadArgs* a, const char* k) {
    const RadParam* p = p_find(a, k);
    if (!p) return NAN;
    if (p->kind == RAD_P_F64) return p->dval;
    if (p->kind == RAD_P_STR && p->sval) return std::strtod(p->sval, nullptr);
    if (p->kind == RAD_P_INT) return (double)p->ival;
    return NAN;
}

/* A call's tiles. */
struct Tiles {
    int64_t R = 0;                     /* rows the tiles cover, segs[n_seg] */
    int64_t v0 = 0;                    /* the video grid's first row */
    int64_t tt = 0, th = 0, tw = 0;    /* the box */
    int64_t F = 0, Hp = 0, Wp = 0;     /* the grid */
    int64_t nb = 0, nc = 0;            /* boxes along h and w */
    int64_t P = 0, V = 0, N = 0, k = 0;
    std::vector<int64_t> start, len;   /* the prefix tiles' rows */
};

/* The tiles of this call; RAD_E_SHAPE where `segs` is not cumulative offsets from 0 covering at
 * least one row and at most `rows`, the last segment is not the grid's F Hp Wp rows, a box extent
 * or `patch` is not positive, or `sparsity` is outside [0, 1). */
int tiles_of(const RadArgs* a, const RadTensor* segs, const RadTensor* geom, int64_t rows,
             Tiles* g) {
    g->tt = p_int(a, "tile_t", 0);
    g->th = p_int(a, "tile_h", 0);
    g->tw = p_int(a, "tile_w", 0);
    const int64_t patch = p_int(a, "patch", 0);
    const double sparsity = p_f64(a, "sparsity");
    if (g->tt <= 0 || g->th <= 0 || g->tw <= 0 || patch <= 0 ||
        !(sparsity >= 0.0 && sparsity < 1.0))
        return RAD_E_SHAPE;
    const int64_t n_seg = numel(segs) - 1;
    if (n_seg < 1 || numel(geom) <= RAD_VGEOM_LAT_W || cu_at(segs, 0) != 0) return RAD_E_SHAPE;
    for (int64_t s = 0; s < n_seg; ++s)
        if (cu_at(segs, s + 1) < cu_at(segs, s)) return RAD_E_SHAPE;
    g->R = cu_at(segs, n_seg);
    g->v0 = cu_at(segs, n_seg - 1);
    if (g->R <= 0 || g->R > rows) return RAD_E_SHAPE;   /* zero rows is not a call */
    g->F = ld_int(geom->dtype, geom->data, offlin(geom, RAD_VGEOM_LAT_FRAMES));
    g->Hp = ld_int(geom->dtype, geom->data, offlin(geom, RAD_VGEOM_LAT_H)) / patch;
    g->Wp = ld_int(geom->dtype, geom->data, offlin(geom, RAD_VGEOM_LAT_W)) / patch;
    /* Each extent at most the rows first, so the product cannot overflow. */
    const int64_t vr = g->R - g->v0;
    if (g->F <= 0 || g->Hp <= 0 || g->Wp <= 0 || g->F > vr || g->Hp > vr || g->Wp > vr ||
        g->F * g->Hp * g->Wp != vr)
        return RAD_E_SHAPE;
    const int64_t TT = g->tt * g->th * g->tw;
    g->start.clear();
    g->len.clear();
    for (int64_t s = 0; s + 1 < n_seg; ++s)
        for (int64_t r = cu_at(segs, s), e = cu_at(segs, s + 1); r < e; r += TT) {
            g->start.push_back(r);
            g->len.push_back(std::min(TT, e - r));
        }
    g->nb = (g->Hp + g->th - 1) / g->th;
    g->nc = (g->Wp + g->tw - 1) / g->tw;
    g->P = (int64_t)g->start.size();
    g->V = (g->F + g->tt - 1) / g->tt * g->nb * g->nc;
    g->N = g->P + g->V;
    const double keep = (1.0 - sparsity) * (double)g->V;
    g->k = std::max<int64_t>(1, std::min<int64_t>((int64_t)std::ceil(keep), g->V));
    return RAD_OK;
}

/* Video tile j's box, [t0, t1) x [h0, h1) x [w0, w1) of the grid. */
struct Box { int64_t t0, t1, h0, h1, w0, w1; };
Box box_of(const Tiles& g, int64_t j) {
    const int64_t b = j - g.P, c = b % g.nc, bh = (b / g.nc) % g.nb, at = b / (g.nc * g.nb);
    return { at * g.tt, std::min(g.F, (at + 1) * g.tt), bh * g.th, std::min(g.Hp, (bh + 1) * g.th),
             c * g.tw, std::min(g.Wp, (c + 1) * g.tw) };
}

int64_t tile_size(const Tiles& g, int64_t j) {
    if (j < g.P) return g.len[(size_t)j];
    const Box x = box_of(g, j);
    return (x.t1 - x.t0) * (x.h1 - x.h0) * (x.w1 - x.w0);
}

/* fn(row) for every row of tile j, in its raster order. */
template <class Fn>
void tile_rows(const Tiles& g, int64_t j, Fn&& fn) {
    if (j < g.P) {
        for (int64_t r = g.start[(size_t)j], e = r + g.len[(size_t)j]; r < e; ++r) fn(r);
        return;
    }
    const Box x = box_of(g, j);
    for (int64_t t = x.t0; t < x.t1; ++t)
        for (int64_t h = x.h0; h < x.h1; ++h)
            for (int64_t w = x.w0; w < x.w1; ++w) fn(g.v0 + (t * g.Hp + h) * g.Wp + w);
}

/* An operand read as [rows, n_head, head_dim]: rank 3, or rank 2 [rows, n_head head_dim] as a
 * column slice of a wider projection at its own row pitch (attn_dense's reading). */
struct Heads {
    const RadTensor* t;
    int64_t rs, hs, ds;
};
Heads heads_of(const RadTensor* t, int64_t nh, int64_t hd) {
    Heads v{ t, 0, 0, 0 };
    v.ds = t->rank >= 2 ? t->stride[t->rank - 1] : 1;
    v.hs = t->rank >= 3 ? t->stride[t->rank - 2] : hd * v.ds;
    v.rs = row_stride(t, nh * hd);
    return v;
}

/* The head count `q` carries, or 0 when its shape is not [rows, n_head, head_dim] or
 * [rows, n_head head_dim]. */
int64_t heads_in(const RadTensor* q, int64_t hd) {
    if (q->rank >= 3) return q->shape[q->rank - 1] == hd ? q->shape[q->rank - 2] : 0;
    if (q->rank == 2) return q->shape[1] % hd == 0 ? q->shape[1] / hd : 0;
    return 0;
}

/* Rows of a per-(head, tile) operand, [n_head, tiles_cap, cols] read as [n_head tiles_cap, cols]:
 * tiles_cap, or 0 when the rows do not split evenly over the heads. */
int64_t tiles_cap(const RadTensor* t, int64_t cols, int64_t nh) {
    const int64_t r = rows_of(t, cols);
    return r % nh == 0 ? r / nh : 0;
}

/* The selection's order: the larger score first, NaN above every number (torch.topk's ranking),
 * equal scores (+0 and -0 among them) tied. */
inline bool above(float a, float b) {
    if (a != a) return b == b;
    return b == b && a > b;
}

}  /* namespace */

/* ------------------------------------------------------------------ vsa_select */
/* Per head h and tile i, the f32 means of q, k and (with `coarse`) v over the tile's rows, each
 * summed in the tile's raster order and divided by its size:
 *
 *   S[h][i][j] = f32(dot(qp[h][i], kp[h][j]) * scale)       the dot an f32 sum in ascending d
 *
 * For a video query tile i, sel[h][i][0 .. k) is the k video tiles j of largest S[h][i][j],
 * ties to the lower tile, in ascending tile order: the k-th largest score is the threshold, every
 * tile above it is taken and the ties at it from the lowest. Prefix query tiles' rows of `sel`
 * and its slots past k are not written.
 *
 * With `coarse` (and `v`, read for it alone), for every tile i: p = softmax over all N tiles of
 * S[h][i][:] (the max subtracted, e_j = exp(S_j - max) summed in ascending j, p_j = e_j / sum),
 * coarse[h][i][d] = sum over ascending j of p_j vp[h][j][d], rounded once to coarse's dtype.
 * `sel` is i32 [n_head, tiles_cap >= N, k_cap >= k], `coarse` [n_head, tiles_cap >= N,
 * head_dim]. */
int ref_vsa_select(const RadArgs* a, RadStream) {
    const RadTensor *q = t_in(a, 0), *k = t_in(a, 1), *v = t_in(a, 2), *segs = t_in(a, 3);
    const RadTensor *geom = t_in(a, 4), *sel = t_in(a, 5), *co = t_in(a, 6);
    if (!q || !k || !segs || !geom || !sel) return RAD_E_INVAL;
    if (!v != !co) return RAD_E_SHAPE;
    if (sel->dtype != RAD_I32) return RAD_E_DTYPE;
    const int64_t hd = p_int(a, "head_dim", 0);
    if (hd <= 0) return RAD_E_SHAPE;
    const int64_t nh = heads_in(q, hd);
    if (nh <= 0) return RAD_E_SHAPE;
    Tiles g;
    RAD_REF_TRY(tiles_of(a, segs, geom, numel(q) / (nh * hd), &g));
    const int64_t need = g.R * nh * hd;
    if (numel(k) < need || (v && numel(v) < need)) return RAD_E_SHAPE;
    const int64_t kc = sel->rank ? sel->shape[sel->rank - 1] : 0;
    const int64_t scap = tiles_cap(sel, kc, nh);
    if (kc < g.k || scap < g.N) return RAD_E_SHAPE;
    const int64_t ccap = co ? tiles_cap(co, hd, nh) : 0;
    if (co && (co->shape[co->rank - 1] != hd || ccap < g.N)) return RAD_E_SHAPE;
    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)hd));
    const Heads Q = heads_of(q, nh, hd), K = heads_of(k, nh, hd);
    const Heads Vh = v ? heads_of(v, nh, hd) : Heads{ nullptr, 0, 0, 0 };
    const int64_t N = g.N, P = g.P;

    /* The pooled means, [n_head][N][head_dim] each. */
    std::vector<float> qp((size_t)(nh * N * hd)), kp(qp.size()), vp(co ? qp.size() : 0);
    REF_PAR_FOR(nh * g.R * hd * (co ? 3 : 2))
    for (int64_t u = 0; u < nh * N; ++u) {
        const int64_t h = u / N, j = u % N;
        float* qs = &qp[(size_t)(u * hd)];
        float* ks = &kp[(size_t)(u * hd)];
        float* vs = co ? &vp[(size_t)(u * hd)] : nullptr;
        for (int64_t d = 0; d < hd; ++d) {
            qs[d] = ks[d] = 0.0f;
            if (vs) vs[d] = 0.0f;
        }
        tile_rows(g, j, [&](int64_t r) {
            for (int64_t d = 0; d < hd; ++d) {
                qs[d] += ldt(q, r * Q.rs + h * Q.hs + d * Q.ds);
                ks[d] += ldt(k, r * K.rs + h * K.hs + d * K.ds);
                if (vs) vs[d] += ldt(v, r * Vh.rs + h * Vh.hs + d * Vh.ds);
            }
        });
        const float n = (float)tile_size(g, j);
        for (int64_t d = 0; d < hd; ++d) {
            qs[d] /= n;
            ks[d] /= n;
            if (vs) vs[d] /= n;
        }
    }

    REF_PAR_FOR(nh * N * N * hd)
    for (int64_t u = 0; u < nh * N; ++u) {
        const int64_t h = u / N, i = u % N;
        if (i < P && !co) continue;
        /* Scores against every tile when the coarse branch needs them, else the video tiles. */
        const int64_t j0 = co ? 0 : P;
        std::vector<float> S((size_t)N, 0.0f);
        const float* qs = &qp[(size_t)(u * hd)];
        for (int64_t j = j0; j < N; ++j) {
            const float* ks = &kp[(size_t)((h * N + j) * hd)];
            float dot = 0.0f;
            for (int64_t d = 0; d < hd; ++d) dot += qs[d] * ks[d];
            S[(size_t)j] = dot * scale;
        }
        if (i >= P) {
            const int64_t V = g.V, kk = g.k;
            std::vector<float> ord(S.begin() + P, S.end());
            std::nth_element(ord.begin(), ord.begin() + (kk - 1), ord.end(), above);
            const float thr = ord[(size_t)(kk - 1)];
            int64_t ties = kk;
            for (int64_t j = 0; j < V; ++j) ties -= above(S[(size_t)(P + j)], thr);
            const int64_t sb = rowoff(sel, h * scap + i, kc), ss = laststride(sel);
            int64_t n = 0;
            for (int64_t j = 0; j < V; ++j) {
                const float s = S[(size_t)(P + j)];
                const bool take = above(s, thr) || (!above(thr, s) && ties-- > 0);
                if (take) st_int(sel->dtype, sel->data, sb + (n++) * ss, P + j);
            }
        }
        if (co) {
            float mx = -INFINITY;
            for (int64_t j = 0; j < N; ++j) if (S[(size_t)j] > mx) mx = S[(size_t)j];
            float sum = 0.0f;
            for (int64_t j = 0; j < N; ++j) {
                S[(size_t)j] = ref_exp(S[(size_t)j] - mx);
                sum += S[(size_t)j];
            }
            std::vector<float> acc((size_t)hd, 0.0f);
            for (int64_t j = 0; j < N; ++j) {
                const float p = S[(size_t)j] / sum;
                const float* vs = &vp[(size_t)((h * N + j) * hd)];
                for (int64_t d = 0; d < hd; ++d) acc[(size_t)d] += p * vs[d];
            }
            const int64_t cb = rowoff(co, h * ccap + i, hd), cs = laststride(co);
            for (int64_t d = 0; d < hd; ++d) stt(co, cb + d * cs, acc[(size_t)d]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_vsa */
/* Block-sparse softmax attention over the tiles, per head (q, k and v share their heads). A query
 * row of tile i sees the rows of every tile when i < P, else of the prefix tiles and the k video
 * tiles sel[h][i][0 .. k); each such row of `sel` must be k ascending video tiles, RAD_E_INVAL
 * otherwise, checked before anything is written. Keys are taken by ascending tile and each tile
 * in its raster order: every score f32(dot * scale) (the dot an f32 sum in ascending d), their
 * max, then e = exp(score - max) and e v accumulated in that order, o = acc * (1 / sum).
 *
 *   out = o                                                   without coarse and gate
 *   out = f32(rd(o) + f32(coarse[h][i] * gate[r, h]))         with them
 *
 * rd rounding to out's dtype and the result rounded again by the store: FastVideo's inference
 * epilogue, the sparse output in bf16 and then out.addcmul_(coarse, gate) in bf16 with f32
 * arithmetic. `coarse` is vsa_select's, `gate` per element like q; both or neither.
 *
 * `qk` "i8" is the int8 score form: `k_i8` and `k_scale` are k's codes and per-(row, head) scales
 * as quant_act_i8g writes them at group head_dim, the query row is quantised here by the same rule,
 * and a score is f32(f32(f32(dot * q_scale) * k_scale) * scale) with the dot the codes' exact sum;
 * `k` is not read.
 *
 * `pv` "fp8" is the fp8 P.V form: `v` holds E4M3 codes with a scale per column in `v_scale`
 * (quant_cols_fp8), each weight p = exp(s - max) is taken to E4M3 at 448 p, and
 * o = (sum w v) v_scale / (448 sum p) in f32. */
int ref_attn_vsa(const RadArgs* a, RadStream) {
    const RadTensor *q = t_in(a, 0), *k = t_in(a, 1), *v = t_in(a, 2), *segs = t_in(a, 3);
    const RadTensor *geom = t_in(a, 4), *sel = t_in(a, 5), *co = t_in(a, 6), *gt = t_in(a, 7);
    const RadTensor* o = t_in(a, 8);
    const RadTensor *k8 = t_in(a, 9), *ks = t_in(a, 10), *vsc = a->n_t > 11 ? t_in(a, 11) : nullptr;
    if (!q || !k || !v || !segs || !geom || !sel || !o) return RAD_E_INVAL;
    const char* pv = p_str(a, "pv", "bf16");
    const bool pv8 = std::strcmp(pv, "fp8") == 0;
    if ((!pv8 && std::strcmp(pv, "bf16") != 0) || pv8 != (vsc != nullptr)) return RAD_E_INVAL;
    if (!co != !gt) return RAD_E_SHAPE;
    const char* qk = p_str(a, "qk", "bf16");
    const bool i8 = std::strcmp(qk, "i8") == 0;
    if ((!i8 && std::strcmp(qk, "bf16") != 0) || i8 != (k8 != nullptr) || i8 != (ks != nullptr))
        return RAD_E_INVAL;
    if (sel->dtype != RAD_I32) return RAD_E_DTYPE;
    const int64_t hd = p_int(a, "head_dim", 0);
    if (hd <= 0) return RAD_E_SHAPE;
    const int64_t nh = heads_in(q, hd);
    if (nh <= 0) return RAD_E_SHAPE;
    Tiles g;
    RAD_REF_TRY(tiles_of(a, segs, geom, numel(q) / (nh * hd), &g));
    const int64_t need = g.R * nh * hd;
    if (numel(k) < need || numel(v) < need || numel(o) < need || (gt && numel(gt) < need))
        return RAD_E_SHAPE;
    const int64_t kc = sel->rank ? sel->shape[sel->rank - 1] : 0;
    const int64_t scap = tiles_cap(sel, kc, nh);
    if (kc < g.k || scap < g.N) return RAD_E_SHAPE;
    const int64_t ccap = co ? tiles_cap(co, hd, nh) : 0;
    if (co && (co->shape[co->rank - 1] != hd || ccap < g.N)) return RAD_E_SHAPE;
    const int64_t N = g.N, P = g.P, kk = g.k, ss = laststride(sel);
    for (int64_t h = 0; h < nh; ++h)
        for (int64_t i = P; i < N; ++i) {
            const int64_t sb = rowoff(sel, h * scap + i, kc);
            int64_t prev = P - 1;
            for (int64_t s = 0; s < kk; ++s) {
                const int64_t j = ld_int(sel->dtype, sel->data, sb + s * ss);
                if (j <= prev || j >= N) return RAD_E_INVAL;
                prev = j;
            }
        }
    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)hd));
    const Heads Q = heads_of(q, nh, hd), K = heads_of(k, nh, hd), Vh = heads_of(v, nh, hd);
    const Heads O = heads_of(o, nh, hd);
    const Heads G = gt ? heads_of(gt, nh, hd) : Heads{ nullptr, 0, 0, 0 };
    const Heads K8 = i8 ? heads_of(k8, nh, hd) : Heads{ nullptr, 0, 0, 0 };
    const int64_t ks_rs = i8 ? row_stride(ks, nh) : 0;
    if (i8 && (numel(k8) < need || numel(ks) < g.R * nh)) return RAD_E_SHAPE;

    REF_PAR_REGION(nh * g.R * hd * 64)
    {
        float* const ws = ws_get((size_t)(g.R + 2 * hd));
        float* const sc = ws;
        float* const qf = ws + g.R;
        float* const acc = qf + hd;
        std::vector<int64_t> vis;
        REF_FOR_DYN
        for (int64_t u = 0; u < nh * N; ++u) {
            const int64_t h = u / N, i = u % N;
            vis.clear();
            for (int64_t j = 0; j < (i < P ? N : P); ++j) vis.push_back(j);
            if (i >= P) {
                const int64_t sb = rowoff(sel, h * scap + i, kc);
                for (int64_t s = 0; s < kk; ++s)
                    vis.push_back(ld_int(sel->dtype, sel->data, sb + s * ss));
            }
            const int64_t cb = co ? rowoff(co, h * ccap + i, hd) : 0;
            const int64_t cs = co ? laststride(co) : 0;
            tile_rows(g, i, [&](int64_t r) {
                const int64_t qb = r * Q.rs + h * Q.hs, ob = r * O.rs + h * O.hs;
                for (int64_t d = 0; d < hd; ++d) qf[d] = ldt(q, qb + d * Q.ds);
                const float qs = i8 ? ref_q8_row(qf, hd, qf) : 1.0f;
                /* Pass one: every visible key's score, and the max. */
                int64_t n = 0;
                float mx = -INFINITY;
                for (int64_t j : vis)
                    tile_rows(g, j, [&](int64_t c) {
                        float dot = 0.0f;
                        if (i8) {
                            /* Integers below 2^24 in magnitude: the f32 sum is exact. */
                            const int64_t kb = c * K8.rs + h * K8.hs;
                            for (int64_t d = 0; d < hd; ++d) dot += qf[d] * ldt(k8, kb + d * K8.ds);
                            dot = dot * qs * ldt(ks, c * ks_rs + h);
                        } else {
                            const int64_t kb = c * K.rs + h * K.hs;
                            for (int64_t d = 0; d < hd; ++d) dot += qf[d] * ldt(k, kb + d * K.ds);
                        }
                        dot *= scale;
                        sc[n++] = dot;
                        if (dot > mx) mx = dot;
                    });
                if (mx == -INFINITY) {
                    for (int64_t d = 0; d < hd; ++d) stt(o, ob + d * O.ds, 0.0f);
                    return;
                }
                /* Pass two: exponentiate and accumulate in the same order. */
                for (int64_t d = 0; d < hd; ++d) acc[d] = 0.0f;
                float sum = 0.0f;
                n = 0;
                for (int64_t j : vis)
                    tile_rows(g, j, [&](int64_t c) {
                        const float p = ref_exp(sc[n++] - mx);
                        sum += p;
                        const float w = pv8 ? round_dt(RAD_F8E4M3, 448.0f * p) : p;
                        const int64_t vb = c * Vh.rs + h * Vh.hs;
                        for (int64_t d = 0; d < hd; ++d) acc[d] += w * ldt(v, vb + d * Vh.ds);
                    });
                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                const float inv8 = sum > 0.0f ? 1.0f / (448.0f * sum) : 0.0f;
                for (int64_t d = 0; d < hd; ++d) {
                    float y = pv8 ? acc[d] * ldt(vsc, offlin(vsc, h * hd + d)) * inv8
                                  : acc[d] * inv;
                    if (co) {
                        const float c = ldt(co, cb + d * cs);
                        const float e = ldt(gt, r * G.rs + h * G.hs + d * G.ds);
                        y = round_dt(o->dtype, y) + c * e;
                    }
                    stt(o, ob + d * O.ds, y);
                }
            });
        }
    }
    return RAD_OK;
}
