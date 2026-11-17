/* avx_vsa.cpp -- video sparse attention (docs/OPS.md, "Video sparse attention"): vsa_select's
 * top-k of pooled tile scores and its coarse branch, and attn_vsa's block-sparse attention.
 *
 * The tiles are libref's (libref/ref_vsa.cpp), read off the same operands with the same refusals
 * in the same order: each dense segment of `segs` in chunks of tile_t tile_h tile_w rows, then the
 * boxes over the video grid `geom` describes, a tile's rows in its (t, h, w) raster order.
 *
 * vsa_select's `sel` is libref's bit for bit: one tile more or less changes the keys a whole query
 * tile sees. The tile means are summed row by row in raster order with the vector lanes across
 * head_dim, each lane libref's sum, then divided by the size; a score's dot runs over d in
 * ascending order with each product rounded before the add (the lanes across tiles, `keep`
 * stopping the pair from fusing), then the scale; the selection is libref's threshold rule. The
 * coarse branch takes the vector exp and is held to the tolerance.
 *
 * attn_vsa: a task is one (head, query tile), up to 64 of its rows at a time. Per visible key tile,
 * 64 keys at a time are widened and packed (K as Kt [hd][64], V as [64][hp]), then S = Q K^T as
 * broadcast-against-tile products, the online softmax (the running max moved at tile boundaries)
 * and O += P V, avx_gen_attn.cpp's f32 form. Results differ from libref's two-pass softmax in
 * summation order and the rescales' roundings, within the attention tolerance. Each output row is
 * one task's, so they do not depend on the thread count.
 */
#include "avx_vec.h"
#include "avx_common.h"
#include "rad_runtime.h"   /* RAD_VGEOM_* */

#include <algorithm>
#include <vector>

using namespace avx;

namespace {

/* ================================================================== the tiles (libref's) */
inline int64_t at(const RadTensor* t, int64_t i) { return ldi(t, offlin(t, i)); }

/* `sparsity` at full precision: k is computed from it in f64. NaN when absent. */
double p_f64(const RadArgs* a, const char* k) {
    const RadParam* p = p_find(a, k);
    if (!p) return NAN;
    if (p->kind == RAD_P_F64) return p->dval;
    if (p->kind == RAD_P_STR && p->sval) return std::strtod(p->sval, nullptr);
    if (p->kind == RAD_P_INT) return (double)p->ival;
    return NAN;
}

struct Tiles {
    int64_t R = 0, v0 = 0;
    int64_t tt = 0, th = 0, tw = 0;
    int64_t F = 0, Hp = 0, Wp = 0, nb = 0, nc = 0;
    int64_t P = 0, V = 0, N = 0, k = 0;
    std::vector<int64_t> start, len;
};

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
    if (n_seg < 1 || numel(geom) <= RAD_VGEOM_LAT_W || at(segs, 0) != 0) return RAD_E_SHAPE;
    for (int64_t s = 0; s < n_seg; ++s)
        if (at(segs, s + 1) < at(segs, s)) return RAD_E_SHAPE;
    g->R = at(segs, n_seg);
    g->v0 = at(segs, n_seg - 1);
    if (g->R <= 0 || g->R > rows) return RAD_E_SHAPE;
    g->F = at(geom, RAD_VGEOM_LAT_FRAMES);
    g->Hp = at(geom, RAD_VGEOM_LAT_H) / patch;
    g->Wp = at(geom, RAD_VGEOM_LAT_W) / patch;
    const int64_t vr = g->R - g->v0;
    if (g->F <= 0 || g->Hp <= 0 || g->Wp <= 0 || g->F > vr || g->Hp > vr || g->Wp > vr ||
        g->F * g->Hp * g->Wp != vr)
        return RAD_E_SHAPE;
    const int64_t TT = g->tt * g->th * g->tw;
    g->start.clear();
    g->len.clear();
    for (int64_t s = 0; s + 1 < n_seg; ++s)
        for (int64_t r = at(segs, s), e = at(segs, s + 1); r < e; r += TT) {
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

struct Box { int64_t t0, t1, h0, h1, w0, w1; };
Box box_of(const Tiles& g, int64_t j) {
    const int64_t b = j - g.P, c = b % g.nc, bh = (b / g.nc) % g.nb, ta = b / (g.nc * g.nb);
    return { ta * g.tt, std::min(g.F, (ta + 1) * g.tt), bh * g.th, std::min(g.Hp, (bh + 1) * g.th),
             c * g.tw, std::min(g.Wp, (c + 1) * g.tw) };
}

int64_t tile_size(const Tiles& g, int64_t j) {
    if (j < g.P) return g.len[(size_t)j];
    const Box x = box_of(g, j);
    return (x.t1 - x.t0) * (x.h1 - x.h0) * (x.w1 - x.w0);
}

/* The n-th row of tile j in its raster order. */
int64_t row_at(const Tiles& g, int64_t j, int64_t n) {
    if (j < g.P) return g.start[(size_t)j] + n;
    const Box x = box_of(g, j);
    const int64_t dh = x.h1 - x.h0, dw = x.w1 - x.w0;
    const int64_t t = x.t0 + n / (dh * dw), h = x.h0 + (n / dw) % dh, w = x.w0 + n % dw;
    return g.v0 + (t * g.Hp + h) * g.Wp + w;
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

/* An operand read as [rows, n_head, head_dim] (libref's reading). */
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

int64_t heads_in(const RadTensor* q, int64_t hd) {
    if (q->rank >= 3) return q->shape[q->rank - 1] == hd ? q->shape[q->rank - 2] : 0;
    if (q->rank == 2) return q->shape[1] % hd == 0 ? q->shape[1] / hd : 0;
    return 0;
}

int64_t tiles_cap(const RadTensor* t, int64_t cols, int64_t nh) {
    const int64_t r = rows_of(t, cols);
    return r % nh == 0 ? r / nh : 0;
}

/* libref's selection order: the larger score first, NaN above every number, equal scores tied. */
inline bool above(float a, float b) {
    if (a != a) return b == b;
    return b == b && a > b;
}

/* A product the compiler may not fuse into the next add: libref rounds every product before the
 * sum (it is built without contraction, this plugin with -ffp-contract=fast). */
inline vf keep(vf x) {
#if AVX_LEVEL == 3
    __asm__("" : "+v"(x));
#else
    __asm__("" : "+x"(x));
#endif
    return x;
}
#if AVX_LEVEL >= 1
inline float keep(float x) {
#if AVX_LEVEL == 3
    __asm__("" : "+v"(x));
#else
    __asm__("" : "+x"(x));
#endif
    return x;
}
#endif

inline void axpy(float* acc, const float* x, float w, int64_t n) {
    const vf vw = vf_set1(w);
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N)
        vf_storeu(acc + i, vf_fma(vw, vf_loadu(x + i), vf_loadu(acc + i)));
    for (; i < n; ++i) acc[i] += w * x[i];
}

/* The checks both ops share, in libref's order. */
struct Call {
    Tiles g;
    int64_t hd = 0, nh = 0, kc = 0, scap = 0, ccap = 0;
};

int common(const RadArgs* a, const RadTensor* q, const RadTensor* segs, const RadTensor* geom,
           Call* c) {
    c->hd = p_int(a, "head_dim", 0);
    if (c->hd <= 0) return RAD_E_SHAPE;
    c->nh = heads_in(q, c->hd);
    if (c->nh <= 0) return RAD_E_SHAPE;
    return tiles_of(a, segs, geom, numel(q) / (c->nh * c->hd), &c->g);
}

int caps(const RadTensor* sel, const RadTensor* co, Call* c) {
    c->kc = sel->rank ? sel->shape[sel->rank - 1] : 0;
    c->scap = tiles_cap(sel, c->kc, c->nh);
    if (c->kc < c->g.k || c->scap < c->g.N) return RAD_E_SHAPE;
    c->ccap = co ? tiles_cap(co, c->hd, c->nh) : 0;
    if (co && (co->shape[co->rank - 1] != c->hd || c->ccap < c->g.N)) return RAD_E_SHAPE;
    return RAD_OK;
}

/* ================================================================== attn_vsa's blocks */
/* Keys per packed block and query rows per block. */
enum { KT = 64, QB = 64 };
#if AVX_LEVEL == 3
constexpr int RB = 6;
#else
constexpr int RB = 2;
#endif
constexpr int64_t CK = 4 * VF_N;

inline int64_t padded(int64_t hd) { return (hd + CK - 1) / CK * CK; }

/* The next row block: `top` rows, the last two kept at four or more (avx_gen_attn.cpp). */
inline int row_block(int64_t left, int top) {
    if (top < 4) return left >= top ? top : (int)left;
    if (left >= top + 4 || left == top) return top;
    return left > top ? (int)(left - 4) : (int)left;
}

#define VSA_ROWS(fn, mb, ...)                             \
    switch (mb) {                                         \
        case 6: fn<6>(__VA_ARGS__); break;                \
        case 5: fn<5>(__VA_ARGS__); break;                \
        case 4: fn<4>(__VA_ARGS__); break;                \
        case 3: fn<3>(__VA_ARGS__); break;                \
        case 2: fn<2>(__VA_ARGS__); break;                \
        default: fn<1>(__VA_ARGS__); break;               \
    }

/* S[i][0, 64) = Q[i] . K for MB_ query rows (pitch ldq) over the packed Kt. */
template <int MB_>
inline void qk_rows(const float* Q, int64_t ldq, const float* Kt, int64_t hd, float* S) {
    for (int64_t c = 0; c < KT; c += CK) {
        vf acc[MB_][4];
        for (int i = 0; i < MB_; ++i)
            for (int v = 0; v < 4; ++v) acc[i][v] = vf_zero();
        for (int64_t d = 0; d < hd; ++d) {
            const float* kr = Kt + d * KT + c;
            const vf k0 = vf_loadu(kr), k1 = vf_loadu(kr + VF_N);
            const vf k2 = vf_loadu(kr + 2 * VF_N), k3 = vf_loadu(kr + 3 * VF_N);
            for (int i = 0; i < MB_; ++i) {
                const vf b = vf_set1(Q[i * ldq + d]);
                acc[i][0] = vf_fma(b, k0, acc[i][0]);
                acc[i][1] = vf_fma(b, k1, acc[i][1]);
                acc[i][2] = vf_fma(b, k2, acc[i][2]);
                acc[i][3] = vf_fma(b, k3, acc[i][3]);
            }
        }
        for (int i = 0; i < MB_; ++i)
            for (int v = 0; v < 4; ++v) vf_storeu(S + i * KT + c + v * VF_N, acc[i][v]);
    }
}

/* acc[i][0, hp) += P[i] . V for MB_ rows over the block's `kt` keys. */
template <int MB_>
inline void pv_rows(float* acc, int64_t hp, const float* Pm, const float* Vt, int64_t kt) {
    for (int64_t c = 0; c < hp; c += CK) {
        vf o[MB_][4];
        for (int i = 0; i < MB_; ++i)
            for (int v = 0; v < 4; ++v) o[i][v] = vf_loadu(acc + i * hp + c + v * VF_N);
        for (int64_t j = 0; j < kt; ++j) {
            const float* vr = Vt + j * hp + c;
            const vf v0 = vf_loadu(vr), v1 = vf_loadu(vr + VF_N);
            const vf v2 = vf_loadu(vr + 2 * VF_N), v3 = vf_loadu(vr + 3 * VF_N);
            for (int i = 0; i < MB_; ++i) {
                const vf p = vf_set1(Pm[i * KT + j]);
                o[i][0] = vf_fma(p, v0, o[i][0]);
                o[i][1] = vf_fma(p, v1, o[i][1]);
                o[i][2] = vf_fma(p, v2, o[i][2]);
                o[i][3] = vf_fma(p, v3, o[i][3]);
            }
        }
        for (int i = 0; i < MB_; ++i)
            for (int v = 0; v < 4; ++v) vf_storeu(acc + i * hp + c + v * VF_N, o[i][v]);
    }
}

/* The online softmax over one block of R rows: m, l and acc (rows of `hp`) rescaled at a new
 * maximum, S replaced by exp(S - m). A row with nothing visible in the block becomes zeros. */
void softmax_block(float* S, int64_t R, float* m, float* l, float* acc, int64_t hp) {
    for (int64_t r = 0; r < R; ++r) {
        float* sr = S + r * KT;
        const float tmax = row_max(sr, KT);
        if (tmax == -INFINITY) {
            for (int64_t j = 0; j < KT; ++j) sr[j] = 0.0f;
            continue;
        }
        if (tmax > m[r]) {
            const float alpha = m[r] == -INFINITY ? 0.0f : sc_exp(m[r] - tmax);
            if (alpha != 1.0f) {
                float* ar = acc + r * hp;
                const vf va = vf_set1(alpha);
                for (int64_t d = 0; d < hp; d += VF_N)
                    vf_storeu(ar + d, vf_mul(vf_loadu(ar + d), va));
                l[r] *= alpha;
            }
            m[r] = tmax;
        }
        const vf vm = vf_set1(m[r]);
        for (int64_t j = 0; j < KT; j += VF_N)
            vf_storeu(sr + j, vf_exp(vf_sub(vf_loadu(sr + j), vm)));
        l[r] += row_sum(sr, KT);
    }
}

struct Att {
    const Call* c;
    const RadTensor *q, *k, *v, *sel, *co, *gt, *o, *k8, *ks;
    Heads Q, K, Vh, O, G, K8;
    int64_t ks_rs;
    float scale;
    bool i8;                                   /* the int8 score form (libref's ref_attn_vsa) */
};

/* quant_act_i8g's rule for one row, in place: the codes rint(clamp(x * (1/scale), +-127)) and the
 * scale amax/127 (1 for an all-zero row). */
float q8_row(float* x, int64_t n) {
    float am = 0.0f;
    for (int64_t d = 0; d < n; ++d) am = std::max(am, std::fabs(x[d]));
    const float sc = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
    const float inv = 1.0f / sc;
    for (int64_t d = 0; d < n; ++d) x[d] = std::rint(std::min(std::max(x[d] * inv, -127.0f), 127.0f));
    return sc;
}

/* Query tile i of head h. */
void attn_task(const Att& t, int64_t h, int64_t i) {
    const Tiles& g = t.c->g;
    const int64_t hd = t.c->hd, hp = padded(hd), P = g.P, kk = g.k;
    float* frame = scratch_f32((size_t)(QB * hd + QB * hp + QB * KT + 2 * QB + hd * KT + KT * hp +
                                        4 * hp + QB + KT) + 64);
    float* Qb = frame;
    float* acc = Qb + QB * hd;
    float* S = acc + QB * hp;
    float* m = S + QB * KT;
    float* l = m + QB;
    float* Kt = l + QB;
    float* Vt = Kt + hd * KT;
    float* row = Vt + KT * hp;                 /* 4 hp: a widened row, the epilogue's rows */
    float* qsc = row + 4 * hp;                 /* the int8 form's query scales, then key scales */
    float* ksc = qsc + QB;
    const int64_t sb = rowoff(t.sel, h * t.c->scap + i, t.c->kc), ss = laststride(t.sel);
    const int64_t nvis = i < P ? g.N : P + kk;
    auto vis = [&](int64_t n) {
        return n < P || i < P ? n : ldi(t.sel, sb + (n - P) * ss);
    };
    const int64_t sz = tile_size(g, i);
    for (int64_t q0 = 0; q0 < sz; q0 += QB) {
        const int64_t nq = std::min<int64_t>(QB, sz - q0);
        for (int64_t r = 0; r < nq; ++r) {
            float* qr = Qb + r * hd;
            const float* src = in_row(t.q, row_at(g, i, q0 + r) * t.Q.rs + h * t.Q.hs, hd, qr);
            if (src != qr) std::memcpy(qr, src, sizeof(float) * (size_t)hd);
            if (t.i8) qsc[r] = q8_row(qr, hd);
            m[r] = -INFINITY;
            l[r] = 0.0f;
        }
        std::memset(acc, 0, sizeof(float) * (size_t)(nq * hp));
        for (int64_t n = 0; n < nvis; ++n) {
            const int64_t j = vis(n), kz = tile_size(g, j);
            for (int64_t c0 = 0; c0 < kz; c0 += KT) {
                const int64_t kt = std::min<int64_t>(KT, kz - c0);
                for (int64_t jj = 0; jj < KT; ++jj) {
                    float* vr = Vt + jj * hp;
                    if (jj >= kt) {
                        for (int64_t d = 0; d < hd; ++d) Kt[d * KT + jj] = 0.0f;
                        std::memset(vr, 0, sizeof(float) * (size_t)hp);
                        continue;
                    }
                    const int64_t kr = row_at(g, j, c0 + jj);
                    const float* kx = t.i8 ? in_row(t.k8, kr * t.K8.rs + h * t.K8.hs, hd, row)
                                           : in_row(t.k, kr * t.K.rs + h * t.K.hs, hd, row);
                    for (int64_t d = 0; d < hd; ++d) Kt[d * KT + jj] = kx[d];
                    if (t.i8) ksc[jj] = ldt(t.ks, kr * t.ks_rs + h);
                    const float* vx = in_row(t.v, kr * t.Vh.rs + h * t.Vh.hs, hd, vr);
                    if (vx != vr) std::memcpy(vr, vx, sizeof(float) * (size_t)hd);
                    for (int64_t d = hd; d < hp; ++d) vr[d] = 0.0f;
                }
                for (int64_t r = 0, mb; r < nq; r += mb) {
                    mb = row_block(nq - r, RB);
                    VSA_ROWS(qk_rows, mb, Qb + r * hd, hd, Kt, hd, S + r * KT);
                }
                const vf vs = vf_set1(t.scale);
                for (int64_t r = 0; r < nq; ++r) {
                    float* sr = S + r * KT;
                    if (t.i8) {
                        /* libref's order: (dot * q scale) * k scale, then the softmax scale. */
                        for (int64_t j2 = kt; j2 < KT; ++j2) ksc[j2] = 0.0f;
                        const vf vq = vf_set1(qsc[r]);
                        for (int64_t j2 = 0; j2 < KT; j2 += VF_N)
                            vf_storeu(sr + j2, keep(vf_mul(keep(vf_mul(keep(vf_mul(
                                                    vf_loadu(sr + j2), vq)), vf_loadu(ksc + j2))),
                                                vs)));
                    } else {
                        for (int64_t j2 = 0; j2 < KT; j2 += VF_N)
                            vf_storeu(sr + j2, vf_mul(vf_loadu(sr + j2), vs));
                    }
                    for (int64_t j2 = kt; j2 < KT; ++j2) sr[j2] = -INFINITY;
                }
                softmax_block(S, nq, m, l, acc, hp);
                for (int64_t r = 0, mb; r < nq; r += mb) {
                    mb = row_block(nq - r, RB);
                    VSA_ROWS(pv_rows, mb, acc + r * hp, hp, S + r * KT, Vt, kt);
                }
            }
        }
        /* o = acc / l; with the coarse term o rounded to out's dtype, then + coarse gate, the
         * product rounded before the add as libref rounds it. */
        const uint32_t odt = t.o->dtype;
        float* tmp = row + hp;
        float* cb = row + 2 * hp;
        float* gb = row + 3 * hp;
        const float* cr =
            t.co ? in_row(t.co, rowoff(t.co, h * t.c->ccap + i, hd), hd, cb) : nullptr;
        for (int64_t r = 0; r < nq; ++r) {
            const int64_t qr = row_at(g, i, q0 + r);
            float* y = acc + r * hp;
            const float inv = (m[r] == -INFINITY || l[r] <= 0.0f) ? 0.0f : 1.0f / l[r];
            const vf vi = vf_set1(inv);
            int64_t d = 0;
            for (; d + VF_N <= hd; d += VF_N) vf_storeu(y + d, vf_mul(vf_loadu(y + d), vi));
            for (; d < hd; ++d) y[d] *= inv;
            if (cr) {
                if (odt != RAD_F32) {
                    row_from_f32(odt, y, tmp, hd);
                    row_to_f32(odt, tmp, y, hd);
                }
                const float* ge = in_row(t.gt, qr * t.G.rs + h * t.G.hs, hd, gb);
                d = 0;
                for (; d + VF_N <= hd; d += VF_N)
                    vf_storeu(y + d, vf_add(vf_loadu(y + d),
                                            keep(vf_mul(vf_loadu(cr + d), vf_loadu(ge + d)))));
                for (; d < hd; ++d) y[d] = y[d] + keep(cr[d] * ge[d]);
            }
            out_row(t.o, qr * t.O.rs + h * t.O.hs, hd, y);
        }
    }
}

/* The fp8 P.V form of query tile i of head h (libref's ref_attn_vsa): per row every visible score
 * (bf16 or int8 form), the max, then p = exp(s - max) in f64 rounded to f32, w = e4m3(448 p), and
 * o = (sum w v) v_scale / (448 sum p), sums in key order; then the epilogue. Two passes, as libref,
 * because the weights are rounded against the row's final max. */
void attn_task_pv8(const Att& t, const RadTensor* vsc, int64_t h, int64_t i) {
    const Tiles& g = t.c->g;
    const int64_t hd = t.c->hd, P = g.P, kk = g.k;
    const int64_t sb = rowoff(t.sel, h * t.c->scap + i, t.c->kc), ss = laststride(t.sel);
    const int64_t nvis = i < P ? g.N : P + kk;
    auto vis = [&](int64_t n) {
        return n < P || i < P ? n : ldi(t.sel, sb + (n - P) * ss);
    };
    int64_t nk = 0;
    for (int64_t n = 0; n < nvis; ++n) nk += tile_size(g, vis(n));
    float* qf = scratch_f32((size_t)(4 * hd + nk));
    float* kf = qf + hd;
    float* acc = kf + hd;
    float* y = acc + hd;
    float* sc = y + hd;
    const uint32_t odt = t.o->dtype;
    const int64_t sz = tile_size(g, i);
    for (int64_t q0 = 0; q0 < sz; ++q0) {
        const int64_t qr = row_at(g, i, q0);
        const float* src = in_row(t.q, qr * t.Q.rs + h * t.Q.hs, hd, qf);
        if (src != qf) std::memcpy(qf, src, sizeof(float) * (size_t)hd);
        const float qs = t.i8 ? q8_row(qf, hd) : 1.0f;
        int64_t m = 0;
        float mx = -INFINITY;
        for (int64_t n = 0; n < nvis; ++n) {
            const int64_t j = vis(n), kz = tile_size(g, j);
            for (int64_t c0 = 0; c0 < kz; ++c0) {
                const int64_t kr = row_at(g, j, c0);
                const float* kx = t.i8 ? in_row(t.k8, kr * t.K8.rs + h * t.K8.hs, hd, kf)
                                       : in_row(t.k, kr * t.K.rs + h * t.K.hs, hd, kf);
                float dot = 0.0f;
                for (int64_t d = 0; d < hd; ++d) dot = keep(dot + keep(qf[d] * kx[d]));
                if (t.i8) dot = keep(keep(dot * qs) * ldt(t.ks, kr * t.ks_rs + h));
                dot = keep(dot * t.scale);
                sc[m++] = dot;
                mx = std::max(mx, dot);
            }
        }
        for (int64_t d = 0; d < hd; ++d) acc[d] = 0.0f;
        float sum = 0.0f;
        m = 0;
        for (int64_t n = 0; n < nvis; ++n) {
            const int64_t j = vis(n), kz = tile_size(g, j);
            for (int64_t c0 = 0; c0 < kz; ++c0) {
                const int64_t kr = row_at(g, j, c0);
                const float p = (float)std::exp((double)(sc[m++] - mx));
                sum += p;
                const float w = narrow1(RAD_F8E4M3, 448.0f * p);
                const float* vx = in_row(t.v, kr * t.Vh.rs + h * t.Vh.hs, hd, kf);
                for (int64_t d = 0; d < hd; ++d) acc[d] = keep(acc[d] + keep(w * vx[d]));
            }
        }
        const float inv8 = sum > 0.0f ? 1.0f / (448.0f * sum) : 0.0f;
        for (int64_t d = 0; d < hd; ++d)
            y[d] = keep(keep(acc[d] * ldt(vsc, offlin(vsc, h * hd + d))) * inv8);
        if (t.co) {
            const int64_t cb = rowoff(t.co, h * t.c->ccap + i, hd), cs = laststride(t.co);
            for (int64_t d = 0; d < hd; ++d) {
                const float c = ldt(t.co, cb + d * cs);
                const float e = ldt(t.gt, qr * t.G.rs + h * t.G.hs + d * t.G.ds);
                y[d] = keep(narrow1(odt, y[d]) + keep(c * e));
            }
        }
        out_row(t.o, qr * t.O.rs + h * t.O.hs, hd, y);
    }
}

}  /* namespace */

/* ------------------------------------------------------------------ vsa_select */
AVX_KERNEL(vsa_select) {
    const RadTensor *q = rad_arg_in(a, 0), *k = rad_arg_in(a, 1), *v = rad_arg_in(a, 2);
    const RadTensor *segs = rad_arg_in(a, 3), *geom = rad_arg_in(a, 4), *sel = rad_arg_in(a, 5);
    const RadTensor* co = rad_arg_in(a, 6);
    if (!q || !k || !segs || !geom || !sel) return RAD_E_INVAL;
    if (!v != !co) return RAD_E_SHAPE;
    if (sel->dtype != RAD_I32) return RAD_E_DTYPE;
    Call c;
    if (const int rc = common(a, q, segs, geom, &c); rc < 0) return rc;
    const Tiles& g = c.g;
    const int64_t hd = c.hd, nh = c.nh, need = g.R * nh * hd;
    if (numel(k) < need || (v && numel(v) < need)) return RAD_E_SHAPE;
    if (const int rc = caps(sel, co, &c); rc < 0) return rc;
    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)hd));
    const Heads Q = heads_of(q, nh, hd), K = heads_of(k, nh, hd);
    const Heads Vh = v ? heads_of(v, nh, hd) : Heads{ nullptr, 0, 0, 0 };
    const int64_t N = g.N, P = g.P, V = g.V, kk = g.k;
    /* The scores run in whole vectors across tiles, through zero key columns up to Np: from tile
     * 0 for the coarse branch, else from the last multiple of the width at or below P. */
    const int64_t Np = (N + VF_N - 1) / VF_N * VF_N;
    const int64_t j0 = co ? 0 : P / VF_N * VF_N;

    /* The tile means [n_head][N][hd], and the key means transposed, [n_head][hd][Np]. Kept at
     * their high-water size; the parallel regions read them through pointers taken here. */
    static thread_local std::vector<float> pool;
    const size_t per = (size_t)(nh * N * hd);
    pool.resize(per * (co ? 3 : 2) + (size_t)(nh * hd * Np));
    float* qp = pool.data();
    float* kp = qp + per;
    float* vp = co ? kp + per : nullptr;
    float* kt = kp + per * (co ? 2 : 1);

    AVX_PARALLEL_FOR_IF(nh * g.R * hd >= AVX_PAR_MIN)
    for (int64_t u = 0; u < nh * N; ++u) {
        const int64_t h = u / N, j = u % N;
        float* scr = scratch_f32((size_t)(hd + 16));
        float* means[3] = { qp + u * hd, kp + u * hd, vp ? vp + u * hd : nullptr };
        const RadTensor* src[3] = { q, k, v };
        const Heads* hv[3] = { &Q, &K, &Vh };
        for (int s = 0; s < 3; ++s) {
            float* mu = means[s];
            if (!mu) continue;
            std::memset(mu, 0, sizeof(float) * (size_t)hd);
            tile_rows(g, j, [&](int64_t r) {
                const float* x = in_row(src[s], r * hv[s]->rs + h * hv[s]->hs, hd, scr);
                int64_t d = 0;
                for (; d + VF_N <= hd; d += VF_N)
                    vf_storeu(mu + d, vf_add(vf_loadu(mu + d), vf_loadu(x + d)));
                for (; d < hd; ++d) mu[d] += x[d];
            });
            const float n = (float)tile_size(g, j);
            const vf vn = vf_set1(n);
            int64_t d = 0;
            for (; d + VF_N <= hd; d += VF_N) vf_storeu(mu + d, vf_div(vf_loadu(mu + d), vn));
            for (; d < hd; ++d) mu[d] /= n;
        }
    }
    AVX_PARALLEL_FOR_IF(nh * Np * hd >= AVX_PAR_MIN)
    for (int64_t hdd = 0; hdd < nh * hd; ++hdd) {
        const int64_t h = hdd / hd, d = hdd % hd;
        float* kr = kt + hdd * Np;
        for (int64_t j = 0; j < N; ++j) kr[j] = kp[(h * N + j) * hd + d];
        for (int64_t j = N; j < Np; ++j) kr[j] = 0.0f;
    }

    const int64_t kc = c.kc, scap = c.scap, ccap = c.ccap, ss = laststride(sel);
    AVX_PARALLEL_FOR_IF(nh * N * Np * hd >= AVX_PAR_MIN)
    for (int64_t u = 0; u < nh * N; ++u) {
        const int64_t h = u / N, i = u % N;
        if (i < P && !co) continue;
        float* frame = scratch_f32((size_t)(Np + V + hd) + 64);
        float* S = frame;
        float* ord = S + Np;
        float* acc = ord + V;
        const float* qs = qp + u * hd;
        const float* kth = kt + h * hd * Np;
        for (int64_t j = j0; j < Np; j += VF_N) vf_storeu(S + j, vf_zero());
        for (int64_t d = 0; d < hd; ++d) {
            const vf b = vf_set1(qs[d]);
            const float* kr = kth + d * Np;
            for (int64_t j = j0; j < Np; j += VF_N)
                vf_storeu(S + j, vf_add(vf_loadu(S + j), keep(vf_mul(b, vf_loadu(kr + j)))));
        }
        const vf vs = vf_set1(scale);
        for (int64_t j = j0; j < Np; j += VF_N) vf_storeu(S + j, vf_mul(vf_loadu(S + j), vs));
        if (i >= P) {
            std::copy(S + P, S + N, ord);
            std::nth_element(ord, ord + (kk - 1), ord + V, above);
            const float thr = ord[kk - 1];
            int64_t ties = kk;
            for (int64_t j = 0; j < V; ++j) ties -= above(S[P + j], thr);
            const int64_t sb = rowoff(sel, h * scap + i, kc);
            int64_t n = 0;
            for (int64_t j = 0; j < V; ++j) {
                const float s = S[P + j];
                if (above(s, thr) || (!above(thr, s) && ties-- > 0))
                    st_int_dt(sel->dtype, sel->data, sb + (n++) * ss, P + j);
            }
        }
        if (co) {
            float mx = -INFINITY;
            for (int64_t j = 0; j < N; ++j) if (S[j] > mx) mx = S[j];
            const vf vm = vf_set1(mx);
            int64_t j = 0;
            for (; j + VF_N <= N; j += VF_N) vf_storeu(S + j, vf_exp(vf_sub(vf_loadu(S + j), vm)));
            for (; j < N; ++j) S[j] = sc_exp(S[j] - mx);
            const float sum = row_sum(S, N);
            std::memset(acc, 0, sizeof(float) * (size_t)hd);
            for (j = 0; j < N; ++j) axpy(acc, vp + (h * N + j) * hd, S[j] / sum, hd);
            out_row(co, rowoff(co, h * ccap + i, hd), hd, acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_vsa */
AVX_KERNEL(attn_vsa) {
    const RadTensor *q = rad_arg_in(a, 0), *k = rad_arg_in(a, 1), *v = rad_arg_in(a, 2);
    const RadTensor *segs = rad_arg_in(a, 3), *geom = rad_arg_in(a, 4), *sel = rad_arg_in(a, 5);
    const RadTensor *co = rad_arg_in(a, 6), *gt = rad_arg_in(a, 7), *o = rad_arg_in(a, 8);
    const RadTensor *k8 = rad_arg_in(a, 9), *ks = rad_arg_in(a, 10);
    const RadTensor* vsc = a->n_t > 11 ? rad_arg_in(a, 11) : nullptr;
    if (!q || !k || !v || !segs || !geom || !sel || !o) return RAD_E_INVAL;
    const char* pv = p_str(a, "pv", "bf16");
    const bool pv8 = std::strcmp(pv, "fp8") == 0;
    if ((!pv8 && std::strcmp(pv, "bf16") != 0) || pv8 != (vsc != nullptr)) return RAD_E_INVAL;
    if (!co != !gt) return RAD_E_SHAPE;
    if (sel->dtype != RAD_I32) return RAD_E_DTYPE;
    const char* qk = p_str(a, "qk", "bf16");
    const bool i8 = std::strcmp(qk, "i8") == 0;
    if ((!i8 && std::strcmp(qk, "bf16") != 0) || i8 != (k8 != nullptr) || i8 != (ks != nullptr))
        return RAD_E_INVAL;
    Call c;
    if (const int rc = common(a, q, segs, geom, &c); rc < 0) return rc;
    const Tiles& g = c.g;
    const int64_t hd = c.hd, nh = c.nh, need = g.R * nh * hd;
    if (numel(k) < need || numel(v) < need || numel(o) < need || (gt && numel(gt) < need))
        return RAD_E_SHAPE;
    if (const int rc = caps(sel, co, &c); rc < 0) return rc;
    const int64_t N = g.N, P = g.P, kk = g.k, ss = laststride(sel);
    for (int64_t h = 0; h < nh; ++h)
        for (int64_t i = P; i < N; ++i) {
            const int64_t sb = rowoff(sel, h * c.scap + i, c.kc);
            int64_t prev = P - 1;
            for (int64_t s = 0; s < kk; ++s) {
                const int64_t j = ldi(sel, sb + s * ss);
                if (j <= prev || j >= N) return RAD_E_INVAL;
                prev = j;
            }
        }
    Att t{};
    t.c = &c;
    t.q = q; t.k = k; t.v = v; t.sel = sel; t.co = co; t.gt = gt; t.o = o;
    t.Q = heads_of(q, nh, hd);
    t.K = heads_of(k, nh, hd);
    t.Vh = heads_of(v, nh, hd);
    t.O = heads_of(o, nh, hd);
    t.G = gt ? heads_of(gt, nh, hd) : Heads{ nullptr, 0, 0, 0 };
    t.scale = p_f32(a, "scale", 1.0f / std::sqrt((float)hd));
    t.i8 = i8;
    if (i8) {
        if (numel(k8) < need || numel(ks) < g.R * nh) return RAD_E_SHAPE;
        t.k8 = k8;
        t.ks = ks;
        t.K8 = heads_of(k8, nh, hd);
        t.ks_rs = row_stride(ks, nh);
    }
    /* Dynamic: a prefix query tile sees every tile, a video one the prefix and k tiles, and a
     * unit's arithmetic does not depend on the thread that runs it. */
#ifdef _OPENMP
    AVX_PRAGMA(omp parallel for schedule(dynamic, 1) if(nh * g.R * hd >= AVX_PAR_MIN ||
               avx_par_forced()) num_threads(avx_team_size()))
#endif
    for (int64_t u = 0; u < nh * N; ++u) {
        if (pv8) attn_task_pv8(t, vsc, u / N, u % N);
        else     attn_task(t, u / N, u % N);
    }
    return RAD_OK;
}
