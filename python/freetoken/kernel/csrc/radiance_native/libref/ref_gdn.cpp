/* ref_gdn.cpp -- the gated delta net: the conv preamble, the chunk preamble, the chunked scan, the
 * recurrent update and the gated norm.
 *
 * libr4d's algorithm without the fusion: the state lives in a per-thread workspace instead of WMMA
 * accumulators, every intermediate is written out, and the gram is recomputed per v head instead
 * of once per k head. The arithmetic is the same.
 *
 * ============================== The algorithm ==============================
 *
 * Per token, per v head, with q and k l2-normalised and the state S shaped [head_v, head_k]:
 *
 *     g    = -exp(A_log) . softplus(a + dt_bias)        beta = sigmoid(b)
 *     S   *= e^g
 *     u    = beta . (v - S k)          S += u k^T          o = S q
 *
 * Decode runs this recurrent form. Prefill runs the chunked form: the same recurrence blocked over
 * `chunk` tokens, with intra-chunk dependencies solved by one triangular inverse (`gdn_kkt_solve`):
 *
 *     ge[j] = e^{g_j}   gs[j] = e^{g_last - g_j}     (g is the intra-chunk cumsum of the gate)
 *     U  = K S^T        W  = beta . (V - ge . U)     V' = A W
 *     P  = Q K^T        sA = scale . e^{g_i - g_j} . tril(P)
 *     O  = scale . ge . (Q S^T) + sA V'
 *     S  = e^{g_last} S + V'^T (gs . K)
 *
 * Invariant: every decay factor formed is <= 1. Intra-chunk gate spans on real prefill inputs reach
 * several hundred nats, so splitting the exponent about any reference c (e^{g_i - c}, e^{c - g_j})
 * overflows fp32's e^88 on one side and underflows on the other. The pairs e^{g_i - g_j} (j <= i)
 * and e^{g_last - g_j} stay in (0, 1].
 *
 * ============================== The layouts ==============================
 *
 *     q, k     [T, head_k_count, head_k]        v, o     [T, head_v_count, head_v]
 *     g, beta  [T, head_v_count]                A        [T, head_v_count, chunk]
 *     h0, ht   [N, head_v_count, head_v, head_k]         cu  [N+1]
 *     conv w   [conv_dim, conv_width]           conv b   [conv_dim]
 *     conv_state [n_slots, conv_dim, state_len]
 *     state      [n_slots, head_v_count, head_v, head_k]
 *
 * with conv_dim = 2 * head_k_count * head_k + head_v_count * head_v, and `x` the fused projection
 * row [q | k | v | a | b]: conv_dim convolved channels, then one `a` and one `b` per v head.
 * gdn_conv_prep's `a` and `b_gate` operands are optional: present, they are separate gate and beta
 * projections; absent, they are the last 2 * head_v_count columns of `x`. gdn_recurrent_update
 * always takes them separately (strided views of a fused row are fine).
 *
 * The speculative rollback contract for the two state caches is in ref_ops.h.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>

using namespace ref;

namespace {

/* The deepest convolution history held in a stack frame. libr4d's width is 4; a wider request
 * returns RAD_E_UNSUPPORTED rather than convolving a truncated window. */
enum { REF_CONV_HIST_MAX = 16 };

struct V3 { const RadTensor* t; int64_t s0, s1, s2; };
struct V2 { const RadTensor* t; int64_t s0, s1; };

static inline V3 v3(const RadTensor* t, int64_t d1, int64_t d2) {
    V3 v{ t, d1 * d2, d2, 1 };
    if (t->rank >= 3) {
        v.s0 = t->stride[t->rank - 3];
        v.s1 = t->stride[t->rank - 2];
        v.s2 = t->stride[t->rank - 1];
    }
    return v;
}
/* v3() for an operand that may be a column view of a wider row (the fp8 delta net's output gate is
 * the tail of a fused [q|k|v|z] projection). A rank-2 view carries its token stride in stride[0],
 * where v3() would assume d1*d2. */
static inline V3 v3_rows(const RadTensor* t, int64_t d1, int64_t d2) {
    if (t->rank != 2) return v3(t, d1, d2);
    return V3{ t, t->stride[0], d2 * t->stride[1], t->stride[1] };
}
static inline V2 v2(const RadTensor* t, int64_t d1) {
    V2 v{ t, d1, 1 };
    if (t->rank >= 2) { v.s0 = t->stride[t->rank - 2]; v.s1 = t->stride[t->rank - 1]; }
    return v;
}
static inline int64_t at3(const V3& v, int64_t i, int64_t j, int64_t k) {
    return i * v.s0 + j * v.s1 + k * v.s2;
}
static inline int64_t at2(const V2& v, int64_t i, int64_t j) { return i * v.s0 + j * v.s1; }

/* [N, heads, head_v, head_k], the shape of the chunked scan's h0/ht and the paged recurrent state.
 * Strides are read off the tensor, not derived: a paged state cache is padded to match the
 * attention page, so a derived stride would misread every slot after the first. */
struct V4 { const RadTensor* t; int64_t s0, s1, s2, s3; };
static inline V4 v4(const RadTensor* t, int64_t d1, int64_t d2, int64_t d3) {
    V4 v{ t, d1 * d2 * d3, d2 * d3, d3, 1 };
    if (t->rank >= 4) {
        v.s0 = t->stride[t->rank - 4];
        v.s1 = t->stride[t->rank - 3];
        v.s2 = t->stride[t->rank - 2];
        v.s3 = t->stride[t->rank - 1];
    }
    return v;
}
static inline int64_t at4(const V4& v, int64_t i, int64_t j, int64_t k, int64_t l) {
    return i * v.s0 + j * v.s1 + k * v.s2 + l * v.s3;
}

/* The conv state is [n_slots, conv_dim, state_len]; a flattened one is read as the tight layout. */
static inline V3 conv_state_view(const RadTensor* cs, int64_t conv_dim, int64_t width,
                                 int64_t* state_len) {
    *state_len = cs->rank >= 3 ? cs->shape[cs->rank - 1] : (width > 1 ? width - 1 : 1);
    return v3(cs, conv_dim, *state_len);
}

/* What gdn_conv_prep and gdn_conv_update share: the fused row `x`, the conv weight and bias, and
 * the q, k and v outputs the channels split into. */
struct ConvIO {
    const RadTensor *x, *w, *bias, *q, *k, *v;
    int64_t xrow, xcol, Hg, H, K, V, width;
};

/* Channel `c` of the depthwise causal convolution with silu over one sequence's `slen` tokens
 * from row `bos`, `hist` holding its width - 1 entries of history. Channels are grouped in whole
 * heads, so a group never straddles the q/k/v boundaries at any tensor-parallel width. */
static void conv_channel(const ConvIO& io, int64_t c, int64_t bos, int64_t slen, const float* hist) {
    const int64_t Hg = io.Hg, K = io.K, V = io.V, width = io.width;
    const RadTensor* out;
    V3 ov;
    int64_t oh, od;
    if (c < Hg * K) {
        out = io.q; ov = v3(out, Hg, K); oh = c / K; od = c % K;
    } else if (c < 2 * Hg * K) {
        out = io.k; ov = v3(out, Hg, K); oh = (c - Hg * K) / K; od = (c - Hg * K) % K;
    } else {
        out = io.v; ov = v3(out, io.H, V); oh = (c - 2 * Hg * K) / V; od = (c - 2 * Hg * K) % V;
    }
    const V2 wv = v2(io.w, width);
    const float bs = io.bias ? ldt(io.bias, c * laststride(io.bias)) : 0.0f;
    for (int64_t t = 0; t < slen; ++t) {
        float acc = bs;
        for (int64_t j = 0; j < width; ++j) {
            const int64_t src = t - (width - 1) + j;
            const float val = src >= 0 ? ldt(io.x, (bos + src) * io.xrow + c * io.xcol)
                                       : hist[src + width - 1];
            acc += val * ldt(io.w, at2(wv, c, j));
        }
        stt(out, at3(ov, bos + t, oh, od), act_silu(acc));
    }
}

}  /* namespace */

/* ------------------------------------------------------------------ gdn_conv_prep */
/* Everything between the qkv projection and the chunked scan: the width-`conv_width` depthwise
 * causal convolution with silu and its state cache, the q/k/v split, the l2 norm of each q and k
 * head, the gate with its per-chunk cumsum, and beta.
 *
 * The conv output is rounded before the norm, as the unfused pair does (the conv writes bf16, the
 * prep reads it back); the reference writes it to its output tensor and reads it back.
 *
 * The l2 norm is rsqrt(sum(x^2) + l2_eps) with no division by head width (a norm, not an RMS);
 * l2_eps defaults to 1e-6, FLA's and libr4d's L2NORM_EPS. */
int ref_gdn_conv_prep(const RadArgs* a, RadStream) {
    if (a->n_t < 16) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *cs = &a->t[5], *cu = &a->t[6];
    const RadTensor *q = &a->t[11], *k = &a->t[12], *v = &a->t[13];
    const RadTensor *g = &a->t[14], *beta = &a->t[15];
    const RadTensor* bias = t_in(a, 2);
    const RadTensor* A_log = t_in(a, 3);
    const RadTensor* dt_bias = t_in(a, 4);
    /* Separate gate and beta projections; absent, they are the last 2*H columns of `x`. */
    const RadTensor* ta = t_in(a, 7);
    const RadTensor* tb = t_in(a, 8);
    const RadTensor* sidx = t_in(a, 9);
    const RadTensor* hini = t_in(a, 10);
    if (!x->data || !w->data || !cu->data || !q->data || !k->data || !v->data ||
        !g->data || !beta->data) return RAD_E_INVAL;
    /* Both or neither, or one read would fall back into a fused row that may not exist. */
    if ((ta != nullptr) != (tb != nullptr)) return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    /* `chunk` is RAD_DERIVED and no operand's shape records it, yet the gate cumsum is cut at the
     * chunk boundary. A call that omits it gets RAD_E_SHAPE rather than a cumsum over the wrong
     * span; the architecture plugins pin it. */
    const int64_t chunk = p_int(a, "chunk", 0), width = p_int(a, "conv_width", 0);
    const float l2eps = p_f32(a, "l2_eps", 1e-6f);
    const float spthr = p_f32(a, "softplus_thr", 20.0f);
    if (K <= 0 || V <= 0 || chunk <= 0 || width <= 0) return RAD_E_SHAPE;
    if (width - 1 > REF_CONV_HIST_MAX) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_E_SHAPE;                  /* zero rows (docs/OPS.md, Conventions) */
    const int64_t Hg = numel(q) / (T * K);
    const int64_t H  = numel(v) / (T * V);
    if (Hg <= 0 || H <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t conv_dim = 2 * Hg * K + H * V;
    if (numel(w) < conv_dim * width) return RAD_E_SHAPE;
    if (numel(g) < T * H || numel(beta) < T * H) return RAD_E_SHAPE;

    const int64_t xrow = x->rank >= 2 ? x->stride[x->rank - 2] : (conv_dim + (ta ? 0 : 2 * H));
    const int64_t xcol = laststride(x);
    /* Bound by the view, not the buffer: `x` may be a column slice of a wider row (the fp8 block
     * passes the left `conv_dim` of a fused [q|k|v|z] projection), whose numel undercounts the
     * span. Rank >= 2 is checked by shape; only a flattened operand is bounded by numel. */
    const int64_t xneed = conv_dim + (ta ? 0 : 2 * H);
    if (x->rank >= 2) {
        if (x->shape[x->rank - 1] < xneed || x->shape[x->rank - 2] < T) return RAD_E_SHAPE;
    } else if (numel(x) < (T - 1) * xrow + xneed) {
        return RAD_E_SHAPE;
    }
    if (ta && (numel(ta) < T * H || numel(tb) < T * H)) return RAD_E_SHAPE;

    int64_t state_len = 0;
    const V3 cvs = conv_state_view(cs, conv_dim, width, &state_len);
    if (cs->data && state_len < width - 1) return RAD_E_SHAPE;

    const V3 qv = v3(q, Hg, K), kv = v3(k, Hg, K);
    const V2 gv = v2(g, H), bv = v2(beta, H);
    const V2 av = ta ? v2(ta, H) : V2{ nullptr, 0, 0 };
    const V2 bgv = tb ? v2(tb, H) : V2{ nullptr, 0, 0 };
    const ConvIO io{ x, w, bias, q, k, v, xrow, xcol, Hg, H, K, V, width };

    /* ---- phase 1: the convolution, straight into q / k / v ---------------------------------- */
    REF_PAR_FOR(T * conv_dim * width)
    for (int64_t nc = 0; nc < N * conv_dim; ++nc) {
        const int64_t n = nc / conv_dim, c = nc % conv_dim;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, n)) : n;
        /* A zero `has_init` means the slot holds no history: the window before the first token is
         * zeros, not a recycled slot's previous contents. */
        const bool warm = !hini || ld_int(hini->dtype, hini->data, offlin(hini, n)) != 0;
        float hist[REF_CONV_HIST_MAX];
        for (int64_t i = 0; i < width - 1; ++i)
            hist[i] = warm && cs->data && slot >= 0 ? ldt(cs, at3(cvs, slot, c, i)) : 0.0f;
        conv_channel(io, c, bos, slen, hist);
    }

    /* ---- phase 2: the l2 norm over each q and k head, reading the rounded conv output -------- */
    REF_PAR_FOR(T * Hg * K * 2)
    for (int64_t th = 0; th < T * Hg * 2; ++th) {
        const int64_t which = th / (T * Hg);
        const int64_t rest = th % (T * Hg);
        const int64_t t = rest / Hg, h = rest % Hg;
        const RadTensor* z = which ? k : q;
        const V3& zv = which ? kv : qv;
        float ss = 0.0f;
        for (int64_t d = 0; d < K; ++d) { float u = ldt(z, at3(zv, t, h, d)); ss += u * u; }
        const float inv = 1.0f / std::sqrt(ss + l2eps);
        for (int64_t d = 0; d < K; ++d)
            stt(z, at3(zv, t, h, d), ldt(z, at3(zv, t, h, d)) * inv);
    }

    /* ---- phase 3: the gate, its per-chunk cumsum, and beta ----------------------------------- */
    /* The cumsum restarts per chunk and per sequence, as the scan consumes it: g[j] is the decay
     * from the start of j's chunk through j. */
    REF_PAR_FOR(T * H * 4)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        const float alog = A_log ? ldt(A_log, h * laststride(A_log)) : 0.0f;
        const float dtb  = dt_bias ? ldt(dt_bias, h * laststride(dt_bias)) : 0.0f;
        float run = 0.0f;
        for (int64_t t = 0; t < slen; ++t) {
            if (t % chunk == 0) run = 0.0f;
            const float av_ = ta ? ldt(ta, at2(av, bos + t, h))
                                 : ldt(x, (bos + t) * xrow + (conv_dim + h) * xcol);
            const float bvl = tb ? ldt(tb, at2(bgv, bos + t, h))
                                 : ldt(x, (bos + t) * xrow + (conv_dim + H + h) * xcol);
            run += -ref_exp(alog) * act_softplus(av_ + dtb, spthr);
            stt(g, at2(gv, bos + t, h), run);
            stt(beta, at2(bv, bos + t, h), act_sigmoid(bvl));
        }
    }

    /* ---- phase 4: rewrite the conv state from the sequence's last width-1 tokens -------------- */
    /* After phase 1, which has read the history this overwrites. A sequence shorter than width-1
     * keeps the old state in front of its tokens, so a later prefill chunk continues from the
     * earlier one. */
    if (cs->data && width > 1) {
        REF_PAR_FOR(N * conv_dim * width)
        for (int64_t nc = 0; nc < N * conv_dim; ++nc) {
            const int64_t n = nc / conv_dim, c = nc % conv_dim;
            const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
            const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, n)) : n;
            if (slen <= 0 || slot < 0) continue;
            const bool warm = !hini || ld_int(hini->dtype, hini->data, offlin(hini, n)) != 0;
            float keep[REF_CONV_HIST_MAX];
            const int64_t st = width - 1;
            for (int64_t i = 0; i < st; ++i) {
                const int64_t src = slen - (width - 1) + i;
                keep[i] = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                   : (warm ? ldt(cs, at3(cvs, slot, c, src + width - 1)) : 0.0f);
            }
            for (int64_t i = 0; i < st; ++i) stt(cs, at3(cvs, slot, c, i), keep[i]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_conv_update */
/* The same convolution for a decode step, against the rolling window (contract in ref_ops.h): the
 * read starts at num_accepted-1, and the rewrite shifts the buffer down by one. A rejection changes
 * the read offset instead of recomputing (spec §10).
 *
 * No l2 norm: on the decode path the recurrent update does it, as in libr4d. */
int ref_gdn_conv_update(const RadArgs* a, RadStream) {
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *cs = &a->t[3], *cu = &a->t[6];
    const RadTensor *q = &a->t[7], *k = &a->t[8], *v = &a->t[9];
    const RadTensor* bias = t_in(a, 2);
    const RadTensor* sidx = t_in(a, 4);
    const RadTensor* nacc = t_in(a, 5);
    if (!x->data || !w->data || !cs->data || !cu->data || !q->data || !k->data || !v->data)
        return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const int64_t width = p_int(a, "conv_width", 0);
    if (K <= 0 || V <= 0 || width <= 0) return RAD_E_SHAPE;
    if (width - 1 > REF_CONV_HIST_MAX) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_E_SHAPE;
    const int64_t Hg = numel(q) / (T * K);
    const int64_t H  = numel(v) / (T * V);
    if (Hg <= 0 || H <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t conv_dim = 2 * Hg * K + H * V;
    if (numel(w) < conv_dim * width) return RAD_E_SHAPE;

    const int64_t xrow = x->rank >= 2 ? x->stride[x->rank - 2] : conv_dim;
    const int64_t xcol = laststride(x);
    int64_t state_len = 0;
    const V3 cvs = conv_state_view(cs, conv_dim, width, &state_len);

    /* Check (not clamp) the read window and the last write slot against the state depth; a too
     * shallow cache is a caller bug that would otherwise silently truncate history. */
    for (int64_t n = 0; n < N; ++n) {
        const int64_t slen = cu_at(cu, n + 1) - cu_at(cu, n);
        const int64_t off = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, n)) - 1 : 0;
        if (off < 0) return RAD_E_INVAL;
        if (off + width - 1 > state_len) return RAD_E_SHAPE;
        if (width > 1 && (width - 2) + slen > state_len) return RAD_E_SHAPE;
    }

    const ConvIO io{ x, w, bias, q, k, v, xrow, xcol, Hg, H, K, V, width };

    REF_PAR_FOR(T * conv_dim * width)
    for (int64_t nc = 0; nc < N * conv_dim; ++nc) {
        const int64_t n = nc / conv_dim, c = nc % conv_dim;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, n)) : n;
        if (slot < 0) continue;
        const int64_t off = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, n)) - 1 : 0;

        float hist[REF_CONV_HIST_MAX];
        for (int64_t i = 0; i < width - 1; ++i) hist[i] = ldt(cs, at3(cvs, slot, c, off + i));
        conv_channel(io, c, bos, slen, hist);

        /* Shift down by one: the tail of the old history, then this step's tokens. Safe in place
         * because the history was read into `hist` first. */
        if (width > 1) {
            for (int64_t i = 0; i + 1 < width - 1; ++i)
                stt(cs, at3(cvs, slot, c, i), hist[i + 1]);
            for (int64_t t = 0; t < slen; ++t)
                stt(cs, at3(cvs, slot, c, t + width - 2),
                    ldt(x, (bos + t) * xrow + c * xcol));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_kkt_solve */
/* A = (I + strict_lower(diag(beta) . K K^T . e^{g_i - g_j}))^{-1}, per chunk, per v head.
 *
 * The decay is applied per element as e^{g_i - g_j}, not split about a per-chunk reference: only
 * i > j is kept and g decreases along the chunk, so the exponent is <= 0 and cannot overflow (see
 * the file header and libr4d's r4d_gdn_kkt_solve).
 *
 * The inverse is a forward substitution instead of libr4d's blocked 16 -> 32 -> 64 WMMA merge. I + M
 * is unit lower triangular, so its inverse is unique; both compute it in fp32. */
int ref_gdn_kkt_solve(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *k = &a->t[0], *beta = &a->t[1], *g = &a->t[2], *cu = &a->t[3], *A = &a->t[4];

    const int64_t K = p_int(a, "head_k", 0);
    /* `chunk` is RAD_DERIVED and may be absent; here A's shape [T, heads, chunk] carries it. */
    const int64_t chunk = p_int(a, "chunk", A->rank >= 3 ? A->shape[A->rank - 1] : 0);
    if (K <= 0 || chunk <= 0) return RAD_E_SHAPE;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_E_SHAPE;
    const int64_t H = numel(beta) / T;
    const int64_t Hg = numel(k) / (T * K);
    if (H <= 0 || Hg <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    if (numel(A) < T * H * chunk) return RAD_E_SHAPE;
    const int64_t grp = H / Hg;

    const V3 kv = v3(k, Hg, K), Av = v3(A, H, chunk);
    const V2 gv = v2(g, H), bv = v2(beta, H);

    /* The chunk's keys are widened and transposed first, so the dot products and the
     * substitution run across j in vector lanes; each element's chain keeps its order. */
    REF_PAR_FOR(T * H * chunk * K)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H, hk = h / grp;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        float* wsp = ws_get((size_t)(chunk * chunk + 3 * chunk + 2 * chunk * K));
        float* Am  = wsp;                 /* [chunk, chunk] the inverse, in fp32 */
        float* row = Am + chunk * chunk;  /* [chunk] */
        float* s   = row + chunk;         /* [chunk] */
        float* gc  = s + chunk;           /* [chunk] */
        float* kc  = gc + chunk;          /* [chunk, K] */
        float* kT  = kc + chunk * K;      /* [K, chunk] */

        for (int64_t c0 = 0; c0 < slen; c0 += chunk) {
            const int64_t L = (c0 + chunk < slen) ? chunk : slen - c0;
            for (int64_t j = 0; j < L; ++j) {
                const int64_t tj = bos + c0 + j;
                gc[j] = ldt(g, at2(gv, tj, h));
                for (int64_t d = 0; d < K; ++d) {
                    kc[j * K + d] = ldt(k, at3(kv, tj, hk, d));
                    kT[d * chunk + j] = kc[j * K + d];
                }
            }
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                const float bi = ldt(beta, at2(bv, ti, h));
                const float gi = gc[i];
                for (int64_t j = 0; j < i; ++j) row[j] = 0.0f;
                for (int64_t d = 0; d < K; ++d) {
                    const float kd = kc[i * K + d];
                    const float* kr = kT + d * chunk;
                    for (int64_t j = 0; j < i; ++j) row[j] += kd * kr[j];
                }
                for (int64_t j = 0; j < i; ++j) row[j] = bi * row[j] * ref_exp(gi - gc[j]);
                /* (I + M) A = I, row by row: A[i][j] = delta - sum_{j<=l<i} M[i][l] A[l][j]. */
                for (int64_t j = 0; j <= i; ++j) s[j] = (i == j) ? 1.0f : 0.0f;
                for (int64_t l = 0; l < i; ++l) {
                    const float rl = row[l];
                    const float* al = Am + l * chunk;
                    for (int64_t j = 0; j <= l; ++j) s[j] -= rl * al[j];
                }
                for (int64_t j = 0; j <= i; ++j) Am[i * chunk + j] = s[j];
                for (int64_t j = i + 1; j < chunk; ++j) Am[i * chunk + j] = 0.0f;
                for (int64_t j = 0; j < chunk; ++j)
                    stt(A, at3(Av, ti, h, j), Am[i * chunk + j]);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_chunk_scan */
int ref_gdn_chunk_scan(const RadArgs* a, RadStream) {
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *A = &a->t[3];
    const RadTensor *g = &a->t[4], *beta = &a->t[5], *cu = &a->t[7];
    const RadTensor *o = &a->t[8], *ht = &a->t[9];
    const RadTensor* h0 = t_in(a, 6);
    /* The state slot, for a state pool not indexed in batch order. Absent, row n owns slot n. */
    const RadTensor* sidx = t_in(a, 10);
    if (!q->data || !k->data || !v->data || !A->data || !g->data || !beta->data ||
        !cu->data || !o->data) return RAD_E_INVAL;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    /* RAD_DERIVED; A is [T, heads, chunk], as gdn_kkt_solve writes it. */
    const int64_t chunk = p_int(a, "chunk", A->rank >= 3 ? A->shape[A->rank - 1] : 0);
    const float scale = p_f32(a, "scale", 1.0f);
    if (K <= 0 || V <= 0 || chunk <= 0) return RAD_E_SHAPE;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    const int64_t T = cu_at(cu, N);
    if (T <= 0) return RAD_E_SHAPE;
    const int64_t H = numel(beta) / T;
    const int64_t Hg = numel(q) / (T * K);
    if (H <= 0 || Hg <= 0 || H % Hg != 0) return RAD_E_SHAPE;
    const int64_t grp = H / Hg;
    if (numel(o) < T * H * V || numel(A) < T * H * chunk) return RAD_E_SHAPE;
    if (!sidx) {
        if (ht->data && numel(ht) < N * H * V * K) return RAD_E_SHAPE;
        if (h0 && numel(h0) < N * H * V * K) return RAD_E_SHAPE;
    } else if (sidx->rank < 1 || sidx->shape[0] < N) {
        return RAD_E_SHAPE;
    }

    const V3 qv = v3(q, Hg, K), kv = v3(k, Hg, K), vv = v3(v, H, V), ov = v3(o, H, V);
    const V3 Av = v3(A, H, chunk);
    const V2 gv = v2(g, H), bv = v2(beta, H);
    const V4 h0v = h0 ? v4(h0, H, V, K) : V4{ nullptr, 0, 0, 0, 0 };
    const V4 htv = ht->data ? v4(ht, H, V, K) : V4{ nullptr, 0, 0, 0, 0 };

    /* The state is held transposed, St[d][e], and the chunk's operands widened to f32 first, so
     * every chain below runs across the V value columns in vector lanes. Each output element is
     * still the same chain in the same order: the u, Vp, o and S sums over d or j ascending, and
     * the products parenthesised as the formulas above write them. */
    REF_PAR_FOR(T * H * (K + chunk) * V)
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t n = nh / H, h = nh % H, hk = h / grp;
        const int64_t bos = cu_at(cu, n), slen = cu_at(cu, n + 1) - bos;
        /* Column 0 of a state_idx row is the state slot. Rows may be wider (speculative decode
         * keeps extra slots beside it); the scan writes only column 0's. */
        const int64_t sw = sidx && sidx->rank >= 1 ? sidx->stride[0] : 1;
        const int64_t sn = sidx ? ld_int(sidx->dtype, sidx->data, n * sw) : n;
        if (sn < 0) continue;                  /* a padded row owns no state */

        /* One workspace request, carved up by hand (see ws_get in ref_common.h). */
        const size_t need = (size_t)(K * V + 3 * chunk * V + 3 * chunk * K + chunk * chunk +
                                     3 * chunk + 2 * V);
        float* wsp = ws_get(need);
        float* St  = wsp;                      /* [K, V] the recurrent state, transposed */
        float* Wm  = St + K * V;               /* [chunk, V] */
        float* Vp  = Wm + chunk * V;           /* [chunk, V] */
        float* vc  = Vp + chunk * V;           /* [chunk, V] v, widened */
        float* kc  = vc + chunk * V;           /* [chunk, K] k, widened */
        float* qc  = kc + chunk * K;           /* [chunk, K] q, widened */
        float* kT  = qc + chunk * K;           /* [K, chunk] k, transposed */
        float* Ac  = kT + chunk * K;           /* [chunk, chunk] A, widened */
        float* gc  = Ac + chunk * chunk;       /* [chunk] g */
        float* bc  = gc + chunk;               /* [chunk] beta */
        float* row = bc + chunk;               /* [chunk] one row of sA */
        float* acc = row + chunk;              /* [V] */
        float* inr = acc + V;                  /* [V] */

        for (int64_t e = 0; e < V; ++e)
            for (int64_t d = 0; d < K; ++d)
                St[d * V + e] = h0 ? ldt(h0, at4(h0v, sn, h, e, d)) : 0.0f;

        for (int64_t c0 = 0; c0 < slen; c0 += chunk) {
            const int64_t L = (c0 + chunk < slen) ? chunk : slen - c0;
            for (int64_t j = 0; j < L; ++j) {
                const int64_t tj = bos + c0 + j;
                gc[j] = ldt(g, at2(gv, tj, h));
                bc[j] = ldt(beta, at2(bv, tj, h));
                for (int64_t d = 0; d < K; ++d) {
                    kc[j * K + d] = ldt(k, at3(kv, tj, hk, d));
                    qc[j * K + d] = ldt(q, at3(qv, tj, hk, d));
                    kT[d * chunk + j] = kc[j * K + d];
                }
                for (int64_t e = 0; e < V; ++e) vc[j * V + e] = ldt(v, at3(vv, tj, h, e));
                for (int64_t i = 0; i <= j; ++i) Ac[j * chunk + i] = ldt(A, at3(Av, tj, h, i));
            }
            const float glast = gc[L - 1];

            /* W = beta . (V - e^{g_j} . (K S^T)) */
            for (int64_t j = 0; j < L; ++j) {
                const float ge = ref_exp(gc[j]);
                for (int64_t e = 0; e < V; ++e) acc[e] = 0.0f;
                for (int64_t d = 0; d < K; ++d) {
                    const float kd = kc[j * K + d];
                    const float* sr = St + d * V;
                    for (int64_t e = 0; e < V; ++e) acc[e] += kd * sr[e];
                }
                for (int64_t e = 0; e < V; ++e)
                    Wm[j * V + e] = bc[j] * (vc[j * V + e] - ge * acc[e]);
            }
            /* V' = A W, with A unit lower triangular so only j <= i contributes. */
            for (int64_t i = 0; i < L; ++i) {
                float* vp = Vp + i * V;
                for (int64_t e = 0; e < V; ++e) vp[e] = 0.0f;
                for (int64_t j = 0; j <= i; ++j) {
                    const float aij = Ac[i * chunk + j];
                    const float* wr = Wm + j * V;
                    for (int64_t e = 0; e < V; ++e) vp[e] += aij * wr[e];
                }
            }
            /* O = scale . e^{g_i} . (Q S^T) + sA V' */
            for (int64_t i = 0; i < L; ++i) {
                const int64_t ti = bos + c0 + i;
                const float gi = gc[i];
                const float ge = ref_exp(gi);
                for (int64_t j = 0; j <= i; ++j) row[j] = 0.0f;
                for (int64_t d = 0; d < K; ++d) {
                    const float qd = qc[i * K + d];
                    const float* kr = kT + d * chunk;
                    for (int64_t j = 0; j <= i; ++j) row[j] += qd * kr[j];
                }
                for (int64_t j = 0; j <= i; ++j) row[j] = scale * ref_exp(gi - gc[j]) * row[j];
                for (int64_t e = 0; e < V; ++e) inr[e] = 0.0f;
                for (int64_t d = 0; d < K; ++d) {
                    const float qd = qc[i * K + d];
                    const float* sr = St + d * V;
                    for (int64_t e = 0; e < V; ++e) inr[e] += qd * sr[e];
                }
                const float sg = scale * ge;
                for (int64_t e = 0; e < V; ++e) acc[e] = sg * inr[e];
                for (int64_t j = 0; j <= i; ++j) {
                    const float rj = row[j];
                    const float* vp = Vp + j * V;
                    for (int64_t e = 0; e < V; ++e) acc[e] += rj * vp[e];
                }
                for (int64_t e = 0; e < V; ++e) stt(o, at3(ov, ti, h, e), acc[e]);
            }
            /* S = e^{g_last} S + V'^T (e^{g_last - g_j} . K) */
            const float el = ref_exp(glast);
            for (int64_t j = 0; j < L; ++j) {
                const float gs = ref_exp(glast - gc[j]);
                for (int64_t d = 0; d < K; ++d) kc[j * K + d] = gs * kc[j * K + d];
            }
            for (int64_t d = 0; d < K; ++d) {
                float* sr = St + d * V;
                for (int64_t e = 0; e < V; ++e) sr[e] = el * sr[e];
                for (int64_t j = 0; j < L; ++j) {
                    const float gk = kc[j * K + d];
                    const float* vp = Vp + j * V;
                    for (int64_t e = 0; e < V; ++e) sr[e] += vp[e] * gk;
                }
            }
        }

        if (ht->data)
            for (int64_t e = 0; e < V; ++e)
                for (int64_t d = 0; d < K; ++d)
                    stt(ht, at4(htv, sn, h, e, d), St[d * V + e]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the anchor state form
 *
 * docs/OPS.md `state_form`: under `anchor` a sequence keeps one committed state, and each step
 * records its tokens' rank-1 update factors beside it; the next step replays the first
 * `num_accepted` of them onto the state before running its own tokens. The cache layout is
 * libr4d's (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip), so either library can continue
 * a cache the other wrote and an oracle can compare them:
 *
 *   state_idx  six columns: 0..2 the anchor, 3..5 the factor block. A paged caller's window can
 *              shift by -1, 0 or +1 pages a step; the anchor is published at column 2 and each
 *              factor copy is stamped with its column (1, 2, 3), so the next step finds the shift
 *              m - 1 from the stamp m at column 4 and reads the anchor at column 3 - m. A zeroed
 *              block reads as no shift. A non-paged caller names one slot in columns 0..2 and
 *              another in 3..5.
 *   factors    per token, from the head's [V, K] block of the factor slot: the normalised k once per
 *              32-row group of V, then v, then (g, beta) once per group, each group's rows
 *              replaying from their own copy; room for `tmax` tokens, the stamp just past them.
 *
 * libr4d's kernel is K = V = 128, where the block holds kTmax = 16 tokens and the stamp sits at
 * 16 * fstr. Square heads from 64 up hold 16 too, so the layout is libr4d's wherever both serve
 * it. A smaller head's block holds fewer tokens' factors (a 16 x 16 head: 7), so `tmax` is as many
 * as fit before the stamp; a step, or an accepted prefix, longer than that is refused. */
struct AnchorLayout {
    static constexpr int64_t kTmax = 16;
    int64_t groups, fstr, vb, gb, tmax, mark;
    AnchorLayout(int64_t K, int64_t V)
        : groups((V + 31) / 32), fstr(groups * K + V + 2 * groups), vb(groups * K),
          gb(groups * K + V), tmax(std::min<int64_t>(kTmax, (V * K - 1) / fstr)),
          mark(tmax * fstr) {}
    /* The anchor's column, from the stamp at the factor block's `mark`. */
    int anchor_col(const RadTensor* st, int64_t fb) const {
        const int m = (int)(ldt(st, fb + mark) + 0.5f) - 1;
        return 3 - ((m >= 0 && m <= 2) ? m : 1);
    }
};

/* ------------------------------------------------------------------ gdn_recurrent_update */
/* Decode's counterpart of the chunked scan, against the paged state cache: the initial state comes
 * from the slot the last accepted token wrote, and every candidate token writes its own slot since
 * survivors are unknown until verification (ref_ops.h). The gated RMS norm is folded into the
 * epilogue, as in libr4d, whose workgroup owns the whole row. */
int ref_gdn_recurrent_update(const RadArgs* a, RadStream) {
    if (a->n_t < 14) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *av = &a->t[3], *bvt = &a->t[4];
    const RadTensor *state = &a->t[7], *o = &a->t[13];
    const RadTensor* A_log = t_in(a, 5);
    const RadTensor* dt_bias = t_in(a, 6);
    const RadTensor* sidx = t_in(a, 8);
    const RadTensor* nacc = t_in(a, 9);
    const RadTensor* cu = t_in(a, 10);
    const RadTensor* z = t_in(a, 11);
    const RadTensor* nw = t_in(a, 12);
    /* The out projection's quantiser, folded in. Optional, all-or-nothing; see the operand list in
     * r4d_rows.cpp for why the group is the row. */
    const RadTensor* oq  = (a->n_t > 14 && a->t[14].data) ? &a->t[14] : nullptr;
    const RadTensor* osc = (a->n_t > 15 && a->t[15].data) ? &a->t[15] : nullptr;
    if (!q->data || !k->data || !v->data || !av->data || !bvt->data || !state->data || !o->data ||
        !cu) return RAD_E_INVAL;

    /* `cu` bounds the sequences, as on gdn_conv_update, which reads the same tensor on the same
     * step. A sequence count carried in the state index only works for rectangular batches, and the
     * fast kernels read `cu[n]` before any null test. */
    const char* sf = p_str(a, "state_form", "per_candidate");
    const bool anchor = sf && std::strcmp(sf, "anchor") == 0;
    if (!anchor && sf && std::strcmp(sf, "per_candidate") != 0) return RAD_E_UNSUPPORTED;

    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const float scale = p_f32(a, "scale", 1.0f);
    const float l2eps = p_f32(a, "l2_eps", 1e-6f);
    const float neps  = p_f32(a, "eps", 1e-6f);
    const float spthr = p_f32(a, "softplus_thr", 20.0f);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int nact = p_enum(a, "act", kActs, 2, 0);
    if (nact != 0 && nact != 1) return RAD_E_INVAL;
    if (K <= 0 || V <= 0) return RAD_E_SHAPE;

    /* Head and token counts come from v, [T, head_v_count, head_v]; `n_head_v` overrides the head
     * count for a flattened view. */
    int64_t Hv = p_int(a, "n_head_v", 0);
    if (Hv <= 0) Hv = v->rank >= 3 ? v->shape[v->rank - 2] : 0;
    if (Hv <= 0) return RAD_E_SHAPE;
    const int64_t Tk = numel(v) / (Hv * V);
    if (Tk <= 0) return RAD_E_SHAPE;
    const int64_t Hg = numel(q) / (Tk * K);
    if (Hg <= 0 || Hv % Hg != 0) return RAD_E_SHAPE;
    const int64_t grp = Hv / Hg;

    /* N comes only from cu ([N+1]); state_idx may have several columns per sequence. */
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    if (cu_at(cu, n_seq) != Tk) return RAD_E_SHAPE;
    const int64_t sw = sidx ? (sidx->rank >= 2 ? sidx->shape[sidx->rank - 1]
                                               : (numel(sidx) / n_seq > 0 ? numel(sidx) / n_seq
                                                                          : 1))
                            : 1;

    const V3 qv = v3(q, Hg, K), kv = v3(k, Hg, K), vv = v3(v, Hv, V), ov = v3(o, Hv, V);
    const V3 zv = z ? v3_rows(z, Hv, V) : vv;
    const V2 a2 = v2(av, Hv), b2 = v2(bvt, Hv);
    const V2 si = sidx ? v2(sidx, sw) : V2{ nullptr, 0, 0 };
    const V4 stv = v4(state, Hv, V, K);

    /* The anchor form keeps its replay factors inside a head's [V, K] block of a state slot, at
     * libr4d's offsets (AnchorLayout), so the block must be dense and hold every token's factors. */
    const AnchorLayout al(K, V);
    if (anchor) {
        if (!sidx || sw < 6) return RAD_E_SHAPE;
        if (stv.s3 != 1 || stv.s2 != K) return RAD_E_STRIDE;
        if (al.tmax < 1) return RAD_E_SHAPE;
        for (int64_t n = 0; n < n_seq; ++n) {
            const int64_t na = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, n)) : 1;
            if (cu_at(cu, n + 1) - cu_at(cu, n) > al.tmax || na > al.tmax) return RAD_E_SHAPE;
        }
    }

    /* The state is held transposed, St[d][e], so the two state passes run across the V value
     * columns in vector lanes; every chain over d keeps its order. */
    REF_PAR_FOR(Tk * Hv * K * V)
    for (int64_t nh = 0; nh < n_seq * Hv; ++nh) {
        const int64_t n = nh / Hv, h = nh % Hv, hk = h / grp;
        const int64_t bos = cu_at(cu, n), q_len = cu_at(cu, n + 1) - bos;
        /* Under `anchor` a sequence with no rows this step still folds its accepted prefix in. */
        if (q_len <= 0 && !anchor) continue;
        const int64_t na = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, n)) : 1;
        const int64_t rd = na > 0 ? na - 1 : 0;
        int64_t col[6] = { 0, 0, 0, 0, 0, 0 };
        int64_t slot_in = 0;
        if (anchor) {
            for (int c = 0; c < 6; ++c) col[c] = ld_int(sidx->dtype, sidx->data, at2(si, n, c));
            bool pad = false;
            for (int c = 0; c < 6; ++c) pad = pad || col[c] < 0;
            if (pad) continue;
            slot_in = col[al.anchor_col(state, at4(stv, col[4], h, 0, 0))];
        } else {
            slot_in = sidx ? ld_int(sidx->dtype, sidx->data, at2(si, n, rd < sw ? rd : sw - 1)) : n;
            if (slot_in < 0) continue;
        }
        /* The factor blocks: column 4's is read, and 3, 4 and 5 are written (each a page the
         * window can shift onto). */
        const int64_t fb  = anchor ? at4(stv, col[4], h, 0, 0) : 0;
        const int64_t fbs[3] = { anchor ? at4(stv, col[3], h, 0, 0) : 0, fb,
                                 anchor ? at4(stv, col[5], h, 0, 0) : 0 };
        const bool dup_f = anchor && (col[3] != col[4] || col[5] != col[4]);

        const size_t need = (size_t)(V * K + 2 * K + 3 * V);
        float* wsp = ws_get(need);
        float* St = wsp;                       /* [K, V] */
        float* qn = St + V * K;
        float* kn = qn + K;
        float* uu = kn + K;
        float* orow = uu + V;
        float* sk2 = orow + V;

        for (int64_t e = 0; e < V; ++e)
            for (int64_t d = 0; d < K; ++d) St[d * V + e] = ldt(state, at4(stv, slot_in, h, e, d));

        const float alog = A_log ? ldt(A_log, h * laststride(A_log)) : 0.0f;
        const float dtb  = dt_bias ? ldt(dt_bias, h * laststride(dt_bias)) : 0.0f;

        /* The delta rule's state update for one token over the state rows [e0, e1): decay by
         * exp(g), then S += beta (v - S k) k^T, `vrow(e)` reading v. A row's update reads only
         * that row, k, g and beta. Shared by the token loop and the replay, so a replayed token
         * updates the state exactly as the step that ran it did. */
        auto update = [&](float g, float bt, auto vrow, int64_t e0, int64_t e1) {
            const float dec = ref_exp(g);
            for (int64_t e = e0; e < e1; ++e) sk2[e] = 0.0f;
            for (int64_t d = 0; d < K; ++d) {
                float* sr = St + d * V;
                const float kd = kn[d];
                for (int64_t e = e0; e < e1; ++e) {
                    sr[e] *= dec;
                    sk2[e] += sr[e] * kd;
                }
            }
            for (int64_t e = e0; e < e1; ++e) uu[e] = bt * (vrow(e) - sk2[e]);
            for (int64_t d = 0; d < K; ++d) {
                float* sr = St + d * V;
                const float kd = kn[d];
                for (int64_t e = e0; e < e1; ++e) sr[e] += uu[e] * kd;
            }
        };

        if (anchor) {
            /* Replay the previous step's accepted tokens from their factors, publish the advanced
             * anchor at column 2, and stamp each factor copy with the column it sits in. */
            for (int64_t t = 0; t < na; ++t) {
                const int64_t f = fb + t * al.fstr;
                /* Each 32-row group of V replays from its own copy of k and its own (g, beta),
                 * as libr4d's kernel reads them. Every writer stores equal copies, so on a block
                 * a step wrote this is the same as reading one copy for all rows. */
                for (int64_t gi = 0; gi < al.groups; ++gi) {
                    for (int64_t d = 0; d < K; ++d) kn[d] = ldt(state, f + gi * K + d);
                    update(ldt(state, f + al.gb + 2 * gi), ldt(state, f + al.gb + 2 * gi + 1),
                           [&](int64_t e) { return ldt(state, f + al.vb + e); },
                           32 * gi, std::min<int64_t>(V, 32 * gi + 32));
                }
            }
            for (int64_t e = 0; e < V; ++e)
                for (int64_t d = 0; d < K; ++d)
                    stt(state, at4(stv, col[2], h, e, d), St[d * V + e]);
            for (int c = 0; c < 3; ++c) stt(state, fbs[c] + al.mark, (float)(c + 1));
        }

        for (int64_t t = 0; t < q_len; ++t) {
            const int64_t tok = bos + t;
            const float gv = -ref_exp(alog) *
                             act_softplus(ldt(av, at2(a2, tok, h)) + dtb, spthr);
            const float bt = act_sigmoid(ldt(bvt, at2(b2, tok, h)));

            float sq = 0.0f, sk = 0.0f;
            for (int64_t d = 0; d < K; ++d) {
                const float u = ldt(q, at3(qv, tok, hk, d)), w = ldt(k, at3(kv, tok, hk, d));
                sq += u * u;
                sk += w * w;
            }
            const float qs = 1.0f / std::sqrt(sq + l2eps) * scale;
            const float ks = 1.0f / std::sqrt(sk + l2eps);
            for (int64_t d = 0; d < K; ++d) {
                qn[d] = ldt(q, at3(qv, tok, hk, d)) * qs;
                kn[d] = ldt(k, at3(kv, tok, hk, d)) * ks;
            }

            update(gv, bt, [&](int64_t e) { return ldt(v, at3(vv, tok, h, e)); }, 0, V);
            for (int64_t e = 0; e < V; ++e) orow[e] = 0.0f;
            for (int64_t d = 0; d < K; ++d) {
                const float* sr = St + d * V;
                const float qd = qn[d];
                for (int64_t e = 0; e < V; ++e) orow[e] += sr[e] * qd;
            }

            /* This token's factors, for the next step to replay: the normalised k once per
             * 32-row group, v, then the decay exponent and beta once per group. */
            if (anchor)
                for (int c = 0; c < 3; ++c) {
                    if (c != 1 && !dup_f) continue;
                    const int64_t f = fbs[c] + t * al.fstr;
                    for (int64_t gi = 0; gi < al.groups; ++gi) {
                        for (int64_t d = 0; d < K; ++d) stt(state, f + gi * K + d, kn[d]);
                        stt(state, f + al.gb + 2 * gi, gv);
                        stt(state, f + al.gb + 2 * gi + 1, bt);
                    }
                    for (int64_t e = 0; e < V; ++e)
                        stt(state, f + al.vb + e, ldt(v, at3(vv, tok, h, e)));
                }

            /* The gated RMS norm, over the head's V channels. */
            float sq2 = 0.0f;
            for (int64_t e = 0; e < V; ++e) sq2 += orow[e] * orow[e];
            const float sc = 1.0f / std::sqrt(sq2 / (float)V + neps);
            for (int64_t e = 0; e < V; ++e) {
                float r = orow[e] * sc;
                if (nw) r *= ldt(nw, e * laststride(nw));
                if (z) {
                    const float zz = ldt(z, at3(zv, tok, h, e));
                    r *= (nact == 1 ? act_sigmoid(zz) : act_silu(zz));
                }
                stt(o, at3(ov, tok, h, e), r);
            }

            /* Quantise the stored `o`, not the f32 value, as the standalone quant_act_fp8 read it
             * back from memory. One group per (token, head), since head_v is the fp8 block width. */
            if (oq && osc) {
                const int64_t qb = rowoff(oq, tok, Hv * V), qs = laststride(oq);
                const float qsc = q8_group([&](int64_t e) { return ldt(o, at3(ov, tok, h, e)); },
                                            V, 448.0f, oq, qb + h * V * qs, qs);
                stt(osc, rowoff(osc, tok, Hv) + h * laststride(osc), qsc);
            }

            if (anchor) continue;
            const int64_t so = sidx ? ld_int(sidx->dtype, sidx->data, at2(si, n, t < sw ? t : sw - 1))
                                    : n;
            if (so >= 0)
                for (int64_t e = 0; e < V; ++e)
                    for (int64_t d = 0; d < K; ++d) stt(state, at4(stv, so, h, e, d), St[d * V + e]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_gated_rmsnorm */
/* out = x * rsqrt(mean_channels(x^2) + eps) * w * act(z), one row per (token, head): the prefill
 * form of the epilogue the recurrent update folds in. */
/* Row offset of the gate. rowoff() handles [tokens, heads, ch] and flat [rows, ch], but not a
 * column slice of a wider row (the fp8 delta net fuses [q|k|v|z] into one GEMM), whose last axis is
 * heads*ch; rowoff would read it as a dense row. */
static inline int64_t gate_rowoff(const RadTensor* t, int64_t r, int64_t ch) {
    if (t->rank == 2 && ch > 0 && t->shape[1] != ch && t->shape[1] % ch == 0) {
        const int64_t heads = t->shape[1] / ch;
        return (r / heads) * t->stride[0] + (r % heads) * ch * t->stride[1];
    }
    return rowoff(t, r, ch);
}

int ref_gdn_gated_rmsnorm(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *z = &a->t[1], *w = &a->t[2], *o = &a->t[3];
    const int64_t ch = p_int(a, "channels", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, 0);
    if (act != 0 && act != 1) return RAD_E_INVAL;
    if (ch <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / ch;
    if (rows <= 0 || numel(o) < rows * ch || numel(z) < rows * ch || numel(w) < ch)
        return RAD_E_SHAPE;

    REF_PAR_FOR(rows * ch)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, ch), zb = gate_rowoff(z, r, ch), ob = rowoff(o, r, ch);
        const int64_t xs = laststride(x), zs = laststride(z), os = laststride(o);
        const int64_t ws = laststride(w);
        float ss = 0.0f;
        for (int64_t i = 0; i < ch; ++i) { float u = ldt(x, xb + i * xs); ss += u * u; }
        const float sc = 1.0f / std::sqrt(ss / (float)ch + eps);
        for (int64_t i = 0; i < ch; ++i) {
            const float zz = ldt(z, zb + i * zs);
            stt(o, ob + i * os, ldt(x, xb + i * xs) * sc * ldt(w, i * ws) *
                                (act == 1 ? act_sigmoid(zz) : act_silu(zz)));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gdn_conv_recurrent_update */
/* gdn_conv_update into staging q, k and v, then gdn_recurrent_update over them; the fused op is
 * defined as the pair. Head geometry comes from `o` ([T, Hv, head_v]) and `x`
 * (2 Hg head_k + Hv head_v columns). */
int ref_gdn_conv_recurrent_update(const RadArgs* a, RadStream st) {
    if (a->n_t < 16) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *o = &a->t[15];
    if (!x->data || !o->data || o->rank != 3 || x->rank < 2) return RAD_E_SHAPE;
    const int64_t K = p_int(a, "head_k", 0), V = p_int(a, "head_v", 0);
    const int64_t T = o->shape[0], Hv = o->shape[1];
    if (K <= 0 || V <= 0 || T <= 0 || Hv <= 0 || o->shape[2] != V) return RAD_E_SHAPE;
    const int64_t conv_dim = x->shape[x->rank - 1];
    if (conv_dim <= Hv * V || (conv_dim - Hv * V) % (2 * K)) return RAD_E_SHAPE;
    const int64_t Hg = (conv_dim - Hv * V) / (2 * K);

    /* The staging takes x's dtype, so q, k and v round as the unfused pair's stores do. */
    const auto bytes = [&](int64_t n) { return (size_t)rad_dtype_bytes(x->dtype, n); };
    std::vector<uint8_t> qb(bytes(T * Hg * K)), kb(bytes(T * Hg * K)), vb(bytes(T * Hv * V));
    auto stage = [&](uint8_t* d, int64_t h, int64_t w) {
        RadTensor t{};
        t.data = d; t.dtype = x->dtype; t.rank = 3;
        t.shape[0] = T; t.shape[1] = h; t.shape[2] = w;
        t.stride[0] = h * w; t.stride[1] = w; t.stride[2] = 1;
        return t;
    };
    const RadTensor tq = stage(qb.data(), Hg, K), tk = stage(kb.data(), Hg, K),
                    tv = stage(vb.data(), Hv, V);

    RadParam cp[4] = {};
    const char* ck[4] = { "q_len", "head_k", "head_v", "conv_width" };
    for (int i = 0; i < 4; ++i) {
        const RadParam* f = rad_param_find(a->p, a->n_p, ck[i]);
        if (!f) return RAD_E_INVAL;
        cp[i] = *f;
    }
    /* x, w, b, conv_state, state_idx, num_accepted, cu, q, k, v */
    RadTensor ct[10] = { a->t[0], a->t[1], a->t[2], a->t[3], a->t[4], a->t[11], a->t[12], tq, tk,
                         tv };
    RadArgs ca = *a;
    ca.t = ct; ca.n_t = 10; ca.p = cp; ca.n_p = 4;
    int rc = ref_gdn_conv_update(&ca, st);
    if (rc != RAD_OK) return rc;

    /* q, k, v, a, b, A_log, dt_bias, state, state_idx, num_accepted, cu, z, norm_w, o, o_q,
     * o_scale. The parameters pass as they are: the recurrence reads its own by name. */
    RadTensor rt[16] = { tq, tk, tv, a->t[5], a->t[6], a->t[7], a->t[8], a->t[9], a->t[10],
                         a->t[11], a->t[12], a->t[13], a->t[14], a->t[15],
                         a->n_t > 16 ? a->t[16] : RadTensor{}, a->n_t > 17 ? a->t[17] : RadTensor{} };
    RadArgs ra = *a;
    ra.t = rt; ra.n_t = 16;
    return ref_gdn_recurrent_update(&ra, st);
}
