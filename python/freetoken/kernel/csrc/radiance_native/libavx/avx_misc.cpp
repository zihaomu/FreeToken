/* avx_misc.cpp -- the PLE pair, the hashed n-gram ids, and the two collectives. Mostly scalar:
 *
 *   ple_gate vectorises the per-stream norms and the gated broadcast of `v`. The dot between the
 *   two normed streams is accumulated in f64, as in libref: it feeds a signed square root, whose
 *   relative sensitivity is unbounded near zero, so an f32 reduction over a 10240-wide product
 *   would move the gate past tolerance on rows near the hinge.
 *
 *   ple_conv is a width-4 depthwise convolution with taps `dilation` apart; channels are the
 *   parallel axis and the four-term inner loop has nothing to vectorise.
 *
 *   ngram_ids is integer hashing with a data-dependent early cut, using torch's arithmetic (a
 *   wrapping 64-bit multiply and a flooring modulo).
 *
 *   all_reduce / all_gather serve world size 1 only and refuse wider worlds (see below).
 */
#include "avx_vec.h"
#include "avx_common.h"

using namespace avx;

/* Reciprocal RMS of one n-wide stream, accumulated in f64 as libref's `group_rsqrt`. The rsqrt is
 * per stream while the gain spans all hc*n; that is what `group_size` means on
 * Qwen4ExpTextRMSNorm. */
static float group_rsqrt_f64(const float* v, int64_t n, float eps) {
    double acc = 0;
    for (int64_t d = 0; d < n; ++d) acc += (double)v[d] * (double)v[d];
    return (float)(1.0 / std::sqrt(acc / (double)n + (double)eps));
}

/* ------------------------------------------------------------------ ple_gate */
/*   k        [M][hc*n]   key_proj(embedding), unnormed
 *   q        [M][hc*n]   the incoming hyper-connection stream, unnormed
 *   v        [M][n]      value_proj(embedding): one row, gated differently per stream
 *   w_key / w_query / w_conv   [hc*n]  gains
 *   gv       [M][hc*n]   out, unnormed: read by the residual add
 *   gvn      [M][hc*n]   out, normed: read by the convolution
 *
 * The third norm is taken over gv as stored (rounded through its dtype), which is why this op owns
 * all three norms.
 *
 * Signed square root in the reference's order: clamp the magnitude to `gate_eps`, take the root,
 * reapply the sign. Other orders differ near zero, where the gate usually sits. */
AVX_KERNEL(ple_gate) {
    if (!rad_args_have(a, 8)) return RAD_E_INVAL;
    const RadTensor* k = &a->t[0];
    const RadTensor* q = &a->t[1];
    const RadTensor* v = &a->t[2];
    const RadTensor* wk = &a->t[3];
    const RadTensor* wq = &a->t[4];
    const RadTensor* wc = &a->t[5];
    const RadTensor* gv = &a->t[6];
    const RadTensor* gvn = &a->t[7];

    const int64_t n = p_int(a, "n", 0);
    /* Rows come from `v` ([M, n]); the `M` parameter is the band's upper bound. */
    const int64_t M = n > 0 ? band_rows(numel(v) / n, p_int(a, "M", 0)) : 0;
    const int64_t hc = p_int(a, "hc", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    /* Floor for the gate's magnitude before the square root. */
    const float geps = p_f32(a, "gate_eps", 1e-6f);
    if (M <= 0 || n <= 0 || hc <= 0) return RAD_E_SHAPE;
    const int64_t hn = hc * n;
    if (numel(k) < M * hn || numel(q) < M * hn || numel(v) < M * n) return RAD_E_SHAPE;
    if (numel(wk) < hn || numel(wq) < hn || numel(wc) < hn) return RAD_E_SHAPE;
    if (numel(gv) < M * hn || numel(gvn) < M * hn) return RAD_E_SHAPE;

    /* The three gains, widened once with `wadd` folded in, in one allocation: a second
     * scratch_raw call could free this one under the worker threads. */
    float* gains = (float*)scratch_raw((size_t)(3 * hn) * sizeof(float) + 64);
    float* gk = gains;
    float* gq = gains + hn;
    float* gc = gains + 2 * hn;
    {
        const int64_t sk = laststride(wk), sq = laststride(wq), sc = laststride(wc);
        for (int64_t i = 0; i < hn; ++i) {
            gk[i] = ldt(wk, i * sk) + wadd;
            gq[i] = ldt(wq, i * sq) + wadd;
            gc[i] = ldt(wc, i * sc) + wadd;
        }
    }


    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t kb = rowoff(k, m, hn), qb = rowoff(q, m, hn), vb = rowoff(v, m, n);
        const int64_t gb = rowoff(gv, m, hn), nb = rowoff(gvn, m, hn);
        float* kv = scratch_f32((size_t)hn);
        float* qv = scratch_f32_b((size_t)hn);
        float* out = scratch_f32_c((size_t)(hn + n) + 16);
        float* vv = out + hn + 8;
        const float* kk = in_row(k, kb, hn, kv);
        const float* qq = in_row(q, qb, hn, qv);
        const float* v0 = in_row(v, vb, n, vv);
        if (v0 != vv) std::memcpy(vv, v0, (size_t)n * sizeof(float));

        for (int64_t c = 0; c < hc; ++c) {
            const float rk = group_rsqrt_f64(kk + c * n, n, eps);
            const float rq = group_rsqrt_f64(qq + c * n, n, eps);
            /* f64: the dot feeds the signed square root (see the file header). */
            double dot = 0;
            for (int64_t d = 0; d < n; ++d) {
                const int64_t j = c * n + d;
                const float a1 = narrow1(k->dtype, kk[j] * rk * gk[j]);
                const float b1 = narrow1(q->dtype, qq[j] * rq * gq[j]);
                dot += (double)a1 * (double)b1;
            }
            float s = (float)(dot / std::sqrt((double)n));
            const float mag = std::sqrt(std::fabs(s) < geps ? geps : std::fabs(s));
            s = s > 0.0f ? mag : (s < 0.0f ? -mag : 0.0f);
            const float g = 1.0f / (1.0f + sc_exp(-s));

            const vf vg = vf_set1(g);
            int64_t d = 0;
            for (; d + VF_N <= n; d += VF_N)
                vf_storeu(out + c * n + d, vf_mul(vg, vf_loadu(vv + d)));
            for (; d < n; ++d) out[c * n + d] = g * vv[d];
        }
        out_row(gv, gb, hn, out);

        /* Third norm over gv as stored: round `out` through gv's dtype in scratch. */
        if (gv->dtype != RAD_F32) {
            row_from_f32(gv->dtype, out, kv, hn);
            row_to_f32(gv->dtype, kv, out, hn);
        }
        for (int64_t c = 0; c < hc; ++c) {
            const float r = group_rsqrt_f64(out + c * n, n, eps);
            for (int64_t d = 0; d < n; ++d) {
                const int64_t j = c * n + d;
                qv[j] = narrow1(gvn->dtype, out[j] * r * gc[j]);
            }
        }
        out_row(gvn, nb, hn, qv);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ple_conv */
/* PLE's dilated depthwise convolution, then silu, with the residual add folded in. Two rules,
 * both silent when broken:
 *
 *   `has_init` zero means the slot holds no history: the window before the first token is zeros.
 *   Otherwise a new sequence on a recycled slot convolves the previous sequence's tail into its
 *   first (width-1)*dilation tokens.
 *
 *   `num_accepted` shifts the read and selects the window left behind (full contract: libref's
 *   ref_ple.cpp). Present, this is a decode step whose drafts may be rejected: the history read is
 *   offset by num_accepted - 1 and the state keeps the shifted history followed by every input of
 *   the step, so a next step that kept k tokens reads at k - 1. Absent, every token commits and the
 *   last (width-1)*dilation inputs land at offset zero.
 *
 * The state is written in a second pass after all reads, since a sequence longer than the window
 * would otherwise overwrite history it still reads.
 *
 * One body serves ple_conv and causal_conv1d (LFM2's short convolution: undilated, no activation,
 * no residual), as in libref, so both use the same window layout. */
struct ConvOps {
    const RadTensor *x, *w, *cs, *cu, *resid, *sidx, *hini, *nacc, *y;
    bool silu;
    const RadTensor* b;   /* per-channel bias inside the activation (Mamba-2's conv1d), or null */
};

static int depthwise_causal_conv(const RadArgs* a, const ConvOps& o) {
    const RadTensor *x = o.x, *w = o.w, *cs = o.cs, *cu = o.cu, *resid = o.resid;
    const RadTensor *sidx = o.sidx, *hini = o.hini, *nacc = o.nacc, *y = o.y, *b = o.b;
    if (!x || !w || !cs || !cu || !y) return RAD_E_INVAL;

    const int64_t n = p_int(a, "n", 0);
    /* Rows come from `x` ([M, n]); the `M` parameter is the band's upper bound. */
    const int64_t M = n > 0 ? band_rows(numel(x) / n, p_int(a, "M", 0)) : 0;
    const int64_t wid = p_int(a, "width", 0);
    const int64_t dil = p_int(a, "dilation", 1);
    if (M <= 0 || n <= 0 || wid <= 0 || dil <= 0) return RAD_E_SHAPE;
    const int64_t hist_len = (wid - 1) * dil;
    /* The read window is a stack array (9 at PLE's shape); wider windows are refused rather than
     * overflowing. Same bound as libref. */
    enum { kHistMax = 64 };
    if (hist_len > kHistMax) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    if (numel(x) < M * n || numel(y) < M * n || numel(w) < n * wid) return RAD_E_SHAPE;
    if (resid && numel(resid) < M * n) return RAD_E_SHAPE;
    if (b && numel(b) != n) return RAD_E_SHAPE;

    const int64_t state_len = cs->rank >= 3 ? cs->shape[cs->rank - 1] : hist_len;
    if (cs->data && state_len < hist_len) return RAD_E_SHAPE;

    /* Validate every sequence before writing any (an OpenMP body cannot return): the state must
     * be deep enough for the offset and for the window a decode step leaves behind. */
    for (int64_t s = 0; s < N; ++s) {
        const int64_t off = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;
        if (off < 0) return RAD_E_INVAL;
        if (off + hist_len > state_len) return RAD_E_SHAPE;
        const int64_t slen = ldi(cu, offlin(cu, s + 1)) - ldi(cu, offlin(cu, s));
        if (nacc && cs->data && hist_len > 0 && slen > 0 && hist_len - 1 + slen > state_len)
            return RAD_E_SHAPE;
    }

    const int64_t xrow = rowoff(x, 1, n), xcol = laststride(x);
    const int64_t yrow = rowoff(y, 1, n), ycol = laststride(y);
    const int64_t rrow = resid ? rowoff(resid, 1, n) : 0;
    const int64_t rcol = resid ? laststride(resid) : 0;
    const int64_t wrow = rowoff(w, 1, wid), wcol = laststride(w);
    /* [n_slots, n, state_len], dense, the same layout as gdn's conv state. */
    const int64_t cs_slot = n * state_len, cs_ch = state_len;

    AVX_PARALLEL_FOR_IF(N * n >= 64)
    for (int64_t sc = 0; sc < N * n; ++sc) {
        const int64_t s = sc / n, c = sc % n;
        const int64_t bos = ldi(cu, offlin(cu, s));
        const int64_t slen = ldi(cu, offlin(cu, s + 1)) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ldi(sidx, offlin(sidx, s)) : s;
        if (slot < 0) continue;
        const bool warm = !hini || ldi(hini, offlin(hini, s)) != 0;
        const int64_t off = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;

        float hist[kHistMax];
        for (int64_t i = 0; i < hist_len; ++i)
            hist[i] = (warm && cs->data) ? ldt(cs, slot * cs_slot + c * cs_ch + off + i) : 0.0f;

        /* Taps read once per channel (libref reads them per output element). Kept in scratch, not
         * on the stack, so `width` needs no bound of its own. */
        float* tap = scratch_f32((size_t)wid);
        for (int64_t j = 0; j < wid; ++j) tap[j] = ldt(w, c * wrow + j * wcol);
        const float bias = b ? ldt(b, c * laststride(b)) : 0.0f;

        for (int64_t t = 0; t < slen; ++t) {
            float acc = bias;
            for (int64_t j = 0; j < wid; ++j) {
                const int64_t src = t - hist_len + j * dil;
                const float vv = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                          : hist[src + hist_len];
                acc += vv * tap[j];
            }
            const float r = resid ? ldt(resid, (bos + t) * rrow + c * rcol) : 0.0f;
            stt(y, (bos + t) * yrow + c * ycol, r + (o.silu ? acc / (1.0f + sc_exp(-acc)) : acc));
        }
    }

    if (cs->data && hist_len > 0) {
        AVX_PARALLEL_FOR_IF(N * n >= 64)
        for (int64_t sc = 0; sc < N * n; ++sc) {
            const int64_t s = sc / n, c = sc % n;
            const int64_t bos = ldi(cu, offlin(cu, s));
            const int64_t slen = ldi(cu, offlin(cu, s + 1)) - bos;
            if (slen <= 0) continue;
            const int64_t slot = sidx ? ldi(sidx, offlin(sidx, s)) : s;
            if (slot < 0) continue;
            const bool warm = !hini || ldi(hini, offlin(hini, s)) != 0;
            const int64_t off = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;
            const int64_t base = slot * cs_slot + c * cs_ch;

            float win[kHistMax];
            for (int64_t i = 0; i < hist_len; ++i) win[i] = warm ? ldt(cs, base + off + i) : 0.0f;

            if (nacc) {
                for (int64_t i = 0; i + 1 < hist_len; ++i) stt(cs, base + i, win[i + 1]);
                for (int64_t t = 0; t < slen; ++t)
                    stt(cs, base + hist_len - 1 + t, ldt(x, (bos + t) * xrow + c * xcol));
            } else {
                float keep[kHistMax];
                for (int64_t i = 0; i < hist_len; ++i) {
                    const int64_t src = slen - hist_len + i;
                    keep[i] = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                       : win[src + hist_len];
                }
                for (int64_t i = 0; i < hist_len; ++i) stt(cs, base + i, keep[i]);
            }
        }
    }
    return RAD_OK;
}

AVX_KERNEL(ple_conv) {
    if (a->n_t < 9) return RAD_E_INVAL;
    return depthwise_causal_conv(a, { rad_arg_in(a, 0), rad_arg_in(a, 1), rad_arg_in(a, 2),
                                      rad_arg_in(a, 3), rad_arg_in(a, 4), rad_arg_in(a, 5),
                                      rad_arg_in(a, 6), rad_arg_in(a, 7), rad_arg_in(a, 8), true });
}

/* ple_conv's operands with `b`, a per-channel bias inside the silu, before `y`: Mamba-2's
 * conv1d at dilation one. */
AVX_KERNEL(ple_conv_bias) {
    if (a->n_t < 10 || !rad_arg_in(a, 8)) return RAD_E_INVAL;
    return depthwise_causal_conv(a, { rad_arg_in(a, 0), rad_arg_in(a, 1), rad_arg_in(a, 2),
                                      rad_arg_in(a, 3), rad_arg_in(a, 4), rad_arg_in(a, 5),
                                      rad_arg_in(a, 6), rad_arg_in(a, 7), rad_arg_in(a, 9), true,
                                      rad_arg_in(a, 8) });
}

/* ple_conv's operands without `resid`, dilation 1, no activation. */
AVX_KERNEL(causal_conv1d) {
    if (a->n_t < 8) return RAD_E_INVAL;
    if (p_int(a, "dilation", 1) != 1) return RAD_E_INVAL;
    return depthwise_causal_conv(a, { rad_arg_in(a, 0), rad_arg_in(a, 1), rad_arg_in(a, 2),
                                      rad_arg_in(a, 3), nullptr, rad_arg_in(a, 4),
                                      rad_arg_in(a, 5), rad_arg_in(a, 6), rad_arg_in(a, 7), false });
}

/* ------------------------------------------------------------------ ngram_ids
 *
 * ids[m][h] = (hash of the n-gram ending at m) % vocab[h] + offset[h], in torch's arithmetic:
 *
 *   Signed overflow is UB in C++ but wraps in torch, so the multiply goes through uint64.
 *   `torch.remainder` floors where C's `%` truncates, so the modulo is written out.
 *
 * Neither matters on a real checkpoint (multipliers are bounded by (2^63-1)//vocab_size, so real
 * products are non-negative); they make random test inputs match torch.
 *
 * Once a shift reaches EOS, every longer shift is EOS too, so the cut is carried. A cold slot's
 * window is EOS, not zero, because zero is a real token. */
static inline int64_t wrap_mul(int64_t x, int64_t y) {
    return (int64_t)((uint64_t)x * (uint64_t)y);
}
static inline int64_t floor_mod(int64_t x, int64_t y) {   /* y > 0 */
    const int64_t r = x % y;
    return r < 0 ? r + y : r;
}

AVX_KERNEL(ngram_ids) {
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor* tok = rad_arg_in(a, 0);
    const RadTensor* st = rad_arg_in(a, 1);
    const RadTensor* cu = rad_arg_in(a, 2);
    const RadTensor* mult = rad_arg_in(a, 3);
    const RadTensor* vsz = rad_arg_in(a, 4);
    const RadTensor* off_ = rad_arg_in(a, 5);
    const RadTensor* sidx = rad_arg_in(a, 6);
    const RadTensor* hini = rad_arg_in(a, 7);
    const RadTensor* nacc = rad_arg_in(a, 8);
    const RadTensor* ids = rad_arg_in(a, 9);
    /* Optional ids for the prompt that follows: `tok` holds that many more tokens after the
     * step's, continuing the last sequence. The state window is still the step's own. */
    const RadTensor* ahead = a->n_t > 10 ? rad_arg_in(a, 10) : nullptr;
    if (!tok || !st || !cu || !mult || !vsz || !off_ || !ids) return RAD_E_INVAL;

    const int64_t heads = p_int(a, "heads", 0);
    if (heads <= 0) return RAD_E_SHAPE;
    /* Rows come from `tok` minus the prompt ahead; the `M` parameter is the band's upper bound. */
    const int64_t A = ahead ? numel(ahead) / heads : 0;
    const int64_t M = band_rows(numel(tok) - A, p_int(a, "M", 0));
    const int64_t ngram = p_int(a, "ngram", 0);
    const int64_t eos = p_int(a, "eos", -1);
    if (M <= 0 || ngram < 2 || eos < 0) return RAD_E_SHAPE;
    /* Heads form `ngram - 1` blocks of `heads_per_ngram`; block b uses the (b+2)-gram. */
    if (heads % (ngram - 1)) return RAD_E_SHAPE;
    const int64_t per = heads / (ngram - 1);
    const int64_t ctx = ngram - 1;
    if (ngram > 8) return RAD_E_UNSUPPORTED;      /* the shift window is a stack array */

    if (numel(mult) < ngram || numel(vsz) < heads || numel(off_) < heads) return RAD_E_SHAPE;
    if (numel(tok) < M + A || numel(ids) < M * heads) return RAD_E_SHAPE;

    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t W = st->rank >= 2 ? st->shape[st->rank - 1] : ctx;
    if (W < ctx) return RAD_E_SHAPE;
    /* The state must hold the read window and the window a decode step leaves behind (`ctx - 1`
     * ids of history plus every id of the step). */
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t o = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;
        if (o < 0) return RAD_E_INVAL;
        if (o + ctx > W) return RAD_E_SHAPE;
        const int64_t T = ldi(cu, offlin(cu, s + 1)) - ldi(cu, offlin(cu, s));
        if (nacc && T > 0 && ctx - 1 + T > W) return RAD_E_SHAPE;
    }

    const int64_t ids_ld = rowoff(ids, 1, heads);
    const int64_t ahd_ld = ahead ? rowoff(ahead, 1, heads) : 0;

    /* Per-head tables read once (libref reads them in the innermost loop). */
    int64_t* tab = (int64_t*)scratch_raw((size_t)(2 * heads + ngram) * sizeof(int64_t) + 64);
    int64_t* vtab = tab;
    int64_t* otab = tab + heads;
    int64_t* mtab = tab + 2 * heads;
    for (int64_t h = 0; h < heads; ++h) {
        vtab[h] = ldi(vsz, h);
        otab[h] = ldi(off_, h);
        if (vtab[h] <= 0 || otab[h] < 0) return RAD_E_SHAPE;
    }
    for (int64_t k = 0; k < ngram; ++k) mtab[k] = ldi(mult, k);

    /* Serial over sequences, as in libref: the per-sequence work is a short integer loop, too
     * little to pay for a parallel region. */
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t bos = ldi(cu, offlin(cu, s));
        const int64_t eot = ldi(cu, offlin(cu, s + 1));
        if (bos < 0 || eot < bos || eot > M) return RAD_E_SHAPE;
        const int64_t T = eot - bos;
        if (T <= 0) continue;
        const int64_t slot = sidx ? ldi(sidx, offlin(sidx, s)) : s;
        if (slot < 0) continue;
        const bool warm = !hini || ldi(hini, offlin(hini, s)) != 0;
        const int64_t o = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;

        int64_t prev[8];
        for (int64_t i = 0; i < ctx; ++i)
            prev[i] = warm ? ldi(st, rowoff(st, slot, W) + o + i) : eos;

        auto hist = [&](int64_t p) -> int64_t {
            return p < ctx ? prev[p] : ldi(tok, offlin(tok, bos + p - ctx));
        };

        const int64_t reach = T + (ahead && s + 1 == n_seq ? A : 0);
        for (int64_t t = 0; t < reach; ++t) {
            const int64_t p = ctx + t;
            int64_t sh[8];
            sh[0] = hist(p);
            bool cut = false;
            for (int64_t k = 1; k < ngram; ++k) {
                if (cut || sh[k - 1] == eos) { cut = true; sh[k] = eos; continue; }
                sh[k] = hist(p - k);
            }

            /* Mixes are extended in place: the (b+2)-gram's is the (b+1)-gram's XOR one more
             * term. */
            int64_t mixed = 0;
            int64_t blk = -1;
            for (int64_t k = 0; k < ngram; ++k) {
                const int64_t term = wrap_mul(sh[k], mtab[k]);
                mixed = k == 0 ? term : (mixed ^ term);
                if (k == 0) continue;           /* a 1-gram addresses no head */
                ++blk;                          /* k = 1 is the 2-gram block, k = 2 the 3-gram */
                for (int64_t j = 0; j < per; ++j) {
                    const int64_t h = blk * per + j;
                    if (t < T)
                        st_int_dt(ids->dtype, ids->data, (bos + t) * ids_ld + h,
                                  floor_mod(mixed, vtab[h]) + otab[h]);
                    else
                        st_int_dt(ahead->dtype, ahead->data, (t - T) * ahd_ld + h,
                                  floor_mod(mixed, vtab[h]) + otab[h]);
                }
            }
        }

        /* State window, written after all reads, in ple_conv's layout: with `num_accepted`, the
         * history shifted by one followed by every id of the step; without, the last `ctx` ids at
         * offset zero. */
        const int64_t base = rowoff(st, slot, W);
        if (nacc) {
            for (int64_t i = 0; i + 1 < ctx; ++i)
                st_int_dt(st->dtype, st->data, base + i, prev[i + 1]);
            for (int64_t t = 0; t < T; ++t)
                st_int_dt(st->dtype, st->data, base + ctx - 1 + t, ldi(tok, offlin(tok, bos + t)));
        } else {
            for (int64_t i = 0; i < ctx; ++i) {
                const int64_t src = T - ctx + i;
                const int64_t vv = src >= 0 ? ldi(tok, offlin(tok, bos + src)) : prev[src + ctx];
                st_int_dt(st->dtype, st->data, base + i, vv);
            }
        }
    }
    return RAD_OK;
}

/* ================================================================== the collectives
 *
 * World size 1 only, by design. libref implements real host collectives (one process, one thread
 * per rank, a shared address space: a rendezvous, read every rank's pointer, rendezvous again),
 * which lets the host backend exercise tensor-parallel graphs without a second card. That state (a
 * barrier, a generation counter, a timeout) must exist once per process: a second copy in this
 * plugin would form a separate rendezvous group, and ranks would deadlock whenever selection put
 * them in different plugins. So above world 1 these return RAD_E_UNSUPPORTED and the selector
 * falls through to libref.
 *
 * `exact` is honoured one way, as in libref: exact=0 permits a lossy wire but there is none here,
 * so both settings get the exact result. */
AVX_KERNEL(all_reduce) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor* x = &a->t[0];
    const RadTensor* y = rad_arg_in(a, 1);
    /* The element count comes from the operand; the `numel` parameter is the band. */
    const int64_t nel = band_rows(numel(x), p_int(a, "numel", 0));
    if (nel <= 0) return RAD_E_SHAPE;
    if (y && numel(y) < nel) return RAD_E_SHAPE;

    const int world = (int)p_int(a, "world_size", a->world_size > 0 ? a->world_size : 1);
    if (world > 1) return RAD_E_UNSUPPORTED;

    /* In place: nothing to do. With a separate destination, copy x into it. */
    if (!y) return RAD_OK;
    if (rad_tensor_is_contiguous(x) && rad_tensor_is_contiguous(y)) {
        const int64_t TILE = 8192;
        for (int64_t off = 0; off < nel; off += TILE) {
            const int64_t m = (nel - off) < TILE ? (nel - off) : TILE;
            float* s = scratch_f32((size_t)TILE);
            out_row(y, off, m, in_row(x, off, m, s));
        }
        return RAD_OK;
    }
    for (int64_t i = 0; i < nel; ++i) stt(y, offlin(y, i), ldt(x, offlin(x, i)));
    return RAD_OK;
}

AVX_KERNEL(all_gather) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    /* The element count comes from the operand; the `numel` parameter is the band. */
    const int64_t nel = band_rows(numel(x), p_int(a, "numel", 0));
    const int world = (int)p_int(a, "world_size", a->world_size > 0 ? a->world_size : 1);
    if (nel <= 0 || numel(y) < nel * world) return RAD_E_SHAPE;
    if (world > 1) return RAD_E_UNSUPPORTED;

    /* `row` has no effect at world 1 but is still validated, so an invalid graph fails the same
     * way at world 1 as at world 2. */
    const int64_t row = p_int(a, "row", 0);
    if (row < 0 || (row > 0 && nel % row)) return RAD_E_SHAPE;

    if (rad_tensor_is_contiguous(x) && rad_tensor_is_contiguous(y) && x->dtype == y->dtype) {
        std::memcpy(y->data, x->data, (size_t)((nel * rad_dtype_bits(x->dtype) + 7) / 8));
        return RAD_OK;
    }
    for (int64_t i = 0; i < nel; ++i) stt(y, offlin(y, i), ldt(x, offlin(x, i)));
    return RAD_OK;
}
