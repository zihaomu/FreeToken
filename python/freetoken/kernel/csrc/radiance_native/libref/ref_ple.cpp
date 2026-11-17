/* ref_ple.cpp -- the gate of Qwen4-Exp's Per-Layer Embedding: everything between the n-gram
 * lookup's two projections and the dilated convolution.
 *
 * ============================== What it is ==============================
 *
 * PLE mixes a hashed n-gram embedding into the residual stream with a mixing coefficient that is a
 * dot product between the embedding and the stream: one scalar per hyper-connection stream per
 * token. With `n` the model width and `hc` the number of streams:
 *
 *   K[c] = grouped_rmsnorm(k)[c] * (w_key[c]   + wadd)      k is key_proj(embedding)
 *   Q[c] = grouped_rmsnorm(q)[c] * (w_query[c] + wadd)      q is the incoming hc stream
 *   s[c] = sum_d K[c][d] * Q[c][d] / sqrt(n)
 *   s[c] = sign(s[c]) * sqrt(max(|s[c]|, gate_eps))                 <- the signed square root
 *   gv[c] = sigmoid(s[c]) * v                               v is value_proj(embedding), one row
 *   gvn   = grouped_rmsnorm(gv) * (w_conv + wadd)
 *
 * and the caller finishes with `out = gv + silu(dilated_conv(gvn))`. Both gv and gvn are outputs:
 * the convolution reads the normed copy and the residual add the un-normed one.
 *
 * ============================== The signed square root ==============================
 *
 * `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()` compresses a score spanning orders of magnitude
 * into a range sigmoid can resolve and keeps the sign, so a stream whose embedding opposes the
 * residual gets a gate below a half.
 *
 * The clamp applies to the magnitude and the sign comes from the original value. torch's `sign(0)`
 * is 0, so an exactly-zero dot stays zero rather than becoming sqrt(gate_eps) = 1e-3; an fp8
 * activation row can produce exact zero.
 *
 * ============================== One op, not six ==============================
 *
 * Three grouped norms, a reduction, a scalar nonlinearity and a broadcast multiply would be six
 * memory-bound launches over [M, hc*n] tensors (10240 wide at this model's shape). `hc_read` is
 * fused for the same reason.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <cmath>

using namespace ref;

namespace {

/* One group's reciprocal RMS: the rsqrt is per `n`-wide stream while the gain spans all hc*n
 * (what `group_size` means on Qwen4ExpTextRMSNorm). */
inline float group_rsqrt(const RadTensor* x, int64_t base, int64_t stride, int64_t n, float eps) {
    double acc = 0;
    for (int64_t d = 0; d < n; ++d) {
        const double v = ldt(x, base + d * stride);
        acc += v * v;
    }
    return (float)(1.0 / std::sqrt(acc / (double)n + (double)eps));
}

}  /* namespace */

/* PLE's gate.
 *
 *   k        [M][hc*n]   key_proj(embedding), un-normed
 *   q        [M][hc*n]   the incoming hyper-connection stream, un-normed
 *   v        [M][n]      value_proj(embedding): one row, gated differently per stream
 *   w_key    [hc*n]   w
 *   w_query  [hc*n]   w
 *   w_conv   [hc*n]   w
 *   gv       [M][hc*n]   out, un-normed: what the residual add reads
 *   gvn      [M][hc*n]   out, normed:    what the convolution reads
 */
int ref_ple_gate(const RadArgs* a, RadStream) {
    if (!have(a, 8)) return RAD_E_INVAL;
    const RadTensor* k   = t_in(a, 0);
    const RadTensor* q   = t_in(a, 1);
    const RadTensor* v   = t_in(a, 2);
    const RadTensor* wk  = t_in(a, 3);
    const RadTensor* wq  = t_in(a, 4);
    const RadTensor* wc  = t_in(a, 5);
    const RadTensor* gv  = t_in(a, 6);
    const RadTensor* gvn = t_in(a, 7);
    if (!k || !q || !v || !wk || !wq || !wc || !gv || !gvn) return RAD_E_INVAL;

    const int64_t n  = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    /* Rows come from `v` [M, n]; `M` is the band (see band_rows in ref_common.h). */
    const int64_t M  = n > 0 ? band_rows(numel(v) / n, p_int(a, "M", 0)) : 0;
    const float eps  = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    /* The floor the gate's magnitude is clamped to before the square root; a model constant. */
    const float geps = p_f32(a, "gate_eps", 1e-6f);
    if (M <= 0 || n <= 0 || hc <= 0) return RAD_E_SHAPE;
    const int64_t hn = hc * n;
    if (numel(k) < M * hn || numel(q) < M * hn || numel(v) < M * n) return RAD_E_SHAPE;
    if (numel(wk) < hn || numel(wq) < hn || numel(wc) < hn) return RAD_E_SHAPE;
    if (numel(gv) < M * hn || numel(gvn) < M * hn) return RAD_E_SHAPE;

    const int64_t kls = laststride(k), qls = laststride(q), vls = laststride(v);
    const int64_t wks = laststride(wk), wqs = laststride(wq), wcs = laststride(wc);
    const int64_t gls = laststride(gv), gnls = laststride(gvn);

    REF_PAR_FOR(M * hn * 4)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t kb = rowoff(k, m, hn), qb = rowoff(q, m, hn), vb = rowoff(v, m, n);
        const int64_t gb = rowoff(gv, m, hn), nb = rowoff(gvn, m, hn);

        for (int64_t c = 0; c < hc; ++c) {
            /* The two normed streams and their dot, not materialised since each element is used
             * once. Each norm is rounded to its input's dtype: upstream's grouped norm returns
             * `.type_as(x)`. */
            const float rk = group_rsqrt(k, kb + c * n * kls, kls, n, eps);
            const float rq = group_rsqrt(q, qb + c * n * qls, qls, n, eps);
            double dot = 0;
            for (int64_t d = 0; d < n; ++d) {
                const int64_t j = c * n + d;
                const float kk = round_dt(k->dtype, ldt(k, kb + j * kls) * rk *
                                                    (ldt(wk, j * wks) + wadd));
                const float qq = round_dt(q->dtype, ldt(q, qb + j * qls) * rq *
                                                    (ldt(wq, j * wqs) + wadd));
                dot += (double)kk * (double)qq;
            }
            float s = (float)(dot / std::sqrt((double)n));

            /* Signed square root in upstream's order: clamp the magnitude, take the root, sign
             * from the original (sign(0) = 0). */
            const float mag = std::sqrt(std::fabs(s) < geps ? geps : std::fabs(s));
            s = s > 0.0f ? mag : (s < 0.0f ? -mag : 0.0f);

            const float g = act_sigmoid(s);
            for (int64_t d = 0; d < n; ++d)
                stt(gv, gb + (c * n + d) * gls, round_dt(gv->dtype, g * ldt(v, vb + d * vls)));
        }

        /* The third norm reads `gv` back after it was narrowed, since the convolution's input is a
         * norm of the stored bytes (the GDN out-projection's epilogue-fold rule). */
        for (int64_t c = 0; c < hc; ++c) {
            const float r = group_rsqrt(gv, gb + c * n * gls, gls, n, eps);
            for (int64_t d = 0; d < n; ++d) {
                const int64_t j = c * n + d;
                stt(gvn, nb + j * gnls,
                    round_dt(gvn->dtype, ldt(gv, gb + j * gls) * r * (ldt(wc, j * wcs) + wadd)));
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ple_conv */
/* PLE's depthwise convolution: width 4 at dilation 3, silu after it, and the residual add folded
 * in; the layer's last piece, `out = gv + silu(conv(gvn))`.
 *
 * ============================== Dilation ==============================
 *
 * A dilated causal convolution reads `width` taps spaced `dilation` apart:
 *
 *     y[t] = silu( sum_j w[c][j] * x[t - (width-1-j)*dilation] )
 *
 * so at Qwen4-Exp's width 4 and dilation 3 the taps are 9, 6, 3 and 0 timesteps back and the
 * carried history is (width-1)*dilation = 9, not 3. The indexing mirrors gdn_conv_prep's: state
 * slot `i` holds relative position `i - state_len`, and a read at `src < 0` becomes
 * `cs[slot][c][src + state_len]`.
 *
 * Upstream pads by `state_len`, slices the last `state_len + seq_len` and convolves unpadded; on a
 * warm cache that is a no-op, and on a cold one the window before the first token is zeros.
 *
 * ============================== One op where the GDN has two ==============================
 *
 * gdn_conv_prep and gdn_conv_update differ in the work around the convolution. PLE's convolution is
 * the same at prefill and decode, so one op takes both contracts: `has_init` says whether the slot
 * holds history, and `num_accepted` shifts the read offset within a deeper window.
 *
 * `num_accepted` absent means every token commits (a prefill chunk): the read is at offset zero.
 * Unlike gdn_conv_update it is optional because this op also serves prefill. The state must still
 * be sized (width-1)*dilation + n_spec, and a window too shallow for the requested offset is
 * refused, not clamped.
 *
 * ============================== The window after a step ==============================
 *
 * The next step reads at its `num_accepted - 1` and must find the `hist` values ending at the last
 * token this step committed. The presence of `num_accepted` picks the layout:
 *
 *   present -- a decode step of `slen` tokens, any past the first rejectable by verify.
 *     The window is rewritten the way gdn_conv_update rewrites its own: slot i holds position
 *     i + 1 of (the window this step read ++ this step's inputs), so the old history shifted down
 *     by one and then token t at slot hist - 1 + t. A next step that kept k of these tokens reads
 *     at k - 1 and finds the history ending at token k - 1. It needs hist - 1 + slen slots, which
 *     is the manager's hist + n_spec for a step of 1 + n_spec tokens.
 *
 *   absent -- every token commits, so the next step reads at offset zero (`num_accepted` 1) and
 *     the window is the step's last `hist` values.
 *
 * The two agree for a one-token step. After a multi-token verify, the absent layout would give a
 * step that accepted only the first token a history ending at a rejected draft.
 */
/* The depthwise causal convolution and its rolling window, shared by `ple_conv` (dilated, silu,
 * optional residual) and `causal_conv1d` (LFM2's short convolution: undilated, no activation, no
 * residual), so the two agree on the window layout. */
struct ConvArgs {
    const RadTensor *x, *w, *cs, *cu, *resid, *sidx, *hini, *nacc, *y;
    bool silu;
    const RadTensor* b;   /* per-channel bias inside the activation (Mamba-2's conv1d), or null */
};

static int depthwise_causal_conv(const RadArgs* a, const ConvArgs& o) {
    const RadTensor *x = o.x, *w = o.w, *cs = o.cs, *cu = o.cu, *resid = o.resid;
    const RadTensor *sidx = o.sidx, *hini = o.hini, *nacc = o.nacc, *y = o.y, *b = o.b;
    if (!x || !w || !cs || !cu || !y) return RAD_E_INVAL;

    const int64_t n    = p_int(a, "n", 0);
    const int64_t wid  = p_int(a, "width", 0);
    /* Rows come from `x` [M, n]; `M` is the band (see band_rows in ref_common.h). */
    const int64_t M    = n > 0 ? band_rows(numel(x) / n, p_int(a, "M", 0)) : 0;
    const int64_t dil  = p_int(a, "dilation", 1);
    if (M <= 0 || n <= 0 || wid <= 0 || dil <= 0) return RAD_E_SHAPE;
    const int64_t hist_len = (wid - 1) * dil;
    /* The read window lives in a stack frame (9 at this model's shape); wider is refused. */
    enum { kHistMax = 64 };
    if (hist_len > kHistMax) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    if (numel(x) < M * n || numel(y) < M * n || numel(w) < n * wid) return RAD_E_SHAPE;
    if (resid && numel(resid) < M * n) return RAD_E_SHAPE;
    if (b && numel(b) != n) return RAD_E_SHAPE;

    const int64_t state_len = cs->rank >= 3 ? cs->shape[cs->rank - 1] : hist_len;
    if (cs->data && state_len < hist_len) return RAD_E_SHAPE;

    /* Check every sequence before writing any: a state too shallow for the requested offset or for
     * a decode step's resulting window is a caller bug that would silently truncate history. */
    for (int64_t s = 0; s < N; ++s) {
        const int64_t off = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, s)) - 1 : 0;
        if (off < 0) return RAD_E_INVAL;
        if (off + hist_len > state_len) return RAD_E_SHAPE;
        const int64_t slen = cu_at(cu, s + 1) - cu_at(cu, s);
        if (nacc && cs->data && hist_len > 0 && slen > 0 && hist_len - 1 + slen > state_len)
            return RAD_E_SHAPE;
    }

    const int64_t xrow = rowoff(x, 1, n), xcol = laststride(x);
    const int64_t yrow = rowoff(y, 1, n), ycol = laststride(y);
    const int64_t rrow = resid ? rowoff(resid, 1, n) : 0;
    const int64_t rcol = resid ? laststride(resid) : 0;
    const int64_t wrow = rowoff(w, 1, wid), wcol = laststride(w);
    /* [n_slots, n, state_len], as gdn's conv state; a flattened one is read as the tight layout. */
    const bool cs3 = cs->rank >= 3;
    const int64_t cs_slot = cs3 ? cs->stride[cs->rank - 3] : n * state_len;
    const int64_t cs_ch = cs3 ? cs->stride[cs->rank - 2] : state_len;
    const int64_t cs_t = cs3 ? cs->stride[cs->rank - 1] : 1;

    REF_PAR_FOR(M * n * wid)
    for (int64_t sc = 0; sc < N * n; ++sc) {
        const int64_t s = sc / n, c = sc % n;
        const int64_t bos = cu_at(cu, s);
        const int64_t slen = cu_at(cu, s + 1) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ld_int(sidx->dtype, sidx->data, offlin(sidx, s)) : s;
        if (slot < 0) continue;
        /* A zero `has_init` means the slot holds no history: the window before the first token
         * is zeros, not a recycled slot's previous contents. */
        const bool warm = !hini || ld_int(hini->dtype, hini->data, offlin(hini, s)) != 0;
        const int64_t off = nacc ? ld_int(nacc->dtype, nacc->data, offlin(nacc, s)) - 1 : 0;
        const int64_t base = slot * cs_slot + c * cs_ch;

        /* Read whole before anything is written back, since a step longer than the window
         * overwrites history it still reads. */
        float hist[kHistMax];
        for (int64_t i = 0; i < hist_len; ++i)
            hist[i] = (warm && cs->data) ? ldt(cs, base + (off + i) * cs_t) : 0.0f;

        const float bias = b ? ldt(b, c * laststride(b)) : 0.0f;
        for (int64_t t = 0; t < slen; ++t) {
            float acc = bias;
            for (int64_t j = 0; j < wid; ++j) {
                const int64_t src = t - hist_len + j * dil;
                const float v = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                         : hist[src + hist_len];
                acc += v * ldt(w, c * wrow + j * wcol);
            }
            const float r = resid ? ldt(resid, (bos + t) * rrow + c * rcol) : 0.0f;
            stt(y, (bos + t) * yrow + c * ycol, r + (o.silu ? act_silu(acc) : acc));
        }

        /* The window after the step, in the layout described above ple_conv. */
        if (!cs->data || hist_len == 0) continue;
        if (nacc) {
            for (int64_t i = 0; i + 1 < hist_len; ++i) stt(cs, base + i * cs_t, hist[i + 1]);
            for (int64_t t = 0; t < slen; ++t)
                stt(cs, base + (hist_len - 1 + t) * cs_t, ldt(x, (bos + t) * xrow + c * xcol));
        } else {
            float keep[kHistMax];
            for (int64_t i = 0; i < hist_len; ++i) {
                const int64_t src = slen - hist_len + i;
                keep[i] = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol) : hist[src + hist_len];
            }
            for (int64_t i = 0; i < hist_len; ++i) stt(cs, base + i * cs_t, keep[i]);
        }
    }
    return RAD_OK;
}

int ref_ple_conv(const RadArgs* a, RadStream) {
    /* Nine positions, four optional; `have()` would refuse an absent optional operand. */
    if (a->n_t < 9) return RAD_E_INVAL;
    return depthwise_causal_conv(a, { t_in(a, 0), t_in(a, 1), t_in(a, 2), t_in(a, 3), t_in(a, 4),
                                      t_in(a, 5), t_in(a, 6), t_in(a, 7), t_in(a, 8), true });
}

/* ple_conv's operands with `b`, a per-channel bias inside the silu, before `y`: Mamba-2's
 * conv1d at dilation one. */
int ref_ple_conv_bias(const RadArgs* a, RadStream) {
    if (a->n_t < 10 || !t_in(a, 8)) return RAD_E_INVAL;
    return depthwise_causal_conv(a, { t_in(a, 0), t_in(a, 1), t_in(a, 2), t_in(a, 3), t_in(a, 4),
                                      t_in(a, 5), t_in(a, 6), t_in(a, 7), t_in(a, 9), true,
                                      t_in(a, 8) });
}

/* ple_conv's operands without `resid`, at dilation one and with no activation. */
int ref_causal_conv1d(const RadArgs* a, RadStream) {
    if (a->n_t < 8) return RAD_E_INVAL;
    if (p_int(a, "dilation", 1) != 1) return RAD_E_INVAL;
    return depthwise_causal_conv(a, { t_in(a, 0), t_in(a, 1), t_in(a, 2), t_in(a, 3), nullptr,
                                      t_in(a, 4), t_in(a, 5), t_in(a, 6), t_in(a, 7), false });
}
