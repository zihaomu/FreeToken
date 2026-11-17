/* ref_hc.cpp -- the gated residual, Qwen4-Exp's replacement for a pre-norm and a residual add.
 *
 * ============================== What it is ==============================
 *
 * The residual stream is `hc` times as wide as the model: hidden_size 2560 at hc_count 4 is a
 * 10240-wide stream through every layer, entered as four identical copies of the embedding and
 * collapsed once at the end. Each block reads a 2560 mix of the four and writes its output back
 * into all four; both the read's mix and the write's per-stream gain are data-dependent ("gated"),
 * so this is a pair of ops rather than a norm.
 *
 *   hc_read    N       = grouped_rmsnorm(h)              one rsqrt per 2560 stream, one gain of
 *                                                        10240 over all four, `1 + w`
 *              g       = sigmoid(W_up @ silu(W_down @ N / hc))
 *              x       = mean_over_streams(g * N)        the block's 2560 input
 *              inj[s]  = 2 * sigmoid(W_inj[s] . N / hc)  four scalars in (0, 2)
 *   hc_write   h[s]   += inj[s] * y                      for each stream s
 *
 * The write reads the unnormed `h`, not `N`: upstream computes `hyper_input + injection` with
 * `hyper_input` the read's argument, so the normalisation feeds only the block input and the gates.
 * Writing into `N` would re-normalise the residual at each of the 96 connections and diverge within
 * a few layers while still producing fluent text.
 *
 * The mix is a mean (`.mean(dim=-2)` over hc), and both gate arguments are separately divided by
 * hc (`/ self.hc_count` before the silu and before the inject sigmoid): three independent
 * divisions, all upstream's.
 *
 * ============================== Why the quantiser is in the read ==============================
 *
 * `q` and `s` are optional outputs. Present, the read also emits the fp8 block-scaled `x` the next
 * projection's GEMM wants, keeping the launch count equal to a pre-norm model's (each launch has a
 * fixed cost, and there are 96 connections). Absent, `x` is bf16 and the caller quantises it: the
 * final mixer feeds the bf16 lm_head.
 *
 * Codes are taken from `x` narrowed to its dtype, not from the f32 accumulator, as a fused kernel
 * must (the GDN out-projection quantiser follows the same rule, libref/ref_gdn.cpp).
 *
 * `rq` and `rs` extend this one consumer further: had_quant_act_fp8 of the stored `x`, for expert
 * planes stored with a 128-wide Hadamard.
 */
#include "ref_common.h"
#include "ref_ops.h"
#include "ref_planes.h"

#include <cmath>

using namespace ref;

/* ------------------------------------------------------------------ hc_enter */
/* The wide stream's initial value: hc identical copies of the embedding. Separate from the
 * embedding lookup, whose output width is n. The streams diverge at the first connection, since
 * each block writes into them with different per-stream gains. */
int ref_hc_enter(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor* x = &a->t[0];
    const RadTensor* h = &a->t[1];
    const int64_t n  = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(h) < rows * hc * n) return RAD_E_SHAPE;
    REF_PAR_FOR(rows * hc * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), hb = rowoff(h, r, hc * n);
        const int64_t xs = laststride(x), hs = laststride(h);
        for (int64_t i = 0; i < n; ++i) {
            const float v = ldt(x, xb + i * xs);
            for (int64_t c = 0; c < hc; ++c) stt(h, hb + (c * n + i) * hs, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_read */
int ref_hc_read(const RadArgs* a, RadStream) {
    /* Not `have(a, 6)`: operand 4, the inject weight, is optional (the final mixer has none).
     * Operands 0..3 and 5 are required. */
    if (a->n_t < 6) return RAD_E_INVAL;
    if (!rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 2) || !rad_arg_in(a, 3) ||
        !rad_arg_in(a, 5)) return RAD_E_INVAL;
    const RadTensor* h    = &a->t[0];
    const RadTensor* w    = &a->t[1];
    const RadTensor* down = &a->t[2];
    const RadTensor* up   = &a->t[3];
    const RadTensor* injw = t_in(a, 4);          /* absent on the final mixer */
    const RadTensor* x    = &a->t[5];
    const RadTensor* inj  = rad_arg_in(a, 6);    /* an OUT; same optional test */
    const RadTensor* q    = rad_arg_in(a, 7);
    const RadTensor* s    = rad_arg_in(a, 8);
    const RadTensor* rq   = rad_arg_in(a, 9);    /* the rotated fp8 form, for a w4a8h consumer */
    const RadTensor* rs   = rad_arg_in(a, 10);

    const int64_t n  = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    const int64_t lr = p_int(a, "lowrank", 0);
    const float  eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0 || hc <= 1 || lr <= 0) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = numel(h) / hn;
    if (rows <= 0 || numel(x) < rows * n || numel(w) < hn) return RAD_E_SHAPE;
    /* The mixing matrices are plain, or codes and a scale plane joined (ref_planes.h; `mix`
     * e4m3), read element by element as the core's decoder gives them. A plain matrix's dtype is
     * what the gate it feeds is rounded to; a joined one's gates are rounded to h's. */
    ScaledRows jd, ju;
    const bool jdown = is_joined(down), jup = is_joined(up);
    if (jdown) RAD_REF_TRY(scaled_rows(down, lr, hn, &jd));
    else if (numel(down) < lr * hn) return RAD_E_SHAPE;
    if (jup) RAD_REF_TRY(scaled_rows(up, hn, lr, &ju));
    else if (numel(up) < hn * lr) return RAD_E_SHAPE;
    const uint32_t ddt = jdown ? h->dtype : down->dtype, udt = jup ? h->dtype : up->dtype;
    if (injw && numel(injw) < hc * hn) return RAD_E_SHAPE;
    /* An inject weight without an inject output would compute scalars nobody reads; refused. */
    if (injw && (!inj || !inj->data || numel(inj) < rows * hc)) return RAD_E_SHAPE;

    int64_t g = 0;
    int64_t ngrp = 0;
    if (q && q->data) {
        g = p_int(a, "group", 0);
        if (g <= 0 || g > n) g = n;
        ngrp = (n + g - 1) / g;
        if (numel(q) < rows * n || !s || !s->data || numel(s) < rows * ngrp) return RAD_E_SHAPE;
    }
    /* The rotated form rotates each scale group whole, so the group must be a power of two that
     * tiles the row; no weight block was rotated across padding. The result is the
     * rotated quantiser's (had_quant_act_i8g's at int8 codes) of the stored `x`. */
    int64_t rg = 0;
    if (rq) {
        rg = p_int(a, "group", 0);
        if (!pow2(rg) || n % rg != 0 || !rs || numel(rq) < rows * n || numel(rs) < rows * (n / rg))
            return RAD_E_SHAPE;
        if (rg > REF_HAD_MAX) return RAD_E_UNSUPPORTED;
    }

    REF_PAR_FOR(rows * hn * (2 * lr + hc))
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = rowoff(h, r, hn), xb = rowoff(x, r, n);
        const int64_t hs = laststride(h), xs = laststride(x), ws = laststride(w);

        /* ---- grouped norm: one rsqrt per stream over its n, then the full-width gain. N is
         * rounded to h's dtype: upstream's norm returns `.type_as(x)`, so every product reads it
         * rounded. */
        std::vector<float> N((size_t)hn);
        for (int64_t c = 0; c < hc; ++c) {
            double ss = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const float v = ldt(h, hb + (c * n + i) * hs);
                ss += (double)v * (double)v;
            }
            const float sc = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
            for (int64_t i = 0; i < n; ++i) {
                const int64_t j = c * n + i;
                N[(size_t)j] = round_dt(h->dtype,
                                        ldt(h, hb + j * hs) * sc * (ldt(w, j * ws) + wadd));
            }
        }

        /* ---- read gate: hc*n -> lowrank -> hc*n, silu between and sigmoid after. */
        std::vector<float> d((size_t)lr);
        for (int64_t j = 0; j < lr; ++j) {
            float acc = 0.0f;
            const int64_t db = jdown ? 0 : rowoff(down, j, hn);
            const int64_t ds = laststride(down);
            for (int64_t i = 0; i < hn; ++i)
                acc += N[(size_t)i] * (jdown ? jd.at(j, i) : ldt(down, db + i * ds));
            d[(size_t)j] = round_dt(ddt, act_silu(acc / (float)hc));
        }

        /* ---- the gated mix, averaged over the streams. */
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int64_t c = 0; c < hc; ++c) {
                const int64_t row = c * n + i;
                const int64_t ub = jup ? 0 : rowoff(up, row, lr);
                const int64_t us = laststride(up);
                float gv = 0.0f;
                for (int64_t j = 0; j < lr; ++j)
                    gv += d[(size_t)j] * (jup ? ju.at(row, j) : ldt(up, ub + j * us));
                acc += round_dt(udt, act_sigmoid(gv)) * N[(size_t)row];
            }
            stt(x, xb + i * xs, acc / (float)hc);
        }

        /* ---- per-stream write gates. */
        if (injw && inj) {
            const int64_t ib = rowoff(inj, r, hc), is = laststride(inj);
            for (int64_t c = 0; c < hc; ++c) {
                float acc = 0.0f;
                const int64_t jb = rowoff(injw, c, hn), js = laststride(injw);
                for (int64_t i = 0; i < hn; ++i) acc += N[(size_t)i] * ldt(injw, jb + i * js);
                stt(inj, ib + c * is, 2.0f * act_sigmoid(acc / (float)hc));
            }
        }

        /* ---- optional fp8 form of x, from the stored values. */
        if (q && q->data) {
            const int64_t qb = rowoff(q, r, n), qs = laststride(q);
            for (int64_t gi = 0; gi < ngrp; ++gi) {
                const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
                const float sc = q8_group([&](int64_t i) { return ldt(x, xb + (k0 + i) * xs); },
                                           k1 - k0, q->dtype == RAD_I8 ? 127.0f : 448.0f, q,
                                           qb + k0 * qs, qs);
                stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
            }
        }

        /* ---- optional rotated form of x, from the same stored values: E4M3 codes, or int8
         * (had_quant_act_i8g's) where the buffer is. */
        if (rq) {
            float buf[REF_HAD_MAX];
            const int64_t qb = rowoff(rq, r, n), qs = laststride(rq), rgrp = n / rg;
            for (int64_t gi = 0; gi < rgrp; ++gi) {
                const int64_t k0 = gi * rg;
                for (int64_t i = 0; i < rg; ++i) buf[i] = ldt(x, xb + (k0 + i) * xs);
                fwht(buf, rg);
                const float sc = q8_group([&](int64_t i) { return buf[i]; }, rg,
                                           rq->dtype == RAD_I8 ? 127.0f : 448.0f, rq,
                                           qb + k0 * qs, qs);
                stt(rs, rowoff(rs, r, rgrp) + gi * laststride(rs), sc);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_write */
/* h[s] += inj[s] * y for each of the hc streams, in place on the stream the paired read was given
 * (which the read left unchanged). */
int ref_hc_write(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor* y   = &a->t[0];
    const RadTensor* inj = &a->t[1];
    const RadTensor* h   = &a->t[2];

    const int64_t n  = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = numel(y) / n;
    if (rows <= 0 || numel(h) < rows * hn || numel(inj) < rows * hc) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * hn)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t yb = rowoff(y, r, n), hb = rowoff(h, r, hn), ib = rowoff(inj, r, hc);
        const int64_t ys = laststride(y), hs = laststride(h), is = laststride(inj);
        for (int64_t c = 0; c < hc; ++c) {
            const float gain = ldt(inj, ib + c * is);
            for (int64_t i = 0; i < n; ++i) {
                const int64_t j = hb + (c * n + i) * hs;
                stt(h, j, ldt(h, j) + gain * ldt(y, yb + i * ys));
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_project */
/* The control vector's projection: h[row, s*n + i] -= scale * (h . v) * v, for each of the hc
 * streams s, on the rows whose `on` flag is 1.
 *
 * `v` is one layer's unit direction (the engine normalises the file's rows when it uploads them)
 * and `scale` is the norm the file stored for that layer, so this is exactly llama.cpp's
 * `--cvec-mode project` with its `llama_adapter_cvec` (the package's 02-cvec-projection-mode
 * patch): the dot is taken per hyper-connection stream, as `mul_mat` over a [n, hc, tokens] view
 * gives it, and not on the stream mean.
 *
 * A row whose flag is 0 is not touched: no read, no write, so an unprojected request's activations
 * are the stock model's. (A deployment that merely *has* the table loaded is not bit-identical to
 * one without it -- the table's VRAM moves the expert/KV plan -- but that is placement, not this
 * op.)
 *
 * `on` is an int32 per row (core derive.cpp's `projection_on_rows`); the direction is f32. */
int ref_hc_project(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor* h  = &a->t[0];
    const RadTensor* v  = &a->t[1];
    const RadTensor* on = &a->t[2];

    const int64_t n  = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    const float   sc = p_f32(a, "scale", 1.0f);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    if (numel(v) < n) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = rows_of(h, hn);
    if (rows <= 0 || numel(on) < rows) return RAD_E_SHAPE;

    /* Rows are independent: each dot is over one row's own streams. The parallel index is the
     * (row, stream) pair, so one row's hc streams may run on different threads without sharing an
     * accumulator. */
    REF_PAR_FOR(rows * hc)
    for (int64_t k = 0; k < rows * hc; ++k) {
        const int64_t r = k / hc, c = k % hc;
        if (ld_int(on->dtype, on->data, offlin(on, r)) == 0) continue;
        const int64_t hb = rowoff(h, r, hn), hs = laststride(h);
        const int64_t vb = rowoff(v, 0, n), vl = laststride(v);
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i)
            acc += (double) ldt(h, hb + (c * n + i) * hs) * (double) ldt(v, vb + i * vl);
        const float g = (float) (acc * (double) sc);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t j = hb + (c * n + i) * hs;
            stt(h, j, ldt(h, j) - g * ldt(v, vb + i * vl));
        }
    }
    return RAD_OK;
}

/* ================================================================== mtp_enter
 *
 * The multi-token-prediction head's entry for a model whose residual stream is `hc` wide.
 *
 * A pre-norm MTP head (Qwen3.5, Qwen3-Next, DeepSeek-V3) joins the trunk's hidden state and the
 * next token's embedding with one `fc` over their concatenation (arch/common/rad_block_mtp_fp8.h:
 * two rmsnorms into one buffer, one gemm_nt). Qwen4-Exp's hidden state is the wide stream and the
 * checkpoint carries two square matrices,
 *
 *     mtp.fc_hidden.weight        [n, n]        mtp.pre_fc_norm_hidden.weight       [hc * n]
 *     mtp.fc_embedding.weight     [n, n]        mtp.pre_fc_norm_embedding.weight    [n]
 *
 * so the hidden half is `fc_h` applied to each of the hc sub-streams (upstream's
 * `.unflatten(-1, (hc_count, hidden_size))` idiom) and the embedding half is one n-wide product
 * added into all of them, the broadcast the trunk enters the stream with (`hidden_states.repeat`).
 *
 *     rs_e      = rsqrt(mean(e^2) + eps)
 *     ne[j]     = e[j] * rs_e * (w_e[j] + wadd)
 *     fe[i]     = sum_j fc_e[i][j] * ne[j]
 *     rs[k]     = rsqrt(mean(h[k*n .. k*n+n)^2) + eps)          -- `ngroup` == hc
 *     nh[k][j]  = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)
 *     x[k*n+i]  = sum_j fc_h[i][j] * nh[k][j] + fe[i]
 *
 * One op rather than six because the intermediates are different views of the same bytes and the
 * ABI has no reinterpreting view: `nh` is [M*hc, n] to the GEMM and [M, hc*n] to the norm, while
 * core/runtime/issue.cpp only narrows dim 0 or the last dimension with strides unchanged, so a
 * row-pitch change is inexpressible. It also saves launches in the latency-critical draft round.
 *
 * `ngroup` is the one thing the checkpoint does not say: a [hc * n] gain fits both a grouped norm
 * (hc rsqrts, one per sub-stream, as Qwen4ExpTextGatedResidual does) and a flat one (one rsqrt over
 * the wide row). No per-op oracle can tell them apart; the draft acceptance rate can. `ngroup`
 * makes it a declare-time choice.
 */
int ref_mtp_enter(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    const RadTensor* h    = &a->t[0];
    const RadTensor* e    = &a->t[1];
    const RadTensor* w_h  = &a->t[2];
    const RadTensor* w_e  = &a->t[3];
    const RadTensor* fc_h = &a->t[4];
    const RadTensor* fc_e = &a->t[5];
    const RadTensor* x    = &a->t[6];

    const int64_t n   = p_int(a, "n", 0);
    const int64_t hc  = p_int(a, "hc", 0);
    const float  eps  = p_f32(a, "eps", 1e-6f);
    const float  wadd = p_f32(a, "wadd", 0.0f);
    int64_t ngroup    = p_int(a, "ngroup", hc);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    if (ngroup != 1 && ngroup != hc) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = numel(h) / hn;
    if (rows <= 0 || numel(e) < rows * n || numel(x) < rows * hn) return RAD_E_SHAPE;
    if (numel(w_h) < hn || numel(w_e) < n) return RAD_E_SHAPE;
    /* The fc matrices are plain, or codes and a scale plane joined (`fc` e4m3), as hc_read's. */
    ScaledRows jh, je;
    const bool jfh = is_joined(fc_h), jfe = is_joined(fc_e);
    if (jfh) RAD_REF_TRY(scaled_rows(fc_h, n, n, &jh));
    else if (numel(fc_h) < n * n) return RAD_E_SHAPE;
    if (jfe) RAD_REF_TRY(scaled_rows(fc_e, n, n, &je));
    else if (numel(fc_e) < n * n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * hn * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = rowoff(h, r, hn), eb = rowoff(e, r, n), xb = rowoff(x, r, hn);
        const int64_t hs = laststride(h), es = laststride(e), xs = laststride(x);
        const int64_t whs = laststride(w_h), wes = laststride(w_e);

        /* ---- embedding half: one norm and one n-wide product, shared by every stream. */
        double ss = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const float v = ldt(e, eb + i * es);
            ss += (double)v * (double)v;
        }
        const float rse = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
        std::vector<float> ne((size_t)n);
        for (int64_t i = 0; i < n; ++i)
            ne[(size_t)i] = round_dt(e->dtype,
                                     ldt(e, eb + i * es) * rse * (ldt(w_e, i * wes) + wadd));

        std::vector<float> fe((size_t)n);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            const int64_t fb = jfe ? 0 : rowoff(fc_e, i, n);
            const int64_t fs = laststride(fc_e);
            for (int64_t j = 0; j < n; ++j)
                acc += ne[(size_t)j] * (jfe ? je.at(i, j) : ldt(fc_e, fb + j * fs));
            fe[(size_t)i] = acc;
        }

        /* ---- hidden half: the norm (grouped, or flat per `ngroup`), then the same square matrix
         * over each stream. */
        float flat = 0.0f;
        if (ngroup == 1) {
            double sw = 0.0;
            for (int64_t j = 0; j < hn; ++j) {
                const float v = ldt(h, hb + j * hs);
                sw += (double)v * (double)v;
            }
            flat = 1.0f / std::sqrt((float)(sw / (double)hn) + eps);
        }

        std::vector<float> nh((size_t)n);
        for (int64_t c = 0; c < hc; ++c) {
            float sc = flat;
            if (ngroup != 1) {
                double sk = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    const float v = ldt(h, hb + (c * n + i) * hs);
                    sk += (double)v * (double)v;
                }
                sc = 1.0f / std::sqrt((float)(sk / (double)n) + eps);
            }
            for (int64_t i = 0; i < n; ++i) {
                const int64_t j = c * n + i;
                nh[(size_t)i] = round_dt(h->dtype,
                                         ldt(h, hb + j * hs) * sc * (ldt(w_h, j * whs) + wadd));
            }
            for (int64_t i = 0; i < n; ++i) {
                float acc = 0.0f;
                const int64_t fb = jfh ? 0 : rowoff(fc_h, i, n);
                const int64_t fs = laststride(fc_h);
                for (int64_t j = 0; j < n; ++j)
                    acc += nh[(size_t)j] * (jfh ? jh.at(i, j) : ldt(fc_h, fb + j * fs));
                stt(x, xb + (c * n + i) * xs, acc + fe[(size_t)i]);
            }
        }
    }
    return RAD_OK;
}
