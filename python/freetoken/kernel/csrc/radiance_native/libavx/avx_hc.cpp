/* avx_hc.cpp -- the gated residual pair (hc_read / hc_write) and the MTP head's entry.
 *
 * Gated residual: Qwen4-Exp's replacement for pre-norm + residual add. The residual stream is `hc`
 * times the model width (hidden 2560 at hc 4 is a 10240-wide stream), entered as hc copies of the
 * embedding and collapsed once at the end. Each block reads a mix of the streams and writes its
 * output back into all of them; both the mix and the per-stream write gain are data-dependent.
 *
 *   hc_read    N       = grouped_rmsnorm(h)              one rsqrt per n-wide stream, one gain of
 *                                                        hc*n over all of them, `1 + w`
 *              g       = sigmoid(W_up @ silu(W_down @ N / hc))
 *              x       = mean_over_streams(g * N)
 *              inj[s]  = 2 * sigmoid(W_inj[s] . N / hc)
 *   hc_write   h[s]   += inj[s] * y
 *
 * Three independent facts any implementation must match (docs/OPS.md); each fails silently:
 *   - The write updates the unnormed `h`, not `N`. Writing into `N` re-normalises the residual at
 *     every connection and diverges within a few layers while still producing fluent text.
 *   - The mix is a mean over the hc dimension, not a sum.
 *   - Each gate argument has its own `/ hc` (before the silu and before the inject sigmoid).
 *
 * Rounding: the reference model's grouped norm returns `.type_as(x)`, so both matrix products and
 * the final multiply read a bf16 `N`. As in libref, N is rounded through h's dtype (per row here,
 * via the row converter); likewise the silu output, the sigmoid gate, and `x` before the optional
 * fp8 quantiser reads it back.
 *
 * hc_read runs as three parallel passes over the mixing matrices' rows, each row decoded in the
 * kernel (E4M3 codes times their scales, or a plain row widened) into the thread's own buffer and
 * met by every token row before the next: `down` [lowrank, hc*n] over its rows, then `up` [hc*n,
 * lowrank] over the n output columns (hc rows each). A decode step reads each matrix once, spread
 * over the team, and no widened copy of either is written. The token rows' norms, gates and mixes
 * live in one scratch_raw block between the passes, since a second scratch_raw call may
 * reallocate the first while workers read it.
 */
#include "avx_vec.h"
#include "avx_common.h"
#include "avx_planes.h"

using namespace avx;

#include "avx_had.h"

/* Row r of a weight plane `cols` wide as f32, addressed as libref does (rowoff / laststride) so
 * strided or rank-3 weights work. A joined plane (codes and a scale, avx_planes.h) decodes; `jr`
 * is its view. */
static void plane_row(const RadTensor* t, const avxp::ScaledRows* jr, int64_t r, int64_t cols,
                      float* dst) {
    if (jr) {
        scaled_row(*jr, r, cols, dst);
        return;
    }
    const int64_t st = laststride(t), b = rowoff(t, r, cols);
    if (st == 1) row_to_f32(t->dtype, byte_at(t->data, t->dtype, b), dst, cols);
    else for (int64_t i = 0; i < cols; ++i) dst[i] = ldt(t, b + i * st);
}


/* Sum of squares in f64, unlike avx_norm.cpp's f32, matching libref's split: `rmsnorm` is checked
 * against fast kernels and accumulates in f32, while this norm is an internal step of a fused op.
 * Scalar: one reduction per stream, dominated by the two matrix products. */
static float rsqrt_meansq_f64(const float* v, int64_t n, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += (double)v[i] * (double)v[i];
    return 1.0f / std::sqrt((float)(ss / (double)n) + eps);
}

/* ------------------------------------------------------------------ hc_enter */
/* The wide stream's initial value: hc identical copies of the embedding, one launch per step. A
 * separate op because the embedding lookup's output width is n. The streams diverge at the first
 * connection, where each gets its own write gain. */
AVX_KERNEL(hc_enter) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor* x = &a->t[0];
    const RadTensor* h = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(h) < rows * hc * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * hc * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), hb = rowoff(h, r, hc * n);
        const int64_t hs = laststride(h);
        float* buf = scratch_f32((size_t)n);
        const float* v = in_row(x, xb, n, buf);
        /* One contiguous stream at a time (libref interleaves streams, writing at stride n). */
        if (hs == 1) for (int64_t c = 0; c < hc; ++c) out_row(h, hb + c * n, n, v);
        else for (int64_t c = 0; c < hc; ++c)
                 for (int64_t i = 0; i < n; ++i) stt(h, hb + (c * n + i) * hs, v[i]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_read */
AVX_KERNEL(hc_read) {
    /* Not `rad_args_have(a, 6)`: operand 4 (the inject weight) is optional and absent on the
     * final mixer. Operands 0..3 and 5 are required. */
    if (a->n_t < 6) return RAD_E_INVAL;
    if (!rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 2) || !rad_arg_in(a, 3) ||
        !rad_arg_in(a, 5)) return RAD_E_INVAL;
    const RadTensor* h = &a->t[0];
    const RadTensor* w = &a->t[1];
    const RadTensor* down = &a->t[2];
    const RadTensor* up = &a->t[3];
    const RadTensor* injw = rad_arg_in(a, 4);      /* absent on the final mixer */
    const RadTensor* x = &a->t[5];
    const RadTensor* inj = rad_arg_in(a, 6);
    const RadTensor* q = rad_arg_in(a, 7);
    const RadTensor* s = rad_arg_in(a, 8);
    const RadTensor* rq = rad_arg_in(a, 9);     /* rotated form, for a w4a8h consumer */
    const RadTensor* rs = rad_arg_in(a, 10);

    const int64_t n = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    const int64_t lr = p_int(a, "lowrank", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0 || hc <= 1 || lr <= 0) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = numel(h) / hn;
    if (rows <= 0 || numel(x) < rows * n || numel(w) < hn) return RAD_E_SHAPE;
    /* Plain mixing matrices, or codes and a scale joined (`mix` e4m3), libref's reading: a plain
     * matrix's dtype is what its gate is narrowed to, a joined one's gates are narrowed to h's. */
    avxp::ScaledRows jd, ju;
    const bool jdown = avxp::is_joined(down), jup = avxp::is_joined(up);
    if (jdown) {
        if (const int st = avxp::scaled_rows(down, lr, hn, &jd); st != RAD_OK) return st;
    } else if (numel(down) < lr * hn) {
        return RAD_E_SHAPE;
    }
    if (jup) {
        if (const int st = avxp::scaled_rows(up, hn, lr, &ju); st != RAD_OK) return st;
    } else if (numel(up) < hn * lr) {
        return RAD_E_SHAPE;
    }
    const uint32_t ddt = jdown ? h->dtype : down->dtype, udt = jup ? h->dtype : up->dtype;
    if (injw && numel(injw) < hc * hn) return RAD_E_SHAPE;
    /* An inject weight without an `inj` output (consumed by hc_write) is a caller error. */
    if (injw && (!inj || !inj->data || numel(inj) < rows * hc)) return RAD_E_SHAPE;

    int64_t g = 0, ngrp = 0;
    if (q && q->data) {
        g = p_int(a, "group", 0);
        if (g <= 0 || g > n) g = n;
        ngrp = (n + g - 1) / g;
        if (numel(q) < rows * n || !s || !s->data || numel(s) < rows * ngrp) return RAD_E_SHAPE;
    }
    /* The rotated form is had_quant_act_fp8 of the stored `x`, or had_quant_act_i8g into an int8
     * buffer: each scale group is rotated whole, so the group must be a power of two that tiles
     * the row. */
    int64_t rg = 0;
    if (rq && rq->data) {
        rg = p_int(a, "group", 0);
        if (!pow2(rg) || n % rg != 0 || !rs || !rs->data || numel(rq) < rows * n ||
            numel(rs) < rows * (n / rg))
            return RAD_E_SHAPE;
        if (rg > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    }

    /* The gain, the inject rows (hc of them, widened once), and the token rows' norms N [rows][hn],
     * gates D [rows][lr] and mixes X [rows][n], in one scratch_raw block (see header). */
    auto pad = [](size_t v) { return (v + 15) & ~(size_t)15; };
    const size_t o_gain = 0;
    const size_t o_inj  = pad(o_gain + (size_t)hn);
    const size_t o_N    = pad(o_inj + (injw ? (size_t)hc * (size_t)hn : 0));
    const size_t o_D    = pad(o_N + (size_t)rows * (size_t)hn);
    const size_t o_X    = pad(o_D + (size_t)rows * (size_t)lr);
    const size_t total  = o_X + (size_t)rows * (size_t)n + 16;
    float* blk = (float*)scratch_raw(total * sizeof(float));
    float* gain = blk + o_gain;
    float* iw = blk + o_inj;
    float* NP = blk + o_N;
    float* DP = blk + o_D;
    float* XP = blk + o_X;
    {
        const int64_t ws = laststride(w);
        if (ws == 1) row_to_f32(w->dtype, w->data, gain, hn);
        else for (int64_t i = 0; i < hn; ++i) gain[i] = ldt(w, i * ws);
        if (wadd != 0.0f) for (int64_t i = 0; i < hn; ++i) gain[i] += wadd;
    }
    if (injw)
        for (int64_t c = 0; c < hc; ++c) plane_row(injw, nullptr, c, hn, iw + c * hn);
    const avxp::ScaledRows* jdp = jdown ? &jd : nullptr;
    const avxp::ScaledRows* jup_p = jup ? &ju : nullptr;
    const float invhc = 1.0f / (float)hc;

    /* ---- grouped norm: one rsqrt per stream, the full-width gain, then one round trip through
     * h's dtype for the whole wide row. */
    AVX_PARALLEL_FOR_IF(rows >= 2)
    for (int64_t r = 0; r < rows; ++r) {
        float* N = NP + r * hn;
        float* tmp = scratch_f32((size_t)hn);
        const float* hv = in_row(h, rowoff(h, r, hn), hn, tmp);
        for (int64_t c = 0; c < hc; ++c) {
            const float sc = rsqrt_meansq_f64(hv + c * n, n, eps);
            const vf vs = vf_set1(sc);
            int64_t i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(N + c * n + i,
                          vf_mul(vf_mul(vf_loadu(hv + c * n + i), vs), vf_loadu(gain + c * n + i)));
            for (; i < n; ++i) N[c * n + i] = hv[c * n + i] * sc * gain[c * n + i];
        }
        if (h->dtype != RAD_F32) {
            row_from_f32(h->dtype, N, tmp, hn);
            row_to_f32(h->dtype, tmp, N, hn);
        }
    }

    /* ---- the read gate's first product, hc*n -> lowrank, and the silu, one row of `down` at a
     * time against every token row. */
    AVX_PARALLEL_FOR_IF(rows * lr * hn >= 65536)
    for (int64_t j = 0; j < lr; ++j) {
        float* wr = scratch_f32((size_t)hn);
        plane_row(down, jdp, j, hn, wr);
        for (int64_t r = 0; r < rows; ++r) {
            const float t = dotf(NP + r * hn, wr, hn) / (float)hc;
            DP[r * lr + j] = narrow1(ddt, t / (1.0f + sc_exp(-t)));    /* silu */
        }
    }

    /* ---- the second product, lowrank -> hc*n, its sigmoid, and the mix: the mean over the
     * streams. Output column i reads `up` rows c*n + i of every stream c. */
    AVX_PARALLEL_FOR_IF(rows * lr * hn >= 65536)
    for (int64_t i = 0; i < n; ++i) {
        float* ur = scratch_f32((size_t)(hc * lr));
        for (int64_t c = 0; c < hc; ++c) plane_row(up, jup_p, c * n + i, lr, ur + c * lr);
        for (int64_t r = 0; r < rows; ++r) {
            const float* N = NP + r * hn;
            float acc = 0.0f;
            for (int64_t c = 0; c < hc; ++c) {
                const float gv = dotf(DP + r * lr, ur + c * lr, lr);
                acc += narrow1(udt, 1.0f / (1.0f + sc_exp(-gv))) * N[c * n + i];
            }
            XP[r * n + i] = acc * invhc;
        }
    }

    AVX_PARALLEL_FOR_IF(rows >= 2)
    for (int64_t r = 0; r < rows; ++r) {
        float* N = NP + r * hn;
        float* xv = XP + r * n;
        float* tmp = scratch_f32((size_t)hn);
        out_row(x, rowoff(x, r, n), n, xv);

        /* ---- the per-branch write gates. */
        if (injw && inj) {
            const int64_t ib = rowoff(inj, r, hc), is = laststride(inj);
            for (int64_t c = 0; c < hc; ++c) {
                const float acc = dotf(N, iw + c * hn, hn) * invhc;
                stt(inj, ib + c * is, 2.0f / (1.0f + sc_exp(-acc)));
            }
        }

        /* ---- optional code form of x, taken from the stored (narrowed) value rather than the
         * accumulator, as a fused kernel must, so the two agree bit for bit. */
        if (((q && q->data) || rg) && x->dtype != RAD_F32) {
            row_from_f32(x->dtype, xv, tmp, n);
            row_to_f32(x->dtype, tmp, xv, n);
        }
        if (q && q->data) {
            float* qv = N;     /* the row's norm is dead here; its storage takes the codes. */
            for (int64_t gi = 0; gi < ngrp; ++gi) {
                const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
                float sc;
                fp8_group(xv, qv, k0, k1, &sc, q->dtype == RAD_I8 ? 127.0f : 448.0f);
                stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
            }
            out_row(q, rowoff(q, r, n), n, qv);
        }

        /* ---- optional rotated form of x, from the same stored values: E4M3 codes, or int8
         * (had_quant_act_i8g's) where the buffer is. */
        if (rg) {
            float* rv = N;     /* N and tmp are dead once the plain codes are stored. */
            float* rc = tmp;
            std::memcpy(rv, xv, (size_t)n * sizeof(float));
            fwht_row(rv, n, rg);
            const int64_t rgrp = n / rg;
            for (int64_t gi = 0; gi < rgrp; ++gi) {
                float sc;
                fp8_group(rv, rc, gi * rg, gi * rg + rg, &sc,
                          rq->dtype == RAD_I8 ? 127.0f : 448.0f);
                stt(rs, rowoff(rs, r, rgrp) + gi * laststride(rs), sc);
            }
            out_row(rq, rowoff(rq, r, n), n, rc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_write */
/* h[s] += inj[s] * y for each of the hc streams, in place, on the unnormed stream the paired read
 * received. */
AVX_KERNEL(hc_write) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor* y = &a->t[0];
    const RadTensor* inj = &a->t[1];
    const RadTensor* h = &a->t[2];

    const int64_t n = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    const int64_t hn = hc * n;
    const int64_t rows = numel(y) / n;
    if (rows <= 0 || numel(h) < rows * hn || numel(inj) < rows * hc) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * hn >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t yb = rowoff(y, r, n), hb = rowoff(h, r, hn), ib = rowoff(inj, r, hc);
        const int64_t hs = laststride(h), is = laststride(inj);
        float* yv = scratch_f32((size_t)n);
        float* hv = scratch_f32_b((size_t)n);
        const float* yy = in_row(y, yb, n, yv);
        for (int64_t c = 0; c < hc; ++c) {
            const float gain = ldt(inj, ib + c * is);
            if (hs != 1) {
                for (int64_t i = 0; i < n; ++i) {
                    const int64_t j = hb + (c * n + i) * hs;
                    stt(h, j, ldt(h, j) + gain * yy[i]);
                }
                continue;
            }
            const float* cur = in_row(h, hb + c * n, n, hv);
            if (cur != hv) std::memcpy(hv, cur, (size_t)n * sizeof(float));
            const vf vg = vf_set1(gain);
            int64_t i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(hv + i, vf_fma(vg, vf_loadu(yy + i), vf_loadu(hv + i)));
            for (; i < n; ++i) hv[i] += gain * yy[i];
            out_row(h, hb + c * n, n, hv);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_project */
/* The control vector's projection (core/cvec.h): h[s] -= scale * (h[s] . v) * v for each of the
 * hc streams of a row whose switch is on. v is f32, the stream may be bf16/f16/f32; a row with a
 * zero switch is skipped without a read, so an unprojected request is byte-identical to one on a
 * deployment with no vector. */
AVX_KERNEL(hc_project) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor* h  = &a->t[0];
    const RadTensor* v  = &a->t[1];
    const RadTensor* on = &a->t[2];

    const int64_t n  = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    const float   sc = p_f32(a, "scale", 1.0f);
    if (n <= 0 || hc <= 1 || numel(v) < n) return RAD_E_SHAPE;
    const int64_t hn = hc * n;
    const int64_t rows = numel(h) / hn;
    if (rows <= 0 || numel(on) < rows) return RAD_E_SHAPE;

    const int64_t vb = rowoff(v, 0, n);
    float* vbuf = scratch_f32_c((size_t) n);
    const float* vec = in_row(v, vb, n, vbuf);

    AVX_PARALLEL_FOR_IF(rows * hn >= AVX_PAR_MIN)
    for (int64_t k = 0; k < rows * hc; ++k) {
        const int64_t r = k / hc, c = k % hc;
        if (ldi(on, offlin(on, r)) == 0) continue;
        const int64_t hb = rowoff(h, r, hn), hs = laststride(h);
        const int64_t base = hb + c * n;
        float* hv = scratch_f32((size_t) n);
        const float* row = in_row(h, base, n, hv);
        if (hs != 1) {
            float dot = 0.0f;
            for (int64_t i = 0; i < n; ++i) dot += row[i] * vec[i];
            const float g = dot * sc;
            for (int64_t i = 0; i < n; ++i)
                stt(h, base + i * hs, row[i] - g * vec[i]);
            continue;
        }
        if (row != hv) std::memcpy(hv, row, (size_t) n * sizeof(float));
        /* The dot reads the row and the direction; the update then rewrites the same elements, so
         * the direction's own copy is not needed (it is never written). */
        vf acc = vf_zero();
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            acc = vf_fma(vf_loadu(hv + i), vf_loadu(vec + i), acc);
        float dot = vf_hsum(acc);
        for (; i < n; ++i) dot += hv[i] * vec[i];
        const vf vg = vf_set1(dot * sc);
        i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(hv + i, vf_sub(vf_loadu(hv + i), vf_mul(vg, vf_loadu(vec + i))));
        for (; i < n; ++i) hv[i] -= (dot * sc) * vec[i];
        out_row(h, base, n, hv);
    }
    return RAD_OK;
}

/* ================================================================== mtp_enter
 *
 * Entry of the multi-token-prediction head for a model with an `hc`-wide residual stream.
 *
 * A pre-norm MTP head usually joins the trunk's hidden state and the next token's embedding with one
 * `fc` over their concatenation. Qwen4-Exp's hidden state is the wide stream and the checkpoint has
 * two square matrices: `fc_h` is applied to each of the hc sub-streams, and one n-wide embedding
 * product is added into all of them.
 *
 *     ne[j]     = e[j] * rsqrt(mean(e^2) + eps) * (w_e[j] + wadd)
 *     fe[i]     = sum_j fc_e[i][j] * ne[j]
 *     nh[k][j]  = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)
 *     x[k*n+i]  = sum_j fc_h[i][j] * nh[k][j] + fe[i]
 *
 * `ngroup` is not determined by the checkpoint: a [hc*n] gain fits both a grouped norm (hc rsqrts)
 * and a flat one (one rsqrt over the wide row). No per-op oracle distinguishes them (the draft
 * acceptance rate does), so it is a parameter fixed at declare.
 *
 * Run as a GEMM. libref converts `fc_h` and `fc_e` inside the innermost loop, `rows * hc` times.
 * Here the norms are first materialised into panels ([rows*hc, n] hidden, [rows, n] embedding),
 * then each output column i widens row i of `fc_h` and `fc_e` once and contracts it against every
 * panel row. The whole [n, n] weight is never widened (26 MiB per matrix at n = 2560); each weight
 * element is still converted once.
 */
AVX_KERNEL(mtp_enter) {
    if (!rad_args_have(a, 7)) return RAD_E_INVAL;
    const RadTensor* h = &a->t[0];
    const RadTensor* e = &a->t[1];
    const RadTensor* w_h = &a->t[2];
    const RadTensor* w_e = &a->t[3];
    const RadTensor* fc_h = &a->t[4];
    const RadTensor* fc_e = &a->t[5];
    const RadTensor* x = &a->t[6];

    const int64_t n = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    const int64_t ngroup = p_int(a, "ngroup", hc);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    if (ngroup != 1 && ngroup != hc) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = numel(h) / hn;
    if (rows <= 0 || numel(e) < rows * n || numel(x) < rows * hn) return RAD_E_SHAPE;
    if (numel(w_h) < hn || numel(w_e) < n) return RAD_E_SHAPE;
    /* Plain fc matrices, or codes and a scale joined (`fc` e4m3), as hc_read's. */
    avxp::ScaledRows jh, je;
    const bool jfh = avxp::is_joined(fc_h), jfe = avxp::is_joined(fc_e);
    if (jfh) {
        if (const int st = avxp::scaled_rows(fc_h, n, n, &jh); st != RAD_OK) return st;
    } else if (numel(fc_h) < n * n) {
        return RAD_E_SHAPE;
    }
    if (jfe) {
        if (const int st = avxp::scaled_rows(fc_e, n, n, &je); st != RAD_OK) return st;
    } else if (numel(fc_e) < n * n) {
        return RAD_E_SHAPE;
    }

    auto pad = [](size_t v) { return (v + 15) & ~(size_t)15; };
    const size_t o_gh = 0;
    const size_t o_ge = pad(o_gh + (size_t)hn);
    const size_t o_nh = pad(o_ge + (size_t)n);
    const size_t o_ne = pad(o_nh + (size_t)rows * (size_t)hn);
    const size_t total = o_ne + (size_t)rows * (size_t)n + 16;
    float* blk = (float*)scratch_raw(total * sizeof(float));
    float* gh = blk + o_gh;
    float* ge = blk + o_ge;
    float* nh = blk + o_nh;     /* [rows * hc, n] */
    float* ne = blk + o_ne;     /* [rows, n] */
    {
        const int64_t whs = laststride(w_h), wes = laststride(w_e);
        if (whs == 1) row_to_f32(w_h->dtype, w_h->data, gh, hn);
        else for (int64_t i = 0; i < hn; ++i) gh[i] = ldt(w_h, i * whs);
        if (wes == 1) row_to_f32(w_e->dtype, w_e->data, ge, n);
        else for (int64_t i = 0; i < n; ++i) ge[i] = ldt(w_e, i * wes);
        if (wadd != 0.0f) {
            for (int64_t i = 0; i < hn; ++i) gh[i] += wadd;
            for (int64_t i = 0; i < n; ++i) ge[i] += wadd;
        }
    }

    /* Phase one: both norms into the panels, rounded to their source tensor's dtype (the
     * reference model's `.type_as(x)`). */
    AVX_PARALLEL_FOR_IF(rows * hn >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)hn);
        float* tmp = scratch_f32_b((size_t)hn);
        const float* ev = in_row(e, rowoff(e, r, n), n, buf);
        const float rse = rsqrt_meansq_f64(ev, n, eps);
        float* dst_e = ne + r * n;
        for (int64_t i = 0; i < n; ++i) dst_e[i] = ev[i] * rse * ge[i];
        if (e->dtype != RAD_F32) {
            row_from_f32(e->dtype, dst_e, tmp, n);
            row_to_f32(e->dtype, tmp, dst_e, n);
        }

        const float* hv = in_row(h, rowoff(h, r, hn), hn, buf);
        const float flat = (ngroup == 1) ? rsqrt_meansq_f64(hv, hn, eps) : 0.0f;
        float* dst_h = nh + r * hn;
        for (int64_t c = 0; c < hc; ++c) {
            const float sc = (ngroup == 1) ? flat : rsqrt_meansq_f64(hv + c * n, n, eps);
            for (int64_t i = 0; i < n; ++i)
                dst_h[c * n + i] = hv[c * n + i] * sc * gh[c * n + i];
        }
        if (h->dtype != RAD_F32) {
            row_from_f32(h->dtype, dst_h, tmp, hn);
            row_to_f32(h->dtype, tmp, dst_h, hn);
        }
    }

    /* Phase two: one output column at a time, so each weight row is widened once. */
    const int64_t xs = laststride(x);
    AVX_PARALLEL_FOR_IF(rows * hc * n >= AVX_PAR_MIN)
    for (int64_t i = 0; i < n; ++i) {
        float* fh = scratch_f32((size_t)n);
        float* fe = scratch_f32_b((size_t)n);
        const int64_t fhb = jfh ? 0 : rowoff(fc_h, i, n), fhs = laststride(fc_h);
        const int64_t feb = jfe ? 0 : rowoff(fc_e, i, n), fes = laststride(fc_e);
        if (jfh) scaled_row(jh, i, n, fh);
        else if (fhs == 1) row_to_f32(fc_h->dtype, byte_at(fc_h->data, fc_h->dtype, fhb), fh, n);
        else for (int64_t j = 0; j < n; ++j) fh[j] = ldt(fc_h, fhb + j * fhs);
        if (jfe) scaled_row(je, i, n, fe);
        else if (fes == 1) row_to_f32(fc_e->dtype, byte_at(fc_e->data, fc_e->dtype, feb), fe, n);
        else for (int64_t j = 0; j < n; ++j) fe[j] = ldt(fc_e, feb + j * fes);

        for (int64_t r = 0; r < rows; ++r) {
            const float fev = dotf(fe, ne + r * n, n);
            const int64_t xb = rowoff(x, r, hn);
            for (int64_t c = 0; c < hc; ++c)
                stt(x, xb + (c * n + i) * xs, dotf(fh, nh + r * hn + c * n, n) + fev);
        }
    }
    return RAD_OK;
}
