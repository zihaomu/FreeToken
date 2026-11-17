/* avx_norm.cpp -- rmsnorm, rmsnorm_add, layernorm and the vision position-table adds.
 *
 * Each norm is a reduction followed by a scale, so two passes over the row. The row is converted
 * to f32 scratch once and both passes read the scratch (libref converts per element, three times
 * per element in rmsnorm_add).
 *
 * The reduction is `row_sumsq` (avx_support.cpp), a four-accumulator vector tree. It does not sum
 * in libref's ascending-index order, which is why the bf16 tolerance for a norm is 4e-3, not 0.
 *
 * Rounding contracts:
 *   `wadd` is GemmaRMSNorm's `1 +`, added in f32 after the weight is widened, never to the stored
 *   bf16 gain (Qwen3.5, Qwen3-Next and Gemma store a zero-centred gain; `1 + w` in bf16 loses 0.3%
 *   per channel). libr4d's fused norms do the same.
 *
 *   `rmsnorm_add` takes the variance from the unrounded f32 sum but multiplies the sum after it is
 *   narrowed into residual_out, as vLLM's fused_add_rms_norm does. Keeping f32 throughout would
 *   disagree with every fused kernel this is checked against.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <vector>

using namespace avx;

/* Widen the gain once per launch, not once per row, into a caller buffer built before the
 * parallel region and shared by all threads. `wadd` is folded in here, once per gain element. */
static void widen_gain(const RadTensor* w, int64_t n, float wadd, float* out) {
    const int64_t ws = laststride(w);
    if (ws == 1) {
        row_to_f32(w->dtype, w->data, out, n);
    } else {
        for (int64_t i = 0; i < n; ++i) out[i] = ldt(w, i * ws);
    }
    if (wadd != 0.0f) {
        const vf va = vf_set1(wadd);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(out + i, vf_add(vf_loadu(out + i), va));
        for (; i < n; ++i) out[i] += wadd;
    }
}

/* ------------------------------------------------------------------ rmsnorm */
/* y = x * rsqrt(mean(x^2) + eps) * (w + wadd) */
AVX_KERNEL(rmsnorm) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[2];
    /* An absent gain is 1 (as in libref); `wadd` requires a gain. */
    const RadTensor* w = a->t[1].data ? &a->t[1] : nullptr;
    const int64_t n = p_int(a, "n", w && w->rank ? w->shape[w->rank - 1] : 0);
    const float eps  = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0) return RAD_E_SHAPE;
    if (!w && wadd != 0.0f) return RAD_E_INVAL;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || (w && numel(w) < n)) return RAD_E_SHAPE;

    float* gain = (float*)scratch_raw((size_t)n * sizeof(float));
    if (w) widen_gain(w, n, wadd, gain);
    else   for (int64_t i = 0; i < n; ++i) gain[i] = 1.0f;
    /* `mult` is folded into the gain, so the row is scaled before its single rounding. */
    const float mult = p_f32(a, "mult", 1.0f);
    if (mult != 1.0f)
        for (int64_t i = 0; i < n; ++i) gain[i] *= mult;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, sx);
        const float ss = row_sumsq(v, n);
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        const vf vs = vf_set1(s);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(sy + i, vf_mul(vf_mul(vf_loadu(v + i), vs), vf_loadu(gain + i)));
        for (; i < n; ++i) sy[i] = v[i] * s * gain[i];
        out_row(y, rowoff(y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rmsnorm_p */
/* y = x * rsqrt(mean(x^2) + eps): rmsnorm without a gain operand. */
AVX_KERNEL(rmsnorm_p) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    const float eps  = p_f32(a, "eps", 1e-6f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, sx);
        const float ss = row_sumsq(v, n);
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        const vf vs = vf_set1(s);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(sy + i, vf_mul(vf_loadu(v + i), vs));
        for (; i < n; ++i) sy[i] = v[i] * s;
        out_row(y, rowoff(y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rmsnorm_add */
/* residual_out = x + residual;  y = rmsnorm(residual_out) * (w + wadd), with the variance from the
 * unrounded sum and the multiply from the narrowed one. */
AVX_KERNEL(rmsnorm_add) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *res = &a->t[1], *w = &a->t[2], *y = &a->t[3], *ro = &a->t[4];
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps  = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || numel(ro) < rows * n ||
        numel(res) < rows * n || numel(w) < n) return RAD_E_SHAPE;

    float* gain = (float*)scratch_raw((size_t)n * sizeof(float));
    widen_gain(w, n, wadd, gain);
    const uint32_t ro_dt = ro->dtype;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sr = scratch_f32_b((size_t)n);
        float* sm = scratch_f32_c((size_t)n);
        const float* xv = in_row(x,   rowoff(x, r, n),   n, sx);
        const float* rv = in_row(res, rowoff(res, r, n), n, sr);

        vf acc0 = vf_zero(), acc1 = vf_zero();
        int64_t i = 0;
        for (; i + 2 * VF_N <= n; i += 2 * VF_N) {
            vf s0 = vf_add(vf_loadu(xv + i), vf_loadu(rv + i));
            vf s1 = vf_add(vf_loadu(xv + i + VF_N), vf_loadu(rv + i + VF_N));
            vf_storeu(sm + i, s0);
            vf_storeu(sm + i + VF_N, s1);
            acc0 = vf_fma(s0, s0, acc0);
            acc1 = vf_fma(s1, s1, acc1);
        }
        float ss = vf_hsum(vf_add(acc0, acc1));
        for (; i < n; ++i) { float v = xv[i] + rv[i]; sm[i] = v; ss += v * v; }

        /* The residual is written from the unrounded sum... */
        out_row(ro, rowoff(ro, r, n), n, sm);
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);

        /* ...and narrowed for the multiply by a round trip through `sr`, rather than re-reading
         * the operand. `sr` is the only free buffer: `gain` lives in the launching thread's
         * scratch_raw arena, and another scratch_raw call inside the parallel region could
         * reallocate it under the other threads. `sr` holds 4n bytes, enough for any dtype, and
         * `rv` (possibly `sr`) is dead once the sum is formed. */
        if (ro_dt != RAD_F32) {
            row_from_f32(ro_dt, sm, sr, n);
            row_to_f32(ro_dt, sr, sm, n);
        }

        const vf vs = vf_set1(s);
        i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(sm + i, vf_mul(vf_mul(vf_loadu(sm + i), vs), vf_loadu(gain + i)));
        for (; i < n; ++i) sm[i] = sm[i] * s * gain[i];
        out_row(y, rowoff(y, r, n), n, sm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ layernorm */
/* y = (x - mean) * rsqrt(var + eps) * w + b, with the biased variance (divide by n). `b` is
 * optional: a model without a bias passes a null operand. */
AVX_KERNEL(layernorm) {
    if (a->n_t < 4 || !a->t[0].data || !a->t[1].data || !a->t[3].data) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *y = &a->t[3];
    const RadTensor* b = rad_arg_in(a, 2);
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-5f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || numel(w) < n) return RAD_E_SHAPE;
    if (b && numel(b) < n) return RAD_E_SHAPE;

    /* Gain and bias are widened once, into one allocation: a second scratch_raw call may
     * reallocate and invalidate the first pointer. */
    float* gb = (float*)scratch_raw((size_t)(2 * n) * sizeof(float));
    float* gain = gb;
    float* bias = gb + n;
    widen_gain(w, n, 0.0f, gain);
    if (b) {
        const int64_t bs = laststride(b);
        if (bs == 1) row_to_f32(b->dtype, b->data, bias, n);
        else for (int64_t i = 0; i < n; ++i) bias[i] = ldt(b, i * bs);
    }
    const bool has_b = b != nullptr;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, sx);

        /* Two passes (mean, then squared deviations), as in libref. The one-pass E[x^2] - E[x]^2
         * cancels catastrophically when the mean is large relative to the spread, as in a
         * post-attention residual; the extra pass reads a row already in L1. */
        const float mean = row_sum(v, n) / (float)n;
        const vf vm = vf_set1(mean);
        vf acc = vf_zero();
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            vf d = vf_sub(vf_loadu(v + i), vm);
            acc = vf_fma(d, d, acc);
        }
        float ss = vf_hsum(acc);
        for (; i < n; ++i) { float d = v[i] - mean; ss += d * d; }

        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        const vf vs = vf_set1(s);
        i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            vf t = vf_mul(vf_mul(vf_sub(vf_loadu(v + i), vm), vs), vf_loadu(gain + i));
            if (has_b) t = vf_add(t, vf_loadu(bias + i));
            vf_storeu(sy + i, t);
        }
        for (; i < n; ++i) {
            float t = (v[i] - mean) * s * gain[i];
            if (has_b) t += bias[i];
            sy[i] = t;
        }
        out_row(y, rowoff(y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_embed */
/* `v` rounded to an activation dtype (f32, bf16 or f16), as the position term is stored. */
static inline float round_act(uint32_t dt, float v) {
    if (dt == RAD_BF16) return rad_bf16_to_f32(rad_f32_to_bf16(v));
    if (dt == RAD_F16)  return rad_f16_to_f32(rad_f32_to_f16(v));
    return v;
}

/* Adds a learned side x side position table, resampled bilinearly to each row's grid (corners
 * aligned, edge taps clamped). Matches libref's grid_embed operation for operation: four taps
 * summed in f32 in the same order, rounded to x's dtype, then added and rounded again. One row per
 * vision-tower patch, off the step path, so it is scalar. */
AVX_KERNEL(grid_embed) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *tab = &a->t[1], *crd = &a->t[2];
    const int64_t n = p_int(a, "n", tab->rank ? tab->shape[tab->rank - 1] : 0);
    const int64_t side = p_int(a, "side", 0);
    if (n <= 0 || side <= 0 || numel(tab) < side * side * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < rows)
        return RAD_E_SHAPE;
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    const int64_t ts = laststride(tab);

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        int64_t g[4];
        for (int c = 0; c < 4; ++c) g[c] = ldi(crd, c * cs + r * ms);
        int64_t th[2], tw[2];
        float wh[2], ww[2];
        auto axis = [&](int64_t idx, int64_t size, int64_t* t, float* w) {
            const float src = (float)idx * (float)(side - 1) / (float)(size - 1 > 1 ? size - 1 : 1);
            const float fl = std::floor(src);
            for (int k = 0; k < 2; ++k) {
                const int64_t tap = (int64_t)fl + k;
                t[k] = tap < 0 ? 0 : (tap > side - 1 ? side - 1 : tap);
                const float d = std::fabs(src - fl - (float)k);
                w[k] = 1.0f - d > 0.0f ? 1.0f - d : 0.0f;
            }
        };
        axis(g[0], g[2], th, wh);
        axis(g[1], g[3], tw, ww);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int ki = 0; ki < 2; ++ki)
                for (int kj = 0; kj < 2; ++kj)
                    acc += ldt(tab, rowoff(tab, th[ki] * side + tw[kj], n) + i * ts) *
                           (wh[ki] * ww[kj]);
            const float pe = round_act(x->dtype, acc);
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_embed_pad */
/* grid_embed with the half-pixel-centre resample (align_corners=False) and grid_sample's
 * padding_mode="zeros": a tap whose UNCLAMPED index falls outside [0, side) carries weight zero
 * instead of reading the clamped edge. Matches libref's grid_embed_pad operation: four taps in
 * f32 in the same order, rounded to x's dtype, then added and rounded again. Scalar, one row per
 * vision-tower patch, as grid_embed's is. */
AVX_KERNEL(grid_embed_pad) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *tab = &a->t[1], *crd = &a->t[2];
    const int64_t n = p_int(a, "n", tab->rank ? tab->shape[tab->rank - 1] : 0);
    const int64_t side = p_int(a, "side", 0);
    if (n <= 0 || side <= 0 || numel(tab) < side * side * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < rows)
        return RAD_E_SHAPE;
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    const int64_t ts = laststride(tab);

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        int64_t g[4];
        for (int c = 0; c < 4; ++c) g[c] = ldi(crd, c * cs + r * ms);
        int64_t th[2], tw[2];
        float wh[2], ww[2];
        auto axis = [&](int64_t idx, int64_t size, int64_t* t, float* w) {
            if (size < 1) size = 1;
            const float src = (float)side / (float)size * ((float)idx + 0.5f) - 0.5f;
            const float fl = std::floor(src);
            for (int k = 0; k < 2; ++k) {
                const int64_t raw = (int64_t)fl + k;
                t[k] = raw < 0 ? 0 : (raw > side - 1 ? side - 1 : raw);
                const float d = std::fabs(src - fl - (float)k);
                const float u = 1.0f - d > 0.0f ? 1.0f - d : 0.0f;
                w[k] = raw >= 0 && raw <= side - 1 ? u : 0.0f;
            }
        };
        axis(g[0], g[2], th, wh);
        axis(g[1], g[3], tw, ww);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int ki = 0; ki < 2; ++ki)
                for (int kj = 0; kj < 2; ++kj)
                    acc += ldt(tab, rowoff(tab, th[ki] * side + tw[kj], n) + i * ts) *
                           (wh[ki] * ww[kj]);
            const float pe = round_act(x->dtype, acc);
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ window_order */
/* The window-attention permutation of one pass from the coordinate planes alone (libref's
 * window_order operation). One thread does the whole pass: the slot of each patch follows from
 * its window's base, computed as a prefix over windows, so the work is two linear walks over P --
 * nothing to vectorise, and off the step path. */
AVX_KERNEL(window_order) {
    if (!rad_args_have(a, 6)) return RAD_E_INVAL;
    const RadTensor *crd = &a->t[0], *cu = &a->t[1];
    RadTensor *idx = const_cast<RadTensor*>(&a->t[2]), *inv = const_cast<RadTensor*>(&a->t[3]),
              *pos = const_cast<RadTensor*>(&a->t[4]), *cuw = const_cast<RadTensor*>(&a->t[5]);
    const int64_t M = p_int(a, "M", 0);
    const int64_t win = p_int(a, "window", 0);
    if (M <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < M) return RAD_E_SHAPE;
    if (numel(cu) < 2 || numel(idx) < M || numel(inv) < M || numel(pos) < 2 * M ||
        numel(cuw) < M + 1)
        return RAD_E_SHAPE;
    const int64_t P = crd->shape[1];
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    auto gx = [&](int c, int64_t p) { return ldi(crd, (int64_t)c * cs + p * ms); };
    auto sti = [](const RadTensor* t, int64_t i, int64_t v) { st_int_dt(t->dtype, t->data, i, v); };
    if (idx->dtype != RAD_I32 || inv->dtype != RAD_I32 || pos->dtype != RAD_I32 ||
        cuw->dtype != RAD_I32)
        return RAD_E_DTYPE;

    auto identity = [&]() {
        for (int64_t p = 0; p < P; ++p) {
            sti(idx, p, p);
            sti(inv, p, p);
            if (p < M) {
                sti(pos, p, gx(1, p) + 1);
                sti(pos, P + p, gx(0, p) + 1);
            } else {
                sti(pos, p, 1);
                sti(pos, P + p, 1);
            }
        }
        int64_t last = 0;
        const int64_t k = numel(cu) < M + 1 ? numel(cu) : M + 1;
        for (int64_t i = 0; i < k; ++i) {
            last = ldi(cu, i);
            sti(cuw, i, last);
        }
        for (int64_t i = k; i < M + 1; ++i) sti(cuw, i, last);
    };
    if (win <= 0) {
        identity();
        return RAD_OK;
    }

    std::vector<int64_t> seg((size_t)M), seg_nw, seg_first_win;
    std::vector<int64_t> wbase((size_t)M);
    int64_t nwin = 0, slot = 0;
    const int64_t S = numel(cu) - 1;
    for (int64_t s = 0; s < S; ++s) {
        const int64_t start = ldi(cu, s);
        if (start >= M) break;
        int64_t end = ldi(cu, s + 1);
        if (end > M) end = M;
        const int64_t gh = start < end ? gx(2, start) : 1;
        const int64_t gw = start < end ? gx(3, start) : 1;
        const int64_t nh = (gh + win - 1) / win, nw = (gw + win - 1) / win;
        seg_first_win.push_back(nwin);
        seg_nw.push_back(nw);
        for (int64_t p = start; p < end; ++p) seg[(size_t)p] = (int64_t)seg_first_win.size() - 1;
        for (int64_t wr = 0; wr < nh; ++wr)
            for (int64_t wc = 0; wc < nw; ++wc) {
                const int64_t rows_in = std::min(win, gh - wr * win);
                const int64_t cols_in = std::min(win, gw - wc * win);
                sti(cuw, nwin, slot);
                wbase[(size_t)nwin] = slot;
                ++nwin;
                slot += rows_in * cols_in;
            }
    }
    sti(cuw, nwin, slot);
    for (int64_t i = nwin + 1; i < M + 1; ++i) sti(cuw, i, slot);

    for (int64_t p = 0; p < P; ++p) {
        if (p >= M) {
            sti(idx, p, p);
            sti(inv, p, p);
            sti(pos, p, 1);
            sti(pos, P + p, 1);
            continue;
        }
        const int64_t r = gx(0, p), c = gx(1, p), gw = gx(3, p);
        int64_t wr = r / win, wc = c / win;
        if (wr < 0) wr = 0;
        if (wc < 0) wc = 0;
        const int64_t cols_in = std::min(win, gw - wc * win);
        const int64_t wid = seg_first_win[(size_t)seg[(size_t)p]] +
                            wr * seg_nw[(size_t)seg[(size_t)p]] + wc;
        const int64_t at = wbase[(size_t)wid] + (r - wr * win) * cols_in + (c - wc * win);
        sti(idx, at, p);
        sti(inv, p, at);
        sti(pos, at, c + 1);
        sti(pos, P + at, r + 1);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_embed_aa */
/* Adds the table resampled as torch's F.interpolate(bilinear, align_corners=False,
 * antialias=True) does (SigLIP2's position embedding); matches libref's grid_embed_aa. Per axis: a
 * triangle filter on half-pixel centres, widened by the scale when shrinking, with the in-table
 * taps normalised. Rows are summed first in f32 and rounded once before the add. Out-of-grid
 * coordinates are clamped. */
static void aa_taps(int64_t at, int64_t size, int64_t side, float* w, int64_t* lo, int64_t* cnt) {
    if (size < 1) size = 1;
    at = at < 0 ? 0 : (at >= size ? size - 1 : at);
    const float scale = (float)side / (float)size;
    const float support = scale > 1.0f ? scale : 1.0f, inv = 1.0f / support;
    const float center = scale * ((float)at + 0.5f);
    int64_t a = (int64_t)(center - support + 0.5f), b = (int64_t)(center + support + 0.5f);
    if (a < 0) a = 0;
    if (b > side) b = side;
    float total = 0.0f;
    for (int64_t j = a; j < b; ++j) {
        const float d = std::fabs(((float)j - center + 0.5f) * inv);
        w[j - a] = d < 1.0f ? 1.0f - d : 0.0f;
        total += w[j - a];
    }
    for (int64_t j = a; j < b; ++j) w[j - a] = total != 0.0f ? w[j - a] / total : 0.0f;
    *lo = a;
    *cnt = b > a ? b - a : 0;
}

AVX_KERNEL(grid_embed_aa) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *tab = &a->t[1], *crd = &a->t[2];
    const int64_t n = p_int(a, "n", tab->rank ? tab->shape[tab->rank - 1] : 0);
    const int64_t side = p_int(a, "side", 0);
    if (n <= 0 || side <= 0 || numel(tab) < side * side * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < rows)
        return RAD_E_SHAPE;
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    const int64_t ts = laststride(tab);

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        int64_t g[4];
        for (int c = 0; c < 4; ++c) g[c] = ldi(crd, c * cs + r * ms);
        float* wh = scratch_f32((size_t)(2 * side));
        float* ww = wh + side;
        int64_t h0 = 0, nh = 0, w0 = 0, nw = 0;
        aa_taps(g[0], g[2], side, wh, &h0, &nh);
        aa_taps(g[1], g[3], side, ww, &w0, &nw);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int64_t y = 0; y < nh; ++y) {
                float row = 0.0f;
                for (int64_t c = 0; c < nw; ++c)
                    row += ldt(tab, rowoff(tab, (h0 + y) * side + w0 + c, n) + i * ts) * ww[c];
                acc += row * wh[y];
            }
            const float pe = round_act(x->dtype, acc);
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_embed_cpe */
/* Adds the table resampled as torch's F.interpolate(bilinear, align_corners=False) does without
 * antialias, onto a square of the grid's larger side and cropped to the grid (RADIO's cropped
 * position embedding); matches libref's grid_embed_cpe operation for operation. */
static void cpe_taps(int64_t at, int64_t d, int64_t side, int64_t t[2], float w[2]) {
    if (d < 1) d = 1;
    at = at < 0 ? 0 : (at >= d ? d - 1 : at);
    float src = (float)side / (float)d * ((float)at + 0.5f) - 0.5f;
    if (src < 0.0f) src = 0.0f;
    int64_t i0 = (int64_t)src;
    if (i0 > side - 1) i0 = side - 1;
    t[0] = i0;
    t[1] = i0 < side - 1 ? i0 + 1 : i0;
    w[1] = src - (float)i0;
    w[0] = 1.0f - w[1];
}

AVX_KERNEL(grid_embed_cpe) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *tab = &a->t[1], *crd = &a->t[2];
    const int64_t n = p_int(a, "n", tab->rank ? tab->shape[tab->rank - 1] : 0);
    const int64_t side = p_int(a, "side", 0);
    if (n <= 0 || side <= 0 || numel(tab) < side * side * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < rows)
        return RAD_E_SHAPE;
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    const int64_t ts = laststride(tab);

    /* The prefix rows, when n_prefix names some; without it the operand is not read. */
    const int64_t npre = p_int(a, "n_prefix", 0);
    const RadTensor* pre = npre > 0 && a->n_t > 3 && a->t[3].data ? &a->t[3] : nullptr;
    if (npre > 0 && (!pre || numel(pre) < npre * n)) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        int64_t g[4];
        for (int c = 0; c < 4; ++c) g[c] = ldi(crd, c * cs + r * ms);
        if (g[0] < 0) {
            /* A row ahead of its segment's patches: the learned row its column names. */
            const int64_t k = g[1] < 0 ? 0 : (g[1] >= npre ? npre - 1 : g[1]);
            const int64_t xb0 = rowoff(x, r, n), xs0 = laststride(x);
            for (int64_t i = 0; npre > 0 && i < n; ++i) {
                const float pe = round_act(x->dtype, ldt(pre, rowoff(pre, k, n) + i * laststride(pre)));
                stt(x, xb0 + i * xs0, ldt(x, xb0 + i * xs0) + pe);
            }
            continue;
        }
        const int64_t d = g[2] > g[3] ? g[2] : g[3];
        int64_t th[2], tw[2];
        float wh[2], ww[2];
        cpe_taps(g[0], d, side, th, wh);
        cpe_taps(g[1], d, side, tw, ww);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        auto at = [&](int64_t y, int64_t c, int64_t i) {
            return ldt(tab, rowoff(tab, y * side + c, n) + i * ts);
        };
        for (int64_t i = 0; i < n; ++i) {
            const float r0 = ww[0] * at(th[0], tw[0], i) + ww[1] * at(th[0], tw[1], i);
            const float r1 = ww[0] * at(th[1], tw[0], i) + ww[1] * at(th[1], tw[1], i);
            const float pe = round_act(x->dtype, wh[0] * r0 + wh[1] * r1);
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
        }
    }
    return RAD_OK;
}
