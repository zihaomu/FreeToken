/* avx_elementwise.cpp -- add, mul, activations, gated products, row gather/scatter, cast, softmax
 * and per-row scaling.
 *
 * These ops are memory-bound: a constant number of touches and a few flops per element.
 *
 *   The dtype switch is per row. `in_row`/`out_row` (avx_support.cpp) convert a whole row once
 *   (returning the operand's own pointer for f32), instead of libref's per-element ld_dt(), whose
 *   branch blocks vectorisation.
 *
 *   Fused forms round where libref rounds. E.g. `softmax` writes the unnormalised exponentials to
 *   `y`, reads them back through the output dtype and multiplies by 1/sum, so a bf16 result has
 *   two roundings; `rmsnorm_add` does the same with `residual_out`. Staying in f32 would be half an
 *   ulp more accurate and disagree with the oracle and the device kernels. The round trip uses a
 *   hot scratch buffer instead of the output operand.
 *
 * Broadcasting: `mul` broadcasts one row down the rows (a bias's shape); `scale_rows` broadcasts one
 * column across them (a per-token gate). They are separate ops because the two readings are
 * ambiguous when rows == n, and a wrong guess gives plausible numbers.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <cmath>

using namespace avx;

/* Round a row through an operand's storage dtype and back, as an unfused store-and-reload would.
 * A no-op for f32. */
static inline void roundtrip(uint32_t dt, float* v, int64_t n, void* tmp) {
    if (dt == RAD_F32) return;
    row_from_f32(dt, v, tmp, n);
    row_to_f32(dt, tmp, v, n);
}

/* ------------------------------------------------------------------ add / mul */
/* y = a op b over the output's extent. A one-row `b` is broadcast down the rows; nothing else
 * broadcasts, so a shape bug fails instead of producing plausible numbers. `add`'s `div` divides
 * the sum rounded through y's dtype, as libref rounds it. */
template <bool MUL>
static int elementwise2(const RadArgs* a) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", numel(Y));
    const float div = MUL ? 1.0f : p_f32(a, "div", 1.0f);
    if (!(div != 0.0f) || !std::isfinite(div)) return RAD_E_INVAL;
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(A) < rows * n) return RAD_E_SHAPE;
    const bool bcast = numel(B) == n && rows > 1;
    if (!bcast && numel(B) < rows * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sa = scratch_f32((size_t)n);
        float* sb = scratch_f32_b((size_t)n);
        float* sy = scratch_f32_c((size_t)n);
        const float* x = in_row(A, rowoff(A, r, n), n, sa);
        const float* y = in_row(B, bcast ? 0 : rowoff(B, r, n), n, sb);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            vf u = vf_loadu(x + i), v = vf_loadu(y + i);
            vf_storeu(sy + i, MUL ? vf_mul(u, v) : vf_add(u, v));
        }
        for (; i < n; ++i) sy[i] = MUL ? x[i] * y[i] : x[i] + y[i];
        if (div != 1.0f) {
            /* sa is free once x is read: the sum's narrowing goes through it. */
            roundtrip(Y->dtype, sy, n, sa);
            const vf vd = vf_set1(div);
            for (i = 0; i + VF_N <= n; i += VF_N) vf_storeu(sy + i, vf_div(vf_loadu(sy + i), vd));
            for (; i < n; ++i) sy[i] /= div;
        }
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

AVX_KERNEL(add) { return elementwise2<false>(a); }
AVX_KERNEL(mul) { return elementwise2<true>(a); }

/* y = x * w, `w` a weight of one element or one row (libref's mul_weight). */
AVX_KERNEL(mul_weight) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(X) < rows * n) return RAD_E_SHAPE;
    const bool scalar = numel(W) == 1;
    if (!scalar && numel(W) != n) return RAD_E_SHAPE;
    /* Widened once; the same row applies to every row of x. */
    float* wrow = (float*)scratch_raw((size_t)n * sizeof(float) + 64);
    if (scalar) { const float v = ldt(W, 0); for (int64_t i = 0; i < n; ++i) wrow[i] = v; }
    else for (int64_t i = 0; i < n; ++i) wrow[i] = ldt(W, i * laststride(W));

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sa = scratch_f32((size_t)n);
        float* sy = scratch_f32_c((size_t)n);
        const float* x = in_row(X, rowoff(X, r, n), n, sa);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(sy + i, vf_mul(vf_loadu(x + i), vf_loadu(wrow + i)));
        for (; i < n; ++i) sy[i] = x[i] * wrow[i];
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ act_mul */
/* y = act(a) * b. As in libref, the activation is rounded to y's dtype before the product,
 * because the unfused pair stores it.
 *
 * silu and gelu_tanh are vectorised on vf_exp: in an encoder's gated FFN this op runs on every
 * token of every layer, and scalar libm made it about 9% of an EmbeddingGemma 2 step. gelu_tanh
 * keeps torch's structure,
 *
 *     0.5 g (1 + tanh z),   z = sqrt(2/pi) (g + 0.044715 g^3),   tanh z = (1 - e^(-2z)) / (1 + e^(-2z))
 *
 * rather than the algebraically equal g / (1 + e^(-2z)), which differs by an ulp or two and after
 * bf16 rounding measurably moves embeddings away from transformers'. vf_exp saturates, so the
 * large-|g| limits come out as g and 0. erf has no vector form here, so gelu stays scalar. */
static inline float round1(uint32_t dt, float v) {
    if (dt == RAD_F32) return v;
    uint16_t t;
    row_from_f32(dt, &v, &t, 1);
    float r;
    row_to_f32(dt, &t, &r, 1);
    return r;
}

static inline float act_of(int act, float g, uint32_t dt) {
    if (act == 0) return g / (1.0f + sc_exp(-g));
    if (act == 1) return 0.5f * g * (1.0f + sc_erf(g * 0.70710678118654752f));
    if (act == 3) return 1.0f / (1.0f + sc_exp(-g));
    if (act == 5) return g > 0.0f ? g : 0.0f;
    if (act == 4)   // quick_gelu: sigmoid of the 1.702*g the unfused mul would have stored
        return 1.0f / (1.0f + sc_exp(-round1(dt, g * round1(RAD_BF16, 1.702f))));
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;
    return 0.5f * g * (1.0f + sc_tanh(k0 * (g + k1 * g * g * g)));
}

/* sy[0..n) = act(g[0..n)) for silu (0) and gelu_tanh (2); the tail is the scalar form. */
static inline void act_row(int act, const float* g, float* sy, int64_t n, uint32_t dt) {
    int64_t i = 0;
    const vf one = vf_set1(1.0f);
    if (act == 0) {
        for (; i + VF_N <= n; i += VF_N) {
            const vf x = vf_loadu(g + i);
            vf_storeu(sy + i, vf_div(x, vf_add(one, vf_exp(vf_sub(vf_zero(), x)))));
        }
    } else if (act == 2) {
        const vf m2k0 = vf_set1(-2.0f * 0.7978845608028654f), k1 = vf_set1(0.044715f);
        const vf half = vf_set1(0.5f);
        for (; i + VF_N <= n; i += VF_N) {
            const vf x = vf_loadu(g + i);
            const vf inner = vf_fma(vf_mul(k1, x), vf_mul(x, x), x);     /* g + k1 g^3 */
            const vf e = vf_exp(vf_mul(m2k0, inner));                     /* e^(-2z) */
            const vf t = vf_div(vf_sub(one, e), vf_add(one, e));           /* tanh z */
            vf_storeu(sy + i, vf_mul(vf_mul(half, x), vf_add(one, t)));
        }
    }
    for (; i < n; ++i) sy[i] = act_of(act, g[i], dt);
}

AVX_KERNEL(act_mul) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(A) < rows * n || numel(B) < rows * n) return RAD_E_SHAPE;
    static const char* kActs[] = { "silu", "gelu", "gelu_tanh", "sigmoid", "quick_gelu", "relu" };
    const int act = p_enum(a, "act", kActs, 6, -1);
    if (act < 0) return RAD_E_INVAL;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sa = scratch_f32((size_t)n);
        float* sb = scratch_f32_b((size_t)n);
        float* sy = scratch_f32_c((size_t)n);
        const float* g = in_row(A, rowoff(A, r, n), n, sa);
        const float* u = in_row(B, rowoff(B, r, n), n, sb);
        act_row(act, g, sy, n, Y->dtype);
        /* sa is free once the activation is taken (g is sa or the tensor); sb may hold u, which
         * is still needed. */
        roundtrip(Y->dtype, sy, n, sa);
        for (int64_t i = 0; i < n; ++i) sy[i] *= u[i];
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ relu / clamp */
/* y = max(x, 0), and y = min(max(x, lo), hi) with lo and hi one-element weights (a clipped
 * linear's range). Row-wise at each operand's own pitch, as in libref. */
AVX_KERNEL(relu) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(X) < rows * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* v = in_row(X, rowoff(X, r, n), n, sx);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(sy + i, vf_max(vf_loadu(v + i), vf_zero()));
        for (; i < n; ++i) sy[i] = v[i] > 0.0f ? v[i] : 0.0f;
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

AVX_KERNEL(clamp) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *L = &a->t[1], *H = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0 || numel(L) < 1 || numel(H) < 1) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(X) < rows * n) return RAD_E_SHAPE;
    const float lo = ldt(L, 0), hi = ldt(H, 0);

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* v = in_row(X, rowoff(X, r, n), n, sx);
        for (int64_t i = 0; i < n; ++i) {
            float t = v[i] < lo ? lo : v[i];
            sy[i] = t > hi ? hi : t;
        }
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv2d */
/* libref's segment-wise convolution over (time, width), channels innermost. The weight is
 * transposed once per launch from PyTorch's [cout, cin, kh, kw] (or [cout, cin, kw, kh] with
 * t_last) to [kh][kw][cin][cout], so the inner loop is a vector FMA over cout rather than a dot
 * product over cin. Same sum in a different order plus the optional bias, one rounding; then
 * `act` and a second rounding, as libref does. */
AVX_KERNEL(conv2d) {
    if (!a || a->n_t < 5 || !a->t[0].data || !a->t[1].data || !a->t[2].data || !a->t[4].data)
        return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Wt = &a->t[1], *cu = &a->t[2], *Y = &a->t[4];
    const RadTensor* V = a->t[3].data ? &a->t[3] : nullptr;
    const RadTensor* B = a->n_t > 5 && a->t[5].data ? &a->t[5] : nullptr;
    const bool tlast = p_int(a, "t_last", 0) != 0;
    const char* act = p_str(a, "act", "none");
    const bool gelu = !std::strcmp(act, "gelu");
    if (!gelu && std::strcmp(act, "none")) return RAD_E_UNSUPPORTED;
    const int64_t W = p_int(a, "width", 0), cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0);
    const int64_t kh = p_int(a, "kh", 0), kw = p_int(a, "kw", 0);
    const int64_t st = p_int(a, "stride", 1), pad = p_int(a, "pad", 0);
    const int64_t cdiv = p_int(a, "cu_div", 1), vst = p_int(a, "valid_stride", 1);
    if (W <= 0 || cin <= 0 || cout <= 0 || kh <= 0 || kw <= 0 || st <= 0 || pad < 0 || cdiv <= 0 ||
        vst <= 0)
        return RAD_E_SHAPE;
    const int64_t Wo = (W + 2 * pad - kw) / st + 1;
    if (Wo <= 0 || numel(Wt) < cout * cin * kh * kw) return RAD_E_SHAPE;
    if (B && numel(B) < cout) return RAD_E_SHAPE;
    const int64_t S = numel(cu) - 1;
    if (S <= 0) return RAD_E_SHAPE;
    const int64_t rin = W * cin, rout = Wo * cout;
    const int64_t Min = rows_of(X, rin), Mout = rows_of(Y, rout);
    if (Min <= 0 || Mout <= 0 || ldi(cu, offlin(cu, S)) <= ldi(cu, offlin(cu, 0)))
        return RAD_E_SHAPE;                         /* zero rows is not a call (docs/OPS.md) */

    float* wt = (float*)scratch_raw((size_t)(kh * kw * cin * cout + cout) * sizeof(float) + 64);
    float* bias = wt + kh * kw * cin * cout;
    for (int64_t co = 0; co < cout; ++co) bias[co] = B ? ldt(B, offlin(B, co)) : 0.0f;
    const int64_t ws = laststride(Wt);
    for (int64_t co = 0; co < cout; ++co)
        for (int64_t ci = 0; ci < cin; ++ci)
            for (int64_t i = 0; i < kh; ++i)
                for (int64_t j = 0; j < kw; ++j)
                    wt[((i * kw + j) * cin + ci) * cout + co] =
                        ldt(Wt, (tlast ? ((co * cin + ci) * kw + j) * kh + i
                                       : ((co * cin + ci) * kh + i) * kw + j) * ws);

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = ldi(cu, offlin(cu, s)), c1 = ldi(cu, offlin(cu, s + 1));
        if (c0 % cdiv || c1 % cdiv || (c0 / cdiv) % st || c1 < c0) return RAD_E_SHAPE;
        const int64_t t0 = c0 / cdiv, T = c1 / cdiv - t0;
        if (T <= 0) continue;
        const int64_t To = (T + 2 * pad - kh) / st + 1, o0 = t0 / st;
        if (To <= 0) continue;
        if (t0 + T > Min || o0 + To > Mout) return RAD_E_SHAPE;
        if (V && numel(V) < (t0 + T - 1) * vst + 1) return RAD_E_SHAPE;

        AVX_PARALLEL_FOR_IF(To * rout * cin * kh * kw >= AVX_PAR_MIN)
        for (int64_t to = 0; to < To; ++to) {
            float* acc = scratch_f32((size_t)rout);
            float* sx = scratch_f32_b((size_t)rin);
            for (int64_t e = 0; e < rout; ++e) acc[e] = 0.0f;
            for (int64_t i = 0; i < kh; ++i) {
                const int64_t ti = to * st - pad + i;
                if (ti < 0 || ti >= T) continue;
                const int64_t tg = t0 + ti;
                if (V && ldi(V, offlin(V, tg * vst)) == 0) continue;
                const float* xr = in_row(X, rowoff(X, tg, rin), rin, sx);
                for (int64_t wo = 0; wo < Wo; ++wo) {
                    float* ar = acc + wo * cout;
                    for (int64_t j = 0; j < kw; ++j) {
                        const int64_t wi = wo * st - pad + j;
                        if (wi < 0 || wi >= W) continue;
                        const float* xv = xr + wi * cin;
                        const float* wb = wt + (i * kw + j) * cin * cout;
                        for (int64_t ci = 0; ci < cin; ++ci) {
                            const vf b = vf_set1(xv[ci]);
                            const float* wr = wb + ci * cout;
                            int64_t c = 0;
                            for (; c + VF_N <= cout; c += VF_N)
                                vf_storeu(ar + c, vf_fma(b, vf_loadu(wr + c), vf_loadu(ar + c)));
                            for (; c < cout; ++c) ar[c] += xv[ci] * wr[c];
                        }
                    }
                }
            }
            if (B)
                for (int64_t wo = 0; wo < Wo; ++wo)
                    for (int64_t co = 0; co < cout; ++co) acc[wo * cout + co] += bias[co];
            out_row(Y, rowoff(Y, o0 + to, rout), rout, acc);
            if (gelu) {
                /* The biased sum as y's dtype holds it, then the activation of the rounded value. */
                const float* yr = in_row(Y, rowoff(Y, o0 + to, rout), rout, acc);
                for (int64_t e = 0; e < rout; ++e) acc[e] = act_of(1, yr[e], RAD_F32);
                out_row(Y, rowoff(Y, o0 + to, rout), rout, acc);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv2d_unfold */
/* libref's receptive fields as rows, a copy: each input row is widened once and the taps are
 * gathered from it. */
AVX_KERNEL(conv2d_unfold) {
    if (!a || a->n_t < 4 || !a->t[0].data || !a->t[1].data || !a->t[3].data) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *cu = &a->t[1], *Y = &a->t[3];
    const RadTensor* V = a->t[2].data ? &a->t[2] : nullptr;
    const int64_t W = p_int(a, "width", 0), cin = p_int(a, "cin", 0);
    const int64_t kh = p_int(a, "kh", 0), kw = p_int(a, "kw", 0);
    const int64_t st = p_int(a, "stride", 1), pad = p_int(a, "pad", 0);
    const int64_t cdiv = p_int(a, "cu_div", 1), vst = p_int(a, "valid_stride", 1);
    const bool tlast = p_int(a, "t_last", 0) != 0;
    if (W <= 0 || cin <= 0 || kh <= 0 || kw <= 0 || st <= 0 || pad < 0 || cdiv <= 0 || vst <= 0)
        return RAD_E_SHAPE;
    const int64_t Wo = (W + 2 * pad - kw) / st + 1, K = cin * kh * kw;
    const int64_t S = numel(cu) - 1;
    if (Wo <= 0 || S <= 0) return RAD_E_SHAPE;
    const int64_t rin = W * cin, Min = rows_of(X, rin), Mout = rows_of(Y, K);
    if (Min <= 0 || Mout <= 0 || ldi(cu, offlin(cu, S)) <= ldi(cu, offlin(cu, 0)))
        return RAD_E_SHAPE;                         /* zero rows is not a call (docs/OPS.md) */

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = ldi(cu, offlin(cu, s)), c1 = ldi(cu, offlin(cu, s + 1));
        if (c0 % cdiv || c1 % cdiv || (c0 / cdiv) % st || c1 < c0) return RAD_E_SHAPE;
        const int64_t t0 = c0 / cdiv, T = c1 / cdiv - t0;
        if (T <= 0) continue;
        const int64_t To = (T + 2 * pad - kh) / st + 1, o0 = t0 / st;
        if (To <= 0) continue;
        if (t0 + T > Min || (o0 + To) * Wo > Mout) return RAD_E_SHAPE;
        if (V && numel(V) < (t0 + T - 1) * vst + 1) return RAD_E_SHAPE;

        AVX_PARALLEL_FOR_IF(To * Wo * K >= AVX_PAR_MIN)
        for (int64_t to = 0; to < To; ++to) {
            float* row = scratch_f32((size_t)K);
            float* sx = scratch_f32_b((size_t)(kh * rin));
            const float** xr = (const float**)scratch_raw((size_t)kh * sizeof(const float*));
            for (int64_t i = 0; i < kh; ++i) {
                const int64_t ti = to * st - pad + i, tg = t0 + ti;
                const bool ok = ti >= 0 && ti < T && !(V && ldi(V, offlin(V, tg * vst)) == 0);
                xr[i] = ok ? in_row(X, rowoff(X, tg, rin), rin, sx + i * rin) : nullptr;
            }
            for (int64_t wo = 0; wo < Wo; ++wo) {
                for (int64_t i = 0; i < kh; ++i)
                    for (int64_t j = 0; j < kw; ++j) {
                        const int64_t wi = wo * st - pad + j;
                        const float* xv = xr[i] && wi >= 0 && wi < W ? xr[i] + wi * cin : nullptr;
                        for (int64_t ci = 0; ci < cin; ++ci) {
                            const int64_t k = tlast ? (ci * kw + j) * kh + i : (ci * kh + i) * kw + j;
                            row[k] = xv ? xv[ci] : 0.0f;
                        }
                    }
                out_row(Y, rowoff(Y, (o0 + to) * Wo + wo, K), K, row);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv1d_dw */
/* libref's causal depthwise convolution over time, plus the optional bias. The weight is
 * transposed to [k][C] once per launch so each tap is a vector FMA over the row. */
AVX_KERNEL(conv1d_dw) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Wt = &a->t[1], *cu = &a->t[2], *Y = &a->t[3];
    const RadTensor* B = rad_arg_in(a, 4);
    const int64_t C = p_int(a, "n", 0), K = p_int(a, "k", 0), cdiv = p_int(a, "cu_div", 1);
    if (C <= 0 || K <= 0 || cdiv <= 0 || numel(Wt) < C * K || (B && numel(B) < C))
        return RAD_E_SHAPE;
    const int64_t S = numel(cu) - 1;
    if (S <= 0) return RAD_E_SHAPE;
    const int64_t Mx = rows_of(X, C), My = rows_of(Y, C);
    if (Mx <= 0 || My <= 0 || ldi(cu, offlin(cu, S)) <= ldi(cu, offlin(cu, 0)))
        return RAD_E_SHAPE;                         /* zero rows is not a call (docs/OPS.md) */

    /* The taps tap-major, then the bias (zeros when unbound), once a launch. */
    float* wt = (float*)scratch_raw((size_t)((K + 1) * C) * sizeof(float) + 64);
    float* bias = wt + K * C;
    for (int64_t c = 0; c < C; ++c)
        for (int64_t j = 0; j < K; ++j) wt[j * C + c] = ldt(Wt, (c * K + j) * laststride(Wt));
    for (int64_t c = 0; c < C; ++c) bias[c] = B ? ldt(B, c) : 0.0f;

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = ldi(cu, offlin(cu, s)), c1 = ldi(cu, offlin(cu, s + 1));
        if (c0 % cdiv || c1 % cdiv || c1 < c0) return RAD_E_SHAPE;
        const int64_t t0 = c0 / cdiv, t1 = c1 / cdiv;
        if (t1 > Mx || t1 > My) return RAD_E_SHAPE;

        AVX_PARALLEL_FOR_IF((t1 - t0) * C * K >= AVX_PAR_MIN)
        for (int64_t t = t0; t < t1; ++t) {
            float* acc = scratch_f32((size_t)C);
            float* sx = scratch_f32_b((size_t)C);
            for (int64_t c = 0; c < C; ++c) acc[c] = 0.0f;
            for (int64_t j = 0; j < K; ++j) {
                const int64_t tt = t - (K - 1) + j;
                if (tt < t0) continue;
                const float* xr = in_row(X, rowoff(X, tt, C), C, sx);
                const float* wr = wt + j * C;
                int64_t c = 0;
                for (; c + VF_N <= C; c += VF_N)
                    vf_storeu(acc + c, vf_fma(vf_loadu(xr + c), vf_loadu(wr + c), vf_loadu(acc + c)));
                for (; c < C; ++c) acc[c] += xr[c] * wr[c];
            }
            if (B)
                for (int64_t c = 0; c < C; ++c) acc[c] += bias[c];
            out_row(Y, rowoff(Y, t, C), C, acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rel_sinusoid */
/* libref's table, with the same f32 steps and libm calls. */
AVX_KERNEL(rel_sinusoid) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor* Y = &a->t[0];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || n % 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0) return RAD_E_SHAPE;
    const int64_t h = n / 2;
    const double mt = (double)p_f32(a, "max_timescale", 10000.0f);
    const float inc = (float)(std::log(mt / 1.0) / (double)(h - 1 > 1 ? h - 1 : 1));

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t d = 0; d < rows; ++d) {
        float* sy = scratch_f32((size_t)n);
        for (int64_t i = 0; i < h; ++i) {
            const float inv = sc_exp((float)i * -inc);
            const float ang = (float)d * inv;
            sy[i] = sc_sin(ang);
            sy[h + i] = sc_cos(ang);
        }
        out_row(Y, rowoff(Y, d, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ seg_sinusoid */
/* libref's per-segment table, to the operation: position (r - start) mod period. */
AVX_KERNEL(seg_sinusoid) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *cu = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), period = p_int(a, "period", 0), cdiv = p_int(a, "cu_div", 1);
    if (n <= 0 || n % 2 || period <= 0 || cdiv <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n), S = numel(cu) - 1;
    if (rows <= 0 || S <= 0 || ldi(cu, offlin(cu, S)) <= ldi(cu, offlin(cu, 0)))
        return RAD_E_SHAPE;                         /* zero rows is not a call (docs/OPS.md) */
    const int64_t h = n / 2;
    const double mt = (double)p_f32(a, "max_timescale", 10000.0f);
    const float inc = (float)(std::log(mt / 1.0) / (double)(h - 1 > 1 ? h - 1 : 1));

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = ldi(cu, offlin(cu, s)), c1 = ldi(cu, offlin(cu, s + 1));
        if (c0 % cdiv || c1 % cdiv || c1 < c0 || c1 / cdiv > rows) return RAD_E_SHAPE;
        const int64_t r0 = c0 / cdiv, r1 = c1 / cdiv;
        AVX_PARALLEL_FOR_IF((r1 - r0) * n >= AVX_PAR_MIN)
        for (int64_t r = r0; r < r1; ++r) {
            float* sy = scratch_f32((size_t)n);
            const int64_t p = (r - r0) % period;
            for (int64_t i = 0; i < h; ++i) {
                const float inv = sc_exp((float)i * -inc);
                const float ang = (float)p * inv;
                sy[i] = sc_sin(ang);
                sy[h + i] = sc_cos(ang);
            }
            out_row(Y, rowoff(Y, r, n), n, sy);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ seg_mean */
/* One output row per cu_seqlens segment: the mean of its rows, summed in double and rounded once,
 * as in libref. */
AVX_KERNEL(seg_mean) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *cu = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t S = numel(cu) - 1;
    if (n <= 0 || S <= 0 || numel(Y) < S * n) return RAD_E_SHAPE;
    const int64_t total = ldi(cu, offlin(cu, S));
    if (total <= 0 || numel(X) < total * n) return RAD_E_SHAPE;   /* zero rows is not a call */

    AVX_PARALLEL_FOR_IF(S >= 2)
    for (int64_t s = 0; s < S; ++s) {
        const int64_t b0 = ldi(cu, offlin(cu, s)), b1 = ldi(cu, offlin(cu, s + 1));
        double* acc = (double*)scratch_raw((size_t)n * sizeof(double) + 64);
        float* sx = scratch_f32_b((size_t)n);
        float* sy = scratch_f32_c((size_t)n);
        for (int64_t i = 0; i < n; ++i) acc[i] = 0.0;
        for (int64_t r = b0; r < b1; ++r) {
            const float* x = in_row(X, rowoff(X, r, n), n, sx);
            for (int64_t i = 0; i < n; ++i) acc[i] += x[i];
        }
        const double inv = b1 > b0 ? 1.0 / (double)(b1 - b0) : 0.0;
        for (int64_t i = 0; i < n; ++i) sy[i] = b1 > b0 ? (float)(acc[i] * inv) : 0.0f;
        out_row(Y, rowoff(Y, s, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ seg_softmax_pool */
/* One output row per cu_seqlens segment: its rows summed with the softmax of their scores as
 * weights, the softmax in f32 with the maximum subtracted, the sum in f32, rounded once, as in
 * libref. A segment is a question's few options, so the scores are read plainly. */
AVX_KERNEL(seg_softmax_pool) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *Sc = &a->t[0], *V = &a->t[1], *cu = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    const int64_t S = numel(cu) - 1;
    if (n <= 0 || S <= 0 || numel(Y) < S * n) return RAD_E_SHAPE;
    const int64_t total = ldi(cu, offlin(cu, S));
    if (total <= 0 || numel(V) < total * n || numel(Sc) < total) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(S >= 2)
    for (int64_t s = 0; s < S; ++s) {
        const int64_t b0 = ldi(cu, offlin(cu, s)), b1 = ldi(cu, offlin(cu, s + 1));
        float* acc = scratch_f32_b((size_t)n);
        float* sy = scratch_f32_c((size_t)n);
        float one = 0.0f;
        float m = -INFINITY, z = 0.0f;
        for (int64_t r = b0; r < b1; ++r) m = std::max(m, in_row(Sc, rowoff(Sc, r, 1), 1, &one)[0]);
        for (int64_t r = b0; r < b1; ++r) z += std::exp(in_row(Sc, rowoff(Sc, r, 1), 1, &one)[0] - m);
        for (int64_t i = 0; i < n; ++i) acc[i] = 0.0f;
        for (int64_t r = b0; r < b1; ++r) {
            const float w = std::exp(in_row(Sc, rowoff(Sc, r, 1), 1, &one)[0] - m) / z;
            const float* x = in_row(V, rowoff(V, r, n), n, sy);
            for (int64_t i = 0; i < n; ++i) acc[i] += w * x[i];
        }
        for (int64_t i = 0; i < n; ++i) sy[i] = b1 > b0 ? acc[i] : 0.0f;
        out_row(Y, rowoff(Y, s, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pool_rows */
/* As libref: the mean of each k consecutive rows, summed in f32 and rounded to y's dtype, then the
 * multiplier and a second rounding (the unfused pair). Parallel over output rows. */
AVX_KERNEL(pool_rows) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), k = p_int(a, "k", 0);
    const float mult = p_f32(a, "mult", 1.0f);
    if (n <= 0 || k <= 0) return RAD_E_SHAPE;
    const int64_t rows_in = rows_of(X, n);
    if (rows_in <= 0 || rows_in % k) return RAD_E_SHAPE;
    const int64_t rows = rows_in / k;
    if (numel(Y) < rows * n) return RAD_E_SHAPE;
    const float inv = 1.0f / (float)k;

    AVX_PARALLEL_FOR_IF(rows * n * k >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* acc = scratch_f32((size_t)n);
        float* sx = scratch_f32_b((size_t)n);
        for (int64_t i = 0; i < n; ++i) acc[i] = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            const float* v = in_row(X, rowoff(X, r * k + j, n), n, sx);
            for (int64_t i = 0; i < n; ++i) acc[i] += v[i];
        }
        for (int64_t i = 0; i < n; ++i) acc[i] *= inv;
        roundtrip(Y->dtype, acc, n, sx);
        for (int64_t i = 0; i < n; ++i) acc[i] *= mult;
        out_row(Y, rowoff(Y, r, n), n, acc);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pool_cols */
/* As libref: the mean of every k consecutive columns of each row, summed in f32 and rounded to
 * y's dtype, then the multiplier and a second rounding (the unfused pair). pool_rows' dual.
 * Parallel over rows. */
AVX_KERNEL(pool_cols) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), k = p_int(a, "k", 0);
    const float mult = p_f32(a, "mult", 1.0f);
    if (n <= 0 || k <= 0 || n % k) return RAD_E_SHAPE;
    const int64_t M = rows_of(X, n), nw = n / k;
    if (M <= 0 || numel(Y) < M * nw) return RAD_E_SHAPE;
    const float inv = 1.0f / (float)k;

    AVX_PARALLEL_FOR_IF(M * nw * k >= AVX_PAR_MIN)
    for (int64_t r = 0; r < M; ++r) {
        float* acc = scratch_f32((size_t)nw);
        float* sx = scratch_f32_b((size_t)n);
        const float* v = in_row(X, rowoff(X, r, n), n, sx);
        for (int64_t c = 0; c < nw; ++c) {
            float s = 0.0f;
            for (int64_t j = 0; j < k; ++j) s += v[c * k + j];
            acc[c] = s * inv;
        }
        roundtrip(Y->dtype, acc, nw, sx);
        for (int64_t c = 0; c < nw; ++c) acc[c] *= mult;
        out_row(Y, rowoff(Y, r, nw), nw, acc);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ point_fourier */
/* As libref: [x | sin(x_i 2^j) | cos(x_i 2^j)] (times pi with `pi` 1), the products, sines and
 * cosines f32 and rounded once into y. Parallel over rows. */
AVX_KERNEL(point_fourier) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), F = p_int(a, "freqs", 0);
    const bool pi = p_int(a, "pi", 0) != 0;
    if (n <= 0 || F <= 0 || F > 30) return RAD_E_SHAPE;
    const int64_t w = n * (2 * F + 1);
    const int64_t M = rows_of(X, n);
    if (M <= 0 || numel(Y) < M * w) return RAD_E_SHAPE;
    float f[30];
    for (int64_t j = 0; j < F; ++j)
        f[j] = pi ? (float)std::ldexp(3.14159265358979323846, (int)j) : std::ldexp(1.0f, (int)j);

    AVX_PARALLEL_FOR_IF(M * w >= AVX_PAR_MIN)
    for (int64_t r = 0; r < M; ++r) {
        float* acc = scratch_f32((size_t)w);
        float* sx = scratch_f32_b((size_t)n);
        const float* v = in_row(X, rowoff(X, r, n), n, sx);
        for (int64_t i = 0; i < n; ++i) {
            acc[i] = v[i];
            for (int64_t j = 0; j < F; ++j) {
                const float t = v[i] * f[j];
                acc[n + i * F + j] = std::sin(t);
                acc[n + n * F + i * F + j] = std::cos(t);
            }
        }
        out_row(Y, rowoff(Y, r, w), w, acc);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ position_scale */
/* Ministral 3's query multiplier is rounded before multiplication, as in libref. */
AVX_KERNEL(position_scale) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *pos = &a->t[1];
    // M in the resolved geometry is a capacity band; issued views carry the active rows.
    const int64_t M = x->rank == 2 ? x->shape[0] : 0;
    const int64_t n = p_int(a, "n", 0), orig = p_int(a, "original", 0);
    const float beta = p_f32(a, "beta", -1);
    if (M <= 0 || M > p_int(a, "M", 0) || n <= 0 || orig <= 0 || !std::isfinite(beta) || beta < 0 ||
        x->rank != 2 || x->shape[0] < M || x->shape[1] != n || pos->rank != 1 || numel(pos) < M)
        return RAD_E_SHAPE;
    if (pos->dtype != RAD_I32 || (x->dtype != RAD_BF16 && x->dtype != RAD_F32)) return RAD_E_DTYPE;
    for (int64_t r = 0; r < M; ++r)
        if (ldi(pos, offlin(pos, r)) < 0) return RAD_E_INVAL;
    AVX_PARALLEL_FOR_IF(M * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t p = ldi(pos, offlin(pos, r));
        const float scale = narrow1(x->dtype, 1.0f + beta * std::log(1.0f + (float)(p / orig)));
        for (int64_t j = 0; j < n; ++j) {
            const int64_t at = rowoff(x, r, n) + j * laststride(x);
            stt(x, at, ldt(x, at) * scale);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ scale_rows */
/* y[m, i] = x[m, i] * act(s[m]), plus add[m, i] when the optional residual is passed.
 *
 * The activation applies to the per-row scalar, as a gate does (Qwen3.5-MoE scales its shared
 * expert by sigmoid(shared_expert_gate(x))).
 *
 * With a residual, the product is narrowed to y's dtype before the add, as libref (and a fused
 * kernel) does by storing the product and reading it back. Done here in a scratch buffer. */
AVX_KERNEL(scale_rows) {
    /* `add` (operand 2) is optional; the other three are required. */
    if (a->n_t < 4 || !rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 3))
        return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *sc = &a->t[1], *y = &a->t[3];
    const RadTensor* ad = rad_arg_in(a, 2);
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(x) < rows * n || numel(sc) < rows) return RAD_E_SHAPE;
    if (ad && numel(ad) < rows * n) return RAD_E_SHAPE;
    const char* act = p_str(a, "act", "none");
    const bool sig = act && !std::strcmp(act, "sigmoid");
    if (act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return RAD_E_INVAL;

    const int64_t ss = laststride(sc);
    const size_t tmpb = (size_t)n * sizeof(float) + 64;
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const float v0 = ldt(sc, r * ss);
        /* One value per row, so use libm's expf as the oracle does, not the vector polynomial. */
        const float v = sig ? 1.0f / (1.0f + sc_exp(-v0)) : v0;
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* xr = in_row(x, rowoff(x, r, n), n, sx);
        const vf vv = vf_set1(v);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(sy + i, vf_mul(vf_loadu(xr + i), vv));
        for (; i < n; ++i) sy[i] = xr[i] * v;
        if (ad) {
            void* tp = scratch_raw(tmpb);
            roundtrip(y->dtype, sy, n, tp);
            float* sa = scratch_f32_c((size_t)n);
            const float* av = in_row(ad, rowoff(ad, r, n), n, sa);
            i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(sy + i, vf_add(vf_loadu(sy + i), vf_loadu(av + i)));
            for (; i < n; ++i) sy[i] += av[i];
        }
        out_row(y, rowoff(y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ silu_mul */
/* y[.., i] = silu(gate_up[.., i]) * gate_up[.., n + i]. The gate is the first half, the order in
 * which rad-convert fuses gate and up into one weight.
 *
 * A bf16 output uses vf_silu_fast (avx_vec.h): the exact path's ~0.6 ulp is wasted on a store that
 * rounds at 3e-3. f32 outputs keep the exact path. Only for ops whose product is consumed once;
 * vf_sigmoid feeds gate chains whose errors accumulate. */

AVX_KERNEL(silu_mul) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *gu = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(gu) < rows * 2 * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        /* One in_row over the whole 2n row: the halves are adjacent, and one conversion saves a
         * call and keeps the prefetcher on one stream. */
        float* s = scratch_f32((size_t)(2 * n));
        float* sy = scratch_f32_b((size_t)n);
        const float* g = in_row(gu, rowoff(gu, r, 2 * n), 2 * n, s);
        const float* u = g + n;
#if AVX_LEVEL >= 2
        /* Keyed on the output dtype (see above). */
        if (y->dtype == RAD_BF16) {
            int64_t i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(sy + i, vf_mul(vf_silu_fast(vf_loadu(g + i)), vf_loadu(u + i)));
            for (; i < n; ++i) {
                const float gi = g[i];
                sy[i] = (gi / (1.0f + sc_exp(-gi))) * u[i];
            }
            out_row(y, rowoff(y, r, n), n, sy);
            continue;
        }
#endif
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(sy + i, vf_mul(vf_silu(vf_loadu(g + i)), vf_loadu(u + i)));
        for (; i < n; ++i) {
            const float gi = g[i];
            sy[i] = (gi / (1.0f + sc_exp(-gi))) * u[i];
        }
        out_row(y, rowoff(y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu_mul
 *
 * y[i] = gelu_tanh(gate_up[i]) * gate_up[n + i], gate first as in silu_mul. Uses libref's scalar
 * gelu_tanh formula (act_gelu_tanh, std::tanh) rather than a second, vectorised definition. */
AVX_KERNEL(gelu_mul) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *gu = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(gu) < rows * 2 * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        /* One in_row over the whole 2n row, as in silu_mul. */
        float* s = scratch_f32((size_t)(2 * n));
        float* sy = scratch_f32_b((size_t)n);
        const float* g = in_row(gu, rowoff(gu, r, 2 * n), 2 * n, s);
        const float* u = g + n;
        for (int64_t i = 0; i < n; ++i) {
            const float x = g[i];
            const float t = sc_tanh(0.7978845608028654f * (x + 0.044715f * x * x * x));
            sy[i] = 0.5f * x * (1.0f + t) * u[i];
        }
        out_row(y, rowoff(y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu / silu / sigmoid
 *
 * These run over the whole tensor, as libref's `unary` does, in fixed L1-sized tiles (the op has
 * no row length).
 *
 * gelu stays scalar on purpose: libref uses libm's erff, and a minimax erf polynomial would be a
 * second definition of gelu. The dtype switch is still hoisted. If gelu lands on a hot path, a
 * polynomial is the next step (--bench gelu); Qwen and Gemma use silu. */
enum { UN_GELU = 0, UN_SILU = 1, UN_SIGMOID = 2 };

template <int WHICH>
static int unary(const RadArgs* a) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel) return RAD_E_SHAPE;

    /* Strided: element at a time through `offlin`, as libref does. */
    if (!dense_row(x) || !dense_row(y) || !rad_tensor_is_contiguous(x) ||
        !rad_tensor_is_contiguous(y)) {
        AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
        for (int64_t i = 0; i < nel; ++i) {
            const float v = ldt(x, offlin(x, i));
            float o;
            if (WHICH == UN_GELU)         o = 0.5f * v * (1.0f + sc_erf(v * 0.70710678118654752f));
            else if (WHICH == UN_SILU)    o = v / (1.0f + sc_exp(-v));
            else                          o = 1.0f / (1.0f + sc_exp(-v));
            stt(y, offlin(y, i), o);
        }
        return RAD_OK;
    }

    const int64_t TILE = 4096;
    const int64_t ntile = (nel + TILE - 1) / TILE;
    AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
    for (int64_t t = 0; t < ntile; ++t) {
        const int64_t off = t * TILE;
        const int64_t m = (nel - off) < TILE ? (nel - off) : TILE;
        float* s  = scratch_f32((size_t)TILE);
        float* so = scratch_f32_b((size_t)TILE);
        const float* v = in_row(x, off, m, s);
        int64_t i = 0;
        if (WHICH != UN_GELU) {
            for (; i + VF_N <= m; i += VF_N) {
                vf u = vf_loadu(v + i);
                vf_storeu(so + i, WHICH == UN_SILU ? vf_silu(u) : vf_sigmoid(u));
            }
        }
        for (; i < m; ++i) {
            const float u = v[i];
            if (WHICH == UN_GELU)      so[i] = 0.5f * u * (1.0f + sc_erf(u * 0.70710678118654752f));
            else if (WHICH == UN_SILU) so[i] = u / (1.0f + sc_exp(-u));
            else                       so[i] = 1.0f / (1.0f + sc_exp(-u));
        }
        out_row(y, off, m, so);
    }
    return RAD_OK;
}

AVX_KERNEL(gelu)    { return unary<UN_GELU>(a); }
AVX_KERNEL(silu)    { return unary<UN_SILU>(a); }
AVX_KERNEL(sigmoid) { return unary<UN_SIGMOID>(a); }

/* ------------------------------------------------------------------ fill */
/* y = value over [M, n] at y's own row pitch, rounded once by the store. */
AVX_KERNEL(fill) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor* y = &a->t[0];
    const int64_t n = p_int(a, "n", y->rank >= 1 ? y->shape[y->rank - 1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(y, n);
    if (M <= 0) return RAD_E_SHAPE;
    const float v = p_f32(a, "value", 0.0f);
    AVX_PARALLEL_FOR_IF(M * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, v);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ logit_softcap */
/* y = cap * tanh(x / cap). Not a unary<WHICH> because `cap` is a runtime parameter. The vector
 * path uses tanh(z) = 2*sigmoid(2z) - 1 on vf_sigmoid; the scalar tail uses libm's tanhf, as
 * libref does. */
AVX_KERNEL(logit_softcap) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel) return RAD_E_SHAPE;
    const float cap = p_f32(a, "cap", 30.0f);
    if (!(cap > 0.0f)) return RAD_E_INVAL;

    if (!dense_row(x) || !dense_row(y) || !rad_tensor_is_contiguous(x) ||
        !rad_tensor_is_contiguous(y)) {
        AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
        for (int64_t i = 0; i < nel; ++i) {
            const float v = ldt(x, offlin(x, i));
            stt(y, offlin(y, i), cap * sc_tanh(v / cap));
        }
        return RAD_OK;
    }

    const int64_t TILE = 4096;
    const int64_t ntile = (nel + TILE - 1) / TILE;
    AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
    for (int64_t t = 0; t < ntile; ++t) {
        const int64_t off = t * TILE;
        const int64_t m = (nel - off) < TILE ? (nel - off) : TILE;
        float* s  = scratch_f32((size_t)TILE);
        float* so = scratch_f32_b((size_t)TILE);
        const float* v = in_row(x, off, m, s);
        const vf z = vf_set1(2.0f / cap);   /* sigmoid argument: 2x / cap */
        const vf c2 = vf_set1(2.0f * cap);  /* 2*cap * sigmoid(...)      */
        const vf c1 = vf_set1(-cap);        /* ... - cap                 */
        int64_t i = 0;
        for (; i + VF_N <= m; i += VF_N)
            vf_storeu(so + i, vf_fma(c2, vf_sigmoid(vf_mul(vf_loadu(v + i), z)), c1));
        for (; i < m; ++i)
            so[i] = cap * sc_tanh(v[i] / cap);
        out_row(y, off, m, so);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ scale */
/* y = x * s, with s a [1] weight broadcast over every element. A separate op rather than `mul`
 * with a [1] operand, for the reason `mul` refuses a scalar broadcast (see the file header). */
AVX_KERNEL(scale) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *s = &a->t[1], *y = &a->t[2];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel || numel(s) < 1) return RAD_E_SHAPE;
    const float sv = ldt(s, 0);

    if (!dense_row(x) || !dense_row(y) || !rad_tensor_is_contiguous(x) ||
        !rad_tensor_is_contiguous(y)) {
        AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
        for (int64_t i = 0; i < nel; ++i)
            stt(y, offlin(y, i), ldt(x, offlin(x, i)) * sv);
        return RAD_OK;
    }

    const int64_t TILE = 4096;
    const int64_t ntile = (nel + TILE - 1) / TILE;
    AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
    for (int64_t t = 0; t < ntile; ++t) {
        const int64_t off = t * TILE;
        const int64_t m = (nel - off) < TILE ? (nel - off) : TILE;
        float* sbuf  = scratch_f32((size_t)TILE);
        float* so = scratch_f32_b((size_t)TILE);
        const float* v = in_row(x, off, m, sbuf);
        const vf sc = vf_set1(sv);
        int64_t i = 0;
        for (; i + VF_N <= m; i += VF_N)
            vf_storeu(so + i, vf_mul(vf_loadu(v + i), sc));
        for (; i < m; ++i)
            so[i] = v[i] * sv;
        out_row(y, off, m, so);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gather_rows */
/* y[i, :] = x[idx[i], :]. A negative index writes a zero row (embed_lookup's padding rule, used
 * when a step has fewer rows than the buffer). An index >= x's row count is RAD_E_INVAL, checked
 * before the loop because an OpenMP body cannot return. */
AVX_KERNEL(gather_rows) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *idx = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", y->rank >= 1 ? y->shape[y->rank - 1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(y, n);
    const int64_t src_rows = rows_of(x, n);
    if (M <= 0 || src_rows <= 0 || numel(idx) < M) return RAD_E_SHAPE;

    for (int64_t m = 0; m < M; ++m)
        if (ldi(idx, offlin(idx, m)) >= src_rows) return RAD_E_INVAL;

    /* Matching whole-byte dtypes (the normal case) copy rows bytewise, which is faster than a
     * widen/narrow round trip and cannot change the bits. A sub-byte row can start mid-byte. */
    const bool same = x->dtype == y->dtype && dense_row(x) && dense_row(y) &&
                      rad_dtype_bits(x->dtype) % 8 == 0;
    const size_t rowb = same ? (size_t)((rad_dtype_bits(x->dtype) * n) / 8) : 0;

    AVX_PARALLEL_FOR_IF(M * n >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r  = ldi(idx, offlin(idx, m));
        const int64_t yb = rowoff(y, m, n), ys = laststride(y);
        if (r < 0) {
            if (same) std::memset((void*)byte_at(y->data, y->dtype, yb), 0, rowb);
            else for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, 0.0f);
            continue;
        }
        const int64_t xb = rowoff(x, r, n);
        if (same) {
            std::memcpy((void*)byte_at(y->data, y->dtype, yb),
                        byte_at(x->data, x->dtype, xb), rowb);
        } else {
            float* s = scratch_f32((size_t)n);
            out_row(y, yb, n, in_row(x, xb, n, s));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ scatter_rows */
/* x[idx[i], :] = v[i, :], the mirror of gather_rows. `x` is INOUT and partially written: rows no
 * index names keep their contents, and a negative index writes nothing. Indices are distinct by
 * contract (RAD_OPD_F_IDX_UNIQUE), so rows can be written by threads in any order. */
AVX_KERNEL(scatter_rows) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *idx = &a->t[1], *x = &a->t[2];
    const int64_t n = p_int(a, "n", v->rank >= 1 ? v->shape[v->rank - 1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(v, n);
    const int64_t dst_rows = rows_of(x, n);
    if (M <= 0 || dst_rows <= 0 || numel(idx) < M) return RAD_E_SHAPE;

    for (int64_t m = 0; m < M; ++m)
        if (ldi(idx, offlin(idx, m)) >= dst_rows) return RAD_E_INVAL;

    /* Bytewise row copy when the dtypes match, as in gather_rows. */
    const bool same = v->dtype == x->dtype && dense_row(v) && dense_row(x) &&
                      rad_dtype_bits(x->dtype) % 8 == 0;
    const size_t rowb = same ? (size_t)((rad_dtype_bits(x->dtype) * n) / 8) : 0;

    AVX_PARALLEL_FOR_IF(M * n >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r = ldi(idx, offlin(idx, m));
        if (r < 0) continue;
        const int64_t vb = rowoff(v, m, n), xb = rowoff(x, r, n);
        if (same) {
            std::memcpy((void*)byte_at(x->data, x->dtype, xb), byte_at(v->data, v->dtype, vb),
                        rowb);
        } else {
            float* s = scratch_f32((size_t)n);
            out_row(x, xb, n, in_row(v, vb, n, s));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ cast */
/* Reads the tensors' own dtypes. `from`/`to` exist in the schema for the selector; a mismatch with
 * the tensors is a caller bug (as in libref).
 *
 * Integer-to-integer casts take an integer path, so token ids and block indices above 2^24 are not
 * rounded through f32. */
AVX_KERNEL(cast) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel) return RAD_E_SHAPE;

    const bool int_pair = (x->dtype == RAD_I32 || x->dtype == RAD_I64 || x->dtype == RAD_U32 ||
                           x->dtype == RAD_I16) &&
                          (y->dtype == RAD_I32 || y->dtype == RAD_I64 || y->dtype == RAD_U32 ||
                           y->dtype == RAD_I16);
    if (int_pair || !rad_tensor_is_contiguous(x) || !rad_tensor_is_contiguous(y)) {
        AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
        for (int64_t i = 0; i < nel; ++i) {
            if (int_pair) st_int_dt(y->dtype, y->data, offlin(y, i),
                                    ld_int_dt(x->dtype, x->data, offlin(x, i)));
            else          stt(y, offlin(y, i), ldt(x, offlin(x, i)));
        }
        return RAD_OK;
    }

    const int64_t TILE = 8192;
    const int64_t ntile = (nel + TILE - 1) / TILE;
    AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
    for (int64_t t = 0; t < ntile; ++t) {
        const int64_t off = t * TILE;
        const int64_t m = (nel - off) < TILE ? (nel - off) : TILE;
        float* s = scratch_f32((size_t)TILE);
        out_row(y, off, m, in_row(x, off, m, s));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ softmax */
/* Max-subtracted softmax in f32 over the last n elements of a row. The max subtraction is needed
 * (a 60-nat logit range overflows) and every kernel this is checked against does it.
 *
 * As in libref, the exponentials are rounded through y's dtype before the 1/sum multiply, so a
 * bf16 output has two roundings; done here in a scratch buffer instead of through `y`. */
AVX_KERNEL(softmax) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(x) < rows * n) return RAD_E_SHAPE;
    const size_t tmpb = (size_t)n * 8 + 64;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* s  = scratch_f32((size_t)n);
        float* e  = scratch_f32_b((size_t)n);
        void*  tp = scratch_raw(tmpb);
        const float* v = in_row(x, rowoff(x, r, n), n, s);
        const float mx = row_max(v, n);

        const vf vmx = vf_set1(mx);
        vf acc = vf_zero();
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            vf t = vf_exp(vf_sub(vf_loadu(v + i), vmx));
            vf_storeu(e + i, t);
            acc = vf_add(acc, t);
        }
        float sum = vf_hsum(acc);
        for (; i < n; ++i) { e[i] = sc_exp(v[i] - mx); sum += e[i]; }

        roundtrip(y->dtype, e, n, tp);
        const vf inv = vf_set1(1.0f / sum);
        i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(e + i, vf_mul(vf_loadu(e + i), inv));
        for (; i < n; ++i) e[i] *= (1.0f / sum);
        out_row(y, rowoff(y, r, n), n, e);
    }
    return RAD_OK;
}
