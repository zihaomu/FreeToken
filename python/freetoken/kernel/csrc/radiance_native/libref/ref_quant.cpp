/* ref_quant.cpp -- activation quantisation, the Walsh-Hadamard rotation, and the fused forms.
 *
 * ============================== What `group` means ==============================
 *
 * docs/OPS.md gives these ops one parameter `group` with different meanings; ref fixes them to
 * match the fast kernels:
 *
 *   quant_act_i8   one scale per row; `group` is the width of the code sums `asum` holds.
 *   quant_act_i8g  contiguous elements sharing one f32 scale, so `scale` is [M, ceil(n/group)].
 *   dequant        the same, inverted: the scale stream's extent gives the group count and
 *                  `group` each group's width.
 *   had_*          the Walsh-Hadamard rotation width (libr4d's `had`); the int8 scale is per row
 *                  and carries the 1/sqrt(group) normalisation, as r4d_had_quant_act_i8 does. A
 *                  rotation block may not cross a tensor-parallel shard, so the width must divide
 *                  the per-rank n.
 *
 * ============================== The rotation ==============================
 *
 * Unnormalised butterflies at ascending strides 1, 2, 4, ... group/2, pairing i with i+h where
 * (i & h) == 0: the order and pairing of r4d_fwht.h, so the two agree bit for bit. The 1/sqrt(group)
 * is folded into the scale, as the fast kernel does; scaling every element by a constant cannot
 * change its int8 code here.
 */
#include "ref_common.h"
#include "ref_gemm.h"   /* Mat and Scales, for gemm_nt_q_gated */
#include "ref_ops.h"

using namespace ref;

/* ------------------------------------------------------------------ quant_act_i8 */
/* The activation of the w4a8, w8a8 and w2a8 GEMMs, with libr4d's arithmetic: one scale per row,
 * max(amax, 1e-8) / 127, and codes x * (1 / scale) rounded to nearest even, stored in the step
 * order (i8_step_pos). `asum`, when passed, receives each row's code sums per `group` columns,
 * group-major [n/group][Mpad] with Mpad = ceil(M/16)*16 (the asymmetric GEMMs read whole 16-row
 * tiles); rows M..Mpad-1 are not written. */
int ref_quant_act_i8(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const RadTensor* as = t_in(a, 3);
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || n % 16) return RAD_E_SHAPE;            /* whole steps */
    if (q->dtype != RAD_I8) return RAD_E_DTYPE;
    if (laststride(q) != 1) return RAD_E_STRIDE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows) return RAD_E_SHAPE;
    const int64_t g = as ? p_int(a, "group", 0) : 0;
    const int64_t mpad = (rows + 15) / 16 * 16;
    if (as && (g <= 0 || n % g || numel(as) < n / g * mpad)) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        int8_t* qr = (int8_t*)q->data + rowoff(q, r, n);
        float amax = 0.0f;
        for (int64_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(ldt(x, xb + i * xs)));
        const float scale = std::max(amax, 1e-8f) / 127.0f;
        const float inv = 1.0f / scale;
        for (int64_t i = 0; i < n; ++i)
            st_dt(RAD_I8, qr, i8_step_pos(i), ldt(x, xb + i * xs) * inv);
        stt(s, rowoff(s, r, 1), scale);
        for (int64_t gi = 0; as && gi < n / g; ++gi) {
            int64_t sum = 0;
            for (int64_t i = gi * g; i < (gi + 1) * g; ++i) sum += qr[i8_step_pos(i)];
            stt(as, gi * mpad + r, (float)sum);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ quant_act_i8g */
/* quant_act_fp8 at int8 codes, libr4d's arithmetic: per `group` contiguous elements a scale
 * amax / 127 (1 for an all-zero group) and codes rounded to nearest even, row-major. What the
 * i8a8 GEMMs read. */
int ref_quant_act_i8g(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;

    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), qb = rowoff(q, r, n);
        const int64_t xs = laststride(x), qs = laststride(q);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
            const float sc = q8_group([&](int64_t i) { return ldt(x, xb + (k0 + i) * xs); },
                                      k1 - k0, 127.0f, q, qb + k0 * qs, qs);
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
        }
    }
    return RAD_OK;
}

/* The rotated forms below are symmetric int8 with one scale per row: q = round(x * 127 / amax),
 * scale = amax / 127, the store rounding to nearest and saturating (ref_common.h). An all-zero row
 * gets scale 0 and codes 0. */

/* ------------------------------------------------------------------ had_quant_act_i8 */
/* The rotated int8 row of the had_*_i8 ops: `v(i)` for i < n rotated in blocks of `g`, its codes
 * stored at q[qb + i * qs] against one scale for the whole row (the amax is over the whole rotated
 * row). The rotation is computed twice, for the amax and then for the codes, rather than held.
 * Returns the row's scale, amax / 127 / sqrt(g). */
template <class V>
static float had_i8_row(const V& v, int64_t n, int64_t g, const RadTensor* q, int64_t qb,
                        int64_t qs) {
    float buf[REF_HAD_MAX];
    float amax = 0.0f;
    for (int64_t b = 0; b < n; b += g) {
        for (int64_t i = 0; i < g; ++i) buf[i] = v(b + i);
        fwht(buf, g);
        for (int64_t i = 0; i < g; ++i) { float m = std::fabs(buf[i]); if (m > amax) amax = m; }
    }
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (int64_t b = 0; b < n; b += g) {
        for (int64_t i = 0; i < g; ++i) buf[i] = v(b + i);
        fwht(buf, g);
        for (int64_t i = 0; i < g; ++i) st_dt(q->dtype, q->data, qb + (b + i) * qs, buf[i] * inv);
    }
    return amax / 127.0f * (1.0f / std::sqrt((float)g));
}

int ref_had_quant_act_i8(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > REF_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n * 8)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        stt(s, r * laststride(s), had_i8_row([&](int64_t i) { return ldt(x, xb + i * xs); }, n, g,
                                             q, rowoff(q, r, n), laststride(q)));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rmsnorm_had_quant_i8 */
/* Fused rmsnorm + hadamard + quant_act_i8, what a w4a8 layer uses in place of a plain rmsnorm
 * (spec §2.3). A plugin asks for this first and falls back to the three unfused ops.
 *
 * With `residual` it is the fused_add_rms_norm form: the sum is written back to `residual` (INOUT;
 * the next layer's shortcut reads it) and the norm is over the sum. The variance comes from the f32
 * sum, but the rotation reads the sum back after narrowing to the residual's dtype, as the unfused
 * pair does.
 *
 * `out_bf16` is the post-norm, pre-rotation tensor for a bf16 consumer; the rotation belongs only
 * to the int8 path. */
int ref_rmsnorm_had_quant_i8(const RadArgs* a, RadStream) {
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[2], *q = &a->t[3], *s = &a->t[4];
    const RadTensor* res = t_in(a, 1);
    const RadTensor* ob  = t_in(a, 5);
    if (!x->data || !w->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const int64_t g = p_int(a, "group", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > REF_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows || numel(w) < n) return RAD_E_SHAPE;
    if (res && numel(res) < rows * n) return RAD_E_SHAPE;
    if (ob && numel(ob) < rows * n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n * 8)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xs = laststride(x), ws = laststride(w);

        /* The variance is taken from the f32 sum as it is written back; everything after reads
         * the narrowed sum. */
        const int64_t rb = res ? rowoff(res, r, n) : 0;
        const int64_t rst = res ? laststride(res) : 1;
        auto src = [&](int64_t i) {
            return res ? ldt(res, rb + i * rst) : ldt(x, xb + i * xs);
        };

        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            const float v = res ? ldt(x, xb + i * xs) + ldt(res, rb + i * rst) : src(i);
            if (res) stt(res, rb + i * rst, v);
            ss += v * v;
        }
        const float sc = 1.0f / std::sqrt(ss / (float)n + eps);

        if (ob) {
            const int64_t obb = rowoff(ob, r, n), obs = laststride(ob);
            for (int64_t i = 0; i < n; ++i)
                stt(ob, obb + i * obs, src(i) * sc * (ldt(w, i * ws) + wadd));
        }

        auto normed = [&](int64_t i) { return src(i) * sc * (ldt(w, i * ws) + wadd); };
        stt(s, r * laststride(s), had_i8_row(normed, n, g, q, rowoff(q, r, n), laststride(q)));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gated_had_quant_i8 */
/* An elementwise gated product, then the rotation and the quantiser. Two of a w4a8 model's four
 * rotated sites have this shape: silu(a)*b is down_proj's input, sigmoid(a)*b is o_proj's on a
 * gated-attention layer (Qwen3_5Attention's attn_output * sigmoid(gate)). `act` picks, default silu
 * (docs/OPS.md's `silu_mul`); libr4d's integer `mode` is accepted where `act` is absent.
 *
 * With `b` absent, `a` is the packed [M, 2n] form (gate first half, up second), vLLM's layout and
 * what a fused gate_up projection produces. */
int ref_gated_had_quant_i8(const RadArgs* a, RadStream) {
    if (a->n_t < 4) return RAD_E_INVAL;
    const RadTensor *ga = &a->t[0], *q = &a->t[2], *s = &a->t[3];
    const RadTensor* up = t_in(a, 1);
    if (!ga->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, (int)p_int(a, "mode", 0));
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (act != 0 && act != 1) return RAD_E_INVAL;
    if (g > REF_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(q) / n;
    if (rows <= 0 || numel(s) < rows) return RAD_E_SHAPE;
    if (numel(ga) < rows * (up ? n : 2 * n)) return RAD_E_SHAPE;
    if (up && numel(up) < rows * n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n * 8)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t gb = rowoff(ga, r, up ? n : 2 * n), gs = laststride(ga);
        const int64_t ub = up ? rowoff(up, r, n) : 0;
        const int64_t us = up ? laststride(up) : 0;
        auto gated = [&](int64_t i) {
            const float gate = ldt(ga, gb + i * gs);
            const float x    = up ? ldt(up, ub + i * us) : ldt(ga, gb + (n + i) * gs);
            return (act == 1 ? act_sigmoid(gate) : act_silu(gate)) * x;
        };
        stt(s, r * laststride(s), had_i8_row(gated, n, g, q, rowoff(q, r, n), laststride(q)));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ dequant */
/* q * scale, the scale stream's extent giving the group count. The inverse of quant_act_i8g only,
 * not of the rotated forms: undoing a rotation is a separate Hadamard the caller must request. */
int ref_dequant(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *s = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(y) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;
    const int64_t ngrp = numel(s) / rows > 0 ? numel(s) / rows : 1;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0) g = (n + ngrp - 1) / ngrp;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t qb = rowoff(q, r, n), yb = rowoff(y, r, n), sb = rowoff(s, r, ngrp);
        const int64_t qs = laststride(q), ys = laststride(y), ss = laststride(s);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t gi = g > 0 ? i / g : 0;
            const float sc = ldt(s, sb + (gi < ngrp ? gi : ngrp - 1) * ss);
            stt(y, yb + i * ys, ldt(q, qb + i * qs) * sc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ dequant_w4
 * The packed 4-bit weight back to a plain row-major bf16 [N, K]. The fragment order is
 * [t][kb][lh][r][s] with eight nibbles along i, where t = n/16, r = n%16, kb = k/64,
 * s = (k%64)/16, and the k inside the step is 4*lh + _KOFF[i] with _KOFF = {0,8,1,9,2,10,3,11}.
 * The scale dword at [t][g][r] holds the f16 scale low and -(1024+zero) high. */
static const int kKOFF[8] = {0, 8, 1, 9, 2, 10, 3, 11};
int ref_dequant_w4(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *wq = &a->t[0], *ws = &a->t[1], *y = &a->t[2];
    const int64_t N = y->shape[0], K = y->shape[1];
    const int64_t group = p_int(a, "group", 128);
    const int64_t twos  = p_int(a, "twos", 1);
    if (N <= 0 || K <= 0 || N % 16 || K % 64 || group < 1 || K % group) return RAD_E_SHAPE;

    const uint32_t* Wq = (const uint32_t*)wq->data;
    const uint32_t* Ws = (const uint32_t*)ws->data;
    const int64_t nkb = K / 64, ng = K / group;
    const int64_t ys = y->stride[1];

    for (int64_t idx = 0; idx < N * K / 8; ++idx) {
        const int s  = (int)(idx & 3);
        const int r  = (int)((idx >> 2) & 15);
        const int lh = (int)((idx >> 6) & 1);
        const int kb = (int)((idx >> 7) % nkb);
        const int t  = (int)((idx >> 7) / nkb);
        const int row_n = t * 16 + r;
        const uint32_t w = Wq[idx];

        const int kbase = kb * 64 + s * 16 + 4 * lh;
        const uint32_t sz = Ws[(size_t)(t * ng + kbase / group) * 16 + r];
        const float sc   = rad_f16_to_f32((uint16_t)(sz & 0xFFFFu));
        const float nz   = rad_f16_to_f32((uint16_t)(sz >> 16));
        const float zero = -nz - 1024.f;

        for (int i = 0; i < 8; i++) {
            const int n = (int)((w >> (4 * i)) & 0xF);
            const float q = twos ? (float)((n ^ 8) - 8) : (float)n;
            const int k = kbase + kKOFF[i];
            stt(y, (size_t)row_n * K + k, (q - (twos ? 0.f : zero)) * sc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ quant_act_fp8
 *
 * The oracle for the block-scaled FP8 family, the activation format the production checkpoints
 * use (spec §17). The arithmetic is r4d_quant_act_fp8.hip's, agreeing on three choices:
 *
 *   448        E4M3's largest finite value; the scale is `amax / 448`, a dequant multiplier like
 *              the checkpoint's `weight_scale_inv`, so the GEMM multiplies the two scales.
 *   scale 1    for an all-zero group (amax / 448 would have an infinite reciprocal): the GEMM
 *              multiplies the two block scales, and a zero would erase the weight's.
 *   the clamp  `amax * (1/scale)` is 448 exactly but can round an ulp above once the reciprocal is
 *              rounded, which would encode as NaN.
 *
 * `M` and `n` are the row count and width with rows along dim 0, so a rank-3 activation
 * ([tokens, heads, width], a GDN block's out-projection input) is one row per token: `n` is the
 * consuming GEMM's K.
 */
int ref_quant_act_fp8(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;

    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), qb = rowoff(q, r, n);
        const int64_t xs = laststride(x), qs = laststride(q);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
            const float sc = q8_group([&](int64_t i) { return ldt(x, xb + (k0 + i) * xs); },
                                       k1 - k0, 448.0f, q, qb + k0 * qs, qs);
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- quant_cols_fp8 */
/* E4M3 codes with ONE scale per column over all M rows: scale[c] = amax_r |x[r, c]| times 1/448
 * in f32 (1 for an all-zero column), y = clamp(x * (1/scale), +-448) rounded to nearest even. The
 * value side of an attention's fp8 P.V: a column is one dimension of one head, so the scale
 * multiplies out of the sum over keys. */
int ref_quant_cols_fp8(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(y) < rows * n || numel(s) < n) return RAD_E_SHAPE;
    const int64_t xs = laststride(x), ys = laststride(y), ss = laststride(s);
    REF_PAR_FOR(rows * n)
    for (int64_t c = 0; c < n; ++c) {
        float amax = 0.0f;
        for (int64_t r = 0; r < rows; ++r) {
            const float m = std::fabs(ldt(x, rowoff(x, r, n) + c * xs));
            if (m > amax) amax = m;
        }
        const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
        const float inv = 1.0f / sc;
        for (int64_t r = 0; r < rows; ++r) {
            float v = ldt(x, rowoff(x, r, n) + c * xs) * inv;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            stt(y, rowoff(y, r, n) + c * ys, v);
        }
        stt(s, c * ss, sc);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- had_quant_act_fp8 */
/* The same quantiser with a `group`-wide Hadamard before the absmax, which lowers the maximum the
 * grid is scaled by.
 *
 * Unlike ref_had_quant_act_i8, the sqrt(g) is not undone here. H is symmetric, so
 * <Hw, Ha> = g <w, a>, and g must be divided out once between the two sides. This op gives the
 * whole factor to the weight, where dividing a bf16 scale by a power of two is exact. Splitting it
 * would store bf16(amax/7/sqrt(g)) instead of bf16(amax/7)/g against codes chosen for amax/7, a
 * ~0.2% error per group (a fifth of the int4 grid's own error).
 *
 * The output is not interchangeable with quant_act_fp8's: only a weight plane stored as w4a8h
 * carries the matching rotation and /g.
 *
 * had_quant_act_i8g is the same at int8 codes (`top` 127), for the i8a8h weight, whose scale
 * carries the /g the same way. Its rotation is at most `rot_cap` (128) wide: a wider group (IQ8R's
 * 256) is whole 128-wide rotations under one scale, the weight's scale carrying /128. */
static int had_quant_act_q8(const RadArgs* a, float top, int64_t rot_cap = 0) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    /* n % g != 0 is refused: no weight block was rotated across the row's end. */
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > REF_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rw = rot_cap > 0 && g > rot_cap ? rot_cap : g;
    const int64_t rows = numel(x) / n;
    const int64_t ngrp = n / g;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n * 8)
    for (int64_t r = 0; r < rows; ++r) {
        float buf[REF_HAD_MAX];
        const int64_t xb = rowoff(x, r, n), qb = rowoff(q, r, n);
        const int64_t xs = laststride(x), qs = laststride(q);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g;
            for (int64_t i = 0; i < g; ++i) buf[i] = ldt(x, xb + (k0 + i) * xs);
            for (int64_t b = 0; b < g; b += rw) fwht(buf + b, rw);
            const float sc = q8_group([&](int64_t i) { return buf[i]; }, g, top, q,
                                      qb + k0 * qs, qs);
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
        }
    }
    return RAD_OK;
}

int ref_had_quant_act_i8g(const RadArgs* a, RadStream) {
    return had_quant_act_q8(a, 127.0f, 128);
}


/* ---------------------------------------------------------------- the Gram accumulator
 *
 * h[k1][k2] += sum over rows of x[m][k1] * x[m][k2], x being the dequantised E4M3 input (or a bf16
 * input as is when no scale is passed): the second moment GPTQ needs, over exactly the operand the
 * expert GEMM contracts. Oracle for libr4d's device kernel.
 *
 * `h` is INOUT and accumulates: the caller zeroes it once and each call adds its block.
 *
 * Rows are dequantised once into a scratch plane, keeping the dtype dispatch out of the
 * O(n^2 * rows) loop. Accumulation is deliberately f64 here (f32 on the device): this oracle is
 * meant to be tighter than the kernel, not bit-identical. */
int ref_gram_accum(const RadArgs* a, RadStream) {
    /* `a_scale` is optional; absent, `a` is read as is (the bf16 form). */
    if (a->n_t < 3) return RAD_E_INVAL;
    const RadTensor *x = t_in(a, 0), *s = t_in(a, 1), *h = t_in(a, 2);
    if (!x || !h) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    if (n <= 0 || (s && g <= 0)) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    const int64_t ngrp = s ? (n + g - 1) / g : 0;
    if (rows <= 0 || (s && numel(s) < rows * ngrp)) return RAD_E_SHAPE;
    if (h->rank != 2 || h->shape[0] != n || h->shape[1] != n) return RAD_E_SHAPE;

    std::vector<float> xs((size_t)rows * (size_t)n);
    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xst = laststride(x);
        const int64_t sb = s ? rowoff(s, r, ngrp) : 0, sst = s ? laststride(s) : 0;
        for (int64_t k = 0; k < n; ++k)
            xs[(size_t)r * (size_t)n + (size_t)k] =
                ldt(x, xb + k * xst) * (s ? ldt(s, sb + (k / g) * sst) : 1.0f);
    }

    /* `upper`: only k2 >= k1. libr4d adds whole 64-wide tiles, so it also writes some entries just
     * below the diagonal; neither form's entries there are part of the contract. */
    const bool upper = p_int(a, "upper", 0) != 0;
    const int64_t hst = laststride(h);
    REF_PAR_FOR(n * n * rows)
    for (int64_t k1 = 0; k1 < n; ++k1) {
        const int64_t hb = rowoff(h, k1, n);
        for (int64_t k2 = upper ? k1 : 0; k2 < n; ++k2) {
            double acc = 0.0;
            for (int64_t r = 0; r < rows; ++r)
                acc += (double)xs[(size_t)r * (size_t)n + (size_t)k1] *
                       (double)xs[(size_t)r * (size_t)n + (size_t)k2];
            const int64_t off = hb + k2 * hst;
            stt(h, off, (float)((double)ldt(h, off) + acc));
        }
    }
    return RAD_OK;
}

/* ======================================================================== the fp8 fusions
 *
 * libr4d fuses the fp8 activation quantiser into whichever kernel already has the row in
 * registers and declares the fused form as its own op. These host kernels for those names let a
 * host-backend run start on an fp8 model; that run is the only check of the graph itself, since the
 * per-op oracle reads the same graph on both sides.
 *
 * Each is the unfused arithmetic in one pass. They need not be bit-identical to libr4d's kernels;
 * the oracle compares within a tolerance. They do follow the bf16 round-trip, which changes the
 * function: the unfused pair writes a bf16 intermediate and quantises those bytes, so codes are
 * taken over the narrowed value and `out_bf16` holds exactly the value the codes came from.
 */

/* One group-scaled fp8 row from the values in `v`; constants and scale layout as quant_act_fp8.
 * An int8 `q` (an int8 linear's input, ActFP8::q8_fed) takes quant_act_i8g's codes instead, scale
 * amax / 127, as libr4d's fused quantisers write them. */
static inline void fp8_row(const float* v, int64_t n, int64_t g, const RadTensor* q,
                           const RadTensor* s, int64_t r) {
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t qb = rowoff(q, r, n), qs = laststride(q);
    const float top = q->dtype == RAD_I8 ? 127.0f : 448.0f;
    for (int64_t gi = 0; gi < ngrp; ++gi) {
        const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
        const float sc = q8_group([&](int64_t i) { return v[k0 + i]; }, k1 - k0, top, q,
                                   qb + k0 * qs, qs);
        stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
    }
}

/* ------------------------------------------------------------- rmsnorm_quant_fp8 */
/* `x`, `residual`:inout?, `w`w -> `q`, `scale`, `out_bf16`?
 *
 * With `residual` it is the fused_add_rms_norm form, and the sum is stored before the variance
 * reads it, so the norm is over the bf16-narrowed sum (as libr4d's kernel and the unfused pair do).
 * Unlike ref_rmsnorm_add, the variance also uses the narrowed sum. `wadd` is added to each gain in
 * f32, as in ref_rmsnorm.
 */
int ref_rmsnorm_quant_fp8(const RadArgs* a, RadStream) {
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[2], *q = &a->t[3], *s = &a->t[4];
    const RadTensor* res = t_in(a, 1);
    const RadTensor* ob  = t_in(a, 5);
    if (!x->data || !w->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n    = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float   eps  = p_f32(a, "eps", 1e-6f);
    const float   wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0) return RAD_E_SHAPE;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows * ngrp || numel(w) < n)
        return RAD_E_SHAPE;
    if (res && numel(res) < rows * n) return RAD_E_SHAPE;
    if (ob  && numel(ob)  < rows * n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        std::vector<float> v((size_t)n);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x), ws = laststride(w);
        const int64_t rb = res ? rowoff(res, r, n) : 0;
        const int64_t rst = res ? laststride(res) : 1;
        if (res)
            for (int64_t i = 0; i < n; ++i)
                stt(res, rb + i * rst, ldt(x, xb + i * xs) + ldt(res, rb + i * rst));
        auto src = [&](int64_t i) { return res ? ldt(res, rb + i * rst) : ldt(x, xb + i * xs); };

        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) { const float t = src(i); ss += t * t; }
        const float rs = 1.0f / std::sqrt(ss / (float)n + eps);

        /* Narrow to bf16, write `out_bf16` if asked, then quantise those values. */
        const int64_t obb = ob ? rowoff(ob, r, n) : 0;
        const int64_t obs = ob ? laststride(ob) : 1;
        for (int64_t i = 0; i < n; ++i) {
            const uint16_t h = f32_to_bf16(src(i) * rs * (ldt(w, i * ws) + wadd));
            if (ob) stt(ob, obb + i * obs, bf16_to_f32(h));
            v[(size_t)i] = bf16_to_f32(h);
        }
        fp8_row(v.data(), n, g, q, s, r);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- gated_quant_fp8 */
/* `gate_up` -> `q`, `scale`, `out_bf16`?
 *
 * Packed form only: `gate_up` is [M, 2n], gate first. `act` picks silu(gate)*up (down_proj's input)
 * or sigmoid(gate)*up (a gated-attention o_proj's), as in ref_gated_had_quant_i8. */
int ref_gated_quant_fp8(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *gu = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const RadTensor* ob = t_in(a, 3);
    const int64_t n = p_int(a, "n", 0);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, (int)p_int(a, "mode", 0));
    if (n <= 0) return RAD_E_SHAPE;
    if (act != 0 && act != 1) return RAD_E_INVAL;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t rows = numel(gu) / (2 * n);
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows * ngrp) return RAD_E_SHAPE;
    if (ob && numel(ob) < rows * n) return RAD_E_SHAPE;

    REF_PAR_FOR(rows * n)
    for (int64_t r = 0; r < rows; ++r) {
        std::vector<float> v((size_t)n);
        const int64_t gb = rowoff(gu, r, 2 * n), gs = laststride(gu);
        const int64_t obb = ob ? rowoff(ob, r, n) : 0;
        const int64_t obs = ob ? laststride(ob) : 1;
        for (int64_t i = 0; i < n; ++i) {
            const float gt = ldt(gu, gb + i * gs);
            const float up = ldt(gu, gb + (n + i) * gs);
            const float sg = act_sigmoid(gt);
            const uint16_t h = f32_to_bf16((act == 0 ? gt * sg : sg) * up);
            if (ob) stt(ob, obb + i * obs, bf16_to_f32(h));
            v[(size_t)i] = bf16_to_f32(h);
        }
        fp8_row(v.data(), n, g, q, s, r);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- gemm_nt_q_gated */
/* `a`, `a_scale`, `b`w, `b_scale`w -> `q`, `scale`, `out_bf16`?
 *
 * gemm_nt_q over an [N, K] gate_up plane (N the weight's rows, twice the fused width, gate half
 * first) with gated_quant_fp8 in the epilogue, so the [M, N] bf16 intermediate is never written.
 * `act` is required here, unlike on the quantiser (docs/OPS.md): an implementation may compile one
 * gate, and an optional parameter would let it resolve for the other.
 *
 * Byte-identical to gemm_nt_q followed by gated_quant_fp8 (ref_test checks it): the product comes
 * from `quant_dot` as gemm_nt_q's does, and both halves are narrowed to bf16 before the gate, as the
 * unfused pair writes and re-reads a bf16 [M, N] tensor.
 *
 * Comparing against a device kernel on the engine's own operands needs the layout inverted first:
 * libr4d stores fp8a8 weights in WMMA fragment order (r4d.fp8.w.frag64) while libref reads [N, K]
 * row-major, and the two readings are uncorrelated (relative error around 1.35). The per-op oracle
 * applies the row's RadUnrelayoutFn before comparing and says so when a row has none.
 *
 * The product is formed one row at a time; [M, N] at a prefill M would be hundreds of MB. */
int ref_gemm_nt_q_gated(const RadArgs* a, RadStream) {
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[2], *q = &a->t[4], *s = &a->t[5];
    const RadTensor* as_ = t_in(a, 1);
    const RadTensor* bs_ = t_in(a, 3);
    const RadTensor* ob  = t_in(a, 6);
    if (!A->data || !B->data || !q->data || !s->data) return RAD_E_INVAL;

    const int64_t K = p_int(a, "K", B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, "N", B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0 || (N & 1)) return RAD_E_SHAPE;
    const int64_t n = N / 2;
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, -1);
    if (act != 0 && act != 1) return RAD_E_INVAL;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t M = numel(q) / n;
    if (M <= 0 || numel(A) < M * K || numel(B) < N * K) return RAD_E_SHAPE;
    if (numel(s) < M * ngrp) return RAD_E_SHAPE;
    if (ob && numel(ob) < M * n) return RAD_E_SHAPE;

    const bool mx = B->dtype == RAD_FP4E2M1;
    const Scales SA = make_scales(as_, M, K, false);
    const Scales SB = make_scales(bs_, N, K, mx);
    const Mat ma = as_mat(A, K), mb = as_mat(B, K);

    REF_PAR_FOR(M * N * K)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> row((size_t)N), v((size_t)n);
        for (int64_t j = 0; j < N; ++j) row[(size_t)j] = quant_dot(ma, mb, SA, SB, m, j, K);
        const int64_t obb = ob ? rowoff(ob, m, n) : 0;
        const int64_t obs = ob ? laststride(ob) : 1;
        /* Both halves narrowed to bf16 before the gate (see above). */
        for (int64_t i = 0; i < n; ++i) {
            const float gt = round_dt(RAD_BF16, row[(size_t)i]);
            const float up = round_dt(RAD_BF16, row[(size_t)(n + i)]);
            const float sg = act_sigmoid(gt);
            const uint16_t h = f32_to_bf16((act == 0 ? gt * sg : sg) * up);
            if (ob) stt(ob, obb + i * obs, bf16_to_f32(h));
            v[(size_t)i] = bf16_to_f32(h);
        }
        fp8_row(v.data(), n, g, q, s, m);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- dflash_select */
/* `cand`, `unary`, `hp`?, `anchor`, `pred`w, `succ`w -> `tokens`
 *
 * DFlash2's greedy walk over the candidate lattice. A block-diffusion drafter proposes every
 * position of a block in one backbone pass, so it cannot condition position l+1 on its choice at
 * l; this restores that dependence.
 *
 *   score[l, c] = unary[l, c] + sum_r pred[pid(l)][r] * hp[l][r] * succ[cand[l, c]][r]
 *   pid(0) = anchor[s * anchor_stride]        the token the target just committed
 *   pid(l) = cand[l - 1, chosen(l - 1)]       the walk's own previous pick
 *
 * `hp` absent is a plane of ones, reducing the trilinear form to a low-rank bigram (DSpark's Markov
 * head).
 *
 * The walk is serial in l, so it is an op rather than a GEMM plus argmax: scoring the full
 * [steps, K, K] edge tensor computes K times the edges ever read, since each predecessor is known
 * only after the previous step. Parallel over sequences. */
int ref_dflash_select(const RadArgs* a, RadStream) {
    if (a->n_t < 7) return RAD_E_INVAL;
    const RadTensor *cand = &a->t[0], *un = &a->t[1], *anc = &a->t[3];
    const RadTensor *pred = &a->t[4], *succ = &a->t[5], *out = &a->t[6];
    const RadTensor* hp = t_in(a, 2);
    if (!cand->data || !un->data || !anc->data || !pred->data || !succ->data || !out->data)
        return RAD_E_INVAL;

    const int64_t steps = p_int(a, "steps", 0);
    const int64_t K     = p_int(a, "top_k", 0);
    /* Rows come from `cand` [M * steps, K]; `M` is the band (see band_rows in ref_common.h). */
    const int64_t M     = steps > 0 && K > 0
                        ? band_rows(numel(cand) / (steps * K), p_int(a, "M", 0)) : 0;
    const int64_t R     = p_int(a, "rank", pred->rank >= 2 ? pred->shape[pred->rank - 1] : 0);
    const int64_t nv    = p_int(a, "n_vocab", pred->rank >= 2 ? pred->shape[pred->rank - 2] : 0);
    const int64_t astr  = p_int(a, "anchor_stride", 1);
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || nv <= 0 || astr <= 0) return RAD_E_SHAPE;
    if (numel(un) < M * steps * K) return RAD_E_SHAPE;
    if (numel(anc) < (M - 1) * astr + 1 || numel(out) < M * steps) return RAD_E_SHAPE;
    if (numel(pred) < nv * R || numel(succ) < nv * R) return RAD_E_SHAPE;
    if (hp && numel(hp) < M * steps * R) return RAD_E_SHAPE;

    const int64_t cs = laststride(cand), us = laststride(un), os = laststride(out);
    const int64_t ps = laststride(pred), sxs = laststride(succ);
    const int64_t hs = hp ? laststride(hp) : 1;

    REF_PAR_FOR(M * steps * K * R)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> ax((size_t)R);
        int64_t pid = ld_int(anc->dtype, anc->data, offlin(anc, m * astr));
        for (int64_t l = 0; l < steps; ++l) {
            const int64_t r     = m * steps + l;
            const int64_t cb    = rowoff(cand, r, K), ub = rowoff(un, r, K);
            const int64_t hb    = hp ? rowoff(hp, r, R) : 0;
            /* The candidate-independent half of the form, computed once and read K times. A
             * predecessor outside the vocabulary (a padding row's anchor) contributes nothing. */
            const bool ok = pid >= 0 && pid < nv;
            for (int64_t i = 0; i < R; ++i)
                ax[(size_t)i] = (ok ? ldt(pred, rowoff(pred, pid, R) + i * ps) : 0.0f) *
                                (hp ? ldt(hp, hb + i * hs) : 1.0f);

            int64_t best_c = 0;
            float   best_s = -INFINITY;
            for (int64_t c = 0; c < K; ++c) {
                const int64_t tok = ld_int(cand->dtype, cand->data, cb + c * cs);
                float sc = ldt(un, ub + c * us);
                if (tok >= 0 && tok < nv) {
                    const int64_t sb = rowoff(succ, tok, R);
                    for (int64_t i = 0; i < R; ++i) sc += ax[(size_t)i] * ldt(succ, sb + i * sxs);
                }
                if (sc > best_s) { best_s = sc; best_c = c; }
            }
            pid = ld_int(cand->dtype, cand->data, cb + best_c * cs);
            st_int(out->dtype, out->data, rowoff(out, m, steps) + l * os, pid);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------ dflash_markov_greedy */
/* `logits`, `anchor`, `pred`w, `succ`w -> tile-local winners and `tokens`.
 *
 * The pinned DSpark reference adds the full-vocabulary bilinear bigram bias to each draft-head
 * score, then takes a vocabulary-wide greedy argmax at every position. This is deliberately not
 * `dflash_select`: no top-K candidate restriction is applied, and the result at one position is
 * the predecessor for the next. Scratch is a 64-token tile decomposition used by the device row;
 * the reference writes the same tile winners before selecting the global winner. */
static int ref_markov_greedy(const RadArgs* a, uint32_t logits_dtype) {
    if (a->n_t < 7) return RAD_E_INVAL;
    const RadTensor *logits = &a->t[0], *anchor = &a->t[1], *pred = &a->t[2];
    const RadTensor *succ = &a->t[3], *tile_scores = &a->t[4], *tile_tokens = &a->t[5];
    const RadTensor *out = &a->t[6];
    if (!logits->data || !anchor->data || !pred->data || !succ->data ||
        !tile_scores->data || !tile_tokens->data || !out->data) return RAD_E_INVAL;
    if (logits->dtype != logits_dtype || pred->dtype != RAD_BF16 || succ->dtype != RAD_BF16 ||
        tile_scores->dtype != RAD_F32 || tile_tokens->dtype != RAD_I32 || out->dtype != RAD_I32)
        return RAD_E_DTYPE;
    /* The device requires I32 anchors and reports invalid anchor tensors as shape errors. */
    if (anchor->dtype != RAD_I32) return RAD_E_SHAPE;

    const int64_t steps = p_int(a, "steps", 0);
    const int64_t R = p_int(a, "rank", 0);
    const int64_t V = p_int(a, "n_vocab", 0);
    const int64_t astr = p_int(a, "anchor_stride", 1);
    const int64_t M = steps > 0 && V > 0
                    ? band_rows(numel(logits) / (steps * V), p_int(a, "M", 0)) : 0;
    if (M <= 0 || steps <= 0 || R <= 0 || V <= 0 || astr <= 0) return RAD_E_SHAPE;
    if (numel(logits) < M * steps * V || numel(anchor) < (M - 1) * astr + 1 ||
        numel(pred) < V * R || numel(succ) < V * R || numel(out) < M * steps)
        return RAD_E_SHAPE;
    const int64_t n_tiles = (V + 63) / 64;
    if (numel(tile_scores) < M * n_tiles || numel(tile_tokens) < M * n_tiles)
        return RAD_E_SHAPE;

    const int64_t ls = laststride(logits), ps = laststride(pred), ss = laststride(succ);
    const int64_t os = laststride(out), tss = laststride(tile_scores), tis = laststride(tile_tokens);
    REF_PAR_FOR(M * steps * V * R)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> previous((size_t)R);
        int64_t pid = ld_int(anchor->dtype, anchor->data, offlin(anchor, m * astr));
        for (int64_t l = 0; l < steps; ++l) {
            const int64_t row = m * steps + l;
            const bool pid_ok = pid >= 0 && pid < V;
            const int64_t pb = pid_ok ? rowoff(pred, pid, R) : 0;
            for (int64_t r = 0; r < R; ++r)
                previous[(size_t)r] = pid_ok ? ldt(pred, pb + r * ps) : 0.0f;

            float global_best = -INFINITY;
            int64_t global_token = 0;
            for (int64_t tile = 0; tile < n_tiles; ++tile) {
                const int64_t first = tile * 64;
                const int64_t end = std::min<int64_t>(first + 64, V);
                float best = -INFINITY;
                int64_t best_token = first;
                const int64_t lb = rowoff(logits, row, V);
                for (int64_t token = first; token < end; ++token) {
                    const float base = ldt(logits, lb + token * ls);
                    float score = logits_dtype == RAD_F32 ? 0.0f : base;
                    const int64_t sb = rowoff(succ, token, R);
                    for (int64_t r = 0; r < R; ++r)
                        score += previous[(size_t)r] * ldt(succ, sb + r * ss);
                    if (logits_dtype == RAD_F32) score += base;
                    if (score > best || (score == best && token < best_token)) {
                        best = score;
                        best_token = token;
                    }
                }
                stt(tile_scores, rowoff(tile_scores, m, n_tiles) + tile * tss, best);
                st_int(tile_tokens->dtype, tile_tokens->data,
                       rowoff(tile_tokens, m, n_tiles) + tile * tis, best_token);
                if (best > global_best || (best == global_best && best_token < global_token)) {
                    global_best = best;
                    global_token = best_token;
                }
            }
            st_int(out->dtype, out->data, rowoff(out, m, steps) + l * os, global_token);
            pid = global_token;
        }
    }
    return RAD_OK;
}

int ref_dflash_markov_greedy(const RadArgs* a, RadStream) {
    return ref_markov_greedy(a, RAD_BF16);
}

int ref_dflash_markov_greedy_f32(const RadArgs* a, RadStream) {
    return ref_markov_greedy(a, RAD_F32);
}

/* ------------------------------------------------------------- dflash_tree */
/* `cand`, `unary`, `hp`?, `anchor`, `pred`w, `succ`w -> `tokens`, `parents`
 *
 * dflash_select's lattice and edge scores, grown into a tree of `nodes` nodes instead of walked
 * into one path (rad_drafter_tree). Node 0 is the anchor, at lattice position -1 with log
 * probability 0. A node at position l with token t has a child for every valid candidate c at
 * position l + 1, scored as dflash_select scores it after t, with log probability its parent's
 * plus log_softmax(score / temp)[c] over the valid candidates there. Best-first: each round
 * selects the not-yet-selected child of any selected node with the largest log probability (ties
 * to the one pushed first: lower parent, then lower c), until `nodes` are selected. Selected node
 * j (1-based, in selection order, so a parent always precedes its child) is written as
 * tokens[j-1] and parents[j-1], the parent's node index (0 the anchor). A frontier that runs dry
 * (every candidate invalid) pads with token 0 under the last node, which the verify rejects.
 *
 * In f32: score as dflash_select; x = score / temp; m = max x; lse = m + log(sum exp(x - m)) over
 * the valid candidates, summed in c order; child = parent + (x - lse). */
int ref_dflash_tree(const RadArgs* a, RadStream) {
    if (a->n_t < 8) return RAD_E_INVAL;
    const RadTensor *cand = &a->t[0], *un = &a->t[1], *anc = &a->t[3];
    const RadTensor *pred = &a->t[4], *succ = &a->t[5], *out = &a->t[6], *par = &a->t[7];
    const RadTensor* hp = t_in(a, 2);
    if (!cand->data || !un->data || !anc->data || !pred->data || !succ->data || !out->data ||
        !par->data)
        return RAD_E_INVAL;

    const int64_t steps = p_int(a, "steps", 0);
    const int64_t K     = p_int(a, "top_k", 0);
    const int64_t M     = steps > 0 && K > 0
                        ? band_rows(numel(cand) / (steps * K), p_int(a, "M", 0)) : 0;
    const int64_t R     = p_int(a, "rank", pred->rank >= 2 ? pred->shape[pred->rank - 1] : 0);
    const int64_t nv    = p_int(a, "n_vocab", pred->rank >= 2 ? pred->shape[pred->rank - 2] : 0);
    const int64_t astr  = p_int(a, "anchor_stride", 1);
    const int64_t N     = p_int(a, "nodes", steps);
    const float   temp  = p_f32(a, "temp", 1.0f);
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || nv <= 0 || astr <= 0 || N <= 0 ||
        !(temp > 0.0f))
        return RAD_E_SHAPE;
    if (numel(un) < M * steps * K) return RAD_E_SHAPE;
    if (numel(anc) < (M - 1) * astr + 1 || numel(out) < M * N || numel(par) < M * N)
        return RAD_E_SHAPE;
    if (numel(pred) < nv * R || numel(succ) < nv * R) return RAD_E_SHAPE;
    if (hp && numel(hp) < M * steps * R) return RAD_E_SHAPE;

    const int64_t cs = laststride(cand), us = laststride(un), os = laststride(out);
    const int64_t pps = laststride(par);
    const int64_t ps = laststride(pred), sxs = laststride(succ);
    const int64_t hs = hp ? laststride(hp) : 1;

    REF_PAR_FOR(M * N * K * R)
    for (int64_t m = 0; m < M; ++m) {
        struct Node { int64_t l, tok, parent; float lp; };
        struct Kid  { float lp; int64_t parent, l, tok; bool taken; };
        std::vector<Node> sel;     /* selected, node 0 the anchor */
        std::vector<Kid>  front;   /* every child pushed so far, in push order */
        std::vector<float> ax((size_t)R), x((size_t)K);
        sel.push_back({ -1, ld_int(anc->dtype, anc->data, offlin(anc, m * astr)), -1, 0.0f });

        auto expand = [&](int64_t n) {
            const Node nd = sel[(size_t)n];
            const int64_t l = nd.l + 1;
            if (l >= steps) return;
            const int64_t r  = m * steps + l;
            const int64_t cb = rowoff(cand, r, K), ub = rowoff(un, r, K);
            const int64_t hb = hp ? rowoff(hp, r, R) : 0;
            const bool ok = nd.tok >= 0 && nd.tok < nv;
            for (int64_t i = 0; i < R; ++i)
                ax[(size_t)i] = (ok ? ldt(pred, rowoff(pred, nd.tok, R) + i * ps) : 0.0f) *
                                (hp ? ldt(hp, hb + i * hs) : 1.0f);
            float mx = -INFINITY;
            for (int64_t c = 0; c < K; ++c) {
                const int64_t tok = ld_int(cand->dtype, cand->data, cb + c * cs);
                float sc = ldt(un, ub + c * us);
                const bool valid = tok >= 0 && tok < nv;
                if (valid) {
                    const int64_t sb = rowoff(succ, tok, R);
                    for (int64_t i = 0; i < R; ++i) sc += ax[(size_t)i] * ldt(succ, sb + i * sxs);
                }
                x[(size_t)c] = valid ? sc / temp : -INFINITY;
                if (x[(size_t)c] > mx) mx = x[(size_t)c];
            }
            if (mx == -INFINITY) return;
            float sum = 0.0f;
            for (int64_t c = 0; c < K; ++c)
                if (x[(size_t)c] > -INFINITY) sum += std::exp(x[(size_t)c] - mx);
            const float lse = mx + std::log(sum);
            for (int64_t c = 0; c < K; ++c) {
                if (!(x[(size_t)c] > -INFINITY)) continue;
                front.push_back({ nd.lp + (x[(size_t)c] - lse), n, l,
                                  ld_int(cand->dtype, cand->data, cb + c * cs), false });
            }
        };

        expand(0);
        for (int64_t j = 1; j <= N; ++j) {
            int64_t best = -1;
            for (size_t f = 0; f < front.size(); ++f)
                if (!front[f].taken && (best < 0 || front[f].lp > front[(size_t)best].lp))
                    best = (int64_t)f;
            if (best < 0) {
                sel.push_back({ steps, 0, (int64_t)sel.size() - 1, -INFINITY });
            } else {
                Kid& k = front[(size_t)best];
                k.taken = true;
                sel.push_back({ k.l, k.tok, k.parent, k.lp });
                expand((int64_t)sel.size() - 1);
            }
            const Node& nd = sel.back();
            st_int(out->dtype, out->data, rowoff(out, m, N) + (j - 1) * os, nd.tok);
            st_int(par->dtype, par->data, rowoff(par, m, N) + (j - 1) * pps, nd.parent);
        }
    }
    return RAD_OK;
}
