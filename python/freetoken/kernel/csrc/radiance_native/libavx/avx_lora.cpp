/* avx_lora.cpp -- runtime LoRA's two halves on the host: lora_shrink and lora_expand.
 *
 * libref/ref_lora.cpp defines the ops; this is the same arithmetic over the same bytes with the dot
 * products vectorised, a tolerance question (avx_check). Which rows are touched is exact: a row
 * whose slot is outside [0, S) is a base row, its `t` row is never written and its `y` row is
 * neither read nor stored, so it stays bit for bit what the GEMM wrote.
 *
 * x(m, k) is gemm_nt_q's activation value: the code times its group's scale, read through the same
 * scale-stream rule as avx_gemm.cpp (copied below for the reason avx_moe.cpp gives for its copy).
 *
 * Threading: a row's dots never split, so its bytes do not depend on the team or on M. A prefill
 * goes wide over rows; a decode step has one row and R columns, so the row is decoded once on the
 * calling thread and the team goes wide over the columns. Either way each output element is one
 * dotf() over the same two f32 rows.
 */
#include "avx_common.h"
#include "avx_vec.h"

using namespace avx;

namespace {

struct Mat { const RadTensor* t; int64_t rs, cs; };

static inline Mat as_mat(const RadTensor* t, int64_t cols) {
    Mat m{ t, cols, 1 };
    if (t->rank < 2) return m;
    m.cs = t->stride[t->rank - 1];
    m.rs = row_stride(t, cols);
    return m;
}

/* gemm_nt_q's activation scale stream, read as libref/ref_gemm.h's make_scales reads it. */
struct Scales {
    const void* p;
    uint32_t dt;
    int64_t rs, cs, group, rblk;
    inline float at(int64_t row, int64_t k) const {
        const int64_t g = group > 0 ? k / group : 0;
        return rad_load_f32(p, dt, (rblk > 1 ? row / rblk : row) * rs + g * cs);
    }
};

static Scales make_scales(const RadTensor* s, int64_t rows, int64_t K) {
    Scales q{};
    q.rblk = 1;
    if (!s || !s->data || rows <= 0 || numel(s) <= 0) {
        q.p = nullptr; q.group = K; q.dt = RAD_F32; return q;
    }
    q.p = s->data;
    q.dt = s->dtype;
    int64_t groups = 0;
    if (s->rank >= 2) {
        const int64_t srows = s->shape[s->rank - 2];
        q.rblk = srows >= rows ? 1 : grid_block(rows, srows);
        groups = s->shape[s->rank - 1];
        q.rs = s->stride[s->rank - 2];
        q.cs = s->stride[s->rank - 1];
    } else {
        const int64_t nel = numel(s);
        if (nel >= rows) { groups = nel / rows; q.rs = groups; }
        else             { groups = 1; q.rblk = grid_block(rows, nel); q.rs = 1; }
        q.cs = 1;
    }
    q.group = grid_block(K, groups);
    return q;
}

/* A pool operand [S, rows, cols]; see libref/ref_lora.cpp. */
struct Pool3 { int64_t s0, s1, s2; };
static inline Pool3 pool3(const RadTensor* t, int64_t rows, int64_t cols) {
    if (t->rank == 3) return Pool3{ t->stride[0], t->stride[1], t->stride[2] };
    return Pool3{ rows * cols, cols, 1 };
}

static inline int64_t row_slot(const RadTensor* slot, int64_t m, int64_t S) {
    const int64_t s = ldi(slot, offlin(slot, m));
    return (s >= 0 && s < S) ? s : -1;
}

static inline bool float_act(uint32_t dt) {
    return dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F32;
}
static inline bool coded_act(uint32_t dt) {
    return dt == RAD_F8E4M3 || dt == RAD_F8E5M2 || dt == RAD_I8;
}

/* Row m of `a` as x(m, k) in f32: the codes widened, then each times its group's scale. */
static inline const float* x_row(const RadTensor* X, const Mat& mx, const Scales& SA, int64_t m,
                                 int64_t K, float* xb) {
    const float* xv = in_row(X, m * mx.rs, K, xb);
    if (!SA.p) return xv;
    for (int64_t k = 0; k < K; ++k) xb[k] = xv[k] * SA.at(m, k);
    return xb;
}

}  /* namespace */

/* operands: a, a_scale?, slot, lora_a, t */
AVX_KERNEL(lora_shrink) {
    if (a->n_t < 5 || !rad_arg_in(a, 0) || !rad_arg_in(a, 2) || !rad_arg_in(a, 3) ||
        !rad_arg_in(a, 4))
        return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *SL = &a->t[2], *LA = &a->t[3], *T = &a->t[4];
    const RadTensor* XS = rad_arg_in(a, 1);
    if (XS && XS->dtype == RAD_DT_INVALID) XS = nullptr;     /* absent, as libref */
    const int64_t K = p_int(a, "K", 0), R = p_int(a, "R", 0), S = p_int(a, "S", 0);
    if (K <= 0 || R <= 0 || S <= 0) return RAD_E_SHAPE;
    const int64_t M = band_rows(numel(T) / R, p_int(a, "M", 0));
    if (M <= 0) return RAD_E_SHAPE;                         /* zero rows is not a call */
    if (numel(X) < M * K || numel(SL) < M || numel(LA) < S * R * K) return RAD_E_SHAPE;
    if (!float_act(X->dtype) && !coded_act(X->dtype)) return RAD_E_DTYPE;
    if (coded_act(X->dtype) && !XS) return RAD_E_INVAL;
    if (T->dtype != RAD_F32) return RAD_E_DTYPE;
    const Scales SA = make_scales(XS, M, K);
    const int64_t gp = p_int(a, "group", 0);
    if (gp > 0 && SA.p && (K + gp - 1) / gp != (K + SA.group - 1) / SA.group) return RAD_E_SHAPE;

    const Mat mx = as_mat(X, K), mt = as_mat(T, R);
    const Pool3 pa = pool3(LA, R, K);
    float* const td = (float*)T->data;

    if (M >= 8) {
        AVX_PARALLEL_FOR_IF(M * R * K >= AVX_PAR_MIN)
        for (int64_t m = 0; m < M; ++m) {
            const int64_t s = row_slot(SL, m, S);
            if (s < 0) continue;                         /* a base row: nothing written */
            float* xb = scratch_f32((size_t)K);
            float* ab = scratch_f32_b((size_t)K);
            const float* xv = x_row(X, mx, SA, m, K, xb);
            for (int64_t j = 0; j < R; ++j) {
                const float* av = in_row(LA, s * pa.s0 + j * pa.s1, K, ab);
                td[m * mt.rs + j * mt.cs] = dotf(xv, av, K);
            }
        }
        return RAD_OK;
    }
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = row_slot(SL, m, S);
        if (s < 0) continue;
        /* The calling thread's first buffer holds the row; every team member's second one holds
         * its column of the pool, so nothing the team writes aliases the shared row. */
        const float* xv = x_row(X, mx, SA, m, K, scratch_f32((size_t)K));
        AVX_PARALLEL_FOR_IF(R * K >= AVX_PAR_MIN)
        for (int64_t j = 0; j < R; ++j) {
            const float* av = in_row(LA, s * pa.s0 + j * pa.s1, K, scratch_f32_b((size_t)K));
            td[m * mt.rs + j * mt.cs] = dotf(xv, av, K);
        }
    }
    return RAD_OK;
}

/* operands: t, slot, lora_b, scale, y (inout). `bf16_t`: lora_expand_mm, t rounded to bf16. */
static int expand_impl(const RadArgs* a, bool bf16_t) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *T = &a->t[0], *SL = &a->t[1], *LB = &a->t[2], *SC = &a->t[3], *Y = &a->t[4];
    const int64_t N = p_int(a, "N", 0), R = p_int(a, "R", 0), S = p_int(a, "S", 0);
    if (N <= 0 || R <= 0 || S <= 0) return RAD_E_SHAPE;
    const int64_t M = band_rows(numel(Y) / N, p_int(a, "M", 0));
    if (M <= 0) return RAD_E_SHAPE;                         /* zero rows is not a call */
    if (numel(T) < M * R || numel(SL) < M || numel(LB) < S * N * R || numel(SC) < S * N)
        return RAD_E_SHAPE;
    if (Y->dtype != RAD_BF16 && Y->dtype != RAD_F16 && Y->dtype != RAD_F32) return RAD_E_DTYPE;
    if (T->dtype != RAD_F32 || SC->dtype != RAD_F32) return RAD_E_DTYPE;

    const Mat mt = as_mat(T, R);
    const int64_t ys_r = row_stride(Y, N), ys_c = laststride(Y);
    const Pool3 pb = pool3(LB, N, R);
    const int64_t sc_r = SC->rank >= 2 ? SC->stride[SC->rank - 2] : N;
    const int64_t sc_c = SC->rank >= 2 ? laststride(SC) : 1;
    const float* const scd = (const float*)SC->data;

    AVX_PARALLEL_FOR_IF(M * N * R >= AVX_PAR_MIN)
    for (int64_t e = 0; e < M * N; ++e) {
        const int64_t m = e / N, n = e - m * N;
        const int64_t s = row_slot(SL, m, S);
        if (s < 0) continue;                             /* a base row: not read, not written */
        const float* tv = in_row(T, m * mt.rs, R, scratch_f32((size_t)R));
        if (bf16_t) {
            float* tr = scratch_f32((size_t)R);
            if (tr != tv) for (int64_t j = 0; j < R; ++j) tr[j] = tv[j];
            for (int64_t j = 0; j < R; ++j) tr[j] = rad_bf16_to_f32(rad_f32_to_bf16(tr[j]));
            tv = tr;
        }
        const float* bv = in_row(LB, s * pb.s0 + n * pb.s1, R, scratch_f32_b((size_t)R));
        const float d = scd[s * sc_r + n * sc_c] * dotf(tv, bv, R);
        const int64_t o = m * ys_r + n * ys_c;
        stt(Y, o, ldt(Y, o) + d);
    }
    return RAD_OK;
}

AVX_KERNEL(lora_expand) { return expand_impl(a, false); }

/* lora_expand with t rounded to bf16 (docs/OPS.md). */
AVX_KERNEL(lora_expand_mm) { return expand_impl(a, true); }
