/* ref_lora.cpp -- runtime LoRA: the two halves of y += scale * (B . (A . x)), per row.
 *
 *     lora_shrink   t[m, j]  = sum_k x(m, k) * A[s, j, k]                     s = slot[m]
 *     lora_expand   y[m, n]  = y[m, n] + scale[s, n] * sum_j t[m, j] * B[s, n, j]
 *
 * The adapter is a per-row operand, since a step mixes rows of several adapters and rows of none.
 * A row whose slot is outside [0, S) is a base row and neither op touches it: shrink writes nothing
 * into its `t` row and expand neither reads nor stores its `y` row, so a base row stays bit for bit
 * what the gemm wrote (`y + 0` would turn -0 into +0).
 *
 * x(m, k) is the value gemm_nt_q uses for that activation element, read the same way: through the
 * row and column strides of `a` (a column slice of a fused projection keeps the whole projection's
 * row stride) and through ref_gemm.h's make_scales for `a_scale`. A code times its group's scale is
 * one f32 product and a float activation is its value. That is one more rounding than gemm_nt_q's
 * group sub-sum, and it is the contract (docs/OPS.md): it keeps the kernels free of a group loop.
 *
 * The sum is per row in a fixed order (k ascending for shrink, j ascending for expand, one f32
 * accumulator, no FMA: libref/CMakeLists.txt), so a row's bytes do not depend on M or on the rows
 * batched with it.
 *
 * Every extent is the operand's and the parameters are cross-checks (ref_ops.h): `M` is the band,
 * and `K`, `N`, `R`, `S` say how to read the tensors and are refused where a tensor is too small.
 */
#include "ref_common.h"
#include "ref_gemm.h"
#include "ref_ops.h"

using namespace ref;

namespace {

/* The slot of row m, or -1 for a base row. The id is read as an integer, never through f32. */
static inline int64_t row_slot(const RadTensor* slot, int64_t m, int64_t S) {
    const int64_t s = ld_int(slot->dtype, slot->data, offlin(slot, m));
    return (s >= 0 && s < S) ? s : -1;
}

/* A pool operand [S, rows, cols]: its three strides. The engine's pools are dense, but a rank-3
 * tensor is read through its own strides: density is a fact about one caller, not about the op. */
struct Pool3 { int64_t s0, s1, s2; };
static inline Pool3 pool3(const RadTensor* t, int64_t rows, int64_t cols) {
    if (t->rank == 3) return Pool3{ t->stride[0], t->stride[1], t->stride[2] };
    return Pool3{ rows * cols, cols, 1 };
}

/* The activation dtypes x(m, k) is defined for. The float ones carry no scale; the coded ones are
 * meaningless without one, and a missing scale is refused rather than read as one. */
static inline bool float_act(uint32_t dt) {
    return dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F32;
}
static inline bool coded_act(uint32_t dt) {
    return dt == RAD_F8E4M3 || dt == RAD_F8E5M2 || dt == RAD_I8;
}

}  /* namespace */

/* operands: a, a_scale?, slot, lora_a, t */
int ref_lora_shrink(const RadArgs* a, RadStream) {
    if (a->n_t < 5 || !rad_arg_in(a, 0) || !rad_arg_in(a, 2) || !rad_arg_in(a, 3) ||
        !rad_arg_in(a, 4))
        return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *SL = &a->t[2], *LA = &a->t[3], *T = &a->t[4];
    const RadTensor* XS = t_in(a, 1);
    if (XS && XS->dtype == RAD_DT_INVALID) XS = nullptr;     /* absent: null data or no dtype */
    const int64_t K = p_int(a, "K", 0), R = p_int(a, "R", 0), S = p_int(a, "S", 0);
    if (K <= 0 || R <= 0 || S <= 0) return RAD_E_SHAPE;
    const int64_t M = band_rows(numel(T) / R, p_int(a, "M", 0));
    if (M <= 0) return RAD_E_SHAPE;                         /* zero rows is not a call */
    if (numel(X) < M * K || numel(SL) < M || numel(LA) < S * R * K) return RAD_E_SHAPE;
    if (!float_act(X->dtype) && !coded_act(X->dtype)) return RAD_E_DTYPE;
    if (coded_act(X->dtype) && !XS) return RAD_E_INVAL;     /* a code without its scale */
    if (T->dtype != RAD_F32) return RAD_E_DTYPE;

    /* gemm_nt_q's reading of the scale stream, and a `group` cross-check: a parameter whose grid
     * (groups along K) disagrees with the stream's describes other bytes. The grid and not the
     * number, because a K no wider than one group has one group whatever `group` says. Without a
     * stream the parameter says nothing: a float activation has no scale. */
    const Scales SA = make_scales(XS, M, K, false);
    const int64_t gp = p_int(a, "group", 0);
    if (gp > 0 && SA.p && (K + gp - 1) / gp != (K + SA.group - 1) / SA.group) return RAD_E_SHAPE;

    const Mat mx = as_mat(X, K), mt = as_mat(T, R);
    const Pool3 pa = pool3(LA, R, K);
    const uint32_t dx = X->dtype;

    /* One thread an output element: the order inside it is k ascending whoever runs it. */
    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < M * R; ++e) {
        const int64_t m = e / R, j = e - m * R;
        const int64_t s = row_slot(SL, m, S);
        if (s < 0) continue;                     /* a base row: nothing written */
        const int64_t x0 = m * mx.rs, a0 = s * pa.s0 + j * pa.s1;
        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            float x = ld_dt(dx, X->data, x0 + k * mx.cs);
            if (SA.p) x = x * SA.at(m, k);
            acc += x * ld_dt(LA->dtype, LA->data, a0 + k * pa.s2);
        }
        ((float*)T->data)[m * mt.rs + j * mt.cs] = acc;
    }
    return RAD_OK;
}

/* operands: t, slot, lora_b, scale, y (inout). `bf16_t`: lora_expand_mm, t rounded to bf16. */
static int lora_expand_impl(const RadArgs* a, bool bf16_t) {
    if (!have(a, 5)) return RAD_E_INVAL;
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

    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < M * N; ++e) {
        const int64_t m = e / N, n = e - m * N;
        const int64_t s = row_slot(SL, m, S);
        if (s < 0) continue;                     /* a base row: not read, not written */
        const int64_t t0 = m * mt.rs, b0 = s * pb.s0 + n * pb.s1;
        float acc = 0.0f;
        for (int64_t j = 0; j < R; ++j) {
            float tv = ((const float*)T->data)[t0 + j * mt.cs];
            if (bf16_t) tv = rad_bf16_to_f32(rad_f32_to_bf16(tv));
            acc += tv * ld_dt(LB->dtype, LB->data, b0 + j * pb.s2);
        }
        const float sc = ((const float*)SC->data)[s * sc_r + n * sc_c];
        const int64_t o = m * ys_r + n * ys_c;
        /* `sc * acc` and then the add, each rounded: the order is the definition. */
        const float d = sc * acc;
        stt(Y, o, ldt(Y, o) + d);
    }
    return RAD_OK;
}

int ref_lora_expand(const RadArgs* a, RadStream) { return lora_expand_impl(a, false); }

/* lora_expand with t rounded to bf16; the order of the sum is free, so j ascending is one of the
 * answers it allows. */
int ref_lora_expand_mm(const RadArgs* a, RadStream) { return lora_expand_impl(a, true); }
