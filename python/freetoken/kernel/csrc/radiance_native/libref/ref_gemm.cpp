/* ref_gemm.cpp -- the GEMM family and the two vocabulary edges.
 *
 *     nt: C[M,N] = A[M,K] @ B[N,K]^T
 *
 * B is stored transposed, the layout transformer weights already have, so both operands are
 * k-contiguous and the inner loop is a dot product over k.
 *
 * This is the one family where the per-element dtype switch would dominate (one element read per
 * FLOP; a prefill chunk is ~10^11 loads even for a small model), so the inner loop is templated on
 * the operand dtypes and selected once per launch through a function pointer
 * (docs/IMPLEMENTATION.md). Everything else pays the switch.
 *
 * Parallelism is over N, not M: decode is M=1 and verify M<=64, while N is thousands in every
 * transformer GEMM. Each thread owns whole output elements, so summation order is independent of
 * thread count.
 */
#include <algorithm>
#include "ref_common.h"
#include "ref_gemm.h"
#include "ref_ops.h"
#include "ref_planes.h"

using namespace ref;

namespace {

/* ------------------------------------------------------------------ the tiled driver */
/* Y[m, n] = quant_dot(m, n) (+ bias[n]) over every output, in quant_tile's tiles (ref_gemm.h).
 * Each tile is owned by one thread and every output's chain is fixed, so the result depends on
 * neither the tiling nor the thread count. */
template <class LA, class LB>
static void gemm_tiles(const Mat& A, const Mat& B, const Scales& SA, const Scales& SB,
                       const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N,
                       int64_t K) {
    const int64_t ys_r = row_stride(Y, N);
    const int64_t ys_c = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;
    const int64_t bs = bias ? laststride(bias) : 0;
    const TileGrid tg = tile_grid(M, N);
    std::vector<int64_t> rows((size_t)M);
    for (int64_t m = 0; m < M; ++m) rows[(size_t)m] = m;
    REF_PAR_REGION(M * N * K)
    {
        float* const ws = ws_get(tile_ws_floats(tg.rg));
        REF_FOR
        for (int64_t it = 0; it < tg.items(); ++it) {
            const int64_t n0 = (it / tg.rgc) * REF_NB, r0 = (it % tg.rgc) * tg.rg;
            const int64_t nb = (n0 + REF_NB < N) ? REF_NB : N - n0;
            const int64_t R  = (r0 + tg.rg < M) ? tg.rg : M - r0;
            const float* acc = quant_tile<LA, LB>(A, B, SA, SB, rows.data() + r0, R, n0, nb, K,
                                                  ws);
            for (int64_t r = 0; r < R; ++r)
                for (int64_t j = 0; j < nb; ++j) {
                    float v = acc[r * REF_NB + j];
                    if (bias) v += ldt(bias, (n0 + j) * bs);
                    stt(Y, (r0 + r) * ys_r + (n0 + j) * ys_c, v);
                }
        }
    }
}

/* ------------------------------------------------------------------ the unquantised core */
/* The tiled driver with no scale streams: each output is acc += a * b over ascending k from
 * zero, then the bias. */
template <class LA, class LB>
static void gemm_plain(const Mat& A, const Mat& B, const RadTensor* Y, const RadTensor* bias,
                       int64_t M, int64_t N, int64_t K) {
    const Scales none = make_scales(nullptr, 0, K, false);
    gemm_tiles<LA, LB>(A, B, none, none, Y, bias, M, N, K);
}

typedef void (*PlainFn)(const Mat&, const Mat&, const RadTensor*, const RadTensor*,
                        int64_t, int64_t, int64_t);

/* The dtype pairs with a compile-time loader; any other pair takes gemm_plain<Dyn, Dyn>. */
#define REF_PLAIN_ROW(DA)                                                \
    case DA:                                                             \
        switch (db) {                                                    \
            case RAD_F32:  return &gemm_plain<Fix<DA>, Fix<RAD_F32>>;    \
            case RAD_F16:  return &gemm_plain<Fix<DA>, Fix<RAD_F16>>;    \
            case RAD_BF16: return &gemm_plain<Fix<DA>, Fix<RAD_BF16>>;   \
            case RAD_I8:   return &gemm_plain<Fix<DA>, Fix<RAD_I8>>;     \
            default:       return &gemm_plain<Dyn, Dyn>;                 \
        }

static PlainFn pick_plain(uint32_t da, uint32_t db) {
    switch (da) {
        REF_PLAIN_ROW(RAD_F32)
        REF_PLAIN_ROW(RAD_F16)
        REF_PLAIN_ROW(RAD_BF16)
        REF_PLAIN_ROW(RAD_I8)
        default: return &gemm_plain<Dyn, Dyn>;
    }
}
#undef REF_PLAIN_ROW

/* ------------------------------------------------------------------ the quantised core
 * `quant_dot` (ref_gemm.h) over a whole plane, with the loaders chosen at compile time where the
 * pair is listed below. */
template <class LA, class LB>
static void gemm_quant(const Mat& A, const Mat& B, const Scales& SA, const Scales& SB,
                       const RadTensor* Y, const RadTensor* bias,
                       int64_t M, int64_t N, int64_t K) {
    const int64_t g = SB.group > 0 ? SB.group : K;
    if (!SA.p || SA.group >= g) {
        gemm_tiles<LA, LB>(A, B, SA, SB, Y, bias, M, N, K);
        return;
    }
    /* An activation grid finer than the weight's, which quant_tile does not serve. */
    const int64_t ys_r = row_stride(Y, N);
    const int64_t ys_c = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;
    REF_PAR_FOR(N * M * K)
    for (int64_t n = 0; n < N; ++n)
        for (int64_t m = 0; m < M; ++m) {
            float acc = quant_dot<LA, LB>(A, B, SA, SB, m, n, K);
            if (bias) acc += ldt(bias, n * laststride(bias));
            stt(Y, m * ys_r + n * ys_c, acc);
        }
}

typedef void (*QuantFn)(const Mat&, const Mat&, const Scales&, const Scales&, const RadTensor*,
                        const RadTensor*, int64_t, int64_t, int64_t);

#define REF_QUANT_ROW(DA)                                                    \
    case DA:                                                                 \
        switch (db) {                                                        \
            case RAD_I4:      return &gemm_quant<Fix<DA>, Fix<RAD_I4>>;      \
            case RAD_I2:      return &gemm_quant<Fix<DA>, Fix<RAD_I2>>;      \
            case RAD_FP4E2M1: return &gemm_quant<Fix<DA>, Fix<RAD_FP4E2M1>>; \
            case RAD_I8:      return &gemm_quant<Fix<DA>, Fix<RAD_I8>>;      \
            case RAD_F8E4M3:  return &gemm_quant<Fix<DA>, Fix<RAD_F8E4M3>>;  \
            case RAD_BF16:    return &gemm_quant<Fix<DA>, Fix<RAD_BF16>>;    \
            case RAD_F32:     return &gemm_quant<Fix<DA>, Fix<RAD_F32>>;     \
            default:          return &gemm_quant<Dyn, Dyn>;                  \
        }

static QuantFn pick_quant(uint32_t da, uint32_t db) {
    switch (da) {
        REF_QUANT_ROW(RAD_I8)
        REF_QUANT_ROW(RAD_F8E4M3)
        REF_QUANT_ROW(RAD_BF16)
        REF_QUANT_ROW(RAD_F32)
        default: return &gemm_quant<Dyn, Dyn>;
    }
}
#undef REF_QUANT_ROW

/* gemm_nt_bias's epilogue, applied to what the GEMM stored: an activation, then a residual. Each
 * step reads and stores, so each rounds to the output dtype, exactly as the unfused chain of three
 * tensors would. `act`: 0 none, 1 exact (erf) GELU, 2 tanh GELU. */
static void gemm_epilogue(const RadTensor* Y, const RadTensor* res, int act, int64_t M, int64_t N) {
    const int64_t ys_r = row_stride(Y, N), ys_c = laststride(Y);
    const int64_t rs_r = res ? row_stride(res, N) : 0, rs_c = res ? laststride(res) : 0;
    REF_PAR_FOR(M * N)
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            const int64_t o = m * ys_r + n * ys_c;
            if (act == 1) stt(Y, o, act_gelu(ldt(Y, o)));
            else if (act == 2) stt(Y, o, act_gelu_tanh(ldt(Y, o)));
            if (res) stt(Y, o, ldt(res, m * rs_r + n * rs_c) + ldt(Y, o));
        }
}

/* ------------------------------------------------------------------ a joined head */
/* Y = A B^T for a B of codes and one scale (ref_planes.h: ScaledRows), decoded a block of rows at
 * a time into f32 and multiplied by the plain core. Each decoded element is the f32 the core's
 * affine decoder gives, so the result is bit-identical to the same product over that decoder's
 * f32 head, which is what a tool hands this plugin as the logical form. */
static int gemm_scaled_rows(const Mat& ma, const RadTensor* Bt, const RadTensor* Y,
                            const RadTensor* bias, int64_t M, int64_t N, int64_t K) {
    ScaledRows sr;
    const int st = scaled_rows(Bt, N, K, &sr);
    if (st != RAD_OK) return st;
    /* Rows per block: about 16 MiB of f32, at least one tile. */
    int64_t ch = ((int64_t)1 << 22) / K;
    ch = std::max<int64_t>(REF_NB, ch / REF_NB * REF_NB);
    std::vector<float> dec((size_t)(std::min(ch, N) * K));
    const int64_t ys_r = row_stride(Y, N), ys_c = laststride(Y);
    const PlainFn plain = pick_plain(ma.t->dtype, RAD_F32);
    for (int64_t n0 = 0; n0 < N; n0 += ch) {
        const int64_t nb = std::min(ch, N - n0);
        REF_PAR_FOR(nb * K)
        for (int64_t r = 0; r < nb; ++r)
            for (int64_t k = 0; k < K; ++k) dec[(size_t)(r * K + k)] = sr.at(n0 + r, k);
        RadTensor Bb{};
        Bb.data = dec.data();
        Bb.dtype = RAD_F32;
        Bb.rank = 2; Bb.shape[0] = nb; Bb.shape[1] = K; Bb.stride[0] = K; Bb.stride[1] = 1;
        RadTensor Yb = *Y;
        Yb.data = (uint8_t*)Y->data + rad_dtype_bytes(Y->dtype, n0 * ys_c);
        Yb.rank = 2; Yb.shape[0] = M; Yb.shape[1] = nb; Yb.stride[0] = ys_r; Yb.stride[1] = ys_c;
        RadTensor bb{};
        if (bias) {
            bb = *bias;
            bb.data = (uint8_t*)bias->data + rad_dtype_bytes(bias->dtype, n0 * laststride(bias));
        }
        plain(ma, as_mat(&Bb, K), &Yb, bias ? &bb : nullptr, M, nb, K);
    }
    return RAD_OK;
}

/* Shared entry for gemm_nt / gemm_nt_bias / logits_gemm. `nkey`/`kkey` name the parameters that
 * carry N and K, since logits_gemm spells them n_vocab and n_embd. */
/* Whether two operands' memory overlaps: from the first element to one past the last element
 * their strides reach. A residual may be the output itself (an in-place residual add). */
static bool spans_overlap(const RadTensor* a, const RadTensor* b) {
    auto span = [](const RadTensor* t, const char** lo, const char** hi) {
        int64_t last = 0;
        for (uint32_t i = 0; i < t->rank; ++i) last += (t->shape[i] - 1) * t->stride[i];
        const int64_t eb = (rad_dtype_bits(t->dtype) + 7) / 8;
        *lo = static_cast<const char*>(t->data);
        *hi = *lo + (last + 1) * eb;
    };
    const char *alo, *ahi, *blo, *bhi;
    span(a, &alo, &ahi);
    span(b, &blo, &bhi);
    return alo < bhi && blo < ahi;
}

/* The residual as f32 [M, N], read before the product overwrites it when it is the output's own
 * memory. Widening is exact for every float dtype, so the epilogue's sum is unchanged. */
static RadTensor snapshot_res(const RadTensor* res, int64_t M, int64_t N, std::vector<float>* buf) {
    const int64_t rs_r = row_stride(res, N), rs_c = laststride(res);
    buf->resize((size_t)(M * N));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) (*buf)[(size_t)(m * N + n)] = ldt(res, m * rs_r + n * rs_c);
    RadTensor t{};
    t.dtype = RAD_F32;
    t.rank = 2;
    t.shape[0] = M;
    t.shape[1] = N;
    t.stride[0] = N;
    t.stride[1] = 1;
    t.data = buf->data();
    return t;
}

static int gemm_nt_impl(const RadArgs* a, bool bias_operand, const char* nkey, const char* kkey) {
    /* gemm_nt_bias is a, b, bias, res?, y; the other two are a, b, y. */
    const int nt = bias_operand ? 5 : 3;
    if (a->n_t < nt || !a->t[0].data || !a->t[1].data || !a->t[nt - 1].data) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[nt - 1];
    const RadTensor* bias = bias_operand ? t_in(a, 2) : nullptr;
    const RadTensor* res  = bias_operand ? t_in(a, 3) : nullptr;
    static const char* kActs[] = { "none", "gelu", "gelu_tanh" };
    const int act = bias_operand ? p_enum(a, "act", kActs, 3, 0) : 0;
    if (act < 0 || act > 2) return RAD_E_INVAL;

    const int64_t K = p_int(a, kkey, B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, nkey, B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / N;
    if (M <= 0) return RAD_E_SHAPE;
    /* A head of codes and scale, joined (ref_planes.h); its extents are its planes'. */
    const bool joined = is_joined(B);
    if (numel(A) < M * K || (!joined && numel(B) < N * K)) return RAD_E_SHAPE;
    if (bias && numel(bias) < N) return RAD_E_SHAPE;

    if (res && numel(res) < M * N) return RAD_E_SHAPE;
    std::vector<float> res_buf;
    RadTensor res_copy{};
    if (res && spans_overlap(res, Y)) {
        res_copy = snapshot_res(res, M, N, &res_buf);
        res = &res_copy;
    }

    const Mat ma = as_mat(A, K);
    if (joined) {
        const int st = gemm_scaled_rows(ma, B, Y, bias, M, N, K);
        if (st != RAD_OK) return st;
        if (act || res) gemm_epilogue(Y, res, act, M, N);
        return RAD_OK;
    }
    const Mat mb = as_mat(B, K);
    pick_plain(A->dtype, B->dtype)(ma, mb, Y, bias, M, N, K);
    if (act || res) gemm_epilogue(Y, res, act, M, N);
    return RAD_OK;
}

/* ------------------------------------------------------------------ the zero-point grids */
/* gemm_nt_q over an asymmetric grid (ref_planes.h, ZpGrid): weight element (n, k) is
 * (code - zero) * scale with one f16 scale and one integer zero per (row, `group` of K), read from
 * `b_scale`'s joined planes. Each group's sub-sum is formed on the integer grid and scaled once, as
 * quant_dot does for the symmetric grids:
 *
 *   ZP_I8_ROW    sub = sum(a * code) - zero * a_sum[g][m]      exact integers below 2^24
 *                acc += (sub * a_scale[m]) * scale
 *   ZP_I8_GROUP  sub = sum(a * (code - zero))                  exact
 *                acc += (sub * a_scale[m][g]) * scale
 *   ZP_F16       BF16 inputs are first narrowed to F16 (the cast kernel's rounding); sub is
 *                the sum over ascending k of a * (code - zero), then acc += sub * scale
 *
 * `a` is already in column order (the caller undoes the step order). `a_sum` is quant_act_i8's,
 * group-major [K/group][Mpad], Mpad = ceil(M/16)*16; the grid's zero is removed through it, as
 * libr4d's kernels do, so it is required. Parts (`pb1`, `pb2`, the ParoQuant grids only): weight
 * rows [0, e1), [e1, e2), [e2, N) meet copies 0, 1, 2 of the activation, side by side in `a` and in
 * `a_scale`. */
static int gemm_zp(const ZpGrid& zg, const RadTensor* A, const RadTensor* as_,
                   const RadTensor* B, const RadTensor* bs_, const RadTensor* asum,
                   const RadTensor* Y, int64_t M, int64_t N, int64_t K, int64_t group,
                   int64_t pb1, int64_t pb2) {
    if (!is_joined(bs_)) return RAD_E_DTYPE;
    const JPlane sc = jplane(bs_, "scale"), zp = jplane(bs_, "zero");
    if (!sc || !zp || !float_scale_dt(sc.dt) || !zero_dt(zp.dt)) return RAD_E_DTYPE;
    if (B->dtype != zg.codes) return RAD_E_DTYPE;
    if (group <= 0 || K % group) return RAD_E_SHAPE;
    const int64_t G = K / group;
    if (sc.rows != N || sc.cols != G || zp.rows != N || zp.cols != G) return RAD_E_SHAPE;
    if (!rad_tensor_is_contiguous(B)) return RAD_E_STRIDE;
    const int64_t brow = rad_enc_row_bytes(B->dtype, K);
    const int cbits = rad_dtype_bits(B->dtype);

    /* The parts' row bounds and the activation's width. */
    const bool parted = pb1 > 0 || pb2 > 0;
    if (parted && zg.act == ZP_I8_ROW) return RAD_E_UNSUPPORTED;
    const int64_t e1 = pb1 > 0 ? std::min<int64_t>(pb1, N) : N;
    const int64_t e2 = pb2 > 0 ? std::max<int64_t>(e1, std::min<int64_t>(pb2, N)) : N;
    const int64_t parts = e2 < N ? 3 : (e1 < N ? 2 : 1);
    if (parted && ((e1 % 16) || (e2 % 16))) return RAD_E_SHAPE;
    const int64_t acols = parts * K;
    if (numel(A) < M * acols) return RAD_E_SHAPE;
    if (A->rank >= 2 && A->shape[A->rank - 1] != acols) return RAD_E_SHAPE;
    int64_t ars = row_stride(A, acols), acs = laststride(A);

    /* A BF16 activation (the W4 decode graph's, which the card narrows on load) is narrowed to F16
     * first, with the cast kernel's rounding, so the F16 grid sees what the card's GEMM sees. */
    std::vector<uint16_t> narrowed;
    RadTensor a_f16{};
    if (zg.act == ZP_F16 && A->dtype == RAD_BF16) {
        narrowed.resize((size_t)(M * acols));
        for (int64_t m = 0; m < M; ++m)
            for (int64_t k = 0; k < acols; ++k)
                narrowed[(size_t)(m * acols + k)] = f32_to_f16(ldt(A, m * ars + k * acs));
        a_f16 = *A;
        a_f16.data = narrowed.data();
        a_f16.dtype = RAD_F16;
        a_f16.rank = 2;
        a_f16.shape[0] = M;
        a_f16.shape[1] = acols;
        a_f16.stride[0] = acols;
        a_f16.stride[1] = 1;
        A = &a_f16;
        ars = acols;
        acs = 1;
    }

    const bool i8 = zg.act != ZP_F16;
    if (i8 && A->dtype != RAD_I8) return RAD_E_DTYPE;
    if (!i8 && !float_scale_dt(A->dtype)) return RAD_E_DTYPE;
    const int64_t mpad = (M + 15) / 16 * 16;
    if (zg.act == ZP_I8_ROW) {
        if (!as_ || numel(as_) < M) return RAD_E_INVAL;
        if (!asum || numel(asum) < G * mpad) return RAD_E_INVAL;
    }
    if (zg.act == ZP_I8_GROUP && (!as_ || numel(as_) < M * parts * G)) return RAD_E_INVAL;
    const int64_t sa_rs = as_ ? row_stride(as_, zg.act == ZP_I8_GROUP ? parts * G : 1) : 0;
    const int64_t sa_cs = as_ ? laststride(as_) : 0;
    const int64_t ys_r = row_stride(Y, N), ys_c = laststride(Y);
    const int64_t asum_s = asum ? laststride(asum) : 0;

    REF_PAR_FOR(N * M * K)
    for (int64_t n = 0; n < N; ++n) {
        const int64_t part = n < e1 ? 0 : (n < e2 ? 1 : 2);
        const uint8_t* crow = (const uint8_t*)B->data + n * brow;
        std::vector<int32_t> q((size_t)K);
        for (int64_t k = 0; k < K; ++k) q[(size_t)k] = (int32_t)rad_load_bits(crow, cbits, k);
        for (int64_t m = 0; m < M; ++m) {
            const int64_t ab = m * ars + part * K * acs;
            float acc = 0.0f;
            for (int64_t g = 0; g < G; ++g) {
                const float sb = sc.at(n, g);
                const int32_t z = (int32_t)zp.at(n, g);
                const int64_t k0 = g * group;
                float sub;
                if (zg.act == ZP_F16) {
                    sub = 0.0f;
                    for (int64_t k = k0; k < k0 + group; ++k)
                        sub += ldt(A, ab + k * acs) * (float)(q[(size_t)k] - z);
                    acc += sub * sb;
                    continue;
                }
                int64_t is = 0;
                const int8_t* ar = (const int8_t*)A->data;
                if (zg.act == ZP_I8_ROW) {
                    for (int64_t k = k0; k < k0 + group; ++k)
                        is += (int64_t)ar[ab + k * acs] * q[(size_t)k];
                    sub = (float)is - (float)z * ldt(asum, (g * mpad + m) * asum_s);
                    acc += sub * ldt(as_, m * sa_rs) * sb;
                } else {
                    for (int64_t k = k0; k < k0 + group; ++k)
                        is += (int64_t)ar[ab + k * acs] * (q[(size_t)k] - z);
                    sub = (float)is;
                    acc += sub * ldt(as_, m * sa_rs + (part * G + g) * sa_cs) * sb;
                }
            }
            stt(Y, m * ys_r + n * ys_c, acc);
        }
    }
    return RAD_OK;
}

static int gemm_ntq_impl(const RadArgs* a, bool bias_operand) {
    const int nt = bias_operand ? 6 : 5;
    if (a->n_t < nt || !a->t[0].data || !a->t[2].data || !a->t[nt - 1].data) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[2], *Y = &a->t[nt - 1];
    const RadTensor* as_ = t_in(a, 1);
    const RadTensor* bs_ = t_in(a, 3);
    const RadTensor* bias = bias_operand ? t_in(a, 4) : nullptr;
    /* The zero-point grids read `b_scale` joined and the w2a8 / w4a8_asym ones `a_sum` too
     * (gemm_zp); they have no bias form. Elsewhere `a_sum` would remove a zero point the grid
     * does not have, so a call passing it is refused rather than answered as if it were absent.
     * `b_ref` is the per-row reference exponent an mxfp4 kernel folds its scales at; it is derived
     * from `b_scale` and does not change the product, so it is accepted beside an e2m1 weight
     * only. */
    const ZpGrid* zg = zp_grid(p_str(a, "dtype", nullptr));
    if (zg && bias_operand) return RAD_E_UNSUPPORTED;
    if (!bias_operand) {
        if (t_in(a, 5) && (!zg || zg->act != ZP_I8_ROW)) return RAD_E_UNSUPPORTED;
        if (t_in(a, 6) && B->dtype != RAD_FP4E2M1) return RAD_E_UNSUPPORTED;
    }

    const int64_t K = p_int(a, "K", B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, "N", B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / N;
    if (M <= 0) return RAD_E_SHAPE;
    if (numel(A) < M * K || numel(B) < N * K) return RAD_E_SHAPE;
    if (bias && numel(bias) < N) return RAD_E_SHAPE;

    /* quant_act_i8's codes arrive in the step order (ref_common.h); read them back into column
     * order. */
    std::vector<int8_t> a_cols;
    RadTensor Ac;
    if (A->dtype == RAD_I8 && i8_steps_dtype(p_str(a, "dtype", nullptr))) {
        if (K % 16) return RAD_E_SHAPE;
        if (laststride(A) != 1) return RAD_E_STRIDE;
        a_cols.resize((size_t)(M * K));
        for (int64_t m = 0; m < M; ++m) {
            const int8_t* row = (const int8_t*)A->data + rowoff(A, m, K);
            for (int64_t c = 0; c < K; ++c) a_cols[(size_t)(m * K + c)] = row[i8_step_pos(c)];
        }
        Ac = *A;
        Ac.data = a_cols.data();
        Ac.rank = 2; Ac.shape[0] = M; Ac.shape[1] = K; Ac.stride[0] = K; Ac.stride[1] = 1;
        A = &Ac;
    }

    /* Parts (`pb1`, `pb2`): weight rows [0, pb1), [pb1, pb2) and [pb2, N) each multiply their own
     * copy of the activation, for a fused projection whose pieces were rotated separately
     * (ParoQuant). The copies sit side by side, so `a` is [M, parts * K] and its scale
     * [M, parts * K / 128]; each part is computed as a plain product over its rows. 0 omits a
     * bound: no `pb1` is one part, no `pb2` ends the second at N. */
    const int64_t pb1 = p_int(a, "pb1", 0), pb2 = p_int(a, "pb2", 0);
    if (zg)
        return gemm_zp(*zg, A, as_, B, bs_, t_in(a, 5), Y, M, N, K, p_int(a, "group", 0), pb1,
                       pb2);
    /* A scale in two levels, joined (ref_layout_gemm_q): an integer per (row, group) under a float
     * a row. Their product in f32 is the scale the loop below reads. */
    std::vector<float> two_level;
    RadTensor bs_two;
    if (is_joined(bs_)) {
        const JPlane s1 = jplane(bs_, "scale"), s2 = jplane(bs_, "scale.1");
        if (!s1 || !s2 || s1.dt != RAD_U8 || !float_scale_dt(s2.dt)) return RAD_E_DTYPE;
        if (s1.rows != N || s1.cols <= 0 || s2.rows != N || s2.cols != 1) return RAD_E_SHAPE;
        two_level.resize((size_t)(N * s1.cols));
        for (int64_t n = 0; n < N; ++n)
            for (int64_t g = 0; g < s1.cols; ++g)
                two_level[(size_t)(n * s1.cols + g)] = s1.at(n, g) * s2.at(n, 0);
        std::memset(&bs_two, 0, sizeof(bs_two));
        bs_two.data = two_level.data();
        bs_two.dtype = RAD_F32;
        bs_two.rank = 2;
        bs_two.shape[0] = N; bs_two.shape[1] = s1.cols;
        bs_two.stride[0] = s1.cols; bs_two.stride[1] = 1;
        bs_ = &bs_two;
    }
    if (pb1 > 0 || pb2 > 0) {
        if (bias_operand || B->dtype != RAD_FP4E2M1 || A->rank < 2) return RAD_E_UNSUPPORTED;
        const int64_t e1 = pb1 > 0 ? std::min<int64_t>(pb1, N) : N;
        const int64_t e2 = pb2 > 0 ? std::max<int64_t>(e1, std::min<int64_t>(pb2, N)) : N;
        const int64_t bound[4] = { 0, e1, e2, N };
        const int64_t acols = A->shape[A->rank - 1];
        const int64_t parts = e2 < N ? 3 : (e1 < N ? 2 : 1);
        if (acols != parts * K || (e1 % 16) || (e2 % 16)) return RAD_E_SHAPE;
        const bool sa_row = as_ && as_->data && as_->rank >= 2;
        if (!sa_row) return RAD_E_INVAL;
        const int64_t sa_cols = as_->shape[as_->rank - 1];
        if (sa_cols % parts) return RAD_E_SHAPE;
        const int64_t sa_part = sa_cols / parts;                 /* K / 128 */
        for (int p = 0; p < 3; ++p) {
            const int64_t n0 = bound[p], n1 = bound[p + 1];
            if (n1 <= n0) continue;
            RadTensor Ap = *A, Sp = *as_, Bp = *B, Bsp = *bs_, Yp = *Y;
            Ap.data = (uint8_t*)A->data + rad_dtype_bytes(A->dtype, p * K);
            Ap.rank = 2; Ap.shape[0] = M; Ap.shape[1] = K;
            Ap.stride[0] = row_stride(A, acols); Ap.stride[1] = 1;
            Sp.data = (uint8_t*)as_->data + rad_dtype_bytes(as_->dtype, p * sa_part);
            Sp.shape[Sp.rank - 1] = sa_part;
            Bp.data = (uint8_t*)B->data + rad_dtype_bytes(B->dtype, n0 * K);
            Bp.rank = 2; Bp.shape[0] = n1 - n0; Bp.shape[1] = K; Bp.stride[0] = K; Bp.stride[1] = 1;
            const int64_t bs_cols = bs_->shape[bs_->rank - 1];
            Bsp.data = (uint8_t*)bs_->data + rad_dtype_bytes(bs_->dtype, n0 * bs_cols);
            Bsp.rank = 2; Bsp.shape[0] = n1 - n0; Bsp.shape[1] = bs_cols;
            Bsp.stride[0] = bs_cols; Bsp.stride[1] = 1;
            const int64_t ys = row_stride(Y, N);
            Yp.data = (uint8_t*)Y->data + rad_dtype_bytes(Y->dtype, n0);
            Yp.rank = 2; Yp.shape[0] = M; Yp.shape[1] = n1 - n0; Yp.stride[0] = ys; Yp.stride[1] = 1;
            Scales SAp = make_scales(&Sp, M, K, false), SBp = make_scales(&Bsp, n1 - n0, K, true);
            const Mat map_ = as_mat(&Ap, K), mbp = as_mat(&Bp, K);
            pick_quant(Ap.dtype, Bp.dtype)(map_, mbp, SAp, SBp, &Yp, nullptr, M, n1 - n0, K);
        }
        return RAD_OK;
    }

    const bool mx = B->dtype == RAD_FP4E2M1;
    Scales SA = make_scales(as_, M, K, false);
    Scales SB = make_scales(bs_, N, K, mx);
    /* `group` is what the selector matches on; the loop uses the group the scale stream implies,
     * which describes the bytes. A disagreement is refused. */
    const int64_t gp = p_int(a, "group", 0);
    if (gp > 0 && SB.p && SB.group != gp) return RAD_E_SHAPE;

    const Mat ma = as_mat(A, K), mb = as_mat(B, K);
    pick_quant(A->dtype, B->dtype)(ma, mb, SA, SB, Y, bias, M, N, K);
    return RAD_OK;
}

}  /* namespace */

int ref_gemm_nt(const RadArgs* a, RadStream) {
    return gemm_nt_impl(a, false, "N", "K");
}
int ref_gemm_nt_bias(const RadArgs* a, RadStream) {
    return gemm_nt_impl(a, true, "N", "K");
}
int ref_gemm_nt_q(const RadArgs* a, RadStream) {
    return gemm_ntq_impl(a, false);
}
int ref_gemm_nt_q_bias(const RadArgs* a, RadStream) {
    return gemm_ntq_impl(a, true);
}

/* ------------------------------------------------------------------ logits_gemm */
/* gemm_nt's arithmetic as a separate op: lm_head is its own access class (RAD_ACCESS_VOCAB) and
 * the sampler fuses against it, so the two must be selectable apart (docs/OPS.md). N and K are
 * spelled n_vocab and n_embd. */
int ref_logits_gemm(const RadArgs* a, RadStream) {
    return gemm_nt_impl(a, false, "n_vocab", "n_embd");
}

/* ------------------------------------------------------------------ embed_lookup */
/* A gather, not a GEMM. A negative token id writes a zero row (padding for short sequences).
 *
 * `vocab_offset` (optional, default 0) makes the table this rank's slice of a vocab-parallel
 * embedding, covering global ids [vocab_offset, vocab_offset + n_vocab). Ids stay global; one
 * outside the slice writes a zero row, so summing the ranks reconstructs x.
 *
 * When this rank holds the whole vocabulary, an id >= n_vocab fails the launch: it is a caller bug,
 * and a plausible embedding is the worst answer. Under a split the kernel cannot know the global
 * vocabulary, so the check is the caller's (the sampler and tokeniser are bounded by it).
 *
 * Every row read is multiplied by `factor`: embed_lookup's `mult`, embed_lookup_q's table scale. */
static int embed_gather(const RadArgs* a, const RadTensor* tok, const RadTensor* wte,
                        const RadTensor* x, float factor) {
    const int64_t n_embd = p_int(a, "n_embd", wte->rank >= 2 ? wte->shape[wte->rank - 1] : 0);
    const int64_t n_vocab = p_int(a, "n_vocab", wte->rank >= 2 ? wte->shape[wte->rank - 2] : 0);
    const int64_t voff = p_int(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / n_embd;
    if (M <= 0 || numel(tok) < M || numel(wte) < n_vocab * n_embd) return RAD_E_SHAPE;

    /* Validated before the loop, since an OpenMP body cannot return. The guard keys on holding the
     * whole vocabulary, not on a zero offset: rank 0 of a vocab-parallel split has offset 0, and
     * its out-of-slice ids are other ranks' tokens that must become zero rows. */
    if (voff == 0 && a->world_size <= 1) {
        for (int64_t m = 0; m < M; ++m)
            if (ld_int(tok->dtype, tok->data, offlin(tok, m)) >= n_vocab) return RAD_E_INVAL;
    }

    const int64_t ws = wte->rank >= 2 ? wte->stride[wte->rank - 2] : n_embd;
    const int64_t wc = wte->rank >= 2 ? wte->stride[wte->rank - 1] : 1;
    REF_PAR_FOR(M * n_embd)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t id = ld_int(tok->dtype, tok->data, offlin(tok, m)) - voff;
        const int64_t xb = rowoff(x, m, n_embd), xs = laststride(x);
        const bool have_row = id >= 0 && id < n_vocab;
        for (int64_t i = 0; i < n_embd; ++i)
            stt(x, xb + i * xs, have_row ? ldt(wte, id * ws + i * wc) * factor : 0.0f);
    }
    return RAD_OK;
}

int ref_embed_lookup(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    return embed_gather(a, &a->t[0], &a->t[1], &a->t[2], p_f32(a, "mult", 1.0f));
}

/* A gather from a quantised table with one scale for the whole table.
 *
 * A separate op rather than a dtype on embed_lookup because operands are positional and this one
 * adds a scale; an optional operand would have moved `x` from slot 2 to slot 3 for every caller
 * (the same reason moe_gemm_q and gemm_nt_q are separate ops).
 *
 * One scale is the checkpoint's format: the n-gram embedding (hundreds of millions of rows) ships
 * as E4M3 with a single bf16 `weight_scale`. E4M3 carries its own exponent, so reconstruction is
 * insensitive to the scale over octaves and a per-row scale gains only a few percent. (INT8 would
 * gain substantially, but that is a different format.)
 */
int ref_embed_lookup_q(const RadArgs* a, RadStream) {
    /* `scale` (operand 2) is optional; the other three are required. */
    if (a->n_t < 4 || !rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 3)) return RAD_E_INVAL;
    const RadTensor* sc = &a->t[2];
    /* An absent scale is 1, so a plain bf16 n-gram embedding read in place from a container takes
     * the same op and host path, keeping a table too large for any device pool on the mapped tier. */
    const float scale = sc->data && numel(sc) >= 1 ? ldt(sc, 0) : 1.0f;
    return embed_gather(a, &a->t[0], &a->t[1], &a->t[3], scale);
}
