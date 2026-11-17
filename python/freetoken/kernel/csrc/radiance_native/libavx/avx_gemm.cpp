/* avx_gemm.cpp -- the GEMM family and the two vocabulary edges.
 *
 *     nt: C[M,N] = A[M,K] @ B[N,K]^T
 *
 * B is stored transposed, as transformer weights are, so both operands are k-contiguous.
 *
 * Which product runs:
 *   gemm_q_bf16    (avx_gemm_tile.cpp) on AVX512_BF16 parts: bf16 x bf16 and every plain
 *                  gemm_nt_q form, including a joined lm head read as codes and a scale grid
 *   gemm_tile_f32  (avx_gemm_tile.cpp) everything else, at every level
 *   gemm_drive's own loop, below: the forms the tile drivers decline (an E8M0 scale they cannot
 *                  fold, a decoded weight with a separate scale stream)
 *   the zero-point grids' small-M paths: their integer and 16-bit forms at M <= 16
 *
 * gemm_drive's loop is a dot-product register block: MR rows of A against NR rows of B, MR*NR
 * vector accumulators across the k loop, one horizontal reduction per output and group. The block
 * was swept (gemm_nt bf16, M=512, N=2048, K=2048, one thread; libavx/README.md names the part):
 *
 *   AVX-512, 32 ZMM         4x4  23.2 ms      4x6  22.1 ms  <-- chosen
 *                           6x4  24.2 ms      6x6  24.0 ms
 *                           8x4  23.8 ms      8x2  35.2 ms      12x2  34.1 ms
 *   AVX2/AVX, 16 YMM        3x3  34.4 ms      2x6  30.7 ms  <-- chosen
 *                           2x5  31.9 ms      2x4  34.8 ms      2x3  40.5 ms
 *                           3x4  58.4 ms      4x3  58.1 ms
 *
 * At AVX2, 3x4 and 4x3 need 12 accumulators + 3-4 A vectors + 1 B vector, 16-17 live registers in a
 * 16-register file: the spill lands inside the k loop and costs 1.9x. The scalar level uses 2x2.
 * Both operands are packed to f32 (A once per M block, each B row once per n-block), so every
 * weight format reaches one microkernel. Parallelism is over N; each thread owns whole outputs, so
 * results do not depend on the thread count.
 */
#include "avx_gemm.h"
#include "avx_rows.h"

#include <algorithm>
#include <vector>

using namespace avx;

#include "avx_had.h"

/* ================================================================== register blocking */
#if   AVX_LEVEL == 3
constexpr int64_t MR = 4, NR = 6;
#elif AVX_LEVEL == 2 || AVX_LEVEL == 1
constexpr int64_t MR = 2, NR = 6;
#else
constexpr int64_t MR = 2, NR = 2;
#endif

/* MR x NR dot products over [k0, k1), added into `out` so the grouped GEMM can call it per group.
 *
 * The k tail is summed scalar after the horizontal reduction rather than in a masked vector (free
 * only at AVX-512), so all four levels share one summation order. */
template <int MB_, int NB_>
static inline void dot_block(const float* A, int64_t lda, const float* B, int64_t ldb,
                             int64_t k0, int64_t k1, float out[MB_][NB_]) {
    vf acc[MB_][NB_];
    for (int i = 0; i < MB_; ++i)
        for (int j = 0; j < NB_; ++j) acc[i][j] = vf_zero();

    int64_t k = k0;
    for (; k + VF_N <= k1; k += VF_N) {
        vf av[MB_];
        for (int i = 0; i < MB_; ++i) av[i] = vf_loadu(A + i * lda + k);
        for (int j = 0; j < NB_; ++j) {
            const vf bv = vf_loadu(B + j * ldb + k);
            for (int i = 0; i < MB_; ++i) acc[i][j] = vf_fma(av[i], bv, acc[i][j]);
        }
    }
    for (int i = 0; i < MB_; ++i)
        for (int j = 0; j < NB_; ++j) {
            float s = vf_hsum(acc[i][j]);
            for (int64_t t = k; t < k1; ++t) s += A[i * lda + t] * B[j * ldb + t];
            out[i][j] += s;
        }
}


/* ================================================================== packing
 * Rows of an operand widened to contiguous f32. Returns a pointer into the operand itself when it
 * is already dense f32 with the right pitch, skipping the pack. */
static const float* pack_rows(const Mat& m, int64_t r0, int64_t nrows, int64_t K, float* dst,
                              int64_t ldd) {
    if (m.dec) {
        for (int64_t r = 0; r < nrows; ++r) m.dec->row(r0 + r, dst + r * ldd);
        return dst;
    }
    const RadTensor* t = m.t;
    if (m.cs == 1 && t->dtype == RAD_F32 && m.rs == ldd)
        return (const float*)t->data + r0 * m.rs;
    for (int64_t r = 0; r < nrows; ++r) widen_cols(m, r0 + r, 0, K, dst + r * ldd);
    return dst;
}

/* ================================================================== joined weights
 *
 * Row decoders for weights stored joined (avx_planes.h). Each writes a row as f32 with its scales
 * folded in, so the driver multiplies it like a plain row in one group:
 *
 *   HeadDec  codes times the scale covering them: an lm head, a mixing or fc row. Each element is
 *            the f32 the core's affine decoder gives, the head libref multiplies.
 *   ZpDec    a zero-point grid's row, (code - zero) * scale. For the grids that remove the zero
 *            through `a_sum` it is code * scale with one more column a group holding
 *            -zero * scale, against an activation whose matching column holds that group's sum.
 *
 * With the scales folded into the rows each product rounds once, where libref rounds each
 * group's sub-sum once; the two differ within the tolerance. */

/* Rows of codes and one scale (avx_planes.h: ScaledRows), `K` wide. */
struct HeadDec final : RowDecoder {
    avxp::ScaledRows sr;
    int64_t K = 0;
    void row(int64_t r, float* dst) const override { scaled_row(sr, r, K, dst); }
    void cols(int64_t r, int64_t k0, int64_t k1, float* dst) const override {
        const uint8_t* c = sr.codes.p + r * sr.codes.row_bytes;
        const int64_t span = sr.span(K);
        for (int64_t c0 = k0; c0 < k1; c0 += span)
            decode_scaled(sr.codes.dt, c + c0, sr.scale_at(r, c0), dst + (c0 - k0),
                          std::min(span, k1 - c0));
    }
    int64_t align() const override { return sr.span(K); }
    void codes(int64_t r, int64_t k0, int64_t k1, const void** p, int64_t* bytes) const override {
        *p = sr.codes.p + r * sr.codes.row_bytes + k0;
        *bytes = k1 - k0;
    }
};

/* A zero-point grid's rows (avx_planes.h: ZpGrid) from row `n0` on: codes at `codes_p`, `rb` bytes
 * a row, with the joined scale and zero planes. `aug` writes code * scale and the extra column of
 * -zero * scale per group, `group + 1` columns a group; otherwise (code - zero) * scale. */
struct ZpDec final : RowDecoder {
    const uint8_t* codes_p = nullptr;
    int64_t rb = 0, K = 0, group = 0, n0 = 0;
    int bits = 0;
    bool aug = false;
    avxp::JPlane sc, zp;
    void row(int64_t r, float* dst) const override { cols(r, 0, K / group * align(), dst); }
    /* Whole groups: [k0, k1) in the decoded row's columns, `group + 1` a group when aug. */
    void cols(int64_t r, int64_t k0, int64_t k1, float* dst) const override {
        const int64_t n = n0 + r;
        const uint8_t* cr = codes_p + n * rb;
        const int64_t gw = align();
        for (int64_t g = k0 / gw; g * gw < k1; ++g) {
            const float s = sc.at(n, g);
            const int32_t z = (int32_t)zp.at(n, g);
            float* d = dst + (g * gw - k0);
            if (aug) {
                decode_zp(cr, bits, g * group, group, 0, s, d);
                d[group] = -(float)z * s;
            } else {
                decode_zp(cr, bits, g * group, group, z, s, d);
            }
        }
    }
    int64_t align() const override { return aug ? group + 1 : group; }
    void codes(int64_t r, int64_t k0, int64_t k1, const void** p, int64_t* bytes) const override {
        const int64_t gw = align();
        *p = codes_p + (n0 + r) * rb + (k0 / gw) * group * bits / 8;
        *bytes = ((k1 - k0 + gw - 1) / gw) * group * bits / 8;
    }
};

/* ================================================================== the driver
 *
 * Null `SB`/`SA` is the unquantised family: one group of width K, scales 1. The grouped form sums
 * raw codes and applies the scales once per group, as libref and an int8 WMMA kernel do; scaling
 * every product would differ in the last bits. */
static int gemm_drive_fine_a(const Mat& ma, const Mat& mb, const Scales* SA, const Scales* SB,
                             const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N,
                             int64_t K);

static int gemm_drive(const Mat& ma, const Mat& mb, const Scales* SA, const Scales* SB,
                      const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N,
                      int64_t K) {
    const int64_t ys_r = row_stride(Y, N);
    const int64_t ys_c = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;
    const int64_t g = (SB && SB->p && SB->group > 0) ? SB->group : K;
    const int64_t ngrp = (K + g - 1) / g;
    /* An activation scale grid finer than the weight's cannot be hoisted out of the group. */
    const bool coarse_a = !SA || !SA->p || SA->group >= g;
    if (!coarse_a) return gemm_drive_fine_a(ma, mb, SA, SB, Y, bias, M, N, K);
    if (gemm_tile_f32(ma, mb, SA, SB, Y, bias, M, N, K)) return RAD_OK;

    /* ================================================== the M block
     *
     * Loop nest: (M block) x (N block) x K. Unblocked, every n-block re-reads the whole packed A
     * (at M=512, K=2048, a 4 MiB panel read 341 times). Blocking M keeps the panel in L2.
     *
     * On a part with 128 MiB of L3 the gain is only ~6% (gemm_nt bf16, M=512, N=2048, K=2048,
     * AVX-512, one thread: 23.4 ms at a 256 KiB panel vs 25.0 ms unblocked), because the panel
     * stays in L3 anyway; at M=64 it is ~2% slower (B is packed twice). It is kept for parts with
     * small L3, where the re-reads go to DRAM.
     *
     * MB is derived from K against a fixed byte target, not tuned (hence no tunable axes in
     * avx_registry.cpp). The target is a constant rather than read from CPUID cache descriptors,
     * which are unreliable; the measured curve is flat, so a wrong guess costs about 10%. */
    enum { AVX_L2_PANEL = 256 * 1024 };
    int64_t MB = (int64_t)AVX_L2_PANEL / ((int64_t)K * (int64_t)sizeof(float));
    /* A decoded weight (a joined head, a zero-point grid) costs a decode per M block on top of
     * the read, so its M block is as tall as a 16 MiB panel allows: an lm head at K = 5120 is
     * decoded once up to M = 819 instead of once every 12 rows. The panel is read-only and
     * shared by the threads through the last level. */
    if (mb.dec) MB = (int64_t)(16 << 20) / ((int64_t)K * (int64_t)sizeof(float));
    if (MB < MR) MB = MR;
    if (MB > M) MB = M;

    const int64_t lda = K;
    float* apack = (float*)scratch_raw((size_t)MB * (size_t)K * sizeof(float) + 64);
    const int64_t nblocks = (N + NR - 1) / NR;

    for (int64_t mb0 = 0; mb0 < M; mb0 += MB) {
        const int64_t mrows = (M - mb0) < MB ? (M - mb0) : MB;
        /* Packed on the calling thread before the parallel region; nothing in the region calls
         * scratch_raw, so the pointer stays valid for the workers. */
        const float* Ap = pack_rows(ma, mb0, mrows, K, apack, lda);

        AVX_PARALLEL_FOR_IF(mrows * N * K >= 65536)
        for (int64_t nb = 0; nb < nblocks; ++nb) {
            const int64_t n0 = nb * NR;
            const int64_t nn = (N - n0) < NR ? (N - n0) : NR;
            const int64_t ldb = K;
            float* bbuf = scratch_f32((size_t)NR * (size_t)K);
            const float* Bp = pack_rows(mb, n0, nn, K, bbuf, ldb);

            for (int64_t m0 = 0; m0 < mrows; m0 += MR) {
                const int64_t mm = (mrows - m0) < MR ? (mrows - m0) : MR;
                float acc[MR][NR];
                for (int i = 0; i < MR; ++i)
                    for (int j = 0; j < NR; ++j) acc[i][j] = 0.0f;

                for (int64_t gi = 0; gi < ngrp; ++gi) {
                    const int64_t k0 = gi * g, k1 = (k0 + g < K) ? k0 + g : K;
                    if (mm == MR && nn == NR) {
                        float sub[MR][NR];
                        for (int i = 0; i < MR; ++i)
                            for (int j = 0; j < NR; ++j) sub[i][j] = 0.0f;
                        dot_block<MR, NR>(Ap + m0 * lda, lda, Bp, ldb, k0, k1, sub);
                        for (int j = 0; j < NR; ++j) {
                            const float sb = (SB && SB->p) ? SB->grp(n0 + j, gi) : 1.0f;
                            for (int i = 0; i < MR; ++i) {
                                const float sa = (SA && SA->p)
                                                     ? SA->grp(mb0 + m0 + i, k0 / SA->group) : 1.0f;
                                acc[i][j] += sub[i][j] * sa * sb;
                            }
                        }
                    } else {
                        for (int64_t j = 0; j < nn; ++j) {
                            const float sb = (SB && SB->p) ? SB->grp(n0 + j, gi) : 1.0f;
                            for (int64_t i = 0; i < mm; ++i) {
                                const float sa = (SA && SA->p)
                                                     ? SA->grp(mb0 + m0 + i, k0 / SA->group) : 1.0f;
                                acc[i][j] += dotf(Ap + (m0 + i) * lda + k0, Bp + j * ldb + k0,
                                                  k1 - k0) * sa * sb;
                            }
                        }
                    }
                }
                for (int64_t i = 0; i < mm; ++i)
                    for (int64_t j = 0; j < nn; ++j) {
                        float v = acc[i][j];
                        if (bias) v += ldt(bias, (n0 + j) * laststride(bias));
                        stt(Y, (mb0 + m0 + i) * ys_r + (n0 + j) * ys_c, v);
                    }
            }
        }
    }
    return RAD_OK;
}

/* gemm_drive over an activation scale grid finer than the weight's. No supported format has one,
 * but libref applies such a scale to each element before the product, so A is scaled into an f32
 * copy here and the product read with no activation scale. The copy is one pass over A. */
static int gemm_drive_fine_a(const Mat& ma, const Mat& mb, const Scales* SA, const Scales* SB,
                             const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N,
                             int64_t K) {
    static thread_local std::vector<float> scaled;
    scaled.resize((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m) {
        float* row = scaled.data() + m * K;
        const float* src = pack_rows(ma, m, 1, K, row, K);
        for (int64_t k = 0; k < K; ++k) row[k] = src[k] * SA->grp(m, k / SA->group);
    }
    RadTensor a{};
    a.data = scaled.data();
    a.dtype = RAD_F32;
    a.rank = 2;
    a.shape[0] = M; a.shape[1] = K;
    a.stride[0] = K; a.stride[1] = 1;
    return gemm_drive(as_mat(&a, K), mb, nullptr, SB, Y, bias, M, N, K);
}

/* A quantised product (gemm_nt_q and its gated and parted forms): the bf16 grouped driver
 * (avx_gemm_tile.cpp) where it applies, which keeps libref's group structure, else gemm_drive. */
static int gemm_q_drive(const Mat& ma, const Mat& mb, const Scales* SA, const Scales* SB,
                        const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N,
                        int64_t K) {
    if (gemm_q_bf16(ma, mb, SA, SB, Y, bias, M, N, K)) return RAD_OK;
    return gemm_drive(ma, mb, SA, SB, Y, bias, M, N, K);
}

/* gemm_nt_bias's epilogue (docs/OPS.md): activation of the stored product, then the residual add,
 * each step read from and stored to the output's dtype (the unfused chain, as in libref). */
static void gemm_epilogue(const RadTensor* Y, const RadTensor* res, int act, int64_t M, int64_t N) {
    const int64_t ys = laststride(Y), yr = Y->rank >= 2 ? Y->stride[Y->rank - 2] : N;
    const int64_t rs = res ? laststride(res) : 0, rr = res && res->rank >= 2 ? res->stride[res->rank - 2] : N;
    AVX_PARALLEL_FOR_IF(M * N >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            const int64_t o = m * yr + n * ys;
            float v = ldt(Y, o);
            if (act == 1) { v = 0.5f * v * (1.0f + sc_erf(v * 0.70710678118654752f)); stt(Y, o, v); }
            else if (act == 2) {
                v = 0.5f * v * (1.0f + sc_tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
                stt(Y, o, v);
            }
            if (res) stt(Y, o, ldt(res, m * rr + n * rs) + ldt(Y, o));
        }
}

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
    const int64_t rs_r = res->rank >= 2 ? res->stride[res->rank - 2] : N, rs_c = laststride(res);
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

/* Shared entry for gemm_nt / gemm_nt_bias / logits_gemm. `nkey`/`kkey` name the N and K
 * parameters (logits_gemm uses n_vocab and n_embd). */
static int gemm_nt_impl(const RadArgs* a, bool bias_operand, const char* nkey, const char* kkey) {
    /* gemm_nt_bias is a, b, bias, res?, y; the other two are a, b, y. */
    const int nt = bias_operand ? 5 : 3;
    if (a->n_t < nt || !a->t[0].data || !a->t[1].data || !a->t[nt - 1].data) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[nt - 1];
    const RadTensor* bias = bias_operand ? rad_arg_in(a, 2) : nullptr;
    const RadTensor* res  = bias_operand ? rad_arg_in(a, 3) : nullptr;
    static const char* kActs[] = { "none", "gelu", "gelu_tanh" };
    const int act = bias_operand ? p_enum(a, "act", kActs, 3, 0) : 0;
    if (act < 0 || act > 2) return RAD_E_INVAL;

    const int64_t K = p_int(a, kkey, B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, nkey, B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / N;
    if (M <= 0) return RAD_E_SHAPE;
    /* A head of codes and scale, joined (avx_planes.h); its extents are its planes'. */
    const bool joined = avxp::is_joined(B);
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
        HeadDec dec;
        int st = avxp::scaled_rows(B, N, K, &dec.sr);
        if (st != RAD_OK) return st;
        dec.K = K;
        /* The codes and their scale grid read as a quantised weight, where the bf16 grouped
         * driver applies (E4M3 or int8 codes against a bf16 activation). */
        RadTensor ct{}, sct{};
        const avxp::JPlane& cp = dec.sr.codes;
        const avxp::JPlane& sp = dec.sr.scale;
        const int64_t ses = rad_dtype_bits(sp.dt) / 8;
        if (ses > 0 && sp.row_bytes % ses == 0) {
            ct.data = (void*)cp.p; ct.dtype = cp.dt; ct.rank = 2;
            ct.shape[0] = N; ct.shape[1] = K; ct.stride[0] = cp.row_bytes; ct.stride[1] = 1;
            sct.data = (void*)sp.p; sct.dtype = sp.dt; sct.rank = 2;
            sct.shape[0] = sp.rows; sct.shape[1] = sp.cols;
            sct.stride[0] = sp.row_bytes / ses; sct.stride[1] = 1;
            const Scales SB = make_scales(&sct, N, K, false);
            if (gemm_q_bf16(ma, as_mat(&ct, K), nullptr, &SB, Y, bias, M, N, K)) {
                if (act || res) gemm_epilogue(Y, res, act, M, N);
                return RAD_OK;
            }
        }
        const Mat mb{ B, K, 1, &dec };
        st = gemm_drive(ma, mb, nullptr, nullptr, Y, bias, M, N, K);
        if (st < 0) return st;
        if (act || res) gemm_epilogue(Y, res, act, M, N);
        return RAD_OK;
    }
    const Mat mb = as_mat(B, K);
    /* bf16 x bf16 takes the grouped driver where the CPU has AVX512_BF16, as one group. */
    if (!gemm_q_bf16(ma, mb, nullptr, nullptr, Y, bias, M, N, K)) {
        const int st = gemm_drive(ma, mb, nullptr, nullptr, Y, bias, M, N, K);
        if (st < 0) return st;
    }
    if (act || res) gemm_epilogue(Y, res, act, M, N);
    return RAD_OK;
}

/* Parts (`pb1`, `pb2`): weight rows [0, pb1), [pb1, pb2) and [pb2, N) each multiply their own
 * copy of the activation, for a fused projection whose pieces were rotated separately (ParoQuant,
 * MXFP4 weights). The copies sit side by side, so `a` is [M, parts * K] and its scale
 * [M, parts * K / 128]; each part is a plain product over its rows. 0 omits a bound: no `pb1` is
 * one part, no `pb2` ends the second at N. libref's reading and refusals (ref_gemm.cpp). */
static int gemm_ntq_parts(const RadTensor* A, const RadTensor* as_, const RadTensor* B,
                          const RadTensor* bs_, const RadTensor* Y, int64_t pb1, int64_t pb2,
                          int64_t M, int64_t N, int64_t K) {
    if (B->dtype != RAD_FP4E2M1 || A->rank < 2) return RAD_E_UNSUPPORTED;
    const int64_t e1 = pb1 > 0 ? std::min<int64_t>(pb1, N) : N;
    const int64_t e2 = pb2 > 0 ? std::max<int64_t>(e1, std::min<int64_t>(pb2, N)) : N;
    const int64_t bound[4] = { 0, e1, e2, N };
    const int64_t acols = A->shape[A->rank - 1];
    const int64_t parts = e2 < N ? 3 : (e1 < N ? 2 : 1);
    if (acols != parts * K || (e1 % 16) || (e2 % 16)) return RAD_E_SHAPE;
    if (!as_ || as_->rank < 2 || !bs_ || bs_->rank < 1) return RAD_E_INVAL;
    const int64_t sa_cols = as_->shape[as_->rank - 1];
    if (sa_cols % parts) return RAD_E_SHAPE;
    const int64_t sa_part = sa_cols / parts;                 /* K / 128 */
    const int64_t bs_cols = bs_->shape[bs_->rank - 1];
    const int64_t ys = row_stride(Y, N);
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
        Bsp.data = (uint8_t*)bs_->data + rad_dtype_bytes(bs_->dtype, n0 * bs_cols);
        Bsp.rank = 2; Bsp.shape[0] = n1 - n0; Bsp.shape[1] = bs_cols;
        Bsp.stride[0] = bs_cols; Bsp.stride[1] = 1;
        Yp.data = (uint8_t*)Y->data + rad_dtype_bytes(Y->dtype, n0);
        Yp.rank = 2; Yp.shape[0] = M; Yp.shape[1] = n1 - n0; Yp.stride[0] = ys; Yp.stride[1] = 1;
        const Scales SAp = make_scales(&Sp, M, K, false);
        const Scales SBp = make_scales(&Bsp, n1 - n0, K, true);
        const int rc = gemm_q_drive(as_mat(&Ap, K), as_mat(&Bp, K), &SAp, &SBp, &Yp, nullptr, M,
                                    n1 - n0, K);
        if (rc != RAD_OK) return rc;
    }
    return RAD_OK;
}

/* ================================================================== the small-M integer path
 *
 * w2a8 and w4a8_asym at decode and draft row counts (M <= 16, a draft head's rows): the weight is
 * read once and never widened. Each 32 bytes of codes split into code planes in registers (four
 * of u2, two of u4, one shift and mask each), and the activation is rearranged once per call into
 * the same plane order, so a plane meets its activation bytes with vpmaddubsw (unsigned code times
 * signed code, pairs summed exactly: at most 2 * 15 * 127) and vpmaddwd into int32. Per row and
 * group of 128 the integer sum is exact; it is widened and scaled into f32 lanes, and the zero is
 * removed per (row, group) through a_sum:
 *
 *   y[m][n] = a_scale[m] * (sum_g scale[n][g] * sum(a * code) - sum_g scale * zero * a_sum[g][m])
 *
 * the arithmetic libref's reference states, in another order. Levels below AVX2 use the decoded
 * driver. At AVX-512 a CPU with VNNI multiplies each plane in one vpdpbusd. */
#if AVX_LEVEL >= 2

/* A joined zero-point grid's scale and zero for (row, group), read directly for the planes the
 * grids are stored in (f16, u8) and through the generic reader otherwise. */
static inline float zp_scale_at(const avxp::JPlane& p, int64_t i, int64_t j) {
    if (p.dt == RAD_F16) return _cvtsh_ss(((const uint16_t*)(p.p + i * p.row_bytes))[j]);
    return p.at(i, j);
}
static inline int32_t zp_zero_at(const avxp::JPlane& p, int64_t i, int64_t j) {
    if (p.dt == RAD_U8) return p.p[i * p.row_bytes + j];
    return (int32_t)p.at(i, j);
}

/* The activation codes in the weight's plane order: per group of 128 and per 32 bytes of codes
 * (`cpb` codes a byte), plane t's byte j meets code cpb * j + t of those 32 bytes' codes. */
static void zp_planes_a(const int8_t* a, int64_t K, int cpb, int8_t* out) {
    const int64_t span = 32 * (int64_t)cpb;              /* codes in 32 bytes */
    for (int64_t c0 = 0; c0 < K; c0 += span)
        for (int t = 0; t < cpb; ++t)
            for (int64_t j = 0; j < 32; ++j) out[c0 + t * 32 + j] = a[c0 + cpb * j + t];
}

/* At AVX-512 the integer path is compiled for VNNI as well, and the VNNI instantiations run only
 * where the CPU reports it; the others use no VNNI instruction. */
#if AVX_LEVEL == 3
#define AVX_ZP_I8_TARGET __attribute__((target("avx512vnni,avx512vl")))
#else
#define AVX_ZP_I8_TARGET
#endif

/* One plane's products into the int32 lanes: vpmaddubsw then vpmaddwd, or with VNNI the one
 * vpdpbusd that does both, exactly (u8 times s8, four products a lane, no saturation either way at
 * these code widths). */
template <bool VNNI>
AVX_ZP_I8_TARGET static inline __m256i zp_dot_plane(__m256i acc, __m256i w, __m256i a,
                                                    __m256i ones) {
#if AVX_LEVEL == 3
    if constexpr (VNNI) return _mm256_dpbusd_epi32(acc, w, a);
#endif
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w, a), ones));
}

template <int MB, int BITS, bool VNNI>
AVX_ZP_I8_TARGET
static void zp_rows_i8(const int8_t* ap, int64_t K, const uint8_t* codes, int64_t rb,
                       const avxp::JPlane& sc, const avxp::JPlane& zp, const float* sa,
                       const float* asum, int64_t asum_ld, int64_t n0, int64_t n1, int64_t m0,
                       const RadTensor* Y, int64_t ys_r, int64_t ys_c) {
    constexpr int CPB = 8 / BITS;                         /* codes a byte */
    constexpr int64_t SPAN = 32 * CPB;                    /* codes a 32-byte chunk */
    const int64_t G = K / 128;
    const __m256i mask = _mm256_set1_epi8((char)((1 << BITS) - 1));
    const __m256i ones = _mm256_set1_epi16(1);
    for (int64_t n = n0; n < n1; ++n) {
        const uint8_t* cr = codes + n * rb;
        __m256 acc[MB];
        for (int i = 0; i < MB; ++i) acc[i] = _mm256_setzero_ps();
        float corr[MB] = {};
        for (int64_t g = 0; g < G; ++g) {
            const float s = zp_scale_at(sc, n, g);
            const float zs = (float)zp_zero_at(zp, n, g) * s;
            const __m256 vs = _mm256_set1_ps(s);
            for (int i = 0; i < MB; ++i) corr[i] += zs * asum[g * asum_ld + m0 + i];
            for (int64_t c0 = g * 128; c0 < (g + 1) * 128; c0 += SPAN) {
                const __m256i x = _mm256_loadu_si256((const __m256i*)(cr + c0 / CPB));
                __m256i w[CPB];
                for (int t = 0; t < CPB; ++t)
                    w[t] = _mm256_and_si256(_mm256_srli_epi16(x, BITS * t), mask);
                for (int i = 0; i < MB; ++i) {
                    const int8_t* ar = ap + (m0 + i) * K + c0;
                    __m256i d = _mm256_setzero_si256();
                    for (int t = 0; t < CPB; ++t)
                        d = zp_dot_plane<VNNI>(
                            d, w[t], _mm256_loadu_si256((const __m256i*)(ar + 32 * t)), ones);
                    acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), vs, acc[i]);
                }
            }
        }
        for (int i = 0; i < MB; ++i) {
            __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[i]),
                                  _mm256_extractf128_ps(acc[i], 1));
            h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            h = _mm_add_ss(h, _mm_movehdup_ps(h));
            stt(Y, (m0 + i) * ys_r + n * ys_c, (_mm_cvtss_f32(h) - corr[i]) * sa[m0 + i]);
        }
    }
}

/* y for rows [0, M) of the integer path; false when it does not apply (M over 16, a group other
 * than 128, codes other than u2 / u4). `a` is column-order int8 [M, K]. */
static bool zp_small_i8(int bits, const int8_t* a, int64_t lda, const float* sa,
                        const RadTensor* asum, int64_t mpad, const uint8_t* codes, int64_t rb,
                        const avxp::JPlane& sc, const avxp::JPlane& zp, int64_t group,
                        const RadTensor* Y, int64_t M, int64_t N, int64_t K) {
    if (M > 16 || group != 128 || K % 128 || (bits != 2 && bits != 4)) return false;
    if (asum->dtype != RAD_F32 || laststride(asum) != 1) return false;
    const int cpb = 8 / bits;
    static thread_local std::vector<int8_t> ap;
    ap.resize((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m) zp_planes_a(a + m * lda, K, cpb, ap.data() + m * K);
    const float* as = (const float*)asum->data;
    const int64_t ys_r = row_stride(Y, N), ys_c = laststride(Y);
    /* Rows a thread takes at a time: enough to amortise the scheduling, few enough to spread a
     * 5120-row projection over every core. */
    const int64_t NB = 16, nblk = (N + NB - 1) / NB;
    using Fn = void (*)(const int8_t*, int64_t, const uint8_t*, int64_t, const avxp::JPlane&,
                        const avxp::JPlane&, const float*, const float*, int64_t, int64_t,
                        int64_t, int64_t, const RadTensor*, int64_t, int64_t);
    static const Fn kFn[2][2][4] = {
        { { zp_rows_i8<1, 2, false>, zp_rows_i8<2, 2, false>, zp_rows_i8<4, 2, false>,
            zp_rows_i8<8, 2, false> },
          { zp_rows_i8<1, 4, false>, zp_rows_i8<2, 4, false>, zp_rows_i8<4, 4, false>,
            zp_rows_i8<8, 4, false> } },
        { { zp_rows_i8<1, 2, true>, zp_rows_i8<2, 2, true>, zp_rows_i8<4, 2, true>,
            zp_rows_i8<8, 2, true> },
          { zp_rows_i8<1, 4, true>, zp_rows_i8<2, 4, true>, zp_rows_i8<4, 4, true>,
            zp_rows_i8<8, 4, true> } },
    };
    /* AVX512-VNNI is not in the level's flags (many AVX-512 parts lack it) and is checked here. */
#if AVX_LEVEL == 3
    static const bool vnni = __builtin_cpu_supports("avx512vnni");
#else
    static const bool vnni = false;
#endif
    const Fn* fn = kFn[vnni ? 1 : 0][bits == 4 ? 1 : 0];
    /* Taken here: inside the region `ap` names each worker's own (empty) thread_local. */
    const int8_t* apd = ap.data();
    AVX_PARALLEL_FOR_IF(M * N * K >= 65536)
    for (int64_t b = 0; b < nblk; ++b) {
        const int64_t n0 = b * NB, n1 = n0 + NB < N ? n0 + NB : N;
        for (int64_t m0 = 0; m0 < M;) {
            const int64_t left = M - m0;
            const int lv = left >= 8 ? 3 : left >= 4 ? 2 : left >= 2 ? 1 : 0;
            fn[lv](apd, K, codes, rb, sc, zp, sa, as, mpad, n0, n1, m0, Y, ys_r, ys_c);
            m0 += (int64_t)1 << lv;
        }
    }
    return true;
}
#endif

/* ================================================================== the small-M 16-bit path
 *
 * w4a16_asym and w5a16 at decode and verify row counts (M <= 16): every weight row decodes in
 * registers, sixteen codes at a time, to (code - zero) * scale, and meets each activation row
 * (widened to f32 once per call) with one FMA, so the weight is read once and never stored widened.
 * The decoded driver writes each row to a packing buffer and reads it back, a pass through L2 this
 * path does not make. u4 is split by mask and shift, u5 by pdep (BMI2, checked at run time: the
 * path is taken only where the CPU has it). */
#if AVX_LEVEL >= 2

/* Sixteen codes of a row's 4- or 5-bit stream from element k (a multiple of 16), one a byte. In the
 * last 16 codes of a row the u5 read takes only the bytes asked for. */
template <int BITS>
__attribute__((target("bmi2"))) static inline __m128i zp_codes16(const uint8_t* row, int64_t k,
                                                                 bool last) {
    if (BITS == 4) {
        const __m128i x = _mm_loadl_epi64((const __m128i*)(row + k / 2));
        const __m128i m4 = _mm_set1_epi8(0x0f);
        return _mm_unpacklo_epi8(_mm_and_si128(x, m4), _mm_and_si128(_mm_srli_epi16(x, 4), m4));
    }
    const uint8_t* p = row + k / 8 * 5;
    uint64_t v0 = 0, v1 = 0;
    if (!last) {
        std::memcpy(&v0, p, 8);
        std::memcpy(&v1, p + 5, 8);
        v0 &= 0xffffffffffull;
        v1 &= 0xffffffffffull;
    } else {
        std::memcpy(&v0, p, 5);
        std::memcpy(&v1, p + 5, 5);
    }
    const unsigned long long kLow5 = 0x1f1f1f1f1f1f1f1full;
    return _mm_set_epi64x((long long)_pdep_u64(v1, kLow5), (long long)_pdep_u64(v0, kLow5));
}

/* Rows [n0, n1) (at most 16) against activation rows [m0, m0 + MB). K is walked in chunks of
 * whole groups, about 512 columns, and within a chunk every weight row of the block: the chunk of
 * activation rows (MB x 512 floats, 16 KiB at MB = 8) stays in L1 while the block's rows stream
 * past it, where a row at a time would read every activation row from L2 per weight row. The
 * accumulators wait in `accb` between chunks. U independent accumulators a row (4 at MB = 1, 2 at
 * MB = 2) cover the FMA latency a single dependent chain would wait on. NR weight rows (2 at
 * AVX-512 from MB = 4, whose 32 registers hold both rows' accumulators) decode side by side and
 * share each activation load. A block never straddles a part: parts end on multiples of 16. */
template <int MB, int BITS, int U, int NR>
__attribute__((target("bmi2"))) static void zp_rows_f16(
        const float* af, int64_t lda, int64_t K, const uint8_t* codes, int64_t rb,
        const avxp::JPlane& sc, const avxp::JPlane& zp, int64_t group, int64_t n0, int64_t n1,
        int64_t m0, int64_t e1, int64_t e2, const RadTensor* Y, int64_t ys_r, int64_t ys_c) {
#if AVX_LEVEL == 3
    using V = __m512;
    using VI = __m512i;
#else
    using V = __m256;
    using VI = __m256i;
#endif
    constexpr int64_t kNB = 16;
    alignas(64) V accb[kNB][MB * U];
    for (int64_t r = 0; r < n1 - n0; ++r)
        for (int i = 0; i < MB * U; ++i) accb[r][i] = (V){};
    const int64_t kc = group * (group >= 512 ? 1 : 512 / group);
    const float* a0 = af + m0 * lda + (n0 < e1 ? 0 : n0 < e2 ? 1 : 2) * K;
    for (int64_t c0 = 0; c0 < K; c0 += kc) {
        const int64_t c1 = c0 + kc < K ? c0 + kc : K;
        for (int64_t n = n0; n < n1; n += NR) {
            const int nr = n + NR <= n1 ? NR : 1;
            const uint8_t* cr[NR];
            V acc[NR][MB * U];
            for (int r = 0; r < NR; ++r) {
                cr[r] = codes + (n + (r < nr ? r : 0)) * rb;
                for (int i = 0; i < MB * U; ++i) acc[r][i] = accb[n - n0 + (r < nr ? r : 0)][i];
            }
            for (int64_t g = c0 / group; g < c1 / group; ++g) {
                VI vz[NR];
                V vs[NR];
                for (int r = 0; r < NR; ++r) {
                    const int64_t row = n + (r < nr ? r : 0);
#if AVX_LEVEL == 3
                    vz[r] = _mm512_set1_epi32(zp_zero_at(zp, row, g));
                    vs[r] = _mm512_set1_ps(zp_scale_at(sc, row, g));
#else
                    vz[r] = _mm256_set1_epi32(zp_zero_at(zp, row, g));
                    vs[r] = _mm256_set1_ps(zp_scale_at(sc, row, g));
#endif
                }
                for (int64_t k0 = g * group; k0 < (g + 1) * group; k0 += 16 * U)
                    for (int u = 0; u < U; ++u) {
                        const int64_t k = k0 + 16 * u;
#if AVX_LEVEL == 3
                        __m512 w[NR];
                        for (int r = 0; r < NR; ++r) {
                            const __m128i b = zp_codes16<BITS>(cr[r], k, k + 16 == K);
                            w[r] = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(
                                                     _mm512_cvtepu8_epi32(b), vz[r])), vs[r]);
                        }
                        for (int i = 0; i < MB; ++i) {
                            const __m512 a = _mm512_loadu_ps(a0 + i * lda + k);
                            for (int r = 0; r < NR; ++r)
                                acc[r][i * U + u] = _mm512_fmadd_ps(a, w[r], acc[r][i * U + u]);
                        }
#else
                        __m256 w0[NR], w1[NR];
                        for (int r = 0; r < NR; ++r) {
                            const __m128i b = zp_codes16<BITS>(cr[r], k, k + 16 == K);
                            w0[r] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(
                                                      _mm256_cvtepu8_epi32(b), vz[r])), vs[r]);
                            w1[r] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(
                                                      _mm256_cvtepu8_epi32(_mm_srli_si128(b, 8)),
                                                      vz[r])), vs[r]);
                        }
                        for (int i = 0; i < MB; ++i) {
                            const __m256 a = _mm256_loadu_ps(a0 + i * lda + k);
                            const __m256 b8 = _mm256_loadu_ps(a0 + i * lda + k + 8);
                            for (int r = 0; r < NR; ++r) {
                                V& c = acc[r][i * U + u];
                                c = _mm256_fmadd_ps(a, w0[r], c);
                                c = _mm256_fmadd_ps(b8, w1[r], c);
                            }
                        }
#endif
                    }
            }
            for (int r = 0; r < nr; ++r)
                for (int i = 0; i < MB * U; ++i) accb[n - n0 + r][i] = acc[r][i];
        }
    }
    for (int64_t n = n0; n < n1; ++n)
        for (int i = 0; i < MB; ++i) {
            V a = accb[n - n0][i * U];
            for (int u = 1; u < U; ++u) a = a + accb[n - n0][i * U + u];
#if AVX_LEVEL == 3
            const float v = _mm512_reduce_add_ps(a);
#else
            __m128 h = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
            h = _mm_add_ps(h, _mm_movehl_ps(h, h));
            h = _mm_add_ss(h, _mm_movehdup_ps(h));
            const float v = _mm_cvtss_f32(h);
#endif
            stt(Y, (m0 + i) * ys_r + n * ys_c, v);
        }
}

/* y over the rows of every part; false when the path does not apply (no BMI2, M over 16, a group
 * that is not whole 64s, codes other than u4 / u5). `af` is the activation as f32 [M, lda]. */
static bool zp_small_f16(int bits, const float* af, int64_t lda, const uint8_t* codes, int64_t rb,
                         const avxp::JPlane& sc, const avxp::JPlane& zp, int64_t group,
                         int64_t e1, int64_t e2, const RadTensor* Y, int64_t M, int64_t N,
                         int64_t K) {
    static const bool bmi2 = __builtin_cpu_supports("bmi2");
    if (!bmi2 || M > 16 || group % 64 || K % group || (bits != 4 && bits != 5)) return false;
    using Fn = void (*)(const float*, int64_t, int64_t, const uint8_t*, int64_t,
                        const avxp::JPlane&, const avxp::JPlane&, int64_t, int64_t, int64_t,
                        int64_t, int64_t, int64_t, const RadTensor*, int64_t, int64_t);
#if AVX_LEVEL == 3
    constexpr int kNR = 2;
#else
    constexpr int kNR = 1;
#endif
    static const Fn kFn[2][4] = {
        { zp_rows_f16<1, 4, 4, 1>, zp_rows_f16<2, 4, 2, 1>, zp_rows_f16<4, 4, 1, kNR>,
          zp_rows_f16<8, 4, 1, kNR> },
        { zp_rows_f16<1, 5, 4, 1>, zp_rows_f16<2, 5, 2, 1>, zp_rows_f16<4, 5, 1, kNR>,
          zp_rows_f16<8, 5, 1, kNR> },
    };
    const Fn* fn = kFn[bits == 5 ? 1 : 0];
    const int64_t ys_r = row_stride(Y, N), ys_c = laststride(Y);
    const int64_t NB = 16, nblk = (N + NB - 1) / NB;
    AVX_PARALLEL_FOR_IF(M * N * K >= 65536)
    for (int64_t b = 0; b < nblk; ++b) {
        const int64_t n0 = b * NB, n1 = n0 + NB < N ? n0 + NB : N;
        for (int64_t m0 = 0; m0 < M;) {
            const int64_t left = M - m0;
            const int lv = left >= 8 ? 3 : left >= 4 ? 2 : left >= 2 ? 1 : 0;
            fn[lv](af, lda, K, codes, rb, sc, zp, group, n0, n1, m0, e1, e2, Y, ys_r, ys_c);
            m0 += (int64_t)1 << lv;
        }
    }
    return true;
}
#endif

/* gemm_nt_q over a zero-point grid (avx_planes.h: ZpGrid), with libref's checks and refusals
 * (libref/ref_gemm.cpp: gemm_zp). Each part of the weight is one plain product over decoded rows
 * (ZpDec), against an f32 activation that carries what the grid scales by:
 *
 *   ZP_F16       the activation as it is; the rows are (code - zero) * scale
 *   ZP_I8_GROUP  code * a_scale[m][g], against (code - zero) * scale
 *   ZP_I8_ROW    code * a_scale[m] with a_sum[g][m] * a_scale[m] after each group, against
 *                code * scale and -zero * scale after each group
 *
 * so every output is a sum over k of a'(k) * w'(k) in one group, which the driver blocks over M
 * and N. `a` arrives in column order (the caller undoes the step order). */
static int gemm_zp(const avxp::ZpGrid& zg, const RadTensor* A, const RadTensor* as_,
                   const RadTensor* B, const RadTensor* bs_, const RadTensor* asum,
                   const RadTensor* Y, int64_t M, int64_t N, int64_t K, int64_t group,
                   int64_t pb1, int64_t pb2) {
    using namespace avxp;
    if (!is_joined(bs_)) return RAD_E_DTYPE;
    const JPlane sc = jplane(bs_, "scale"), zp = jplane(bs_, "zero");
    if (!sc || !zp || !float_scale_dt(sc.dt) || !zero_dt(zp.dt)) return RAD_E_DTYPE;
    if (B->dtype != zg.codes) return RAD_E_DTYPE;
    if (group <= 0 || K % group) return RAD_E_SHAPE;
    const int64_t G = K / group;
    if (sc.rows != N || sc.cols != G || zp.rows != N || zp.cols != G) return RAD_E_SHAPE;
    if (!rad_tensor_is_contiguous(B)) return RAD_E_STRIDE;

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
                narrowed[(size_t)(m * acols + k)] = rad_f32_to_f16(ldt(A, m * ars + k * acs));
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
    const int64_t ys = row_stride(Y, N), yc = laststride(Y);
    const bool aug = zg.act == ZP_I8_ROW;
    const int64_t Kw = aug ? G * (group + 1) : K;
#if AVX_LEVEL >= 2
    if (zg.act == ZP_F16 && acs == 1 && M <= 16) {
        static thread_local std::vector<float> afv;
        afv.resize((size_t)(M * acols));
        for (int64_t m = 0; m < M; ++m)
            row_to_f32(A->dtype, (const uint8_t*)A->data + rad_dtype_bytes(A->dtype, m * ars),
                       afv.data() + m * acols, acols);
        if (zp_small_f16(rad_dtype_bits(B->dtype), afv.data(), acols, (const uint8_t*)B->data,
                         rad_enc_row_bytes(B->dtype, K), sc, zp, group, e1, e2, Y, M, N, K))
            return RAD_OK;
    }
    if (aug && acs == 1) {
        static thread_local std::vector<float> sav;
        sav.resize((size_t)M);
        for (int64_t m = 0; m < M; ++m) sav[(size_t)m] = ldt(as_, m * sa_rs);
        if (zp_small_i8(rad_dtype_bits(B->dtype), (const int8_t*)A->data, ars, sav.data(), asum,
                        mpad, (const uint8_t*)B->data, rad_enc_row_bytes(B->dtype, K), sc, zp,
                        group, Y, M, N, K))
            return RAD_OK;
    }
#endif

    /* The int8-activation grids on the bf16 grouped driver (avx_gemm_tile.cpp): code - zero and
     * the int8 codes are exact in bf16, so each group's sub-sum is libref's exact integer, then
     * (sub * a_scale) * scale. One job a part, each part against its own copy of the activation;
     * where a part is declined, the decoded driver below computes all of Y. */
    if (zg.act == ZP_I8_GROUP && acs == 1 && zp.dt == RAD_U8 &&
        rad_enc_row_bytes(B->dtype, K) * 8 == K * rad_dtype_bits(B->dtype)) {
        RadTensor ct{}, st{};
        ct.data = B->data; ct.dtype = B->dtype; ct.rank = 2;
        ct.shape[0] = N; ct.shape[1] = K; ct.stride[0] = K; ct.stride[1] = 1;
        const int64_t ses = rad_dtype_bits(sc.dt) / 8;
        st.data = (void*)sc.p; st.dtype = sc.dt; st.rank = 2;
        st.shape[0] = N; st.shape[1] = G; st.stride[0] = sc.row_bytes / ses; st.stride[1] = 1;
        const Scales SBz = make_scales(&st, N, K, false);
        const int64_t bound[4] = { 0, e1, e2, N };
        bool ok = true;
        for (int p = 0; p < 3 && ok; ++p) {
            const int64_t n0 = bound[p], n1 = bound[p + 1];
            if (n1 <= n0) continue;
            RadTensor Ap = *A, Sp = *as_;
            Ap.data = (uint8_t*)A->data + p * K * acs;
            Ap.rank = 2; Ap.shape[0] = M; Ap.shape[1] = K; Ap.stride[0] = ars; Ap.stride[1] = 1;
            Sp.data = (uint8_t*)as_->data + rad_dtype_bytes(as_->dtype, p * G * sa_cs);
            Sp.rank = 2; Sp.shape[0] = M; Sp.shape[1] = G; Sp.stride[0] = sa_rs;
            Sp.stride[1] = sa_cs;
            const Scales SAz = make_scales(&Sp, M, K, false);
            TileJob jb{};
            jb.mb = as_mat(&ct, K);
            jb.SB = SBz;
            jb.b0 = n0; jb.a0 = 0; jb.M = M; jb.N = n1 - n0; jb.K = K;
            jb.y0 = 0; jb.yc0 = n0;
            jb.zero = zp;
            ok = gemm_q_bf16_jobs(as_mat(&Ap, K), nullptr, M, &SAz, &jb, 1, Y, N);
        }
        if (ok) return RAD_OK;
    }

    ZpDec dec;
    dec.codes_p = (const uint8_t*)B->data;
    dec.rb = rad_enc_row_bytes(B->dtype, K);
    dec.K = K;
    dec.group = group;
    dec.bits = rad_dtype_bits(B->dtype);
    dec.aug = aug;
    dec.sc = sc;
    dec.zp = zp;

    static thread_local std::vector<float> aw;
    const int64_t bound[4] = { 0, e1, e2, N };
    for (int p = 0; p < 3; ++p) {
        const int64_t n0 = bound[p], n1 = bound[p + 1];
        if (n1 <= n0) continue;
        dec.n0 = n0;
        RadTensor Ap{}, Yp = *Y;
        Yp.data = (uint8_t*)Y->data + rad_dtype_bytes(Y->dtype, n0 * yc);
        Yp.rank = 2; Yp.shape[0] = M; Yp.shape[1] = n1 - n0; Yp.stride[0] = ys; Yp.stride[1] = yc;
        if (i8) {
            /* The scaled activation, f32 [M, Kw], which the driver reads in place. */
            aw.resize((size_t)(M * Kw));
            const int8_t* ar = (const int8_t*)A->data;
            for (int64_t m = 0; m < M; ++m) {
                float* d = aw.data() + m * Kw;
                const int64_t ab = m * ars + p * K * acs;
                for (int64_t g = 0; g < G; ++g) {
                    const float sa = zg.act == ZP_I8_ROW
                                         ? ldt(as_, m * sa_rs)
                                         : ldt(as_, m * sa_rs + (p * G + g) * sa_cs);
                    float* dg = d + g * (aug ? group + 1 : group);
                    for (int64_t k = 0; k < group; ++k)
                        dg[k] = (float)ar[ab + (g * group + k) * acs] * sa;
                    if (aug) dg[group] = ldt(asum, (g * mpad + m) * laststride(asum)) * sa;
                }
            }
            Ap.data = aw.data();
            Ap.dtype = RAD_F32;
            Ap.rank = 2; Ap.shape[0] = M; Ap.shape[1] = Kw; Ap.stride[0] = Kw; Ap.stride[1] = 1;
        } else {
            Ap = *A;
            Ap.data = (uint8_t*)A->data + rad_dtype_bytes(A->dtype, p * K * acs);
            Ap.rank = 2; Ap.shape[0] = M; Ap.shape[1] = K; Ap.stride[0] = ars; Ap.stride[1] = acs;
        }
        const Mat mb{ B, Kw, 1, &dec };
        const int rc = gemm_drive(as_mat(&Ap, Kw), mb, nullptr, nullptr, &Yp, nullptr, M,
                                  n1 - n0, Kw);
        if (rc != RAD_OK) return rc;
    }
    return RAD_OK;
}

static int gemm_ntq_impl(const RadArgs* a, bool bias_operand) {
    const int nt = bias_operand ? 6 : 5;
    if (a->n_t < nt || !a->t[0].data || !a->t[2].data || !a->t[nt - 1].data) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[2], *Y = &a->t[nt - 1];
    const RadTensor* as_ = rad_arg_in(a, 1);
    const RadTensor* bs_ = rad_arg_in(a, 3);
    const RadTensor* bias = bias_operand ? rad_arg_in(a, 4) : nullptr;
    /* libref's refusals (ref_gemm.cpp): the zero-point grids have no bias form, `a_sum` belongs
     * to the grids that remove a zero through it, `b_ref` to an e2m1 weight. */
    const avxp::ZpGrid* zg = avxp::zp_grid(p_str(a, "dtype", nullptr));
    if (zg && bias_operand) return RAD_E_UNSUPPORTED;
    if (!bias_operand) {
        if (rad_arg_in(a, 5) && (!zg || zg->act != avxp::ZP_I8_ROW)) return RAD_E_UNSUPPORTED;
        if (rad_arg_in(a, 6) && B->dtype != RAD_FP4E2M1) return RAD_E_UNSUPPORTED;
    }

    const int64_t K = p_int(a, "K", B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, "N", B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / N;
    if (M <= 0) return RAD_E_SHAPE;
    if (numel(A) < M * K || numel(B) < N * K) return RAD_E_SHAPE;
    if (bias && numel(bias) < N) return RAD_E_SHAPE;

    /* quant_act_i8's codes arrive with the middle two dwords of every 16-byte step exchanged;
     * exchange them back into a contiguous copy. */
    RadTensor Ac;
    if (A->dtype == RAD_I8 && i8_steps_dtype(p_str(a, "dtype", nullptr))) {
        if (K % 16) return RAD_E_SHAPE;
        if (!dense_row(A)) return RAD_E_STRIDE;
        static thread_local std::vector<int8_t> cols;
        cols.resize((size_t)(M * K));
        for (int64_t m = 0; m < M; ++m) {
            const int8_t* row = (const int8_t*)A->data + rowoff(A, m, K);
            int8_t* dst = cols.data() + m * K;
            for (int64_t k = 0; k < K; k += 16) {
                std::memcpy(dst + k,      row + k,      4);
                std::memcpy(dst + k + 4,  row + k + 8,  4);
                std::memcpy(dst + k + 8,  row + k + 4,  4);
                std::memcpy(dst + k + 12, row + k + 12, 4);
            }
        }
        Ac = *A;
        Ac.data = cols.data();
        Ac.rank = 2; Ac.shape[0] = M; Ac.shape[1] = K; Ac.stride[0] = K; Ac.stride[1] = 1;
        A = &Ac;
    }

    const int64_t pb1 = p_int(a, "pb1", 0), pb2 = p_int(a, "pb2", 0);
    if (zg)
        return gemm_zp(*zg, A, as_, B, bs_, rad_arg_in(a, 5), Y, M, N, K, p_int(a, "group", 0),
                       pb1, pb2);
    /* A scale in two levels, joined (avx_layout_gemm_q): an integer per (row, group) under a float
     * a row. Their product in f32 is the scale the loop below reads, as libref forms it. */
    std::vector<float> two_level;
    RadTensor bs_two;
    if (avxp::is_joined(bs_)) {
        const avxp::JPlane s1 = avxp::jplane(bs_, "scale"), s2 = avxp::jplane(bs_, "scale.1");
        if (!s1 || !s2 || s1.dt != RAD_U8 || !avxp::float_scale_dt(s2.dt)) return RAD_E_DTYPE;
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
        if (bias_operand) return RAD_E_UNSUPPORTED;
        return gemm_ntq_parts(A, as_, B, bs_, Y, pb1, pb2, M, N, K);
    }

    const bool mx = B->dtype == RAD_FP4E2M1;
    Scales SA = make_scales(as_, M, K, false);
    Scales SB = make_scales(bs_, N, K, mx);
    /* `group` is what the selector matches on; the loop uses the group the scale stream implies.
     * A disagreement is refused. */
    const int64_t gp = p_int(a, "group", 0);
    if (gp > 0 && SB.p && SB.group != gp) return RAD_E_SHAPE;

    const Mat ma = as_mat(A, K), mb = as_mat(B, K);
    return gemm_q_drive(ma, mb, &SA, &SB, Y, bias, M, N, K);
}

AVX_KERNEL(gemm_nt)        { return gemm_nt_impl(a, false, "N", "K"); }
AVX_KERNEL(gemm_nt_bias)   { return gemm_nt_impl(a, true,  "N", "K"); }
AVX_KERNEL(gemm_nt_q)      { return gemm_ntq_impl(a, false); }
AVX_KERNEL(gemm_nt_q_bias) { return gemm_ntq_impl(a, true); }

/* ------------------------------------------------------------------ gemm_nt_q_gated */
/* gemm_nt_q over an [N, K] gate_up weight (N is twice the fused width, gate half first), with
 * gated_quant_fp8 as the epilogue; output [M, N/2].
 *
 * Byte-identical to gemm_nt_q into a bf16 [M, N] tensor followed by gated_quant_fp8: same driver
 * and blocking, both halves narrowed to bf16 with the ABI's converter (as gemm_nt_q's bf16 store
 * does), then avx_had.h's gate and codes.
 *
 * The f32 plane between the two is materialised whole in a thread-kept buffer, since the driver
 * parallelises over columns and the epilogue over rows. `act` is required here (the quantiser
 * defaults it) so an implementation that compiles one gate cannot match a call for the other. */
AVX_KERNEL(gemm_nt_q_gated) {
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[2], *q = &a->t[4], *s = &a->t[5];
    const RadTensor* as_ = rad_arg_in(a, 1);
    const RadTensor* bs_ = rad_arg_in(a, 3);
    const RadTensor* ob  = rad_arg_in(a, 6);
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

    static thread_local std::vector<float> plane;
    if (plane.size() < (size_t)(M * N)) plane.resize((size_t)(M * N));
    RadTensor Y{};
    Y.data = plane.data();
    Y.dtype = RAD_F32;
    Y.rank = 2;
    Y.shape[0] = M; Y.shape[1] = N;
    Y.stride[0] = N; Y.stride[1] = 1;
    const int rc = gemm_q_drive(ma, mb, &SA, &SB, &Y, nullptr, M, N, K);
    if (rc != RAD_OK) return rc;

    const float* P = plane.data();
    AVX_PARALLEL_FOR_IF(M * N >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* row = scratch_f32((size_t)N);
        float* out = scratch_f32_b((size_t)n);
        float* tmp = scratch_f32_c((size_t)n);
        for (int64_t j = 0; j < N; ++j) row[j] = rad_bf16_to_f32(rad_f32_to_bf16(P[m * N + j]));
        gate_bf16(row, row + n, out, n, act, tmp);
        fp8_emit(out, tmp, n, g, q, s, ob, m);
    }
    return RAD_OK;
}

/* gemm_nt's arithmetic as a separate op: lm_head has its own access class (RAD_ACCESS_VOCAB) and
 * the sampler fuses against it, so the two must be selectable apart. N and K are n_vocab and
 * n_embd. */
AVX_KERNEL(logits_gemm) { return gemm_nt_impl(a, false, "n_vocab", "n_embd"); }

/* gemm_nt's product for this plugin's composite kernels (the image convolution, avx_gen.cpp), over
 * operands they assembled: Y = A B^T + bias, f32 accumulation, one rounding. Runs on the calling
 * thread's team like gemm_nt and uses the same scratch arenas, so a caller keeps its own buffers
 * elsewhere. */
namespace avx {
inline namespace AVX_SUFFIX {
int gemm_nt_into(const RadTensor* A, const RadTensor* B, const RadTensor* bias, const RadTensor* Y,
                 int64_t M, int64_t N, int64_t K) {
    const Mat ma = as_mat(A, K), mb = as_mat(B, K);
    if (gemm_q_bf16(ma, mb, nullptr, nullptr, Y, bias, M, N, K)) return RAD_OK;
    return gemm_drive(ma, mb, nullptr, nullptr, Y, bias, M, N, K);
}
}  /* inline namespace AVX_SUFFIX */
}  /* namespace avx */

/* ================================================================== the vocabulary edges
 *
 * Embedding gather. A negative token id writes a zero row (padding). `vocab_offset` makes the table
 * this rank's slice of a vocab-parallel embedding: ids stay global, ids outside the slice write
 * zero rows, and summing the ranks reconstructs x.
 *
 * When this rank holds the whole vocabulary, an id >= n_vocab fails the launch (a caller bug).
 * Under a split, rank 0 included, the kernel cannot know the global vocabulary, so bounds are the
 * caller's responsibility. */
template <bool QUANT>
static int embed_impl(const RadArgs* a) {
    const int nt = QUANT ? 4 : 3;
    /* embed_lookup_q's scale (operand 2) is optional; the rest are required. */
    for (int i = 0; i < nt; ++i)
        if (!(QUANT && i == 2) && !rad_arg_in(a, i)) return RAD_E_INVAL;
    const RadTensor* tok = &a->t[0];
    const RadTensor* wte = &a->t[1];
    const RadTensor* sc  = QUANT ? &a->t[2] : nullptr;
    const RadTensor* x   = &a->t[nt - 1];
    const int64_t n_embd = p_int(a, "n_embd", wte->rank >= 2 ? wte->shape[wte->rank - 1] : 0);
    const int64_t n_vocab = p_int(a, "n_vocab", wte->rank >= 2 ? wte->shape[wte->rank - 2] : 0);
    const int64_t voff = p_int(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / n_embd;
    if (M <= 0 || numel(tok) < M || numel(wte) < n_vocab * n_embd) return RAD_E_SHAPE;
    /* An absent scale is 1 (see ref_embed_lookup_q). embed_lookup's `mult` is the same multiply
     * as a parameter. */
    const float scale = QUANT ? (sc->data && numel(sc) >= 1 ? ldt(sc, 0) : 1.0f)
                              : p_f32(a, "mult", 1.0f);

    /* Validated before the loop; an OpenMP body cannot return. */
    if (voff == 0 && a->world_size <= 1)
        for (int64_t m = 0; m < M; ++m)
            if (ldi(tok, offlin(tok, m)) >= n_vocab) return RAD_E_INVAL;

    const int64_t ws = wte->rank >= 2 ? wte->stride[wte->rank - 2] : n_embd;
    const int64_t wc = wte->rank >= 2 ? wte->stride[wte->rank - 1] : 1;
    /* Byte copy when nothing changes (same dtype, no scale), the usual case for the unquantised
     * gather. Whole-byte dtypes only: a sub-byte row can start mid-byte. */
    const bool copy = !QUANT && scale == 1.0f && wte->dtype == x->dtype && wc == 1 &&
                      laststride(x) == 1 && rad_dtype_bits(x->dtype) % 8 == 0;
    const size_t rowb = copy ? (size_t)((rad_dtype_bits(x->dtype) * n_embd) / 8) : 0;

    AVX_PARALLEL_FOR_IF(M * n_embd >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t id = ldi(tok, offlin(tok, m)) - voff;
        const int64_t xb = rowoff(x, m, n_embd);
        const bool have_row = id >= 0 && id < n_vocab;
        if (copy) {
            void* dst = (void*)byte_at(x->data, x->dtype, xb);
            if (have_row) std::memcpy(dst, byte_at(wte->data, wte->dtype, id * ws), rowb);
            else          std::memset(dst, 0, rowb);
            continue;
        }
        float* s = scratch_f32((size_t)n_embd);
        if (!have_row) {
            for (int64_t i = 0; i < n_embd; ++i) s[i] = 0.0f;
        } else if (wc == 1) {
            row_to_f32(wte->dtype, byte_at(wte->data, wte->dtype, id * ws), s, n_embd);
            if (scale != 1.0f) {
                const vf vs = vf_set1(scale);
                int64_t i = 0;
                for (; i + VF_N <= n_embd; i += VF_N) vf_storeu(s + i, vf_mul(vf_loadu(s + i), vs));
                for (; i < n_embd; ++i) s[i] *= scale;
            }
        } else {
            for (int64_t i = 0; i < n_embd; ++i)
                s[i] = ld_elem(wte->data, wte->dtype, id * ws + i * wc) * scale;
        }
        out_row(x, xb, n_embd, s);
    }
    return RAD_OK;
}

AVX_KERNEL(embed_lookup)   { return embed_impl<false>(a); }

/* ================================================================== the E4M3 table into bf16
 *
 * The served form (E4M3 table, bf16 activation) gathers rows that are usually not in memory; they
 * are fetched by avx_ngram.cpp (shared row cache, direct reads for misses, a reader one step
 * ahead). This level only decodes a row of codes to bf16 through the call's 256-entry table.
 *
 * At AVX-512 the table (256 bf16, 512 bytes) lives in eight registers. vpermt2w indexes 64 words
 * across a register pair, so four of them look up each code in its quarter and the code's top two
 * bits select the quarter: about a dozen instructions per 32 codes. Lower levels have no
 * in-register form (vpshufb indexes 16 bytes; a gather is one load per lane, see README.md) and
 * look up one code at a time.
 *
 * Every row is a cache miss, so each row's lines are prefetched kAhead rows before its decode. */
static void decode_e4m3_rows(const uint8_t* const* src, uint16_t* const* dst, int64_t count,
                             int64_t n, const uint16_t* lut) {
    constexpr int64_t kAhead = 8;
#if AVX_LEVEL >= 3
    __m512i t[8];
    for (int k = 0; k < 8; ++k) t[k] = _mm512_loadu_si512((const void*)(lut + 32 * k));
    const __m512i bit6 = _mm512_set1_epi16(0x40), bit7 = _mm512_set1_epi16(0x80);
    auto pick = [&](__m512i ix) {
        const __m512i  q0 = _mm512_permutex2var_epi16(t[0], ix, t[1]);
        const __m512i  q1 = _mm512_permutex2var_epi16(t[2], ix, t[3]);
        const __m512i  q2 = _mm512_permutex2var_epi16(t[4], ix, t[5]);
        const __m512i  q3 = _mm512_permutex2var_epi16(t[6], ix, t[7]);
        const __mmask32 h6 = _mm512_test_epi16_mask(ix, bit6);
        const __mmask32 h7 = _mm512_test_epi16_mask(ix, bit7);
        return _mm512_mask_blend_epi16(h7, _mm512_mask_blend_epi16(h6, q0, q1),
                                       _mm512_mask_blend_epi16(h6, q2, q3));
    };
#endif
    for (int64_t r = 0; r < count; ++r) {
        if (r + kAhead < count) {
            const uintptr_t p = (uintptr_t)src[r + kAhead];
            for (uintptr_t q = p & ~(uintptr_t)63; q < p + (uintptr_t)n; q += 64)
                __builtin_prefetch((const void*)q);
        }
        const uint8_t* s = src[r];
        uint16_t*      d = dst[r];
#if AVX_LEVEL >= 3
        int64_t i = 0;
        for (; i + 32 <= n; i += 32) {
            const __m512i ix = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(s + i)));
            _mm512_storeu_si512((void*)(d + i), pick(ix));
        }
        if (i < n) {
            const __mmask32 k = (__mmask32)((1u << (n - i)) - 1u);
            const __m512i ix = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(k, s + i));
            _mm512_mask_storeu_epi16((void*)(d + i), k, pick(ix));
        }
#else
        for (int64_t i = 0; i < n; ++i) d[i] = lut[s[i]];
#endif
    }
}

/* A bf16 table's rows are the output: the decode is a copy. */
static void decode_bf16_rows(const uint8_t* const* src, uint16_t* const* dst, int64_t count,
                             int64_t n, const uint16_t*) {
    for (int64_t r = 0; r < count; ++r) std::memcpy(dst[r], src[r], (size_t)n * sizeof(uint16_t));
}

/* Fills `g` for avx_ngram.cpp's gather, or returns false: i32 ids, a dense E4M3 table with a
 * scale or a dense bf16 table without, bf16 output rows at one pitch. */
static bool rows_form(const RadArgs* a, RowGather* g) {
    /* The scale (operand 2) is optional. */
    if (!rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 3)) return false;
    const RadTensor* tok = &a->t[0];
    const RadTensor* wte = &a->t[1];
    const RadTensor* sc  = &a->t[2];
    const RadTensor* x   = &a->t[3];
    if (tok->dtype != RAD_I32 || !rad_tensor_is_contiguous(tok)) return false;
    if (!rad_tensor_is_contiguous(wte)) return false;
    const bool e4m3 = wte->dtype == RAD_F8E4M3;
    if (!e4m3 && wte->dtype != RAD_BF16) return false;
    const bool scaled = sc->data && numel(sc) >= 1;
    if (x->dtype != RAD_BF16 || scaled != e4m3) return false;
    const int64_t n_embd = p_int(a, "n_embd", wte->rank >= 2 ? wte->shape[wte->rank - 1] : 0);
    const int64_t n_vocab = p_int(a, "n_vocab", wte->rank >= 2 ? wte->shape[wte->rank - 2] : 0);
    if (n_embd <= 0 || n_vocab <= 0) return false;
    int64_t ld = -1;
    if (rad_tensor_is_contiguous(x))                                        ld = n_embd;
    else if (x->rank == 2 && x->shape[1] == n_embd && x->stride[1] == 1)   ld = x->stride[0];
    if (ld < n_embd) return false;
    g->ids = (const int32_t*)tok->data;
    g->M = numel(x) / n_embd;
    g->tab = (const uint8_t*)wte->data;
    g->n_vocab = n_vocab;
    g->n_embd = n_embd;
    g->voff = p_int(a, "vocab_offset", 0);
    g->scale = e4m3 ? ldt(sc, 0) : 1.0f;
    g->row_bytes = e4m3 ? n_embd : 2 * n_embd;
    g->out = (uint16_t*)x->data;
    g->x_ld = ld;
    const RadTensor* ahd = rad_arg_in(a, 4);
    const bool ahead = ahd && ahd->dtype == RAD_I32 && rad_tensor_is_contiguous(ahd) && numel(ahd);
    g->ahead = ahead ? (const int32_t*)ahd->data : nullptr;
    g->n_ahead = ahead ? numel(ahd) : 0;
    g->rank = a->rank;
    g->world_size = a->world_size;
    return true;
}

AVX_KERNEL(embed_lookup_q) {
    RowGather g;
    if (!rows_form(a, &g)) return embed_impl<true>(a);
    if (g.voff < 0 || g.M <= 0 || numel(&a->t[0]) < g.M || numel(&a->t[1]) < g.n_vocab * g.n_embd)
        return RAD_E_SHAPE;
    /* Checked before the gather so a refusal never leaves the output half written. Same bound
     * as embed_impl. */
    if (g.voff == 0 && g.world_size <= 1)
        for (int64_t m = 0; m < g.M; ++m)
            if ((int64_t)g.ids[m] >= g.n_vocab) return RAD_E_INVAL;
    return row_gather(g, g.row_bytes == g.n_embd ? decode_e4m3_rows : decode_bf16_rows);
}
