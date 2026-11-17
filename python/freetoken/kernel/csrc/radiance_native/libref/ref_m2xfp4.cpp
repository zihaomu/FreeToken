/* ref_m2xfp4.cpp -- gemm_nt_q at dtype m2xfp4: M2XFP's metadata-augmented MXFP4 (arXiv
 * 2601.19213), the scheme libquant's `m2xfp4` quantiser writes.
 *
 * The weight is three TILED planes over 32-column groups: e2m1 codes, an E8M0 exponent per group
 * with the adaptive bias already folded in, and a meta byte per group whose two-bit field k_s
 * [2s, 2s+2) refines the exponent of subgroup s, columns [8s, 8s+8). The exact element is
 * e2m1 code . 2^(e-127) . (1 + k_s/4).
 *
 * gemm_nt_q's weight operand takes the codes and the meta, stored as the two planes' canonical
 * bytes concatenated -- the codes rows, then the meta rows -- the form the core's selection reader
 * hands a kernel that decodes the scheme itself (core/format/encoding.cpp). `b_scale` and `b_ref`
 * are the exponent plane as it lies; `b_ref` is accepted as at mxfp4a8 and does not change the
 * product.
 *
 * libref keeps one row an op, so ref_gemm_nt_q's row (ref_registry.cpp) takes its hooks from the
 * end of this file: each answers dtype m2xfp4 here and hands every other dtype, unchanged, to the
 * hook the row had (ref_gemm.cpp, ref_shapes.cpp, ref_planes.h).
 */
#include <algorithm>
#include <cstring>
#include <vector>
#include "ref_common.h"
#include "ref_gemm.h"
#include "ref_ops.h"
#include "ref_planes.h"

using namespace ref;

namespace {

/* The scheme as libquant writes it: codes [1 x 1], scale and meta [1 x 32]. A rotated weight
 * (ParoQuant) carries the same three planes in the basis its "givens" transform names, with the
 * rotation's tables beside them; the rotation is the activation's, so the planes are read as they
 * are. */
bool m2_enc_planes(const RadEncoding* enc) {
    if (!enc || !rad_enc_is(enc, "m2xfp4")) return false;
    if (enc->transform[0] && std::strncmp(enc->transform, "givens", RAD_ENC_STR) != 0) return false;
    const RadEncPlane* sc = rad_enc_plane(enc, "scale");
    const RadEncPlane* mt = rad_enc_plane(enc, "meta");
    const bool planes = enc->plane[0].kind == RAD_PLANE_TILED &&
                        enc->plane[0].dtype == RAD_FP4E2M1 && enc->plane[0].block[0] == 1 &&
                        enc->plane[0].block[1] == 1 && sc && mt && sc->kind == RAD_PLANE_TILED &&
                        sc->dtype == RAD_E8M0 && sc->block[0] == 1 && sc->block[1] == 32 &&
                        mt->kind == RAD_PLANE_TILED && mt->dtype == RAD_U8 &&
                        mt->block[0] == 1 && mt->block[1] == 32;
    if (!planes) return false;
    if (!enc->transform[0]) return enc->n_planes == 3;
    return enc->n_planes >= 6 && rad_enc_plane(enc, "t.pairs") && rad_enc_plane(enc, "t.angle") &&
           rad_enc_plane(enc, "t.scale");
}

/* Each product is one f32 multiply and each output one f32 accumulation in k order: the exact
 * decode's own arithmetic, since the multipliers take the codes off the integer grid an mxfp4
 * group sub-sum would use.
 *
 * Parts (`pb1`, `pb2`), a rotated (ParoQuant) fused projection's pieces, as gemm_ntq_impl's mxfp4
 * arm reads them (ref_gemm.cpp): weight rows [0, pb1), [pb1, pb2) and [pb2, N) each meet their own
 * copy of the activation, side by side in a row of `a` ([M, parts * K]) and of its scale
 * ([M, parts * K / 128]). 0 omits a bound. */
int m2_gemm(const RadArgs* a) {
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[2], *Y = &a->t[4];
    const RadTensor *as_ = t_in(a, 1), *bs_ = t_in(a, 3);
    if (!A->data || !B->data || !Y->data) return RAD_E_INVAL;
    if (t_in(a, 5)) return RAD_E_UNSUPPORTED;            /* a_sum belongs to the asym grids */
    if (A->dtype != RAD_F8E4M3 || B->dtype != RAD_U8 || B->rank != 1) return RAD_E_DTYPE;
    if (!as_ || !bs_ || bs_->dtype != RAD_E8M0 || bs_->rank < 2) return RAD_E_DTYPE;
    const int64_t K = p_int(a, "K", 0), N = p_int(a, "N", 0);
    if (K <= 0 || K % 32 || N <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / N;
    if (M <= 0) return RAD_E_SHAPE;
    const int64_t pb1 = p_int(a, "pb1", 0), pb2 = p_int(a, "pb2", 0);
    const int64_t e1 = pb1 > 0 ? std::min<int64_t>(pb1, N) : N;
    const int64_t e2 = pb2 > 0 ? std::max<int64_t>(e1, std::min<int64_t>(pb2, N)) : N;
    const int64_t parts = e2 < N ? 3 : (e1 < N ? 2 : 1);
    if (parts > 1 && (e1 % 16 || e2 % 16 || A->rank < 2 || A->shape[A->rank - 1] != parts * K))
        return RAD_E_SHAPE;
    const int64_t kb = K / 32, akb = (K + 127) / 128;
    if (numel(A) < M * parts * K || numel(B) != N * (K / 2 + kb) || numel(bs_) < N * kb ||
        numel(as_) < M * parts * akb)
        return RAD_E_SHAPE;
    const int64_t ars = row_stride(A, parts * K), acs = laststride(A);
    const int64_t srs = row_stride(as_, parts * akb), scs = laststride(as_);
    const int64_t brs = row_stride(bs_, kb);
    const int64_t ys_r = row_stride(Y, N), ys_c = laststride(Y);
    const uint8_t* meta0 = (const uint8_t*)B->data + N * (K / 2);
    REF_PAR_FOR(N)
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t* codes = (const uint8_t*)B->data + n * (K / 2);
        const uint8_t* meta  = meta0 + n * kb;
        const uint8_t* expe  = (const uint8_t*)bs_->data + n * brs;
        const int64_t  part  = n < e1 ? 0 : (n < e2 ? 1 : 2);
        const int64_t  ak0   = part * K, sk0 = part * akb;   /* this part's copy of `a` */
        /* The weight row decoded once, so the meta and exponent reads are not repeated per
         * activation row. */
        std::vector<float> w((size_t)K);
        for (int64_t k = 0; k < K; ++k) {
            const uint8_t c  = (uint8_t)((codes[k >> 1] >> (4 * (k & 1))) & 0xF);
            const float cv   = mxfp4_elem_to_f32(c);
            const float ev   = e8m0_to_f32(expe[k >> 5]);
            const float km   = 1.0f + 0.25f * (float)((meta[k >> 5] >> (2 * ((k & 31) >> 3))) & 3);
            w[(size_t)k]     = cv * ev * km;
        }
        for (int64_t m = 0; m < M; ++m) {
            const int64_t ab = m * ars;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                acc += ldt(A, ab + (ak0 + k) * acs) * ldt(as_, m * srs + (sk0 + (k >> 7)) * scs) *
                       w[(size_t)k];
            stt(Y, m * ys_r + n * ys_c, acc);
        }
    }
    return RAD_OK;
}

/* mxfp4a8's operands (ref_shapes.cpp), the weight being the codes and meta concatenated. */
int m2_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const long long N = rad_param_geti(p, n_p, "N", 0), K = rad_param_geti(p, n_p, "K", 0);
    if (rad_param_geti(p, n_p, "group", 0) != 32 || N <= 0 || K <= 0 || K % 32)
        return RAD_E_UNSUPPORTED;
    std::vector<RadParam> q(p, p + n_p);
    for (RadParam& r : q)
        if (r.key && std::strcmp(r.key, "dtype") == 0) r.sval = "mxfp4a8";
    const int rc = ref_gemm_nt_q_shape(q.data(), n_p, operand, out);
    if (rc != RAD_OK || operand != 2) return rc;
    out->dtype    = (uint32_t)RAD_U8;
    out->rank     = 1;
    out->shape[0] = N * (K / 2 + K / 32);
    out->shape[1] = 0;
    return RAD_OK;
}

/* The weight operand is codes and meta, concatenated; the scale and reference operands are the
 * exponent plane, stored as it is. Any other encoding is refused, a transform other than
 * "givens" included. */
int m2_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
              const RadTensor* planes, int n, RadLayout* out) {
    const long long group = rad_param_geti(p, n_p, "group", 0);
    const long long N = rad_param_geti(p, n_p, "N", 0), K = rad_param_geti(p, n_p, "K", 0);
    if (!m2_enc_planes(enc) || group != 32 || N <= 0 || K <= 0 || K % 32) return RAD_E_DTYPE;
    if (operand == 2) {
        if (!sel_roles(enc, sel, n, "codes", "meta")) return RAD_E_DTYPE;
        if (planes[0].shape[0] != N || planes[0].shape[1] != K ||
            planes[1].shape[0] != N || planes[1].shape[1] != K / 32)
            return RAD_E_SHAPE;
        out->tag = "m2.codes.meta";
        out->dtype = RAD_U8;
        out->rank = 1;
        out->shape[0] = N * (K / 2 + K / 32);
        out->align = 0;
        out->bytes = out->shape[0];
        return RAD_OK;
    }
    if (operand == 3 || operand == 6) {
        if (n != 1 || sel[0] < 0 || sel[0] >= enc->n_planes ||
            std::strncmp(enc->plane[sel[0]].role, "scale", RAD_ENC_STR) != 0)
            return RAD_E_DTYPE;
        return RAD_E_UNSUPPORTED;   /* one plane, stored as it is */
    }
    return RAD_E_UNSUPPORTED;
}

int m2_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                const RadTensor* planes, int n, void* dst, int64_t dst_bytes) {
    RadLayout lay{};
    const int st = m2_layout(p, n_p, operand, enc, sel, planes, n, &lay);
    if (st != RAD_OK) return st;
    if (dst_bytes != lay.bytes) return RAD_E_SHAPE;
    uint8_t* d = (uint8_t*)dst;
    for (int i = 0; i < n; ++i) {
        const int64_t b =
            planes[i].shape[0] * rad_enc_row_bytes(planes[i].dtype, planes[i].shape[1]);
        std::memcpy(d, planes[i].data, (size_t)b);
        d += b;
    }
    return RAD_OK;
}

int m2_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const void* src, int64_t src_bytes, const RadTensor* planes, int n) {
    RadLayout lay{};
    const int st = m2_layout(p, n_p, operand, enc, sel, planes, n, &lay);
    if (st != RAD_OK) return st;
    if (src_bytes != lay.bytes) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    for (int i = 0; i < n; ++i) {
        const int64_t b =
            planes[i].shape[0] * rad_enc_row_bytes(planes[i].dtype, planes[i].shape[1]);
        std::memcpy(planes[i].data, s, (size_t)b);
        s += b;
    }
    return RAD_OK;
}

}  /* namespace */

/* ---- ref_gemm_nt_q's row hooks: m2xfp4 here, every other dtype the row's own. */

static bool is_m2xfp4(const RadParam* p, int n_p) {
    const char* dt = rad_param_gets(p, n_p, "dtype", nullptr);
    return dt && std::strcmp(dt, "m2xfp4") == 0;
}

int ref_gemm_nt_q_m2(const RadArgs* a, RadStream s) {
    return is_m2xfp4(a->p, a->n_p) ? m2_gemm(a) : ref_gemm_nt_q(a, s);
}

int ref_gemm_nt_q_m2_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return is_m2xfp4(p, n_p) ? m2_shape(p, n_p, operand, out)
                             : ref_gemm_nt_q_shape(p, n_p, operand, out);
}

int ref_layout_gemm_q_m2(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                         const int* sel, const RadTensor* planes, int n, RadLayout* out) {
    return is_m2xfp4(p, n_p) ? m2_layout(p, n_p, operand, enc, sel, planes, n, out)
                             : ref_layout_gemm_q(p, n_p, operand, enc, sel, planes, n, out);
}

int ref_relayout_gemm_q_m2(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                           const int* sel, const RadTensor* planes, int n, void* dst,
                           int64_t dst_bytes) {
    return is_m2xfp4(p, n_p)
               ? m2_relayout(p, n_p, operand, enc, sel, planes, n, dst, dst_bytes)
               : ref_relayout_planes<ref_layout_gemm_q>(p, n_p, operand, enc, sel, planes, n, dst,
                                                        dst_bytes);
}

int ref_unrelayout_gemm_q_m2(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                             const int* sel, const void* src, int64_t src_bytes,
                             const RadTensor* planes, int n) {
    return is_m2xfp4(p, n_p)
               ? m2_unrelayout(p, n_p, operand, enc, sel, src, src_bytes, planes, n)
               : ref_unrelayout_planes<ref_layout_gemm_q>(p, n_p, operand, enc, sel, src,
                                                          src_bytes, planes, n);
}
