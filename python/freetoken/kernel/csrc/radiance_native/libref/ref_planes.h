/* ref_planes.h -- libref's reading of a weight operand joined over several planes
 * (abi/rad_planes.h), and the layout hooks that ask for that form.
 *
 * A weight declared over several planes of one stored tensor reaches a kernel as one operand: an
 * fp8 or int8 lm head's codes and scale, Flash-Next's E4M3 mixing rows with their f32 group
 * scales, a zero-point grid's scale and zero. On the host it is stored joined: a header naming each
 * plane, then the planes as the container holds them. A launch recognises the form by its header
 * and reads the planes by role, so the bytes are the canonical ones and nothing is decoded at
 * load.
 *
 * The hooks join only a selection of several planes; one plane is stored as it is (rad_planes.h
 * declines it), so a row with these hooks still reads the plain forms a tool describes (the
 * logical f32 head rad-kbench and the op oracle hand over). They refuse with RAD_E_DTYPE a joined
 * selection the op does not compute, which keeps the row from being selected for it.
 *
 * libavx/avx_planes.h states the same rules for libavx; the two must accept the same encodings,
 * since a selector may resolve either library for a band.
 */
#ifndef REF_PLANES_H
#define REF_PLANES_H

#include "ref_common.h"
#include "rad_planes.h"

namespace ref {

/* ================================================================== reading at launch */

/* One plane of a joined operand, as its entry describes it. */
struct JPlane {
    const uint8_t* p = nullptr;
    uint32_t dt = 0;
    int64_t  rows = 0, cols = 0, row_bytes = 0;
    int64_t  br = 0, bc = 0;          /* logical rows x columns one element covers; 0 is all */

    explicit operator bool() const { return p != nullptr; }
    float at(int64_t i, int64_t j) const { return rad_load_f32(p + i * row_bytes, dt, j); }
    /* The element of a TILED plane covering logical (r, c). */
    float cover(int64_t r, int64_t c) const {
        return at(br > 0 ? r / br : 0, bc > 0 ? c / bc : 0);
    }
};

static inline bool is_joined(const RadTensor* t) {
    return t && t->data && rad_planes_header(t->data) != nullptr;
}

/* The plane of `t`'s joined form with this role; false when there is none. */
static inline JPlane jplane(const RadTensor* t, const char* role) {
    JPlane v;
    const RadPlanesEntry* e = nullptr;
    const void* d = t && t->data ? rad_planes_find(t->data, role, &e) : nullptr;
    if (!d) return v;
    v.p = (const uint8_t*)d;
    v.dt = e->dtype;
    v.rows = e->rows;
    v.cols = e->cols;
    v.row_bytes = e->row_bytes;
    v.br = e->block[0];
    v.bc = e->block[1];
    return v;
}

static inline int64_t cdiv_or_1(int64_t n, int64_t b) { return b > 0 ? (n + b - 1) / b : 1; }

static inline bool float_scale_dt(uint32_t dt) {
    return dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F32;
}

/* Codes with one scale over [rows, cols], read element by element as the core's affine decoder
 * gives it: the code times the scale covering it, one f32 rounding. An E4M3 lm head with a bf16
 * scale per 128x128 block, an int8 one per (row, 128 of K), E4M3 rows with an f32 scale per group
 * of their own. */
struct ScaledRows {
    JPlane codes, scale;
    float at(int64_t r, int64_t c) const { return codes.at(r, c) * scale.cover(r, c); }
};

/* RAD_OK and the view, RAD_E_DTYPE for a joined form other than codes and a scale, RAD_E_SHAPE
 * for planes that do not cover [rows, cols]. */
static inline int scaled_rows(const RadTensor* t, int64_t rows, int64_t cols, ScaledRows* v) {
    const RadPlanesHeader* h = t && t->data ? rad_planes_header(t->data) : nullptr;
    if (!h) return RAD_E_INVAL;
    v->codes = jplane(t, "codes");
    v->scale = jplane(t, "scale");
    if (h->n_planes != 2 || !v->codes || !v->scale) return RAD_E_DTYPE;
    if ((v->codes.dt != RAD_F8E4M3 && v->codes.dt != RAD_I8) || !float_scale_dt(v->scale.dt) ||
        v->codes.br != 1 || v->codes.bc != 1)
        return RAD_E_DTYPE;
    if (v->codes.rows != rows || v->codes.cols != cols ||
        v->scale.rows != cdiv_or_1(rows, v->scale.br) ||
        v->scale.cols != cdiv_or_1(cols, v->scale.bc))
        return RAD_E_SHAPE;
    return RAD_OK;
}

/* ================================================================== the zero-point grids
 *
 * gemm_nt_q's asymmetric grids (docs/OPS.md): unsigned codes read as (code - zero) * scale, an f16
 * scale and an integer zero per (row, `group` of K), joined in `b_scale` from the weight's "scale"
 * and "zero" planes; `b` is the codes plane. The activation side is one of three:
 *
 *   ZP_I8_ROW    quant_act_i8's codes in the step order, an f32 scale a row and the code sums
 *                `a_sum` the zero is removed with (w2a8, w4a8_asym)
 *   ZP_I8_GROUP  row-major int8 codes with an f32 scale per (row, 128 of K), as givens_dense_quant
 *                writes them (w4a8i8_asym, w5a8i8)
 *   ZP_F16       a 16-bit activation, givens_rotate's (w4a16_asym, w5a16) */
enum ZpAct { ZP_I8_ROW = 0, ZP_I8_GROUP = 1, ZP_F16 = 2 };

struct ZpGrid {
    const char* name;
    uint32_t    codes;
    ZpAct       act;
};

static inline const ZpGrid* zp_grid(const char* dtype) {
    static const ZpGrid kGrids[] = {
        { "w2a8", RAD_U2, ZP_I8_ROW },        { "w4a8_asym", RAD_U4, ZP_I8_ROW },
        { "w4a16_asym", RAD_U4, ZP_F16 },     { "w5a16", RAD_U5, ZP_F16 },
        { "w4a8i8_asym", RAD_U4, ZP_I8_GROUP }, { "w5a8i8", RAD_U5, ZP_I8_GROUP },
    };
    if (!dtype) return nullptr;
    for (const ZpGrid& g : kGrids)
        if (std::strcmp(g.name, dtype) == 0) return &g;
    return nullptr;
}

/* An unsigned integer zero point a code is compared against: the same width as the codes or
 * wider, never signed. */
static inline bool zero_dt(uint32_t dt) {
    return dt == RAD_U8 || dt == RAD_U2 || dt == RAD_U4 || dt == RAD_U5;
}

/* ================================================================== the layout hooks */

/* The encoding's planes `sel` names, in order: one role per name. */
static inline bool sel_roles(const RadEncoding* enc, const int* sel, int n, const char* r0,
                             const char* r1) {
    if (!enc || !sel || n != 2) return false;
    for (int i = 0; i < 2; ++i)
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return false;
    return std::strncmp(enc->plane[sel[0]].role, r0, RAD_ENC_STR) == 0 &&
           std::strncmp(enc->plane[sel[1]].role, r1, RAD_ENC_STR) == 0;
}

/* Whether `enc` and its selection are codes and one scale, untransformed: what ScaledRows reads. */
static inline bool enc_scaled_rows(const RadEncoding* enc, const int* sel, int n) {
    if (!rad_enc_is(enc, "affine") || enc->transform[0] || enc->n_planes != 2) return false;
    if (!sel_roles(enc, sel, n, "codes", "scale")) return false;
    const RadEncPlane& c = enc->plane[sel[0]];
    const RadEncPlane& s = enc->plane[sel[1]];
    return c.kind == RAD_PLANE_TILED && c.block[0] == 1 && c.block[1] == 1 &&
           (c.dtype == RAD_F8E4M3 || c.dtype == RAD_I8) && s.kind == RAD_PLANE_TILED &&
           float_scale_dt(s.dtype);
}

/* The joined form for a weight operand that takes codes and one scale (`rows_opd` and `rows_opd2`
 * are the operands that may be one, -1 for none). Every other operand, and a single plane, is
 * stored as it is. */
static inline int layout_scaled(int operand, int rows_opd, int rows_opd2, const RadEncoding* enc,
                                const int* sel, const RadTensor* planes, int n, RadLayout* out) {
    if (n < 2 || (operand != rows_opd && operand != rows_opd2)) return RAD_E_UNSUPPORTED;
    if (!enc_scaled_rows(enc, sel, n)) return RAD_E_DTYPE;
    return rad_planes_layout(enc, sel, planes, n, out);
}

/* logits_gemm and logit_rerank: the head is operand 1 of both. */
static inline int ref_layout_head(const RadParam*, int, int operand, const RadEncoding* enc,
                                  const int* sel, const RadTensor* planes, int n, RadLayout* out) {
    return layout_scaled(operand, 1, -1, enc, sel, planes, n, out);
}

/* hc_read: mix_down and mix_up (operands 2 and 3). */
static inline int ref_layout_hc(const RadParam*, int, int operand, const RadEncoding* enc,
                                const int* sel, const RadTensor* planes, int n, RadLayout* out) {
    return layout_scaled(operand, 2, 3, enc, sel, planes, n, out);
}

/* mtp_enter: fc_h and fc_e (operands 4 and 5). */
static inline int ref_layout_mtp(const RadParam*, int, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* planes, int n, RadLayout* out) {
    return layout_scaled(operand, 4, 5, enc, sel, planes, n, out);
}

/* gemm_nt_q: `b_scale` (operand 3) as the scale and zero of the zero-point grid `dtype` names,
 * one each per (row, `group` of K), over codes of that grid's width. The codes (operand 2) are one
 * plane and stored as they are. A "givens" transform is the activation's (givens_rotate,
 * givens_dense_quant), so the planes are read the same way under it.
 *
 * A symmetric int8 grid's scale in two levels is joined the same way: an unsigned integer "scale"
 * per (row, `group` of K) under a float "scale.1" a row (IQ8R), whose product is the scale. */
static inline int ref_layout_gemm_q(const RadParam* p, int n_p, int operand,
                                    const RadEncoding* enc, const int* sel,
                                    const RadTensor* planes, int n, RadLayout* out) {
    if (n < 2 || operand != 3) return RAD_E_UNSUPPORTED;
    const ZpGrid* zg = zp_grid(rad_param_gets(p, n_p, "dtype", nullptr));
    const long long group = rad_param_geti(p, n_p, "group", 0);
    if (!zg && sel_roles(enc, sel, n, "scale", "scale.1")) {
        if (group <= 0 || !rad_enc_is(enc, "affine") || enc->n_planes != 3) return RAD_E_DTYPE;
        const RadEncPlane& c = enc->plane[0];
        const RadEncPlane& s = enc->plane[sel[0]];
        const RadEncPlane& s2 = enc->plane[sel[1]];
        if (std::strncmp(c.role, "codes", RAD_ENC_STR) != 0 || c.dtype != RAD_I8 ||
            c.block[0] != 1 || c.block[1] != 1)
            return RAD_E_DTYPE;
        if (s.kind != RAD_PLANE_TILED || s.dtype != RAD_U8 || s.block[0] != 1 ||
            s.block[1] != group || s2.kind != RAD_PLANE_TILED || !float_scale_dt(s2.dtype) ||
            s2.block[0] != 1 || s2.block[1] != 0)
            return RAD_E_DTYPE;
        return rad_planes_layout(enc, sel, planes, n, out);
    }
    if (!zg || group <= 0 || !rad_enc_is(enc, "affine")) return RAD_E_DTYPE;
    if (enc->transform[0] && std::strncmp(enc->transform, "givens", RAD_ENC_STR) != 0)
        return RAD_E_DTYPE;
    if (!sel_roles(enc, sel, n, "scale", "zero") || enc->n_planes < 3) return RAD_E_DTYPE;
    const RadEncPlane& c = enc->plane[0];
    const RadEncPlane& s = enc->plane[sel[0]];
    const RadEncPlane& z = enc->plane[sel[1]];
    if (std::strncmp(c.role, "codes", RAD_ENC_STR) != 0 || c.dtype != zg->codes ||
        c.block[0] != 1 || c.block[1] != 1)
        return RAD_E_DTYPE;
    if (s.kind != RAD_PLANE_TILED || s.dtype != RAD_F16 || s.block[0] != 1 || s.block[1] != group ||
        z.kind != RAD_PLANE_TILED || !zero_dt(z.dtype) || z.block[0] != 1 || z.block[1] != group)
        return RAD_E_DTYPE;
    return rad_planes_layout(enc, sel, planes, n, out);
}

/* RadRelayoutFn and RadUnrelayoutFn for any of the layout hooks above: the hook decides whether
 * the planes are joined, and the copies are rad_planes.h's. */
template <RadLayoutFn L>
static int ref_relayout_planes(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                               const int* sel, const RadTensor* planes, int n, void* dst,
                               int64_t dst_bytes) {
    RadLayout lay{};
    const int st = L(p, n_p, operand, enc, sel, planes, n, &lay);
    if (st != RAD_OK) return st;
    return rad_planes_relayout(enc, sel, planes, n, dst, dst_bytes);
}

template <RadLayoutFn L>
static int ref_unrelayout_planes(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const void* src, int64_t src_bytes,
                                 const RadTensor* planes, int n) {
    RadLayout lay{};
    const int st = L(p, n_p, operand, enc, sel, planes, n, &lay);
    if (st != RAD_OK) return st;
    return rad_planes_unrelayout(enc, sel, src, src_bytes, planes, n);
}

}  /* namespace ref */

#endif /* REF_PLANES_H */
