/* avx_planes.h -- libavx's reading of a weight operand joined over several planes
 * (abi/rad_planes.h), and the layout hooks that ask for that form.
 *
 * The rules are libref's (libref/ref_planes.h), stated again because a plugin links only abi/:
 * the same selections are joined and the same ones refused, so the selector can resolve either
 * library for a band of an op that reads such a weight, and both read one stored form. A single
 * plane is stored as it is, so these rows still read the plain forms a tool describes.
 *
 * Plain functions with internal linkage, no vector code: avx_registry.cpp (baseline) installs the
 * hooks and every ISA level's kernels read the planes through the same helpers.
 */
#ifndef RAD_AVX_PLANES_H
#define RAD_AVX_PLANES_H

#include "rad_abi.h"
#include "rad_plugin.h"
#include "rad_planes.h"

#include <cstdint>
#include <cstring>

namespace avxp {

/* ================================================================== reading at launch */

/* One plane of a joined operand, as its entry describes it. */
struct JPlane {
    const uint8_t* p = nullptr;
    uint32_t dt = 0;
    int64_t  rows = 0, cols = 0, row_bytes = 0;
    int64_t  br = 0, bc = 0;          /* logical rows x columns one element covers; 0 is all */

    explicit operator bool() const { return p != nullptr; }
    float at(int64_t i, int64_t j) const { return rad_load_f32(p + i * row_bytes, dt, j); }
};

static inline bool is_joined(const RadTensor* t) {
    return t && t->data && rad_planes_header(t->data) != nullptr;
}

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

/* Codes with one scale over [rows, cols]; element (r, c) is code(r, c) * scale covering it, one
 * f32 rounding, as the core's affine decoder gives it (libref/ref_planes.h: ScaledRows). */
struct ScaledRows {
    JPlane codes, scale;
    /* Columns one scale covers, the whole row for a block extent of 0. */
    int64_t span(int64_t cols) const { return scale.bc > 0 ? scale.bc : cols; }
    float scale_at(int64_t r, int64_t c) const {
        return scale.at(scale.br > 0 ? r / scale.br : 0, scale.bc > 0 ? c / scale.bc : 0);
    }
};

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
 * As libref/ref_planes.h: unsigned codes read as (code - zero) * scale, one f16 scale and one
 * integer zero per (row, `group` of K) joined in `b_scale`, against one of three activations. */
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

static inline bool zero_dt(uint32_t dt) {
    return dt == RAD_U8 || dt == RAD_U2 || dt == RAD_U4 || dt == RAD_U5;
}

/* ================================================================== the layout hooks */

static inline bool sel_roles(const RadEncoding* enc, const int* sel, int n, const char* r0,
                             const char* r1) {
    if (!enc || !sel || n != 2) return false;
    for (int i = 0; i < 2; ++i)
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return false;
    return std::strncmp(enc->plane[sel[0]].role, r0, RAD_ENC_STR) == 0 &&
           std::strncmp(enc->plane[sel[1]].role, r1, RAD_ENC_STR) == 0;
}

static inline bool enc_scaled_rows(const RadEncoding* enc, const int* sel, int n) {
    if (!rad_enc_is(enc, "affine") || enc->transform[0] || enc->n_planes != 2) return false;
    if (!sel_roles(enc, sel, n, "codes", "scale")) return false;
    const RadEncPlane& c = enc->plane[sel[0]];
    const RadEncPlane& s = enc->plane[sel[1]];
    return c.kind == RAD_PLANE_TILED && c.block[0] == 1 && c.block[1] == 1 &&
           (c.dtype == RAD_F8E4M3 || c.dtype == RAD_I8) && s.kind == RAD_PLANE_TILED &&
           float_scale_dt(s.dtype);
}

static inline int layout_scaled(int operand, int rows_opd, int rows_opd2, const RadEncoding* enc,
                                const int* sel, const RadTensor* planes, int n, RadLayout* out) {
    if (n < 2 || (operand != rows_opd && operand != rows_opd2)) return RAD_E_UNSUPPORTED;
    if (!enc_scaled_rows(enc, sel, n)) return RAD_E_DTYPE;
    return rad_planes_layout(enc, sel, planes, n, out);
}

/* logits_gemm and logit_rerank: the head is operand 1 of both. */
static inline int avx_layout_head(const RadParam*, int, int operand, const RadEncoding* enc,
                                  const int* sel, const RadTensor* planes, int n, RadLayout* out) {
    return layout_scaled(operand, 1, -1, enc, sel, planes, n, out);
}

/* hc_read: mix_down and mix_up (operands 2 and 3). */
static inline int avx_layout_hc(const RadParam*, int, int operand, const RadEncoding* enc,
                                const int* sel, const RadTensor* planes, int n, RadLayout* out) {
    return layout_scaled(operand, 2, 3, enc, sel, planes, n, out);
}

/* mtp_enter: fc_h and fc_e (operands 4 and 5). */
static inline int avx_layout_mtp(const RadParam*, int, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* planes, int n, RadLayout* out) {
    return layout_scaled(operand, 4, 5, enc, sel, planes, n, out);
}

/* gemm_nt_q: `b_scale` (operand 3) as the scale and zero of the grid `dtype` names, or a symmetric
 * int8 grid's scale in two levels, an unsigned integer per (row, `group` of K) under a float a row
 * (IQ8R), whose product is the scale: libref's two forms (ref_planes.h). */
static inline int avx_layout_gemm_q(const RadParam* p, int n_p, int operand,
                                    const RadEncoding* enc, const int* sel,
                                    const RadTensor* planes, int n, RadLayout* out) {
    /* libref serves M2XFP4 (ref_m2xfp4.cpp); refusing it here passes the op on to it. */
    if (enc && rad_enc_is(enc, "m2xfp4")) return RAD_E_DTYPE;
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

template <RadLayoutFn L>
static int avx_relayout_planes(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                               const int* sel, const RadTensor* planes, int n, void* dst,
                               int64_t dst_bytes) {
    RadLayout lay{};
    const int st = L(p, n_p, operand, enc, sel, planes, n, &lay);
    if (st != RAD_OK) return st;
    return rad_planes_relayout(enc, sel, planes, n, dst, dst_bytes);
}

template <RadLayoutFn L>
static int avx_unrelayout_planes(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const void* src, int64_t src_bytes,
                                 const RadTensor* planes, int n) {
    RadLayout lay{};
    const int st = L(p, n_p, operand, enc, sel, planes, n, &lay);
    if (st != RAD_OK) return st;
    return rad_planes_unrelayout(enc, sel, src, src_bytes, planes, n);
}

}  /* namespace avxp */

#ifdef AVX_LEVEL
/* The per-level decoders (avx_planes.cpp). */
namespace avx {
inline namespace AVX_SUFFIX {

/* n E4M3 or int8 codes, each times `s`: the code's exact value, then one rounding. */
void decode_scaled(uint32_t dt, const uint8_t* src, float s, float* dst, int64_t n);

/* `n` unsigned codes of `bits` from element k0 of a packed row, each (code - z) * s. */
void decode_zp(const uint8_t* row, int bits, int64_t k0, int64_t n, int32_t z, float s,
               float* dst);

/* Row `r` of a ScaledRows weight, `K` wide, as f32. */
void scaled_row(const avxp::ScaledRows& sr, int64_t r, int64_t K, float* dst);

}  /* inline namespace AVX_SUFFIX */
}  /* namespace avx */
#endif

#endif /* RAD_AVX_PLANES_H */
