/* rad_plugin.h -- everything a kernel or architecture plugin needs and cannot link.
 *
 * ===================================================================== WHY THIS FILE EXISTS
 *
 * A plugin exports the symbols in rad_abi.h and links nothing of radiance's, so it has only the
 * raw structs. This header supplies the helpers every plugin would otherwise rewrite: an
 * optional-operand fetch, parameters with defaults, row pitch, and f32<->bf16/f16/fp8
 * conversions.
 *
 * Two of them are contracts, not conveniences. The narrowing conversions define what a result is:
 * libref's contract is f32 accumulation, IEEE round-to-nearest-even, no FMA and no reassociation,
 * and a plugin whose bf16 rounding truncates cannot be checked against the reference. And an
 * absent optional operand is a null RadTensor.data, not a short operand count; getting that wrong
 * silently shifts every later operand by one.
 *
 * ===================================================================== WHAT IS AND IS NOT HERE
 *
 * Here: pure, total, header-only functions over the ABI's own types. Nothing allocates, holds
 * state or calls back into the core, and nothing may start to.
 *
 * Not here: anything that needs the core. Device allocations, streams and builder handles come
 * from rad_device.h / rad_builder.h / rad_runtime.h, which the host exports; the loader's ABI
 * check covers them.
 *
 * This header is optional. A plugin that includes only rad_abi.h is complete; this is
 * convenience plus one shared definition of the rounding.
 *
 * Pure C, like the rest of abi/. See spec.md 2.
 */
#ifndef RAD_PLUGIN_H
#define RAD_PLUGIN_H

#include "rad_abi.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================== parameters
 *
 * Two families: `launch` and `scratch` get a RadArgs; `layout`, `relayout`, `unrelayout` and
 * `opd_shape` get a bare (const RadParam*, int) pair with no tensors. */

static inline const RadParam* rad_param_find(const RadParam* p, int n_p, const char* key) {
    if (!p || !key) return 0;
    for (int i = 0; i < n_p; ++i)
        if (p[i].key && strcmp(p[i].key, key) == 0) return &p[i];
    return 0;
}

/* An int parameter, or `dflt` if the key is absent or not an int.
 *
 * A range is deliberately not accepted: by the time a kernel runs, the ranged axis has been
 * resolved to a value, so a range reaching a launch is a selection bug that returning its low
 * bound would hide. */
static inline long long rad_param_geti(const RadParam* p, int n_p, const char* key,
                                       long long dflt) {
    const RadParam* f = rad_param_find(p, n_p, key);
    return (f && f->kind == RAD_P_INT) ? f->ival : dflt;
}

/* The same read for a shape hook, the one place a range is legitimate; it reads as its high
 * bound. `opd_shape` sizes buffers before selection has collapsed anything, so over a banded op it
 * describes the whole band, and the high end covers the low. */
static inline long long rad_param_getdim(const RadParam* p, int n_p, const char* key,
                                         long long dflt) {
    const RadParam* f = rad_param_find(p, n_p, key);
    if (!f) return dflt;
    if (f->kind == RAD_P_INT)   return f->ival;
    if (f->kind == RAD_P_RANGE) return f->ihi;
    return dflt;
}

static inline const char* rad_param_gets(const RadParam* p, int n_p, const char* key,
                                         const char* dflt) {
    const RadParam* f = rad_param_find(p, n_p, key);
    return (f && f->kind == RAD_P_STR && f->sval) ? f->sval : dflt;
}

static inline double rad_param_getf(const RadParam* p, int n_p, const char* key, double dflt) {
    const RadParam* f = rad_param_find(p, n_p, key);
    return (f && f->kind == RAD_P_F64) ? f->dval : dflt;
}

/* The dtype a parameter names, parsed. RAD_DT_INVALID if the key is absent or unrecognised, which
 * a kernel should treat as a refusal. */
static inline uint32_t rad_param_getdt(const RadParam* p, int n_p, const char* key) {
    return rad_dtype_parse(rad_param_gets(p, n_p, key, 0));
}

/* The RadArgs forms. rad_args_geti/gets are in rad_abi.h and return presence; these return the
 * value with a default, which is what a launch almost always wants. */
static inline long long rad_args_geti_or(const RadArgs* a, const char* key, long long dflt) {
    long long v = 0;
    return (a && rad_args_geti(a, key, &v)) ? v : dflt;
}

static inline const char* rad_args_gets_or(const RadArgs* a, const char* key, const char* dflt) {
    const char* s = a ? rad_args_gets(a, key) : 0;
    return s ? s : dflt;
}

static inline double rad_args_getf_or(const RadArgs* a, const char* key, double dflt) {
    return a ? rad_param_getf(a->p, a->n_p, key, dflt) : dflt;
}

/* ================================================================== operands
 *
 * The absent-operand rule: an optional operand a call does not pass is a RadTensor whose `data`
 * is null, not a shorter `n_t`. Operands are positional, so treating absence as a shorter array
 * would read operand i+1 for operand i and produce wrong numbers without a fault.
 *
 * There is always one entry per schema operand. This returns null both past the end and for a
 * null `data`; a caller that sees the former has a schema disagreement, not an absent operand. */
static inline const RadTensor* rad_arg_in(const RadArgs* a, int i) {
    if (!a || !a->t || i < 0 || i >= a->n_t) return 0;
    return a->t[i].data ? &a->t[i] : 0;
}


/* There is deliberately no rad_arg_out. RadArgs::t is a `const RadTensor*`: the descriptors are
 * const, and a kernel writes an output through the non-const `void* data` inside one, so outputs
 * are fetched with rad_arg_in too. */

/* Are the first `n` operands all present? For a launch whose required operands are a prefix. */
static inline int rad_args_have(const RadArgs* a, int n) {
    for (int i = 0; i < n; ++i) if (!rad_arg_in(a, i)) return 0;
    return 1;
}

/* ================================================================== tensor shape helpers */

/* [rows..., K] flattened to (M, K): every leading axis is a row count and only the last is
 * structural, as a row-owning kernel (one workgroup per row) wants. Returns 0 on a rank-0
 * tensor. */
static inline int64_t rad_tensor_rows(const RadTensor* t) {
    if (!t || t->rank == 0) return 0;
    int64_t m = 1;
    for (uint32_t i = 0; i + 1 < t->rank; ++i) m *= t->shape[i];
    return m;
}

static inline int64_t rad_tensor_lastdim(const RadTensor* t) {
    return (t && t->rank > 0) ? t->shape[t->rank - 1] : 0;
}

/* The element distance between consecutive rows: the stride of the second-to-last axis (the last
 * extent when contiguous). Strides are in elements, not bytes (rad_types.h). */
static inline int64_t rad_tensor_row_pitch(const RadTensor* t) {
    if (!t || t->rank == 0) return 0;
    if (t->rank == 1) return t->shape[0];
    return t->stride[t->rank - 2];
}

/* Is `p` aligned to `bytes`? The ABI does not promise alignment and a constraint row cannot
 * express it, so a kernel issuing wide loads checks it in the entry point (rad_types.h). */
static inline int rad_ptr_aligned(const void* p, int64_t bytes) {
    return bytes > 0 && ((uintptr_t)p % (uintptr_t)bytes) == 0;
}

/* ================================================================== narrowing conversions
 *
 * These define what a result is, so they are stated once here.
 *
 * libref's contract, which every kernel is checked against, is f32 accumulation, IEEE
 * round-to-nearest-ties-to-even, no fused multiply-add and no reassociation; `-ffp-contract=off`
 * is a correctness flag in this project. A plugin whose bf16 narrowing truncates is off by up to
 * one ulp on every element and cannot be checked against the reference.
 *
 * Every conversion is exact when widening and RTNE when narrowing, and keeps subnormals rather
 * than flushing them: the hardware decodes subnormals, and flushing loses the smallest block of
 * every quantised weight. */

static inline float rad_bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* `(u >> 16) & 1` is the tie-break: it adds one more than the halfway point when the surviving
 * bit is odd, so ties go to even. Overflow carries into 0x7f80, the bf16 infinity. A NaN is forced
 * quiet so it cannot round into an infinity. */
static inline uint16_t rad_f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static inline float rad_f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31;
    uint32_t e = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t m = (uint32_t)h & 0x3ffu;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;                       /* subnormal: renormalise into f32's range */
            while (!(m & 0x400u)) { m <<= 1; ++sh; }
            m &= 0x3ffu;
            u = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | (m << 13);
        }
    } else if (e == 0x1fu) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e - 15u + 127u) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* RTNE, with subnormals and overflow-to-infinity handled explicitly rather than by a hardware
 * convert whose rounding mode a caller could have changed. */
static inline uint16_t rad_f32_to_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t s = (u >> 16) & 0x8000u;
    uint32_t a = u & 0x7fffffffu;

    if (a >= 0x7f800000u)                                     /* inf or nan */
        return (uint16_t)(s | 0x7c00u | (a > 0x7f800000u ? ((a >> 13) & 0x3ffu) | 0x200u : 0u));
    if (a >= 0x47800000u) return (uint16_t)(s | 0x7c00u);     /* overflow -> inf */
    if (a <  0x33000000u) return (uint16_t)s;                 /* underflow -> zero */

    {
        int e = (int)(a >> 23) - 127;
        uint32_t m = a & 0x7fffffu;
        if (e < -14) {                                        /* subnormal half */
            int sh = -14 - e + 13;
            uint32_t lo, q, half;
            m |= 0x800000u;
            lo   = m & ((1u << sh) - 1u);
            q    = m >> sh;
            half = 1u << (sh - 1);
            if (lo > half || (lo == half && (q & 1u))) ++q;
            return (uint16_t)(s | q);
        }
        {
            uint32_t q  = m >> 13;
            uint32_t lo = m & 0x1fffu;
            uint32_t he;
            if (lo > 0x1000u || (lo == 0x1000u && (q & 1u))) ++q;
            he = (uint32_t)(e + 15);
            if (q == 0x400u) { q = 0; ++he; }                 /* rounding carried into the exp */
            if (he >= 0x1fu) return (uint16_t)(s | 0x7c00u);
            return (uint16_t)(s | (he << 10) | q);
        }
    }
}

/* OCP fp8 e4m3 ("e4m3fn"): 1-4-3, bias 7, no infinity, S1111111 is the only NaN, max 448. The
 * KV-cache and block-scaled weight format, so the attention oracle and every fp8 GEMM depend on
 * its rounding. */
static inline float rad_fp8e4m3_to_f32(uint8_t b) {
    uint32_t s = (uint32_t)(b >> 7) << 31;
    uint32_t e = (uint32_t)(b >> 3) & 0xfu;
    uint32_t m = (uint32_t)b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((uint32_t)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;                                  /* the single NaN encoding */
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* Saturates at 448 rather than overflowing: a NaN for one large activation would spread through
 * the whole row. Every fp8 cast in this project clamps for that reason. */
static inline uint8_t rad_f32_to_fp8e4m3(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    {
        uint8_t  s = (uint8_t)((u >> 24) & 0x80u);
        uint32_t a = u & 0x7fffffffu;
        int e;
        uint32_t m;
        if (a > 0x7f800000u) return (uint8_t)(s | 0x7fu);     /* nan */
        if (a >= 0x43e00000u) return (uint8_t)(s | 0x7eu);    /* >= 448: clamp to the largest */
        e = (int)(a >> 23) - 127;
        m = a & 0x7fffffu;
        /* The smallest subnormal is 2^-9, so everything below half of it (exponent -11 and
         * below) rounds to zero. Exponent -10 is [2^-10, 2^-9): its tie point 2^-10 rounds to the
         * even zero and the rest up to 2^-9, which the subnormal branch does with a 24-bit
         * shift. */
        if (e < -10) return s;
        if (e < -6) {                                         /* subnormal fp8 */
            int sh = -6 - e + 20;
            uint32_t lo, q, half;
            m |= 0x800000u;
            lo   = m & ((1u << sh) - 1u);
            q    = m >> sh;
            half = 1u << (sh - 1);
            if (lo > half || (lo == half && (q & 1u))) ++q;
            return (uint8_t)(s | q);
        }
        {
            uint32_t q  = m >> 20;
            uint32_t lo = m & 0xfffffu;
            int ee;
            if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
            ee = e + 7;
            if (q == 0x8u) { q = 0; ++ee; }
            if (ee > 15 || (ee == 15 && q >= 7u)) return (uint8_t)(s | 0x7eu);
            return (uint8_t)(s | ((uint32_t)ee << 3) | q);
        }
    }
}

/* fp8 e5m2: 1-5-2, bias 15, IEEE-shaped (it has infinities and NaNs, unlike e4m3fn). It is the
 * top byte of an f16, so both directions go through the f16 path.
 *
 * Narrowing rounds to f16 and then drops the low byte. Every e5m2 tie point is an f16 value, so
 * the f16 step can move `f` onto a tie but never across one; when it lands on a tie, `f`'s side
 * of it decides, and only an exact tie goes to even. That makes the result one RTNE of `f`. */
static inline float rad_fp8e5m2_to_f32(uint8_t b) {
    return rad_f16_to_f32((uint16_t)((uint16_t)b << 8));
}

static inline uint8_t rad_f32_to_fp8e5m2(float f) {
    uint16_t h  = rad_f32_to_f16(f);
    uint16_t lo = (uint16_t)(h & 0xffu);
    uint16_t q  = (uint16_t)(h >> 8);
    /* A NaN keeps its top payload bits and stays quiet; rounding its low byte could carry it out
     * of the exponent into a zero of the other sign. */
    if ((h & 0x7fffu) > 0x7c00u) return (uint8_t)(q | 0x02u);
    if (lo == 0x80u && (h & 0x7c00u) != 0x7c00u) {
        /* Magnitudes of non-negative floats order as their bit patterns. */
        float t = rad_f16_to_f32(h);
        uint32_t uf, ut;
        memcpy(&uf, &f, 4);
        memcpy(&ut, &t, 4);
        uf &= 0x7fffffffu;
        ut &= 0x7fffffffu;
        if (uf != ut) return (uint8_t)(uf > ut ? q + 1u : q);
    }
    if (lo > 0x80u || (lo == 0x80u && (q & 1u))) ++q;
    return (uint8_t)q;
}

/* OCP FP4 e2m1: 1-2-1, bias 1, no infinity or NaN; all sixteen codes are numbers, max 6. */
static inline float rad_fp4e2m1_to_f32(uint8_t nib) {
    static const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    return (nib & 8u) ? -mag[nib & 7u] : mag[nib & 7u];
}

/* OCP FP6, both forms. No infinity or NaN; e2m3 has bias 1 and max 7.5, e3m2 bias 3 and max 28. */
static inline float rad_fp6_to_f32(uint8_t c, int ebits) {
    const int mbits = 5 - ebits, bias = ebits == 2 ? 1 : 3;
    const int e = (c >> mbits) & ((1 << ebits) - 1), m = c & ((1 << mbits) - 1);
    const float frac = (float)m / (float)(1 << mbits);
    const float pow2 = e >= bias ? (float)(1 << (e - bias)) : 1.0f / (float)(1 << (bias - e));
    const float mag = e == 0 ? frac / (float)(1 << (bias - 1)) : (1.0f + frac) * pow2;
    return (c & 0x20) ? -mag : mag;
}

/* The bit stream, for every element narrower than a byte or not a power of two wide.
 *
 * Element `i` of a b-bit row occupies bits [i*b, (i+1)*b) counted from bit 0 of `base`, lowest
 * first, so it may straddle a byte; the read touches only the bytes the element is in. Widths to
 * 16 bits. rad_encoding.h is where the order is defined. */
static inline uint32_t rad_load_bits(const void* base, int bits, int64_t i) {
    const uint8_t* b8 = (const uint8_t*)base;
    const int64_t o = i * (int64_t)bits;
    const int64_t byte = o >> 3;
    const int sh = (int)(o & 7);
    uint32_t w = b8[byte];
    if (sh + bits > 8)  w |= (uint32_t)b8[byte + 1] << 8;
    if (sh + bits > 16) w |= (uint32_t)b8[byte + 2] << 16;
    return (w >> sh) & ((1u << bits) - 1u);
}

static inline void rad_store_bits(void* base, int bits, int64_t i, uint32_t code) {
    uint8_t* b8 = (uint8_t*)base;
    const int64_t o = i * (int64_t)bits;
    int64_t byte = o >> 3;
    int sh = (int)(o & 7), left = bits;
    code &= (1u << bits) - 1u;
    while (left > 0) {
        const int take = 8 - sh < left ? 8 - sh : left;
        const uint8_t m = (uint8_t)(((1u << take) - 1u) << sh);
        b8[byte] = (uint8_t)((b8[byte] & ~m) | ((code << sh) & m));
        code >>= take;
        left -= take;
        sh = 0;
        ++byte;
    }
}

/* Signed widths from the table, sign-extended from the stream; 0 for a dtype that is not a
 * packed integer. */
static inline int rad_dtype_packed_int(uint32_t dt, int* is_signed) {
    switch (dt) {
        case RAD_I2: case RAD_I3: case RAD_I4: case RAD_I5: case RAD_I6: case RAD_I7:
            *is_signed = 1; return rad_dtype_bits(dt);
        case RAD_U1: case RAD_U2: case RAD_U3: case RAD_U4: case RAD_U5: case RAD_U6:
        case RAD_U7: case RAD_U9: case RAD_U10: case RAD_U11: case RAD_U12:
            *is_signed = 0; return rad_dtype_bits(dt);
        default:
            *is_signed = 0; return 0;
    }
}

/* OCP E8M0: a bare biased exponent, 2^(e-127), with 0xFF the NaN. Exact in f32 except at the two
 * ends, where 2^-127 is subnormal and still exact and 2^128 does not exist. */
static inline float rad_e8m0_to_f32(uint8_t e) {
    uint32_t u;
    float f;
    if (e == 0xffu) u = 0x7fc00000u;
    else if (e == 0u) u = 0x00400000u;              /* 2^-127, the one subnormal */
    else u = (uint32_t)e << 23;
    memcpy(&f, &u, 4);
    return f;
}

/* ================================================================== dtype-generic access
 *
 * Load or store element `i` of a plane as f32, whatever it is stored as. This is what a host
 * kernel and a converter both need, and the one place the dtype enum and the conversions above
 * meet. Returns 0 for a dtype it cannot read, which a caller must treat as a refusal, not a value.
 *
 * For sub-byte dtypes `i` counts elements from `base`, which is valid only within one row: rows
 * start on a byte (rad_encoding.h), so a caller crossing rows steps `base` by a row's bytes. */
static inline float rad_load_f32(const void* base, uint32_t dtype, int64_t i) {
    const uint8_t* b8 = (const uint8_t*)base;
    int sgn = 0;
    const int pb = rad_dtype_packed_int(dtype, &sgn);
    if (pb) {
        const uint32_t c = rad_load_bits(base, pb, i);
        return sgn ? (float)((int32_t)(c ^ (1u << (pb - 1))) - (int32_t)(1u << (pb - 1)))
                   : (float)c;
    }
    switch (dtype) {
        case RAD_FP4E2M1: return rad_fp4e2m1_to_f32((uint8_t)rad_load_bits(base, 4, i));
        case RAD_FP6E2M3: return rad_fp6_to_f32((uint8_t)rad_load_bits(base, 6, i), 2);
        case RAD_FP6E3M2: return rad_fp6_to_f32((uint8_t)rad_load_bits(base, 6, i), 3);
        case RAD_E8M0:    return rad_e8m0_to_f32(b8[i]);
        case RAD_U16:     return (float)((const uint16_t*)base)[i];
        case RAD_F32:    return ((const float*)base)[i];
        case RAD_F16:    return rad_f16_to_f32(((const uint16_t*)base)[i]);
        case RAD_BF16:   return rad_bf16_to_f32(((const uint16_t*)base)[i]);
        case RAD_F8E4M3: return rad_fp8e4m3_to_f32(((const uint8_t*)base)[i]);
        case RAD_F8E5M2: return rad_fp8e5m2_to_f32(((const uint8_t*)base)[i]);
        case RAD_I8:     return (float)((const int8_t*)base)[i];
        case RAD_U8:     return (float)((const uint8_t*)base)[i];
        case RAD_I16:    return (float)((const int16_t*)base)[i];
        case RAD_I32:    return (float)((const int32_t*)base)[i];
        case RAD_U32:    return (float)((const uint32_t*)base)[i];
        case RAD_I64:    return (float)((const int64_t*)base)[i];
        case RAD_BOOL:   return ((const uint8_t*)base)[i] ? 1.0f : 0.0f;
        default:         return 0.0f;
    }
}

/* Store the raw code of element `i` of a sub-byte or byte plane (the chosen bit pattern, masked to
 * the dtype's width), leaving neighbours in the same byte alone. A signed code is passed as its
 * value (-8..7 for i4) and stored two's complement. The quantisers' counterpart to rad_load_f32;
 * nothing rounds here. Returns 0 for a dtype wider than 16 bits. */
static inline int rad_store_code(void* base, uint32_t dtype, int64_t i, int code) {
    const int bits = rad_dtype_bits(dtype);
    if (bits < 1 || bits > 16) return 0;
    if (bits == 16) { ((uint16_t*)base)[i] = (uint16_t)code; return 1; }
    if (bits == 8)  { ((uint8_t*)base)[i]  = (uint8_t)code;  return 1; }
    rad_store_bits(base, bits, i, (uint32_t)code);
    return 1;
}

/* `v` rounded to the nearest integer, ties to even, and clamped to [lo, hi]; NaN is 0. Exact and
 * independent of the rounding mode: a float of magnitude 2^23 or more is already an integer, and
 * below that its fraction is exact in a double. */
static inline double rad_round_even_sat(float v, double lo, double hi) {
    double d = (double)v, r;
    if (v != v) return 0.0;
    if (d >= 8388608.0 || d <= -8388608.0) {
        r = d;
    } else {
        const long long t = (long long)d;        /* toward zero */
        const double    f = d - (double)t;
        long long       n = t;
        if (f > 0.5 || (f == 0.5 && (t & 1))) ++n;
        else if (f < -0.5 || (f == -0.5 && (t & 1))) --n;
        r = (double)n;
    }
    return r < lo ? lo : (r > hi ? hi : r);
}

/* Store `v` as element `i` of a byte-aligned plane. A float dtype rounds as the narrowings above
 * do; an integer dtype rounds to nearest, ties to even, and saturates to its range, NaN storing 0.
 * Sub-byte dtypes are stored with rad_store_code; any other dtype is left unwritten. */
static inline void rad_store_f32(void* base, uint32_t dtype, int64_t i, float v) {
    switch (dtype) {
        case RAD_F32:    ((float*)base)[i]    = v; break;
        case RAD_F16:    ((uint16_t*)base)[i] = rad_f32_to_f16(v); break;
        case RAD_BF16:   ((uint16_t*)base)[i] = rad_f32_to_bf16(v); break;
        case RAD_F8E4M3: ((uint8_t*)base)[i]  = rad_f32_to_fp8e4m3(v); break;
        case RAD_F8E5M2: ((uint8_t*)base)[i]  = rad_f32_to_fp8e5m2(v); break;
        case RAD_I8:  ((int8_t*)base)[i]  = (int8_t)rad_round_even_sat(v, -128.0, 127.0); break;
        case RAD_U8:  ((uint8_t*)base)[i] = (uint8_t)rad_round_even_sat(v, 0.0, 255.0); break;
        case RAD_U16:
            ((uint16_t*)base)[i] = (uint16_t)rad_round_even_sat(v, 0.0, 65535.0);
            break;
        case RAD_I16:
            ((int16_t*)base)[i] = (int16_t)rad_round_even_sat(v, -32768.0, 32767.0);
            break;
        case RAD_I32:
            ((int32_t*)base)[i] = (int32_t)rad_round_even_sat(v, -2147483648.0, 2147483647.0);
            break;
        case RAD_U32:
            ((uint32_t*)base)[i] = (uint32_t)rad_round_even_sat(v, 0.0, 4294967295.0);
            break;
        case RAD_I64: {
            /* 2^63 is the first float past the range, so it is compared, not clamped to. */
            const double r = rad_round_even_sat(v, -9223372036854775808.0, 9223372036854775808.0);
            ((int64_t*)base)[i] = r >= 9223372036854775808.0 ? INT64_MAX : (int64_t)r;
            break;
        }
        case RAD_BOOL:   ((uint8_t*)base)[i]  = v != 0.0f ? 1u : 0u; break;
        default:         break;
    }
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* RAD_PLUGIN_H */
