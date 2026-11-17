/* rad_encoding.h -- what a stored weight's numbers are, independent of how any kernel wants them
 * arranged. spec.md §4.1.
 *
 * An encoding is the code type, the codebook the codes index (if any), the grouping of the scales,
 * zero points and minimums over the weight, and the transform applied before quantisation. It is a
 * property of the model: choosing one is lossy and can be expensive (GPTQ needs a calibration
 * run), and changing one means quantising again. A layout (fragment order, a scale carried in each
 * row's tail, a nibble order that suits one unpack) is a property of a kernel, lossless and cheap,
 * applied when the model loads (rad_abi.h). A container holds encodings only, so a kernel library
 * can change any layout without a reconvert.
 *
 * An encoding is data rather than an enum value, so a quantiser plugin can invent one without the
 * core knowing it. The core can size, slice and place any encoding from this struct alone, and
 * decodes every encoding of the one scheme it defines, "affine". That scheme covers the common
 * formats: per-channel and grouped integers of any width, block FP8, MXFP4/6/8, NVFP4, llama.cpp's
 * legacy, K- and I-quants, AWQ's column scales, GPTQ's act-order permutation, Hadamard and Givens
 * rotations.
 *
 * ===================================================================== PLANES
 *
 * An encoding is up to RAD_ENC_MAX_PLANES planes, each a dense row-major array of one core dtype.
 * A plane is one of two kinds:
 *
 *   TILED   it covers the weight. A weight of logical shape [d0, ..., dn] is seen as [rows, cols]
 *           with rows = d0 * ... * d(n-1) and cols = dn (a rank-1 weight is one row); a TILED
 *           plane whose block is [br, bc] has ceil(rows/br) x ceil(cols/bc) elements, element
 *           (i, j) covering logical rows [i*br, (i+1)*br) and columns [j*bc, (j+1)*bc). A block
 *           extent of 0 is the whole extent, so a per-tensor scale is one element and a
 *           per-column scale is a block of [0 x 1].
 *   TABLE   it does not: a codebook, a lattice grid, a rotation's parameters. Its extents are its
 *           own, [extent0 x extent1], and every slice of the weight carries all of it.
 *
 * Sub-byte and odd-width elements are a bit stream along the row, lowest bits first: element j of a
 * b-bit row occupies bits [j*b, (j+1)*b) from bit 0 of the row's first byte, so two 4-bit codes
 * share a byte with element 0 in bits 0-3, and a 3-bit code may straddle a byte. Every row starts
 * on a byte. A kernel that wants its nibbles permuted for its own unpack does so in its relayout.
 *
 * ===================================================================== THE AFFINE SCHEME
 *
 * The planes are found by role, and only "codes" is required. A weight element decodes as
 *
 *   q  = code                        codes is TILED; its block is [1 x 1] -- one code a value -- or
 *                                    [1 x V] with a "grid", one code a V-vector
 *   q  = grid[code][k]               grid  TABLE [entries x V]: a vector codebook (the IQ2/IQ3/IQ1
 *                                    lattices); value k of the code's V-vector
 *   q  = table[code]                 table TABLE [1 x entries]: a scalar codebook (IQ4_NL)
 *   q  = -q if the sign bit is set   signs TILED u1 [1 x 1], one bit a value
 *   q  = q - zero                    zero  TILED; an integer or real offset in the code's domain
 *   v  = q * scale * scale.1 * ...   scale, scale.1, ... TILED: every scale level, multiplied
 *   v  = v - min * min.1 * ...       min, min.1, ... TILED: a minimum in the value's domain
 *   w  = v . T^-1                    the transform, per row, over the columns (below)
 *
 * so NVFP4 is fp4 codes, an e4m3 scale per [1 x 16] and an f32 scale.1 per tensor; Q4_K is u4
 * codes, a u6 scale and u6 min per [1 x 32] each times an f16 per [1 x 256]; IQ2_XXS is u8 codes
 * per [1 x 8] into a 256 x 8 grid with a sign bit a value and two scale levels. Any other scheme
 * is a quantiser's own, and that quantiser decodes it (rad_quant.h).
 *
 * ===================================================================== TRANSFORMS
 *
 * `transform` names what was done to each row before it was quantised: the stored row is v = w . T,
 * so a kernel computes <w, x> = <v, x . T^-T> and transforms its activation, never the weight.
 *
 *   ""         none.
 *   "fwhtN"    T = H_N / N block-diagonally over groups of N columns, H_N the unnormalised
 *              Sylvester Hadamard (entries +-1, H H = N I). Decoding applies H_N once (w = H v);
 *              the activation is rotated by the unnormalised H_N with nothing folded at runtime:
 *              the 1/N lives in the stored scale.
 *   "perm"     T is a column permutation: TABLE "t.perm" i32 [1 x cols], v[j] = w[perm[j]] --
 *              GPTQ's act-order, which quantises the columns in a different order than stored.
 *   "givens"   T = D . G_1 . ... . G_S: D = diag(t.scale) when TABLE "t.scale" f32 [1 x cols]
 *              is present, and G_s a stage of disjoint Givens rotations -- TABLE "t.pairs" i32
 *              [S x cols] lists stage s's pairs as (pairs[s][2p], pairs[s][2p+1]), TABLE "t.angle"
 *              f32 [S x cols/2] their angles, and a row vector's pair (a, b) at angle t becomes
 *              (a cos t + b sin t, b cos t - a sin t). ParoQuant's scaled pairwise rotation.
 *              A weight whose row parts were rotated independently (a fused gate_up, a fused
 *              q|k|v: each projection has its own rotation) carries P parts: TABLE "t.part" i32
 *              [1 x P] holds the end row (exclusive) of each part, and the other tables stack the
 *              parts in order -- t.pairs and t.angle S rows a part (P*S rows), t.scale one row a
 *              part (P rows). Without "t.part" the weight is one part. The stages of a part are
 *              applied in table order, and a part's pairs address columns of the whole row.
 *              TABLE "t.dense" f16 [P * cols/128 x 128*128], where a weight carries it, is the same
 *              rotation composed: row p * (cols/128) + g holds part p's stages over the 128 columns
 *              of group g as one matrix, element j * 128 + i the output j for the unit input e_i,
 *              composed exactly (f64) and rounded to f16. It is stored because a kernel that
 *              applies the rotation as a matrix must not compute one at load (rad_abi.h).
 *
 * Pure C, header-only, like the rest of abi/.
 */
#ifndef RAD_ENCODING_H
#define RAD_ENCODING_H

#include "rad_types.h"

#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { RAD_ENC_MAX_PLANES = 8, RAD_ENC_STR = 24 };

enum { RAD_PLANE_TILED = 0, RAD_PLANE_TABLE = 1 };

typedef struct RadEncPlane {
    char     role[RAD_ENC_STR];   /* "codes", "scale", "scale.1", "zero", "min", "grid", "t.perm" */
    uint32_t dtype;               /* a core dtype -- never a plugin-private one */
    uint32_t kind;                /* RAD_PLANE_TILED | RAD_PLANE_TABLE */
    int64_t  block[2];            /* TILED: logical rows x cols one element covers; 0 = whole */
    int64_t  extent[2];           /* TABLE: the plane's own rows x cols */
} RadEncPlane;

typedef struct RadEncoding {
    char        scheme[RAD_ENC_STR];      /* "plain" | "affine" | a quantiser's own */
    char        transform[RAD_ENC_STR];   /* "" | "fwhtN" | "perm" | "givens" */
    int32_t     n_planes;
    int32_t     reserved;
    RadEncPlane plane[RAD_ENC_MAX_PLANES];
} RadEncoding;

/* ================================================================== building one */

static inline void rad_enc_copy_str(char* dst, const char* src) {
    size_t n = src ? strlen(src) : 0;
    if (n >= RAD_ENC_STR) n = RAD_ENC_STR - 1;
    memset(dst, 0, RAD_ENC_STR);
    if (n) memcpy(dst, src, n);
}

static inline void rad_enc_clear(RadEncoding* e) { memset(e, 0, sizeof *e); }

/* Append a TILED plane. Returns its index, or -1 when the encoding is full. */
static inline int rad_enc_add_plane(RadEncoding* e, const char* role, uint32_t dtype,
                                    int64_t block_rows, int64_t block_cols) {
    RadEncPlane* p;
    if (e->n_planes >= RAD_ENC_MAX_PLANES) return -1;
    p = &e->plane[e->n_planes];
    memset(p, 0, sizeof *p);
    rad_enc_copy_str(p->role, role);
    p->dtype = dtype;
    p->kind = RAD_PLANE_TILED;
    p->block[0] = block_rows;
    p->block[1] = block_cols;
    return e->n_planes++;
}

/* Append a TABLE plane of its own [rows x cols]. */
static inline int rad_enc_add_table(RadEncoding* e, const char* role, uint32_t dtype,
                                    int64_t rows, int64_t cols) {
    const int i = rad_enc_add_plane(e, role, dtype, 0, 0);
    if (i < 0) return i;
    e->plane[i].kind = RAD_PLANE_TABLE;
    e->plane[i].extent[0] = rows;
    e->plane[i].extent[1] = cols;
    return i;
}

/* A weight stored as it is: one plane of `dtype`, one element a value. */
static inline RadEncoding rad_enc_plain(uint32_t dtype) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "plain");
    rad_enc_add_plane(&e, "codes", dtype, 1, 1);
    return e;
}

/* value = code * scale, one scale per [block_rows x block_cols]. Further planes go on with
 * rad_enc_add_plane / rad_enc_add_table, and a transform with
 * rad_enc_copy_str(e.transform, ...). */
static inline RadEncoding rad_enc_affine(uint32_t codes, uint32_t scale, int64_t block_rows,
                                         int64_t block_cols) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "codes", codes, 1, 1);
    rad_enc_add_plane(&e, "scale", scale, block_rows, block_cols);
    return e;
}

/* ================================================================== reading one */

static inline int rad_enc_is(const RadEncoding* e, const char* scheme) {
    return e && strncmp(e->scheme, scheme, RAD_ENC_STR) == 0;
}

/* The index of the plane with this role, or -1. */
static inline int rad_enc_find(const RadEncoding* e, const char* role) {
    int i;
    if (!e || !role) return -1;
    for (i = 0; i < e->n_planes; ++i)
        if (strncmp(e->plane[i].role, role, RAD_ENC_STR) == 0) return i;
    return -1;
}

/* The plane with this role, or null. */
static inline const RadEncPlane* rad_enc_plane(const RadEncoding* e, const char* role) {
    const int i = rad_enc_find(e, role);
    return i < 0 ? 0 : &e->plane[i];
}

/* The index of level `level` of a multiplied chain -- "scale", "scale.1", "scale.2" -- or -1. */
static inline int rad_enc_find_level(const RadEncoding* e, const char* base, int level) {
    char r[RAD_ENC_STR];
    if (level == 0) return rad_enc_find(e, base);
    snprintf(r, sizeof r, "%s.%d", base, level);
    return rad_enc_find(e, r);
}

/* Does `role` belong to the chain `base` ("scale" -> scale, scale.1, ...)? */
static inline int rad_enc_role_in_chain(const char* role, const char* base) {
    const size_t n = strlen(base);
    int lv = 0;
    if (strncmp(role, base, n) != 0) return 0;
    if (role[n] == 0) return 1;
    return role[n] == '.' && sscanf(role + n + 1, "%d", &lv) == 1 && lv > 0;
}

/* The order N of an "fwhtN" transform, 0 when the transform is not one, -1 when it is malformed. */
static inline int64_t rad_enc_fwht(const RadEncoding* e) {
    long long n = 0;
    if (!e || strncmp(e->transform, "fwht", 4) != 0) return 0;
    if (sscanf(e->transform + 4, "%lld", &n) != 1 || n < 2 || (n & (n - 1)) != 0) return -1;
    return (int64_t)n;
}

/* The number of independently rotated row parts of a "givens" weight: the width of its "t.part"
 * table, or 1 without one. 0 when the table is malformed. */
static inline int64_t rad_enc_givens_parts(const RadEncoding* e) {
    const int i = rad_enc_find(e, "t.part");
    if (i < 0) return 1;
    return e->plane[i].kind == RAD_PLANE_TABLE && e->plane[i].extent[0] == 1 &&
           e->plane[i].extent[1] >= 1 ? e->plane[i].extent[1] : 0;
}

/* Byte-for-byte equality. Valid because the strings are always zero-padded, so memcmp sees only
 * meaningful bytes. */
static inline int rad_enc_equal(const RadEncoding* a, const RadEncoding* b) {
    return a && b && memcmp(a, b, sizeof *a) == 0;
}

/* ================================================================== geometry */

/* The [rows, cols] a logical shape is seen as. */
static inline void rad_enc_view(uint32_t rank, const int64_t* shape, int64_t* rows,
                                int64_t* cols) {
    uint32_t i;
    int64_t r = 1;
    if (rank == 0) { *rows = 1; *cols = 1; return; }
    for (i = 0; i + 1 < rank; ++i) r *= shape[i];
    *rows = r;
    *cols = shape[rank - 1];
}

/* A plane's element extents over a [rows, cols] weight. */
static inline void rad_enc_plane_dims(const RadEncPlane* p, int64_t rows, int64_t cols,
                                      int64_t* prows, int64_t* pcols) {
    if (p->kind == RAD_PLANE_TABLE) {
        *prows = p->extent[0];
        *pcols = p->extent[1];
        return;
    }
    *prows = p->block[0] > 0 ? (rows + p->block[0] - 1) / p->block[0] : 1;
    *pcols = p->block[1] > 0 ? (cols + p->block[1] - 1) / p->block[1] : 1;
}

/* Bytes of one row of `n` elements: every row starts on a byte. 0 for a dtype the core does not
 * size, which an encoding may never hold. */
static inline int64_t rad_enc_row_bytes(uint32_t dtype, int64_t n) {
    return rad_dtype_bytes(dtype, n);
}

static inline int64_t rad_enc_plane_bytes(const RadEncPlane* p, int64_t rows, int64_t cols) {
    int64_t pr, pc;
    rad_enc_plane_dims(p, rows, cols, &pr, &pc);
    return pr * rad_enc_row_bytes(p->dtype, pc);
}

/* How many values one code stands for: 1, or V for a vector code into a grid. */
static inline int64_t rad_enc_code_width(const RadEncoding* e) {
    return (e && e->n_planes > 0 && e->plane[0].block[1] > 1) ? e->plane[0].block[1] : 1;
}

/* Is the encoding well formed? 1 if so.
 *
 * Every plane has a role and a core dtype, the "codes" plane comes first, is TILED and covers one
 * row of one value -- or of V values when a grid of width V is there to expand it -- and the
 * planes the affine scheme reads have the kinds it reads them as. A scheme the core does not
 * define is checked for the first two only: its planes are its quantiser's business. */
static inline int rad_enc_valid(const RadEncoding* e) {
    int i, g;
    int64_t v;
    const char* t;
    if (!e || e->n_planes < 1 || e->n_planes > RAD_ENC_MAX_PLANES || !e->scheme[0]) return 0;
    for (i = 0; i < e->n_planes; ++i) {
        const RadEncPlane* p = &e->plane[i];
        if (!p->role[0] || rad_dtype_bits(p->dtype) == 0) return 0;
        if (p->kind == RAD_PLANE_TILED && (p->block[0] < 0 || p->block[1] < 0)) return 0;
        if (p->kind == RAD_PLANE_TABLE && (p->extent[0] < 1 || p->extent[1] < 1)) return 0;
        if (p->kind != RAD_PLANE_TILED && p->kind != RAD_PLANE_TABLE) return 0;
        if (rad_enc_find(e, p->role) != i) return 0;              /* roles are unique */
    }
    if (strncmp(e->plane[0].role, "codes", RAD_ENC_STR) != 0) return 0;
    if (e->plane[0].kind != RAD_PLANE_TILED || e->plane[0].block[0] != 1) return 0;
    if (rad_enc_is(e, "plain"))
        return e->n_planes == 1 && !e->transform[0] && e->plane[0].block[1] == 1;
    if (!rad_enc_is(e, "affine")) return e->plane[0].block[1] >= 1;

    /* nothing beyond the codes is "plain", and one weight has one spelling */
    if (e->n_planes == 1 && !e->transform[0]) return 0;
    v = e->plane[0].block[1];
    g = rad_enc_find(e, "grid");
    if (v < 1) return 0;
    if (g >= 0) {
        if (e->plane[g].kind != RAD_PLANE_TABLE || e->plane[g].extent[1] != v) return 0;
    } else if (v != 1) {
        return 0;
    }
    if (rad_enc_find(e, "table") >= 0 && (g >= 0 || e->plane[rad_enc_find(e, "table")].kind
                                                     != RAD_PLANE_TABLE)) return 0;
    for (i = 1; i < e->n_planes; ++i) {
        const RadEncPlane* p = &e->plane[i];
        const int tiled = rad_enc_role_in_chain(p->role, "scale") ||
                          rad_enc_role_in_chain(p->role, "min") ||
                          !strncmp(p->role, "zero", RAD_ENC_STR) ||
                          !strncmp(p->role, "signs", RAD_ENC_STR);
        const int table = !strncmp(p->role, "grid", RAD_ENC_STR) ||
                          !strncmp(p->role, "table", RAD_ENC_STR) || !strncmp(p->role, "t.", 2);
        if (!tiled && !table) return 0;                           /* a role nothing reads */
        if (tiled && p->kind != RAD_PLANE_TILED) return 0;
        if (table && p->kind != RAD_PLANE_TABLE) return 0;
    }
    /* a chain level exists only above the one before it */
    for (i = 1; i < e->n_planes; ++i) {
        int lv = 0;
        const char* r = e->plane[i].role;
        const char* base = rad_enc_role_in_chain(r, "scale") ? "scale"
                         : rad_enc_role_in_chain(r, "min") ? "min" : 0;
        if (!base || !strchr(r, '.')) continue;
        sscanf(strchr(r, '.') + 1, "%d", &lv);
        if (rad_enc_find_level(e, base, lv - 1) < 0) return 0;
    }
    if (rad_enc_find(e, "signs") >= 0 &&
        (e->plane[rad_enc_find(e, "signs")].dtype != RAD_U1 ||
         e->plane[rad_enc_find(e, "signs")].block[0] != 1 ||
         e->plane[rad_enc_find(e, "signs")].block[1] != 1)) return 0;

    t = e->transform;
    if (!t[0]) return 1;
    if (!strncmp(t, "fwht", 4)) return rad_enc_fwht(e) > 0;
    if (!strcmp(t, "perm")) return rad_enc_find(e, "t.perm") > 0;
    if (!strcmp(t, "givens")) {
        const int ip = rad_enc_find(e, "t.pairs"), ia = rad_enc_find(e, "t.angle");
        const int is = rad_enc_find(e, "t.scale");
        const int64_t np = rad_enc_givens_parts(e);
        if (ip <= 0 || ia <= 0 || np < 1) return 0;
        if (e->plane[ip].kind != RAD_PLANE_TABLE || e->plane[ia].kind != RAD_PLANE_TABLE) return 0;
        if (e->plane[ip].extent[0] % np || e->plane[ia].extent[0] != e->plane[ip].extent[0]) return 0;
        if (e->plane[ia].extent[1] * 2 != e->plane[ip].extent[1]) return 0;
        if (is > 0 && (e->plane[is].kind != RAD_PLANE_TABLE || e->plane[is].extent[0] != np ||
                       e->plane[is].extent[1] != e->plane[ip].extent[1])) return 0;
        {
            const int id = rad_enc_find(e, "t.dense");
            if (id > 0 && (e->plane[id].kind != RAD_PLANE_TABLE || e->plane[id].dtype != RAD_F16 ||
                           e->plane[ip].extent[1] % 128 ||
                           e->plane[id].extent[0] != np * (e->plane[ip].extent[1] / 128) ||
                           e->plane[id].extent[1] != 128 * 128)) return 0;
        }
        return 1;
    }
    return 0;
}

/* ================================================================== the canonical spelling
 *
 * What a person, a log line and rad-info read. Kernels read the struct.
 *
 *   plain                     <dtype>                             bf16
 *   affine, codes and scales  <codes>*<scale>[RxC]*<scale.1>[RxC] fp4_e2m1*fp8_e4m3[1x16]*f32[*x*]
 *     with an integer zero    ...-<zero>[RxC]                     u2*f16[1x128]-u8[1x128]
 *   anything else             <scheme>:<role>=<dtype><shape>,...  affine:codes=u4[1x1],...
 *   and a transform           .../<transform>                     i4*bf16[1x128]/fwht128
 *
 * A block extent of 0 -- the whole extent -- is written `*`, and a TABLE plane's shape is written
 * `{RxC}`. Returns the length written, like snprintf; the output is always terminated. */
static inline int rad_enc_shape_str(const RadEncPlane* p, char* buf, size_t n) {
    char r[24], c[24];
    if (p->kind == RAD_PLANE_TABLE)
        return snprintf(buf, n, "{%lldx%lld}", (long long)p->extent[0], (long long)p->extent[1]);
    if (p->block[0] > 0) snprintf(r, sizeof r, "%lld", (long long)p->block[0]);
    else snprintf(r, sizeof r, "*");
    if (p->block[1] > 0) snprintf(c, sizeof c, "%lld", (long long)p->block[1]);
    else snprintf(c, sizeof c, "*");
    return snprintf(buf, n, "[%sx%s]", r, c);
}

/* Codes, a scale chain and at most an integer zero: the shape that has a short spelling. */
static inline int rad_enc_is_simple_affine(const RadEncoding* e) {
    int i, lv;
    if (!rad_enc_is(e, "affine") || e->plane[0].block[1] != 1) return 0;
    for (i = 1; i < e->n_planes; ++i) {
        const char* r = e->plane[i].role;
        if (!rad_enc_role_in_chain(r, "scale") && strncmp(r, "zero", RAD_ENC_STR) != 0 &&
            strncmp(r, "t.", 2) != 0) return 0;
    }
    /* and the scales in level order, so the spelling is read the way it is multiplied */
    for (i = 1, lv = 0; i < e->n_planes; ++i) {
        if (!rad_enc_role_in_chain(e->plane[i].role, "scale")) continue;
        if (rad_enc_find_level(e, "scale", lv++) != i) return 0;
    }
    return lv > 0;
}

static inline int rad_enc_format(const RadEncoding* e, char* buf, size_t n) {
    char shp[64];
    size_t at = 0;
    int i;
#define RAD_ENC_PUT(...)                                                              \
    do {                                                                              \
        int w_ = snprintf(buf + (at < n ? at : n), at < n ? n - at : 0, __VA_ARGS__); \
        if (w_ > 0) at += (size_t)w_;                                                 \
    } while (0)
    if (!buf || n == 0) return 0;
    buf[0] = 0;
    if (!e || e->n_planes < 1) {
        RAD_ENC_PUT("(none)");
    } else if (rad_enc_is(e, "plain") && e->n_planes == 1) {
        RAD_ENC_PUT("%s", rad_dtype_name(e->plane[0].dtype));
    } else if (rad_enc_is_simple_affine(e)) {
        RAD_ENC_PUT("%s", rad_dtype_name(e->plane[0].dtype));
        for (i = 1; i < e->n_planes; ++i) {
            if (!rad_enc_role_in_chain(e->plane[i].role, "scale")) continue;
            rad_enc_shape_str(&e->plane[i], shp, sizeof shp);
            RAD_ENC_PUT("*%s%s", rad_dtype_name(e->plane[i].dtype), shp);
        }
        i = rad_enc_find(e, "zero");
        if (i > 0) {
            rad_enc_shape_str(&e->plane[i], shp, sizeof shp);
            RAD_ENC_PUT("-%s%s", rad_dtype_name(e->plane[i].dtype), shp);
        }
    } else {
        RAD_ENC_PUT("%s:", e->scheme);
        for (i = 0; i < e->n_planes; ++i) {
            rad_enc_shape_str(&e->plane[i], shp, sizeof shp);
            RAD_ENC_PUT("%s%s=%s%s", i ? "," : "", e->plane[i].role,
                        rad_dtype_name(e->plane[i].dtype), shp);
        }
    }
    if (e && e->transform[0]) RAD_ENC_PUT("/%s", e->transform);
#undef RAD_ENC_PUT
    return (int)at;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* RAD_ENCODING_H */
