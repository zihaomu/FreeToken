/* rad_planes.h -- the joined form of a weight operand that takes several planes, for kernels that
 * read the planes as the container stores them. spec.md §4.3.
 *
 * A weight can be declared over several planes of one stored tensor (RadWeightDecl::planes:
 * "scale,zero", "codes,table", or every plane of an fp8 lm head), and it reaches a kernel as one
 * operand, so something has to say where each plane lies. A kernel that wants the planes
 * rearranged for its own reads says so in its layout hook (rad_abi.h, RadLayoutFn). One that reads
 * them unchanged uses this form instead:
 *
 *     [header][entry 0][entry 1]...   padded to RAD_PLANES_ALIGN
 *     plane 0                         dense rows, padded to RAD_PLANES_ALIGN
 *     plane 1 ...
 *
 * The header and the entries describe each selected plane (its role, dtype, block, this rank's
 * extents and where its bytes start), so a launch finds its planes from the operand alone, without
 * the encoding. Each plane is the canonical bytes (rad_encoding.h), unchanged: the conversion is a
 * copy and its inverse is one too.
 *
 * A library implements its layout, relayout and unrelayout hooks with the three functions below
 * and reads at launch with rad_planes_find. Every library that uses this form stores the same bytes
 * under the same tag, so two such libraries serving different bands of one op read one stored
 * weight. A selection of one plane is not joined: rad_planes_layout declines it, and the weight is
 * stored as it is.
 *
 * Pure C, header-only, like the rest of abi/.
 */
#ifndef RAD_PLANES_H
#define RAD_PLANES_H

#include "rad_abi.h"
#include "rad_encoding.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAD_PLANES_TAG   "rad.planes"
#define RAD_PLANES_MAGIC 0x534e4c50u   /* "PLNS" */
enum { RAD_PLANES_ALIGN = 64 };

/* One selected plane. `offset` counts from the start of the joined bytes. */
typedef struct RadPlanesEntry {
    char     role[RAD_ENC_STR];
    uint32_t dtype;
    uint32_t kind;          /* RAD_PLANE_TILED | RAD_PLANE_TABLE */
    int64_t  block[2];      /* the encoding's block (TILED) or extent (TABLE) */
    int64_t  rows, cols;    /* this rank's extents of the plane, in its own elements */
    int64_t  row_bytes;     /* rad_dtype_bytes(dtype, cols): every row starts on a byte */
    int64_t  offset;
} RadPlanesEntry;

typedef struct RadPlanesHeader {
    uint32_t magic;
    int32_t  n_planes;
    char     transform[RAD_ENC_STR];   /* the encoding's, for a kernel that checks it */
    int64_t  bytes;                    /* the whole joined form, padding included */
} RadPlanesHeader;

static inline int64_t rad_planes_round(int64_t n) {
    return (n + RAD_PLANES_ALIGN - 1) / RAD_PLANES_ALIGN * RAD_PLANES_ALIGN;
}

/* The joined form's entries for the planes a layout hook was handed, and its total size; -1 when
 * the planes cannot be joined (no planes, more than RAD_ENC_MAX_PLANES, an index outside the
 * encoding, or a plane with no byte size). */
static inline int64_t rad_planes_plan(const RadEncoding* enc, const int* sel,
                                      const RadTensor* planes, int n_planes, RadPlanesEntry* e) {
    int64_t off;
    int i;
    if (!enc || !sel || !planes || n_planes < 1 || n_planes > RAD_ENC_MAX_PLANES) return -1;
    off = rad_planes_round((int64_t)sizeof(RadPlanesHeader) +
                           (int64_t)n_planes * (int64_t)sizeof(RadPlanesEntry));
    for (i = 0; i < n_planes; ++i) {
        const RadEncPlane* p;
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return -1;
        p = &enc->plane[sel[i]];
        memset(&e[i], 0, sizeof e[i]);
        memcpy(e[i].role, p->role, RAD_ENC_STR);
        e[i].dtype = planes[i].dtype;
        e[i].kind = p->kind;
        e[i].block[0] = p->kind == RAD_PLANE_TABLE ? p->extent[0] : p->block[0];
        e[i].block[1] = p->kind == RAD_PLANE_TABLE ? p->extent[1] : p->block[1];
        e[i].rows = planes[i].rank >= 2 ? planes[i].shape[0] : 1;
        e[i].cols = planes[i].rank >= 2 ? planes[i].shape[1]
                                        : (planes[i].rank == 1 ? planes[i].shape[0] : 0);
        e[i].row_bytes = rad_dtype_bytes(e[i].dtype, e[i].cols);
        if (e[i].rows < 0 || e[i].row_bytes <= 0) return -1;
        e[i].offset = off;
        off = rad_planes_round(off + e[i].rows * e[i].row_bytes);
    }
    return off;
}

/* RadLayoutFn's answer for the joined form. RAD_E_UNSUPPORTED for a single plane, which is stored
 * as it is. The stored dtype is the first plane's, so a launch that dispatches on the operand's
 * dtype still sees the codes' (or the scale's) type; the stored shape is the byte count. */
static inline int rad_planes_layout(const RadEncoding* enc, const int* sel, const RadTensor* planes,
                                    int n_planes, RadLayout* out) {
    RadPlanesEntry e[RAD_ENC_MAX_PLANES];
    int64_t bytes;
    if (n_planes == 1) return RAD_E_UNSUPPORTED;
    bytes = rad_planes_plan(enc, sel, planes, n_planes, e);
    if (bytes < 0 || !out) return RAD_E_SHAPE;
    memset(out, 0, sizeof *out);
    out->tag = RAD_PLANES_TAG;
    out->dtype = planes[0].dtype;
    out->rank = 1;
    out->shape[0] = bytes;
    out->align = RAD_PLANES_ALIGN;
    out->bytes = bytes;
    return RAD_OK;
}

/* RadRelayoutFn: the header, the entries and each plane's rows, copied. The planes are dense, as
 * the loader hands them. */
static inline int rad_planes_relayout(const RadEncoding* enc, const int* sel,
                                      const RadTensor* planes, int n_planes, void* dst,
                                      int64_t dst_bytes) {
    RadPlanesEntry e[RAD_ENC_MAX_PLANES];
    RadPlanesHeader h;
    uint8_t* d = (uint8_t*)dst;
    int i;
    const int64_t bytes = rad_planes_plan(enc, sel, planes, n_planes, e);
    if (bytes < 0 || !dst || dst_bytes < bytes) return RAD_E_SHAPE;
    memset(d, 0, (size_t)bytes);
    memset(&h, 0, sizeof h);
    h.magic = RAD_PLANES_MAGIC;
    h.n_planes = n_planes;
    memcpy(h.transform, enc->transform, RAD_ENC_STR);
    h.bytes = bytes;
    memcpy(d, &h, sizeof h);
    memcpy(d + sizeof h, e, (size_t)n_planes * sizeof e[0]);
    for (i = 0; i < n_planes; ++i) {
        if (!planes[i].data && e[i].rows * e[i].row_bytes > 0) return RAD_E_INVAL;
        if (e[i].rows * e[i].row_bytes > 0)
            memcpy(d + e[i].offset, planes[i].data, (size_t)(e[i].rows * e[i].row_bytes));
    }
    return RAD_OK;
}

/* RadUnrelayoutFn: the planes back out of the joined form, into the buffers `planes` carry. */
static inline int rad_planes_unrelayout(const RadEncoding* enc, const int* sel, const void* src,
                                        int64_t src_bytes, const RadTensor* planes, int n_planes) {
    RadPlanesEntry e[RAD_ENC_MAX_PLANES];
    const uint8_t* s = (const uint8_t*)src;
    int i;
    const int64_t bytes = rad_planes_plan(enc, sel, planes, n_planes, e);
    if (bytes < 0 || !src || src_bytes < bytes) return RAD_E_SHAPE;
    for (i = 0; i < n_planes; ++i) {
        if (!planes[i].data && e[i].rows * e[i].row_bytes > 0) return RAD_E_INVAL;
        if (e[i].rows * e[i].row_bytes > 0)
            memcpy(planes[i].data, s + e[i].offset, (size_t)(e[i].rows * e[i].row_bytes));
    }
    return RAD_OK;
}

/* ================================================================== reading one at launch */

/* The joined form's header, or null when `base` does not hold one. */
static inline const RadPlanesHeader* rad_planes_header(const void* base) {
    const RadPlanesHeader* h = (const RadPlanesHeader*)base;
    if (!h || h->magic != RAD_PLANES_MAGIC || h->n_planes < 1 ||
        h->n_planes > RAD_ENC_MAX_PLANES)
        return 0;
    return h;
}

/* Entry `i`, or null. */
static inline const RadPlanesEntry* rad_planes_entry(const void* base, int i) {
    const RadPlanesHeader* h = rad_planes_header(base);
    if (!h || i < 0 || i >= h->n_planes) return 0;
    return (const RadPlanesEntry*)((const uint8_t*)base + sizeof(RadPlanesHeader)) + i;
}

/* The plane with this role: its bytes, and its entry through `out` when given; null when the
 * joined form has no such plane. */
static inline const void* rad_planes_find(const void* base, const char* role,
                                          const RadPlanesEntry** out) {
    const RadPlanesHeader* h = rad_planes_header(base);
    int i;
    if (!h || !role) return 0;
    for (i = 0; i < h->n_planes; ++i) {
        const RadPlanesEntry* e = rad_planes_entry(base, i);
        if (strncmp(e->role, role, RAD_ENC_STR) == 0) {
            if (out) *out = e;
            return (const uint8_t*)base + e->offset;
        }
    }
    return 0;
}

#ifdef __cplusplus
}   /* extern "C" */
#endif
#endif /* RAD_PLANES_H */
