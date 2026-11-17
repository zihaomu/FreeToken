/* r4d_enc.h -- the encoding checks libr4d's relayouts make before rearranging a weight.
 *
 * A relayout that rearranged an encoding its kernel does not read would hand the kernel bytes
 * that decode to other numbers; the layout hook refuses such a declaration by name instead. Pure
 * host C++, like the rest of the declaration side.
 */
#pragma once
#include "rad_abi.h"

#include <cstring>

namespace r4d_enc {

/* The affine encoding a kernel reads: `codes`, a `scale` of dtype `scale` over [br x bc], an
 * integer zero of dtype `zero` when `zero` is not RAD_DT_INVALID, `transform` exactly ("" for
 * none), and no other plane. */
inline bool affine(const RadEncoding* e, uint32_t codes, uint32_t scale, int64_t br, int64_t bc,
                   uint32_t zero = RAD_DT_INVALID, const char* transform = "") {
    if (!e || !rad_enc_is(e, "affine")) return false;
    if (std::strncmp(e->transform, transform, RAD_ENC_STR) != 0) return false;
    if (e->n_planes != (zero != RAD_DT_INVALID ? 3 : 2)) return false;
    if (e->plane[0].dtype != codes || e->plane[0].block[1] != 1) return false;
    const int s = rad_enc_find(e, "scale");
    if (s < 0 || e->plane[s].dtype != scale || e->plane[s].kind != RAD_PLANE_TILED ||
        e->plane[s].block[0] != br || e->plane[s].block[1] != bc) return false;
    if (zero != RAD_DT_INVALID) {
        const int z = rad_enc_find(e, "zero");
        if (z < 0 || e->plane[z].dtype != zero || e->plane[z].block[0] != br ||
            e->plane[z].block[1] != bc) return false;
    }
    return true;
}

/* The same with a scalar codebook: `codes` indexing a `table` plane of `entries` values of dtype
 * `tab`, the `scale` as above, `transform` exactly, and no other plane. A declaration carries no
 * plane bytes, so a kernel that decodes one fixed table compares the plane's contents at
 * relayout. */
inline bool affine_table(const RadEncoding* e, uint32_t codes, uint32_t scale, int64_t br,
                         int64_t bc, uint32_t tab, int64_t entries, const char* transform = "") {
    if (!e || !rad_enc_is(e, "affine")) return false;
    if (std::strncmp(e->transform, transform, RAD_ENC_STR) != 0) return false;
    if (e->n_planes != 3) return false;
    if (e->plane[0].dtype != codes || e->plane[0].block[1] != 1) return false;
    const int s = rad_enc_find(e, "scale");
    if (s < 0 || e->plane[s].dtype != scale || e->plane[s].kind != RAD_PLANE_TILED ||
        e->plane[s].block[0] != br || e->plane[s].block[1] != bc) return false;
    const int t = rad_enc_find(e, "table");
    return t >= 0 && e->plane[t].dtype == tab && e->plane[t].kind == RAD_PLANE_TABLE &&
           e->plane[t].extent[0] == 1 && e->plane[t].extent[1] == entries;
}

/* affine_table's codebook with a second scale level: `scale.1` of dtype `scale2`, one value for
 * the whole tensor (libquant's block2=*x*). Four planes: codes, scale, scale.1, table. */
inline bool affine_table2(const RadEncoding* e, uint32_t codes, uint32_t scale, int64_t br,
                          int64_t bc, uint32_t scale2, uint32_t tab, int64_t entries,
                          const char* transform = "") {
    if (!e || !rad_enc_is(e, "affine")) return false;
    if (std::strncmp(e->transform, transform, RAD_ENC_STR) != 0) return false;
    if (e->n_planes != 4) return false;
    if (e->plane[0].dtype != codes || e->plane[0].block[1] != 1) return false;
    const int s = rad_enc_find(e, "scale");
    if (s < 0 || e->plane[s].dtype != scale || e->plane[s].kind != RAD_PLANE_TILED ||
        e->plane[s].block[0] != br || e->plane[s].block[1] != bc) return false;
    const int s2 = rad_enc_find(e, "scale.1");
    if (s2 < 0 || e->plane[s2].dtype != scale2 || e->plane[s2].kind != RAD_PLANE_TILED ||
        e->plane[s2].block[0] != 0 || e->plane[s2].block[1] != 0) return false;
    const int t = rad_enc_find(e, "table");
    return t >= 0 && e->plane[t].dtype == tab && e->plane[t].kind == RAD_PLANE_TABLE &&
           e->plane[t].extent[0] == 1 && e->plane[t].extent[1] == entries;
}

/* The role of selected plane i. */
inline const char* role(const RadEncoding* e, const int* sel, int i) {
    return (e && sel && sel[i] >= 0 && sel[i] < e->n_planes) ? e->plane[sel[i]].role : "";
}

/* Does the selection name exactly these roles, in this order? */
inline bool selects(const RadEncoding* e, const int* sel, int n, const char* a,
                    const char* b = nullptr) {
    const int want = b ? 2 : 1;
    if (n != want) return false;
    if (std::strncmp(role(e, sel, 0), a, RAD_ENC_STR) != 0) return false;
    return !b || std::strncmp(role(e, sel, 1), b, RAD_ENC_STR) == 0;
}

/* A selected plane's bytes, row r. */
inline const uint8_t* row(const RadTensor& t, int64_t r) {
    return (const uint8_t*)t.data + r * rad_dtype_bytes(t.dtype, t.shape[1]);
}

/* The planes' extents match [rows, cols]. */
inline bool dims(const RadTensor& t, int64_t rows, int64_t cols) {
    return t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols;
}

}  /* namespace r4d_enc */
