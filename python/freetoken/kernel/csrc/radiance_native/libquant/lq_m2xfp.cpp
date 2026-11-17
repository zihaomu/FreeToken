/* lq_m2xfp.cpp -- the `m2xfp4` quantiser: M2XFP, a metadata-augmented Microscaling format
 * (Weiming Hu et al., "M2XFP: A Metadata-Augmented Microscaling Data Format for Efficient
 * Low-bit Quantization", ASPLOS '26, arXiv 2601.19213), weight side only -- the paper's
 * Sg-EM scheme, subgroup-level extra mantissa with an adaptive shared scale.
 *
 * An OCP MXFP4 group of 32 columns carries, beside its E8M0 exponent, one metadata byte: four
 * 2-bit extra mantissas, one a subgroup of 8 columns. Subgroup s (columns [8s, 8s+8)) refines the
 * group's shared scale to (1 + k/4) . 2^E with k its two bits, so a group decodes as
 *
 *   w[j] = e2m1(codes[j]) . 2^(scale[g] - 127) . (1 + k_s/4),   g = j/32, s = (j%32)/8,
 *
 * 4.5 bits an element. The three planes sit apart, as the paper's memory organisation keeps them
 * (its §5.2): codes fp4_e2m1 [1 x 1] as a bit stream along the row (abi/rad_encoding.h), the scale
 * e8m0 [1 x 32] and the metadata u8 [1 x 32] one byte a group, the metadata packed little-endian,
 * subgroup s in bits [2s, 2s+2). The scheme is "m2xfp4", which the core does not decode; the
 * quantiser's own decode hook reads it back (abi/rad_quant.h).
 *
 * The search is the paper's §4.4.2, Eqs. 3-4, hierarchical because a subgroup's k is chosen
 * independently once the bias is fixed and only the bias couples the subgroups. E0 is the OCP
 * rule floor(log2(amax / 4)); the group's scale candidates are 2^(E0+b), b in {-1, 0, 1}, and
 * each subgroup's are (1 + k/4) times the group's, k in {0, 1, 2, 3}. For a bias, every
 * subgroup takes the k of least squared error against its own 8 columns; the group takes the
 * bias of least squared error over the four. adaptive=0 pins the bias at 0 and searches the
 * subgroups only, which is where a container that wants the plain exponent still wants the
 * refinement; wide=1 widens adaptive's sweep to b in {-2, ..., 2}, still lowest-first so a tie
 * keeps the lowest bias. The bias costs nothing stored: it is absorbed into the exponent byte.
 *
 * Conventions shared with the MXFP4 grid (`rtn codes=fp4_e2m1 group=32 scale=e8m0`,
 * lq_grid.cpp's rule=mx), whose containers this format is a superset of:
 *  - E0 is taken as the f32 exponent field of amax less 129, exactly as rule=mx takes it (for
 *    every normal amax that is floor(log2(amax)) - 2); the paper's reference takes floor(log2
 *    of an amax clamped to 1e-8, which differs only below the f32 normals;
 *  - a zero group stores scale byte 0 and zero codes, rule=mx's exponent-floor clamp; the
 *    reference's clamp gives a different byte whose codes and decode are the same zeros;
 *  - codes round to nearest, ties to the even code -- f32_to_e2m1, the same rounding the grid
 *    quantises with (and the reference's nudged midpoints);
 *  - the exponent byte is clamped to [0, 254]; 255 is E8M0's NaN;
 *  - a final group shorter than 32 columns is its own group over the columns it has, the way
 *    the grid quantises a row's tail (the reference requires a whole number of groups).
 *
 * Under the (b=0, k=0) candidate every group reproduces rule=mx's scale and codes exactly, so
 * an adaptive weight's error is that plain MXFP4's or better, group by group.
 *
 * The error summed is in double; the per-candidate dequantised value is computed exactly as the
 * decode computes it (the code through rad_fp4e2m1_to_f32, times the candidate's scale), so what
 * the search minimises is what a reader gets. Rows are independent, so the stored bytes do not
 * depend on the thread count.
 */
#include "lq_common.h"
#include "lq_quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace lq {

namespace {

const char kScheme[] = "m2xfp4";

const RadQuantOption kM2xfpOptions[] = {
    { "adaptive", RAD_P_INT, "1",
      "1: search the group's exponent bias b in {-1, 0, 1} beside the subgroups (the paper's "
      "Eq. 4); 0: the bias pinned at 0, the subgroup search still on" },
    { "wide", RAD_P_INT, "0",
      "1: widen adaptive's bias search to b in {-2, -1, 0, 1, 2}; 0: {-1, 0, 1}. Ignored at "
      "adaptive=0, which pins the bias at 0 whatever the sweep width" },
};

/* One subgroup's least squared error over k, with k itself through `best` when asked: the
 * subgroups of one bias are independent (the paper's Eq. 4 inner argmin). The dequantised value
 * is the code through the same table the decode reads, times the candidate's scale. */
double subgroup_sse(const float* w, int64_t b0, int64_t b1, float s, int* best) {
    double best_sse = -1.0;
    int best_k = 0;
    for (int k = 0; k < 4; ++k) {
        const float sm = s * (1.0f + 0.25f * (float)k);
        double sse = 0.0;
        for (int64_t i = b0; i < b1; ++i) {
            const float dq = rad_fp4e2m1_to_f32(f32_to_e2m1(w[i] / sm)) * sm;
            const double d = (double)dq - (double)w[i];
            sse += d * d;
        }
        if (best_sse < 0.0 || sse < best_sse) {
            best_sse = sse;
            best_k = k;
        }
    }
    if (best) *best = best_k;
    return best_sse;
}

/* One row's group of [c0, c1): the OCP exponent of its amax, the bias of least subgroup error,
 * then the winning bias's subgroups, whose k go into the metadata byte and whose codes and
 * exponent byte go into the planes. Ties keep the first candidate, the k and then the b of
 * lowest index, as the paper's argmin does. */
void quant_group(const float* w, int64_t c0, int64_t c1, int adaptive, int wide, uint8_t* codes,
                 uint8_t* scale, uint8_t* meta, uint32_t codes_dt) {
    float amax = 0.0f;
    for (int64_t c = c0; c < c1; ++c) {
        const float a = w[c] < 0.0f ? -w[c] : w[c];
        if (a > amax) amax = a;
    }
    /* E0 = floor(log2(amax / 4)), as rule=mx takes it: the exponent field less emax + 127. A
     * zero group keeps -127, the field's underflow, whose byte clamps to 0 like rule=mx's. */
    int e0 = -127;
    if (amax > 0.0f) {
        uint32_t u;
        std::memcpy(&u, &amax, 4);
        e0 = (int)((u >> 23) & 0xFFu) - 127 - 2;
    }

    /* The bias: every subgroup's best k given it, summed over the group (the paper averages the
     * four, the same argmin). adaptive=0 sees only b = 0; wide sweeps b over {-2, ..., 2}
     * instead of {-1, 0, 1}, lowest first either way, so a tie keeps the lowest bias as the
     * paper's argmin does. */
    const int lo = adaptive ? (wide ? -2 : -1) : 0;
    const int hi = adaptive ? (wide ? 2 : 1) : 0;
    int best_b = 0;
    double best_err = -1.0;
    for (int b = lo; b <= hi; ++b) {
        const float s = rad_e8m0_to_f32((uint8_t)clampi(e0 + b + 127, 0, 254));
        double err = 0.0;
        for (int64_t c = c0; c < c1; c += 8)
            err += subgroup_sse(w, c, c + 8 < c1 ? c + 8 : c1, s, nullptr);
        if (best_err < 0.0 || err < best_err) {
            best_err = err;
            best_b = b;
        }
    }

    /* The winning bias's scale, its subgroups' k, and the codes those k were chosen against. */
    const int e8 = clampi(e0 + best_b + 127, 0, 254);
    const float s = rad_e8m0_to_f32((uint8_t)e8);
    uint8_t mb = 0;
    for (int sl = 0; sl < 4; ++sl) {
        const int64_t c = c0 + 8 * sl;
        if (c >= c1) break;                      /* a short final group's absent subgroups */
        int k = 0;
        subgroup_sse(w, c, c + 8 < c1 ? c + 8 : c1, s, &k);
        mb |= (uint8_t)k << (2 * sl);
        const float sm = s * (1.0f + 0.25f * (float)k);
        for (int64_t j = c; j < c + 8 && j < c1; ++j)
            rad_store_code(codes, codes_dt, j, (int)f32_to_e2m1(w[j] / sm));
    }
    *scale = (uint8_t)e8;
    *meta = mb;
}

int m2xfp_encoding(const RadParam*, int, const RadQuantWeight* w, RadEncoding* out) {
    if (declines(w)) return RAD_E_UNSUPPORTED;
    if (!w || w->cols <= 0 || w->rows <= 0) return RAD_E_SHAPE;
    rad_enc_clear(out);
    rad_enc_copy_str(out->scheme, kScheme);
    /* The three streams of the paper's §5.2: the codes, the group's exponent, the group's
     * metadata -- a final group shorter than 32 columns covers the columns it has. The codes and
     * the scale carry the MXFP4 grid's dtypes, whose bytes these are; the exponent byte is an
     * E8M0 biased exponent, never the integer a bare u8 would read as. */
    rad_enc_add_plane(out, "codes", RAD_FP4E2M1, 1, 1);
    rad_enc_add_plane(out, "scale", RAD_E8M0, 1, 32);
    rad_enc_add_plane(out, "meta", RAD_U8, 1, 32);
    return RAD_OK;
}

/* Groups are column-local and rows independent, so a call may take any rows: one, as the MXFP4
 * grid's row block is, and the converter streams by its band, not by the format. */
int64_t m2xfp_row_block(const RadParam*, int, const RadQuantWeight*) { return 1; }

int m2xfp_quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* src, int64_t row0,
                   int64_t rows, void* const* planes) {
    if (!w || w->cols <= 0 || w->rows <= 0) return RAD_E_SHAPE;
    if (row0 < 0 || rows < 0 || row0 + rows > w->rows) return RAD_E_SHAPE;
    const int adaptive = (int)rad_param_geti(o, n_o, "adaptive", 1);
    const int wide = (int)rad_param_geti(o, n_o, "wide", 0);
    const int64_t C = w->cols;
    const int64_t ng = (C + 31) / 32;                 /* groups a row, the last one maybe short */
    uint8_t* codes = (uint8_t*)planes[0];
    uint8_t* scale = (uint8_t*)planes[1];
    uint8_t* meta = (uint8_t*)planes[2];
    const int64_t codes_row = rad_enc_row_bytes(RAD_FP4E2M1, C);
    const uint32_t codes_dt = RAD_FP4E2M1;           /* the plane's dtype: same four bits as u4 */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < rows; ++r) {
        const float* wr = src + r * C;
        uint8_t* cr = codes + r * codes_row;
        uint8_t* sr = scale + r * ng;
        uint8_t* mr = meta + r * ng;
        for (int64_t g = 0; g < ng; ++g)
            quant_group(wr, g * 32, std::min(g * 32 + 32, C), adaptive, wide, cr, sr + g, mr + g,
                        codes_dt);
    }
    return RAD_OK;
}

/* The encoding's own decode (abi/rad_quant.h): the equation of the scheme, the only place the
 * core cannot follow it. RAD_E_UNSUPPORTED for an encoding this quantiser did not write. */
int m2xfp_decode(const RadEncoding* e, const RadQuantWeight* w, const void* const* planes,
                 int64_t row0, int64_t rows, float* out) {
    if (!rad_enc_is(e, kScheme) || e->n_planes != 3) return RAD_E_UNSUPPORTED;
    const RadEncPlane* p = e->plane;
    if (p[0].dtype != RAD_FP4E2M1 || p[0].kind != RAD_PLANE_TILED || p[0].block[0] != 1 ||
        p[0].block[1] != 1 || std::strcmp(p[0].role, "codes") != 0)
        return RAD_E_UNSUPPORTED;
    for (int i = 1; i < 3; ++i)
        if (p[i].dtype != (i == 1 ? RAD_E8M0 : RAD_U8) || p[i].kind != RAD_PLANE_TILED ||
            p[i].block[0] != 1 || p[i].block[1] != 32 ||
            (i == 1 ? std::strcmp(p[i].role, "scale") != 0
                    : std::strcmp(p[i].role, "meta") != 0))
            return RAD_E_UNSUPPORTED;
    if (!w || w->cols <= 0) return RAD_E_SHAPE;

    const int64_t C = w->cols;
    const int64_t ng = (C + 31) / 32;
    const uint8_t* codes = (const uint8_t*)planes[0];
    const uint8_t* scale = (const uint8_t*)planes[1];
    const uint8_t* meta = (const uint8_t*)planes[2];
    const int64_t codes_row = rad_enc_row_bytes(RAD_FP4E2M1, C);
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t* cr = codes + (row0 + r) * codes_row;
        const uint8_t* sr = scale + (row0 + r) * ng;
        const uint8_t* mr = meta + (row0 + r) * ng;
        float* orow = out + r * C;
        for (int64_t j = 0; j < C; ++j) {
            const uint8_t c = (uint8_t)rad_load_bits(cr, 4, j);
            const uint8_t mb = mr[j >> 5];
            const int k = (mb >> (2 * ((j & 31) >> 3))) & 3;
            orow[j] = rad_fp4e2m1_to_f32(c) * rad_e8m0_to_f32(sr[j >> 5]) *
                      (1.0f + 0.25f * (float)k);
        }
    }
    return RAD_OK;
}

}  // namespace

const RadQuantizerInfo& m2xfp_info() {
    static const RadQuantizerInfo q = {
        "m2xfp4",
        "M2XFP: OCP MXFP4 whose group of 32 also carries one metadata byte of four 2-bit "
        "subgroup extra mantissas, each a subgroup of 8 columns refining the shared exponent to "
        "(1 + k/4) . 2^E, the exponent itself searched over a bias {-1, 0, 1}, widened to "
        "{-2, ..., 2} by wide=1, and absorbed into its byte (adaptive=0 pins the bias at 0); "
        "4.5 bits an element, weights only, adapted from arXiv 2601.19213",
        kM2xfpOptions, (int)(sizeof kM2xfpOptions / sizeof kM2xfpOptions[0]),
        m2xfp_encoding, m2xfp_row_block, m2xfp_quantize, m2xfp_decode,
    };
    return q;
}

}  // namespace lq
