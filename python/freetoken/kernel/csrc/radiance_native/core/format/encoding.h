/* encoding.h -- the core's half of a weight encoding: decoding to f32, and cutting a plane for
 * tensor parallelism. The encoding itself is abi/rad_encoding.h; spec.md §4.1.
 *
 * The core decodes its two schemes, "plain" and "affine", under every ABI transform; a quantiser
 * that defines another scheme supplies its decoder (abi/rad_quant.h). The oracle, rad-convert's
 * error measurement and re-quantisation all decode through this one implementation.
 */
#pragma once
#include "../rad_internal.h"
#include "rad_encoding.h"
#include "rad_quant.h"

#include <string>

namespace rad {

/* One whole plane of a weight, canonical: row-major, `row_bytes` a row. */
struct PlaneView {
    const void* data = nullptr;
    int64_t     rows = 0, cols = 0;   /* elements */
    int64_t     row_bytes = 0;
};

/* The planes of an encoding over a [rows, cols] weight, in the encoding's order. */
struct EncodedView {
    RadEncoding enc{};
    int64_t     rows = 0, cols = 0;   /* the logical view */
    PlaneView   plane[RAD_ENC_MAX_PLANES];
};

/* Fill `v`'s plane geometry for a weight of [rows, cols] in `e`, pointing plane i at `data[i]`. */
void enc_view(const RadEncoding& e, int64_t rows, int64_t cols, const void* const* data,
              EncodedView* v);

/* Decode logical rows [row0, row0 + n) into `out`, f32 [n, cols]. RAD_OK; RAD_E_UNSUPPORTED for
 * a scheme nobody decodes (`why` names it); RAD_E_SHAPE when a transform does not fit the row;
 * RAD_E_FORMAT for a malformed encoding or a code outside its codebook (rather than a read past
 * the table). */
int enc_decode_rows(const EncodedView& v, int64_t row0, int64_t n, float* out,
                    std::string* why = nullptr);

/* Decoders for schemes the core does not define (RadQuantDecodeFn), registered by the plugin
 * loader per quantiser; enc_decode_rows asks them in order for any scheme but plain and affine. */
void enc_add_decoder(RadQuantDecodeFn fn);
void enc_remove_decoder(RadQuantDecodeFn fn);

/* Unnormalised Walsh-Hadamard transform of `n` values in place, `n` a power of two. The butterfly
 * keeps (a + b) in the lower index and (a - b) in the upper: H_n in natural (Sylvester) order, as
 * libr4d's r4d_fwht_wave computes and an "fwhtN" encoding names. H_n is symmetric and
 * H_n H_n = n I. */
void fwht_inplace(float* v, int64_t n);

/* The canonical spelling of an encoding (rad_enc_format), as a string. */
std::string enc_name(const RadEncoding& e);

/* ================================================================== a selection, as a reader reads it
 *
 * The `n_sel` planes of `e` a declaration selects (RadWeightDecl::planes; plane `sel[k]` at
 * `data[k]`, dense [rows[k], cols[k]] in its own elements) as `n` elements of dtype `dt` at `out`:
 * the tensor a reference implementation reads in place of the stored form. A single plane converts
 * element by element; codes with their scalar codebook become table entries; a scale with its
 * chain's higher levels becomes their product; the whole encoding is decoded. RAD_OK, or
 * RAD_E_SHAPE with `why` when the planes do not form that tensor (an incomplete encoding, or
 * extents disagreeing with `n`), or the decoder's refusal. */
int enc_selection_logical(const RadEncoding& e, const int* sel, int n_sel, const int64_t* rows,
                          const int64_t* cols, const void* const* data, uint32_t dt, int64_t n,
                          void* out, std::string* why);

/* ================================================================== slicing */
/* A rectangle of a plane, in that plane's elements. */
struct PlaneRect {
    int64_t row0 = 0, rows = 0;
    int64_t col0 = 0, cols = 0;
};

/* The part of plane `p` covering logical rows [r0, r1) and columns [c0, c1) of a [rows, cols]
 * weight. Refused, with `why`, when a bound would split one of the plane's blocks (a scale shared
 * by two ranks' rows belongs to neither). The weight's end never splits: a last block may be
 * short. */
int enc_plane_rect(const RadEncPlane& p, int64_t rows, int64_t cols, int64_t r0, int64_t r1,
                   int64_t c0, int64_t c1, PlaneRect* out, std::string* why);

}  /* namespace rad */
