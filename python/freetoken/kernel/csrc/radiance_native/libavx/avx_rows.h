/* avx_rows.h -- the quantised embedding gather, split in two: the row source, compiled once at
 * the baseline, and the row decode, compiled per ISA level.
 *
 * embed_lookup_q over an E4M3 table into bf16 serves the n-gram table, tens of gigabytes in a
 * mapped container that is never loaded, so fetching the rows is the whole cost. That half (a
 * shared row cache, direct reads for misses, a reader thread one step ahead) is the same at every
 * level and lives in avx_ngram.cpp, with one row cache per process. Decoding codes to bf16 differs
 * per level, so each level's kernel passes its own decoder. */
#ifndef RAD_AVX_ROWS_H
#define RAD_AVX_ROWS_H

#include "rad_abi.h"

#include <cstdint>

namespace avx {

/* dst[i][0, n) = lut[src[i][0, n)] for every i < count: E4M3 codes through a table of all 256
 * codes already scaled and rounded to bf16. */
typedef void (*RowDecode)(const uint8_t* const* src, uint16_t* const* dst, int64_t count,
                          int64_t n, const uint16_t* lut);

/* One call's gather, validated by the caller: `ids` holds M row ids, `tab` n_vocab rows of n_embd
 * elements -- E4M3 codes, or bf16 as stored -- `row_bytes` apart, `out` M rows of n_embd bf16
 * `x_ld` elements apart. The row cache and the reads deal in `row_bytes`; the decode in elements.
 * `ahead` holds n_ahead ids a later call will gather, or is null. rank and world_size are the
 * call's. */
struct RowGather {
    const int32_t* ids;
    int64_t        M;
    const uint8_t* tab;
    int64_t        n_vocab, n_embd, voff;
    int64_t        row_bytes;
    float          scale;
    uint16_t*      out;
    int64_t        x_ld;
    const int32_t* ahead;
    int64_t        n_ahead;
    int            rank, world_size;
};

/* The gather, through the row cache when a file backs the table and from the table's memory when
 * none does. Returns a RAD_* status; on anything but RAD_OK the output is not complete. */
int row_gather(const RowGather& g, RowDecode decode);

}  /* namespace avx */

/* The table must be an E4M3 plane with one scale, or plain bf16, stored as is (avx_ngram.cpp). */
extern "C" int avx_layout_ngram(const RadParam*, int, int, const RadEncoding*, const int*,
                                const RadTensor*, int, RadLayout*);

/* Refuses, with the reason on stderr, when this process may not create an io_uring ring and so
 * cannot read the table's rows (avx_ngram.cpp). */
extern "C" int avx_init_ngram(const RadParam*, int, int, int, void**);

#endif /* RAD_AVX_ROWS_H */
