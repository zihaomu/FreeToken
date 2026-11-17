/* lq_ggml_port.h -- llama.cpp's quantisers for its legacy, K- and I-quant formats, ported to C++.
 *
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===================================================================== Overview
 *
 * The block layouts, value tables and quantisation searches of llama.cpp's ggml library at commit
 * 06938ac12, so that the `ggml` quantiser (lq_ggml.cpp) chooses the same codes and scales as
 * llama.cpp for the same weight and imatrix:
 *
 *   ggml/src/ggml-common.h   the block structs below, and the tables in lq_ggml_tables.cpp:
 *                            kmask_iq2xs, ksigns_iq2xs, iq2xxs_grid, iq2xs_grid, iq2s_grid,
 *                            iq3xxs_grid, iq3s_grid, kvalues_iq4nl, iq1s_grid
 *   ggml/src/ggml-impl.h     the f32 <-> f16 conversion (Maratyszcza's, round to nearest even)
 *   ggml/src/ggml-quants.c   everything in lq_ggml_port.cpp: best_index_int8, nearest_int,
 *                            make_qx_quants, make_q3_quants, make_qkx2_quants, make_qkx3_quants,
 *                            make_qp_quants, get_scale_min_k4, quantize_row_<t>_ref and _impl for
 *                            the legacy and K types, iq2xs_init_impl and iq3xs_init_impl with
 *                            their kgrid tables, iq2/iq3_find_best_neighbour,
 *                            iq1_find_best_neighbour2, iq1_sort_helper, quantize_row_iq4_nl_impl,
 *                            the quantize_row_<t>_impl of every I type, and the quantize_<t>
 *                            entry points below
 *   ggml/src/ggml.c          ggml_quantize_init's choice of table per type, and
 *                            ggml_quantize_requires_imatrix's list
 *
 * The numerics match llama.cpp bit for bit: the same float operations in the same order, the same
 * f16 rounding, the same `nearest_int` magic-number round and the same qsort comparators, so each
 * quantize_<t> writes the bytes llama.cpp writes (pinned in tests/quant_ggml_test.cpp). The port
 * departs from upstream only where C++ requires it (a cast from void*), where a library inside a
 * converter must (a failed assertion prints to stderr and aborts instead of calling ggml_abort;
 * the Oops diagnostics go to stderr, not stdout), and in the one-time table build (std::call_once
 * instead of ggml's critical section; the tables live for the process and are never freed).
 *
 * The match is with llama.cpp as its own build compiles ggml-quants.c: into ggml-base with no
 * -march, so without fused multiply-add. libquant's -ffp-contract=off holds the port there on
 * every target. A llama.cpp built with FMA contraction (-march=native on x86-64, or any aarch64
 * build, where FMA is baseline) rounds inside the searches and writes different bytes for most
 * types.
 *
 * Each quantize_<t> keeps llama.cpp's dispatch: legacy and K types take the _ref round-to-nearest
 * path when `quant_weights` is null and the _impl search otherwise; I types always take _impl,
 * and three of them assert `quant_weights` is set (ggml_quantize_requires_imatrix).
 * `quant_weights` is the imatrix's per-column importance, n_per_row values shared by every row.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace lq::ggml {

typedef uint16_t ggml_half;
typedef uint16_t ggml_fp16_t;

/* The subset of ggml_type this port quantises. The values are ggml.h's, so a type read out of a
 * GGUF names the same thing here. */
enum ggml_type {
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_IQ1_M   = 29,
};

/* ================================================================== ggml-common.h's blocks */

constexpr int QK_K = 256;
constexpr int K_SCALE_SIZE = 12;

constexpr int QK4_0 = 32;
struct block_q4_0 {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
};
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

constexpr int QK4_1 = 32;
struct block_q4_1 {
    ggml_half d;           // delta
    ggml_half m;           // min
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
};
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_half) + QK4_1 / 2, "wrong q4_1 block size/padding");

constexpr int QK5_0 = 32;
struct block_q5_0 {
    ggml_half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
};
static_assert(sizeof(block_q5_0) == sizeof(ggml_half) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

constexpr int QK5_1 = 32;
struct block_q5_1 {
    ggml_half d;           // delta
    ggml_half m;           // min
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
};
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_half) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

constexpr int QK8_0 = 32;
struct block_q8_0 {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
};
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
struct block_q2_K {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    ggml_half d;             // super-block scale for quantized scales
    ggml_half dmin;          // super-block scale for quantized mins
};
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_half) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight
struct block_q3_K {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
};
static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
struct block_q4_K {
    ggml_half d;                  // super-block scale for quantized scales
    ggml_half dmin;               // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
};
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");

// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
struct block_q5_K {
    ggml_half d;                  // super-block scale for quantized scales
    ggml_half dmin;               // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
};
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
struct block_q6_K {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
};
static_assert(sizeof(block_q6_K) == sizeof(ggml_half) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

// (Almost) "true" 2-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 2.0625 bpw because of the 16-bit scale for each block of 256.
struct block_iq2_xxs {
    ggml_half d;
    uint16_t qs[QK_K/8];
};
static_assert(sizeof(block_iq2_xxs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t), "wrong iq2_xxs block size/padding");

// 2.3125 bpw quants
struct block_iq2_xs {
    ggml_half d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
};
static_assert(sizeof(block_iq2_xs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t) + QK_K/32, "wrong iq2_xs block size/padding");

// 2.5625 bpw quants
struct block_iq2_s {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
};
static_assert(sizeof(block_iq2_s) == sizeof(ggml_half) + QK_K/4 + QK_K/16, "wrong iq2_s block size/padding");

// (Almost) "true" 3-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 3.0625 bpw because of the 16-bit scale for each block of 256.
struct block_iq3_xxs {
    ggml_half d;
    uint8_t qs[3*QK_K/8];
};
static_assert(sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3*(QK_K/8), "wrong iq3_xxs block size/padding");

// 3.4375 bpw
constexpr int IQ3S_N_SCALE = QK_K/64;
struct block_iq3_s {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
};
static_assert(sizeof(block_iq3_s) == sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");

// 1.5625 bpw
struct block_iq1_s {
    ggml_half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
};
static_assert(sizeof(block_iq1_s) == sizeof(ggml_half) + QK_K/8 + QK_K/16, "wrong iq1_s block size/padding");

// 1.75 bpw
struct block_iq1_m {
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits
    uint8_t  qh[QK_K/16];     // grid index, high 3 bits + grid shift bit (for two groups of 8)
    uint8_t  scales[QK_K/32]; // 3-bit block scales (4-bit if QK_K == 64)
};
static_assert(sizeof(block_iq1_m) == QK_K/8 + QK_K/16 + QK_K/32, "wrong iq1_m block size/padding");

// Used by IQ1_M quants
typedef union {
    ggml_half f16;
    uint16_t  u16;
} iq1m_scale_t;

// Non-linear quants
constexpr int QK4_NL = 32;
struct block_iq4_nl {
    ggml_half d;
    uint8_t qs[QK4_NL/2];
};
static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2, "wrong iq4_nl block size/padding");

struct block_iq4_xs {
    ggml_half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
};
static_assert(sizeof(block_iq4_xs) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/64 + QK_K/2, "wrong iq4_xs block size/padding");

/* ================================================================== ggml-common.h's tables */

constexpr int   NGRID_IQ1S = 2048;
constexpr float IQ1S_DELTA = 0.125f;
constexpr float IQ1M_DELTA = 0.125f;

extern const uint8_t  kmask_iq2xs[8];
extern const uint8_t  ksigns_iq2xs[128];
extern const uint64_t iq2xxs_grid[256];
extern const uint64_t iq2xs_grid[512];
extern const uint64_t iq2s_grid[1024];
extern const uint32_t iq3xxs_grid[256];
extern const uint32_t iq3s_grid[512];
extern const int8_t   kvalues_iq4nl[16];
extern const uint64_t iq1s_grid[NGRID_IQ1S];

/* ================================================================== the quantisers */

/* Build the search tables an I type needs (the lattice, its index map and each off-grid point's
 * nearest neighbours) once per process, on whichever thread asks first; a no-op for other types.
 * Every quantize_<t> below requires it to have run for its type. */
void ggml_quantize_init(enum ggml_type type);

/* ggml_quantize_requires_imatrix: the types whose search has no path without `quant_weights`. */
bool ggml_quantize_requires_imatrix(enum ggml_type type);

/* Bytes of one row of `ne` values; `ne` a multiple of the type's block. */
size_t ggml_row_size(enum ggml_type type, int64_t ne);

/* llama.cpp's quantize_<t>: `nrow` rows of `n_per_row` values from `src` into `dst`, returning
 * the bytes written. Rows are independent. */
size_t quantize_q4_0(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
size_t quantize_q4_1(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
size_t quantize_q5_0(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
size_t quantize_q5_1(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
size_t quantize_q8_0(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
size_t quantize_q2_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
size_t quantize_q3_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
size_t quantize_q4_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
size_t quantize_q5_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
size_t quantize_q6_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                     const float * quant_weights);
size_t quantize_iq2_xxs(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                        const float * quant_weights);
size_t quantize_iq2_xs(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                       const float * quant_weights);
size_t quantize_iq2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                      const float * quant_weights);
size_t quantize_iq3_xxs(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                        const float * quant_weights);
size_t quantize_iq3_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                      const float * quant_weights);
size_t quantize_iq1_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                      const float * quant_weights);
size_t quantize_iq1_m(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                      const float * quant_weights);
size_t quantize_iq4_nl(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                       const float * quant_weights);
size_t quantize_iq4_xs(const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                       const float * quant_weights);

}  /* namespace lq::ggml */
