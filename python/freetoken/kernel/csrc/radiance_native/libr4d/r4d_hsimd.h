/* r4d_hsimd.h -- libr4d's host-side hot loops (the load-time relayouts), dispatched by CPU feature
 * as the core's util/simd.h is: the plugin runs on machines it was not built on, so no
 * -march=native. r4d_hsimd_kernels.cpp is compiled once per level into separate objects with their
 * own -m flags (cmake/RadSimdLevels.cmake); r4d_hsimd.cpp (baseline, no -m flags) picks one at
 * first use through __builtin_cpu_supports, which also checks the OS has enabled the register
 * state.
 *
 * Every function here moves bits and has one exact result: all levels agree bit for bit on every
 * input, which tests/relayout_iqr_test.cpp and tests/relayout_parity_test.cpp check by forcing
 * each level (force_level) against the scalar level and a reference written in the test.
 *
 * Levels, as util/simd.h: 0 scalar, 1 SSE4.1/4.2, 2 AVX (integer work in 128-bit halves), 3 AVX2,
 * 4 AVX-512 F/BW/DQ/VL. On a non-x86 build only the scalar level exists.
 */
#pragma once

#include <cstdint>

namespace r4d_hsimd {

enum Level { kScalar = 0, kSse4 = 1, kAvx = 2, kAvx2 = 3, kAvx512 = 4, kLevels = 5 };

/* The highest level this CPU and OS can execute, computed once. */
int         detect_level();
/* The level in use (detect_level() unless forced). */
int         level();
const char* level_name(int lvl);
/* Forces a level, clamped to detect_level(); for tests only. Returns the level now in use. */
int         force_level(int lvl);

/* MoE w4 codes between canonical and kernel nibble order, `bytes` (a multiple of four) from `s` to
 * `d`: in every dword, canonical nibble i (element i, two a byte, the first low) goes to nibble
 * 2i (i < 4) or 2(i - 4) + 1, a perfect shuffle of the dword's halves (r4d_layout.cpp); `forward`
 * false undoes it. */
void moew4_permute(const uint8_t* s, uint8_t* d, int64_t bytes, bool forward);

/* One 16-row tile of an IQR trellis weight into its `nkb` stored units (r4d_iqr.h, "the codes:
 * lane segments"): `rows` is the tile's first row of B-bit codes (2 <= B <= 7, LSB first), rows
 * `rb` bytes apart; unit kb, at `out` + kb * 256 * B, holds for each lane l and column
 * e = 0..3 the 16 codes of column 128 kb + 4 l + e bottom-up (row 15 first). Reads only the
 * tile's own bytes. */
void iqr_trellis_tile(const uint8_t* rows, int64_t rb, int B, int64_t nkb, uint8_t* out);

/* One 16-row tile of a w2a8 weight's u2 codes into its `kpb` stored units (r4d_layout.cpp,
 * w2_slot): `rows` is the tile's first row of codes (four a byte, the first in the low bits), rows
 * `rb` bytes apart; unit kb, at `out` + 512 kb, holds for half lh and row r, at
 * 16 (16 lh + r), four dwords d, whose field 4j + b + 2h is the code of column
 * 128 kb + 16 (2d + h) + 8b + 4 lh + j. Reads only the tile's own bytes. */
void w2a8_tile(const uint8_t* rows, int64_t rb, int64_t kpb, uint8_t* out);

/* A 16-row tile of dwords transposed: out[16 c + r] = in[r * stride + c] for r < 16, c < cols
 * (dword units; `in` rows `stride` dwords apart). MXFP4's stored codes are this (r4d_layout.cpp,
 * mx_slot): each 16-code group's two 4-byte halves go to the n-tile's slots whole. */
void dword_transpose16(const uint32_t* in, int64_t stride, int64_t cols, uint32_t* out);

/* A rows x cols byte matrix transposed: out[c * rows + r] = in[r * cols + c]. MXFP4's exponents go
 * from canonical [N][K/32] to the stored [K/32][N] so. */
void byte_transpose(const uint8_t* in, int64_t rows, int64_t cols, uint8_t* out);

/* One 16-row tile of a block-FP8 or int8 weight into fragment order (r4d_fp8_frag.h): `rows` is
 * the tile's first row of byte codes, rows K bytes apart; the `ksuper` 1024-byte units at `out`
 * hold, for each 64-byte K step, lane l's 32 bytes at 32 l: row l % 16's qwords 2t + l / 16 for
 * t = 0..3. Reads only the tile's own bytes. */
void fp8_frag_tile(const uint8_t* rows, int64_t K, int64_t ksuper, uint8_t* out);

}  // namespace r4d_hsimd
