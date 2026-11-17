/* r4d_bf16_gemm_rows.h -- host-side row chunks for the BF16 WMMA GEMMs.
 *
 * The operands' dimensions use the kernels' signed-int ABI. Each launch uses 32-bit element
 * offsets for A and W, but successive launches rebase A and Y with 64-bit pointers. Thus M*K
 * need not fit in 32 bits. W is reused whole and still must fit; a wider weight needs a different
 * addressing implementation. This is a matrix-shape constraint, not a deployment token budget.
 */
#ifndef R4D_BF16_GEMM_ROWS_H
#define R4D_BF16_GEMM_ROWS_H

#include <algorithm>
#include <climits>
#include <cstdint>

static inline int r4d_bf16_gemm_chunk_rows(int64_t M, int64_t N, int64_t K, int row_tile,
                                           int col_tile) {
    if (M < 1 || M > INT_MAX || N < 1 || N > INT_MAX || K < 16 || K > INT_MAX || K % 16 ||
        row_tile < 1 || col_tile < 1)
        return 0;
    constexpr int64_t offsets = UINT32_MAX;
    if (N > offsets / K) return 0;
    const int64_t gx = (N + col_tile - 1) / col_tile;
    /* Bound each launch's row grid, not the full matrix. The tiled kernel flattens block IDs
     * in unsigned arithmetic, so the product must fit too. No input is rejected for exceeding
     * this workgroup budget: it becomes another launch on the same stream. */
    const int64_t gy = std::min<int64_t>(65535, offsets / gx);
    int64_t rows = std::min(offsets / K, gy * row_tile);
    if (rows >= row_tile) rows -= rows % row_tile;
    return (int)std::min(M, rows);
}

#endif
