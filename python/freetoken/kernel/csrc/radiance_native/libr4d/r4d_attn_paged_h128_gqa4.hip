// R4D attention dispatch for the DFlash2 drafter: head_dim 128, 4 queries per KV head, paged
// block 16, sliding window. Instantiations of the h256/gqa6 kernel templates with WIN = 1.
//
// Decode only. Queries long enough for the query-tiled prefill kernel occur on a drafter only in
// prefill steps; the rows declare the decode band (q_len * gqa <= 64), so those resolve elsewhere.
//
// Rationale: a general unified-attention kernel (vLLM's) launches (q_blocks, kv_heads, segments)
// with q_blocks padded for capture, so most workgroups return immediately; on this shape it is
// 8-9x slower than the window-capped DRAM bound. R4D keeps the query inside the workgroup (up to 64 rows of
// q_len * gqa), so its grid is (splits, kv_heads, num_seqs) with no dead workgroups.
#include "r4d.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
// The decode kernel uses helpers from the prefill unit, so prefill is included first.
// Nothing here instantiates r4d_attn_prefill_kernel.
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

// Tile, block size, row cap and merge: r4d_decode_launch_win (r4d_attn_decode_h256_gqa6.hip).
#define B_HEAD_DIM   128
#define B_GQA          4
#define B_TILE 16
// The h256 decode option set, including K prefetch 2 (head_dim 128 would not spill at 4, but 2 is
// 0.3-1.3% faster here with identical output), plus O_KRAW: one-round K loads
// (r4d_attn_decode_h256_gqa6.hip), 16 VGPRs at head_dim 128 with no occupancy loss.
#define B_OPT  20207163

extern "C" long r4d_attn_decode_h128_gqa4_scratch_bytes(const R4DArgs* a) {
    return r4d_decode_scratch(a, r4d_decode_splits_win(a, B_TILE), B_OPT);
}

extern "C" int r4d_attn_decode_h128_gqa4_bf16kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT, 1>(a, s);
}
// The int8 cache (int8 K and V, an f16 scale each per token and head): K is widened from the paged
// cache a fragment at a time, so without the one-round raw K loads.
extern "C" int r4d_attn_decode_h128_gqa4_i8kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, (B_OPT & ~O_KRAW) | O_KVI8, 0>(a, s);
}
// The same over a cache holding each kv head once for kv_rep query groups (O_KVREP), q at
// R4DArgs::q_ld (O_QLD): gqa 8 on this gqa-4 kernel, under attn_paged_qrow.
extern "C" int r4d_attn_decode_h128_gqa4_rep_i8kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, (B_OPT & ~O_KRAW) | O_KVI8 | O_QLD | O_KVREP,
                                 0>(a, s);
}
extern "C" int r4d_attn_decode_h128_gqa4_rep_bf16kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT | O_QLD | O_KVREP, 1>(a, s);
}
