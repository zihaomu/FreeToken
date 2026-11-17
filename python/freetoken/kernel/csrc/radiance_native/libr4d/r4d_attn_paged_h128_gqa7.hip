// R4D attention at seven queries per KV head, head_dim 128: Nanbeige4.3-33B (56 query heads over
// 8 KV heads). Geometry only; r4d_attn_paged_gqa8.hip is the body. The ratio changes two bounds,
// both stated in the row table:
//
//   decode verify width   64 / gqa = 9 query positions (63 rows of q_len * gqa)
//   prefill block width   336 / gqa = 48 query positions
//
// The gqa-8 unit runs 24 warps because ROWS = NWARPS * 16 is both the tile's query rows and a
// multiple of the ratio. Seven divides neither 384 (24 warps) nor 64, so this unit takes 21 warps
// -- 336 rows, the same 48 query positions a block as the gqa-8 unit, at seven eighths of the
// rows -- and a 63-row decode cap, which is nine query positions rather than eight. Both numbers
// are compile-time geometry (BLOCK_Q = ROWS / GQA in the prefill kernel); a 24-warp instantiation
// would leave six rows mapping outside the tile's query positions, so they are not tuning knobs.
//
// No head_dim 256 leg: Nanbeige is 128 throughout, and a second instantiation would widen the
// module for a geometry nothing serves.
#define R4D_ATTN_GQA         7
#define R4D_ATTN_H128        1
#define R4D_ATTN_H256        0
#define G8_MAX_DECODE_ROWS   63
#define G8_P_NWARPS          21

#include "r4d_attn_paged_gqa8.hip"
