// R4D attention at sixteen queries per KV head, head_dim 128: Nemotron-H's attention layers (32
// query heads over 2 KV heads) at one and two ranks. Geometry only; r4d_attn_paged_gqa8.hip is the
// body. The ratio changes two bounds, both stated in the row table:
//
//   decode verify width   128 / gqa = 8 (a workgroup of eight waves holds 128 (query, head) rows).
//   prefill block width   384 / gqa = 24 query positions.
//
// The verify width is the drafter's: LiLiCorr proposes 7, so a verify is 8 query rows. At 64
// rows (q_len <= 4) those rows ran the query-tiled prefill kernel while a plain decode runs this
// split-KV one, so no verify row was the single-row decode at its position, and greedy text with
// the drafter left the text without it at one and two ranks within a few hundred characters (gqa
// 8, four ranks, already verified here). A wave holds one query's 16 heads in both launches.
//
// No head_dim 256 leg (R4D_ATTN_H256 0): nothing in scope is gqa 16 at 256. Page size 16.
#define R4D_ATTN_GQA  16
#define G8_MAX_DECODE_ROWS 128
#define R4D_ATTN_H128 1
#define R4D_ATTN_H256 0

#include "r4d_attn_paged_gqa8.hip"
