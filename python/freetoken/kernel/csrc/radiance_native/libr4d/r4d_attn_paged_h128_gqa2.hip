// R4D attention at two queries per KV head, head_dim 128: Qwen3-1.7B's 16 query heads over 8 KV
// heads, Qwen3-ASR's decoder (arch/qwen3asr_bf16). Geometry only; r4d_attn_paged_gqa8.hip is the
// body. The ratio changes two bounds, both stated in the row table:
//
//   decode verify width   64 / gqa = 32 (a workgroup holds 64 (query, head) rows).
//   prefill block width   384 / gqa = 192 query positions.
//
// No head_dim 256 leg (R4D_ATTN_H256 0): gqa 2 at 256 is r4d_attn_paged_h256_gqa2.hip's, a
// windowed kernel of its own. Page size 16.
#define R4D_ATTN_GQA  2
#define R4D_ATTN_H128 1
#define R4D_ATTN_H256 0

#include "r4d_attn_paged_gqa8.hip"
