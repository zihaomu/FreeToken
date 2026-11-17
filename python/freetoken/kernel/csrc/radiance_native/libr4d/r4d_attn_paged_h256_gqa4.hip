// R4D attention at four queries per KV head, head_dim 256: Qwen3.5-0.8B and siblings (8 query heads
// over 2 KV heads). Geometry only; r4d_attn_paged_gqa8.hip is the body. The ratio changes two
// bounds, both stated in the row table:
//
//   decode verify width   64 / gqa = 16.
//   prefill block width   384 / gqa = 96 query positions.
//
// No head_dim 128 leg: gqa 4 at 128 is the DFlash2 drafter's windowed geometry, served by
// r4d_attn_paged_h128_gqa4.hip. A head_dim 64 prefill leg serves LFM2 (32 query heads over 8) at
// every query length, since the split-KV decode kernel has no 64-wide form. Page size 16.
#define R4D_ATTN_GQA  4
#define R4D_ATTN_H128 0
#define R4D_ATTN_H64  1

#include "r4d_attn_paged_gqa8.hip"
