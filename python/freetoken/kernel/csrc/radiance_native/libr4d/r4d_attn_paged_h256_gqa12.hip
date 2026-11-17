// R4D attention at twelve queries per KV head, head_dim 256: Qwen3.8-Flash-Next's QSA layers
// (24 query heads over 2 KV heads). Geometry only; r4d_attn_paged_gqa8.hip is the body. The ratio
// changes two bounds, both stated in the row table:
//
//   decode verify width   64 / gqa = 5. A wider speculative verify takes the prefill kernel.
//   prefill block width   384 / gqa = 32 query positions.
//
// No head_dim 128 leg: nothing in scope is gqa 12 at 128, so those entry points are not built.
#define R4D_ATTN_GQA  12
#define R4D_ATTN_H128 0
// KV page of 4 tokens (as in the gqa6-bs4 unit for the same layers at four ranks; 16 elsewhere).
// The QSA indexer selects 4-token blocks, so a selected block is a page and sparse attention is
// this kernel with a different block table (docs/QSA.md). The only cost is reading NB = TILE / BS
// table entries per tile (4 at TILE 16), within step noise; output is identical.
//
// Note: the rows must agree. C_EQ("block_size", 4) on all four gqa12 rows makes the core size the
// pages at 4 (RadKVGroupDecl takes the block size from the resolved kernel). A row saying 16 here
// would refuse at issue rather than at declare.
#define G8_BLOCK_SIZE 4

#include "r4d_attn_paged_gqa8.hip"
