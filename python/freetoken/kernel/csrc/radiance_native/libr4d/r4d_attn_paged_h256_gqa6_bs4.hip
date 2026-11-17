// R4D attention at six queries per KV head, head_dim 256: Qwen3.8-Flash-Next's QSA layers at four
// ranks (r4d_attn_paged_h256_gqa12.hip serves them at two). The two KV heads are replicated across
// rank pairs, so each rank serves 6 queries for its one KV head. Geometry only;
// r4d_attn_paged_gqa8.hip is the body. The ratio changes two bounds, both stated in the row table:
//
//   decode verify width   64 / gqa = 10. MTP's K=3 verify (q_len 4) fits either ratio; a draft
//                         block of six or seven positions fits decode here but not at gqa 12.
//   prefill block width   384 / gqa = 64 query positions.
//
// No head_dim 128 leg: nothing in scope is gqa 6 at 128, so those entry points are not built.
#define R4D_ATTN_GQA  6
#define R4D_ATTN_H128 0
// KV page of 4 tokens, as in the gqa12 unit (the model's page; TP4 changes only queries per head).
// The QSA indexer selects 4-token blocks, so a selected block is a page and sparse attention is
// this kernel with a different block table (docs/QSA.md). The only cost is reading NB = TILE / BS
// table entries per tile, within step noise; output is identical.
//
// Note: the rows must agree. C_EQ("block_size", 4) on all four gqa6-bs4 rows makes the core size
// the pages at 4 (RadKVGroupDecl takes the block size from the resolved kernel). A row saying 16
// here would refuse at issue rather than at declare.
#define G8_BLOCK_SIZE 4

#include "r4d_attn_paged_gqa8.hip"
