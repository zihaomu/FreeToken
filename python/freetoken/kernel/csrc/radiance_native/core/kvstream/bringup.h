/* bringup.h -- the engine side of a streamed group's bring-up (docs/KV-STREAMING.md §4.1).
 *
 * `--kv-resident N` is the only switch: absent or 0 nothing streams, and a ring that covers the
 * group's context streams nothing either, so the group keeps the resident path at no cost. This
 * module chooses the group, refuses what cannot stream, applies the ring's floors, gates the host
 * mirrors against host RAM, fills the maps and reports, so core/engine_bringup.cpp makes one call
 * at each seam.
 */
#pragma once
#include "mem/kv.h"

#include <string>
#include <vector>

namespace rad {

struct Config;

/* What --kv-resident resolved to. `group` is -1 when nothing streams; the other fields then say
 * what was asked and why, for the report. */
struct KVStreamSetup {
    int32_t     group = -1;            /* the group that streams; -1 when none does */
    std::string group_name;            /* the group that would have */
    int64_t     asked_tokens = 0;      /* --kv-resident as given */
    int64_t     raised_tokens = 0;     /* the ring a floor raised it to; 0 when not raised */
    bool        raised_by_union = false; /* raised to the decode-step bound, not the floor */
    int64_t     context_tokens = 0;    /* --max-model-len x --max-num-seqs */
    int64_t     ring_tokens = 0;       /* the carved ring, in tokens */
    int64_t     ring_blocks = 0;       /* the carved ring, in blocks */
    int64_t     block_size = 0;        /* the group's page, in tokens */
    int64_t     ctx_blocks = 0;        /* the group's id space, in blocks */
    int64_t     stage_blocks = 0;      /* the stage view's depth in blocks; 0 when none */
};

/* The host mirrors summed over the ranks; `max_rank_bytes` is the largest one rank pins (an uneven
 * tensor-parallel split gives ranks different shards). */
struct KVStreamHostUse {
    KVStreamCost sum;
    int64_t      max_rank_bytes = 0;
};

/* Whether `op` must not be declared: the streamed-only ops (kv_store_streamed, its folded form,
 * kv_resolve) exist only for a carved ring, so a run that carves none must not have them in its
 * graph. False when the architecture streams (RadBuildCtx::kv_stream_ok) and --kv-resident asks
 * for a ring. Every architecture's rad_op_declared check then falls through to its resident path. */
bool kvstream_op_gated(const RadBuildCtx& ctx, const char* op);

/* Read --kv-resident against rank 0's declarations and fill `out`; RAD_OK with group -1 when
 * nothing streams.
 *
 *   - The group is kv_stream_group's. A ring that covers its context streams nothing.
 *   - An architecture with no streamed path for the group (kvstream_ops: no kv_resolve, or no
 *     streamed store) ignores the flag with one warning naming the missing op; `out` is then
 *     exactly the no-flag configuration.
 *   - A core dtype other than int8, bf16 or E4M3 is refused: the streamed store has rows for
 *     those only.
 *   - A ring below the decode-step bound or kStreamFloorTokens is raised, with one warning.
 *     --kv-resident itself is left as given; `out` holds both numbers for the report. */
int kvstream_configure(const Config& cfg, int64_t ctx, const std::vector<KVGroupInfo>& groups,
                       const Program& prog, KVStreamSetup* out);

/* Add one rank's stream_cost() to `use`. */
void kvstream_host_add(KVStreamHostUse* use, const KVManager& kv);

/* The host-RAM gate for the mirrors, checked once before any rank pins one. A mirror is one full
 * context of the group per sequence per rank, the largest host allocation after the experts'
 * pool, so it is checked against everything still to be pinned after it (`still`). A refusal
 * names --max-model-len and --max-num-seqs, which size it; --host-pool-mib sizes the experts' pool
 * and cannot make room here. */
int kvstream_host_gate(const KVStreamHostUse& use, int n_ranks, int64_t still, int64_t headroom);

/* After the carve: fill the maps on a bring-up stream and drain it. */
int kvstream_fill(KVManager& kv);

/* The startup lines: the raised ring, which group streams and what it took on both tiers, the
 * stage, or which context a ring given by the flag covers. */
void kvstream_report(const KVStreamSetup& s, const KVStreamHostUse& host, int n_ranks);

}  /* namespace rad */
