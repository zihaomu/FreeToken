/* stage.h -- the one-layer prefill stage: per pass and layer, copy the live part of the layer's
 * mirror row into the group's stage view, on the card's DMA engine, one layer ahead.
 *
 * Ordering rule. Every layer aliases offset 0 of the stage, which is safe only because for each
 * layer L these happen in order:
 *
 *   1. the stage copy of L has landed;
 *   2. L's store writes this chunk's tokens into the stage (the copy reads the mirror, which does
 *      not hold them yet, so a copy landing later would put older bytes back over them);
 *   3. L's attention reads the stage;
 *   4. the copy of L+1 starts.
 *
 * The copy is a DMA list, which holds no CU slot, so the order cannot rest on one stream's issue
 * order:
 *
 *   - 1 before 2: the compute stream waits on an event the list itself completes (rad_event_dma
 *     hands the list the event's completion word as its `done`), issued immediately before L's
 *     store. The wait is a barrier packet, so nothing between it and the store can pass it.
 *   - 4 after 3: the list polls a ride (rad_ride) taken on the compute stream after every
 *     attention launch of L (rad_kv_stage_next), so it starts when L's reads are done and crosses
 *     the link under L's MLP and L+1's projections.
 *
 * What breaks it:
 *
 *   - taking the ride before all of L's attention launches (a prefill layer's attention is issued
 *     in `qsa_gq_rows`-wide chunks, so the seam is after the loop, not in the resolve);
 *   - waiting for the copy after the layer's store instead of before it;
 *   - arming the completion word from the host instead of in stream order: an older 0 would then
 *     release this layer's wait early, or a fresh 1 cut the previous wait short;
 *   - the copy on lane 1, or on a second stream with no recorded event pair (docs/KV-STREAMING.md
 *     §3.5.1);
 *   - replaying a recorded pass without re-running the arm and the submit: the list is not a
 *     queue packet, so Ctx::tape_kv_stage re-runs them with the tape paused, as xprefetch.cpp does.
 *
 * The hazard pass sees the same byte range for every aliased layer, and it never sees the DMA
 * copy at all, so the barriers above are the copy's whole ordering (core/runtime/issue.cpp,
 * haz_mark).
 *
 * Volume. The copy moves only the blocks the step's own block tables name, not the whole row
 * (`n_blocks * block_bytes` is the context allocation): each layer's list carries one range per
 * run of those ids (kvstream_runs), mirror offset to the same stage offset. The ranges must never
 * miss a block the pass reads: the attention would read whatever the previous layer left there,
 * which is fluent wrong text, not a fault. They cover every read because
 *
 *   - the pass's kernels reach the stage only through the step's block tables (the store through
 *     the slots derived from them), so an id no table of the step names is never read;
 *   - the stage is identity-addressed, block b at b * block_bytes, as the mirror row is;
 *   - they are taken from the step's own sequences after its blocks are ensured
 *     (kvstream_step_runs, per step), so a block a chunk appends, a prefix hit's cached blocks or a
 *     speculative token's block is among them; a partial block is copied whole and its unwritten
 *     tail is the mirror's zeros.
 *
 * Allocation is not assumed ascending or dense: the prefix cache keeps retained conversations'
 * blocks, often the low ids, and a new sequence takes ids above them, so a bound on the highest id
 * a step names would still copy every retained conversation below it. A step whose sequences are
 * not known (a batch the builder did not produce) or whose trunk pass takes the ring copies up to
 * KVManager::held_top_blocks, one past the highest held block; nothing held copies the whole row.
 *
 * A backend with no DMA queue (the host backend, a test) copies on the compute stream, in the same
 * order.
 */
#pragma once
#include "mem/kv.h"
#include "runtime/ctx.h"

#include <cstdint>
#include <vector>

namespace rad {

/* Select the view of every streamed group the pass names and reset the stage's per-pass state.
 * Called before each pass (core/runtime/ctx.cpp). Not in bind.h, so a caller that only builds a
 * step's view (core/sched/batch.cpp) does not link the driver. */
void kvstream_enter(Ctx* c, const RadBatch* b);

/* At most this many ranges a layer's list: gaps of at most kStageGapBlocks are copied through, and
 * past the cap the smallest gaps are, so a fragmented table stays one short list. */
constexpr int     kStageMaxRuns   = 256;
constexpr int64_t kStageGapBlocks = 16;

/* Into `*out`, the ranges of a layer's row the block ids in `tables` occupy at `block_bytes` a
 * block: each table's runs of consecutive ids, sorted and joined where they overlap, touch or lie
 * at most kStageGapBlocks apart, then across the smallest remaining gaps until at most
 * kStageMaxRuns are left. Negative ids are skipped. Empty when no table names a block. Its
 * scratch is the calling thread's and keeps its capacity, so a step reuses it. */
void kvstream_runs(const std::vector<const std::vector<int32_t>*>& tables, int64_t block_bytes,
                   std::vector<KVStageRun>* out);

/* Into `*runs`, by group index, every group's ranges for the step `b` about to run (empty for a
 * group that does not stream): from the tables of `seqs`, the step's sequences, when its trunk
 * pass stages; otherwise [0, held_top_blocks) of `kv` (rank 0's manager, which hands out the
 * ids). The vectors are refilled in place. */
void kvstream_step_runs(const KVManager& kv, const RadBatch* b, const std::vector<uint64_t>* seqs,
                        std::vector<std::vector<KVStageRun>>* runs);

/* Install `runs` (kvstream_step_runs) as the step's stage ranges of every streamed group of `c`.
 * Called for every rank from Engine::prepare_release, with the ranks parked and the step's blocks
 * ensured. */
void kvstream_bound_live(Ctx* c, const std::vector<std::vector<KVStageRun>>& runs);

/* The driver, also re-run by a recorded pass (Ctx::tape_kv_stage). `after` is the call after the
 * layer's attention (rad_kv_stage_next) rather than before its store. */
int kvstage_run(Ctx* c, int group, int layer, bool after);

}  /* namespace rad */
