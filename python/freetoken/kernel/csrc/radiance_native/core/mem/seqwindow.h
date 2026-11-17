/* seqwindow.h -- a sequence's per-sequence runtime window (R-SWA), shared by the KV manager
 * (mem/kv.h, which owns the policy) and the scheduler's view of it (sched/mem_iface.h). The full
 * semantics are at KVManager::set_window in kv.h: a retained reference [0, visible_lo) plus a
 * ring of ring_blocks whole blocks starting at the block-aligned ring_pos, rewritten in place.
 */
#pragma once

#include <cstdint>

namespace rad {

struct SeqWindow {
    bool    set = false;
    int64_t window = 0;       /* W as set */
    int64_t visible_lo = 0;   /* the retained reference ends here */
    int64_t ring_pos = 0;     /* block-aligned first ring token position */
    int64_t ring_blocks = 0;  /* whole blocks the ring holds */
    int64_t ring_span(int64_t block) const { return ring_blocks * block; }
};

/* Everything the batch builder reads about one sequence in one group, fetched with one lookup
 * of the sequence (IKVManager::seq_group): block_table, first_block_pos, runtime_window,
 * state_slot and take_table_changes were five hash lookups of the same sequence per group per
 * step. `bt` points into the manager's table and stays valid for the builder's extent (the
 * sequence is alive and the step is being built under the scheduler's lock). Defaults are the
 * accessors' own answers for a sequence or group the manager does not know. */
struct SeqGroupView {
    const int32_t* bt = nullptr;
    size_t         bt_n = 0;
    int64_t        first = 0;
    SeqWindow      win{};
    bool           has_win = false;
    int32_t        state = -1;
    int64_t        changed_from = 0;   /* take_table_changes, when taken */
};

}  /* namespace rad */
