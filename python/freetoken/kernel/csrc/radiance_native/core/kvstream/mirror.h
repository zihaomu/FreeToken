/* mirror.h -- a streamed group's host mirror and residency maps (docs/KV-STREAMING.md §4.1).
 *
 * A streamed group's device pool backs a resident window, not its context. Every block lives in a
 * pinned host mirror laid out layer-major and block-indexed like the pool, and a device residency
 * map gives each block's ring slot (-1 when absent). KVStreamStore owns the window caps, the
 * mirrors, the maps and their sizing, so mem/kv.{h,cpp} keeps one member and one call per method.
 * Only init() touches a device: it fills the maps on the stream it is given.
 */
#pragma once
#include "mem/pools.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace rad {

struct KVGroupPlan;

/* What the streamed groups need, before anything is taken: `host_bytes` of pinned mirror (each rank
 * its own shard) and `map_bytes` of residency map, reverse map, slot state and control block, all
 * taken from the KV pool itself and budgeted with it. Valid between plan_groups() and carve(). */
struct KVStreamCost {
    int64_t groups     = 0;
    int64_t host_bytes = 0;
    int64_t map_bytes  = 0;
};

/* The resolve's counters for the metrics; zeros on a server that does not stream. */
struct KVStreamStats {
    int64_t  resident_blocks = 0;
    uint64_t misses = 0;        /* blocks placed in the ring by resolve launches */
    uint64_t bytes_in = 0;      /* bytes those placements copied from the host mirror */
};

/* Bytes of one streamed group's maps: the residency map (one entry a block, per layer), the
 * ring's reverse map and slot state (per layer), and the shared control block. */
int64_t kvstream_map_bytes(int64_t blocks, int64_t slots, int64_t layers);

/* The streamed groups' state by group handle: window caps, mirrors, maps and the resolve's
 * counters. A group's Map is the ABI's RadKVStream, so the run phase reads the fields alloc()
 * fills. Empty on a manager that does not stream. */
class KVStreamStore {
public:
    using Map = RadKVStream;

    /* The tokens of `group`'s context its device pool backs; 0 (the default) is all of it. Only the
     * named group: its neighbours (an indexer's block keys, read over the whole history) stay
     * resident. */
    void cap(int32_t group, int64_t tokens);

    /* Whether cap() narrowed `group` to a window: its block ids run over the context and only
     * `n_slots` of them are in device memory. */
    bool windowed(int32_t group) const;

    /* The window in whole blocks; at least one, since a pool that holds nothing cannot run. */
    int64_t resident_blocks(const KVGroupPlan& p) const;

    /* Blocks the group's context addresses (kv_context_blocks). The block table is already pitched
     * for them, whether the pool backs them or not. */
    int64_t context_blocks(const KVGroupPlan& p, int64_t max_ctx, int64_t max_seqs) const;

    void set_map(int32_t group, const Map& m);
    const Map* map(int32_t group) const;

    /* The ABI view of a streamed group, or null for a resident one. Its addresses are fixed from
     * bring-up, which lets a recorded pass that reads it replay. */
    const RadKVStream* view(const KVGroupPlan* p, int32_t group) const;

    KVStreamCost  cost(const std::vector<KVGroupPlan>& plans, int64_t max_ctx, int64_t max_seqs) const;

    /* Take the mirror and the maps for every streamed group. Call after plan_groups() and before
     * carve(), which refuses a streamed group with no map and whose paged split must see the maps'
     * bytes gone. The mirror is zeroed: a block nothing has written yet is still read on a miss.
     * The maps are filled by init() after the carve backs the pool. On failure the mirrors taken
     * so far are released. */
    int  alloc(const std::vector<KVGroupPlan>& plans, int64_t max_ctx, int64_t max_seqs,
               MemAllocator& host_alloc, Pool& kv_pool);

    /* Fill the maps as a resolve expects them: `res` and `slot_block` -1, `slot_state` and `ctl`
     * zero. Call after carve(), then sync `s` before the first pass. */
    int  init(const std::vector<KVGroupPlan>& plans, RadStream s);

    /* Free the mirrors; the maps go with the KV pool. Safe twice and after a failed bring-up. */
    void release(MemAllocator& host_alloc);

    KVStreamStats stats(const std::vector<KVGroupPlan>& plans) const;

    /* Charged by the engine, which reads the resolve's counters where a step has drained. Atomic
     * (relaxed): stats() is also read by the dashboard's snapshot on an HTTP thread while a step
     * may be in flight; only the values carry meaning, not their order. */
    void note_traffic(int64_t misses, int64_t bytes) {
        if (misses > 0) misses_.fetch_add((uint64_t)misses, std::memory_order_relaxed);
        if (bytes  > 0) bytes_in_.fetch_add((uint64_t)bytes, std::memory_order_relaxed);
    }

private:
    std::vector<int64_t> resident_tokens_;   /* by group handle: its pool's depth in tokens */
    std::vector<Map>     maps_;              /* set_map(), by group handle */
    std::atomic<uint64_t> misses_{0};        /* note_traffic() */
    std::atomic<uint64_t> bytes_in_{0};
};

}  /* namespace rad */
