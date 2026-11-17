/* hostarena.h -- fixed-size slots of pinned host memory.
 *
 * Backs two stores: the KV tiers' block arena (core/mem/kvtier.h, one slot per cache entry) and
 * the linear-state snapshot store (KVManager::set_checkpoint_store, one slot per snapshot). Under
 * tensor parallelism a slot holds every rank's shard of its object, each at its own offset, so
 * one slot id (handed out by rank 0) names the object on every rank.
 *
 * Pinned, because every byte is a PCIe transfer endpoint: the driver stages async copies to
 * pageable memory through its own buffer and may synchronise. Every object in a store has the same
 * size for a given model, so a bitmap is the whole allocator.
 *
 * Transfers are per run: with 4-token blocks a 200K-token conversation is 50K entries, and one
 * copy per entry makes the driver call rate the cost of a restore. Slots allocated together get
 * consecutive ids where possible, so a batch crosses the link as one strided copy per run. */
#pragma once
#include "rad_core.h"

#include <cstdint>
#include <vector>

namespace rad {

class HostArena {
public:
    ~HostArena();
    int  open(int64_t slot_bytes, int64_t cap_bytes);
    void close();
    bool enabled() const { return base_ != nullptr; }

    /* Not thread-safe: the owner serialises every call (IdleTiers under its mutex, the snapshot
     * store on the engine thread). */
    int64_t slot_bytes() const { return slot_bytes_; }
    int64_t capacity()   const { return cap_slots_; }   /* fixed at open */
    int64_t used_slots() const { return used_; }
    int64_t cap_bytes()  const { return slot_bytes_ * capacity(); }
    int64_t free_slots() const { return cap_slots_ - used_; }

    /* alloc: -1 when full. The caller evicts and retries; victim policy belongs to the store's
     * owner, not the allocator. */
    bool    full() const { return used_ >= cap_slots_; }
    int32_t alloc() { int32_t got = 0; return alloc_run(1, &got); }
    /* Up to `want` consecutive slots: the first free run at or after the cursor. `*got` is the
     * count; the caller asks again for the rest. -1 when full. */
    int32_t alloc_run(int32_t want, int32_t* got);
    void    free(int32_t slot);
    bool    in_use(int32_t slot) const { return slot >= 0 && slot < cap_slots_ && is_used(slot); }
    void*   at(int32_t slot) const;

private:
    bool    is_used(int64_t s) const { return (bits_[(size_t)(s >> 6)] >> (s & 63)) & 1ull; }
    void*   base_ = nullptr;
    int64_t slot_bytes_ = 0;
    int64_t used_ = 0;
    int64_t cap_slots_ = 0;
    int64_t cursor_ = 0;
    std::vector<uint64_t> bits_;
};

}  /* namespace rad */
