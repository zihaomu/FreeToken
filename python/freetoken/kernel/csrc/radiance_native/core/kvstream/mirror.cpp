/* mirror.cpp -- see mirror.h. */
#include "kvstream/mirror.h"

#include <sys/mman.h>

#include "kvstream/plan.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace rad {

namespace {
int64_t ceil_div(int64_t a, int64_t b) { return b > 0 ? (a + b - 1) / b : 0; }

/* Each array starts 8-byte aligned: `ctl`'s counters are read as u64 pairs, and an odd block
 * count or ring depth would misalign the next array. The pool base is RAD_ALIGN_UNIT aligned. */
int64_t align8(int64_t n) { return (n + 7) & ~(int64_t)7; }

/* The maps' one layout, shared by alloc(), init() and kvstream_map_bytes(). */
struct MapLayout {
    int64_t res_bytes = 0, slot_block_bytes = 0, slot_state_bytes = 0, ctl_bytes = 0;
    int64_t res = 0, slot_block = 0, slot_state = 0, ctl = 0;
    int64_t bytes = 0;   /* the whole reservation */
};

MapLayout map_layout(int64_t blocks, int64_t slots, int64_t layers) {
    if (layers < 1) layers = 1;
    MapLayout l;
    l.res_bytes        = layers * blocks * (int64_t)sizeof(int32_t);
    l.slot_block_bytes = layers * slots  * (int64_t)sizeof(int32_t);
    l.slot_state_bytes = layers * slots * 2 * (int64_t)sizeof(int32_t);
    l.ctl_bytes        = 16 * (int64_t)sizeof(int32_t);
    l.res        = 0;                                        /* res, block -> slot, one map a layer */
    l.slot_block = l.res        + align8(l.res_bytes);       /* slot_block, slot -> block */
    l.slot_state = l.slot_block + align8(l.slot_block_bytes);/* slot_state: epoch, heat */
    l.ctl        = l.slot_state + align8(l.slot_state_bytes);/* ctl, shared by the layers */
    l.bytes      = l.ctl        + align8(l.ctl_bytes);
    return l;
}
}  /* namespace */

int64_t kvstream_map_bytes(int64_t blocks, int64_t slots, int64_t layers) {
    return map_layout(blocks, slots, layers).bytes;
}

void KVStreamStore::cap(int32_t group, int64_t tokens) {
    if (group < 0) return;
    if ((size_t)group >= resident_tokens_.size()) resident_tokens_.resize((size_t)group + 1, 0);
    resident_tokens_[(size_t)group] = tokens > 0 ? tokens : 0;
}

bool KVStreamStore::windowed(int32_t group) const {
    return group >= 0 && (size_t)group < resident_tokens_.size() &&
           resident_tokens_[(size_t)group] > 0;
}

int64_t KVStreamStore::resident_blocks(const KVGroupPlan& p) const {
    const size_t g = (size_t)p.index;
    if (g >= resident_tokens_.size() || resident_tokens_[g] <= 0 || p.block_size <= 0) return 0;
    return ceil_div(resident_tokens_[g], p.block_size);
}

int64_t KVStreamStore::context_blocks(const KVGroupPlan& p, int64_t max_ctx,
                                      int64_t max_seqs) const {
    return kv_context_blocks(p.block_size, max_ctx, max_seqs);
}

void KVStreamStore::set_map(int32_t group, const Map& m) {
    if (group < 0) return;
    if ((size_t)group >= maps_.size()) maps_.resize((size_t)group + 1);
    maps_[(size_t)group] = m;
}

const KVStreamStore::Map* KVStreamStore::map(int32_t group) const {
    if (group < 0 || (size_t)group >= maps_.size()) return nullptr;
    return &maps_[(size_t)group];
}

const RadKVStream* KVStreamStore::view(const KVGroupPlan* p, int32_t group) const {
    /* A ring that covers the context is never capped, so such a group gets null here. */
    if (!p || !p->streamed() || !windowed(group)) return nullptr;
    const Map* m = map(group);
    return m && m->host && m->res ? m : nullptr;
}

KVStreamCost KVStreamStore::cost(const std::vector<KVGroupPlan>& plans, int64_t max_ctx,
                                 int64_t max_seqs) const {
    KVStreamCost c;
    for (const auto& p : plans) {
        if (!windowed(p.index) || p.zero || p.kind != RAD_KV_FULL) continue;
        const int64_t blocks = context_blocks(p, max_ctx, max_seqs);
        const int64_t slots  = std::min(resident_blocks(p), blocks);
        if (blocks <= 0 || slots <= 0) continue;
        c.host_bytes += blocks * p.bytes_per_block;
        c.map_bytes  += kvstream_map_bytes(blocks, slots, p.n_layers);
        ++c.groups;
    }
    return c;
}

int KVStreamStore::alloc(const std::vector<KVGroupPlan>& plans, int64_t max_ctx, int64_t max_seqs,
                         MemAllocator& host_alloc, Pool& kv_pool) {
    for (const auto& p : plans) {
        if (!windowed(p.index) || p.zero || p.kind != RAD_KV_FULL) continue;
        const int64_t blocks = context_blocks(p, max_ctx, max_seqs);
        const int64_t slots  = std::min(resident_blocks(p), blocks);
        if (blocks <= 0 || slots <= 0) continue;

        Map m;
        m.slots             = slots;
        m.block_bytes       = p.layer_bytes_per_block();
        m.host_layer_stride = blocks * m.block_bytes;
        m.host_bytes        = blocks * p.bytes_per_block;
        m.host = static_cast<uint8_t*>(host_alloc.alloc(m.host_bytes, RAD_MEM_HOST_MAPPED));
        if (!m.host) {
            RAD_ERR("KV group '%s': the host mirror of %s (%lld blocks of %s) could not be "
                    "allocated", p.name.c_str(), humanb(m.host_bytes).c_str(), (long long)blocks,
                    humanb(p.bytes_per_block).c_str());
            release(host_alloc);
            return RAD_E_NOMEM;
        }
        /* A block nothing has written is still copied in on a miss, so its bytes must be defined. */
        std::memset(m.host, 0, (size_t)m.host_bytes);

        /* One reservation for all of the group's maps, from the KV pool so its report names them. */
        const int64_t layers = p.n_layers > 0 ? p.n_layers : 1;
        const MapLayout lay  = map_layout(blocks, slots, layers);
        const int64_t map_bytes = lay.bytes;
        std::string what = "kv_stream_maps:" + p.name;
        char* at = static_cast<char*>(kv_pool.reserve(what, map_bytes, RAD_ALIGN_UNIT, nullptr));
        if (!at) {
            RAD_ERR("KV group '%s': its residency map and ring bookkeeping (%s) did not fit the "
                    "KV pool", p.name.c_str(), humanb(map_bytes).c_str());
            host_alloc.free(m.host, RAD_MEM_HOST_MAPPED);
            release(host_alloc);
            return RAD_E_NOMEM;
        }
        /* Residency is per layer, because the resolve copies one layer's block: a group-wide map
         * would call a block resident in a layer whose bytes were never copied. The control
         * block is the group's. */
        m.res        = reinterpret_cast<int32_t*>(at + lay.res);
        m.slot_block = reinterpret_cast<int32_t*>(at + lay.slot_block);
        m.slot_state = reinterpret_cast<int32_t*>(at + lay.slot_state);
        m.ctl        = reinterpret_cast<int32_t*>(at + lay.ctl);
        set_map(p.index, m);
    }
    return RAD_OK;
}

int KVStreamStore::init(const std::vector<KVGroupPlan>& plans, RadStream s) {
    if (!s) {
        RAD_ERR("kv streaming: the maps must be filled on a stream, and none was given");
        return RAD_E_INVAL;
    }
    for (const auto& p : plans) {
        const Map* m = map(p.index);
        if (!m || !m->host || !m->res) continue;
        /* The state a resolve expects (libref/ref_kvstream.cpp): both maps -1, slot state and
         * control block 0. Every block starts absent. */
        const MapLayout lay = map_layout((int64_t)p.n_blocks, m->slots,
                                         p.n_layers > 0 ? p.n_layers : 1);
        int rc = RAD_OK;
        rc = rad_memset_async(m->res, 0xff, lay.res_bytes, s);
        if (rc == RAD_OK) rc = rad_memset_async(m->slot_block, 0xff, lay.slot_block_bytes, s);
        if (rc == RAD_OK) rc = rad_memset_async(m->slot_state, 0, lay.slot_state_bytes, s);
        if (rc == RAD_OK) rc = rad_memset_async(m->ctl, 0, lay.ctl_bytes, s);
        if (rc != RAD_OK) {
            RAD_ERR("KV group '%s': the residency map could not be initialised: %s",
                    p.name.c_str(), rad_dev_last_error());
            return rc;
        }
    }
    return RAD_OK;
}

void KVStreamStore::release(MemAllocator& host_alloc) {
    for (auto& m : maps_) {
        if (!m.host) continue;
        host_alloc.free(m.host, RAD_MEM_HOST_MAPPED);
        /* Cleared, so a second call is a no-op. */
        m = Map{};
    }
}

KVStreamStats KVStreamStore::stats(const std::vector<KVGroupPlan>& plans) const {
    KVStreamStats st;
    st.misses   = (int64_t)misses_.load(std::memory_order_relaxed);
    st.bytes_in = (int64_t)bytes_in_.load(std::memory_order_relaxed);
    for (const auto& p : plans)
        if (p.streamed()) st.resident_blocks += p.pool_blocks();
    return st;
}

}  /* namespace rad */
