/* carve.cpp -- see carve.h. */
#include "kvstream/carve.h"

#include "kvstream/mirror.h"
#include "kvstream/plan.h"
#include "mem/kv.h"

namespace rad {

int64_t kvstream_maps_charged(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                              int64_t max_ctx, int64_t max_seqs) {
    return st.cost(plans, max_ctx, max_seqs).map_bytes;
}

bool kvstream_dense(const KVGroupPlan& p, const std::vector<KVGroupPlan>& plans,
                    const KVStreamStore& st) {
    if (!p.paged() || p.zero || p.kind != RAD_KV_FULL || !p.dense_read) return false;
    for (const auto& q : plans)
        if (!q.zero && q.paged() && st.windowed(q.index)) return true;
    return false;
}

int64_t kvstream_dense_charged(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                               int64_t max_ctx, int64_t max_seqs) {
    int64_t bytes = 0;
    for (const auto& p : plans) {
        if (!kvstream_dense(p, plans, st)) continue;
        bytes += align_up(kv_context_blocks(p.block_size, max_ctx, max_seqs) * p.bytes_per_block,
                          RAD_ALIGN_UNIT);
    }
    return bytes;
}

void kvstream_split_windows(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                            int64_t max_ctx, int64_t max_seqs, int64_t* remaining,
                            std::vector<int64_t>* out) {
    for (const auto& p : plans) {
        if (!p.paged() || p.zero || !st.windowed(p.index)) continue;
        const int64_t slots = st.resident_blocks(p);
        *remaining -= kv_pool_bytes(p, slots, st.context_blocks(p, max_ctx, max_seqs));
        (*out)[p.index] = slots;
    }
    /* The maps, which the probe cannot reserve yet, or the pool it sizes falls short by them. */
    *remaining -= kvstream_maps_charged(plans, st, max_ctx, max_seqs);
    /* A dense-read group's floor is every sequence's context, fixed by the configuration. */
    for (const auto& p : plans) {
        if (!kvstream_dense(p, plans, st)) continue;
        const int64_t blocks = kv_context_blocks(p.block_size, max_ctx, max_seqs);
        *remaining -= align_up(blocks * p.bytes_per_block, RAD_ALIGN_UNIT);
        (*out)[p.index] = blocks;
    }
}

int kvstream_window_fit(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                        int64_t fixed, int64_t budget, int64_t max_ctx, int64_t max_seqs) {
    int64_t window_bytes = 0;
    int64_t window_groups = 0;
    for (const auto& p : plans) {
        if (!p.paged() || p.zero || !st.windowed(p.index)) continue;
        if (p.kind != RAD_KV_FULL) {
            RAD_ERR("KV group '%s' is a %s group and streaming is set for it: only a full "
                    "attention group streams (a window's block ids are its window's, not its "
                    "context's)",
                    p.name.c_str(), kv_kind_name(p.kind));
            return RAD_E_INVAL;
        }
        window_bytes += kv_pool_bytes(p, st.resident_blocks(p),
                                      st.context_blocks(p, max_ctx, max_seqs));
        ++window_groups;
    }
    const int64_t map_bytes   = kvstream_maps_charged(plans, st, max_ctx, max_seqs);
    const int64_t dense_bytes = kvstream_dense_charged(plans, st, max_ctx, max_seqs);
    int64_t dense_groups = 0;
    for (const auto& p : plans) if (kvstream_dense(p, plans, st)) ++dense_groups;
    if (fixed + window_bytes + map_bytes + dense_bytes > budget) {
        RAD_ERR("the resident KV windows and dense-read groups do not fit vram_kv:\n"
                "  resident windows   %10s  (%lld group(s) streaming)\n"
                "  ring maps          %10s  (the streamed groups' residency maps)\n"
                "  dense-read groups  %10s  (%lld group(s) at %lld sequences x %lld tokens)\n"
                "  linear/conv state  %10s  (%lld sequences)\n"
                "  vram_kv            %10s\n"
                "  short by           %10s\n"
                "Lower --kv-resident (the ring) or --max-model-len / --max-num-seqs (the one-layer "
                "stage and the dense-read floor), or give the cache more of the card: a smaller "
                "--gpu-headroom-mib or a larger --host-pool-mib.",
                humanb(window_bytes).c_str(), (long long)window_groups,
                humanb(map_bytes).c_str(), humanb(dense_bytes).c_str(), (long long)dense_groups,
                (long long)max_seqs, (long long)max_ctx, humanb(fixed).c_str(),
                (long long)max_seqs, humanb(budget).c_str(),
                humanb(fixed + window_bytes + map_bytes + dense_bytes - budget).c_str());
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

int kvstream_apply(std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                   int64_t max_ctx, int64_t max_seqs) {
    for (auto& p : plans) {
        if (!p.paged()) continue;
        if (st.windowed(p.index) && !p.zero) {
            /* The split gave a streamed group its window; its ids run over its context. */
            p.n_blocks = st.context_blocks(p, max_ctx, max_seqs);
            p.n_slots  = st.resident_blocks(p);
            /* A ring that covers the context never reaches here (kvstream_ring), so the group has a
             * stage. */
            p.depth    = p.streamed() ? 1 : 0;
        } else {
            p.n_slots = p.n_blocks;
        }
        if (p.n_slots > p.n_blocks) p.n_slots = p.n_blocks;
        /* A streamed group has no tier entry (its mirror is its tier), so replication, which
         * describes the tier slot, does not apply. */
        if (p.streamed()) p.replicated = false;
        /* An unbacked block id would be written past the layer stride: fluent wrong output, so
         * refuse here. RAD_E_INVAL, since RAD_E_STATE reads as "called in the wrong phase". */
        if (p.streamed()) {
            const KVStreamStore::Map* m = st.map(p.index);
            const int64_t need = p.n_blocks * p.bytes_per_block;
            if (!m || !m->host || !m->res || m->host_bytes < need) {
                RAD_ERR("KV group '%s' streams (%lld of its %lld blocks resident) but has no host "
                        "mirror of %s and residency map to hold the rest: an engine bug",
                        p.name.c_str(), (long long)p.n_slots, (long long)p.n_blocks,
                        humanb(need).c_str());
                return RAD_E_INVAL;
            }
        }
    }
    return RAD_OK;
}

std::string kvstream_report_line(const KVGroupPlan& p) {
    return fmt("    %-14s %-6s streaming: %7lld blocks resident in vram_kv at "
               "+%lld = %9s, %7lld blocks = %9s mirrored in pinned host memory, "
               "maps %s\n", "", "", (long long)p.pool_blocks(),
               (long long)p.pool_offset, humanb(p.pool_bytes).c_str(),
               (long long)p.n_blocks,
               humanb(p.n_blocks * p.bytes_per_block).c_str(),
               humanb(kvstream_map_bytes(p.n_blocks, p.pool_blocks(), p.n_layers)).c_str());
}

}  /* namespace rad */
