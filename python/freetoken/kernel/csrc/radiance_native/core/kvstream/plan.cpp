/* plan.cpp -- see plan.h. */
#include "kvstream/plan.h"

#include <algorithm>

namespace rad {

int64_t kv_pool_bytes(const KVGroupPlan& p, int64_t slots, int64_t context_blocks) {
    if (!p.paged()) return p.n_states * p.bytes_per_state;
    const int64_t ring = kv_ring_bytes(p, slots);
    if (slots <= 0 || slots >= context_blocks) return ring;
    return std::max(ring, context_blocks * p.layer_bytes_per_block());
}

int64_t kv_pool_bytes(const KVGroupPlan& p) {
    return kv_pool_bytes(p, p.pool_blocks(), p.n_blocks);
}

int64_t kv_ceiling_bytes(const KVGroupPlan& p, int64_t blocks, int64_t slots) {
    if (!p.paged() || p.block_size <= 0) return 0;
    if (slots <= 0 || slots >= blocks) return blocks * p.bytes_per_block;
    return kv_pool_bytes(p, slots, blocks);
}

namespace {
/* Both overloads' rule: from group 1 on, the widest full group that owns bytes, is neither zero
 * nor dense-read, and passes the caller's test. */
bool kv_stream_candidate(int32_t kind, bool zero, bool dense_read, int64_t bytes_per_block) {
    return kind == RAD_KV_FULL && !zero && !dense_read && bytes_per_block > 0;
}

template <class Candidate>
int32_t kv_stream_widest(size_t n, const Candidate& candidate) {
    int32_t best = -1;
    int64_t widest = 0;
    for (size_t g = 1; g < n; ++g) {
        int64_t bytes = 0;
        if (!candidate(g, &bytes)) continue;
        if (bytes > widest) {
            widest = bytes;
            best = (int32_t)g;
        }
    }
    return best;
}
}  /* namespace */

int32_t kv_stream_group(const KVManager& kv, const Program* prog) {
    const std::vector<KVGroupPlan>& ps = kv.plans();
    return kv_stream_widest(ps.size(), [&](size_t g, int64_t* bytes) {
        const KVGroupPlan& p = ps[g];
        if (!kv_stream_candidate(p.kind, p.zero, p.dense_read, p.bytes_per_block)) return false;
        const KVGroupInfo* gi = g < prog->kv_groups.size() ? &prog->kv_groups[g] : nullptr;
        if (!kvstream_ops(*prog, gi).streams()) return false;
        *bytes = p.bytes_per_block;
        return true;
    });
}

int32_t kv_stream_group(const std::vector<KVGroupInfo>& gs, const Program* prog) {
    return kv_stream_widest(gs.size(), [&](size_t g, int64_t* bytes) {
        const KVGroupInfo& gi = gs[g];
        if (!kv_stream_candidate(gi.decl.kind, gi.zero, gi.dense_read, gi.bytes_per_block))
            return false;
        if (prog && !kvstream_ops(*prog, &gi).streams()) return false;
        *bytes = gi.bytes_per_block;
        return true;
    });
}

int64_t kvstream_ceiling_blocks(const Config& cfg, int64_t sel_pitch) {
    if (sel_pitch <= 0 || cfg.max_seqs <= 0) return 0;
    /* One selection per row; a verify step is 1 + n_spec rows a sequence. */
    const int64_t rows = cfg.max_seqs * (1 + (cfg.n_spec > 0 ? cfg.n_spec : 0));
    return rows * sel_pitch;
}

KVStreamOps kvstream_ops(const Program& prog, const KVGroupInfo* group) {
    KVStreamOps ops;
    /* Whether a declaration has a kernel in some domain: an op no row matched keeps a placeholder band
     * with both domains null (Builder::op_resolved asks the same). */
    const auto resolved = [](const OpInfo& o) {
        for (const Band& b : o.bands)
            for (int d = 0; d < RAD_N_DOMAINS; ++d)
                if (b.dom[d]) return true;
        return false;
    };
    /* The group's per-layer block pitch, which kv_resolve states as `block_bytes`. */
    const int64_t n_bound = group && !group->layers.empty() ? (int64_t)group->layers.size() : 1;
    const int64_t layer_bytes =
        group && group->bytes_per_block > 0 ? group->bytes_per_block / n_bound : 0;
    for (const OpInfo& op : prog.ops) {
        /* A declaration with no kernel issues nothing (a packed cache layout has no streamed store
         * row); counting it would stream a group whose mirror is never written. */
        if (!resolved(op)) continue;
        const bool is_resolve = op.op == "kv_resolve";
        const bool is_store   = op.op == "kv_store_streamed" ||
                                op.op == "qk_norm_rope_gate_kv_store_streamed";
        if (!is_resolve && !is_store) continue;
        /* A declaration stating another group's geometry does not stream this one; one stating
         * none is taken at its word. */
        long long stated = 0;
        if (is_resolve && layer_bytes > 0 && op.base.get_i("block_bytes", &stated) &&
            stated != layer_bytes)
            continue;
        if (is_store && group && group->block_size > 0 &&
            op.base.get_i("block_size", &stated) && stated != group->block_size)
            continue;
        if (is_resolve) {
            long long pitch = 0;
            if (!op.base.get_i("sel_pitch", &pitch) || pitch <= 0) continue;
            /* Every layer of the group declares the same pitch; the first wins. */
            if (!ops.resolve) ops.sel_pitch = (int64_t)pitch;
            ops.resolve = true;
        } else if (op.op == "kv_store_streamed") {
            ops.store = true;
        } else {
            ops.folded = true;
        }
    }
    return ops;
}

KVStreamRing kvstream_ring_shape(const Config& cfg, int64_t ctx, int64_t block_size) {
    KVStreamRing out;
    const int64_t bs = block_size > 0 ? block_size : 1;
    out.ctx_blocks  = kv_context_blocks(bs, ctx, cfg.max_seqs);
    out.ring_blocks = cfg.kv_resident > 0 ? (cfg.kv_resident + bs - 1) / bs : 0;
    /* The floor, at most the context: a context under it streams nothing. */
    if (out.ring_blocks > 0) {
        out.floor_blocks = (kStreamFloorTokens + bs - 1) / bs;
        if (out.floor_blocks > out.ctx_blocks) out.floor_blocks = out.ctx_blocks;
        if (out.ring_blocks < out.floor_blocks) {
            out.ring_blocks    = out.floor_blocks;
            out.raised_blocks  = out.floor_blocks;
        }
    }
    if (out.ring_blocks > out.ctx_blocks) out.ring_blocks = out.ctx_blocks;
    return out;
}

void kvstream_ring(const Config& cfg, int64_t ctx, int64_t block_size, int64_t sel_pitch,
                   KVStreamRing* out) {
    *out = kvstream_ring_shape(cfg, ctx, block_size);
    if (!out->streams()) return;   /* absent, 0, or a ring that covers the context */

    /* The decode-step bound is always positive here: a group reaches this only when kvstream_ops
     * found a kv_resolve with a sel_pitch, and --max-num-seqs is at least 1. Silent, because this
     * runs on every rank's probe and again in bring-up; kvstream_configure prints the one line. */
    const int64_t ceiling = kvstream_ceiling_blocks(cfg, sel_pitch);
    if (out->ring_blocks < ceiling) {
        out->ring_blocks   = ceiling < out->ctx_blocks ? ceiling : out->ctx_blocks;
        out->raised_blocks = out->ring_blocks;
        out->by_union      = true;
    }
}

/* `cfg.kv_streams` is set by kvstream_cap_probe only when a group really streams. */
int64_t kvstream_prefix_host_off(Config& cfg) {
    if (!cfg.kv_streams || cfg.prefix_cache_mib <= 0) return 0;
    const int64_t was = cfg.prefix_cache_mib;
    cfg.prefix_cache_mib = 0;
    return was;
}

int kvstream_cap_probe(KVManager& probe, Config& cfg, const Program& prog, int64_t ctx) {
    cfg.kv_streams = false;
    if (cfg.kv_resident <= 0) return RAD_OK;
    const int32_t g = kv_stream_group(probe, &prog);
    if (g < 0) return RAD_OK;
    const KVGroupPlan* p = probe.plan(g);
    if (!p || p->block_size <= 0) return RAD_OK;
    /* No streamed path: priced as without the flag. kvstream_configure warns, once. */
    const KVStreamOps ops =
        kvstream_ops(prog, g < (int32_t)prog.kv_groups.size() ? &prog.kv_groups[(size_t)g] : nullptr);
    if (!ops.streams()) return RAD_OK;
    KVStreamRing ring;
    /* The raised ring, so the budget and the carve take the same depth. */
    kvstream_ring(cfg, ctx, p->block_size, ops.sel_pitch, &ring);
    if (!ring.streams()) return RAD_OK;
    cfg.kv_streams = true;
    probe.cap_resident_tokens(g, ring.ring_blocks * p->block_size);
    return RAD_OK;
}

}  /* namespace rad */
