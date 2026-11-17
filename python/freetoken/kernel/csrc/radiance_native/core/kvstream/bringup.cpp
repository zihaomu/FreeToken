/* bringup.cpp -- see bringup.h. */
#include "kvstream/bringup.h"

#include "device/device.h"
#include "engine_hostmem.h"
#include "kvstream/plan.h"

#include <algorithm>
#include <cstring>

namespace rad {

/* The ops whose only job is the host mirror or the ring. `kv_store` and the fused
 * `qk_norm_rope_gate_kv_store` serve resident groups too. */
static const char* const kStreamOnlyOps[] = {
    "kv_store_streamed",
    "qk_norm_rope_gate_kv_store_streamed",
    "kv_resolve",
};

bool kvstream_op_gated(const RadBuildCtx& ctx, const char* op) {
    if (!op) return false;
    if (ctx.kv_stream_ok != 0 && ctx.kv_resident_blocks > 0) return false; /* a ring is carved */
    for (const char* s : kStreamOnlyOps)
        if (std::strcmp(s, op) == 0) return true;
    return false;
}

int kvstream_configure(const Config& cfg, int64_t ctx, const std::vector<KVGroupInfo>& groups,
                       const Program& prog, KVStreamSetup* out) {
    *out = KVStreamSetup{};
    if (cfg.kv_resident <= 0) return RAD_OK;
    /* On Flash-Next the group is `kv_attn`. The indexer's `kv_qsa_bkey` is full too, but qsa_score
     * reads it over the whole history every step, so it is dense-read and stays resident
     * (docs/KV-STREAMING.md §2). A model with only windowed groups has nothing to stream. */
    int32_t g = kv_stream_group(groups, &prog);
    /* With no streamed path anywhere, the widest full group is still chosen, so the warning below
     * can name it and the op it lacks. */
    if (g < 0) g = kv_stream_group(groups);
    if (g < 0) return RAD_OK;
    const KVGroupInfo& gi = groups[(size_t)g];
    const int64_t bs = gi.block_size > 0 ? gi.block_size : 1;
    /* A ring that covers the context is not a refusal: the group stays a resident pool. */
    const KVStreamRing shape = kvstream_ring_shape(cfg, ctx, bs);
    out->asked_tokens   = cfg.kv_resident;
    out->context_tokens = ctx * cfg.max_seqs;
    out->group_name     = gi.name;
    out->ring_blocks    = shape.ring_blocks;
    out->ring_tokens    = shape.ring_blocks * bs;
    out->block_size     = bs;
    out->ctx_blocks     = shape.ctx_blocks;
    if (!shape.streams()) return RAD_OK;
    /* Before anything is allocated: the group needs the ops its streamed path issues (KVStreamOps).
     * The gated attention (rad_block_attn_gated_fp8.h) and GLM-5.3's MLA block
     * (rad_block_glm_mla.h) declare both. Without them the flag is ignored and the configuration is
     * exactly the one without it. */
    const KVStreamOps ops = kvstream_ops(prog, &gi);
    if (!ops.streams()) {
        RAD_WARN("--kv-resident %lld ignored: architecture '%s' declares no %s for group '%s', "
                 "so the group stays resident",
                 (long long)cfg.kv_resident,
                 prog.meta.arch_id && *prog.meta.arch_id ? prog.meta.arch_id : "(unnamed)",
                 ops.missing(), gi.name.c_str());
        *out = KVStreamSetup{};
        return RAD_OK;
    }
    /* The streamed store has int8, bf16 and E4M3 rows; this refuses a core dtype no architecture
     * ships a full group in (f32, f16). */
    if (gi.decl.dtype != RAD_I8 && gi.decl.dtype != RAD_BF16 && gi.decl.dtype != RAD_F8E4M3) {
        RAD_ERR("--kv-resident %lld: group '%s' has a %s cache, and the streamed store has int8, "
                "bf16 and E4M3 rows only; leave --kv-resident off",
                (long long)cfg.kv_resident, out->group_name.c_str(),
                rad_dtype_name(gi.decl.dtype));
        return RAD_E_INVAL;
    }
    KVStreamRing ring;
    kvstream_ring(cfg, ctx, bs, ops.sel_pitch, &ring);
    if (ring.raised_blocks > 0) {
        /* The line names the floor that bound: one decode step's selection (a correctness bound) or
         * kStreamFloorTokens (a cost bound). */
        const int64_t asked_blocks = (out->asked_tokens + bs - 1) / bs;
        const int64_t step = kvstream_ceiling_blocks(cfg, ops.sel_pitch);
        if (ring.by_union && step > asked_blocks) {
            RAD_WARN("--kv-resident %lld raised to %lld tokens: one decode step can select %lld "
                     "blocks of %lld tokens (--max-num-seqs %lld x %lld rows x %lld), and the ring "
                     "must hold them",
                     (long long)out->asked_tokens, (long long)(ring.ring_blocks * bs),
                     (long long)step, (long long)bs,
                     (long long)(cfg.max_seqs > 0 ? cfg.max_seqs : 1),
                     (long long)(1 + (cfg.n_spec > 0 ? cfg.n_spec : 0)),
                     (long long)ops.sel_pitch);
        } else {
            RAD_WARN("--kv-resident %lld raised to %lld tokens: a ring under %lld tokens (%lld "
                     "blocks of %lld) misses more than it saves",
                     (long long)out->asked_tokens, (long long)(ring.ring_blocks * bs),
                     (long long)kStreamFloorTokens, (long long)ring.raised_blocks, (long long)bs);
        }
    }
    /* The raised ring may cover the whole context, which does not stream. */
    out->ring_blocks   = ring.ring_blocks;
    out->ring_tokens   = ring.ring_blocks * bs;
    out->raised_tokens = ring.raised_blocks > 0 ? ring.ring_blocks * bs : 0;
    out->raised_by_union = ring.raised_blocks > 0 && ring.by_union;
    out->group         = ring.streams() ? g : -1;
    /* Prefill takes the stage and never resolves: above `qsa_exact_to` a launch selects a chunk's
     * union, which no ring could hold. The stage is the group's context one layer deep, addressed
     * by block id (kvstream/bind.h says which passes take it). */
    out->stage_blocks = ring.streams() ? ring.ctx_blocks : 0;
    return RAD_OK;
}

void kvstream_host_add(KVStreamHostUse* use, const KVManager& kv) {
    const KVStreamCost c = kv.stream_cost();
    use->sum.groups     += c.groups;
    use->sum.host_bytes += c.host_bytes;
    use->sum.map_bytes  += c.map_bytes;
    use->max_rank_bytes = std::max(use->max_rank_bytes, c.host_bytes);
}

int kvstream_host_gate(const KVStreamHostUse& use, int n_ranks, int64_t still, int64_t headroom) {
    if (use.sum.host_bytes <= 0) return RAD_OK;
    std::string why;
    if (lockable_room(use.sum.host_bytes, still, headroom, false, &why)) return RAD_OK;
    RAD_ERR("kv streaming: the host mirror for %lld streamed KV group(s) needs up to %s of "
            "pinned host memory a rank (%s over %d ranks), and %s. It holds one full context a "
            "sequence: lower --max-model-len or --max-num-seqs, or set --kv-resident 0 "
            "(--host-pool-mib sizes the experts' tier, not this one)",
            (long long)use.sum.groups, humanb(use.max_rank_bytes).c_str(),
            humanb(use.sum.host_bytes).c_str(), n_ranks, why.c_str());
    return RAD_E_NOMEM;
}

int kvstream_fill(KVManager& kv) {
    /* After the carve, which backs an elastic pool's range; drained before the first pass reads the
     * maps. */
    RadStream st = nullptr;
    int rc = rad_stream_create(&st, 0);
    if (rc == RAD_OK) rc = kv.init_stream(st);
    if (rc == RAD_OK) rc = rad_stream_sync(st);
    rad_stream_destroy(st);
    return rc;
}

void kvstream_report(const KVStreamSetup& s, const KVStreamHostUse& host, int n_ranks) {
    const int64_t flag = s.raised_tokens > 0 ? s.raised_tokens : s.asked_tokens;
    if (s.raised_tokens > 0 && s.block_size > 0)
        RAD_INFO("kv streaming: --kv-resident %lld became %lld tokens (%lld -> %lld blocks of "
                 "%lld tokens), %s",
                 (long long)s.asked_tokens, (long long)s.raised_tokens,
                 (long long)((s.asked_tokens + s.block_size - 1) / s.block_size),
                 (long long)(s.raised_tokens / s.block_size), (long long)s.block_size,
                 s.raised_by_union ? "the blocks one decode step can select"
                                   : "the floor below which misses cost more than the ring saves");
    if (host.sum.groups > 0)
        RAD_INFO("kv streaming: %lld group(s) stream (%s): up to %s of pinned host mirror a "
                 "rank (%s over %d), %s of ring maps in the KV pools",
                 (long long)host.sum.groups, s.group_name.c_str(),
                 humanb(host.max_rank_bytes).c_str(), humanb(host.sum.host_bytes).c_str(), n_ranks,
                 humanb(host.sum.map_bytes).c_str());
    if (s.stage_blocks > 0)
        RAD_INFO("kv streaming: prefill staging, group '%s' ring %lld blocks, one-layer stage %lld "
                 "blocks (one layer of its %lld-block context); decode passes use the ring",
                 s.group_name.c_str(), (long long)s.ring_blocks,
                 (long long)s.stage_blocks, (long long)s.ctx_blocks);
    if (s.group < 0 && s.asked_tokens > 0 && s.ctx_blocks > 0)
        RAD_INFO("kv streaming: nothing streams: --kv-resident %lld covers group '%s''s "
                 "%lld-token context (%lld blocks), so the group stays resident; below %lld "
                 "tokens it would stream",
                 (long long)flag, s.group_name.c_str(), (long long)s.context_tokens,
                 (long long)s.ctx_blocks, (long long)s.context_tokens);
}

}  /* namespace rad */
