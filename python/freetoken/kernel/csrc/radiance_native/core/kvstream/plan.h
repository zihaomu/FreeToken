/* plan.h -- the KV streaming policy, as arithmetic on a KVGroupPlan. Nothing here touches a device.
 *
 * A streamed group has two device views over one allocation (docs/KV-STREAMING.md §4.1):
 *
 *   ring   n_slots blocks deep, `n_layers` layer stripes, addressed through the residency map;
 *   stage  n_blocks blocks deep, one layer stripe, addressed by block id.
 *
 * The allocation is max(ring, stage), never the sum. The plan's `depth` is the staged view's stripe
 * count (1); the ring keeps the group's `n_layers` stripes.
 *
 * The VRAM budget is resolved from the plan's ceiling before cap_resident_tokens() runs, so
 * kvstream_cap_probe() must teach the budget probe the streamed plan first; otherwise the pool is
 * sized for the whole cache and streaming frees nothing.
 */
#pragma once
#include "mem/kv.h"

namespace rad {

/* Bytes of the ring view: `slots` blocks, every layer. Inline so that a caller that only lays out a
 * view (kvstream/bind.cpp, linked by tests with weak device fakes) need not link KVManager. */
inline int64_t kv_ring_bytes(const KVGroupPlan& p, int64_t slots) {
    return slots > 0 ? slots * p.bytes_per_block : 0;
}

/* Blocks the group's context addresses: `max_seqs` sequences of `max_ctx` tokens, each rounded up
 * to whole blocks. The one definition of the id space, for the mirror's size and the ring decision
 * alike. */
inline int64_t kv_context_blocks(int64_t block_size, int64_t max_ctx, int64_t max_seqs) {
    if (block_size <= 0) return 0;
    const int64_t seqs = max_seqs > 0 ? max_seqs : 1;
    const int64_t ctx  = max_ctx > 0 ? max_ctx : 0;
    return seqs * ((ctx + block_size - 1) / block_size);
}

/* The group's one allocation: max(ring, stage) for `slots` resident blocks of `context_blocks`.
 * `slots >= context_blocks` is a group that does not stream: `slots` blocks of every layer. The
 * one-argument form is the carved plan's. A stateful group is its states. */
int64_t kv_pool_bytes(const KVGroupPlan& p, int64_t slots, int64_t context_blocks);
int64_t kv_pool_bytes(const KVGroupPlan& p);

/* The pool cost ceiling() sums for `blocks` blocks of context and `slots` resident blocks:
 * blocks * bytes_per_block unless the group streams. */
int64_t kv_ceiling_bytes(const KVGroupPlan& p, int64_t blocks, int64_t slots);

/* The group --kv-resident streams: the widest paged full group, not dense-read (a step reads those
 * over the whole history), for which the program declares a streamed path (kvstream_ops). -1 when
 * there is none. */
int32_t kv_stream_group(const KVManager& kv, const Program* prog);

/* The same choice from a rank's declarations, before the plans exist. With no program the
 * declared-path test is skipped. */
int32_t kv_stream_group(const std::vector<KVGroupInfo>& groups, const Program* prog = nullptr);

/* What --kv-resident makes of one group. */
struct KVStreamRing {
    int64_t ring_blocks = 0;   /* the ring: the flag in whole blocks, raised to the floors */
    int64_t raised_blocks = 0; /* the depth a floor raised the flag to; 0 when not raised */
    int64_t floor_blocks = 0;  /* kStreamFloorTokens in blocks, at most the context */
    bool by_union = false;     /* the raise was the decode-step bound, not kStreamFloorTokens */
    int64_t ctx_blocks  = 0;   /* max_seqs x ceil(ctx / block_size): the group's context */
    /* A ring that covers the context does not stream: the group keeps the resident path, with no
     * mirror, maps, stage or resolve, and a null RadKVStream. */
    bool streams() const { return ring_blocks > 0 && ring_blocks < ctx_blocks; }
};

/* The blocks one decode step's selections can name, which the ring must hold: `max_seqs x
 * (1 + n_spec)` rows of at most `sel_pitch` distinct blocks each. A decode-classified pass with
 * more rows a sequence than that (the serial drafter's history pass) takes the stage and is not
 * counted. 0 when the program declares no kv_resolve. */
int64_t kvstream_ceiling_blocks(const Config& cfg, int64_t sel_pitch);

/* The streamed ops the program declares for `group` (null: any group). A streamed group needs
 * `kv_resolve`, which rewrites `sel` from block ids to ring slots, and a store that writes the host
 * mirror (`kv_store_streamed`, or its fold into the prologue). Without the resolve the attention
 * reads block ids as ring slots, past the ring; without the store a miss copies zeros. A
 * declaration whose stated geometry (`block_bytes` on the resolve, `block_size` on the store)
 * belongs to another group does not count, nor does one that resolved to no kernel (a packed cache
 * layout has no streamed store row). */
struct KVStreamOps {
    bool    resolve   = false;   /* kv_resolve with this group's layer bytes and a sel_pitch */
    bool    store     = false;   /* kv_store_streamed */
    bool    folded    = false;   /* qk_norm_rope_gate_kv_store_streamed */
    int64_t sel_pitch = 0;       /* the blocks one selection row can name */
    bool streams() const { return resolve && (store || folded); }
    /* The first op missing, for the startup line; null when streams(). */
    const char* missing() const {
        if (!resolve) return "kv_resolve";
        if (!store && !folded) return "kv_store_streamed";
        return nullptr;
    }
};
KVStreamOps kvstream_ops(const Program& prog, const KVGroupInfo* group);

/* The depth a ring is never carved below, in tokens (at most the group's context): below it the
 * misses of a decode step cost more than the VRAM the ring gives back (docs/KV-STREAMING.md §3.9).
 * A floor, not a default: --kv-resident 0 is still off. */
constexpr int64_t kStreamFloorTokens = 32768;

/* The ring and the id space from the flag and kStreamFloorTokens alone, so a caller can tell a
 * model with no streamed path from a ring below the decode-step bound. */
KVStreamRing kvstream_ring_shape(const Config& cfg, int64_t ctx, int64_t block_size);

/* kvstream_ring_shape, then raised to kvstream_ceiling_blocks: a ring below what one decode step
 * can select would serve a selection table that no longer addresses the pool. Never refuses; the
 * caller prints the one warning. `block_size` is the group's page (1 when unknown). */
void kvstream_ring(const Config& cfg, int64_t ctx, int64_t block_size, int64_t sel_pitch,
                   KVStreamRing* out);

/* Turn the prefix cache's host tier off when a group streams: no entry holding a streamed group
 * can be demoted to it. Returns the MiB turned off, 0 when nothing changed. Call after
 * kvstream_cap_probe, before anything is sized from --prefix-cache-mib. */
int64_t kvstream_prefix_host_off(Config& cfg);

/* Teach the budget probe the streamed plan (the ring from --kv-resident, the stage from the
 * context) before probe.ceiling(ctx), and set `cfg.kv_streams` when a group streams. A model with
 * no streamed path is left exactly as without the flag. */
int kvstream_cap_probe(KVManager& probe, Config& cfg, const Program& prog, int64_t ctx);

}  /* namespace rad */
