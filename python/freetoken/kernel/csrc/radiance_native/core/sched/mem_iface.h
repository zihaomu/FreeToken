// FreeToken modification: share cache and checkpoint key construction.
/* mem_iface.h -- the seam between core/sched and core/mem.
 *
 * The scheduler binds to pure-virtual mirrors of core/mem's `KVManager` (kv.h) and `PrefixCache`
 * (prefix.h) instead of the classes themselves:
 *
 *   - tests/sched_test.cpp can drive the scheduler against an arbitrarily small block pool, the
 *     only way to test admission refusal and preemption;
 *   - the cost is one indirect call per sequence per step, not per token.
 *
 * `KVBinding` / `PCBinding` at the bottom adapt the concrete classes. `CacheHit` comes from
 * core/mem/prefix.h; do not redefine it here (ODR).
 */
#pragma once
#include <functional>
#include "mem/prefix.h"
#include "mem/seqwindow.h"
#include "rad_core.h"
#include "mm/processor.h"

#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the KV manager */
/* Keyed by sequence id (the request id). The manager owns the block tables; the scheduler does
 * not read Request::blocks, so there is one owner and a freed state slot cannot be handed out
 * while its previous sequence still reads it. */
struct IKVManager {
    virtual ~IKVManager() = default;

    virtual int  add_sequence(uint64_t seq) = 0;
    virtual bool has_sequence(uint64_t seq) const = 0;
    virtual void free_sequence(uint64_t seq) = 0;

    /* Grow the sequence's mapping to cover `n_tokens` in every group. RAD_E_FULL when a group's
     * free list is empty; the caller preempts a victim and retries. */
    virtual int  ensure(uint64_t seq, int64_t n_tokens) = 0;
    /* As ensure(), for a step whose first query is at `keep_from` (the request's n_computed): a
     * window group keeps every block that query can still read (KVManager::ensure). The default
     * suits a manager with no windowed group. */
    virtual int  ensure_from(uint64_t seq, int64_t n_tokens, int64_t keep_from) {
        (void)keep_from;
        return ensure(seq, n_tokens);
    }
    /* After a committed step, a window group frees what no later query can read
     * (KVManager::trim_window). */
    virtual void trim_window(uint64_t seq, int64_t keep_from) { (void)seq; (void)keep_from; }
    /* R-SWA (KVManager::declare_runtime_window): the W this group arms a sequence with, 0 when it
     * carries no runtime windows. The scheduler arms every sequence it admits with
     * set_window(seq, prompt length, W); a manager with no runtime window refuses it and the
     * scheduler never asks. */
    virtual int64_t runtime_window_tokens(int32_t group) const { (void)group; return 0; }
    virtual int  set_window(uint64_t seq, int64_t visible_lo, int64_t window_tokens) {
        (void)seq; (void)visible_lo; (void)window_tokens;
        return RAD_E_UNSUPPORTED;
    }
    /* The sequence's runtime window in `group`, if it has one (KVManager::runtime_window). A
     * sequence with one is excluded from prefix publish, checkpoint snapshots and speculation:
     * the window is runtime data and the ring rewrites its slots in place. */
    virtual bool runtime_window(uint64_t seq, int32_t group, SeqWindow* out) const {
        (void)seq; (void)group; (void)out;
        return false;
    }

    virtual bool has_runtime_window(uint64_t seq) const { (void)seq; return false; }
    /* Window snapshot of a prefix hit (KVManager::has_window_snapshot / seed_window). The
     * defaults mean no hit on a windowed trunk is ever resumed. */
    virtual bool has_window_snapshot() const { return false; }
    virtual int  seed_window(uint64_t seq, int64_t pos) { (void)seq; (void)pos; return -1; }

    /* Adopt blocks a prefix-cache hit found. `n_tokens` must be a multiple of the group's block
     * size; a shared partial block would be corrupted by the next token written into it. */
    virtual int  adopt(uint64_t seq, int32_t group, const std::vector<int32_t>& blocks,
                       int64_t n_tokens) = 0;

    /* Free the blocks and mark the sequence for re-prefill. */
    virtual int  preempt(uint64_t seq) = 0;

    /* Whether a sequence of this length fits an empty pool. If not, the request fails at once
     * instead of deadlocking the engine (spec §17). */
    virtual bool can_ever_fit(int64_t n_tokens) const = 0;

    /* Whether `seq` can be grown to `n_tokens` without taking blocks from another sequence.
     * A sequence the manager does not know is not a refusal: the caller adds it first. */
    virtual bool can_grow(uint64_t seq, int64_t n_tokens) const = 0;
    /* Return state slots no running sequence holds, and any slot neither free nor held.
     * The caller guarantees nothing is running. A slot held by a zeroed or forgotten sequence
     * would otherwise keep an overdue preempted prefill waiting forever (spec §17). Returns
     * how many slots were put back. A manager with no state leaves this as a no-op. */
    virtual int release_idle_states() { return 0; }

    /* How many blocks a paged group can still hand out, including those lent to the expert slab
     * and recallable on demand. */
    virtual int64_t available_blocks(int32_t group) const = 0;

    virtual const std::vector<int32_t>& block_table(uint64_t seq, int32_t group) const = 0;
    virtual int32_t state_slot(uint64_t seq, int32_t group) const = 0;

    /* The sequence's state_index row: normally one entry (its state slot), plus any scratch
     * slots the group's kernel asked for. The manager owns the column-to-slot mapping; the
     * scheduler stages the row without reading it. Returns the width written (0 for a paged
     * group), at most `max`. */
    virtual int state_index_row(uint64_t seq, int32_t group, int32_t* out, int max) const = 0;
    virtual int64_t state_index_width(int32_t group) const = 0;

    virtual int64_t free_blocks(int32_t group) const = 0;
    virtual int64_t total_blocks(int32_t group) const = 0;
    /* Blocks held by anything, as opposed to free_blocks (what can be handed out now). On an
     * elastic pool they differ by what the cache released (KVManager::held_blocks). */
    virtual int64_t held_blocks(int32_t group) const = 0;
    /* KVManager::held_top_blocks; a manager that cannot answer says the whole row. */
    virtual int64_t held_top_blocks(int32_t group) const { (void)group; return INT64_MAX; }

    /* The pool in bytes, the only unit that sums across groups. Asked of the manager because its
     * plan knows whether a group counts blocks or states. */
    virtual int64_t pool_bytes(int32_t group) const = 0;
    virtual int64_t pool_bytes_used(int32_t group) const = 0;
    /* What the carve gave the group; an elastic pool may have released part of it. */
    virtual int64_t pool_bytes_carved(int32_t group) const = 0;

    /* A streamed group's RadKVStream for the step batch, or null for a resident group; its addresses
     * are fixed from bring-up. */
    virtual const RadKVStream* stream_view(int32_t group) const { (void)group; return nullptr; }
    /* The resolve's cumulative traffic and the rings' depth, for the metrics; zero without
     * streaming. */
    virtual int64_t  stream_misses() const { return 0; }
    virtual int64_t  stream_bytes_in() const { return 0; }
    virtual int64_t  stream_resident_blocks() const { return 0; }

    /* Tokens one block covers, or 0 for a stateful group. */
    virtual int64_t block_tokens(int32_t group) const = 0;
    /* A checkpoint slot carries this group's window (KVGroupPlan::win_snap): a prefix hit, which
     * resumes from such a slot, restores it whole. */
    virtual bool window_snapshot(int32_t group) const { (void)group; return false; }

    /* Speculative rejection rollback (spec §10). A verify writes KV for 1 + n_spec positions and
     * the sampler accepts a prefix; this shrinks the sequence to `n_tokens` and frees blocks past
     * it, which the monotone ensure() cannot. Only the slot table changes: linear and conv groups
     * roll back by read offset inside their kernels. */
    virtual int  rollback(uint64_t seq, int64_t n_tokens) = 0;

    /* Absolute token position that block_table(seq, group)[0] covers: zero for a full group,
     * non-zero for a window group once blocks below the window are recycled. The step batch maps
     * a slot as `(pos - first_block_pos) / block_size`.
     *
     * A sequence with a runtime window (runtime_window) keeps first_block_pos 0 even once the ring
     * recycles: its table is [reference][ring] from position 0 and the ring's positions map
     * modulo ring_blocks (SeqWindow), so only the slot index recycles, never the table. */
    virtual int64_t first_block_pos(uint64_t seq, int32_t group) const = 0;

    /* Index of the first entry of block_table(seq, group) changed since the previous call;
     * INT64_MAX if none.
     * Reading clears it, so the batch builder, which keeps the device copy and stages only the
     * changed tail, must be the only caller. 0 is always correct (restage the whole table). */
    virtual int64_t take_table_changes(uint64_t seq, int32_t group) const = 0;

    /* The batch builder's reads about (seq, group) in one call (SeqGroupView, mem/seqwindow.h).
     * The default composes the accessors above, so a manager that does not override it answers
     * identically; the concrete manager overrides it to find the sequence once. `take_changes`
     * additionally performs take_table_changes (batch build only, as there). */
    virtual void seq_group(uint64_t seq, int32_t group, bool take_changes,
                           SeqGroupView* out) const {
        const std::vector<int32_t>& bt = block_table(seq, group);
        out->bt    = bt.data();
        out->bt_n  = bt.size();
        out->first = first_block_pos(seq, group);
        out->has_win = runtime_window(seq, group, &out->win);
        if (!out->has_win) out->win = SeqWindow{};
        out->state = state_slot(seq, group);
        out->changed_from = take_changes ? take_table_changes(seq, group) : 0;
    }

    /* Whether the snapshot store still holds `slot`. The scheduler checks before building a
     * restore step, since a released slot may hold another session's state. */
    virtual bool checkpoint_held(int32_t slot) const = 0;
};

/* ------------------------------------------------------------------ the prefix cache */
/* What IPrefixCache::restore did: brought back every entry the tiers hold for the prompt (or there
 * were none), brought back part of them because the pool could not hold the chain, or, asked for
 * the whole chain only, left it on the host. */
enum class RestoreResult : uint8_t { Whole, Cut, Declined };

struct IPrefixCache {
    virtual ~IPrefixCache() = default;

    /* Bring back whatever the idle tiers hold for this prompt, before the lookup. The engine
     * performs it (it allocates blocks and runs a device scatter). A failure only leaves a
     * shorter prefix, which is always safe. With `whole_only`, a chain the pool cannot be made to
     * hold whole is left on the host (RestoreResult::Declined). */
    virtual RestoreResult restore(const Request&, bool whole_only) {
        (void)whole_only;
        return RestoreResult::Whole;
    }
    /* A step's admissions begin; the last step's lookup results have been consumed. */
    virtual void begin_step() {}
    /* Keep this waiting request's chain in VRAM until release(). Called when admission is refused
     * after restore(), so the restored chain is not moved out again before the next attempt. */
    virtual void hold(const Request&) {}
    /* The request finished, failed or was cancelled and holds nothing any more. */
    virtual void release(const Request&) {}

    /* Longest cached prefix, plus the checkpoint covering the linear state for a prefix of it.
     * `blocks_out` receives block ids per cached group, in cached_groups() order, for
     * IKVManager::adopt: n_attn_tokens over the group's own block size of them. `cap`, when not
     * null, ends the hit early (HitCap, core/mem/prefix.h). */
    virtual CacheHit lookup(const Request& r, std::vector<std::vector<int32_t>>* blocks_out,
                            const HitCap* cap) const = 0;
    virtual const std::vector<int32_t>& cached_groups() const = 0;
    /* The drafter's refill window (Scheduler::init): retention keeps a snapshot that deep below a
     * session's tip, so a refill-capped hit has one to end on (PrefixCache::set_drafter_floor). */
    virtual void set_drafter_floor(int64_t) {}

    /* Publish what this sequence computed so a later request sharing the prefix can adopt it.
     * `blocks[i]` is the block table of cached_groups()[i]; `n_tokens` is how many tokens are
     * backed. Takes a KV reference on every newly indexed block, so call it before the sequence's
     * own blocks are released. */
    virtual int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
                       int64_t n_tokens) = 0;

    /* Release the least recently used entries and their KV references. The scheduler tries this
     * before preempting a running request. */
    virtual int64_t evict(int64_t n_entries) = 0;
    /* Free up to `n_entries` for the reserve without losing anything a later turn needs. Unlike
     * evict(), it skips entries a live sequence still holds (no block would be freed) and entries
     * whose only copy is in VRAM. An entry is one block in each cached group, or more in a group
     * whose blocks are shorter than the entry (PrefixCache::blocks_per_entry). Returns the number
     * of entries freed. */
    virtual int64_t trim(int64_t n_entries) = 0;

    /* Drop every entry in every tier. Implementations with idle tiers must override this: there,
     * evict() keeps an entry whose host copy has landed. Returns the number of entries dropped. */
    virtual int64_t reset() {
        int64_t dropped = 0;
        for (int guard = 0; guard < 65536; ++guard) {
            const int64_t n = evict(1024);
            if (n <= 0) break;
            dropped += n;
        }
        return dropped;
    }

    /* Entries evicted, cumulative. Once the pool fills past the scheduler's reserve (a quarter of
     * it), the cache evicts prefixes about to be reused; this count rises before the hit rate
     * falls. */
    virtual int64_t evictions() const = 0;
    virtual int64_t ckpt_evictions() const = 0;
    /* ckpt_evictions by reason, and reservations that found nothing to give up
     * (PrefixCache::Stats). */
    struct CkptCounts { int64_t retired = 0, lru = 0, refused = 0; };
    virtual CkptCounts ckpt_counts() const { return {}; }
    /* Entries expired by --prefix-cache-ttl; not counted as evictions. */
    virtual int64_t expired() const { return 0; }
    virtual int64_t ckpt_expired() const { return 0; }
    /* Entries and snapshots a client released on purpose (release_session). */
    virtual int64_t released() const { return 0; }
    virtual int64_t ckpt_released() const { return 0; }

    /* Forget one stored conversation now, by the client's label or the engine's session id
     * (PrefixCache::release_session). RAD_E_UNSUPPORTED from a cache without a session table. */
    virtual int release_session(const std::string& /*label*/, uint64_t /*id*/,
                                int64_t /*keep_tokens*/, SessionRelease* /*out*/) {
        return RAD_E_UNSUPPORTED;
    }

    /* Tokens of context held only by the cache right now. Blocks a running sequence also reads
     * are excluded. Separates a working cache from a leak in the pool figures. */
    virtual int64_t held_tokens() const = 0;

    /* The next position strictly after `pos` at which the scheduler must split a chunk so that a
     * checkpoint lands on a boundary. */
    virtual int64_t next_checkpoint_after(int64_t pos) const = 0;

    /* Reserve the snapshot slot the scheduler is about to write. Retention is budgeted (§7.3: on
     * Qwen3.5-27B a snapshot is 144 MiB, and a 128K session at the default interval is 64 of them,
     * about 9 GiB), so RAD_E_FULL is ordinary: no checkpoint is written at that boundary and a
     * later hit replays from the previous one. */
    virtual int reserve_checkpoint(const Request& r, int64_t pos, int32_t* slot_out) = 0;

    /* The reserved state is now in its slot. Reservation happens at planning, before the step
     * computes the state, so lookup does not offer the slot until this is called. */
    virtual void commit_checkpoint(const Request& r, int64_t pos) = 0;
};

/* ------------------------------------------------------------------ block geometry */
/* What the step batch needs to turn a token position into a KV slot, read from the declared
 * program: KVGroupInfo::block_size comes from the resolved attention kernel's constraints
 * (§7.2). */
struct KVGeom {
    struct G {
        int     kind = RAD_KV_FULL;
        int64_t block_size = 1;
        int64_t window = 0;
        bool    paged = true;
        /* Columns in this group's state_index row. The manager decides it, so KVGeom::of leaves
         * it at 1 and the scheduler fills it in from the configured pool. */
        int64_t sidx_width = 1;
    };
    std::vector<G> g;

    static KVGeom of(const std::vector<KVGroupInfo>& groups) {
        KVGeom k;
        k.g.reserve(groups.size());
        for (const KVGroupInfo& gi : groups) {
            G e;
            e.kind = gi.decl.kind;
            e.paged = (e.kind == RAD_KV_FULL || e.kind == RAD_KV_WINDOW || e.kind == RAD_KV_RING);
            e.block_size = gi.block_size > 0 ? gi.block_size : 1;
            e.window = gi.decl.window;
            k.g.push_back(e);
        }
        return k;
    }

    /* Flat KV slot for absolute token position `pos`, given the sequence's block table for this
     * group and the position its first block covers. */
    int32_t slot(int gi, const std::vector<int32_t>& bt, int64_t first_pos, int64_t pos) const {
        const G& e = g[(size_t)gi];
        int64_t bi = (pos - first_pos) / e.block_size;
        if (bi < 0 || bi >= (int64_t)bt.size()) return -1;
        return (int32_t)(bt[(size_t)bi] * e.block_size + (pos % e.block_size));
    }

    /* The same slot for a runtime window (SeqWindow): the reference maps linearly (first_pos is
     * 0 there), and a position at or past the ring's start lives in block
     * ring_pos/block + ((pos - ring_pos)/block) % ring_blocks, which is what rewrites the ring's
     * slots in place. ring_pos is block-aligned, so `pos % block` stays the in-block offset. */
    int32_t slot_ring(int gi, const std::vector<int32_t>& bt, int64_t first_pos, int64_t pos,
                      const SeqWindow& w) const {
        const G& e = g[(size_t)gi];
        int64_t bi;
        if (w.set && pos >= w.ring_pos) {
            bi = w.ring_pos / e.block_size + ((pos - w.ring_pos) / e.block_size) % w.ring_blocks;
        } else {
            bi = (pos - first_pos) / e.block_size;
        }
        if (bi < 0 || bi >= (int64_t)bt.size()) return -1;
        return (int32_t)(bt[(size_t)bi] * e.block_size + (pos % e.block_size));
    }

    /* Steps until trunc(x / bs) next changes as x increases: the run over which a quotient (and
     * the matching C++ remainder, x - trunc(x/bs)*bs) stays affine. Truncation toward zero makes
     * the negative side differ from floor: the quotient of a negative x next changes one past a
     * multiple, not at it. */
    static int64_t trunc_steps(int64_t x, int64_t bs) {
        const int64_t q = x / bs;
        return (q >= 0 ? (q + 1) * bs : q * bs + 1) - x;
    }

    /* slot() at n consecutive positions pos0, pos0+1, ..., identical values with one division a
     * segment instead of two a token. A segment ends where the block index or the in-block
     * offset would change rule (trunc_steps of (pos - first_pos) and of pos); inside it the slot
     * is (block base + off + k), or -1 throughout when the block index is outside the table.
     * bt is the table's data and length where slot() takes the vector. */
    void slots(int gi, const int32_t* bt, size_t bt_n, int64_t first_pos, int64_t pos0,
               int64_t n, int32_t* out) const {
        const int64_t bs = g[(size_t)gi].block_size;
        int64_t done = 0;
        while (done < n) {
            const int64_t pos = pos0 + done;
            const int64_t rel = pos - first_pos;
            const int64_t bi = rel / bs;
            int64_t len = trunc_steps(rel, bs);
            const int64_t lo = trunc_steps(pos, bs);
            if (lo < len) len = lo;
            if (len > n - done) len = n - done;
            if (bi < 0 || (size_t)bi >= bt_n) {
                for (int64_t k = 0; k < len; ++k) out[done + k] = -1;
            } else {
                const int64_t base = (int64_t)bt[(size_t)bi] * bs + pos % bs;
                for (int64_t k = 0; k < len; ++k) out[done + k] = (int32_t)(base + k);
            }
            done += len;
        }
    }

    /* slot_ring() at n consecutive positions, the same treatment. Positions before ring_pos go
     * through slots(); in the ring the block index advances where (pos - ring_pos) crosses a
     * block multiple, wrapping modulo ring_blocks at those same points. A window with no ring
     * blocks cannot be armed (KVManager::set_window clamps the ring to at least one block and
     * slot_ring's modulo would be by zero), so it takes the plain mapping. */
    void slots_ring(int gi, const int32_t* bt, size_t bt_n, int64_t first_pos, int64_t pos0,
                    int64_t n, const SeqWindow& w, int32_t* out) const {
        if (!w.set || w.ring_blocks <= 0) {
            slots(gi, bt, bt_n, first_pos, pos0, n, out);
            return;
        }
        const int64_t bs = g[(size_t)gi].block_size;
        int64_t done = 0;
        if (pos0 < w.ring_pos) {
            done = w.ring_pos - pos0 < n ? w.ring_pos - pos0 : n;
            slots(gi, bt, bt_n, first_pos, pos0, done, out);
        }
        const int64_t ring_bi0 = w.ring_pos / bs;
        while (done < n) {
            const int64_t pos = pos0 + done;
            const int64_t rp = pos - w.ring_pos;
            const int64_t bi = ring_bi0 + (rp / bs) % w.ring_blocks;
            int64_t len = trunc_steps(rp, bs);
            const int64_t lo = trunc_steps(pos, bs);
            if (lo < len) len = lo;
            if (len > n - done) len = n - done;
            if (bi < 0 || (size_t)bi >= bt_n) {
                for (int64_t k = 0; k < len; ++k) out[done + k] = -1;
            } else {
                const int64_t base = (int64_t)bt[(size_t)bi] * bs + pos % bs;
                for (int64_t k = 0; k < len; ++k) out[done + k] = (int32_t)(base + k);
            }
            done += len;
        }
    }
};

/* ------------------------------------------------------------------ binding the concrete classes */
/* Templates, so a mismatch with core/mem is reported where the engine instantiates them. */
template <class KV>
struct KVBinding final : IKVManager {
    KV* m;
    explicit KVBinding(KV* k) : m(k) {}

    int  add_sequence(uint64_t s) override          { return m->add_sequence(s); }
    bool has_sequence(uint64_t s) const override    { return m->has_sequence(s); }
    void free_sequence(uint64_t s) override         { m->free_sequence(s); }
    int  ensure(uint64_t s, int64_t n) override     { return m->ensure(s, n); }
    int  ensure_from(uint64_t s, int64_t n, int64_t k) override { return m->ensure(s, n, k); }
    void trim_window(uint64_t s, int64_t k) override { m->trim_window(s, k); }
    int64_t runtime_window_tokens(int32_t g) const override { return m->runtime_window_tokens(g); }
    int  set_window(uint64_t s, int64_t lo, int64_t w) override { return m->set_window(s, lo, w); }
    bool runtime_window(uint64_t s, int32_t g, SeqWindow* o) const override {
        return m->runtime_window(s, g, o);
    }
    bool has_runtime_window(uint64_t s) const override { return m->has_runtime_window(s); }
    bool has_window_snapshot() const override { return m->has_window_snapshot(); }
    int  seed_window(uint64_t s, int64_t p) override { return m->seed_window(s, p); }
    int  adopt(uint64_t s, int32_t g, const std::vector<int32_t>& b, int64_t n) override {
        return m->adopt(s, g, b, n);
    }
    int  preempt(uint64_t s) override               { return m->preempt(s); }
    bool can_ever_fit(int64_t n) const override     { return m->can_ever_fit(n); }
    bool can_grow(uint64_t s, int64_t n) const override { return m->can_grow(s, n); }
    int  release_idle_states() override                 { return m->release_idle_states(); }
    const std::vector<int32_t>& block_table(uint64_t s, int32_t g) const override {
        return m->block_table(s, g);
    }
    int32_t state_slot(uint64_t s, int32_t g) const override { return m->state_slot(s, g); }
    int state_index_row(uint64_t s, int32_t g, int32_t* o, int n) const override {
        return m->state_index_row(s, g, o, n);
    }
    int64_t state_index_width(int32_t g) const override { return m->state_index_width(g); }
    int64_t free_blocks(int32_t g) const override   { return m->free_blocks(g); }
    int64_t available_blocks(int32_t g) const override { return m->available_blocks(g); }
    int64_t total_blocks(int32_t g) const override  { return m->total_blocks(g); }
    int64_t held_blocks(int32_t g) const override   { return m->held_blocks(g); }
    int64_t held_top_blocks(int32_t g) const override { return m->held_top_blocks(g); }
    int64_t pool_bytes(int32_t g) const override      { return m->pool_bytes(g); }
    int64_t pool_bytes_used(int32_t g) const override { return m->pool_bytes_used(g); }
    int64_t pool_bytes_carved(int32_t g) const override { return m->pool_bytes_carved(g); }
    const RadKVStream* stream_view(int32_t g) const override { return m->stream_view(g); }
    int64_t stream_misses() const override { return (int64_t)m->stream_stats().misses; }
    int64_t stream_bytes_in() const override { return (int64_t)m->stream_stats().bytes_in; }
    int64_t stream_resident_blocks() const override {
        return m->stream_stats().resident_blocks;
    }
    int64_t block_tokens(int32_t g) const override {
        const KVGroupPlan* p = m->plan(g);
        return p && p->paged() ? p->block_size : 0;
    }
    bool window_snapshot(int32_t g) const override {
        const KVGroupPlan* p = m->plan(g);
        return p && p->win_snap;
    }
    int     rollback(uint64_t s, int64_t n) override { return m->rollback(s, n); }
    int64_t first_block_pos(uint64_t s, int32_t g) const override {
        return m->first_block_pos(s, g);
    }
    int64_t take_table_changes(uint64_t s, int32_t g) const override {
        return m->take_table_changes(s, g);
    }
    void seq_group(uint64_t s, int32_t g, bool take, SeqGroupView* o) const override {
        m->seq_group(s, g, take, o);
    }
    bool checkpoint_held(int32_t slot) const override { return m->checkpoint_held(slot); }
};

/* Content keys for a request's media runs. Placeholder ids are identical for every picture, so
 * the prefix cache also hashes each run's item content key (core/mm, over the processed patches)
 * and segment: the same picture at the same place hits, two pictures never alias. */
inline void media_keys(const Request& r, std::vector<MMContentKey>* out,
                       uint64_t control_vector_salt = 0) {
    out->clear();
    /* The runtime LoRA adapter is part of every block's key: KV written under an adapter is not
     * the base's. It enters the chain as one span over the whole sequence, keyed by the adapter's
     * load uid, so every block overlaps it; the base (uid 0) adds no span and its hashes are
     * unchanged. The uid and not the name, because a reload under the same name is new weights. */
    if (r.lora_uid != 0) {
        MMContentKey k;
        k.tok_begin = 0;
        k.tok_end = INT32_MAX;
        k.hash = (r.lora_uid * 0x9e3779b97f4a7c15ull) ^ 0x4c6f5241ull;   /* "LoRA" */
        out->push_back(k);
    }
    /* A control vector is part of every block's key for the same reason: a projected step's hidden
     * states are not the stock model's, so its KV is not the stock model's either. The vector's own
     * identity rides in as one span over the whole sequence, so every block overlaps it, and a
     * request that runs without the vector (or a deployment with none) adds no span and hashes
     * exactly as it did. An adapter and a vector are independent spans and compose. */
    if (control_vector_salt != 0 && r.sp.use_control_vector) {
        MMContentKey k;
        k.tok_begin = 0;
        k.tok_end = INT32_MAX;
        k.hash = control_vector_salt ^ 0x43455643ull;   /* "CVEC" */
        out->push_back(k);
    }
    if (!r.media) return;
    const mm::PromptMedia& pm = *r.media;
    for (const mm::RopeLayout::Run& run : pm.rope.runs) {
        const mm::Item& it = *pm.items[(size_t)run.item].item;
        MMContentKey k;
        k.tok_begin = run.tok;
        k.tok_end = run.tok + run.n;
        k.hash = it.key_hi ^ (it.key_lo * 0x9e3779b97f4a7c15ull) ^
                 ((uint64_t)(uint32_t)run.row0 * 0xc2b2ae3d27d4eb4full);
        out->push_back(k);
    }
}

template <class PC>
struct PCBinding final : IPrefixCache {
    PC* m;
    /* The control vector's identity (Engine::cvec_salt_for), or 0 with none loaded. A request that
     * runs with it on keys its blocks and its linear-state snapshots under a chain that includes it,
     * so its KV is never handed to a request that runs without it: the hidden states differ, and
     * the cache stores hidden states' consequences. Per request, not per cache: two sequences of one
     * step may differ. */
    uint64_t salt_id = 0;
    /* Scratch for a request's media keys, reused so a text request's lookup allocates nothing. */
    mutable std::vector<MMContentKey> mm_;
    explicit PCBinding(PC* p) : m(p) {}

    CacheHit lookup(const Request& r, std::vector<std::vector<int32_t>>* out,
                    const HitCap* cap) const override {
        media_keys(r, &mm_, salt_id);
        /* r.id: the admission takes its conversation's snapshots over. */
        return m->lookup(r.prompt, mm_, out, cap, r.id);
    }

    /* Installed by the engine when the idle tiers are on; empty otherwise. */
    std::function<RestoreResult(const Request&, bool whole_only)> restore_fn;
    RestoreResult restore(const Request& r, bool whole_only) override {
        return restore_fn ? restore_fn(r, whole_only) : RestoreResult::Whole;
    }
    /* The entries restore_fn brings back, in chain order: the whole chain if the pool can be made
     * to hold it, otherwise cut (PrefixCache::restorable_from). Room is made as evict() makes it,
     * dropping entries with a landed host copy, then landing pending copies and dropping those,
     * but never erasing one, so no other conversation is lost to a restore. The pool's free
     * quarter can go to it; the reserve's trim after the next publish frees it again. `*res`
     * says whether the chain was cut; with `whole_only`, a chain that would be cut comes back
     * empty and Declined, and the room made stays for the next attempt. */
    std::vector<BlockHash> restore_plan(const Request& r, uint64_t salt, bool whole_only = false,
                                        RestoreResult* res = nullptr) {
        if (res) *res = RestoreResult::Whole;
        media_keys(r, &mm_, salt);
        std::vector<BlockHash> want = m->restorable_from(r.prompt, mm_);
        const int64_t need = (int64_t)want.size() - m->restore_room();
        if (need <= 0) return want;
        const int64_t k = m->trim_for(r.prompt, mm_, need);
        if (k < need && flush_fn && flush_fn()) (void)m->trim_for(r.prompt, mm_, need - k);
        /* Landing copies may have moved records; take the chain again against the room made. */
        if (whole_only) {
            want = m->restorable_from(r.prompt, mm_);
            if ((int64_t)want.size() <= m->restore_room()) return want;
            if (res) *res = RestoreResult::Declined;
            return {};
        }
        bool cut = false;
        want = m->restorable_from(r.prompt, mm_, m->restore_room(), &cut);
        if (cut && res) *res = RestoreResult::Cut;
        return want;
    }
    void begin_step() override { m->begin_step(); }
    void hold(const Request& r) override {
        media_keys(r, &mm_, salt_id);
        m->hold(r.id, r.prompt, mm_);
    }
    void release(const Request& r) override { m->release(r.id); }
    const std::vector<int32_t>& cached_groups() const override { return m->cached_groups(); }
    void set_drafter_floor(int64_t tokens) override { m->set_drafter_floor(tokens); }

    /* Indexes prompt ++ output: the next turn sends the reply back in its prompt, so indexing the
     * prompt alone would stop hitting at the first assistant token. */
    int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
               int64_t n_tokens) override {
        std::vector<int32_t> toks;
        toks.reserve(r.prompt.size() + r.output.size());
        toks.insert(toks.end(), r.prompt.begin(), r.prompt.end());
        toks.insert(toks.end(), r.output.begin(), r.output.end());
        if (n_tokens > (int64_t)toks.size()) n_tokens = (int64_t)toks.size();
        /* r.id is the session, the same id reserve_checkpoint keys its snapshots by. */
        media_keys(r, &mm_, salt_id);
        return m->insert(r.id, toks, mm_, blocks, n_tokens, r.session_label);
    }
    /* Installed by the engine when the idle tiers are on: lands every pending host copy now and
     * returns whether any landed. Runs on the step thread under the scheduler's lock. */
    std::function<bool()> flush_fn;
    /* Under pressure, first drop entries whose host copy has landed (trim), then land pending
     * copies and trim again. Only the rest is erased; with tiers on, an erase is warned about,
     * since a later turn will prefill that conversation again. */
    int64_t evict(int64_t n) override {
        int64_t k = m->trim(n);
        if (k < n && flush_fn && flush_fn()) k += m->trim(n - k);
        if (k >= n) return k;
        const int64_t before = m->stats().evictions;
        k += m->evict_lru(n - k);
        const int64_t lost = m->stats().evictions - before;
        if (lost > 0 && flush_fn) {
            const uint64_t now = rad_mono_ns();
            if (now - lost_warned_ns_ >= 60ull * 1000000000ull) {
                lost_warned_ns_ = now;
                RAD_WARN("prefix cache: %lld entries erased with no copy off the card -- the host "
                         "tier had no room for them in time (--prefix-cache-mib); "
                         "radiance:prefix_cache_evictions_total counts every one",
                         (long long)lost);
            }
        }
        return k;
    }
    int64_t trim(int64_t n) override { return m->trim(n); }
    int64_t reset() override {
        const int64_t n = m->size();
        m->clear();
        return n;
    }
    int64_t evictions() const override { return m->stats().evictions; }
    int64_t ckpt_evictions() const override { return m->stats().ckpt_evictions; }
    CkptCounts ckpt_counts() const override {
        const auto& st = m->stats();
        return { st.ckpt_retired, st.ckpt_lru, st.ckpt_refused };
    }
    int64_t expired() const override { return m->stats().expired; }
    int64_t ckpt_expired() const override { return m->stats().ckpt_expired; }
    int64_t released() const override { return m->stats().released; }
    int64_t ckpt_released() const override { return m->stats().ckpt_released; }
    int release_session(const std::string& label, uint64_t id, int64_t keep_tokens,
                        SessionRelease* out) override {
        return m->release_session(label, id, keep_tokens, out);
    }
    int64_t held_tokens() const override { return m->unshared_tokens(); }

    int64_t next_checkpoint_after(int64_t p) const override { return m->next_checkpoint_after(p); }
    int reserve_checkpoint(const Request& r, int64_t pos, int32_t* slot_out) override {
        /* A snapshot is keyed by the chained hash of the prefix it covers, built here because the
         * cache does not hold the tokens. It must match the chain lookup() walks, or the linear
         * cache never hits. The prefix is prompt ++ output, so continuations of one prompt do not
         * share a snapshot; whole blocks only, since a partial block's key would change. */
        return m->reserve_checkpoint(r.id, pos, prefix_hash(r, pos), slot_out);
    }

    void commit_checkpoint(const Request& r, int64_t pos) override {
        m->commit_checkpoint(prefix_hash(r, pos));
    }

private:
    uint64_t lost_warned_ns_ = 0;
    mutable std::vector<int32_t> blk_;   /* prefix_hash's block of tokens, kept for its capacity */

    /* The chain reserve_checkpoint and commit_checkpoint both key on; shared so they cannot
     * drift apart. */
    BlockHash prefix_hash(const Request& r, int64_t pos) const {
        media_keys(r, &mm_, salt_id);
        const int64_t have = (int64_t)r.prompt.size() + (int64_t)r.output.size();
        const int64_t end = std::min(pos, have);
        blk_.clear();
        blk_.reserve((size_t)end);
        for (int64_t i = 0; i < end; ++i)
            blk_.push_back(i < (int64_t)r.prompt.size() ? r.prompt[(size_t)i]
                : r.output[(size_t)(i - (int64_t)r.prompt.size())]);
        // Use the cache's own chain, including exact-mode whole-chunk namespaces.
        return m->checkpoint_key(blk_, mm_, pos);
    }
};

}  /* namespace rad */
