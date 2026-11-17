// Modified for FreeToken: exact hybrid prefix chunk namespaces.
/* prefix.h -- prefix caching (spec §7.3). Full and linear attention need separate mechanisms.
 *
 * Full attention: content-hashed blocks with a chained hash, so a block's identity includes its
 * whole prefix. LRU eviction and copy-on-write on fork. Multimodal spans are keyed by their
 * processed patches, not the file bytes (MMContentKey).
 *
 * Linear attention: state checkpoints. A GDN layer's recurrent state at position N depends on all
 * of 0..N and is one fixed-size object per sequence (on Qwen3.5-27B, 48 heads of 128x128 fp32 is
 * 3 MiB a layer, 144 MiB across its 48 linear layers), far too large to snapshot per 16-token
 * block. This follows vLLM's `--mamba-cache-mode align` with two changes:
 *
 *   (a) The checkpoint interval is decoupled from the attention block size. vLLM inflates the
 *       attention block to the mamba page (528 tokens for Qwen3.5, 2240 for the 35B-A3B), so
 *       shorter prompts get no attention hits either. Here the attention block stays 16 (§7.2
 *       takes it from the kernel; libr4d's paged attention is compiled for 16) and
 *       Config::checkpoint_interval is a separate, larger number. A hit between checkpoints
 *       reuses the attention blocks and replays only the linear layers from the last checkpoint:
 *       in CacheHit, `n_attn_tokens` is free, `[n_linear_tokens, n_attn_tokens)` costs a
 *       linear-layers-only pass, and the rest is ordinary prefill.
 *
 *   (b) Checkpoints are taken deliberately. vLLM writes one only when a step happens to end on a
 *       boundary. The scheduler here owns chunk boundaries (spec §7.1) and splits a chunk at the
 *       interval (next_checkpoint_after()), so a checkpoint is always written, at the cost of an
 *       occasional short chunk.
 *
 * Block sizes: an entry covers block_size() tokens, the lcm of the cached groups' block sizes, and
 * holds block_size() / B blocks of a group whose blocks are B tokens. A model whose layers keep KV
 * at different strides (a compressed group at one row per r tokens beside one per token) is cached
 * in every group; a hit is then a multiple of every group's block. The chunk quantum is a multiple
 * of every paged group's block (core/sched/geometry.cpp), so checkpoints land on entries.
 *
 * Sizing: the cache holds a KV reference on every block it indexes, so pool demand scales with
 * live sessions, not running requests. Hit rate falls off a cliff with pool utilisation: intact
 * below about 75%, half gone by about 81%, and zero once the sessions no longer fit, because each
 * turn evicts the prefixes the next one needs. Output is unaffected; TTFT grows 10-100x. A context
 * of T tokens occupies about 1.8 x T pool-tokens across the paged groups. The budget gives the
 * cache what the experts can spare (core/mem/vram_budget.h); --host-pool-mib moves that.
 *
 * Monitoring: `kv_util` (/stats, or vllm:kv_cache_usage_perc) predicts trouble and
 * `radiance:prefix_cache_evictions_total` counts it. The hit rate lags: it falls only after a
 * reusable prefix is already gone.
 */
#pragma once
#include "mem/kv.h"
#include "rad_core.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rad {

class IdleTiers;

/* ------------------------------------------------------------------ block identity */
/* 128 bits, chained: identical spans under different prefixes are different blocks. The width
 * makes a collision improbable; the token compare on lookup makes a wrong hit impossible. */
struct BlockHash {
    uint64_t hi = 0, lo = 0;
    bool operator==(const BlockHash& o) const { return hi == o.hi && lo == o.lo; }
    bool operator!=(const BlockHash& o) const { return !(*this == o); }
    bool valid() const { return hi != 0 || lo != 0; }
    std::string str() const;
};

struct BlockHashHash {
    size_t operator()(const BlockHash& h) const { return (size_t)(h.lo ^ (h.hi * 0x9e3779b97f4a7c15ull)); }
};

/* A multimodal span, keyed by content. Placeholder token ids are the same for every image, so
 * `hash` is the item's content key (core/mm: its processed patches, grid and tiling) mixed with
 * the run's row offset, by media_keys in core/sched/mem_iface.h (spec §7.3, §11). The same picture
 * at the same resolution and place hits even if the file bytes differ, and different pictures
 * never alias. Hashing the file instead would hit with stale KV whenever preprocessing changes. */
struct MMContentKey {
    int32_t  tok_begin = 0;    /* [begin, end) in prompt-token coordinates */
    int32_t  tok_end = 0;
    uint64_t hash = 0;
};

/* The chained hash of one block. `parent` is the previous block's hash (a zeroed BlockHash for
 * block 0), `block_begin` is the block's first token position. */
BlockHash chain_block_hash(const BlockHash& parent, const int32_t* toks, int32_t n,
                           const MMContentKey* mm, int32_t n_mm, int64_t block_begin);

/* ------------------------------------------------------------------ the split lookup */
struct CacheHit {
    int32_t n_attn_tokens   = 0;   /* attention KV reused directly, always a whole block multiple */
    int32_t n_linear_tokens = 0;   /* linear state reused; <= n_attn_tokens, a checkpoint multiple */
    int32_t checkpoint_slot = -1;  /* the snapshot holding that state, -1 if none */
    int32_t given_back      = 0;   /* reusable tokens a HitCap took off the whole hit */
    /* Where the prompt leaves a head at least two cached prompts already left differently (the
     * same instructions in front of different texts), or 0. One cached prompt past it is not
     * enough: the next could be the same conversation going on. A snapshot there serves every
     * request that starts with that head (Scheduler::admit_one). Uncapped by a HitCap. */
    int32_t branch_at       = 0;
};

/* Ends a hit early on purpose: at or below `tokens`, on a block and (with recurrent state) on the
 * deepest checkpoint at or below it, unless that gives back more than `max_give_back` reusable
 * tokens, in which case the whole hit stands. For a drafter whose KV the cache does not hold
 * (Scheduler::init, --draft-refill): recomputing the prompt's tail gives it context, and the
 * give-back limit stops a far-off checkpoint from turning that into a long recompute. */
struct HitCap {
    int64_t tokens = 0;
    int64_t max_give_back = 0;
};

/* ------------------------------------------------------------------ retention policy */
/* Which checkpoints a session keeps. Spec §19.1 treats geometric backoff as unmeasured, so the
 * policy is a replaceable class with two alternatives alongside it. */
class CheckpointPolicy {
public:
    virtual ~CheckpointPolicy() = default;
    virtual const char* name() const = 0;
    /* `positions` ascending, one per checkpoint the session holds; fill `keep` with the subset to
     * retain. The tip, positions.back(), must always be kept: an appending transcript resumes
     * from it. */
    virtual void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const = 0;
    /* The most snapshots the policy keeps for one session that grows to `ctx` tokens with a
     * checkpoint on every `interval`, retention applied after each: the session's standing share
     * of the store. The base applies select() to its own output one checkpoint at a time, which
     * is cheap only for a policy that keeps few. */
    virtual int64_t held(int64_t ctx, int64_t interval) const;
    /* Tokens before the tip a hit must be able to end at: a drafter whose KV the cache does not
     * hold recomputes that much of the prompt's tail (Scheduler::init, HitCap). 0: none. A policy
     * that would otherwise keep nothing that deep keeps one snapshot there. */
    virtual void set_floor(int64_t) {}
};

/* Keep the tip and, at every scale S = interval x 2^k, the shallowest checkpoint in the tip's
 * bucket [j*S, (j+1)*S) and in the bucket before it. Depths behind the tip grow geometrically, a
 * session keeps about log2(tip / interval) + 2 snapshots, and a turn that diverges d tokens behind
 * the tip resumes less than max(2d, 2 x interval) below the divergence (spec §7.3). Buckets are
 * fixed token ranges, so the rule applied again after more checkpoints keeps the same ladder, and
 * a snapshot it drops is never one it would select later. */
class GeometricBackoff final : public CheckpointPolicy {
public:
    /* `interval` is the resolved checkpoint interval, the smallest scale. */
    explicit GeometricBackoff(int64_t interval) : interval_(interval > 0 ? interval : 1) {}
    const char* name() const override { return "geometric-backoff"; }
    void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const override;

private:
    int64_t interval_;
};

/* The two ends of the trade, for comparison. tip-only under a drafter floor also keeps the deepest
 * snapshot at or below tip - floor: without it a refill-capped hit finds nothing to end on, keeps
 * the whole hit, and the request decodes with no drafts (8 sessions of 22k tokens: 5 ran
 * unspeculated, decode 424-521 t/s against 738-754 under geometric backoff). */
class KeepTipOnly final : public CheckpointPolicy {
public:
    const char* name() const override { return "tip-only"; }
    void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const override;
    int64_t held(int64_t, int64_t) const override { return floor_ > 0 ? 2 : 1; }
    void set_floor(int64_t tokens) override { floor_ = tokens > 0 ? tokens : 0; }
private:
    int64_t floor_ = 0;
};
class KeepAll final : public CheckpointPolicy {
public:
    const char* name() const override { return "keep-all"; }
    void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const override;
    int64_t held(int64_t ctx, int64_t interval) const override;
};

/* The policy --checkpoint-policy names, at the resolved checkpoint interval. config_parse admits
 * only the three names, so any other is geometric backoff. */
std::unique_ptr<CheckpointPolicy> make_checkpoint_policy(const std::string& name, int64_t interval);

/* ------------------------------------------------------------------ the store's size */
/* --checkpoint-slots auto. Every sequence --max-num-seqs admits gets its session's standing share
 * at --max-model-len (CheckpointPolicy::held) plus one slot for the checkpoint its running turn
 * writes while the previous turn's snapshots are still held. The store is pinned for the life of
 * the process, so the count is bounded by two byte budgets the engine measures: `share`, the part
 * of host memory the store may take, and `room`, what the host can pin after everything still to
 * be allocated. Under the share it never drops below kCheckpointSlotsFloor (or `want`, if fewer);
 * the room is a hard bound. */
constexpr int64_t kCheckpointSlotsFloor = 8;

/* The least --host-headroom-mib is reduced to for a store's floor (host_reserve_for_floor). */
constexpr int64_t kHostReserveFloor = (int64_t)1 << 30;

struct CheckpointSlots {
    int64_t per_seq = 0;      /* snapshots a sequence is given */
    int64_t want = 0;         /* --max-num-seqs x per_seq */
    int64_t slots = 0;        /* the store's size */
    enum class Bound { Want, Share, Room } bound = Bound::Want;
};

/* What --checkpoint-slots auto wants before any byte budget applies: the policy's hold for one
 * --max-model-len session plus one for its running turn, for each of --max-num-seqs. */
int64_t checkpoint_slots_want(const CheckpointPolicy& policy, int64_t max_seqs, int64_t ctx,
                              int64_t interval);

CheckpointSlots checkpoint_slots_auto(const CheckpointPolicy& policy, int64_t max_seqs,
                                      int64_t ctx, int64_t interval, int64_t slot_bytes,
                                      int64_t share_bytes, int64_t room_bytes);

/* The reserve to charge a store whose floor `need_bytes` must fit. `room_bytes` is what the host
 * can pin with `margin_bytes` (--host-headroom-mib) held free; it is what the engine measures, not
 * what a model or a KV encoding is assumed to cost.
 *
 * The reserve is a courtesy to the system -- the process's unpooled allocations, the load's
 * staging windows, the page cache -- and a store that turns prefix reuse on or off outranks it: a
 * store too small for one snapshot re-prefills the whole conversation on every turn, while the
 * reserve it displaced is memory the host would merely have kept. So the reserve gives back
 * exactly the shortfall, never below kHostReserveFloor, and never above the margin asked for.
 * It answers `margin_bytes` unchanged whenever the floor already fits, so a host with room to
 * spare never sees the reserve move. */
int64_t host_reserve_for_floor(int64_t need_bytes, int64_t room_bytes, int64_t margin_bytes);

/* ------------------------------------------------------------------ a release */
/* What PrefixCache::release_session took out and what it left. Blocks are left for three
 * reasons, counted apart so a caller can tell a half-shared conversation from a busy one: the
 * caller asked for them (`keep_tokens`), another stored session's chain runs through them (a
 * system prompt two conversations open with belongs to both), or a live sequence holds them. */
struct SessionRelease {
    int64_t sessions = 0;        /* rows the label or id named */
    int64_t entries = 0;         /* blocks forgotten, from VRAM or the host tier */
    int64_t tokens = 0;          /* entries x block size */
    int64_t device_bytes = 0;    /* pool bytes given back */
    int64_t host_bytes = 0;      /* host-tier slots given back */
    int64_t snapshots = 0;
    int64_t kept_tokens = 0;     /* the chain's start, as asked, whole blocks */
    int64_t kept_shared = 0;     /* blocks beyond that on another stored session's chain */
    int64_t kept_live = 0;       /* blocks a running or waiting request holds */
};

/* ------------------------------------------------------------------ the cache */
class PrefixCache {
public:
    /* `kv` must already be configured. The cache holds a KV reference on every block it indexes,
     * so an indexed block is never on the free list and cannot be handed to another sequence.
     *
     * `checkpoint_interval` is the scheduler's resolved interval, not cfg.checkpoint_interval:
     * chunk_geometry_resolve rounds the flag up to the chunk quantum. Using the raw value would
     * make reserve_checkpoint refuse every checkpoint and lookups search the wrong stride. */
    int configure(const Config& cfg, KVManager* kv, int64_t checkpoint_interval);
    /* FreeToken: exact hybrid pages are keyed by their complete cold prefill chunk. */
    void set_exact_chunk_grid(int64_t grid, int64_t origin) {
        exact_grid_ = grid > 0 ? grid : 0;
        exact_origin_ = origin;
    }
    BlockHash checkpoint_key(const std::vector<int32_t>& tokens,
                             const std::vector<MMContentKey>& mm, int64_t pos) const {
        const int64_t end = std::min<int64_t>(pos, (int64_t)tokens.size());
        const auto hs = chain_all(tokens, mm, block_size_ > 0 ? end / block_size_ : 0);
        return hs.empty() ? BlockHash{} : hs.back();
    }
    bool enabled() const { return enabled_; }
    /* Tokens one entry covers: the lcm of the cached groups' block sizes. Hits, publishes and
     * checkpoint keys are whole entries. */
    int64_t block_size() const { return block_size_; }
    int64_t checkpoint_interval() const { return interval_; }
    const std::vector<int32_t>& cached_groups() const { return groups_; }
    /* Whether the host tier can hold this cache's entries. False when a cached group streams: its
     * rows are addressed by the sequence's block ids, so entries stay in VRAM and are evicted
     * whole (docs/KV-STREAMING.md §4.1). */
    bool demotable() const { return demotable_; }
    /* Blocks of cached_groups()[i] in one entry: block_size() over that group's block size. */
    int64_t blocks_per_entry(size_t i) const { return i < per_.size() ? per_[i] : 0; }
    /* The group of each block an entry holds, in the order of its block list (Entry::blocks,
     * TierJob::blocks, rebind): cached_groups()[i] repeated blocks_per_entry(i) times. The idle
     * tiers move an entry as these blocks, one per slot. Equal to cached_groups() when every
     * cached group has the same block size. */
    const std::vector<int32_t>& entry_groups() const { return entry_groups_; }
    std::string report() const;

    /* Overrides the policy configure() installed from Config::checkpoint_policy (custom policies,
     * tests). Call after configure(), which would undo it. */
    void set_policy(std::unique_ptr<CheckpointPolicy> p);
    /* The drafter's refill window (CheckpointPolicy::set_floor), held across set_policy. */
    void set_drafter_floor(int64_t tokens);
    const CheckpointPolicy& policy() const { return *policy_; }

    /* ---------------- full attention ---------------- */
    /* Longest cached prefix, in whole entries, plus the checkpoint that covers the linear state
     * for a prefix of it. `blocks_out`, when given, receives the block ids per cached group, in
     * cached_groups() order, ready for KVManager::adopt(): blocks_per_entry(i) of them an entry.
     *
     * Never returns the whole prompt: one token must remain to compute a logit from, so a full
     * hit is backed off by one entry. `cap`, when given, ends the hit early (HitCap). `session`,
     * when not 0, is the request being admitted: it takes over the snapshots on the chain it hit
     * (take_over). */
    CacheHit lookup(const std::vector<int32_t>& tokens,
                    const std::vector<MMContentKey>& mm,
                    std::vector<std::vector<int32_t>>* blocks_out = nullptr,
                    const HitCap* cap = nullptr, uint64_t session = 0) const;

    /* Publish the whole entries a sequence just filled. `blocks[i]` is the block table of
     * cached_groups()[i]. Takes a KV reference on every newly indexed block. */
    int insert(uint64_t session,
               const std::vector<int32_t>& tokens,
               const std::vector<MMContentKey>& mm,
               const std::vector<std::vector<int32_t>>& blocks,
               int64_t n_tokens,
               const std::string& label = std::string());

    /* Release up to `n_entries` least recently used entries' KV references; returns how many went.
     * Entries a live sequence also holds are skipped, since releasing them frees nothing. With the
     * idle tiers on, an entry whose host copy has landed moves to the host tier instead of being
     * forgotten; only one with no host copy is lost. */
    int64_t evict_lru(int64_t n_entries);
    /* Give back up to `n_entries` entries' blocks for the scheduler's reserve without losing
     * anything a later turn could reuse. One oldest-first walk that skips entries a live sequence
     * also holds and, with the tiers on, entries whose host copy has not landed yet. Pressure
     * that cannot wait goes to evict_lru. Returns how many entries went: each gave back
     * blocks_per_entry(i) blocks of cached group i. */
    int64_t trim(int64_t n_entries);
    /* Give up every VRAM copy that costs nothing to drop: host copy landed and no running sequence
     * reading it. Used when the memory has a better use (the expert slab's loan). Entries still
     * shared are retried after a sequence frees blocks. Returns how many entries left VRAM.
     *
     * `loan_goal` is the loan the slab is short of, in tokens (0: none). An entry with no host
     * copy coming (the host tier is full of context worth more) never becomes clean, and the
     * loan reaches down only to the highest held block: a few such entries high in the pool keep
     * the slab from everything below them for as long as the server runs. Where they are
     * stragglers (KVManager::straggler_mark: giving them up lends at least sixteen blocks for
     * each one lost) and stand inside `loan_goal`, they are forgotten, and their tokens are
     * prefilled again if their conversation returns. Context that fills its part of the pool
     * stays: there the cache is worth more than the loan it holds back. */
    int64_t drop_clean(int64_t loan_goal = 0);
    int64_t size() const { return (int64_t)map_.size(); }
    /* Entries holding blocks in the pool, as opposed to only a copy off the card. */
    int64_t resident() const { return (int64_t)lru_.size(); }
    /* Pool held by the cache alone, in tokens: blocks nothing else references, i.e. what an
     * eviction would actually free. The entry count overstates this, because entries share blocks
     * with every live sequence that hit them. */
    int64_t unshared_tokens() const;
    /* Forget every entry, in VRAM or off it, and give back whatever the tiers held for them. */
    void    clear();
    /* Diagnostic: the LRU order, oldest first. */
    std::vector<BlockHash> lru_order() const;
    /* Diagnostic: clean entries drop_clean() is waiting to recheck, each listed once. */
    size_t drop_waiting() const { return drop_wait_.size(); }

    /* ---------------- idle session tiers (core/mem/kvtier.h) ---------------- */
    /* Installed after configure() when the tiers are on. Null means a plain VRAM-only cache;
     * every use is guarded. */
    void set_tiers(IdleTiers* t) { tiers_ = t; }

    /* The off-device entries this prompt's chain would hit if restored, in order, from where
     * lookup stopped. The caller promotes them and looks up again; empty means the prefix ended
     * on a miss or token mismatch. Separate from lookup because lookup is const and has no
     * device; a promotion allocates blocks and runs a scatter between steps.
     *
     * `fit` is how many entries the pool can take. A longer chain is cut, and `*cut` set: with
     * recurrent state at the deepest snapshot the shorter prefix reaches, since the hit resumes
     * there and blocks past it would come back for nothing; without, at `fit`. */
    std::vector<BlockHash> restorable_from(const std::vector<int32_t>& tokens,
                                           const std::vector<MMContentKey>& mm,
                                           int64_t fit = INT64_MAX, bool* cut = nullptr) const;
    /* Entries a restore can place now: free blocks against the carve over the blocks an entry
     * takes in the group, the least over the cached groups. Lent blocks count, since
     * KVManager::reserve_blocks recalls the loan for them. */
    int64_t restore_room() const;
    /* Room for a restore of this prompt's chain: trim(), sparing the chain itself (its blocks
     * would only have to come back too) and every chain a waiting request holds (hold()), whose
     * next attempt would restore it again. Only entries whose host copy has landed leave VRAM,
     * so nothing is lost; with no host tier it takes nothing. In entries, as trim(). */
    int64_t trim_for(const std::vector<int32_t>& tokens, const std::vector<MMContentKey>& mm,
                     int64_t n_entries);

    /* The tiers lost the bytes. Erases the entry without releasing blocks: an off-device entry
     * owns none, and its recorded ids now belong to other sequences. Installed as
     * IdleTiers::set_drop_sink. */
    void drop_offdevice(const BlockHash& key);

    /* A restore finished and the entry's blocks now live at these ids, one per entry_groups()
     * slot. The restore's raw reservations (KVManager::reserve_block) become cache-held
     * references. */
    int rebind(const BlockHash& key, const std::vector<int32_t>& blocks);


    /* ---------------- the linear half of an idle session ----------------
     *
     * On a hybrid model a session also needs its recurrent state back, held in a checkpoint slot:
     * the engine's snapshot store in pinned host memory, KVManager::checkpoint_bytes() per rank,
     * --checkpoint-slots in total for the server. Without a reachable snapshot a session reuses
     * nothing: the scheduler clamps reuse to the last reachable checkpoint because RadBatch has
     * one query length for all KV groups (Scheduler::admit_one). A snapshot is restored straight
     * from its slot wherever the blocks are. There is no tier below the store; a retired snapshot
     * is gone and the session replays from an older one. */

    /* Keep a waiting request's chain on the card. An admission that restored a chain and was then
     * refused gives its hit back with clean entries; without a hold, drop_clean would drop them
     * and every retry would restore the chain again. drop_clean skips held entries until
     * release(); trim and evict_lru still take them. Idempotent per request. */
    void hold(uint64_t req, const std::vector<int32_t>& tokens, const std::vector<MMContentKey>& mm);
    void release(uint64_t req);
    /* Start of a step's admissions. Snapshots the previous step's lookups handed out have been
     * read by its forward by now, and the ones it reserved have been saved into (the engine
     * issues the save before the next step is planned); until this call they keep their slots
     * against retention, reserve_checkpoint's eviction and expiry. */
    void begin_step() { handed_.clear(); reserved_.clear(); }

    /* ---------------- expiry ----------------
     *
     * Remove everything no request has used for `ttl_ns`: VRAM entries, host-only entries and
     * snapshots, each returned to its pool. An entry is used by a hit or a publish; a snapshot by
     * a use of its prefix entry or a lookup handing it out. Nothing is demoted first.
     *
     * A hit or publish stamps every block of the chain with one clock, so a prefix is never older
     * than what follows it: conversations expire from the tip inwards and a shared system prompt
     * outlives its readers. Returns how many entries went. */
    int64_t expire(uint64_t now_ns, uint64_t ttl_ns);

    /* ---------------- release on request ----------------
     *
     * Forget one stored conversation now instead of at the TTL: a client that deleted it, or
     * replaced everything past a prefix (a compaction) and knows the rest is dead. `label` names
     * every row carrying it, `id` (nonzero) one row by its engine id (IdleTiers::SessionInfo).
     * The chain goes from the tip inwards like expire(): VRAM blocks to the pool, the host copy
     * with them, the session's snapshots on the released blocks to the store. `keep_tokens`
     * leaves the chain's first blocks standing (rounded up to whole blocks), so the prefix the
     * client kept still hits. A block another stored session's chain shares stays, as does one a
     * live sequence holds or a waiting request has restored; snapshots handed out or reserved
     * this step stay too. RAD_E_NOTFOUND when nothing matches, RAD_E_UNSUPPORTED without the
     * session table (no tiers attached). */
    int release_session(const std::string& label, uint64_t id, int64_t keep_tokens,
                        SessionRelease* out);
    /* ---------------- linear checkpoints ---------------- */
    /* The next position strictly after `pos` at which the scheduler must split a chunk so that a
     * checkpoint lands on a boundary (spec §7.1, §7.3). */
    int64_t next_checkpoint_after(int64_t pos) const;
    bool    is_checkpoint(int64_t pos) const { return interval_ > 0 && pos > 0 && pos % interval_ == 0; }

    /* Reserve a snapshot slot for `session` at `pos`. When the store is full, applies the
     * retention policy, then evicts the least recently used snapshot (a use is its reservation or
     * a lookup resuming from it), a tip like any other. A snapshot handed out this step is never
     * released, by retention or eviction: its admission restores from the slot ahead of the step.
     * Nor is one this call reserved and commit_checkpoint has not yet committed: the engine saves
     * into the slot after the step, and a later reservation in the same step must not run
     * retention over a save target (a second request reserving a snapshot a first session holds
     * below its tip). RAD_E_FULL when nothing can be given up. */
    int  reserve_checkpoint(uint64_t session, int64_t pos, const BlockHash& prefix,
                            int32_t* slot_out);
    /* The state for `prefix` is now in its slot; lookup() offers it only after this (Ckpt::valid). */
    void commit_checkpoint(const BlockHash& prefix);
    /* Every checkpoint of a session, ascending. */
    std::vector<int64_t> checkpoints(uint64_t session) const;
    /* Apply the retention policy to one session now, keeping also what a lookup resumed from
     * since the session's tip was written. Returns how many snapshots were released. A session
     * left with none is erased from the index. */
    int64_t retain(uint64_t session);

    /* Sessions in the checkpoint index. Invariant: since a session is a request id, this tracks
     * sessions holding a live snapshot, not every request served. */
    int64_t sessions_tracked() const { return (int64_t)ck_by_session_.size(); }

    struct Stats {
        int64_t lookups = 0, block_hits = 0, block_misses = 0;
        int64_t ckpt_hits = 0, ckpt_misses = 0;
        int64_t evictions = 0, ckpt_evictions = 0;
        /* ckpt_evictions by reason: released by the retention policy, or taken as the least
         * recently used of a store retention left full. `ckpt_refused` counts reservations that
         * found every slot handed out this step and went without a snapshot. */
        int64_t ckpt_retired = 0, ckpt_lru = 0, ckpt_refused = 0;
        /* Entries whose VRAM copy was given up with the host copy kept; not evictions, since
         * nothing was lost. */
        int64_t drops = 0;
        /* Entries and snapshots removed by --prefix-cache-ttl, from any tier. */
        int64_t expired = 0, ckpt_expired = 0;
        /* Entries and snapshots a client released (release_session). */
        int64_t released = 0, ckpt_released = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    struct Entry {
        /* blocks_per_entry(i) per cached group, in groups_ order: entry_groups_[k] owns blocks[k] */
        std::vector<int32_t> blocks;
        std::vector<int32_t> tokens;    /* the entry's token ids, compared on hit */
        /* On the LRU exactly while the cache holds the blocks; `lru` is valid only then. An
         * off-device entry owns no block, so eviction must not reach it. */
        bool in_lru = false;
        bool drop_wait = false;         /* in drop_wait_: clean, but a sequence still reads it */
        /* Blocks ever published right after this one, saturating: the ways prompts went on from
         * this head (CacheHit::branch_at). An evicted child still counts. */
        uint8_t n_children = 0;
        /* Last hit or publish, monotonic clock. Every path that moves an entry to the LRU's back
         * stamps it, so the list is in this order and expiry stops at the first entry it keeps. */
        uint64_t used_ns = 0;
        std::list<BlockHash>::iterator lru;
    };
    struct Ckpt {
        uint64_t  session = 0;
        int64_t   pos = 0;
        BlockHash prefix;
        int32_t   slot = -1;
        /* `clock_` values: the last use (reservation or hand-out, the eviction order), the
         * reservation, and the last hand-out (0 if none). */
        uint64_t  used = 0, made = 0, resumed = 0;
        uint64_t  used_ns = 0;          /* reserved, or handed out by a lookup */
        /* The state has been written into the slot. Reservation happens when the chunk is
         * planned and the copy after the step that computes it; in between the slot still holds
         * the previous tenant's state and must not be handed out. */
        bool      valid = false;
    };

    int64_t exact_grid_ = 0, exact_origin_ = 0;
    std::vector<BlockHash> chain_all(const std::vector<int32_t>& tokens,
                                     const std::vector<MMContentKey>& mm,
                                     int64_t n_blocks) const;
    void touch(Entry& e, uint64_t now_ns) const;
    void touch(const BlockHash& h, uint64_t now_ns) const;
    /* trim(), leaving the entries in `spare` (when given) where they are. */
    int64_t trim_sparing(int64_t n_entries,
                         const std::unordered_set<BlockHash, BlockHashHash>* spare);
    /* A live sequence also holds one of this entry's blocks, so releasing it frees nothing. */
    bool shared_with_live(const Entry& e) const;
    /* Whether `e` holds the block_size_ token ids at `src`, so a hash collision is a miss. */
    bool same_tokens(const Entry& e, const int32_t* src) const;
    /* Gives the entry's blocks in every group back to the KV manager. */
    void release_blocks(const Entry& e);
    /* Takes the cache's reference on the blocks of entry `b` of a sequence's tables `bt` (one per
     * cached group) into `out`, entry_groups_ order; on a refusal releases exactly what it took
     * and returns false. */
    bool retain_entry(const std::vector<std::vector<int32_t>>& bt, int64_t b,
                      std::vector<int32_t>* out);
    /* Retire one snapshot: free its slot and tell the tiers. The session index is left to the
     * caller. */
    void retire_ckpt(std::unordered_map<BlockHash, Ckpt, BlockHashHash>::iterator it);
    /* Move every snapshot on the chain `hs` at or below `upto` tokens under `session`. A
     * conversation's snapshots belong to its latest turn: sessions are request ids, so otherwise
     * each earlier turn's deepest snapshot would stay a protected tip, and a running turn would be
     * a second session beside its conversation, two ladders where retention should keep one.
     * The admission's lookup and the publish both call it. */
    void take_over(uint64_t session, const std::vector<BlockHash>& hs, int64_t upto) const;
    /* Take one key out of a session's index entry, and the entry when it empties. */
    void unindex_ckpt(uint64_t session, const BlockHash& key);
    /* For the debug lines: whether `c` is its session's deepest snapshot, and seconds since a
     * monotonic stamp. */
    bool is_tip(const Ckpt& c) const;
    /* The deepest valid snapshot on the chain `hs` at or below `limit` tokens; end() if none.
     * Checkpoints sit on any block boundary (the interval or a prompt's tail), so this probes the
     * distinct snapshot positions deepest-first (ck_positions_) rather than either every snapshot
     * or every block of the chain. */
    std::unordered_map<BlockHash, Ckpt, BlockHashHash>::iterator
    deepest_ckpt(const std::vector<BlockHash>& hs, int64_t limit) const;
    static double idle_s(uint64_t used_ns);

    KVManager*  kv_ = nullptr;
    IdleTiers*  tiers_ = nullptr;
    bool        enabled_ = false;
    int64_t     block_size_ = 0;
    int64_t     interval_ = 0;
    std::vector<int32_t> groups_;         /* paged groups this cache indexes */
    bool demotable_ = false;              /* the host tier can hold every group in an entry */
    std::vector<int32_t> per_;            /* blocks of groups_[i] an entry holds */
    std::vector<int32_t> entry_groups_;   /* the group of each of an entry's blocks */
    bool        has_linear_ = false;

    mutable std::unordered_map<BlockHash, Entry, BlockHashHash> map_;
    mutable std::list<BlockHash> lru_;    /* front = oldest; resident entries only */
    /* Clean entries a sequence was still reading at drop_clean's last pass, and the pool's
     * sequence-free count then. They are rechecked only once that count moves. */
    std::vector<BlockHash> drop_wait_;
    std::vector<BlockHash> drop_scratch_;
    /* drop_stragglers: the marks and held tops it last walked the list for, and when. It walks
     * again when a sequence frees blocks, or once they have moved and a second has passed. */
    std::vector<int64_t> strag_mark_, strag_seen_;
    uint64_t    strag_ns_ = 0;
    int64_t drop_stragglers(int64_t loan_goal, bool recheck);
    uint64_t    frees_seen_ = 0;
    /* hold(): keys per waiting request, and holders per key. A release also triggers a recheck
     * of drop_wait_. */
    std::unordered_map<uint64_t, std::vector<BlockHash>> holds_;
    std::unordered_map<BlockHash, int32_t, BlockHashHash> held_;
    bool        released_ = false;
    mutable std::unordered_map<BlockHash, Ckpt, BlockHashHash> ck_by_prefix_;
    mutable std::unordered_map<uint64_t, std::vector<BlockHash>> ck_by_session_;
    /* Snapshots per token position, for deepest_ckpt's deepest-first probe. Maintained where
     * ck_by_prefix_ gains (reserve_checkpoint) and loses (retire_ckpt) an entry. */
    std::map<int64_t, int64_t> ck_positions_;
    /* Snapshots lookup handed out since begin_step; kept in their slots until the next one. */
    mutable std::vector<BlockHash> handed_;
    bool handed(const BlockHash& h) const {
        return std::find(handed_.begin(), handed_.end(), h) != handed_.end();
    }
    /* Snapshots reserve_checkpoint reserved and commit_checkpoint has not committed: the step's
     * save targets, kept in their slots until the commit, or until the next step begins when the
     * save was skipped (its request gone). */
    mutable std::vector<BlockHash> reserved_;
    bool reserved(const BlockHash& h) const {
        return std::find(reserved_.begin(), reserved_.end(), h) != reserved_.end();
    }
    mutable uint64_t clock_ = 0;
    std::unique_ptr<CheckpointPolicy> policy_;
    int64_t drafter_floor_ = 0;
    mutable Stats stats_;
    std::string report_;
};

}  /* namespace rad */
