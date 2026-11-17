/* The all-reduce shims, and the init/fini they need.
 *
 * init/fini exist in the ABI for this (spec §2.1): libr4d's all-reduces handshake through peer
 * signal buffers and a scratch region that must be allocated on every rank, published to every
 * other rank and made peer-accessible before the first launch. That allocates, synchronises and
 * needs every rank, so it cannot run on the step path.
 *
 * No IPC handles: radiance runs one process with a thread per rank sharing an address space
 * (spec §1). Once hipDeviceEnablePeerAccess has run, a device pointer from rank 0's card is valid
 * on rank 1's card (unified virtual addressing), so the exchange is a pointer written to a shared
 * array plus a barrier (`r4d_ar_ipc_enable_peer`). This avoids handle marshalling and
 * hipIpcOpenMemHandle, which maps a second VA range for the same pages and must be closed in order
 * at shutdown.
 *
 * The barrier assumes every rank runs the same declare: rad_arch_declare runs once per rank with
 * the same plugin and model, so the Nth all-reduce instance is the Nth on every rank and the
 * rendezvous is keyed on arrival order alone. A rank declaring a different graph fails at startup
 * with a 60-second timeout and a named error.
 */

#include "r4d_args.h"
#include "r4d.h"
#include "r4d_ar_shards.h"
#include "r4d_ar_links.h"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <chrono>

static_assert(R4D_AR_TWOSHOT_WIDE_MAX_ELEMS == 268435456,
              "r4d_rows.cpp restates this bound; libr4d changed it");
static_assert(R4D_MAX_RANKS == 8, "the 8-rank entry point names exactly seven peers");

/* ------------------------------------------------------------------ throw conversion
 *
 * The all-reduce entry points throw; the ABI returns error codes, so every call is wrapped. The
 * shim pre-checks everything the entry points throw about, so an escaping throw is a library or
 * device fault: it becomes RAD_E_DEVICE and is printed once. */
static int r4d_caught(const char* who, const char* what) {
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "[r4d] %s threw: %s\n", who, what ? what : "(null)");
    }
    return RAD_E_DEVICE;
}

#define R4D_TRY(who, ...)                                                       \
    do {                                                                        \
        try { __VA_ARGS__; }                                                    \
        catch (const std::exception& e_) { return r4d_caught(who, e_.what()); } \
        catch (...) { return r4d_caught(who, "unknown exception"); }            \
    } while (0)

/* ------------------------------------------------------------------ the rendezvous */

namespace {

struct Rendezvous {
    std::mutex              m;
    std::condition_variable cv;
    int  arrived = 0;
    int  epoch   = 0;
    long scratch[R4D_MAX_RANKS] = { 0 };
    long flags  [R4D_MAX_RANKS] = { 0 };
    long seq    [R4D_MAX_RANKS] = { 0 };
    int  dev    [R4D_MAX_RANKS] = { 0 };
    /* Each rank's card's link, which the eight-rank two-shot's weights are derived from. */
    R4DArLink ar8_link[R4D_MAX_RANKS];
    int  failed = 0;
};

/* One rendezvous per tensor-parallel group. Several engines may run in one process
 * (docs/MULTIMODEL.md), each with its own ranks 0..world-1, so the ranks that meet are the ranks of
 * one group: the core names it in init's `tp_group` parameter (absent, the one group 0). Entries
 * live for the process; a group's id is reused only by a later engine with the same ranks. */
std::mutex                                       g_rv_mu;
std::map<long long, std::unique_ptr<Rendezvous>> g_rvs;

Rendezvous& rv_for(long long group) {
    std::lock_guard<std::mutex> lk(g_rv_mu);
    std::unique_ptr<Rendezvous>& r = g_rvs[group];
    if (!r) r.reset(new Rendezvous());
    return *r;
}

/* Sense-reversing barrier on a monotone epoch, so a rank that wakes and re-enters for the next
 * instance cannot be counted twice. The timeout is not recovery; it turns a startup deadlock into
 * a named error instead of a silent hang. */
int rv_barrier(Rendezvous& rv, int world) {
    std::unique_lock<std::mutex> lk(rv.m);
    const int e = rv.epoch;
    if (++rv.arrived == world) {
        rv.arrived = 0;
        rv.epoch++;
        rv.cv.notify_all();
        return RAD_OK;
    }
    const bool ok = rv.cv.wait_for(lk, std::chrono::seconds(60),
                                   [&] { return rv.epoch != e; });
    /* A rank that gave up takes its arrival back: a peer that failed before arriving would
     * otherwise leave it counted, and the group's next engine (its id is reused) would be released
     * one rank early. */
    if (!ok) --rv.arrived;
    return ok ? RAD_OK : RAD_E_STATE;
}

/* One resolved instance's view of the world: its own buffers and its peers', the latter in
 * ascending global rank with this rank removed (the order libr4d's wide entry points take). */
struct ArInst {
    int  rank = 0, world = 0;
    long long group = 0;      /* the tensor-parallel group it rendezvoused in (rv_for) */
    long my_scratch = 0, my_flags = 0, my_seq = 0, locpack = 0;
    /* A 2-rank host-mailbox instance's VRAM mirror of its slots, 0 otherwise (r4d_ar_seq.h). */
    long my_stage = 0;
    long peer_scratch[R4D_MAX_PEERS] = { 0 };
    long peer_flags  [R4D_MAX_PEERS] = { 0 };
    long slot_bytes = 0;      /* per-slot capacity, 16-byte aligned                  */
    long slot16     = 0;      /* the same in 16-byte words                           */
    long max_numel  = 0;      /* the declared bound the sizing was computed from     */
    int  esize      = 2;
    int  dtype      = 0;      /* libr4d code: 0 bf16, 1 fp16, 2 fp32                 */
    int  max_blocks = 0;      /* what the flag array was sized for; a hard bound      */
    /* Block count a launch may use: a tuning, unlike the max_blocks capacity. Each block drives its
     * own flag handshake, so more blocks help only until the link saturates, then add round trips
     * (at a 2048-token chunk the capacity allows 936 blocks; 24 is used). */
    int  nb_cap     = 0;
    /* Exchange buffers: 0 in each rank's VRAM, written by peers over P2P; 1 in pinned host memory
     * each card reaches over its own link (see build). */
    int  host       = 0;
    long gate_ctr   = 0;      /* the blocks that have seen their peer's bytes land (r4d_ar_seq.h) */
    /* The eight-rank two-shot's per-rank shard weights (r4d_ar_shards.h), fixed at build so the
     * scratch was sized for them and a recorded launch keeps its table. */
    int  shard_weighted = 0;
    int  shard_w[R4D_MAX_RANKS] = { 0 };
    /* Its piece count, 0 for the size rule (r4d_ar8_auto_pieces); a launch's `pieces` parameter
     * overrides it. */
    int  pieces = 0;
    /* Messages under these sizes (elements) run even and in one piece, whatever is configured. */
    long long weights_min = R4D_AR8_MIN_ELEMS_DEFAULT;
    long long pieces_min  = R4D_AR8_MIN_ELEMS_DEFAULT;
};

/* A parameter that may be declared as a range: init gets one band's frozen geometry, and buffers
 * are sized for the band's upper bound. */
long long pgeti_hi(const RadParam* p, int n_p, const char* k, long long dflt) {
    if (!p || !k) return dflt;
    for (int i = 0; i < n_p; ++i) {
        if (std::strcmp(p[i].key, k) != 0) continue;
        if (p[i].kind == RAD_P_INT)   return p[i].ival;
        if (p[i].kind == RAD_P_RANGE) return p[i].ihi;
    }
    return dflt;
}

long align16(long v) { return (v + 15) & ~15L; }

/* The eight-rank two-shot's settings for the next build: the engine's defaults unless
 * r4d_ar8_test_settings set others. */
std::mutex     g_ar8_m;
R4DAr8Settings g_ar8;

R4DAr8Settings ar8_settings() {
    std::lock_guard<std::mutex> lk(g_ar8_m);
    return g_ar8;
}

/* This rank's card's slowest PCIe hop, from sysfs by the runtime's bus id. */
bool ar8_own_link(R4DArLink* l) {
    int dev = 0;
    char bus[64] = { 0 };
    if (hipGetDevice(&dev) != hipSuccess ||
        hipDeviceGetPCIBusId(bus, (int)sizeof bus, dev) != hipSuccess)
        return false;
    unsigned dom = 0, b = 0, d = 0, f = 0;
    if (std::sscanf(bus, "%x:%x:%x.%x", &dom, &b, &d, &f) != 4) return false;
    char bdf[32];
    std::snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.%x", dom, b, d, f);
    return r4d_ar_link_bottleneck("", bdf, l);
}

/* The eight-rank two-shot's choice for one message, a function of its size and the instance
 * alone: a recorded launch keeps the table and piece count it was recorded with, and replays of
 * the same size agree with it. */
const int* ar8_weights_for(const ArInst* in, long numel) {
    return in->shard_weighted && numel >= in->weights_min ? in->shard_w : nullptr;
}
long long ar8_pieces_for(const ArInst* in, long numel, long long asked) {
    if (numel < in->pieces_min) return 1;
    return asked > 0 ? asked : r4d_ar8_auto_pieces(numel);
}

/* The eight-rank two-shot's settings, fixed before anything is sized: each rank publishes its own
 * card's link, and after one barrier every rank derives the same weights from the same eight
 * entries. */
int ar8_settle(ArInst* in, int rank, int world) {
    Rendezvous& rv = rv_for(in->group);
    static std::once_flag said_links, said_unknown, said_set;
    const R4DAr8Settings set = ar8_settings();
    R4DArLink own;
    if (set.weights == R4D_AR8_WEIGHTS_AUTO && !ar8_own_link(&own)) own = R4DArLink();
    {
        std::lock_guard<std::mutex> lk(rv.m);
        rv.ar8_link[rank] = own;
    }
    const int rc = rv_barrier(rv, world);
    if (rc != RAD_OK) return rc;

    const R4DArLink* all = rv.ar8_link;
    in->pieces = set.pieces;
    in->weights_min = set.weights_min;
    in->pieces_min = set.pieces_min;

    int w[R4D_MAX_RANKS] = { 0 };
    if (set.weights == R4D_AR8_WEIGHTS_SET) {
        std::memcpy(w, set.w, sizeof(set.w));
        std::call_once(said_set, [&w] {
            std::fprintf(stderr, "[r4d] eight-rank two-shot shards weighted "
                         "%d,%d,%d,%d,%d,%d,%d,%d (r4d_ar8_test_settings)\n", w[0], w[1], w[2],
                         w[3], w[4], w[5], w[6], w[7]);
        });
    } else if (set.weights == R4D_AR8_WEIGHTS_AUTO) {
        double gbs[R4D_MAX_RANKS] = { 0 };
        for (int r = 0; r < world; ++r) gbs[r] = r4d_ar_link_gbs(all[r]);
        if (!r4d_ar_weights_from_rates(gbs, world, w)) {
            std::call_once(said_unknown, [all, world] {
                std::string who;
                for (int r = 0; r < world; ++r)
                    if (r4d_ar_link_gbs(all[r]) <= 0.0) who += " " + std::to_string(r);
                std::fprintf(stderr, "[r4d] eight-rank two-shot: no PCIe link speed in sysfs for "
                             "rank%s; shards stay even\n", who.c_str());
            });
            return RAD_OK;
        }
        std::call_once(said_links, [all, world, &w] {
            std::string l;
            for (int r = 0; r < world; ++r) {
                char one[96];
                std::snprintf(one, sizeof one, "%s r%d %.1f GT/s x%d %.1f GB/s w%d",
                              r ? "," : "", r, all[r].gts, all[r].width,
                              r4d_ar_link_gbs(all[r]), w[r]);
                l += one;
            }
            std::fprintf(stderr, "[r4d] eight-rank two-shot shard weights from the slowest PCIe "
                         "hop of each card:%s\n", l.c_str());
        });
        bool same = true;
        for (int r = 1; r < world; ++r) same = same && w[r] == w[0];
        if (same) return RAD_OK;   /* equal links: the even split, on the unweighted path */
    } else {
        return RAD_OK;
    }
    in->shard_weighted = 1;
    std::memcpy(in->shard_w, w, sizeof(in->shard_w));
    return RAD_OK;
}

/* Sequence counter allocation in words: 2 * nbw (r4d_ar_seq.h), then r4d_ar_link_landed's block
 * counter, padded to a cache line. */
long kSeqWords(long nbw) { return 2L * nbw + 32; }

/* Build the world. `max_numel` is the largest message any launch on this instance carries.
 *
 * One sizing rule for the whole family: a slot holds a full message. The one-shots need exactly
 * that; the two-shot's two regions (2 * message / world_size) and the tiered-int8 wire's two
 * images (2 * ceil(shard/32) * 38 bytes) are smaller. RadInitFn is not told which row resolved,
 * so sizing for the maximum avoids reconstructing which kernel this instance serves. The one
 * exception is a weighted eight-rank two-shot whose largest weight is over half the sum: its two
 * regions of the longest shard outgrow a message, and the slot grows to hold them.
 *
 * Cost: at a 4096-token chunk on a 5120-wide residual, 40 MiB a slot and 80 MiB for the pair. One
 * set serves every op with the same (rank, world, dtype, bound); see the cache below. */
int build(ArInst* in, long long group, int rank, int world, long long max_numel, int dtype_code,
          int host) {
    if (world < 2 || world > R4D_MAX_RANKS) return RAD_E_SHAPE;
    if (rank < 0 || rank >= world) return RAD_E_INVAL;
    if (max_numel <= 0) return RAD_E_INVAL;

    in->group = group;
    in->rank = rank;
    in->world = world;
    in->host = host;
    in->dtype = dtype_code;
    in->esize = r4d_ar_esize(dtype_code);
    in->max_numel = (long)max_numel;
    in->slot_bytes = align16((long)max_numel * in->esize);
    in->slot16 = in->slot_bytes / 16;
    if (world == 8) {
        const int rc8 = ar8_settle(in, rank, world);
        if (rc8 != RAD_OK) return rc8;
    }
    if (in->shard_weighted) {
        const long need16 =
            2 * r4d_ar_shard_max_bound((long)max_numel * in->esize / 16, 8, in->shard_w);
        /* Sized only if some message this instance carries is weighted. */
        if (max_numel >= in->weights_min && need16 > in->slot16) {
            in->slot16 = need16;
            in->slot_bytes = need16 * 16;
        }
    }

    int maxb = 0, nb_design = 0, max_peers = 0;
    if (world == 2) {
        maxb = r4d_ar_max_blocks();
    } else {
        r4d_ar_wide_dims(&maxb, &nb_design, &max_peers);
        if (world - 1 > max_peers) return RAD_E_SHAPE;
    }
    if (maxb <= 0) return RAD_E_DEVICE;
    in->max_blocks = maxb;
    /* R4D_AR_MAX_BLOCKS: a ceiling on every collective's block count, for two engines sharing one
     * set of cards. Each block of a collective spins until its twin on the peer card has pushed, and
     * a prefill-sized ar_rmsnorm_quant_fp8 launches up to 256 blocks of 16 waves, twice what an
     * R9700 holds. One engine survives that (both cards dispatch blocks in the same order, so the
     * lowest unfinished block is resident on both); two engines do not: one engine's spinning blocks
     * fill a card while their twins wait behind the other engine's kernel on the peer card, which
     * waits for a slot on the full card. Not capped: the hyper-connection gathers of the MoE paths
     * (r4d_ar_gather_hc_write without its inverse scratch, r4d_ar_hc_write_wht6), whose LDS table
     * needs a minimum block count, which may exceed the ceiling (dense models do not use them).
     * Two TP2 engines on two R9700s (Qwen3.8-27B and Gemma 3 27B) hung within a minute of
     * concurrent prefill; MES then failed to evict the queues and the failed reset froze the
     * host. Capped in both engines so that their spinners together cannot
     * fill a card, the cycle cannot close. The flag and counter arrays keep their full size (maxb);
     * only the launch counts shrink, and each element's sum is the same. Unset, nothing changes. */
    if (const char* e = std::getenv("R4D_AR_MAX_BLOCKS")) {
        const int cap = std::atoi(e);
        if (cap > 0 && cap < in->max_blocks) in->max_blocks = cap;
    }
    /* Launch ceilings: 24 at two ranks, the library's design count at width. The compressed
     * 2-rank wire uses up to 48, set at its own call site: it moves three eighths of the bytes, so
     * it needs more blocks to saturate the link. */
    in->nb_cap = (world == 2) ? 24 : (nb_design > 0 ? nb_design : maxb);
    if (in->nb_cap > in->max_blocks) in->nb_cap = in->max_blocks;

    /* Slots: one per (parity, source rank) at width; two at world 2, where there is one remote
     * writer. Flags: one word per (block, source rank) at width, one per block at world 2.
     * Sequence counters: a handshake and a parity word per block of capacity (r4d_ar_seq.h),
     * device-resident so a captured graph replays. */
    const int  nslots     = (world == 2) ? 2 : 2 * world;
    const long flag_words = (long)maxb * ((world == 2) ? 1 : world);

    char handle[R4D_AR_HANDLE_BYTES];
    int  hlen = 0;
    long sc = 0, fl = 0, sq = 0, lp = 0;
    int  dev = 0;

    /* Fine-grained: the peer writes the scratch and this rank reads it bypassing L2, so a
     * coarse-grained allocation could leave the peer's bytes in a cache this rank never
     * invalidates, and the launches run without an acquire fence. r4d_ar_ipc_alloc throws when it
     * cannot allocate fine-grained memory; there is no coarse fallback.
     *
     * Without peer access the scratch and flags move to pinned host memory and nothing else
     * changes: kernels touch other ranks only through those two, so no card addresses another's
     * VRAM. A message then crosses two links instead of one; only rows requiring device.p2p == 0
     * take this path (r4d_rows.cpp). Sequence counters stay in VRAM, since only their own rank
     * reads them, and coarse-grained: only this rank's kernels touch them, with device- and
     * agent-scope atomics, which then complete in L2 rather than past it -- the first returning
     * atomic of every launch (r4d_ar_seq_take) is on the push's path. */
    R4D_TRY("ar_ipc_alloc",
        if (hipGetDevice(&dev) != hipSuccess) return RAD_E_DEVICE;
        if (host) {
            sc = r4d_ar_host_alloc((long)nslots * in->slot_bytes);
            fl = r4d_ar_host_alloc((flag_words + 2L * maxb) * 4);
        } else {
            sc = r4d_ar_ipc_alloc((long)nslots * in->slot_bytes, 1, handle, &hlen);
            fl = r4d_ar_ipc_alloc((flag_words + 2L * maxb) * 4, 1, handle, &hlen);
        }
        sq = r4d_ar_ipc_alloc(kSeqWords(maxb) * 4, 0, handle, &hlen));

    /* A host mailbox is read over the link past every cache, so the 2-rank kernels' push check
     * copies what it reads into this VRAM mirror for the reduce, crossing the link once
     * (r4d_ar_seq.h). Coarse-grained: only this rank's kernels touch it, each block its own words.
     * The wide family has no push check and reads its slots once. Host-memory slots free the same
     * VRAM over P2P, so the mirror keeps the footprint where the P2P instance has it. */
    long stg = 0;
    if (host && world == 2) {
        R4D_TRY("ar_ipc_alloc(stage)",
            stg = r4d_ar_ipc_alloc((long)nslots * in->slot_bytes, 0, handle, &hlen));
    }

    /* This rank's own packed copy, for the rotated 6-bit wire only: the reduce must fold exactly
     * the bytes it sent. The 6-bit payload is 3/8 of the bf16 message and the f32 scales 1/32 of
     * it, laid out from half the slot; a row padded to whole 512-element chunks
     * (ar_rmsnorm_quant_fp8) adds up to 7 groups' scales a row, so the scales get a sixteenth
     * (enough for any row of 512 or more) plus a page. ar_rmsnorm_quant checks the fit. */
    if (in->dtype != 2) {
        R4D_TRY("ar_ipc_alloc(locpack)",
            lp = r4d_ar_ipc_alloc(in->slot_bytes / 2 + in->slot_bytes / 16 + 4096, 0,
                                  handle, &hlen));
    }

    in->my_scratch = sc;
    in->my_flags   = fl;
    in->my_seq     = sq;
    in->locpack    = lp;
    in->my_stage   = stg;
    in->gate_ctr   = sq + 2L * maxb * 4;   /* the word after the sequence counters (r4d_ar_seq.h) */

    Rendezvous& rv = rv_for(group);
    {
        std::lock_guard<std::mutex> lk(rv.m);
        rv.scratch[rank] = sc;
        rv.flags[rank]   = fl;
        rv.seq[rank]     = sq;
        rv.dev[rank]     = dev;
    }
    int rc = rv_barrier(rv, world);
    if (rc != RAD_OK) return rc;

    /* Peer access is per context, so each rank enables its own, after the publish barrier where
     * peer device ordinals are known. A host can expose more HIP devices than ranks (e.g. an
     * integrated GPU), so the ordinal comes from the peer rather than the rank. A peer on this
     * rank's own card needs none (the runtime refuses to enable it): r4d_selftest runs eight ranks
     * on one card that way, each on its own stream. */
    int j = 0;
    for (int r = 0; r < world; ++r) {
        if (r == rank) continue;
        if (!rv.scratch[r] || !rv.flags[r]) return RAD_E_STATE;
        in->peer_scratch[j] = rv.scratch[r];
        in->peer_flags[j]   = rv.flags[r];
        ++j;
        if (!host && rv.dev[r] != dev)
            R4D_TRY("ar_ipc_enable_peer", r4d_ar_ipc_enable_peer((long)rv.dev[r]));
    }

    /* Second barrier: no rank may launch until every rank has enabled peer access, or the first
     * message would be pushed into a mapping the receiver has not opened. */
    return rv_barrier(rv, world);
}

/* ------------------------------------------------------------------ one world, shared
 *
 * One instance per (rank, world, dtype, message bound), not per op. The core calls init per
 * resolved instance, and a 64-layer tensor-parallel model declares 128 identical all-reduces.
 * Scratch is sized by the largest message (at a 2048-token chunk on a 5120-wide residual, ~53 MiB
 * per instance), so per-op instances would cost 6.7 GiB a card. It comes from hipMalloc, outside
 * the VRAM budget, so over-allocation shows up as the KV cache or a prefill chunk failing to fit.
 *
 * Sharing is safe. Slots are double-buffered by the parity of the launch on this instance, and a
 * rank's all-reduces run on one stream in declared order, so all-reduce k+2 (the next user of k's
 * slot) cannot launch until k+1 completed, which needed the peer's k+1 flag, set only after the
 * peer's k kernel finished reading that slot. This needs the launch's parity, not a block's: ops
 * sharing an instance use different block counts (r4d_ar_seq.h).
 *
 * The rendezvous stays symmetric because every rank declares the same graph in the same order:
 * the Nth all-reduce misses the cache on every rank or on none. */
struct CacheKey {
    long long group;          /* the tensor-parallel group: another engine's ranks never share */
    int       rank, world, dtype;
    long long numel;
    int       host;
    bool operator==(const CacheKey& o) const {
        return group == o.group && rank == o.rank && world == o.world && dtype == o.dtype &&
               numel == o.numel && host == o.host;
    }
};

struct CacheSlot {
    CacheKey key;
    ArInst*  in   = nullptr;
    int      refs = 0;
};

std::mutex             g_cache_mu;
std::vector<CacheSlot> g_cache;

int acquire(const CacheKey& k, ArInst** out) {
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (CacheSlot& s : g_cache)
            if (s.key == k && s.in) { ++s.refs; *out = s.in; return RAD_OK; }
    }
    /* Built outside the lock: build() rendezvouses with the other ranks, and holding the mutex
     * across that barrier would deadlock with the peer's lookup. Keys include the rank and each
     * rank has one declare thread, so no two builders race for one key. */
    ArInst* in = new (std::nothrow) ArInst();
    if (!in) return RAD_E_NOMEM;
    int rc = build(in, k.group, k.rank, k.world, k.numel, k.dtype, k.host);
    if (rc != RAD_OK) { delete in; return rc; }
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        g_cache.push_back(CacheSlot{ k, in, 1 });
    }
    *out = in;
    return RAD_OK;
}

void release(ArInst* in) {
    if (!in) return;
    {
        std::lock_guard<std::mutex> lk(g_cache_mu);
        for (size_t i = 0; i < g_cache.size(); ++i) {
            if (g_cache[i].in != in) continue;
            if (--g_cache[i].refs > 0) return;
            g_cache.erase(g_cache.begin() + (long)i);
            break;
        }
    }
    /* Free only this rank's own allocations; peer pointers belong to the peer. */
    if (in->host) {
        r4d_ar_host_free(in->my_scratch);
        r4d_ar_host_free(in->my_flags);
    } else {
        r4d_ar_ipc_free(in->my_scratch);
        r4d_ar_ipc_free(in->my_flags);
    }
    r4d_ar_ipc_free(in->my_seq);
    r4d_ar_ipc_free(in->locpack);
    r4d_ar_ipc_free(in->my_stage);
    delete in;
}

}  /* namespace */

/* ------------------------------------------------------------------ init / fini */

static int ar_init(const RadParam* p, int n_p, int rank, int world, int host, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;

    /* world_size is both an ABI argument and a declared parameter. The parameter is what the
     * constraint matched; the argument is what the engine runs. A mismatch means the graph and the
     * process describe different deployments, so refuse here instead of at the first handshake. */
    const long long ws_param = pgeti_hi(p, n_p, "world_size", world);
    if (ws_param != world) return RAD_E_SHAPE;

    const int dt = r4d_ar_dtype_code(r4d_pgets(p, n_p, "dtype"));
    if (dt < 0) return RAD_E_DTYPE;

    const long long numel = pgeti_hi(p, n_p, "numel", 0);
    if (numel <= 0) return RAD_E_INVAL;

    ArInst* in = nullptr;
    const long long group = pgeti_hi(p, n_p, "tp_group", 0);
    int rc = acquire(CacheKey{ group, rank, world, dt, numel, host }, &in);
    if (rc != RAD_OK) return rc;
    *out = in;
    return RAD_OK;
}

extern "C" void r4d_ar8_test_settings(const R4DAr8Settings* s) {
    std::lock_guard<std::mutex> lk(g_ar8_m);
    g_ar8 = s ? *s : R4DAr8Settings();
}

extern "C" int r4d_rad_ar_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    return ar_init(p, n_p, rank, world, 0, out);
}
extern "C" int r4d_rad_ar_host_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    return ar_init(p, n_p, rank, world, 1, out);
}

/* The fused all-reduce + norm declares `M` (range) and `n` instead of `numel`; the bound is
 * M * n, over the same world and buffers. */
static int ar_mn_acquire(const RadParam* p, int n_p, int rank, int world, int host,
                         void** out) {
    const long long M = pgeti_hi(p, n_p, "M", 0);
    const long long K = pgeti_hi(p, n_p, "n", 0);
    if (M <= 0 || K <= 0) return RAD_E_INVAL;

    ArInst* in = nullptr;
    const long long group = pgeti_hi(p, n_p, "tp_group", 0);
    int rc = acquire(CacheKey{ group, rank, world, 0 /* bf16 */, M * K, host }, &in);
    if (rc != RAD_OK) return rc;
    *out = in;
    return RAD_OK;
}

static int ar_ln_init(const RadParam* p, int n_p, int rank, int world, int host, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;

    const long long ws_param = pgeti_hi(p, n_p, "world_size", world);
    if (ws_param != world) return RAD_E_SHAPE;
    if (world != 2) return RAD_E_SHAPE;   /* the handshake is one peer flag per block */

    return ar_mn_acquire(p, n_p, rank, world, host, out);
}

extern "C" int r4d_rad_ar_ln_init(const RadParam* p, int n_p, int rank, int world, void** out) {
    return ar_ln_init(p, n_p, rank, world, 0, out);
}
extern "C" int r4d_rad_ar_ln_host_init(const RadParam* p, int n_p, int rank, int world,
                                       void** out) {
    return ar_ln_init(p, n_p, rank, world, 1, out);
}

/* The four-rank ar_hc_write, ar_gather_hc_write and ar_rmsnorm_quant_i8 (the norm declares no
 * `wire`). The key is {rank, 4, bf16, M_hi * n}, shared with a standalone all_reduce band only where
 * the bounds agree. Either way is safe on one stream: every wide kernel advances the instance's seq
 * counters once per all-reduce and writes only its (parity, source) slot (r4d_ar_wide.h), so ops on
 * one instance interleave in any order all ranks share. Exact wire only; `wire` 6 is refused. */
static int ar_hc_wide_init(const RadParam* p, int n_p, int rank, int world, int host,
                           void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;

    const long long ws_param = pgeti_hi(p, n_p, "world_size", world);
    if (ws_param != world) return RAD_E_SHAPE;
    if (world != 4) return RAD_E_SHAPE;   /* the peer bundle is compiled for four */
    if (pgeti_hi(p, n_p, "wire", 0) != 0) return RAD_E_UNSUPPORTED;

    return ar_mn_acquire(p, n_p, rank, world, host, out);
}

extern "C" int r4d_rad_ar_hc_wide_init(const RadParam* p, int n_p, int rank, int world,
                                       void** out) {
    return ar_hc_wide_init(p, n_p, rank, world, 0, out);
}
extern "C" int r4d_rad_ar_hc_wide_host_init(const RadParam* p, int n_p, int rank, int world,
                                            void** out) {
    return ar_hc_wide_init(p, n_p, rank, world, 1, out);
}

extern "C" void r4d_rad_ar_fini(void* instance) {
    /* One fini per successful init; the instance is shared, so the last release frees it. */
    release((ArInst*)instance);
}

/* ------------------------------------------------------------------ launch helpers */

/* The engine's link word, if it passed one (rad_abi.h, `link_gate`). Over P2P the fused kernels
 * end the hold when the peer's push lands, on either wire, and return RAD_LINK_ENDED. Through the
 * host mailbox the reduce itself reads across the link, so those launches return RAD_OK and leave
 * the hold to the engine. */
static long link_gate(const RadArgs* a) {
    long long g = 0;
    return rad_args_geti(a, "link_gate", &g) ? (long)g : 0L;
}

enum { AR_X = 0, AR_Y };

struct ArCall {
    const ArInst* in;
    long inp, out;
    long numel;
    int  nblocks, nthreads, drain, acq, pub;
};

/* Block count. PCIe saturates with very few blocks and more only help the local reduce, so the
 * count scales with the message and then stops: 1400 sixteen-byte words per block, floor 4,
 * ceiling max_nb. A tuned `nblocks` overrides it. */
static int nb_for(long words16, int max_nb) {
    long nb = words16 / 1400;
    if (nb < 4) nb = 4;
    if (nb > max_nb) nb = max_nb;
    if (nb > words16) nb = words16 > 0 ? words16 : 1;
    return (int)nb;
}

/* The four-rank exact two-shot at decode and verify heights (the one-shot row's band from ts_min,
 * the junction's tail form, the two-shot row up to kTsDecodeShard words a shard), over the words of
 * one shard: one block per 2688 words, floor 2, ceiling 8 (and max_nb). A two-shot takes two
 * handshakes, each a PCIe round trip a block, so a block more costs more than its share of the
 * bytes saves until the shards are large. On 4x R9700, one launch at a time on the 31B's payload,
 * 64 rows of 5376 (8 users x DFlash2's 8 rows) took 85.6 us at 16 blocks against 75.3 at 4, 32
 * rows 47.1 at 8 against 44.2 at 4, and 128 rows 143.5 at 16 against 137.9 at 8. The block count
 * moves no byte: every word is folded by one thread in ascending rank order whatever the grid. */
/* The chained two-shot (r4d_chain.h) where the row resolved with `chain`, for a decode-sized
 * message only (64 rows of an 8192-wide residual): DFlash2's verify-height reductions. */
static long ar_chain(const RadArgs* a, long numel) {
    return r4d_geti_or(a, "chain", 0) != 0 && numel <= 64L * 8192L ? 1L : 0L;
}

static constexpr long kTsDecodeShard = 43008;   /* 256 rows of 5376 at tp 4 */
static int nb_for_shard(long shard16, int max_nb) {
    long nb = shard16 / 2688;
    if (nb < 2) nb = 2;
    if (nb > 8) nb = 8;
    if (nb > max_nb) nb = max_nb;
    if (nb > shard16) nb = shard16 > 0 ? shard16 : 1;
    return (int)nb;
}

/* Shared front half of every all-reduce launch: find the instance, check the message against
 * init's sizing, resolve the tuned point. Every kernel of the family is safe in place (`y` absent
 * or equal to `x`); each entry point says why. */
static int ar_setup(const RadArgs* a, ArCall* c, int shard_nb) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;                  /* init did not run: a core sequencing bug */
    const RadTensor* x = r4d_opt(a, AR_X);
    if (!x) return RAD_E_INVAL;
    if (!r4d_contig(x)) return RAD_E_STRIDE;

    const RadTensor* y = r4d_opt(a, AR_Y);
    if (y) {
        if (!r4d_contig(y)) return RAD_E_STRIDE;
        if (r4d_numel(y) != r4d_numel(x)) return RAD_E_SHAPE;
    }

    const long n = (long)r4d_numel(x);
    if (n <= 0) return RAD_E_SHAPE;
    if (n > in->max_numel) return RAD_E_SHAPE;    /* the peer scratch was sized for max_numel */

    /* The push uses 16-byte vector stores and libr4d throws on a partial word; check it here. */
    if (((long long)n * in->esize) % 16) return RAD_E_ALIGN;
    if (!r4d_aligned(x->data, 16)) return RAD_E_ALIGN;
    if (y && !r4d_aligned(y->data, 16)) return RAD_E_ALIGN;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);

    c->in    = in;
    c->inp   = r4d_addr(x);
    c->out   = y ? r4d_addr(y) : r4d_addr(x);
    c->numel = n;
    c->nthreads = cfgv.nthreads;
    c->drain    = cfgv.drain;
    c->acq      = cfgv.acq;
    c->pub      = cfgv.pub;

    const long words16 = (long)n * in->esize / 16;
    long basis = shard_nb ? words16 / in->world : words16;
    if (shard_nb && ar8_weights_for(in, n)) {
        long off[R4D_MAX_RANKS + 1];
        basis = r4d_ar_shard_table(words16, in->world, in->shard_w, off);
    }
    /* The two-shots move shards, so they use the smaller basis and a tighter ceiling; the
     * full-message rule over-launches across the mid band. */
    int cap = in->nb_cap;
    if (shard_nb && cap > 24) cap = 24;
    /* Every block handshakes with all seven peers at eight ranks; past eight blocks the extra
     * flag rounds cost more than the added reduce bandwidth (8 against 16 blocks measured up to 3 %
     * faster from 2097152 elements on). */
    if (shard_nb && in->world == 8 && cap > 8) cap = 8;
    /* a decode-sized shard takes the decode rule (nb_for_shard), a prompt-sized one nb_for's */
    c->nblocks = cfgv.nblocks > 0 ? cfgv.nblocks
               : (shard_nb && basis <= kTsDecodeShard) ? nb_for_shard(basis, cap) : nb_for(basis, cap);
    if (c->nblocks > in->max_blocks) c->nblocks = in->max_blocks;
    return RAD_OK;
}

/* ------------------------------------------------------------------ all-gather
 *
 * The instance, scratch and flags are the all-reduce's; a slot holds a full input message, so
 * r4d_rad_ar_init sizes it from the same `numel` parameter. The wide form uses the wide family's
 * peer arrays and r4d_ar_handshake (r4d_ar_wide.h), as the wide one-shots do. Its launchers live in
 * r4d_ar_gather_Nrank.hip and are declared here, not in r4d.h, since their only caller is this
 * plugin. */
extern "C" void r4d_ar_gather_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                    long peer_flags0, long peer_flags1, long peer_flags2,
                                    long my_scratch, long my_flags, long seq_ctrs,
                                    long slot_stride16, long inp, long out, long n_elem,
                                    long dtype, long rank, long row_elems, long stream,
                                    long nblocks, long nthreads, long drain, long acq, long pub,
                                    long gate, long gate_ctr, long dpitch, long spitch, long chain);
extern "C" void r4d_ar_gather_8rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                    long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                    long peer_scratch6, long peer_flags0, long peer_flags1,
                                    long peer_flags2, long peer_flags3, long peer_flags4,
                                    long peer_flags5, long peer_flags6, long my_scratch,
                                    long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                    long out, long n_elem, long dtype, long rank, long row_elems,
                                    long stream, long nblocks, long nthreads, long drain, long acq,
                                    long pub, long dpitch, long spitch);

/* Not ar_setup: an all-gather's output is world_size times its input. With `row`, `y` may also be
 * the leading world_size * row columns of a wider buffer and `x` a row-wide window of one
 * (docs/OPS.md); `*dpitch_out` and `*spitch_out` are then their row pitches in elements, else 0. */
static int ar_gather_wide_setup(const RadArgs* a, int ws, const ArInst** out_in, ArCall* c,
                                long long* row_out, long long* dpitch_out,
                                long long* spitch_out) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;                  /* init did not run: a core sequencing bug */
    if (in->world != ws) return RAD_E_SHAPE;      /* the peer bundle is compiled for this width */

    const RadTensor* x = r4d_opt(a, AR_X);
    const RadTensor* y = r4d_opt(a, AR_Y);
    if (!x || !y) return RAD_E_INVAL;             /* the gather has no in-place form */
    if (y->data == x->data) return RAD_E_INVAL;

    const long n = (long)r4d_numel(x);
    if (n <= 0 || n > in->max_numel) return RAD_E_SHAPE;
    /* y is [world_size * numel], rank r's contribution at offset r * numel (docs/OPS.md). */
    if ((long long)r4d_numel(y) != (long long)n * in->world) return RAD_E_SHAPE;

    const long long row = r4d_pgeti_or(a->p, a->n_p, "row", 0);
    if (row < 0 || (row > 0 && (n % row || (row * in->esize) % 16))) return RAD_E_SHAPE;
    /* A row-interleaved gather's [rows][world * row] may sit in the leading columns of a wider
     * buffer, the pitch in whole 16-byte words: rows land at that pitch, the rest of each row is
     * left alone. */
    long long dpitch = 0;
    if (!r4d_contig(y)) {
        const long long p = r4d_row_pitch(y);
        if (row <= 0 || y->rank != 2 || p <= 0 || y->shape[1] != (int64_t)row * in->world ||
            (p * in->esize) % 16)
            return RAD_E_STRIDE;
        dpitch = p;
    }
    /* Likewise x may be [rows][row] read at a wider pitch: a rank's window of a row it computed
     * wider (a split linear's block-aligned superset, arch/common/rad_block_dflash2.h). */
    long long spitch = 0;
    if (!r4d_contig(x)) {
        const long long p = r4d_row_pitch(x);
        if (row <= 0 || x->rank != 2 || p <= 0 || x->shape[1] != (int64_t)row ||
            (p * in->esize) % 16)
            return RAD_E_STRIDE;
        spitch = p;
    }

    if (((long long)n * in->esize) % 16) return RAD_E_ALIGN;
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(y->data, 16)) return RAD_E_ALIGN;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long words16 = (long)n * in->esize / 16;
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    *out_in = in;
    c->in = in;
    c->inp = r4d_addr(x);
    c->out = r4d_addr(y);
    c->numel = n;
    c->nblocks = nblocks;
    c->nthreads = cfgv.nthreads;
    c->drain = cfgv.drain;
    c->acq = cfgv.acq;
    c->pub = cfgv.pub;
    *row_out = row;
    *dpitch_out = dpitch;
    *spitch_out = spitch;
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_2rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0, dpitch = 0, spitch = 0;
    int rc = ar_gather_wide_setup(a, 2, &in, &c, &row, &dpitch, &spitch);
    if (rc != RAD_OK) return rc;
    R4D_TRY("ar_gather_2rank",
        r4d_ar_gather_2rank(in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags,
                            in->my_seq, in->slot16, c.inp, c.out, c.numel, in->dtype, in->rank,
                            (long)row, (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain,
                            c.acq, (long)dpitch, (long)spitch, in->my_stage));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_3rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0, dpitch = 0, spitch = 0;
    int rc = ar_gather_wide_setup(a, 3, &in, &c, &row, &dpitch, &spitch);
    if (rc != RAD_OK) return rc;
    R4D_TRY("ar_gather_3rank",
        r4d_ar_gather_3rank(in->peer_scratch[0], in->peer_scratch[1],
                            in->peer_flags[0], in->peer_flags[1],
                            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                            c.inp, c.out, c.numel, in->dtype, in->rank, (long)row,
                            (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq, c.pub,
                            (long)dpitch, (long)spitch));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0, dpitch = 0, spitch = 0;
    int rc = ar_gather_wide_setup(a, 4, &in, &c, &row, &dpitch, &spitch);
    if (rc != RAD_OK) return rc;
    /* One handshake, placement after it local: the release is the one-shot's. */
    const long gate = link_gate(a), early = in->host ? 0L : gate;
    R4D_TRY("ar_gather_4rank",
        r4d_ar_gather_4rank(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                            in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                            c.inp, c.out, c.numel, in->dtype, in->rank, (long)row,
                            (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq, c.pub,
                            early, early ? in->gate_ctr : 0L, (long)dpitch, (long)spitch,
                            ar_chain(a, c.numel)));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

extern "C" int r4d_rad_ar_gather_8rank(const RadArgs* a, RadStream s) {
    const ArInst* in = nullptr;
    ArCall c;
    long long row = 0, dpitch = 0, spitch = 0;
    int rc = ar_gather_wide_setup(a, 8, &in, &c, &row, &dpitch, &spitch);
    if (rc != RAD_OK) return rc;
    R4D_TRY("ar_gather_8rank",
        r4d_ar_gather_8rank(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                            in->peer_scratch[3], in->peer_scratch[4], in->peer_scratch[5],
                            in->peer_scratch[6],
                            in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                            in->peer_flags[3], in->peer_flags[4], in->peer_flags[5],
                            in->peer_flags[6],
                            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                            c.inp, c.out, c.numel, in->dtype, in->rank, (long)row,
                            (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq, c.pub,
                            (long)dpitch, (long)spitch));
    return RAD_OK;
}

/* ------------------------------------------------------------------ 2-rank one-shot */

extern "C" int r4d_rad_ar_oneshot_2rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 2) return RAD_E_SHAPE;
    R4D_TRY("ar_oneshot_2rank_exact",
        r4d_ar_oneshot_2rank_exact(c.in->peer_scratch[0], c.in->my_scratch,
                                   c.in->peer_flags[0], c.in->my_flags, c.in->my_seq,
                                   c.in->slot16, c.inp, c.out, c.numel, c.in->dtype,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.in->my_stage, ar_chain(a, c.numel)));
    return RAD_OK;
}

/* Below this size the compressed wire does not pay. A decode all-reduce (eight residual rows,
 * 80 KiB) already runs at ~92% of one-way link bandwidth on the exact kernel, leaving nothing for a
 * 2.4x smaller payload to recover against rotate, pack and unpack. A prefill chunk is 40x larger
 * and bandwidth-bound. So the row is chosen at declare and the wire per launch. */
enum { R4D_AR_WHT6_MIN_BYTES = 128 * 1024 };

/* Block ceiling for the compressed path: it moves three eighths of the bytes, so it needs more
 * blocks to saturate the same link. */
enum { R4D_AR_WHT6_NB_CAP = 48 };

/* The rotated wire's fused writes run 512 threads per block, not 1024. Their waves are
 * latency-bound on the gather's routed rows and the write's streams; at a 2048-row chunk, twice as
 * many resident blocks of half the size is 5-9% faster per call, while 96 or 192 blocks of 1024
 * do not help. */
enum { R4D_AR_WHT6_FUSED_THREADS = 512 };

extern "C" int r4d_rad_ar_oneshot_2rank_wht6(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 2) return RAD_E_SHAPE;
    if (c.in->dtype == 2) return RAD_E_DTYPE;      /* the rotated wire is bf16/fp16 only */
    if (!c.in->locpack) return RAD_E_STATE;

    int group = 0, bits = 0, chunk = 0;
    r4d_ar_wht6_dims(&group, &bits, &chunk);

    /* Size gate: below the floor, or when the rotation's group does not tile the message, use the
     * exact kernel. Both share this instance's scratch, flags and sequence counters and one flag
     * value space, so mixing them on a stream is protocol-safe. A floor of 0 forces the rotated
     * wire. */
    const long long floor_b = r4d_pgeti_or(a->p, a->n_p, "min_bytes", R4D_AR_WHT6_MIN_BYTES);
    const long long msg_b   = (long long)c.numel * c.in->esize;
    if (group <= 0 || c.numel % group || msg_b < floor_b) {
        R4D_TRY("ar_oneshot_2rank_exact",
            r4d_ar_oneshot_2rank_exact(c.in->peer_scratch[0], c.in->my_scratch,
                                       c.in->peer_flags[0], c.in->my_flags, c.in->my_seq,
                                       c.in->slot16, c.inp, c.out, c.numel, c.in->dtype,
                                       (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                       c.in->my_stage, ar_chain(a, c.numel)));
        return RAD_OK;
    }

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    if (cfgv.nblocks <= 0) {
        int nb = nb_for((long)msg_b / 16, R4D_AR_WHT6_NB_CAP);
        if (nb > c.in->max_blocks) nb = c.in->max_blocks;
        c.nblocks = nb;
    }

    R4D_TRY("ar_oneshot_2rank_wht6",
        r4d_ar_oneshot_2rank_wht6(c.in->peer_scratch[0], c.in->my_scratch,
                                  c.in->peer_flags[0], c.in->my_flags, c.in->my_seq,
                                  c.in->locpack, c.in->slot_bytes, c.in->slot_bytes / 2,
                                  c.inp, c.out, c.numel, c.in->dtype,
                                  (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                  c.in->my_stage));
    return RAD_OK;
}

/* ------------------------------------------------------------------ wide (3 / 4 / 8 rank)
 *
 * Peer arguments are in ascending global rank with the current rank removed, the order build()
 * filled peer_scratch/peer_flags in. */

/* The one-shots are safe in place: the thread that stores output word k is the one that read
 * input word k (for its push and its own term of the fold), and peers read only their own scratch
 * (r4d_ar_oneshot_Nrank_exact.hip). */
extern "C" int r4d_rad_ar_oneshot_3rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 3) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    R4D_TRY("ar_oneshot_3rank_exact",
        r4d_ar_oneshot_3rank_exact(in->peer_scratch[0], in->peer_scratch[1],
                                   in->peer_flags[0], in->peer_flags[1],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_oneshot_4rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 4) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    /* From `ts_min` elements, the exact two-shot on this instance: the same fold per element, so
     * the same bytes, for half the bytes a rank (a 16-row verify's message, 516 KiB a card at
     * 3x, is what the link waits on). It shares the wide family's flags, counters and slots, and
     * a shard pair fits any slot that holds the whole message. */
    const long ts_min = r4d_cfg_ar_ts_min(a);
    const long words16 = c.numel * in->esize / 16;
    if (ts_min > 0 && c.numel >= ts_min && words16 % in->world == 0 &&
        2 * (c.numel * in->esize) / in->world <= in->slot_bytes &&
        c.numel <= R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) {
        R4dArCfg cfgv;
        r4d_cfg_ar(a, &cfgv);
        int nb = cfgv.nblocks > 0 ? cfgv.nblocks
                                  : nb_for_shard(words16 / in->world,
                                                 in->nb_cap > 24 ? 24 : in->nb_cap);
        if (nb > in->max_blocks) nb = in->max_blocks;
        const long gate = link_gate(a), early = in->host ? 0L : gate;
        R4D_TRY("ar_oneshot_4rank_exact (two-shot)",
            r4d_ar_twoshot_4rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                       in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                       in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                       c.inp, c.out, c.numel, in->dtype, in->rank,
                                       (long)(uintptr_t)s, nb, c.nthreads, c.drain, c.acq,
                                       c.pub, early, early ? in->gate_ctr : 0L,
                                       ar_chain(a, c.numel)));
        return early ? RAD_LINK_ENDED : RAD_OK;
    }
    /* The wide release, the two-rank family's rule: over P2P the kernel ends the engine's hold
     * itself once the peers' pushes have landed (the kernel after the handshake is local VRAM
     * traffic); through the host mailbox the reduce reads the bytes across the link, so early
     * stays 0 and the engine releases by barrier. */
    const long gate = link_gate(a), early = in->host ? 0L : gate;
    R4D_TRY("ar_oneshot_4rank_exact",
        r4d_ar_oneshot_4rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub, early, early ? in->gate_ctr : 0L));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

extern "C" int r4d_rad_ar_oneshot_8rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 0);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 8) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    R4D_TRY("ar_oneshot_8rank_exact",
        r4d_ar_oneshot_8rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_scratch[3], in->peer_scratch[4], in->peer_scratch[5],
                                   in->peer_scratch[6],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->peer_flags[3], in->peer_flags[4], in->peer_flags[5],
                                   in->peer_flags[6],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

/* The two-shots shard the message, so their block count scales with the shard.
 *
 * In place is safe (r4d_ar_twoshot_Nrank_exact.hip): a thread uses the same word offsets k in all
 * three phases (push reads word k of every peer's shard, reduce overwrites word k of its own,
 * assemble writes word k of every peer's shard), so each input word is overwritten only by the
 * thread that already read it, and peers read only their own scratch. The selector gives the
 * two-shot messages above the one-shot's size bound (r4d_rows.cpp, cArTsExact4). */
/* At three ranks the shards are uneven, so there is no divisibility check: the first
 * (words % 3) shards get one extra word and the slot holds two of the longest. The fold is the
 * one-shot's (fp32, ascending rank), so the band edge between the two rows changes no bits. */
extern "C" int r4d_rad_ar_twoshot_3rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 3) return RAD_E_SHAPE;
    if (c.numel > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return RAD_E_SHAPE;
    const long words16 = c.numel * c.in->esize / 16;
    if (words16 < c.in->world) return RAD_E_SHAPE;
    const long sh_max = (words16 + c.in->world - 1) / c.in->world;
    if (2 * sh_max * 16 > c.in->slot_bytes) return RAD_E_SCRATCH;
    const ArInst* in = c.in;
    R4D_TRY("ar_twoshot_3rank_exact",
        r4d_ar_twoshot_3rank_exact(in->peer_scratch[0], in->peer_scratch[1],
                                   in->peer_flags[0], in->peer_flags[1],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_twoshot_4rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 4) return RAD_E_SHAPE;
    if (c.numel > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return RAD_E_SHAPE;
    /* The message must shard evenly in whole 16-byte words and 2*shard must fit the slot. */
    const long words16 = c.numel * c.in->esize / 16;
    if (words16 % c.in->world) return RAD_E_SHAPE;
    if (2 * (c.numel * c.in->esize) / c.in->world > c.in->slot_bytes) return RAD_E_SCRATCH;
    const ArInst* in = c.in;
    /* A reduction beside compute (overlap 1) carries its pushes coded at ~12.2 bits a bf16 value
     * (r4d_ar_twoshot_Nrank_exact.hip, the coded form), at 4 blocks of 1024 threads, wherever the
     * message fits that form; the same bytes either way. On 4x R9700 4 x 1024 gave the 31B's
     * prefill the most (+3.9%) against 8 x 512, 16 x 256 and 6 x 1024, whose blocks crowd the GEMM
     * beside them. `coded` 0, a launch parameter no declaration carries, keeps the plain wire:
     * r4d_selftest's plain arm. */
    const bool lane = r4d_geti_or(a, "overlap", 0) == 1;
    if (lane && r4d_geti_or(a, "coded", 1) != 0 &&
        r4d_ar_twoshot_coded_ok(c.numel, in->dtype, 4, in->slot16)) {
        const long gate = link_gate(a), early = in->host ? 0L : gate;
        R4D_TRY("ar_twoshot_4rank_exact (coded)",
            r4d_ar_twoshot_4rank_exact_coded(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                             in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                             in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                             c.inp, c.out, c.numel, in->rank, (long)(uintptr_t)s,
                                             4, 1024, c.drain, c.acq, c.pub, early,
                                             early ? in->gate_ctr : 0L, ar_chain(a, c.numel)));
        return early ? RAD_LINK_ENDED : RAD_OK;
    }
    /* The kernel ends the hold after its second handshake (its reduce is fused with the phase-2
     * push); through the host mailbox early stays 0 and the engine releases by barrier. */
    const long gate = link_gate(a), early = in->host ? 0L : gate;
    R4D_TRY("ar_twoshot_4rank_exact",
        r4d_ar_twoshot_4rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub, early, early ? in->gate_ctr : 0L,
                                   ar_chain(a, c.numel)));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

extern "C" int r4d_rad_ar_twoshot_8rank_exact(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 8) return RAD_E_SHAPE;
    if (c.numel > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return RAD_E_SHAPE;
    const ArInst* in = c.in;
    const int* weights = ar8_weights_for(in, c.numel);
    /* Even shards keep the four-rank rule (whole words a rank); weighted ones are uneven anyway. */
    const long words16 = c.numel * in->esize / 16;
    if (!weights && words16 % in->world) return RAD_E_SHAPE;
    long off[R4D_MAX_RANKS + 1];
    const long sh_max = r4d_ar_shard_table(words16, in->world, weights, off);
    if (2 * sh_max * 16 > in->slot_bytes) return RAD_E_SCRATCH;
    /* Pieces only move the handshakes; the regions stay at 0 and sh_max, so scratch is as above. */
    const long long asked = r4d_pgeti_or(a->p, a->n_p, "pieces", in->pieces);
    if (asked < 0 || asked > R4D_AR_TS_MAX_PIECES) return RAD_E_INVAL;
    const long long pieces = ar8_pieces_for(in, c.numel, asked);
    /* Both for measuring a board (r4d_selftest --perfar8), never set by the engine: `timeline` a
     * device buffer of nblocks * R4D_AR_TL_WORDS int64 stamps (r4d.h), `form` a schedule other than
     * lockstep (r4d_ar_shards.h). Lockstep is the serving form: on eight R9700 the ring form takes
     * 2-6% less time for 8 and 16 MiB messages, but at TP8 the split form, which includes the ring
     * order, moves neither prefill nor decode beyond run-to-run noise. */
    const long long timeline = r4d_pgeti_or(a->p, a->n_p, "timeline", 0);
    const long long form = r4d_pgeti_or(a->p, a->n_p, "form", R4D_AR8_FORM_LOCKSTEP);
    if (form < R4D_AR8_FORM_LOCKSTEP || form > R4D_AR8_FORM_SPLIT) return RAD_E_INVAL;
    R4D_TRY("ar_twoshot_8rank_exact",
        r4d_ar_twoshot_8rank_exact(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                   in->peer_scratch[3], in->peer_scratch[4], in->peer_scratch[5],
                                   in->peer_scratch[6],
                                   in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                   in->peer_flags[3], in->peer_flags[4], in->peer_flags[5],
                                   in->peer_flags[6],
                                   in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                                   c.inp, c.out, c.numel, in->dtype, in->rank,
                                   (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                   c.pub, weights, (long)pieces, (long)timeline, (long)form));
    return RAD_OK;
}

/* The tiered-int8 wire. Its slot stride is in bytes, not 16-byte words: the wire is 38 bytes per
 * group of 32.
 *
 * It is the only lossy row at three and four ranks, so a lossy request resolves it at every size.
 * As in r4d_rad_ar_oneshot_2rank_wht6, a message under `min_bytes` (or not whole groups) goes to
 * the exact one-shot. The default floor is the two-rank wire's; blocks declare `min_bytes` from
 * --tp-wire-min-kb, and no four-rank crossover has been measured. In place is safe for the exact
 * two-shot's reason: a thread owns the same 8 elements of a shard in all phases and reads them
 * before writing them, and no peer reads this rank's input. */
/* Three ranks run over the uneven two-shot (r4d_ar_twoshot_Nrank_ti8.hip), so the group count
 * need not divide by three. */
extern "C" int r4d_rad_ar_twoshot_3rank_ti8(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 3) return RAD_E_SHAPE;
    if (c.in->dtype == 2) return RAD_E_DTYPE;

    int group = 0, group_bytes = 0;
    r4d_ar_ti8_dims(&group, &group_bytes);
    if (group <= 0) return RAD_E_DEVICE;
    const long long floor_b = r4d_pgeti_or(a->p, a->n_p, "min_bytes", R4D_AR_WHT6_MIN_BYTES);
    const long n_groups = c.numel / group;
    if ((long long)c.numel * c.in->esize < floor_b || c.numel % group || n_groups < c.in->world)
        return r4d_rad_ar_oneshot_3rank_exact(a, s);
    /* The slot holds two wire images of the longest shard. */
    const long sg = (n_groups + c.in->world - 1) / c.in->world;
    const long wire = (sg * group_bytes + 15) / 16 * 16;
    if (2 * wire > c.in->slot_bytes) return RAD_E_SCRATCH;
    /* nthreads must be a multiple of the encoder's group width; the entry point throws otherwise. */
    if (c.nthreads % 32) return RAD_E_INVAL;

    const ArInst* in = c.in;
    R4D_TRY("ar_twoshot_3rank_ti8",
        r4d_ar_twoshot_3rank_ti8(in->peer_scratch[0], in->peer_scratch[1],
                                 in->peer_flags[0], in->peer_flags[1],
                                 in->my_scratch, in->my_flags, in->my_seq, in->slot_bytes,
                                 c.inp, c.out, c.numel, in->dtype, in->rank,
                                 (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                 c.pub));
    return RAD_OK;
}

extern "C" int r4d_rad_ar_twoshot_4rank_ti8(const RadArgs* a, RadStream s) {
    ArCall c;
    int rc = ar_setup(a, &c, 1);
    if (rc != RAD_OK) return rc;
    if (c.in->world != 4) return RAD_E_SHAPE;
    if (c.in->dtype == 2) return RAD_E_DTYPE;

    int group = 0, group_bytes = 0;
    r4d_ar_ti8_dims(&group, &group_bytes);
    if (group <= 0) return RAD_E_DEVICE;
    const long long floor_b = r4d_pgeti_or(a->p, a->n_p, "min_bytes", R4D_AR_WHT6_MIN_BYTES);
    if ((long long)c.numel * c.in->esize < floor_b || c.numel % ((long)group * c.in->world))
        return r4d_rad_ar_oneshot_4rank_exact(a, s);
    /* The slot holds two wire images of one shard, each 16-byte aligned. */
    const long wire = ((c.numel / c.in->world / group) * group_bytes + 15) / 16 * 16;
    if (2 * wire > c.in->slot_bytes) return RAD_E_SCRATCH;
    if (c.nthreads % 32) return RAD_E_INVAL;

    const ArInst* in = c.in;
    /* Released after the second handshake, as in the exact two-shot. */
    const long gate = link_gate(a), early = in->host ? 0L : gate;
    R4D_TRY("ar_twoshot_4rank_ti8",
        r4d_ar_twoshot_4rank_ti8(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                                 in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                                 in->my_scratch, in->my_flags, in->my_seq, in->slot_bytes,
                                 c.inp, c.out, c.numel, in->dtype, in->rank,
                                 (long)(uintptr_t)s, c.nblocks, c.nthreads, c.drain, c.acq,
                                 c.pub, early, early ? in->gate_ctr : 0L));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

/* ------------------------------------------------------------------ the fused all-reduce + norm */

enum { AL_X = 0, AL_RESID, AL_W, AL_Q, AL_SCALE, AL_OUTBF16, AL_WPOST };

extern "C" int r4d_rad_ar_ln_had_quant_i8(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;

    const RadTensor* x  = r4d_opt(a, AL_X);
    const RadTensor* rs = r4d_opt(a, AL_RESID);
    const RadTensor* w  = r4d_opt(a, AL_W);
    const RadTensor* q  = r4d_opt(a, AL_Q);
    const RadTensor* sc = r4d_opt(a, AL_SCALE);
    const RadTensor* ob = r4d_opt(a, AL_OUTBF16);
    if (!x || !w || !q || !sc) return RAD_E_INVAL;
    if (x->rank < 2 || q->rank < 2) return RAD_E_SHAPE;
    /* A workgroup owns whole rows and walks them with vector loads, so nothing may be pitched. */
    if (!r4d_contig(x) || !r4d_contig(q) || !r4d_contig(w) || !r4d_contig(sc))
        return RAD_E_STRIDE;
    if (rs && !r4d_contig(rs)) return RAD_E_STRIDE;
    if (ob && !r4d_contig(ob)) return RAD_E_STRIDE;

    const long long K = x->shape[x->rank - 1];
    long long M = 1;
    for (uint32_t i = 0; i + 1 < x->rank; ++i) M *= x->shape[i];
    if (M < 1 || K < 1) return RAD_E_SHAPE;
    if (M * K > in->max_numel) return RAD_E_SHAPE;

    const long long had  = r4d_geti_or(a, "group", 0);
    const int       wire = (int)r4d_geti_or(a, "wire", 0);
    const float     eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float     wadd = (float)r4d_getf(a, "wadd", 0.0);

    /* Everything the entry point throws about, checked here. */
    if (K % 16) return RAD_E_SHAPE;
    if (had < 2 || (had & (had - 1))) return RAD_E_INVAL;
    if (had != 128 && had != 256 && had != 512) return RAD_E_SHAPE;
    if (K % had) return RAD_E_SHAPE;
    if (wire != 0 && wire != 6) return RAD_E_INVAL;
    if (wire == 6) {
        int stage_bits = 0, chunk_elems = 0, chunk_bytes = 0, scale_bytes = 0;
        r4d_ar_ln_wire_dims(&stage_bits, &chunk_elems, &chunk_bytes, &scale_bytes);
        if (chunk_elems <= 0 || K % chunk_elems) return RAD_E_SHAPE;
    }

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    int nthreads = cfgv.nthreads;
    if (nthreads != 128 && nthreads != 256 && nthreads != 512 && nthreads != 1024)
        return RAD_E_INVAL;
    /* The entry point clamps nblocks to M (a block owns whole rows); clamping here too keeps the
     * shim's count equal to the kernel's. */
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(M * K * 2 / 16, in->max_blocks);
    if (nblocks > (int)M) nblocks = (int)M;
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;   // a tuned count under R4D_AR_MAX_BLOCKS too
    /* LDS floor on the block count. The compressed wire stages every row a block owns, so a count
     * chosen for the push alone can exceed LDS (at K = 5120, more than two rows per block). Raise
     * the count to the kernel's floor; refuse shapes no count within the flag capacity serves.
     * Block count changes no bytes, since every reduction closes over one row. */
    const int nb_min = r4d_ar_ln_blocks_min((int)M, (int)K, (int)had, nthreads, wire);
    if (nb_min < 1 || nb_min > in->max_blocks) return RAD_E_SHAPE;
    if (nblocks < nb_min) nblocks = nb_min;

    R4D_TRY("ar_ln_had_quant_i8",
        r4d_ar_ln_had_quant_i8(in->peer_scratch[0], in->my_scratch, in->peer_flags[0],
                               in->my_flags, in->my_seq, in->slot16,
                               r4d_addr(x), r4d_addr(rs), r4d_addr(w), r4d_addr(ob),
                               r4d_addr(q), r4d_addr(sc),
                               (int)M, (int)K, (int)had, eps, wadd,
                               (long)(uintptr_t)s, nblocks, nthreads, cfgv.drain, cfgv.acq,
                               wire, in->my_stage));
    return RAD_OK;
}

/* The fp8 counterpart: the int8 shim without the Hadamard rotation.
 *
 * `x` is inout here, unlike the int8 form (which keeps the reduced row in LDS): the reduced row is
 * stored back as the standalone all-reduce does, so other readers of `x` see the unfused pair's
 * bytes. This holds on both wires.
 *
 * `wire` is passed through: 0 for exact bf16, 6 for the rotated 6-bit payload. The architecture
 * chooses it as the standalone `all_reduce` row would (lossy wire allowed, group tiles the message,
 * size above --tp-wire-min-kb), and both ranks decide on the same T. A wire-6 chunk that would
 * straddle a row is refused as a shape. */
/* `hi8`: ar_rmsnorm_had_quant_i8g, the same with had_quant_act_i8g's quantiser on the norm's pass
 * (int8 codes, K whole scale groups of 128 or IQ8R's 256, no post-norm). */
static int rad_ar_rmsnorm_quant(const RadArgs* a, RadStream s, bool hi8) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;

    const RadTensor* x  = r4d_opt(a, AL_X);
    const RadTensor* rs = r4d_opt(a, AL_RESID);
    const RadTensor* w  = r4d_opt(a, AL_W);
    const RadTensor* q  = r4d_opt(a, AL_Q);
    const RadTensor* sc = r4d_opt(a, AL_SCALE);
    const RadTensor* ob = r4d_opt(a, AL_OUTBF16);
    /* ar_rmsnorm_quant_fp8's optional post-norm (Gemma 3's sandwich); ar_ln_had_quant_i8 has no
     * seventh operand, and this function serves only the fp8 op and its rotated int8 twin. */
    const RadTensor* wpo = r4d_opt(a, AL_WPOST);
    if (!x || !w || !q || !sc) return RAD_E_INVAL;
    if (hi8 && q->dtype != RAD_I8) return RAD_E_DTYPE;
    if (hi8 && wpo) return RAD_E_UNSUPPORTED;
    if (!r4d_contig(w)) return RAD_E_STRIDE;
    if (wpo && (!rs || !r4d_contig(wpo) || r4d_numel(wpo) != r4d_numel(w) || wpo->dtype != w->dtype))
        return RAD_E_SHAPE;

    /* Rank 2 everywhere, as in rmsnorm_quant_fp8: a norm has a gain per channel, so flattening
     * everything after dim 0 into the row would put one gain across several heads. */
    if (x->rank != 2 || q->rank != 2 || sc->rank != 2) return RAD_E_SHAPE;
    if (ob && ob->rank != 2) return RAD_E_SHAPE;
    const long long M = x->shape[0], K = x->shape[1];
    if (M < 1 || K < 1) return RAD_E_SHAPE;
    if (M * K > in->max_numel) return RAD_E_SHAPE;
    /* A row must be whole 16B words, since a block pushes its rows as words. */
    if (K % 8) return RAD_E_SHAPE;

    const long long g0 = r4d_quant_act_fp8_group();
    const long long grp = r4d_geti_or(a, "group", g0);
    if (grp != g0 && !(hi8 && grp == R4D_I8_GROUP_WIDE)) return RAD_E_SHAPE;
    const long long nkb = (K + grp - 1) / grp;
    if (hi8 && K % grp) return RAD_E_SHAPE;
    if (r4d_numel(w) != K) return RAD_E_SHAPE;
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (q->shape[0] != M || q->shape[1] != K) return RAD_E_SHAPE;
    if (sc->shape[0] != M || sc->shape[1] != nkb) return RAD_E_SHAPE;
    if (ob && (ob->shape[0] != M || ob->shape[1] != K)) return RAD_E_SHAPE;
    if (rs && (rs->rank != 2 || rs->shape[0] != M || rs->shape[1] != K)) return RAD_E_SHAPE;
    /* x is the message, so it must be tight; other operands may be pitched. */
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = ob ? r4d_row_pitch(ob) : 0, rp = rs ? r4d_row_pitch(rs) : 0;
    if (xp < 0 || qp < 0 || sp < 0 || yp < 0 || rp < 0) return RAD_E_STRIDE;
    if (xp != K) return RAD_E_STRIDE;

    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    /* Gain dtype comes from the operand, as in rmsnorm_quant_fp8's shim. */
    const int w_f32 = w->dtype == RAD_F32;

    const int wire = (int)r4d_geti_or(a, "wire", 0);
    if (wire != 0 && wire != 6) return RAD_E_INVAL;
    if (wire == 6) {
        const int ce = r4d_ar_rmsnorm_quant_fp8_chunk_elems();
        /* Whole 64-element groups a row; the last 512-element chunk may be partial (padded). */
        if (ce <= 0 || K % 64) return RAD_E_SHAPE;
        /* This rank's packed-half side buffer exists for every bf16 instance; missing means a
         * broken init. */
        if (!in->locpack) return RAD_E_STATE;
        /* The padded message must fit: payload (3/4 byte an element) in the peer slot's half and
         * in locpack's, the bf16 scales in the slot's other half, the f32 ones in locpack after
         * half the slot (its allocation: a sixteenth of the slot plus a page). */
        const long long chunks = M * ((K + ce - 1) / ce), groups = chunks * (ce / 64);
        if (chunks * (ce * 6 / 8) > in->slot_bytes / 2 || groups * 2 > in->slot_bytes / 2 ||
            groups * 4 > in->slot_bytes / 16 + 4096)
            return RAD_E_SHAPE;
    }

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    /* The width is the standalone rmsnorm_quant_fp8's, r4d_act_threads(M, n) (1024 up to 32 rows,
     * 512 above): zero means that, resolved from M by the entry point. The sums fold in
     * kR4dSumLanes' partition at every width, so the width changes no bytes; it is not tuned here.
     * The tuned block count, drain and acq change no bytes either and are honoured. */
    const int nthreads = 0;
    /* One block per row, the norm's shape rather than the all-reduce's. The push saturates the
     * fabric at four blocks, but the norm serialises the rows a block owns, which costs more than
     * the saved dispatch; M is 8 to 64 at decode, a block count the fabric handles. nb_cap (24, the
     * push's number) is the wrong ceiling: at M = 64 it would serialise three rows per block.
     *
     * Cap at 256. The wire alone prefers ~64 blocks, but a 2048-row chunk at 64 gives each block 32
     * serial rows, costing several times what the wire gains; 256 is the combined knee and 512 is
     * within noise. At decode, nblocks is simply M.
     *
     * Block count changes no bytes: a block owns whole rows, the wire's 512-element chunk tiles a
     * row, and every reduction closes over one row. A tuned `nblocks` may name a smaller count; the
     * entry point refuses one whose LDS does not fit. The caps bound the flag array. */
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : (int)M;
    if (nblocks > (int)M) nblocks = (int)M;
    constexpr int kNbCap = 256;
    if (nblocks > kNbCap) nblocks = kNbCap;
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    if (nblocks < 1) nblocks = 1;

    const int e = hi8
        ? r4d_ar_rmsnorm_had_quant_i8g(
              in->peer_scratch[0], in->my_scratch,
              in->peer_flags[0], in->my_flags, in->my_seq,
              in->slot16, r4d_addr(x), r4d_addr(rs), r4d_addr(w), r4d_addr(ob), r4d_addr(q),
              r4d_addr(sc), (int)M, (int)K, eps, wadd, w_f32,
              (int)rp, (int)yp, (int)qp, (int)sp,
              (long)(uintptr_t)s, nblocks, nthreads, cfgv.drain, cfgv.acq, wire,
              in->locpack, in->slot_bytes / 2, in->my_stage, (int)grp,
              r4d_geti_or(a, "chain", 0) != 0 ? 1 : 0)
        : r4d_ar_rmsnorm_quant_fp8(
              in->peer_scratch[0], in->my_scratch,
              in->peer_flags[0], in->my_flags, in->my_seq,
              in->slot16, r4d_addr(x), r4d_addr(rs), r4d_addr(w), r4d_addr(ob), r4d_addr(q),
              r4d_addr(sc), (int)M, (int)K, eps, wadd, w_f32,
              (int)rp, (int)yp, (int)qp, (int)sp,
              (long)(uintptr_t)s, nblocks, nthreads, cfgv.drain, cfgv.acq, wire,
              in->locpack, in->slot_bytes / 2, wpo ? r4d_addr(wpo) : 0, in->my_stage);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_ar_rmsnorm_quant_fp8(const RadArgs* a, RadStream s) {
    return rad_ar_rmsnorm_quant(a, s, false);
}

extern "C" int r4d_rad_ar_rmsnorm_had_quant_i8g(const RadArgs* a, RadStream s) {
    return rad_ar_rmsnorm_quant(a, s, true);
}

/* The all-reduce and the sandwich-norm junction after it, int8 out: exact `all_reduce` of `x`, then
 * `rmsnorm_quant_i8` (post-norm of the reduced x, residual add, scalar, norm and the per-row int8
 * quantiser the w4a8 skinny GEMM reads). Operands are rmsnorm_quant_i8's with `x` inout, since the
 * reduced sum lands back in it. Exact wire only; a lossy deployment keeps the standalone
 * all-reduce. Width and block count as in ar_rmsnorm_quant_fp8. */
enum { AQ8_X = 0, AQ8_RES, AQ8_W, AQ8_Q, AQ8_S, AQ8_Y, AQ8_WP, AQ8_SC };

/* The operands and their rules, the same at every width. `grp`: quant_act_i8g's grid
 * (ar_rmsnorm_quant_i8g), a scale per (row, 128 columns) and K in whole groups. */
struct ArQ8Args {
    const RadTensor *x = nullptr, *rs = nullptr, *w = nullptr, *q = nullptr, *sc = nullptr;
    const RadTensor *ob = nullptr, *wp = nullptr, *sl = nullptr;
    long long M = 0, K = 0, yp = 0, rp = 0;
};

static int ar_q8_args(const RadArgs* a, const ArInst* in, ArQ8Args* o, bool grp = false) {
    const RadTensor* x  = o->x  = r4d_opt(a, AQ8_X);
    const RadTensor* rs = o->rs = r4d_opt(a, AQ8_RES);
    const RadTensor* w  = o->w  = r4d_opt(a, AQ8_W);
    const RadTensor* q  = o->q  = r4d_opt(a, AQ8_Q);
    const RadTensor* sc = o->sc = r4d_opt(a, AQ8_S);
    const RadTensor* ob = o->ob = r4d_opt(a, AQ8_Y);
    const RadTensor* wp = o->wp = r4d_opt(a, AQ8_WP);
    const RadTensor* sl = o->sl = r4d_opt(a, AQ8_SC);
    if (!x || !w || !q || !sc) return RAD_E_INVAL;
    if ((wp || sl) && !rs) return RAD_E_INVAL;
    if (!r4d_contig(w) || (wp && !r4d_contig(wp))) return RAD_E_STRIDE;
    if (x->dtype != RAD_BF16 || q->dtype != RAD_I8 || sc->dtype != RAD_F32) return RAD_E_DTYPE;
    long long M = 0, K = 0;
    if (r4d_rows_of(x, &M, &K) != RAD_OK) return RAD_E_SHAPE;
    if (M * K > in->max_numel) return RAD_E_SHAPE;
    if (K % 16 || (grp && K % 128)) return RAD_E_SHAPE;
    if (r4d_numel(w) != K || (w->dtype != RAD_F32 && w->dtype != RAD_BF16)) return RAD_E_DTYPE;
    if (wp && (r4d_numel(wp) != K || wp->dtype != w->dtype)) return RAD_E_SHAPE;
    if (sl && (r4d_numel(sl) != 1 || (sl->dtype != RAD_F32 && sl->dtype != RAD_BF16))) return RAD_E_SHAPE;
    if (r4d_numel(q) != M * K || !r4d_contig(q)) return RAD_E_SHAPE;
    if (r4d_numel(sc) != M * (grp ? K / 128 : 1) || !r4d_contig(sc)) return RAD_E_SHAPE;
    if (ob && r4d_numel(ob) != M * K) return RAD_E_SHAPE;
    if (rs && r4d_numel(rs) != M * K) return RAD_E_SHAPE;
    const long long xp = r4d_row_pitch(x), yp = ob ? r4d_row_pitch(ob) : 0, rp = rs ? r4d_row_pitch(rs) : 0;
    if (xp != K || yp < 0 || rp < 0) return RAD_E_STRIDE;   /* x is the message: tight */
    if (r4d_geti_or(a, "wire", 0) != 0) return RAD_E_INVAL;
    o->M = M; o->K = K; o->yp = yp; o->rp = rp;
    return RAD_OK;
}

static int rad_ar_rmsnorm_quant_i8(const RadArgs* a, RadStream s, bool grp) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;
    ArQ8Args v;
    const int arc = ar_q8_args(a, in, &v, grp);
    if (arc != RAD_OK) return arc;
    const long long M = v.M, K = v.K;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : (int)M;
    if (nblocks > (int)M) nblocks = (int)M;
    if (nblocks > 256) nblocks = 256;
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    if (nblocks < 1) nblocks = 1;
    const int e = r4d_ar_rmsnorm_quant_i8(
        in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
        in->slot16, r4d_addr(v.x), r4d_addr(v.rs), r4d_addr(v.w), r4d_addr(v.wp), r4d_addr(v.sl),
        v.sl ? (v.sl->dtype == RAD_F32) : 0, r4d_addr(v.ob), r4d_addr(v.q), r4d_addr(v.sc), (int)M,
        (int)K, (float)r4d_getf(a, "eps", 1e-6), (float)r4d_getf(a, "wadd", 0.0),
        v.w->dtype == RAD_F32, (int)v.rp, (int)v.yp, (long)(uintptr_t)s, nblocks, cfgv.drain,
        cfgv.acq, grp ? 1 : 0, in->my_stage, r4d_geti_or(a, "chain", 0) != 0 ? 1 : 0);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_ar_rmsnorm_quant_i8(const RadArgs* a, RadStream s) {
    return rad_ar_rmsnorm_quant_i8(a, s, false);
}

/* The same at quant_act_i8g's grid (rmsnorm_quant_i8g's junction): two ranks only. A four-rank
 * deployment declares it, finds no row, and issues the all-reduce apart. */
extern "C" int r4d_rad_ar_rmsnorm_quant_i8g(const RadArgs* a, RadStream s) {
    return rad_ar_rmsnorm_quant_i8(a, s, true);
}

/* The same at four ranks, on the wide family's one-shot transport (r4d_rmsnorm_quant_i8.hip's
 * ar_rmsnorm_quant_i8_wide_kernel), the one ar_hc_write_4rank rides, with the junction as its
 * epilogue. The operands and the norm's width are the two-rank form's. The block count is one per
 * row, capped by the wide family's flag array in the launcher and by the instance here. The
 * instance is the four-rank family's {rank, 4, bf16, M_hi * n} (ar_hc_wide_init), so it is a
 * standalone all_reduce band's where the two bounds agree, which is safe on one stream for the
 * reason that init gives. The plugin issues it only at one-shot heights: each rank sends three
 * times the message, which the two-shot the standalone takes above them does not. */
extern "C" int r4d_rad_ar_rmsnorm_quant_i8_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 4) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;
    ArQ8Args v;
    const int arc = ar_q8_args(a, in, &v);
    if (arc != RAD_OK) return arc;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : (int)v.M;
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    /* Above one-shot heights (the plugin's fold band at 5 rows and up), the `tail` axis's form:
     * the standalone two-shot over words on the blocks the standalone all_reduce would take for
     * this message (nb_for_shard), with the junction as its tail; its two counters are words of the
     * instance's zeroed sequence allocation past the landed counter (r4d_ar_seq.h). P2P only. */
    int tail_nbt = 0;
    if (r4d_cfg_ar_tail(a) && v.M >= 5 && !in->host) {
        const long words16 = (long)v.M * v.K / 8;
        tail_nbt = nb_for_shard(words16 / in->world, in->nb_cap > 24 ? 24 : in->nb_cap);
        if (tail_nbt > in->max_blocks) tail_nbt = in->max_blocks;
    }
    const long tail_ctr = in->my_seq + (2L * in->max_blocks + 8) * 4;
    /* `sp` (0 unless asked): the all-reduce split around the junction, P2P only. The host mailbox
     * runs the whole form, which leaves every byte the split one does. */
    const long long sp = r4d_geti_or(a, "sp", 0);
    if (sp < 0 || sp > 2) return RAD_E_INVAL;
    /* The junction after the last handshake is local, so either form releases there. */
    const long gate = link_gate(a), early = in->host ? 0L : gate;
    const int e = r4d_ar_rmsnorm_quant_i8_4rank(
        in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
        in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
        in->my_scratch, in->my_flags, in->my_seq, in->slot16,
        r4d_addr(v.x), r4d_addr(v.rs), r4d_addr(v.w), r4d_addr(v.wp), r4d_addr(v.sl),
        v.sl ? (v.sl->dtype == RAD_F32) : 0, r4d_addr(v.ob), r4d_addr(v.q), r4d_addr(v.sc),
        (int)v.M, (int)v.K, (float)r4d_getf(a, "eps", 1e-6), (float)r4d_getf(a, "wadd", 0.0),
        v.w->dtype == RAD_F32, (int)v.rp, (int)v.yp, (int)in->rank, (long)(uintptr_t)s, nblocks,
        cfgv.drain, cfgv.acq, cfgv.pub, early, early ? in->gate_ctr : 0L, tail_ctr, tail_nbt,
        r4d_geti_or(a, "chain", 0) != 0 ? 1 : 0, in->host ? 0 : (int)sp);
    if (e != 0) return RAD_E_SHAPE;
    return early ? RAD_LINK_ENDED : RAD_OK;
}

/* The all-reduce with the gated residual write: `all_reduce` of `y`, then `hc_write` of the reduced
 * `y` into `h`, in one kernel. A block whose output feeds a hyper-connection write skips its own
 * collective when this runs.
 *
 * `y` is the message, pushed as 16B words, so it must be tight. The streams may be pitched; n and
 * the pitch being multiples of eight keep every write whole 16B words.
 *
 * `wire` (0 exact bf16, 6 rotated 6-bit) belongs to the op, not the message: the architecture
 * declares one op per wire and issues the one ar_wire_is_exact picks for the step, decided on both
 * ranks from the same T. A wire-6 op is 6 at every size, with no floor. */
enum { AH_Y = 0, AH_INJ, AH_H };

/* ---- ar_gather_hc_write: moe_gather, then ar_hc_write, in one kernel. Operands:
 *   [0] y_expert  [1] expert_w  [2] sorted_tok  [3] shared?  [4] shared_gate?  [5] y  [6] inj  [7] h
 * The first five are moe_gather's and the last three ar_hc_write's. `wire` is as in ar_hc_write;
 * wire 6 builds its slot table in LDS at every size and needs no scratch. */
enum { AG_YE = 0, AG_EW, AG_SORTED, AG_SH, AG_SG, AG_Y, AG_INJ, AG_H };

/* ar_gather_hc_write's operand and shape rules at every width: moe_gather's five operands,
 * ar_hc_write's three, and the inverse table's scratch. */
struct GatherHcWriteArgs {
    const RadTensor *ye = nullptr, *ew = nullptr, *so = nullptr, *sh = nullptr, *sg = nullptr;
    const RadTensor *y = nullptr, *inj = nullptr, *h = nullptr;
    long long M = 0, n = 0, hc = 0, top_k = 0, irow = 0, hrow = 0;
    long long ye_ld = 0, ew_ld = 0, sh_ld = 0, sg_ld = 0;
    int sig = 0, wire = 0;
    bool has_inv = false;
};

static int ar_gather_hc_write_args(const RadArgs* a, const ArInst* in, GatherHcWriteArgs* o) {
    const RadTensor* ye  = r4d_opt(a, AG_YE);
    const RadTensor* ew  = r4d_opt(a, AG_EW);
    const RadTensor* so  = r4d_opt(a, AG_SORTED);
    const RadTensor* sh  = r4d_opt(a, AG_SH);
    const RadTensor* sg  = r4d_opt(a, AG_SG);
    const RadTensor* y   = r4d_opt(a, AG_Y);
    const RadTensor* inj = r4d_opt(a, AG_INJ);
    const RadTensor* h   = r4d_opt(a, AG_H);
    if (!ye || !ew || !so || !y || !inj || !h) return RAD_E_INVAL;
    if ((sh != nullptr) != (sg != nullptr)) return RAD_E_INVAL;
    if (y->rank != 2 || inj->rank != 2 || h->rank != 2) return RAD_E_SHAPE;
    const long long M = y->shape[0], n = y->shape[1];
    const long long hc = r4d_geti_or(a, "hc", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    if (M < 1 || n < 1 || hc < 1 || hc > 8 || top_k < 1) return RAD_E_SHAPE;
    if (M * n > in->max_numel) return RAD_E_SHAPE;
    if (n % 8) return RAD_E_SHAPE;
    if (inj->shape[0] < M || inj->shape[1] != hc) return RAD_E_SHAPE;
    if (h->shape[0] < M || h->shape[1] != hc * n) return RAD_E_SHAPE;
    if (!r4d_contig(y)) return RAD_E_STRIDE;
    const long long irow = r4d_row_pitch(inj), hrow = r4d_row_pitch(h);
    if (irow < 0 || hrow < 0) return RAD_E_STRIDE;
    if (hrow % 8) return RAD_E_ALIGN;
    if (!r4d_aligned(y->data, 16) || !r4d_aligned(h->data, 16)) return RAD_E_ALIGN;

    /* moe_gather's side: its pitches and the shared-arm pair. */
    const long long T = M * top_k;
    long long Te = 0, Mw = 0;
    const long long ye_ld = r4d_rows_pitch(ye, n, &Te);
    const long long ew_ld = r4d_rows_pitch(ew, top_k, &Mw);
    if (ye_ld < 0 || ew_ld < 0 || Te < T || Mw < M) return RAD_E_STRIDE;
    if (so->dtype != RAD_I32 || !r4d_contig(so) || r4d_numel(so) < T) return RAD_E_SHAPE;
    if (ye->dtype != RAD_BF16 || ew->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (ye_ld % 8 || !r4d_aligned(ye->data, 16)) return RAD_E_ALIGN;
    long long sh_ld = 0, sg_ld = 0, shrows = 0, sgrows = 0;
    int sig = 0;
    if (sh) {
        if (sh->dtype != RAD_BF16 || sg->dtype != RAD_BF16) return RAD_E_DTYPE;
        sh_ld = r4d_rows_pitch(sh, n, &shrows);
        sg_ld = r4d_rows_pitch(sg, 1, &sgrows);
        if (sh_ld < 0 || sg_ld < 0 || shrows < M || sgrows < M) return RAD_E_STRIDE;
        const char* const act = r4d_gets(a, "act");
        if (act && !std::strcmp(act, "sigmoid"))  sig = 1;
        else if (act && std::strcmp(act, "none")) return RAD_E_UNSUPPORTED;
    }

    const int wire = (int)r4d_geti_or(a, "wire", 0);
    if (wire != 0 && wire != 6) return RAD_E_INVAL;

    o->ye = ye; o->ew = ew; o->so = so; o->sh = sh; o->sg = sg;
    o->y = y; o->inj = inj; o->h = h;
    o->M = M; o->n = n; o->hc = hc; o->top_k = top_k; o->irow = irow; o->hrow = hrow;
    o->ye_ld = ye_ld; o->ew_ld = ew_ld; o->sh_ld = sh_ld; o->sg_ld = sg_ld;
    o->sig = sig; o->wire = wire;
    o->has_inv = a->scratch && a->scratch_bytes >= T * (long long)sizeof(int);
    return RAD_OK;
}

extern "C" int r4d_rad_ar_gather_hc_write(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    GatherHcWriteArgs g;
    const int arc = ar_gather_hc_write_args(a, in, &g);
    if (arc != RAD_OK) return arc;
    const RadTensor* const ye = g.ye;
    const RadTensor* const ew = g.ew;
    const RadTensor* const so = g.so;
    const RadTensor* const sh = g.sh;
    const RadTensor* const sg = g.sg;
    const RadTensor* const y = g.y;
    const RadTensor* const inj = g.inj;
    const RadTensor* const h = g.h;
    const long long M = g.M, n = g.n, hc = g.hc, top_k = g.top_k, irow = g.irow, hrow = g.hrow;
    const long long ye_ld = g.ye_ld, ew_ld = g.ew_ld, sh_ld = g.sh_ld, sg_ld = g.sg_ld;
    const int sig = g.sig, wire = g.wire;
    const bool has_inv = g.has_inv;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long words16 = (long)(M * n * 2 / 16);
    if (wire == 6) {
        if (!in->locpack) return RAD_E_STATE;
        int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, R4D_AR_WHT6_NB_CAP);
        if (nblocks > in->max_blocks) nblocks = in->max_blocks;
        const long gate = link_gate(a), early = in->host ? 0L : gate;
        const int rc = r4d_ar_hc_write_wht6(
            in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
            in->locpack, in->slot_bytes, in->slot_bytes / 2, r4d_addr(y), r4d_addr(inj),
            r4d_addr(h), (long)M, (long)n, (long)hc, (long)irow, (long)hrow, r4d_addr(ye),
            r4d_addr(ew), r4d_addr(so), sh ? r4d_addr(sh) : 0L, sh ? r4d_addr(sg) : 0L,
            (long)ye_ld, (long)ew_ld, (long)sh_ld, (long)sg_ld, (long)top_k, (long)sig,
            (long)(uintptr_t)s, nblocks, R4D_AR_WHT6_FUSED_THREADS, cfgv.drain, cfgv.acq,
            early, early ? in->gate_ctr : 0L, in->my_stage);
        if (rc != 0) return RAD_E_SHAPE;
        return early ? RAD_LINK_ENDED : RAD_OK;
    }
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    const long gate = link_gate(a), early = in->host ? 0L : gate;
    const int rc = r4d_ar_gather_hc_write(
        in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
        in->slot16, r4d_addr(y), r4d_addr(inj), r4d_addr(h), (long)M, (long)n, (long)hc,
        (long)irow, (long)hrow, r4d_addr(ye), r4d_addr(ew), r4d_addr(so),
        sh ? r4d_addr(sh) : 0L, sh ? r4d_addr(sg) : 0L, (long)ye_ld, (long)ew_ld, (long)sh_ld,
        (long)sg_ld, (long)top_k, (long)sig, (long)(uintptr_t)s, nblocks, cfgv.nthreads,
        cfgv.drain, cfgv.acq, has_inv ? (long)(uintptr_t)a->scratch : 0L, early,
        early ? in->gate_ctr : 0L, in->my_stage, (long)r4d_geti_or(a, "chain", 0));
    if (rc != 0) return RAD_E_SHAPE;
    return early ? RAD_LINK_ENDED : RAD_OK;
}

/* Four ranks, exact wire only: r4d_rad_ar_hc_write_4rank with the gather in front, on an instance
 * from r4d_rad_ar_hc_wide_init. Issued only on steps whose write takes the exact fused form
 * (ar_exact_write_takes, arch/common/rad_fp8.h) and whose gather is a single pass. The block count
 * is the write's rule, widened by the launcher until a block's slot table fits in LDS, or, when
 * that needs more blocks and scratch is present, kept and served from the inverse table. */
extern "C" int r4d_rad_ar_gather_hc_write_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 4) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    GatherHcWriteArgs g;
    const int arc = ar_gather_hc_write_args(a, in, &g);
    if (arc != RAD_OK) return arc;
    if (g.wire != 0) return RAD_E_UNSUPPORTED;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long words16 = (long)(g.M * g.n * 2 / 16);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    const long gate = link_gate(a), early = in->host ? 0L : gate;

    R4D_TRY("ar_gather_hc_write_4rank",
        r4d_ar_gather_hc_write_4rank(
            in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
            in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
            in->my_scratch, in->my_flags, in->my_seq, in->slot16,
            r4d_addr(g.y), r4d_addr(g.inj), r4d_addr(g.h), (long)g.M, (long)g.n, (long)g.hc,
            (long)g.irow, (long)g.hrow, r4d_addr(g.ye), r4d_addr(g.ew), r4d_addr(g.so),
            g.sh ? r4d_addr(g.sh) : 0L, g.sh ? r4d_addr(g.sg) : 0L, (long)g.ye_ld,
            (long)g.ew_ld, (long)g.sh_ld, (long)g.sg_ld, (long)g.top_k, (long)g.sig,
            (long)in->rank, (long)(uintptr_t)s, nblocks, cfgv.nthreads, cfgv.drain, cfgv.acq,
            cfgv.pub, g.has_inv ? (long)(uintptr_t)a->scratch : 0L, early,
            early ? in->gate_ctr : 0L));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

/* Inverse slot table for the exact wire's prefill form: one int per (token, slot) at the band's M.
 * The rotated wire keeps its table in LDS. */
extern "C" int64_t r4d_rad_ar_gather_hc_write_scratch(const RadArgs* a) {
    if (r4d_geti_or(a, "wire", 0) == 6) return 0;
    const long long M = r4d_geti_or(a, "M", 0), k = r4d_geti_or(a, "top_k", 0);
    return M > 0 && k > 0 ? M * k * (int64_t)sizeof(int) : 0;
}

/* ar_hc_write's operand and shape rules at every width. */
struct HcWriteArgs {
    const RadTensor* y = nullptr;
    const RadTensor* inj = nullptr;
    const RadTensor* h = nullptr;
    long long M = 0, n = 0, hc = 0, irow = 0, hrow = 0;
    int wire = 0;
};

static int ar_hc_write_args(const RadArgs* a, const ArInst* in, HcWriteArgs* o) {
    const RadTensor* y   = r4d_opt(a, AH_Y);
    const RadTensor* inj = r4d_opt(a, AH_INJ);
    const RadTensor* h   = r4d_opt(a, AH_H);
    if (!y || !inj || !h) return RAD_E_INVAL;
    if (y->rank != 2 || inj->rank != 2 || h->rank != 2) return RAD_E_SHAPE;
    const long long M = y->shape[0], n = y->shape[1];
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (M < 1 || n < 1 || hc < 1 || hc > 8) return RAD_E_SHAPE;
    if (M * n > in->max_numel) return RAD_E_SHAPE;
    if (n % 8) return RAD_E_SHAPE;
    if (inj->shape[0] < M || inj->shape[1] != hc) return RAD_E_SHAPE;
    if (h->shape[0] < M || h->shape[1] != hc * n) return RAD_E_SHAPE;
    if (!r4d_contig(y)) return RAD_E_STRIDE;
    const long long irow = r4d_row_pitch(inj), hrow = r4d_row_pitch(h);
    if (irow < 0 || hrow < 0) return RAD_E_STRIDE;
    if (hrow % 8) return RAD_E_ALIGN;
    if (!r4d_aligned(y->data, 16) || !r4d_aligned(h->data, 16)) return RAD_E_ALIGN;

    const int wire = (int)r4d_geti_or(a, "wire", 0);
    if (wire != 0 && wire != 6) return RAD_E_INVAL;

    o->y = y; o->inj = inj; o->h = h;
    o->M = M; o->n = n; o->hc = hc; o->irow = irow; o->hrow = hrow;
    o->wire = wire;
    return RAD_OK;
}

extern "C" int r4d_rad_ar_hc_write(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 2) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    HcWriteArgs w;
    const int arc = ar_hc_write_args(a, in, &w);
    if (arc != RAD_OK) return arc;
    const RadTensor* const y = w.y;
    const RadTensor* const inj = w.inj;
    const RadTensor* const h = w.h;
    const long long M = w.M, n = w.n, hc = w.hc, irow = w.irow, hrow = w.hrow;
    const int wire = w.wire;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long words16 = (long)(M * n * 2 / 16);
    if (wire == 6) {
        if (!in->locpack) return RAD_E_STATE;
        int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, R4D_AR_WHT6_NB_CAP);
        if (nblocks > in->max_blocks) nblocks = in->max_blocks;
        const long gate = link_gate(a), early = in->host ? 0L : gate;
        const int rc = r4d_ar_hc_write_wht6(
            in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags, in->my_seq,
            in->locpack, in->slot_bytes, in->slot_bytes / 2, r4d_addr(y), r4d_addr(inj),
            r4d_addr(h), (long)M, (long)n, (long)hc, (long)irow, (long)hrow, 0L, 0L, 0L, 0L, 0L,
            0L, 0L, 0L, 0L, 0L, 0L, (long)(uintptr_t)s, nblocks, R4D_AR_WHT6_FUSED_THREADS,
            cfgv.drain, cfgv.acq, early, early ? in->gate_ctr : 0L, in->my_stage);
        if (rc != 0) return RAD_E_SHAPE;
        return early ? RAD_LINK_ENDED : RAD_OK;
    }
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;

    const long gate = link_gate(a), early = in->host ? 0L : gate;
    R4D_TRY("ar_hc_write",
        r4d_ar_hc_write(in->peer_scratch[0], in->my_scratch, in->peer_flags[0], in->my_flags,
                        in->my_seq, in->slot16, r4d_addr(y), r4d_addr(inj), r4d_addr(h),
                        (long)M, (long)n, (long)hc, (long)irow, (long)hrow, (long)(uintptr_t)s,
                        nblocks, cfgv.nthreads, cfgv.drain, cfgv.acq, early,
                        early ? in->gate_ctr : 0L, in->my_stage,
                        (long)r4d_geti_or(a, "chain", 0)));
    return early ? RAD_LINK_ENDED : RAD_OK;
}

/* Four ranks, exact wire only: the wide one-shot with the write as its epilogue. The four-rank
 * lossy wire (the ti8 two-shot) has no fused form, so `wire` 6 is refused. Any message up to the
 * instance's bound is correct here as a one-shot; the architecture issues it on decode and verify
 * steps (ar_wide_write_exact, arch/common/rad_fp8.h), since a message the standalone bands would
 * send as a two-shot costs more traffic here.
 *
 * Block count follows the plain one-shot's rule (nb_for over the whole message under nb_cap);
 * words, flags and slot match the one-shot's, so the two can share an instance. */
extern "C" int r4d_rad_ar_hc_write_4rank(const RadArgs* a, RadStream s) {
    const ArInst* in = (const ArInst*)(a ? a->instance : nullptr);
    if (!in) return RAD_E_STATE;
    if (in->world != 4) return RAD_E_SHAPE;
    if (in->dtype != 0) return RAD_E_DTYPE;

    HcWriteArgs w;
    const int arc = ar_hc_write_args(a, in, &w);
    if (arc != RAD_OK) return arc;
    if (w.wire != 0) return RAD_E_UNSUPPORTED;

    R4dArCfg cfgv;
    r4d_cfg_ar(a, &cfgv);
    const long words16 = (long)(w.M * w.n * 2 / 16);
    int nblocks = cfgv.nblocks > 0 ? cfgv.nblocks : nb_for(words16, in->nb_cap);
    if (nblocks > in->max_blocks) nblocks = in->max_blocks;
    /* The two-rank fused write's release, at width: the reduce and the write behind the
     * handshake are local VRAM traffic, so the kernel ends the hold once the pushes land. */
    const long gate = link_gate(a), early = in->host ? 0L : gate;

    R4D_TRY("ar_hc_write_4rank",
        r4d_ar_hc_write_4rank(in->peer_scratch[0], in->peer_scratch[1], in->peer_scratch[2],
                              in->peer_flags[0], in->peer_flags[1], in->peer_flags[2],
                              in->my_scratch, in->my_flags, in->my_seq, in->slot16,
                              r4d_addr(w.y), r4d_addr(w.inj), r4d_addr(w.h),
                              (long)w.M, (long)w.n, (long)w.hc, (long)w.irow, (long)w.hrow,
                              (long)in->rank, (long)(uintptr_t)s, nblocks, cfgv.nthreads,
                              cfgv.drain, cfgv.acq, cfgv.pub, early,
                              early ? in->gate_ctr : 0L));
    return early ? RAD_LINK_ENDED : RAD_OK;
}
