/* device.h -- the device layer's internal interface.
 *
 * `abi/rad_device.h` is what the engine and plugins call. This header is the seam between its
 * backend implementations, plus queries for which backend is active.
 *
 * The seam is a vtable, not an `#ifdef` per call site. The indirect call is unmeasurable next to
 * any copy worth enqueuing, and in exchange each `rad_dev_*` symbol has one definition, so the
 * backends cannot drift into different contracts. That matters because the host backend is where
 * everything above the kernels is developed (docs/IMPLEMENTATION.md).
 */
#pragma once
#include "rad_abi.h"
#include "rad_device.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <functional>
#include <string>
#include <utility>
#include <vector>

/* An event that orders work on one card only. Its record makes the stream's writes visible to
 * that card and nothing else: another stream of the same card may wait on it and the host may
 * query or sync it for timing, but the host must not read device-written memory on its strength
 * and no other card may wait on it. The gain is the cheaper release: a system-scope release waits
 * for every in-flight write to host memory, ~1-10 us per record while the link streams weights. */
extern "C" int rad_event_create_local(RadEvent* out);
/* Event purpose, fixed at creation. RAD_EVENT_LOCAL is rad_event_create_local's scope.
 * RAD_EVENT_HOST_WAIT records complete with an interrupt a host thread can block on; without it
 * records complete silently and only a stream can wait on them. rad_event_create is
 * RAD_EVENT_HOST_WAIT, rad_event_create_local is RAD_EVENT_LOCAL alone, and rad_event_sync refuses
 * an event made without RAD_EVENT_HOST_WAIT. */
enum : unsigned {
    RAD_EVENT_LOCAL     = 1u,
    RAD_EVENT_HOST_WAIT = 2u,
};
extern "C" int rad_event_create_as(RadEvent* out, unsigned flags);
/* The host has just written memory that a kernel issued after this call reads, with no host wait
 * in between (e.g. a host-site op's output read over the link by the next device op). */
extern "C" void rad_dev_host_wrote(void);
/* Chained launches: chainable rows run their chained forms (RadKernelChainFn), and the backend
 * lets a chained kernel (named r4dchain*) wait in its prologue instead of behind the barrier bit.
 * Only on the AQL backend, whose queues keep the completion counter, and only where the model's
 * plugin measured a gain (RadArchProbe::chain_pays): with its drafter decoding speculatively, or
 * with and without one (rad_dev_chain_infer, from the engine before any row resolves). */
extern "C" int  rad_dev_chain_enabled(void);
extern "C" void rad_dev_chain_infer(int pays);

/* One range of a ranged copy: `bytes` from source offset `src` to destination offset `dst`, all
 * three multiples of four. */
struct RadCopyRange {
    int64_t src;
    int64_t dst;
    int64_t bytes;
};
/* Copies src[0, prefix) to dst[0, prefix) (a multiple of 16, possibly zero), then every range,
 * in one dispatch. The kernel reads `ranges` when it runs, so the list must be device-reachable
 * (pinned host memory beside `src`) and stay unchanged until the stream passes the copy. One
 * workgroup copies one range, so callers cut long runs into ranges of a few KB. */
extern "C" int rad_memcpy_ranges_async(void* dst, const void* src, int64_t prefix,
                                       const RadCopyRange* ranges, int n, RadStream s);

/* One copy of a list: `bytes` from `src` to `dst`, both absolute addresses. */
struct RadCopySeg {
    const void* src;
    void*       dst;
    int64_t     bytes;
};
/* Many unrelated copies, in list order on the stream, on the card's DMA engine where the backend
 * has one (otherwise segment by segment). The list itself is read at the call; the bytes it names
 * must live until the copy runs. An empty list enqueues nothing but performs the card's first-list
 * setup, so a caller can pay that cost at bring-up.
 *
 * For bulk link copies while the card computes (e.g. a layer's pooled experts in the prefill
 * stager). A copy kernel holds a CU slot for the link's duration and waits for a free one at
 * launch: beside busy kernels, 1.3 MB pinned-host copies run at 2.3 GB/s as kernels versus
 * 28.7 GB/s (link rate) on the DMA engine, at no cost to the kernels (aql.cpp, "the DMA lane").
 *
 * RAD_COPY_YIELD makes the list yield to the card's collectives: between ~1 MB pieces it waits
 * while a stream of that card holds the link (rad_link_hold). */
#define RAD_COPY_YIELD 1u
extern "C" int rad_memcpy_list_async(const RadCopySeg* segs, int n, unsigned flags, RadStream s);

/* Holds the card's link for the work `s` issues between two calls. `on` = 1 where step-critical
 * link traffic (a collective) starts, 0 where it ends, or 2 where the work in between released the
 * hold itself (rad_link_gate). RAD_COPY_YIELD lists for that card pause between pieces while held.
 * A card's holds alternate on/off in issue order across all its streams; they never nest or
 * overlap. Between the two calls the holding stream must not wait on anything behind a yielding
 * list, or the list and the hold deadlock. A backend without a DMA engine holds nothing.
 *
 * Why: collective peer bytes and stager copies share the inbound link, and the DMA engine takes
 * its share at full rate. A 4 MB all-reduce message takes 165 us alone and 301 us beside a layer's
 * copies; the step waits on the collective, while the copies have the rest of the layer. */
extern "C" int rad_link_hold(RadStream s, int on);

/* The launches this thread issues between `on` 1 and `on` 0 are a collective's: their blocks wait
 * for the other ranks' blocks of the same launch on another card. A recorded pass keeps the mark.
 *
 * rad_dev_cards_claim tells the backend which cards an engine's collectives run on (`on` 1, once
 * per engine with more than one rank) and that the engine is gone (`on` 0). On a card more than one
 * engine has claimed, nothing may wait on the card behind a collective except the collective's own
 * blocks (aql.cpp, "collectives on a shared card"). A claim that makes a card shared returns once
 * everything issued before it on that card's streams has finished. */
extern "C" void rad_dev_peer_launch(int on);
extern "C" int  rad_dev_cards_claim(const int* cards, int n, int on);

/* The hold word, for a launch that ends its own hold: a 64-bit word reachable by every engine and
 * kernel of the card, odd while the link is held. A launch handed this address releases the hold
 * by decrementing the word once, only if odd, as soon as its awaited bytes have landed; follow it
 * with rad_link_hold(s, 2). Null where the backend holds nothing. */
extern "C" void* rad_link_gate(RadStream s);

/* Device half of the expert prefetch (core/runtime/xprefetch.cpp). A backend that cannot have a
 * dispatch complete a signal, or has no spare DMA engine, leaves these out; rad_xp_supported is
 * then 0 and the prefetch is off.
 *
 * A ride is a signal completed by the next dispatch on a stream, instead of a separate barrier
 * packet. That dispatch keeps its ordering bit, so the signal covers everything before it.
 * rad_ride_wait puts a barrier waiting on it onto another stream and recycles the signal behind
 * that barrier. A ride handed to rad_xp_list as its mark is returned with rad_ride_release once
 * the caller is done with it, `after` null; the backend keeps it out of reuse until the list's
 * engine is past its poll. One never handed to a list (a dropped request's) is returned with
 * `after` and `packets` reaching past the dispatch it rides on: the signal is reused only once
 * `after` has consumed `packets` more packets.
 *
 * rad_xp_list gives the card's prefetch DMA queue a list: wait for `mark` (a ride, or null), copy
 * the `n` ranges, then write `done_val` to `done` (pinned host memory, or null). rad_xp_behind
 * counts lists handed over and not yet finished. The prefetch ring is uncached device memory
 * (rad_dev_alloc_uncached): the engine writes it and kernels read it, so no cache may hold a
 * slot's previous contents. */
/* An event a DMA list completes instead of a dispatch. `*word` is its completion word, for
 * rad_xp_list to write 0 to as the list's `done`; the caller arms it with a 1 in stream order (the
 * weight table's patch dispatch). `rad_event_wait(s, ev)` then holds `s` on the device until the
 * copy has landed. The caller waits for the previous copy before arming the next, so an older 0
 * cannot release a later wait. RAD_E_STATE while a tape records: the caller re-runs it with the
 * tape paused (core/kvstream/stage.cpp). */
extern "C" int     rad_event_dma(RadEvent ev, RadStream s, uint32_t** word);
extern "C" int     rad_xp_supported(int device);
extern "C" int     rad_xp_stream_create(RadStream* out);
extern "C" int     rad_ride(RadStream s, void** token);
extern "C" int     rad_ride_wait(RadStream s, void* token);
extern "C" void    rad_ride_release(void* token, RadStream after, int64_t packets);
/* `flags` RAD_XP_NO_YIELD: the list does not yield the link to a hold by a stream of this card. A
 * copy the compute stream waits on must set it, since the hold's release is behind that wait. */
#define RAD_XP_NO_YIELD 1u
extern "C" int     rad_xp_list(int device, void* mark, int n, const uint64_t* src, const uint64_t* dst,
                               const uint64_t* bytes, uint32_t* done, uint32_t done_val,
                               unsigned flags);
extern "C" int64_t rad_xp_behind(int device);
extern "C" void*   rad_dev_alloc_uncached(int64_t bytes);
extern "C" void    rad_dev_free_uncached(void* p);

/* [foff, foff + bytes) of the open file `fd`, mapped MAP_SHARED and pinned for every card as a
 * RAD_MEM_HOST_MAPPED allocation is: one range at the same address on host and cards. For a table
 * that engines on separate cards share through a tmpfs file (format/table_share.h). `foff` is a
 * multiple of 2 MiB and the file holds the whole range. Freed with rad_dev_free(p,
 * RAD_MEM_HOST_MAPPED), which unpins and unmaps and leaves the file alone. Null with the reason in
 * rad_dev_last_error() on a backend without it or a failure. */
extern "C" void*   rad_dev_map_file(int fd, int64_t foff, int64_t bytes);

/* A gate on a stream: the stream stops at the point rad_stream_gate enqueues and goes on once the
 * host calls rad_gate_open. Everything enqueued behind the gate meanwhile is then ready at once, so
 * the device runs it back to back however long the host took to submit it. For timing a batch of
 * launches without the host's submission rate in the measurement (tools/kbench_timer.cpp); the
 * engine has no use for it.
 *
 * Behind a closed gate the queue only fills. A caller enqueues at most rad_gate_room(s) more
 * submissions, counting each launch's own (a kernel may issue several), and must not wait on the
 * stream before opening the gate. The backends refuse with RAD_E_STATE, rather than deadlock, a
 * submission that would wait for room and a rad_stream_sync on a gated stream; an event sync on a
 * record behind the gate is not caught. Open every gate before destroying its stream.
 *
 * The gate itself acquires at system scope, so nothing behind it takes the acquire a first
 * dispatch after a host wait otherwise takes. rad_gate_room counts the submissions that fit
 * before the queue is full or before a dispatch would take the periodic system-scope acquire
 * (an L2 invalidate, see aql.cpp "fences"); on a stream without a closed gate it is the same
 * count with the device still draining. `gate` is single-use: opening it frees it. Not inside a
 * tape capture. RAD_E_UNSUPPORTED (and a room of 0) on a backend without gates. */
extern "C" int     rad_stream_gate(RadStream s, void** gate);
extern "C" int     rad_gate_open(void* gate);
extern "C" int64_t rad_gate_room(RadStream s);

namespace rad {

/* A recorded pass, owned by the backend that recorded it. See DeviceBackend::tape_begin. */
typedef void* RadTape;

/* ------------------------------------------------------------------ active backend */
/* Decided once at first use and cached. Used by the planner (a host backend has only a budget,
 * no VRAM tier), the selector (only RAD_DOMAIN_HOST can run) and tests that skip device-only
 * assertions. */
bool        device_is_host();
const char* device_backend_name();      /* "host" or "aql" */

/* ------------------------------------------------------------------ two contract notes
 *
 * Properties of rad_device.h that the frozen header does not state; tests/device_test.cpp pins
 * them against a real card.
 *
 *  1. `rad_memset_async` requires a device-accessible destination: RAD_MEM_DEVICE,
 *     RAD_MEM_HOST_PINNED or RAD_MEM_HOST_MAPPED. hipMemsetAsync refuses pageable RAD_MEM_HOST.
 *     The host backend cannot reproduce that refusal, so this is the one asymmetry between the
 *     two (see d_memset_async in hip.cpp). `rad_memcpy_async` has no such restriction.
 *
 *  2. Both backends refuse a null RadStream. HIP's stream 0 serialises against every blocking
 *     stream, contrary to spec §5.4 (no host synchronisation in a transfer). The engine never uses
 *     the default stream, so null is a bug.
 */

/* ------------------------------------------------------------------ the backend vtable */
/* One entry per call in rad_device.h. `name` is what device_backend_name() returns. */
struct DeviceBackend {
    const char* name;

    int   (*count)(void);
    int   (*set)(int device);
    int   (*props)(int device, RadDeviceProps* out);
    int   (*enable_peer)(int self, int peer);

    void* (*alloc)(int64_t bytes, int kind);
    void  (*free)(void* p, int kind);
    void* (*host_ptr)(void* p);
    void* (*device_ptr)(void* p);

    int   (*stream_create)(RadStream* out, int high_priority);
    void  (*stream_destroy)(RadStream s);
    int   (*stream_sync)(RadStream s);

    /* `flags`: see rad_event_create_as. A backend with no scope to choose ignores RAD_EVENT_LOCAL. */
    int   (*event_create)(RadEvent* out, unsigned flags);
    void  (*event_destroy)(RadEvent e);
    int   (*event_record)(RadEvent e, RadStream s);
    int   (*event_wait)(RadStream s, RadEvent e);
    int   (*event_query)(RadEvent e);
    int   (*event_sync)(RadEvent e);
    int   (*event_elapsed_ms)(RadEvent a, RadEvent b, float* out);

    /* The host wrote memory that kernels issued from here on read, with no host wait in between:
     * do whatever a wait would have done to make host writes visible to the device. */
    void  (*host_wrote)(void);

    int   (*memcpy_async)(void* dst, const void* src, int64_t bytes, RadStream s);
    int   (*memcpy_2d_async)(void* dst, int64_t dpitch, const void* src, int64_t spitch,
                             int64_t width, int64_t height, RadStream s);
    int   (*memset_async)(void* dst, int value, int64_t bytes, RadStream s);
    int   (*memcpy_ranges_async)(void* dst, const void* src, int64_t prefix,
                                 const RadCopyRange* ranges, int n, RadStream s);
    int   (*memcpy_list_async)(const RadCopySeg* segs, int n, unsigned flags, RadStream s);
    int   (*link_hold)(RadStream s, int on);
    void* (*link_gate)(RadStream s);

    /* ---- recorded passes. Null on a backend without them; the run phase then issues every
     * pass. See Ctx::run_step for what a tape is for.
     *
     * `tape_begin` captures every submission this thread makes (dispatches, event records and
     * waits) outside `tape_pause` (which nests). `tape_mark` returns the record count, so the
     * caller can cut the tape at its own host work. `tape_end` returns the tape, or null when
     * `keep` is 0 or the pass did something a tape cannot reproduce (reason in
     * rad_dev_last_error). `tape_play` resubmits records [from, to) onto their recorded queues.
     * `tape_destroy` may be called while a play is queued; the tape is freed once unreachable. */
    int     (*tape_begin)(void);
    int     (*tape_mark)(void);
    void    (*tape_pause)(int on);
    RadTape (*tape_end)(int keep);
    int     (*tape_len)(RadTape t);
    int     (*tape_play)(RadTape t, int from, int to);
    /* First record at which two tapes differ, or -1 if they agree (same submissions to the same
     * queues, same packets apart from argument addresses, same argument bytes). The difference is
     * described into `why`. */
    int     (*tape_diff)(RadTape a, RadTape b, char* why, int cap);
    void    (*tape_destroy)(RadTape t);
    /* `tape_record`: record `i`'s kind (0 dispatch, 1 event record or wait) and queue.
     * `tape_set_overlap`, the one edit a closed tape accepts: play dispatch `i` without the barrier
     * bit, so it launches while earlier packets on its queue still run. The caller sets it only
     * where the dispatch touches nothing those packets write, and on at most kTapeOverlapRun
     * consecutive dispatches; see Ctx::haz_mark. */
    int     (*tape_record)(RadTape t, int i, int* kind, uint32_t* queue);
    int     (*tape_set_overlap)(RadTape t, int i, int on);


    /* ---- elastic memory: an address range of fixed size whose backing comes and goes.
     *
     * Lets two pools that take turns share memory: the KV cache backs only what live sessions
     * hold, and the expert plane keeps the rest resident instead of streaming it over the link.
     *
     * Free-then-reallocate does not work: the KV pool is layer-major (core/mem/kv.h), layer i at
     * `base + i * n_blocks * layer_bytes_per_block`, so unused blocks are a stripe inside each
     * layer, not a tail. Reserving the maximum range and backing part of it keeps every stride
     * where the kernels expect it.
     *
     * A backend without elastic memory leaves these null and pools are sized once at startup;
     * that is a supported configuration.
     *
     * Contract:
     *   `vmem_reserve` takes address space, no pages.
     *   `vmem_commit` backs one granule. The driver unmaps a mapping only exactly as it was
     *   mapped (four granules mapped as one cannot be unmapped singly), so VMem maps a mapping per
     *   granule, a run of them in one `vmem_commit_units` call where the backend has it. It
     *   returns 1 when the backing is recycled memory that holds what its last mapping left, 0
     *   when it reads zero.
     *   `vmem_decommit` unbacks a range that is one mapping or a contiguous run of whole ones, in
     *   one unmap: each unmap costs a driver round trip that grows with the mappings the process
     *   holds (8.8 us a granule alone, 48 beside 6000 others; 26 for a run, per granule).
     *   `vmem_protect` grants this device read/write access to an already-backed range. It is
     *   separate from commit because access set on a granule adjacent to an accessible one is
     *   refused, while the whole contiguous run is accepted; VMem names the run.
     *   `vmem_release` takes a range with nothing committed in it.
     *
     * Overlap bookkeeping lives once in VMem below, not in each backend. */
    int64_t (*vmem_granularity)(void);                     /* commit quantum; 0 = unsupported */
    void*   (*vmem_reserve)(int64_t bytes, int64_t align);
    void    (*vmem_release)(void* base, int64_t bytes);
    int     (*vmem_commit)(void* addr, int64_t bytes);
    int     (*vmem_decommit)(void* addr, int64_t bytes);
    int     (*vmem_protect)(void* addr, int64_t bytes);

    /* ---- observation. VRAM in use on a card by all processes, as the driver counts it, read
     * without entering the device runtime (a dashboard polls it, and a runtime call can block on
     * a lock the launch path holds). Available once props() has been read for the device;
     * otherwise, or without such a counter, RAD_E_UNSUPPORTED. */
    int     (*vram_used)(int device, int64_t* out);

    /* ---- plugin device code. `code_mark` counts fat-binary registrations so far; `code_runs`
     * reports whether every registration in [from, to) has a code object `device` runs (1), not
     * (0, reason in `why`), or unknown (< 0). The plugin loader brackets each dlopen with marks.
     * Null on a backend that loads no device code. */
    size_t  (*code_mark)(void);
    int     (*code_runs)(size_t from, size_t to, int device, char* why, int cap);

    /* ---- 1 when the device is integrated graphics: its "device memory" is host memory shared
     * with every process. Null (answers 0) on a backend with no such devices. */
    int     (*integrated)(int device);

    /* ---- 1 when `self` can access `peer`'s memory directly, 0 when not, < 0 when unknown. A
     * query only; rad_dev_enable_peer maps the memory. Null (answers 1) on a backend whose devices
     * share one address space. */
    int     (*peer_access)(int self, int peer);

    /* ---- the card's PCI address, "0000:03:00.0", into `out`; RAD_E_UNSUPPORTED for non-PCI
     * devices. The startup report (core/diag/diag.h) uses it to join the runtime's device index to
     * the kernel's view of the card (link, power limits, IOMMU group). */
    int     (*pci_address)(int device, char* out, int cap);
    /* ---- the device runtime's name and version, "HIP 7.2.25403", into `out`. Null on a backend
     * with no runtime. */
    int     (*runtime_version)(char* out, int cap);
    /* ---- what carries rad_memcpy_list_async for `device`, into `out`: the card's DMA engine on
     * the process's own queue, or why not. Null on a backend that copies segment by segment. */
    int     (*dma_lane)(int device, char* out, int cap);

    /* ---- expert prefetch device half: see rad_xp_supported. Last in the table, so a backend
     * listing its entries positionally leaves them null. */
    int     (*xp_supported)(int device);
    int     (*xp_stream_create)(RadStream* out);
    int     (*ride)(RadStream s, void** token);
    int     (*ride_wait)(RadStream s, void* token);
    void    (*ride_release)(void* token, RadStream after, int64_t packets);
    int     (*xp_list)(int device, void* mark, int n, const uint64_t* src, const uint64_t* dst,
                       const uint64_t* bytes, uint32_t* done, uint32_t done_val, unsigned flags);
    int64_t (*xp_behind)(int device);
    void*   (*alloc_uncached)(int64_t bytes);
    void    (*free_uncached)(void* p);

    /* ---- gates: see rad_stream_gate. Null on a backend without them. */
    int     (*stream_gate)(RadStream s, void** gate);
    int     (*gate_open)(void* gate);
    int64_t (*gate_room)(RadStream s);

    /* ---- rad_event_dma; null on a backend with no DMA queue. */
    int     (*event_dma)(RadEvent e, RadStream s, uint32_t** word);

    /* ---- see rad_dev_map_file. Null on a backend with no pinned host memory to map a file into. */
    void*   (*map_file)(int fd, int64_t foff, int64_t bytes);

    /* ---- see dev_stuck_ms. Null on a backend that cannot tell. */
    int64_t (*stuck_ms)(void);

    /* ---- host-backed mappings, for parking (paging.cpp). `vmem_commit_host` backs a reserved
     * granule with pinned host memory that the reservation's card reads and writes in place, at
     * that address, and the host can too. `vmem_swap` exchanges what backs two mapped ranges of
     * equal size (each keeps its kind: a range of device memory stays device memory wherever it
     * is mapped). Remapped addresses need dev_settle_maps before the card writes them. Null on a
     * backend without them; parking then keeps no host copies. */
    int     (*vmem_commit_host)(void* addr, int64_t bytes);
    int     (*vmem_swap)(void* a, void* b, int64_t bytes);

    /* ---- [addr, addr + bytes) backed as one mapping per `unit` bytes, each of which can be
     * unbacked alone later, with access granted once over the run: per granule, granting access
     * costs about as much as the map (41.9 us a granule backed singly, 24.6 as a run). All or
     * nothing: on failure nothing of the run stays backed. Null: VMem backs a granule at a time. */
    int     (*vmem_commit_units)(void* addr, int64_t bytes, int64_t unit);

    /* ---- device memory a decommit kept for the next commit of its size (counted as free in
     * props), given back to the driver: card `device`'s, or every card's for -1; the bytes. Null
     * on a backend that keeps none. */
    int64_t (*vmem_trim)(int device);
    /* rad_dev_peer_launch and rad_dev_cards_claim. Null on a backend that never launches a kernel
     * ahead of the one before it. */
    void  (*peer_launch)(int on);
    int   (*cards_claim)(const int* cards, int n, int on);
    /* ---- the hardware queues card `device` has open and how many of them no stream holds (a
     * closed stream's queue is kept for the card's next stream); and the kept ones destroyed
     * until `keep` are left, the number destroyed, none while any queue of the card holds a
     * packet. Null on a backend that keeps none. */
    void    (*stream_counts)(int device, int* open, int* idle);
    int     (*stream_trim)(int device, int keep);
};


const DeviceBackend* host_backend(void);

/* The current backend's recorded passes; RAD_E_UNSUPPORTED (or null) from one that has none. */
bool    tape_supported();
int     tape_begin();
int     tape_mark();
void    tape_pause(bool on);
RadTape tape_end(bool keep);
int     tape_len(RadTape t);
int     tape_play(RadTape t, int from, int to);
int     tape_diff(RadTape a, RadTape b, std::string* why);
void    tape_destroy(RadTape t);
int     tape_record(RadTape t, int i, int* kind, uint32_t* queue);
int     tape_set_overlap(RadTape t, int i, bool on);

/* The current backend's DeviceBackend::peer_access. */
int     dev_peer_access(int self, int peer);

/* The current backend's DeviceBackend::vram_used. */
int     dev_vram_used(int device, int64_t* out);

/* The current backend's DeviceBackend::code_mark and code_runs. Without them the mark is 0 and
 * every plugin runs (1). */
size_t  dev_code_mark();
int     dev_code_runs(size_t from, size_t to, int device, std::string* why);

/* The current backend's DeviceBackend::integrated. */
bool    dev_integrated(int device);

/* The current backend's DeviceBackend::pci_address and runtime_version, "" where it has none. */
std::string dev_pci_address(int device);
std::string dev_runtime_version();
/* The current backend's DeviceBackend::dma_lane, "" where it has none. */
std::string dev_dma_lane(int device);

/* How long the device has been stuck, in ms, or 0: since the last moment anything advanced, once a
 * host wait has seen nothing advance for its stall period and named the stuck packet (the queue
 * dump), or since a queue faulted. Anything advancing again clears it. 0 where the backend cannot
 * tell. */
int64_t dev_stuck_ms();

/* How long the device may be stuck (dev_stuck_ms) before the backend ends the process, in ms; 0
 * never. Set from `scheduler.stuck-exit` before serving; a backend that can tell reads it when a
 * stall is standing. A device that has advanced nothing for a minute is not coming back: the
 * reported hangs stood for hours, and a stall that clears does so within seconds. */
extern std::atomic<int64_t> dev_stuck_exit_ms;

/* Longest run of dispatches a played tape leaves unordered. Of any kTapeOverlapRun + 1
 * consecutive packets on a queue one carries the barrier bit, so a launched packet still proves
 * that everything more than that many packets earlier has completed. Argument slots and tape
 * lifetimes are reclaimed on that guarantee. */
constexpr int kTapeOverlapRun = 6;

/* ------------------------------------------------------------------ the allocation ledger
 * By RAD_MEM_* kind: the bytes rad_dev_alloc has handed out and rad_dev_free has not taken back,
 * and the number of allocations since process start. It does not see a
 * kernel plugin that calls hipMalloc directly; see the ledger in dispatch.cpp. */
void rad_dev_alloc_totals(int kind, int64_t* bytes, int64_t* count);

/* `fn` is called after every device-memory free (RAD_MEM_DEVICE) with the range freed: a kernel
 * library's rad_plugin_freed (abi/rad_abi.h), registered by the plugin loader. Registrations are
 * counted: every registry that loads the library registers its hook, and rad_dev_off_free, which
 * the loader calls before it closes the library, takes one back. The hook goes with the last,
 * once no free is calling it: a library opened again can land at another address, and a hook
 * left behind would call into memory nothing maps. */
void rad_dev_on_free(void (*fn)(const void* base, int64_t bytes));
void rad_dev_off_free(void (*fn)(const void* base, int64_t bytes));

/* Event handles created and not yet destroyed, process-wide. */
int64_t rad_dev_live_events(void);
/* Calls to rad_event_sync and rad_stream_sync since process start, from any thread: the host
 * blocks the ABI makes. */
int64_t rad_dev_host_blocks(void);

/* A parked allocation's contents (docs/MULTIMODEL.md §7): in pinned host memory, in a file of the
 * disk tier, or both. A copy written to disk can let its host memory go and be read back later. */
struct HostCopy {
    void*       host = nullptr;   /* pinned, `bytes` rounded up to the direct-I/O block */
    int64_t     bytes = 0;
    std::string file;             /* the disk tier's copy, when written */
    bool        on_disk = false;  /* `file` holds exactly what `host` holds (or held) */

    int64_t ram() const { return host ? bytes : 0; }
    /* Pinned memory for `n` bytes; what the copy held before, host memory and file, is freed. */
    int  alloc(int64_t n);
    /* Writes the host bytes to a new file under `dir` (O_DIRECT). Between pieces it asks `gate`
     * (when given): 0 goes on, 1 waits (a model loading reads its container from the same drive),
     * -1 abandons the write (RAD_E_STATE, no file left). */
    int  spill(const std::string& dir, const std::function<int()>& gate = {});
    /* Frees the host memory of a copy that is on disk. */
    void drop_host();
    /* Reads the file back into fresh pinned memory when the host copy is gone. */
    int  load();
    /* Frees everything: the host memory and the file. */
    void release();
};

/* ------------------------------------------------------------------ elastic memory
 * One reserved address range and which granules inside it are backed.
 *
 * The interval bookkeeping lives here, not in the backends, because the driver refuses a commit
 * overlapping a commit and a decommit covering a hole. Backends get granule-aligned, disjoint
 * ranges and make no decisions.
 *
 * The roundings differ on purpose: commit rounds outward, so every requested byte is backed;
 * decommit rounds inward, so a granule holding any byte still in use is kept. A shared rounding
 * would either strand memory or unmap live data (a kernel fault at a valid address).
 *
 * Not thread-safe. A range is owned by the owner of the pool it backs, and commit/decommit is
 * policy, not step-path work. */
class VMem {
public:
    /* Whether the backend has elastic memory. Without it callers size pools whole (a supported
     * configuration; see DeviceBackend). */
    static bool    supported();
    static int64_t granularity();          /* 0 when unsupported */

    VMem() = default;
    ~VMem() { release(); }
    VMem(const VMem&) = delete;
    VMem& operator=(const VMem&) = delete;
    /* Movable, since owners live in vectors. The source is left empty so its destructor does not
     * unmap the range the destination now owns. */
    VMem(VMem&& o) noexcept
        : base_(o.base_), size_(o.size_), committed_(o.committed_), ext_(std::move(o.ext_)),
          device_(o.device_), copy_(std::move(o.copy_)), parked_(std::move(o.parked_)),
          discarded_(o.discarded_) {
        rekey(&o, this);
        o.base_ = nullptr;
        o.size_ = o.committed_ = 0;
        o.ext_.clear();
        o.copy_ = HostCopy{};
        o.parked_.clear();
    }
    VMem& operator=(VMem&& o) noexcept {
        if (this != &o) {
            release();
            base_ = o.base_;
            size_ = o.size_;
            committed_ = o.committed_;
            ext_ = std::move(o.ext_);
            device_ = o.device_;
            copy_ = std::move(o.copy_);
            parked_ = std::move(o.parked_);
            discarded_ = o.discarded_;
            rekey(&o, this);
            o.base_ = nullptr;
            o.size_ = o.committed_ = 0;
            o.ext_.clear();
            o.copy_ = HostCopy{};
            o.parked_.clear();
        }
        return *this;
    }

    /* Address space only: no pages, no VRAM. `align` defaults to the granularity. Reserve the
     * maximum the pool may ever reach; the range cannot grow later without moving addresses. */
    int   reserve(int64_t bytes, int64_t align = 0);
    void  release();                       /* decommits whatever is still backed, then frees the VA */

    void*   base() const { return base_; }
    int64_t size() const { return size_; }
    int64_t committed() const { return committed_; }   /* bytes actually backed right now */

    /* Idempotent and clamped to the range. commit() backs every unbacked granule touching
     * [off, off+bytes); decommit() unbacks every backed granule wholly inside it. Doing nothing is
     * success. */
    int  commit(int64_t off, int64_t bytes);
    int  decommit(int64_t off, int64_t bytes);

    /* Whether every byte of [off, off+bytes) is backed; check before handing a range to a
     * kernel. */
    bool is_committed(int64_t off, int64_t bytes) const;

    /* Parking (dev_page_out): every backed extent copied to pinned host memory on `s`, then
     * unbacked; the bytes held on the host, or an error. page_in backs the same extents again and
     * copies them back. With `discard` nothing in the range is kept (a cache with no live
     * sequence and its index reset): page_out copies nothing and returns 0, and page_in backs the
     * extents again without copying anything back. */
    int64_t page_out(RadStream s, bool discard = false);
    int     page_in(RadStream s);
    bool    parked() const { return !parked_.empty(); }
    int64_t parked_bytes() const {
        int64_t n = 0;
        for (const auto& e : parked_) n += e.second;
        return n;
    }
    /* The card the range was reserved on, for the copies' stream. */
    int     device() const { return device_; }
    /* Host bytes held for the parked extents (zero once they live only on disk). */
    int64_t host_bytes() const { return parked_.empty() ? 0 : copy_.ram(); }
    /* The disk tier (dev_page_spill): write the parked copy out, or let its host memory go. */
    int     spill(const std::string& dir, const std::function<int()>& gate = {}) {
        return parked_.empty() ? RAD_OK : copy_.spill(dir, gate);
    }
    int64_t drop_host() {
        if (parked_.empty() || !copy_.on_disk || !copy_.host) return 0;
        const int64_t n = copy_.bytes;
        copy_.drop_host();
        return n;
    }

private:
    /* The owner registry follows the object when it moves (reserve() files it under the thread's
     * engine owner). */
    static void rekey(VMem* from, VMem* to);
    void*   base_ = nullptr;
    int64_t size_ = 0;
    int64_t committed_ = 0;
    /* Backed extents (offset, bytes): sorted, disjoint and never adjacent, since a touching
     * commit merges. Length tracks live regions, not calls. */
    std::vector<std::pair<int64_t, int64_t>> ext_;
    int     device_ = 0;
    HostCopy copy_;                                       /* parked: the extents' contents */
    std::vector<std::pair<int64_t, int64_t>> parked_;     /* parked: the extents to back again */
    bool    discarded_ = false;                           /* parked without a copy */

    /* Adds a newly mapped piece and merges it with whatever it touches. */
    void add_extent(int64_t off, int64_t bytes);
    void drop_extent(int64_t off, int64_t bytes);
};

/* ------------------------------------------------------------------ parking (paging.cpp)
 * docs/MULTIMODEL.md §7. A device allocation of 2 MiB or more made on a thread working for an
 * engine (engine_owner, rad_internal.h) is reserved as one address range and mapped in chunks, and
 * an elastic range reserved on such a thread is filed under the engine too, so the engine's memory
 * can leave the card in part or whole and come back at the same addresses.
 *
 * A chunk out of the card is somewhere else: in pinned host memory, in a file of the disk tier,
 * both, or nowhere when it was discarded (its contents not needed: an expert slab's free slots).
 * Restoring maps every chunk out and copies back the ones that had a copy. The engine must be
 * quiet for both: nothing it issued still runs, nothing new is issued until the page-in. */
constexpr int64_t kPageMin = (int64_t)2 << 20;
/* For rad_dev_alloc and rad_dev_free: an engine's pageable allocation, and its release (false when
 * `p` is not one). */
void* page_alloc(int64_t bytes, int owner, int device);
bool  page_free(void* p);
/* Device bytes `owner` holds on card `device`: its allocations and elastic backing (a load in
 * flight, for the fleet's reservation). */
int64_t dev_owner_bytes(int owner, int device);
/* FreeToken: peak of the same owner ledger over its lifetime; device -1 is the simultaneous
 * total, not a sum of card peaks. Excludes plugins allocating outside the device ABI. */
int64_t dev_owner_high_water(int owner, int device);
/* The backend rad_dev_* calls go to. */
const DeviceBackend* active_backend();
/* The owner's elastic ranges (dispatch.cpp's registry). */
std::vector<VMem*> vmems_of(int owner);

struct PageOut {
    /* Bytes to take off each card; empty takes everything. */
    std::map<int, int64_t> want;
    /* New host memory the copies may take; past it they go to `dir`, and without a `dir` the
     * page-out stops there. */
    int64_t     host_budget = INT64_MAX;
    std::string dir;
    /* The elastic ranges (the KV cache) are dropped, not copied. */
    bool        discard_elastic = false;
};
struct PageResult {
    std::map<int, int64_t> freed;   /* bytes unmapped on each card */
    int64_t host = 0;               /* host bytes the copies took */
    int64_t disk = 0;               /* bytes written straight to the disk tier */
    int64_t dropped = 0;            /* bytes unmapped with nothing kept (discarded) */
    int64_t reused = 0;             /* bytes unmapped against a copy made earlier */
    int64_t file = 0;               /* bytes unmapped against the container they were loaded from */
    double  quiet_s = 0;            /* the engine coming to rest (responses drained, loop held) */
    double  slab_s = 0;             /* lent slots recalled and the expert slab emptied */
    double  out_s = 0;              /* the page-out itself */
};
int     dev_page_out(int owner, const PageOut& req, PageResult* out);
/* A stretch of memory a page-in mapped for its owner to fill (dev_page_refilled). */
struct PageSpan { char* p; int64_t bytes; int device; };
/* Every chunk of the owner that is out comes back. Those of an allocation the owner refills are
 * only mapped, and appended to `refill` for the owner to fill before anything reads them; a
 * page-in with such chunks out and no `refill` fails (RAD_E_STATE) and maps nothing. */
int     dev_page_in(int owner, std::vector<PageSpan>* refill = nullptr, bool elastic = true);
/* The owner's elastic ranges backed again (and their copies, if they were kept), what a page-in
 * does last; then the cards' translations settled. A page-in with `elastic` false leaves this to
 * the caller, which can run it beside the refill: nothing of the refill reads those ranges, and
 * an image model's KV cache is ten thousand commits (0.5 s). */
int     dev_page_in_elastic(int owner);
/* The owner's chunks wholly inside these ranges hold nothing that needs keeping: the next page-out
 * unmaps them first and copies nothing (an expert slab's slots after it dropped its units). */
void    dev_page_discard(int owner, const std::vector<std::pair<const void*, int64_t>>& ranges);
/* Bytes mapped on cards, and bytes of host copies (parked or kept). */
void    dev_page_stats(int owner, int64_t* device, int64_t* host);
/* Bytes of the owner's pageable memory still on each card. */
std::map<int, int64_t> dev_page_mapped_bytes(int owner);
/* Bytes the owner has out of the card right now, by card. With `live`, only what dev_page_in_live
 * would map: the sealed chunks parked in host memory stay out of the sum, read in place. */
std::map<int, int64_t> dev_page_out_bytes(int owner, bool live = false);
/* [off, off + bytes) of the pageable allocation `p` (all of it for bytes < 0) is never written after
 * the load (a static plan's weights pool, a tiered one's statics): a copy, once made, is kept after
 * the page-in, and later page-outs of those chunks only unmap. */
void    dev_page_seal(void* p, int64_t off = 0, int64_t bytes = -1);
/* The pageable allocation `p` is rebuilt by its owner from its source (a weights pool from its
 * container) after a page-in: its sealed chunks leave the card with no copy at all, an unmap, and
 * dev_page_in hands them back to be filled. Its chunks are no longer checked against the
 * container (dev_page_source). */
void    dev_page_refilled(void* p);
/* [p, p + bytes) is being written (an expert moved into a slab slot): the kept copies of the
 * chunks it touches are stale. A no-op for memory that is not pageable. Called when the write is
 * issued and again when it lands, so a copy made across it is never taken as exact. */
void    dev_page_invalidate(const void* p, int64_t bytes);
/* Memory mapped again at an address that was mapped before (a page-in, a pool growing back) is not
 * written until the cards' translation caches have been flushed in full: a map flushes them only
 * lightly, and copies into such memory can land nowhere, so the memory reads back its previous
 * owner's bytes or zeros (Linux 7.1 with gfx1201; tests/device_test.cpp
 * two_allocations_of_one_owner_come_back_exact_from_the_disk_tier). Unpinning host memory makes
 * the driver flush every card it was mapped on in full, so this pins one page and unpins it.
 * Called after a batch of maps, before anything writes them. */
/* Where address `a` lies among the elastic reservations and pageable allocations, for a fault report:
 * its owner, card, offset, whether it is backed and what its paging chunk holds. Empty when it is in
 * none. Takes no lock it would wait for, since a fault can arrive while one is held. */
std::string dev_fault_where(const void* a);

int     dev_settle_maps();

/* VMem::commit on this thread leaves its flush (dev_settle_maps) for the end of this scope, so a
 * run of ranges mapped together flushes once. Nothing may write the new ranges before end(), which
 * reports the flush's result; the destructor ends a batch left open by an early return. */
class MapBatch {
public:
    MapBatch();
    ~MapBatch();
    MapBatch(const MapBatch&) = delete;
    MapBatch& operator=(const MapBatch&) = delete;
    int end();
private:
    bool ended_ = false;
};
/* The backend's kept device memory on card `device` (-1: every card) back to the driver, the
 * bytes; its pages come free as the driver wipes them (~95 GB/s). For a caller about to allocate
 * outside the elastic calls, or done with the swap that would have reused it. */
int64_t dev_vmem_trim(int device);
/* The hardware queues card `device` has open, and how many of them are kept for its next streams
 * (0 and 0 on a backend that keeps none). */
void    dev_stream_counts(int device, int* open, int* idle);
/* Card `device`'s kept queues destroyed until `keep` are left: the number destroyed, about 41 MiB
 * of the card each. Nothing is destroyed while any queue of the card holds a packet, and the
 * caller sees to it that nothing is launched on the card during the call: destroying a queue
 * beside a busy one can reset the card. */
int     dev_stream_trim(int device, int keep);
/* The device this thread last selected with rad_dev_set. */
int     dev_current();
/* The container at `path`, by an id for dev_page_source (the same path gives the same id). */
int     dev_page_file(const char* path);
/* The load copied container `file`'s rows [0, ny) of wb bytes at foff + y * sp verbatim onto
 * [dst, dst + ny * wb). A chunk of pageable memory filled this way can go out against the container
 * instead of a copy: a page-out (or backup) reads the device bytes back, compares them with the
 * container (zero where nothing was loaded) and writes nothing when they match; the page-in reads
 * the container. A chunk written since (dev_page_invalidate) is copied as any other. A no-op for
 * memory that is not pageable. */
void    dev_page_source(const void* dst, int file, int64_t foff, int64_t wb, int64_t ny, int64_t sp);
/* Bytes of the owner's mapped chunks a page-out would have to copy (no discard, no kept copy,
 * not loaded verbatim from a container, not refilled by the owner). */
int64_t dev_page_uncopied(int owner);
/* A restore that serves at once: like dev_page_in, but a sealed chunk parked in host memory stays
 * there, read in place over the link, until dev_page_promote moves it; *left is how much. */
int dev_page_in_live(int owner, int64_t* left, std::vector<PageSpan>* refill = nullptr,
                     bool elastic = true);
/* dev_page_in (`left` null) or dev_page_in_live in two halves, for an owner that refills: the begin
 * maps every chunk, hands back those to refill and starts the copies of the chunks held in host
 * memory; the end waits for those copies and swaps their memory in, and must follow every begin
 * that succeeded. The refill runs between the two, beside the copies: it writes none of those
 * chunks. Elastic ranges are left to dev_page_in_elastic. */
int dev_page_in_begin(int owner, int64_t* left, std::vector<PageSpan>* refill);
int dev_page_in_end(int owner);
/* Between the two halves: waits for the clears the begin issued on recycled refill chunks, which
 * it does not wait for itself. The refill calls it before its first write into them; the end
 * waits for them too. The calling thread's current device is kept. */
int dev_page_in_cleared(int owner);
/* Moves up to `max_bytes` of the chunks dev_page_in_live left in host memory onto their cards,
 * while the engine serves: each is copied to device memory first, then `quiet` runs the swap of
 * mappings with nothing of the engine in flight (Engine::run_quiet). Returns the bytes moved, 0
 * when none is left, or an error (no device memory for them: the rest stays in host memory). */
int64_t dev_page_promote(int owner, int64_t max_bytes,
                         const std::function<int(const std::function<int()>&)>& quiet);
/* Bytes of `owner`'s chunks served from host memory in place, on `device` (-1: all). */
int64_t dev_page_live_bytes(int owner, int device = -1);
/* Host memory all parked and kept copies hold (they are host-backed mappings, not pinned
 * allocations, so rad_dev_alloc_totals does not count them). */
int64_t dev_page_host_bytes();
/* Frees host copies of mapped sealed chunks (a file copy stays). The host bytes freed. */
int64_t dev_page_drop_copies(int owner);
/* Writes every host copy without a file copy to the disk tier; `gate` as HostCopy::spill. */
int64_t dev_page_spill(int owner, const std::string& dir, const std::function<int()>& gate = {});
/* Frees host copies that have a file copy. The host bytes freed. */
int64_t dev_page_drop_host(int owner);
/* A parked owner's chunks whose only copy is the disk tier or its container: their bytes. */
int64_t dev_page_fillable(int owner);
/* The owner's parked chunks of allocations it refills that have no host memory yet, one span a
 * chunk, in address order. */
std::vector<PageSpan> dev_page_refill_chunks(int owner);
/* A host fill by the owner (Engine::fill_host): host memory mapped at each of these chunks (from
 * dev_page_refill_chunks), as a park to host memory leaves a chunk, for the owner to fill from its
 * source; then dev_page_fill_end: `filled`, the chunks' host copies are theirs and the next restore
 * copies them over the link (exact until written, so a later park keeps them); otherwise the host
 * memory goes again and the chunks stay to be refilled. */
int     dev_page_fill_begin(int owner, const std::vector<PageSpan>& chunks);
void    dev_page_fill_end(int owner, const std::vector<PageSpan>& chunks, bool filled);
/* Up to max_bytes of those read into host memory and mapped at their own addresses, as a park to
 * host memory leaves them, so the next restore copies over the link instead of reading the drive
 * (and a live restore serves from them in place). The owner must be parked and idle. Returns the
 * bytes filled, or a negative status. */
int64_t dev_page_fill_host(int owner, int64_t max_bytes);

/* Bytes backed by every VMem in the process, for Engine::start's receipt. VMem commits bypass the
 * rad_dev_alloc ledger, so without this the engine's accounting would miss the elastic pools. */
int64_t rad_dev_vmem_committed(void);

#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
/* Null when the build has HIP but the machine shows no device, so a machine with ROCm installed
 * but no card runs the host backend instead of failing at the first hipMalloc. Configure decides
 * which kernels are compiled; this decides whether there is a card to run them. */
const DeviceBackend* hip_backend_if_present(void);
/* The HIP backend with streams, events, copies and fills replaced by AQL packets the engine writes
 * itself (aql.cpp). Aborts with the reason on a machine it cannot run on. */
const DeviceBackend* aql_backend_if_present(void);
/* For tests: every card keeps its kernel argument rings and recorded-pass argument blocks in pinned
 * host memory, the placement a card whose PCI BAR is smaller than its VRAM gets, so that path runs
 * on any card. Takes effect only before the backend starts; RAD_E_STATE after. */
int     aql_force_host_kargs(void);
/* Bytes copied to or from pageable host memory so far; each such copy is a host wait. */
int64_t aql_pageable_bytes(void);
/* Fat binaries registered so far, counted whichever backend runs: the registration hooks are this
 * build's exported symbols and see every dlopen'ed plugin's device code, card or not. Serves as
 * the host backend's code_mark in a HIP build. */
size_t  aql_code_registered(void);
#endif

/* ------------------------------------------------------------------ the error slot */
/* Both backends write here; rad_dev_last_error() drains it. Thread-local because spec §1 gives each
 * tensor-parallel rank and the mover their own thread, and an error must be reported on the thread
 * that caused it. */
void device_set_error(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

}  /* namespace rad */
