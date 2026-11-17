/* dispatch.cpp -- picks a backend once, then forwards.
 *
 * The only definition of every rad_dev_* / rad_stream_* / rad_event_* symbol. host.cpp and hip.cpp
 * export vtables only, so there is no #ifdef above this file and both backends share one contract.
 *
 * A build with HIP still runs the host backend when no device is visible (e.g. a CI container
 * with ROCm headers and no /dev/kfd); the chosen backend is logged at Info.
 */
#include "device.h"
#include "directio.h"
#include "../rad_internal.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdint>
#include <unistd.h>

/* The device rad_dev_set last bound on this thread. */
static thread_local int t_device = 0;

namespace rad {

/* ------------------------------------------------------------------ the error slot */
/* Thread-local and cleared by reading, per the ABI. Each rank and the mover run on their own
 * thread (spec §1), so an error must stay on the thread that raised it. */
static thread_local std::string t_last_error;

void device_set_error(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    t_last_error.assign(buf);
    /* Debug, not Error: callers often recover (e.g. a pool retrying smaller), and fatal ones are
     * reported by the caller with more context. */
    RAD_DEBUG("device: %s", buf);
}

/* ------------------------------------------------------------------ selection */

/* The startup line's list of every visible device by index and arch. A machine may report more
 * devices than accelerators: an APU shows up beside discrete cards and claims system RAM as
 * "VRAM". The arch tells them apart. Filtering is the engine's decision (via rad_dev_props). */
static std::string device_inventory(const DeviceBackend* b) {
    std::string s;
    const int n = b->count();
    for (int i = 0; i < n; ++i) {
        RadDeviceProps p{};
        if (b->props(i, &p) < 0) continue;
        s += fmt("%s%d:%s", i ? " " : "", i, p.arch);
    }
    return s;
}

#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
/* With amdgpu.cwsr_enable=0 (compute wave save/restore off) a queue eviction hangs the GPU. The
 * driver evicts a process's queues when card-mapped host memory is moved or reclaimed, and a serve
 * maps gigabytes of it. Without CWSR the driver waits for the queue to drain, a busy queue never
 * does, MES fails REMOVE_QUEUE and the card resets. Warned here as the backend comes up, not only
 * in the startup report: a hang during load kills the process first, and rad-kbench and rad-tune
 * print no report. */
static void warn_cwsr_off() {
    FILE* f = std::fopen("/sys/module/amdgpu/parameters/cwsr_enable", "r");
    if (!f) return;
    int v = -1;
    const int got = std::fscanf(f, "%d", &v);
    std::fclose(f);
    if (got == 1 && v == 0)
        RAD_WARN("device: amdgpu cwsr_enable is 0, so the driver cannot preempt a running queue: "
                 "any queue eviction (host memory the card maps, moved under memory pressure) waits "
                 "for a queue a serve never lets drain, and ends in a GPU hang and reset. Set "
                 "amdgpu.cwsr_enable=1 (module option, then rebuild the initramfs and reboot)");
}

/* The host backend in a HIP build still counts plugin GPU code. This build's fat-binary
 * registration hooks see every dlopen'ed kernel library, card or no card, so the plugin loader can
 * tell which library brought device code. A serving engine drops those device rows from selection
 * (Registry::set_no_device_code); a selection-only tool keeps them. */
static DeviceBackend g_host_beside_hip;
#endif

static const DeviceBackend* select_backend() {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    if (const DeviceBackend* b = aql_backend_if_present()) {
        RAD_INFO("device: AQL backend, %d device(s) [%s]", b->count(),
                 device_inventory(b).c_str());
        warn_cwsr_off();
        return b;
    }
    RAD_INFO("device: built with HIP but no device is visible -- using the host backend");
    g_host_beside_hip = *host_backend();
    g_host_beside_hip.code_mark = aql_code_registered;
    const DeviceBackend* b = &g_host_beside_hip;
#else
    const DeviceBackend* b = host_backend();
#endif
    RAD_INFO("device: host backend, %d device(s) [%s]", b->count(), device_inventory(b).c_str());
    return b;
}

/* Selected exactly once, under std::call_once; read from every rank thread. */
static const DeviceBackend* g_backend = nullptr;
static std::once_flag       g_once;

static inline const DeviceBackend* dev() {
    std::call_once(g_once, [] { g_backend = select_backend(); });
    return g_backend;
}

bool        device_is_host()      { return std::strcmp(dev()->name, "host") == 0; }
const char* device_backend_name() { return dev()->name; }

}  /* namespace rad */

/* ------------------------------------------------------------------ the ledger
 *
 * Counts every device allocation the engine makes. The VRAM budget (core/mem/vram_budget.cpp)
 * spends the card down to --gpu-headroom-mib, so whatever lands outside the two pools and is not
 * derived from the buffer plan or run configuration must be counted here.
 *
 * Not seen: plugins that call hipMalloc directly (libr4d's split-tail workspaces and all-reduce
 * peer scratch). Engine::start reads the card back after allocation for that reason.
 *
 * RADIANCE_TRACE_ALLOC=N logs each allocation of at least N bytes.
 *
 * RADIANCE_DEBUG_POISON_ALLOC=B fills every fresh DEVICE allocation with byte B (0..255), so a
 * read-before-write shows up as garbage instead of lucky zeros (as after a driver reset, "VRAM is
 * lost"). A correct engine serves the same output with it set. RADIANCE_DEBUG_POISON_RANGE=lo:hi
 * limits the fill to device allocations with ordinal in [lo, hi), counted from 1 per allocating
 * thread (the trace's `#n on this thread`), for bisecting. The fill is synchronous on one stream
 * per thread and device, kept until exit: a stream is a hardware queue, and per-allocation queue
 * churn is slow and faults in ROCr's Trim path. Bad values abort at the first allocation. */
namespace rad {
namespace {

struct Ledger {
    std::atomic<int64_t> bytes{0};
    std::atomic<int64_t> count{0};
};
Ledger g_ledger[4];        /* indexed by RAD_MEM_*; rad_dev_alloc bounds the kind */
/* Each live allocation's size, so rad_dev_free (which has none) takes it back off the ledger's
 * bytes: pinned ranges are freed and taken again during bring-up, and the pin-cap check reads the
 * ledger as what the process holds. Allocations are rare, never per step. */
struct LiveAlloc { int64_t bytes; int kind; int owner; int device; };
/* Device bytes each engine holds on each card (dev_owner_bytes): its rad_dev_alloc allocations and
 * its elastic ranges' backing. */
std::mutex                               g_owner_mu;
std::map<std::pair<int, int>, int64_t>   g_owner_bytes;
// FreeToken preserves lifetime peaks across pool frees and backing rebuilds.
std::map<std::pair<int, int>, int64_t>   g_owner_peaks;
void owner_add(int owner, int device, int64_t n) {
    if (owner <= 0 || n == 0) return;
    std::lock_guard<std::mutex> lk(g_owner_mu);
    for (int card : {device, -1}) {
        const auto key = std::make_pair(owner, card);
        const int64_t live = (g_owner_bytes[key] += n);
        g_owner_peaks[key] = std::max(g_owner_peaks[key], live);
    }
}
std::mutex                           g_ledger_mu;
std::unordered_map<void*, LiveAlloc> g_ledger_live;

void ledger_add(void* p, int64_t bytes, int kind, int owner, int device) {
    g_ledger[kind].bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (kind == RAD_MEM_DEVICE) owner_add(owner, device, bytes);
    std::lock_guard<std::mutex> lk(g_ledger_mu);
    g_ledger_live[p] = LiveAlloc{ bytes, kind, owner, device };
}
/* The allocation's bytes, or 0 when the ledger does not know it. */
int64_t ledger_drop(void* p) {
    if (!p) return 0;
    std::lock_guard<std::mutex> lk(g_ledger_mu);
    const auto it = g_ledger_live.find(p);
    if (it == g_ledger_live.end()) return 0;
    const int64_t bytes = it->second.bytes;
    g_ledger[it->second.kind].bytes.fetch_sub(bytes, std::memory_order_relaxed);
    if (it->second.kind == RAD_MEM_DEVICE) owner_add(it->second.owner, it->second.device, -bytes);
    g_ledger_live.erase(it);
    return bytes;
}

/* rad_dev_on_free's hooks and how many registrations each has. Registered at plugin load and
 * read on every device free, which holds the lock shared while it calls them. */
std::shared_mutex                                                 g_free_hooks_mu;
std::vector<std::pair<void (*)(const void*, int64_t), int>>       g_free_hooks;

/* Live event handles, counted here so the number is backend-independent; the only place an
 * event leaked at teardown shows. */
std::atomic<int64_t> g_live_events{0};
/* Host blocks through the ABI (rad_event_sync, rad_stream_sync), counted here so a test can
 * assert that a path never stops the host, whichever backend runs it. */
std::atomic<int64_t> g_host_blocks{0};

int64_t trace_floor() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_TRACE_ALLOC");
        if (!e || !*e) return (int64_t)-1;
        const long long x = std::atoll(e);
        return x >= 0 ? (int64_t)x : (int64_t)0;
    }();
    return v;
}

/* -1 when off; otherwise the byte. */
int poison_byte() {
    static const int v = [] {
        const char* e = std::getenv("RADIANCE_DEBUG_POISON_ALLOC");
        if (!e || !*e) return -1;
        char* end = nullptr;
        const long x = std::strtol(e, &end, 0);
        if (*end || x < 0 || x > 255) {
            RAD_ERR("RADIANCE_DEBUG_POISON_ALLOC=%s is not a byte (0..255)", e);
            std::abort();
        }
        return (int)x;
    }();
    return v;
}

/* [lo, hi) over device-allocation ordinals; the whole run when unset. */
bool poison_in_range(int64_t ordinal) {
    static const std::pair<int64_t, int64_t> r = [] {
        const char* e = std::getenv("RADIANCE_DEBUG_POISON_RANGE");
        long long lo = 0, hi = INT64_MAX;
        if (e && *e && (std::sscanf(e, "%lld:%lld", &lo, &hi) != 2 || lo < 0 || hi <= lo)) {
            RAD_ERR("RADIANCE_DEBUG_POISON_RANGE=%s is not lo:hi with 0 <= lo < hi", e);
            std::abort();
        }
        return std::make_pair((int64_t)lo, (int64_t)hi);
    }();
    return ordinal >= r.first && ordinal < r.second;
}

const char* kind_name(int kind) {
    switch (kind) {
        case RAD_MEM_DEVICE:      return "device";
        case RAD_MEM_HOST:        return "host";
        case RAD_MEM_HOST_PINNED: return "host-pinned";
        case RAD_MEM_HOST_MAPPED: return "host-mapped";
        default:                  return "?";
    }
}

}  /* namespace */

void rad_dev_on_free(void (*fn)(const void* base, int64_t bytes)) {
    if (!fn) return;
    std::unique_lock<std::shared_mutex> lk(g_free_hooks_mu);
    for (auto& h : g_free_hooks)
        if (h.first == fn) { ++h.second; return; }
    g_free_hooks.push_back({ fn, 1 });
}

void rad_dev_off_free(void (*fn)(const void* base, int64_t bytes)) {
    if (!fn) return;
    std::unique_lock<std::shared_mutex> lk(g_free_hooks_mu);
    for (auto it = g_free_hooks.begin(); it != g_free_hooks.end(); ++it)
        if (it->first == fn) {
            if (--it->second == 0) g_free_hooks.erase(it);
            return;
        }
}

void rad_dev_alloc_totals(int kind, int64_t* bytes, int64_t* count) {
    if (kind < 0 || kind >= (int)std::size(g_ledger)) {
        if (bytes) *bytes = 0;
        if (count) *count = 0;
        return;
    }
    if (bytes) *bytes = g_ledger[kind].bytes.load(std::memory_order_relaxed);
    if (count) *count = g_ledger[kind].count.load(std::memory_order_relaxed);
}

int64_t rad_dev_live_events(void) { return g_live_events.load(std::memory_order_relaxed); }
int64_t rad_dev_host_blocks(void) { return g_host_blocks.load(std::memory_order_relaxed); }


/* ------------------------------------------------------------------ elastic memory */
/* See device.h. Interval arithmetic and rounding here; the backends do the mapping. */

namespace {
std::atomic<int64_t> g_vmem_committed{0};

/* Elastic ranges (VMem) reserved on a thread working for an engine are filed under it, so its
 * park (paging.cpp) finds them. */
std::mutex                 g_page_mu;
std::map<VMem*, int>       g_vmem_owner;
}  /* namespace */

static int vmem_owner_of(const VMem* v) {
    std::lock_guard<std::mutex> lk(g_page_mu);
    auto it = g_vmem_owner.find(const_cast<VMem*>(v));
    return it == g_vmem_owner.end() ? 0 : it->second;
}

int64_t rad_dev_vmem_committed(void) { return g_vmem_committed.load(std::memory_order_relaxed); }

/* Both checks: a backend may have the entries on a driver with no virtual memory manager
 * (granularity 0). Checking only the pointers would reserve a range that can never be backed. */
bool    VMem::supported()   { return dev()->vmem_reserve != nullptr && VMem::granularity() > 0; }
int64_t VMem::granularity() {
    if (!dev()->vmem_granularity) return 0;
    const int64_t g = dev()->vmem_granularity();
    return g > 0 ? g : 0;
}

int VMem::reserve(int64_t bytes, int64_t align) {
    release();
    if (bytes <= 0) return RAD_E_INVAL;
    if (!supported()) {
        device_set_error("elastic memory: the %s backend has none", device_backend_name());
        return RAD_E_UNSUPPORTED;
    }
    const int64_t g = granularity();
    if (g <= 0) return RAD_E_UNSUPPORTED;
    if (align <= 0) align = g;
    /* Round up to a granule: a partial last granule could never be committed. */
    const int64_t want = align_up(bytes, g);
    void* p = dev()->vmem_reserve(want, align);
    if (!p) return RAD_E_NOMEM;
    base_ = p;
    size_ = want;
    committed_ = 0;
    ext_.clear();
    device_ = t_device;
    if (const int owner = engine_owner(); owner > 0) {
        std::lock_guard<std::mutex> lk(g_page_mu);
        g_vmem_owner[this] = owner;
    }
    return RAD_OK;
}

void VMem::rekey(VMem* from, VMem* to) {
    std::lock_guard<std::mutex> lk(g_page_mu);
    auto it = g_vmem_owner.find(from);
    if (it == g_vmem_owner.end()) return;
    const int owner = it->second;
    g_vmem_owner.erase(it);
    if (to) g_vmem_owner[to] = owner;
}

void VMem::release() {
    owner_add(vmem_owner_of(this), device_, -committed_);
    rekey(this, nullptr);
    copy_.release();
    parked_.clear();
    if (!base_) { size_ = 0; committed_ = 0; ext_.clear(); return; }
    /* Decommit before release: releasing a range with pages still mapped leaks them, and some
     * drivers refuse the free. An extent is a run of whole mappings (one a granule), so it goes in
     * one unmap. */
    for (const auto& e : ext_) {
        if (dev()->vmem_decommit) (void)dev()->vmem_decommit((char*)base_ + e.first, e.second);
        g_vmem_committed.fetch_sub(e.second, std::memory_order_relaxed);
    }
    if (dev()->vmem_release) dev()->vmem_release(base_, size_);
    base_ = nullptr;
    size_ = 0;
    committed_ = 0;
    ext_.clear();
}

bool VMem::is_committed(int64_t off, int64_t bytes) const {
    if (bytes <= 0) return true;
    if (off < 0 || off + bytes > size_) return false;
    for (const auto& e : ext_)
        if (off >= e.first && off + bytes <= e.first + e.second) return true;
    return false;                       /* extents are merged, so one of them covers it or none does */
}
/* The granules to map are gathered before mapping any: add_extent merges, so walking ext_ while
 * mapping would skip entries and map the tail twice (the driver refuses a double map).
 *
 * Driver rules:
 *   - A mapping can only be unbacked exactly as it was backed, so each granule is mapped
 *     separately; that lets the pool hand back the top of its range.
 *   - Access cannot be granted per granule: a granule adjacent to an accessible one is refused on
 *     its own. Access is granted once per merged extent the new granules belong to; re-granting
 *     access to a range is free. */
namespace {
thread_local int  t_map_batch = 0;
thread_local bool t_map_due = false;
}

MapBatch::MapBatch() { ++t_map_batch; }

MapBatch::~MapBatch() { (void)end(); }

int MapBatch::end() {
    if (ended_) return RAD_OK;
    ended_ = true;
    if (--t_map_batch > 0 || !t_map_due) return RAD_OK;
    t_map_due = false;
    return dev_settle_maps();
}

int VMem::commit(int64_t off, int64_t bytes) {
    if (!base_ || bytes <= 0) return bytes <= 0 ? RAD_OK : RAD_E_INVAL;
    const int64_t g = granularity();
    if (g <= 0) return RAD_E_UNSUPPORTED;
    /* Round outward: every requested byte must be committed. */
    const int64_t lo = (off < 0 ? 0 : off / g * g);
    int64_t hi = align_up(off + bytes, g);
    if (hi > size_) hi = size_;
    if (lo >= hi) return RAD_OK;

    std::vector<int64_t> want;
    want.reserve((size_t)((hi - lo) / g));
    for (int64_t o = lo; o < hi; o += g)
        if (!is_committed(o, g)) want.push_back(o);
    if (want.empty()) return RAD_OK;

    /* On a failure part way, what already mapped stays mapped and recorded: a partial grow is
     * usable, untracked mappings are not. A contiguous run goes in one call; one that fails (the
     * card lacks room for all of it) is backed a granule at a time, so it grows by what fits.
     * `took`: the granules of `want` backed, a prefix of it. */
    int rc = RAD_OK;
    size_t took = 0;
    while (took < want.size() && rc >= 0) {
        size_t j = took + 1;
        while (j < want.size() && want[j] == want[j - 1] + g) ++j;
        const int64_t run = (int64_t)(j - took) * g;
        if (j - took > 1 && dev()->vmem_commit_units &&
            dev()->vmem_commit_units((char*)base_ + want[took], run, g) >= 0) {
            add_extent(want[took], run);
            took = j;
            continue;
        }
        for (; took < j; ++took) {
            rc = dev()->vmem_commit((char*)base_ + want[took], g);
            if (rc < 0) break;
            rc = RAD_OK;   /* recycled memory (1) is fine here: a pool's contents are written before they are read */
            add_extent(want[took], g);
        }
    }

    /* Grant access even if the grow stopped short, so the granules that did map are usable. */
    int prc = RAD_OK;
    if (took > 0 && dev()->vmem_protect) {
        for (const auto& e : ext_) {
            if (e.first + e.second <= lo || e.first >= hi) continue;
            prc = dev()->vmem_protect((char*)base_ + e.first, e.second);
            if (prc < 0) break;
        }
    }
    if (prc < 0) {
        /* Mapped but inaccessible memory faults the first kernel that touches it. Undo exactly
         * what this call mapped; earlier extents were already accessible. */
        for (size_t i = 0; i < took; ++i) {
            if (dev()->vmem_decommit) (void)dev()->vmem_decommit((char*)base_ + want[i], g);
            drop_extent(want[i], g);
        }
        return prc;
    }
    if (took > 0) {
        if (t_map_batch > 0) t_map_due = true;
        else if (const int src = dev_settle_maps(); src < 0) return src;
    }
    return rc;
}

int VMem::decommit(int64_t off, int64_t bytes) {
    if (!base_ || bytes <= 0) return bytes <= 0 ? RAD_OK : RAD_E_INVAL;
    const int64_t g = granularity();
    if (g <= 0) return RAD_E_UNSUPPORTED;
    /* Round inward: a granule holding any byte outside the range stays. */
    const int64_t lo = align_up(off < 0 ? 0 : off, g);
    int64_t hi = (off + bytes) / g * g;
    if (hi > size_) hi = size_;
    if (lo >= hi) return RAD_OK;

    /* Gathered first; each contiguous run of granules (a mapping each) goes in one unmap. */
    std::vector<int64_t> drop;
    for (int64_t o = lo; o < hi; o += g)
        if (is_committed(o, g)) drop.push_back(o);

    for (size_t i = 0; i < drop.size();) {
        size_t j = i + 1;
        while (j < drop.size() && drop[j] == drop[j - 1] + g) ++j;
        const int64_t run = (int64_t)(j - i) * g;
        if (const int rc = dev()->vmem_decommit((char*)base_ + drop[i], run); rc < 0) return rc;
        drop_extent(drop[i], run);
        i = j;
    }
    return RAD_OK;
}

int64_t dev_owner_bytes(int owner, int device) {
    std::lock_guard<std::mutex> lk(g_owner_mu);
    auto it = g_owner_bytes.find({ owner, device });
    return it == g_owner_bytes.end() ? 0 : it->second;
}

int64_t dev_owner_high_water(int owner, int device) {
    std::lock_guard<std::mutex> lk(g_owner_mu);
    const auto it = g_owner_peaks.find({ owner, device });
    return it == g_owner_peaks.end() ? 0 : it->second;
}

void VMem::add_extent(int64_t off, int64_t bytes) {
    owner_add(vmem_owner_of(this), device_, bytes);
    committed_ += bytes;
    g_vmem_committed.fetch_add(bytes, std::memory_order_relaxed);
    /* Insert in order, then merge touching neighbours, so is_committed() stays a short scan
     * although pools grow one granule at a time. */
    size_t i = 0;
    while (i < ext_.size() && ext_[i].first < off) ++i;
    ext_.insert(ext_.begin() + (long)i, { off, bytes });
    for (size_t j = 0; j + 1 < ext_.size();) {
        if (ext_[j].first + ext_[j].second >= ext_[j + 1].first) {
            const int64_t end = std::max(ext_[j].first + ext_[j].second,
                                         ext_[j + 1].first + ext_[j + 1].second);
            ext_[j].second = end - ext_[j].first;
            ext_.erase(ext_.begin() + (long)j + 1);
        } else {
            ++j;
        }
    }
}

int64_t VMem::page_out(RadStream s, bool discard) {
    if (!base_ || ext_.empty() || !parked_.empty()) return 0;
    int64_t total = 0;
    for (const auto& e : ext_) total += e.second;
    if (discard) {
        /* Nothing in the range is kept: back it again on page_in with whatever the pages hold. */
        parked_ = ext_;
        discarded_ = true;
        for (const auto& e : parked_)
            if (const int rc = decommit(e.first, e.second); rc < 0) return rc;
        return 0;
    }
    discarded_ = false;
    if (const int rc = copy_.alloc(total); rc < 0) return rc;
    int64_t at = 0;
    for (const auto& e : ext_) {
        if (const int rc = rad_memcpy_async((char*)copy_.host + at, (char*)base_ + e.first, e.second, s); rc < 0)
            return rc;
        at += e.second;
    }
    if (const int rc = rad_stream_sync(s); rc < 0) return rc;
    parked_ = ext_;
    for (const auto& e : parked_)
        if (const int rc = decommit(e.first, e.second); rc < 0) return rc;
    return total;
}

int dev_current() { return t_device; }

int64_t dev_vmem_trim(int device) {
    return dev()->vmem_trim ? dev()->vmem_trim(device) : 0;
}

void dev_stream_counts(int device, int* open, int* idle) {
    if (open) *open = 0;
    if (idle) *idle = 0;
    if (dev()->stream_counts) dev()->stream_counts(device, open, idle);
}

int dev_stream_trim(int device, int keep) {
    return dev()->stream_trim ? dev()->stream_trim(device, keep) : 0;
}

std::string dev_page_where(uintptr_t a);

std::string dev_fault_where(const void* p) {
    const uintptr_t a = (uintptr_t)p;
    std::string s = dev_page_where(a);
    std::unique_lock<std::mutex> lk(g_page_mu, std::try_to_lock);
    if (!lk.owns_lock()) return s;
    for (const auto& [v, owner] : g_vmem_owner) {
        const uintptr_t b = (uintptr_t)v->base();
        if (!b || a < b || a >= b + (uintptr_t)v->size()) continue;
        const int64_t off = (int64_t)(a - b), g = std::max<int64_t>(VMem::granularity(), 1);
        return (s.empty() ? s : s + "; ") + fmt("elastic reservation of owner %d card %d, %s long, %s backed, +%s %s", owner, v->device(),
                   humanb(v->size()).c_str(), humanb(v->committed()).c_str(), humanb(off).c_str(),
                   v->is_committed(off / g * g, g) ? "is backed" : "is NOT backed");
    }
    return s;
}

int dev_settle_maps() {
    if (device_is_host()) return RAD_OK;
    void* x = dev()->alloc(4096, RAD_MEM_HOST_PINNED);
    if (!x) {
        device_set_error("vmem: no page of pinned host memory to flush the cards' translations with");
        return RAD_E_NOMEM;
    }
    dev()->free(x, RAD_MEM_HOST_PINNED);
    return RAD_OK;
}

int VMem::page_in(RadStream s) {
    if (parked_.empty()) return RAD_OK;
    MapBatch maps;
    if (discarded_) {
        for (const auto& e : parked_)
            if (const int rc = commit(e.first, e.second); rc < 0) return rc;
        if (const int rc = maps.end(); rc < 0) return rc;
        parked_.clear();
        discarded_ = false;
        return RAD_OK;
    }
    if (const int rc = copy_.load(); rc < 0) return rc;
    for (const auto& e : parked_)
        if (const int rc = commit(e.first, e.second); rc < 0) return rc;
    if (const int rc = maps.end(); rc < 0) return rc;
    int64_t at = 0;
    for (const auto& e : parked_) {
        if (const int rc = rad_memcpy_async((char*)base_ + e.first, (char*)copy_.host + at, e.second, s); rc < 0)
            return rc;
        at += e.second;
    }
    if (const int rc = rad_stream_sync(s); rc < 0) return rc;
    parked_.clear();
    copy_.release();
    return RAD_OK;
}

/* ---- HostCopy */

namespace {
std::atomic<uint64_t> g_copy_seq{0};
/* The unit of a copy's file I/O: what one write asks the gate about, what one reader takes. */
constexpr int64_t kCopyPiece = (int64_t)64 << 20;
}

int HostCopy::alloc(int64_t n) {
    release();
    bytes = n;
    host = rad_dev_alloc(align_up(n, DirectFile::alignment()), RAD_MEM_HOST_PINNED);
    if (!host) {
        device_set_error("parking: no pinned host memory for %s", humanb(n).c_str());
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

int HostCopy::spill(const std::string& dir, const std::function<int()>& gate) {
    if (on_disk || !host || bytes <= 0) return RAD_OK;
    /* Not the page files' name: their counter is another one (paging.cpp). */
    const std::string path = dir + "/radiance-vmem-" + std::to_string((long long)getpid()) + "-" +
                             std::to_string((unsigned long long)g_copy_seq.fetch_add(1)) + ".bin";
    DirectFile f;
    if (const int rc = f.open(path.c_str(), RAD_DIO_WRITE | RAD_DIO_CREATE | RAD_DIO_TRUNC); rc < 0)
        return rc;
    const int64_t n = align_up(bytes, DirectFile::alignment());
    int rc = RAD_OK;
    for (int64_t at = 0; at < n; at += kCopyPiece) {
        int g = gate ? gate() : 0;
        while (g == 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            g = gate();
        }
        if (g < 0) { rc = RAD_E_STATE; break; }
        if ((rc = f.write_at((const char*)host + at, at, std::min(kCopyPiece, n - at))) < 0) break;
    }
    f.close();
    if (rc < 0) {
        ::unlink(path.c_str());
        return rc;
    }
    file = path;
    on_disk = true;
    return RAD_OK;
}

void HostCopy::drop_host() {
    if (!on_disk || !host) return;
    rad_dev_free(host, RAD_MEM_HOST_PINNED);
    host = nullptr;
}

int HostCopy::load() {
    if (host) return RAD_OK;
    if (!on_disk) {
        device_set_error("parking: a copy has neither host memory nor a file");
        return RAD_E_STATE;
    }
    const int64_t n = align_up(bytes, DirectFile::alignment());
    host = rad_dev_alloc(n, RAD_MEM_HOST_PINNED);
    if (!host) {
        device_set_error("parking: no pinned host memory to read %s back", humanb(bytes).c_str());
        return RAD_E_NOMEM;
    }
    /* A read that fails leaves no host memory behind, so the next load() reads again instead of
     * taking a partial copy as the copy. */
    auto fail = [&](int rc) {
        rad_dev_free(host, RAD_MEM_HOST_PINNED);
        host = nullptr;
        return rc;
    };
    DirectFile f;
    if (const int rc = f.open(file.c_str(), RAD_DIO_READ); rc < 0) return fail(rc);
    /* Four readers over the file: one stream of O_DIRECT reads leaves an NVMe drive half idle. */
    std::atomic<int64_t> next{0};
    std::atomic<int> err{RAD_OK};
    auto reader = [&] {
        DirectFile g;
        if (g.open(file.c_str(), RAD_DIO_READ) < 0) { err.store(RAD_E_IO); return; }
        for (;;) {
            const int64_t at = next.fetch_add(kCopyPiece);
            if (at >= n || err.load() < 0) return;
            if (g.read_at((char*)host + at, at, std::min(kCopyPiece, n - at)) < 0) { err.store(RAD_E_IO); return; }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < 4; ++i) th.emplace_back(reader);
    for (auto& t : th) t.join();
    if (err.load() < 0) {
        device_set_error("parking: reading %s back failed", file.c_str());
        return fail(err.load());
    }
    return RAD_OK;
}

void HostCopy::release() {
    if (host) rad_dev_free(host, RAD_MEM_HOST_PINNED);
    host = nullptr;
    if (!file.empty()) ::unlink(file.c_str());
    file.clear();
    on_disk = false;
    bytes = 0;
}

/* Precondition: the piece lies wholly inside one extent (decommit only drops committed granules).
 * What remains of that extent is nothing, a head, a tail, or both. */
void VMem::drop_extent(int64_t off, int64_t bytes) {
    owner_add(vmem_owner_of(this), device_, -bytes);
    committed_ -= bytes;
    g_vmem_committed.fetch_sub(bytes, std::memory_order_relaxed);
    for (size_t i = 0; i < ext_.size(); ++i) {
        const int64_t es = ext_[i].first, ee = es + ext_[i].second;
        if (off < es || off + bytes > ee) continue;
        ext_.erase(ext_.begin() + (long)i);
        if (off + bytes < ee) ext_.insert(ext_.begin() + (long)i, { off + bytes, ee - off - bytes });
        if (es < off)         ext_.insert(ext_.begin() + (long)i, { es, off - es });
        return;
    }
}

const DeviceBackend* active_backend() { return dev(); }

std::vector<VMem*> vmems_of(int owner) {
    std::vector<VMem*> v;
    std::lock_guard<std::mutex> lk(g_page_mu);
    for (auto& [m, o] : g_vmem_owner)
        if (o == owner) v.push_back(m);
    return v;
}

}  /* namespace rad */

/* ------------------------------------------------------------------ the ABI surface */

using rad::dev;

extern "C" {

int  rad_dev_count(void)                          { return dev()->count(); }
/* This thread's stream for the poison fill (rad_dev_alloc), one per device. */
static thread_local std::map<int, RadStream> t_poison_stream;
int  rad_dev_set(int device) {
    const int st = dev()->set(device);
    if (st == RAD_OK) t_device = device;
    return st;
}
int  rad_dev_props(int device, RadDeviceProps* o)  { return dev()->props(device, o); }
int  rad_dev_enable_peer(int self, int peer)      { return dev()->enable_peer(self, peer); }

void* rad_dev_alloc(int64_t bytes, int kind) {
    /* Poison ordinals are per thread: each rank allocates in a fixed order on its own thread, but
     * threads interleave, so a process-wide ordinal would not be repeatable. */
    thread_local int64_t t_device_ordinal = 0;
    void* p = nullptr;
    const int owner = rad::engine_owner();
    /* A large device allocation for an engine is mapped in chunks it can be parked by
     * (paging.cpp). Where the backend has no virtual memory manager it is an ordinary allocation,
     * and the model parks only by unloading. */
    if (kind == RAD_MEM_DEVICE && bytes >= rad::kPageMin && owner > 0 && rad::VMem::supported())
        p = rad::page_alloc(bytes, owner, t_device);
    if (!p) p = dev()->alloc(bytes, kind);
    if (p && kind >= 0 && kind < (int)std::size(rad::g_ledger)) {
        rad::ledger_add(p, bytes, kind, owner, t_device);
        const int64_t total = rad::g_ledger[kind].count.fetch_add(1, std::memory_order_relaxed) + 1;
        const int64_t ordinal = kind == RAD_MEM_DEVICE ? ++t_device_ordinal : total;
        const int64_t floor_ = rad::trace_floor();
        if (floor_ >= 0 && bytes >= floor_)
            RAD_INFO("alloc %s %s (#%lld on this thread; %lld live in %lld allocations)",
                     rad::kind_name(kind), rad::humanb(bytes).c_str(), (long long)ordinal,
                     (long long)rad::g_ledger[kind].bytes.load(std::memory_order_relaxed),
                     (long long)total);
        /* Poison fill (see the ledger header): synchronous, so it lands before p is returned. */
        const int poison = rad::poison_byte();
        if (poison >= 0 && kind == RAD_MEM_DEVICE && rad::poison_in_range(ordinal)) {
            RadStream& s = t_poison_stream[t_device];
            if (!s && dev()->stream_create(&s, 0) != RAD_OK) s = nullptr;
            if (!s || dev()->memset_async(p, poison, bytes, s) != RAD_OK ||
                dev()->stream_sync(s) != RAD_OK)
                RAD_WARN("poison: device allocation #%lld (%s) was not filled",
                         (long long)ordinal, rad::humanb(bytes).c_str());
            else if (floor_ >= 0 && bytes >= floor_)
                RAD_INFO("poison: device allocation #%lld filled with 0x%02x", (long long)ordinal,
                         poison);
        }
    }
    return p;
}
void  rad_dev_free(void* p, int kind) {
    const int64_t bytes = rad::ledger_drop(p);
    if (!(p && kind == RAD_MEM_DEVICE && rad::page_free(p))) dev()->free(p, kind);
    /* Kernel libraries drop state they keep for addresses in the freed range (rad_plugin_freed). */
    if (p && bytes > 0 && kind == RAD_MEM_DEVICE) {
        std::shared_lock<std::shared_mutex> lk(rad::g_free_hooks_mu);
        for (const auto& h : rad::g_free_hooks) h.first(p, bytes);
    }
}
void* rad_dev_host_ptr(void* p)                   { return dev()->host_ptr(p); }
void* rad_dev_device_ptr(void* p)                 { return dev()->device_ptr(p); }

int  rad_stream_create(RadStream* out, int hi)    { return dev()->stream_create(out, hi); }
void rad_stream_destroy(RadStream s)              { dev()->stream_destroy(s); }
int  rad_stream_sync(RadStream s) {
    rad::g_host_blocks.fetch_add(1, std::memory_order_relaxed);
    return dev()->stream_sync(s);
}

}  /* extern "C" */

namespace rad {

bool    tape_supported()       { return dev()->tape_begin != nullptr; }
int     dev_vram_used(int device, int64_t* out) {
    return dev()->vram_used ? dev()->vram_used(device, out) : RAD_E_UNSUPPORTED;
}
int     tape_begin()           { return dev()->tape_begin ? dev()->tape_begin() : RAD_E_UNSUPPORTED; }
int     tape_mark()            { return dev()->tape_mark ? dev()->tape_mark() : RAD_E_UNSUPPORTED; }
void    tape_pause(bool on)    { if (dev()->tape_pause) dev()->tape_pause(on ? 1 : 0); }
RadTape tape_end(bool keep)    { return dev()->tape_end ? dev()->tape_end(keep ? 1 : 0) : nullptr; }
int     tape_len(RadTape t)    { return dev()->tape_len ? dev()->tape_len(t) : RAD_E_UNSUPPORTED; }
int     tape_play(RadTape t, int from, int to) {
    return dev()->tape_play ? dev()->tape_play(t, from, to) : RAD_E_UNSUPPORTED;
}
int     tape_diff(RadTape a, RadTape b, std::string* why) {
    if (!dev()->tape_diff) return RAD_E_UNSUPPORTED;
    char buf[512];
    buf[0] = 0;
    const int at = dev()->tape_diff(a, b, buf, (int)sizeof buf);
    if (why) *why = buf;
    return at;
}
void    tape_destroy(RadTape t) { if (dev()->tape_destroy) dev()->tape_destroy(t); }
int     tape_record(RadTape t, int i, int* kind, uint32_t* queue) {
    return dev()->tape_record ? dev()->tape_record(t, i, kind, queue) : RAD_E_UNSUPPORTED;
}
int     tape_set_overlap(RadTape t, int i, bool on) {
    return dev()->tape_set_overlap ? dev()->tape_set_overlap(t, i, on ? 1 : 0) : RAD_E_UNSUPPORTED;
}
bool    dev_integrated(int device) { return dev()->integrated && dev()->integrated(device) > 0; }
/* These getters answer "" on failure and restore the error slot, so the next
 * rad_dev_last_error reader does not see an unrelated error. `get` fills a buffer of `n` bytes. */
template <class Get> static std::string dev_string(int n, Get get) {
    std::string buf((size_t)n, '\0');
    const std::string before = t_last_error;
    if (get(buf.data(), n) < 0) {
        t_last_error = before;
        return std::string();
    }
    buf.resize(std::strlen(buf.c_str()));
    return buf;
}
std::string dev_pci_address(int device) {
    if (!dev()->pci_address) return std::string();
    return dev_string(64, [&](char* b, int n) { return dev()->pci_address(device, b, n); });
}
std::atomic<int64_t> dev_stuck_exit_ms{60000};
int64_t dev_stuck_ms() {
    const auto f = dev()->stuck_ms;
    return f ? f() : 0;
}
std::string dev_dma_lane(int device) {
    if (!dev()->dma_lane) return std::string();
    return dev_string(256, [&](char* b, int n) { return dev()->dma_lane(device, b, n); });
}
std::string dev_runtime_version() {
    if (!dev()->runtime_version) return std::string();
    return dev_string(128, [&](char* b, int n) { return dev()->runtime_version(b, n); });
}
int     dev_peer_access(int self, int peer) {
    if (self == peer) return 1;
    return dev()->peer_access ? dev()->peer_access(self, peer) : 1;
}
size_t  dev_code_mark() { return dev()->code_mark ? dev()->code_mark() : 0; }
int     dev_code_runs(size_t from, size_t to, int device, std::string* why) {
    if (!dev()->code_runs || from >= to) return 1;
    char buf[1024];
    buf[0] = 0;
    const int r = dev()->code_runs(from, to, device, buf, (int)sizeof buf);
    if (why) *why = buf;
    return r;
}

}  /* namespace rad */

extern "C" {

int  rad_dev_peer_access(int self, int peer)      { return rad::dev_peer_access(self, peer); }

int  rad_event_create_as(RadEvent* out, unsigned flags) {
    const int s = dev()->event_create(out, flags);
    if (s >= 0 && out && *out) rad::g_live_events.fetch_add(1, std::memory_order_relaxed);
    return s;
}
int  rad_event_create(RadEvent* out)              { return rad_event_create_as(out, RAD_EVENT_HOST_WAIT); }
int  rad_event_create_local(RadEvent* out)        { return rad_event_create_as(out, RAD_EVENT_LOCAL); }
void rad_event_destroy(RadEvent e) {
    if (!e) return;
    rad::g_live_events.fetch_sub(1, std::memory_order_relaxed);
    dev()->event_destroy(e);
}
int  rad_event_record(RadEvent e, RadStream s)    { return dev()->event_record(e, s); }
int  rad_event_dma(RadEvent e, RadStream s, uint32_t** w) {
    if (const auto f = dev()->event_dma) return f(e, s, w);
    rad::device_set_error("rad_event_dma: the device backend has no DMA queue to complete an event");
    return RAD_E_UNSUPPORTED;
}
int  rad_event_wait(RadStream s, RadEvent e)      { return dev()->event_wait(s, e); }
int  rad_event_query(RadEvent e)                  { return dev()->event_query(e); }
int  rad_event_sync(RadEvent e) {
    rad::g_host_blocks.fetch_add(1, std::memory_order_relaxed);
    return dev()->event_sync(e);
}
void rad_dev_host_wrote(void)                     { dev()->host_wrote(); }

/* Whether launches chain, by engine: rad_dev_chain_infer on a thread of an engine (EngineScope)
 * records that engine's answer and rad_dev_chain_enabled on any of its threads reads it. A process
 * serves several models, each with its own plugin's answer; a single answer for the process was
 * the answer of whichever model started last. Owner 0 is a thread outside every engine (a tool, a
 * test). Only where the backend chains: the AQL backend writes the record a chained form reads,
 * while a HIP launch carries only the kernel's own arguments. */
static std::mutex              g_chain_mu;
static std::map<int, int>      g_chain_pays;    /* by owner */
static std::atomic<uint64_t>   g_chain_gen{1};
void rad_dev_chain_infer(int pays) {
    {
        std::lock_guard<std::mutex> lk(g_chain_mu);
        g_chain_pays[rad::engine_owner()] = pays ? 1 : 0;
    }
    g_chain_gen.fetch_add(1, std::memory_order_release);
}
int rad_dev_chain_enabled(void) {
    /* Asked on every launch: a thread keeps its engine's answer until some answer changes. */
    thread_local uint64_t seen = 0;
    thread_local int      owner = -1, on = 0;
    const uint64_t gen = g_chain_gen.load(std::memory_order_acquire);
    const int      o = rad::engine_owner();
    if (seen != gen || owner != o) {
        int pays = 0;
        {
            std::lock_guard<std::mutex> lk(g_chain_mu);
            const auto it = g_chain_pays.find(o);
            if (it != g_chain_pays.end()) pays = it->second;
        }
        on = pays && std::strcmp(rad::device_backend_name(), "aql") == 0 ? 1 : 0;
        seen = gen;
        owner = o;
    }
    return on;
}

int  rad_event_elapsed_ms(RadEvent a, RadEvent b, float* out) {
    return dev()->event_elapsed_ms(a, b, out);
}

int rad_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream s) {
    /* One address space (unified memory, or a host view of a device allocation): a copy
     * onto itself is already done. */
    if (dst == src) return RAD_OK;
    return dev()->memcpy_async(dst, src, bytes, s);
}
int rad_memcpy_2d_async(void* dst, int64_t dpitch, const void* src, int64_t spitch,
                        int64_t width, int64_t height, RadStream s) {
    if (dst == src && dpitch == spitch) return RAD_OK;
    return dev()->memcpy_2d_async(dst, dpitch, src, spitch, width, height, s);
}
int rad_memset_async(void* dst, int value, int64_t bytes, RadStream s) {
    return dev()->memset_async(dst, value, bytes, s);
}
int rad_memcpy_ranges_async(void* dst, const void* src, int64_t prefix,
                            const RadCopyRange* ranges, int n, RadStream s) {
    return dev()->memcpy_ranges_async(dst, src, prefix, ranges, n, s);
}
int rad_memcpy_list_async(const RadCopySeg* segs, int n, unsigned flags, RadStream s) {
    if (const auto f = dev()->memcpy_list_async) return f(segs, n, flags, s);
    for (int i = 0; i < n; ++i) RAD_TRY(dev()->memcpy_async(segs[i].dst, segs[i].src, segs[i].bytes, s));
    return RAD_OK;
}
int rad_link_hold(RadStream s, int on) {
    if (const auto f = dev()->link_hold) return f(s, on);
    return RAD_OK;
}
void* rad_link_gate(RadStream s) {
    if (const auto f = dev()->link_gate) return f(s);
    return nullptr;
}
void rad_dev_peer_launch(int on) {
    if (const auto f = dev()->peer_launch) f(on);
}
int rad_dev_cards_claim(const int* cards, int n, int on) {
    if (!cards || n <= 0) return RAD_E_INVAL;
    if (const auto f = dev()->cards_claim) return f(cards, n, on);
    return RAD_OK;
}
int rad_xp_supported(int device) {
    const auto* d = dev();
    return d->xp_supported && d->xp_stream_create && d->ride && d->ride_wait && d->ride_release && d->xp_list &&
           d->xp_behind && d->alloc_uncached && d->free_uncached && d->xp_supported(device);
}
int rad_xp_stream_create(RadStream* out) {
    if (const auto f = dev()->xp_stream_create) return f(out);
    return RAD_E_UNSUPPORTED;
}
int rad_ride(RadStream s, void** token) {
    if (const auto f = dev()->ride) return f(s, token);
    return RAD_E_UNSUPPORTED;
}
int rad_ride_wait(RadStream s, void* token) {
    if (const auto f = dev()->ride_wait) return f(s, token);
    return RAD_E_UNSUPPORTED;
}
void rad_ride_release(void* token, RadStream after, int64_t packets) {
    if (const auto f = dev()->ride_release) f(token, after, packets);
}
int rad_xp_list(int device, void* mark, int n, const uint64_t* src, const uint64_t* dst, const uint64_t* bytes,
                uint32_t* done, uint32_t done_val, unsigned flags) {
    if (const auto f = dev()->xp_list) return f(device, mark, n, src, dst, bytes, done, done_val, flags);
    return RAD_E_UNSUPPORTED;
}
int64_t rad_xp_behind(int device) {
    if (const auto f = dev()->xp_behind) return f(device);
    return 0;
}
void* rad_dev_alloc_uncached(int64_t bytes) {
    if (const auto f = dev()->alloc_uncached) return f(bytes);
    return nullptr;
}
void* rad_dev_map_file(int fd, int64_t foff, int64_t bytes) {
    const auto f = dev()->map_file;
    if (!f) {
        rad::device_set_error("rad_dev_map_file: the %s backend has no pinned host memory to map a "
                              "file into", dev()->name);
        return nullptr;
    }
    void* p = f(fd, foff, bytes);
    if (p) {
        rad::ledger_add(p, bytes, RAD_MEM_HOST_MAPPED, 0, t_device);
        rad::g_ledger[RAD_MEM_HOST_MAPPED].count.fetch_add(1, std::memory_order_relaxed);
    }
    return p;
}
void rad_dev_free_uncached(void* p) {
    if (const auto f = dev()->free_uncached) f(p);
}
int rad_stream_gate(RadStream s, void** gate) {
    if (gate) *gate = nullptr;
    if (const auto f = dev()->stream_gate) return f(s, gate);
    rad::device_set_error("rad_stream_gate: the %s backend has no gates", dev()->name);
    return RAD_E_UNSUPPORTED;
}
int rad_gate_open(void* gate) {
    if (const auto f = dev()->gate_open) return f(gate);
    return RAD_E_UNSUPPORTED;
}
int64_t rad_gate_room(RadStream s) {
    if (const auto f = dev()->gate_room) return f(s);
    return 0;
}

/* Cleared by reading. Never null; "" when nothing failed. */
const char* rad_dev_last_error(void) {
    static thread_local std::string drained;
    drained.swap(rad::t_last_error);
    rad::t_last_error.clear();
    return drained.c_str();
}

}  /* extern "C" */
