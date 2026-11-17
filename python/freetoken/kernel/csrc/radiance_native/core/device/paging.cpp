/* paging.cpp -- parking an engine's memory in chunks (device.h, "parking"; docs/MULTIMODEL.md §7).
 *
 * An engine's large device allocation is one reserved address range mapped in kChunk pieces, each
 * its own mapping, so a piece can leave the card and come back alone at the same address. A model
 * that has to make room gives up exactly what is asked of it, chunks with nothing worth keeping
 * first, then chunks whose copy already exists (a sealed weights pool's), then chunks whose copy is
 * stale in a few blocks, then any other; what is not asked for stays on the card, and a restore
 * moves back only what left.
 *
 * A chunk out of the card is held in host memory while the host budget allows, else written
 * straight to a file of the disk tier through a small pinned staging ring. An allocation its owner
 * refills (dev_page_refilled: a weights pool its container rebuilds) is never copied: its sealed
 * chunks leave with an unmap, and a page-in maps them and hands them back to the owner to fill. Files are one per
 * allocation, a chunk at its own offset (sparse), O_DIRECT both ways so the page cache neither
 * doubles them nor evicts the containers' pages for them.
 *
 * A host copy is itself a mapping (vmem_commit_host): every allocation has a shadow reservation
 * of its size, and a chunk's host copy is mapped at the shadow while the chunk is on the card and
 * at the chunk's own address while it is out, so the card can still read and write a parked
 * chunk in place, over the link. A park is a copy to the shadow and a swap of the two mappings; a
 * restore the reverse; a sealed chunk whose copy is kept leaves with the swap alone.
 *
 * Locking: g_mu guards the map of pages and every chunk's fields. Copies and file I/O run without
 * it on chunks of one owner, which only that owner's park or restore touches (the fleet runs at
 * most one at a time per model, and the engine is quiet).
 */
#include "device.h"
#include "directio.h"
#include "../rad_internal.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rad {

namespace {

constexpr int64_t kChunk = (int64_t)64 << 20;
/* Dirty tracking within a chunk: the expert mover rewrites one slot (about a megabyte) at a time,
 * and a park after a few hundred moves would otherwise copy the whole slab again. */
constexpr int64_t kSub = (int64_t)64 << 10;
constexpr int     kSubWords = (int)(kChunk / kSub / 64);

/* Where a chunk's host copy is mapped. */
enum : uint8_t { kNoHost = 0, kAtShadow = 1, kAtBase = 2 };

struct Chunk {
    uint8_t hloc = kNoHost;   /* its host copy: at the shadow (the chunk on the card) or at the
                               * chunk's own address (out, read in place) */
    bool    sealed = false;   /* kept copies stay exact until dev_page_invalidate */
    uint32_t gen = 0;         /* bumped by every dev_page_invalidate: a copy made across one is stale */
    bool    on_disk = false;  /* the page file holds this chunk at off */
    bool    src = false;      /* the load copied container bytes onto part of it (dev_page_source) */
    bool    backed = false;   /* checked equal to its sources (zero elsewhere): the container is
                               * its copy */
    bool    nofile = false;   /* written since the load: its sources no longer describe it */
    bool    out = false;      /* not in device memory: unmapped, or its host copy mapped there */
    bool    discard = false;  /* nothing to keep (dev_page_discard), for the next page-out */
    bool    dropped = false;  /* out with no copy: comes back with whatever the pages hold */
    bool    refill = false;   /* out with no copy, and the owner refills it after the page-in */
    /* kSub blocks written since the host or disk copy was made; the copy is exact elsewhere. A
     * park or a backup rewrites only these into it. */
    uint64_t dirty[kSubWords] = {};

    bool any_dirty() const {
        for (uint64_t w : dirty) if (w) return true;
        return false;
    }
    int64_t dirty_bytes() const {
        int64_t n = 0;
        for (uint64_t w : dirty) n += __builtin_popcountll(w);
        return n * kSub;
    }
    void clear_dirty() { std::memset(dirty, 0, sizeof dirty); }
    /* Marks the blocks [a, b) touches, offsets within the chunk. */
    void mark(int64_t a, int64_t b) {
        for (int64_t k = a / kSub; k < (b + kSub - 1) / kSub && k < kSubWords * 64; ++k)
            dirty[k / 64] |= (uint64_t)1 << (k % 64);
    }
};

/* The dirty stretches of a chunk of `len` bytes, as (offset, bytes), adjacent blocks merged. */
std::vector<std::pair<int64_t, int64_t>> dirty_runs(const Chunk& c, int64_t len) {
    std::vector<std::pair<int64_t, int64_t>> out;
    for (int64_t k = 0; k < kSubWords * 64;) {
        if (!(c.dirty[k / 64] >> (k % 64) & 1)) { ++k; continue; }
        int64_t e = k + 1;
        while (e < kSubWords * 64 && (c.dirty[e / 64] >> (e % 64) & 1)) ++e;
        const int64_t a = k * kSub, b = std::min(e * kSub, len);
        if (b > a) out.push_back({ a, b - a });
        k = e;
    }
    return out;
}

/* Rows [0, ny) of wb bytes at `off` in the allocation came from container `fid` at foff + y * sp. */
struct Src {
    int64_t off = 0;
    int     fid = 0;
    int64_t foff = 0, wb = 0, ny = 1, sp = 0;
    int64_t end() const { return off + wb * ny; }
};

/* A container range that lands at `d0` within a chunk. */
struct Piece { int fid; int64_t f0, n, d0; };

struct Page {
    int64_t span = 0;
    int     owner = 0, device = 0;
    char*   shadow = nullptr;   /* where host copies of chunks on the card are mapped; null: none */
    std::string file;
    std::vector<Chunk> ch;
    std::vector<Src> src;
    bool    src_sorted = true;
    bool    refill = false;     /* its owner rebuilds its sealed chunks (dev_page_refilled) */

    int64_t off(size_t i) const { return (int64_t)i * kChunk; }
    int64_t len(size_t i) const { return std::min(kChunk, span - off(i)); }
    /* An exact copy exists: the container, or a host or disk copy with nothing written since. */
    bool has_copy(size_t i) const {
        return ch[i].backed || ((ch[i].hloc || ch[i].on_disk) && !ch[i].any_dirty());
    }
    /* A host or disk copy that is exact but for some dirty blocks. */
    bool patchable(size_t i) const {
        return ch[i].sealed && !ch[i].backed && (ch[i].hloc || ch[i].on_disk) && ch[i].any_dirty();
    }
    /* Worth checking against the container instead of copying out. */
    bool checkable(size_t i) const { return ch[i].src && !ch[i].nofile; }
};

std::mutex                 g_mu;
std::map<char*, Page>      g_pages;
std::atomic<uint64_t>      g_file_seq{0};
/* Containers named by dev_page_file, opened for reading on first use. */
std::vector<std::string>                  g_files;
std::vector<std::unique_ptr<DirectFile>>  g_dio;

const DeviceBackend* be() { return active_backend(); }

/* One copy stream per card, created on first use. on() also selects that card on the calling
 * thread; nothing restores the previous selection. */
struct Streams {
    std::map<int, RadStream> s;
    RadStream on(int device) {
        rad_dev_set(device);
        RadStream& x = s[device];
        if (!x && rad_stream_create(&x, 0) < 0) x = nullptr;
        return x;
    }
    int sync_all() {
        for (auto& [d, x] : s)
            if (x) {
                rad_dev_set(d);
                if (const int rc = rad_stream_sync(x); rc < 0) return rc;
            }
        return RAD_OK;
    }
    ~Streams() {
        for (auto& [d, x] : s)
            if (x) { rad_dev_set(d); rad_stream_destroy(x); }
    }
};

/* Pinned staging buffers of kChunk, reused across calls. */
std::mutex          g_stage_mu;
std::vector<void*>  g_stage_free;
void* stage_take() {
    {
        std::lock_guard<std::mutex> lk(g_stage_mu);
        if (!g_stage_free.empty()) { void* p = g_stage_free.back(); g_stage_free.pop_back(); return p; }
    }
    return rad_dev_alloc(kChunk, RAD_MEM_HOST_PINNED);
}
void stage_give(void* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_stage_mu);
    if (g_stage_free.size() < 16) g_stage_free.push_back(p);
    else rad_dev_free(p, RAD_MEM_HOST_PINNED);
}

/* Asks `gate` (when given) until it stops answering 1 (wait), sleeping `ms` between asks: 0 goes
 * on, a negative answer abandons the work. */
int gate_wait(const std::function<int()>& gate, int ms) {
    int g = gate ? gate() : 0;
    while (g == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        g = gate();
    }
    return g;
}

std::string new_file(const std::string& dir) {
    return dir + "/radiance-park-" + std::to_string((long long)getpid()) + "-" +
           std::to_string((unsigned long long)g_file_seq.fetch_add(1)) + ".bin";
}

int file_io(const std::string& path, bool write, void* buf, int64_t off, int64_t n) {
    DirectFile f;
    const int fl = write ? (RAD_DIO_WRITE | RAD_DIO_CREATE) : RAD_DIO_READ;
    if (const int rc = f.open(path.c_str(), fl); rc < 0) return rc;
    const int64_t a = align_up(n, DirectFile::alignment());
    return write ? f.write_at(buf, off, a) : f.read_at(buf, off, a);
}

/* Device memory for chunk i at `at` (its own address, or its shadow's). 1: recycled memory that
 * holds what its last mapping left (vmem_commit), for a caller that writes only part of it. */
int map_dev(Page& pg, char* at, size_t i) {
    const DeviceBackend* d = be();
    const int rc = d->vmem_commit(at, pg.len(i));
    if (rc < 0) return rc;
    if (d->vmem_protect)
        if (const int prc = d->vmem_protect(at, pg.len(i)); prc < 0) return prc;
    return rc;
}

/* Zeros what recycled chunks hold: a chunk whose writer covers only its tensors reads zero in
 * between, as a fresh one does, so a later park still finds it equal to its sources. */
struct Clear { char* at; int64_t bytes; int device; };
/* The clears on `st`, not waited for. */
int clear_issue(const std::vector<Clear>& v, Streams& st) {
    for (const Clear& c : v) {
        RadStream s = st.on(c.device);
        if (!s) return RAD_E_DEVICE;
        if (const int rc = rad_memset_async(c.at, 0, c.bytes, s); rc < 0) return rc;
    }
    return RAD_OK;
}
int clear_all(const std::vector<Clear>& v, Streams& st) {
    if (v.empty()) return RAD_OK;
    if (const int rc = clear_issue(v, st); rc < 0) return rc;
    return st.sync_all();
}
int map_chunk(Page& pg, char* base, size_t i) { return map_dev(pg, base + pg.off(i), i); }

/* Host memory held by host copies, all owners. */
std::atomic<int64_t> g_host_bytes{0};

int host_commit(char* at, int64_t n) {
    const int rc = be()->vmem_commit_host(at, n);
    if (rc >= 0) g_host_bytes.fetch_add(n);
    return rc;
}
void host_release(char* at, int64_t n) {
    if (be()->vmem_decommit(at, n) >= 0) g_host_bytes.fetch_sub(n);
}
/* Host copies to let go, collected under g_mu and released after it. */
struct Gone { char* at; int64_t n; int device; };
void release_all(const std::vector<Gone>& g) {
    for (const Gone& x : g) {
        rad_dev_set(x.device);
        host_release(x.at, x.n);
    }
}

int container_fd(int fid);

/* One aligned read covering every piece (all of one container, within kScratch), copied out. */
int fill_expected_part(const std::vector<Piece>& g, uint8_t* out, uint8_t* scr) {
    constexpr int64_t A = DirectFile::alignment();
    const int fd = container_fd(g.front().fid);
    if (fd < 0) return RAD_E_IO;
    const int64_t lo = g.front().f0 / A * A;
    int64_t hi = 0;
    for (const Piece& q : g) hi = std::max(hi, q.f0 + q.n);
    const int64_t want = align_up(hi, A) - lo;
    int64_t got = 0;
    while (got < want) {
        const ssize_t r = ::pread(fd, scr + got, (size_t)(want - got), (off_t)(lo + got));
        if (r < 0) {
            if (errno == EINTR) continue;
            return RAD_E_IO;
        }
        if (r == 0) break;   /* the end of the file, past its last block */
        got += r;
    }
    if (got < hi - lo) return RAD_E_IO;
    for (const Piece& q : g) std::memcpy(out + q.d0, scr + (q.f0 - lo), (size_t)q.n);
    return RAD_OK;
}

/* The container pieces chunk i was loaded from; empty when it has none or was written since.
 * Under g_mu. */
std::vector<Piece> chunk_pieces(Page& pg, size_t i) {
    std::vector<Piece> out;
    if (!pg.checkable(i)) return out;
    if (!pg.src_sorted) {
        std::sort(pg.src.begin(), pg.src.end(), [](const Src& a, const Src& b) { return a.off < b.off; });
        pg.src_sorted = true;
    }
    const int64_t lo = pg.off(i), hi = lo + pg.len(i);
    auto it = std::partition_point(pg.src.begin(), pg.src.end(), [&](const Src& x) { return x.end() <= lo; });
    for (; it != pg.src.end() && it->off < hi; ++it) {
        const Src& x = *it;
        const int64_t y0 = x.ny <= 1 ? 0 : std::max<int64_t>((lo - x.off) / x.wb, 0);
        for (int64_t y = y0; y < x.ny; ++y) {
            const int64_t r0 = x.off + y * x.wb;
            if (r0 >= hi) break;
            const int64_t a = std::max(lo, r0), b = std::min(hi, r0 + x.wb);
            if (b > a) out.push_back({ x.fid, x.foff + y * x.sp + (a - r0), b - a, a - lo });
        }
    }
    return out;
}

/* Pageable scratch for container reads (kScratch) and checks (kChunk), reused across calls. */
constexpr int64_t kScratch = (int64_t)16 << 20;
std::mutex                              g_scr_mu;
std::map<int64_t, std::vector<void*>>   g_scr_free;
void* scratch_take(int64_t n) {
    {
        std::lock_guard<std::mutex> lk(g_scr_mu);
        auto& v = g_scr_free[n];
        if (!v.empty()) { void* p = v.back(); v.pop_back(); return p; }
    }
    void* p = nullptr;
    return posix_memalign(&p, (size_t)DirectFile::alignment(), (size_t)n) == 0 ? p : nullptr;
}
void scratch_give(void* p, int64_t n) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_scr_mu);
    auto& v = g_scr_free[n];
    if (v.size() < 8) v.push_back(p);
    else std::free(p);
}

int container_fd(int fid) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (fid < 0 || (size_t)fid >= g_files.size()) return -1;
    if (g_dio.size() < g_files.size()) g_dio.resize(g_files.size());
    auto& f = g_dio[(size_t)fid];
    if (!f) {
        f = std::make_unique<DirectFile>();
        if (f->open(g_files[(size_t)fid].c_str(), RAD_DIO_READ | RAD_DIO_ALLOW_BUFFERED) < 0) {
            f.reset();
            return -1;
        }
    }
    return f->fd();
}

/* Fills out[0, n) with what the pieces say the chunk holds: their container bytes, zero elsewhere.
 * Nearby pieces share one aligned read through `scr` (kScratch bytes). */
int fill_expected(std::vector<Piece> pieces, uint8_t* out, int64_t n, uint8_t* scr) {
    std::memset(out, 0, (size_t)n);
    std::sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& b) {
        return a.fid != b.fid ? a.fid < b.fid : a.f0 < b.f0;
    });
    constexpr int64_t A = DirectFile::alignment();
    constexpr int64_t kGap = 1 << 20;
    for (size_t k = 0; k < pieces.size();) {
        const int fid = pieces[k].fid;
        const int64_t lo = pieces[k].f0 / A * A;
        int64_t hi = pieces[k].f0 + pieces[k].n;
        size_t e = k + 1;
        for (; e < pieces.size(); ++e) {
            const Piece& q = pieces[e];
            if (q.fid != fid || q.f0 - hi > kGap) break;
            const int64_t h = std::max(hi, q.f0 + q.n);
            if (align_up(h, A) - lo > kScratch) break;
            hi = h;
        }
        if (align_up(hi, A) - lo > kScratch) {
            /* One piece wider than the scratch: read it in parts straight into place. */
            const Piece& q = pieces[k];
            for (int64_t at = 0; at < q.n;) {
                const int64_t f = q.f0 + at;
                const int64_t a = f / A * A;
                const int64_t take = std::min(q.n - at, kScratch - (f - a) - A);
                std::vector<Piece> one{ { fid, f, take, q.d0 + at } };
                if (const int rc = fill_expected_part(one, out, scr); rc < 0) return rc;
                at += take;
            }
            k = e;
            continue;
        }
        std::vector<Piece> grp(pieces.begin() + (ptrdiff_t)k, pieces.begin() + (ptrdiff_t)e);
        if (const int rc = fill_expected_part(grp, out, scr); rc < 0) return rc;
        k = e;
    }
    return RAD_OK;
}

}  // namespace

void* page_alloc(int64_t bytes, int owner, int device) {
    const DeviceBackend* d = be();
    const int64_t g = VMem::granularity();
    if (g <= 0 || !d->vmem_reserve) return nullptr;
    Page pg;
    pg.span = align_up(bytes, g);
    pg.owner = owner;
    pg.device = device;
    pg.ch.resize((size_t)((pg.span + kChunk - 1) / kChunk));
    char* base = static_cast<char*>(d->vmem_reserve(pg.span, g));
    if (!base) return nullptr;
    std::vector<Clear> clear;
    auto undo = [&](size_t n) {
        for (size_t j = 0; j < n; ++j) d->vmem_decommit(base + pg.off(j), pg.len(j));
        d->vmem_release(base, pg.span);
        return nullptr;
    };
    for (size_t i = 0; i < pg.ch.size(); ++i) {
        const int rc = map_chunk(pg, base, i);
        if (rc < 0) return undo(i);
        if (rc == 1) clear.push_back({ base + pg.off(i), pg.len(i), device });
    }
    if (!clear.empty()) {
        Streams st;
        if (dev_settle_maps() < 0 || clear_all(clear, st) < 0) return undo(pg.ch.size());
        rad_dev_set(device);
    }
    /* Without a shadow (a backend with no host-backed mappings) a park goes to disk alone. */
    if (d->vmem_commit_host && d->vmem_swap) pg.shadow = static_cast<char*>(d->vmem_reserve(pg.span, g));
    std::lock_guard<std::mutex> lk(g_mu);
    g_pages[base] = std::move(pg);
    return base;
}

bool page_free(void* p) {
    Page pg;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_pages.find(static_cast<char*>(p));
        if (it == g_pages.end()) return false;
        pg = std::move(it->second);
        g_pages.erase(it);
    }
    const DeviceBackend* d = be();
    char* base = static_cast<char*>(p);
    for (size_t i = 0; i < pg.ch.size(); ++i) {
        const Chunk& c = pg.ch[i];
        if (!c.out) d->vmem_decommit(base + pg.off(i), pg.len(i));
        else if (c.hloc == kAtBase) host_release(base + pg.off(i), pg.len(i));
        if (c.hloc == kAtShadow) host_release(pg.shadow + pg.off(i), pg.len(i));
    }
    d->vmem_release(base, pg.span);
    if (pg.shadow) d->vmem_release(pg.shadow, pg.span);
    if (!pg.file.empty()) ::unlink(pg.file.c_str());
    return true;
}

int dev_page_file(const char* path) {
    if (!path) return -1;
    std::lock_guard<std::mutex> lk(g_mu);
    for (size_t k = 0; k < g_files.size(); ++k)
        if (g_files[k] == path) return (int)k;
    g_files.emplace_back(path);
    return (int)g_files.size() - 1;
}

void dev_page_source(const void* dst, int fid, int64_t foff, int64_t wb, int64_t ny, int64_t sp) {
    if (!dst || fid < 0 || wb <= 0 || ny <= 0) return;
    const char* lo = static_cast<const char*>(dst);
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_pages.upper_bound(const_cast<char*>(lo));
    if (it == g_pages.begin()) return;
    --it;
    Page& pg = it->second;
    const int64_t off = lo - it->first;
    if (off >= pg.span || pg.refill) return;
    Src x{ off, fid, foff, wb, ny, ny > 1 ? sp : 0 };
    if (x.end() > pg.span) return;
    if (!pg.src.empty() && pg.src.back().off > off) pg.src_sorted = false;
    pg.src.push_back(x);
    for (size_t i = (size_t)(off / kChunk); i < pg.ch.size() && pg.off(i) < x.end(); ++i) pg.ch[i].src = true;
}

void dev_page_discard(int owner, const std::vector<std::pair<const void*, int64_t>>& ranges) {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [base, pg] : g_pages) {
        if (pg.owner != owner) continue;
        for (size_t i = 0; i < pg.ch.size(); ++i) {
            const char* lo = base + pg.off(i);
            const char* hi = lo + pg.len(i);
            for (const auto& [r, n] : ranges)
                if (lo >= static_cast<const char*>(r) && hi <= static_cast<const char*>(r) + n) {
                    pg.ch[i].discard = true;
                    break;
                }
        }
    }
}

int dev_page_out(int owner, const PageOut& req, PageResult* out) {
    PageResult res;
    struct Pick { char* base; Page* pg; size_t i; int cls; int64_t dirt; };
    std::vector<Pick> picks;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages) {
            if (pg.owner != owner) continue;
            for (size_t i = 0; i < pg.ch.size(); ++i) {
                if (pg.ch[i].out) continue;
                /* Cheapest first: nothing to keep, then a chunk its owner refills (one whose host
                 * copy, filled from the container, is still exact leaves by the swap), then a copy
                 * that already exists, then a copy exact but for a few blocks, then a copy. */
                const Chunk& c = pg.ch[i];
                const int cls = c.discard ? 0
                              : (pg.refill && c.sealed) ? (c.hloc == kAtShadow && !c.any_dirty() ? 1 : 4)
                              : (c.sealed && pg.has_copy(i)) ? 1
                              : pg.patchable(i) ? 2 : 3;
                picks.push_back({ base, &pg, i, cls, cls == 2 ? pg.ch[i].dirty_bytes() : 0 });
            }
        }
    }
    /* Within a class, the fewest dirty blocks first, then from the top of the largest
     * allocations down. */
    auto rank = [](int cls) { return cls == 4 ? 1 : cls < 1 ? cls : cls + 1; };
    std::stable_sort(picks.begin(), picks.end(), [&](const Pick& a, const Pick& b) {
        if (a.cls != b.cls) return rank(a.cls) < rank(b.cls);
        if (a.dirt != b.dirt) return a.dirt < b.dirt;
        if (a.pg->span != b.pg->span) return a.pg->span > b.pg->span;
        return a.i > b.i;
    });
    const bool all = req.want.empty();
    std::map<int, int64_t> left = req.want;
    int64_t host_left = req.host_budget;
    Streams st;
    const DeviceBackend* d = be();

    /* Chunks to copy are copied in batches: host copies committed, copies issued, one sync, then
     * the swaps and unmaps. A chunk the load filled from the container is checked against it
     * first, and goes out with nothing written when it still matches.
     *
     * A host job leaves its chunk with the host copy mapped at the chunk's address: a fresh copy
     * is committed at the shadow and filled there, a kept one only patched where dirty, and the
     * two mappings are swapped. A disk job copies through a staging buffer. */
    struct Job {
        Pick p; void* buf; bool to_disk; std::vector<Piece> pieces; bool backed = false;
        bool patch = false, staged = false;
        bool host = false;    /* leaves through the shadow: buf is its host copy */
        bool fresh = false;   /* ...a copy made now, committed by flush */
        bool committed = false;   /* flush mapped its host memory at the shadow */
        std::vector<std::pair<int64_t, int64_t>> runs;
    };
    std::vector<Job> batch;
    /* A batch that fails before its swaps leaves nothing behind: the host memory it mapped at the
     * shadows and its staging buffers go back, and its chunks stay on the card as they were. */
    auto undo = [&](int rc) {
        for (Job& j : batch) {
            if (j.committed) {
                rad_dev_set(j.p.pg->device);
                host_release((char*)j.buf, j.p.pg->len(j.p.i));
            }
            if (j.staged && j.buf) stage_give(j.buf);
        }
        batch.clear();
        return rc;
    };
    auto flush = [&]() -> int {
        if (batch.empty()) return RAD_OK;
        /* Fresh host copies first, then the flush a remapped address needs before it is written:
         * a shadow granule has held other mappings before. A chunk whose host memory cannot be
         * had goes to the disk tier instead; without one (or without a staging buffer) the
         * page-out fails with RAD_E_NOMEM. */
        bool remapped = false;
        for (Job& j : batch) {
            if (!j.fresh) continue;
            rad_dev_set(j.p.pg->device);
            const int64_t n = j.p.pg->len(j.p.i);
            if (host_commit((char*)j.buf, n) >= 0) { remapped = j.committed = true; continue; }
            j.host = j.fresh = false;
            res.host -= n;
            j.buf = req.dir.empty() ? nullptr : stage_take();
            if (!j.buf) return undo(RAD_E_NOMEM);
            j.staged = j.to_disk = true;
            res.disk += n;
        }
        if (remapped)
            if (const int rc = dev_settle_maps(); rc < 0) return undo(rc);
        /* Copies are issued once the batch's buffers all exist: none is allocated while one runs. */
        for (Job& j : batch) {
            if (j.host && !j.fresh && !j.patch) continue;   /* a kept copy, exact: nothing to copy */
            RadStream s = st.on(j.p.pg->device);
            if (!s) return undo(RAD_E_DEVICE);
            char* at = j.p.base + j.p.pg->off(j.p.i);
            if (j.patch) {
                for (const auto& [o, n] : j.runs)
                    if (const int rc = rad_memcpy_async((char*)j.buf + o, at + o, n, s); rc < 0) return undo(rc);
            } else if (const int rc = rad_memcpy_async(j.buf, at, j.p.pg->len(j.p.i), s); rc < 0) {
                return undo(rc);
            }
        }
        if (const int rc = st.sync_all(); rc < 0) return undo(rc);
        /* The staged ones go to disk, four writers at once. */
        std::vector<Job*> disk;
        for (Job& j : batch) if (j.to_disk) disk.push_back(&j);
        std::atomic<size_t> next{0};
        std::atomic<int> err{RAD_OK};
        auto writer = [&] {
            uint8_t* want = nullptr;
            uint8_t* scr = nullptr;
            for (size_t k; (k = next.fetch_add(1)) < disk.size();) {
                Job& j = *disk[k];
                const int64_t n = j.p.pg->len(j.p.i);
                if (j.patch) {
                    for (const auto& [o, m] : j.runs)
                        if (file_io(j.p.pg->file, true, (char*)j.buf + o, j.p.pg->off(j.p.i) + o, m) < 0)
                            err.store(RAD_E_IO);
                    continue;
                }
                if (!j.pieces.empty()) {
                    if (!want) want = (uint8_t*)scratch_take(kChunk);
                    if (!scr) scr = (uint8_t*)scratch_take(kScratch);
                    if (want && scr && fill_expected(j.pieces, want, n, scr) == RAD_OK &&
                        std::memcmp(want, j.buf, (size_t)n) == 0) {
                        j.backed = true;
                        continue;
                    }
                }
                if (j.p.pg->file.empty()) {
                    std::lock_guard<std::mutex> lk(g_mu);
                    if (j.p.pg->file.empty()) j.p.pg->file = new_file(req.dir);
                }
                if (file_io(j.p.pg->file, true, j.buf, j.p.pg->off(j.p.i), n) < 0)
                    err.store(RAD_E_IO);
            }
            scratch_give(want, kChunk);
            scratch_give(scr, kScratch);
        };
        std::vector<std::thread> th;
        for (int t = 0; t < std::min<int>(4, (int)disk.size()); ++t) th.emplace_back(writer);
        for (auto& t : th) t.join();
        if (err.load() < 0) {
            device_set_error("parking: writing to the disk tier failed");
            return undo(err.load());
        }
        bool swapped = false;
        for (Job& j : batch) {
            char* addr = j.p.base + j.p.pg->off(j.p.i);
            const int64_t n = j.p.pg->len(j.p.i);
            rad_dev_set(j.p.pg->device);
            if (j.host) {
                /* The host copy moves to the chunk's address; the device memory, now at the
                 * shadow, is let go. */
                char* sh = j.p.pg->shadow + j.p.pg->off(j.p.i);
                if (const int rc = d->vmem_swap(addr, sh, n); rc < 0) return rc;
                if (const int rc = d->vmem_decommit(sh, n); rc < 0) return rc;
                swapped = true;
            } else if (const int rc = d->vmem_decommit(addr, n); rc < 0) {
                return rc;
            }
            std::lock_guard<std::mutex> lk(g_mu);
            Chunk& c = j.p.pg->ch[j.p.i];
            c.out = true;
            if (j.host) {
                c.hloc = kAtBase;
                c.clear_dirty();
            } else if (j.patch) {
                c.clear_dirty();
                if (j.staged) stage_give(j.buf);
            } else if (j.backed) {
                c.backed = true;
                res.disk -= n;
                res.file += n;
                stage_give(j.buf);
            } else {
                c.on_disk = true;
                if (!j.pieces.empty()) c.nofile = true;   /* checked and differs: not the container's */
                stage_give(j.buf);
            }
        }
        batch.clear();
        if (swapped)
            if (const int rc = dev_settle_maps(); rc < 0) return rc;
        return RAD_OK;
    };
    /* Queues a job; a full batch (16) is flushed at once. */
    auto enqueue = [&](Job&& j) -> int {
        batch.push_back(std::move(j));
        return batch.size() >= 16 ? flush() : RAD_OK;
    };

    for (const Pick& p : picks) {
        const int dev = p.pg->device;
        if (!all) {
            auto it = left.find(dev);
            if (it == left.end() || it->second <= 0) continue;
        }
        const int64_t n = p.pg->len(p.i);
        char* addr = p.base + p.pg->off(p.i);
        char* sh = p.pg->shadow ? p.pg->shadow + p.pg->off(p.i) : nullptr;
        Chunk& c = p.pg->ch[p.i];
        uint8_t hloc;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            hloc = c.hloc;
        }
        if (p.cls == 0 || p.cls == 4 || (p.cls == 1 && hloc != kAtShadow)) {
            /* Nothing to copy: unmap at once. A chunk with nothing to keep, or one its owner
             * refills, lets its host copy go. */
            rad_dev_set(dev);
            if (const int rc = d->vmem_decommit(addr, n); rc < 0) return rc;
            if ((p.cls == 0 || p.cls == 4) && hloc == kAtShadow) host_release(sh, n);
            std::lock_guard<std::mutex> lk(g_mu);
            c.out = true;
            c.dropped = p.cls == 0;
            c.refill = p.cls == 4;
            if (p.cls == 0 || p.cls == 4) c.hloc = kNoHost;
            (p.cls == 0 ? res.dropped : p.cls == 4 ? res.file : res.reused) += n;
        } else if (p.cls == 1) {
            /* A kept host copy, exact: it leaves by the swap alone. */
            Job j{ p, sh, false, {} };
            j.host = true;
            res.reused += n;
            if (const int rc = enqueue(std::move(j)); rc < 0) return rc;
        } else if (p.cls == 2) {
            /* Only the dirty runs move: into the host copy when there is one (the disk copy, now
             * behind it, is let go rather than written on this path), else onto the disk copy. */
            Job j{ p, nullptr, false, {} };
            {
                std::lock_guard<std::mutex> lk(g_mu);
                j.runs = dirty_runs(c, n);
                if (c.hloc == kAtShadow) {
                    j.buf = sh;
                    j.host = true;
                    c.on_disk = false;
                }
            }
            if (!j.buf) {
                j.buf = stage_take();
                if (!j.buf) break;
                j.staged = j.to_disk = true;
            }
            j.patch = true;
            const int64_t dirt = std::min(p.dirt, n);
            (j.to_disk ? res.disk : res.host) += dirt;
            res.reused += n - dirt;
            if (const int rc = enqueue(std::move(j)); rc < 0) return rc;
        } else {
            Job j{ p, nullptr, false, {} };
            if (!req.dir.empty()) {
                std::lock_guard<std::mutex> lk(g_mu);
                j.pieces = chunk_pieces(*p.pg, p.i);
            }
            /* A checkable chunk goes through staging: it most likely needs no copy at all, and
             * one that fails the check goes to the disk tier. */
            if (j.pieces.empty() && host_left >= n && sh) {
                j.buf = sh;
                j.host = j.fresh = true;
                host_left -= n;
                res.host += n;
            } else if (!req.dir.empty()) {
                j.buf = stage_take();
                j.staged = j.to_disk = j.buf != nullptr;
                if (j.to_disk) res.disk += n;
            }
            if (!j.buf) break;   /* no host room and no disk tier: the rest stays on the card */
            if (const int rc = enqueue(std::move(j)); rc < 0) return rc;
        }
        res.freed[dev] += n;
        if (!all) left[dev] -= n;
    }
    if (const int rc = flush(); rc < 0) return rc;

    /* The elastic ranges go whole, when everything is asked for or a card still wants more. */
    bool more = all;
    for (auto& [dv, l] : left) more = more || l > 0;
    if (more)
        for (VMem* v : vmems_of(owner)) {
            if (v->parked()) continue;
            if (!all && left[v->device()] <= 0) continue;
            RadStream s = st.on(v->device());
            if (!s) return RAD_E_DEVICE;
            const int64_t had = v->committed();
            const int64_t n = v->page_out(s, req.discard_elastic);
            if (n < 0) return (int)n;
            res.host += n;
            if (req.discard_elastic) res.dropped += had;
            res.freed[v->device()] += had;
            if (!all) left[v->device()] -= had;
        }
    if (out) *out = res;
    return RAD_OK;
}

namespace {

struct InPick { char* base; Page* pg; size_t i; };

/* The disk copies back onto their cards as one pipeline: readers fill a ring of staging buffers
 * from the drive while the filled ones copy over each card's link; a buffer goes back to the
 * readers when its own copy is done, so neither the drive nor the links wait on a batch. */
int read_back(const std::vector<InPick>& disk, Streams& st) {
    /* What each chunk reads from: its page file, or (backed) the container pieces it came from. */
    std::vector<std::vector<Piece>> from(disk.size());
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (size_t k = 0; k < disk.size(); ++k) {
            Page& pg = *disk[k].pg;
            if (!pg.ch[disk[k].i].on_disk) from[k] = chunk_pieces(pg, disk[k].i);
        }
    }
    if (disk.empty()) return RAD_OK;
    constexpr int kSlots = 16, kReaders = 8;
    struct Slot { void* buf = nullptr; std::map<int, RadEvent> ev; int dev = -1; size_t pick = 0; };
    std::vector<Slot> slots(kSlots);
    std::mutex mu;
    std::condition_variable cv;
    std::deque<int> free_q, filled;
    size_t next = 0;
    bool stop = false;
    int err = RAD_OK;
    for (int k = 0; k < kSlots; ++k) {
        slots[k].buf = stage_take();
        if (!slots[k].buf) {
            for (Slot& x : slots) stage_give(x.buf);
            return RAD_E_NOMEM;
        }
        free_q.push_back(k);
    }
    auto reader = [&] {
        uint8_t* scr = nullptr;
        struct Give { uint8_t*& p; ~Give() { scratch_give(p, kScratch); } } give{ scr };
        for (;;) {
            int k;
            size_t at;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || next >= disk.size() || !free_q.empty(); });
                if (stop || next >= disk.size()) return;
                k = free_q.front();
                free_q.pop_front();
                at = next++;
            }
            const InPick& p = disk[at];
            int rc;
            if (from[at].empty()) {
                rc = file_io(p.pg->file, false, slots[k].buf, p.pg->off(p.i), p.pg->len(p.i));
            } else {
                if (!scr) scr = (uint8_t*)scratch_take(kScratch);
                rc = scr ? fill_expected(from[at], (uint8_t*)slots[k].buf, p.pg->len(p.i), scr) : RAD_E_NOMEM;
            }
            std::lock_guard<std::mutex> lk(mu);
            if (rc < 0) { err = RAD_E_IO; stop = true; }
            slots[k].pick = at;
            filled.push_back(k);
            cv.notify_all();
        }
    };
    std::vector<std::thread> th;
    for (int r = 0; r < kReaders; ++r) th.emplace_back(reader);
    std::deque<int> flying;
    int rc = RAD_OK;
    auto land = [&](int k, bool wait) -> int {
        Slot& x = slots[k];
        rad_dev_set(x.dev);
        const int q = wait ? rad_event_sync(x.ev[x.dev]) : rad_event_query(x.ev[x.dev]);
        if (q < 0) return q;
        if (!wait && q == 0) return 0;
        std::lock_guard<std::mutex> lk(mu);
        free_q.push_back(k);
        cv.notify_all();
        return 1;
    };
    for (size_t issued = 0; issued < disk.size() && rc >= 0;) {
        while (!flying.empty()) {   /* buffers whose copies finished go back to the readers */
            const int q = land(flying.front(), false);
            if (q < 0) { rc = q; break; }
            if (q == 0) break;
            flying.pop_front();
        }
        if (rc < 0) break;
        int k = -1;
        {
            std::unique_lock<std::mutex> lk(mu);
            if (filled.empty() && !flying.empty() && free_q.empty()) {
                lk.unlock();
                /* Every buffer is filled or copying and none is ready: the oldest copy is next. */
                if (const int q = land(flying.front(), true); q < 0) { rc = q; break; }
                flying.pop_front();
                continue;
            }
            cv.wait_for(lk, std::chrono::microseconds(200), [&] { return stop || !filled.empty(); });
            if (err < 0) { rc = err; break; }
            if (filled.empty()) continue;
            k = filled.front();
            filled.pop_front();
        }
        Slot& x = slots[k];
        const InPick& p = disk[x.pick];
        x.dev = p.pg->device;
        RadStream s = st.on(x.dev);
        RadEvent& ev = x.ev[x.dev];
        if (!s || (!ev && rad_event_create(&ev) < 0)) { rc = RAD_E_DEVICE; break; }
        if ((rc = rad_memcpy_async(p.base + p.pg->off(p.i), x.buf, p.pg->len(p.i), s)) < 0) break;
        if ((rc = rad_event_record(ev, s)) < 0) break;
        flying.push_back(k);
        ++issued;
    }
    {
        std::lock_guard<std::mutex> lk(mu);
        stop = true;
        cv.notify_all();
    }
    for (auto& t : th) t.join();
    const int sync_rc = st.sync_all();
    for (Slot& x : slots) {
        for (auto& [dv, e] : x.ev) if (e) { rad_dev_set(dv); rad_event_destroy(e); }
        stage_give(x.buf);
    }
    if (rc == RAD_E_IO || err < 0) {
        device_set_error("parking: reading back from the disk tier failed");
        return RAD_E_IO;
    }
    return rc < 0 ? rc : sync_rc;
}

/* `live`: a sealed chunk read in place from its host copy stays there (dev_page_promote moves it
 * later); its bytes are added to *left. The chunks the owner refills are mapped and appended to
 * `refill`, adjacent ones merged. */
int page_in_elastic(int owner, Streams& st) {
    for (VMem* v : vmems_of(owner)) {
        RadStream s = st.on(v->device());
        if (!s) return RAD_E_DEVICE;
        if (const int rc = v->page_in(s); rc < 0) return rc;
    }
    return RAD_OK;
}

/* A page-in between its two halves: the copies of the chunks read back from their host copies are
 * in flight on `st`, and those chunks are swapped in and recorded as back only at the end. */
struct InPending {
    Streams st;
    /* The refill chunks' clears, waited for before the refill writes (dev_page_in_cleared). */
    Streams clear_st;
    std::vector<InPick> host;
};
std::mutex g_in_mu;
std::map<int, std::unique_ptr<InPending>> g_in;   /* by owner */

/* What a page-in has done for chunk `c` once its memory holds its bytes again. A sealed chunk
 * keeps its copies for the next park; any other's go stale the moment a kernel writes, and its
 * host copy at the shadow is appended to `gone`. Under g_mu. */
void chunk_back(const Page& pg, size_t i, Chunk& c, std::vector<Gone>& gone) {
    c.out = false;
    c.discard = false;
    c.dropped = false;
    c.refill = false;
    if (c.hloc == kAtBase) c.hloc = kAtShadow;
    if (!c.sealed) {
        if (c.hloc == kAtShadow) gone.push_back({ pg.shadow + pg.off(i), pg.len(i), pg.device });
        c.hloc = kNoHost;
        c.on_disk = false;
        c.backed = false;
        c.nofile = true;
        c.clear_dirty();
    }
}

int page_in_begin(int owner, bool live, int64_t* left, std::vector<PageSpan>* refill) {
    using Pick = InPick;
    std::vector<Pick> picks;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages)
            if (pg.owner == owner)
                for (size_t i = 0; i < pg.ch.size(); ++i) {
                    const Chunk& c = pg.ch[i];
                    if (!c.out) continue;
                    if (c.refill && !refill) {
                        device_set_error("parking: a page-in of chunks the owner refills was not "
                                         "asked to hand them back");
                        return RAD_E_STATE;
                    }
                    if (live && c.sealed && c.hloc == kAtBase) {
                        if (left) *left += pg.len(i);
                        continue;
                    }
                    picks.push_back({ base, &pg, i });
                }
    }
    auto pend = std::make_unique<InPending>();
    Streams& st = pend->st;
    std::vector<InPick>& host = pend->host;
    /* Every chunk's device memory mapped first, then the copies: a chunk read in place from its
     * host copy gets its device memory at the shadow, filled over the link and swapped in at the
     * end (page_in_end), once its copy has landed; the disk ones go through the staging ring. No
     * mapping of these chunks changes while a copy into them runs. */
    std::vector<Pick> disk;
    /* Host copies and disk reads write whole chunks; the owner's refill writes its tensors only. */
    std::vector<Clear> clear;
    for (const Pick& p : picks) {
        const Chunk& c = p.pg->ch[p.i];
        rad_dev_set(p.pg->device);
        if (c.hloc == kAtBase) {
            if (const int rc = map_dev(*p.pg, p.pg->shadow + p.pg->off(p.i), p.i); rc < 0) return rc;
            host.push_back(p);
            continue;
        }
        const int mrc = map_chunk(*p.pg, p.base, p.i);
        if (mrc < 0) return mrc;
        if (c.refill) {
            char* at = p.base + p.pg->off(p.i);
            if (mrc == 1) clear.push_back({ at, p.pg->len(p.i), p.pg->device });
            if (!refill->empty() && refill->back().p + refill->back().bytes == at &&
                refill->back().device == p.pg->device)
                refill->back().bytes += p.pg->len(p.i);
            else
                refill->push_back({ at, p.pg->len(p.i), p.pg->device });
            continue;
        }
        if ((c.on_disk || c.backed) && !c.dropped) disk.push_back(p);
    }
    if (!picks.empty())
        if (const int rc = dev_settle_maps(); rc < 0) return rc;
    /* Only refill chunks are cleared, and the refill writes nothing before its first read lands, so
     * the clears run beside that read (~20-40 ms of a large restore's) instead of ahead of it. */
    if (const int rc = clear_issue(clear, pend->clear_st); rc < 0) return rc;
    for (const Pick& p : host) {
        RadStream s = st.on(p.pg->device);
        if (!s) return RAD_E_DEVICE;
        if (const int rc = rad_memcpy_async(p.pg->shadow + p.pg->off(p.i), p.base + p.pg->off(p.i),
                                            p.pg->len(p.i), s); rc < 0)
            return rc;
    }
    /* Round-robin over the cards, so each card's link has a chunk in flight while the drive
     * reads the next. */
    {
        std::map<int, std::vector<Pick>> by;
        for (const Pick& p : disk) by[p.pg->device].push_back(p);
        disk.clear();
        for (size_t k = 0;; ++k) {
            bool any = false;
            for (auto& [dv, v] : by)
                if (k < v.size()) { disk.push_back(v[k]); any = true; }
            if (!any) break;
        }
    }
    /* With anything on disk, read_back waits for the stream, the host copies with it. */
    if (const int rc = read_back(disk, st); rc < 0) return rc;
    std::vector<Gone> gone;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (const Pick& p : picks) {
            Chunk& c = p.pg->ch[p.i];
            if (c.hloc != kAtBase) chunk_back(*p.pg, p.i, c, gone);
        }
    }
    release_all(gone);
    std::lock_guard<std::mutex> lk(g_in_mu);
    g_in[owner] = std::move(pend);
    return RAD_OK;
}

int page_in_end(int owner, bool elastic) {
    std::unique_ptr<InPending> pend;
    {
        std::lock_guard<std::mutex> lk(g_in_mu);
        auto it = g_in.find(owner);
        if (it == g_in.end()) {
            device_set_error("parking: a page-in ended that had not begun");
            return RAD_E_STATE;
        }
        pend = std::move(it->second);
        g_in.erase(it);
    }
    Streams& st = pend->st;
    const DeviceBackend* d = be();
    if (const int rc = pend->clear_st.sync_all(); rc < 0) return rc;
    if (const int rc = st.sync_all(); rc < 0) return rc;
    for (const InPick& p : pend->host) {
        rad_dev_set(p.pg->device);
        if (const int rc = d->vmem_swap(p.base + p.pg->off(p.i), p.pg->shadow + p.pg->off(p.i),
                                        p.pg->len(p.i)); rc < 0)
            return rc;
    }
    if (!pend->host.empty())
        if (const int rc = dev_settle_maps(); rc < 0) return rc;
    std::vector<Gone> gone;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (const InPick& p : pend->host) chunk_back(*p.pg, p.i, p.pg->ch[p.i], gone);
    }
    release_all(gone);
    return elastic ? page_in_elastic(owner, st) : RAD_OK;
}

}  // namespace

int dev_page_in(int owner, std::vector<PageSpan>* refill, bool elastic) {
    if (const int rc = page_in_begin(owner, false, nullptr, refill); rc < 0) return rc;
    return page_in_end(owner, elastic);
}

int dev_page_in_live(int owner, int64_t* left, std::vector<PageSpan>* refill, bool elastic) {
    if (left) *left = 0;
    if (const int rc = page_in_begin(owner, true, left, refill); rc < 0) return rc;
    return page_in_end(owner, elastic);
}

int dev_page_in_begin(int owner, int64_t* left, std::vector<PageSpan>* refill) {
    if (left) *left = 0;
    return page_in_begin(owner, left != nullptr, left, refill);
}

int dev_page_in_end(int owner) { return page_in_end(owner, false); }

int dev_page_in_cleared(int owner) {
    InPending* pend = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_in_mu);
        auto it = g_in.find(owner);
        if (it == g_in.end()) return RAD_OK;
        pend = it->second.get();
    }
    const int dev = dev_current();
    const int rc = pend->clear_st.sync_all();
    (void)rad_dev_set(dev);
    return rc;
}

int dev_page_in_elastic(int owner) {
    Streams st;
    if (const int rc = page_in_elastic(owner, st); rc < 0) return rc;
    /* The ranges are backed at the addresses they had: same-VA maps need the translations
     * flushed before a card writes them. */
    return dev_settle_maps();
}

int64_t dev_page_promote(int owner, int64_t max_bytes, const std::function<int(const std::function<int()>&)>& quiet) {
    struct Pick { char* base; Page* pg; size_t i; uint32_t gen; };
    std::vector<Pick> picks;
    int64_t want = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages) {
            if (pg.owner != owner) continue;
            for (size_t i = 0; i < pg.ch.size() && want < max_bytes; ++i)
                if (pg.ch[i].out && pg.ch[i].hloc == kAtBase) {
                    picks.push_back({ base, &pg, i, 0 });
                    want += pg.len(i);
                }
        }
    }
    if (picks.empty()) return 0;
    const DeviceBackend* d = be();
    /* Device memory at the shadows, filled over the link while the engine runs on; the copy is
     * exact unless the engine wrote the chunk meanwhile (its generation moved), and then it is
     * made again inside the quiet window. */
    size_t mapped = 0;
    int rc = RAD_OK;
    for (; mapped < picks.size(); ++mapped) {
        const Pick& p = picks[mapped];
        rad_dev_set(p.pg->device);
        if ((rc = map_dev(*p.pg, p.pg->shadow + p.pg->off(p.i), p.i)) < 0) break;
    }
    auto unmap_rest = [&](size_t from) {
        for (size_t k = from; k < mapped; ++k) {
            rad_dev_set(picks[k].pg->device);
            (void)d->vmem_decommit(picks[k].pg->shadow + picks[k].pg->off(picks[k].i), picks[k].pg->len(picks[k].i));
        }
    };
    if (mapped == 0) return rc;   /* the first map failed */
    picks.resize(mapped);
    if ((rc = dev_settle_maps()) < 0) { unmap_rest(0); return rc; }
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (Pick& p : picks) p.gen = p.pg->ch[p.i].gen;
    }
    Streams st;
    auto copy = [&](const Pick& p) -> int {
        RadStream s = st.on(p.pg->device);
        if (!s) return RAD_E_DEVICE;
        return rad_memcpy_async(p.pg->shadow + p.pg->off(p.i), p.base + p.pg->off(p.i), p.pg->len(p.i), s);
    };
    for (const Pick& p : picks)
        if ((rc = copy(p)) < 0) break;
    if (rc >= 0) rc = st.sync_all();
    if (rc < 0) { unmap_rest(0); return rc; }
    int64_t moved = 0;
    size_t swapped = 0;
    const int qrc = quiet([&]() -> int {
        bool again = false;
        for (const Pick& p : picks) {
            bool stale;
            {
                std::lock_guard<std::mutex> lk(g_mu);
                stale = p.pg->ch[p.i].gen != p.gen;
            }
            if (stale) {
                if (const int r = copy(p); r < 0) return r;
                again = true;
            }
        }
        if (again)
            if (const int r = st.sync_all(); r < 0) return r;
        for (; swapped < picks.size(); ++swapped) {
            const Pick& p = picks[swapped];
            rad_dev_set(p.pg->device);
            if (const int r = d->vmem_swap(p.base + p.pg->off(p.i), p.pg->shadow + p.pg->off(p.i), p.pg->len(p.i)); r < 0)
                return r;
        }
        return dev_settle_maps();
    });
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (size_t k = 0; k < swapped; ++k) {
            Chunk& c = picks[k].pg->ch[picks[k].i];
            c.out = false;
            c.hloc = kAtShadow;
            moved += picks[k].pg->len(picks[k].i);
        }
    }
    unmap_rest(swapped);
    return qrc < 0 ? qrc : moved;
}

int64_t dev_page_live_bytes(int owner, int device) {
    int64_t n = 0;
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [base, pg] : g_pages)
        if (pg.owner == owner && (device < 0 || pg.device == device))
            for (size_t i = 0; i < pg.ch.size(); ++i)
                if (pg.ch[i].out && pg.ch[i].hloc == kAtBase) n += pg.len(i);
    return n;
}

void dev_page_stats(int owner, int64_t* device, int64_t* host) {
    int64_t dv = 0, h = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages) {
            if (pg.owner != owner) continue;
            for (size_t i = 0; i < pg.ch.size(); ++i) {
                if (!pg.ch[i].out) dv += pg.len(i);
                if (pg.ch[i].hloc) h += pg.len(i);
            }
        }
    }
    for (VMem* v : vmems_of(owner)) {
        dv += v->committed();
        h += v->host_bytes();
    }
    if (device) *device = dv;
    if (host) *host = h;
}

std::map<int, int64_t> dev_page_out_bytes(int owner, bool live) {
    std::map<int, int64_t> m;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages) {
            if (pg.owner != owner) continue;
            for (size_t i = 0; i < pg.ch.size(); ++i) {
                const Chunk& c = pg.ch[i];
                if (c.out && !(live && c.sealed && c.hloc == kAtBase)) m[pg.device] += pg.len(i);
            }
        }
    }
    for (VMem* v : vmems_of(owner))
        if (v->parked()) m[v->device()] += v->parked_bytes();
    return m;
}

void dev_page_invalidate(const void* p, int64_t bytes) {
    if (!p || bytes <= 0) return;
    const char* lo = static_cast<const char*>(p);
    std::vector<Gone> gone;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_pages.upper_bound(const_cast<char*>(lo));
        if (it == g_pages.begin()) return;
        --it;
        char* base = it->first;
        Page& pg = it->second;
        if (lo >= base + pg.span) return;
        const int64_t a = lo - base, b = std::min<int64_t>(a + bytes, pg.span);
        for (size_t i = (size_t)(a / kChunk); i < pg.ch.size() && pg.off(i) < b; ++i) {
            Chunk& c = pg.ch[i];
            ++c.gen;
            /* Out and read in place (a live restore, a fill from the drive): the write lands in its
             * host copy, which stays exact and is now its only one. Its disk copy and its sources
             * no longer describe it, so nothing frees the host copy before it is written again. */
            if (c.out) {
                if (c.hloc == kAtBase) {
                    c.backed = false;
                    c.nofile = true;
                    c.on_disk = false;
                    c.clear_dirty();
                }
                continue;
            }
            c.backed = false;
            c.nofile = true;
            if (!c.hloc && !c.on_disk) continue;
            /* A sealed chunk's copy stays, exact but for the blocks written; a chunk written
             * whole, or one that is not sealed, has no copy left. */
            if (c.sealed) {
                c.mark(std::max(a, pg.off(i)) - pg.off(i), std::min(b, pg.off(i) + pg.len(i)) - pg.off(i));
                if (c.dirty_bytes() < pg.len(i)) continue;
            }
            if (c.hloc == kAtShadow) gone.push_back({ pg.shadow + pg.off(i), pg.len(i), pg.device });
            c.hloc = kNoHost;
            c.on_disk = false;
            c.clear_dirty();
        }
    }
    release_all(gone);
}

std::map<int, int64_t> dev_page_mapped_bytes(int owner) {
    std::map<int, int64_t> m;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages) {
            if (pg.owner != owner) continue;
            for (size_t i = 0; i < pg.ch.size(); ++i)
                if (!pg.ch[i].out) m[pg.device] += pg.len(i);
        }
    }
    for (VMem* v : vmems_of(owner)) m[v->device()] += v->committed();
    return m;
}

void dev_page_seal(void* p, int64_t off, int64_t bytes) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_pages.find(static_cast<char*>(p));
    if (it == g_pages.end()) return;
    Page& pg = it->second;
    const int64_t hi = bytes < 0 ? pg.span : off + bytes;
    for (size_t i = 0; i < pg.ch.size(); ++i)
        if (pg.off(i) >= off && pg.off(i) + pg.len(i) <= hi) pg.ch[i].sealed = true;
}

void dev_page_refilled(void* p) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_pages.find(static_cast<char*>(p));
    if (it == g_pages.end()) return;
    Page& pg = it->second;
    pg.refill = true;
    /* The container check is never made on these: the owner rebuilds them. */
    pg.src.clear();
    pg.src.shrink_to_fit();
    for (Chunk& c : pg.ch) c.src = false;
}

int64_t dev_page_uncopied(int owner) {
    int64_t n = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages) {
            if (pg.owner != owner) continue;
            for (size_t i = 0; i < pg.ch.size(); ++i) {
                const Chunk& c = pg.ch[i];
                /* A chunk the load filled from the container is counted as free: the check
                 * against it almost always passes. */
                if (c.out || c.discard || (c.sealed && (pg.refill || pg.has_copy(i))) ||
                    pg.checkable(i))
                    continue;
                n += pg.patchable(i) ? c.dirty_bytes() : pg.len(i);
            }
        }
    }
    return n;
}

int64_t dev_page_drop_copies(int owner) {
    int64_t n = 0;
    std::vector<Gone> gone;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages) {
            if (pg.owner != owner) continue;
            for (size_t i = 0; i < pg.ch.size(); ++i) {
                Chunk& c = pg.ch[i];
                if (c.out || c.hloc != kAtShadow) continue;
                gone.push_back({ pg.shadow + pg.off(i), pg.len(i), pg.device });
                c.hloc = kNoHost;
                n += pg.len(i);
            }
        }
    }
    release_all(gone);
    return n;
}

int64_t dev_page_spill(int owner, const std::string& dir, const std::function<int()>& gate) {
    struct Pick { Page* pg; size_t i; char* at; };
    std::vector<Pick> picks;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages) {
            if (pg.owner != owner) continue;
            /* The weights of an allocation its owner refills are never written to the drive. */
            if (pg.refill) continue;
            for (size_t i = 0; i < pg.ch.size(); ++i) {
                const Chunk& c = pg.ch[i];
                if (!c.hloc || c.on_disk || c.backed) continue;
                if (pg.file.empty()) pg.file = new_file(dir);
                picks.push_back({ &pg, i, (c.hloc == kAtBase ? base : pg.shadow) + pg.off(i) });
            }
        }
    }
    int64_t n = 0;
    void* buf = picks.empty() ? nullptr : stage_take();
    if (!picks.empty() && !buf) return RAD_E_NOMEM;
    for (const Pick& p : picks) {
        if (gate_wait(gate, 20) < 0) { stage_give(buf); return RAD_E_STATE; }
        /* The host reads its copy through the mapping; O_DIRECT writes from a staging buffer. */
        const int64_t len = p.pg->len(p.i);
        uint8_t hloc;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            hloc = p.pg->ch[p.i].hloc;
            if (!hloc) continue;
        }
        std::memcpy(buf, p.at, (size_t)len);
        if (const int rc = file_io(p.pg->file, true, buf, p.pg->off(p.i), len); rc < 0) {
            stage_give(buf);
            return rc;
        }
        std::lock_guard<std::mutex> lk(g_mu);
        if (p.pg->ch[p.i].hloc == hloc) p.pg->ch[p.i].on_disk = true;
        n += len;
    }
    stage_give(buf);
    for (VMem* v : vmems_of(owner)) {
        const int64_t before = v->host_bytes();
        if (const int rc = v->spill(dir, gate); rc < 0) return rc;
        n += before;
    }
    return n;
}

int64_t dev_page_drop_host(int owner) {
    int64_t n = 0;
    std::vector<Gone> gone;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages) {
            if (pg.owner != owner) continue;
            for (size_t i = 0; i < pg.ch.size(); ++i) {
                Chunk& c = pg.ch[i];
                if (!c.hloc || !(c.on_disk || c.backed || (pg.refill && c.sealed))) continue;
                /* Out and read in place: unmapped now, it comes back from its other copy (on an
                 * allocation its owner refills, the container: the page-in hands it back). */
                gone.push_back({ (c.hloc == kAtBase ? base : pg.shadow) + pg.off(i), pg.len(i), pg.device });
                if (c.hloc == kAtBase && pg.refill && !c.on_disk && !c.backed) c.refill = true;
                c.hloc = kNoHost;
                n += pg.len(i);
            }
        }
    }
    release_all(gone);
    for (VMem* v : vmems_of(owner)) n += v->drop_host();
    return n;
}

int64_t dev_page_host_bytes() { return g_host_bytes.load(); }

std::string dev_page_where(uintptr_t a) {
    std::unique_lock<std::mutex> lk(g_mu, std::try_to_lock);
    if (!lk.owns_lock()) return "paging state locked";
    auto describe = [&](const Page& pg, uintptr_t base, const char* what) {
        const int64_t off = (int64_t)(a - base);
        const size_t i = (size_t)(off / kChunk);
        if (i >= pg.ch.size()) return fmt("%s of owner %d card %d +%s", what, pg.owner, pg.device, humanb(off).c_str());
        const Chunk& c = pg.ch[i];
        return fmt("%s of owner %d card %d, %s long, +%s, chunk %zu: out=%d host=%d disk=%d backed=%d dropped=%d refill=%d discard=%d sealed=%d",
                   what, pg.owner, pg.device, humanb(pg.span).c_str(), humanb(off).c_str(), i, (int)c.out, (int)c.hloc,
                   (int)c.on_disk, (int)c.backed, (int)c.dropped, (int)c.refill, (int)c.discard, (int)c.sealed);
    };
    for (const auto& [base, pg] : g_pages) {
        const uintptr_t b = (uintptr_t)base;
        if (a >= b && a < b + (uintptr_t)pg.span) return describe(pg, b, "pageable");
        const uintptr_t sh = (uintptr_t)pg.shadow;
        if (sh && a >= sh && a < sh + (uintptr_t)pg.span) return describe(pg, sh, "pageable's host shadow");
    }
    return std::string();
}

int64_t dev_page_fillable(int owner) {
    int64_t n = 0;
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [base, pg] : g_pages) {
        if (pg.owner != owner || !pg.shadow) continue;
        for (size_t i = 0; i < pg.ch.size(); ++i) {
            const Chunk& c = pg.ch[i];
            if (c.out && !c.hloc && !c.dropped && (c.on_disk || c.backed || c.refill)) n += pg.len(i);
        }
    }
    return n;
}

std::vector<PageSpan> dev_page_refill_chunks(int owner) {
    std::vector<PageSpan> out;
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto& [base, pg] : g_pages) {
        if (pg.owner != owner || !pg.shadow) continue;
        for (size_t i = 0; i < pg.ch.size(); ++i) {
            const Chunk& c = pg.ch[i];
            if (c.out && c.refill && !c.hloc) out.push_back({ base + pg.off(i), pg.len(i), pg.device });
        }
    }
    return out;
}

namespace {
/* The chunk whose first byte is `p`, under g_mu; null when `p` starts none. */
Chunk* chunk_at(char* p, Page** pg_out, char** base_out) {
    auto it = g_pages.upper_bound(p);
    if (it == g_pages.begin()) return nullptr;
    --it;
    Page& pg = it->second;
    const int64_t off = p - it->first;
    if (off < 0 || off >= pg.span || off % kChunk) return nullptr;
    *pg_out = &pg;
    *base_out = it->first;
    return &pg.ch[(size_t)(off / kChunk)];
}
}  // namespace

int dev_page_fill_begin(int owner, const std::vector<PageSpan>& chunks) {
    size_t mapped = 0;
    auto undo = [&] {
        for (size_t k = 0; k < mapped; ++k) {
            rad_dev_set(chunks[k].device);
            host_release(chunks[k].p, chunks[k].bytes);
        }
    };
    for (; mapped < chunks.size(); ++mapped) {
        const PageSpan& c = chunks[mapped];
        {
            std::lock_guard<std::mutex> lk(g_mu);
            Page* pg = nullptr;
            char* base = nullptr;
            Chunk* ch = chunk_at(c.p, &pg, &base);
            if (!ch || pg->owner != owner || !ch->out || !ch->refill || ch->hloc || pg->len((size_t)((c.p - base) / kChunk)) != c.bytes) {
                undo();
                device_set_error("parking: a host fill named a chunk that is not parked for its owner to refill");
                return RAD_E_INVAL;
            }
        }
        rad_dev_set(c.device);
        if (const int rc = host_commit(c.p, c.bytes); rc < 0) {
            undo();
            return rc;
        }
    }
    if (const int rc = dev_settle_maps(); rc < 0) {
        undo();
        return rc;
    }
    return RAD_OK;
}

void dev_page_fill_end(int owner, const std::vector<PageSpan>& chunks, bool filled) {
    (void)owner;
    std::vector<Gone> gone;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (const PageSpan& c : chunks) {
            Page* pg = nullptr;
            char* base = nullptr;
            Chunk* ch = chunk_at(c.p, &pg, &base);
            if (!ch) continue;
            if (filled) {
                ch->hloc = kAtBase;
                ch->refill = false;
                ch->clear_dirty();
            } else {
                gone.push_back({ c.p, c.bytes, c.device });
            }
        }
    }
    release_all(gone);
    if (!filled) (void)dev_settle_maps();
}

int64_t dev_page_fill_host(int owner, int64_t max_bytes) {
    std::vector<InPick> picks;
    int64_t n = 0;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto& [base, pg] : g_pages) {
            if (pg.owner != owner || !pg.shadow) continue;
            for (size_t i = 0; i < pg.ch.size() && n < max_bytes; ++i) {
                const Chunk& c = pg.ch[i];
                if (!c.out || c.hloc || c.dropped || !(c.on_disk || c.backed)) continue;
                picks.push_back({ base, &pg, i });
                n += pg.len(i);
            }
        }
    }
    if (picks.empty()) return 0;
    /* Host memory at each chunk's own address, as a park to host memory leaves it, then the
     * drive's bytes copied into it through the staging ring. */
    size_t mapped = 0;
    auto undo = [&] {
        for (size_t k = 0; k < mapped; ++k) {
            rad_dev_set(picks[k].pg->device);
            host_release(picks[k].base + picks[k].pg->off(picks[k].i), picks[k].pg->len(picks[k].i));
        }
    };
    for (; mapped < picks.size(); ++mapped) {
        const InPick& p = picks[mapped];
        rad_dev_set(p.pg->device);
        if (const int rc = host_commit(p.base + p.pg->off(p.i), p.pg->len(p.i)); rc < 0) {
            undo();
            return rc;
        }
    }
    int rc = dev_settle_maps();
    Streams st;
    if (rc >= 0) rc = read_back(picks, st);
    if (rc >= 0) rc = st.sync_all();
    if (rc < 0) {
        undo();
        (void)dev_settle_maps();
        return rc;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    for (const InPick& p : picks) p.pg->ch[p.i].hloc = kAtBase;
    return n;
}

}  // namespace rad
