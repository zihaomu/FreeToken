/* avx_ngram.cpp -- the row source for embed_lookup_q, and the table's stored layout.
 * Baseline-compiled (no -m flags), like avx_dispatch.cpp; the vector decode comes from the calling
 * level (avx_rows.h).
 *
 * Why a host kernel. The n-gram embedding is 320,001,536 rows of 160 E4M3 bytes (51.2 GB), read 16
 * rows per token. The engine's mechanisms for oversized weights do not fit that access pattern:
 * Tier::SSD + DeviceStaged stages whole movement units, Site::DeviceZeroCopy needs pinned host RAM,
 * and hipHostRegister on the mapping would pin all 51.2 GB. core/format/radfile.cpp mmaps the
 * container, so the table already sits at a host address backed by an NVMe file. The gather runs
 * on the host over that mapping and only the [T, n_embd] bf16 result crosses the link. The placement
 * planner puts an op on the host when all its weight operands are Site::Host (core/runtime/ctx.cpp),
 * so the table must be a weight operand (a plain `gather_rows` input would not trigger it), and a
 * production host kernel is needed since libref is not served (rad_gate_reference_kernels).
 *
 * Row cache plus direct reads. Decoding a row takes a few dozen cycles; fetching it is the cost,
 * and at decode about half the rows a step wants have never been read before. The page cache is
 * the wrong cache: a buffered miss costs the submitting thread ~4 us (page allocation, insertion,
 * block mapping, bio), serialising a call's cold rows, and it keeps a 4 KiB page per 160-byte row
 * (pages average ~1 useful row out of 25). So RowCache (below) holds rows, shared by every gathering
 * thread, and a miss is an O_DIRECT read of the aligned span holding the row (~2 us to submit,
 * 512 bytes instead of 4096). With eight varied sequences a 2^19-row cache (80 MiB) misses 51.0% of
 * rows against 50.6% for an unbounded one.
 *
 * Each miss is read once. Every rank gathers every row. A thread claims a missing row by marking
 * its slot LOADING, reads it and publishes it READY; others wait instead of re-reading. Ranks start
 * at different rows of a call, so the misses are split between them with no barrier.
 *
 * The file is found from the mapping: the ABI passes a pointer, so /proc/self/maps gives the file
 * and offset, once per table. A table not backed by a file (a test's heap buffer) is read as
 * memory. A file on a filesystem without direct I/O is refused; direct_granule finds the read
 * alignment, also where the filesystem reports none. */
#include "avx_rows.h"
#include "avx_common.h"

#include "rad_plugin.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

/* Prefetch distance, in rows, for cache lookups and decodes. Rows are scattered over an 80 MiB
 * cache and the set words over 4 MiB, so each is an L3 or DRAM miss. */
constexpr long long kPrefetchRows = 8;

/* One read of a row's aligned span: row m of the call, its key, and its cache slot (-1: decoded
 * straight from the read, not cached). */
struct SpanRead {
    long long m;
    uint64_t  key;
    int64_t   slot;
};

/* Per-thread working state for a call, kept between calls so steady-state calls do not
 * allocate. */
struct Gather {
    std::vector<int64_t>   slot;     /* per row: its cache slot */
    std::vector<uint8_t>   kind;     /* per row: a RowCache::Found, or PAD / DONE below */
    std::vector<SpanRead>  reads;    /* the rows this thread reads */
    std::vector<long long> waits;    /* rows another thread is reading */
    std::vector<long long> later;    /* the other ranks' share of a decode gather, unclaimed so far */
    std::vector<long long> again;    /* rows whose slot changed under the decode */
    /* One decode batch: each row's source and destination and, for cached rows, its call index
     * and the slot word it was read under. */
    std::vector<const uint8_t*> src;
    std::vector<uint16_t*>      dst;
    std::vector<long long>      dm;
    std::vector<uint64_t>       dw;
    /* Sources and destinations for rows decoded from a read's bounce buffer. */
    std::vector<const uint8_t*> bsrc;
    std::vector<uint16_t*>      bdst;
    uint8_t*              bounce = nullptr;
    size_t                bounce_bytes = 0;
    size_t                bounce_align = 0;

    enum : uint8_t { PAD = 16, DONE };

    /* A buffer of at least `bytes`, aligned to max(page, `align`) (a power of two): direct reads
     * need the file's granule, which can exceed a page on large-block filesystems. */
    uint8_t* bounce_for(size_t bytes, size_t align) {
        if (align < 4096) align = 4096;
        if (bytes > bounce_bytes || align > bounce_align) {
            std::free(bounce);
            bounce = static_cast<uint8_t*>(std::aligned_alloc(align, (bytes + align - 1) & ~(align - 1)));
            bounce_bytes = bounce ? bytes : 0;
            bounce_align = bounce ? align : 0;
        }
        return bounce;
    }
    ~Gather() { std::free(bounce); }
};

inline Gather& gather_state() {
    static thread_local Gather g;
    return g;
}

#if defined(__linux__)

/* The file behind a table: its mapping's address range, the file offset of the range's first byte,
 * an O_DIRECT fd, and the granule reads must align to. `fd` is -1 for memory with no file. */
struct TableFile {
    const uint8_t* lo    = nullptr;
    const uint8_t* hi    = nullptr;
    int            fd    = -1;
    long long      off   = 0;
    long long      align = 0;
};

/* Granule of a direct read of `t`'s file (`t.fd`): the alignment the file offset, length and buffer
 * address must all keep. Returns 0, with `*why`, if the file cannot be read that way. [p, p + n) is
 * the table inside the mapping; `ino` is the mapped file's inode.
 *
 * STATX_DIOALIGN is used when reported. btrfs (and tmpfs since Linux 6.6) does direct reads but
 * reports nothing; then the granule is stx_blksize. No block filesystem needs more (blocks are never
 * smaller than device sectors), and btrfs needs exactly its sector size: below it, it silently
 * serves direct reads through the page cache.
 *
 * One test read then proves the granule: a span at an offset and into a buffer aligned to the
 * granule and no more, which must return the bytes the mapping holds. This catches a refused
 * granule and a file that is not the one mapped. */
long long direct_granule(const TableFile& t, const uint8_t* p, long long n, unsigned long long ino,
                         const char* path, const char** why) {
    struct statx sx;
    std::memset(&sx, 0, sizeof sx);
    if (::statx(t.fd, "", AT_EMPTY_PATH, STATX_INO | STATX_DIOALIGN, &sx) != 0) {
        *why = std::strerror(errno);
        return 0;
    }
    if (!(sx.stx_mask & STATX_INO) || sx.stx_ino != ino) {
        *why = "the file opened is not the one mapped";
        return 0;
    }
    const bool reported = (sx.stx_mask & STATX_DIOALIGN) != 0;
    if (reported && sx.stx_dio_offset_align == 0) {
        *why = "the filesystem says this file has no direct I/O";
        return 0;
    }
    const long long a = !reported ? (long long)sx.stx_blksize
                      : sx.stx_dio_offset_align > sx.stx_dio_mem_align
                      ? (long long)sx.stx_dio_offset_align : (long long)sx.stx_dio_mem_align;
    if (a <= 0 || (a & (a - 1))) {
        *why = reported ? "the filesystem's direct-I/O alignment is not a power of two"
                        : "the filesystem reports no direct-I/O alignment, and its block size is "
                          "not a power of two";
        return 0;
    }

    /* An odd multiple of the granule, so neither offset nor buffer is over-aligned. */
    const long long tab_lo = t.off + (long long)(p - t.lo);
    long long o = (tab_lo + a - 1) & ~(a - 1);
    if ((o / a) % 2 == 0) o += a;
    if (o + a > tab_lo + n) {
        *why = "the table is smaller than one direct read";
        return 0;
    }
    uint8_t* buf = static_cast<uint8_t*>(std::aligned_alloc((size_t)(2 * a), (size_t)(2 * a)));
    if (!buf) {
        *why = "no memory for a read";
        return 0;
    }
    const ssize_t got = ::pread(t.fd, buf + a, (size_t)a, (off_t)o);
    const int e = errno;
    const bool same = got == a && std::memcmp(buf + a, t.lo + (o - t.off), (size_t)a) == 0;
    std::free(buf);
    if (!same) {
        *why = got < 0  ? std::strerror(e)
             : got != a ? "a read of one granule came back short"
                        : "a read returned other bytes than the mapping holds";
        return 0;
    }
    if (!reported)
        std::fprintf(stderr, "libavx: %s: the filesystem reports no direct-I/O alignment, so the "
                     "n-gram table is read in spans of its block size, %lld bytes; a direct read "
                     "of one returned the file's bytes\n", path, a);
    return a;
}

/* The file behind [p, p + n): the mapping holding `p`, extended over following mappings of the same
 * file (madvise on a sub-range splits a mapping). Looked up once per table and kept for the life of
 * the process; rank threads share the entry and fd (reads use explicit offsets). */
const TableFile* table_file(const uint8_t* p, long long n, int* err) {
    static std::mutex mu;
    static std::vector<std::unique_ptr<TableFile>> known;
    std::lock_guard<std::mutex> lk(mu);
    for (const auto& t : known)
        if (p >= t->lo && p + n <= t->hi) return t.get();

    auto t = std::make_unique<TableFile>();
    t->lo = p;
    t->hi = p + n;
    std::FILE* f = std::fopen("/proc/self/maps", "re");
    if (!f) { *err = RAD_E_IO; return nullptr; }
    char line[4352];
    bool found = false;
    uintptr_t end = 0;
    unsigned long long ino = 0, next_off = 0;
    std::string path;
    while (std::fgets(line, sizeof line, f)) {
        unsigned long long lo = 0, hi = 0, off = 0, inode = 0;
        char perms[8];
        unsigned dmaj = 0, dmin = 0;
        int at = 0;
        if (std::sscanf(line, "%llx-%llx %7s %llx %x:%x %llu %n", &lo, &hi, perms, &off, &dmaj,
                        &dmin, &inode, &at) < 7)
            continue;
        std::string name(line + at);
        while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) name.pop_back();
        if (!found) {
            if ((uintptr_t)p < lo || (uintptr_t)p >= hi) continue;
            if (inode == 0 || name.empty() || name[0] != '/') break;     /* not file-backed */
            found = true;
            t->lo = (const uint8_t*)(uintptr_t)lo;
            t->off = (long long)off;
            end = (uintptr_t)hi;
            ino = inode;
            next_off = off + (hi - lo);
            path = name;
        } else if ((uintptr_t)lo == end && inode == ino && name == path && off == next_off) {
            end = (uintptr_t)hi;
            next_off = off + (hi - lo);
        } else {
            break;
        }
        if ((uintptr_t)(p + n) <= end) break;
    }
    std::fclose(f);
    if (found) {
        if ((uintptr_t)(p + n) > end) { *err = RAD_E_IO; return nullptr; }
        t->hi = (const uint8_t*)end;
        t->fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        const char* why = t->fd < 0 ? std::strerror(errno) : nullptr;
        if (t->fd >= 0) t->align = direct_granule(*t, p, n, ino, path.c_str(), &why);
        if (t->align == 0) {
            std::fprintf(stderr, "libavx: %s cannot be read with direct I/O (%s); the n-gram table "
                         "is read that way. A local filesystem with direct I/O -- ext4, XFS, "
                         "btrfs -- can hold the container\n", path.c_str(), why);
            if (t->fd >= 0) ::close(t->fd);
            *err = RAD_E_UNSUPPORTED;
            return nullptr;
        }
    }
    known.push_back(std::move(t));
    return known.back().get();
}

/* ============================== the row cache ==============================
 *
 * Set-associative: a row's key (its byte offset in the file) hashes to a set of kWays slots; a slot
 * is one atomic word plus the row's bytes. The word is the slot's whole state:
 *
 *      key << 10 | owner << 2 | state          state: EMPTY (word 0), LOADING, READY
 *
 * `owner` is the loading thread, so a thread meeting its own claim (a row twice in one call) does
 * not wait on itself. Readers copy a READY slot's bytes and re-read the word: bytes are rewritten
 * only after the word leaves READY, so an unchanged word means a valid copy (a seqlock). Victims are
 * chosen by CLOCK over per-slot reference bits kept outside the word (so marking a row used never
 * fails a reader's check); LOADING slots are never victims.
 *
 * A set's eight words are one 64-byte line. The arrays are anonymous mappings, so memory is
 * committed only as slots fill. */
struct RowCache {
    static constexpr int      kWays = 8;
    static constexpr uint64_t kRows = 1ull << 19;
    enum : uint64_t { EMPTY = 0, LOADING = 1, READY = 2 };

    const TableFile*       tf       = nullptr;
    long long              n        = 0;
    uint64_t               set_mask = 0;
    std::atomic<uint64_t>* word     = nullptr;
    std::atomic<uint8_t>*  ref      = nullptr;
    std::atomic<uint8_t>*  hand     = nullptr;
    uint8_t*               data     = nullptr;
    std::atomic<uint32_t>  published{0};    /* bumped (and waited on) as each batch lands */

    static uint64_t ready(uint64_t key) { return key << 10 | READY; }

    static uint64_t mix(uint64_t x) {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
        x ^= x >> 27; x *= 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    const void* set_line(uint64_t key) const { return &word[(mix(key) & set_mask) * kWays]; }

    enum Found { HIT, CLAIMED, MINE, THEIRS, NONE };

    /* The slot holding `key`: READY (HIT), loading by this thread (MINE) or another (THEIRS); or a
     * slot claimed for it now (CLAIMED); or NONE when every slot of the set is loading. */
    Found find_or_claim(uint64_t key, uint64_t me, int64_t* slot) {
        const uint64_t set = mix(key) & set_mask;
        const uint64_t s0  = set * kWays;
        for (;;) {
            int64_t  victim = -1;
            uint64_t vw     = 0;
            for (int w = 0; w < kWays; ++w) {
                const uint64_t x = word[s0 + w].load(std::memory_order_acquire);
                if (x != EMPTY && (x >> 10) == key) {
                    *slot = (int64_t)(s0 + w);
                    if ((x & 3) == READY) {
                        ref[s0 + w].store(1, std::memory_order_relaxed);
                        return HIT;
                    }
                    return ((x >> 2) & 255) == me ? MINE : THEIRS;
                }
                if (x == EMPTY && victim < 0) victim = (int64_t)(s0 + w);
            }
            /* CLOCK: clear reference bits until a READY slot without one comes round. Two turns
             * clear them all, so this fails only if every slot is LOADING. */
            for (int k = 0; victim < 0 && k < 2 * kWays; ++k) {
                const uint8_t  h = hand[set].load(std::memory_order_relaxed);
                hand[set].store((uint8_t)((h + 1) % kWays), std::memory_order_relaxed);
                const uint64_t s = s0 + h % kWays;
                const uint64_t x = word[s].load(std::memory_order_acquire);
                if ((x & 3) != READY) continue;
                if (ref[s].load(std::memory_order_relaxed)) {
                    ref[s].store(0, std::memory_order_relaxed);
                    continue;
                }
                victim = (int64_t)s;
                vw = x;
            }
            if (victim < 0) { *slot = -1; return NONE; }
            if (word[victim].compare_exchange_strong(vw, key << 10 | me << 2 | LOADING,
                                                     std::memory_order_acq_rel)) {
                /* Bytes are rewritten only after this (seqlock, above). */
                std::atomic_thread_fence(std::memory_order_release);
                *slot = victim;
                return CLAIMED;
            }
            /* The set changed between the scan and the claim: retry. */
        }
    }

    /* Where `key` sits, without claiming it: HIT with *slot (decode from it), THEIRS with *slot (being
     * read), or NONE with *slot = -1. */
    Found lookup(uint64_t key, int64_t* slot) const {
        const uint64_t set = mix(key) & set_mask;
        const uint64_t s0  = set * kWays;
        for (int w = 0; w < kWays; ++w) {
            const uint64_t x = word[s0 + w].load(std::memory_order_acquire);
            if (x != EMPTY && (x >> 10) == key) {
                *slot = (int64_t)(s0 + w);
                return (x & 3) == READY ? HIT : THEIRS;
            }
        }
        *slot = -1;
        return NONE;
    }
};

template <class T>
T* anon_array(size_t count) {
    void* p = mmap(nullptr, count * sizeof(T), PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<T*>(p);
}

/* The cache for `n`-byte rows of `tf`, created on first use and kept for the process lifetime. */
RowCache* row_cache(const TableFile* tf, long long n) {
    static std::mutex mu;
    static std::vector<std::unique_ptr<RowCache>> caches;
    std::lock_guard<std::mutex> lk(mu);
    for (const auto& c : caches)
        if (c->tf == tf && c->n == n) return c.get();
    auto c = std::make_unique<RowCache>();
    c->tf       = tf;
    c->n        = n;
    c->set_mask = RowCache::kRows / RowCache::kWays - 1;
    c->word     = anon_array<std::atomic<uint64_t>>(RowCache::kRows);
    c->ref      = anon_array<std::atomic<uint8_t>>(RowCache::kRows);
    c->hand     = anon_array<std::atomic<uint8_t>>(RowCache::kRows / RowCache::kWays);
    c->data     = anon_array<uint8_t>(RowCache::kRows * (size_t)n);
    if (!c->word || !c->ref || !c->hand || !c->data) return nullptr;
    caches.push_back(std::move(c));
    return caches.back().get();
}

/* This thread's owner tag for LOADING words, 1..255, or 0 while 255 live threads hold one (such a
 * thread cannot gather through the cache). A thread holds LOADING words only inside a call, so its
 * tag is returned for reuse when it exits. */
inline uint64_t owner_tag() {
    struct Pool {
        std::mutex mu;
        std::vector<uint32_t> free;
        uint32_t next = 0;
    };
    static Pool& pool = *new Pool;     /* never destroyed: threads may exit after statics are */
    struct Tag {
        uint32_t v = 0;
        Tag() {
            std::lock_guard<std::mutex> lk(pool.mu);
            if (!pool.free.empty()) { v = pool.free.back(); pool.free.pop_back(); }
            else if (pool.next < 255) v = ++pool.next;
        }
        ~Tag() {
            if (!v) return;
            std::lock_guard<std::mutex> lk(pool.mu);
            pool.free.push_back(v);
        }
    };
    static thread_local const Tag me;
    return me.v;
}

struct ReadRing {
    static constexpr unsigned kEntries = 4096;
    int            fd       = -1;
    unsigned       entries  = 0;
    unsigned      *sq_tail  = nullptr, *sq_mask = nullptr, *sq_array = nullptr;
    unsigned      *cq_head  = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
    io_uring_sqe  *sqes     = nullptr;
    io_uring_cqe  *cqes     = nullptr;
    void          *sq_ring  = nullptr, *cq_ring = nullptr;
    size_t         sq_sz = 0, cq_sz = 0, sqe_sz = 0;
    int            err      = 0;     /* errno from a failed init(), for ring_refusal() */

    /* SINGLE_ISSUER and DEFER_TASKRUN: the ring is thread-local, and its thread only waits for
     * batches it has just submitted, so completions run when it asks for them. */
    bool init() {
        io_uring_params p;
        std::memset(&p, 0, sizeof p);
        p.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
        const long r = syscall(__NR_io_uring_setup, kEntries, &p);
        if (r < 0) { err = errno; return false; }
        fd      = (int)r;
        entries = p.sq_entries;
        sq_sz   = p.sq_off.array + (size_t)p.sq_entries * sizeof(unsigned);
        cq_sz   = p.cq_off.cqes  + (size_t)p.cq_entries * sizeof(io_uring_cqe);
        if (p.features & IORING_FEAT_SINGLE_MMAP) {
            if (cq_sz > sq_sz) sq_sz = cq_sz;
            cq_sz = sq_sz;
        }
        sq_ring = mmap(nullptr, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                       fd, IORING_OFF_SQ_RING);
        if (sq_ring == MAP_FAILED) { err = errno; sq_ring = nullptr; shut(); return false; }
        cq_ring = (p.features & IORING_FEAT_SINGLE_MMAP)
                ? sq_ring
                : mmap(nullptr, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                       fd, IORING_OFF_CQ_RING);
        if (cq_ring == MAP_FAILED) { err = errno; cq_ring = nullptr; shut(); return false; }
        sqe_sz = (size_t)p.sq_entries * sizeof(io_uring_sqe);
        sqes   = (io_uring_sqe*)mmap(nullptr, sqe_sz, PROT_READ | PROT_WRITE,
                                     MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
        if (sqes == MAP_FAILED) { err = errno; sqes = nullptr; shut(); return false; }
        sq_tail  = (unsigned*)((char*)sq_ring + p.sq_off.tail);
        sq_mask  = (unsigned*)((char*)sq_ring + p.sq_off.ring_mask);
        sq_array = (unsigned*)((char*)sq_ring + p.sq_off.array);
        cq_head  = (unsigned*)((char*)cq_ring + p.cq_off.head);
        cq_tail  = (unsigned*)((char*)cq_ring + p.cq_off.tail);
        cq_mask  = (unsigned*)((char*)cq_ring + p.cq_off.ring_mask);
        cqes     = (io_uring_cqe*)((char*)cq_ring + p.cq_off.cqes);
        return true;
    }

    void shut() {
        if (sqes    && sqes    != MAP_FAILED) munmap(sqes, sqe_sz);
        if (cq_ring && cq_ring != sq_ring)    munmap(cq_ring, cq_sz);
        if (sq_ring)                          munmap(sq_ring, sq_sz);
        if (fd >= 0)                          ::close(fd);
        sqes = nullptr; sq_ring = cq_ring = nullptr; fd = -1;
    }
    ~ReadRing() { shut(); }

    /* Reads the aligned span holding each row of `r[0, count)` (count <= entries) into `bounce`,
     * one span per `stride` bytes, submitting and waiting in one call. Short reads fail. */
    int read_spans(const TableFile& tf, const SpanRead* r, unsigned count, long long n,
                   long long stride, uint8_t* bounce) {
        const long long A = tf.align;
        unsigned tail = __atomic_load_n(sq_tail, __ATOMIC_RELAXED);
        for (unsigned i = 0; i < count; ++i) {
            const long long a = (long long)r[i].key & ~(A - 1);
            const long long e = ((long long)r[i].key + n + A - 1) & ~(A - 1);
            const unsigned idx = tail & *sq_mask;
            io_uring_sqe* s = &sqes[idx];
            std::memset(s, 0, sizeof *s);
            s->opcode    = IORING_OP_READ;
            s->fd        = tf.fd;
            s->off       = (uint64_t)a;
            s->addr      = (uint64_t)(uintptr_t)(bounce + (size_t)i * stride);
            s->len       = (unsigned)(e - a);
            s->user_data = (uint64_t)(e - a);
            sq_array[idx] = idx;
            ++tail;
        }
        __atomic_store_n(sq_tail, tail, __ATOMIC_RELEASE);
        int status = RAD_OK;
        unsigned to_submit = count, got = 0;
        while (got < count) {
            const long rc = syscall(__NR_io_uring_enter, fd, to_submit, count - got,
                                    IORING_ENTER_GETEVENTS, nullptr, (size_t)0);
            if (rc < 0 && errno != EINTR && errno != EAGAIN && errno != EBUSY) return RAD_E_IO;
            if (rc > 0) to_submit -= (unsigned)rc < to_submit ? (unsigned)rc : to_submit;
            unsigned h = __atomic_load_n(cq_head, __ATOMIC_RELAXED);
            const unsigned tl = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
            for (; h != tl; ++h, ++got) {
                const io_uring_cqe& c = cqes[h & *cq_mask];
                if (c.res < 0 || (uint64_t)c.res != c.user_data) status = RAD_E_IO;
            }
            __atomic_store_n(cq_head, h, __ATOMIC_RELEASE);
        }
        return status;
    }
};

/* This thread's ring, created on its first gather over a file and destroyed at thread exit (each
 * rank thread has its own, so they never contend). Null if it could not be made, with the errno in
 * `*why`. */
inline ReadRing* read_ring(int* why) {
    static thread_local bool tried = false;
    static thread_local int  err   = 0;
    static thread_local std::unique_ptr<ReadRing> r;
    if (!tried) {
        tried = true;
        std::unique_ptr<ReadRing> p(new ReadRing());
        if (p->init()) r = std::move(p);
        else           err = p->err ? p->err : EIO;
    }
    if (!r && why) *why = err;
    return r.get();
}

/* Explains a ring failure in terms of the fix. EPERM: Docker's default seccomp profile or the
 * kernel.io_uring_disabled sysctl. EINVAL: a kernel without SINGLE_ISSUER / DEFER_TASKRUN.
 * ENOMEM: the ring's memory refused, e.g. over a locked-memory limit shared with other processes. */
std::string ring_refusal(int e) {
    std::string s = "libavx: the n-gram table is read through io_uring, and this process cannot "
                    "make a ring (" + std::string(std::strerror(e)) + "). ";
    if (e == EPERM || e == EACCES)
        s += "Docker's default seccomp profile refuses io_uring: run the container with "
             "--security-opt seccomp=unconfined (docs/DOCKER.md). Outside a container, the "
             "kernel.io_uring_disabled sysctl refuses it the same way.";
    else if (e == ENOSYS)
        s += "This kernel is built without io_uring.";
    else if (e == EINVAL)
        s += "The ring needs IORING_SETUP_SINGLE_ISSUER and IORING_SETUP_DEFER_TASKRUN, which "
             "Linux 6.1 added.";
    else if (e == ENOMEM)
        s += "The kernel refused the ring's memory. It can count against the locked-memory limit "
             "(ulimit -l), which every process of this user shares: raise it (LimitMEMLOCK= in a "
             "systemd unit, --ulimit memlock=-1 for Docker) or stop other io_uring users.";
    return s;
}

/* 0 if this process can create a ring, else the errno. Tested once by creating and closing one. */
int ring_errno() {
    static const int e = [] {
        ReadRing r;
        return r.init() ? 0 : (r.err ? r.err : EIO);
    }();
    return e;
}

/* Rows read from table files by this process, every thread together (avx_ngram_reads). */
std::atomic<uint64_t> g_rows_read{0};

/* Reads `reads[0, total)` through the ring in batches: rows with a slot are copied into the cache
 * and published READY; rows without one are decoded directly into their output rows. On a failed
 * read, every unpublished claim is released (word back to EMPTY) so no thread waits on it. */
int read_and_publish(const TableFile& tf, RowCache& rc, ReadRing& rr, Gather& g,
                     const SpanRead* reads, size_t total, long long n, long long ne,
                     avx::RowDecode decode, const uint16_t* lut, uint16_t* out, long long x_ld) {
    const long long A      = tf.align;
    const long long stride = ((n + A - 1) / A + 1) * A;
    for (size_t b = 0; b < total;) {
        const unsigned cnt = (unsigned)(total - b < rr.entries ? total - b : rr.entries);
        uint8_t* bounce = g.bounce_for((size_t)cnt * (size_t)stride, (size_t)A);
        const int rs = bounce ? rr.read_spans(tf, &reads[b], cnt, n, stride, bounce) : RAD_E_NOMEM;
        if (rs != RAD_OK) {
            for (size_t i = b; i < total; ++i)
                if (reads[i].slot >= 0)
                    rc.word[reads[i].slot].store(RowCache::EMPTY, std::memory_order_release);
            rc.published.fetch_add(1, std::memory_order_release);
            rc.published.notify_all();
            return rs;
        }
        g_rows_read.fetch_add(cnt, std::memory_order_relaxed);
        g.bsrc.clear();
        g.bdst.clear();
        for (unsigned i = 0; i < cnt; ++i) {
            const SpanRead& r = reads[b + i];
            const uint8_t* row = bounce + (size_t)i * stride + (r.key & (uint64_t)(A - 1));
            if (r.slot < 0) {
                g.bsrc.push_back(row);
                g.bdst.push_back(out + r.m * x_ld);
                continue;
            }
            std::memcpy(rc.data + (size_t)r.slot * (size_t)n, row, (size_t)n);
            rc.ref[r.slot].store(1, std::memory_order_relaxed);
            rc.word[r.slot].store(RowCache::ready(r.key), std::memory_order_release);
        }
        rc.published.fetch_add(1, std::memory_order_release);
        rc.published.notify_all();
        if (!g.bsrc.empty()) decode(g.bsrc.data(), g.bdst.data(), (int64_t)g.bsrc.size(), ne, lut);
        b += cnt;
    }
    return RAD_OK;
}

/* The decode regime: a step wants a hundred-odd rows now, and never waits on the reader thread's
 * read. Rows READY in the cache decode from it; a missing row whose slot is free to take is claimed
 * and read into the cache, so the ranks' gathers and the later steps that meet the same trigram
 * find it READY instead of reading the file again (Flash-Next decode at c4 lost 2.5% when these
 * reads were never cached); a row the reader thread holds, or one with no slot, is read straight
 * into its output row, and one another rank's gather is reading is waited for. The ranks split a
 * step's missing rows between them (gather_ondemand). */
constexpr long long kOnDemandMaxRows = 4096;

/* Decode-sized gathers reading now. The reader thread starts no batch while any is (see the reader
 * below); such a gather never waits on another thread's claim, so the pause cannot close a cycle. */
std::atomic<int> g_urgent_reads{0};

/* The read-ahead reader thread's owner tag (0 before it starts): a decode gather reads a row that
 * thread holds directly instead of waiting on it, so the g_urgent_reads pause cannot close a cycle. */
std::atomic<uint64_t> g_reader_tag{0};

struct UrgentReads {
    UrgentReads() { g_urgent_reads.fetch_add(1, std::memory_order_acq_rel); }
    ~UrgentReads() {
        if (g_urgent_reads.fetch_sub(1, std::memory_order_acq_rel) == 1) g_urgent_reads.notify_all();
    }
};

int gather_ondemand(const TableFile& tf, RowCache& rc, ReadRing& rr, const avx::RowGather& q,
                    avx::RowDecode decode, const uint16_t* lut) {
    Gather& g = gather_state();
    const uint64_t me = owner_tag();
    if (!me) return RAD_E_UNSUPPORTED;
    const long long M = q.M, n = q.row_bytes, ne = q.n_embd, x_ld = q.x_ld;
    const long long base = tf.off + (long long)(q.tab - tf.lo);
    auto row_of = [&](long long m) { return (long long)q.ids[m] - q.voff; };
    auto pad    = [&](long long id) { return id < 0 || id >= q.n_vocab; };
    auto key_of = [&](long long m) { return (uint64_t)(base + row_of(m) * n); };

    g.src.clear();
    g.dst.clear();
    g.dm.clear();
    g.dw.clear();
    g.slot.resize((size_t)M);
    g.reads.clear();
    g.again.clear();
    g.waits.clear();
    g.later.clear();
    const uint64_t reader = g_reader_tag.load(std::memory_order_acquire);
    auto hit = [&](long long m, int64_t s, uint64_t key) {
        g.src.push_back(rc.data + (size_t)s * (size_t)n);
        g.dst.push_back(q.out + m * x_ld);
        g.dm.push_back(m);
        g.dw.push_back(RowCache::ready(key));
    };
    auto theirs_waits = [&](int64_t s) {
        return s >= 0 && ((rc.word[s].load(std::memory_order_acquire) >> 2) & 255) != reader;
    };
    /* Finds or claims row m: READY decodes from the cache, a row another rank thread is reading is
     * waited for below instead of read a second time (both ranks gather the same rows, so reading
     * it again doubled decode's reads), and the reader's THEIRS, MINE (a row twice in one call)
     * and NONE are read directly. */
    auto claim = [&](long long m) {
        const uint64_t key = key_of(m);
        int64_t s = -1;
        const RowCache::Found f = rc.find_or_claim(key, me, &s);
        g.slot[m] = s;
        if (f == RowCache::HIT) hit(m, s, key);
        else if (f == RowCache::THEIRS && theirs_waits(s)) g.waits.push_back(m);
        else g.reads.push_back({m, key, f == RowCache::CLAIMED ? s : -1});
    };
    /* A direct read costs its thread a few microseconds to submit and to complete, and the ranks'
     * threads gather the same rows at the same moment: whichever scanned first used to claim every
     * missing row and read them all alone while the other waited (47 rows, 380 us, on Flash-Next
     * at one sequence under the draft head, longer than the first layer runs: the card idled). So
     * a rank first claims only its share, the rows m with m % world_size == rank, and leaves the
     * rest, which it takes up after its own reads: by then the other ranks have claimed theirs and
     * it waits for those, and whatever nobody claimed (a rank that is not there) it reads itself. */
    const long long share = q.world_size > 1 ? q.world_size : 1;
    for (long long m = 0; m < M; ++m) {
        if (pad(row_of(m))) {
            std::memset(q.out + m * x_ld, 0, (size_t)ne * sizeof(uint16_t));
            continue;
        }
        if (m % share == q.rank % share) { claim(m); continue; }
        const uint64_t key = key_of(m);
        int64_t s = -1;
        const RowCache::Found f = rc.lookup(key, &s);
        g.slot[m] = s;
        if (f == RowCache::HIT) {
            rc.ref[s].store(1, std::memory_order_relaxed);
            hit(m, s, key);
        } else if (f == RowCache::THEIRS && theirs_waits(s)) {
            g.waits.push_back(m);
        } else {
            g.later.push_back(m);
        }
    }
    const UrgentReads urgent;
    /* This rank's share, then what is left of the others' (see above): each batch is read and
     * published before the next is claimed. */
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            g.reads.clear();
            for (long long m : g.later) claim(m);
        }
        if (g.reads.empty()) continue;
        const int rs = read_and_publish(tf, rc, rr, g, g.reads.data(), g.reads.size(), n, ne,
                                        decode, lut, q.out, x_ld);
        if (rs != RAD_OK) return rs;
        /* The claimed rows were published into the cache, not decoded: decode them from it. */
        for (const SpanRead& r : g.reads) {
            if (r.slot < 0) continue;
            g.src.push_back(rc.data + (size_t)r.slot * (size_t)n);
            g.dst.push_back(q.out + r.m * x_ld);
            g.dm.push_back(r.m);
            g.dw.push_back(RowCache::ready(r.key));
        }
    }
    /* Wait for the other ranks' reads, as gather_rows does. This thread's own reads are done, so
     * no cycle: every rank reads its claims before it waits. A row that turns READY decodes from
     * the cache; one released or taken over is read directly below. */
    while (!g.waits.empty()) {
        const uint32_t e = rc.published.load(std::memory_order_acquire);
        size_t keep = 0;
        for (long long m : g.waits) {
            const uint64_t key = key_of(m);
            const uint64_t x   = rc.word[g.slot[m]].load(std::memory_order_acquire);
            if (x == RowCache::ready(key)) {
                g.src.push_back(rc.data + (size_t)g.slot[m] * (size_t)n);
                g.dst.push_back(q.out + m * x_ld);
                g.dm.push_back(m);
                g.dw.push_back(x);
                continue;
            }
            if ((x & 3) == RowCache::LOADING && (x >> 10) == key) { g.waits[keep++] = m; continue; }
            g.again.push_back(m);
        }
        g.waits.resize(keep);
        if (keep) rc.published.wait(e, std::memory_order_acquire);
    }
    if (!g.src.empty()) {
        /* The seqlock, as gather_rows reads it: a row whose word changed during the decode was
         * taken over, and is read again directly. */
        decode(g.src.data(), g.dst.data(), (int64_t)g.src.size(), ne, lut);
        std::atomic_thread_fence(std::memory_order_acquire);
        for (size_t i = 0; i < g.dm.size(); ++i)
            if (rc.word[g.slot[g.dm[i]]].load(std::memory_order_relaxed) != g.dw[i])
                g.again.push_back(g.dm[i]);
    }
    if (g.again.empty()) return RAD_OK;
    g.reads.clear();
    for (long long m : g.again) g.reads.push_back({m, key_of(m), -1});
    return read_and_publish(tf, rc, rr, g, g.reads.data(), g.reads.size(), n, ne, decode, lut,
                            q.out, x_ld);
}

/* out[m] = decoded row ids[m] - voff for every m < M, through the cache: find or claim each row
 * starting at row `first`, read this thread's claims, wait for rows other threads are reading, then
 * decode from the cache. A row whose slot is taken over during the decode is read again directly. */
int gather_rows(const TableFile& tf, RowCache& rc, ReadRing& rr, const avx::RowGather& q,
                long long first, avx::RowDecode decode, const uint16_t* lut) {
    /* A decode step's rows are read on demand; a prefill chunk's tens of thousands, whose future is
     * known, go through the claims and the read-ahead queue. */
    if (q.M <= kOnDemandMaxRows) return gather_ondemand(tf, rc, rr, q, decode, lut);
    Gather& g = gather_state();
    const uint64_t me = owner_tag();
    if (!me) return RAD_E_UNSUPPORTED;
    /* `n` is a row's bytes (cache and file offsets); `ne` its elements (decode and padding). */
    const long long M = q.M, n = q.row_bytes, ne = q.n_embd, x_ld = q.x_ld;
    const long long base = tf.off + (long long)(q.tab - tf.lo);
    auto row_of = [&](long long m) { return (long long)q.ids[m] - q.voff; };
    auto pad    = [&](long long id) { return id < 0 || id >= q.n_vocab; };
    auto key_of = [&](long long m) { return (uint64_t)(base + row_of(m) * n); };
    auto at     = [&](long long j) { return first + j < M ? first + j : first + j - M; };

    g.slot.resize((size_t)M);
    g.kind.resize((size_t)M);
    g.reads.clear();
    g.waits.clear();
    g.again.clear();

    for (long long j = 0; j < M; ++j) {
        if (j + kPrefetchRows < M) {
            const long long mp = at(j + kPrefetchRows);
            if (!pad(row_of(mp))) __builtin_prefetch(rc.set_line(key_of(mp)));
        }
        const long long m = at(j);
        if (pad(row_of(m))) { g.kind[m] = Gather::PAD; continue; }
        const uint64_t key = key_of(m);
        int64_t s = -1;
        const RowCache::Found f = rc.find_or_claim(key, me, &s);
        g.slot[m] = s;
        g.kind[m] = (uint8_t)f;
        if (f == RowCache::CLAIMED)     g.reads.push_back({m, key, s});
        else if (f == RowCache::NONE) { g.reads.push_back({m, key, -1}); g.kind[m] = Gather::DONE; }
        else if (f == RowCache::THEIRS) g.waits.push_back(m);
    }

    int rs = read_and_publish(tf, rc, rr, g, g.reads.data(), g.reads.size(), n, ne, decode, lut,
                              q.out, x_ld);
    if (rs != RAD_OK) return rs;

    /* Wait for rows other threads claimed to turn READY. `published` is read before the words, so
     * a batch landing after the scan makes the wait return at once. Released claims (failed reads)
     * and reused slots are read by this thread instead. */
    while (!g.waits.empty()) {
        const uint32_t e = rc.published.load(std::memory_order_acquire);
        size_t keep = 0;
        for (long long m : g.waits) {
            const uint64_t key = key_of(m);
            const uint64_t x   = rc.word[g.slot[m]].load(std::memory_order_acquire);
            if (x == RowCache::ready(key)) continue;
            if ((x & 3) == RowCache::LOADING && (x >> 10) == key) { g.waits[keep++] = m; continue; }
            g.again.push_back(m);
            g.kind[m] = Gather::DONE;
        }
        g.waits.resize(keep);
        if (keep) rc.published.wait(e, std::memory_order_acquire);
    }

    /* Decode all READY rows in one batch: each word is read before the bytes and again after the
     * batch (the seqlock), and rows whose word changed are read again. */
    g.src.clear();
    g.dst.clear();
    g.dm.clear();
    g.dw.clear();
    for (long long m = 0; m < M; ++m) {
        uint16_t* dst = q.out + m * x_ld;
        const uint8_t k = g.kind[m];
        if (k == Gather::PAD)  { std::memset(dst, 0, (size_t)ne * sizeof(uint16_t)); continue; }
        if (k == Gather::DONE) continue;
        const int64_t  s  = g.slot[m];
        const uint64_t w1 = rc.word[s].load(std::memory_order_acquire);
        if (w1 != RowCache::ready(key_of(m))) { g.again.push_back(m); continue; }
        g.src.push_back(rc.data + (size_t)s * (size_t)n);
        g.dst.push_back(dst);
        g.dm.push_back(m);
        g.dw.push_back(w1);
    }
    if (!g.src.empty()) decode(g.src.data(), g.dst.data(), (int64_t)g.src.size(), ne, lut);
    std::atomic_thread_fence(std::memory_order_acquire);
    for (size_t i = 0; i < g.dm.size(); ++i)
        if (rc.word[g.slot[g.dm[i]]].load(std::memory_order_relaxed) != g.dw[i])
            g.again.push_back(g.dm[i]);

    if (g.again.empty()) return RAD_OK;
    g.reads.clear();
    for (long long m : g.again) g.reads.push_back({m, key_of(m), -1});
    return read_and_publish(tf, rc, rr, g, g.reads.data(), g.reads.size(), n, ne, decode, lut,
                            q.out, x_ld);
}

/* ============================== the reader that runs a step ahead ==============================
 *
 * A prefill chunk gathers `heads` rows per token (32,768 for a 2048-token chunk); on unseen text
 * most are file reads (~30 ms per chunk on NVMe), during which the issuing thread waits and the card
 * idles. The next chunks' ids are known a step early (RadBatch::n_ahead, ngram_ids' `ahead_ids`), so
 * the call hands them to one process-wide reader thread, which claims, reads and publishes the
 * missing rows while the chunk runs. The next call waits for LOADING rows as it would for the other
 * rank's.
 *
 * The rows go over in jobs of at most kJobRows keys, claimed and published one job at a time in
 * list order. A claim is a promise a gather may wait on, so jobs stay small: a gather that outruns
 * the reader finds unclaimed rows and reads them itself.
 *
 * Each id is handed over once, and only as far ahead as the cache can hold it. A call hands over
 * at most kAheadRows ids, a quarter of the cache, so a row read ahead is still there when its chunk
 * gathers it. Each step's list is the previous one moved on by the chunk just gathered, so only the
 * ids past the end of the previous list are new; both ranks hand over the same list, and a repeated
 * list adds nothing. A list that does not continue the previous one (another request's prompt)
 * drops the jobs still queued for the previous one.
 *
 * The reader keeps up to kReaderBatch reads in the drive, starting one as each lands, and starts
 * none while a decode-sized gather is reading (g_urgent_reads), so a decode step's few reads do not queue in the drive behind a
 * prefetch's thousands. A failed read releases its claims and the next call reads those rows
 * itself. The reader is detached and never destroyed. */

/* Ids of the following prompt a call hands over, at most: a quarter of the cache. */
constexpr long long kAheadRows = (long long)(RowCache::kRows / 4);

/* Keys per reader job, about two ring batches: a job's claims are promises a consuming gather may
 * wait on, so it stays short. */
constexpr size_t kJobRows = 8192;

/* Reads the reader has in the drive at once. A drive serves a batch in arrival order, so this is
 * the most a decode step's reads can find ahead of them; 128 still keeps an NVMe queue busy. */
constexpr size_t kReaderBatch = 128;

/* The reader's claimed rows `reads[0, total)` (each with a slot), with at most `depth` in the drive
 * at once, a new read submitted as each completes, and each row published READY as it lands: waiting
 * for whole batches let the drive's queue drain at the end of each (40k reads/s against 46k).  While a decode-sized gather is reading, no new read starts;
 * the ones in flight drain first, so at most `depth` reads are ever ahead of a decode step's (as
 * with the stock batches). On a failed read every unpublished claim is released. */
int stream_publish(const TableFile& tf, RowCache& rc, ReadRing& rr, Gather& g,
                   const SpanRead* reads, size_t total, long long n, size_t depth) {
    const long long A      = tf.align;
    const long long stride = ((n + A - 1) / A + 1) * A;
    if (depth > rr.entries) depth = rr.entries;
    if (depth > 0xffff) depth = 0xffff;
    static thread_local std::vector<uint32_t> freebuf;
    static thread_local std::vector<uint8_t>  landed;
    landed.assign(total, 0);
    uint8_t* bounce = g.bounce_for(depth * (size_t)stride, (size_t)A);
    int status = bounce ? RAD_OK : RAD_E_NOMEM;
    freebuf.clear();
    for (size_t d = 0; d < depth; ++d) freebuf.push_back((uint32_t)(depth - 1 - d));
    size_t next = 0, inflight = 0;
    unsigned pending = 0;                       /* queued in the SQ, not yet submitted */
    while (status == RAD_OK && (next < total || inflight)) {
        if (inflight == 0 && pending == 0)
            for (int u; (u = g_urgent_reads.load(std::memory_order_acquire)) > 0;)
                g_urgent_reads.wait(u, std::memory_order_acquire);
        const bool may_start = g_urgent_reads.load(std::memory_order_acquire) == 0;
        unsigned tail = __atomic_load_n(rr.sq_tail, __ATOMIC_RELAXED);
        unsigned add = 0;
        while (may_start && next < total && !freebuf.empty()) {
            const uint32_t buf = freebuf.back();
            freebuf.pop_back();
            const SpanRead& r = reads[next];
            const long long a = (long long)r.key & ~(A - 1);
            const long long e = ((long long)r.key + n + A - 1) & ~(A - 1);
            const unsigned idx = tail & *rr.sq_mask;
            io_uring_sqe* s = &rr.sqes[idx];
            std::memset(s, 0, sizeof *s);
            s->opcode    = IORING_OP_READ;
            s->fd        = tf.fd;
            s->off       = (uint64_t)a;
            s->addr      = (uint64_t)(uintptr_t)(bounce + (size_t)buf * (size_t)stride);
            s->len       = (unsigned)(e - a);
            s->user_data = (uint64_t)next << 16 | buf;
            rr.sq_array[idx] = idx;
            ++tail; ++add; ++next; ++inflight;
        }
        if (add) __atomic_store_n(rr.sq_tail, tail, __ATOMIC_RELEASE);
        pending += add;
        if (!inflight) continue;
        const long rc_ = syscall(__NR_io_uring_enter, rr.fd, pending, 1u, IORING_ENTER_GETEVENTS,
                                 nullptr, (size_t)0);
        if (rc_ < 0 && errno != EINTR && errno != EAGAIN && errno != EBUSY) { status = RAD_E_IO; break; }
        if (rc_ > 0) pending -= (unsigned)rc_ < pending ? (unsigned)rc_ : pending;
        unsigned h = __atomic_load_n(rr.cq_head, __ATOMIC_RELAXED);
        const unsigned tl = __atomic_load_n(rr.cq_tail, __ATOMIC_ACQUIRE);
        unsigned got = 0;
        for (; h != tl; ++h) {
            const io_uring_cqe& c = rr.cqes[h & *rr.cq_mask];
            const size_t   i   = (size_t)(c.user_data >> 16);
            const uint32_t buf = (uint32_t)(c.user_data & 0xffff);
            --inflight;
            freebuf.push_back(buf);
            const SpanRead& r = reads[i];
            const long long a = (long long)r.key & ~(A - 1);
            const long long e = ((long long)r.key + n + A - 1) & ~(A - 1);
            if (c.res < 0 || (long long)c.res != e - a) { status = RAD_E_IO; continue; }
            const uint8_t* row = bounce + (size_t)buf * (size_t)stride + (r.key & (uint64_t)(A - 1));
            std::memcpy(rc.data + (size_t)r.slot * (size_t)n, row, (size_t)n);
            rc.ref[r.slot].store(1, std::memory_order_relaxed);
            rc.word[r.slot].store(RowCache::ready(r.key), std::memory_order_release);
            landed[i] = 1;
            ++got;
        }
        __atomic_store_n(rr.cq_head, h, __ATOMIC_RELEASE);
        if (got) {
            g_rows_read.fetch_add(got, std::memory_order_relaxed);
            rc.published.fetch_add(1, std::memory_order_release);
            rc.published.notify_all();
        }
    }
    /* After an error, drain what is still in flight so the ring is clean for the next job. */
    while (inflight) {
        const long rc_ = syscall(__NR_io_uring_enter, rr.fd, pending, 1u, IORING_ENTER_GETEVENTS,
                                 nullptr, (size_t)0);
        if (rc_ < 0 && errno != EINTR && errno != EAGAIN && errno != EBUSY) break;
        if (rc_ > 0) pending -= (unsigned)rc_ < pending ? (unsigned)rc_ : pending;
        unsigned h = __atomic_load_n(rr.cq_head, __ATOMIC_RELAXED);
        const unsigned tl = __atomic_load_n(rr.cq_tail, __ATOMIC_ACQUIRE);
        for (; h != tl; ++h) --inflight;
        __atomic_store_n(rr.cq_head, h, __ATOMIC_RELEASE);
    }
    if (status != RAD_OK) {
        for (size_t i = 0; i < total; ++i)
            if (!landed[i]) rc.word[reads[i].slot].store(RowCache::EMPTY, std::memory_order_release);
        rc.published.fetch_add(1, std::memory_order_release);
        rc.published.notify_all();
    }
    return status;
}

/* Tries at matching a list against the previous one (read_ahead) before it counts as new. */
constexpr int kShiftTries = 64;

/* One table's following ids as last handed over: the table's cache and offset in the file, and
 * the list, clamped to kAheadRows. */
struct AheadStream {
    const RowCache*      rc   = nullptr;
    long long            base = 0;
    std::vector<int32_t> ids;
};

struct AheadJob {
    const TableFile*      tf = nullptr;
    RowCache*             rc = nullptr;
    long long             n  = 0;
    std::vector<uint64_t> keys;
    bool                  retry = false;            /* the keys a full set refused once already */
    const AheadStream*    stream = nullptr;
};

struct AheadReader {
    std::mutex              mu;
    std::condition_variable cv;
    std::condition_variable idle;     /* the queues are empty and no job runs */
    std::deque<AheadJob>    urgent;   /* keys a full set refused, tried once more first */
    std::deque<AheadJob>    q;        /* the read-ahead queue: future chunks, in order */
    std::vector<std::unique_ptr<AheadStream>> streams;
    bool                    running = false;

    /* Called with `mu` held. */
    AheadStream* stream(const RowCache* rc, long long base) {
        for (const auto& st : streams)
            if (st->rc == rc && st->base == base) return st.get();
        streams.push_back(std::make_unique<AheadStream>());
        streams.back()->rc = rc;
        streams.back()->base = base;
        return streams.back().get();
    }

    /* Drops the queued jobs of `st`. Called with `mu` held. */
    void drop(const AheadStream* st) {
        for (std::deque<AheadJob>* d : { &urgent, &q })
            for (auto it = d->begin(); it != d->end();)
                it = it->stream == st ? d->erase(it) : it + 1;
    }

    void run() {
        Gather&        g  = gather_state();
        ReadRing*      rr = read_ring(nullptr);
        const uint64_t me = owner_tag();
        g_reader_tag.store(me, std::memory_order_release);
        for (;;) {
            AheadJob j;
            {
                std::unique_lock<std::mutex> lk(mu);
                running = false;
                if (urgent.empty() && q.empty()) idle.notify_all();
                cv.wait(lk, [&] { return !urgent.empty() || !q.empty(); });
                if (!urgent.empty()) { j = std::move(urgent.front()); urgent.pop_front(); }
                else                 { j = std::move(q.front());      q.pop_front(); }
                running = true;
            }
            if (!rr || !me) continue;
            g.reads.clear();
            std::vector<uint64_t> retry;   /* sets full of LOADING rows: claim once more */
            for (size_t i = 0; i < j.keys.size(); ++i) {
                if (i + kPrefetchRows < j.keys.size())
                    __builtin_prefetch(j.rc->set_line(j.keys[i + kPrefetchRows]));
                int64_t slot = -1;
                const auto f = j.rc->find_or_claim(j.keys[i], me, &slot);
                if (f == RowCache::CLAIMED)
                    g.reads.push_back({0, j.keys[i], slot});
                else if (f == RowCache::NONE && !j.retry)
                    retry.push_back(j.keys[i]);
            }
            /* Every read has a slot, so nothing is decoded and there is no output; a failed read
             * releases every claim not yet published. */
            stream_publish(*j.tf, *j.rc, *rr, g, g.reads.data(), g.reads.size(), j.n, kReaderBatch);
            if (!retry.empty()) {
                std::lock_guard<std::mutex> lk(mu);
                AheadJob r;
                r.tf = j.tf; r.rc = j.rc; r.n = j.n; r.keys = std::move(retry); r.retry = true;
                r.stream = j.stream;
                urgent.push_front(std::move(r));
                cv.notify_one();
            }
        }
    }
};

/* The reader, started by the first call that hands it rows; null before. */
std::atomic<AheadReader*> g_reader{nullptr};

AheadReader* ahead_reader() {
    static AheadReader* const r = [] {
        AheadReader* a = new AheadReader();
        std::thread([a] { a->run(); }).detach();
        g_reader.store(a, std::memory_order_release);
        return a;
    }();
    return r;
}

/* Hands the ids of `q.ahead` (gathered by later calls) that the previous list did not hold to the
 * reader, as kJobRows-key jobs in list order, so the nearest chunk's rows come first. */
void read_ahead(const TableFile& tf, RowCache& rc, const avx::RowGather& q) {
    const long long base = tf.off + (long long)(q.tab - tf.lo);
    const long long na   = q.n_ahead < kAheadRows ? q.n_ahead : kAheadRows;
    const int32_t*  w    = q.ahead;
    /* Keys before the lock; a padding id has none. */
    constexpr uint64_t kPad = ~0ull;
    std::vector<uint64_t> keys((size_t)na);
    for (long long m = 0; m < na; ++m) {
        const long long id = (long long)w[m] - q.voff;
        keys[(size_t)m] = id < 0 || id >= q.n_vocab ? kPad : (uint64_t)(base + id * q.row_bytes);
    }
    AheadReader* r = ahead_reader();
    std::lock_guard<std::mutex> lk(r->mu);
    AheadStream* st = r->stream(&rc, base);
    const std::vector<int32_t>& p = st->ids;
    const long long np = (long long)p.size();
    if (np == na && std::equal(w, w + na, p.data())) return;   /* the other rank's, or a repeat */
    /* The list continues the previous one moved on by `s` ids if w[0, np - s) == p[s, np); the
     * smallest such s leaves the fewest ids to hand over. An id repeats only where the text does,
     * so few shifts pass the first comparison; kShiftTries bounds a text that repeats a lot. */
    long long from = -1;
    for (long long s = 1, tries = 0; s < np && tries < kShiftTries; ++s) {
        if (p[(size_t)s] != w[0]) continue;
        ++tries;
        const long long k = std::min(np - s, na);
        if (std::equal(w, w + k, p.data() + s)) { from = k; break; }
    }
    if (from < 0) {
        r->drop(st);
        from = 0;
    }
    st->ids.assign(w, w + na);
    AheadJob j;
    j.tf = &tf;
    j.rc = &rc;
    j.n  = q.row_bytes;
    j.stream = st;
    j.keys.reserve(kJobRows);
    for (long long m = from; m < na; ++m) {
        if (keys[(size_t)m] == kPad) continue;
        j.keys.push_back(keys[(size_t)m]);
        if (j.keys.size() == kJobRows) {
            r->q.push_back(j);
            j.keys.clear();
        }
    }
    if (!j.keys.empty()) r->q.push_back(std::move(j));
    r->cv.notify_one();
}

#else   /* not Linux: no file reads; a table is read as memory */
struct TableFile { int fd = -1; };
inline const TableFile* table_file(const uint8_t*, long long, int*) { static TableFile m; return &m; }
#endif

}  // namespace

/* Rows read from table files by this process, once the reader thread has nothing queued or
 * running: rad-avx-check counts what a prefill reads with it (avx_check_rows.cpp). */
extern "C" uint64_t avx_ngram_reads(void) {
#if defined(__linux__)
    if (AheadReader* r = g_reader.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lk(r->mu);
        r->idle.wait(lk, [&] { return !r->running && r->q.empty() && r->urgent.empty(); });
    }
    return g_rows_read.load(std::memory_order_relaxed);
#else
    return 0;
#endif
}

int avx::row_gather(const RowGather& q, RowDecode decode) {
    if (!decode || !q.ids || !q.tab || !q.out || q.M <= 0 || q.n_embd <= 0 || q.n_vocab <= 0 ||
        q.row_bytes < q.n_embd)
        return RAD_E_INVAL;

    /* Cost has two regimes: repeated prompts hit the cache and read nothing, while varied serving
     * makes about half of each step's rows new NVMe reads. A benchmark that repeats one prompt
     * only measures the first. /proc/<pid>/io shows which regime applied. */

    /* bf16 value of every E4M3 code at this call's scale, built with the ABI's converters (so it
     * matches libref); the decode is a table lookup. */
    uint16_t lut[256];
    for (int b = 0; b < 256; ++b)
        lut[b] = rad_f32_to_bf16(rad_fp8e4m3_to_f32((uint8_t)b) * q.scale);

    /* File-backed tables go through the shared cache; others are read from memory. The caller
     * has already refused ids past the table, which matters once they become file offsets. */
    int ferr = RAD_OK;
    const TableFile* tfile = table_file(q.tab, q.n_vocab * q.row_bytes, &ferr);
    if (!tfile) return ferr;
#if defined(__linux__)
    if (tfile->fd >= 0) {
        int why = 0;
        ReadRing* rr = read_ring(&why);
        if (!rr) {
            static std::once_flag said;
            std::call_once(said, [why] { std::fprintf(stderr, "%s\n", ring_refusal(why).c_str()); });
            return RAD_E_UNSUPPORTED;
        }
        RowCache* rc = row_cache(tfile, q.row_bytes);
        if (!rc) return RAD_E_NOMEM;
        const long long ws    = q.world_size > 1 ? q.world_size : 1;
        const long long first = q.rank > 0 && q.rank < ws ? q.M * q.rank / ws : 0;
        const int rs = gather_rows(*tfile, *rc, *rr, q, first, decode, lut);
        if (rs != RAD_OK) return rs;
        /* Queue the rows a later call will gather, after this call's own rows. */
        if (q.ahead && q.n_ahead > 0) read_ahead(*tfile, *rc, q);
        return RAD_OK;
    }
#endif

    Gather& g = gather_state();
    g.src.clear();
    g.dst.clear();
    for (long long m = 0; m < q.M; ++m) {
        const long long id = (long long)q.ids[m] - q.voff;
        uint16_t* dst = q.out + m * q.x_ld;
        if (id < 0 || id >= q.n_vocab) {
            std::memset(dst, 0, (size_t)q.n_embd * sizeof(uint16_t));
            continue;
        }
        g.src.push_back(q.tab + id * q.row_bytes);
        g.dst.push_back(dst);
    }
    if (!g.src.empty()) decode(g.src.data(), g.dst.data(), (int64_t)g.src.size(), q.n_embd, lut);
    return RAD_OK;
}

/* ============================== the table's encoding and its one scale ==============================
 *
 * Checked at declare (spec §4.3).
 *
 * The table is row-major E4M3 with one bf16 scale for the whole table, `fp8_e4m3*bf16[*x*]` (the
 * shipped format). E4M3 carries its own exponent, a row's dynamic range is only ~2.3x, and the
 * error is flat across scales from 2^-13 to 2^-10, so a per-row scale would buy a few percent of
 * error for 1.25% more bytes. (INT8 in the same byte would do much better; that is a different
 * format.)
 *
 * The table is stored as is, so it can be read in place from the container's mapping (a relayout
 * would copy 51 GB). This hook only validates; any other quantisation is refused. The one other
 * accepted form is the checkpoint's plain bf16 table (102 GB), read the same way with no scale. */
namespace {

enum { EQ_TOK = 0, EQ_WTE, EQ_SCALE, EQ_X, EQ_AHEAD };

}  /* namespace */

/* RAD_E_UNSUPPORTED is the accepted answer: the plane is stored as is (rad_abi.h, RadLayoutFn). */
extern "C" int avx_layout_ngram(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                const int* sel, const RadTensor* pl, int n, RadLayout*) {
    if (operand != EQ_WTE && operand != EQ_SCALE) return RAD_E_UNSUPPORTED;
    const long long rows = rad_param_geti(p, n_p, "n_vocab", 0);
    const long long dim  = rad_param_geti(p, n_p, "n_embd", 0);
    if (rows <= 0 || dim <= 0) return RAD_E_SHAPE;
    /* Or the checkpoint's plain bf16 table, with no scale plane; rows go through the same
     * cache. */
    if (rad_enc_is(enc, "plain") && enc->n_planes == 1 && enc->plane[0].dtype == RAD_BF16) {
        if (operand != EQ_WTE) return RAD_E_DTYPE;
        if (n != 1 || !pl) return RAD_E_SHAPE;
        if (pl[0].rank != 2 || pl[0].shape[0] != rows || pl[0].shape[1] != dim) return RAD_E_SHAPE;
        return RAD_E_UNSUPPORTED;
    }
    const RadEncPlane* sc = rad_enc_plane(enc, "scale");
    if (!rad_enc_is(enc, "affine") || enc->n_planes != 2 || enc->transform[0] ||
        enc->plane[0].dtype != RAD_F8E4M3 || !sc || sc->dtype != RAD_BF16 || sc->block[0] != 0 ||
        sc->block[1] != 0)
        return RAD_E_DTYPE;
    if (n != 1 || !pl || !sel) return RAD_E_SHAPE;
    const char* want = operand == EQ_WTE ? "codes" : "scale";
    if (std::strncmp(enc->plane[sel[0]].role, want, RAD_ENC_STR) != 0) return RAD_E_SHAPE;
    const long long r = operand == EQ_WTE ? rows : 1, c = operand == EQ_WTE ? dim : 1;
    if (pl[0].rank != 2 || pl[0].shape[0] != r || pl[0].shape[1] != c) return RAD_E_SHAPE;
    return RAD_E_UNSUPPORTED;
}

/* ============================== the instance: can this process read the table ==============================
 *
 * A bf16 gather reads misses through io_uring, so a process that cannot create a ring would fail
 * its first request after the whole model loaded. The instance checks at declare and refuses with
 * the reason. Gathers into other dtypes read the table as memory and need no ring. rad-convert
 * declares without instances, so a sandbox without io_uring can still convert models. */
extern "C" int avx_init_ngram(const RadParam* p, int n_p, int rank, int world_size, void** out) {
    (void)rank;
    (void)world_size;
    if (out) *out = nullptr;
#if defined(__linux__)
    const char* dt = rad_param_gets(p, n_p, "dtype", nullptr);
    if (!dt || std::strcmp(dt, "bf16") != 0) return RAD_OK;
    if (const int e = ring_errno()) {
        static std::once_flag said;
        std::call_once(said, [e] { std::fprintf(stderr, "%s\n", ring_refusal(e).c_str()); });
        return RAD_E_UNSUPPORTED;
    }
#else
    (void)p;
    (void)n_p;
#endif
    return RAD_OK;
}
