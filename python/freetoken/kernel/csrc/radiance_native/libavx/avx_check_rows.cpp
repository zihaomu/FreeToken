/* avx_check_rows.cpp -- embed_lookup_q over a file-backed table, as a model serves it.
 *
 * The geometry table checks the op against libref over a heap buffer. A table in a container is
 * read differently: through a shared row cache, direct file reads for misses and a reader thread
 * one call ahead (avx_ngram.cpp). The failures there are about which bytes arrive (a slot claimed
 * twice, a row decoded from a slot taken over, a wait on an abandoned claim), and libref has no
 * such path, so the reference is the file's own bytes.
 *
 * Two threads act as ranks 0 and 1 of 2 over the same ids, released together, so each claims rows,
 * waits on the other's claims and meets its own claim again when an id repeats. Half the ids come
 * from a small hot set; the rest from a table four times the cache's 2^19 rows, so sets fill and
 * rows are evicted while the other thread may be reading them. Each call also passes the next
 * call's ids as `ahead`, so the reader thread races both ranks.
 *
 * Each output row is compared bit for bit with the file's bytes through the call's scale; a padding
 * id must give a zero row. Rows are 40 codes, so the AVX-512 decode runs one 32-code step plus a
 * masked tail. Direct reads align to the filesystem's reported DIO alignment (ext4, XFS) or to its
 * block size where none is reported (btrfs, tmpfs since Linux 6.6); the case reports which.
 *
 * Each table is written once per process under a name of its own (mkostemp), stays mapped for the
 * rest of the process and is unlinked once the gather has opened it: the ISA levels share the file
 * reads and differ only in the decode. */
#include "avx_check.h"

#include <barrier>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>

#include <cstddef>

#include <fcntl.h>
#include <linux/filter.h>
#include <linux/io_uring.h>
#include <linux/seccomp.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

/* The errno of creating a ring as the gather does (avx_ngram.cpp: ReadRing::init), or 0. */
static int ring_probe() {
    io_uring_params p{};
    p.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
    const long ring = syscall(__NR_io_uring_setup, 4096, &p);
    if (ring < 0) return errno;
    ::close((int)ring);
    return 0;
}

/* Ring memory can count against the locked-memory limit, which every process of the user shares,
 * so a run beside other io_uring users can be refused one (ENOMEM) for a while. That is the
 * machine, not the gather: the cases name it and skip. */
static const char* const kRingMemory = "io_uring ring memory refused (ENOMEM); the locked-memory "
                                       "limit is shared with this user's other processes";

/* embed_lookup_q's init must refuse a process that cannot create an io_uring ring, and say why
 * (Docker's default seccomp profile returns EPERM for io_uring_setup). A child installs that
 * refusal as a seccomp filter; init must then refuse a bf16 gather (read through the ring), name
 * the Docker option on stderr, and accept an f32 gather (read as memory). This process must then be
 * accepted where a ring can be created and refused where it cannot, as in a build sandbox. */
bool ngram_init_case(const Lib& lib) {
    const char* what = "embed_lookup_q's init";
    const RadKernelInfo* r = lib.row("embed_lookup_q");
    if (!r || !r->init) {
        std::printf("  %s: the row has no init\n", what);
        return false;
    }
    RadParam bf16[2]{}, f32[2]{};
    bf16[0].key = f32[0].key = "n_embd";
    bf16[0].kind = f32[0].kind = RAD_P_INT;
    bf16[0].ival = f32[0].ival = 160;
    bf16[1].key = f32[1].key = "dtype";
    bf16[1].kind = f32[1].kind = RAD_P_STR;
    bf16[1].sval = "bf16";
    f32[1].sval = "f32";

    int out[2];
    if (::pipe(out) != 0) {
        std::printf("  %s: pipe: %s\n", what, std::strerror(errno));
        return false;
    }
    std::fflush(nullptr);
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::dup2(out[1], 2);
        ::close(out[0]);
        sock_filter f[] = {
            BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_io_uring_setup, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA)),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        };
        sock_fprog prog{ (unsigned short)(sizeof f / sizeof f[0]), f };
        if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
            ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0)
            ::_exit(3);
        void* inst = nullptr;
        const int a = r->init(bf16, 2, 0, 1, &inst);
        const int b = r->init(f32, 2, 0, 1, &inst);
        std::fflush(nullptr);
        ::_exit(a == RAD_E_UNSUPPORTED && b == RAD_OK ? 0 : 1);
    }
    ::close(out[1]);
    std::string said;
    char buf[512];
    for (ssize_t k; (k = ::read(out[0], buf, sizeof buf)) > 0;) said.append(buf, (size_t)k);
    ::close(out[0]);
    int ws = 0;
    if (pid < 0 || ::waitpid(pid, &ws, 0) != pid || !WIFEXITED(ws)) {
        std::printf("  %s: the child did not run to the end\n", what);
        return false;
    }
    bool ok = true;
    if (WEXITSTATUS(ws) == 3) {
        std::printf("  %s, where io_uring is refused: SKIPPED -- no seccomp filter can be installed "
                    "here\n", what);
    } else if (WEXITSTATUS(ws) != 0 ||
               said.find("--security-opt seccomp=unconfined") == std::string::npos) {
        std::printf("  %s, where io_uring is refused: FAIL -- %s; it said: %s\n", what,
                    WEXITSTATUS(ws) ? "the statuses were wrong" : "the reason was not named",
                    said.c_str());
        ok = false;
    } else {
        std::printf("  %s, where io_uring is refused: refused, naming the Docker option\n", what);
    }

    /* The init tests the ring once per process, so the probes bracket it. */
    const int before = ring_probe();
    void* inst = nullptr;
    const int st = r->init(bf16, 2, 0, 1, &inst);
    const int after = ring_probe();
    if (st == RAD_E_UNSUPPORTED && (before == ENOMEM || after == ENOMEM)) {
        std::printf("  %s, in this process: SKIPPED -- %s\n", what, kRingMemory);
        return ok;
    }
    const bool here = before == 0;
    const bool right = here ? st == RAD_OK : st == RAD_E_UNSUPPORTED;
    std::printf("  %s, in this process (io_uring %s): %s\n", what, here ? "available" : "refused",
                right ? (here ? "accepted" : "refused") : "FAIL");
    return ok && right;
}

/* A table file, written and mapped once per process. */
struct RowTable {
    enum { OK, SKIP, FAIL } state = FAIL;
    std::string path, note;       /* note: the skip or failure reason, or the read granule */
    const uint8_t* tab = nullptr;
    bool unlinked = false;
};

static RowTable* row_table(const std::string& dir, bool bf16, int64_t V, int64_t rb) {
    static std::map<std::pair<std::string, bool>, RowTable> tables;
    const auto key = std::make_pair(dir, bf16);
    if (auto it = tables.find(key); it != tables.end()) return &it->second;
    RowTable& t = tables[key];

    /* A name of its own, so a concurrent run cannot truncate the file under this one's mapping. */
    std::string path = dir + "/rad-avx-check-rows.XXXXXX";
    uint64_t x = 0x9e3779b97f4a7c15ull;
    auto rnd = [&x]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    {
        const int fd = ::mkostemp(path.data(), O_CLOEXEC);
        if (fd < 0) { t.note = "cannot create " + path; return &t; }
        std::vector<uint64_t> chunk(1 << 16);
        bool wrote = true;
        for (int64_t done = 0; done < V * rb && wrote; done += (int64_t)(chunk.size() * 8)) {
            for (auto& w : chunk) w = rnd();
            const size_t want = (size_t)std::min<int64_t>(V * rb - done, (int64_t)chunk.size() * 8);
            wrote = ::write(fd, chunk.data(), want) == (ssize_t)want;
        }
        wrote = wrote && ::fsync(fd) == 0;
        ::close(fd);
        if (!wrote) { ::unlink(path.c_str()); t.note = "short write"; return &t; }
    }
    /* The gather refuses a filesystem without direct I/O (the O_DIRECT open fails, as on tmpfs
     * before Linux 6.6, or it reports a zero DIO alignment), so the case is skipped there. Every
     * other filesystem must work, with or without a reported alignment. */
    bool reported = false;
    {
        const int dfd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        struct statx sx {};
        const bool dio = dfd >= 0 &&
                         ::statx(dfd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 &&
                         !((sx.stx_mask & STATX_DIOALIGN) && sx.stx_dio_offset_align == 0);
        reported = (sx.stx_mask & STATX_DIOALIGN) != 0;
        if (dfd >= 0) ::close(dfd);
        if (!dio) {
            ::unlink(path.c_str());
            t.state = RowTable::SKIP;
            t.note = dir + " cannot do direct I/O";
            return &t;
        }
    }
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    void* map = fd < 0 ? MAP_FAILED : mmap(nullptr, (size_t)(V * rb), PROT_READ, MAP_SHARED, fd, 0);
    if (fd >= 0) ::close(fd);
    if (map == MAP_FAILED) { ::unlink(path.c_str()); t.note = "mmap"; return &t; }
    /* The mapping stays for the life of the process. libavx resolves a table to its file once, by
     * the mapping's address range (avx_ngram.cpp: table_file), so no other file may be mapped at
     * these addresses while the process runs. */
    t.tab = static_cast<const uint8_t*>(map);
    t.path = path;
    t.note = reported ? "the alignment the filesystem reports"
                      : "no alignment reported, read by the block size";
    t.state = RowTable::OK;
    return &t;
}

bool row_gather_case(const Lib& lib, const std::string& dir, int64_t calls, uint64_t seed,
                     bool verbose, bool bf16) {
    const RadKernelInfo* r = lib.row("embed_lookup_q");
    const char* what = bf16 ? "embed_lookup_q from a file, bf16 table" : "embed_lookup_q from a file";
    if (!r || !r->launch) {
        std::printf("  %s: no row\n", what);
        return false;
    }
    /* A bf16 table has twice the row bytes and no scale, so it exercises the cache and the reads
     * at a row size the E4M3 table never has. */
    /* Misses are io_uring reads. Where a sandbox forbids io_uring the gather refuses every call by
     * design, so the case is skipped with the reason. */
    if (const int e = ring_probe()) {
        std::printf("  %s: SKIPPED -- %s\n", what,
                    e == ENOMEM ? kRingMemory
                                : (std::string("io_uring is unavailable here (") +
                                   std::strerror(e) + ")").c_str());
        return true;
    }
    const int64_t n = 40, V = int64_t(1) << 21, M = 512, hot = 4096;
    const int64_t rb = bf16 ? 2 * n : n;
    const float scale = 0.375f;

    RowTable* rt = row_table(dir, bf16, V, rb);
    if (rt->state == RowTable::SKIP) {
        std::printf("  %s: SKIPPED -- %s\n", what, rt->note.c_str());
        return true;
    }
    if (rt->state != RowTable::OK) {
        std::printf("  %s: %s\n", what, rt->note.c_str());
        return false;
    }
    const char* granule = rt->note.c_str();
    const uint8_t* tab = rt->tab;
    uint64_t x = seed ? seed : 1;
    auto rnd = [&x]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };

    std::vector<int32_t> hotset((size_t)hot);
    for (auto& h : hotset) h = (int32_t)(rnd() % (uint64_t)V);
    std::vector<int32_t> ids((size_t)(calls * M));
    for (int64_t c = 0; c < calls; ++c)
        for (int64_t m = 0; m < M; ++m) {
            const uint64_t v = rnd();
            int32_t& id = ids[(size_t)(c * M + m)];
            if (m > 0 && v % 29 == 0)  id = ids[(size_t)(c * M + m - 1)];   /* repeated in a call */
            else if (v % 16 == 0)      id = -1;                             /* padding */
            else if (v % 16 < 8)       id = hotset[(v >> 8) % (uint64_t)hot];
            else                       id = (int32_t)((v >> 8) % (uint64_t)V);
        }
    uint16_t lut[256];
    for (int b = 0; b < 256; ++b) lut[b] = rad_f32_to_bf16(rad_fp8e4m3_to_f32((uint8_t)b) * scale);

    auto tensor = [](void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
        RadTensor t{};
        t.data = p;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int i = 0;
        for (int64_t s : shape) t.shape[i++] = s;
        int64_t st = 1;
        for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = st; st *= t.shape[k]; }
        return t;
    };

    std::barrier<> go(2);
    int64_t bad[2] = {0, 0}, refused[2] = {0, 0}, failed[2] = {0, 0};
    /* Each rank thread makes its own ring on its first call. */
    auto rank = [&](int rk) {
        uint16_t sc = rad_f32_to_bf16(scale);
        std::vector<uint16_t> out((size_t)(M * n));
        std::vector<RadTensor> ts = {
            tensor(nullptr, RAD_I32, {M}),
            tensor(const_cast<uint8_t*>(tab), bf16 ? RAD_BF16 : RAD_F8E4M3, {V, n}),
            tensor(bf16 ? nullptr : &sc, RAD_BF16, {1}), tensor(out.data(), RAD_BF16, {M, n}),
            tensor(nullptr, RAD_I32, {M}) };
        RadParam ps[3] = {};
        ps[0].key = "M";       ps[0].kind = RAD_P_INT; ps[0].ival = M;
        ps[1].key = "n_embd";  ps[1].kind = RAD_P_INT; ps[1].ival = n;
        ps[2].key = "n_vocab"; ps[2].kind = RAD_P_INT; ps[2].ival = V;
        RadArgs a{};
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps; a.n_p = 3;
        a.rank = rk; a.world_size = 2;
        for (int64_t c = 0; c < calls; ++c) {
            ts[0].data = &ids[(size_t)(c * M)];
            /* The last call passes no `ahead` (absent operand). */
            ts[4].data = c + 1 < calls ? &ids[(size_t)((c + 1) * M)] : nullptr;
            std::fill(out.begin(), out.end(), (uint16_t)0xdead);
            go.arrive_and_wait();
            if (const int rc = r->launch(&a, nullptr); rc != RAD_OK) {
                ++refused[rk];
                if (rc != RAD_E_UNSUPPORTED) ++failed[rk];
                continue;
            }
            for (int64_t m = 0; m < M; ++m) {
                const int32_t id = ids[(size_t)(c * M + m)];
                for (int64_t i = 0; i < n; ++i) {
                    uint16_t want = 0;
                    if (id >= 0 && bf16) std::memcpy(&want, tab + ((int64_t)id * n + i) * 2, 2);
                    else if (id >= 0)    want = lut[tab[(int64_t)id * n + i]];
                    if (out[(size_t)(m * n + i)] != want) { ++bad[rk]; break; }
                }
            }
        }
    };
    auto race = [&] {
        bad[0] = bad[1] = refused[0] = refused[1] = failed[0] = failed[1] = 0;
        std::thread t0(rank, 0), t1(rank, 1);
        t0.join();
        t1.join();
    };
    /* Calls refused as unsupported with every delivered row right mean a ring the kernel would
     * not make. Under contention for ring memory the case runs again on fresh threads, which make
     * fresh rings, up to three times; it is skipped by name only if a probe saw ENOMEM meanwhile.
     * Refusals without that evidence fail. */
    bool pressure = false;
    for (int attempt = 0; attempt < 3; ++attempt) {
        race();
        const bool ring_refused = refused[0] + refused[1] && bad[0] + bad[1] == 0 &&
                                  failed[0] + failed[1] == 0;
        if (!ring_refused) break;
        if (ring_probe() == ENOMEM) pressure = true;
    }
    /* The gather holds its own descriptor from the first call on, so the name can go. */
    if (!rt->unlinked) {
        ::unlink(rt->path.c_str());
        rt->unlinked = true;
    }
    if (pressure && refused[0] + refused[1] && bad[0] + bad[1] + failed[0] + failed[1] == 0) {
        std::printf("  %s: SKIPPED -- %s\n", what, kRingMemory);
        return true;
    }

    const bool pass = bad[0] + bad[1] + refused[0] + refused[1] == 0;
    if (verbose || !pass)
        std::printf("  %s in %s (%s): %lld calls x 2 ranks, rows wrong %lld/%lld, "
                    "calls refused %lld/%lld %s\n", what, dir.c_str(), granule, (long long)calls,
                    (long long)bad[0], (long long)bad[1], (long long)refused[0],
                    (long long)refused[1], pass ? "ok" : "FAIL");
    else
        std::printf("  %s in %s (%s): %lld calls x 2 ranks, every row exact\n", what, dir.c_str(),
                    granule, (long long)calls);
    return pass;
}

/* A prefill's read-ahead, as the engine drives it (#355): each call gathers one chunk of 65,536
 * rows (4096 tokens of 16 heads) and passes the following eight chunks as `ahead`, the window
 * moving on by one chunk a call, with both ranks passing the same lists. Each row the calls gather
 * must be read from the file about once: each window adds one chunk to what the reader was handed,
 * and what is read ahead must still be in the cache when its chunk is gathered. Handing each window
 * over whole queued every chunk eight times and read ahead the cache's whole 2^19 rows, so rows
 * were evicted before their chunk came and read again, and the reader went on reading after the
 * last call. avx_ngram_reads counts the rows read from table files once the reader is idle; the
 * table is row_gather_case's, so some rows are already cached and read less than once. */
bool row_ahead_case(const Lib& lib, const std::string& dir, bool verbose) {
    const char* what = "embed_lookup_q's read-ahead over a prefill";
    const RadKernelInfo* r = lib.row("embed_lookup_q");
    auto reads = (uint64_t (*)())dlsym(lib.h, "avx_ngram_reads");
    if (!r || !r->launch || !reads) {
        std::printf("  %s: %s\n", what, !reads ? "no avx_ngram_reads export" : "no row");
        return false;
    }
    if (const int e = ring_probe()) {
        std::printf("  %s: SKIPPED -- %s\n", what,
                    e == ENOMEM ? kRingMemory
                                : (std::string("io_uring is unavailable here (") +
                                   std::strerror(e) + ")").c_str());
        return true;
    }
    const int64_t n = 40, V = int64_t(1) << 21, M = 65536, chunks = 12, window = 8;
    RowTable* rt = row_table(dir, false, V, n);
    if (rt->state == RowTable::SKIP) {
        std::printf("  %s: SKIPPED -- %s\n", what, rt->note.c_str());
        return true;
    }
    if (rt->state != RowTable::OK) {
        std::printf("  %s: %s\n", what, rt->note.c_str());
        return false;
    }
    const uint8_t* tab = rt->tab;
    const float scale = 0.375f;
    uint64_t x = 0x2545f4914f6cdd1dull;
    auto rnd = [&x]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    std::vector<int32_t> ids((size_t)(chunks * M));
    for (auto& id : ids) id = (int32_t)(rnd() % (uint64_t)V);
    std::vector<int32_t> distinct(ids);
    std::sort(distinct.begin(), distinct.end());
    const int64_t U = std::unique(distinct.begin(), distinct.end()) - distinct.begin();
    uint16_t lut[256];
    for (int b = 0; b < 256; ++b) lut[b] = rad_f32_to_bf16(rad_fp8e4m3_to_f32((uint8_t)b) * scale);

    auto tensor = [](void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
        RadTensor t{};
        t.data = p;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int i = 0;
        for (int64_t s : shape) t.shape[i++] = s;
        int64_t st = 1;
        for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = st; st *= t.shape[k]; }
        return t;
    };
    const uint64_t before = reads();
    std::barrier<> go(2);
    int64_t bad[2] = {0, 0}, refused[2] = {0, 0};
    auto rank = [&](int rk) {
        uint16_t sc = rad_f32_to_bf16(scale);
        std::vector<uint16_t> out((size_t)(M * n));
        RadParam ps[3] = {};
        ps[0].key = "M";       ps[0].kind = RAD_P_INT; ps[0].ival = M;
        ps[1].key = "n_embd";  ps[1].kind = RAD_P_INT; ps[1].ival = n;
        ps[2].key = "n_vocab"; ps[2].kind = RAD_P_INT; ps[2].ival = V;
        for (int64_t c = 0; c < chunks; ++c) {
            const int64_t na = std::min(window, chunks - 1 - c) * M;
            std::vector<RadTensor> ts = {
                tensor(&ids[(size_t)(c * M)], RAD_I32, {M}),
                tensor(const_cast<uint8_t*>(tab), RAD_F8E4M3, {V, n}),
                tensor(&sc, RAD_BF16, {1}), tensor(out.data(), RAD_BF16, {M, n}),
                tensor(na ? &ids[(size_t)((c + 1) * M)] : nullptr, RAD_I32, {na}) };
            RadArgs a{};
            a.t = ts.data(); a.n_t = (int)ts.size();
            a.p = ps; a.n_p = 3;
            a.rank = rk; a.world_size = 2;
            go.arrive_and_wait();
            if (r->launch(&a, nullptr) != RAD_OK) { ++refused[rk]; continue; }
            for (int64_t m = 0; m < M; ++m) {
                const uint8_t* row = tab + (int64_t)ids[(size_t)(c * M + m)] * n;
                for (int64_t i = 0; i < n; ++i)
                    if (out[(size_t)(m * n + i)] != lut[row[i]]) { ++bad[rk]; break; }
            }
        }
    };
    std::thread t0(rank, 0), t1(rank, 1);
    t0.join();
    t1.join();
    const uint64_t read = reads() - before;
    const bool exact = bad[0] + bad[1] + refused[0] + refused[1] == 0;
    const bool once = (int64_t)read <= U + U / 10;
    const bool pass = exact && once;
    std::printf("  %s in %s: %lld chunks of %lld rows x 2 ranks, %llu rows read for %lld distinct "
                "(%.2f each)%s%s\n", what, dir.c_str(), (long long)chunks, (long long)M,
                (unsigned long long)read, (long long)U, (double)read / (double)U,
                exact ? ", every row exact" : "", pass ? "" : " FAIL");
    if (!exact || verbose)
        std::printf("    rows wrong %lld/%lld, calls refused %lld/%lld\n", (long long)bad[0],
                    (long long)bad[1], (long long)refused[0], (long long)refused[1]);
    return pass;
}
