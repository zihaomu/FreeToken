/* avx_bench.cpp -- rad-avx-bench: this machine's measured ceilings, and what libavx reaches.
 *
 * A speedup over libref (a scalar loop nest with a per-element dtype switch) says more about
 * libref than about libavx. So each op is also reported as a rate to compare against this
 * machine's ceilings, measured by --peak rather than taken from a spec sheet (memory population,
 * thermal limits and firmware settings only show up when measured). The libref ratio stays: it
 * shows an op is wired at all, and a kernel slower than the reference is a defect.
 *
 * Measurement hygiene:
 *   - INOUT operands are reset to the drawn bytes between iterations (h_reset), outside the
 *     clock; otherwise each iteration computes something different (rope rotates again, samplers
 *     narrow an already narrowed set).
 *   - Loops are cache-warm, so bandwidth-bound kernels are ranked by in-cache behaviour; the
 *     output says so.
 *   - Warm-up iterations run first, since the thread-local scratch arenas allocate on first use.
 *   - Per-op times are medians, robust to a preempted iteration.
 *   - --cold flushes the case's operands from every cache before each timed launch, for an op
 *     whose working set fits this part's L3 at model shape but not beside the rest of a model's
 *     step (the QSA indexer's block keys); the flush is outside the clock.
 */
#include "avx_harness.h"
#include "avx_isa.h"
#include "avx_planes.h"

#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <memory>

#include <emmintrin.h>   /* _mm_clflush, _mm_mfence: SSE2, the x86-64 baseline */

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::steady_clock;

static double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* ================================================================== the machine's ceilings
 *
 * Measured ceilings, one per class of op:
 *
 *   FMA throughput bounds the GEMMs: independent in-register FMA chains, at the clock the core
 *   holds under vector load (below advertised boost on parts with an AVX-512 offset).
 *
 *   L1 bandwidth bounds elementwise ops at shapes that fit, most of a decode step.
 *
 *   DRAM bandwidth bounds prefill shapes, and always bounds `logits_gemm` (a 248K-vocabulary
 *   weight fits no cache).
 *
 * The memory triads use --threads (default 1); the FMA probe is single-threaded.
 */
static void bench_peak(Lib& avx, int threads) {
    std::printf("\n== machine ceilings (measured here, not quoted) ==\n");

    /* ---- FMA per ISA level, measured inside the plugin: this file is baseline-compiled and
     * could only measure SSE2. `avx_peak_fma` runs the level's probe and returns the flops
     * performed; timing happens here. */
    auto peak = (double (*)(int, int64_t))dlsym(avx.h, "avx_peak_fma");
    if (peak) {
        for (int lvl = 0; lvl < AVX_N_LEVELS; ++lvl) {
            if (avx.isa_detect && lvl > avx.isa_detect()) {
                std::printf("  FMA %-18s %8s              (not executable on this CPU)\n",
                            avx.isa_name(lvl), "n/a");
                continue;
            }
            std::vector<double> rates;
            for (int rep = 0; rep < 5; ++rep) {
                const int64_t iters = 1000000;
                const double t0 = now_s();
                const double fl = peak(lvl, iters);
                const double dt = now_s() - t0;
                rates.push_back(fl / dt);
            }
            /* Best of five, since this is a ceiling (a preempted or down-clocked repetition says
             * nothing about the limit). Per-op timings use the median. */
            std::sort(rates.begin(), rates.end());
            std::printf("  FMA %-18s %8.2f GFLOP/s  (1 thread, in-register, no memory traffic)\n",
                        avx.isa_name(lvl), rates.back() / 1e9);
        }
    }

    /* ---- memory at three working-set sizes. A triad (a[i] = b[i] + s*c[i]): two streams in,
     * one out, the traffic shape of the elementwise ops. */
    auto triad = [&](size_t bytes_per_array, const char* label) {
        const size_t n = bytes_per_array / sizeof(float);
        std::vector<float> A(n), B(n), C(n);
        for (size_t i = 0; i < n; ++i) { B[i] = (float)i * 1e-6f; C[i] = 1.0f; }
        std::vector<double> ts;
        for (int rep = 0; rep < 7; ++rep) {
            const double t0 = now_s();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
            for (int64_t i = 0; i < (int64_t)n; ++i) A[(size_t)i] = B[(size_t)i] + 0.5f * C[(size_t)i];
            ts.push_back(now_s() - t0);
        }
        const double dt = median(ts);
        /* Counts three arrays (two read, one written); the store's read-for-ownership, which the
         * compiler may or may not elide, is not counted. */
        std::printf("  triad %-15s %8.2f GB/s     (%zu threads, %.0f KiB working set)\n", label,
                    3.0 * (double)n * sizeof(float) / dt / 1e9, (size_t)threads,
                    3.0 * (double)bytes_per_array / 1024.0);
    };
    triad(16 * 1024, "L1");
    triad(512 * 1024, "L2");
    triad(256u * 1024 * 1024, "DRAM");

    /* ---- a read-only stream over 768 MiB: the ceiling of an op that only reads, as the QSA
     * indexer does its block keys. On parts whose core complexes reach memory through links
     * narrower for writes than for reads, it exceeds the triad. */
    {
        /* An integer sum: this file is baseline-compiled, and a float sum may not be reassociated,
         * so it would measure one add's latency instead of the memory. */
        const size_t n = (768u << 20) / sizeof(uint64_t);
        std::vector<uint64_t> A(n, 1);
        std::vector<double> ts;
        uint64_t sink = 0;
        for (int rep = 0; rep < 7; ++rep) {
            const double t0 = now_s();
            uint64_t acc = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) reduction(+ : acc)
#endif
            for (int64_t i = 0; i < (int64_t)n; ++i) acc += A[(size_t)i];
            ts.push_back(now_s() - t0);
            sink += acc;
        }
        std::printf("  read  %-15s %8.2f GB/s     (%zu threads, %.0f KiB, sum %llu)\n", "DRAM",
                    (double)n * sizeof(uint64_t) / median(ts) / 1e9, (size_t)threads,
                    (double)n * sizeof(uint64_t) / 1024.0, (unsigned long long)sink);
    }
}

/* ================================================================== conversion rates
 * Timed through the plugin's own `cast` op at each ISA level, so the conversions measured are the
 * ones the kernels use (avx_support.cpp). */
static void bench_cvt(Lib& avx, Lib& ref) {
    std::printf("\n== dtype conversion, through `cast` (MB/s of SOURCE bytes) ==\n");
    const char* dts[] = { "bf16", "f16", "fp8_e4m3", "i8" };
    const int64_t N = 1 << 16;
    std::printf("  %-12s %10s %10s %10s %10s %10s\n", "pair", "scalar", "avx", "avx2", "avx512",
                "libref");
    for (const char* dt : dts)
        for (int dir = 0; dir < 2; ++dir) {
            const char* from = dir ? dt : "f32";
            const char* to   = dir ? "f32" : dt;
            Case c{ "cast", { HI("M", 1), HI("n", N), HS("from", from), HS("to", to) }, 0, 0 };
            const RadKernelInfo* kr = ref.row("cast");
            const RadKernelInfo* ka = avx.row("cast");
            if (!kr || !ka) continue;
            Prepared pr = h_prepare(ref, kr, c, kr, ka);
            if (!pr.ok) continue;
            const uint32_t sdt = pr.opd[0].d.dtype;
            const double src_bytes = (double)dt_bytes(sdt, N);
            char line[512];
            int off = std::snprintf(line, sizeof line, "  %-12s", (std::string(from) + "->" + to).c_str());
            for (int lvl = 0; lvl < AVX_N_LEVELS; ++lvl) {
                if (avx.isa_force && avx.isa_force(lvl) != lvl) {
                    off += std::snprintf(line + off, sizeof line - off, " %10s", "n/a");
                    continue;
                }
                for (int w = 0; w < 20; ++w) ka->launch(&pr.ab, nullptr);
                std::vector<double> ts;
                for (int r = 0; r < 9; ++r) {
                    const double t0 = now_s();
                    for (int it = 0; it < 50; ++it) ka->launch(&pr.ab, nullptr);
                    ts.push_back((now_s() - t0) / 50);
                }
                off += std::snprintf(line + off, sizeof line - off, " %10.0f",
                                     src_bytes / median(ts) / 1e6);
            }
            {
                for (int w = 0; w < 3; ++w) kr->launch(&pr.aa, nullptr);
                std::vector<double> ts;
                for (int r = 0; r < 5; ++r) {
                    const double t0 = now_s();
                    kr->launch(&pr.aa, nullptr);
                    ts.push_back(now_s() - t0);
                }
                off += std::snprintf(line + off, sizeof line - off, " %10.0f",
                                     src_bytes / median(ts) / 1e6);
            }
            std::printf("%s\n", line);
        }
    if (avx.isa_force) avx.isa_force(AVX_N_LEVELS - 1);
}

/* ================================================================== the codebook experts
 *
 * moe_gemm_q over Qwen3.8-Flash-Next's served experts at the model's geometry on one rank: 512
 * experts in two parity tables, top_k 10, gate_up [1280, 2560] and down [2560, 640], stored through
 * libavx's own layout hooks. Every expert of a layer is resident (1.3 GB at w4nl64a8h) and each
 * timed call draws a new routing, so a decode step reads its experts from DRAM as a served step
 * does, not from the cache a repeated call would leave them in. Reported: the expert bytes a call
 * reads (codes and scales of the experts it routes to) per second, against the DRAM triad, and
 * FLOP/s against the FMA peak (--peak measures both). */
namespace moebench {

struct Form { const char* dtype; int bits; int64_t group; uint32_t sdt; bool level; };
const Form kForms[] = {
    { "w4nla8h",   4, 128, RAD_BF16,   false },
    { "w4nl64a8h", 4, 64,  RAD_F8E4M3, true },
    { "w4nl32a8h", 4, 32,  RAD_F8E4M3, true },
    { "w5nl64a8h", 5, 64,  RAD_F8E4M3, true },
};
const float kW4nl[16] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f,
                          -1.125f, -3.25f, -5.5f, -8.0f, -11.0f, -14.0f, -18.0f, -22.0f };
const float kW5nl[32] = { 0.8125f, 2.5f, 4.0f, 5.5f, 7.5f, 9.0f, 11.0f, 12.0f,
                          14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 30.0f,
                          -0.8125f, -2.5f, -4.0f, -5.5f, -7.5f, -9.0f, -11.0f, -12.0f,
                          -14.0f, -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -30.0f };

constexpr int64_t kEmbd = 2560, kFF = 640, kExperts = 512, kTopK = 10;

struct Buf {
    uint8_t* p = nullptr;
    int64_t  n = 0;
    explicit Buf(int64_t bytes) : n(bytes) {
        p = (uint8_t*)std::aligned_alloc(64, (size_t)((bytes + 63) / 64 * 64));
    }
    ~Buf() { std::free(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
};

/* One projection's experts, each view stored as libavx's row stores it. */
struct Proj {
    int64_t N = 0, K = 0;
    std::vector<std::unique_ptr<Buf>> codes, scales;
    uint32_t cdt = 0, sdt = 0;
    int64_t  bytes_per_expert = 0;
};

/* The experts of one projection, every view through the row's layout hooks. False on a refusal. */
bool build(const RadKernelInfo* k, const Form& f, int64_t N, int64_t K, const char* order,
           Proj* out) {
    out->N = N;
    out->K = K;
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_copy_str(e.transform, "fwht128");
    const uint32_t cdt = f.bits == 5 ? RAD_U5 : RAD_U4;
    const int ic = rad_enc_add_plane(&e, "codes", cdt, 1, 1);
    const int is = rad_enc_add_plane(&e, "scale", f.sdt, 1, f.group);
    const int il = f.level ? rad_enc_add_plane(&e, "scale.1", RAD_F32, 0, 0) : -1;
    const int it = rad_enc_add_table(&e, "table", RAD_F32, 1, (int64_t)1 << f.bits);
    const int64_t SC = (K + f.group - 1) / f.group;
    const int64_t rbc = rad_dtype_bytes(cdt, K), rbs = rad_dtype_bytes(f.sdt, SC);
    std::vector<uint8_t> pc((size_t)(N * rbc)), ps((size_t)(N * rbs)), pl(4), pt(4u << f.bits);
    const float lv = 0x1p-13f;
    std::memcpy(pl.data(), &lv, 4);
    std::memcpy(pt.data(), f.bits == 5 ? kW5nl : kW4nl, pt.size());
    const RadParam p[] = { RadParam{ "dtype", RAD_P_STR, 0, 0, f.dtype, 0 },
                           RadParam{ "a_order", RAD_P_STR, 0, 0, order, 0 } };
    auto plane = [](void* d, uint32_t dt, int64_t r, int64_t c) {
        RadTensor t{};
        t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c;
        t.stride[0] = c; t.stride[1] = 1;
        return t;
    };
    uint64_t s = 0x1234567ull + (uint64_t)N;
    for (int64_t ex = 0; ex < kExperts; ++ex) {
        /* Any bit pattern is a valid code, and the scales span four binades as the model's do. */
        for (size_t i = 0; i < pc.size(); i += 8) {
            const uint64_t r = hsplitmix(s);
            std::memcpy(pc.data() + i, &r, std::min<size_t>(8, pc.size() - i));
        }
        for (int64_t i = 0; i < N * SC; ++i)
            rad_store_f32(ps.data(), f.sdt, i,
                          f.sdt == RAD_F8E4M3 ? std::ldexp(1.0f + (float)(i % 7) / 8.0f,
                                                           (int)(hsplitmix(s) % 4) - 2)
                                              : 0.0005f + 0.0001f * (float)(i % 13));
        const int opd = ex % 2 ? 6 : 2;
        for (int v = 0; v < 2; ++v) {
            int sel[2];
            RadTensor t[2];
            int n = 0;
            if (v == 0) {
                sel[0] = ic; sel[1] = it; n = 2;
                t[0] = plane(pc.data(), cdt, N, K);
                t[1] = plane(pt.data(), RAD_F32, 1, (int64_t)1 << f.bits);
            } else {
                sel[0] = is; t[0] = plane(ps.data(), f.sdt, N, SC); n = 1;
                if (f.level) { sel[1] = il; t[1] = plane(pl.data(), RAD_F32, 1, 1); n = 2; }
            }
            RadLayout L{};
            const int rc = k->layout ? k->layout(p, 2, opd + v, &e, sel, t, n, &L)
                                     : RAD_E_UNSUPPORTED;
            std::unique_ptr<Buf> b;
            if (rc == RAD_E_UNSUPPORTED && n == 1) {
                b.reset(new Buf((int64_t)ps.size()));
                std::memcpy(b->p, ps.data(), ps.size());
                if (ex == 0) out->sdt = f.sdt;
            } else if (rc == RAD_OK && k->relayout) {
                b.reset(new Buf(L.bytes));
                if (k->relayout(p, 2, opd + v, &e, sel, t, n, b->p, L.bytes) != RAD_OK) return false;
                if (ex == 0) (v ? out->sdt : out->cdt) = L.dtype;
            } else {
                std::fprintf(stderr, "  %s: %s's layout refused %s (%d)\n", f.dtype, k->name,
                             v ? "the scales" : "the codes", rc);
                return false;
            }
            (v ? out->scales : out->codes).push_back(std::move(b));
        }
    }
    out->bytes_per_expert = N * rbc + N * rbs;
    return true;
}

/* A routing of M tokens to kTopK distinct experts each, drawn from `seed`, as moe_scatter leaves
 * it: the flattened slots grouped by expert, and the offsets. Returns the experts it touches. */
int64_t route(int64_t M, uint64_t seed, std::vector<int32_t>* sorted, std::vector<int32_t>* off) {
    const int64_t T = M * kTopK;
    std::vector<int32_t> eid((size_t)T);
    uint64_t s = seed;
    for (int64_t t = 0; t < M; ++t)
        for (int64_t j = 0; j < kTopK; ++j) {
            int32_t e;
            bool dup;
            do {
                e = (int32_t)(hsplitmix(s) % (uint64_t)kExperts);
                dup = false;
                for (int64_t i = 0; i < j; ++i) dup = dup || eid[(size_t)(t * kTopK + i)] == e;
            } while (dup);
            eid[(size_t)(t * kTopK + j)] = e;
        }
    sorted->assign((size_t)T, -1);
    off->assign((size_t)kExperts + 1, 0);
    std::vector<int64_t> cnt((size_t)kExperts, 0);
    for (int32_t e : eid) ++cnt[(size_t)e];
    int64_t touched = 0;
    for (int64_t e = 0; e < kExperts; ++e) {
        (*off)[(size_t)e + 1] = (*off)[(size_t)e] + (int32_t)cnt[(size_t)e];
        touched += cnt[(size_t)e] > 0;
    }
    std::vector<int32_t> cur(off->begin(), off->end() - 1);
    for (int64_t i = 0; i < T; ++i) (*sorted)[(size_t)cur[(size_t)eid[(size_t)i]]++] = (int32_t)i;
    return touched;
}

/* Seconds per call, median of `reps` calls, each on a fresh routing; the touched experts' bytes
 * averaged over the calls. */
double time_proj(const RadKernelInfo* k, const Form& f, const Proj& pj, bool gate_up, int64_t M,
                 int reps, double* bytes, int* status) {
    const int64_t T = M * kTopK, AR = gate_up ? M : T, N = pj.N, K = pj.K, SG = (K + 127) / 128;
    std::vector<uint8_t> aq((size_t)(AR * K)), y((size_t)(T * N * 2));
    std::vector<float> as((size_t)(AR * SG));
    uint64_t s = 99;
    for (auto& b : aq) {
        b = (uint8_t)(hsplitmix(s) & 0xff);
        if ((b & 0x7f) == 0x7f) b = 0x38;       /* not the format's one NaN */
    }
    for (size_t i = 0; i < as.size(); ++i) as[i] = 0.01f + 0.001f * (float)(i % 11);
    std::vector<void*> tab[4];
    for (int64_t e = 0; e < kExperts; ++e) {
        tab[(e % 2) * 2].push_back(pj.codes[(size_t)e]->p);
        tab[(e % 2) * 2 + 1].push_back(pj.scales[(size_t)e]->p);
    }
    const int64_t scols = f.level ? K : SG;
    auto table = [&](int i) {
        RadTensor t{};
        t.data = tab[i].data();
        t.dtype = i % 2 ? pj.sdt : pj.cdt;
        t.rank = 3;
        t.shape[0] = kExperts / 2; t.shape[1] = N; t.shape[2] = i % 2 ? scols : K;
        t.stride[0] = 0; t.stride[1] = t.shape[2]; t.stride[2] = 1;
        return t;
    };
    auto dense = [](void* d, uint32_t dt, std::initializer_list<int64_t> sh) {
        RadTensor t{};
        t.data = d; t.dtype = dt; t.rank = (uint32_t)sh.size();
        int i = 0;
        for (int64_t x : sh) t.shape[i++] = x;
        rad_tensor_pack(&t);
        return t;
    };
    std::vector<int32_t> sorted, off;
    const RadParam p[] = {
        RadParam{ "M", RAD_P_INT, M, 0, nullptr, 0 }, RadParam{ "N", RAD_P_INT, N, 0, nullptr, 0 },
        RadParam{ "K", RAD_P_INT, K, 0, nullptr, 0 },
        RadParam{ "n_expert", RAD_P_INT, kExperts, 0, nullptr, 0 },
        RadParam{ "top_k", RAD_P_INT, kTopK, 0, nullptr, 0 },
        RadParam{ "group", RAD_P_INT, 128, 0, nullptr, 0 },
        RadParam{ "a_order", RAD_P_STR, 0, 0, gate_up ? "token" : "sorted", 0 },
        RadParam{ "dtype", RAD_P_STR, 0, 0, f.dtype, 0 },
        RadParam{ "N_odd", RAD_P_INT, N, 0, nullptr, 0 },
        RadParam{ "K_odd", RAD_P_INT, K, 0, nullptr, 0 },
        RadParam{ "parts", RAD_P_INT, gate_up ? 2 : 1, 0, nullptr, 0 } };
    std::vector<double> ts;
    double total = 0;
    *status = RAD_OK;
    for (int r = -1; r < reps; ++r) {
        const int64_t touched = route(M, 1000 + (uint64_t)r, &sorted, &off);
        RadTensor t[9] = { dense(aq.data(), RAD_F8E4M3, { AR, K }),
                           dense(as.data(), RAD_F32, { AR, SG }), table(0), table(1),
                           dense(sorted.data(), RAD_I32, { T }),
                           dense(off.data(), RAD_I32, { kExperts + 1 }), table(2), table(3),
                           dense(y.data(), RAD_BF16, { T, N }) };
        RadArgs a{};
        a.t = t; a.n_t = 9; a.p = p; a.n_p = (int)(sizeof p / sizeof p[0]); a.world_size = 1;
        const double t0 = now_s();
        const int rc = k->launch(&a, nullptr);
        const double dt = now_s() - t0;
        if (rc < 0) { *status = rc; return 0; }
        if (r < 0) continue;                      /* the warm-up: scratch arenas grow */
        ts.push_back(dt);
        total += (double)touched * (double)pj.bytes_per_expert;
    }
    *bytes = total / reps;
    return median(ts);
}

}  /* namespace moebench */

static void bench_moe(Lib& avx, Lib& ref, const std::string& dtypes, int threads) {
    using namespace moebench;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    const RadKernelInfo* ka = avx.row("moe_gemm_q");
    const RadKernelInfo* kr = ref.row("moe_gemm_q");
    if (!ka || !kr) return;
    std::printf("\n== moe_gemm_q, Qwen3.8-Flash-Next's experts on one rank: %lld experts, top_k "
                "%lld, gate_up [%lld, %lld], down [%lld, %lld]; %d thread(s), each call a fresh "
                "routing ==\n", (long long)kExperts, (long long)kTopK, (long long)(2 * kFF),
                (long long)kEmbd, (long long)kEmbd, (long long)kFF, threads);
    std::printf("  %-10s %5s %-8s %11s %11s %11s %11s %11s %10s %11s\n", "dtype", "M", "proj",
                "libref us", "scalar us", "avx us", "avx2 us", "avx512 us", "GB/s", "GFLOP/s");
    for (const Form& f : kForms) {
        if (("," + dtypes + ",").find("," + std::string(f.dtype) + ",") == std::string::npos)
            continue;
        Proj gu, dn;
        if (!build(ka, f, 2 * kFF, kEmbd, "token", &gu) || !build(ka, f, kEmbd, kFF, "sorted", &dn))
            continue;
        for (int64_t M : { 1, 4, 64, 2048 })
            for (int pj = 0; pj < 2; ++pj) {
                const Proj& P = pj ? dn : gu;
                const int reps = M >= 2048 ? 3 : M >= 64 ? 7 : 21;
                double t[AVX_N_LEVELS + 1] = {}, bytes = 0;
                int st = RAD_OK;
                /* libref decodes per element through its table: decode sizes only. */
                if (M <= 4) t[0] = time_proj(kr, f, P, !pj, M, 3, &bytes, &st);
                for (int lvl = 0; lvl < AVX_N_LEVELS && st == RAD_OK; ++lvl) {
                    if (avx.isa_force && avx.isa_force(lvl) != lvl) continue;
                    if (lvl < 2 && M >= 64) continue;
                    t[lvl + 1] = time_proj(ka, f, P, !pj, M, reps, &bytes, &st);
                }
                if (avx.isa_force) avx.isa_force(AVX_N_LEVELS - 1);
                if (st != RAD_OK) {
                    std::printf("  %-10s %5lld %-8s refused: %d\n", f.dtype, (long long)M,
                                pj ? "down" : "gate_up", st);
                    continue;
                }
                const double best = t[AVX_N_LEVELS];
                const double flops = 2.0 * (double)(M * kTopK) * (double)P.N * (double)P.K;
                char line[256];
                int o = std::snprintf(line, sizeof line, "  %-10s %5lld %-8s", f.dtype,
                                      (long long)M, pj ? "down" : "gate_up");
                for (int i = 0; i <= AVX_N_LEVELS; ++i)
                    o += t[i] > 0 ? std::snprintf(line + o, sizeof line - o, " %11.0f", t[i] * 1e6)
                                  : std::snprintf(line + o, sizeof line - o, " %11s", "-");
                std::snprintf(line + o, sizeof line - o, " %10.1f %11.1f", bytes / best / 1e9,
                              flops / best / 1e9);
                std::printf("%s\n", line);
            }
    }
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

/* ================================================================== weights of several planes
 *
 * The lm head and the zero-point grids at model size, the weight stored joined (abi/rad_planes.h)
 * through libavx's own hooks, on every team thread: these are the decode step's largest reads, a
 * memory benchmark at small M, so each rate is beside the DRAM triad on the same threads. The case
 * table cannot describe them (libref describes logical forms), so they are built here. */

/* A weight's canonical planes over [rows, cols]: codes and the planes after them, each dense. */
struct BenchPlanes {
    RadEncoding enc{};
    int64_t rows = 0, cols = 0;
    std::vector<std::vector<uint8_t>> data;
    std::vector<int64_t> pr, pc;
    BenchPlanes(const RadEncoding& e, int64_t r, int64_t c) : enc(e), rows(r), cols(c) {
        for (int i = 0; i < e.n_planes; ++i) {
            int64_t a = 0, b = 0;
            rad_enc_plane_dims(&e.plane[i], r, c, &a, &b);
            pr.push_back(a);
            pc.push_back(b);
            data.emplace_back((size_t)(a * rad_enc_row_bytes(e.plane[i].dtype, b)), 0);
        }
    }
    /* Plane i filled with element values v(r, c): floats in their dtype, codes as integers. */
    template <class F> void fill(int i, F v) {
        const uint32_t dt = enc.plane[i].dtype;
        const int64_t rb = rad_enc_row_bytes(dt, pc[(size_t)i]);
        for (int64_t r = 0; r < pr[(size_t)i]; ++r) {
            uint8_t* row = data[(size_t)i].data() + r * rb;
            for (int64_t c = 0; c < pc[(size_t)i]; ++c) {
                const float x = v(r, c);
                if (dt == RAD_F32 || dt == RAD_F16 || dt == RAD_BF16) rad_store_f32(row, dt, c, x);
                else rad_store_code(row, dt, c, (int)x);
            }
        }
    }
    /* The joined bytes of planes `sel` for operand `opd`, through `k`'s hooks. */
    bool join(const RadKernelInfo* k, const std::vector<RadParam>& p, int opd,
              const std::vector<int>& sel, std::vector<uint8_t>* out, uint32_t* dt) {
        std::vector<RadTensor> t;
        for (int i : sel) {
            RadTensor x{};
            x.dtype = enc.plane[i].dtype;
            x.rank = 2;
            x.shape[0] = pr[(size_t)i];
            x.shape[1] = pc[(size_t)i];
            x.stride[0] = x.shape[1];
            x.stride[1] = 1;
            x.data = data[(size_t)i].data();
            t.push_back(x);
        }
        RadLayout L{};
        if (!k->layout || k->layout(p.data(), (int)p.size(), opd, &enc, sel.data(), t.data(),
                                    (int)t.size(), &L) != RAD_OK)
            return false;
        out->assign((size_t)L.bytes, 0);
        *dt = L.dtype;
        return k->relayout(p.data(), (int)p.size(), opd, &enc, sel.data(), t.data(),
                           (int)t.size(), out->data(), L.bytes) == RAD_OK;
    }
};

static uint64_t bench_mix(uint64_t s, int64_t i) {
    uint64_t x = s * 0x9E3779B97F4A7C15ull + (uint64_t)i;
    x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
    return x;
}

static RadTensor bench_tensor(void* data, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = data;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) t.shape[i++] = s;
    rad_tensor_pack(&t);
    return t;
}

/* Seconds of `reps` launches after `warm`, launch i over copy i of the weight: a working set past
 * the last-level cache (128 MiB here), so a weight smaller than it is not timed from it
 * (libr4d/r4d_gemm_w4a16_nt_m64.hip makes the same point). The best of them, and the median in
 * `med`: on a machine with other load the best is the kernel's, the median the machine's. */
static double bench_time(const RadKernelInfo* k, const std::vector<RadArgs>& a, int warm,
                         int reps, double* med = nullptr) {
    for (int w = 0; w < warm; ++w) k->launch(&a[(size_t)w % a.size()], nullptr);
    std::vector<double> ts;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        const int rc = k->launch(&a[(size_t)(warm + r) % a.size()], nullptr);
        ts.push_back(now_s() - t0);
        if (rc < 0) { std::printf("    launch returned %d\n", rc); return -1; }
    }
    if (med) *med = median(ts);
    return *std::min_element(ts.begin(), ts.end());
}

/* Copies of a weight of `bytes` that make a working set of at least 1 GiB. */
static int bench_copies(double bytes) {
    const double want = (double)(1u << 30);
    return bytes >= want ? 1 : (int)std::ceil(want / bytes);
}

static double dram_triad(int threads) {
    const size_t n = (size_t)256 * 1024 * 1024 / sizeof(float);
    std::vector<float> A(n), B(n, 1.0f), C(n, 2.0f);
    std::vector<double> ts;
    for (int rep = 0; rep < 5; ++rep) {
        const double t0 = now_s();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int64_t i = 0; i < (int64_t)n; ++i) A[(size_t)i] = B[(size_t)i] + 0.5f * C[(size_t)i];
        ts.push_back(now_s() - t0);
    }
    return 3.0 * (double)n * sizeof(float) / median(ts) / 1e9;
}

/* A read-only stream, the traffic shape of a GEMM's weight at small M: 1 GiB summed as integers
 * (an integer sum vectorises without reassociating floats). The best of five: it is a ceiling. */
static double dram_read(int threads) {
    const size_t n = ((size_t)1 << 30) / sizeof(uint64_t);
    std::vector<uint64_t> A(n, 1);
    std::vector<double> ts;
    uint64_t sink = 0;
    for (int rep = 0; rep < 5; ++rep) {
        uint64_t sum = 0;
        const double t0 = now_s();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) reduction(+ : sum)
#endif
        for (int64_t i = 0; i < (int64_t)n; ++i) sum += A[(size_t)i];
        ts.push_back(now_s() - t0);
        sink += sum;
    }
    if (sink == 0) std::printf("(empty read)\n");
    return (double)n * sizeof(uint64_t) / *std::min_element(ts.begin(), ts.end()) / 1e9;
}

static void bench_joined(Lib& avx, Lib& ref, int threads, int reps, bool with_ref,
                         const std::string& only_op, int level) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    const int lvl = level >= 0 ? level : AVX_N_LEVELS - 1;
    if (avx.isa_force && avx.isa_force(lvl) != lvl) {
        std::printf("  level %d does not execute on this CPU\n", lvl);
        return;
    }
    /* Other load on the machine lowers a ceiling measured once; it is taken again after the
     * kernels, and the higher reading kept. */
    const double triad = dram_triad(threads);
    double rd = dram_read(threads);
    std::printf("\n== weights of several planes at model size, %s, %d thread(s), best of %d ==\n",
                avx.isa_name(lvl), threads, reps);
    std::printf("  DRAM on these threads: triad %.1f GB/s, read-only %.1f GB/s (measured again at "
                "the end)\n\n", triad, rd);
    /* The weight's stored bytes over the time, against the read; libref once, at M = 1. */
    struct Line { std::string op, geom; double t, med, tr, wbytes, flops; };
    std::vector<Line> lines;
    auto row = [&](const char* op, const char* geom, const RadKernelInfo* ka,
                   const RadKernelInfo* kr, const std::vector<RadArgs>& a, double wbytes,
                   double flops) {
        double med = 0;
        const double t = bench_time(ka, a, 2, reps, &med);
        const double tr = with_ref && kr ? bench_time(kr, a, 0, 1) : 0;
        if (t > 0) lines.push_back({ op, geom, t, med, tr, wbytes, flops });
    };

    /* ---- the lm head: the 27B's fp8 (block 128 x 128, bf16 scales) and Flash-Next's int8
     * (bf16 scale per row and 128 of K), against a bf16 activation into f32 logits. */
    struct Head { const char* name; uint32_t codes; int64_t br, N, K; };
    static const Head kHeads[] = {
        { "fp8_e4m3*bf16[128x128]", RAD_F8E4M3, 128, 248320, 5120 },
        { "i8*bf16[1x128]", RAD_I8, 1, 248320, 5120 },
        { "i8*bf16[1x128]", RAD_I8, 1, 248320, 2560 },
    };
    const RadKernelInfo* ka = avx.row("logits_gemm");
    const RadKernelInfo* kr = ref.row("logits_gemm");
    for (const Head& hd : kHeads) {
        if (!only_op.empty() && only_op != "logits_gemm") break;
        BenchPlanes w(rad_enc_affine(hd.codes, RAD_BF16, hd.br, 128), hd.N, hd.K);
        w.fill(0, [&](int64_t r, int64_t c) {
            const int v = (int)(bench_mix(7, r * hd.K + c) & 0xffu);
            return hd.codes == RAD_I8 ? (float)(int8_t)v : (float)(v == 0x7f || v == 0xff ? 0 : v);
        });
        w.fill(1, [](int64_t r, int64_t c) { return 0.001f * (float)(1 + (r + c) % 13); });
        for (int64_t M : { 1, 4, 64 }) {
            std::vector<RadParam> p = {
                RadParam{ "M", RAD_P_INT, M, 0, nullptr, 0 },
                RadParam{ "n_vocab", RAD_P_INT, hd.N, 0, nullptr, 0 },
                RadParam{ "n_embd", RAD_P_INT, hd.K, 0, nullptr, 0 },
                RadParam{ "dtype", RAD_P_STR, 0, 0, "bf16", 0 } };
            std::vector<uint8_t> j;
            uint32_t jdt = 0;
            if (!w.join(ka, p, 1, { 0, 1 }, &j, &jdt)) { std::printf("  join failed\n"); return; }
            std::vector<std::vector<uint8_t>> pool((size_t)bench_copies((double)j.size()) - 1, j);
            std::vector<uint16_t> x((size_t)(M * hd.K));
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = rad_f32_to_bf16(
                    (float)((int64_t)(bench_mix(9, (int64_t)i) % 2001) - 1000) * 1e-3f);
            std::vector<float> y((size_t)(M * hd.N));
            std::vector<std::vector<RadTensor>> ts;
            std::vector<RadArgs> a;
            for (size_t c = 0; c <= pool.size(); ++c) {
                void* wd = c ? pool[c - 1].data() : j.data();
                ts.push_back({ bench_tensor(x.data(), RAD_BF16, { M, hd.K }),
                               bench_tensor(wd, jdt, { hd.N, hd.K }),
                               bench_tensor(y.data(), RAD_F32, { M, hd.N }) });
            }
            for (auto& t : ts) {
                RadArgs x1{};
                x1.t = t.data(); x1.n_t = 3; x1.p = p.data(); x1.n_p = (int)p.size();
                x1.world_size = 1;
                a.push_back(x1);
            }
            char g[96];
            std::snprintf(g, sizeof g, "%s %lldx%lld M=%lld", hd.name, (long long)hd.N,
                          (long long)hd.K, (long long)M);
            row("logits_gemm", g, ka, M == 1 ? kr : nullptr, a, (double)j.size(),
                2.0 * (double)M * (double)hd.N * (double)hd.K);
        }
    }

    /* ---- gemm_nt_q's zero-point grids: the 2-bit draft head over the 27B's vocabulary, and
     * the 27B ParoQuant int4 / int5 MLP (gate|up in two rotated parts, then down). */
    struct Zp { const char* dtype; uint32_t codes; int64_t N, K, parts; std::vector<int64_t> Ms; };
    const Zp kZp[] = {
        { "w2a8", RAD_U2, 248320, 5120, 1, { 1, 4, 8 } },
        { "w4a16_asym", RAD_U4, 34816, 5120, 2, { 1, 4, 8 } },
        { "w4a16_asym", RAD_U4, 5120, 17408, 1, { 1, 4, 8 } },
        { "w5a16", RAD_U5, 34816, 5120, 2, { 1, 4, 8 } },
        { "w5a16", RAD_U5, 5120, 17408, 1, { 1, 4, 8 } },
        { "w4a8i8_asym", RAD_U4, 34816, 5120, 2, { 256 } },
        { "w5a8i8", RAD_U5, 34816, 5120, 2, { 256 } },
    };
    const RadKernelInfo* qa = avx.row("gemm_nt_q");
    const RadKernelInfo* qr = ref.row("gemm_nt_q");
    for (const Zp& z : kZp) {
        if (!only_op.empty() && only_op != "gemm_nt_q" && only_op != z.dtype) continue;
        const int64_t GS = 128, G = z.K / GS, P = z.parts;
        const int64_t top = (1 << rad_dtype_bits(z.codes)) - 1;
        RadEncoding e = rad_enc_affine(z.codes, RAD_F16, 1, GS);
        rad_enc_add_plane(&e, "zero", RAD_U8, 1, GS);
        BenchPlanes w(e, z.N, z.K);
        w.fill(0, [&](int64_t r, int64_t c) {
            return (float)(bench_mix(11, r * z.K + c) % (uint64_t)(top + 1));
        });
        w.fill(1, [](int64_t r, int64_t c) { return 0.01f * (float)(1 + (r + c) % 7); });
        w.fill(2, [&](int64_t r, int64_t c) { return (float)((r + c) % (top + 1)); });
        for (int64_t M : z.Ms) {
            std::vector<RadParam> p;
            p.reserve(6);
            p.push_back(RadParam{ "M", RAD_P_INT, M, 0, nullptr, 0 });
            p.push_back(RadParam{ "N", RAD_P_INT, z.N, 0, nullptr, 0 });
            p.push_back(RadParam{ "K", RAD_P_INT, z.K, 0, nullptr, 0 });
            p.push_back(RadParam{ "group", RAD_P_INT, GS, 0, nullptr, 0 });
            p.push_back(RadParam{ "dtype", RAD_P_STR, 0, 0, z.dtype, 0 });
            if (P > 1) p.push_back(RadParam{ "pb1", RAD_P_INT, z.N / 2, 0, nullptr, 0 });
            std::vector<uint8_t> j;
            uint32_t jdt = 0;
            if (!w.join(qa, p, 3, { 1, 2 }, &j, &jdt)) { std::printf("  join failed\n"); return; }
            const avxp::ZpGrid* zg = avxp::zp_grid(z.dtype);
            const int64_t AC = P * z.K, mpad = (M + 15) / 16 * 16;
            std::vector<uint8_t> a8((size_t)(M * AC));
            std::vector<uint16_t> a16((size_t)(M * AC));
            for (int64_t i = 0; i < M * AC; ++i) {
                a8[(size_t)i] = (uint8_t)(bench_mix(13, i) % 255 + 129);
                a16[(size_t)i] =
                    rad_f32_to_f16((float)((int64_t)(bench_mix(13, i) % 2001) - 1000) * 1e-3f);
            }
            std::vector<float> as((size_t)(M * P * G), 0.004f), asum((size_t)(G * mpad), 3.0f);
            std::vector<uint16_t> y((size_t)(M * z.N));
            const double wb = (double)(w.data[0].size() + j.size());
            const size_t ncopy = (size_t)bench_copies(wb);
            std::vector<std::vector<uint8_t>> cpool(ncopy - 1, w.data[0]), jpool(ncopy - 1, j);
            std::vector<std::vector<RadTensor>> ts;
            for (size_t c = 0; c < ncopy; ++c) {
                std::vector<RadTensor> t(7);
                if (zg->act == avxp::ZP_F16) t[0] = bench_tensor(a16.data(), RAD_F16, { M, AC });
                else                         t[0] = bench_tensor(a8.data(), RAD_I8, { M, AC });
                if (zg->act == avxp::ZP_I8_ROW) t[1] = bench_tensor(as.data(), RAD_F32, { M, 1 });
                if (zg->act == avxp::ZP_I8_GROUP)
                    t[1] = bench_tensor(as.data(), RAD_F32, { M, P * G });
                t[2] = bench_tensor(c ? cpool[c - 1].data() : w.data[0].data(), z.codes,
                                    { z.N, z.K });
                t[3] = bench_tensor(c ? jpool[c - 1].data() : j.data(), jdt, { z.N, z.K });
                t[4] = bench_tensor(y.data(), RAD_BF16, { M, z.N });
                if (zg->act == avxp::ZP_I8_ROW)
                    t[5] = bench_tensor(asum.data(), RAD_F32, { G, mpad });
                ts.push_back(std::move(t));
            }
            std::vector<RadArgs> a;
            for (auto& t : ts) {
                RadArgs x1{};
                x1.t = t.data(); x1.n_t = 7; x1.p = p.data(); x1.n_p = (int)p.size();
                x1.world_size = 1;
                a.push_back(x1);
            }
            char g[96];
            std::snprintf(g, sizeof g, "%s %lldx%lld%s M=%lld", z.dtype, (long long)z.N,
                          (long long)z.K, P > 1 ? " 2 parts" : "", (long long)M);
            row("gemm_nt_q", g, qa, M == 1 ? qr : nullptr, a, wb,
                2.0 * (double)M * (double)z.N * (double)z.K);
        }
    }
    rd = std::max(rd, dram_read(threads));
    std::printf("  read-only stream, the better of the two readings: %.1f GB/s\n\n", rd);
    std::printf("  %-12s %-44s %9s %9s %9s %8s %10s %10s\n", "op", "geometry", "ms", "median",
                "GB/s", "of read", "GFLOP/s", "libref ms");
    for (const Line& l : lines)
        std::printf("  %-12s %-44s %9.3f %9.3f %9.1f %7.0f%% %10.1f %10.1f\n", l.op.c_str(),
                    l.geom.c_str(), l.t * 1e3, l.med * 1e3, l.wbytes / l.t / 1e9,
                    100.0 * l.wbytes / l.t / 1e9 / rd, l.flops / l.t / 1e9, l.tr * 1e3);
    if (avx.isa_force) avx.isa_force(AVX_N_LEVELS - 1);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

/* ================================================================== dense GEMM forms at model size
 *
 * gemm_nt and every plain gemm_nt_q form a shipped model's linears use, at the linears' own shapes,
 * on --threads threads at one ISA level. A small M (decode, a verify window) reads each weight once
 * per launch, so it is timed over copies of the weight past the last-level cache and reported
 * against the read-only stream; a prefill M is compute-bound and is reported in GFLOP/s, beside the
 * level's FMA probe times the team (the bf16 dot product's ceiling is twice that). */
struct DenseForm {
    const char* name;      /* gemm_nt_q's dtype, or "bf16" for gemm_nt */
    uint32_t adt;          /* activation codes */
    int      a_scale;      /* 0 none, 1 one per row, 2 one per (row, 128 of K) */
    uint32_t bdt;          /* weight codes */
    uint32_t sdt;          /* weight scale dtype (0: none) */
    int64_t  sbr, sbc;     /* weight rows x columns one scale covers */
    int64_t  group;
};
static const DenseForm kDenseForms[] = {
    { "bf16",    RAD_BF16,   0, RAD_BF16,    0,        0,   0,   0 },
    { "fp8a8",   RAD_F8E4M3, 2, RAD_F8E4M3,  RAD_BF16, 128, 128, 128 },
    { "fp8a16",  RAD_BF16,   0, RAD_F8E4M3,  RAD_BF16, 128, 128, 128 },
    { "i8a8",    RAD_I8,     2, RAD_I8,      RAD_BF16, 1,   128, 128 },
    { "mxfp4a8", RAD_F8E4M3, 2, RAD_FP4E2M1, RAD_E8M0, 1,   32,  32 },
    { "w4a8",    RAD_I8,     1, RAD_I4,      RAD_F16,  1,   128, 128 },
    { "w8a8",    RAD_I8,     1, RAD_I8,      RAD_F16,  1,   128, 128 },
};

/* The linears each form serves, [N, K] (docs/ARCHITECTURES.md): the 27B's gate|up, down and
 * q|gate for bf16, block fp8 and MXFP4; Flash-Next's q|gate and o for its int8 trunk; Gemma 4
 * 31B's MLP for the int4 and int8 weights. */
static std::vector<std::pair<int64_t, int64_t>> dense_shapes(const DenseForm& f) {
    if (!std::strcmp(f.name, "i8a8")) return { { 12288, 2560 }, { 2560, 6144 } };
    if (!std::strcmp(f.name, "w4a8") || !std::strcmp(f.name, "w8a8"))
        return { { 21504, 5376 }, { 5376, 21504 } };
    return { { 34816, 5120 }, { 5120, 17408 }, { 12288, 5120 } };
}

static void bench_dense(Lib& avx, int threads, int level, const std::string& only,
                        const std::vector<int64_t>& Ms, int64_t shape_n, int64_t shape_k,
                        int reps) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    const int lvl = level >= 0 ? level : AVX_N_LEVELS - 1;
    if (avx.isa_force && avx.isa_force(lvl) != lvl) {
        std::printf("  level %d does not execute on this CPU\n", lvl);
        return;
    }
    double fma = 0;
    if (auto peak = (double (*)(int, int64_t))dlsym(avx.h, "avx_peak_fma")) {
        for (int r = 0; r < 3; ++r) {
            const double t0 = now_s();
            const double fl = peak(lvl, 1000000);
            fma = std::max(fma, fl / (now_s() - t0));
        }
    }
    double rd = dram_read(threads);
    std::printf("\n== dense GEMM forms at model size, %s, %d thread(s), best of %d ==\n",
                avx.isa_name(lvl), threads, reps);
    std::printf("  ceilings: FMA %.0f GFLOP/s x %d threads = %.0f; read-only DRAM %.1f GB/s\n\n",
                fma / 1e9, threads, fma * threads / 1e9, rd);
    std::printf("  %-8s %-14s %6s %10s %10s %10s %9s %8s\n", "form", "N x K", "M", "ms", "median",
                "GFLOP/s", "GB/s", "of read");
    for (const DenseForm& f : kDenseForms) {
        if (!only.empty() && only != f.name) continue;
        const bool plain = !std::strcmp(f.name, "bf16");
        const RadKernelInfo* k = avx.row(plain ? "gemm_nt" : "gemm_nt_q");
        if (!k) continue;
        auto shapes = dense_shapes(f);
        if (shape_n > 0 && shape_k > 0) shapes = { { shape_n, shape_k } };
        for (const auto& sh : shapes) {
            const int64_t N = sh.first, K = sh.second;
            const int64_t wbytes = rad_dtype_bytes(f.bdt, N * K);
            const int64_t sr = f.sdt ? (N + f.sbr - 1) / f.sbr : 0;
            const int64_t sc = f.sdt ? (K + f.sbc - 1) / f.sbc : 0;
            const int64_t sbytes = f.sdt ? rad_dtype_bytes(f.sdt, sr * sc) : 0;
            /* One weight, its copies made on demand for the small M that read it from DRAM. */
            std::vector<uint8_t> w0((size_t)wbytes), s0((size_t)sbytes);
            for (int64_t i = 0; i < wbytes; i += 8) {
                uint64_t r = bench_mix(17, i);
                if (f.bdt == RAD_F8E4M3)        /* no NaN codes: 0x7f and 0xff become 0x7e, 0xfe */
                    for (int b = 0; b < 8; ++b)
                        if (((r >> (8 * b)) & 0x7f) == 0x7f) r ^= (uint64_t)1 << (8 * b);
                if (f.bdt == RAD_BF16)          /* normal and finite: |w| in [2^-7, 2) */
                    r = (r & 0x80ff80ff80ff80ffull) | 0x3c003c003c003c00ull;
                std::memcpy(w0.data() + i, &r, (size_t)std::min<int64_t>(8, wbytes - i));
            }
            for (int64_t i = 0; i < sr * sc; ++i) {
                if (f.sdt == RAD_E8M0) s0[(size_t)i] = (uint8_t)(120 + bench_mix(19, i) % 8);
                else rad_store_f32(s0.data(), f.sdt, i, 0.001f * (float)(1 + i % 13));
            }
            for (int64_t M : Ms) {
                const size_t ncopy = M < 256 ? (size_t)bench_copies((double)(wbytes + sbytes)) : 1;
                std::vector<std::vector<uint8_t>> wc(ncopy - 1, w0), scp(ncopy - 1, s0);
                std::vector<uint8_t> a((size_t)rad_dtype_bytes(f.adt, M * K));
                for (int64_t i = 0; i < (int64_t)a.size(); ++i) {
                    uint8_t v = (uint8_t)bench_mix(23, i);
                    if (f.adt == RAD_F8E4M3 && (v & 0x7f) == 0x7f) v ^= 1;
                    if (f.adt == RAD_BF16 && (i & 1)) v = (uint8_t)(0x3c + (v & 3));  /* finite */
                    a[(size_t)i] = v;
                }
                const int64_t asc = f.a_scale == 2 ? (K + 127) / 128 : 1;
                std::vector<float> as((size_t)(M * asc), 0.01f);
                std::vector<uint16_t> y((size_t)(M * N));
                std::vector<std::vector<RadTensor>> ts;
                for (size_t c = 0; c < ncopy; ++c) {
                    void* wd = c ? wc[c - 1].data() : w0.data();
                    void* sd = c ? scp[c - 1].data() : s0.data();
                    if (plain) {
                        ts.push_back({ bench_tensor(a.data(), f.adt, { M, K }),
                                       bench_tensor(wd, f.bdt, { N, K }),
                                       bench_tensor(y.data(), RAD_BF16, { M, N }) });
                        continue;
                    }
                    std::vector<RadTensor> t(7);
                    t[0] = bench_tensor(a.data(), f.adt, { M, K });
                    if (f.a_scale) t[1] = bench_tensor(as.data(), RAD_F32, { M, asc });
                    t[2] = bench_tensor(wd, f.bdt, { N, K });
                    t[3] = bench_tensor(sd, f.sdt, { sr, sc });
                    t[4] = bench_tensor(y.data(), RAD_BF16, { M, N });
                    ts.push_back(std::move(t));
                }
                std::vector<RadParam> p = {
                    RadParam{ "M", RAD_P_INT, M, 0, nullptr, 0 },
                    RadParam{ "N", RAD_P_INT, N, 0, nullptr, 0 },
                    RadParam{ "K", RAD_P_INT, K, 0, nullptr, 0 },
                    RadParam{ "dtype", RAD_P_STR, 0, 0, f.name, 0 },
                    RadParam{ "group", RAD_P_INT, f.group, 0, nullptr, 0 } };
                std::vector<RadArgs> args;
                for (auto& t : ts) {
                    RadArgs x{};
                    x.t = t.data(); x.n_t = (int)t.size(); x.p = p.data();
                    x.n_p = plain ? 4 : 5;
                    x.world_size = 1;
                    args.push_back(x);
                }
                const double flops = 2.0 * (double)M * (double)N * (double)K;
                /* At least a second of launches, at most `reps`. */
                double med = 0;
                const double t1 = bench_time(k, args, 1, 1);
                const int r = t1 <= 0 ? 0 : std::max(3, std::min(reps, (int)(1.0 / t1)));
                const double t = r ? bench_time(k, args, 1, r, &med) : -1;
                if (t <= 0) continue;
                char g[32];
                std::snprintf(g, sizeof g, "%lldx%lld", (long long)N, (long long)K);
                std::printf("  %-8s %-14s %6lld %10.3f %10.3f %10.1f %9.1f %7.0f%%\n", f.name, g,
                            (long long)M, t * 1e3, med * 1e3, flops / t / 1e9,
                            (double)(wbytes + sbytes) / t / 1e9,
                            100.0 * (double)(wbytes + sbytes) / t / 1e9 / rd);
                std::fflush(stdout);
            }
        }
    }
    if (avx.isa_force) avx.isa_force(AVX_N_LEVELS - 1);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
}

/* ================================================================== per-op timing */
struct Row {
    std::string op, geom;
    double t_ref = 0;
    double t[AVX_N_LEVELS] = { 0, 0, 0, 0 };
    double bytes = 0, flops = 0;
};

/* Flushes every line of the operands one side of the case launches with, from every cache level,
 * so the launch reads its inputs from DRAM and writes its outputs into lines it does not hold.
 * Flushing the operands themselves is exact; sweeping a large buffer instead leaves some of them
 * behind on a part whose L3 favours reused lines over streamed ones. */
static void evict_operands(Prepared& pr, bool side_a) {
    for (Opd& o : pr.opd) {
        if (o.absent || o.n <= 0) continue;
        GBuf& g = side_a ? o.a : o.b;
        const uint8_t* p = g.data();
        for (size_t off = 0; off < g.size(); off += 64) _mm_clflush(p + off);
        if (g.size()) _mm_clflush(p + g.size() - 1);
    }
    GBuf& s = side_a ? pr.sa : pr.sb;
    for (size_t off = 0; off < s.size(); off += 64) _mm_clflush(s.data() + off);
    _mm_mfence();
}

static double time_launch(const RadKernelInfo* k, RadArgs* args, Prepared& pr, int warm,
                          int iters, int reps, bool cold) {
    for (int w = 0; w < warm; ++w) { k->launch(args, nullptr); h_reset(pr); }
    std::vector<double> ts;
    for (int r = 0; r < reps; ++r) {
        double acc = 0;
        for (int it = 0; it < iters; ++it) {
            if (cold) evict_operands(pr, args == &pr.aa);
            const double t0 = now_s();
            k->launch(args, nullptr);
            acc += now_s() - t0;
            /* Outside the clock: INOUT operands must start from the drawn bytes each time. */
            h_reset(pr);
        }
        ts.push_back(acc / iters);
    }
    return median(ts);
}

/* The indexer's descriptions draw each sequence's complete-block count and each query's position
 * at random (libref/ref_shapes.cpp), so a timing would cover a random share of the context. At
 * model shape every block is complete and every query sees all of them. */
static void model_shape(const Case& c, Prepared& pr) {
    auto param = [&](const char* k) {
        for (const P& p : c.p)
            if (!std::strcmp(p.k, k)) return (int64_t)p.i;
        return (int64_t)0;
    };
    auto set_all = [&](size_t k, int64_t v) {
        Opd& o = pr.opd[k];
        if (o.absent || o.d.dtype != RAD_I32) return;
        int32_t* w = (int32_t*)o.master.data();
        for (int64_t i = 0; i < o.n; ++i) w[i] = (int32_t)v;
        std::memcpy(o.a.data(), o.master.data(), o.master.size());
        std::memcpy(o.b.data(), o.master.data(), o.master.size());
    };
    if (!std::strcmp(c.op, "qsa_score")) {
        set_all(4, param("blocks"));                                   /* nc */
    } else if (!std::strcmp(c.op, "qsa_select")) {
        set_all(2, param("blocks"));                                   /* nc */
        set_all(3, param("blocks") * param("ratio") - 1);              /* pos */
    }
}

int main(int argc, char** argv) {
    std::string ref_path = "libref.so", avx_path = "libavx.so", only_op;
    bool do_peak = false, do_cvt = false, do_ops = true, do_moe = false, no_ref = false;
    bool cold = false, do_joined = false, joined_ref = false, do_dense = false;
    std::vector<int64_t> dense_m = { 1, 4, 8, 512 };
    int64_t shape_n = 0, shape_k = 0;
    std::string moe_dtypes = "w4nl64a8h";
    int threads = 1, only_level = -1;
    int iters = 20, reps = 7;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if      (s == "--ref")  ref_path = next();
        else if (s == "--avx")  avx_path = next();
        else if (s == "--op")   only_op = next();
        else if (s == "--peak") { do_peak = true; do_ops = false; }
        else if (s == "--cvt")  { do_cvt = true; do_ops = false; }
        else if (s == "--moe")  { do_moe = true; do_ops = false; }
        else if (s == "--moe-dtype") moe_dtypes = next();
        else if (s == "--all")  { do_peak = do_cvt = do_ops = true; }
        else if (s == "--joined") { do_joined = true; do_ops = false; }
        else if (s == "--joined-ref") { do_joined = joined_ref = true; do_ops = false; }
        else if (s == "--dense") { do_dense = true; do_ops = false; }
        else if (s == "--m") {
            dense_m.clear();
            for (const char* q = next(); *q;) {
                dense_m.push_back(std::strtoll(q, nullptr, 10));
                while (*q && *q != ',') ++q;
                if (*q == ',') ++q;
            }
        }
        else if (s == "--shape") {
            const char* q = next();
            shape_n = std::strtoll(q, nullptr, 10);
            const char* x = std::strchr(q, 'x');
            shape_k = x ? std::strtoll(x + 1, nullptr, 10) : 0;
        }
        else if (s == "--threads") threads = std::atoi(next());
        else if (s == "--level") only_level = std::atoi(next());
        else if (s == "--no-ref") no_ref = true;
        else if (s == "--cold") cold = true;
        else if (s == "--iters") iters = std::atoi(next());
        else if (s == "-h" || s == "--help") {
            std::printf(
                "rad-avx-bench -- what this machine can do, and what libavx gets of it.\n\n"
                "  --ref PATH      the reference plugin (default libref.so)\n"
                "  --avx PATH      the plugin under test (default libavx.so)\n"
                "  --peak          measure this machine's FMA and memory ceilings and stop\n"
                "  --cvt           measure the dtype conversions per ISA level and stop\n"
                "  --all           ceilings, conversions and every op\n"
                "  --joined        the lm head and the zero-point grids at model size, stored\n"
                "                  joined, on --threads threads (--joined-ref also times libref;\n"
                "                  --op narrows it to logits_gemm, gemm_nt_q or one grid;\n"
                "                  --level picks the ISA level, default the highest)\n"
                "  --dense         gemm_nt and the plain gemm_nt_q forms at their models' linear\n"
                "                  shapes, on --threads threads at --level (default the highest);\n"
                "                  --op narrows it to one form (bf16, fp8a8, fp8a16, i8a8,\n"
                "                  mxfp4a8, w4a8, w8a8)\n"
                "  --m LIST        --dense's row counts, comma-separated (default 1,4,8,512)\n"
                "  --shape NxK     --dense at this one weight shape\n"
                "  --op NAME       only this op\n"
                "  --moe           moe_gemm_q over Qwen3.8-Flash-Next's codebook experts at the\n"
                "                  model's geometry (1.3 GB of experts), and stop\n"
                "  --moe-dtype L   the codebook forms --moe runs, comma-separated (default\n"
                "                  w4nl64a8h; also w4nla8h, w4nl32a8h, w5nl64a8h)\n"
                "  --threads N     OpenMP threads for the ceilings, the kernels, --moe and --joined\n"
                "                  (default 1)\n"
                "  --level L       time only ISA level L (0 scalar .. 3 avx512)\n"
                "  --no-ref        do not time libref (it is not timed past 2e10 FLOP anyway)\n"
                "  --cold          flush the operands from every cache before each timed launch\n"
                "  --iters N       timed iterations per rep (default 20)\n");
            return 0;
        }
    }

#ifdef _OPENMP
    /* One thread by default: most case geometries are small (to keep the checker fast), and a wide
     * fork/join on them would measure the barrier. --threads sets the team for a scaling
     * measurement, of the kernels, the ceilings and --moe alike. */
    omp_set_num_threads(threads > 0 ? threads : 1);
#endif

    Lib ref, avx;
    if (!ref.open(ref_path.c_str()) || !avx.open(avx_path.c_str())) return 2;
    std::printf("oracle : %s %s\nunder  : %s %s\n", ref.info->name, ref.info->version,
                avx.info->name, avx.info->version);
    if (avx.isa_detect)
        std::printf("cpu    : highest executable level is %s\n", avx.isa_name(avx.isa_detect()));

    if (do_peak) bench_peak(avx, threads);
    if (do_cvt)  bench_cvt(avx, ref);
    if (do_moe)  bench_moe(avx, ref, moe_dtypes, threads);
    if (do_joined) bench_joined(avx, ref, threads, 7, joined_ref, only_op, only_level);
    if (do_dense) bench_dense(avx, threads, only_level, only_op, dense_m, shape_n, shape_k, 7);
    if (!do_ops) return 0;

    const std::vector<Case> cases = build_cases();
    std::vector<Row> rows;

    for (const Case& c : cases) {
        if (!only_op.empty() && only_op != c.op) continue;
        const RadKernelInfo* kr = ref.row(c.op);
        const RadKernelInfo* ka = avx.row(c.op);
        if (!kr || !ka) continue;
        Prepared pr = h_prepare(ref, kr, c, kr, ka);
        if (!pr.ok) continue;
        model_shape(c, pr);
        if (ka->launch(&pr.ab, nullptr) == RAD_E_UNSUPPORTED) continue;
        h_reset(pr);

        Row r;
        r.op = c.op;
        r.geom = case_str(c);
        r.bytes = c.bytes;
        r.flops = c.flops;
        /* Fewer iterations for libref, which is one to two orders of magnitude slower, and none
         * for a case whose one launch would take it minutes. */
        if (!no_ref && c.flops <= 2e10) r.t_ref = time_launch(kr, &pr.aa, pr, 2, 3, 3, cold);
        /* A case of a second or more is timed fewer times. */
        const bool big = c.flops > 2e10;
        for (int lvl = 0; lvl < AVX_N_LEVELS; ++lvl) {
            if (only_level >= 0 && lvl != only_level) { r.t[lvl] = 0; continue; }
            if (avx.isa_force && avx.isa_force(lvl) != lvl) { r.t[lvl] = 0; continue; }
            r.t[lvl] = big ? time_launch(ka, &pr.ab, pr, 1, 2, 3, cold)
                           : time_launch(ka, &pr.ab, pr, 5, iters, reps, cold);
        }
        rows.push_back(r);
    }

    std::printf("\n== per op, microseconds per launch (median of %d reps x %d iters, %d thread%s, "
                "%s) ==\n", reps, iters, std::max(threads, 1), threads > 1 ? "s" : "",
                cold ? "operands flushed before each launch" : "cache-warm");
    std::printf("%-22s %9s %9s %9s %9s %9s %8s  %s\n", "op", "libref", "scalar", "avx", "avx2",
                "avx512", "x(ref)", "rate");
    for (const Row& r : rows) {
        char rate[64] = "";
        /* The highest level timed. */
        double best = 0;
        for (int lvl = 0; lvl < AVX_N_LEVELS; ++lvl)
            if (r.t[lvl] > 0) best = r.t[lvl];
        if (best > 0 && r.flops > 0)
            std::snprintf(rate, sizeof rate, "%.1f GFLOP/s", r.flops / best / 1e9);
        else if (best > 0 && r.bytes > 0)
            std::snprintf(rate, sizeof rate, "%.1f GB/s", r.bytes / best / 1e9);
        if (r.t_ref > 0) std::printf("%-22s %9.2f", r.op.c_str(), r.t_ref * 1e6);
        else             std::printf("%-22s %9s", r.op.c_str(), "n/a");
        for (int lvl = 0; lvl < AVX_N_LEVELS; ++lvl) {
            if (r.t[lvl] > 0) std::printf(" %9.2f", r.t[lvl] * 1e6);
            else              std::printf(" %9s", "n/a");
        }
        std::printf(" %8.1f  %-14s  %s\n", best > 0 ? r.t_ref / best : 0.0, rate, r.geom.c_str());
    }
    std::printf("\nx(ref) is the speedup over libref's scalar loop; judge libavx by the rate column\n"
                "against the --peak ceilings.\n");
    return 0;
}
