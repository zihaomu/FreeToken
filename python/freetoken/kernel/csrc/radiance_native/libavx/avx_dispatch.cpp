/* avx_dispatch.cpp -- ISA detection and level selection; must contain no vector instruction.
 *
 * Every other object in the plugin is built with -mavx/-mavx2/-mavx512f and may only execute once
 * this file has established that the CPU supports it. So this file is built with no -m flags, set
 * as a target property in CMakeLists.txt. Avoiding intrinsics is not enough: under -mavx512f the
 * compiler may auto-vectorise anything, such as the table copies below, and a SIGILL here would
 * hit before main on exactly the machines the fallback is for.
 *
 * Detection uses __builtin_cpu_supports, which also reads XCR0 via XGETBV: the OS must enable
 * YMM/ZMM/mask state, and a VM or older kernel may leave it off while CPUID still reports AVX-512.
 * AVX-512 needs F, BW, DQ and VL together: BW for the byte and word shuffles of the bf16 pack, DQ
 * for the 64-bit integer forms, VL for the 128/256-bit encodings.
 */
#include "avx_isa.h"
#include "avx_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <new>
#include <vector>

/* ================================================================== the fp8 decode tables
 *
 * Built at load from the ABI's converters rather than typed out, so they cannot drift from
 * rad_fp8e4m3_to_f32 / rad_fp8e5m2_to_f32. */
extern "C" { float avx_fp8e4m3_tab[256] = { 0 };
float avx_fp8e5m2_tab[256] = { 0 }; }

static void fp8_tables_init(void) {
    for (int i = 0; i < 256; ++i) {
        avx_fp8e4m3_tab[i] = rad_fp8e4m3_to_f32((uint8_t)i);
        avx_fp8e5m2_tab[i] = rad_fp8e5m2_to_f32((uint8_t)i);
    }
}

/* ================================================================== detection
 * Atomics, since the first launches may run on several threads at once; every racer computes the
 * same values. */
static std::atomic<int> g_detected{-1};
static std::atomic<int> g_level{-1};

static int detect(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl"))
        return AVX_LEVEL_AVX512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        __builtin_cpu_supports("f16c"))
        return AVX_LEVEL_AVX2;
    if (__builtin_cpu_supports("avx")) return AVX_LEVEL_AVX;
    return AVX_LEVEL_SCALAR;
#else
    /* Not x86: the scalar level is a full implementation, just slow; rad_plugin_info().build_target
     * reports "host". */
    return AVX_LEVEL_SCALAR;
#endif
}

extern "C" int avx_isa_detect(void) {
    int d = g_detected.load(std::memory_order_relaxed);
    if (d < 0) g_detected.store(d = detect(), std::memory_order_relaxed);
    return d;
}

extern "C" const char* avx_isa_name(int level) {
    switch (level) {
        case AVX_LEVEL_SCALAR: return "scalar";
        case AVX_LEVEL_AVX:    return "avx";
        case AVX_LEVEL_AVX2:   return "avx2";
        case AVX_LEVEL_AVX512: return "avx512";
        default:               return "?";
    }
}

extern "C" const AvxKernelTable* avx_table_for(int level) {
    switch (level) {
        case AVX_LEVEL_AVX512: return &avx_table_v3;
        case AVX_LEVEL_AVX2:   return &avx_table_v2;
        case AVX_LEVEL_AVX:    return &avx_table_v1;
        default:               return &avx_table_sc;
    }
}

static std::atomic<const AvxKernelTable*> g_tab{nullptr};

extern "C" const AvxKernelTable* avx_table_live(void) {
    const AvxKernelTable* t = g_tab.load(std::memory_order_acquire);
    if (t) return t;
    const int level = avx_isa_detect();
    g_level.store(level, std::memory_order_relaxed);
    t = avx_table_for(level);
    g_tab.store(t, std::memory_order_release);
    return t;
}

extern "C" int avx_isa_level(void) {
    if (g_level.load(std::memory_order_relaxed) < 0) (void)avx_table_live();
    return g_level.load(std::memory_order_relaxed);
}

extern "C" int avx_isa_force(int level) {
    const int cap = avx_isa_detect();
    /* Clamp to what the CPU can execute and return the level selected, so a checker can report
     * that it ran a lower level than it asked for. */
    if (level < 0) level = 0;
    if (level > cap) level = cap;
    g_level.store(level, std::memory_order_relaxed);
    g_tab.store(avx_table_for(level), std::memory_order_release);
    return level;
}

/* ================================================================== the thread override
 *
 * Every check geometry is below the kernels' parallel thresholds, so without this the checker
 * would run each OpenMP body on one thread only. */
#ifdef _OPENMP
std::atomic<int> avx_par_threads{0};
#endif

extern "C" int avx_par_force(int threads) {
#ifdef _OPENMP
    if (threads < 0) threads = 0;
    avx_par_threads.store(threads, std::memory_order_relaxed);
    return threads;
#else
    (void)threads;
    return 0;
#endif
}

/* ================================================================== the thunks
 *
 * One baseline-compiled function per op, pointed at by the registry row, making one indirect call
 * into the selected level's table. The rows themselves are not repointed: rad_kernel_at returns
 * them as const from a static table, and the loader may read them before rad_plugin_open runs, so
 * a row patched in open() would be briefly null. The extra indirect branch is not measurable. */
#define AVX_THUNK(nm)                                                                             \
    extern "C" int avx_thunk_##nm(const RadArgs* a, RadStream s) {                                \
        return avx_table_live()->nm(a, s);                                                        \
    }
AVX_OP_LIST(AVX_THUNK)
#undef AVX_THUNK

/* ================================================================== scratch
 *
 * One growable arena per thread per slot. Only a call larger than any before it allocates, so the
 * hot path is allocation-free.
 *
 * Aligned to 64 so _mm512_load_ps is legal on it (`new float[]` guarantees only 16). Kernels mostly
 * use unaligned loads, which cost nothing extra on aligned addresses since Nehalem, but the GEMM
 * packing buffers use aligned stores. */
namespace {

struct Arena {
    void*  p = nullptr;
    size_t bytes = 0;
    ~Arena() { std::free(p); }
    void* get(size_t want) {
        if (want <= bytes) return p;
        /* Grow geometrically so increasing sizes do not realloc every call. A request past half
         * the address space is taken as asked, so the doubling cannot wrap to zero and spin. */
        size_t nb = bytes ? bytes : 4096;
        while (nb < want) nb = nb > SIZE_MAX / 2 ? want : nb * 2;
        void* q = nb <= SIZE_MAX - 63 ? std::aligned_alloc(64, (nb + 63) & ~(size_t)63) : nullptr;
        /* Allocation failure aborts. Callers do not check the result (most run inside an OpenMP
         * body that cannot return a status), so returning the old smaller buffer would overflow
         * the heap and returning null would fault with no diagnostic. */
        if (!q) {
            std::fprintf(stderr, "libavx: could not allocate %zu bytes of kernel scratch\n", want);
            std::abort();
        }
        std::free(p);
        p = q; bytes = nb;
        return p;
    }
};

thread_local Arena g_a, g_b, g_c, g_raw;

}  /* namespace */

namespace avx {
float* scratch_f32  (size_t n) { return (float*)g_a.get(n * sizeof(float)); }
float* scratch_f32_b(size_t n) { return (float*)g_b.get(n * sizeof(float)); }
float* scratch_f32_c(size_t n) { return (float*)g_c.get(n * sizeof(float)); }
void*  scratch_raw  (size_t b) { return g_raw.get(b); }
}

/* ================================================================== plugin open/close
 * The loader resolves these exact names (rad_abi.h); a misspelling would silently skip init and
 * leave the fp8 tables zero. The typedef checks below make a signature drift a compile error. */
extern "C" int rad_plugin_open(void) {
    fp8_tables_init();
    (void)avx_table_live();
    return RAD_OK;
}

extern "C" void rad_plugin_close(void) {}

[[maybe_unused]] static const RadPluginOpenFn  kOpenCheck  = rad_plugin_open;
[[maybe_unused]] static const RadPluginCloseFn kCloseCheck = rad_plugin_close;

/* ================================================================== the measured FMA ceiling
 *
 * Exported for rad-avx-bench, which is baseline-compiled and cannot emit vector code itself. The
 * per-level probes live in avx_support.cpp, built with the kernels' flags, so they measure the
 * ceiling the kernels see, including any AVX-512 frequency offset under sustained load.
 *
 * Returns the FLOPs performed, not a rate; the caller owns the clock and takes the median. */
extern "C" double avx_peak_fma_sc(int64_t);
extern "C" double avx_peak_fma_v1(int64_t);
extern "C" double avx_peak_fma_v2(int64_t);
extern "C" double avx_peak_fma_v3(int64_t);

extern "C" double avx_peak_fma(int level, int64_t iters) {
    switch (level) {
        case AVX_LEVEL_AVX512: return avx_peak_fma_v3(iters);
        case AVX_LEVEL_AVX2:   return avx_peak_fma_v2(iters);
        case AVX_LEVEL_AVX:    return avx_peak_fma_v1(iters);
        default:               return avx_peak_fma_sc(iters);
    }
}
