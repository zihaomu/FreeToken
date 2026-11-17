/* r4d_hsimd.cpp -- CPU detection and dispatch for r4d_hsimd.h.
 *
 * Baseline-compiled (no -m flags): it decides which levels may run, so it must not itself contain
 * an instruction above the baseline. The tables are the per-level objects of r4d_hsimd_kernels.cpp;
 * off x86-64 only the scalar one exists. Detection as the core's util/simd.cpp.
 */
#include "r4d_hsimd.h"

#include <atomic>

#if defined(__x86_64__)
#define R4D_HSIMD_X86 1
#endif

extern "C" {
#define R4D_HSIMD_DECLS(sfx)                                                                   \
    void r4d_hsimd_moew4_permute##sfx(const uint8_t*, uint8_t*, int64_t, bool);               \
    void r4d_hsimd_iqr_trellis_tile##sfx(const uint8_t*, int64_t, int, int64_t, uint8_t*);          \
    void r4d_hsimd_w2a8_tile##sfx(const uint8_t*, int64_t, int64_t, uint8_t*);                    \
    void r4d_hsimd_dword_transpose16##sfx(const uint32_t*, int64_t, int64_t, uint32_t*);          \
    void r4d_hsimd_byte_transpose##sfx(const uint8_t*, int64_t, int64_t, uint8_t*);               \
    void r4d_hsimd_fp8_frag_tile##sfx(const uint8_t*, int64_t, int64_t, uint8_t*)
R4D_HSIMD_DECLS(_sc);
#ifdef R4D_HSIMD_X86
R4D_HSIMD_DECLS(_s4);
R4D_HSIMD_DECLS(_v1);
R4D_HSIMD_DECLS(_v2);
R4D_HSIMD_DECLS(_v3);
#endif
#undef R4D_HSIMD_DECLS
}

namespace r4d_hsimd {
namespace {

struct Table {
    void (*moew4_permute)(const uint8_t*, uint8_t*, int64_t, bool);
    void (*iqr_trellis_tile)(const uint8_t*, int64_t, int, int64_t, uint8_t*);
    void (*w2a8_tile)(const uint8_t*, int64_t, int64_t, uint8_t*);
    void (*dword_transpose16)(const uint32_t*, int64_t, int64_t, uint32_t*);
    void (*byte_transpose)(const uint8_t*, int64_t, int64_t, uint8_t*);
    void (*fp8_frag_tile)(const uint8_t*, int64_t, int64_t, uint8_t*);
};

#define R4D_HSIMD_TABLE(sfx) \
    { r4d_hsimd_moew4_permute##sfx, r4d_hsimd_iqr_trellis_tile##sfx, r4d_hsimd_w2a8_tile##sfx, \
      r4d_hsimd_dword_transpose16##sfx, r4d_hsimd_byte_transpose##sfx, r4d_hsimd_fp8_frag_tile##sfx }
const Table kTables[kLevels] = {
    R4D_HSIMD_TABLE(_sc),
#ifdef R4D_HSIMD_X86
    R4D_HSIMD_TABLE(_s4), R4D_HSIMD_TABLE(_v1), R4D_HSIMD_TABLE(_v2), R4D_HSIMD_TABLE(_v3),
#else
    R4D_HSIMD_TABLE(_sc), R4D_HSIMD_TABLE(_sc), R4D_HSIMD_TABLE(_sc), R4D_HSIMD_TABLE(_sc),
#endif
};
#undef R4D_HSIMD_TABLE

/* The live level, read on every call (a relaxed load) so force_level takes effect between calls. */
std::atomic<int> g_level{ -1 };

const Table& live() {
    int l = g_level.load(std::memory_order_relaxed);
    if (l < 0) {
        l = detect_level();
        g_level.store(l, std::memory_order_relaxed);
    }
    return kTables[l];
}

}  // namespace

int detect_level() {
    static const int lvl = [] {
#ifdef R4D_HSIMD_X86
        __builtin_cpu_init();
        /* AVX-512 as the conjunction F && BW && DQ && VL: Knights Landing has F without the
         * others. */
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
            __builtin_cpu_supports("avx2"))
            return (int)kAvx512;
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
            __builtin_cpu_supports("f16c"))
            return (int)kAvx2;
        if (__builtin_cpu_supports("avx")) return (int)kAvx;
        if (__builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("sse4.1") &&
            __builtin_cpu_supports("popcnt"))
            return (int)kSse4;
#endif
        return (int)kScalar;
    }();
    return lvl;
}

int level() {
    const int l = g_level.load(std::memory_order_relaxed);
    return l < 0 ? detect_level() : l;
}

const char* level_name(int lvl) {
    switch (lvl) {
        case kScalar: return "scalar";
        case kSse4:   return "sse4";
        case kAvx:    return "avx";
        case kAvx2:   return "avx2";
        case kAvx512: return "avx512";
        default:      return "?";
    }
}

int force_level(int lvl) {
    const int top = detect_level();
    if (lvl > top) lvl = top;
    if (lvl < 0) lvl = 0;
    g_level.store(lvl, std::memory_order_relaxed);
    return lvl;
}

void moew4_permute(const uint8_t* s, uint8_t* d, int64_t bytes, bool forward) {
    live().moew4_permute(s, d, bytes, forward);
}

void iqr_trellis_tile(const uint8_t* rows, int64_t rb, int B, int64_t nkb, uint8_t* out) {
    live().iqr_trellis_tile(rows, rb, B, nkb, out);
}

void w2a8_tile(const uint8_t* rows, int64_t rb, int64_t kpb, uint8_t* out) {
    live().w2a8_tile(rows, rb, kpb, out);
}

void dword_transpose16(const uint32_t* in, int64_t stride, int64_t cols, uint32_t* out) {
    live().dword_transpose16(in, stride, cols, out);
}

void byte_transpose(const uint8_t* in, int64_t rows, int64_t cols, uint8_t* out) {
    live().byte_transpose(in, rows, cols, out);
}

void fp8_frag_tile(const uint8_t* rows, int64_t K, int64_t ksuper, uint8_t* out) {
    live().fp8_frag_tile(rows, K, ksuper, out);
}

}  // namespace r4d_hsimd
