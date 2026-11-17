/* simd.cpp -- CPU detection and dispatch for util/simd.h.
 *
 * Baseline-compiled (no -m flags): it decides which levels may run, so it must not itself
 * contain an instruction above the baseline. The tables come from the per-level objects of
 * simd_kernels.cpp; levels this build does not have (a non-x86 build has only the scalar one)
 * point at the scalar table. Detection mirrors libavx's avx_dispatch.cpp: __builtin_cpu_supports
 * rather than raw CPUID, because it also checks XGETBV -- a CPU may enumerate AVX-512 while the
 * OS has not enabled ZMM state, and executing it then faults.
 */
#include "util/simd.h"

#include <atomic>

extern "C" {
#define RAD_SIMD_DECLS(sfx)                                                                     \
    int32_t rad_simd_max_i32##sfx(const int32_t*, size_t);                                      \
    size_t  rad_simd_count_eq_i32##sfx(const int32_t*, size_t, int32_t);                        \
    int64_t rad_simd_find_eq_i64##sfx(const int64_t*, size_t, int64_t);                         \
    size_t  rad_simd_first_non_ascii##sfx(const char*, size_t);                                 \
    size_t  rad_simd_first_json_special##sfx(const char*, size_t);                              \
    void    rad_simd_bitstream_shr##sfx(uint32_t*, const uint32_t*, size_t, unsigned);          \
    void    rad_simd_bitmap_marks##sfx(uint8_t*, const int32_t*, size_t, const uint32_t*, size_t); \
    void    rad_simd_or3_u8##sfx(uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, size_t); \
    void    rad_simd_or_adjacent_u8##sfx(uint8_t*, const uint8_t*, size_t);                     \
    void    rad_simd_madd_i32_i64##sfx(int64_t*, const int32_t*, int32_t, size_t);              \
    size_t  rad_simd_b64_encode_run##sfx(const uint8_t*, size_t, char*);                         \
    size_t  rad_simd_b64_decode_run##sfx(const uint8_t*, size_t, uint8_t*);                      \
    void    rad_simd_stripe_acc_u64x8##sfx(uint64_t*, const uint8_t*, size_t);                 \
    void    rad_simd_level_bits_f32##sfx(const float*, size_t, float, float, uint64_t*, uint64_t*, \
                                         uint64_t*, uint64_t*)
RAD_SIMD_DECLS(_sc);
#ifdef RAD_SIMD_HAVE_X86_LEVELS
RAD_SIMD_DECLS(_s4);
RAD_SIMD_DECLS(_v1);
RAD_SIMD_DECLS(_v2);
RAD_SIMD_DECLS(_v3);
#endif
#undef RAD_SIMD_DECLS
}

namespace rad {
namespace simd {
namespace {

struct Table {
    int32_t (*max_i32)(const int32_t*, size_t);
    size_t  (*count_eq_i32)(const int32_t*, size_t, int32_t);
    int64_t (*find_eq_i64)(const int64_t*, size_t, int64_t);
    size_t  (*first_non_ascii)(const char*, size_t);
    size_t  (*first_json_special)(const char*, size_t);
    void    (*bitstream_shr)(uint32_t*, const uint32_t*, size_t, unsigned);
    void    (*bitmap_marks)(uint8_t*, const int32_t*, size_t, const uint32_t*, size_t);
    void    (*or3_u8)(uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, size_t);
    void    (*or_adjacent_u8)(uint8_t*, const uint8_t*, size_t);
    void    (*madd_i32_i64)(int64_t*, const int32_t*, int32_t, size_t);
    size_t  (*b64_encode_run)(const uint8_t*, size_t, char*);
    size_t  (*b64_decode_run)(const uint8_t*, size_t, uint8_t*);
    void    (*stripe_acc_u64x8)(uint64_t*, const uint8_t*, size_t);
    void    (*level_bits_f32)(const float*, size_t, float, float, uint64_t*, uint64_t*, uint64_t*,
                              uint64_t*);
};

#define RAD_SIMD_TABLE(sfx)                                                                     \
    { rad_simd_max_i32##sfx, rad_simd_count_eq_i32##sfx, rad_simd_find_eq_i64##sfx,             \
      rad_simd_first_non_ascii##sfx, rad_simd_first_json_special##sfx,                         \
      rad_simd_bitstream_shr##sfx, rad_simd_bitmap_marks##sfx, rad_simd_or3_u8##sfx,          \
      rad_simd_or_adjacent_u8##sfx, rad_simd_madd_i32_i64##sfx, rad_simd_b64_encode_run##sfx, \
      rad_simd_b64_decode_run##sfx, rad_simd_stripe_acc_u64x8##sfx, rad_simd_level_bits_f32##sfx }

const Table kTables[kLevels] = {
    RAD_SIMD_TABLE(_sc),
#ifdef RAD_SIMD_HAVE_X86_LEVELS
    RAD_SIMD_TABLE(_s4), RAD_SIMD_TABLE(_v1), RAD_SIMD_TABLE(_v2), RAD_SIMD_TABLE(_v3),
#else
    RAD_SIMD_TABLE(_sc), RAD_SIMD_TABLE(_sc), RAD_SIMD_TABLE(_sc), RAD_SIMD_TABLE(_sc),
#endif
};
#undef RAD_SIMD_TABLE

}  /* namespace */

int detect_level() {
    static const int lvl = [] {
#if defined(__x86_64__) && defined(RAD_SIMD_HAVE_X86_LEVELS)
        __builtin_cpu_init();
        /* popcnt rides along from the sse4 level up (count_eq_i32 uses it); it has its own
         * CPUID bit, paired with SSE4.2 on every shipping part but checked rather than assumed. */
        if (!__builtin_cpu_supports("popcnt")) return (int)kScalar;
        /* AVX-512 as the conjunction F && BW && DQ && VL, as in libavx: Knights Landing has F
         * without the others. */
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
            __builtin_cpu_supports("avx2"))
            return (int)kAvx512;
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
            __builtin_cpu_supports("f16c"))
            return (int)kAvx2;
        if (__builtin_cpu_supports("avx")) return (int)kAvx;
        if (__builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("sse4.1"))
            return (int)kSse4;
#endif
        return (int)kScalar;
    }();
    return lvl;
}

namespace {
/* The live level. Read on every call (one relaxed atomic load) so force_level takes effect for
 * the test that flips it between calls; production never writes it after static init. */
std::atomic<int> g_level{ -1 };

const Table& live() {
    int l = g_level.load(std::memory_order_relaxed);
    if (l < 0) {
        l = detect_level();
        g_level.store(l, std::memory_order_relaxed);
    }
    return kTables[l];
}
}  /* namespace */

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

int32_t max_i32(const int32_t* v, size_t n) { return live().max_i32(v, n); }
size_t  count_eq_i32(const int32_t* v, size_t n, int32_t x) {
    return live().count_eq_i32(v, n, x);
}
int64_t find_eq_i64(const int64_t* v, size_t n, int64_t x) {
    return live().find_eq_i64(v, n, x);
}
size_t  first_non_ascii_large(const char* p, size_t n) { return live().first_non_ascii(p, n); }
size_t  first_json_special_large(const char* p, size_t n) { return live().first_json_special(p, n); }
void    bitstream_shr(uint32_t* dst, const uint32_t* src, size_t n_words, unsigned sh) {
    live().bitstream_shr(dst, src, n_words, sh);
}
void    bitmap_marks(uint8_t* dst, const int32_t* ids, size_t n, const uint32_t* bits,
                     size_t n_words) {
    live().bitmap_marks(dst, ids, n, bits, n_words);
}

void    or3_u8_impl(uint8_t* dst, const uint8_t* a, const uint8_t* b, const uint8_t* c, size_t n) {
    live().or3_u8(dst, a, b, c, n);
}
void    or_adjacent_u8(uint8_t* dst, const uint8_t* s, size_t n) { live().or_adjacent_u8(dst, s, n); }
void    madd_i32_i64(int64_t* acc, const int32_t* h, int32_t c, size_t n) {
    live().madd_i32_i64(acc, h, c, n);
}
size_t  b64_encode_run(const uint8_t* in, size_t n, char* out) { return live().b64_encode_run(in, n, out); }
size_t  b64_decode_run(const uint8_t* in, size_t n, uint8_t* out) {
    return live().b64_decode_run(in, n, out);
}
void    stripe_acc_u64x8(uint64_t acc[8], const uint8_t* p, size_t stripes) {
    live().stripe_acc_u64x8(acc, p, stripes);
}

void    level_bits_f32(const float* v, size_t n, float lv, float band, uint64_t* gt, uint64_t* lt,
                       uint64_t* nan, uint64_t* near) {
    live().level_bits_f32(v, n, lv, band, gt, lt, nan, near);
}

}  /* namespace simd */
}  /* namespace rad */
