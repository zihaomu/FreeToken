/* simd.h -- small exact integer/bit primitives for the core's host hot paths, dispatched by CPU
 * feature like libavx (libavx/avx_isa.h): the shipped binary is built once and runs on machines
 * it was not built on, so no -march=native. simd_kernels.cpp is compiled once per level into
 * separate objects with their own -m flags; simd.cpp (baseline, no -m flags) picks a table once
 * at first use via __builtin_cpu_supports, which also checks the OS has enabled the register
 * state (XGETBV), so a level the CPU enumerates but the OS has not enabled is never selected.
 *
 * Every function here is integer or bit arithmetic, or float comparison, with a single exact
 * result: all levels must agree bit for bit on every input, which simd_test cases in tests/hostbench_test.cpp check by
 * forcing each level in turn (simd_force_level) against a scalar reference written in the test.
 * Nothing with floating-point reduction order belongs here.
 *
 * Levels:
 *   0 scalar  x86-64 baseline (SSE2 exists but nothing is hand-vectorised)
 *   1 sse4    SSE4.1 + SSE4.2 (pmaxsd, pmovzx, pcmpeqq)
 *   2 avx     AVX (256-bit float only; integer helpers split into 128-bit halves)
 *   3 avx2    AVX2 + FMA + F16C (256-bit integer)
 *   4 avx512  AVX-512 F/BW/DQ/VL (the conjunction libavx checks)
 *
 * On a non-x86 build only the scalar level exists and the dispatcher selects it.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rad {
namespace simd {

enum Level {
    kScalar = 0,
    kSse4   = 1,
    kAvx    = 2,
    kAvx2   = 3,
    kAvx512 = 4,
    kLevels = 5
};

/* The highest level this CPU+OS can execute. Computed once. */
int         detect_level();
/* The level in use (detect_level() unless forced). */
int         level();
const char* level_name(int lvl);
/* Force a level, clamped to detect_level(); for tests only, like avx_isa_force. Returns the
 * level now in use. */
int         force_level(int lvl);

/* Largest of v[0..n). n == 0 returns INT32_MIN (callers size-check; the value is the identity
 * for max, so folding it into a running maximum is harmless). */
int32_t max_i32(const int32_t* v, size_t n);

/* How many of v[0..n) equal x. */
size_t count_eq_i32(const int32_t* v, size_t n, int32_t x);

/* First index i with v[i] == x, or -1. */
int64_t find_eq_i64(const int64_t* v, size_t n, int64_t x);

size_t first_non_ascii_large(const char* p, size_t n);
size_t first_json_special_large(const char* p, size_t n);

/* First index i with (unsigned char)p[i] >= 0x80, or n when the whole range is ASCII.
 * Short ranges -- a tokenizer piece, a token of streamed text -- are scanned inline: the
 * dispatched kernels pay an atomic load and an indirect call to look at a handful of bytes,
 * which at a call per piece cost more than the scan. The word-at-a-time check is portable
 * scalar code, exact at every level like the kernels behind it. */
inline size_t first_non_ascii(const char* p, size_t n) {
    if (n >= 32) return first_non_ascii_large(p, n);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        if (w & 0x8080808080808080ull) break;
    }
    for (; i < n; ++i)
        if ((unsigned char)p[i] >= 0x80) return i;
    return n;
}

/* First index i whose byte cannot go into a JSON string as it stands -- below 0x20, '"' or
 * '\\' (an escape), or >= 0x80 (UTF-8 to validate) -- or n. The streaming escaper's run scan
 * (server/json_partial.cpp), with the same short-range inline path as first_non_ascii. */
inline size_t first_json_special(const char* p, size_t n) {
    if (n >= 32) return first_json_special_large(p, n);
    for (size_t i = 0; i < n; ++i) {
        const unsigned char c = (unsigned char)p[i];
        if (c < 0x20 || c >= 0x80 || c == '"' || c == '\\') return i;
    }
    return n;
}

/* dst[i] = (src[i] >> sh) | (src[i+1] << (32 - sh)) for i in [0, n_words), reading src[n_words]
 * as 0. sh must be in [1, 31]. dst and src must not overlap. This is the bit-window realignment
 * a sharded vocabulary mask needs (sampler shard_mask). */
void bitstream_shr(uint32_t* dst, const uint32_t* src, size_t n_words, unsigned sh);

/* dst[i] = bit ids[i] of the bitmap `bits` (id < n_words*32), 1 or 0; an id outside
 * [0, n_words*32) reads as 0. The DRY breaker-mark loop. */
void bitmap_marks(uint8_t* dst, const int32_t* ids, size_t n, const uint32_t* bits,
                  size_t n_words);

/* dst[i] = a[i] | b[i] | c[i] over n bytes; b or c null stands for no term. dst may be a, b or c
 * (each byte is read before it is written). A voxel grid's dilation along one axis (the shape
 * generator's octree refinement). */
void or3_u8_impl(uint8_t* dst, const uint8_t* a, const uint8_t* b, const uint8_t* c, size_t n);
inline void or3_u8(uint8_t* dst, const uint8_t* a, const uint8_t* b, const uint8_t* c, size_t n) {
    or3_u8_impl(dst, a, b ? b : a, c ? c : a, n);
}

/* dst[i] = s[i - 1] | s[i] | s[i + 1] for i in [0, n), a neighbour past either end standing for
 * no term. dst and s must not overlap. The same dilation along a contiguous line. */
void or_adjacent_u8(uint8_t* dst, const uint8_t* s, size_t n);

/* acc[i] += (int64_t)h[i] * c for i in [0, n), c within int32. Exact (no product or sum rounds;
 * the caller keeps the sums within int64). A fixed-point separable filter's vertical taps. */
void madd_i32_i64(int64_t* acc, const int32_t* h, int32_t c, size_t n);

/* Base64 (RFC 4648 standard alphabet) over the input's leading whole blocks of the vector width:
 * encodes the leading bytes of in[0..n) to out, returning the bytes consumed (a multiple of 3;
 * out holds 4/3 as many digits). Reads stay inside in[0..n). Level 0 consumes nothing: the
 * caller's scalar loop does the rest, so every level gives the caller's exact output. */
size_t b64_encode_run(const uint8_t* in, size_t n, char* out);
/* Decodes leading digits of in[0..n) while every one of a vector block is a digit (no padding,
 * whitespace or other byte), returning the digits consumed (a multiple of 4; out holds 3/4 as
 * many bytes). Stops at the first block holding anything else, for the caller's scalar loop. */
size_t b64_decode_run(const uint8_t* in, size_t n, uint8_t* out);

/* Accumulates `stripes` 64-byte stripes of p into eight 64-bit lanes, as XXH3 does: lane j of
 * stripe s takes d = the stripe's j-th word, k = d ^ key[s % 16 + j], adds lo32(k) * hi32(k) to
 * itself and d to lane j ^ 1; after every 16 stripes each lane is scrambled (xorshift 47, a key
 * word, times 0x9E3779B1). The keys slide a word a stripe so reordered stripes accumulate
 * differently. Integer arithmetic only, the same at every level. The bulk of a content hash;
 * the caller seeds the lanes and folds them. */
void stripe_acc_u64x8(uint64_t acc[8], const uint8_t* p, size_t stripes);

/* Bitmaps of v[0..n) against a level, bit i of word i / 64 for element i, bits past n zero
 * ((n + 63) / 64 words each): gt v[i] > lv, lt v[i] < lv, nan v[i] is a NaN, near
 * |v[i] - lv| < band (the difference rounded to float first). A NaN sets only its nan bit. A
 * float comparison has one answer, so the levels agree bit for bit. One row of a scalar field
 * classified at a time (the shape generator's marching cubes and octree refinement). */
void level_bits_f32(const float* v, size_t n, float lv, float band, uint64_t* gt, uint64_t* lt,
                    uint64_t* nan, uint64_t* near);

}  /* namespace simd */
}  /* namespace rad */
