/* simd_kernels.cpp -- the per-level bodies behind simd.h. Compiled once per level with
 * RAD_SIMD_LEVEL=0..4 and matching -m flags (core/CMakeLists.txt); the globbed copy built
 * without the definition is empty. Names carry the level suffix; simd.cpp holds the table.
 *
 * Correctness rule for every body: the result is exact integer arithmetic, identical at every
 * level. Where a vector trick would change a corner (signed compare of bit patterns, a masked
 * tail reading past the end), the scalar tail below is the level-0 code repeated, so the test
 * that forces each level compares like with like.
 */
#ifdef RAD_SIMD_LEVEL

#include "util/simd.h"

#include <algorithm>
#include <cstring>

#if RAD_SIMD_LEVEL >= 1
#include <immintrin.h>
#endif

#if   RAD_SIMD_LEVEL == 0
#define RAD_SIMD_SUFFIX _sc
#elif RAD_SIMD_LEVEL == 1
#define RAD_SIMD_SUFFIX _s4
#elif RAD_SIMD_LEVEL == 2
#define RAD_SIMD_SUFFIX _v1
#elif RAD_SIMD_LEVEL == 3
#define RAD_SIMD_SUFFIX _v2
#elif RAD_SIMD_LEVEL == 4
#define RAD_SIMD_SUFFIX _v3
#else
#error "RAD_SIMD_LEVEL out of range"
#endif

#define RAD_SIMD_CAT2(a, b) a##b
#define RAD_SIMD_CAT(a, b)  RAD_SIMD_CAT2(a, b)
#define RAD_SIMD_FN(name)   RAD_SIMD_CAT(rad_simd_##name, RAD_SIMD_SUFFIX)

extern "C" {

/* ---------------------------------------------------------------- max_i32 */
int32_t RAD_SIMD_FN(max_i32)(const int32_t* v, size_t n) {
    int32_t m = INT32_MIN;
    size_t i = 0;
#if RAD_SIMD_LEVEL == 4
    __m512i a = _mm512_set1_epi32(INT32_MIN);
    for (; i + 16 <= n; i += 16) a = _mm512_max_epi32(a, _mm512_loadu_si512(v + i));
    m = _mm512_reduce_max_epi32(a);
#elif RAD_SIMD_LEVEL == 3
    __m256i a = _mm256_set1_epi32(INT32_MIN);
    for (; i + 8 <= n; i += 8) a = _mm256_max_epi32(a, _mm256_loadu_si256((const __m256i*)(v + i)));
    __m128i x = _mm_max_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    x = _mm_max_epi32(x, _mm_shuffle_epi32(x, 0x4e));
    x = _mm_max_epi32(x, _mm_shuffle_epi32(x, 0xb1));
    m = _mm_cvtsi128_si32(x);
#elif RAD_SIMD_LEVEL == 2
    /* AVX without AVX2 has no 256-bit integer max: two 128-bit halves (pmaxsd is SSE4.1, part of
     * every AVX part's SSE set, and this object is compiled -mavx only, so the 128-bit integer
     * instructions it uses are the SSE2..SSE4.1 ones __builtin_cpu_supports("avx") implies:
     * AVX requires SSE4.2 in every shipping part, and the compiler is told only -mavx, which
     * enables up to SSE4.2). */
    __m128i a = _mm_set1_epi32(INT32_MIN), b = _mm_set1_epi32(INT32_MIN);
    for (; i + 8 <= n; i += 8) {
        a = _mm_max_epi32(a, _mm_loadu_si128((const __m128i*)(v + i)));
        b = _mm_max_epi32(b, _mm_loadu_si128((const __m128i*)(v + i + 4)));
    }
    __m128i x = _mm_max_epi32(a, b);
    x = _mm_max_epi32(x, _mm_shuffle_epi32(x, 0x4e));
    x = _mm_max_epi32(x, _mm_shuffle_epi32(x, 0xb1));
    m = _mm_cvtsi128_si32(x);
#elif RAD_SIMD_LEVEL == 1
    __m128i a = _mm_set1_epi32(INT32_MIN);
    for (; i + 4 <= n; i += 4) a = _mm_max_epi32(a, _mm_loadu_si128((const __m128i*)(v + i)));
    a = _mm_max_epi32(a, _mm_shuffle_epi32(a, 0x4e));
    a = _mm_max_epi32(a, _mm_shuffle_epi32(a, 0xb1));
    m = _mm_cvtsi128_si32(a);
#endif
    for (; i < n; ++i) m = v[i] > m ? v[i] : m;
    return m;
}

/* ---------------------------------------------------------------- count_eq_i32 */
size_t RAD_SIMD_FN(count_eq_i32)(const int32_t* v, size_t n, int32_t x) {
    size_t c = 0, i = 0;
#if RAD_SIMD_LEVEL == 4
    const __m512i xv = _mm512_set1_epi32(x);
    for (; i + 16 <= n; i += 16)
        c += (size_t)__builtin_popcount(
            (unsigned)_mm512_cmpeq_epi32_mask(_mm512_loadu_si512(v + i), xv));
#elif RAD_SIMD_LEVEL == 3
    const __m256i xv = _mm256_set1_epi32(x);
    for (; i + 8 <= n; i += 8)
        c += (size_t)__builtin_popcount((unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(
            _mm256_cmpeq_epi32(_mm256_loadu_si256((const __m256i*)(v + i)), xv))));
#elif RAD_SIMD_LEVEL == 2
    const __m128i xv = _mm_set1_epi32(x);
    for (; i + 8 <= n; i += 8) {
        c += (size_t)__builtin_popcount((unsigned)_mm_movemask_ps(_mm_castsi128_ps(
            _mm_cmpeq_epi32(_mm_loadu_si128((const __m128i*)(v + i)), xv))));
        c += (size_t)__builtin_popcount((unsigned)_mm_movemask_ps(_mm_castsi128_ps(
            _mm_cmpeq_epi32(_mm_loadu_si128((const __m128i*)(v + i + 4)), xv))));
    }
#elif RAD_SIMD_LEVEL == 1
    const __m128i xv = _mm_set1_epi32(x);
    for (; i + 4 <= n; i += 4)
        c += (size_t)__builtin_popcount((unsigned)_mm_movemask_ps(_mm_castsi128_ps(
            _mm_cmpeq_epi32(_mm_loadu_si128((const __m128i*)(v + i)), xv))));
#endif
    for (; i < n; ++i) c += v[i] == x;
    return c;
}

/* ---------------------------------------------------------------- find_eq_i64 */
int64_t RAD_SIMD_FN(find_eq_i64)(const int64_t* v, size_t n, int64_t x) {
    size_t i = 0;
#if RAD_SIMD_LEVEL == 4
    const __m512i xv = _mm512_set1_epi64(x);
    for (; i + 8 <= n; i += 8) {
        const unsigned mk = (unsigned)_mm512_cmpeq_epi64_mask(_mm512_loadu_si512(v + i), xv);
        if (mk) return (int64_t)(i + (size_t)__builtin_ctz(mk));
    }
#elif RAD_SIMD_LEVEL == 3
    const __m256i xv = _mm256_set1_epi64x(x);
    for (; i + 4 <= n; i += 4) {
        const unsigned mk = (unsigned)_mm256_movemask_pd(_mm256_castsi256_pd(
            _mm256_cmpeq_epi64(_mm256_loadu_si256((const __m256i*)(v + i)), xv)));
        if (mk) return (int64_t)(i + (size_t)__builtin_ctz(mk));
    }
#elif RAD_SIMD_LEVEL == 2
    const __m128i xv = _mm_set1_epi64x(x);
    for (; i + 4 <= n; i += 4) {
        unsigned mk = (unsigned)_mm_movemask_pd(_mm_castsi128_pd(
            _mm_cmpeq_epi64(_mm_loadu_si128((const __m128i*)(v + i)), xv)));
        if (mk) return (int64_t)(i + (size_t)__builtin_ctz(mk));
        mk = (unsigned)_mm_movemask_pd(_mm_castsi128_pd(
            _mm_cmpeq_epi64(_mm_loadu_si128((const __m128i*)(v + i + 2)), xv)));
        if (mk) return (int64_t)(i + 2 + (size_t)__builtin_ctz(mk));
    }
#elif RAD_SIMD_LEVEL == 1
    const __m128i xv = _mm_set1_epi64x(x);
    for (; i + 2 <= n; i += 2) {
        const unsigned mk = (unsigned)_mm_movemask_pd(_mm_castsi128_pd(
            _mm_cmpeq_epi64(_mm_loadu_si128((const __m128i*)(v + i)), xv)));
        if (mk) return (int64_t)(i + (size_t)__builtin_ctz(mk));
    }
#endif
    for (; i < n; ++i)
        if (v[i] == x) return (int64_t)i;
    return -1;
}

/* ---------------------------------------------------------------- first_non_ascii */
size_t RAD_SIMD_FN(first_non_ascii)(const char* p, size_t n) {
    size_t i = 0;
#if RAD_SIMD_LEVEL == 4
    for (; i + 64 <= n; i += 64) {
        const __m512i v = _mm512_loadu_si512(p + i);
        /* Bytes >= 0x80 are negative as epi8 lanes: test against zero with the signed compare. */
        const uint64_t mk = (uint64_t)_mm512_cmplt_epi8_mask(v, _mm512_setzero_si512());
        if (mk) return i + (size_t)__builtin_ctzll(mk);
    }
    if (i < n) {
        const __mmask64 tail = (__mmask64)((1ull << (n - i)) - 1ull);
        const __m512i v = _mm512_maskz_loadu_epi8(tail, p + i);
        const uint64_t mk = (uint64_t)_mm512_mask_cmplt_epi8_mask(tail, v, _mm512_setzero_si512());
        if (mk) return i + (size_t)__builtin_ctzll(mk);
        return n;
    }
    return n;
#elif RAD_SIMD_LEVEL == 3
    const __m256i hi = _mm256_set1_epi8((char)0x80);
    for (; i + 32 <= n; i += 32) {
        const __m256i v = _mm256_loadu_si256((const __m256i*)(p + i));
        const unsigned mk = (unsigned)_mm256_movemask_epi8(_mm256_and_si256(v, hi));
        if (mk) return i + (size_t)__builtin_ctz(mk);
    }
#elif RAD_SIMD_LEVEL == 2
    /* No 256-bit integer ops at this level: 128-bit halves, as in libavx. */
    const __m128i hi = _mm_set1_epi8((char)0x80);
    for (; i + 16 <= n; i += 16) {
        const unsigned mk = (unsigned)_mm_movemask_epi8(
            _mm_and_si128(_mm_loadu_si128((const __m128i*)(p + i)), hi));
        if (mk) return i + (size_t)__builtin_ctz(mk);
    }
#elif RAD_SIMD_LEVEL == 1
    const __m128i hi = _mm_set1_epi8((char)0x80);
    for (; i + 16 <= n; i += 16) {
        const unsigned mk = (unsigned)_mm_movemask_epi8(
            _mm_and_si128(_mm_loadu_si128((const __m128i*)(p + i)), hi));
        if (mk) return i + (size_t)__builtin_ctz(mk);
    }
#endif
    for (; i < n; ++i)
        if ((unsigned char)p[i] >= 0x80) return i;
    return n;
}

/* ---------------------------------------------------------------- first_json_special */
size_t RAD_SIMD_FN(first_json_special)(const char* p, size_t n) {
    size_t i = 0;
    /* Signed byte compare against 0x20 catches the controls and, being signed, every byte
     * >= 0x80 in the same mask; '"' and '\\' are equality compares OR'ed in. */
#if RAD_SIMD_LEVEL == 4
    const __m512i sp = _mm512_set1_epi8(0x20), qu = _mm512_set1_epi8('"'),
                  bs = _mm512_set1_epi8('\\');
    for (; i + 64 <= n; i += 64) {
        const __m512i v = _mm512_loadu_si512(p + i);
        const uint64_t mk = (uint64_t)(_mm512_cmplt_epi8_mask(v, sp) |
                                       _mm512_cmpeq_epi8_mask(v, qu) |
                                       _mm512_cmpeq_epi8_mask(v, bs));
        if (mk) return i + (size_t)__builtin_ctzll(mk);
    }
    if (i < n) {
        const __mmask64 tail = (__mmask64)((1ull << (n - i)) - 1ull);
        const __m512i v = _mm512_maskz_loadu_epi8(tail, p + i);
        const uint64_t mk = (uint64_t)(_mm512_mask_cmplt_epi8_mask(tail, v, sp) |
                                       _mm512_mask_cmpeq_epi8_mask(tail, v, qu) |
                                       _mm512_mask_cmpeq_epi8_mask(tail, v, bs));
        if (mk) return i + (size_t)__builtin_ctzll(mk);
        return n;
    }
    return n;
#elif RAD_SIMD_LEVEL == 3
    const __m256i sp = _mm256_set1_epi8(0x20), qu = _mm256_set1_epi8('"'),
                  bs = _mm256_set1_epi8('\\');
    for (; i + 32 <= n; i += 32) {
        const __m256i v = _mm256_loadu_si256((const __m256i*)(p + i));
        const __m256i m = _mm256_or_si256(_mm256_cmpgt_epi8(sp, v),
                          _mm256_or_si256(_mm256_cmpeq_epi8(v, qu),
                                          _mm256_cmpeq_epi8(v, bs)));
        const unsigned mk = (unsigned)_mm256_movemask_epi8(m);
        if (mk) return i + (size_t)__builtin_ctz(mk);
    }
#elif RAD_SIMD_LEVEL == 2 || RAD_SIMD_LEVEL == 1
    const __m128i sp = _mm_set1_epi8(0x20), qu = _mm_set1_epi8('"'), bs = _mm_set1_epi8('\\');
    for (; i + 16 <= n; i += 16) {
        const __m128i v = _mm_loadu_si128((const __m128i*)(p + i));
        const __m128i m = _mm_or_si128(_mm_cmplt_epi8(v, sp),
                          _mm_or_si128(_mm_cmpeq_epi8(v, qu), _mm_cmpeq_epi8(v, bs)));
        const unsigned mk = (unsigned)_mm_movemask_epi8(m);
        if (mk) return i + (size_t)__builtin_ctz(mk);
    }
#endif
    for (; i < n; ++i) {
        const unsigned char c = (unsigned char)p[i];
        if (c < 0x20 || c == '"' || c == '\\' || c >= 0x80) return i;
    }
    return n;
}

/* ---------------------------------------------------------------- bitstream_shr */
void RAD_SIMD_FN(bitstream_shr)(uint32_t* dst, const uint32_t* src, size_t n_words, unsigned sh) {
    if (n_words == 0) return;
    size_t i = 0;
    const unsigned lsh = 32 - sh;
#if RAD_SIMD_LEVEL == 4
    /* Variable per-lane shift with a broadcast count; the (i+1) load goes one word past, which
     * the masked final store below keeps inside the caller's buffer: the main loop stops with
     * at least 16 words left, so src[i+16] is in bounds whenever it loads. */
    const __m512i cs = _mm512_set1_epi32((int)sh), cl = _mm512_set1_epi32((int)lsh);
    for (; i + 17 <= n_words; i += 16) {
        const __m512i a = _mm512_srlv_epi32(_mm512_loadu_si512(src + i), cs);
        const __m512i b = _mm512_sllv_epi32(_mm512_loadu_si512(src + i + 1), cl);
        _mm512_storeu_si512(dst + i, _mm512_or_si512(a, b));
    }
#elif RAD_SIMD_LEVEL == 3
    const __m256i cs = _mm256_set1_epi32((int)sh), cl = _mm256_set1_epi32((int)lsh);
    for (; i + 9 <= n_words; i += 8) {
        const __m256i a = _mm256_srlv_epi32(_mm256_loadu_si256((const __m256i*)(src + i)), cs);
        const __m256i b = _mm256_sllv_epi32(_mm256_loadu_si256((const __m256i*)(src + i + 1)), cl);
        _mm256_storeu_si256((__m256i*)(dst + i), _mm256_or_si256(a, b));
    }
#elif RAD_SIMD_LEVEL == 2 || RAD_SIMD_LEVEL == 1
    /* Shifts by a runtime count use the xmm count form; four words a step. The loop stops with
     * at least four words left, so src[i+4] is in bounds. */
    const __m128i cs = _mm_set_epi64x(0, (int64_t)sh), cl = _mm_set_epi64x(0, (int64_t)lsh);
    for (; i + 5 <= n_words; i += 4) {
        const __m128i a = _mm_srl_epi32(_mm_loadu_si128((const __m128i*)(src + i)), cs);
        const __m128i b = _mm_sll_epi32(_mm_loadu_si128((const __m128i*)(src + i + 1)), cl);
        _mm_storeu_si128((__m128i*)(dst + i), _mm_or_si128(a, b));
    }
#endif
    for (; i < n_words; ++i) {
        const uint32_t hi = i + 1 < n_words ? src[i + 1] : 0u;
        dst[i] = (src[i] >> sh) | (hi << lsh);
    }
}

/* ---------------------------------------------------------------- bitmap_marks */
void RAD_SIMD_FN(bitmap_marks)(uint8_t* dst, const int32_t* ids, size_t n, const uint32_t* bits,
                               size_t n_words) {
    /* A gather of the containing word, a variable shift and a mask: exact, with out-of-range ids
     * clamped to a zero word by index arithmetic rather than a branch per element. The clamp is
     * done in the scalar tail's arithmetic and, on the vector levels, by giving the gather a
     * zeroed guard word past the bitmap: indices land at n_words (the caller guarantees bits has
     * one readable word there? No -- it does not, so the vector levels clamp the index into
     * [0, n_words) and clear the result where the id was out of range). */
    if (n_words == 0) {
        for (size_t i = 0; i < n; ++i) dst[i] = 0;
        return;
    }
    size_t i = 0;
    /* A bitmap wider than int32 ids can name: the vector levels' range arithmetic is in int32,
     * so take the scalar loop (a >256 MiB bitmap; never a vocabulary mask). */
    [[maybe_unused]] const bool huge = (uint64_t)n_words * 32u > (uint64_t)INT32_MAX;
    [[maybe_unused]] const int32_t lim = huge ? 0 : (int32_t)n_words;
#if RAD_SIMD_LEVEL == 4
    const __m512i top = _mm512_set1_epi32(lim - 1);
    for (; !huge && i + 16 <= n; i += 16) {
        const __m512i id = _mm512_loadu_si512(ids + i);
        const __m512i w = _mm512_min_epi32(_mm512_max_epi32(_mm512_srai_epi32(id, 5),
                                                            _mm512_setzero_si512()), top);
        const __m512i g = _mm512_i32gather_epi32(w, bits, 4);
        const __m512i s = _mm512_and_si512(_mm512_srlv_epi32(g, _mm512_and_si512(id, _mm512_set1_epi32(31))), _mm512_set1_epi32(1));
        const __mmask16 in = _kand_mask16(_mm512_cmpge_epi32_mask(id, _mm512_setzero_si512()),
                                          _mm512_cmplt_epi32_mask(id, _mm512_set1_epi32(lim * 32)));
        _mm_storeu_si128((__m128i*)(dst + i),
                         _mm512_cvtepi32_epi8(_mm512_maskz_mov_epi32(in, s)));
    }
#elif RAD_SIMD_LEVEL == 3
    const __m256i top = _mm256_set1_epi32(lim - 1);
    const __m256i nbits = _mm256_set1_epi32(lim * 32);
    for (; !huge && i + 8 <= n; i += 8) {
        const __m256i id = _mm256_loadu_si256((const __m256i*)(ids + i));
        const __m256i w = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(id, 5),
                                                            _mm256_setzero_si256()), top);
        const __m256i g = _mm256_i32gather_epi32((const int*)bits, w, 4);
        __m256i s = _mm256_and_si256(_mm256_srlv_epi32(g, _mm256_and_si256(id, _mm256_set1_epi32(31))), _mm256_set1_epi32(1));
        const __m256i in = _mm256_and_si256(
            _mm256_cmpgt_epi32(id, _mm256_set1_epi32(-1)),
            _mm256_cmpgt_epi32(nbits, id));
        s = _mm256_and_si256(s, in);
        __m128i pk = _mm_packus_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        pk = _mm_packus_epi16(pk, pk);
        _mm_storel_epi64((__m128i*)(dst + i), pk);
    }
#endif
    for (; i < n; ++i) {
        const int32_t id = ids[i];
        dst[i] = id >= 0 && (size_t)(id >> 5) < n_words
                     ? (uint8_t)((bits[(size_t)(id >> 5)] >> (uint32_t)(id & 31)) & 1u)
                     : (uint8_t)0;
    }
}

/* ---------------------------------------------------------------- or3_u8 */
void RAD_SIMD_FN(or3_u8)(uint8_t* dst, const uint8_t* a, const uint8_t* b, const uint8_t* c,
                         size_t n) {
    size_t i = 0;
#if RAD_SIMD_LEVEL == 4
    for (; i + 64 <= n; i += 64) {
        const __m512i v = _mm512_or_si512(_mm512_or_si512(_mm512_loadu_si512(a + i),
                                                          _mm512_loadu_si512(b + i)),
                                          _mm512_loadu_si512(c + i));
        _mm512_storeu_si512(dst + i, v);
    }
#elif RAD_SIMD_LEVEL == 3
    for (; i + 32 <= n; i += 32) {
        const __m256i v = _mm256_or_si256(
            _mm256_or_si256(_mm256_loadu_si256((const __m256i*)(a + i)),
                            _mm256_loadu_si256((const __m256i*)(b + i))),
            _mm256_loadu_si256((const __m256i*)(c + i)));
        _mm256_storeu_si256((__m256i*)(dst + i), v);
    }
#elif RAD_SIMD_LEVEL == 2
    /* AVX has 256-bit bitwise ops only on the float types; an OR of their bits is exact. */
    for (; i + 32 <= n; i += 32) {
        const __m256 v = _mm256_or_ps(_mm256_or_ps(_mm256_loadu_ps((const float*)(a + i)),
                                                   _mm256_loadu_ps((const float*)(b + i))),
                                      _mm256_loadu_ps((const float*)(c + i)));
        _mm256_storeu_ps((float*)(dst + i), v);
    }
#elif RAD_SIMD_LEVEL == 1
    for (; i + 16 <= n; i += 16) {
        const __m128i v = _mm_or_si128(_mm_or_si128(_mm_loadu_si128((const __m128i*)(a + i)),
                                                    _mm_loadu_si128((const __m128i*)(b + i))),
                                       _mm_loadu_si128((const __m128i*)(c + i)));
        _mm_storeu_si128((__m128i*)(dst + i), v);
    }
#endif
    for (; i < n; ++i) dst[i] = (uint8_t)(a[i] | b[i] | c[i]);
}

/* ---------------------------------------------------------------- or_adjacent_u8 */
void RAD_SIMD_FN(or_adjacent_u8)(uint8_t* dst, const uint8_t* s, size_t n) {
    if (n == 0) return;
    if (n == 1) { dst[0] = s[0]; return; }
    dst[0] = (uint8_t)(s[0] | s[1]);
    /* The interior: three overlapping unaligned loads a vector, at i - 1, i and i + 1. */
    size_t i = 1;
    const size_t end = n - 1;
#if RAD_SIMD_LEVEL == 4
    for (; i + 64 <= end; i += 64) {
        const __m512i v = _mm512_or_si512(_mm512_or_si512(_mm512_loadu_si512(s + i - 1),
                                                          _mm512_loadu_si512(s + i)),
                                          _mm512_loadu_si512(s + i + 1));
        _mm512_storeu_si512(dst + i, v);
    }
#elif RAD_SIMD_LEVEL == 3
    for (; i + 32 <= end; i += 32) {
        const __m256i v = _mm256_or_si256(
            _mm256_or_si256(_mm256_loadu_si256((const __m256i*)(s + i - 1)),
                            _mm256_loadu_si256((const __m256i*)(s + i))),
            _mm256_loadu_si256((const __m256i*)(s + i + 1)));
        _mm256_storeu_si256((__m256i*)(dst + i), v);
    }
#elif RAD_SIMD_LEVEL == 2
    for (; i + 32 <= end; i += 32) {
        const __m256 v = _mm256_or_ps(_mm256_or_ps(_mm256_loadu_ps((const float*)(s + i - 1)),
                                                   _mm256_loadu_ps((const float*)(s + i))),
                                      _mm256_loadu_ps((const float*)(s + i + 1)));
        _mm256_storeu_ps((float*)(dst + i), v);
    }
#elif RAD_SIMD_LEVEL == 1
    for (; i + 16 <= end; i += 16) {
        const __m128i v = _mm_or_si128(_mm_or_si128(_mm_loadu_si128((const __m128i*)(s + i - 1)),
                                                    _mm_loadu_si128((const __m128i*)(s + i))),
                                       _mm_loadu_si128((const __m128i*)(s + i + 1)));
        _mm_storeu_si128((__m128i*)(dst + i), v);
    }
#endif
    for (; i < end; ++i) dst[i] = (uint8_t)(s[i - 1] | s[i] | s[i + 1]);
    dst[n - 1] = (uint8_t)(s[n - 2] | s[n - 1]);
}

/* ---------------------------------------------------------------- madd_i32_i64 */
void RAD_SIMD_FN(madd_i32_i64)(int64_t* acc, const int32_t* h, int32_t c, size_t n) {
    size_t i = 0;
    /* Each int32 widened to a 64-bit lane, then pmuldq: the signed product of the lanes' low
     * halves, exact in 64 bits. */
#if RAD_SIMD_LEVEL == 4
    const __m512i k = _mm512_set1_epi64(c);
    for (; i + 8 <= n; i += 8) {
        const __m512i x = _mm512_cvtepi32_epi64(_mm256_loadu_si256((const __m256i*)(h + i)));
        _mm512_storeu_si512(acc + i, _mm512_add_epi64(_mm512_loadu_si512(acc + i),
                                                      _mm512_mul_epi32(x, k)));
    }
#elif RAD_SIMD_LEVEL == 3
    const __m256i k = _mm256_set1_epi64x(c);
    for (; i + 4 <= n; i += 4) {
        const __m256i x = _mm256_cvtepi32_epi64(_mm_loadu_si128((const __m128i*)(h + i)));
        _mm256_storeu_si256((__m256i*)(acc + i),
                            _mm256_add_epi64(_mm256_loadu_si256((const __m256i*)(acc + i)),
                                             _mm256_mul_epi32(x, k)));
    }
#elif RAD_SIMD_LEVEL == 2 || RAD_SIMD_LEVEL == 1
    /* No 256-bit integer below AVX2: SSE4.1's pmovsxdq and pmuldq, two lanes a step. */
    const __m128i k = _mm_set1_epi64x(c);
    for (; i + 4 <= n; i += 4) {
        const __m128i x = _mm_loadu_si128((const __m128i*)(h + i));
        const __m128i lo = _mm_cvtepi32_epi64(x), hi = _mm_cvtepi32_epi64(_mm_srli_si128(x, 8));
        _mm_storeu_si128((__m128i*)(acc + i),
                         _mm_add_epi64(_mm_loadu_si128((const __m128i*)(acc + i)), _mm_mul_epi32(lo, k)));
        _mm_storeu_si128((__m128i*)(acc + i + 2),
                         _mm_add_epi64(_mm_loadu_si128((const __m128i*)(acc + i + 2)), _mm_mul_epi32(hi, k)));
    }
#endif
    for (; i < n; ++i) acc[i] += (int64_t)h[i] * c;
}

/* ---------------------------------------------------------------- base64
 *
 * Encode: each 3-byte group spread over one 32-bit lane as [b1 b0 b2 b1] (pshufb), its four
 * 6-bit fields moved to the low bits of four bytes with a 16-bit multiply-high and multiply-low,
 * then each field's digit as field + offset, the offset from four compares: +65 (A-Z), +6 more
 * past 25 (a-z), -75 past 51 (0-9), -15 past 61 ('+'), +3 past 62 ('/').
 *
 * Decode: a block is taken only if every byte is in A-Z, a-z, 0-9, '+' or '/', checked with
 * signed range compares (a byte >= 0x80 is negative and in no range); each digit's value is the
 * byte plus its range's offset; pmaddubsw and pmaddwd fold four 6-bit values into 24 bits a
 * lane, and pshufb packs the lanes' three bytes. */
#if RAD_SIMD_LEVEL >= 1
namespace {

inline __m128i b64_fields_128(__m128i in) {
    const __m128i spread = _mm_shuffle_epi8(in, _mm_setr_epi8(1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7,
                                                              10, 9, 11, 10));
    const __m128i t0 = _mm_and_si128(spread, _mm_set1_epi32(0x0fc0fc00));
    const __m128i t1 = _mm_mulhi_epu16(t0, _mm_set1_epi32(0x04000040));
    const __m128i t2 = _mm_and_si128(spread, _mm_set1_epi32(0x003f03f0));
    const __m128i t3 = _mm_mullo_epi16(t2, _mm_set1_epi32(0x01000010));
    return _mm_or_si128(t1, t3);
}
inline __m128i b64_digits_128(__m128i f) {
    __m128i off = _mm_set1_epi8(65);
    off = _mm_add_epi8(off, _mm_and_si128(_mm_cmpgt_epi8(f, _mm_set1_epi8(25)), _mm_set1_epi8(6)));
    off = _mm_add_epi8(off, _mm_and_si128(_mm_cmpgt_epi8(f, _mm_set1_epi8(51)), _mm_set1_epi8(-75)));
    off = _mm_add_epi8(off, _mm_and_si128(_mm_cmpgt_epi8(f, _mm_set1_epi8(61)), _mm_set1_epi8(-15)));
    off = _mm_add_epi8(off, _mm_and_si128(_mm_cmpgt_epi8(f, _mm_set1_epi8(62)), _mm_set1_epi8(3)));
    return _mm_add_epi8(f, off);
}
/* c in [lo, hi] as a signed byte compare; lo >= 1, so a byte >= 0x80 is never in it. */
inline __m128i in_range_128(__m128i c, char lo, char hi) {
    return _mm_and_si128(_mm_cmpgt_epi8(c, _mm_set1_epi8((char)(lo - 1))),
                         _mm_cmpgt_epi8(_mm_set1_epi8((char)(hi + 1)), c));
}
/* The block's 6-bit values, or false when any byte is not a digit. */
inline bool b64_values_128(__m128i c, __m128i* v) {
    const __m128i up = in_range_128(c, 'A', 'Z'), lo = in_range_128(c, 'a', 'z'),
                  dg = in_range_128(c, '0', '9'), pl = _mm_cmpeq_epi8(c, _mm_set1_epi8('+')),
                  sl = _mm_cmpeq_epi8(c, _mm_set1_epi8('/'));
    const __m128i ok = _mm_or_si128(_mm_or_si128(_mm_or_si128(up, lo), _mm_or_si128(dg, pl)), sl);
    if (_mm_movemask_epi8(ok) != 0xffff) return false;
    __m128i off = _mm_and_si128(up, _mm_set1_epi8(-65));
    off = _mm_or_si128(off, _mm_and_si128(lo, _mm_set1_epi8(-71)));
    off = _mm_or_si128(off, _mm_and_si128(dg, _mm_set1_epi8(4)));
    off = _mm_or_si128(off, _mm_and_si128(pl, _mm_set1_epi8(19)));
    off = _mm_or_si128(off, _mm_and_si128(sl, _mm_set1_epi8(16)));
    *v = _mm_add_epi8(c, off);
    return true;
}
/* Sixteen 6-bit values to twelve bytes in the lane's first twelve. */
inline __m128i b64_pack_128(__m128i v) {
    const __m128i ab = _mm_maddubs_epi16(v, _mm_set1_epi32(0x01400140));
    const __m128i abc = _mm_madd_epi16(ab, _mm_set1_epi32(0x00011000));
    return _mm_shuffle_epi8(abc, _mm_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1));
}

#if RAD_SIMD_LEVEL >= 3
inline __m256i in_range_256(__m256i c, char lo, char hi) {
    return _mm256_and_si256(_mm256_cmpgt_epi8(c, _mm256_set1_epi8((char)(lo - 1))),
                            _mm256_cmpgt_epi8(_mm256_set1_epi8((char)(hi + 1)), c));
}
#endif

}  /* namespace */
#endif

size_t RAD_SIMD_FN(b64_encode_run)(const uint8_t* in, size_t n, char* out) {
    size_t i = 0, o = 0;
#if RAD_SIMD_LEVEL == 4
    /* Four 12-byte groups a register, one a 128-bit lane; the last lane loads 16 bytes from 36. */
    for (; i + 52 <= n; i += 48, o += 64) {
        __m512i x = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)(in + i)));
        x = _mm512_inserti32x4(x, _mm_loadu_si128((const __m128i*)(in + i + 12)), 1);
        x = _mm512_inserti32x4(x, _mm_loadu_si128((const __m128i*)(in + i + 24)), 2);
        x = _mm512_inserti32x4(x, _mm_loadu_si128((const __m128i*)(in + i + 36)), 3);
        const __m512i sp = _mm512_shuffle_epi8(x, _mm512_broadcast_i32x4(_mm_setr_epi8(
            1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10)));
        const __m512i t1 = _mm512_mulhi_epu16(_mm512_and_si512(sp, _mm512_set1_epi32(0x0fc0fc00)),
                                              _mm512_set1_epi32(0x04000040));
        const __m512i t3 = _mm512_mullo_epi16(_mm512_and_si512(sp, _mm512_set1_epi32(0x003f03f0)),
                                              _mm512_set1_epi32(0x01000010));
        const __m512i f = _mm512_or_si512(t1, t3);
        __m512i off = _mm512_set1_epi8(65);
        off = _mm512_mask_add_epi8(off, _mm512_cmpgt_epi8_mask(f, _mm512_set1_epi8(25)), off, _mm512_set1_epi8(6));
        off = _mm512_mask_add_epi8(off, _mm512_cmpgt_epi8_mask(f, _mm512_set1_epi8(51)), off, _mm512_set1_epi8(-75));
        off = _mm512_mask_add_epi8(off, _mm512_cmpgt_epi8_mask(f, _mm512_set1_epi8(61)), off, _mm512_set1_epi8(-15));
        off = _mm512_mask_add_epi8(off, _mm512_cmpgt_epi8_mask(f, _mm512_set1_epi8(62)), off, _mm512_set1_epi8(3));
        _mm512_storeu_si512(out + o, _mm512_add_epi8(f, off));
    }
#elif RAD_SIMD_LEVEL == 3
    /* Two 12-byte groups a register, one a lane; the high lane loads 16 bytes from 12. */
    for (; i + 28 <= n; i += 24, o += 32) {
        __m256i x = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(in + i)));
        x = _mm256_inserti128_si256(x, _mm_loadu_si128((const __m128i*)(in + i + 12)), 1);
        const __m256i sp = _mm256_shuffle_epi8(x, _mm256_setr_epi8(
            1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
            1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10));
        const __m256i t1 = _mm256_mulhi_epu16(_mm256_and_si256(sp, _mm256_set1_epi32(0x0fc0fc00)),
                                              _mm256_set1_epi32(0x04000040));
        const __m256i t3 = _mm256_mullo_epi16(_mm256_and_si256(sp, _mm256_set1_epi32(0x003f03f0)),
                                              _mm256_set1_epi32(0x01000010));
        const __m256i f = _mm256_or_si256(t1, t3);
        __m256i off = _mm256_set1_epi8(65);
        off = _mm256_add_epi8(off, _mm256_and_si256(_mm256_cmpgt_epi8(f, _mm256_set1_epi8(25)), _mm256_set1_epi8(6)));
        off = _mm256_add_epi8(off, _mm256_and_si256(_mm256_cmpgt_epi8(f, _mm256_set1_epi8(51)), _mm256_set1_epi8(-75)));
        off = _mm256_add_epi8(off, _mm256_and_si256(_mm256_cmpgt_epi8(f, _mm256_set1_epi8(61)), _mm256_set1_epi8(-15)));
        off = _mm256_add_epi8(off, _mm256_and_si256(_mm256_cmpgt_epi8(f, _mm256_set1_epi8(62)), _mm256_set1_epi8(3)));
        _mm256_storeu_si256((__m256i*)(out + o), _mm256_add_epi8(f, off));
    }
#elif RAD_SIMD_LEVEL >= 1
    for (; i + 16 <= n; i += 12, o += 16)
        _mm_storeu_si128((__m128i*)(out + o),
                         b64_digits_128(b64_fields_128(_mm_loadu_si128((const __m128i*)(in + i)))));
#endif
    (void)in; (void)n; (void)out; (void)o;
    return i;
}

size_t RAD_SIMD_FN(b64_decode_run)(const uint8_t* in, size_t n, uint8_t* out) {
    size_t i = 0;
#if RAD_SIMD_LEVEL >= 1
    size_t o = 0;
    alignas(64) uint8_t tmp[64];
#if RAD_SIMD_LEVEL == 4
    for (; i + 64 <= n; i += 64, o += 48) {
        const __m512i c = _mm512_loadu_si512(in + i);
        auto rng = [&](char lo, char hi) {
            return _kand_mask64(_mm512_cmpgt_epi8_mask(c, _mm512_set1_epi8((char)(lo - 1))),
                                _mm512_cmpgt_epi8_mask(_mm512_set1_epi8((char)(hi + 1)), c));
        };
        const __mmask64 up = rng('A', 'Z'), lo = rng('a', 'z'), dg = rng('0', '9'),
                        pl = _mm512_cmpeq_epi8_mask(c, _mm512_set1_epi8('+')),
                        sl = _mm512_cmpeq_epi8_mask(c, _mm512_set1_epi8('/'));
        if ((up | lo | dg | pl | sl) != ~(__mmask64)0) break;
        __m512i off = _mm512_maskz_mov_epi8(up, _mm512_set1_epi8(-65));
        off = _mm512_mask_mov_epi8(off, lo, _mm512_set1_epi8(-71));
        off = _mm512_mask_mov_epi8(off, dg, _mm512_set1_epi8(4));
        off = _mm512_mask_mov_epi8(off, pl, _mm512_set1_epi8(19));
        off = _mm512_mask_mov_epi8(off, sl, _mm512_set1_epi8(16));
        const __m512i v = _mm512_add_epi8(c, off);
        const __m512i ab = _mm512_maddubs_epi16(v, _mm512_set1_epi32(0x01400140));
        const __m512i abc = _mm512_madd_epi16(ab, _mm512_set1_epi32(0x00011000));
        const __m512i pk = _mm512_shuffle_epi8(abc, _mm512_broadcast_i32x4(_mm_setr_epi8(
            2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1)));
        _mm512_store_si512(tmp, pk);
        for (int l = 0; l < 4; ++l) std::memcpy(out + o + 12 * l, tmp + 16 * l, 12);
    }
#elif RAD_SIMD_LEVEL == 3
    for (; i + 32 <= n; i += 32, o += 24) {
        const __m256i c = _mm256_loadu_si256((const __m256i*)(in + i));
        const __m256i up = in_range_256(c, 'A', 'Z'), lo = in_range_256(c, 'a', 'z'),
                      dg = in_range_256(c, '0', '9'),
                      pl = _mm256_cmpeq_epi8(c, _mm256_set1_epi8('+')),
                      sl = _mm256_cmpeq_epi8(c, _mm256_set1_epi8('/'));
        const __m256i ok = _mm256_or_si256(_mm256_or_si256(_mm256_or_si256(up, lo),
                                                           _mm256_or_si256(dg, pl)), sl);
        if ((uint32_t)_mm256_movemask_epi8(ok) != 0xffffffffu) break;
        __m256i off = _mm256_and_si256(up, _mm256_set1_epi8(-65));
        off = _mm256_or_si256(off, _mm256_and_si256(lo, _mm256_set1_epi8(-71)));
        off = _mm256_or_si256(off, _mm256_and_si256(dg, _mm256_set1_epi8(4)));
        off = _mm256_or_si256(off, _mm256_and_si256(pl, _mm256_set1_epi8(19)));
        off = _mm256_or_si256(off, _mm256_and_si256(sl, _mm256_set1_epi8(16)));
        const __m256i v = _mm256_add_epi8(c, off);
        const __m256i ab = _mm256_maddubs_epi16(v, _mm256_set1_epi32(0x01400140));
        const __m256i abc = _mm256_madd_epi16(ab, _mm256_set1_epi32(0x00011000));
        const __m256i pk = _mm256_shuffle_epi8(abc, _mm256_setr_epi8(
            2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1,
            2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1));
        _mm256_store_si256((__m256i*)tmp, pk);
        std::memcpy(out + o, tmp, 12);
        std::memcpy(out + o + 12, tmp + 16, 12);
    }
#else
    for (; i + 16 <= n; i += 16, o += 12) {
        __m128i v;
        if (!b64_values_128(_mm_loadu_si128((const __m128i*)(in + i)), &v)) break;
        _mm_store_si128((__m128i*)tmp, b64_pack_128(v));
        std::memcpy(out + o, tmp, 12);
    }
#endif
#endif
    (void)in; (void)n; (void)out;
    return i;
}

/* ---------------------------------------------------------------- stripe_acc_u64x8 */
namespace {
/* 24 key words from splitmix64 over a fixed seed: words 0..22 are the stripes' sliding keys (16
 * stripes, eight lanes), 16..23 the scramble's. */
constexpr uint64_t stripe_key_word(int i) {
    uint64_t z = 0x5241444943414e45ull + 0x9e3779b97f4a7c15ull * (uint64_t)(i + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
struct StripeKeys {
    uint64_t w[24];
    constexpr StripeKeys() : w() { for (int i = 0; i < 24; ++i) w[i] = stripe_key_word(i); }
};
constexpr StripeKeys kStripeKeys{};
constexpr uint64_t kStripePrime = 0x9E3779B1ull;
constexpr size_t kStripesBlock = 16;
}  /* namespace */

void RAD_SIMD_FN(stripe_acc_u64x8)(uint64_t acc[8], const uint8_t* p, size_t stripes) {
    const uint64_t* K = kStripeKeys.w;
#if RAD_SIMD_LEVEL == 4
    __m512i a = _mm512_loadu_si512(acc);
    const __m512i prime = _mm512_set1_epi64((long long)kStripePrime);
    for (size_t s = 0; s < stripes; ++s) {
        const size_t r = s % kStripesBlock;
        const __m512i d = _mm512_loadu_si512(p + 64 * s);
        const __m512i k = _mm512_xor_si512(d, _mm512_loadu_si512(K + r));
        a = _mm512_add_epi64(a, _mm512_mul_epu32(k, _mm512_srli_epi64(k, 32)));
        a = _mm512_add_epi64(a, _mm512_shuffle_epi32(d, (_MM_PERM_ENUM)0x4E));
        if (r == kStripesBlock - 1) {
            a = _mm512_xor_si512(_mm512_xor_si512(a, _mm512_srli_epi64(a, 47)), _mm512_loadu_si512(K + 16));
            a = _mm512_add_epi64(_mm512_mul_epu32(a, prime),
                                 _mm512_slli_epi64(_mm512_mul_epu32(_mm512_srli_epi64(a, 32), prime), 32));
        }
    }
    _mm512_storeu_si512(acc, a);
#elif RAD_SIMD_LEVEL == 3
    __m256i a0 = _mm256_loadu_si256((const __m256i*)acc), a1 = _mm256_loadu_si256((const __m256i*)(acc + 4));
    const __m256i prime = _mm256_set1_epi64x((long long)kStripePrime);
    auto step = [](__m256i a, __m256i d, __m256i key) {
        const __m256i k = _mm256_xor_si256(d, key);
        a = _mm256_add_epi64(a, _mm256_mul_epu32(k, _mm256_srli_epi64(k, 32)));
        return _mm256_add_epi64(a, _mm256_shuffle_epi32(d, 0x4E));
    };
    auto scramble = [&](__m256i a, __m256i key) {
        a = _mm256_xor_si256(_mm256_xor_si256(a, _mm256_srli_epi64(a, 47)), key);
        return _mm256_add_epi64(_mm256_mul_epu32(a, prime),
                                _mm256_slli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), prime), 32));
    };
    for (size_t s = 0; s < stripes; ++s) {
        const size_t r = s % kStripesBlock;
        a0 = step(a0, _mm256_loadu_si256((const __m256i*)(p + 64 * s)), _mm256_loadu_si256((const __m256i*)(K + r)));
        a1 = step(a1, _mm256_loadu_si256((const __m256i*)(p + 64 * s + 32)), _mm256_loadu_si256((const __m256i*)(K + r + 4)));
        if (r == kStripesBlock - 1) {
            a0 = scramble(a0, _mm256_loadu_si256((const __m256i*)(K + 16)));
            a1 = scramble(a1, _mm256_loadu_si256((const __m256i*)(K + 20)));
        }
    }
    _mm256_storeu_si256((__m256i*)acc, a0);
    _mm256_storeu_si256((__m256i*)(acc + 4), a1);
#elif RAD_SIMD_LEVEL >= 1
    /* Two lanes a register (no 256-bit integer below AVX2); pmuludq is SSE2. */
    __m128i a[4];
    for (int q = 0; q < 4; ++q) a[q] = _mm_loadu_si128((const __m128i*)(acc + 2 * q));
    const __m128i prime = _mm_set1_epi64x((long long)kStripePrime);
    for (size_t s = 0; s < stripes; ++s) {
        const size_t r = s % kStripesBlock;
        for (int q = 0; q < 4; ++q) {
            const __m128i d = _mm_loadu_si128((const __m128i*)(p + 64 * s + 16 * q));
            const __m128i k = _mm_xor_si128(d, _mm_loadu_si128((const __m128i*)(K + r + 2 * q)));
            a[q] = _mm_add_epi64(a[q], _mm_mul_epu32(k, _mm_srli_epi64(k, 32)));
            a[q] = _mm_add_epi64(a[q], _mm_shuffle_epi32(d, 0x4E));
        }
        if (r == kStripesBlock - 1)
            for (int q = 0; q < 4; ++q) {
                __m128i x = _mm_xor_si128(_mm_xor_si128(a[q], _mm_srli_epi64(a[q], 47)),
                                          _mm_loadu_si128((const __m128i*)(K + 16 + 2 * q)));
                a[q] = _mm_add_epi64(_mm_mul_epu32(x, prime),
                                     _mm_slli_epi64(_mm_mul_epu32(_mm_srli_epi64(x, 32), prime), 32));
            }
    }
    for (int q = 0; q < 4; ++q) _mm_storeu_si128((__m128i*)(acc + 2 * q), a[q]);
#else
    for (size_t s = 0; s < stripes; ++s) {
        const size_t r = s % kStripesBlock;
        uint64_t d[8];
        std::memcpy(d, p + 64 * s, 64);
        for (int j = 0; j < 8; ++j) {
            const uint64_t k = d[j] ^ K[r + (size_t)j];
            acc[j] += (k & 0xffffffffull) * (k >> 32);
            acc[j ^ 1] += d[j];
        }
        if (r == kStripesBlock - 1)
            for (int j = 0; j < 8; ++j) {
                const uint64_t x = acc[j] ^ (acc[j] >> 47) ^ K[16 + j];
                acc[j] = (x & 0xffffffffull) * kStripePrime + (((x >> 32) * kStripePrime) << 32);
            }
    }
#endif
}

/* ---------------------------------------------------------------- level_bits_f32 */
void RAD_SIMD_FN(level_bits_f32)(const float* v, size_t n, float lv, float band, uint64_t* gt,
                                 uint64_t* lt, uint64_t* nan, uint64_t* near) {
    size_t w = 0;
    /* Whole words: 64 elements, a lane group's comparison masks placed side by side. The
     * difference is a float subtraction, as the scalar tail's. */
#if RAD_SIMD_LEVEL == 4
    const __m512 L = _mm512_set1_ps(lv), B = _mm512_set1_ps(band);
    for (; (w + 1) * 64 <= n; ++w) {
        uint64_t g = 0, l = 0, q = 0, r = 0;
        for (int h = 0; h < 4; ++h) {
            const __m512 x = _mm512_loadu_ps(v + w * 64 + h * 16);
            const __m512 d = _mm512_abs_ps(_mm512_sub_ps(x, L));
            g |= (uint64_t)_mm512_cmp_ps_mask(x, L, _CMP_GT_OQ) << (16 * h);
            l |= (uint64_t)_mm512_cmp_ps_mask(x, L, _CMP_LT_OQ) << (16 * h);
            q |= (uint64_t)_mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q) << (16 * h);
            r |= (uint64_t)_mm512_cmp_ps_mask(d, B, _CMP_LT_OQ) << (16 * h);
        }
        gt[w] = g; lt[w] = l; nan[w] = q; near[w] = r;
    }
#elif RAD_SIMD_LEVEL == 2 || RAD_SIMD_LEVEL == 3
    const __m256 L = _mm256_set1_ps(lv), B = _mm256_set1_ps(band);
    const __m256 abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (; (w + 1) * 64 <= n; ++w) {
        uint64_t g = 0, l = 0, q = 0, r = 0;
        for (int h = 0; h < 8; ++h) {
            const __m256 x = _mm256_loadu_ps(v + w * 64 + h * 8);
            const __m256 d = _mm256_and_ps(_mm256_sub_ps(x, L), abs);
            g |= (uint64_t)(unsigned)_mm256_movemask_ps(_mm256_cmp_ps(x, L, _CMP_GT_OQ)) << (8 * h);
            l |= (uint64_t)(unsigned)_mm256_movemask_ps(_mm256_cmp_ps(x, L, _CMP_LT_OQ)) << (8 * h);
            q |= (uint64_t)(unsigned)_mm256_movemask_ps(_mm256_cmp_ps(x, x, _CMP_UNORD_Q)) << (8 * h);
            r |= (uint64_t)(unsigned)_mm256_movemask_ps(_mm256_cmp_ps(d, B, _CMP_LT_OQ)) << (8 * h);
        }
        gt[w] = g; lt[w] = l; nan[w] = q; near[w] = r;
    }
#elif RAD_SIMD_LEVEL == 1
    const __m128 L = _mm_set1_ps(lv), B = _mm_set1_ps(band);
    const __m128 abs = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    for (; (w + 1) * 64 <= n; ++w) {
        uint64_t g = 0, l = 0, q = 0, r = 0;
        for (int h = 0; h < 16; ++h) {
            const __m128 x = _mm_loadu_ps(v + w * 64 + h * 4);
            const __m128 d = _mm_and_ps(_mm_sub_ps(x, L), abs);
            g |= (uint64_t)(unsigned)_mm_movemask_ps(_mm_cmpgt_ps(x, L)) << (4 * h);
            l |= (uint64_t)(unsigned)_mm_movemask_ps(_mm_cmplt_ps(x, L)) << (4 * h);
            q |= (uint64_t)(unsigned)_mm_movemask_ps(_mm_cmpunord_ps(x, x)) << (4 * h);
            r |= (uint64_t)(unsigned)_mm_movemask_ps(_mm_cmplt_ps(d, B)) << (4 * h);
        }
        gt[w] = g; lt[w] = l; nan[w] = q; near[w] = r;
    }
#endif
    for (; w * 64 < n; ++w) {
        uint64_t g = 0, l = 0, q = 0, r = 0;
        const size_t end = std::min(n, (w + 1) * 64);
        for (size_t i = w * 64; i < end; ++i) {
            const float x = v[i];
            const uint64_t b = 1ull << (i - w * 64);
            if (x > lv) g |= b;
            if (x < lv) l |= b;
            if (x != x) q |= b;
            float d = x - lv;
            d = d < 0.0f ? -d : d;
            if (d < band) r |= b;
        }
        gt[w] = g; lt[w] = l; nan[w] = q; near[w] = r;
    }
}

}  /* extern "C" */

#endif  /* RAD_SIMD_LEVEL */
