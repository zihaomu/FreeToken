/* avx_vec.h -- the vector abstraction every kernel is written against, and the dtype conversions
 * that feed it.
 *
 * Compiled four times, once per ISA level, with AVX_LEVEL defined on the command line and matching
 * -m flags. The level is a compile-time constant here, so `vf` is one concrete type and each
 * operation is one instruction or a short fixed sequence. The runtime choice is made once, in
 * avx_dispatch.cpp, between tables of compiled functions.
 *
 * One source per kernel instead of four copies: the AVX-512 path is the same algorithm as the
 * scalar path, and the checker only has to prove the arithmetic agrees. The abstraction cannot
 * express what differs in kind between levels (AVX-512 write masks, 32 registers,
 * compress/expand), so where that matters a kernel has an `#if AVX_LEVEL >= 3` block with a
 * comment (e.g. the GEMM microkernel's register blocking, top-k's compress).
 *
 * Everything computes in f32, as libref does (f32 accumulation, IEEE RTNE narrowing;
 * libref/ref_common.h). A whole row is converted to f32 scratch once, computed, and converted
 * back, instead of libref's per-element dtype switch, which blocks vectorisation.
 *
 * Unlike libref, this plugin reassociates reductions and uses FMA, so avx_check.cpp uses the kbench
 * tolerances rather than zero. The narrowing conversions are bit-exact, since they define the
 * stored result.
 */
#ifndef RAD_AVX_VEC_H
#define RAD_AVX_VEC_H

#ifndef AVX_LEVEL
#error "AVX_LEVEL must be defined on the command line: 0 scalar, 1 avx, 2 avx2, 3 avx512"
#endif

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cstdint>
#include <cstring>
#include <cmath>

#if AVX_LEVEL >= 1
#include <immintrin.h>
#endif

/* The symbol suffix for this level. Every kernel defines avx_<op>_<suffix>. */
#if   AVX_LEVEL == 0
#define AVX_SUFFIX sc
#define AVX_LEVEL_NAME "scalar"
#elif AVX_LEVEL == 1
#define AVX_SUFFIX v1
#define AVX_LEVEL_NAME "avx"
#elif AVX_LEVEL == 2
#define AVX_SUFFIX v2
#define AVX_LEVEL_NAME "avx2"
#elif AVX_LEVEL == 3
#define AVX_SUFFIX v3
#define AVX_LEVEL_NAME "avx512"
#else
#error "AVX_LEVEL out of range"
#endif

#define AVX_CAT2(a, b) a##_##b
#define AVX_CAT(a, b)  AVX_CAT2(a, b)
/* avx_rmsnorm -> avx_rmsnorm_v2 at AVX_LEVEL 2. */
#define AVX_FN(name)   AVX_CAT(name, AVX_SUFFIX)

/* Signature of every kernel. A host kernel ignores `stream`, but every entry point is
 * (const RadArgs*, RadStream) in every domain (libr4d/r4d_plugin.h). */
#define AVX_KERNEL(nm) \
    extern "C" int AVX_FN(avx_##nm)(const RadArgs* a, [[maybe_unused]] RadStream stream)

namespace avx {
/* One inline namespace per level, so kernels write `avx::vf` without naming the level. Without it
 * the four objects would define the same `avx::row_to_f32` symbol and the linker would silently
 * pick one (e.g. the AVX-512 one) for all four tables. */
inline namespace AVX_SUFFIX {

/* ================================================================== the vector type */
#if AVX_LEVEL == 0
typedef float vf;
enum { VF_N = 1 };
static inline vf vf_zero()                    { return 0.0f; }
static inline vf vf_set1(float x)             { return x; }
static inline vf vf_load(const float* p)      { return *p; }
static inline vf vf_loadu(const float* p)     { return *p; }
static inline void vf_store(float* p, vf v)   { *p = v; }
static inline void vf_storeu(float* p, vf v)  { *p = v; }
static inline vf vf_add(vf a, vf b)           { return a + b; }
static inline vf vf_sub(vf a, vf b)           { return a - b; }
static inline vf vf_mul(vf a, vf b)           { return a * b; }
static inline vf vf_div(vf a, vf b)           { return a / b; }
static inline vf vf_max(vf a, vf b)           { return a > b ? a : b; }
static inline vf vf_min(vf a, vf b)           { return a < b ? a : b; }
static inline vf vf_fma(vf a, vf b, vf c)     { return a * b + c; }
static inline vf vf_sqrt(vf a)                { return std::sqrt(a); }
static inline vf vf_abs(vf a)                 { return std::fabs(a); }
static inline float vf_hsum(vf a)             { return a; }
static inline float vf_hmax(vf a)             { return a; }

#elif AVX_LEVEL == 1 || AVX_LEVEL == 2
typedef __m256 vf;
enum { VF_N = 8 };
static inline vf vf_zero()                    { return _mm256_setzero_ps(); }
static inline vf vf_set1(float x)             { return _mm256_set1_ps(x); }
static inline vf vf_load(const float* p)      { return _mm256_load_ps(p); }
static inline vf vf_loadu(const float* p)     { return _mm256_loadu_ps(p); }
static inline void vf_store(float* p, vf v)   { _mm256_store_ps(p, v); }
static inline void vf_storeu(float* p, vf v)  { _mm256_storeu_ps(p, v); }
static inline vf vf_add(vf a, vf b)           { return _mm256_add_ps(a, b); }
static inline vf vf_sub(vf a, vf b)           { return _mm256_sub_ps(a, b); }
static inline vf vf_mul(vf a, vf b)           { return _mm256_mul_ps(a, b); }
static inline vf vf_div(vf a, vf b)           { return _mm256_div_ps(a, b); }
static inline vf vf_max(vf a, vf b)           { return _mm256_max_ps(a, b); }
static inline vf vf_min(vf a, vf b)           { return _mm256_min_ps(a, b); }
#if AVX_LEVEL >= 2
static inline vf vf_fma(vf a, vf b, vf c)     { return _mm256_fmadd_ps(a, b, c); }
#else
/* AVX has no FMA (it arrived with Haswell). Two roundings, which is closer to libref than AVX2's
 * fused form. */
static inline vf vf_fma(vf a, vf b, vf c)     { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
#endif
static inline vf vf_sqrt(vf a)                { return _mm256_sqrt_ps(a); }
static inline vf vf_abs(vf a) {
    return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
}
static inline float vf_hsum(vf a) {
    /* Fixed order (cross-lane, then within 128 bits) so the result does not depend on the
     * compiler's choice of tree. */
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 hi = _mm256_extractf128_ps(a, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}
static inline float vf_hmax(vf a) {
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 hi = _mm256_extractf128_ps(a, 1);
    __m128 s  = _mm_max_ps(lo, hi);
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

#else  /* AVX_LEVEL == 3 */
typedef __m512 vf;
enum { VF_N = 16 };
static inline vf vf_zero()                    { return _mm512_setzero_ps(); }
static inline vf vf_set1(float x)             { return _mm512_set1_ps(x); }
static inline vf vf_load(const float* p)      { return _mm512_load_ps(p); }
static inline vf vf_loadu(const float* p)     { return _mm512_loadu_ps(p); }
static inline void vf_store(float* p, vf v)   { _mm512_store_ps(p, v); }
static inline void vf_storeu(float* p, vf v)  { _mm512_storeu_ps(p, v); }
static inline vf vf_add(vf a, vf b)           { return _mm512_add_ps(a, b); }
static inline vf vf_sub(vf a, vf b)           { return _mm512_sub_ps(a, b); }
static inline vf vf_mul(vf a, vf b)           { return _mm512_mul_ps(a, b); }
static inline vf vf_div(vf a, vf b)           { return _mm512_div_ps(a, b); }
static inline vf vf_max(vf a, vf b)           { return _mm512_max_ps(a, b); }
static inline vf vf_min(vf a, vf b)           { return _mm512_min_ps(a, b); }
static inline vf vf_fma(vf a, vf b, vf c)     { return _mm512_fmadd_ps(a, b, c); }
static inline vf vf_sqrt(vf a)                { return _mm512_sqrt_ps(a); }
static inline vf vf_abs(vf a)                 { return _mm512_abs_ps(a); }
static inline float vf_hsum(vf a)             { return _mm512_reduce_add_ps(a); }
static inline float vf_hmax(vf a)             { return _mm512_reduce_max_ps(a); }
#endif

/* ================================================================== narrowing conversions
 *
 * Bit-exact. rad_plugin.h defines f32->bf16 as RTNE with ties to the even surviving bit, NaN
 * forced quiet, and overflow carrying into the bf16 infinity. The vector forms below are the same
 * integer arithmetic lane for lane; avx_check.cpp checks `cast` over every dtype pair against
 * libref at 1e-6, i.e. exactly.
 *
 * Not `_mm512_cvtneps_pbh`: AVX512-BF16 (e.g. Zen 5) also rounds RTNE, but its NaN handling is the
 * hardware's, not the ABI's.
 */
static inline void cvt_bf16_to_f32(const uint16_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16) {
        __m256i h = _mm256_loadu_si256((const __m256i*)(src + i));
        __m512i w = _mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16);
        _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(w));
    }
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(src + i));
        __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(w));
    }
#elif AVX_LEVEL == 1
    /* AVX has no 256-bit integer ALU: shift in two 128-bit halves and join with vinsertf128, a
     * float-domain instruction. Every integer helper at this level has this shape. */
    for (; i + 8 <= n; i += 8) {
        __m128i h  = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i lo = _mm_slli_epi32(_mm_cvtepu16_epi32(h), 16);
        __m128i hi = _mm_slli_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(h, 8)), 16);
        __m256 v = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(lo)),
                                        _mm_castsi128_ps(hi), 1);
        _mm256_storeu_ps(dst + i, v);
    }
#endif
    for (; i < n; ++i) dst[i] = rad_bf16_to_f32(src[i]);
}

#if AVX_LEVEL >= 1
/* RTNE core on f32 bit patterns, leaving the 16-bit result in the low half of each lane. Same
 * arithmetic at all three vector levels; only the pack differs. */
#if AVX_LEVEL == 3
static inline __m512i bf16_round_bits(__m512 f) {
    __m512i u    = _mm512_castps_si512(f);
    __m512i mag  = _mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff));
    __mmask16 nan = _mm512_cmpgt_epi32_mask(mag, _mm512_set1_epi32(0x7f800000));
    __m512i lsb  = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
    __m512i rnd  = _mm512_srli_epi32(
                       _mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7fff)), lsb), 16);
    __m512i qnan = _mm512_or_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(0x0040));
    return _mm512_mask_blend_epi32(nan, rnd, qnan);
}
#elif AVX_LEVEL == 2
static inline __m256i bf16_round_bits(__m256 f) {
    __m256i u    = _mm256_castps_si256(f);
    __m256i mag  = _mm256_and_si256(u, _mm256_set1_epi32(0x7fffffff));
    __m256i nan  = _mm256_cmpgt_epi32(mag, _mm256_set1_epi32(0x7f800000));
    __m256i lsb  = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    __m256i rnd  = _mm256_srli_epi32(
                       _mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7fff)), lsb), 16);
    __m256i qnan = _mm256_or_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(0x0040));
    return _mm256_blendv_epi8(rnd, qnan, nan);
}
#else  /* AVX_LEVEL == 1 */
static inline __m128i bf16_round_bits128(__m128 f) {
    __m128i u    = _mm_castps_si128(f);
    __m128i mag  = _mm_and_si128(u, _mm_set1_epi32(0x7fffffff));
    __m128i nan  = _mm_cmpgt_epi32(mag, _mm_set1_epi32(0x7f800000));
    __m128i lsb  = _mm_and_si128(_mm_srli_epi32(u, 16), _mm_set1_epi32(1));
    __m128i rnd  = _mm_srli_epi32(_mm_add_epi32(_mm_add_epi32(u, _mm_set1_epi32(0x7fff)), lsb), 16);
    __m128i qnan = _mm_or_si128(_mm_srli_epi32(u, 16), _mm_set1_epi32(0x0040));
    return _mm_blendv_epi8(rnd, qnan, nan);
}
#endif
#endif

static inline void cvt_f32_to_bf16(const float* src, uint16_t* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16) {
        __m512i w = bf16_round_bits(_mm512_loadu_ps(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), _mm512_cvtepi32_epi16(w));
    }
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8) {
        __m256i w  = bf16_round_bits(_mm256_loadu_ps(src + i));
        __m128i lo = _mm256_castsi256_si128(w);
        __m128i hi = _mm256_extracti128_si256(w, 1);
        /* Every lane is <= 0xffff, so packus does not saturate. */
        _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi32(lo, hi));
    }
#elif AVX_LEVEL == 1
    for (; i + 8 <= n; i += 8) {
        __m256   v  = _mm256_loadu_ps(src + i);
        __m128i  lo = bf16_round_bits128(_mm256_castps256_ps128(v));
        __m128i  hi = bf16_round_bits128(_mm256_extractf128_ps(v, 1));
        _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi32(lo, hi));
    }
#endif
    for (; i < n; ++i) dst[i] = rad_f32_to_bf16(src[i]);
}

/* f16 is hardware (F16C) from the AVX2 level up and the ABI's software convert below. F16C is a
 * separate CPUID bit that arrived a generation after AVX, so the `avx` level cannot assume it.
 * F16C rounds RTNE, as rad_f32_to_f16 does, so the two agree bit for bit (`cast` at 1e-6). */
static inline void cvt_f16_to_f32(const uint16_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(src + i))));
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(src + i))));
#endif
    for (; i < n; ++i) dst[i] = rad_f16_to_f32(src[i]);
}

static inline void cvt_f32_to_f16(const float* src, uint16_t* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16)
        _mm256_storeu_si256((__m256i*)(dst + i),
                            _mm512_cvtps_ph(_mm512_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i*)(dst + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
#endif
    for (; i < n; ++i) dst[i] = rad_f32_to_f16(src[i]);
}

/* ---- fp8. Decode: a 256-entry table at the scalar level and for tails, built at load from the
 * ABI's converters (avx_dispatch.cpp: fp8_tables_init); e4m3 is a bit reconstruction on the vector
 * levels (see cvt_fp8e4m3_to_f32), e5m2 a table gather.
 *
 * Encode: the bit-exact algorithm has three branches (subnormal, normal, saturate). Only the normal
 * case, |x| in [2^-6, 448), which covers a normalised activation, is vectorised; a mask of the
 * other elements drives a scalar fixup through the ABI's converter. */
}  /* The tables have C linkage at global scope: defined once in the baseline avx_dispatch.cpp and
   read by all four levels, so they must stay outside the per-level inline namespace. */
}
extern "C" {
/* Not const: rad_plugin_open fills them at load. A const declaration against the non-const
   definition would still link (C linkage does not mangle the type), unreported. */
extern float avx_fp8e4m3_tab[256];
extern float avx_fp8e5m2_tab[256];
}
namespace avx {
inline namespace AVX_SUFFIX {

/* e4m3 decode by bit reconstruction rather than a vpgatherdd table lookup. A gather issues one L1
 * access per lane even for a 1 KiB table, and `rad-avx-bench --cvt` shows it no faster than the
 * scalar lookup; the reconstruction is about 2.5x faster at AVX2 and 3.3x at AVX-512.
 *
 *   u = (b & 0x7f) << 20  +  120 << 23    e into the f32 exponent field (bias 7 -> 127), m into
 *                                         the top three mantissa bits, in one add
 *   e == 0                                fp8 subnormal: (f - 2^-7) * 2 removes the implicit 1
 *                                         and rescales, exactly
 *   mag == 0x7f                           the format's only NaN (S1111111); blended in
 *   sign                                  (b & 0x80) << 24
 *
 * The table remains for the scalar level and tails. The checker compares all 256 input bytes,
 * NaN and subnormals included. */
static inline void cvt_fp8e4m3_to_f32(const uint8_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    const __m512i c7f = _mm512_set1_epi32(0x7f);
    const __m512i cbias = _mm512_set1_epi32(120 << 23);
    const __m512  chalf = _mm512_set1_ps(0.0078125f);      /* 2^-7 */
    const __m512  cnan = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000));
    for (; i + 16 <= n; i += 16) {
        const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(src + i)));
        const __m512i mag = _mm512_and_si512(b, c7f);
        const __m512i u = _mm512_add_epi32(_mm512_slli_epi32(mag, 20), cbias);
        __m512 f = _mm512_castsi512_ps(u);
        const __mmask16 sub = _mm512_cmplt_epi32_mask(mag, _mm512_set1_epi32(8));
        f = _mm512_mask_blend_ps(sub, f,
                                 _mm512_mul_ps(_mm512_sub_ps(f, chalf), _mm512_set1_ps(2.0f)));
        f = _mm512_mask_blend_ps(_mm512_cmpeq_epi32_mask(mag, c7f), f, cnan);
        const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x80)), 24);
        _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(
                                      _mm512_or_si512(_mm512_castps_si512(f), sign)));
    }
#elif AVX_LEVEL == 2
    const __m256i c7f = _mm256_set1_epi32(0x7f);
    const __m256i cbias = _mm256_set1_epi32(120 << 23);
    const __m256  chalf = _mm256_set1_ps(0.0078125f);
    const __m256  cnan = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));
    for (; i + 8 <= n; i += 8) {
        const __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(src + i)));
        const __m256i mag = _mm256_and_si256(b, c7f);
        const __m256i u = _mm256_add_epi32(_mm256_slli_epi32(mag, 20), cbias);
        __m256 f = _mm256_castsi256_ps(u);
        const __m256i sub = _mm256_cmpgt_epi32(_mm256_set1_epi32(8), mag);
        f = _mm256_blendv_ps(f, _mm256_mul_ps(_mm256_sub_ps(f, chalf), _mm256_set1_ps(2.0f)),
                             _mm256_castsi256_ps(sub));
        f = _mm256_blendv_ps(f, cnan, _mm256_castsi256_ps(_mm256_cmpeq_epi32(mag, c7f)));
        const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x80)), 24);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(
                                      _mm256_or_si256(_mm256_castps_si256(f), sign)));
    }
#endif
    for (; i < n; ++i) dst[i] = avx_fp8e4m3_tab[src[i]];
}

static inline void cvt_fp8e5m2_to_f32(const uint8_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16) {
        __m512i i32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(src + i)));
        _mm512_storeu_ps(dst + i, _mm512_i32gather_ps(i32, avx_fp8e5m2_tab, 4));
    }
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8) {
        __m256i i32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(src + i)));
        _mm256_storeu_ps(dst + i, _mm256_i32gather_ps(avx_fp8e5m2_tab, i32, 4));
    }
#endif
    for (; i < n; ++i) dst[i] = avx_fp8e5m2_tab[src[i]];
}

static inline void cvt_f32_to_fp8e4m3(const float* src, uint8_t* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    /* q = ((e+7) << 3) | rtne(mantissa >> 20), valid for finite 2^-6 <= |x| < 448; anything
     * else takes the scalar fixup.
     *
     * The lower bound must be exactly 2^-6, e4m3's smallest normal. Below it the format is
     * subnormal and needs a variable renormalising shift; a lower bound would send subnormals
     * through this path with an exponent field <= 0, wrong by orders of magnitude. */
    const __m512i c_lo  = _mm512_set1_epi32(0x3c800000);  /* 2^-6, the smallest e4m3 normal */
    const __m512i c_hi  = _mm512_set1_epi32(0x43e00000);  /* 448,  the largest e4m3 finite  */
    for (; i + 16 <= n; i += 16) {
        __m512  f   = _mm512_loadu_ps(src + i);
        __m512i u   = _mm512_castps_si512(f);
        __m512i mag = _mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff));
        __mmask16 fast = _kand_mask16(_mm512_cmpge_epi32_mask(mag, c_lo),
                                      _mm512_cmplt_epi32_mask(mag, c_hi));
        __m512i sign = _mm512_srli_epi32(_mm512_and_si512(u, _mm512_set1_epi32(0x80000000u)), 24);
        /* RTNE on the 20 dropped bits: add half, plus one when the surviving bit is odd (bf16's
         * tie rule). A mantissa carry propagates into the exponent in the same add. */
        __m512i lsb  = _mm512_and_si512(_mm512_srli_epi32(mag, 20), _mm512_set1_epi32(1));
        __m512i r    = _mm512_add_epi32(_mm512_add_epi32(mag, _mm512_set1_epi32(0x7ffff)), lsb);
        __m512i e    = _mm512_sub_epi32(_mm512_srli_epi32(r, 23), _mm512_set1_epi32(127 - 7));
        __m512i m    = _mm512_and_si512(_mm512_srli_epi32(r, 20), _mm512_set1_epi32(7));
        __m512i q    = _mm512_or_si512(sign, _mm512_or_si512(_mm512_slli_epi32(e, 3), m));
        /* A carry can push e past 15, and (e, m) = (15, 7) is the NaN encoding. Neither occurs
         * for inputs below 448, but such lanes are sent to the saturating fixup rather than risk
         * a silent NaN. */
        __mmask16 ovf = _kor_mask16(
            _mm512_cmpgt_epi32_mask(e, _mm512_set1_epi32(15)),
            _kand_mask16(_mm512_cmpeq_epi32_mask(e, _mm512_set1_epi32(15)),
                         _mm512_cmpeq_epi32_mask(m, _mm512_set1_epi32(7))));
        fast = _kandn_mask16(ovf, fast);
        /* All lanes fast (the normal case for activations): one narrowing store. Without this
         * exit, the per-lane fixup loop below would run on every vector and dominate the op. */
        if (fast == (__mmask16)0xffff) {
            _mm_storeu_si128((__m128i*)(dst + i), _mm512_cvtepi32_epi8(q));
            continue;
        }
        alignas(64) int32_t tmp[16];
        _mm512_store_si512((__m512i*)tmp, q);
        for (int k = 0; k < 16; ++k)
            dst[i + k] = (fast >> k) & 1 ? (uint8_t)tmp[k] : rad_f32_to_fp8e4m3(src[i + k]);
    }
#elif AVX_LEVEL == 2
    /* The same arithmetic at 256 bits (the scalar converter is an order of magnitude slower).
     * The mask is a movemask instead of a k-register; the all-fast exit narrows eight lanes to
     * eight bytes with two packs. */
    for (; i + 8 <= n; i += 8) {
        __m256  f   = _mm256_loadu_ps(src + i);
        __m256i u   = _mm256_castps_si256(f);
        __m256i mag = _mm256_and_si256(u, _mm256_set1_epi32(0x7fffffff));
        const __m256i c_lo = _mm256_set1_epi32(0x3c800000);   /* 2^-6 */
        const __m256i c_hi = _mm256_set1_epi32(0x43e00000);   /* 448  */
        /* AVX2 has only signed cmpgt: `mag >= c_lo` is `!(c_lo > mag)`, `mag < c_hi` is
         * `c_hi > mag`. Both operands are non-negative. */
        __m256i ge = _mm256_andnot_si256(_mm256_cmpgt_epi32(c_lo, mag),
                                         _mm256_cmpgt_epi32(c_hi, mag));
        __m256i sign = _mm256_srli_epi32(_mm256_and_si256(u, _mm256_set1_epi32(0x80000000u)), 24);
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(mag, 20), _mm256_set1_epi32(1));
        __m256i r = _mm256_add_epi32(_mm256_add_epi32(mag, _mm256_set1_epi32(0x7ffff)), lsb);
        __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(r, 23), _mm256_set1_epi32(127 - 7));
        __m256i m = _mm256_and_si256(_mm256_srli_epi32(r, 20), _mm256_set1_epi32(7));
        __m256i q = _mm256_or_si256(sign, _mm256_or_si256(_mm256_slli_epi32(e, 3), m));
        __m256i ovf = _mm256_or_si256(
            _mm256_cmpgt_epi32(e, _mm256_set1_epi32(15)),
            _mm256_and_si256(_mm256_cmpeq_epi32(e, _mm256_set1_epi32(15)),
                             _mm256_cmpeq_epi32(m, _mm256_set1_epi32(7))));
        ge = _mm256_andnot_si256(ovf, ge);
        const int fast = _mm256_movemask_ps(_mm256_castsi256_ps(ge));
        if (fast == 0xff) {
            __m128i p = _mm_packus_epi32(_mm256_castsi256_si128(q),
                                         _mm256_extracti128_si256(q, 1));
            p = _mm_packus_epi16(p, p);
            _mm_storel_epi64((__m128i*)(dst + i), p);
            continue;
        }
        alignas(32) int32_t tmp[8];
        _mm256_store_si256((__m256i*)tmp, q);
        for (int k = 0; k < 8; ++k)
            dst[i + k] = (fast >> k) & 1 ? (uint8_t)tmp[k] : rad_f32_to_fp8e4m3(src[i + k]);
    }
#endif
    for (; i < n; ++i) dst[i] = rad_f32_to_fp8e4m3(src[i]);
}

static inline void cvt_f32_to_fp8e5m2(const float* src, uint8_t* dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] = rad_f32_to_fp8e5m2(src[i]);
}

/* ---- int8. `_mm*_cvtps_epi32` rounds under MXCSR (round-to-nearest-even by default), the same
 * rule as libref's `std::nearbyintf`.
 *
 * The clamp is applied in float before conversion. The saturating packs are not enough: +inf and
 * values beyond int32 convert to 0x80000000 ("integer indefinite"), which packs to -128.
 * Clamp-then-round matches libref's round-then-clamp for every input, including .5 ties at the
 * ends, so `quant_act` holds at 1e-6. NaN stores 0, as in the scalar tail. */
#if AVX_LEVEL == 3
static inline __m512 i8_range_ps(__m512 x) {
    const __mmask16 ord = _mm512_cmp_ps_mask(x, x, _CMP_ORD_Q);
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-128.0f)), _mm512_set1_ps(127.0f));
    return _mm512_maskz_mov_ps(ord, x);
}
#elif AVX_LEVEL == 2
static inline __m256 i8_range_ps(__m256 x) {
    const __m256 ord = _mm256_cmp_ps(x, x, _CMP_ORD_Q);
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-128.0f)), _mm256_set1_ps(127.0f));
    return _mm256_and_ps(x, ord);
}
#endif

static inline int8_t i8_round_sat(float v) {
    if (std::isnan(v)) return 0;
    const float r = std::nearbyintf(v);
    return (int8_t)(r < -128.f ? -128.f : (r > 127.f ? 127.f : r));
}

static inline void cvt_f32_to_i8(const float* src, int8_t* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16) {
        __m512i q = _mm512_cvtps_epi32(i8_range_ps(_mm512_loadu_ps(src + i)));
        _mm_storeu_si128((__m128i*)(dst + i), _mm512_cvtsepi32_epi8(q));
    }
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8) {
        __m256i q  = _mm256_cvtps_epi32(i8_range_ps(_mm256_loadu_ps(src + i)));
        __m128i p  = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        p = _mm_packs_epi16(p, p);
        _mm_storel_epi64((__m128i*)(dst + i), p);
    }
#endif
    for (; i < n; ++i) dst[i] = i8_round_sat(src[i]);
}

/* Fused scale + int8 encode for the quant_act_i8 family: src * inv in a register feeds the same
 * clamp/cvtps/packs sequence, so the bytes match scale-then-encode without an f32 round trip
 * through memory. Not done for fp8: folding the scale into its longer integer chain is slower. */
static inline void cvt_scaled_f32_to_i8(const float* src, float inv, int8_t* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    const __m512 vinv = _mm512_set1_ps(inv);
    for (; i + 16 <= n; i += 16) {
        __m512 t = _mm512_mul_ps(_mm512_loadu_ps(src + i), vinv);
        __m512i q = _mm512_cvtps_epi32(i8_range_ps(t));
        _mm_storeu_si128((__m128i*)(dst + i), _mm512_cvtsepi32_epi8(q));
    }
#elif AVX_LEVEL == 2
    const __m256 vinv = _mm256_set1_ps(inv);
    for (; i + 8 <= n; i += 8) {
        __m256 t = _mm256_mul_ps(_mm256_loadu_ps(src + i), vinv);
        __m256i q  = _mm256_cvtps_epi32(i8_range_ps(t));
        __m128i p  = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
        p = _mm_packs_epi16(p, p);
        _mm_storel_epi64((__m128i*)(dst + i), p);
    }
#endif
    for (; i < n; ++i) dst[i] = i8_round_sat(src[i] * inv);
}

static inline void cvt_i8_to_f32(const int8_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(dst + i,
                         _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                             _mm_loadu_si128((const __m128i*)(src + i)))));
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i,
                         _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                             _mm_loadl_epi64((const __m128i*)(src + i)))));
#endif
    for (; i < n; ++i) dst[i] = (float)src[i];
}

/* ================================================================== the dtype-generic row
 *
 * One dtype switch per row instead of libref's per-element ld_dt(); for memory-bound ops this
 * accounts for most of the speedup. The f32 case can be skipped entirely (row_f32_in).
 *
 * The packed weight codes W4 / W2 / MXFP4 decode here too (for `dequant` and the quantised GEMMs),
 * as raw codes: the caller applies the group scale (the same split as libref/ref_common.h). */
void row_to_f32  (uint32_t dt, const void* src, float* dst, int64_t n);
void row_from_f32(uint32_t dt, const float* src, void* dst, int64_t n);

/* Element i at src[i * stride]. Separate so the contiguous path carries no stride multiply. */
void row_to_f32_strided(uint32_t dt, const void* src, int64_t stride, float* dst, int64_t n);

/* Returns `src` itself for RAD_F32, otherwise `scratch` filled with the converted row. */
static inline const float* row_f32_in(uint32_t dt, const void* src, float* scratch, int64_t n) {
    if (dt == RAD_F32) return (const float*)src;
    row_to_f32(dt, src, scratch, n);
    return scratch;
}

/* ================================================================== math
 *
 * exp and sigmoid. Softmax, gates and attention rows all go through these, and a per-element libm
 * expf would dominate those ops.
 *
 * Minimax degree 5 on [-ln2/2, ln2/2] after range reduction, ~0.6 ulp. That fits softmax's 1e-6
 * because normalisation cancels the relative error common to all terms.
 *
 * Hand-written because SVML's _mm512_exp_ps needs Intel's runtime, and GCC's libmvec only
 * vectorises expf where it can prove the call safe, which fails in kernels that mix integer work. */
static inline vf vf_exp(vf x) {
#if AVX_LEVEL == 0
    return std::exp(x);
#else
    /* Clamp first: outside this range the integer exponent construction overflows. The top
     * saturates, which every consumer here prefers to an infinity. Below lo (under the smallest
     * normal float) the result is set to zero, and NaN is passed through, after the polynomial;
     * the clamp alone would turn e^-inf into e^lo and lose a NaN, so a fully masked softmax row
     * would give 1/n instead of libref's NaN. */
    const vf hi = vf_set1(88.3762626647949f), lo = vf_set1(-87.3365447504f);
    const vf x0 = x;
    x = vf_min(vf_max(x, lo), hi);
    const vf log2e = vf_set1(1.44269504088896341f);
    const vf c1 = vf_set1(-0.693359375f), c2 = vf_set1(2.12194440e-4f);
#if AVX_LEVEL == 3
    vf n = _mm512_roundscale_ps(vf_mul(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    vf n = _mm256_round_ps(vf_mul(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
    /* x - n*ln2, split so the product is exact in f32. */
    vf r = vf_fma(n, c1, x);
    r = vf_fma(n, c2, r);
    vf r2 = vf_mul(r, r);
    /* exp(r) on [-ln2/2, ln2/2], Horner form. */
    vf p = vf_set1(1.9875691500e-4f);
    p = vf_fma(p, r, vf_set1(1.3981999507e-3f));
    p = vf_fma(p, r, vf_set1(8.3334519073e-3f));
    p = vf_fma(p, r, vf_set1(4.1665795894e-2f));
    p = vf_fma(p, r, vf_set1(1.6666665459e-1f));
    p = vf_fma(p, r, vf_set1(5.0000001201e-1f));
    p = vf_fma(p, r2, vf_add(r, vf_set1(1.0f)));
    /* 2^n by building the exponent field directly. */
#if AVX_LEVEL == 3
    __m512i k = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
    vf e = vf_mul(p, _mm512_castsi512_ps(k));
    e = _mm512_mask_mov_ps(e, _mm512_cmp_ps_mask(x0, lo, _CMP_LT_OQ), _mm512_setzero_ps());
    return _mm512_mask_mov_ps(e, _mm512_cmp_ps_mask(x0, x0, _CMP_UNORD_Q), x0);
#else
#if AVX_LEVEL == 2
    __m256i k = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    vf e = vf_mul(p, _mm256_castsi256_ps(k));
#else
    __m128i nlo = _mm_cvtps_epi32(_mm256_castps256_ps128(n));
    __m128i nhi = _mm_cvtps_epi32(_mm256_extractf128_ps(n, 1));
    nlo = _mm_slli_epi32(_mm_add_epi32(nlo, _mm_set1_epi32(127)), 23);
    nhi = _mm_slli_epi32(_mm_add_epi32(nhi, _mm_set1_epi32(127)), 23);
    __m256 k = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(nlo)),
                                    _mm_castsi128_ps(nhi), 1);
    vf e = vf_mul(p, k);
#endif
    e = _mm256_andnot_ps(_mm256_cmp_ps(x0, lo, _CMP_LT_OQ), e);
    return _mm256_blendv_ps(e, x0, _mm256_cmp_ps(x0, x0, _CMP_UNORD_Q));
#endif
#endif
}

static inline vf vf_sigmoid(vf x) {
    /* 1 / (1 + e^-x), as libref computes it. The tanh form differs in the last ulp, which
     * `hc_read`'s chained sigmoid gates accumulate. */
    return vf_div(vf_set1(1.0f), vf_add(vf_set1(1.0f), vf_exp(vf_sub(vf_zero(), x))));
}

static inline vf vf_silu(vf x) { return vf_mul(x, vf_sigmoid(x)); }

#if AVX_LEVEL >= 2
/* Cheap silu, only for consumers whose output is bf16: a Taylor-4 exp (truncation ~6e-5) and a
 * raw reciprocal seed (2^-14 avx512, 2^-12 avx2, no Newton step). Worst lane ~3e-4, well inside the
 * bf16 tolerance of 6e-3 but not f32's 1e-5, so callers key on their output dtype. silu_mul uses
 * it; in the frontend-bound conv kernels it makes no measurable difference. */
static inline vf vf_exp_t4(vf x) {
    const vf hi = vf_set1(88.3762626647949f), lo = vf_set1(-87.3365447504f);
    x = vf_min(vf_max(x, lo), hi);
    const vf log2e = vf_set1(1.44269504088896341f);
    const vf c1 = vf_set1(-0.693359375f), c2 = vf_set1(2.12194440e-4f);
#if AVX_LEVEL == 3
    vf n = _mm512_roundscale_ps(vf_mul(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    vf n = _mm256_round_ps(vf_mul(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
    vf r = vf_fma(n, c1, x);
    r = vf_fma(n, c2, r);
    /* Taylor-4: 1 + r(1 + r(1/2 + r(1/6 + r/24))). Same range reduction and exponent build as
     * vf_exp. */
    vf p = vf_fma(r, vf_set1(1.0f / 24.0f), vf_set1(1.0f / 6.0f));
    p = vf_fma(p, r, vf_set1(0.5f));
    p = vf_fma(p, r, vf_set1(1.0f));
    p = vf_fma(p, r, vf_set1(1.0f));
#if AVX_LEVEL == 3
    __m512i k = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
    return vf_mul(p, _mm512_castsi512_ps(k));
#else
    __m256i k = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return vf_mul(p, _mm256_castsi256_ps(k));
#endif
}

static inline vf vf_silu_fast(vf x) {
#if AVX_LEVEL == 3
    vf e = vf_exp_t4(vf_sub(vf_zero(), x));
    vf d = vf_add(vf_set1(1.0f), e);
    vf r = _mm512_rcp14_ps(d);
#else
    vf e = vf_exp_t4(vf_sub(vf_zero(), x));
    vf d = vf_add(vf_set1(1.0f), e);
    vf r = _mm256_rcp_ps(d);
#endif
    /* No Newton step: d is in [1, emax], so the seed is in range and nothing overflows. */
    return vf_mul(x, r);
}
#endif

/* ================================================================== exact top-R
 * One step of an exact top-R selection over an f32 row: descending by value, ties to the lower
 * index (libr4d's rule; a router that broke ties differently would pick a different expert).
 * `pv`/`pi` carry the previous pick; seed them with +infinity and -1. Returns false when nothing
 * is left. Shared by row_topk, router_topk and sample_topk. */
bool topk_next_f32(const float* v, int64_t n, float* pv, int64_t* pi);

/* ================================================================== an operand's row, as f32
 * `in_row` returns `n` floats holding elements (base + i * laststride) of the operand: `t->data`
 * itself when it is dense f32, otherwise `scr` filled. `out_row` is the inverse. Every op must
 * accept strided views (libref/ref_ops.h), and this is the one place the dense/strided split is
 * made. */
const float* in_row (const struct RadTensor* t, int64_t base, int64_t n, float* scr);
void         out_row(const struct RadTensor* t, int64_t base, int64_t n, const float* src);

/* A dot product with four accumulators to cover FMA latency, combined (a0 + a1) + (a2 + a3) before
 * the scalar tail: a fixed summation tree. Inline, since the attention and GEMM edges call it per
 * element. */
static inline float dotf(const float* a, const float* b, int64_t n) {
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t i = 0;
    for (; i + 4 * VF_N <= n; i += 4 * VF_N) {
        a0 = vf_fma(vf_loadu(a + i), vf_loadu(b + i), a0);
        a1 = vf_fma(vf_loadu(a + i + VF_N), vf_loadu(b + i + VF_N), a1);
        a2 = vf_fma(vf_loadu(a + i + 2 * VF_N), vf_loadu(b + i + 2 * VF_N), a2);
        a3 = vf_fma(vf_loadu(a + i + 3 * VF_N), vf_loadu(b + i + 3 * VF_N), a3);
    }
    for (; i + VF_N <= n; i += VF_N) a0 = vf_fma(vf_loadu(a + i), vf_loadu(b + i), a0);
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* Row reductions with a fixed summation tree, independent of compiler unrolling. */
float row_sumsq(const float* x, int64_t n);
float row_sum  (const float* x, int64_t n);
float row_amax (const float* x, int64_t n);
float row_max  (const float* x, int64_t n);

}  /* inline namespace AVX_SUFFIX */
}  /* namespace avx */
#endif /* RAD_AVX_VEC_H */
