/* avx_planes.cpp -- decoding the rows of a weight joined over several planes (avx_planes.h),
 * compiled once per ISA level. The GEMM's packing, the head rescore and the mixing matrices widen
 * such a row to f32 here, scale folded in, where a plain row would go through row_to_f32. */
#include "avx_vec.h"
#include "avx_common.h"
#include "avx_planes.h"

namespace avx {
inline namespace AVX_SUFFIX {

/* n E4M3 or int8 codes, each times `s`. The E4M3 decode is avx_vec.h's reconstruction
 * (cvt_fp8e4m3_to_f32), so the code's value is bit-exact before the one rounding of the
 * product. */
void decode_scaled(uint32_t dt, const uint8_t* src, float s, float* dst, int64_t n) {
    int64_t i = 0;
    if (dt == RAD_I8) {
        const int8_t* q = (const int8_t*)src;
#if AVX_LEVEL == 3
        const __m512 vs = _mm512_set1_ps(s);
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                                          _mm_loadu_si128((const __m128i*)(q + i)))), vs));
#elif AVX_LEVEL == 2
        const __m256 vs = _mm256_set1_ps(s);
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                          _mm_loadl_epi64((const __m128i*)(q + i)))), vs));
#endif
        for (; i < n; ++i) dst[i] = (float)q[i] * s;
        return;
    }
#if AVX_LEVEL == 3
    const __m512i c7f = _mm512_set1_epi32(0x7f);
    const __m512i cbias = _mm512_set1_epi32(120 << 23);
    const __m512  chalf = _mm512_set1_ps(0.0078125f);
    const __m512  cnan = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000));
    const __m512  vs = _mm512_set1_ps(s);
    for (; i + 16 <= n; i += 16) {
        const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(src + i)));
        const __m512i mag = _mm512_and_si512(b, c7f);
        __m512 f = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_slli_epi32(mag, 20), cbias));
        const __mmask16 sub = _mm512_cmplt_epi32_mask(mag, _mm512_set1_epi32(8));
        f = _mm512_mask_blend_ps(sub, f,
                                 _mm512_mul_ps(_mm512_sub_ps(f, chalf), _mm512_set1_ps(2.0f)));
        f = _mm512_mask_blend_ps(_mm512_cmpeq_epi32_mask(mag, c7f), f, cnan);
        const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x80)), 24);
        f = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(f), sign));
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(f, vs));
    }
#elif AVX_LEVEL == 2
    const __m256i c7f = _mm256_set1_epi32(0x7f);
    const __m256i cbias = _mm256_set1_epi32(120 << 23);
    const __m256  chalf = _mm256_set1_ps(0.0078125f);
    const __m256  cnan = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000));
    const __m256  vs = _mm256_set1_ps(s);
    for (; i + 8 <= n; i += 8) {
        const __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(src + i)));
        const __m256i mag = _mm256_and_si256(b, c7f);
        __m256 f = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(mag, 20), cbias));
        const __m256i sub = _mm256_cmpgt_epi32(_mm256_set1_epi32(8), mag);
        f = _mm256_blendv_ps(f, _mm256_mul_ps(_mm256_sub_ps(f, chalf), _mm256_set1_ps(2.0f)),
                             _mm256_castsi256_ps(sub));
        f = _mm256_blendv_ps(f, cnan, _mm256_castsi256_ps(_mm256_cmpeq_epi32(mag, c7f)));
        const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x80)), 24);
        f = _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(f), sign));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(f, vs));
    }
#endif
    for (; i < n; ++i) dst[i] = avx_fp8e4m3_tab[src[i]] * s;
}

#if AVX_LEVEL >= 2
/* Sixteen codes, one per byte of `b` in natural order, each (code - z) * s, into dst. */
static inline void zp_emit16(__m128i b, int32_t z, float s, float* dst) {
#if AVX_LEVEL == 3
    const __m512i v = _mm512_sub_epi32(_mm512_cvtepu8_epi32(b), _mm512_set1_epi32(z));
    _mm512_storeu_ps(dst, _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(s)));
#else
    const __m256i v0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(b), _mm256_set1_epi32(z));
    const __m256i v1 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(b, 8)),
                                        _mm256_set1_epi32(z));
    _mm256_storeu_ps(dst, _mm256_mul_ps(_mm256_cvtepi32_ps(v0), _mm256_set1_ps(s)));
    _mm256_storeu_ps(dst + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(v1), _mm256_set1_ps(s)));
#endif
}

/* u5 codes by pdep: eight codes are 40 bits, and one pdep deposits them into the low five bits of
 * eight bytes. BMI2 is not in the level's flags (it is a CPUID bit of its own) and is checked at
 * run time; Zen 1 and 2 microcode pdep, slowly but correctly. Each 40 bits are read as one
 * 8-byte load, masked, except in the last 16 codes, where 3 of those bytes would lie past the
 * ones asked for. Returns the codes decoded, a multiple of 16. */
__attribute__((target("bmi2"))) static int64_t zp_u5_pdep(const uint8_t* src, int64_t n,
                                                          int32_t z, float s, float* dst) {
    const unsigned long long kLow5 = 0x1f1f1f1f1f1f1f1full, k40 = 0xffffffffffull;
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const uint8_t* p = src + i / 8 * 5;
        uint64_t v0 = 0, v1 = 0;
        if (i + 16 < n) {
            std::memcpy(&v0, p, 8);
            std::memcpy(&v1, p + 5, 8);
            v0 &= k40;
            v1 &= k40;
        } else {
            std::memcpy(&v0, p, 5);
            std::memcpy(&v1, p + 5, 5);
        }
        const __m128i b = _mm_set_epi64x((long long)_pdep_u64(v1, kLow5),
                                         (long long)_pdep_u64(v0, kLow5));
        zp_emit16(b, z, s, dst + i);
    }
    return i;
}
#endif

/* `n` unsigned codes of `bits` from element k0 of a packed row, each (code - z) * s. Grid groups
 * start on a byte for every width here (k0 a multiple of 128); other starts take the bit reader. */
void decode_zp(const uint8_t* row, int bits, int64_t k0, int64_t n, int32_t z, float s,
               float* dst) {
    int64_t i = 0;
#if AVX_LEVEL >= 2
    const bool aligned = (k0 * bits) % 8 == 0;
    const uint8_t* src = row + (k0 * bits) / 8;
    if (aligned && bits == 4) {
        const __m128i m4 = _mm_set1_epi8(0x0f);
        for (; i + 16 <= n; i += 16) {
            const __m128i x = _mm_loadl_epi64((const __m128i*)(src + i / 2));
            const __m128i lo = _mm_and_si128(x, m4);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), m4);
            zp_emit16(_mm_unpacklo_epi8(lo, hi), z, s, dst + i);
        }
    } else if (aligned && bits == 2) {
        const __m128i m2 = _mm_set1_epi8(0x03);
        for (; i + 16 <= n; i += 16) {
            int32_t w;
            std::memcpy(&w, src + i / 4, 4);
            const __m128i x = _mm_cvtsi32_si128(w);
            const __m128i p0 = _mm_and_si128(x, m2);
            const __m128i p1 = _mm_and_si128(_mm_srli_epi16(x, 2), m2);
            const __m128i p2 = _mm_and_si128(_mm_srli_epi16(x, 4), m2);
            const __m128i p3 = _mm_and_si128(_mm_srli_epi16(x, 6), m2);
            zp_emit16(_mm_unpacklo_epi16(_mm_unpacklo_epi8(p0, p1), _mm_unpacklo_epi8(p2, p3)),
                      z, s, dst + i);
        }
    } else if (aligned && bits == 5) {
        static const bool bmi2 = __builtin_cpu_supports("bmi2");
        if (bmi2) {
            i = zp_u5_pdep(src, n, z, s, dst);
        } else {
            for (; i + 16 <= n; i += 16) {
                alignas(16) uint8_t q[16];
                for (int h = 0; h < 2; ++h) {
                    uint64_t v = 0;
                    std::memcpy(&v, src + (i + 8 * h) / 8 * 5, 5);
                    for (int t = 0; t < 8; ++t) q[8 * h + t] = (uint8_t)((v >> (5 * t)) & 31u);
                }
                zp_emit16(_mm_load_si128((const __m128i*)q), z, s, dst + i);
            }
        }
    }
#endif
    for (; i < n; ++i) dst[i] = (float)((int32_t)rad_load_bits(row, bits, k0 + i) - z) * s;
}

void scaled_row(const avxp::ScaledRows& sr, int64_t r, int64_t K, float* dst) {
    const uint8_t* c = sr.codes.p + r * sr.codes.row_bytes;
    const int64_t span = sr.span(K);
    for (int64_t c0 = 0; c0 < K; c0 += span) {
        const int64_t c1 = c0 + span < K ? c0 + span : K;
        decode_scaled(sr.codes.dt, c + c0, sr.scale_at(r, c0), dst + c0, c1 - c0);
    }
}

}  /* inline namespace AVX_SUFFIX */
}  /* namespace avx */
