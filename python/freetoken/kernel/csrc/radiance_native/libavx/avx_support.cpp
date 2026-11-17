/* avx_support.cpp -- dtype-generic row conversion and row reductions, compiled once per ISA level.
 *
 * Out of line rather than in the header: every kernel calls these with a runtime dtype, so inlining
 * would paste the dtype switch into every call site. Out of line, the switch is taken once per row
 * and lands in a straight vector loop.
 *
 * Conversion rates per level come from `rad-avx-bench --cvt` (libavx/README.md). The two fp8
 * decodes differ (avx_vec.h explains): e4m3 is a branchless bit reconstruction, e5m2 a table
 * gather; the scalar level and tail elements use the table for both.
 */
#include "avx_vec.h"
#include "avx_common.h"

namespace avx {
inline namespace AVX_SUFFIX {

/* ================================================================== packed weight codes
 * The canonical planes a container stores (abi/rad_encoding.h), read as stored: libavx rearranges
 * no plane, its relayouts only join whole planes (avx_planes.h). Definition: libref/ref_common.h.
 * W4: two nibbles a byte, element 2i in the low nibble, code (n ^ 8) - 8. W2: four codes a byte,
 * code (c ^ 2) - 2. MXFP4: two e2m1 nibbles, as avx_common.h: ld_elem decodes them.
 *
 * W4 unpacks 32 elements from one 128-bit load: AND for the low nibbles, shift for the high, then
 * interleave with punpcklbw; MXFP4 takes the same unpack and a register table. W2 stays scalar: no
 * plain W2 weight is served. */

static void w4_to_f32(const uint8_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL >= 2
    const __m128i lomask = _mm_set1_epi8(0x0f);
    for (; i + 32 <= n; i += 32) {
        __m128i b  = _mm_loadu_si128((const __m128i*)(src + (i >> 1)));
        __m128i lo = _mm_and_si128(b, lomask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), lomask);
        lo = _mm_sub_epi8(_mm_xor_si128(lo, _mm_set1_epi8(8)), _mm_set1_epi8(8));
        hi = _mm_sub_epi8(_mm_xor_si128(hi, _mm_set1_epi8(8)), _mm_set1_epi8(8));
        __m128i e0 = _mm_unpacklo_epi8(lo, hi);   /* codes 0..15  */
        __m128i e1 = _mm_unpackhi_epi8(lo, hi);   /* codes 16..31 */
#if AVX_LEVEL >= 3
        _mm512_storeu_ps(dst + i,      _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(e0)));
        _mm512_storeu_ps(dst + i + 16, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(e1)));
#else
        _mm256_storeu_ps(dst + i,      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(e0)));
        _mm256_storeu_ps(dst + i + 8,
                         _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(e0, 8))));
        _mm256_storeu_ps(dst + i + 16, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(e1)));
        _mm256_storeu_ps(dst + i + 24,
                         _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(e1, 8))));
#endif
    }
#endif
    for (; i < n; ++i) dst[i] = ld_elem(src, RAD_I4, i);
}

/* MXFP4: the same nibble unpack, then the e2m1 value of each code from a register table. AVX-512
 * looks all sixteen up in one vpermps; AVX2's vpermps indexes eight, so it takes the magnitude by
 * the low three bits and ors in the sign (bit 3). Code 8 is -0, as ld_elem gives it. */
static void fp4_to_f32(const uint8_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL >= 2
    const __m128i lomask = _mm_set1_epi8(0x0f);
#if AVX_LEVEL >= 3
    const __m512 tab = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f,
                                      -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
#else
    const __m256 mag = _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f);
    auto emit8 = [&](__m128i codes8, float* d) {
        const __m256i ix = _mm256_cvtepu8_epi32(codes8);
        const __m256 m = _mm256_permutevar8x32_ps(mag, ix);
        const __m256i sg = _mm256_slli_epi32(_mm256_and_si256(ix, _mm256_set1_epi32(8)), 28);
        _mm256_storeu_ps(d, _mm256_or_ps(m, _mm256_castsi256_ps(sg)));
    };
#endif
    for (; i + 32 <= n; i += 32) {
        const __m128i b  = _mm_loadu_si128((const __m128i*)(src + (i >> 1)));
        const __m128i lo = _mm_and_si128(b, lomask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), lomask);
        const __m128i e0 = _mm_unpacklo_epi8(lo, hi);   /* codes 0..15  */
        const __m128i e1 = _mm_unpackhi_epi8(lo, hi);   /* codes 16..31 */
#if AVX_LEVEL >= 3
        _mm512_storeu_ps(dst + i, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(e0), tab));
        _mm512_storeu_ps(dst + i + 16, _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(e1), tab));
#else
        emit8(e0, dst + i);
        emit8(_mm_srli_si128(e0, 8), dst + i + 8);
        emit8(e1, dst + i + 16);
        emit8(_mm_srli_si128(e1, 8), dst + i + 24);
#endif
    }
#endif
    for (; i < n; ++i) dst[i] = ld_elem(src, RAD_FP4E2M1, i);
}

/* ================================================================== the generic row */
void row_to_f32(uint32_t dt, const void* src, float* dst, int64_t n) {
    switch (dt) {
        case RAD_F32:    std::memcpy(dst, src, (size_t)n * 4); break;
        case RAD_BF16:   cvt_bf16_to_f32((const uint16_t*)src, dst, n); break;
        case RAD_F16:    cvt_f16_to_f32((const uint16_t*)src, dst, n); break;
        case RAD_F8E4M3: cvt_fp8e4m3_to_f32((const uint8_t*)src, dst, n); break;
        case RAD_F8E5M2: cvt_fp8e5m2_to_f32((const uint8_t*)src, dst, n); break;
        case RAD_I8:     cvt_i8_to_f32((const int8_t*)src, dst, n); break;
        case RAD_I4:     w4_to_f32((const uint8_t*)src, dst, n); break;
        case RAD_I2:     for (int64_t i = 0; i < n; ++i) dst[i] = ld_elem(src, RAD_I2, i); break;
        case RAD_FP4E2M1: fp4_to_f32((const uint8_t*)src, dst, n); break;
        default:
            /* Remaining integer and bool dtypes are not hot; use the ABI accessor. */
            for (int64_t i = 0; i < n; ++i) dst[i] = rad_load_f32(src, dt, i);
            break;
    }
}

void row_from_f32(uint32_t dt, const float* src, void* dst, int64_t n) {
    switch (dt) {
        case RAD_F32:    std::memcpy(dst, src, (size_t)n * 4); break;
        case RAD_BF16:   cvt_f32_to_bf16(src, (uint16_t*)dst, n); break;
        case RAD_F16:    cvt_f32_to_f16(src, (uint16_t*)dst, n); break;
        case RAD_F8E4M3: cvt_f32_to_fp8e4m3(src, (uint8_t*)dst, n); break;
        case RAD_F8E5M2: cvt_f32_to_fp8e5m2(src, (uint8_t*)dst, n); break;
        case RAD_I8:     cvt_f32_to_i8(src, (int8_t*)dst, n); break;
        default:
            /* Every other integer width rounds and saturates as i8 does (avx_common.h: st_elem). */
            for (int64_t i = 0; i < n; ++i) st_elem(dst, dt, i, src[i]);
            break;
    }
}

void row_to_f32_strided(uint32_t dt, const void* src, int64_t stride, float* dst, int64_t n) {
    if (stride == 1) { row_to_f32(dt, src, dst, n); return; }
    /* Scalar gather. vpgatherdd issues one cache line per lane, while scalar loads coalesce in
     * the load queue; and strided operands here are only column slices of a fused projection
     * (docs/OPS.md on `z`), a few rows per launch. */
    for (int64_t i = 0; i < n; ++i) dst[i] = ld_elem(src, dt, i * stride);
}

/* ================================================================== row reductions
 * Fixed summation trees. The accumulators are written out rather than left to the unroller so the
 * summation order is fixed by the source, not the optimiser, and results do not move by an ulp
 * between builds. Four accumulators cover the FMA latency on current cores (`rad-avx-bench
 * --reduce`). */
float row_sumsq(const float* x, int64_t n) {
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t i = 0;
    for (; i + 4 * VF_N <= n; i += 4 * VF_N) {
        vf v0 = vf_loadu(x + i), v1 = vf_loadu(x + i + VF_N);
        vf v2 = vf_loadu(x + i + 2 * VF_N), v3 = vf_loadu(x + i + 3 * VF_N);
        a0 = vf_fma(v0, v0, a0);
        a1 = vf_fma(v1, v1, a1);
        a2 = vf_fma(v2, v2, a2);
        a3 = vf_fma(v3, v3, a3);
    }
    for (; i + VF_N <= n; i += VF_N) { vf v = vf_loadu(x + i); a0 = vf_fma(v, v, a0); }
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; i < n; ++i) s += x[i] * x[i];
    return s;
}

float row_sum(const float* x, int64_t n) {
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t i = 0;
    for (; i + 4 * VF_N <= n; i += 4 * VF_N) {
        a0 = vf_add(a0, vf_loadu(x + i));
        a1 = vf_add(a1, vf_loadu(x + i + VF_N));
        a2 = vf_add(a2, vf_loadu(x + i + 2 * VF_N));
        a3 = vf_add(a3, vf_loadu(x + i + 3 * VF_N));
    }
    for (; i + VF_N <= n; i += VF_N) a0 = vf_add(a0, vf_loadu(x + i));
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; i < n; ++i) s += x[i];
    return s;
}

/* A NaN element is skipped, as libref's std::max(amax, |x|) skips it: vmaxps returns its second
 * operand when either is NaN, so the running maximum goes second. */
float row_amax(const float* x, int64_t n) {
    vf m0 = vf_zero(), m1 = vf_zero();
    int64_t i = 0;
    for (; i + 2 * VF_N <= n; i += 2 * VF_N) {
        m0 = vf_max(vf_abs(vf_loadu(x + i)), m0);
        m1 = vf_max(vf_abs(vf_loadu(x + i + VF_N)), m1);
    }
    for (; i + VF_N <= n; i += VF_N) m0 = vf_max(vf_abs(vf_loadu(x + i)), m0);
    float m = vf_hmax(vf_max(m0, m1));
    for (; i < n; ++i) { float av = x[i] < 0 ? -x[i] : x[i]; if (av > m) m = av; }
    return m;
}

float row_max(const float* x, int64_t n) {
    if (n <= 0) return -3.402823466e+38f;
    vf m0 = vf_set1(x[0]), m1 = vf_set1(x[0]);
    int64_t i = 0;
    for (; i + 2 * VF_N <= n; i += 2 * VF_N) {
        m0 = vf_max(m0, vf_loadu(x + i));
        m1 = vf_max(m1, vf_loadu(x + i + VF_N));
    }
    for (; i + VF_N <= n; i += VF_N) m0 = vf_max(m0, vf_loadu(x + i));
    float m = vf_hmax(vf_max(m0, m1));
    for (; i < n; ++i) if (x[i] > m) m = x[i];
    return m;
}

/* ================================================================== an operand's row, as f32 */
/* Whether a dense row starting at element `base` can be addressed by a byte pointer. A sub-byte
 * dtype starting mid-byte cannot (`byte_at` maps elements 2i and 2i+1 to the same byte), so it
 * takes the element path rather than reading the row shifted by a nibble. Real W4 rows start on a
 * group boundary. */
static inline bool byte_aligned(const RadTensor* t, int64_t base) {
    const int sub = (t->dtype == RAD_I4 || t->dtype == RAD_FP4E2M1) ? 1
                  : (t->dtype == RAD_I2) ? 3 : 0;
    return (base & sub) == 0;
}

const float* in_row(const RadTensor* t, int64_t base, int64_t n, float* scr) {
    const int64_t st = laststride(t);
    if (st == 1 && byte_aligned(t, base))
        return row_f32_in(t->dtype, byte_at(t->data, t->dtype, base), scr, n);
    for (int64_t i = 0; i < n; ++i) scr[i] = ldt(t, base + i * st);
    return scr;
}

void out_row(const RadTensor* t, int64_t base, int64_t n, const float* src) {
    const int64_t st = laststride(t);
    if (st == 1 && byte_aligned(t, base)) {
        row_from_f32(t->dtype, src, (void*)byte_at(t->data, t->dtype, base), n);
        return;
    }
    for (int64_t i = 0; i < n; ++i) stt(t, base + i * st, src[i]);
}

/* ================================================================== exact top-R
 *
 * One step of an exact selection: the next element in descending value order, ties to the lower
 * index (libref's `topk_next`).
 *
 * Eligibility ("not above the last pick, and not the last pick or an equal element before it")
 * splits at the previous pick into two value-only tests, which vectorise as compare + blend + max:
 *
 *     i <= pi    eligible iff  !(v > pv)  and  v != pv
 *     i >  pi    eligible iff  !(v > pv)
 *
 * A second pass finds the index of the maximum and exits at the first match.
 *
 * `!(v > pv)` (the unordered NGT compare) rather than `v <= pv` matches libref, which treats a NaN
 * as eligible. One known difference: libref keeps an eligible NaN that it meets first as the
 * winner, while the vector max returns the largest non-NaN. No path produces NaN logits (a masked
 * logit is -inf, which is handled). */
bool topk_next_f32(const float* v, int64_t n, float* pv, int64_t* pi) {
    const float prev = *pv;
    const int64_t pidx = *pi;

#if AVX_LEVEL == 0
    /* Scalar level: libref's single pass. The two-pass form below only pays when the first pass
     * is vector work; scalar, it is slower than one pass. */
    float best = 0.0f;
    int64_t bi = -1;
    for (int64_t j = 0; j < n; ++j) {
        const float x = v[j];
        if (x > prev) continue;
        if (x == prev && j <= pidx) continue;
        if (bi < 0 || x > best) { best = x; bi = j; }
    }
    if (bi < 0) return false;
    *pv = best;
    *pi = bi;
    return true;
#else
    const float NEG = -INFINITY;
    /* Indices below `split` (at or below `pidx`) must also differ from `prev`. */
    const int64_t split = (pidx + 1 > 0) ? (pidx + 1 < n ? pidx + 1 : n) : 0;
    float best = NEG;
    int64_t i = 0;

    {
        const vf vprev = vf_set1(prev), vneg = vf_set1(NEG);
        vf acc = vf_set1(NEG);
        /* At or below the previous pick: equality excluded. */
        for (; i + VF_N <= split; i += VF_N) {
            const vf x = vf_loadu(v + i);
#if AVX_LEVEL == 3
            const __mmask16 ok = _kand_mask16(_mm512_cmp_ps_mask(x, vprev, _CMP_NGT_UQ),
                                              _mm512_cmp_ps_mask(x, vprev, _CMP_NEQ_UQ));
            acc = vf_max(acc, _mm512_mask_blend_ps(ok, vneg, x));
#else
            const __m256 ok = _mm256_and_ps(_mm256_cmp_ps(x, vprev, _CMP_NGT_UQ),
                                            _mm256_cmp_ps(x, vprev, _CMP_NEQ_UQ));
            acc = vf_max(acc, _mm256_blendv_ps(vneg, x, ok));
#endif
        }
        for (; i < split; ++i) {
            const float x = v[i];
            if (x > prev || x == prev) continue;
            if (x > best) best = x;
        }
        /* Past the previous pick: equality allowed. */
        for (; i + VF_N <= n; i += VF_N) {
            const vf x = vf_loadu(v + i);
#if AVX_LEVEL == 3
            const __mmask16 ok = _mm512_cmp_ps_mask(x, vprev, _CMP_NGT_UQ);
            acc = vf_max(acc, _mm512_mask_blend_ps(ok, vneg, x));
#else
            const __m256 ok = _mm256_cmp_ps(x, vprev, _CMP_NGT_UQ);
            acc = vf_max(acc, _mm256_blendv_ps(vneg, x, ok));
#endif
        }
        const float vm = vf_hmax(acc);
        if (vm > best) best = vm;
    }
    for (; i < n; ++i) {
        const float x = v[i];
        if (x > prev) continue;
        if (x == prev && i <= pidx) continue;
        if (x > best) best = x;
    }

    /* Lowest index holding the maximum (ties to the lower column). Vectorised as well: a scalar
     * scan reads n/2 elements on average and would dominate `sample_argmax`.
     *
     * The range below the previous pick must be scanned first, since the lowest index over the
     * whole row wins. Scanning the vector range first returns the wrong index when the maximum
     * occurs on both sides, which is common at bf16.
     *
     * That low range needs the extra `x != prev` test and is at most pidx+1 long (0 for an argmax,
     * R-1 for the R-th pick), so it stays scalar. Past it, `x == best` implies eligibility, since
     * best <= prev. */
    for (int64_t j = 0; j < split; ++j) {
        const float x = v[j];
        if (x != best || x == prev) continue;
        *pv = best;
        *pi = j;
        return true;
    }
    {
        int64_t j = split;
        const vf vbest = vf_set1(best);
        for (; j + VF_N <= n; j += VF_N) {
            const vf x = vf_loadu(v + j);
#if AVX_LEVEL == 3
            const unsigned msk = (unsigned)_mm512_cmpeq_ps_mask(x, vbest);
#else
            const unsigned msk =
                (unsigned)_mm256_movemask_ps(_mm256_cmp_ps(x, vbest, _CMP_EQ_OQ));
#endif
            if (msk) { *pv = best; *pi = j + __builtin_ctz(msk); return true; }
        }
        for (; j < n; ++j)
            if (v[j] == best) { *pv = best; *pi = j; return true; }
    }
    return false;
#endif
}

}  /* inline namespace */
}  /* namespace avx */

/* ================================================================== the machine's FMA ceiling
 *
 * Measured here, with the kernels' per-level flags, because rad-avx-bench is baseline-compiled
 * and could only measure a scalar chain far below what a GEMM at this level faces.
 *
 * Independent FMA accumulators in registers, no memory traffic. The result is what the core
 * retires at the clock it holds under sustained vector load (below advertised boost on parts with
 * an AVX-512 frequency offset).
 *
 * The accumulator count must fit the register file; a spill inside the loop understates the
 * ceiling. 16 accumulators plus 2 constants fit AVX-512's 32 registers but spill on AVX/AVX2's 16.
 * 10 covers a 4-5 cycle FMA latency on 2 FMA ports and leaves six registers free.
 *
 * The `volatile` sink keeps the compiler from deleting the loop. */
extern "C" double AVX_FN(avx_peak_fma)(int64_t iters) {
    using namespace avx;
#if AVX_LEVEL == 1 || AVX_LEVEL == 2
    enum { NACC = 10 };          /* 16 architectural YMM registers */
#else
    enum { NACC = 16 };          /* 32 ZMM at AVX-512; scalar has no such limit */
#endif
    vf acc[NACC];
    for (int i = 0; i < NACC; ++i) acc[i] = vf_set1((float)(i + 1) * 1e-6f);
    const vf m = vf_set1(1.0000001f), b = vf_set1(1e-7f);
    for (int64_t it = 0; it < iters; ++it)
        for (int i = 0; i < NACC; ++i) acc[i] = vf_fma(acc[i], m, b);
    float sink = 0;
    for (int i = 0; i < NACC; ++i) sink += vf_hsum(acc[i]);
    volatile float keep = sink;
    (void)keep;
    /* 2 flops per FMA, NACC FMAs per iteration, VF_N lanes each. */
    return 2.0 * (double)iters * (double)NACC * (double)VF_N;
}
