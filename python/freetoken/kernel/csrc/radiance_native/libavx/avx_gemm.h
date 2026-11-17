/* avx_gemm.h -- how the GEMM family reads its operands, and the two large-M drivers.
 *
 * Shared by avx_gemm.cpp (the ops, the decode-shaped driver) and avx_gemm_tile.cpp (the tile
 * drivers). Compiled once per ISA level like every kernel source, so everything here lives in the
 * level's inline namespace (avx_vec.h): the four objects never share a definition.
 *
 * The readings are copied from libref/ref_gemm.h and must match it: where an operand's (row, col)
 * lives, and which scale covers it. Note `row_stride` (avx_common.h): a rank-3 [tokens, heads,
 * width] activation read as [tokens, heads*width] has its row stride on the tokens axis, not
 * stride[rank-2].
 */
#ifndef RAD_AVX_GEMM_H
#define RAD_AVX_GEMM_H

#include "avx_vec.h"
#include "avx_common.h"
#include "avx_planes.h"

namespace avx {
inline namespace AVX_SUFFIX {

/* A weight whose rows are produced by a decoder instead of read as stored: codes and their scales
 * joined (avx_planes.h), widened to f32 with the scales folded in. Called from parallel regions;
 * an implementation only writes `dst`.
 *
 *   row    the whole row, `K` columns
 *   cols   columns [k0, k1) into dst[0, k1 - k0); k0 is a multiple of align(), and k1 is too or
 *          is the row's end */
struct RowDecoder {
    virtual void row(int64_t r, float* dst) const = 0;
    virtual void cols(int64_t r, int64_t k0, int64_t k1, float* dst) const = 0;
    virtual int64_t align() const = 0;
    /* The stored bytes cols(r, k0, k1) reads first (the codes), for a prefetch ahead of it. */
    virtual void codes(int64_t r, int64_t k0, int64_t k1, const void** p, int64_t* bytes) const = 0;
};

struct Mat {
    const RadTensor* t;
    int64_t rs, cs;
    const RowDecoder* dec = nullptr;
};

static inline Mat as_mat(const RadTensor* t, int64_t cols) {
    Mat m{ t, cols, 1 };
    if (t->rank < 2) return m;
    m.cs = t->stride[t->rank - 1];
    m.rs = row_stride(t, cols);
    return m;
}

/* n E8M0 exponents as f32, rad_e8m0_to_f32's values: e << 23, with 0 the subnormal 2^-127 and
 * 0xFF a NaN. */
static inline void e8m0_to_f32(const uint8_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL == 3
    for (; i + 16 <= n; i += 16) {
        const __m512i e = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(src + i)));
        __m512i u = _mm512_slli_epi32(e, 23);
        u = _mm512_mask_mov_epi32(u, _mm512_cmpeq_epi32_mask(e, _mm512_setzero_si512()),
                                  _mm512_set1_epi32(0x00400000));
        u = _mm512_mask_mov_epi32(u, _mm512_cmpeq_epi32_mask(e, _mm512_set1_epi32(0xff)),
                                  _mm512_set1_epi32(0x7fc00000));
        _mm512_storeu_si512((void*)(dst + i), u);
    }
#elif AVX_LEVEL == 2
    for (; i + 8 <= n; i += 8) {
        const __m256i e = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(src + i)));
        __m256i u = _mm256_slli_epi32(e, 23);
        u = _mm256_blendv_epi8(u, _mm256_set1_epi32(0x00400000),
                               _mm256_cmpeq_epi32(e, _mm256_setzero_si256()));
        u = _mm256_blendv_epi8(u, _mm256_set1_epi32(0x7fc00000),
                               _mm256_cmpeq_epi32(e, _mm256_set1_epi32(0xff)));
        _mm256_storeu_si256((__m256i*)(dst + i), u);
    }
#endif
    for (; i < n; ++i) dst[i] = rad_e8m0_to_f32(src[i]);
}

/* A quantised operand's scale stream, as defined in libref/ref_gemm.h. `rblk` makes a
 * DeepSeek-style [ceil(N/128)][ceil(K/128)] block grid readable: without it a 17408-row weight's
 * 136-row scale plane would be read as 17408 rows. MXFP4 weights may carry e8m0 (u8) scales. */
struct Scales {
    const void* p;
    uint32_t dt;
    int64_t rs, cs, group, rblk;
    bool e8m0;
    inline float grp(int64_t row, int64_t g) const {
        const int64_t i = (rblk > 1 ? row / rblk : row) * rs + g * cs;
        if (e8m0) {
            const uint8_t e = ((const uint8_t*)p)[i];
            return e == 0xff ? std::nanf("") : std::ldexp(1.0f, (int)e - 127);
        }
        return rad_load_f32(p, dt, i);
    }
    /* Groups [g0, g0 + n) of `row` into dst, the same values grp() returns, with the dtype
     * resolved once rather than per scale. */
    inline void row_groups(int64_t row, int64_t g0, int64_t n, float* dst) const {
        const int64_t base = (rblk > 1 ? row / rblk : row) * rs + g0 * cs;
        const uint8_t* b = (const uint8_t*)p;
        if (cs == 1) {
            switch (e8m0 ? (uint32_t)RAD_E8M0 : dt) {
                case RAD_E8M0: e8m0_to_f32(b + base, dst, n); return;
                case RAD_BF16: cvt_bf16_to_f32((const uint16_t*)p + base, dst, n); return;
                case RAD_F16:  cvt_f16_to_f32((const uint16_t*)p + base, dst, n); return;
                case RAD_F32:  std::memcpy(dst, (const float*)p + base, (size_t)n * 4); return;
                default: break;
            }
        }
        if (e8m0 || dt == RAD_E8M0) {
            for (int64_t g = 0; g < n; ++g) dst[g] = rad_e8m0_to_f32(b[base + g * cs]);
        } else if (dt == RAD_BF16) {
            for (int64_t g = 0; g < n; ++g)
                dst[g] = rad_bf16_to_f32(((const uint16_t*)p)[base + g * cs]);
        } else if (dt == RAD_F16) {
            for (int64_t g = 0; g < n; ++g)
                dst[g] = rad_f16_to_f32(((const uint16_t*)p)[base + g * cs]);
        } else if (dt == RAD_F32) {
            for (int64_t g = 0; g < n; ++g) dst[g] = ((const float*)p)[base + g * cs];
        } else {
            for (int64_t g = 0; g < n; ++g) dst[g] = rad_load_f32(p, dt, base + g * cs);
        }
    }
};

static inline Scales make_scales(const RadTensor* s, int64_t rows, int64_t K, bool mxfp4_weight) {
    Scales q{};
    q.rblk = 1;
    if (!s || !s->data || rows <= 0 || numel(s) <= 0) {
        q.p = nullptr; q.group = K; q.dt = RAD_F32; return q;
    }
    q.p = s->data;
    q.dt = s->dtype;
    /* As libref: a rank-2 stream with fewer rows than the operand is a block grid, and a flat
     * stream shorter than the row count is one scale per block of rows. */
    int64_t groups = 0;
    if (s->rank >= 2) {
        const int64_t srows = s->shape[s->rank - 2];
        q.rblk = srows >= rows ? 1 : grid_block(rows, srows);
        groups = s->shape[s->rank - 1];
        q.rs = s->stride[s->rank - 2];
        q.cs = s->stride[s->rank - 1];
    } else {
        const int64_t nel = numel(s);
        if (nel >= rows) { groups = nel / rows; q.rs = groups; }
        else             { groups = 1; q.rblk = grid_block(rows, nel); q.rs = 1; }
        q.cs = 1;
    }
    q.group = grid_block(K, groups);
    q.e8m0 = mxfp4_weight && (s->dtype == RAD_U8);
    return q;
}

/* Columns [k0, k0 + n) of row r of a plain operand as raw f32 codes: the row converter where the
 * columns are contiguous and start on a byte, element by element otherwise. */
static inline void widen_cols(const Mat& m, int64_t r, int64_t k0, int64_t n, float* dst) {
    const RadTensor* t = m.t;
    const int64_t base = r * m.rs + k0 * m.cs;
    if (m.cs == 1) {
        const int sub = (t->dtype == RAD_I4 || t->dtype == RAD_FP4E2M1) ? 1
                      : (t->dtype == RAD_I2) ? 3 : 0;
        if (!sub || (base & sub) == 0) {
            row_to_f32(t->dtype, byte_at(t->data, t->dtype, base), dst, n);
            return;
        }
    }
    for (int64_t k = 0; k < n; ++k) dst[k] = ld_elem(t->data, t->dtype, base + k * m.cs);
}

/* ================================================================== the tile drivers
 * (avx_gemm_tile.cpp). Y = A B^T (+ bias), the product gemm_drive forms, at any M. Each returns
 * false when it does not apply, and the caller then computes all of Y another way.
 *
 *   gemm_q_bf16    AVX512_BF16 parts: weight codes exact in bf16 (E4M3, int8, int4, e2m1) against
 *                  an E4M3, int8 or bf16 activation; the group structure kept, as libref's
 *   gemm_tile_f32  any operands, both scales folded into the f32 operands (it may have written Y
 *                  when it returns false: an E8M0 scale it cannot fold is found as it runs)
 *
 * SA must be coarse (no finer than the weight's group); gemm_drive takes the other case first. */
bool gemm_q_bf16(const Mat& ma, const Mat& mb, const Scales* SA, const Scales* SB,
                 const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N, int64_t K);
bool gemm_tile_f32(const Mat& ma, const Mat& mb, const Scales* SA, const Scales* SB,
                   const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N, int64_t K);

/* A batch of independent products in one parallel region, as a grouped GEMM's experts are. Job j
 * computes Y[y0 + i][yc0 + c] = sum_k A[a0 + i][k] * B[b0 + c][k] (+ bias[c]) for i < M, c < N,
 * k < K: rows of one batch activation, columns of its own weight. Batch row r is A's row
 * arow[r] (or r with no map), and a negative entry a zero row. Every job's weight is the same
 * format at the same group (SB.p null: unscaled); SA scales A's rows as libref's Scales. `ycols`
 * is Y's logical width, which sets its row pitch. With `zero` (a zero-point grid's u8 plane, one
 * per row and group of SB) the weight is (code - zero) * scale over u4 or u5 codes, which only the
 * bf16 grouped driver takes. */
struct TileJob {
    Mat mb;
    Scales SB;
    int64_t b0, a0, M, N, K, y0, yc0;
    const RadTensor* bias;
    avxp::JPlane zero;
};
bool gemm_q_bf16_jobs(const Mat& ma, const int64_t* arow, int64_t arows, const Scales* SA,
                      const TileJob* jobs, int64_t njobs, const RadTensor* Y, int64_t ycols);
bool gemm_tile_f32_jobs(const Mat& ma, const int64_t* arow, int64_t arows, const Scales* SA,
                        const TileJob* jobs, int64_t njobs, const RadTensor* Y, int64_t ycols);

}  /* inline namespace AVX_SUFFIX */
}  /* namespace avx */

#endif /* RAD_AVX_GEMM_H */
