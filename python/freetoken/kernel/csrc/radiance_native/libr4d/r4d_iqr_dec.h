/* r4d_iqr_dec.h -- device decoders of the IQR weights (r4d_iqr.h), decoded from their codes where
 * they are read.
 *
 * A decoder is fetch (a lane's raw code dwords of one 16-row x 128-K unit, issued ahead like any
 * operand load) and rows (raw to the int8 values: for each of the tile's 16 rows, the lane's four
 * columns packed in a dword, the operand of one sudot4). A GEMM on the WMMA tiles takes the unit
 * through `frag`: the 16 x 128 bytes through this wave's LDS tile, read back as the int8 tiles'
 * fragment order (r4d_fp8_frag.h), two 64-K supertiles.
 *
 * The forms' tables are read from LDS, staged once a block by `stage`: Table and Trellis entries
 * one byte each, Grid entries one V-vector each.
 *
 * A trellis column's 16 codes are followed in its stream by the first ones of the same column in
 * the unit above, so each row's state is a field of that stream at a fixed position. On libquant's
 * tcq_rb grid (LRB > 0) a value v is rint(v * own / B), own the row's scale and B its block's, the
 * int8 libquant's grid_tcq_decode reads; the ratio is one IEEE division a (row, 128-K block), the
 * host's f32 operations, computed by the lane of its row and read by the others. */
#pragma once
#include "r4d_fp8_wmma.h"
#include "r4d_iqr.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace r4d_iqr_dev {

using namespace r4d_fp8;
using namespace r4d_iqr;

template <uint32_t KIND_, uint32_t B_, uint32_t LV_ = 0, uint32_t L_ = 0, uint32_t LRB_ = 0,
          uint32_t SC_ = kBf16>
struct FormT {
    static constexpr uint32_t KIND = KIND_, B = B_, LV = LV_, L = L_, LRB = LRB_, SC = SC_;
    static constexpr uint32_t word() {
        return r4d_iqr::word(Form{ KIND, B, LV, L, LRB, SC, 0 });
    }
};

/* What a decoder reads besides the codes. */
struct Ctx {
    const uint8_t* W;       /* the weight buffer */
    const uint8_t* S;       /* the scale buffer */
    const int8_t* tab;      /* the form's table, staged in LDS */
    uint32_t N, nkb;
    uint32_t run_tiles;     /* a trellis run's 16-row tiles */
};

/* The LDS tile a unit is transposed through: 16 rows of 128 bytes and 8 of padding, so the sixteen
 * rows' 8-byte fragment reads fall on distinct banks. */
constexpr uint32_t kTrPitch = 136, kTrBytes = 16 * kTrPitch;

/* Bits [s, s + n) of the 64 bits hi:lo, n <= 32 (constants after unrolling). On byte bounds one
 * v_perm, which also zeroes the bytes above; else a funnel shift and a mask. */
__device__ __forceinline__ uint32_t window(uint32_t lo, uint32_t hi, uint32_t s, uint32_t n) {
    if (s % 8u == 0u && n % 8u == 0u && n < 32u) {
        uint32_t sel = 0x0C0C0C0Cu;
        for (uint32_t i = 0; i < n / 8u; ++i)   /* byte s / 8 + i of hi:lo: lo's 0-3, hi's 4-7 */
            sel = (sel & ~(0xFFu << (8u * i))) | ((s / 8u + i) << (8u * i));
        return __builtin_amdgcn_perm(hi, lo, sel);
    }
    const uint32_t m = n == 32u ? 0xFFFFFFFFu : ((1u << n) - 1u);
    return __builtin_amdgcn_alignbit(hi, lo, s) & m;
}

/* `n` <= 32 bits of a little-endian dword stream from bit `o` (constant after unrolling). */
template <uint32_t ND>
__device__ __forceinline__ uint32_t field(const uint32_t (&d)[ND], uint32_t o, uint32_t n) {
    const uint32_t q = o >> 5, s = o & 31u;
    const uint32_t m = n == 32u ? 0xFFFFFFFFu : ((1u << n) - 1u);
    if (s + n <= 32u) return (d[q] >> s) & m;
    return window(d[q], q + 1u < ND ? d[q + 1u] : 0u, s, n);
}

/* Low bytes of two, then of two pairs, into one dword. */
__device__ __forceinline__ uint32_t pack2(uint32_t lo, uint32_t hi) {
    return __builtin_amdgcn_perm(hi, lo, 0x0c0c0400u);
}
__device__ __forceinline__ uint32_t pack4(uint32_t p01, uint32_t p23) {
    return __builtin_amdgcn_perm(p23, p01, 0x05040100u);
}

template <uint32_t N>
__device__ __forceinline__ void load_dw(uint32_t (&d)[N], const uint8_t* p, bool live) {
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) d[i] = 0u;
    if (!live) return;
    uint32_t i = 0;
#pragma unroll
    for (; i + 4u <= N; i += 4u) {
        const uint4 v = *(const uint4*)(p + 4 * i);
        d[i] = v.x; d[i + 1] = v.y; d[i + 2] = v.z; d[i + 3] = v.w;
    }
    if constexpr (N % 4u == 3u) {
        const uint3 v = *(const uint3*)(p + 4 * i);
        d[i] = v.x; d[i + 1] = v.y; d[i + 2] = v.z;
    } else if constexpr (N % 4u == 2u) {
        const uint2 v = *(const uint2*)(p + 4 * i);
        d[i] = v.x; d[i + 1] = v.y;
    } else if constexpr (N % 4u == 1u) {
        d[i] = *(const uint32_t*)(p + 4 * i);
    }
}

/* The bf16 bits of x rounded to nearest even, as the host relayout and libquant round. */
__device__ __forceinline__ uint32_t bf16_bits(float x) {
    const uint32_t u = __float_as_uint(x);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

/* A row's own scale for 128-K block kb, as bf16 bits. */
template <class F>
__device__ __forceinline__ uint32_t own_scale(const Ctx& c, uint32_t r, uint32_t kb) {
    const size_t i = ((size_t)(r / 16u) * c.nkb + kb) * 16u + r % 16u;
    if constexpr (F::SC == kBf16) {
        return ((const uint16_t*)c.S)[i];
    } else {
        constexpr Form f{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
        const float* s2 = (const float*)(c.S + s2_off(f, c.N, (int64_t)c.nkb * 128));
        return bf16_bits(s2[F::SC == kE4m3Row ? r : 0u] * e4m3(c.S[i]));
    }
}
/* The scale the GEMM folds a row's block kb at: its own, or with LRB its block's. */
template <class F>
__device__ __forceinline__ uint32_t fold_scale(const Ctx& c, uint32_t r, uint32_t kb) {
    if constexpr (F::LRB != 0) {
        constexpr Form f{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
        const uint16_t* bp = (const uint16_t*)(c.S + blk_off(f, c.N, (int64_t)c.nkb * 128));
        return bp[(size_t)(r >> F::LRB) * c.nkb + kb];
    } else {
        return own_scale<F>(c, r, kb);
    }
}
/* The f32 a GEMM folds row r's block kb at (with the activation's per-128 scale): the bf16 fold
 * scale, or under one scale a row (kRowF32) the block's shared factor, the row's own then taken
 * in the epilogue (row_scale). */
template <class F>
__device__ __forceinline__ float fold_f32(const Ctx& c, uint32_t r, uint32_t kb) {
    if constexpr (F::SC == kRowF32) {
        constexpr Form f{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
        (void)r;
        return ((const float*)(c.S + cf_off(f, c.N, (int64_t)c.nkb * 128)))[kb];
    } else {
        return bscale(fold_scale<F>(c, r, kb));
    }
}
/* The factor row r's sums take in the epilogue: its kRowF32 scale, else 1. */
template <class F>
__device__ __forceinline__ float row_scale(const Ctx& c, uint32_t r) {
    if constexpr (F::SC == kRowF32) return ((const float*)c.S)[r];
    else { (void)c; (void)r; return 1.0f; }
}

/* The tcq_rb ratio own / B of a row's block kb (1 off that grid). */
template <class F>
__device__ __forceinline__ float rb_ratio(const Ctx& c, uint32_t r, uint32_t kb) {
    if constexpr (F::LRB != 0) {
        const float so = __uint_as_float(own_scale<F>(c, r, kb) << 16);
        const float sb = __uint_as_float(fold_scale<F>(c, r, kb) << 16);
        return sb > 0.0f ? __fdiv_rn(so, sb) : 0.0f;
    } else {
        return 1.0f;
    }
}

/* H16: a tcq_rb trellis's state values staged as f16 (exact for int8), so a value's product with
 * its ratio is one mixed-precision fma instead of a conversion and a multiply; twice the LDS, which
 * the GEMV has to spare and the tiles do not. */
template <class F, bool H16 = false>
struct Dec {
    static constexpr uint32_t B = F::B;
    static constexpr bool TR = is_trellis(F::KIND), GRID = F::KIND == kGrid;
    static constexpr bool MAD = F::KIND == kTrellisMad;   /* values computed, no table */
    static constexpr uint32_t LANE_BITS = GRID ? (64u >> F::LV) * B : 64u * B;
    /* A grid lane segment may start mid-dword: read the dword-aligned window covering it. */
    static constexpr bool ALIGNED = LANE_BITS % 32u == 0u;
    static constexpr uint32_t ND = ALIGNED ? LANE_BITS / 32u : (LANE_BITS + 24u + 31u) / 32u;
    /* A trellis column's continuation: its first L - B bits in the unit above, which lie in one
     * dword of that unit's lane segment for every width (column e starts at bit 16 B e), so a lane
     * reads that dword alone of each of its four columns: p[e]. */
    static constexpr uint32_t CB = 16u * B;
    static constexpr uint32_t NP = TR ? 4u : 1u;
    static constexpr uint32_t pdw(uint32_t e) { return e * CB / 32u; }
    static constexpr uint32_t pbit(uint32_t e) { return e * CB % 32u; }
    struct Raw { uint32_t d[ND]; uint32_t p[NP]; };
    /* A trellis column's stream: its CB code bits, then its continuation, as NW dwords, which
     * its sixteen rows' states are windows of (row q's at bit q B). Built once a unit: a dword of
     * a column that does not start on a dword bound is one funnel shift or v_perm, where reading
     * each state from the lane's codes would join every state that crosses a bound, and every
     * continued one, separately. */
    static constexpr uint32_t NW = TR ? (15u * B + F::L + 31u) / 32u : 1u;
    __device__ static uint32_t colw(const Raw& r, uint32_t e, uint32_t i) {
        const uint32_t a = e * CB + 32u * i, oe = e * CB + CB;
        const uint32_t q = a >> 5, s = a & 31u;
        if (a + 32u <= oe) return s == 0u ? r.d[q] : __builtin_amdgcn_alignbit(r.d[q + 1u], r.d[q], s);
        if (a >= oe) return r.p[e] >> (pbit(e) + (a - oe));
        const uint32_t n0 = oe - a;   /* the column's own last bits, then its continuation */
        if (s + n0 == 32u && pbit(e) == 0u) return __builtin_amdgcn_alignbit(r.p[e], r.d[q], s);
        if (s % 8u == 0u && n0 % 8u == 0u && pbit(e) % 8u == 0u && s + n0 <= 32u) {
            uint32_t sel = 0u;
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t b = j < n0 / 8u ? s / 8u + j : 4u + pbit(e) / 8u + (j - n0 / 8u);
                sel |= (b < 8u ? b : 0x0Cu) << (8u * j);
            }
            return __builtin_amdgcn_perm(r.p[e], r.d[q], sel);
        }
        return field<ND>(r.d, a, n0) | (r.p[e] >> pbit(e)) << n0;
    }
    __device__ static void columns(const Raw& r, uint32_t (&cw)[4][NW]) {
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e)
#pragma unroll
            for (uint32_t i = 0; i < NW; ++i) cw[e][i] = colw(r, e, i);
    }
    /* Bits 16 .. 47 of words i and i + 1 of each column's stream. A state that crosses that
     * bound starts past bit 16 of word i (L is at most 16), so it is a field of this word: one
     * extract a state, where a funnel shift and a mask each would be two. */
    __device__ static void mids(const uint32_t (&cw)[4][NW], uint32_t (&mw)[4][NW]) {
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e)
#pragma unroll
            for (uint32_t i = 0; i < NW; ++i)
                mw[e][i] = i + 1u < NW ? __builtin_amdgcn_alignbit(cw[e][i + 1u], cw[e][i], 16u) : 0u;
    }
    /* Row q's state of column e: L bits from bit q B of the column's stream. */
    __device__ static uint32_t state(const uint32_t (&cw)[4][NW], const uint32_t (&mw)[4][NW],
                                     uint32_t q, uint32_t e) {
        constexpr uint32_t L = F::L;
        static_assert(L <= 16u, "a crossing state within bits 16 .. 47");
        const uint32_t o = q * B, i = o >> 5, s = o & 31u;
        if (s + L <= 32u) return (cw[e][i] >> s) & ((1u << L) - 1u);
        if (s % 8u == 0u && L % 8u == 0u) return window(cw[e][i], cw[e][i + 1u], s, L);
        return (mw[e][i] >> (s - 16u)) & ((1u << L) - 1u);
    }
    static constexpr bool HALF = H16 && TR && !MAD && F::LRB != 0;
    static constexpr uint32_t kTab = F::KIND == kTable ? 256u
                                   : GRID ? (1u << B) << F::LV
                                   : TR && !MAD ? (1u << F::L) * (HALF ? 2u : 1u) : 4u;

    /* Copy the form's table into LDS; the caller then barriers. */
    __device__ static void stage(int8_t* tab, const Ctx& c, uint32_t tid, uint32_t nthr) {
        if constexpr (F::KIND == kTable || GRID || (TR && !MAD)) {
            constexpr Form f{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
            const uint32_t* src = (const uint32_t*)(c.W + aux_off(f, c.N, (int64_t)c.nkb * 128));
            if constexpr (HALF) {
                for (uint32_t i = tid; i < (1u << F::L) / 4u; i += nthr) {
                    const uint32_t q = src[i];
#pragma unroll
                    for (uint32_t j = 0; j < 4; ++j)
                        ((_Float16*)tab)[4 * i + j] = (_Float16)(float)(int8_t)(q >> (8 * j));
                }
            } else {
                for (uint32_t i = tid; i < kTab / 4u; i += nthr) ((uint32_t*)tab)[i] = src[i];
            }
        }
    }

    __device__ static Raw fetch(const Ctx& c, uint32_t rt, uint32_t kb, uint32_t lane, bool live) {
        Raw r;
        constexpr size_t ub = 4u * LANE_BITS;
        const uint8_t* unit = c.W + ((size_t)rt * c.nkb + kb) * ub;
        if constexpr (ALIGNED) {
            load_dw<ND>(r.d, unit + (size_t)lane * (LANE_BITS / 8u), live);
        } else {
            const uint32_t off = lane * (LANE_BITS / 8u);   /* LANE_BITS is whole bytes */
            uint32_t w[ND];
            load_dw<ND>(w, unit + (off & ~3u), live);
            const uint32_t sh = (off & 3u) * 8u;
#pragma unroll
            for (uint32_t i = 0; i < ND; ++i)
                r.d[i] = __builtin_amdgcn_alignbit(i + 1u < ND ? w[i + 1u] : 0u, w[i], sh);
        }
        if constexpr (TR) {
            static_assert(!TR || (pbit(1) + F::L - B <= 32u && pbit(2) + F::L - B <= 32u &&
                                  pbit(3) + F::L - B <= 32u),
                          "a column's continuation in one dword");
            const uint32_t rp = rt % c.run_tiles == 0u ? rt + c.run_tiles - 1u : rt - 1u;
            const uint32_t* up = (const uint32_t*)(c.W + ((size_t)rp * c.nkb + kb) * ub +
                                                   (size_t)lane * (LANE_BITS / 8u));
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) r.p[e] = live ? up[pdw(e)] : 0u;
        } else {
            r.p[0] = 0u;
        }
        return r;
    }

    /* The lane's four values of each of the tile's 16 rows, packed. `ratio` is row (lane & 15)'s
     * tcq_rb ratio for this block (rb_ratio). UNS (a computed trellis off the tcq_rb grid): each
     * value as its unsigned byte sum, the value plus 126, which the caller corrects in its sums. */
    template <bool UNS = false>
    __device__ static void rows(const Raw& r, const Ctx& c, float ratio, uint32_t (&w)[16]) {
        static_assert(!UNS || (MAD && F::LRB == 0), "unsigned values: a computed trellis, one scale a row");
        if constexpr (TR) {
            /* All 64 states, then all 64 lookups, then the values: the LDS reads go out together
             * instead of each waiting on the last's use. */
            uint32_t st[64], cw[4][NW], mw[4][NW];
            columns(r, cw);
            mids(cw, mw);
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q)
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) st[q * 4 + e] = state(cw, mw, q, e);
            [[maybe_unused]] int32_t t[HALF ? 1 : 64];
            [[maybe_unused]] _Float16 th[HALF ? 64 : 1];
            if constexpr (MAD) {
                /* bytesum((st * A + B) & 0x3F3F3F3F) - 126: st < 2^16 and A < 2^24, so the
                 * 24-bit multiply is the full product mod 2^32 */
                constexpr Form f{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
                const uint32_t* ab = (const uint32_t*)(c.W + aux_off(f, c.N, (int64_t)c.nkb * 128));
                const uint32_t A = ab[0], Bc = ab[1];
#pragma unroll
                for (uint32_t i = 0; i < 64; ++i) {
                    /* the byte sum less the code's bias, in the dot's own accumulator */
                    const uint32_t h = (__umul24(st[i], A) + Bc) & 0x3F3F3F3Fu;
                    t[i] = __builtin_amdgcn_sudot4(false, (int)h, false, 0x01010101, UNS ? 0 : -126, false);
                }
            } else {
#pragma unroll
                for (uint32_t i = 0; i < 64; ++i) {
                    if constexpr (HALF) th[i] = ((const _Float16*)c.tab)[st[i]];
                    else t[i] = c.tab[st[i]];
                }
            }
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) {
                uint32_t v[4];
                if constexpr (F::LRB != 0) {
                    /* rint(t * rq) as the low byte of t * rq + 1.5 * 2^23: the add rounds the f32
                     * product to an integer, nearest even as rintf does, and |t * rq| <= 127
                     * leaves it in the mantissa's low bits, two's complement in its low byte. */
                    const float rq = __builtin_bit_cast(
                        float, __builtin_amdgcn_readlane(__float_as_int(ratio), 15 - q));
#pragma unroll
                    for (uint32_t e = 0; e < 4; ++e) {
                        /* the f16 value times rq, rounded once: the f32 product of the int8; a
                         * computed code rounds the exact product, one fma */
                        if constexpr (MAD) {
                            v[e] = __float_as_uint(__builtin_fmaf((float)t[q * 4 + e], rq, 12582912.0f));
                        } else {
                            const float pr = HALF ? __builtin_fmaf((float)th[q * 4 + e], rq, 0.0f)
                                                  : __fmul_rn((float)t[HALF ? 0 : q * 4 + e], rq);
                            v[e] = __float_as_uint(__fadd_rn(pr, 12582912.0f));
                        }
                    }
                } else {
#pragma unroll
                    for (uint32_t e = 0; e < 4; ++e) v[e] = (uint32_t)t[HALF ? 0 : q * 4 + e];
                }
                w[15u - q] = pack4(pack2(v[0], v[1]), pack2(v[2], v[3]));
            }
        } else if constexpr (GRID) {
            constexpr uint32_t V = 1u << F::LV, PER = 4u / V;
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) {
                if constexpr (V == 4u) {
                    w[q] = *(const uint32_t*)(c.tab + field<ND>(r.d, q * B, B) * 4u);
                } else {
                    const uint32_t g0 = *(const uint16_t*)(c.tab + field<ND>(r.d, (q * PER) * B, B) * 2u);
                    const uint32_t g1 = *(const uint16_t*)(c.tab + field<ND>(r.d, (q * PER + 1) * B, B) * 2u);
                    w[q] = g0 | g1 << 16;
                }
            }
        } else {
            (void)ratio;
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) {
                uint32_t v[4];
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) {
                    const uint32_t code = field<ND>(r.d, (q * 4u + e) * B, B);
                    if constexpr (F::KIND == kTable) v[e] = (uint8_t)c.tab[code];
                    else if constexpr (F::KIND == kZero) v[e] = code;
                    else v[e] = (uint32_t)((int32_t)(code << (32u - B)) >> (32u - B));
                }
                w[q] = pack4(pack2(v[0], v[1]), pack2(v[2], v[3]));
            }
        }
    }

    /* A computed trellis's masked hashes, one a value: h[q][e] for the lane's column e of row q,
     * whose four bytes sum to the value plus 126. A dot of h with an activation byte repeated in
     * each byte is the value plus 126 times the activation, the sum and the product in one. */
    __device__ static void hashes(const Raw& r, const Ctx& c, uint32_t (&h)[16][4]) {
        static_assert(MAD, "hashes: a computed trellis");
        constexpr Form f{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
        const uint32_t* ab = (const uint32_t*)(c.W + aux_off(f, c.N, (int64_t)c.nkb * 128));
        const uint32_t A = ab[0], Bc = ab[1];
        uint32_t cw[4][NW], mw[4][NW];
        columns(r, cw);
        mids(cw, mw);
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q)
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) h[15u - q][e] = (__umul24(state(cw, mw, q, e), A) + Bc) & 0x3F3F3F3Fu;
    }

    /* Zero: the zero of row (rt*16 + q) for block kb is subtracted bytewise after `rows`. */
    __device__ static void unzero(const Ctx& c, uint32_t rt, uint32_t kb, uint32_t (&w)[16]) {
        if constexpr (F::KIND == kZero) {
            constexpr Form f{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
            const uint8_t* zp = c.W + aux_off(f, c.N, (int64_t)c.nkb * 128) +
                                ((size_t)rt * c.nkb + kb) * 16u;
            const uint4 z4 = *(const uint4*)zp;   /* the tile's 16 rows, one byte each */
            const uint32_t zd[4] = { z4.x, z4.y, z4.z, z4.w };
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) {
                const uint32_t z = (zd[q / 4u] >> (8u * (q % 4u))) & 0xFFu;
                /* codes and zero are below 128, so bytewise subtraction borrows nothing across
                 * bytes once 0x80 is added to each code byte and taken off the result */
                w[q] = ((w[q] | 0x80808080u) - z * 0x01010101u) ^ 0x80808080u;
            }
        } else {
            (void)c; (void)rt; (void)kb; (void)w;
        }
    }
};

/* The unit's int8 tile in fragment order, two 64-K supertiles, through this wave's LDS tile `tr`.
 * A wave's LDS accesses complete in order, so only the compiler must not move the reads above the
 * writes; the second fence keeps the next unit's writes below these reads. */
__device__ __forceinline__ void frag(const uint32_t (&w)[16], uint8_t* tr, uint32_t lane,
                                     Stage8 (&s)[2]) {
#pragma unroll
    for (uint32_t q = 0; q < 16; ++q) *(uint32_t*)(tr + q * kTrPitch + 4u * lane) = w[q];
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
    const uint8_t* row = tr + (lane & 15u) * kTrPitch + (lane >> 4) * 8u;
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
        uint32_t o[8];
#pragma unroll
        for (uint32_t t = 0; t < 4; ++t) {
            const uint2 g = *(const uint2*)(row + h * 64u + t * 16u);
            o[2 * t] = g.x;
            o[2 * t + 1] = g.y;
        }
        s[h].v = make_uint4(o[0], o[1], o[2], o[3]);
        s[h].w = make_uint4(o[4], o[5], o[6], o[7]);
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
}

}  // namespace r4d_iqr_dev
