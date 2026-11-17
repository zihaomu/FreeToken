/* r4d_hsimd_kernels.cpp -- the per-level bodies behind r4d_hsimd.h. Compiled once per level with
 * R4D_HSIMD_LEVEL=0..4 and matching -m flags (cmake/RadSimdLevels.cmake); names carry the level's
 * suffix and r4d_hsimd.cpp holds the table. Every body moves bits and is exact: each level gives
 * the scalar level's bytes on every input.
 *
 * The trellis tile is three passes over a 16-row x 128-K unit, all in L1: each row's 128 B-bit
 * codes are unpacked to bytes, the 16 x 128 byte matrix is transposed with its rows reversed (the
 * stored order is column-major, row 15 first), and the 2048 bytes are packed back to B bits, which
 * is the unit: lane l's segment is columns 4l .. 4l + 3, 16 codes each, so the transposed matrix's
 * rows in order are the segments in order.
 *
 * The w2a8 tile moves 2-bit codes within a row: each stored dword is four bytes of the row's unit
 * gathered and transposed as a 4 x 4 matrix of 2-bit fields, a shuffle and two delta swaps.
 */
#include "r4d_hsimd.h"

#include <cstring>

#if R4D_HSIMD_LEVEL >= 1
#include <immintrin.h>
#endif

#if   R4D_HSIMD_LEVEL == 0
#define R4D_HSIMD_SUFFIX _sc
#elif R4D_HSIMD_LEVEL == 1
#define R4D_HSIMD_SUFFIX _s4
#elif R4D_HSIMD_LEVEL == 2
#define R4D_HSIMD_SUFFIX _v1
#elif R4D_HSIMD_LEVEL == 3
#define R4D_HSIMD_SUFFIX _v2
#elif R4D_HSIMD_LEVEL == 4
#define R4D_HSIMD_SUFFIX _v3
#else
#error "R4D_HSIMD_LEVEL out of range"
#endif

#define R4D_HSIMD_CAT2(a, b) a##b
#define R4D_HSIMD_CAT(a, b)  R4D_HSIMD_CAT2(a, b)
#define R4D_HSIMD_FN(name)   R4D_HSIMD_CAT(r4d_hsimd_##name, R4D_HSIMD_SUFFIX)

namespace {

/* `n` bytes from L1 scratch to a destination the host never reads back (a pinned staging ring or a
 * pool slot the card reads), with streaming stores where both are aligned: an ordinary store first
 * reads each destination line in, which during a restore is DRAM traffic beside the drive's DMA.
 * The caller fences once (stream_fence) before the bytes are handed on. */
inline bool stream_copy(uint8_t* d, const uint8_t* s, int64_t n) {
#if R4D_HSIMD_LEVEL == 4
    if ((((uintptr_t)d | (uintptr_t)s | (uintptr_t)n) & 63) == 0) {
        for (int64_t i = 0; i < n; i += 64)
            _mm512_stream_si512((__m512i*)(d + i), _mm512_load_si512((const __m512i*)(s + i)));
        return true;
    }
#elif R4D_HSIMD_LEVEL == 3
    if ((((uintptr_t)d | (uintptr_t)s | (uintptr_t)n) & 31) == 0) {
        for (int64_t i = 0; i < n; i += 32)
            _mm256_stream_si256((__m256i*)(d + i), _mm256_load_si256((const __m256i*)(s + i)));
        return true;
    }
#elif R4D_HSIMD_LEVEL >= 1
    if ((((uintptr_t)d | (uintptr_t)s | (uintptr_t)n) & 15) == 0) {
        for (int64_t i = 0; i < n; i += 16)
            _mm_stream_si128((__m128i*)(d + i), _mm_load_si128((const __m128i*)(s + i)));
        return true;
    }
#endif
    std::memcpy(d, s, (size_t)n);
    return false;
}

/* Orders the streaming stores before whatever publishes the bytes. */
inline void stream_fence(bool streamed) {
#if R4D_HSIMD_LEVEL >= 1
    if (streamed) _mm_sfence();
#else
    (void)streamed;
#endif
}

/* ---- the MoE w4 nibble shuffle ------------------------------------------------------------- */

/* Scalar, two dwords at a time: forward swaps each dword's middle bytes, then each half's middle
 * nibbles; back does the steps in reverse. */
inline uint64_t mw_bytes(uint64_t x) {
    return (x & 0xFF0000FFFF0000FFull) | ((x & 0x0000FF000000FF00ull) << 8) |
           ((x >> 8) & 0x0000FF000000FF00ull);
}
inline uint64_t mw_nibbles(uint64_t x) {
    return (x & 0xF00FF00FF00FF00Full) | ((x & 0x00F000F000F000F0ull) << 4) |
           ((x >> 4) & 0x00F000F000F000F0ull);
}
void mw_scalar(const uint8_t* s, uint8_t* d, int64_t bytes, bool fwd) {
    int64_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t x;
        std::memcpy(&x, s + i, 8);
        x = fwd ? mw_nibbles(mw_bytes(x)) : mw_bytes(mw_nibbles(x));
        std::memcpy(d + i, &x, 8);
    }
    if (i < bytes) {   /* one dword left; no step crosses dwords */
        uint32_t w;
        std::memcpy(&w, s + i, 4);
        uint64_t x = w;
        x = fwd ? mw_nibbles(mw_bytes(x)) : mw_bytes(mw_nibbles(x));
        w = (uint32_t)x;
        std::memcpy(d + i, &w, 4);
    }
}

/* Vector form: output byte b of a dword is a low nibble of input byte A[b] and a high nibble of
 * input byte B[b], each from the input byte's low or high half. Forward, byte b takes element b
 * low and b + 4 high: A = (0, 0, 1, 1), B = (2, 2, 3, 3), odd bytes from the high halves. Back,
 * A = (0, 2, 0, 2), B = (1, 3, 1, 3), bytes 2 and 3 from the high halves. */
#if R4D_HSIMD_LEVEL >= 1
inline int8_t mw_idx(int k, bool fwd) {   /* A's byte for output byte k of a 16-byte lane */
    const int dw = k & ~3, b = k & 3;
    return (int8_t)(dw + (fwd ? (b >> 1) : (b & 1) * 2));
}
/* Whether a permute streams its output (stream_copy says why): a whole weight into its pinned or
 * pool destination does, aligned to the vector. A row into scratch the caller reads straight back
 * (moew4_frag_permute, moew5_row: a few KiB) keeps ordinary stores, which leave it in cache. */
inline bool mw_stream(const uint8_t* d, int64_t bytes, int vec) {
    return bytes >= ((int64_t)64 << 10) && ((uintptr_t)d & (uintptr_t)(vec - 1)) == 0;
}
#endif

#if R4D_HSIMD_LEVEL == 1 || R4D_HSIMD_LEVEL == 2
void mw_vec(const uint8_t* s, uint8_t* d, int64_t bytes, bool fwd) {
    alignas(16) int8_t t[16];
    for (int k = 0; k < 16; ++k) t[k] = mw_idx(k, fwd);
    const __m128i ia = _mm_load_si128((const __m128i*)t);
    const __m128i ib = _mm_add_epi8(ia, _mm_set1_epi8(fwd ? 2 : 1));
    const __m128i m = fwd ? _mm_set1_epi16((short)0xFF00) : _mm_set1_epi32((int)0xFFFF0000);
    const __m128i lo = _mm_set1_epi8(0x0F), hi = _mm_set1_epi8((char)0xF0);
    const bool nt = mw_stream(d, bytes, 16);
    int64_t i = 0;
    for (; i + 16 <= bytes; i += 16) {
        const __m128i x = _mm_loadu_si128((const __m128i*)(s + i));
        const __m128i a = _mm_shuffle_epi8(x, ia), b = _mm_shuffle_epi8(x, ib);
        const __m128i a2 = _mm_blendv_epi8(a, _mm_srli_epi16(a, 4), m);
        const __m128i b2 = _mm_blendv_epi8(_mm_slli_epi16(b, 4), b, m);
        const __m128i v = _mm_or_si128(_mm_and_si128(a2, lo), _mm_and_si128(b2, hi));
        if (nt) _mm_stream_si128((__m128i*)(d + i), v);
        else    _mm_storeu_si128((__m128i*)(d + i), v);
    }
    stream_fence(nt);
    mw_scalar(s + i, d + i, bytes - i, fwd);
}
#elif R4D_HSIMD_LEVEL == 3
void mw_vec(const uint8_t* s, uint8_t* d, int64_t bytes, bool fwd) {
    alignas(32) int8_t t[32];
    for (int k = 0; k < 32; ++k) t[k] = mw_idx(k & 15, fwd);
    const __m256i ia = _mm256_load_si256((const __m256i*)t);
    const __m256i ib = _mm256_add_epi8(ia, _mm256_set1_epi8(fwd ? 2 : 1));
    const __m256i m = fwd ? _mm256_set1_epi16((short)0xFF00) : _mm256_set1_epi32((int)0xFFFF0000);
    const __m256i lo = _mm256_set1_epi8(0x0F), hi = _mm256_set1_epi8((char)0xF0);
    const bool nt = mw_stream(d, bytes, 32);
    int64_t i = 0;
    for (; i + 32 <= bytes; i += 32) {
        const __m256i x = _mm256_loadu_si256((const __m256i*)(s + i));
        const __m256i a = _mm256_shuffle_epi8(x, ia), b = _mm256_shuffle_epi8(x, ib);
        const __m256i a2 = _mm256_blendv_epi8(a, _mm256_srli_epi16(a, 4), m);
        const __m256i b2 = _mm256_blendv_epi8(_mm256_slli_epi16(b, 4), b, m);
        const __m256i v = _mm256_or_si256(_mm256_and_si256(a2, lo), _mm256_and_si256(b2, hi));
        if (nt) _mm256_stream_si256((__m256i*)(d + i), v);
        else    _mm256_storeu_si256((__m256i*)(d + i), v);
    }
    stream_fence(nt);
    mw_scalar(s + i, d + i, bytes - i, fwd);
}
#elif R4D_HSIMD_LEVEL == 4
void mw_vec(const uint8_t* s, uint8_t* d, int64_t bytes, bool fwd) {
    alignas(64) int8_t t[64];
    for (int k = 0; k < 64; ++k) t[k] = mw_idx(k & 15, fwd);
    const __m512i ia = _mm512_load_si512(t);
    const __m512i ib = _mm512_add_epi8(ia, _mm512_set1_epi8(fwd ? 2 : 1));
    const __mmask64 m = fwd ? 0xAAAAAAAAAAAAAAAAull : 0xCCCCCCCCCCCCCCCCull;
    const __m512i lo = _mm512_set1_epi8(0x0F);
    const bool nt = mw_stream(d, bytes, 64);
    int64_t i = 0;
    for (; i + 64 <= bytes; i += 64) {
        const __m512i x = _mm512_loadu_si512(s + i);
        const __m512i a = _mm512_shuffle_epi8(x, ia), b = _mm512_shuffle_epi8(x, ib);
        const __m512i a2 = _mm512_mask_mov_epi8(a, m, _mm512_srli_epi16(a, 4));
        const __m512i b2 = _mm512_mask_mov_epi8(_mm512_slli_epi16(b, 4), m, b);
        /* lo ? a2 : b2, bit by bit */
        const __m512i v = _mm512_ternarylogic_epi64(lo, a2, b2, 0xCA);
        if (nt) _mm512_stream_si512((__m512i*)(d + i), v);
        else    _mm512_storeu_si512(d + i, v);
    }
    stream_fence(nt);
    mw_scalar(s + i, d + i, bytes - i, fwd);
}
#endif

/* ---- the trellis tile ------------------------------------------------------------------------ */

/* A row's codes for one unit: 128 codes, 16 B bytes. */
constexpr int kCodes = 128;

#if R4D_HSIMD_LEVEL == 0
/* 128 codes of B bits from `s` to one byte each. */
void unpack_row(const uint8_t* s, int B, uint8_t* u) {
    const uint64_t m = (1u << B) - 1u;
    for (int g = 0; g < kCodes / 8; ++g) {
        uint64_t x = 0;
        std::memcpy(&x, s + g * B, (size_t)B);
        for (int i = 0; i < 8; ++i) u[g * 8 + i] = (uint8_t)((x >> (i * B)) & m);
    }
}
/* T[k][q] = U[15 - q][k]. */
void transpose_rev(const uint8_t (*U)[kCodes], uint8_t* T) {
    for (int k = 0; k < kCodes; ++k)
        for (int q = 0; q < 16; ++q) T[k * 16 + q] = U[15 - q][k];
}
/* `n` bytes of B-bit values to the stream, eight values (B bytes) at a time. */
void pack(const uint8_t* t, int n, int B, uint8_t* out) {
    for (int g = 0; g < n / 8; ++g) {
        uint64_t x = 0;
        for (int i = 0; i < 8; ++i) x |= (uint64_t)t[g * 8 + i] << (i * B);
        std::memcpy(out + g * B, &x, (size_t)B);
    }
}
#else
/* 128-bit lanes, shared by every vector level: one lane unpacks 16 codes (2B bytes) or packs 16
 * bytes. Unpacking spreads the lane's two B-byte halves to its 64-bit halves, then halves each
 * width in turn (64 -> 32 -> 16 -> 8 bits a field); packing runs the same steps backwards. The
 * shift counts follow B, so they go in a register. */
struct Lanes {
    __m128i up_idx, pk_idx, c4, c2, c1, c32, c16, c8;
    __m128i m4, m2, m1, k32, k16, k8;
    explicit Lanes(int B) {
        alignas(16) int8_t u[16], p[16];
        for (int i = 0; i < 16; ++i) {
            u[i] = (int8_t)(i < 8 ? (i < B ? i : -128) : (i - 8 < B ? B + (i - 8) : -128));
            p[i] = (int8_t)(i < B ? i : i < 2 * B ? 8 + (i - B) : -128);
        }
        up_idx = _mm_load_si128((const __m128i*)u);
        pk_idx = _mm_load_si128((const __m128i*)p);
        c4 = _mm_cvtsi32_si128(4 * B);
        c2 = _mm_cvtsi32_si128(2 * B);
        c1 = _mm_cvtsi32_si128(B);
        c32 = _mm_cvtsi32_si128(32);
        c16 = _mm_cvtsi32_si128(16);
        c8 = _mm_cvtsi32_si128(8);
        m4 = _mm_set1_epi64x((long long)((1ull << (4 * B)) - 1ull));
        m2 = _mm_set1_epi32((int)((1u << (2 * B)) - 1u));
        m1 = _mm_set1_epi16((short)((1u << B) - 1u));
        k32 = _mm_set1_epi64x(0xFFFFFFFFll);
        k16 = _mm_set1_epi32(0xFFFF);
        k8 = _mm_set1_epi16(0xFF);
    }
};
#endif

#if R4D_HSIMD_LEVEL == 1 || R4D_HSIMD_LEVEL == 2
inline __m128i unpack16(const Lanes& L, __m128i x) {
    x = _mm_shuffle_epi8(x, L.up_idx);
    x = _mm_or_si128(_mm_and_si128(x, L.m4), _mm_sll_epi64(_mm_srl_epi64(x, L.c4), L.c32));
    x = _mm_or_si128(_mm_and_si128(x, L.m2), _mm_sll_epi32(_mm_srl_epi32(x, L.c2), L.c16));
    return _mm_or_si128(_mm_and_si128(x, L.m1), _mm_sll_epi16(_mm_srl_epi16(x, L.c1), L.c8));
}
inline __m128i pack16(const Lanes& L, __m128i x) {
    x = _mm_or_si128(_mm_and_si128(x, L.k8), _mm_sll_epi16(_mm_srl_epi16(x, L.c8), L.c1));
    x = _mm_or_si128(_mm_and_si128(x, L.k16), _mm_sll_epi32(_mm_srl_epi32(x, L.c16), L.c2));
    x = _mm_or_si128(_mm_and_si128(x, L.k32), _mm_sll_epi64(_mm_srl_epi64(x, L.c32), L.c4));
    return _mm_shuffle_epi8(x, L.pk_idx);
}
/* `s` may be read 16 bytes past each group: the caller pads where the plane ends. */
void unpack_row(const Lanes& L, const uint8_t* s, int B, uint8_t* u) {
    for (int g = 0; g < kCodes / 16; ++g)
        _mm_storeu_si128((__m128i*)(u + g * 16),
                         unpack16(L, _mm_loadu_si128((const __m128i*)(s + g * 2 * B))));
}
/* Four rounds of interleaving rows i and i + 8 transpose a 16 x 16 byte block: each round rotates
 * an element's 8-bit (row, column) address left by one bit. */
void transpose_rev(const uint8_t (*U)[kCodes], uint8_t* T) {
    for (int c = 0; c < kCodes / 16; ++c) {
        __m128i x[16], y[16];
        for (int q = 0; q < 16; ++q) x[q] = _mm_loadu_si128((const __m128i*)(U[15 - q] + 16 * c));
        for (int round = 0; round < 4; ++round) {
            for (int i = 0; i < 8; ++i) {
                y[2 * i] = _mm_unpacklo_epi8(x[i], x[i + 8]);
                y[2 * i + 1] = _mm_unpackhi_epi8(x[i], x[i + 8]);
            }
            for (int i = 0; i < 16; ++i) x[i] = y[i];
        }
        for (int j = 0; j < 16; ++j) _mm_storeu_si128((__m128i*)(T + (16 * c + j) * 16), x[j]);
    }
}
/* Writes up to 16 bytes past the 2B n / 16 the stream holds: `out` has room. */
void pack(const Lanes& L, const uint8_t* t, int n, int B, uint8_t* out) {
    for (int g = 0; g < n / 16; ++g)
        _mm_storeu_si128((__m128i*)(out + g * 2 * B), pack16(L, _mm_loadu_si128((const __m128i*)(t + g * 16))));
}
#elif R4D_HSIMD_LEVEL == 3
inline __m256i bc(__m128i x) { return _mm256_broadcastsi128_si256(x); }
inline __m256i unpack32(const Lanes& L, __m256i x) {
    x = _mm256_shuffle_epi8(x, bc(L.up_idx));
    x = _mm256_or_si256(_mm256_and_si256(x, bc(L.m4)), _mm256_sll_epi64(_mm256_srl_epi64(x, L.c4), L.c32));
    x = _mm256_or_si256(_mm256_and_si256(x, bc(L.m2)), _mm256_sll_epi32(_mm256_srl_epi32(x, L.c2), L.c16));
    return _mm256_or_si256(_mm256_and_si256(x, bc(L.m1)), _mm256_sll_epi16(_mm256_srl_epi16(x, L.c1), L.c8));
}
inline __m256i pack32(const Lanes& L, __m256i x) {
    x = _mm256_or_si256(_mm256_and_si256(x, bc(L.k8)), _mm256_sll_epi16(_mm256_srl_epi16(x, L.c8), L.c1));
    x = _mm256_or_si256(_mm256_and_si256(x, bc(L.k16)), _mm256_sll_epi32(_mm256_srl_epi32(x, L.c16), L.c2));
    x = _mm256_or_si256(_mm256_and_si256(x, bc(L.k32)), _mm256_sll_epi64(_mm256_srl_epi64(x, L.c32), L.c4));
    return _mm256_shuffle_epi8(x, bc(L.pk_idx));
}
void unpack_row(const Lanes& L, const uint8_t* s, int B, uint8_t* u) {
    for (int g = 0; g < kCodes / 16; g += 2) {
        const __m256i x = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(s + g * 2 * B))),
            _mm_loadu_si128((const __m128i*)(s + (g + 1) * 2 * B)), 1);
        _mm256_storeu_si256((__m256i*)(u + g * 16), unpack32(L, x));
    }
}
/* Two 16 x 16 blocks at once, one a 128-bit lane. */
void transpose_rev(const uint8_t (*U)[kCodes], uint8_t* T) {
    for (int c = 0; c < kCodes / 16; c += 2) {
        __m256i x[16], y[16];
        for (int q = 0; q < 16; ++q) x[q] = _mm256_loadu_si256((const __m256i*)(U[15 - q] + 16 * c));
        for (int round = 0; round < 4; ++round) {
            for (int i = 0; i < 8; ++i) {
                y[2 * i] = _mm256_unpacklo_epi8(x[i], x[i + 8]);
                y[2 * i + 1] = _mm256_unpackhi_epi8(x[i], x[i + 8]);
            }
            for (int i = 0; i < 16; ++i) x[i] = y[i];
        }
        for (int j = 0; j < 16; ++j) {
            _mm_storeu_si128((__m128i*)(T + (16 * c + j) * 16), _mm256_castsi256_si128(x[j]));
            _mm_storeu_si128((__m128i*)(T + (16 * (c + 1) + j) * 16), _mm256_extracti128_si256(x[j], 1));
        }
    }
}
void pack(const Lanes& L, const uint8_t* t, int n, int B, uint8_t* out) {
    for (int g = 0; g < n / 16; g += 2) {
        const __m256i x = pack32(L, _mm256_loadu_si256((const __m256i*)(t + g * 16)));
        _mm_storeu_si128((__m128i*)(out + g * 2 * B), _mm256_castsi256_si128(x));
        _mm_storeu_si128((__m128i*)(out + (g + 1) * 2 * B), _mm256_extracti128_si256(x, 1));
    }
}
#elif R4D_HSIMD_LEVEL == 4
inline __m512i bc(__m128i x) { return _mm512_broadcast_i32x4(x); }
inline __m512i unpack64(const Lanes& L, __m512i x) {
    x = _mm512_shuffle_epi8(x, bc(L.up_idx));
    x = _mm512_or_si512(_mm512_and_si512(x, bc(L.m4)), _mm512_sll_epi64(_mm512_srl_epi64(x, L.c4), L.c32));
    x = _mm512_or_si512(_mm512_and_si512(x, bc(L.m2)), _mm512_sll_epi32(_mm512_srl_epi32(x, L.c2), L.c16));
    return _mm512_or_si512(_mm512_and_si512(x, bc(L.m1)), _mm512_sll_epi16(_mm512_srl_epi16(x, L.c1), L.c8));
}
inline __m512i pack64(const Lanes& L, __m512i x) {
    x = _mm512_or_si512(_mm512_and_si512(x, bc(L.k8)), _mm512_sll_epi16(_mm512_srl_epi16(x, L.c8), L.c1));
    x = _mm512_or_si512(_mm512_and_si512(x, bc(L.k16)), _mm512_sll_epi32(_mm512_srl_epi32(x, L.c16), L.c2));
    x = _mm512_or_si512(_mm512_and_si512(x, bc(L.k32)), _mm512_sll_epi64(_mm512_srl_epi64(x, L.c32), L.c4));
    return _mm512_shuffle_epi8(x, bc(L.pk_idx));
}
void unpack_row(const Lanes& L, const uint8_t* s, int B, uint8_t* u) {
    for (int g = 0; g < kCodes / 16; g += 4) {
        __m512i x = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)(s + g * 2 * B)));
        x = _mm512_inserti32x4(x, _mm_loadu_si128((const __m128i*)(s + (g + 1) * 2 * B)), 1);
        x = _mm512_inserti32x4(x, _mm_loadu_si128((const __m128i*)(s + (g + 2) * 2 * B)), 2);
        x = _mm512_inserti32x4(x, _mm_loadu_si128((const __m128i*)(s + (g + 3) * 2 * B)), 3);
        _mm512_storeu_si512(u + g * 16, unpack64(L, x));
    }
}
/* Four 16 x 16 blocks at once, one a 128-bit lane. */
void transpose_rev(const uint8_t (*U)[kCodes], uint8_t* T) {
    for (int c = 0; c < kCodes / 16; c += 4) {
        __m512i x[16], y[16];
        for (int q = 0; q < 16; ++q) x[q] = _mm512_loadu_si512(U[15 - q] + 16 * c);
        for (int round = 0; round < 4; ++round) {
            for (int i = 0; i < 8; ++i) {
                y[2 * i] = _mm512_unpacklo_epi8(x[i], x[i + 8]);
                y[2 * i + 1] = _mm512_unpackhi_epi8(x[i], x[i + 8]);
            }
            for (int i = 0; i < 16; ++i) x[i] = y[i];
        }
        for (int j = 0; j < 16; ++j) {
            _mm_storeu_si128((__m128i*)(T + (16 * c + j) * 16), _mm512_extracti32x4_epi32(x[j], 0));
            _mm_storeu_si128((__m128i*)(T + (16 * (c + 1) + j) * 16), _mm512_extracti32x4_epi32(x[j], 1));
            _mm_storeu_si128((__m128i*)(T + (16 * (c + 2) + j) * 16), _mm512_extracti32x4_epi32(x[j], 2));
            _mm_storeu_si128((__m128i*)(T + (16 * (c + 3) + j) * 16), _mm512_extracti32x4_epi32(x[j], 3));
        }
    }
}
void pack(const Lanes& L, const uint8_t* t, int n, int B, uint8_t* out) {
    for (int g = 0; g < n / 16; g += 4) {
        const __m512i x = pack64(L, _mm512_loadu_si512(t + g * 16));
        _mm_storeu_si128((__m128i*)(out + g * 2 * B), _mm512_extracti32x4_epi32(x, 0));
        _mm_storeu_si128((__m128i*)(out + (g + 1) * 2 * B), _mm512_extracti32x4_epi32(x, 1));
        _mm_storeu_si128((__m128i*)(out + (g + 2) * 2 * B), _mm512_extracti32x4_epi32(x, 2));
        _mm_storeu_si128((__m128i*)(out + (g + 3) * 2 * B), _mm512_extracti32x4_epi32(x, 3));
    }
}
#endif

/* ---- the w2a8 tile --------------------------------------------------------------------------- */

/* Dword d of half lh gathers four bytes of the unit's 32: bytes 8d + lh + {0, 2, 4, 6}, which
 * are codes 4 lh .. 4 lh + 3 of columns 16 (2d + h) + 8b for (b, h) = (0, 0), (1, 0), (0, 1),
 * (1, 1), a 4 x 4 matrix of 2-bit fields with group s = b + 2h in byte s. The stored dword has
 * code j of group s in field 4j + s, its transpose: two delta swaps, the 2 x 2 blocks' corners
 * then the blocks. */
inline uint32_t w2_transpose(uint32_t x) {
    uint32_t t = (x ^ (x >> 6)) & 0x00CC00CCu;
    x ^= t ^ (t << 6);
    t = (x ^ (x >> 12)) & 0x0000F0F0u;
    return x ^ t ^ (t << 12);
}

#if R4D_HSIMD_LEVEL == 0
void w2_row(const uint8_t* s, int64_t kpb, uint8_t* o) {
    for (int64_t kb = 0; kb < kpb; ++kb, s += 32, o += 512)
        for (int lh = 0; lh < 2; ++lh)
            for (int d = 0; d < 4; ++d) {
                const uint8_t* b = s + 8 * d + lh;
                const uint32_t x = (uint32_t)b[0] | (uint32_t)b[2] << 8 | (uint32_t)b[4] << 16 |
                                   (uint32_t)b[6] << 24;
                const uint32_t w = w2_transpose(x);
                std::memcpy(o + 256 * lh + 4 * d, &w, 4);
            }
}
#else
/* A 16-byte lane of a unit holds the bytes of dwords d = 2m, 2m + 1: the shuffle gathers half 0's
 * two dwords, then half 1's, and the transpose runs on all four. */
inline __m128i w2_idx() { return _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15); }
#endif

#if R4D_HSIMD_LEVEL == 1 || R4D_HSIMD_LEVEL == 2
inline __m128i w2_lane(__m128i x, __m128i idx) {
    x = _mm_shuffle_epi8(x, idx);
    __m128i t = _mm_and_si128(_mm_xor_si128(x, _mm_srli_epi32(x, 6)), _mm_set1_epi32(0x00CC00CC));
    x = _mm_xor_si128(x, _mm_xor_si128(t, _mm_slli_epi32(t, 6)));
    t = _mm_and_si128(_mm_xor_si128(x, _mm_srli_epi32(x, 12)), _mm_set1_epi32(0x0000F0F0));
    return _mm_xor_si128(x, _mm_xor_si128(t, _mm_slli_epi32(t, 12)));
}
void w2_row(const uint8_t* s, int64_t kpb, uint8_t* o) {
    const __m128i idx = w2_idx();
    for (int64_t kb = 0; kb < kpb; ++kb, s += 32, o += 512) {
        const __m128i a = w2_lane(_mm_loadu_si128((const __m128i*)s), idx);
        const __m128i b = w2_lane(_mm_loadu_si128((const __m128i*)(s + 16)), idx);
        _mm_storeu_si128((__m128i*)o, _mm_unpacklo_epi64(a, b));
        _mm_storeu_si128((__m128i*)(o + 256), _mm_unpackhi_epi64(a, b));
    }
}
#elif R4D_HSIMD_LEVEL == 3
void w2_row(const uint8_t* s, int64_t kpb, uint8_t* o) {
    const __m256i idx = _mm256_broadcastsi128_si256(w2_idx());
    const __m256i k6 = _mm256_set1_epi32(0x00CC00CC), k12 = _mm256_set1_epi32(0x0000F0F0);
    for (int64_t kb = 0; kb < kpb; ++kb, s += 32, o += 512) {
        __m256i x = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i*)s), idx);
        __m256i t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi32(x, 6)), k6);
        x = _mm256_xor_si256(x, _mm256_xor_si256(t, _mm256_slli_epi32(t, 6)));
        t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi32(x, 12)), k12);
        x = _mm256_xor_si256(x, _mm256_xor_si256(t, _mm256_slli_epi32(t, 12)));
        /* Quadwords (a0, a1, b0, b1) to (a0, b0, a1, b1): half 0, then half 1. */
        x = _mm256_permute4x64_epi64(x, 0xD8);
        _mm_storeu_si128((__m128i*)o, _mm256_castsi256_si128(x));
        _mm_storeu_si128((__m128i*)(o + 256), _mm256_extracti128_si256(x, 1));
    }
}
#elif R4D_HSIMD_LEVEL == 4
void w2_row(const uint8_t* s, int64_t kpb, uint8_t* o) {
    const __m512i idx = _mm512_broadcast_i32x4(w2_idx());
    const __m512i k6 = _mm512_set1_epi32(0x00CC00CC), k12 = _mm512_set1_epi32(0x0000F0F0);
    const __m512i perm = _mm512_setr_epi64(0, 2, 1, 3, 4, 6, 5, 7);
    int64_t kb = 0;
    /* Two units a step; xor-and-xor is one ternary op. */
    for (; kb + 2 <= kpb; kb += 2, s += 64, o += 1024) {
        __m512i x = _mm512_shuffle_epi8(_mm512_loadu_si512(s), idx);
        __m512i t = _mm512_and_si512(_mm512_xor_si512(x, _mm512_srli_epi32(x, 6)), k6);
        x = _mm512_ternarylogic_epi32(x, t, _mm512_slli_epi32(t, 6), 0x96);
        t = _mm512_and_si512(_mm512_xor_si512(x, _mm512_srli_epi32(x, 12)), k12);
        x = _mm512_ternarylogic_epi32(x, t, _mm512_slli_epi32(t, 12), 0x96);
        x = _mm512_permutexvar_epi64(perm, x);
        _mm_storeu_si128((__m128i*)o, _mm512_extracti32x4_epi32(x, 0));
        _mm_storeu_si128((__m128i*)(o + 256), _mm512_extracti32x4_epi32(x, 1));
        _mm_storeu_si128((__m128i*)(o + 512), _mm512_extracti32x4_epi32(x, 2));
        _mm_storeu_si128((__m128i*)(o + 768), _mm512_extracti32x4_epi32(x, 3));
    }
    if (kb < kpb) {
        const __m256i idx2 = _mm512_castsi512_si256(idx);
        __m256i x = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i*)s), idx2);
        __m256i t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi32(x, 6)), _mm512_castsi512_si256(k6));
        x = _mm256_ternarylogic_epi32(x, t, _mm256_slli_epi32(t, 6), 0x96);
        t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi32(x, 12)), _mm512_castsi512_si256(k12));
        x = _mm256_ternarylogic_epi32(x, t, _mm256_slli_epi32(t, 12), 0x96);
        x = _mm256_permute4x64_epi64(x, 0xD8);
        _mm_storeu_si128((__m128i*)o, _mm256_castsi256_si128(x));
        _mm_storeu_si128((__m128i*)(o + 256), _mm256_extracti128_si256(x, 1));
    }
}
#endif

/* ---- the 16-row dword transpose ---------------------------------------------------------------- */

void dt_scalar(const uint32_t* in, int64_t stride, int64_t c0, int64_t cols, uint32_t* out) {
    for (int64_t c = c0; c < cols; ++c)
        for (int r = 0; r < 16; ++r) out[16 * c + r] = in[r * stride + c];
}

#if R4D_HSIMD_LEVEL >= 1
/* Rows 4g .. 4g + 3 of columns c .. c + 3: v[i] becomes column c + i of those rows. */
inline void dt_4x4(const uint32_t* in, int64_t stride, int64_t c, int g, __m128i v[4]) {
    const uint32_t* p = in + 4 * g * stride + c;
    const __m128i a = _mm_loadu_si128((const __m128i*)p);
    const __m128i b = _mm_loadu_si128((const __m128i*)(p + stride));
    const __m128i e = _mm_loadu_si128((const __m128i*)(p + 2 * stride));
    const __m128i f = _mm_loadu_si128((const __m128i*)(p + 3 * stride));
    const __m128i ab0 = _mm_unpacklo_epi32(a, b), ab1 = _mm_unpackhi_epi32(a, b);
    const __m128i ef0 = _mm_unpacklo_epi32(e, f), ef1 = _mm_unpackhi_epi32(e, f);
    v[0] = _mm_unpacklo_epi64(ab0, ef0);
    v[1] = _mm_unpackhi_epi64(ab0, ef0);
    v[2] = _mm_unpacklo_epi64(ab1, ef1);
    v[3] = _mm_unpackhi_epi64(ab1, ef1);
}
#endif

#if R4D_HSIMD_LEVEL == 1 || R4D_HSIMD_LEVEL == 2
void dt_vec(const uint32_t* in, int64_t stride, int64_t cols, uint32_t* out) {
    int64_t c = 0;
    for (; c + 4 <= cols; c += 4)
        for (int g = 0; g < 4; ++g) {
            __m128i v[4];
            dt_4x4(in, stride, c, g, v);
            for (int i = 0; i < 4; ++i) _mm_storeu_si128((__m128i*)(out + 16 * (c + i) + 4 * g), v[i]);
        }
    dt_scalar(in, stride, c, cols, out);
}
#elif R4D_HSIMD_LEVEL == 3
/* Eight columns a step: the 4 x 4 transpose in each 128-bit lane, the high lane columns c + 4 ..
 * c + 7. */
void dt_vec(const uint32_t* in, int64_t stride, int64_t cols, uint32_t* out) {
    int64_t c = 0;
    for (; c + 8 <= cols; c += 8)
        for (int g = 0; g < 4; ++g) {
            const uint32_t* p = in + 4 * g * stride + c;
            const __m256i a = _mm256_loadu_si256((const __m256i*)p);
            const __m256i b = _mm256_loadu_si256((const __m256i*)(p + stride));
            const __m256i e = _mm256_loadu_si256((const __m256i*)(p + 2 * stride));
            const __m256i f = _mm256_loadu_si256((const __m256i*)(p + 3 * stride));
            const __m256i ab0 = _mm256_unpacklo_epi32(a, b), ab1 = _mm256_unpackhi_epi32(a, b);
            const __m256i ef0 = _mm256_unpacklo_epi32(e, f), ef1 = _mm256_unpackhi_epi32(e, f);
            const __m256i v[4] = { _mm256_unpacklo_epi64(ab0, ef0), _mm256_unpackhi_epi64(ab0, ef0),
                                   _mm256_unpacklo_epi64(ab1, ef1), _mm256_unpackhi_epi64(ab1, ef1) };
            for (int i = 0; i < 4; ++i) {
                _mm_storeu_si128((__m128i*)(out + 16 * (c + i) + 4 * g), _mm256_castsi256_si128(v[i]));
                _mm_storeu_si128((__m128i*)(out + 16 * (c + 4 + i) + 4 * g), _mm256_extracti128_si256(v[i], 1));
            }
        }
    for (; c + 4 <= cols; c += 4)
        for (int g = 0; g < 4; ++g) {
            __m128i v[4];
            dt_4x4(in, stride, c, g, v);
            for (int i = 0; i < 4; ++i) _mm_storeu_si128((__m128i*)(out + 16 * (c + i) + 4 * g), v[i]);
        }
    dt_scalar(in, stride, c, cols, out);
}
#elif R4D_HSIMD_LEVEL == 4
/* Four columns a step, each one whole 64-byte column of the tile: four 4 x 4 transposes, one per
 * group of rows, joined lane by lane. */
void dt_vec(const uint32_t* in, int64_t stride, int64_t cols, uint32_t* out) {
    int64_t c = 0;
    for (; c + 4 <= cols; c += 4) {
        __m128i v[4][4];
        for (int g = 0; g < 4; ++g) dt_4x4(in, stride, c, g, v[g]);
        for (int i = 0; i < 4; ++i) {
            __m512i z = _mm512_castsi128_si512(v[0][i]);
            z = _mm512_inserti32x4(z, v[1][i], 1);
            z = _mm512_inserti32x4(z, v[2][i], 2);
            z = _mm512_inserti32x4(z, v[3][i], 3);
            _mm512_storeu_si512(out + 16 * (c + i), z);
        }
    }
    dt_scalar(in, stride, c, cols, out);
}
#endif

/* ---- the byte transpose ----------------------------------------------------------------------- */

void bt_scalar(const uint8_t* in, int64_t rows, int64_t cols, int64_t r0, int64_t r1, int64_t c0,
               int64_t c1, uint8_t* out) {
    for (int64_t r = r0; r < r1; ++r)
        for (int64_t c = c0; c < c1; ++c) out[c * rows + r] = in[r * cols + c];
}

#if R4D_HSIMD_LEVEL >= 1
/* A 16 x 16 block at (r, c): four rounds of interleaving rows i and i + 8, as the trellis tile's
 * transpose (without its row reversal). */
void bt_block(const uint8_t* in, int64_t rows, int64_t cols, int64_t r, int64_t c, uint8_t* out) {
    __m128i x[16], y[16];
    for (int q = 0; q < 16; ++q) x[q] = _mm_loadu_si128((const __m128i*)(in + (r + q) * cols + c));
    for (int round = 0; round < 4; ++round) {
        for (int i = 0; i < 8; ++i) {
            y[2 * i] = _mm_unpacklo_epi8(x[i], x[i + 8]);
            y[2 * i + 1] = _mm_unpackhi_epi8(x[i], x[i + 8]);
        }
        for (int i = 0; i < 16; ++i) x[i] = y[i];
    }
    for (int j = 0; j < 16; ++j) _mm_storeu_si128((__m128i*)(out + (c + j) * rows + r), x[j]);
}
#endif

#if R4D_HSIMD_LEVEL == 3 || R4D_HSIMD_LEVEL == 4
/* Two 16 x 16 blocks side by side, one a 128-bit lane: rows r .. r + 15, columns c .. c + 31. */
void bt_block2(const uint8_t* in, int64_t rows, int64_t cols, int64_t r, int64_t c, uint8_t* out) {
    __m256i x[16], y[16];
    for (int q = 0; q < 16; ++q) x[q] = _mm256_loadu_si256((const __m256i*)(in + (r + q) * cols + c));
    for (int round = 0; round < 4; ++round) {
        for (int i = 0; i < 8; ++i) {
            y[2 * i] = _mm256_unpacklo_epi8(x[i], x[i + 8]);
            y[2 * i + 1] = _mm256_unpackhi_epi8(x[i], x[i + 8]);
        }
        for (int i = 0; i < 16; ++i) x[i] = y[i];
    }
    for (int j = 0; j < 16; ++j) {
        _mm_storeu_si128((__m128i*)(out + (c + j) * rows + r), _mm256_castsi256_si128(x[j]));
        _mm_storeu_si128((__m128i*)(out + (c + 16 + j) * rows + r), _mm256_extracti128_si256(x[j], 1));
    }
}
#endif

void bt_run(const uint8_t* in, int64_t rows, int64_t cols, uint8_t* out) {
#if R4D_HSIMD_LEVEL == 0
    bt_scalar(in, rows, cols, 0, rows, 0, cols, out);
#else
    const int64_t R = rows / 16 * 16, C = cols / 16 * 16;
    for (int64_t r = 0; r < R; r += 16) {
        int64_t c = 0;
#if R4D_HSIMD_LEVEL >= 3
        for (; c + 32 <= C; c += 32) bt_block2(in, rows, cols, r, c, out);
#endif
        for (; c < C; c += 16) bt_block(in, rows, cols, r, c, out);
    }
    bt_scalar(in, rows, cols, 0, R, C, cols, out);
    bt_scalar(in, rows, cols, R, rows, 0, cols, out);
#endif
}

}  // namespace

extern "C" void R4D_HSIMD_FN(byte_transpose)(const uint8_t* in, int64_t rows, int64_t cols, uint8_t* out) {
    bt_run(in, rows, cols, out);
}

extern "C" void R4D_HSIMD_FN(dword_transpose16)(const uint32_t* in, int64_t stride, int64_t cols,
                                                uint32_t* out) {
#if R4D_HSIMD_LEVEL == 0
    dt_scalar(in, stride, 0, cols, out);
#else
    dt_vec(in, stride, cols, out);
#endif
}

extern "C" void R4D_HSIMD_FN(w2a8_tile)(const uint8_t* rows, int64_t rb, int64_t kpb, uint8_t* out) {
    for (int r = 0; r < 16; ++r) w2_row(rows + r * rb, kpb, out + 16 * r);
}

extern "C" void R4D_HSIMD_FN(moew4_permute)(const uint8_t* s, uint8_t* d, int64_t bytes, bool fwd) {
#if R4D_HSIMD_LEVEL == 0
    mw_scalar(s, d, bytes, fwd);
#else
    mw_vec(s, d, bytes, fwd);
#endif
}

extern "C" void R4D_HSIMD_FN(iqr_trellis_tile)(const uint8_t* rows, int64_t rb, int B, int64_t nkb,
                                               uint8_t* out) {
    alignas(64) uint8_t U[16][kCodes];
    alignas(64) uint8_t T[kCodes * 16];
    const int64_t per = 16 * B, ub = 256 * B;
#if R4D_HSIMD_LEVEL >= 1
    alignas(64) uint8_t pad[16 * 8 + 64];   /* a row's 16 B bytes with room to read past */
    alignas(64) uint8_t pk[256 * 8 + 64];   /* a unit with room to write past */
    const Lanes L(B);
    bool streamed = false;
#endif
    for (int64_t kb = 0; kb < nkb; ++kb) {
        for (int r = 0; r < 16; ++r) {
            const uint8_t* s = rows + r * rb + kb * per;
#if R4D_HSIMD_LEVEL == 0
            unpack_row(s, B, U[r]);
#else
            /* The vector unpack reads up to 16 bytes past a group; only the tile's last row of
             * its last unit may end where the plane does. */
            if (r == 15 && kb == nkb - 1) {
                std::memcpy(pad, s, (size_t)per);
                s = pad;
            }
            unpack_row(L, s, B, U[r]);
#endif
        }
        transpose_rev(U, T);
#if R4D_HSIMD_LEVEL == 0
        pack(T, kCodes * 16, B, out + kb * ub);
#else
        pack(L, T, kCodes * 16, B, pk);
        streamed |= stream_copy(out + kb * ub, pk, ub);
#endif
    }
#if R4D_HSIMD_LEVEL >= 1
    stream_fence(streamed);
#endif
}

/* A row's eight qwords of one 64-byte K step: the even ones to lane r, the odd ones to lane
 * 16 + r, 32 bytes each (r4d_fp8_frag.h, frag_src). */
/* With `out` aligned the 32-byte pieces go out as streaming stores (stream_copy says why); the two
 * of a destination line are written by consecutive rows, so the line leaves whole. */
template <bool kStream>
inline void fp8_frag_body(const uint8_t* rows, int64_t K, int64_t ksuper, uint8_t* out) {
#if R4D_HSIMD_LEVEL == 4
    const __m512i idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
#endif
#if R4D_HSIMD_LEVEL >= 3
    auto put = [](uint8_t* d, __m256i v) {
        if (kStream) _mm256_stream_si256((__m256i*)d, v);
        else         _mm256_storeu_si256((__m256i*)d, v);
    };
#elif R4D_HSIMD_LEVEL >= 1
    auto put = [](uint8_t* d, __m128i v) {
        if (kStream) _mm_stream_si128((__m128i*)d, v);
        else         _mm_storeu_si128((__m128i*)d, v);
    };
#endif
    for (int64_t ks = 0; ks < ksuper; ++ks) {
        uint8_t* t = out + ks * 1024;
        for (int r = 0; r < 16; ++r) {
            const uint8_t* s = rows + r * K + ks * 64;
            uint8_t* lo = t + 32 * r;
            uint8_t* hi = t + 32 * (16 + r);
#if R4D_HSIMD_LEVEL == 4
            const __m512i x = _mm512_permutexvar_epi64(idx, _mm512_loadu_si512(s));
            put(lo, _mm512_castsi512_si256(x));
            put(hi, _mm512_extracti64x4_epi64(x, 1));
#elif R4D_HSIMD_LEVEL == 3
            const __m256i a = _mm256_permute4x64_epi64(_mm256_loadu_si256((const __m256i*)s), 0xD8);
            const __m256i b = _mm256_permute4x64_epi64(_mm256_loadu_si256((const __m256i*)(s + 32)), 0xD8);
            put(lo, _mm256_permute2x128_si256(a, b, 0x20));
            put(hi, _mm256_permute2x128_si256(a, b, 0x31));
#elif R4D_HSIMD_LEVEL >= 1
            const __m128i a = _mm_loadu_si128((const __m128i*)s);
            const __m128i b = _mm_loadu_si128((const __m128i*)(s + 16));
            const __m128i c = _mm_loadu_si128((const __m128i*)(s + 32));
            const __m128i d = _mm_loadu_si128((const __m128i*)(s + 48));
            put(lo, _mm_unpacklo_epi64(a, b));
            put(lo + 16, _mm_unpacklo_epi64(c, d));
            put(hi, _mm_unpackhi_epi64(a, b));
            put(hi + 16, _mm_unpackhi_epi64(c, d));
#else
            for (int q = 0; q < 4; ++q) {
                std::memcpy(lo + 8 * q, s + 16 * q, 8);
                std::memcpy(hi + 8 * q, s + 16 * q + 8, 8);
            }
#endif
        }
    }
}

extern "C" void R4D_HSIMD_FN(fp8_frag_tile)(const uint8_t* rows, int64_t K, int64_t ksuper,
                                            uint8_t* out) {
#if R4D_HSIMD_LEVEL >= 1
    if (((uintptr_t)out & 63) == 0) {
        fp8_frag_body<true>(rows, K, ksuper, out);
        stream_fence(true);
        return;
    }
#endif
    fp8_frag_body<false>(rows, K, ksuper, out);
}
