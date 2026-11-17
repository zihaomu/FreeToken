// 16-bit operand traits. The QK and PV matmuls take a 16-bit A/B pair, and both sources are exact
// in either format: an fp8 e4m3 K/V has 3 mantissa bits (bf16 has 7, f16 10), and a bf16 query's
// 7 bits are exact in f16 for |q| in [6.1e-5, 65504].
// The costs differ. gfx1201 has no bf16 convert, so each f32->bf16 is ~6 VALU of software RTNE,
// while v_cvt_pkrtz_f16_f32 converts two f32 in one; on the softmax path that is 24 VALU per
// 8-element P fragment against 4. f16 also has 3 more mantissa bits; it gives up exponent range,
// which GROW below accounts for.
#pragma once
#include "r4d_common.h"

typedef __bf16   v8bf __attribute__((ext_vector_type(8)));
typedef __fp16   v8hf __attribute__((ext_vector_type(8)));  // what the f16 WMMA builtin takes
typedef __fp16   v2hf __attribute__((ext_vector_type(2)));  // what cvt_pkrtz returns

// GROW = how many octaves the running row max may exceed the reference max before the accumulator
// must be rescaled. p = 2^(s - m_ref) must stay inside the format, so GROW is log2(format max) with
// margin. Eager online softmax pays 128 multiplies every m-tile; with a reference max they are paid
// only when the max moves by more than GROW, which at long context is rare.
template<int F16> struct DT16;

template<> struct DT16<0> {                       // bf16: exponent range of f32
    typedef v8bf frag;
    static constexpr float SHIFT = 0.0f;
    static constexpr float GROW  = 60.0f;         // 2^60 * |v| * n stays far under f32 max
    __device__ __forceinline__ static frag mk(uint2 lo, uint2 hi) {
        union { uint32_t w[4]; frag f; } u;       // union, never a pointer cast
        u.w[0] = lo.x; u.w[1] = lo.y; u.w[2] = hi.x; u.w[3] = hi.y; return u.f;
    }
    __device__ __forceinline__ static v8f wmma(frag a, frag b, v8f c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    }
    __device__ __forceinline__ static uint32_t pk(float x, float y) {
        return (uint32_t)f32_to_bf16(x) | ((uint32_t)f32_to_bf16(y) << 16);
    }
    __device__ __forceinline__ static uint32_t pkv(v2f x) {   // what cvt_pk_f32_fp8 returns
        union { v2f f; uint32_t u[2]; } q; q.f = x;
        return __builtin_amdgcn_perm(q.u[1], q.u[0], 0x07060302u);
    }
    __device__ __forceinline__ static uint32_t from_bf16w(uint32_t w) { return w; }  // identity
};

template<> struct DT16<1> {                       // f16: 3 more mantissa bits, 1 convert instead of 6
    typedef v8hf frag;
    static constexpr float SHIFT = 0.0f;
    static constexpr float GROW  = 14.0f;         // p <= 2^14 = 16384 < 65504
    __device__ __forceinline__ static frag mk(uint2 lo, uint2 hi) {
        union { uint32_t w[4]; frag f; } u;
        u.w[0] = lo.x; u.w[1] = lo.y; u.w[2] = hi.x; u.w[3] = hi.y; return u.f;
    }
    __device__ __forceinline__ static v8f wmma(frag a, frag b, v8f c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    }
    __device__ __forceinline__ static uint32_t pk(float x, float y) {
        union { v2hf h; uint32_t u; } q; q.h = __builtin_amdgcn_cvt_pkrtz(x, y); return q.u;
    }
    __device__ __forceinline__ static uint32_t pkv(v2f x) {
        union { v2f f; float e[2]; } s; s.f = x;
        union { v2hf h; uint32_t u; } q; q.h = __builtin_amdgcn_cvt_pkrtz(s.e[0], s.e[1]); return q.u;
    }
    // two bf16 in a dword -> two f16 in a dword. bf16->f32 is a shift, so this is 3 VALU.
    __device__ __forceinline__ static uint32_t from_bf16w(uint32_t w) {
        union { v2hf h; uint32_t u; } q;
        q.h = __builtin_amdgcn_cvt_pkrtz(__builtin_bit_cast(float, w << 16),
                                         __builtin_bit_cast(float, w & 0xffff0000u));
        return q.u;
    }
};

// Widen 8 fp8 (2 dwords) to 8 16-bit values (4 dwords) in element order.
// Eight K dims of the k6_v4 cache as four f16 dwords, dims (2i, 2i+1) in dword i. `lo` holds
// their low nibbles (dim 2i in bits 8i..8i+3, 2i+1 in 8i+4..8i+7) and `hb` their 16-dim chunk's
// high-bit word (byte j: bits 0-1 dim 2j, 2-3 dim 2j+1, 4-5 dim 8+2j, 6-7 dim 9+2j); `half` says
// which eight of the sixteen these are. One mask and shift put the high bits beside the low
// nibbles, a byte a dim. Codes are 6-bit two's complement: ^ 32 makes them offset binary u,
// 0x6400 | u is the f16 1024 + u, and (x - 1056) * s is the value times the group scale `s2` (the
// f16 in both halves), rounded once.
__device__ __forceinline__ void k6x8_to_f16x4w(uint32_t lo, uint32_t hb, int half, uint32_t s2,
                                               uint32_t* w4) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const uint32_t ev = ((lo & 0x0F0F0F0Fu) | ((half ? hb : hb << 4) & 0x30303030u)) ^ 0x20202020u;
    const uint32_t od = (((lo >> 4) & 0x0F0F0F0Fu) | ((half ? hb >> 2 : hb << 2) & 0x30303030u))
                      ^ 0x20202020u;
    const h2 s = __builtin_bit_cast(h2, s2);
    const h2 bias = { (_Float16)1056.0f, (_Float16)1056.0f };
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        // Byte b of the even dims low, of the odd dims high; perm takes bytes 0-3 from its second
        // operand, 4-7 from its first, and 0x0c is a zero byte.
        const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
        const uint32_t x = __builtin_amdgcn_perm(od, ev, sel) | 0x64006400u;
        w4[b] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, x) - bias) * s);
    }
}

// Eight int8 codes (two dwords, element order) as four f16 dwords, dims (2i, 2i+1) in dword i,
// each the code times the scale `s2` (the f16 in both halves): the i8 cache's K and V
// (a RAD_I8 group). ^ 0x80 makes a code offset binary u, 0x6400 | u is the f16 1024 + u, and
// (x - 1152) * s is the value times the scale, rounded once.
__device__ __forceinline__ void i8x8_to_f16x4w(uint2 raw, uint32_t s2, uint32_t* w4) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const uint32_t lo = raw.x ^ 0x80808080u, hi = raw.y ^ 0x80808080u;
    const h2 s = __builtin_bit_cast(h2, s2);
    const h2 bias = { (_Float16)1152.0f, (_Float16)1152.0f };
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        // Bytes 2b and 2b+1 of the eight, each into the low byte of a half; perm takes bytes 0-3
        // from its second operand, 4-7 from its first, and 0x0c is a zero byte.
        const uint32_t sel = (uint32_t)(2 * b) | 0x0c00u | ((uint32_t)(2 * b + 1) << 16) | 0x0c000000u;
        const uint32_t x = __builtin_amdgcn_perm(hi, lo, sel) | 0x64006400u;
        w4[b] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, x) - bias) * s);
    }
}

template<class D>
__device__ __forceinline__ void fp8x8_to_16x4w(uint2 raw, uint32_t* w4) {
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int wv = (q == 0) ? (int)raw.x : (int)raw.y;
        w4[q * 2 + 0] = D::pkv(__builtin_amdgcn_cvt_pk_f32_fp8(wv, false));
        w4[q * 2 + 1] = D::pkv(__builtin_amdgcn_cvt_pk_f32_fp8(wv, true));
    }
}

__device__ __forceinline__ int wave_any(bool c) {
    return (int)__builtin_amdgcn_ballot_w32(c);
}
