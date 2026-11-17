// r4d_q8.h -- the 8-bit activation codes the trunk GEMMs read, written by every quantising producer:
// E4M3 for the fp8a8 GEMMs, int8 for their int8 twins (gemm_i8a8_*). One scale per (row, 128
// columns); only the largest code and the conversion differ.
//
// E4M3: scale `amax * (1 / 448)`, clamp to +-448, two packed conversions. The clamp is required:
// v_cvt_pk_fp8_f32 returns NaN for out-of-range input instead of saturating, and
// `amax * (1 / scale)` can land an ulp above 448.
//
// Int8: as quant_act_i8g (r4d_quant_act_fp8.hip), scale `amax * (1 / 127)` and round-to-nearest-
// even into -127..127, so a fused producer writes the same codes as the standalone quantiser.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "r4d_common.h"   // r4d_wave_max, for q8_row_i8g
#include "r4d_fwht.h"     // the rotation q8_row_had_i8g puts before the absmax

template <bool I8>
__device__ __forceinline__ constexpr float q8_top() {
    return I8 ? 127.0f : 448.0f;
}

// Four values times `inv` (the reciprocal of the scale), as four codes in one dword, value 0 in
// the low byte.
template <bool I8>
__device__ __forceinline__ uint32_t q8_pack4(float a, float b, float c, float d, float inv) {
    if constexpr (I8) {
        const int ia = (int)rintf(fminf(fmaxf(a * inv, -127.0f), 127.0f));
        const int ib = (int)rintf(fminf(fmaxf(b * inv, -127.0f), 127.0f));
        const int ic = (int)rintf(fminf(fmaxf(c * inv, -127.0f), 127.0f));
        const int id = (int)rintf(fminf(fmaxf(d * inv, -127.0f), 127.0f));
        return ((uint32_t)ia & 0xFFu) | (((uint32_t)ib & 0xFFu) << 8) |
               (((uint32_t)ic & 0xFFu) << 16) | (((uint32_t)id & 0xFFu) << 24);
    } else {
        int pk = 0;
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(a * inv, -448.0f), 448.0f),
                                             fminf(fmaxf(b * inv, -448.0f), 448.0f), pk, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(c * inv, -448.0f), 448.0f),
                                             fminf(fmaxf(d * inv, -448.0f), 448.0f), pk, true);
        return (uint32_t)pk;
    }
}

// The int8 codes of a whole bf16 row as quant_act_i8g writes them, for a producer whose row is
// staged in LDS (`row`, n values, n % 128 == 0) because it crossed threads on the way: the norm and
// gelu junctions of a Gemma trunk on an int8-per-(row, 128) grid (rmsnorm_quant_i8g,
// gelu_mul_quant_i8g). A wave takes a 128-column group and a lane four consecutive columns, as
// r4d_quant_act_fp8.hip's I8 form maps them; the absmax is a max, which any partition gives, and
// the scale (1 for an all-zero group), rounding and clamp are that form's. Codes row-major into `q`,
// the n / 128 scales into `s`. The caller publishes `row` behind a barrier first.
template <uint32_t THR>
__device__ __forceinline__ void q8_row_i8g(const uint16_t* row, signed char* __restrict__ q,
                                           float* __restrict__ s, uint32_t n) {
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    for (uint32_t kb = wave; kb < n / 128u; kb += THR / 32u) {
        const uint32_t c0 = kb * 128u + lane * 4u;
        float v[4];
        float m = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = __uint_as_float((uint32_t)row[c0 + j] << 16);
            m = fmaxf(m, fabsf(v[j]));
        }
        m = r4d_wave_max(m);
        const float sc = m > 0.0f ? m * (1.0f / q8_top<true>()) : 1.0f;
        if (lane == 0) s[kb] = sc;
        *(uint32_t*)(q + c0) = q8_pack4<true>(v[0], v[1], v[2], v[3], 1.0f / sc);
    }
}

// q8_row_i8g with the 128-wide Hadamard between the staged values and the absmax: the codes
// had_quant_act_i8g writes over the same bf16 row (r4d_fwht_wave's mapping is this one's, a wave a
// group and four consecutive columns a lane), for a producer whose row crossed threads on the way
// (norm_add_norm_route_i8g). The caller publishes `row` behind a barrier first.
template <uint32_t THR>
__device__ __forceinline__ void q8_row_had_i8g(const uint16_t* row, signed char* __restrict__ q,
                                               float* __restrict__ s, uint32_t n) {
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    for (uint32_t kb = wave; kb < n / 128u; kb += THR / 32u) {
        const uint32_t c0 = kb * 128u + lane * 4u;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __uint_as_float((uint32_t)row[c0 + j] << 16);
        r4d_fwht_wave<128>(v, (int)lane);
        float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
        m = r4d_wave_max(m);
        const float sc = m > 0.0f ? m * (1.0f / q8_top<true>()) : 1.0f;
        if (lane == 0) s[kb] = sc;
        *(uint32_t*)(q + c0) = q8_pack4<true>(v[0], v[1], v[2], v[3], 1.0f / sc);
    }
}
