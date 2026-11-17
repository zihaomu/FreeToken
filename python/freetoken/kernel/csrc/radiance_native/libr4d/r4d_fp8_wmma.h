// Shared device code for the block-scaled FP8 family: the 16x16x16 fp8 WMMA fragment map, the
// E4M3 decoders and the bf16 block-scale load.
//
// ---- lane->element mapping (undocumented by AMD; ROCm issue #6025) ------------------------------
//
// With lane `l`, `c = l % 16`, `h = l / 16`:
//
//   A, row-major A[m][k]     lane l holds A[c][h*8 + j],  j = 0..7
//   B, row-major B[k][n]     lane l holds B[h*8 + j][c],  j = 0..7
//   C/D accumulator          acc[j] IS D[h*8 + j][c]
//
// The accumulator is column-distributed: a lane owns one column and eight consecutive rows (lanes
// 0-15 rows 0-7, lanes 16-31 rows 8-15). Assuming a lane owns a row gives a silently transposed
// tile. Identity, diagonal and symmetric test matrices cannot detect that; tests of fragment code
// must use asymmetric operands.
//
// ---- K labelling --------------------------------------------------------------------------------
//
// The hardware K-order is not contiguous: lanes 0-15 hold k = 0,1,2,3,8,9,10,11. Labelling slot j
// of lane-half h as "k = 8h + j" is a permutation of K applied identically to A and B, so the
// product is bit-identical. Two rules keep it that way:
//
//   1. Both operands must use this labelling. Mixing GLOBAL_LOAD_TR_B128 (canonical order) with a
//      plain load misaligns K silently. Every kernel here stages both operands through `stage8`
//      and reads both through `r4d_fp8::frag`.
//   2. A sparse path may not use it: SWMMAC's index bits are tied to the hardware K groups.
//
// ---- hazards ------------------------------------------------------------------------------------
//
// A WMMA whose A, B or index overlaps the previous WMMA's D needs an independent VALU instruction
// in between (required for correctness; the compiler inserts it). Feeding D back as C of the same
// opcode is exempt, so these kernels keep one accumulator chain of a single opcode.
//
// WMMA does not co-issue with VALU on RDNA4, so dequantisation and address math are serial with
// matrix work. v_wmma_f32_16x16x16_fp8_fp8 issues at 412 TF/s (2.99 cycles), the same rate as
// v_wmma_i32_16x16x16_iu8, and is reached here because both operands are already E4M3 bytes and
// nothing is unpacked between the LDS read and the mma.
#pragma once
#include "r4d_common.h"
#include "r4d_fp8_frag.h"

#include <hip/hip_runtime.h>
#include <cstdint>

namespace r4d_fp8 {

// A/B operand widths are RDNA4's: half of RDNA3's, since RDNA4 dropped the cross-half
// replication. A gfx11-width vector fails to compile here.
typedef int   i2v __attribute__((ext_vector_type(2)));   // fp8 A or B, 2 VGPRs
typedef float f8v __attribute__((ext_vector_type(8)));   // f32 accumulator, 8 VGPRs

constexpr uint32_t kM = 16, kN = 16, kK = 16;

// Always the _gfx12-suffixed builtin; the unsuffixed RDNA3 form has different operand widths.
__device__ __forceinline__ f8v mma(i2v a, i2v b, f8v c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// Int8 twin for the same kernels at i8 codes: v_wmma_i32_16x16x16_iu8, both operands signed, at
// the fp8 rate. The int32 accumulator is carried in f8v registers as bits so one accumulator type
// serves both; acc_val reads a slot back as a number. A 128-K block sums at most 128 * 127 * 127,
// well inside int32.
typedef int i8v __attribute__((ext_vector_type(8)));
template <bool I8>
__device__ __forceinline__ f8v mma8(i2v a, i2v b, f8v c) {
    if constexpr (I8)
        return __builtin_bit_cast(f8v, __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                           true, a, true, b, __builtin_bit_cast(i8v, c), false));
    else
        return mma(a, b, c);
}
template <bool I8>
__device__ __forceinline__ float acc_val(float x) {
    if constexpr (I8) return (float)__builtin_bit_cast(int, x);
    else return x;
}

__device__ __forceinline__ uint32_t frag_c (uint32_t lane) { return lane & 15u; }
__device__ __forceinline__ uint32_t frag_k0(uint32_t lane) { return (lane >> 4) * 8u; }
// Row of D that accumulator slot `j` holds; the column is frag_c(lane).
__device__ __forceinline__ uint32_t acc_row(uint32_t lane, uint32_t j) {
    return (lane >> 4) * 8u + j;
}

__device__ __forceinline__ f8v acc_zero() {
    f8v z;
#pragma unroll
    for (int j = 0; j < 8; ++j) z[j] = 0.0f;
    return z;
}

// ---- E4M3 -> f32 -------------------------------------------------------------------------------
//
// One instruction on gfx1201 (v_cvt_f32_fp8); the software fallback is ~11 VALU ops. They agree on
// all bytes except 0x7F and 0xFF, E4M3-fn's NaN encodings: hardware returns NaN, the fallback
// +-480. Both quantisers clamp to +-448 = 0x7E, so real data never holds them.
//
// No subnormal-free fast path: a checkpoint weight may hold subnormals, which such a path would
// decode as 2^-7 * 1.MMM instead of MMM * 2^-9.
__device__ __forceinline__ float e4m3(uint32_t b) {
#if __has_builtin(__builtin_amdgcn_cvt_f32_fp8)
    return __builtin_amdgcn_cvt_f32_fp8(b, 0);
#else
    const uint32_t sign = (b & 0x80u) << 24;
    const uint32_t ex   = (b >> 3) & 0x0Fu;
    const uint32_t mn   = b & 0x07u;
    const float nrm = __uint_as_float(((ex + 120u) << 23) | (mn << 20));
    const float sub = (float)mn * (1.0f / 512.0f);
    return __uint_as_float(__float_as_uint(ex ? nrm : sub) | sign);
#endif
}

// Two E4M3 bytes of a packed dword to two f32 in one instruction (v_cvt_pk_f32_fp8). `HI` selects
// bytes 2,3 over 0,1 and is a template parameter because the builtin's word select is a constant.
// Returns r4d_common.h's `v2f`, the builtin's own return type.
template <bool HI>
__device__ __forceinline__ v2f e4m3_pk(uint32_t q) {
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
    return __builtin_amdgcn_cvt_pk_f32_fp8(q, HI);
#else
    constexpr uint32_t s = HI ? 16u : 0u;
    v2f r;
    r.x = e4m3((q >> s) & 0xFFu);
    r.y = e4m3((q >> (s + 8u)) & 0xFFu);
    return r;
#endif
}

// ---- block scale --------------------------------------------------------------------------------
//
// The checkpoint's BF16 `weight_scale_inv` (one per 128x128 tile, row-major) is read natively:
// decode is one shift and one bitcast. Converting to UE8M0 would round every tile's gain to a
// power of two and force requantising the E4M3 bytes, for no saving (the scale plane is 21.8 KiB
// against an 89 MiB weight at gate_proj's shape). Unlike UE8M0, bf16 needs no zero test:
// `0u << 16` is +0.0f.
__device__ __forceinline__ float bscale(uint16_t s) {
    return __uint_as_float((uint32_t)s << 16);
}
/* The same scale zero-extended in a full register (see u16_load_w in r4d_moe.hip). The upper
 * sixteen bits must be zero. */
__device__ __forceinline__ float bscale(uint32_t s) {
    return __uint_as_float(s << 16);
}

// ---- LDS staging, shared by both fp8a8 instantiations -------------------------------------------
//
// A fragment is 8 B per lane, so a 64 B row holds 8 granules, and a fragment read is one granule
// column over sixteen consecutive rows. LDS has 32 banks of 4 B; a row pitch of P bytes puts row r
// at bank (r * P/4) % 32. With the tile width as pitch (64 or 256 B) rows collide and the read
// serialises.
//
// An XOR granule swizzle `(k >> 3) ^ (row & 7)` only permutes within a row, so it separates at most
// 8 rows: rows r and r+8 still collide and the read takes two bank cycles.
//
// A padded pitch P = FK + 8 separates all sixteen. At FK = 64, P/4 = 18 words and 18r % 32 over
// r = 0..15 is
//
//     0 18 4 22 8 26 12 30 16 2 20 6 24 10 28 14
//
// sixteen distinct even banks, so the read is one bank cycle. At FK = 256, 66r % 32 = 2r % 32,
// likewise distinct. The pad must be a multiple of 8 to keep the granule aligned for ds_read_b64;
// +16 (4r % 32, period 8) collides again. The cost is 8 B per row: the prefill tile is 36 KiB per
// block instead of 32, allowing 3 resident blocks per WGP instead of 4, which the saved bank
// cycle outweighs.
template <uint32_t FK>
__device__ __forceinline__ constexpr uint32_t lds_pitch() { return FK + 8u; }

template <uint32_t FK>
__device__ __forceinline__ i2v frag(const uint8_t* __restrict__ lds, uint32_t row0, uint32_t k0,
                                    uint32_t lane) {
    const uint32_t row = row0 + frag_c(lane);
    const uint32_t k = k0 + frag_k0(lane);              // 8-aligned, so (k & 7) == 0
    const uint2 v = *(const uint2*)(lds + (size_t)row * lds_pitch<FK>() + k);
    i2v f;
    f[0] = (int)v.x;
    f[1] = (int)v.y;
    return f;
}

// The 32 bytes a thread stages, held in registers between the global load and the LDS store.
// Load and store are split so the next load can be issued before the barrier; as one call, a
// thread has a single 32-byte request in flight and the K loop becomes a chain of latencies.
struct Stage8 { uint4 v, w; };

// Fragment `t` (bytes 8t) of a 32-byte fragment-order load: 0 and 1 from `v`, 2 and 3 from `w`.
// `t` is constant after unrolling, so this is a register rename, not a branch.
__device__ __forceinline__ i2v frag_of(const Stage8& s, uint32_t t) {
    const uint2 g[4] = { make_uint2(s.v.x, s.v.y), make_uint2(s.v.z, s.v.w),
                         make_uint2(s.w.x, s.w.y), make_uint2(s.w.z, s.w.w) };
    i2v f;
    f[0] = (int)g[t].x;
    f[1] = (int)g[t].y;
    return f;
}

__device__ __forceinline__ Stage8 stage8_load(const uint8_t* __restrict__ src, bool live) {
    Stage8 r;
    r.v = make_uint4(0, 0, 0, 0);
    r.w = make_uint4(0, 0, 0, 0);
    if (live) {
        r.v = *(const uint4*)src;
        r.w = *(const uint4*)(src + 16);
    }
    return r;
}

template <uint32_t FK>
__device__ __forceinline__ void stage8_store(uint8_t* __restrict__ lds, uint32_t row, uint32_t k0,
                                             Stage8 r) {
    const uint4 v = r.v, w = r.w;
    const uint2 gr[4] = { make_uint2(v.x, v.y), make_uint2(v.z, v.w), make_uint2(w.x, w.y),
                          make_uint2(w.z, w.w) };
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t k = k0 + i * 8;
        *(uint2*)(lds + (size_t)row * lds_pitch<FK>() + k) = gr[i];
    }
}

}  // namespace r4d_fp8
