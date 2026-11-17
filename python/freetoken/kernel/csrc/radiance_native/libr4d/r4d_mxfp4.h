// Shared device code for the OCP-MXFP4 GEMMs (r4d_gemm_mxfp4a8_nt_m64, _tiled, _decode): the
// folded-magnitude table and the eight-element unpack that reads it. Shared so all three fold every
// weight element to the same e4m3 byte, and a step crossing M bands sees the same weight.
//
// The fold is explained in r4d_gemm_mxfp4a8_nt_m64.hip: d = Wref[n] - E8M0[n][blk] is applied
// through this table, and 2^(Wref[n]-127) once in the epilogue.
#pragma once

#include <hip/hip_runtime.h>

#define R4D_MXFP4_GROUP 32   // K per E8M0 exponent, fixed by the OCP MX format

// kMxfp4Mag[d] is the eight e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} scaled by 2^-d, as packed
// e4m3 bytes (low word, high word), indexed by the 3-bit magnitude through v_perm_b32. Entries are
// the round-to-nearest-even e4m3 of the product, matching the hardware conversion: exact for
// d <= 8; for d >= 9 the ties at 1.5*2^-9 (d=9 mag 1.5, d=10 mag 3, d=11 mag 6) round to 2^-8.
// d >= 13 is zero.
//
// Static: each kernel's code object carries its own copy.
static __device__ __constant__ unsigned int kMxfp4Mag[16][2] = {
  {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u}, {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u}, {0x02010000u, 0x06040302u},
  {0x01000000u, 0x03020201u}, {0x00000000u, 0x02010100u},
  {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
};

// The exponent difference a block is folded at, clamped to the table. d < 0 cannot occur (the
// reference is the row's maximum); d >= 15 is zero.
static __device__ __forceinline__ int r4d_mxfp4_fold_d(int ref, int e8) {
  const int d = ref - e8;
  return d < 0 ? 0 : (d > 15 ? 15 : d);
}

// Eight e2m1 codes (four packed bytes, element 2j in the low nibble of byte j) -> eight e4m3 bytes
// in k order (word 0 = elements 0..3, word 1 = 4..7), with exponent difference `d` folded in. Even
// and odd nibbles are each mapped through the table by a v_perm_b32 on the 3-bit magnitude, bit 3
// is shifted into the e4m3 sign, and two more v_perm_b32 interleave the halves.
//
// `_t` takes a table row (t0, t1) the caller already holds, e.g. from an LDS copy, which keeps the
// lookup off the global load counter the weight and activation loads wait on.
static __device__ __forceinline__ void r4d_mxfp4_unpack8_t(unsigned int wv, unsigned int t0,
                                                           unsigned int t1, unsigned int* lo,
                                                           unsigned int* hi) {
  const unsigned int ev = wv & 0x0F0F0F0Fu;          // low nibble of each byte  -> even elements
  const unsigned int od = (wv >> 4) & 0x0F0F0F0Fu;   // high nibble             -> odd elements
  const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
  const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
  *lo = __builtin_amdgcn_perm(bo, be, 0x05010400u);
  *hi = __builtin_amdgcn_perm(bo, be, 0x07030602u);
}
static __device__ __forceinline__ void r4d_mxfp4_unpack8(unsigned int wv, int d,
                                                         unsigned int* lo, unsigned int* hi) {
  r4d_mxfp4_unpack8_t(wv, kMxfp4Mag[d][0], kMxfp4Mag[d][1], lo, hi);
}
// Row d = 0 as immediates, for kernels that fold the E8M0 scale in F32 and need the exact e2m1
// values: no table load, v_perm_b32 takes the two words as literals. tests/r4d_emu/test_moemx
// holds them to row 0 of the table.
#define R4D_MXFP4_MAG0_LO 0x3c383000u
#define R4D_MXFP4_MAG0_HI 0x4c484440u
static __device__ __forceinline__ void r4d_mxfp4_unpack8_exact(unsigned int wv, unsigned int* lo,
                                                               unsigned int* hi) {
  r4d_mxfp4_unpack8_t(wv, R4D_MXFP4_MAG0_LO, R4D_MXFP4_MAG0_HI, lo, hi);
}

// The activation scale a row's accumulator is carried in, for the per-(row, 128 of K) grid: the
// last non-zero entry of `row` at or before block `kb`, or 1 if none. A zero-scale block has zero
// codes and contributes nothing, so it keeps the previous scale instead of dividing by zero.
static __device__ __forceinline__ float r4d_mxfp4_carried(const float* __restrict__ row, int kb) {
  for (int j = kb; j >= 0; --j) {
    const float s = row[j];
    if (s > 0.f) return s;
  }
  return 1.f;
}
