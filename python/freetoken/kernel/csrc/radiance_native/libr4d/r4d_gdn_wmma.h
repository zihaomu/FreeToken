// R4D: gfx1201 wave32 bf16 WMMA fragment helpers, shared by the GDN kernels.
// Layout (probed on this hardware, not documented by AMD):
//   16-bit A/B : idx = lane%16 , k = 8*(e>>2) + 4*(lane>>4) + (e&3)   ("two runs of four")
//   f32   C/D  : n   = lane%16 , m = 8*(lane>>4) + e
// So A wants [M][K] and B wants [N][K] (both K-contiguous), and D stored to an [N][M] buffer is
// one contiguous 8-element store, while [M][N] costs eight scalar ones.
#pragma once
#include <hip/hip_runtime.h>

typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef float  v8f  __attribute__((ext_vector_type(8)));

// Software f32 -> bf16 (gfx1201 has no convert). Round-half-away (u + 0x8000) is one add where
// RTNE needs bfe + add3 + shift; they differ only when the discarded bits are exactly 0x8000.
__device__ __forceinline__ unsigned short f2bf(float f) {
  return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}
// Pack two floats into one dword of bf16 in 3 instructions (2 adds + v_perm_b32)
// instead of 2 converts plus a shift/or.
__device__ __forceinline__ unsigned f2bf2(float a, float b) {
  unsigned ua = __builtin_bit_cast(unsigned, a) + 0x8000u;
  unsigned ub = __builtin_bit_cast(unsigned, b) + 0x8000u;
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);   // {b_hi16, a_hi16}
}
__device__ __forceinline__ uint2 f2bf4(float a, float b, float c, float d) {
  return make_uint2(f2bf2(a, b), f2bf2(c, d));
}
__device__ __forceinline__ float bf2f(unsigned short h) {
  return __builtin_bit_cast(float, (unsigned)h << 16);
}

// K-contiguous fragment from a row pointer already advanced to this lane's row and half-K.
// Indexed off one base so the loads take immediate offsets (`p + 8` makes LLVM materialise a
// second address register per fragment). The data is two 8-byte groups 16 bytes apart; reading
// them as uint2 gives one ds_load_2addr_b64 instead of two ds_load_2addr_b32. LLVM pairs into b64
// only with provable 8-byte alignment, which holds: row pitches are multiples of 4 shorts and the
// lane's k offset is 4*(lane>>4).
__device__ __forceinline__ v8bf fragP(const unsigned short* p) {
  const uint2* pw = (const uint2*)p;
  union { uint2 w[2]; v8bf f; } u;
  u.w[0] = pw[0]; u.w[1] = pw[2];
  return u.f;
}
// Row pointer for the fragment above: LDS source, `pitch` in elements.
__device__ __forceinline__ const unsigned short* rowP(const unsigned short* src,
                                                      int pitch, int base, int lane) {
  return src + (base + (lane & 15)) * pitch + 4 * (lane >> 4);
}
// 32-bit element offset for a global fragment, split from a uniform base. `lim` clamps the row so a
// partial last chunk cannot read past the tensor (those rows are discarded downstream). A per-lane
// 64-bit row pointer costs ~13 instructions per fragment under register pressure (64-bit stride
// math, exec-mask ternary, rematerialised per load) and can emit flat_load, which on gfx12 bumps
// both loadcnt and dscnt so LDS drains also wait on memory. A 32-bit offset makes the k loop
// global_load_b64 vdst, voff, s[base] with a 13-bit immediate.
__device__ __forceinline__ unsigned rowGo(unsigned rowstride, int row0, int lim,
                                          int lane) {
  int r = row0 + (lane & 15); r = r < lim ? r : lim;
  return (unsigned)r * rowstride + 4u * (unsigned)(lane >> 4);
}
__device__ __forceinline__ v8bf fragGo(const unsigned short* base, unsigned off) {
  const uint2* pw = (const uint2*)(base + off);      // 8-byte aligned: see fragP
  union { uint2 w[2]; v8bf f; } u;
  u.w[0] = pw[0]; u.w[1] = pw[2];
  return u.f;
}
// Fragment from a 32-bit LDS byte offset off the shared-block base. Buffers in a large __shared__
// block are further apart than ds_read2_b32's immediate reach (1020 B), so LLVM rematerialises
// `v_add_nc_u32 vaddr, <buffer_const>, vbase` before every fragment. An opaque offset register
// (LDSOPAQUE) keeps all 8 k steps as immediates (8 dwords apart, 61 max).
__device__ __forceinline__ v8bf fragO(const char* sb, unsigned off) {
  const uint2* pw = (const uint2*)(sb + off);
  union { uint2 w[2]; v8bf f; } u;
  u.w[0] = pw[0]; u.w[1] = pw[2];
  return u.f;
}
#define LDSOPAQUE(x) asm("" : "+v"(x))

// K-contiguous fragment: src[idx][k], row pitch `pitch` (in elements)
__device__ __forceinline__ v8bf frag(const unsigned short* src, int pitch,
                                     int base, int kbase, int lane) {
  return fragP(rowP(src, pitch, base, lane) + kbase);
}

// Same fragment but the source is stored transposed: src[k][idx]. Costs 8 scalar reads.
__device__ __forceinline__ v8bf fragT(const unsigned short* src, int pitch,
                                      int base, int kbase, int lane) {
  int idx = base + (lane & 15);
  int k0 = kbase + 4 * (lane >> 4);
  union { unsigned short h[8]; v8bf f; } u;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    u.h[e]     = src[(size_t)(k0 + e) * pitch + idx];
    u.h[4 + e] = src[(size_t)(k0 + 8 + e) * pitch + idx];
  }
  return u.f;
}

__device__ __forceinline__ v8f mma(v8bf a, v8bf b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// C[mt*16 .. ][nt*16 .. ] += A[M][KA] @ B[N][KA]^T over KSTEP 16-wide steps
template <int KSTEP>
__device__ __forceinline__ v8f gemm(const unsigned short* A, int Ap, int mt,
                                    const unsigned short* B, int Bp, int nt,
                                    int lane, v8f acc) {
#pragma unroll
  for (int s = 0; s < KSTEP; ++s)
    acc = mma(frag(A, Ap, mt * 16, s * 16, lane), frag(B, Bp, nt * 16, s * 16, lane), acc);
  return acc;
}
