#pragma once
// r4d_ar_ti8.h: the three-tier INT8 all-reduce wire codec shared by the wide quantized kernels.
// A group's record tags its tier: Tier 1 is plain amax-scaled int8, Tier 2 lifts one dominant
// element out of the scale and ships it exactly in the record, Tier 3 applies an orthonormal
// WHT-32 first. A massive outlier stays exact while ordinary groups take the cheapest path.
// Wire layout per slot, ng = numel/32 groups (fixed size, no escapes, no atomics):
//   codes  u8 [32*ng]  at 0        (uint2 per 8-element lane vector)
//   rec    u32[ng]     at 32*ng
//   scale  fp16[ng]    at 36*ng    -> 38 bytes/group = 9.5 bits/element
// Slot base must be 16B-aligned; 32*ng keeps the u32 records 4B-aligned for any ng.
//
// A group of 32 = TI8_LPG=4 adjacent lanes x TI8_EPL=8 consecutive elements. Cross-lane
// exchanges use masks < 4, so a group stays in its 4-lane cluster and never straddles a wave.
// Kernels partition work in whole groups so a cluster branches together (the shuffles sit inside
// tier branches).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cfloat>

// TI8_-prefixed (unlike the 2-rank header's bare GROUP/EPL) so this header can share a TU with
// other codec headers without macro collisions.
#define TI8_GROUP 32
#define TI8_EPL 8
#define TI8_LPG 4
// Per-group wire layout: codes u8[GROUP] at 0, the u32 record at TI8_REC_OFF, the fp16 scale at
// TI8_SC_OFF.
#define TI8_REC_OFF TI8_GROUP
#define TI8_SC_OFF (TI8_GROUP + 4)
#define TI8_GROUP_BYTES (TI8_SC_OFF + 2)
#define TI8_RLIM 253.0f               // the ratio limit: the int8 zero-line
#define TI8_FLOOR_INV (1.0f / 96.0f)  // floor divisor, influences rotation use rate

__device__ __forceinline__ float ti8_to_f(const float& x) { return x; }
__device__ __forceinline__ float ti8_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float ti8_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void ti8_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void ti8_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// one 16-byte load of a lane's 8 consecutive payload elements, widened to fp32
__device__ __forceinline__ void ti8_load8(const __hip_bfloat16* p, float* out) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const unsigned int w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    out[2 * j] = __uint_as_float((w[j] & 0xFFFFu) << 16);
    out[2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
  }
}
__device__ __forceinline__ void ti8_load8(const __half* p, float* out) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __half2* h = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __half22float2(h[j]);
    out[2 * j] = f.x;
    out[2 * j + 1] = f.y;
  }
}
template <typename T>
__device__ __forceinline__ void ti8_store8(T* p, const float* v) {
  union {
    uint4 u;
    T t[8];
  } o;
#pragma unroll
  for (int i = 0; i < 8; ++i) ti8_from_f(v[i], o.t[i]);
  *reinterpret_cast<uint4*>(p) = o.u;
}

// 16-bit payload-dtype bit pattern for the Tier 2 record value
template <typename T>
__device__ __forceinline__ unsigned short ti8_bits16(float f) {
  union {
    T t;
    unsigned short u;
  } c;
  ti8_from_f(f, c.t);
  return c.u;
}
template <typename T>
__device__ __forceinline__ float ti8_from_bits16(unsigned short u) {
  union {
    T t;
    unsigned short u;
  } c;
  c.u = u;
  return ti8_to_f(c.t);
}

// orthonormal WHT-32 over one group (4 lanes x 8 elements): h=1,2,4 in-register,
// h=8,16 cross-lane; its own inverse (1/sqrt(32) each way). Must be reached by all
// 4 lanes of the group together.
__device__ __forceinline__ void ti8_wht32(float* v, int lig) {
#pragma unroll
  for (int h = 1; h < TI8_EPL; h <<= 1)
#pragma unroll
    for (int i = 0; i < TI8_EPL; ++i)
      if ((i & h) == 0) {
        const float x = v[i], y = v[i ^ h];
        v[i] = x + y;
        v[i ^ h] = x - y;
      }
#pragma unroll
  for (int m = 1; m < TI8_LPG; m <<= 1) {
    const bool lo = (lig & m) == 0;
#pragma unroll
    for (int i = 0; i < TI8_EPL; ++i) {
      const float o = __shfl_xor(v[i], m);
      v[i] = lo ? (v[i] + o) : (o - v[i]);
    }
  }
  const float nrm = 0.17677669529663687f;               // 1/sqrt(32)
#pragma unroll
  for (int i = 0; i < TI8_EPL; ++i) v[i] *= nrm;
}

// Encode one lane's 8 elements of a group: this lane's 8 int8 codes, plus the group record and
// scale (identical on all 4 lanes; the lig==0 lane stores them). All 4 lanes of the group must
// call together (cross-lane classifier and Tier 3 rotation); the caller fans the words out to every
// destination slot. The _v form consumes values already in registers (the phase-2 reduced vector):
// Tier 3 rotates v in place, so a caller wanting the reconstruction must decode the wire words as
// every peer does.
template <typename T>
__device__ __forceinline__ void ti8_encode_group_v(float* v, int lig, uint2* codes,
                                                   unsigned int* rec, __half* scale) {
  // classifier: max1(+idx,+value), max2, sumsq in one scan, 2-shuffle merges
  float m1 = -1.f, m2 = 0.f, sumsq = 0.f;
  int li = 0;
#pragma unroll
  for (int i = 0; i < TI8_EPL; ++i) {
    const float a = fabsf(v[i]);
    sumsq += v[i] * v[i];
    if (a > m1) { m2 = m1; m1 = a; li = i; }
    else if (a > m2) m2 = a;
  }
  int gidx = lig * TI8_EPL + li;
  float mval = v[li];
#pragma unroll
  for (int m = 1; m < TI8_LPG; m <<= 1) {
    const float om1 = __shfl_xor(m1, m);
    const float om2 = __shfl_xor(m2, m);
    const float osq = __shfl_xor(sumsq, m);
    const int ogidx = __shfl_xor(gidx, m);
    const float omval = __shfl_xor(mval, m);
    m2 = fmaxf(fminf(m1, om1), fmaxf(m2, om2));
    if (om1 > m1) { m1 = om1; gidx = ogidx; mval = omval; }
    sumsq += osq;
  }
  const float rms_rest = sqrtf(fmaxf(0.f, sumsq - m1 * m1) * (1.0f / 31.0f));
  const float delta = rms_rest * TI8_FLOOR_INV;
  float mn = FLT_MAX;
#pragma unroll
  for (int i = 0; i < TI8_EPL; ++i) {
    const float a = fabsf(v[i]);
    if (a >= delta && a > 0.f && a < mn) mn = a;
  }
#pragma unroll
  for (int m = 1; m < TI8_LPG; m <<= 1) mn = fminf(mn, __shfl_xor(mn, m));

  // tier choice; degenerate groups (all zero / nothing significant) hit Tier 1
  float amax;
  if (m1 <= TI8_RLIM * mn) {
    amax = m1;
    *rec = 0u;
  } else if (m2 <= TI8_RLIM * mn) {
    amax = m2;
    *rec = 1u | ((unsigned int)gidx << 4) |
           ((unsigned int)ti8_bits16<T>(mval) << 16);
  } else {
    ti8_wht32(v, lig);
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < TI8_EPL; ++i) a = fmaxf(a, fabsf(v[i]));
#pragma unroll
    for (int m = 1; m < TI8_LPG; m <<= 1) a = fmaxf(a, __shfl_xor(a, m));
    amax = a;
    *rec = 2u;
  }
  *scale = __float2half(amax * (1.0f / 127.0f));
  float s = __half2float(*scale);
  if (!(s > 0.f)) { s = 1.0f; *scale = __float2half(1.0f); }
  const float inv = 1.0f / s;
  unsigned int w[2] = {0u, 0u};
#pragma unroll
  for (int i = 0; i < TI8_EPL; ++i) {
    int qi = (int)rintf(v[i] * inv);
    qi = (qi < -127) ? -127 : ((qi > 127) ? 127 : qi);
    w[i >> 2] |= ((unsigned int)(unsigned char)(signed char)qi) << ((i & 3) << 3);
  }
  *codes = make_uint2(w[0], w[1]);
}

template <typename T>
__device__ __forceinline__ void ti8_encode_group(const T* p, int lig, uint2* codes,
                                                 unsigned int* rec, __half* scale) {
  float v[TI8_EPL];
  ti8_load8(p, v);
  ti8_encode_group_v<T>(v, lig, codes, rec, scale);
}

// Decode one lane's 8 elements back to the original domain (dequant, then the Tier 2
// patch or the Tier 3 inverse rotation per the tag). Group-uniform tags keep the Tier 3
// shuffles convergent.
template <typename T>
__device__ __forceinline__ void ti8_decode_group(uint2 codes, unsigned int rec, float s,
                                                 int lig, float* v) {
  const unsigned int w[2] = {codes.x, codes.y};
#pragma unroll
  for (int i = 0; i < TI8_EPL; ++i) {
    const signed char n = (signed char)((w[i >> 2] >> ((i & 3) << 3)) & 0xFFu);
    v[i] = (float)n * s;
  }
  const unsigned int tag = rec & 3u;
  if (tag == 2u) {
    ti8_wht32(v, lig);
  } else if (tag == 1u) {
    const int oi = (int)((rec >> 4) & 31u);
    if (oi / TI8_EPL == lig)
      v[oi % TI8_EPL] = ti8_from_bits16<T>((unsigned short)(rec >> 16));
  }
}
