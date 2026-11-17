// r4d_ar_ln_had_quant_i8.hip -- r4d_ar_ln_had_quant_i8: the tensor-parallel all-reduce plus the
// residual add, RMS norm, block Hadamard and int8 quantiser that always follow it, in one kernel.
//
// Every row-parallel linear is followed by an all-reduce and then a norm (128 of each on a
// 64-layer decode step):
//
//     out_proj / o_proj / down_proj  ->  all_reduce  ->  (residual add, RMS norm, rotate, quantise)
//
// and the norm's output is the next quantised linear's A. Decode is launch-bound (~2.3 us gap per
// dispatch), and the standalone norm is latency-bound (4.1 us for an 80 KiB row set against a
// ~2.6 us launch floor). Fusing removes 128 dispatches and most of the norm's fixed cost. The
// all-reduce output is never materialised: the peer's contribution is a second operand on the
// norm's load, `t = mine + peer + residual`.
//
// Partition by row. r4d_ar_oneshot_2rank_exact gives each block a chunk of 16B words and an
// independent flag with no grid barrier. A norm needs whole rows for its sum of squares, but rows
// are contiguous, so giving block b a contiguous row range makes its words exactly those rows'
// bytes. Both ranks derive the range from (M, gridDim.x), so the handshake stays per-block with no
// grid barrier. This caps nb at M, which costs nothing here: the standalone all-reduce uses 4
// blocks on an 80 KiB decode message and M is 8 or more.
//
// Below the handshake this is r4d_rmsnorm_had_quant_i8: one pass (the FWHT is linear and inv_rms
// is a per-row scalar, so the rotation runs before the norm closes and the scale absorbs it),
// register staging, `wadd` inside the fp32 arithmetic for GemmaRMSNorm's `x_hat * (1 + w)`, and an
// optional unrotated bf16 output for a GDN block's bf16 in_proj_ba. The differences: the row is
// loaded from two places, and LDS is reused across a block's rows, which costs one more barrier
// per row.
#include "r4d.h"
#include "r4d_ar_seq.h"
#include "r4d_common.h"
#include "r4d_fwht.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

#define R4D_ARN_SPIN_MAX 4000000000ULL
#define R4D_ARN_MAXVALS 72

__device__ __forceinline__ void arn_store_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int arn_load_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// ---------------------------------------------------------------------------------------------
// Rotated 6-bit wire: same format and arithmetic as r4d_ar_oneshot_2rank_wht6 (orthonormal
// Walsh-Hadamard over groups of 64, amax-scaled, 6 uniform bits plus a bf16 scale), but packed in
// chunks of ARN_CE elements instead of 2048 so a chunk never straddles a row. K is a whole number
// of chunks, so a block's rows [r0, r1) are too, and the row partition holds.
//
// Above ~200 KiB the message is bandwidth-bound and the compressed payload is 2.6x smaller, which
// outweighs the saved launch; carrying it here gets both savings.
//
// The helpers are duplicated from r4d_ar_oneshot_2rank_wht6.h on purpose: that header also holds
// the standalone kernel and launcher, which would be instantiated again in this unit.
#define ARN_GROUP 64
#define ARN_BITS 6
#define ARN_EPL 8
#define ARN_LPG (ARN_GROUP / ARN_EPL)                   // 8 lanes span one group
#define ARN_WV 32
#define ARN_GPW (ARN_WV * ARN_EPL / ARN_GROUP)          // 4 groups per wave-iteration
// Groups per chunk, the wire's one tuning knob: 8 here against the standalone kernel's 32. A wave
// owns a whole chunk, so chunk size sets how many waves have work (a 5120-group message is 160
// chunks at 32, 640 at 8). More blocks cannot substitute, since each pays its own flag round trip
// (24 -> 32 blocks costs ~3%, 64 ~4%, 160 ~24%). The wire's cost falls monotonically from 32 groups
// to 8.
#define ARN_CG 8
#define ARN_CE (ARN_CG * ARN_GROUP)                     // elements per chunk (512 at ARN_CG 8)
#define ARN_ITERS (ARN_CE / (ARN_WV * ARN_EPL))         // wave-iterations per chunk
#define ARN_NWORD (ARN_ITERS * ARN_EPL * ARN_BITS / 32) // dwords a lane holds per chunk
#define ARN_CBYTES (ARN_CE * ARN_BITS / 8)              // wire bytes per chunk
#define ARN_QMAX ((1 << (ARN_BITS - 1)) - 1)
#define ARN_QMIN (-(1 << (ARN_BITS - 1)))
#define ARN_MASK ((1u << ARN_BITS) - 1u)

// RTNE to bf16 and back: the scale ships as its top 16 bits, and the sender's own copy must equal
// what the receiver reconstructs.
__device__ __forceinline__ float arn_bf16_round(float v) {
  unsigned u = __float_as_uint(v);
  const unsigned lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return __uint_as_float(u & 0xffff0000u);
}


// `lo ? (v + o) : (o - v)` written as `o + (lo ? v : -v)`. The sign flip is one XOR with a mask
// uniform over the stage: two instructions per element (xor, DPP-sourced add) instead of add,
// subtract and v_cndmask. Bit-identical, since IEEE negation is exact and addition commutes.
template <int M>
__device__ __forceinline__ void arn_xstage(float* v, int lig) {
  const unsigned int sm = (lig & M) ? 0x80000000u : 0u;
#pragma unroll
  for (int i = 0; i < ARN_EPL; ++i) {
    const float o = r4d_xor_lane<M>(v[i]);
    v[i] = __int_as_float(__float_as_int(v[i]) ^ sm) + o;
  }
}

// Orthonormal Walsh-Hadamard over one group; its own inverse, so this routine also undoes it.
// A group is ARN_LPG lanes x ARN_EPL elements: the in-register butterflies are free ALU, and with
// ROW_XMASK the cross-lane stages are free too.
__device__ __forceinline__ void arn_wht(float* v, int lig) {
  static_assert(ARN_LPG == 8, "the cross-lane stages are unrolled for a group of 8 lanes");
#pragma unroll
  for (int h = 1; h < ARN_EPL; h <<= 1)
#pragma unroll
    for (int i = 0; i < ARN_EPL; ++i)
      if ((i & h) == 0) { const float x = v[i], y = v[i ^ h]; v[i] = x + y; v[i ^ h] = x - y; }
  arn_xstage<1>(v, lig);
  arn_xstage<2>(v, lig);
  arn_xstage<4>(v, lig);
  const float nrm = rsqrtf((float)ARN_GROUP);
#pragma unroll
  for (int i = 0; i < ARN_EPL; ++i) v[i] *= nrm;
}

template <int HAD, int THR, bool WOUT, int WIRE>
__global__ __launch_bounds__(THR) void r4d_ar_ln_had_quant_i8_kernel(
    const __hip_bfloat16* __restrict__ X, __hip_bfloat16* __restrict__ RES,
    const __hip_bfloat16* __restrict__ Wg, __hip_bfloat16* __restrict__ OUTB,
    signed char* __restrict__ Q, float* __restrict__ S,
    int4* peer_base, const int4* my_base,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int M, int K, int slot_stride16, int drain, int acq,
    float eps, float wadd, float rnorm, unsigned int nbw, long stage) {
  constexpr int VPL = HAD / 32, NW = THR / 32;
  constexpr int MAXB = R4D_ARN_MAXVALS / VPL;
  extern __shared__ float lds[];
  // One staging slice per owned row on the compressed wire (the exact wire needs one, since the
  // norm reads rows from memory). The compressed wire must reduce first; staging all rows makes
  // the reduce one pass over the block's chunks behind one barrier instead of one per row.
  const int r0_ = (int)(((long)blockIdx.x * M) / gridDim.x);
  const int nrows = (int)(((long)(blockIdx.x + 1) * M) / gridDim.x) - r0_;
  float* red = lds + (size_t)(WIRE ? nrows : 1) * K;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x;
  const int lane = tid & 31, wave = tid >> 5;
  const int nblk = K / HAD;
  // Contiguous row range. Integer arithmetic only, so both ranks land on the same split.
  const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);

  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int sq = s_seq;
  int4* ps = peer_base + (size_t)s_slot * slot_stride16;
  const int4* ms = my_base + (size_t)s_slot * slot_stride16;
  const long md = r4d_archk_md(stage, my_base);   // the host mailbox's VRAM mirror (r4d_ar_seq.h)

  // WIRE == 6 only. The slot is split as in the standalone compressed all-reduce: packed symbols
  // from the base, bf16 group scales from halfway. Chunk indices are global over the [M, K]
  // message, so both ranks address the same bytes; a block owns the chunks its rows cover.
  const int cpr = K / ARN_CE;                    // chunks a row spans (K % ARN_CE == 0, checked)
  const int c00 = r0 * cpr, cpb = (r1 - r0) * cpr;
  char* p_pay = (char*)ps;
  const char* m_pay = (const char*)ms;
  unsigned short* p_sc = (unsigned short*)((char*)ps + (size_t)(slot_stride16 >> 1) * 16);
  const unsigned short* m_sc =
      (const unsigned short*)((const char*)ms + (size_t)(slot_stride16 >> 1) * 16);
  // This rank's own packed half, kept in LDS so the reduce folds exactly the bytes it sent without
  // a global round trip (the standalone kernel uses a side buffer instead).
  unsigned int* Lpk = (unsigned int*)(red + 2 * NW);
  float* Lsc = (float*)(Lpk + (size_t)cpb * ARN_WV * ARN_NWORD);
  const int lig = lane & (ARN_LPG - 1), gw = lane / ARN_LPG;

  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)
  if constexpr (WIRE == 0) {
    // Push my rows to the peer slot. A row is K/8 16B words and rows are contiguous, so a row
    // range is a word range.
    const int wpr = K >> 3;
    const int wstart = r0 * wpr, wend = r1 * wpr;
    const int4* in4 = (const int4*)X;
    int k = wstart + tid;
    for (; k + 3 * THR < wend; k += 4 * THR) {
      int4 v0 = in4[k], v1 = in4[k + THR], v2 = in4[k + 2 * THR], v3 = in4[k + 3 * THR];
      ps[k] = v0; ps[k + THR] = v1; ps[k + 2 * THR] = v2; ps[k + 3 * THR] = v3;
      hck += r4d_archk_u4(v0, 4u * (unsigned)k) + r4d_archk_u4(v1, 4u * (unsigned)(k + THR)) +
             r4d_archk_u4(v2, 4u * (unsigned)(k + 2 * THR)) + r4d_archk_u4(v3, 4u * (unsigned)(k + 3 * THR));
    }
    for (; k < wend; k += THR) { const int4 v = in4[k]; ps[k] = v; hck += r4d_archk_u4(v, 4u * (unsigned)k); }
  } else {
    // Rotate, amax, quantise and pack straight into the peer's slot, keeping the same words in
    // LDS. A wave owns whole chunks, so every lane emits exactly ARN_NWORD dwords per chunk and the
    // flush is lane-uniform.
    for (int ci = wave; ci < cpb; ci += NW) {
      const int c = c00 + ci;
      unsigned int acc[ARN_NWORD];
#pragma unroll
      for (int w = 0; w < ARN_NWORD; ++w) acc[w] = 0u;
#pragma unroll
      for (int t = 0; t < ARN_ITERS; ++t) {
        const int gl = t * ARN_GPW + gw;                 // group within the chunk
        const long off = (long)(c * ARN_CG + gl) * ARN_GROUP + (long)lig * ARN_EPL;
        float v[ARN_EPL];
#pragma unroll
        for (int i = 0; i < ARN_EPL; ++i) v[i] = __bfloat162float(X[off + i]);
        arn_wht(v, lig);
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < ARN_EPL; ++i) a = fmaxf(a, fabsf(v[i]));
        a = fmaxf(a, r4d_xor_lane<1>(a));
        a = fmaxf(a, r4d_xor_lane<2>(a));
        a = fmaxf(a, r4d_xor_lane<4>(a));
        float scl = (a > 0.f) ? arn_bf16_round(a * (1.0f / (float)ARN_QMAX)) : 1.0f;
        if (!(scl > 0.f)) scl = 1.0f;
        if (lig == 0) {
          const unsigned short sc16 = (unsigned short)(__float_as_uint(scl) >> 16);
          p_sc[c * ARN_CG + gl] = sc16;
          Lsc[(size_t)ci * ARN_CG + gl] = scl;
          hck += r4d_archk_mix(sc16, R4D_ARCHK_U16 + (unsigned)(c * ARN_CG + gl));
        }
        const float inv = 1.0f / scl;
#pragma unroll
        for (int i = 0; i < ARN_EPL; ++i) {
          const unsigned int sym =
              (unsigned int)(min(max((int)rintf(v[i] * inv), ARN_QMIN), ARN_QMAX) - ARN_QMIN);
          const int bit = (t * ARN_EPL + i) * ARN_BITS, wi = bit >> 5, bo = bit & 31;
          acc[wi] |= sym << bo;
          if (bo + ARN_BITS > 32) acc[wi + 1] |= sym >> (32 - bo);
        }
      }
      // Flush lane-contiguously region by region, widest first, so each region is one coalesced
      // store of ARN_WV x width bytes (ARN_NWORD is 3).
      char* pb = p_pay + (size_t)c * ARN_CBYTES;
      const unsigned int cd = (unsigned)c * (ARN_CBYTES / 4);   // the chunk's first dword
#pragma unroll
      for (int w4 = 0; w4 + 4 <= ARN_NWORD; w4 += 4) {
        *(uint4*)(pb + w4 * 4 * ARN_WV + lane * 16) =
            make_uint4(acc[w4], acc[w4 + 1], acc[w4 + 2], acc[w4 + 3]);
        hck += r4d_archk_u4(make_uint4(acc[w4], acc[w4 + 1], acc[w4 + 2], acc[w4 + 3]),
                            cd + (unsigned)(w4 * ARN_WV + lane * 4));
      }
      if constexpr (ARN_NWORD % 4 >= 2) {
        *(uint2*)(pb + (ARN_NWORD & ~3) * 4 * ARN_WV + lane * 8) =
            make_uint2(acc[ARN_NWORD & ~3], acc[(ARN_NWORD & ~3) + 1]);
        const unsigned int e = cd + (unsigned)((ARN_NWORD & ~3) * ARN_WV + lane * 2);
        hck += r4d_archk_mix(acc[ARN_NWORD & ~3], e) + r4d_archk_mix(acc[(ARN_NWORD & ~3) + 1], e + 1u);
      }
      if constexpr (ARN_NWORD % 2 == 1) {
        *(unsigned int*)(pb + (ARN_NWORD - 1) * 4 * ARN_WV + lane * 4) = acc[ARN_NWORD - 1];
        hck += r4d_archk_mix(acc[ARN_NWORD - 1], cd + (unsigned)((ARN_NWORD - 1) * ARN_WV + lane));
      }
#pragma unroll
      for (int w = 0; w < ARN_NWORD; ++w)
        Lpk[((size_t)ci * ARN_NWORD + w) * ARN_WV + lane] = acc[w];
    }
  }
  r4d_archk_put(hck, peer_flags, nbw, s_slot, (unsigned)b, sq);
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    arn_store_rel(&peer_flags[b], sq);
    unsigned long long z = 0;
    // A handshake that cannot complete is a lockstep violation; trap rather than use stale bytes.
    while (r4d_ar_seq_behind(arn_load_acq(&my_flags[b]), sq)) {
      if (++z > R4D_ARN_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  r4d_archk_wait(my_flags, nbw, s_slot, (unsigned)b, sq, [&](unsigned int& h) {
    if constexpr (WIRE == 0) {
      const int wpr = K >> 3;
      for (int k = r0 * wpr + tid; k < r1 * wpr; k += THR)
        h += r4d_archk_u4(r4d_archk_keep(&ms[k], md), 4u * (unsigned)k);
    } else {
      for (int ci = wave; ci < cpb; ci += NW) {
        const int c = c00 + ci;
        const char* pb = m_pay + (size_t)c * ARN_CBYTES;
        const unsigned int cd = (unsigned)c * (ARN_CBYTES / 4);
#pragma unroll
        for (int w4 = 0; w4 + 4 <= ARN_NWORD; w4 += 4)
          h += r4d_archk_u4(r4d_archk_keep((const uint4*)(pb + w4 * 4 * ARN_WV + lane * 16), md),
                            cd + (unsigned)(w4 * ARN_WV + lane * 4));
        if constexpr (ARN_NWORD % 4 >= 2) {
          const uint2 u =
              r4d_archk_keep((const uint2*)(pb + (ARN_NWORD & ~3) * 4 * ARN_WV + lane * 8), md);
          const unsigned int e = cd + (unsigned)((ARN_NWORD & ~3) * ARN_WV + lane * 2);
          h += r4d_archk_mix(u.x, e) + r4d_archk_mix(u.y, e + 1u);
        }
        if constexpr (ARN_NWORD % 2 == 1)
          h += r4d_archk_mix(
              r4d_archk_keep((const unsigned int*)(pb + (ARN_NWORD - 1) * 4 * ARN_WV + lane * 4), md),
                             cd + (unsigned)((ARN_NWORD - 1) * ARN_WV + lane));
#pragma unroll
        for (int t = 0; t < ARN_ITERS; ++t) {
          const int gl = t * ARN_GPW + gw;
          if (lig == 0)
            h += r4d_archk_mix(r4d_archk_keep(&m_sc[c * ARN_CG + gl], md),
                               R4D_ARCHK_U16 + (unsigned)(c * ARN_CG + gl));
        }
      }
    }
  });

  ms = r4d_archk_moved(ms, md);
  m_pay = r4d_archk_moved(m_pay, md);
  m_sc = r4d_archk_moved(m_sc, md);
  // The peer's rows are now in my scratch, in the same [M, K] layout as X.
  const __hip_bfloat16* P = (const __hip_bfloat16*)ms;

  if constexpr (WIRE == 6) {
    // Reduce: unpack both halves of every owned chunk, add in fp32, inverse-rotate, and leave the
    // result in the norm's staging slice. The norm reads it where it would read `x + p` and
    // overwrites it with its rotated result; one lane owns a slot in both passes, so the aliasing
    // is safe.
    for (int ci = wave; ci < cpb; ci += NW) {
      const int c = c00 + ci;
      const char* pb = m_pay + (size_t)c * ARN_CBYTES;
      unsigned int pac[ARN_NWORD], lac[ARN_NWORD];
#pragma unroll
      for (int w4 = 0; w4 + 4 <= ARN_NWORD; w4 += 4) {
        const uint4 pq = *(const uint4*)(pb + w4 * 4 * ARN_WV + lane * 16);
        pac[w4] = pq.x; pac[w4 + 1] = pq.y; pac[w4 + 2] = pq.z; pac[w4 + 3] = pq.w;
      }
      if constexpr (ARN_NWORD % 4 >= 2) {
        const uint2 pu = *(const uint2*)(pb + (ARN_NWORD & ~3) * 4 * ARN_WV + lane * 8);
        pac[ARN_NWORD & ~3] = pu.x; pac[(ARN_NWORD & ~3) + 1] = pu.y;
      }
      if constexpr (ARN_NWORD % 2 == 1)
        pac[ARN_NWORD - 1] = *(const unsigned int*)(pb + (ARN_NWORD - 1) * 4 * ARN_WV + lane * 4);
#pragma unroll
      for (int w = 0; w < ARN_NWORD; ++w)
        lac[w] = Lpk[((size_t)ci * ARN_NWORD + w) * ARN_WV + lane];
#pragma unroll
      for (int t = 0; t < ARN_ITERS; ++t) {
        const int gl = t * ARN_GPW + gw;
        const float psc = __uint_as_float((unsigned int)m_sc[c * ARN_CG + gl] << 16);
        const float lsc = Lsc[(size_t)ci * ARN_CG + gl];
        float v[ARN_EPL];
#pragma unroll
        for (int i = 0; i < ARN_EPL; ++i) {
          const int bit = (t * ARN_EPL + i) * ARN_BITS, wi = bit >> 5, bo = bit & 31;
          unsigned int psy = pac[wi] >> bo, lsy = lac[wi] >> bo;
          if (bo + ARN_BITS > 32) {
            psy |= pac[wi + 1] << (32 - bo);
            lsy |= lac[wi + 1] << (32 - bo);
          }
          // The ranks fold the same pair in opposite order; fp32 add commutes, so they stay
          // bit-identical as long as neither product becomes an FMA (libr4d/CMakeLists.txt passes
          // -ffp-contract=off).
          v[i] = (float)((int)(psy & ARN_MASK) + ARN_QMIN) * psc
               + (float)((int)(lsy & ARN_MASK) + ARN_QMIN) * lsc;
        }
        arn_wht(v, lig);
        const int e0 = (int)((long)(c * ARN_CG + gl) * ARN_GROUP + (long)lig * ARN_EPL
                             - (long)r0 * K);
        // Rounded to bf16, as the unfused compressed all-reduce stores its sum and the norm reads
        // it back, so this matches the pair exactly.
#pragma unroll
        for (int i = 0; i < ARN_EPL; ++i)
          lds[e0 + i] = __bfloat162float(__float2bfloat16(v[i]));
      }
    }
    __syncthreads();
  }

  for (int row = r0; row < r1; ++row) {
    float* rl = lds + (size_t)(WIRE ? (row - r0) : 0) * K;
    const __hip_bfloat16* x = X + (size_t)row * K;
    const __hip_bfloat16* p = P + (size_t)row * K;
    __hip_bfloat16* res = RES ? RES + (size_t)row * K : nullptr;
    (void)x; (void)p;
    float u[WOUT ? MAXB : 1][VPL];
    float ss = 0.f, m = 0.f;
    int nbk = 0;
    for (int blk = wave; blk < nblk; blk += NW, nbk++) {
      const int base = blk * HAD + lane * VPL;
      float v[VPL];
#pragma unroll
      for (int j = 0; j < VPL; j++) {
        // WIRE == 0 rounds to bf16 on purpose: the standalone all-reduce stores its fp32 sum as
        // bf16 and the norm reads that, so skipping the rounding would change the token stream.
        // Bit-identity with the unfused pair is preferred over the extra accuracy; the software
        // RTNE (gfx1201 has no bf16 convert) is cheap on this launch-bound kernel. WIRE == 6 reads
        // the reduced value from LDS, already rounded.
        float t;
        if constexpr (WIRE == 0) {
          t = __bfloat162float(__float2bfloat16(
              __bfloat162float(x[base + j]) + __bfloat162float(p[base + j])));
        } else {
          t = rl[base + j];
        }
        if (res) {
          t += __bfloat162float(res[base + j]);
          res[base + j] = __float2bfloat16(t);
        }
        ss += t * t;
        v[j] = t * (__bfloat162float(Wg[base + j]) + wadd);
        if (WOUT) u[nbk][j] = v[j];
      }
      r4d_fwht_wave<HAD>(v, lane);
#pragma unroll
      for (int j = 0; j < VPL; j++) {
        rl[base + j] = v[j];
        m = fmaxf(m, fabsf(v[j]));
      }
    }
    ss = r4d_wave_sum(ss);
    m = r4d_wave_max(m);
    if (lane == 0) { red[wave] = ss; red[NW + wave] = m; }
    __syncthreads();
    ss = 0.f; m = 0.f;
#pragma unroll
    for (int i = 0; i < NW; i++) { ss += red[i]; m = fmaxf(m, red[NW + i]); }
    const float inv_rms = rsqrtf(ss / (float)K + eps);
    m = fmaxf(m, 1e-8f);
    const float inv = 127.f / m;
    if (tid == 0) S[row] = m * inv_rms * rnorm / 127.f;

    if (WOUT) {
      nbk = 0;
      for (int blk = wave; blk < nblk; blk += NW, nbk++) {
        const int base = blk * HAD + lane * VPL;
#pragma unroll
        for (int j = 0; j < VPL; j++)
          OUTB[(size_t)row * K + base + j] = __float2bfloat16(u[nbk][j] * inv_rms);
      }
    }

    signed char* q = Q + (size_t)row * K;
    for (int i = tid; i < K; i += THR) {
      const int s = i >> 4, j = i & 15;
      const int sr = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
      const int t = __float2int_rn(rl[sr] * inv);
      q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
    }
    // lds and red are reused by the block's next row.
    __syncthreads();
  }
}

// LDS cost of the compressed wire, so the caller can choose a block count. A block owns whole
// rows and holds, per row, a staging slice at `stage_bits` bits per element plus the packed half
// of its own chunks; a block count that does not fit is refused.
void r4d_ar_ln_wire_dims(int* stage_bits, int* chunk_elems, int* chunk_bytes, int* scale_bytes) {
  // 32: reduced values are staged as fp32 in the norm's own array, so staging costs no extra LDS.
  // bf16 staging would fit more rows per block, but only lowers the block count, which this
  // kernel does not want; at the row counts it runs, the LDS is the same.
  if (stage_bits)  *stage_bits  = 32;
  if (chunk_elems) *chunk_elems = ARN_CE;
  if (chunk_bytes) *chunk_bytes = ARN_WV * ARN_NWORD * (int)sizeof(unsigned int);
  if (scale_bytes) *scale_bytes = ARN_CG * (int)sizeof(float);
}

// LDS per launch: the norm's fp32 staging (one row on the exact wire, every owned row on the
// compressed one), the two reduction planes, and on the compressed wire this rank's packed half of
// every owned chunk. Shared by the launcher and r4d_ar_ln_blocks_min so they agree.
static size_t ar_ln_lds_bytes(int rows_max, int K, int thr, int wire) {
  size_t b = (size_t)(wire ? rows_max : 1) * K * sizeof(float) + 2 * (thr / 32) * sizeof(float);
  if (wire) {
    const size_t cpb = (size_t)rows_max * (K / ARN_CE);
    b += cpb * ARN_WV * ARN_NWORD * sizeof(unsigned int) + cpb * ARN_CG * sizeof(float);
  }
  return b;
}
// Dynamic LDS ceiling: a workgroup's 64 KiB less the kernel's static words, which the runtime places
// in front of it: seq and slot, and the push check's (r4d_ar_seq.h: two 32-word sums and a flag,
// 260 bytes), rounded up.
static constexpr size_t kArnLdsCap = 65536 - 16 - 512;

// Fewest blocks at which this shape fits: each block owns ceil(M / nb) rows, and on the compressed
// wire each row costs a staging slice and its chunks' packed half. -1 when no block count fits (a
// row too wide to stage alone, or for the register staging), where the launcher would throw.
int r4d_ar_ln_blocks_min(int M, int K, int had, int nthreads, int wire) {
  if (M < 1 || K < 1 || had < 32 || K % had) return -1;
  if (wire != 0 && wire != 6) return -1;
  if (wire && K % ARN_CE) return -1;
  const int thr = nthreads ? nthreads : r4d_act_threads(M, K);
  if (thr < 32) return -1;
  if ((K / had + thr / 32 - 1) / (thr / 32) * (had / 32) > R4D_ARN_MAXVALS) return -1;
  if (!wire) return ar_ln_lds_bytes(1, K, thr, 0) <= kArnLdsCap ? 1 : -1;
  const size_t fixed = ar_ln_lds_bytes(0, K, thr, wire);
  const size_t row = ar_ln_lds_bytes(1, K, thr, wire) - fixed;
  if (fixed + row > kArnLdsCap) return -1;
  const long fit = (long)((kArnLdsCap - fixed) / row);
  return (int)(((long)M + fit - 1) / fit);
}

void r4d_ar_ln_had_quant_i8(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                            long seq_ctrs, long slot_stride16,
                            long x, long residual, long wgain, long out_bf16, long q, long s,
                            int M, int K, int had, float eps, float wadd,
                            long stream, int nblocks, int nthreads, int drain, int acq, int wire,
                            long stage) {
  if (K % 16) throw std::runtime_error("r4d ar_ln_had_quant_i8: K must be a multiple of 16");
  if (M < 1) throw std::runtime_error("r4d ar_ln_had_quant_i8: M must be positive");
  if (had < 2 || (had & (had - 1))) throw std::runtime_error(
      "r4d ar_ln_had_quant_i8: had must be a power of two");
  if (K % had) throw std::runtime_error(
      "r4d ar_ln_had_quant_i8: had must divide K -- a rotation block cannot straddle a "
      "tensor-parallel shard boundary");
  int nb = nblocks < 1 ? 1 : nblocks;
  if (nb > M) nb = M;                       // row-aligned: a block owns whole rows
  const int nbw = r4d_ar_max_blocks();      // the counter capacity; see r4d_ar_seq.h
  if (nb > nbw) nb = nbw;
  const int thr = nthreads ? nthreads : r4d_act_threads(M, K);
  if ((K / had + thr / 32 - 1) / (thr / 32) * (had / 32) > R4D_ARN_MAXVALS)
    throw std::runtime_error(
        "r4d ar_ln_had_quant_i8: K too large for the register staging at this width");
  const int rows_max = (M + nb - 1) / nb;
  if (wire) {
    if (wire != 6) throw std::runtime_error("r4d ar_ln_had_quant_i8: wire must be 0 or 6");
    // A chunk may not straddle a row, or the ranks would disagree on byte ownership.
    if (K % ARN_CE) throw std::runtime_error(
        "r4d ar_ln_had_quant_i8: the 6-bit wire needs K to be a multiple of " +
        std::to_string(ARN_CE));
  }
  const size_t shmem = ar_ln_lds_bytes(rows_max, K, thr, wire);
  if (shmem > kArnLdsCap) throw std::runtime_error(
      "r4d ar_ln_had_quant_i8: the rows a block owns do not fit LDS at this block count");
  const float rn = 1.f / sqrtf((float)had);
#define R4D_ARN_LAUNCH(H, T, W, R) \
    hipLaunchKernelGGL((r4d_ar_ln_had_quant_i8_kernel<H, T, W, R>), \
    dim3(nb), dim3(T), shmem, (hipStream_t)stream, (const __hip_bfloat16*)x, \
    (__hip_bfloat16*)residual, (const __hip_bfloat16*)wgain, (__hip_bfloat16*)out_bf16, \
    (signed char*)q, (float*)s, (int4*)peer_scratch, (const int4*)my_scratch, \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, M, K, (int)slot_stride16, \
    drain, acq, eps, wadd, rn, (unsigned)nbw, stage)
#define R4D_ARN_T(H, W, R) switch (thr) { \
    case 128: R4D_ARN_LAUNCH(H, 128, W, R); break; \
    case 256: R4D_ARN_LAUNCH(H, 256, W, R); break; \
    case 512: R4D_ARN_LAUNCH(H, 512, W, R); break; \
    case 1024: R4D_ARN_LAUNCH(H, 1024, W, R); break; \
    default: throw std::runtime_error( \
        "r4d ar_ln_had_quant_i8: workgroup width must be 128, 256, 512 or 1024"); }
#define R4D_ARN_W(W, R) switch (had) { \
    case 128: R4D_ARN_T(128, W, R); break; \
    case 256: R4D_ARN_T(256, W, R); break; \
    case 512: R4D_ARN_T(512, W, R); break; \
    default: throw std::runtime_error( \
        "r4d ar_ln_had_quant_i8: only widths 128, 256 and 512 are compiled"); }
  if (wire) {
    if (out_bf16) { R4D_ARN_W(true, 6); } else { R4D_ARN_W(false, 6); }
  } else {
    if (out_bf16) { R4D_ARN_W(true, 0); } else { R4D_ARN_W(false, 0); }
  }
#undef R4D_ARN_W
#undef R4D_ARN_T
#undef R4D_ARN_LAUNCH
}
