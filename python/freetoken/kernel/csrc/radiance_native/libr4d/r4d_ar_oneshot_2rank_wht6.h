#pragma once
// r4d_ar_oneshot_2rank_wht6.h: device code for r4d_ar_oneshot_2rank_wht6, the rotated 6-bit payload of the
// one-shot 2-rank P2P all-reduce on dual R9700 (gfx1201/RDNA4, PCIe). Reuses the IPC scratch,
// per-block flags and device-resident seq counters of r4d_ar_oneshot_2rank_exact.hip. Used for
// large messages, where wire time dominates and 6.25 bits per element saves more than the rotation
// and packing cost.
//
// Design:
//  * Each group of GROUP=64 is rotated by an orthonormal Walsh-Hadamard, scaled by its own amax
//    and stored as BITS=6 uniform bits plus a bf16 scale (6.25 bits/element). The rotation
//    spreads the outlier channel, which is what makes 6 uniform bits enough.
//  * Push rotates, quantizes and packs straight into the peer's scratch and keeps a local copy
//    of the same words (cheaper than repeating the rotation and amax). Reduce unpacks both
//    halves, adds in fp32, inverse-rotates (the transform is its own inverse) and stores
//    bf16/fp16.
//  * Both ranks fold the same pair and fp32 add commutes, so the ranks stay bit-identical. Must
//    be built with -ffp-contract=off: the ranks see the two products in opposite order, and an
//    FMA on either makes them differ by ~1 ULP.
//  * Graph-safe and double-buffered as in r4d_ar_oneshot_2rank_exact.hip: per-block
//    device-resident seq counter, slot from the per-launch parity counters (r4d_ar_seq.h).
//
// Layout, per slot, inside the shared 2*max_bytes scratch:
//   [ packed symbols, chunk-interleaved ] ... [ scale_off_bytes ] ... [ bf16 scales: n_groups ]
// A chunk is 32 groups = 2048 elements by default. One wave owns a chunk and every lane emits 64
// symbols = 384 bits = three uint4, so flush points are lane-uniform and lane L's u-th uint4 lands
// at c*96 + u*WAVE + L: one coalesced 512B store per wave. Smaller chunks (Wht6Chunk) pack a
// lane's NW words the same way, as dwords when NW is not a multiple of four.
//
// n_groups need not tile a chunk: groups past the end quantize zeros and are skipped on output.
// This is safe only because a group spans LPG contiguous lanes and every cross-lane exchange in
// wht() stays inside the group (m < LPG), so no shuffle is split across the guard.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>
#include "r4d_ar_seq.h"
#include "r4d_common.h"   // r4d_xor_lane

#define R4D_AR_SPIN_MAX 4000000000ULL
#define R4D_AR_MAX_BLOCKS 512
#define GROUP 64                  // elements per rotation + scaling group
#define BITS 6                    // bits per symbol
#define EPL 8                     // elements per lane per group
#define LPG (GROUP / EPL)         // 8 lanes span one group
#define WAVE 32
#define GPW (WAVE * EPL / GROUP)  // 4 groups per wave per iteration
#define CHUNK_GROUPS 32           // groups per wave-chunk (makes 64 symbols/lane = 3 uint4)
#define CHUNK_ELEMS (CHUNK_GROUPS * GROUP)

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void do_drain(int drain) {
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
}

__device__ __forceinline__ float rd_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float rd_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void rd_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void rd_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// RTNE to bf16 and back: the scale ships as its top 16 bits, and the sender's local copy must
// equal what the receiver reconstructs.
__device__ __forceinline__ float bf16_round(float v) {
  unsigned u = __float_as_uint(v);
  const unsigned lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return __uint_as_float(u & 0xffff0000u);
}


// `lo ? (v + o) : (o - v)` written as `o + (lo ? v : -v)`: the sign flip is one XOR with a mask
// uniform over the stage, so the per-element select disappears. Bit-identical, since IEEE negation
// is exact and addition commutes.
template <int M>
__device__ __forceinline__ void ar6_xstage(float* v, int lig) {
  const unsigned int sm = (lig & M) ? 0x80000000u : 0u;
#pragma unroll
  for (int i = 0; i < EPL; ++i) {
    const float o = r4d_xor_lane<M>(v[i]);
    v[i] = __int_as_float(__float_as_int(v[i]) ^ sm) + o;
  }
}

// Orthonormal Walsh-Hadamard over one group; its own inverse, so this routine also undoes it.
// A group is LPG lanes x EPL elements: the in-register butterflies are free ALU, and with
// ROW_XMASK the cross-lane stages are free too.
__device__ __forceinline__ void wht(float* v, int lig) {
  static_assert(LPG == 8, "the cross-lane stages are unrolled for a group of 8 lanes");
#pragma unroll
  for (int h = 1; h < EPL; h <<= 1)
#pragma unroll
    for (int i = 0; i < EPL; ++i)
      if ((i & h) == 0) { const float x = v[i], y = v[i ^ h]; v[i] = x + y; v[i ^ h] = x - y; }
  ar6_xstage<1>(v, lig);
  ar6_xstage<2>(v, lig);
  ar6_xstage<4>(v, lig);
  const float nrm = rsqrtf((float)GROUP);
#pragma unroll
  for (int i = 0; i < EPL; ++i) v[i] *= nrm;
}

// Push and reduce over one wave-chunk, shared by every kernel carrying the rotated payload (the
// all-reduce below and the fused gated-residual writes in r4d_ar_oneshot_2rank_wht6.hip), so all
// pack and unpack the same bytes. `ld(off, v)` fills the EPL elements of a live group starting at
// message element `off`; `st(off, v)` takes the reduced EPL elements there. Each runs once per
// live group per lane; a dead group quantizes zeros and is skipped on output.
//
// CG is the chunk in groups: 32 by default (2048 elements, three uint4 per lane), fewer where a
// kernel wants more waves. It sets byte ownership and placement in the slot, never values (the
// rotation and amax are per group), so two kernels agree on the wire iff they agree on CG. A
// lane's NW words go out as dwords when NW is not a multiple of four.
template <int CG>
struct Wht6Chunk {
  static constexpr int ITERS = CG * GROUP / (WAVE * EPL);    // group-iterations a chunk
  static constexpr int NW = ITERS * EPL * BITS / 32;         // packed words a lane holds
  static constexpr int ELEMS = CG * GROUP;
  static constexpr bool U4 = NW % 4 == 0;
  static_assert(ITERS >= 1 && ITERS * WAVE * EPL == CG * GROUP, "a chunk is whole wave-iterations");
  static_assert(ITERS * EPL * BITS % 32 == 0, "a lane's symbols fill whole words");
};

// Push: rotate, amax, quantize, pack into peer scratch, plus a local copy for the reduce.
template <int CG = CHUNK_GROUPS, typename Ld>
__device__ __forceinline__ void wht6_push_chunk(const Ld& ld, int c, int n_groups, int lane,
                                                int lig, int gw, uint4* p_pay,
                                                unsigned short* p_sc, uint4* l_pay, float* l_sc,
                                                unsigned int& hck) {
  using Ch = Wht6Chunk<CG>;
  const int qmin = -(1 << (BITS - 1)), qmax = (1 << (BITS - 1)) - 1;
  unsigned int acc[Ch::NW];
#pragma unroll
  for (int k = 0; k < Ch::NW; ++k) acc[k] = 0u;
#pragma unroll
  for (int t = 0; t < Ch::ITERS; ++t) {
    const int g = c * CG + t * GPW + gw;
    const bool live = (g < n_groups);                     // uniform across a group's lanes
    const long off = (long)g * GROUP + (long)lig * EPL;
    float v[EPL];
    if (live) {
      ld(off, v);
    } else {
#pragma unroll
      for (int i = 0; i < EPL; ++i) v[i] = 0.f;
    }
    wht(v, lig);
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < EPL; ++i) a = fmaxf(a, fabsf(v[i]));
    a = fmaxf(a, r4d_xor_lane<1>(a));
    a = fmaxf(a, r4d_xor_lane<2>(a));
    a = fmaxf(a, r4d_xor_lane<4>(a));
    float sc = (a > 0.f) ? bf16_round(a * (1.0f / (float)qmax)) : 1.0f;
    if (!(sc > 0.f)) sc = 1.0f;
    if (lig == 0 && live) {
      const unsigned short sc16 = (unsigned short)(__float_as_uint(sc) >> 16);
      p_sc[g] = sc16;
      l_sc[g] = sc;
      hck += r4d_archk_mix(sc16, R4D_ARCHK_U16 + (unsigned)g);
    }
    const float inv = 1.0f / sc;
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
      const int q = min(max((int)rintf(v[i] * inv), qmin), qmax) - qmin;
      const unsigned int sym = (unsigned int)q;
      const int bit = (t * EPL + i) * BITS, wi = bit >> 5, bo = bit & 31;
      acc[wi] |= sym << bo;
      if (bo + BITS > 32) acc[wi + 1] |= sym >> (32 - bo);
    }
  }
  if constexpr (Ch::U4) {
#pragma unroll
    for (int u = 0; u < Ch::NW / 4; ++u) {
      const uint4 q = make_uint4(acc[4 * u], acc[4 * u + 1], acc[4 * u + 2], acc[4 * u + 3]);
      const long o = (long)c * (Ch::NW / 4 * WAVE) + (long)u * WAVE + lane;
      p_pay[o] = q;
      l_pay[o] = q;
      hck += r4d_archk_u4(q, 4u * (unsigned)o);
    }
  } else {
    unsigned int* pp = reinterpret_cast<unsigned int*>(p_pay);
    unsigned int* lp = reinterpret_cast<unsigned int*>(l_pay);
#pragma unroll
    for (int u = 0; u < Ch::NW; ++u) {
      const long o = (long)c * (Ch::NW * WAVE) + (long)u * WAVE + lane;
      pp[o] = acc[u];
      lp[o] = acc[u];
      hck += r4d_archk_mix(acc[u], (unsigned)o);
    }
  }
}

// Handshake: drain the pushes into the fabric, set a per-block release flag and spin on the peer's.
// A handshake that cannot complete is a lockstep violation; trap rather than sum stale bytes.
__device__ __forceinline__ void wht6_handshake(unsigned int* peer_flags, unsigned int* my_flags,
                                               int b, unsigned int s, int tid, int drain,
                                               int acq) {
  do_drain(drain);
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
}

// The push check's re-hash of one chunk (r4d_ar_seq.h): the payload words and live scales the
// reduce below reads, at the indices wht6_push_chunk hashed them.
template <int CG = CHUNK_GROUPS>
__device__ __forceinline__ void wht6_rehash_chunk(unsigned int& h, int c, int n_groups, int lane,
                                                  int lig, int gw, const uint4* m_pay,
                                                  const unsigned short* m_sc, long md) {
  using Ch = Wht6Chunk<CG>;
  if constexpr (Ch::U4) {
#pragma unroll
    for (int u = 0; u < Ch::NW / 4; ++u) {
      const long o = (long)c * (Ch::NW / 4 * WAVE) + (long)u * WAVE + lane;
      h += r4d_archk_u4(r4d_archk_keep(&m_pay[o], md), 4u * (unsigned)o);
    }
  } else {
    const unsigned int* mp = reinterpret_cast<const unsigned int*>(m_pay);
#pragma unroll
    for (int u = 0; u < Ch::NW; ++u) {
      const long o = (long)c * (Ch::NW * WAVE) + (long)u * WAVE + lane;
      h += r4d_archk_mix(r4d_archk_keep(&mp[o], md), (unsigned)o);
    }
  }
#pragma unroll
  for (int t = 0; t < Ch::ITERS; ++t) {
    const int g = c * CG + t * GPW + gw;
    if (lig == 0 && g < n_groups)
      h += r4d_archk_mix(r4d_archk_keep(&m_sc[g], md), R4D_ARCHK_U16 + (unsigned)g);
  }
}

// Reduce: unpack both halves, add in fp32, inverse-rotate, pass each live group's EPL elements to st.
template <int CG = CHUNK_GROUPS, typename St>
__device__ __forceinline__ void wht6_reduce_chunk(const St& st, int c, int n_groups, int lane,
                                                  int lig, int gw, const uint4* m_pay,
                                                  const unsigned short* m_sc,
                                                  const uint4* l_pay, const float* l_sc) {
  using Ch = Wht6Chunk<CG>;
  const int qmin = -(1 << (BITS - 1));
  const unsigned int MASK = (1u << BITS) - 1u;
  unsigned int pac[Ch::NW], lac[Ch::NW];
  if constexpr (Ch::U4) {
#pragma unroll
    for (int u = 0; u < Ch::NW / 4; ++u) {
      const long o = (long)c * (Ch::NW / 4 * WAVE) + (long)u * WAVE + lane;
      const uint4 pq = m_pay[o], lq = l_pay[o];
      pac[4 * u] = pq.x; pac[4 * u + 1] = pq.y; pac[4 * u + 2] = pq.z; pac[4 * u + 3] = pq.w;
      lac[4 * u] = lq.x; lac[4 * u + 1] = lq.y; lac[4 * u + 2] = lq.z; lac[4 * u + 3] = lq.w;
    }
  } else {
    const unsigned int* mp = reinterpret_cast<const unsigned int*>(m_pay);
    const unsigned int* lp = reinterpret_cast<const unsigned int*>(l_pay);
#pragma unroll
    for (int u = 0; u < Ch::NW; ++u) {
      const long o = (long)c * (Ch::NW * WAVE) + (long)u * WAVE + lane;
      pac[u] = mp[o];
      lac[u] = lp[o];
    }
  }
#pragma unroll
  for (int t = 0; t < Ch::ITERS; ++t) {
    const int g = c * CG + t * GPW + gw;
    if (g >= n_groups) continue;                          // whole group: no shuffle is split
    const long off = (long)g * GROUP + (long)lig * EPL;
    const float psc = __uint_as_float((unsigned int)m_sc[g] << 16);
    const float lsc = l_sc[g];
    float v[EPL];
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
      const int bit = (t * EPL + i) * BITS, wi = bit >> 5, bo = bit & 31;
      unsigned int ps = pac[wi] >> bo, ls = lac[wi] >> bo;
      if (bo + BITS > 32) { ps |= pac[wi + 1] << (32 - bo); ls |= lac[wi + 1] << (32 - bo); }
      v[i] = (float)((int)(ps & MASK) + qmin) * psc + (float)((int)(ls & MASK) + qmin) * lsc;
    }
    wht(v, lig);
    st(off, v);
  }
}

// The slot's regions, as the kernels below address them.
struct Wht6Slot {
  uint4* p_pay; unsigned short* p_sc;               // the peer's scratch: payload, bf16 scales
  const uint4* m_pay; const unsigned short* m_sc;   // this rank's scratch, which the peer filled
  uint4* l_pay; float* l_sc;                        // this rank's own half, local only
  // Onto the host mailbox's VRAM mirror once the check has passed (r4d_ar_seq.h).
  __device__ __forceinline__ void moved(long md) {
    m_pay = r4d_archk_moved(m_pay, md);
    m_sc = r4d_archk_moved(m_sc, md);
  }
};
__device__ __forceinline__ Wht6Slot wht6_slot(char* peer_scratch, const char* my_scratch,
                                              char* loc_pack, int slot, long slot_stride_bytes,
                                              long scale_off_bytes) {
  char* pbase = peer_scratch + (size_t)slot * slot_stride_bytes;
  const char* mbase = my_scratch + (size_t)slot * slot_stride_bytes;
  Wht6Slot w;
  w.p_pay = (uint4*)pbase;
  w.p_sc = (unsigned short*)(pbase + scale_off_bytes);
  w.m_pay = (const uint4*)mbase;
  w.m_sc = (const unsigned short*)(mbase + scale_off_bytes);
  w.l_pay = (uint4*)loc_pack;
  w.l_sc = (float*)(loc_pack + scale_off_bytes);
  return w;
}

template <typename T>
__global__ void r4d_ar_oneshot_2rank_wht6_kernel(
    const T* __restrict__ in, char* peer_scratch, const char* my_scratch, char* loc_pack,
    T* __restrict__ out, unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n_groups, int n_chunks, long slot_stride_bytes, long scale_off_bytes, int drain, int acq,
    unsigned int nbw, long stage) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  const int lane = tid & (WAVE - 1), wid = tid / WAVE, nwarp = nt / WAVE;
  const int lig = lane & (LPG - 1), gw = lane / LPG;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  Wht6Slot w = wht6_slot(peer_scratch, my_scratch, loc_pack, (int)s_slot,
                         slot_stride_bytes, scale_off_bytes);
  const long md = r4d_archk_md(stage, my_scratch);   // the host mailbox's VRAM mirror

  const int cpb = (n_chunks + nb - 1) / nb;                 // block b owns chunks [c0,c1)
  const int c0 = b * cpb; int c1 = c0 + cpb; if (c1 > n_chunks) c1 = n_chunks;

  const auto ld = [&](long off, float* v) {
#pragma unroll
    for (int i = 0; i < EPL; ++i) v[i] = rd_to_f(in[off + i]);
  };
  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)
  for (int c = c0 + wid; c < c1; c += nwarp)
    wht6_push_chunk(ld, c, n_groups, lane, lig, gw, w.p_pay, w.p_sc, w.l_pay, w.l_sc, hck);
  r4d_archk_put(hck, peer_flags, nbw, s_slot, (unsigned)b, s);

  wht6_handshake(peer_flags, my_flags, b, s, tid, drain, acq);
  r4d_archk_wait(my_flags, nbw, s_slot, (unsigned)b, s, [&](unsigned int& h) {
    for (int c = c0 + wid; c < c1; c += nwarp)
      wht6_rehash_chunk(h, c, n_groups, lane, lig, gw, w.m_pay, w.m_sc, md);
  });
  w.moved(md);

  const auto st = [&](long off, const float* v) {
#pragma unroll
    for (int i = 0; i < EPL; ++i) rd_from_f(v[i], out[off + i]);
  };
  for (int c = c0 + wid; c < c1; c += nwarp)
    wht6_reduce_chunk(st, c, n_groups, lane, lig, gw, w.m_pay, w.m_sc, w.l_pay, w.l_sc);
}

static int clamp_blocks(int nb, int upper) {
  if (nb < 1) nb = 1; if (nb > upper) nb = upper;
  if (nb > R4D_AR_MAX_BLOCKS) nb = R4D_AR_MAX_BLOCKS; return nb;
}
static int clamp_threads(int nt) { if (nt < 64) nt = 64; if (nt > 1024) nt = 1024; return (nt / 64) * 64; }

// Operates on scratch/flags/seq allocated by r4d_ar_ipc_alloc; slot_stride_bytes = max_bytes,
// scale_off_bytes = max_bytes/2. loc_pack is a plain local buffer (not IPC) holding this rank's
// packed half and its fp32 scales, split at the same scale_off_bytes.
// bf16/fp16 payloads only; n_elem must be a multiple of GROUP.
static void r4d_ar_wht6_launch(int64_t peer_scratch, int64_t my_scratch, int64_t peer_flags,
                            int64_t my_flags, int64_t seq_ctrs, int64_t loc_pack,
                            int64_t slot_stride_bytes, int64_t scale_off_bytes, int64_t inp,
                            int64_t out, int64_t n_elem, int64_t dtype, int64_t stream_i,
                            int64_t nblocks, int64_t nthreads, int64_t drain, int64_t acq,
                            int64_t stage = 0) {
  hipStream_t st = (hipStream_t)stream_i;
  if (dtype != 0 && dtype != 1) throw std::runtime_error("pack ar: only bf16/fp16 payloads");
  if (n_elem % GROUP != 0) throw std::runtime_error("pack ar: n_elem not a multiple of GROUP");
  const int n_groups = (int)(n_elem / GROUP);
  const int n_chunks = (n_groups + CHUNK_GROUPS - 1) / CHUNK_GROUPS;
  const int nb = clamp_blocks((int)nblocks, n_chunks);
  const int nt = clamp_threads((int)nthreads);
#define LF(T) r4d_ar_oneshot_2rank_wht6_kernel<T><<<nb, nt, 0, st>>>( \
    (const T*)inp, (char*)peer_scratch, (const char*)my_scratch, (char*)loc_pack, (T*)out, \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n_groups, n_chunks, \
    (long)slot_stride_bytes, (long)scale_off_bytes, (int)drain, (int)acq, \
    (unsigned)R4D_AR_MAX_BLOCKS, (long)stage)
  if (dtype == 0) LF(__hip_bfloat16);
  else LF(__half);
#undef LF
}
