// r4d_ar_twoshot_Nrank_exact.hip: r4d_ar_twoshot_{3,4,8}rank_exact -- reduce-scatter +
// all-gather at 3, 4 or 8 ranks, exact bf16/fp16/fp32 wire. Per-block pipelined handshake
// instead of a grid arrival election. Sends 2*(ws-1)/ws*N bytes per rank in two hops, so it
// covers the sizes where the one-shot's (ws-1)*N send is too heavy.
//
// Shard s = a contiguous run of the message's 16-byte words, owned by rank s. Phase 1: push my
// contribution to each peer-owned shard into the owner's slot[me] region A, flag 2s-1; reduce my
// shard fp32 in ascending rank order (own contribution from the input). Phase 2: reduced words go
// to my out and every peer's slot[me] region B, flag 2s; assemble out from region B. Region A at
// 0, B at sh_max; the slot stride must hold 2*sh_max.
//
// The shard table (r4d_ar_shards.h) comes in as a kernel argument. By default it is the even
// split (the first n16 % ws shards one word longer); at 3 ranks engine residuals rarely divide (a
// 2048-wide bf16 row is 256 words). At 8 ranks it is weighted per link (r4d_ar_links.h); a rank with weight 0 owns nothing and only pushes, gathers and
// handshakes. Over both phases a rank moves N + (ws-2)*len_r each way, but phase 1 only sends
// N - len_r (receiving (ws-1)*len_r) and phase 2 the reverse, so a short shard on a slow link pays
// only as far as the phases overlap on that link (docs/GUIDE.md, 4.7). Each word is the same fp32
// fold in ascending rank order whoever owns it, so the bits match the one-shot's.
//
// At 8 ranks the message may also run in pieces (a count from the message size, r4d_ar_shards.h): each block's words split into `pieces` runs pipelined so
// piece j+1's phase 1 shares the link with piece j's phase 2. A rank with a short shard then
// sends and receives at once instead of one direction per phase.
//
// Also at 8 ranks, two more forms of the same work, chosen by the `form` launch parameter to time
// a board (the engine runs lockstep, r4d_ar_entry.hip):
//  * ring sends the first hop to the peers in ring order from rank + 1 instead of ascending, so
//    no card receives from all seven at once, as the lowest rank does when every sender starts
//    at its lowest peer;
//  * split also replaces the one handshake a step, which waits for every peer to finish the whole
//    step, by a flag after each half (the first hop of the next piece, the reduce of this one),
//    each awaited only before its data is read: before reducing piece j the peers' first hop of
//    j, before assembling j their reduce of j. A rank whose own half is done goes on with the
//    next piece's first hop while a slower one still reduces, so the links run at their own pace
//    instead of in lockstep. 2 * pieces + 1 flag values, pieces + 1 steps reserved.
// The words, their owners and their folds are the lockstep form's, so the bits are too.
//
// Shares the wide family's flag/seq/scratch set (r4d_ar_wide.h): it publishes in the 2*seq flag
// space plus the 2s-1 half-step (with pieces: pieces+1 values from 2s-1 on, after reserving one
// step a piece), so one buffer set can carry any mix of one-shot and two-shot launches. The slot
// stride must hold the largest requirement in the mix.
#include "r4d.h"
#include "r4d_ar_wide.h"
#include "r4d_ar_shards.h"
#include "r4d_chain.h"
#include <stdexcept>
#include <string>

static_assert(R4D_AR_TL_WORDS >= 4 + 2 * R4D_AR_TS_MAX_PIECES,
              "a timeline holds the start, a stamp pair a handshake and the end");

template <int WS>
struct R4DArShardOff {
  int off[WS + 1];
  int sh_max;
};

// PIECED false fixes pieces at 1, so the unpipelined launch keeps the straight-line kernel.
// CHAIN: the chained form's wait (before the sequence take, so a block of the next launch on the
// instance never starts ahead of this one's) and add (r4d_chain.h); no block returns early.
template <typename T, int LANES, int WS, bool PIECED, int FORM, bool CHAIN>
__device__ __forceinline__ void r4d_ar_twoshot_wide_exact_body(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,
    const int4* __restrict__ my_base, unsigned int* __restrict__ my_flags,
    int4* __restrict__ out4, unsigned int* __restrict__ seq_ctrs, R4DArShardOff<WS> sh,
    int slot_stride16, int rank, int drain, int acq, int pub,
    long long* gate, unsigned int* gate_ctr, int pieces_arg, long long* tl_all) {
  if (CHAIN) r4d_chain_wait();
  constexpr int NPEERS = WS - 1;
  const int pieces = PIECED ? pieces_arg : 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  // device-resident, replay-safe; `pieces` steps reserve the 2*pieces flag values this launch uses
  // (the split form publishes 2 * pieces + 1 and reserves one step more)
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot,
                  (unsigned int)(FORM == R4D_AR8_FORM_SPLIT ? pieces + 1 : pieces));
  const unsigned int v0 = 2u * s_seq - 1u;
  const int parity = (int)s_slot;
  // the split form awaits 2 * pieces times, which the timeline holds up to 8 pieces
  long long* const tl = tl_all && (FORM != R4D_AR8_FORM_SPLIT || 4 * pieces + 2 <= R4D_AR_TL_WORDS)
                            ? tl_all + (size_t)b * R4D_AR_TL_WORDS : nullptr;
  r4d_ar_tl_stamp(tl, 0);

  const int sh_max = sh.sh_max;
#define R4D_TS_OFF(s) (sh.off[s])
#define R4D_TS_LEN(s) (sh.off[(s) + 1] - sh.off[s])

  const int w = (sh_max + nb - 1) / nb;    // block b owns words [w0,w1) of the SHARD space
  const int w0 = b * w;
  int w1 = w0 + w; if (w1 > sh_max) w1 = sh_max;
  const int pw = (w + pieces - 1) / pieces;  // piece j of the block: [w0 + j*pw, +pw) within w1

  // region A of my slot in every peer's scratch (B sits at +sh_max)
  int4* pa[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    pa[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;

  const int4* src[WS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    src[r] = (r == rank) ? (in4 + (size_t)R4D_TS_OFF(rank))
                         : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);
  int4* my_out = out4 + (size_t)R4D_TS_OFF(rank);

  // phase 1: my contribution to each peer-owned shard -> owner's region A
  auto push1 = [&](int j) {
    const int p0 = w0 + j * pw, p1 = p0 + pw < w1 ? p0 + pw : w1;
    for (int i = 0; i < NPEERS; ++i) {
      const int p = FORM == R4D_AR8_FORM_LOCKSTEP ? i : (i + rank) % NPEERS;
      const int owner = peers.rank[p];
      const int4* s1 = in4 + (size_t)R4D_TS_OFF(owner);
      int4* dst = pa[p];
      const int lim = p1 < R4D_TS_LEN(owner) ? p1 : R4D_TS_LEN(owner);
      for (int k = p0 + tid; k < lim; k += nt) dst[k] = s1[k];
    }
  };
  // reduce my shard (fp32, ascending rank, own contribution from the input) fused with the phase-2
  // push: the reduced word goes to my out and every peer's region B in the same pass
  auto reduce2 = [&](int j) {
    const int p0 = w0 + j * pw, p1 = p0 + pw < w1 ? p0 + pw : w1;
    const int my_lim = p1 < R4D_TS_LEN(rank) ? p1 : R4D_TS_LEN(rank);
    for (int k = p0 + tid; k < my_lim; k += nt) {
      float acc[LANES];
#pragma unroll
      for (int l = 0; l < LANES; ++l) acc[l] = 0.0f;
#pragma unroll
      for (int r = 0; r < WS; ++r) {
        const int4 v = src[r][k];
        const T* pv = reinterpret_cast<const T*>(&v);
#pragma unroll
        for (int l = 0; l < LANES; ++l) acc[l] += r4d_ar_to_f(pv[l]);
      }
      int4 o;
      T* po = reinterpret_cast<T*>(&o);
#pragma unroll
      for (int l = 0; l < LANES; ++l) r4d_ar_from_f(acc[l], po[l]);
      my_out[k] = o;
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) pa[p][sh_max + k] = o;  // region B
    }
  };
  // every peer-owned shard from my region B copies
  auto assemble = [&](int j) {
    const int p0 = w0 + j * pw, p1 = p0 + pw < w1 ? p0 + pw : w1;
    for (int p = 0; p < NPEERS; ++p) {
      const int owner = peers.rank[p];
      const int4* srcB = my_base + ((size_t)parity * WS + owner) * (size_t)slot_stride16 + sh_max;
      int4* dst = out4 + (size_t)R4D_TS_OFF(owner);
      const int lim = p1 < R4D_TS_LEN(owner) ? p1 : R4D_TS_LEN(owner);
      for (int k = p0 + tid; k < lim; k += nt) dst[k] = srcB[k];
    }
  };

  /* Pieces run as a pipeline so both directions of a link carry bytes at once: step j pushes
   * piece j+1's phase 1 and piece j's phase 2. One flag word per (block, source) is a monotone
   * counter, so a step's single handshake (v0 + 1 + j) also vouches for every earlier one. A rank
   * owning little sends nearly all its bytes in phase 1 and receives them in phase 2; without the
   * overlap one direction idles in each phase. With one piece the sequence is push, flag 2s-1,
   * reduce, flag 2s, assemble. A word is touched by the same thread in all three steps, which keeps
   * the in-place form safe. */
  if constexpr (FORM != R4D_AR8_FORM_SPLIT) {
    push1(0);
    r4d_ar_handshake<WS>(peers, my_flags, b, rank, v0, drain, acq, pub, tl ? tl + 1 : nullptr);
    for (int j = 0; j < pieces; ++j) {
      if (j + 1 < pieces) push1(j + 1);
      reduce2(j);
      if (j > 0) assemble(j - 1);
      r4d_ar_handshake<WS>(peers, my_flags, b, rank, v0 + 1u + (unsigned int)j, drain, acq, pub,
                           tl ? tl + 3 + 2 * j : nullptr);
    }
    /* The hold ends after the last handshake, not the first: the reduce is fused with the phase-2
     * push (every reduced word goes to each peer's region B), so until then the link still carries
     * this launch's phase-2 bytes. Here every phase-1 and phase-2 remote store has retired behind
     * its flag store and every peer's pushes to this rank have been waited on; only the local
     * assemble of the last piece is left (r4d_ar_seq.h). */
    r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);
    assemble(pieces - 1);
    if (tl) {
      __syncthreads();
      r4d_ar_tl_stamp(tl, 3 + 2 * pieces);
    }
  } else {
    /* The same steps with a flag after each half. Published in program order on one monotone
     * word: first hop of piece 0 at v0, of piece j + 1 at v0 + 2j + 1, the reduce of piece j at
     * v0 + 2j + 2; so "peer's first hop of j done" is the word at v0 (j = 0) or v0 + 2j - 1, and
     * "peer's reduce of j done" at v0 + 2j + 2. A launch completes only after every peer has
     * published its last reduce, which it does inside this launch, so the slot parity argument
     * of the lockstep form holds unchanged (r4d_ar_seq.h). */
    int n = 0;
    auto tlp = [&]() { long long* p = tl ? tl + 1 + 2 * n : nullptr; ++n; return p; };
    push1(0);
    r4d_ar_publish<WS>(peers, b, rank, v0, drain, pub);
    for (int j = 0; j < pieces; ++j) {
      if (j + 1 < pieces) {
        push1(j + 1);
        r4d_ar_publish<WS>(peers, b, rank, v0 + 2u * (unsigned int)j + 1u, drain, pub);
      }
      r4d_ar_await<WS>(peers, my_flags, b, j == 0 ? v0 : v0 + 2u * (unsigned int)j - 1u, acq,
                       tlp());
      reduce2(j);
      r4d_ar_publish<WS>(peers, b, rank, v0 + 2u * (unsigned int)j + 2u, drain, pub);
      if (j > 0) {
        r4d_ar_await<WS>(peers, my_flags, b, v0 + 2u * (unsigned int)j, acq, tlp());
        assemble(j - 1);
      }
    }
    r4d_ar_await<WS>(peers, my_flags, b, v0 + 2u * (unsigned int)pieces, acq, tlp());
    r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);
    assemble(pieces - 1);
    if (tl) {
      __syncthreads();
      r4d_ar_tl_stamp(tl, 1 + 2 * n);
    }
  }
#undef R4D_TS_OFF
#undef R4D_TS_LEN
  if (CHAIN) r4d_chain_done();
}

// ---- the coded form: both pushes carry ~12.2 bits a bf16 value instead of 16, losslessly
//
// The prefill overlap's reductions are link-bound, and a residual stream's bf16 words have few
// distinct exponents: on the 31B (4x R9700, 64 groups of 512 x 5376 values from a 9.9k-token
// prefill) 2.5-3.2 bits of exponent entropy against 8 stored. A chunk is one int4 word (8
// values); 8 consecutive chunks (64 values) form a group that shares a base exponent, and each
// value keeps its sign, its 7 mantissa bits and a 4-bit exponent offset (0 = an exact zero, 1..15 =
// base .. base + 14), 8 values to 3 dwords. A group holding a subnormal, an inf or NaN, or an
// exponent range past 14 (0.02% of groups there) is stored raw in a shadow area, its base byte
// 0xFF. A region (one source's shard in an owner's scratch) is [3 dwords a chunk][a base byte a
// group][the shadow, 16 bytes a chunk]; A and B are two regions in a slot. A wave codes 32
// consecutive chunks: the group's exponents meet by lane shuffles, and the packed dwords are
// redistributed so every store a wave makes is contiguous, keeping the P2P writes coalesced.
//
// Decoding gives back the same bf16 values, so the reduce folds the same numbers in the same rank
// order: the bytes are the plain kernel's. Four-card microbenchmark (the real payload above, the
// two-shot of a 1024-row slice beside the w4a8 prefill GEMM): byte-identical; the reduction 1383
// us at the plain lane launch (2 x 256) against 996 us coded at 4 x 1024, with the GEMM beside it
// 746 -> 724 us. Below 1024 threads a launch codes too slowly (2 x 256: 2492 us).
__device__ __forceinline__ unsigned r4d_arc_shfl(unsigned v, int src) {
  return (unsigned)__builtin_amdgcn_ds_bpermute(src << 2, (int)v);
}
struct R4dArcEnc { unsigned w[3]; unsigned base; };
// Code this lane's chunk; every lane of the wave calls (a lane past the end with valid false).
__device__ __forceinline__ R4dArcEnc r4d_arc_enc(int4 v, bool valid) {
  const unsigned d[4] = {(unsigned)v.x, (unsigned)v.y, (unsigned)v.z, (unsigned)v.w};
  unsigned emin = 255u, emax = 0u, bad = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned u = (d[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
    const unsigned e = (u >> 7) & 0xffu, m = u & 0x7fu;
    if (e) { emin = e < emin ? e : emin; emax = e > emax ? e : emax; }
    bad |= (unsigned)(e == 0u && m != 0u) | (unsigned)(e == 255u);
  }
  if (!valid) { emin = 255u; emax = 0u; bad = 0u; }
  const int lane = (int)(threadIdx.x & 31u);
#pragma unroll
  for (int s = 1; s < 8; s <<= 1) {
    const unsigned a = r4d_arc_shfl(emin, lane ^ s), b = r4d_arc_shfl(emax, lane ^ s);
    const unsigned c = r4d_arc_shfl(bad, lane ^ s);
    emin = a < emin ? a : emin; emax = b > emax ? b : emax; bad |= c;
  }
  R4dArcEnc r;
  const unsigned base = emin == 255u ? 1u : emin;
  r.base = (bad || emax > base + 14u) ? 0xFFu : base;
  unsigned c[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned u = (d[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
    const unsigned e = (u >> 7) & 0xffu, m = u & 0x7fu, sg = u >> 15;
    c[i] = (sg << 11) | (m << 4) | (e ? (e - base + 1u) & 15u : 0u);
  }
  r.w[0] = c[0] | (c[1] << 12) | (c[2] << 24);
  r.w[1] = (c[2] >> 8) | (c[3] << 4) | (c[4] << 16) | (c[5] << 28);
  r.w[2] = (c[5] >> 4) | (c[6] << 8) | (c[7] << 20);
  return r;
}
__device__ __forceinline__ int4 r4d_arc_dec(const unsigned (&w)[3], unsigned base) {
  unsigned c[8];
  c[0] = w[0] & 0xfffu; c[1] = (w[0] >> 12) & 0xfffu; c[2] = (w[0] >> 24) | ((w[1] & 0xfu) << 8);
  c[3] = (w[1] >> 4) & 0xfffu; c[4] = (w[1] >> 16) & 0xfffu; c[5] = (w[1] >> 28) | ((w[2] & 0xffu) << 4);
  c[6] = (w[2] >> 8) & 0xfffu; c[7] = w[2] >> 20;
  unsigned u[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned off = c[i] & 15u;
    u[i] = ((c[i] >> 11) << 15) | ((off ? base + off - 1u : 0u) << 7) | ((c[i] >> 4) & 0x7fu);
  }
  int4 v;
  v.x = (int)(u[0] | (u[1] << 16)); v.y = (int)(u[2] | (u[3] << 16));
  v.z = (int)(u[4] | (u[5] << 16)); v.w = (int)(u[6] | (u[7] << 16));
  return v;
}
struct R4dArcRegion { unsigned* packed; unsigned* bases; int4* shadow; };
__host__ __device__ constexpr size_t r4d_arc_shadow_off(long SL) {
  return ((size_t)12 * SL + (size_t)SL / 8 + 15) & ~(size_t)15;
}
__host__ __device__ constexpr size_t r4d_arc_region_bytes(long SL) {
  return (r4d_arc_shadow_off(SL) + (size_t)16 * SL + 255) & ~(size_t)255;
}
__device__ __forceinline__ R4dArcRegion r4d_arc_region(const void* p, int SL) {
  char* b = (char*)p;
  return R4dArcRegion{(unsigned*)b, (unsigned*)(b + (size_t)12 * SL), (int4*)(b + r4d_arc_shadow_off(SL))};
}
// A wave's coded chunks [k0, k0 + 32) (k0 a multiple of 32) into ND regions, every store contiguous.
template <int ND>
__device__ __forceinline__ void r4d_arc_put(R4dArcRegion (&dst)[ND], int k0, const R4dArcEnc& e,
                                            int4 raw, bool valid) {
  const int lane = (int)(threadIdx.x & 31u);
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int p = 32 * j + lane, sl = p / 3, wi = p - sl * 3;
    const unsigned a0 = r4d_arc_shfl(e.w[0], sl), a1 = r4d_arc_shfl(e.w[1], sl);
    const unsigned a2 = r4d_arc_shfl(e.w[2], sl), vs = r4d_arc_shfl(valid ? 1u : 0u, sl);
    const unsigned val = wi == 0 ? a0 : wi == 1 ? a1 : a2;
    if (vs) {
#pragma unroll
      for (int q = 0; q < ND; ++q) dst[q].packed[(size_t)k0 * 3 + p] = val;
    }
  }
  const unsigned b0 = r4d_arc_shfl(e.base, 0), b1 = r4d_arc_shfl(e.base, 8);
  const unsigned b2 = r4d_arc_shfl(e.base, 16), b3 = r4d_arc_shfl(e.base, 24);
  if (lane == 0 && valid) {
#pragma unroll
    for (int q = 0; q < ND; ++q) dst[q].bases[k0 / 32] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
  }
  if (valid && e.base == 0xFFu) {
#pragma unroll
    for (int q = 0; q < ND; ++q) dst[q].shadow[k0 + lane] = raw;
  }
}
// A wave's chunks [k0, k0 + 32) back from a region; every lane calls.
__device__ __forceinline__ int4 r4d_arc_get(const R4dArcRegion& src, int k0, bool valid) {
  const int lane = (int)(threadIdx.x & 31u);
  unsigned ld[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) ld[j] = valid ? src.packed[(size_t)k0 * 3 + 32 * j + lane] : 0u;
  unsigned w[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int q = 3 * lane + i, sl = q & 31, j = q >> 5;
    const unsigned a0 = r4d_arc_shfl(ld[0], sl), a1 = r4d_arc_shfl(ld[1], sl);
    const unsigned a2 = r4d_arc_shfl(ld[2], sl);
    w[i] = j == 0 ? a0 : j == 1 ? a1 : a2;
  }
  const unsigned base = ((valid ? src.bases[k0 / 32] : 0u) >> ((lane >> 3) * 8)) & 0xffu;
  if (base == 0xFFu) return valid ? src.shadow[k0 + lane] : make_int4(0, 0, 0, 0);
  return r4d_arc_dec(w, base);
}

// The coded two-shot, bf16 only; the launcher checks that each shard is a whole number of waves'
// 32 chunks and that a slot holds two regions.
template <int WS, bool CHAIN>
__device__ __forceinline__ void r4d_ar_twoshot_coded_body(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,
    const int4* __restrict__ my_base, unsigned int* __restrict__ my_flags,
    int4* __restrict__ out4, unsigned int* __restrict__ seq_ctrs, int n16,
    int slot_stride16, int rank, int drain, int acq, int pub,
    long long* gate, unsigned int* gate_ctr) {
  if (CHAIN) r4d_chain_wait();
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;
  const int SL = n16 / WS;                       /* every shard: a multiple of 32 chunks */
  int w = (SL + nb - 1) / nb;
  w = (w + 31) & ~31;
  const int w0 = b * w;
  int w1 = w0 + w; if (w1 > SL) w1 = SL;
  const size_t RB = r4d_arc_region_bytes(SL);
  const int lane = tid & 31;

  // phase 1: my contribution to each peer-owned shard, coded, into the owner's region A for me
  for (int p = 0; p < NPEERS; ++p) {
    const int owner = peers.rank[p];
    const int4* src = in4 + (size_t)owner * SL;
    R4dArcRegion dst[1] = {r4d_arc_region(peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16, SL)};
    for (int k = w0 + tid; k - lane < w1; k += nt) {
      const bool valid = k < w1;
      const int4 v = valid ? src[k] : make_int4(0, 0, 0, 0);
      r4d_arc_put<1>(dst, k - lane, r4d_arc_enc(v, valid), v, valid);
    }
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s - 1u, drain, acq, pub);

  // reduce my shard (fp32, ascending rank, my own contribution raw from the input), code it once,
  // and push it to every peer's region B
  R4dArcRegion srcA[WS], dstB[NPEERS];
#pragma unroll
  for (int r = 0; r < WS; ++r)
    srcA[r] = r4d_arc_region(my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16, SL);
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    dstB[p] = r4d_arc_region((const char*)(peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16) + RB, SL);
  const int4* own = in4 + (size_t)rank * SL;
  int4* my_out = out4 + (size_t)rank * SL;
  for (int k = w0 + tid; k - lane < w1; k += nt) {
    const bool valid = k < w1;
    const int k0 = k - lane;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < WS; ++r) {
      const int4 v = (r == rank) ? (valid ? own[k] : make_int4(0, 0, 0, 0)) : r4d_arc_get(srcA[r], k0, valid);
      const __hip_bfloat16* pv = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += (float)pv[j];
    }
    int4 o;
    __hip_bfloat16* po = reinterpret_cast<__hip_bfloat16*>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) po[j] = (__hip_bfloat16)acc[j];
    if (valid) my_out[k] = o;
    r4d_arc_put<NPEERS>(dstB, k0, r4d_arc_enc(o, valid), o, valid);
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  // assemble: every peer-owned shard from my region B
  for (int p = 0; p < NPEERS; ++p) {
    const int owner = peers.rank[p];
    const R4dArcRegion src = r4d_arc_region((const char*)(my_base + ((size_t)parity * WS + owner) * (size_t)slot_stride16) + RB, SL);
    int4* dst = out4 + (size_t)owner * SL;
    for (int k = w0 + tid; k - lane < w1; k += nt) {
      const bool valid = k < w1;
      const int4 v = r4d_arc_get(src, k - lane, valid);
      if (valid) dst[k] = v;
    }
  }
  if (CHAIN) r4d_chain_done();
}

#define R4D_ARTS_PARAMS const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,       \
    const int4* __restrict__ my_base, unsigned int* __restrict__ my_flags,                      \
    int4* __restrict__ out4, unsigned int* __restrict__ seq_ctrs, R4DArShardOff<WS> sh,         \
    int slot_stride16, int rank, int drain, int acq, int pub, long long* gate,                  \
    unsigned int* gate_ctr, int pieces_arg, long long* tl_all
#define R4D_ARTS_ARGS in4, peers, my_base, my_flags, out4, seq_ctrs, sh, slot_stride16, rank,    \
    drain, acq, pub, gate, gate_ctr, pieces_arg, tl_all
template <typename T, int LANES, int WS, bool PIECED, int FORM>
__global__ void r4d_ar_twoshot_wide_exact_kernel(R4D_ARTS_PARAMS) {
  r4d_ar_twoshot_wide_exact_body<T, LANES, WS, PIECED, FORM, false>(R4D_ARTS_ARGS);
}
template <typename T, int LANES, int WS, bool PIECED, int FORM>
__global__ void r4dchain_ar_twoshot_wide_exact(R4D_ARTS_PARAMS) {
  r4d_ar_twoshot_wide_exact_body<T, LANES, WS, PIECED, FORM, true>(R4D_ARTS_ARGS);
}
#undef R4D_ARTS_ARGS
#undef R4D_ARTS_PARAMS
// The coded form keeps the even split (its launcher checks every shard is whole waves' chunks), so
// it takes the message length, not the plain form's shard table and pieces.
#define R4D_ARTC_PARAMS const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers,       \
    const int4* __restrict__ my_base, unsigned int* __restrict__ my_flags,                      \
    int4* __restrict__ out4, unsigned int* __restrict__ seq_ctrs, int n16, int slot_stride16,   \
    int rank, int drain, int acq, int pub, long long* gate, unsigned int* gate_ctr
#define R4D_ARTC_ARGS in4, peers, my_base, my_flags, out4, seq_ctrs, n16, slot_stride16, rank,   \
    drain, acq, pub, gate, gate_ctr
template <int WS>
__global__ void r4d_ar_twoshot_coded_kernel(R4D_ARTC_PARAMS) {
  r4d_ar_twoshot_coded_body<WS, false>(R4D_ARTC_ARGS);
}
template <int WS>
__global__ void r4dchain_ar_twoshot_coded(R4D_ARTC_PARAMS) {
  r4d_ar_twoshot_coded_body<WS, true>(R4D_ARTC_ARGS);
}
#undef R4D_ARTC_ARGS
#undef R4D_ARTC_PARAMS

// --- host side ---
template <int WS>
static void ar_twoshot_launch(const long* peer_scratch, const long* peer_flags, long my_scratch,
                              long my_flags, long seq_ctrs, long slot_stride16, long inp,
                              long out, long n_elem, long dtype, long rank, long stream_i,
                              long nblocks, long nthreads, long drain, long acq, long pub,
                              long gate, long gate_ctr, const int* shard_weights,
                              long pieces, long timeline, long form, bool chain = false) {
  constexpr int NPEERS = WS - 1;
  if (rank < 0 || rank >= WS)
    throw std::runtime_error("all_reduce: rank " + std::to_string(rank) +
                             " out of range for the " + std::to_string(WS) +
                             " ranks this entry point is compiled for");
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  if (n_elem > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS)
    throw std::runtime_error("all_reduce: n_elem " + std::to_string(n_elem) +
                             " exceeds the twoshot band (" +
                             std::to_string(R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) +
                             " elems); use the fallback path");
  const long n16 = nbytes / 16;
  if (!shard_weights && n16 < WS)
    throw std::runtime_error("all_reduce: message (" + std::to_string(n16) +
                             " 16B words) is shorter than one word a rank");
  if (n16 < 1) throw std::runtime_error("all_reduce: empty message");
  if (shard_weights) {
    long wsum = 0;
    for (int s = 0; s < WS; ++s) {
      if (shard_weights[s] < 0 || shard_weights[s] > R4D_AR_SHARD_WEIGHT_MAX)
        throw std::runtime_error("all_reduce: shard weight out of range");
      wsum += shard_weights[s];
    }
    if (wsum <= 0) throw std::runtime_error("all_reduce: shard weights sum to zero");
  }
  long off[WS + 1];
  const long sh_max = r4d_ar_shard_table(n16, WS, shard_weights, off);
  R4DArShardOff<WS> sh{};
  for (int s = 0; s <= WS; ++s) sh.off[s] = (int)off[s];
  sh.sh_max = (int)sh_max;
  // regions A and B each hold the longest shard
  if (2 * sh_max > (long)slot_stride16)
    throw std::runtime_error("all_reduce: shard (" + std::to_string(sh_max) +
                             " 16B words) needs 2x inside the per-source slot stride (" +
                             std::to_string(slot_stride16) + ")");

  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > sh_max) nb = (int)sh_max;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  if (pieces < 1 || pieces > R4D_AR_TS_MAX_PIECES)
    throw std::runtime_error("all_reduce: pieces " + std::to_string(pieces) + " outside 1.." +
                             std::to_string((int)R4D_AR_TS_MAX_PIECES));
  if (form < R4D_AR8_FORM_LOCKSTEP || form > R4D_AR8_FORM_SPLIT ||
      (WS != 8 && form != R4D_AR8_FORM_LOCKSTEP))
    throw std::runtime_error("all_reduce: form " + std::to_string(form) + " at " +
                             std::to_string(WS) + " ranks (ring and split are eight-rank forms)");
  // a block's share of the longest shard bounds the pieces worth a handshake each
  const long words_a_block = (sh_max + nb - 1) / nb;
  const int np = (int)(pieces < words_a_block ? pieces : words_a_block);

  R4DArWidePeers<int4, NPEERS> peers{};
  for (int i = 0; i < NPEERS; ++i) {
    if (!peer_scratch[i] || !peer_flags[i])
      throw std::runtime_error("all_reduce: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (int4*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }

  // Checked against the allocations: scratch backs 2*WS slots of the stride, flags one word per
  // (block, source rank), seq a handshake and a parity word per block of capacity (r4d_ar_seq.h).
  const long slot_bytes = 2L * WS * (long)slot_stride16 * 16;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "all_reduce", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "all_reduce", "my_flags");
  r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                      "all_reduce", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, "all_reduce", "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, "all_reduce", "peer flags");
  }

  hipStream_t st = (hipStream_t)stream_i;
#define R4D_AR_TS_LAUNCH_F(T, LN, P, F)                                               \
  (chain ? r4dchain_ar_twoshot_wide_exact<T, LN, WS, P, F>                            \
         : r4d_ar_twoshot_wide_exact_kernel<T, LN, WS, P, F>)<<<nb, nt, 0, st>>>(     \
      (const int4*)inp, peers, (const int4*)my_scratch, (unsigned int*)my_flags,      \
      (int4*)out, (unsigned int*)seq_ctrs, sh, (int)slot_stride16,                    \
      (int)rank, (int)drain, (int)acq, (int)pub, (long long*)gate,                    \
      (unsigned int*)gate_ctr, np, (long long*)timeline)
#define R4D_AR_TS_LAUNCH_P(T, LN, P)                                                  \
  do {                                                                                \
    if constexpr (WS == 8) {                                                          \
      if (form == R4D_AR8_FORM_RING)                                                  \
        R4D_AR_TS_LAUNCH_F(T, LN, P, R4D_AR8_FORM_RING);                              \
      else if (form == R4D_AR8_FORM_SPLIT)                                            \
        R4D_AR_TS_LAUNCH_F(T, LN, P, R4D_AR8_FORM_SPLIT);                             \
      else                                                                            \
        R4D_AR_TS_LAUNCH_F(T, LN, P, R4D_AR8_FORM_LOCKSTEP);                          \
    } else {                                                                          \
      R4D_AR_TS_LAUNCH_F(T, LN, P, R4D_AR8_FORM_LOCKSTEP);                            \
    }                                                                                 \
  } while (0)
#define R4D_AR_TS_LAUNCH(T, LN)                                                       \
  do {                                                                                \
    if (np > 1) R4D_AR_TS_LAUNCH_P(T, LN, true);                                      \
    else R4D_AR_TS_LAUNCH_P(T, LN, false);                                            \
  } while (0)
  if (dtype == 0) R4D_AR_TS_LAUNCH(__hip_bfloat16, 8);
  else if (dtype == 1) R4D_AR_TS_LAUNCH(__half, 8);
  else if (dtype == 2) R4D_AR_TS_LAUNCH(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef R4D_AR_TS_LAUNCH
#undef R4D_AR_TS_LAUNCH_P
#undef R4D_AR_TS_LAUNCH_F
}

void r4d_ar_twoshot_3rank_exact(long peer_scratch0, long peer_scratch1, long peer_flags0,
                                long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub) {
  const long ps[2] = {peer_scratch0, peer_scratch1};
  const long pf[2] = {peer_flags0, peer_flags1};
  /* No link gate at three ranks: the row answers 1, the engine releases by barrier. */
  ar_twoshot_launch<3>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub, 0L, 0L, nullptr,
                       1L, 0L, 0L);
}

void r4d_ar_twoshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_flags0, long peer_flags1, long peer_flags2,
                                long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub, long gate, long gate_ctr, long chain) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  ar_twoshot_launch<4>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub, gate, gate_ctr,
                       nullptr, 1L, 0L, 0L, chain != 0);
}

/* The coded form's conditions: bf16, a message that shards evenly into whole waves' 32 chunks
 * (256 values) a rank, and a slot that holds two coded regions. */
int r4d_ar_twoshot_coded_ok(long n_elem, long dtype, long world, long slot_stride16) {
  if (dtype != 0 || world <= 0 || n_elem % (world * 256) != 0) return 0;
  const long SL = n_elem / 8 / world;
  return 2 * r4d_arc_region_bytes(SL) <= (size_t)slot_stride16 * 16 ? 1 : 0;
}

void r4d_ar_twoshot_4rank_exact_coded(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                      long peer_flags0, long peer_flags1, long peer_flags2,
                                      long my_scratch, long my_flags, long seq_ctrs,
                                      long slot_stride16, long inp, long out, long n_elem,
                                      long rank, long stream, long nblocks, long nthreads,
                                      long drain, long acq, long pub, long gate, long gate_ctr,
                                      long chain) {
  constexpr int WS = 4;
  if (!r4d_ar_twoshot_coded_ok(n_elem, 0, WS, slot_stride16))
    throw std::runtime_error("all_reduce (coded): message or slot outside the coded form");
  if (rank < 0 || rank >= WS) throw std::runtime_error("all_reduce (coded): rank out of range");
  const long n16 = n_elem / 8, SL = n16 / WS;
  int nb = (int)nblocks;
  if (nb < 1) nb = 4;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  if (nb > SL / 32) nb = (int)(SL / 32);
  int nt = (int)nthreads;
  if (nt < 256) nt = 256;
  if (nt > 1024) nt = 1024;
  nt &= ~31;
  R4DArWidePeers<int4, WS - 1> peers{};
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  for (int i = 0; i < WS - 1; ++i) {
    if (!ps[i] || !pf[i]) throw std::runtime_error("all_reduce (coded): null peer pointer");
    peers.scratch[i] = (int4*)ps[i];
    peers.flags[i] = (unsigned int*)pf[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }
  hipStream_t st = (hipStream_t)stream;
  (chain ? r4dchain_ar_twoshot_coded<WS> : r4d_ar_twoshot_coded_kernel<WS>)<<<nb, nt, 0, st>>>(
      (const int4*)inp, peers, (const int4*)my_scratch, (unsigned int*)my_flags, (int4*)out,
      (unsigned int*)seq_ctrs, (int)n16, (int)slot_stride16, (int)rank, (int)drain, (int)acq,
      (int)pub, (long long*)gate, (unsigned int*)gate_ctr);
}

void r4d_ar_twoshot_8rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                long peer_scratch6, long peer_flags0, long peer_flags1,
                                long peer_flags2, long peer_flags3, long peer_flags4,
                                long peer_flags5, long peer_flags6, long my_scratch,
                                long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                long out, long n_elem, long dtype, long rank, long stream,
                                long nblocks, long nthreads, long drain, long acq, long pub,
                                const int* shard_weights, long pieces, long timeline,
                                long form) {
  const long ps[7] = {peer_scratch0, peer_scratch1, peer_scratch2, peer_scratch3,
                      peer_scratch4, peer_scratch5, peer_scratch6};
  const long pf[7] = {peer_flags0, peer_flags1, peer_flags2, peer_flags3,
                      peer_flags4, peer_flags5, peer_flags6};
  /* No link gate at eight ranks: the row answers 1, the engine releases by barrier. */
  ar_twoshot_launch<8>(ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride16, inp, out, n_elem,
                       dtype, rank, stream, nblocks, nthreads, drain, acq, pub, 0L, 0L,
                       shard_weights, pieces, timeline, form);
}
