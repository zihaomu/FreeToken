// r4d_gdn: the gated-delta-net chunked scan, gfx1201 WMMA. Entry point
// r4d_gdn_chunk_scan_k128_v128_c64_bf16 -- head_k 128, head_v 128, chunk 64, bf16 in.
// Used by prefill and chunked prefill; decode uses the recurrent update kernel. Replaces FLA's
// recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h + chunk_fwd_o: the recurrent state stays in
// WMMA accumulators across the chunk loop, so h / w / u / v_new never touch HBM.
//
// Invariant: every decay factor formed is <= 1. The intra-chunk gate span reaches 558 nats on
// real Qwen3.8 prefill activations, so splitting the exponent about a per-chunk reference c
// (e^{g_i-c} * e^{c-g_j}) overflows fp32 on one side and underflows on the other for any c.
// e^{g_i-g_j} (j <= i) and e^{g_last-g_j} stay in (0, 1].
//
//   ge[j]=e^{g_j}   gg[j]=g_j   gs[j]=e^{g_last-g_j}
//   U  = K@S            W  = beta*(V - ge*U)      V' = A@W          (undecayed)
//   P  = Q@K^T          sA = scale*e^{g_i-g_j}*tril(P)
//   O  = scale*ge*(Q@S) + sA@V'
//   S  = e^{g_last}*S + V'^T@(gs*K)
// A is FLA's solve_tril output (decay folded in), so no decay enters W or V'. The (i,j) weight
// is applied per element in the score step (eight __expf per tile), and the state's
// e^{g_last-g_j} is applied to K, which is otherwise dead by then. V' stays undecayed because its
// consumers need different references (e^{g_i-g_j} for the output, e^{g_last-g_j} for the state),
// and serving both from one scaling would need the unsafe split.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_gdn_args.h"
#include "r4d_gdn_wmma.h"
// Barriers are lds_barrier(): __syncthreads() fences all address spaces, so the barrier after the
// global O store would wait on s_wait_storecnt_dscnt 0x0.

#define BT   64
#define KD  128
#define VD  128
#define BV   64                 // V per workgroup; also halves the Q@K^T redundancy
#define KP  132                 // row pitch of the [.][KD] buffers, in shorts
#define AP   68                 // row pitch of the [.][BT] buffers, in shorts
#define WP  AP
// Waves per workgroup. The [4 token][4 v] tile grid is fixed by BT and BV, so 8 waves carry two
// tiles each. The kernel is at 242 VGPRs; one more live address spills.
#define NW    8
#define NTHR (NW * 32)
#define NTV   2                 // n tiles per wave in the main / V' / O loops
// A wave's slice of the 8 k-tile x 4 v-tile state grid is 2 x 2. 1 x 4 halves the transposed-K
// reads (fragT) but is ~20% slower: holding four V' fragments live costs more than they save.
#define SKT   2
#define SVT   2
#define NST (SKT * SVT)         // state tiles per wave

struct Smem {
  unsigned short k[BT][KP];     // A: K   /  B: K^T for P=Q@K^T
  unsigned short A[BT][AP];     // B-source for sA (written as [i][j])
  unsigned short S[BV][KP];     // B: state, V-major (matches FLA's [V,K] layout)
  unsigned short W[BV][WP];     // B: W (transposed)
  unsigned short D[BV][AP];     // staged V as [token][v]; then V' as [v][token].
                                // V dies in the W update; V' is written after barrier (2).
  float ge[BT], gv[BT], gg[BT], bt[BT];   // gv = gs, gg = raw g
};

// Choices measured against their alternatives on gfx1201, kept so they are not undone:
//   - LDS rows in the main loop are opaque 32-bit byte offsets (fragO), global q / A fragments a
//     32-bit offset off a uniform base (fragGo). The V' / O / state loops keep plain pointers:
//     offsets there spill at 8 waves.
//   - k and v are staged eight shorts per lane (global_load_b128), issued as a block before the
//     main loop: interleaving them into it is ~5% slower.
//   - The V' loop's A fragments are not preloaded: 16 more live VGPRs cost ~16%.
//   - Every bf16 pack rounds (f2bf4). Truncating W, sA and V' is ~2% faster but roughly triples
//     the fp64-oracle error, since they feed the recurrence.
#define SB 2                    // k steps per sched group in the V' and O += loops
#define SS 1                    // k steps per sched group in the state update
// LDS reads and WMMAs per main-loop k step, for the sched groups: one A fragment (K row mt) shared
// by every tile, plus S and K per n tile, one ds_load_2addr_b64 each (fragP). These must match the
// loop exactly; a wrong count schedules worse than no pattern.
#define MLDS (1 + 2 * NTV)
#define MWMA (3 * NTV)
// byte offset of buffer[base + lo][4*hi] from the base of the shared block
#define ROWO(buf, pitch, base) \
  ((unsigned)((const char*)&(buf)[0][0] - (const char*)&s) + \
   (unsigned)((((base) + lo) * (pitch) + 4 * hi) * 2))

// SDT is the state cache's element type (0 = fp32, 1 = fp16) and nothing else; all arithmetic is
// fp32. The cache is shared with the recurrent update (whose time is the state traffic, so it
// favours fp16) and allocated from vLLM's --mamba-ssm-cache-dtype, so both kernels must agree.
typedef _Float16 cs_h8 __attribute__((ext_vector_type(8)));
template <int SDT> struct cs_state { using T = float; };
template <> struct cs_state<1>     { using T = _Float16; };

// H0 / HT: whether the optional initial and final state operands are present (template parameters,
// so the common both-present form has no runtime test). As in libref, no initial state starts from
// zero and no final state stores nothing.
template <int SDT, bool H0 = true, bool HT = true>
__global__ __launch_bounds__(NTHR) void r4d_gdn_chunk_scan_kernel(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, const unsigned short* __restrict__ Amat,
    const float* __restrict__ gcs, const float* __restrict__ beta,
    const typename cs_state<SDT>::T* __restrict__ h0, unsigned short* __restrict__ o,
    typename cs_state<SDT>::T* __restrict__ ht, const int* __restrict__ cu,
    const int* __restrict__ sidx, long sidx_stride, long st_slot, long st_head,
    int H, int Hg, float scale)
{
  __shared__ Smem s;
  const int bv = blockIdx.x, hv = blockIdx.y, nq = blockIdx.z;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int hq = hv / (H / Hg), v0 = bv * BV;
  const int tb = cu[nq], te = cu[nq + 1];
  // The state slot comes from the cache manager (sidx), not the batch position; without sidx the
  // kernel uses state[nq], valid only while slots match batch order. A row may name several slots
  // (speculative decode adds scratch slots); column 0 is the state and the only one used here.
  const long si = sidx ? (long)sidx[(size_t)nq * sidx_stride] : (long)nq;
  if (si < 0) return;                                        // a padded row owns no state
  const size_t st_base = (size_t)si * st_slot + (size_t)hv * st_head;
  const int nchunk = (te - tb + BT - 1) / BT;
  const int lo = lane & 15, hi = lane >> 4;

  const int mt = w >> 1, ntb = (w & 1) * 2;                 // [BT][BV]: 2 tiles/wave
  // The state accumulator is held transposed, C[m=k-dim][n=v]. A and B fragments share a lane
  // layout, so swapping the state update's operands is free, and k-dim becomes the per-lane
  // contiguous axis: the per-chunk writeback to s.S is four 8-short stores instead of 32 scalar.
  const int kt0 = (w >> 1) * 2, vt0 = (w & 1) * 2;           // 2 x 2

#define PFW 8                   // shorts staged per lane per step
#define PFZERO make_uint4(0u, 0u, 0u, 0u)
#define NKB (BT * KD / (NTHR * PFW))
#define NVB (BT * BV / (NTHR * PFW))
  uint4 kb[NKB], vb[NVB];
  // Staging index, split into a wave-uniform row and a loop-invariant lane column. As
  // `i = tid*4 + u*NTHR*4; r = i>>7` the row looks divergent to LLVM and each load costs ~13
  // instructions of 64-bit address math; readfirstlane makes the row provably uniform (SALU).
  // Identities (exact, since u*NTHR*4 is a multiple of both 128 and 64):
  //   k: r = (tid>>5) + NW*u          c = (tid*4) & 127
  //   v: r = 2*(tid>>5) + vh + 2*NW*u  c = (tid*4) & 63   vh = (tid>>4) & 1
  const int wu = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int kc = (tid * PFW) & (KD - 1), vc = (tid * PFW) & (BV - 1);
  const int kr0 = 2 * wu + ((tid >> 4) & 1), vr0 = 4 * wu + ((tid >> 3) & 3);
#define KRU (2 * NW)
#define VRU (4 * NW)
  // Do not change: staged rows past the chunk end are zeroed, not clamped to the last live row
  // (~4% faster). A clamp looks harmless (beta and gv are 0 there, the tril mask drops padding j,
  // the O store is guarded), yet with it any prompt whose trailing chunk is not a multiple of 16
  // tokens returns one repeated token. Zeroing k alone fixes it, so the leak is through some
  // consumer of s.k whose path is not yet understood.
#define PFSTORE(dst, val)                                                            \
  do {                                                                               \
    unsigned short* dp_ = (dst);                                                     \
    *(uint2*)dp_ = make_uint2((val).x, (val).y);                                      \
    *(uint2*)(dp_ + 4) = make_uint2((val).z, (val).w);                                \
  } while (0)
#define PREFETCH(CH)                                                                 \
  {                                                                                  \
    const int pc0 = tb + (CH) * BT, pnv = min(BT, te - pc0);                         \
    const unsigned short* kb_ = k + ((size_t)pc0 * Hg + hq) * KD;                   \
    const unsigned short* vb_ = v + ((size_t)pc0 * H + hv) * VD + v0;               \
    _Pragma("unroll") for (int u = 0; u < NKB; ++u) {                                \
      const int r = kr0 + KRU * u;                                                   \
      kb[u] = r < pnv ? *(const uint4*)(kb_ + (unsigned)(r * (Hg * KD)) + kc) : PFZERO; \
    }                                                                                \
    _Pragma("unroll") for (int u = 0; u < NVB; ++u) {                                \
      const int r = vr0 + VRU * u;                                                   \
      vb[u] = r < pnv ? *(const uint4*)(vb_ + (unsigned)(r * (H * VD)) + vc) : PFZERO; \
    }                                                                                \
  }

  v8f St[NST];                  // state: SKT k-tiles x SVT v-tiles
#define KT(t) (kt0 + (t) / SVT)
#define VT(t) (vt0 + (t) % SVT)
#define SROW(t) ((size_t)(v0 + VT(t) * 16 + lo) * KD + KT(t) * 16 + 8 * hi)
  if constexpr (!H0) {
#pragma unroll
    for (int t = 0; t < NST; ++t)
#pragma unroll
      for (int e = 0; e < 8; ++e) St[t][e] = 0.0f;
  } else {
    const typename cs_state<SDT>::T* p = h0 + st_base;
#pragma unroll
    for (int t = 0; t < NST; ++t) {
      // Eight elements per row-tile: two 16-byte accesses in fp32, one in fp16 (16-byte
      // aligned since SROW ends in 8*hi).
      if constexpr (SDT) {
        const cs_h8 x = *(const cs_h8*)(p + SROW(t));
#pragma unroll
        for (int e = 0; e < 8; ++e) St[t][e] = (float)x[e];
      } else {
        const float4 a = *(const float4*)(p + SROW(t));
        const float4 b = *(const float4*)(p + SROW(t) + 4);
        St[t][0]=a.x; St[t][1]=a.y; St[t][2]=a.z; St[t][3]=a.w;
        St[t][4]=b.x; St[t][5]=b.y; St[t][6]=b.z; St[t][7]=b.w;
      }
    }
  }
#define SWRITE()                                                                      \
  _Pragma("unroll") for (int t = 0; t < NST; ++t) {                                     \
    unsigned short* sp = &s.S[VT(t) * 16 + lo][KT(t) * 16 + 8 * hi];                  \
    *(uint2*)sp       = f2bf4(St[t][0], St[t][1], St[t][2], St[t][3]);                \
    *(uint2*)(sp + 4) = f2bf4(St[t][4], St[t][5], St[t][6], St[t][7]);                \
  }
  SWRITE();
  lds_barrier();
  PREFETCH(0);

  const char* const sbase = (const char*)&s;
  // scale goes inside the score exponent, exp(g_i + log(scale) - g_j), saving eight multiplies
  // per tile.
  const float lgs = __logf(scale);
  for (int ci = 0; ci < nchunk; ++ci) {
    const int c0 = tb + ci * BT, nval = min(BT, te - c0);
    const int lim = nval - 1;
    const unsigned short* qg = q + ((size_t)c0 * Hg + hq) * KD;
    const unsigned short* ag = Amat + ((size_t)c0 * H + hv) * BT;
    const size_t qrs = (size_t)Hg * KD, ars = (size_t)H * BT;
    const float gl = gcs[(size_t)(c0 + lim) * H + hv];

    // Issued before the staging store so the LDS writes cover their latency. Only waves 0-1
    // (tid < BT) run this; the others wait at (1).
    float ggr = 0.f, btr = 0.f;
    if (tid < BT) {
      bool ok = tid < nval;
      ggr = ok ? gcs[(size_t)(c0 + tid) * H + hv] : 0.f;
      btr = ok ? beta[(size_t)(c0 + tid) * H + hv] : 0.f;
    }
    { // consume the previous iteration's prefetch, issued a whole chunk of compute ago
#pragma unroll
      for (int u = 0; u < NKB; ++u) PFSTORE(&s.k[kr0 + KRU * u][kc], kb[u]);
#pragma unroll
      for (int u = 0; u < NVB; ++u) PFSTORE(&s.D[vr0 + VRU * u][vc], vb[u]);
    }
    if (tid < BT) {
      bool ok = tid < nval;
      s.bt[tid] = btr;
      s.ge[tid] = __expf(ggr);
      // g is a within-chunk cumsum of log-sigmoid gates, so it is non-increasing and gl is its
      // minimum: gl - ggr <= 0, and gs underflowing to 0 is correct.
      // A padding row gets gl, not 0: it is scored against live j with very negative g_j, and
      // g_i = 0 would give exp(+400) = inf in s.A. With gl (the chunk minimum) the exponent
      // stays in (0, scale]. The row is discarded anyway (O store is guarded).
      s.gg[tid] = ok ? ggr : gl;
      s.gv[tid] = ok ? __expf(gl - ggr) : 0.f;                           // gs
    }
    lds_barrier();                                           // (1)
    PREFETCH(min(ci + 1, nchunk - 1));       // issue only; consumed next iteration

    // ---- U = K@S, O^T = S@Q^T and P^T = K@Q^T share the q and k fragments.
    // Tiles above the diagonal (A is unit lower triangular, sA is masked) are computed anyway:
    // skipping them is 2.4% slower, since some wave still carries two tiles, so the barrier
    // arrival does not move, and the guards stop fragment loads batching ahead of the WMMAs.
    v8f O[NTV], u[NTV], p[NTV];
#pragma unroll
    for (int i = 0; i < NTV; ++i) { O[i] = (v8f){}; u[i] = (v8f){}; p[i] = (v8f){}; }
    {
      const char* sb = sbase;
      unsigned oSn[NTV], okn[NTV];
      unsigned okm = ROWO(s.k, KP, mt * 16);
      LDSOPAQUE(okm);
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        oSn[i] = ROWO(s.S, KP, (ntb + i) * 16);
        okn[i] = ROWO(s.k, KP, (ntb + i) * 16);
        LDSOPAQUE(oSn[i]); LDSOPAQUE(okn[i]);
      }
#define pSnf(i, ko) fragO(sb, oSn[i] + (ko) * 2u)
#define pkmf(ko)    fragO(sb, okm + (ko) * 2u)
#define pknf(i, ko) fragO(sb, okn[i] + (ko) * 2u)
      const unsigned qo = rowGo((unsigned)qrs, mt * 16, lim, lane);
#define pqf(ko) fragGo(qg, qo + (ko))
#pragma unroll
      for (int sk = 0; sk < KD / 16; ++sk) {
        const int ko = sk * 16;
        v8bf ak = pkmf(ko), aq = pqf(ko);
#pragma unroll
        for (int i = 0; i < NTV; ++i) {
          v8bf b = pSnf(i, ko);
          u[i] = mma(ak, b, u[i]);
          O[i] = mma(b, aq, O[i]);                     // O^T: m = v-tile, n = token-tile
          p[i] = mma(pknf(i, ko), aq, p[i]);           // P^T: m = j
        }
      }
      // Issue one k step's LDS reads against the previous step's WMMAs (per step: 5 LDS
      // fragments, 1 global q fragment = 2 loads, 6 WMMA). Otherwise the scheduler sinks each
      // read onto its consumer and every WMMA waits on a full s_wait_dscnt 0x0 drain.
      // GRAN k steps per group. Coarser is better: the group barrier pins the order at every
      // boundary, and a per-step pattern over-constrains a loop the scheduler can interleave.
#define GRAN 2
#pragma unroll
      for (int i = 0; i < KD / (16 * GRAN); ++i) {
        __builtin_amdgcn_sched_group_barrier(0x020, 2 * GRAN, 0);
        __builtin_amdgcn_sched_group_barrier(0x100, MLDS * GRAN, 0);
        __builtin_amdgcn_sched_group_barrier(0x008, MWMA * GRAN, 0);
      }
    }
    {
      float g = scale * s.ge[mt * 16 + lo];             // token index is constant per lane
#pragma unroll
      for (int i = 0; i < NTV; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) O[i][e] *= g;
    }
    // W = beta * (V - ge*U)   (beta folded here: (A*beta_col)@W == A@(beta_row*W))
    {
      float wa[NTV][8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {           // beta / ge read once for both tiles
        const int tok = mt * 16 + 8 * hi + e;
        const float f = s.bt[tok], g = s.ge[tok];
#pragma unroll
        for (int i = 0; i < NTV; ++i)
          wa[i][e] = f * (bf2f(s.D[tok][(ntb + i) * 16 + lo]) - g * u[i][e]);
      }
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        unsigned short* wp = &s.W[(ntb + i) * 16 + lo][mt * 16 + 8 * hi];
        *(uint2*)wp       = f2bf4(wa[i][0], wa[i][1], wa[i][2], wa[i][3]);
        *(uint2*)(wp + 4) = f2bf4(wa[i][4], wa[i][5], wa[i][6], wa[i][7]);
      }
    }
    // sA = scale * e^{g_i-g_j} * tril(P). Row ii is constant per lane and the eight g_j are
    // uniform per 16 lanes: eight broadcast LDS reads, v_sub and v_exp per tile (log(scale) is
    // in the exponent). Where kept, g_i - g_j <= 0, so the exp cannot overflow.
    {
      const int ii = mt * 16 + lo;
      const float gis = s.gg[ii] + lgs;
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        const int j0 = (ntb + i) * 16 + 8 * hi;
        float a[8];
#pragma unroll
        for (int e = 0; e < 8; ++e)
          a[e] = (j0 + e <= ii) ? p[i][e] * __expf(gis - s.gg[j0 + e]) : 0.f;
        *(uint2*)&s.A[ii][j0]     = f2bf4(a[0], a[1], a[2], a[3]);
        *(uint2*)&s.A[ii][j0 + 4] = f2bf4(a[4], a[5], a[6], a[7]);
      }
    }
    lds_barrier();                                           // (2)

    // ---- V' = gv * (A @ W)  -> D (transposed store is one contiguous pair of dwords)
    {
      v8f d[NTV];
#pragma unroll
      for (int i = 0; i < NTV; ++i) d[i] = (v8f){};
      const unsigned ao = rowGo((unsigned)ars, mt * 16, lim, lane);
#define paf(ko) fragGo(ag, ao + (ko))
      const unsigned short* pwn[NTV];
#pragma unroll
      for (int i = 0; i < NTV; ++i) pwn[i] = rowP(&s.W[0][0], WP, (ntb + i) * 16, lane);
#define pwnf(i, ko) fragP(pwn[i] + (ko))
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) {
        const int ko = sk * 16;
        v8bf a = paf(ko);
#pragma unroll
        for (int i = 0; i < NTV; ++i) d[i] = mma(a, pwnf(i, ko), d[i]);
      }
      // Per group: SB steps x (1 global A fragment, NTV LDS, NTV WMMA).
#pragma unroll
      for (int i = 0; i < BT / (16 * SB); ++i) {
        __builtin_amdgcn_sched_group_barrier(0x020, 2 * SB, 0);
        __builtin_amdgcn_sched_group_barrier(0x100, NTV * SB, 0);
        __builtin_amdgcn_sched_group_barrier(0x008, NTV * SB, 0);
      }
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        unsigned short* dp = &s.D[(ntb + i) * 16 + lo][mt * 16 + 8 * hi];
        *(uint2*)dp       = f2bf4(d[i][0], d[i][1], d[i][2], d[i][3]);
        *(uint2*)(dp + 4) = f2bf4(d[i][4], d[i][5], d[i][6], d[i][7]);
      }
    }
    // K <- gs * K in place, for the state's V'^T @ K: (diag(gs) V')^T K == V'^T (diag(gs) K),
    // and K is otherwise dead. All raw-K reads (U, P) retired before barrier (2); s.k is
    // restaged next chunk; barrier (3) publishes the scaled K. A decayed copy of V' would need
    // an extra barrier, since s.S is the only free buffer and SWRITE races the state matmul.
#pragma unroll
    for (int u = 0; u < BT * (KD / 4) / NTHR; ++u) {
      const int idx = tid + u * NTHR;
      const int t = idx / (KD / 4), c = (idx % (KD / 4)) * 4;
      const float f = s.gv[t];
      unsigned short* kp = &s.k[t][c];
      *(uint2*)kp = f2bf4(bf2f(kp[0]) * f, bf2f(kp[1]) * f,
                           bf2f(kp[2]) * f, bf2f(kp[3]) * f);
    }
    lds_barrier();                                           // (3)

    // ---- O += sA @ V' ; store.  V' also feeds the state update, unchanged.
    {
      const unsigned short* pA = rowP(&s.A[0][0], AP, mt * 16, lane);
      const unsigned short* pdn[NTV];
#pragma unroll
      for (int i = 0; i < NTV; ++i) pdn[i] = rowP(&s.D[0][0], AP, (ntb + i) * 16, lane);
#define pAf(ko)     fragP(pA + (ko))
#define pdnf(i, ko) fragP(pdn[i] + (ko))
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) {
        const int ko = sk * 16;
        v8bf b = pAf(ko);
#pragma unroll
        for (int i = 0; i < NTV; ++i) O[i] = mma(pdnf(i, ko), b, O[i]);
      }
      // Per group: SB steps x (1 + NTV LDS fragments, NTV WMMA).
#pragma unroll
      for (int i = 0; i < BT / (16 * SB); ++i) {
        __builtin_amdgcn_sched_group_barrier(0x100, (1 + NTV) * SB, 0);
        __builtin_amdgcn_sched_group_barrier(0x008, NTV * SB, 0);
      }
    }
    {
      int tok = mt * 16 + lo;
      if (tok < nval) {
        unsigned short* op = o + ((size_t)(c0 + tok) * H + hv) * VD + v0;
#pragma unroll
        for (int i = 0; i < NTV; ++i) {
          unsigned short* q2 = op + (ntb + i) * 16 + 8 * hi;
          *(uint2*)q2       = f2bf4(O[i][0], O[i][1], O[i][2], O[i][3]);
          *(uint2*)(q2 + 4) = f2bf4(O[i][4], O[i][5], O[i][6], O[i][7]);
        }
      }
    }

    // ---- S = e^{gl} * S + V'^T @ (gs*K)   (B needs K as [dim][token] -> strided fragT)
    {
      const float ec = __expf(gl);            // gl <= 0: no lam, the decay is already on K
#pragma unroll
      for (int t = 0; t < NST; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) St[t][e] *= ec;
      const unsigned short* pvn[SVT];
#pragma unroll
      for (int j = 0; j < SVT; ++j) pvn[j] = rowP(&s.D[0][0], AP, (vt0 + j) * 16, lane);
#define pvnf(j, ko) fragP(pvn[j] + (ko))
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) {
        v8bf bv[SVT];
#pragma unroll
        for (int j = 0; j < SVT; ++j) bv[j] = pvnf(j, sk * 16);
#pragma unroll
        for (int ki = 0; ki < SKT; ++ki) {
          v8bf ak = fragT(&s.k[0][0], KP, (kt0 + ki) * 16, sk * 16, lane);  // K^T: m = k-dim
#pragma unroll
          for (int j = 0; j < SVT; ++j)
            St[ki * SVT + j] = mma(ak, bv[j], St[ki * SVT + j]);
        }
      }
      // Per group: SS steps x (SKT fragT of 8 scalar reads each + SVT fragP, NST WMMA).
#pragma unroll
      for (int i = 0; i < BT / (16 * SS); ++i) {
        __builtin_amdgcn_sched_group_barrier(0x100, (8 * SKT + SVT) * SS, 0);
        __builtin_amdgcn_sched_group_barrier(0x008, NST * SS, 0);
      }
    }
    // No barrier needed: s.S aliases nothing, and barrier (2) proves every wave finished its
    // U/O reads of s.S this chunk.
    SWRITE();
    lds_barrier();                                           // (4)
  }

  if constexpr (!HT) return;
  typename cs_state<SDT>::T* p = ht + st_base;
#pragma unroll
  for (int t = 0; t < NST; ++t) {
    if constexpr (SDT) {
      cs_h8 x;
#pragma unroll
      for (int e = 0; e < 8; ++e) x[e] = (_Float16)St[t][e];
      *(cs_h8*)(p + SROW(t)) = x;
    } else {
      *(float4*)(p + SROW(t))     = make_float4(St[t][0], St[t][1], St[t][2], St[t][3]);
      *(float4*)(p + SROW(t) + 4) = make_float4(St[t][4], St[t][5], St[t][6], St[t][7]);
    }
  }
}

// Slot-indexed form (the one radiance calls). `sidx` names each batch row's state slot; without
// it the cache is indexed by row. `st_slot` and `st_head` are the cache strides in elements, which
// may exceed H*V*K when a caller pads its slots (vLLM pads to the attention page size). Pass
// sidx = null and strides 0 for the packed [N, H, V, K] layout.
extern "C" int r4d_gdn_chunk_scan_k128_v128_c64_bf16_sidx(
    const void* q, const void* k, const void* v, const void* A, const void* g,
    const void* beta, const void* h0, void* o, void* ht, const void* cu,
    const void* ssm_state_indices, long indices_stride, long state_slot_stride,
    long state_head_stride,
    int N, int H, int Hg, int K, int V, int bt, float scale, int state_fp16, void* stream)
{
  if (K != KD || V != VD || bt != BT) return -1;
  // hq = hv / (H / Hg) truncates: H not a multiple of Hg would silently map v heads to the wrong
  // k heads. Refused with -2, as in the kkt_solve and recurrent_update siblings.
  if (Hg <= 0 || H % Hg) return -2;
  const long st_head = state_head_stride > 0 ? state_head_stride : (long)VD * KD;
  const long st_slot = state_slot_stride > 0 ? state_slot_stride : (long)H * VD * KD;
  dim3 grid(V / BV, H, N), blk(NTHR);
#define CS_LAUNCH(SDTC, H0C, HTC)                                                          \
  hipLaunchKernelGGL((r4d_gdn_chunk_scan_kernel<SDTC, H0C, HTC>), grid, blk, 0,            \
      (hipStream_t)stream,                                                                 \
      (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,        \
      (const unsigned short*)A, (const float*)g, (const float*)beta,                       \
      (const cs_state<SDTC>::T*)h0, (unsigned short*)o, (cs_state<SDTC>::T*)ht,            \
      (const int*)cu, (const int*)ssm_state_indices, (indices_stride > 0 ? indices_stride : 1),  \
      st_slot, st_head, H, Hg, scale)
#define CS_PICK(SDTC)                                                                      \
  do { if (h0 && ht) CS_LAUNCH(SDTC, true, true);                                          \
       else if (h0)  CS_LAUNCH(SDTC, true, false);                                         \
       else if (ht)  CS_LAUNCH(SDTC, false, true);                                         \
       else          CS_LAUNCH(SDTC, false, false); } while (0)
  if (state_fp16) CS_PICK(1); else CS_PICK(0);
#undef CS_PICK
#undef CS_LAUNCH
  return (int)hipGetLastError();
}

// Packed-layout entry point for the out-of-tree vLLM caller: one state per batch row,
// [N, H, V, K].
extern "C" int r4d_gdn_chunk_scan_k128_v128_c64_bf16(
    const void* q, const void* k, const void* v, const void* A, const void* g,
    const void* beta, const void* h0, void* o, void* ht, const void* cu,
    int N, int H, int Hg, int K, int V, int bt, float scale, int state_fp16, void* stream)
{
  return r4d_gdn_chunk_scan_k128_v128_c64_bf16_sidx(q, k, v, A, g, beta, h0, o, ht, cu,
                                                    nullptr, 0, 0, 0, N, H, Hg, K, V, bt, scale,
                                                    state_fp16, stream);
}

// The compiled geometry, so callers can check a model's GDN shape against it.
extern "C" void r4d_gdn_dims(int* head_k, int* head_v, int* chunk) {
  if (head_k) *head_k = KD;
  if (head_v) *head_v = VD;
  if (chunk)  *chunk  = BT;
}

// ============================================================== the plugin ABI entry points
// Shared operand helpers: r4d_gdn_args.h.
/* ------------------------------------------------------------------ chunk scan */

enum { CS_Q = 0, CS_K, CS_V, CS_A, CS_G, CS_BETA, CS_H0, CS_CU, CS_O, CS_HT, CS_SIDX };

extern "C" int r4d_rad_gdn_chunk_scan(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* q  = r4d_opt(a, CS_Q);
    const RadTensor* k  = r4d_opt(a, CS_K);
    const RadTensor* v  = r4d_opt(a, CS_V);
    const RadTensor* A  = r4d_opt(a, CS_A);
    const RadTensor* g  = r4d_opt(a, CS_G);
    const RadTensor* bt = r4d_opt(a, CS_BETA);
    const RadTensor* h0 = r4d_opt(a, CS_H0);
    const RadTensor* cu = r4d_opt(a, CS_CU);
    const RadTensor* o  = r4d_opt(a, CS_O);
    const RadTensor* ht = r4d_opt(a, CS_HT);
    const RadTensor* sidx = r4d_opt(a, CS_SIDX);
    if (!q || !k || !v || !A || !g || !bt || !cu || !o) return R4D_NO(RAD_E_INVAL);
    if (q->rank != 3 || k->rank != 3 || v->rank != 3 || o->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (A->rank != 3 || g->rank != 2 || bt->rank != 2) return R4D_NO(RAD_E_SHAPE);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;

    /* The entry point takes no pitches: everything must be dense. */
    if (!r4d_contig(q) || !r4d_contig(k) || !r4d_contig(v) || !r4d_contig(A) ||
        !r4d_contig(g) || !r4d_contig(bt) || !r4d_contig(o)) return R4D_NO(RAD_E_STRIDE);

    /* bf16 q/k/v/A/out, fp32 gate/beta. */
    R4D_DT(q, RAD_BF16); R4D_DT(k, RAD_BF16); R4D_DT(v, RAD_BF16); R4D_DT(A, RAD_BF16);
    R4D_DT(o, RAD_BF16); R4D_DT(g, RAD_F32);  R4D_DT(bt, RAD_F32);
    /* The state cache is f32 or f16, taken from the operand's dtype (a property of the
     * allocation), so this kernel and the recurrent update, which share the cache, agree on it. */
    const uint32_t st_dt = h0 ? h0->dtype : (ht ? ht->dtype : (uint32_t)RAD_F32);
    if (st_dt != RAD_F32 && st_dt != RAD_F16) return R4D_NO(RAD_E_DTYPE);
    if (h0) R4D_DT(h0, st_dt);
    if (ht) R4D_DT(ht, st_dt);
    /* Slot strides come from the tensor, not from H, V and K, so a cache with padded slots is
     * served too; the engine's own state cache is dense ([n_states, H, V, K] per layer). h0 and
     * ht must share strides, since the kernel uses one base offset for both. */
    long long st_slot = 0, st_head = 0;
    if (h0) {
        if (h0->rank < 2) return R4D_NO(RAD_E_SHAPE);
        st_slot = h0->stride[0];
        st_head = h0->stride[1];
    }
    if (ht) {
        if (ht->rank < 2) return R4D_NO(RAD_E_SHAPE);
        if (h0 && (ht->stride[0] != st_slot || ht->stride[1] != st_head))
            return R4D_NO(RAD_E_STRIDE);
        st_slot = ht->stride[0];
        st_head = ht->stride[1];
    }
    if (sidx) {
        R4D_DT(sidx, RAD_I32);
        if (sidx->rank < 1 || sidx->shape[0] != N) return R4D_NO(RAD_E_SHAPE);
        if (sidx->stride[sidx->rank - 1] != 1) return R4D_NO(RAD_E_STRIDE);
    }

    const long long T  = q->shape[0];
    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    const long long bt_chunk = A->shape[2];
    if (k->shape[0] != T || k->shape[1] != Hg || k->shape[2] != K) return R4D_NO(RAD_E_SHAPE);
    if (v->shape[0] != T || o->shape[0] != T || o->shape[1] != H || o->shape[2] != V)
        return R4D_NO(RAD_E_SHAPE);
    if (A->shape[0] != T || A->shape[1] != H) return R4D_NO(RAD_E_SHAPE);
    if (g->shape[0] != T || g->shape[1] != H) return R4D_NO(RAD_E_SHAPE);
    if (bt->shape[0] != T || bt->shape[1] != H) return R4D_NO(RAD_E_SHAPE);
    if (H == 0 || Hg == 0 || H % Hg) return R4D_NO(RAD_E_SHAPE);

    const float scale = (float)r4d_getf(a, "scale", 1.0);
    const int state_fp16 = st_dt == RAD_F16;

    return R4D_RAW(r4d_gdn_chunk_scan_k128_v128_c64_bf16_sidx(
        q->data, k->data, v->data, A->data, g->data, bt->data,
        h0 ? h0->data : nullptr, o->data, ht ? ht->data : nullptr, cu->data,
        sidx ? sidx->data : nullptr, sidx && sidx->rank >= 2 ? (long)sidx->stride[0] : 1,
        (long)st_slot, (long)st_head,
        N, (int)H, (int)Hg, (int)K, (int)V, (int)bt_chunk, scale, state_fp16, s));
}

