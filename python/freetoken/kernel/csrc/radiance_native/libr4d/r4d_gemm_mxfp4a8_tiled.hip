// r4d_gemm_mxfp4a8_tiled.hip - r4d_gemm_mxfp4a8_tiled: the OCP-MXFP4 x fp8 GEMM for prefill row
// counts, gfx1201. C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with the operands, weight layout and fold
// of r4d_gemm_mxfp4a8_nt_m64 and r4d_gemm_mxfp4a8_decode, for the band above theirs.
//
// The decode kernels cover at most 128 rows per block, so at prefill they would re-read the weight
// ceil(M/128) times. Here a block is a 256 x (32*TN) tile and each weight byte is unpacked once per
// K slab into LDS, shared by the block's four row-waves.
//
// A goes from global straight into registers. A WMMA A fragment (lane l holds row l%16, k-bytes
// 8*(l/16)..+8) spans 16 rows of a row-major plane, and staging it through LDS would cost a write,
// a barrier and a second read. So A is first re-tiled into the op's scratch in fragment order:
//
//     AT[((m/16) * (K/16) + k/16) * 256 + ((m%16) + 16 * ((k/8)%2)) * 8 + k%8] = A[m][k]
//
// A fragment is 256 contiguous bytes and a lane's eight are one aligned 8-byte load, so a slab's A
// fragments issue as one clause of global loads off a wave-uniform SGPR base (SADDR form). The
// re-tile costs one read and write of A, far less than it saves on large projections, and is
// internal to the op. Rows past M in the last fragment are written as zeros.
//
// Weight and fold as in the skinny kernel: the fragment-order weight from r4d_rad_layout_mxfp4,
// folded through r4d_mxfp4.h's table, so all three MXFP4 kernels produce identical e4m3 bytes. The
// table is copied to LDS per block; from global it would be a dependent load behind the A
// fragments, and the in-order load counter would make the fold wait for all of them.
// 2^(Wref[n]-127) is applied in the epilogue (see r4d_gemm_mxfp4a8_nt_m64.hip).
//
// Tile: 8 waves as 4 (M) x 2 (N), each owning TM=4 x TN fragments, so a block covers BMF = 256
// rows and BNF = 2*TN*16 columns. TN=4 halves the column-block count and wins from M = 1024 on
// (on gfx1201 at a 27B's per-rank weights, mixed below), so tn=0 picks TN=4 from M >= 1024. The K
// slab is 128 at TN=2 (64 if K % 128 != 0) and 64 at TN=4, which holds twice the accumulators.
// TN does not change an output's bits: each fragment runs the same WMMAs in the same K order.
//
// Barriers fence LDS only: __syncthreads() also drains the global load counter, and nothing in
// global memory is shared between the waves, so the next slab's A loads stay in flight.
//
// The weight words and exponents a thread stages are loaded a slab ahead, into registers, right
// after the staging barrier, so their DRAM round trip runs under the slab's WMMAs instead of in
// front of the next unpack (+1 VGPR at TN=4). Loading the A fragments a slab ahead too (+38 VGPRs)
// is slower.
//
// A thin last row of blocks (1..160 rows, under 1024 rows in all) runs a second kernel whose
// blocks of that row skip the WMMAs of their padding fragments past M. Every other M runs the
// plain kernel, whose body declares its own LDS: a guard inside the one kernel, or the body
// taking the LDS by pointer, slows the full blocks.
//
// Staging is clamped, never predicated: a predicated load gets its own exec-mask region and the
// compiler cannot carry a counted s_wait_loadcnt across the merge. Clamped loads (tile past M ->
// last real tile, row past N -> last aligned group) issue back to back; the epilogue drops them.
//
// The accumulator is C transposed: the WMMA takes the weight fragment as A and the activation as B
// (same register layout), so a lane's accumulator for (i, j) holds one row and eight consecutive
// columns. A lane needs one activation scale per row, and each (i, j) is one 16-byte store.
//
// Two activation scale grids: one f32 per row (ABLK=false) or one per (row, 128 of K) (ABLK=true,
// as every fused producer in arch/common writes). With ABLK the accumulator is carried in units of
// the current block's scale: at each 128 boundary every row is multiplied by s_prev / s_next
// (~K/128 fp32 ulps of extra rounding), and the epilogue applies the last scale. Thread t owns row
// t of the block, loads the next boundary's two scales a boundary early and writes the ratio to
// LDS; a lane reads one ratio per row.
//
// Benchmarking note: as for the skinny kernel, a looped weight stays Infinity-Cache resident.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_mxfp4.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_MXT_TM    4         // 16-row fragments per wave
#define R4D_MXT_WM    4         // waves along M
#define R4D_MXT_WN    2         // waves along N
#define R4D_MXT_PAD   8         // LDS row pad: 16 lanes reading rows 64 B apart collide 8 ways
#define R4D_MXT_NTHR  (R4D_MXT_WM * R4D_MXT_WN * 32)
#define R4D_MXT_BMF   (R4D_MXT_WM * R4D_MXT_TM * 16)
#define R4D_MXT_KUNIT 64        // the narrowest K slab, so K must be a multiple of it
#define R4D_MXT_TN4_MIN_M 1024  // tn=0's crossover, see above

typedef int      mxt_v2i  __attribute__((ext_vector_type(2)));
typedef unsigned mxt_u2   __attribute__((ext_vector_type(2)));
typedef unsigned mxt_u4   __attribute__((ext_vector_type(4)));

// The re-tile: one thread per 8-byte slot of the fragment-ordered plane; writes are contiguous.
__global__ __launch_bounds__(256) void r4d_mxfp4a8_tile_a_kernel(
    const unsigned char* __restrict__ A, unsigned char* __restrict__ AT, int M, int K,
    long long slots, int lda) {
  const long long o = (long long)blockIdx.x * 256 + threadIdx.x;
  if (o >= slots) return;
  const int ks = K / 16;
  const int lane = (int)(o & 31);
  const long long f = o >> 5;
  const int mt = (int)(f / ks), kst = (int)(f % ks);
  const int r = mt * 16 + (lane & 15), k = kst * 16 + (lane >> 4) * 8;
  mxt_u2 v = { 0u, 0u };
  if (r < M) v = *(const mxt_u2*)(A + (size_t)r * lda + k);
  *(mxt_u2*)(AT + o * 8) = v;
}

// The kernel's body. OWN: it declares its LDS arrays itself (the plain kernel); otherwise it takes
// the caller's (the two-body kernel, one set for both bodies). PART: a block the M edge cuts,
// which skips the WMMAs of its padding fragments past M.
template <int TN, int LBK, bool ABLK, bool PART, bool OWN>
__device__ __forceinline__ void r4d_gemm_mxfp4a8_tiled_body(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int as_rs, int ldws, int ldc, unsigned char* __restrict__ sW_in,
    unsigned int* __restrict__ sMag_in, float* __restrict__ sR_in) {
  constexpr int TM = R4D_MXT_TM, WN = R4D_MXT_WN, NTHR = R4D_MXT_NTHR, BMF = R4D_MXT_BMF;
  constexpr int NS = LBK / 16, STR = LBK + R4D_MXT_PAD, BNF = WN * TN * 16;
  constexpr int NTILES_T = BNF / 16, TOT_SLOTS = NTILES_T * NS * 32;
  // 4 weight slots (one 16 B read) wherever the tile has enough of them, else 2 (8 B), else 1.
  constexpr int WSLOTS = (TOT_SLOTS >= NTHR * 4) ? 4 : (TOT_SLOTS >= NTHR * 2 ? 2 : 1);
  constexpr int NW = TOT_SLOTS / (NTHR * WSLOTS);
  static_assert(NW * NTHR * WSLOTS == TOT_SLOTS, "the W tile must split evenly over the block");
  __shared__ unsigned char sW_own[OWN ? BNF * STR : 1];
  __shared__ unsigned int sMag_own[OWN ? 32 : 1];
  __shared__ float sR_own[OWN && ABLK ? BMF : 1];   // ABLK: a boundary's ratio a row, then each row's final scale
  unsigned char* __restrict__ sW = OWN ? sW_own : sW_in;
  unsigned int* __restrict__ sMag = OWN ? sMag_own : sMag_in;
  float* __restrict__ sR = OWN ? sR_own : sR_in;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF;
  const int ksteps = K / 16, Mt = (M + 15) >> 4;
  // PART: this wave's M fragments that hold a row below M, 0..TM and wave-uniform; the rest are
  // padding whose WMMAs are skipped (their rows are dropped at the store). Their A loads stay,
  // clamped, in the slab's one clause.
  const int live = PART ? __builtin_amdgcn_readfirstlane(min(TM, max(0, Mt - ((m0 >> 4) + wm * TM))))
                        : TM;

  if (tid < 32) sMag[tid] = (&kMxfp4Mag[0][0])[tid];

  // One A base per M fragment, wave-uniform: readfirstlane keeps it in an SGPR so loads take the
  // SADDR form with a 32-bit lane offset. A tile past M reads the last real one.
  const unsigned char* abase[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int mt = (m0 >> 4) + wm * TM + i;
    mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps * 256);
  }
  const unsigned int aoff = (unsigned int)lane * 8;

  // The W slots this thread stages, fixed across slabs: WSLOTS adjacent rows of one (n-tile,
  // k-step), so the wave reads contiguous bytes; the clamp keeps the vector read aligned.
  int wrow[NW], wkloc[NW], wrc[NW], wkst[NW], wlanec[NW];
  unsigned int wref[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    const int sl = it * NTHR * WSLOTS + tid * WSLOTS;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % NS, ntl = rest / NS;
    wrow[it] = ntl * 16 + (lane_ & 15);
    wkloc[it] = kst * 16 + (lane_ >> 4) * 8;
    const int gn = n0 + wrow[it];
    wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;
    wlanec[it] = (lane_ & 16) | (wrc[it] & 15);
    wkst[it] = kst;
    wref[it] = WSLOTS == 4 ? *(const unsigned int*)(Wref + wrc[it])
             : WSLOTS == 2 ? *(const unsigned short*)(Wref + wrc[it])
                           : Wref[wrc[it]];
  }

  v8f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

  // ABLK: thread t owns row m0 + t (BMF == NTHR) and loads the next 128 boundary's two scales (the
  // block starting there and the one before it) a boundary early.
  const int trow = m0 + tid < M ? m0 + tid : M - 1;
  const float* __restrict__ asrow = As + (size_t)trow * as_rs;
  float ns = 0.f, np = 0.f;
  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  lds_barrier();   // sMag

  // The slab's weight words and exponents, loaded a slab ahead (the first one here).
  unsigned int wqn[NW][WSLOTS], wswn[NW];
  auto wload = [&](int k0) {
    const int kstep0 = k0 / 16;
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      const unsigned int* src =
          &Wp[((size_t)(wrc[it] >> 4) * ksteps + kstep0 + wkst[it]) * 32 + wlanec[it]];
      const int blk = (k0 + wkst[it] * 16) / R4D_MXFP4_GROUP;
      if constexpr (WSLOTS == 4) {
        const mxt_u4 v = *(const mxt_u4*)src;
        wqn[it][0] = v[0]; wqn[it][1] = v[1]; wqn[it][2] = v[2]; wqn[it][3] = v[3];
        wswn[it] = *(const unsigned int*)(Ws + (size_t)blk * ldws + wrc[it]);
      } else if constexpr (WSLOTS == 2) {
        const mxt_u2 v = *(const mxt_u2*)src;
        wqn[it][0] = v[0]; wqn[it][1] = v[1];
        wswn[it] = *(const unsigned short*)(Ws + (size_t)blk * ldws + wrc[it]);
      } else {
        wqn[it][0] = *src;
        wswn[it] = Ws[(size_t)blk * ldws + wrc[it]];
      }
    }
  };
  wload(0);

  for (int k0 = 0; k0 < K; k0 += LBK) {
    // ABLK: at a 128 boundary each row's ratio goes to LDS, published by the staging barrier below
    // (sR's previous readers finished before the last slab's closing barrier). A zero block keeps
    // the carried scale; after a zero block the carried scale is looked up from the grid.
    const bool bnd = ABLK && k0 && (k0 & 127) == 0;
    if constexpr (ABLK) {
      if (bnd)
        sR[tid] = ns > 0.f ? (np > 0.f ? np : r4d_mxfp4_carried(asrow, (k0 >> 7) - 2)) / ns : 1.f;
    }
    // A fragments for the whole slab, one clause, consumed only at the WMMAs.
    mxt_v2i af[TM][NS];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const mxt_v2i*)(abase[i] +
                                      (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));
    // W slab -> LDS, folded to e4m3, from the words loaded a slab ago.
#pragma unroll
    for (int it = 0; it < NW; ++it) {
#pragma unroll
      for (int q = 0; q < WSLOTS; ++q) {
        const int d = r4d_mxfp4_fold_d((int)((wref[it] >> (8 * q)) & 0xFF),
                                       (int)((wswn[it] >> (8 * q)) & 0xFF));
        unsigned int lo, hi;
        r4d_mxfp4_unpack8_t(wqn[it][q], sMag[d * 2], sMag[d * 2 + 1], &lo, &hi);
        *(mxt_u2*)(&sW[(wrow[it] + q) * STR + wkloc[it]]) = mxt_u2{lo, hi};
      }
    }
    lds_barrier();
    if (k0 + LBK < K) wload(k0 + LBK);   // the next slab's weight flies across this slab's WMMAs

    if constexpr (ABLK) {
      // Re-express each row's partial sum in the new block's scale: one ratio a lane a row.
      if (bnd) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          const float r = sR[wm * TM * 16 + i * 16 + col];
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] *= r;
        }
      }
      // The next boundary's scales, issued behind this slab's loads and needed a boundary later.
      if ((k0 & 127) == 0 && k0 + 128 < K) {
        ns = asrow[(k0 >> 7) + 1];
        np = asrow[k0 >> 7];
      }
    }

#pragma unroll
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      mxt_v2i wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char* p = &sW[(wn * TN * 16 + j * 16 + col) * STR + kk];
        wf[j][0] = *(const int*)p; wf[j][1] = *(const int*)(p + 4);
      }
      // Keep one step's W fragments live at a time; otherwise the compiler hoists every step's
      // loads above the first WMMA.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        if (PART && i >= live) break;   // wave-uniform: padding fragments past M
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wf[j], af[i][step],
                                                                          acc[i][j]);
      }
    }
    lds_barrier();
  }

  // Epilogue. A lane holds row m0 + 64*wm + 16i + col of M fragment i and eight consecutive columns
  // of N fragment j, one 16-byte store per (i, j). 2^(Wref[n]-127) is the float whose exponent
  // field is Wref[n]; N % 16 == 0 keeps a lane's eight columns all inside or all outside N.
  float sc[TM];
  if constexpr (ABLK) {
    // the scale each row ends in, once a row; the K loop's closing barrier retired the ratio reads
    sR[tid] = r4d_mxfp4_carried(asrow, (K - 1) >> 7);
    lds_barrier();
#pragma unroll
    for (int i = 0; i < TM; ++i) sc[i] = sR[wm * TM * 16 + i * 16 + col];
  } else {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int m = m0 + wm * TM * 16 + i * 16 + col;
      sc[i] = As[(size_t)(m < M ? m : M - 1) * as_rs];
    }
  }
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int nb = n0 + wn * TN * 16 + j * 16 + kb8;
    if (nb >= N) continue;
    const mxt_u2 rb = *(const mxt_u2*)(Wref + nb);
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int m = m0 + wm * TM * 16 + i * 16 + col;
      if (m >= M) continue;
      // rounded as (__hip_bfloat16) rounds, packed in whole registers (see bf16x2_rne_nan)
      float v[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float rf = __int_as_float((int)((rb[e >> 2] >> (8 * (e & 3))) & 0xFF) << 23);
        v[e] = acc[i][j][e] * rf * sc[i];
      }
      mxt_u4 o;
      o.x = bf16x2_rne_nan(v[0], v[1]);
      o.y = bf16x2_rne_nan(v[2], v[3]);
      o.z = bf16x2_rne_nan(v[4], v[5]);
      o.w = bf16x2_rne_nan(v[6], v[7]);
      *(mxt_u4*)(C + (size_t)m * ldc + nb) = o;
    }
  }
}

// The plain kernel: every block the unguarded body, over the body's own LDS.
template <int TN, int LBK, bool ABLK>
__global__ __launch_bounds__(R4D_MXT_NTHR) void r4d_gemm_mxfp4a8_tiled_kernel(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int as_rs, int ldws, int ldc) {
  r4d_gemm_mxfp4a8_tiled_body<TN, LBK, ABLK, false, true>(AT, As, W, Ws, Wref, C, M, K, N, as_rs, ldws,
                                                          ldc, nullptr, nullptr, nullptr);
}

// The two-body kernel, for a thin last row of blocks: a block of that row takes the guarded body,
// every other block the unguarded one, uniform over the block, over one set of LDS arrays.
template <int TN, int LBK, bool ABLK>
__global__ __launch_bounds__(R4D_MXT_NTHR) void r4d_gemm_mxfp4a8_tiled_part_kernel(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int as_rs, int ldws, int ldc) {
  constexpr int BNF = R4D_MXT_WN * TN * 16, STR = LBK + R4D_MXT_PAD;
  __shared__ unsigned char sW[BNF * STR];
  __shared__ unsigned int sMag[32];
  __shared__ float sR[ABLK ? R4D_MXT_BMF : 1];
  if ((int)(blockIdx.y + 1) * R4D_MXT_BMF > M)
    r4d_gemm_mxfp4a8_tiled_body<TN, LBK, ABLK, true, false>(AT, As, W, Ws, Wref, C, M, K, N, as_rs,
                                                            ldws, ldc, sW, sMag, sR);
  else
    r4d_gemm_mxfp4a8_tiled_body<TN, LBK, ABLK, false, false>(AT, As, W, Ws, Wref, C, M, K, N, as_rs,
                                                             ldws, ldc, sW, sMag, sR);
}

extern "C" int r4d_gemm_mxfp4a8_tiled_group(void) { return R4D_MXFP4_GROUP; }

extern "C" long r4d_mxfp4a8_tile_a_bytes(int M, int K) {
  return M < 1 || K < 16 ? 0 : (long)((M + 15) / 16) * 16 * K;
}

extern "C" void r4d_mxfp4a8_tile_a(long a, long at, int M, int K, long stream, int lda) {
  if (!lda) lda = K;
  if (M < 1) throw std::runtime_error("r4d mxfp4a8_tile_a: M must be >= 1");
  if (K < 16 || K % 16) throw std::runtime_error("r4d mxfp4a8_tile_a: K must be a multiple of 16");
  const long long slots = (long long)((M + 15) / 16) * (K / 16) * 32;
  hipLaunchKernelGGL(r4d_mxfp4a8_tile_a_kernel, dim3((unsigned)((slots + 255) / 256)), dim3(256), 0,
                     (hipStream_t)stream, (const unsigned char*)a, (unsigned char*)at, M, K, slots, lda);
}

extern "C" void r4d_gemm_mxfp4a8_tiled(long at, long ascale, long wq, long ws, long wref, long c,
                                       int M, int K, int N, int TN, int as_group, int as_rs,
                                       long stream, int ldws, int ldc) {
  if (!ldws) ldws = N;
  if (!ldc) ldc = N;
  if (M < 1) throw std::runtime_error("r4d gemm_mxfp4a8_tiled: M must be >= 1");
  if (N < 16 || N % 16) throw std::runtime_error("r4d gemm_mxfp4a8_tiled: N must be a multiple of 16");
  if (K < R4D_MXT_KUNIT || K % R4D_MXT_KUNIT)
    throw std::runtime_error("r4d gemm_mxfp4a8_tiled: K must be a multiple of 64");
  // tn=0: TN=4 from the M crossover on, unless its 128-wide tile leaves fewer than 128 blocks
  // (narrow N), where TN=2's extra blocks win.
  if (TN == 0)
    TN = M >= R4D_MXT_TN4_MIN_M && (long)((N + 127) / 128) * ((M + R4D_MXT_BMF - 1) / R4D_MXT_BMF) >= 128
             ? 4 : 2;
  if (TN != 2 && TN != 4) throw std::runtime_error("r4d gemm_mxfp4a8_tiled: TN must be 2 or 4");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_tiled: the activation scale is per row or per 128 of K");
  const bool ablk = as_group == 128;
  if (ablk && as_rs < (K + 127) / 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_tiled: activation scale rows shorter than ceil(K/128)");
  const bool wide_slab = TN == 2 && K % 128 == 0;
  const int bnf = R4D_MXT_WN * TN * 16;
  dim3 grid((N + bnf - 1) / bnf, (M + R4D_MXT_BMF - 1) / R4D_MXT_BMF), block(R4D_MXT_NTHR);
  auto Ap = (const unsigned char*)at;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
  // The two-body kernel only for a thin last row of blocks (1..160 rows) under 1024 rows; see the
  // header.
  const int rem = M % R4D_MXT_BMF;
  const bool part = rem > 0 && rem <= 160 && M < 1024;
#define MXT_LAUNCH(TNv, LBKv, ABv) do {                                                          \
    if (part)                                                                                     \
      hipLaunchKernelGGL((r4d_gemm_mxfp4a8_tiled_part_kernel<TNv, LBKv, ABv>), grid, block, 0, st, \
                         Ap, Sp, Wp, Zp, Rp, Cp, M, K, N, as_rs, ldws, ldc);                      \
    else                                                                                          \
      hipLaunchKernelGGL((r4d_gemm_mxfp4a8_tiled_kernel<TNv, LBKv, ABv>), grid, block, 0, st, Ap,  \
                         Sp, Wp, Zp, Rp, Cp, M, K, N, as_rs, ldws, ldc);                          \
  } while (0)
  if (TN == 4)        { if (ablk) MXT_LAUNCH(4, 64, true);  else MXT_LAUNCH(4, 64, false); }
  else if (wide_slab) { if (ablk) MXT_LAUNCH(2, 128, true); else MXT_LAUNCH(2, 128, false); }
  else                { if (ablk) MXT_LAUNCH(2, 64, true);  else MXT_LAUNCH(2, 64, false); }
#undef MXT_LAUNCH
}

// ============================================================== the plugin ABI entry point
// The row-major activation is re-tiled into the op's scratch and the GEMM reads it there. For a
// partitioned weight (mx_parts), each part's activation copy is re-tiled and multiplied in turn.
extern "C" int r4d_rad_gemm_mxfp4a8_tiled(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (g.N < 16 || g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    /* The per-row reference exponent is required, as in r4d_rad_gemm_mxfp4a8_m64. */
    if (!g.bref) return R4D_NO(RAD_E_INVAL);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    if (g.K % R4D_MXT_KUNIT) return R4D_NO(RAD_E_SHAPE);
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs, mp.n);
    if (rc != RAD_OK) return rc;
    const long long slab = r4d_mxfp4a8_tile_a_bytes((int)g.M, (int)g.K);
    if (!a->scratch || a->scratch_bytes < slab) return R4D_NO(RAD_E_INVAL);
    R4dMxTileCfg t;
    r4d_cfg_mxfp4_tiled(a, &t);
    const long kb = (long)((g.K + 127) / 128);
    const long at = (long)(uintptr_t)a->scratch;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        R4D_TRY("gemm_mxfp4a8_tiled",
            r4d_mxfp4a8_tile_a(r4d_addr(g.a) + p * g.K, at, (int)g.M, (int)g.K,
                               (long)(uintptr_t)s, (int)(mp.lda ? mp.lda : g.K));
            r4d_gemm_mxfp4a8_tiled(at, r4d_addr(g.ascale) + p * kb * 4, r4d_addr(g.b) + n0 * g.K / 2,
                                   r4d_addr(g.bscale) + n0, r4d_addr(g.bref) + n0,
                                   r4d_addr(g.y) + n0 * 2, (int)g.M, (int)g.K, (int)np, t.tn,
                                   as_group, as_rs, (long)(uintptr_t)s, (int)g.N, (int)g.N));
    }
    return RAD_OK;
}

/* The op's scratch, sized at the band's widest M: the fragment-tiled activation in whole 16-row
 * fragments. One copy serves every part: parts run in order on the op's stream, so a re-tile
 * cannot overtake the previous part's GEMM. */
extern "C" int64_t r4d_rad_gemm_mxfp4a8_tiled_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), K = r4d_geti_or(a, "K", 0);
    return (int64_t)r4d_mxfp4a8_tile_a_bytes((int)M, (int)K);
}
