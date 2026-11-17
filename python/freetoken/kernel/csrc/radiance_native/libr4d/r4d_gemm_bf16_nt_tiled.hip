// r4d_gemm_bf16_nt_tiled.hip - r4d_gemm_bf16_nt_tiled: the bf16 GEMM at prefill row counts, gfx1201.
//
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 in and out, fp32 accumulate: gemm_bf16_nt_m64's arithmetic in a
// block shape that reuses what it loads.
//
// The skinny kernel reads every fragment from global memory and re-reads the whole activation per
// N tile: right at decode, where the weight is the cost, but at a 2048-row prefill chunk the
// activation is read N/16 times and no configuration reaches 50 TF/s.
//
// Here a block of eight waves stages a BM x 64 activation tile and a BN x 64 weight tile through
// LDS, and each wave computes a (BM/2) x (BN/4) patch, so each staged fragment feeds TM or TN WMMA.
// The next K tile is loaded into registers while the current one is computed.
//
// Contract: byte-identical to gemm_bf16_nt_m64. Both accumulate each of SK slices from zero with
// one WMMA per 16 K in ascending order, using the same fragment layout, and sum the slices in
// ascending order from 0.f. SK comes from the same rule (cfg_for), so an op moving between the two
// rows changes no output bit.
//
// Fragment rows are clamped as in the skinny kernel: rows past M or N re-read the last valid row
// and their results are never stored.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_bf16_gemm_rows.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <cstdint>

#define BT_BK 64                // K per staged tile: four WMMA k steps
// LDS row pitch in elements (144 bytes). A fragment read is sixteen lanes on sixteen rows, 8 bytes
// each; at 36 dwords per row they land on distinct bank pairs, and the upper half-wave's +2 dwords
// fall between them. 144 is a multiple of 16, so a staged chunk is one aligned ds_store_b128.
#define BT_PITCH (BT_BK + 8)
#define BT_WM 2                 // waves over M
#define BT_WN 4                 // waves over N

// TM x TN WMMA tiles a wave; BM = 2*TM*16, BN = 4*TN*16. SPLIT carries the slice sum.
template <int TM, int TN, bool SPLIT>
__global__ __launch_bounds__(256) void r4d_gemm_bf16_nt_tiled_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int spk) {
  constexpr int BM = BT_WM * TM * 16, BN = BT_WN * TN * 16;
  constexpr int CPR = BT_BK / 8;                  // 16-byte chunks a staged row
  constexpr int AL = BM * CPR / 256, WL = BN * CPR / 256;
  static_assert(AL >= 1 && WL >= 1 && BM * CPR % 256 == 0 && BN * CPR % 256 == 0,
                "a tile is whole 16-byte chunks for every thread");
  __shared__ __attribute__((aligned(16))) unsigned short sA[BM * BT_PITCH];
  __shared__ __attribute__((aligned(16))) unsigned short sW[BN * BT_PITCH];

  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int wm = wave / BT_WN, wn = wave % BT_WN;
  // Blocks are remapped into groups of 8 activation bands, M fastest within a group. Blocks
  // dispatch in flattened-id order; x-fastest order would read the whole weight once per 128 rows,
  // which halves the rate once the weight exceeds the 64 MB MALL. A group shares weight tiles among
  // blocks in flight and reads the weight once per group; with fewer bands than a group the order
  // is plain M-fastest.
  constexpr unsigned kGroupM = 8;
  const unsigned per = kGroupM * gridDim.x, lin = blockIdx.y * gridDim.x + blockIdx.x;
  const unsigned first = lin / per * kGroupM, r = lin % per;
  const unsigned gm = min(gridDim.y - first, kGroupM);
  const int m0 = (int)(first + r % gm) * BM, n0 = (int)(r / gm) * BN;

  // Staging: chunk c = tid + 256 i is row c / CPR, elements (c % CPR) * 8. Eight threads cover a
  // row's 128 bytes, so a wave's loads are four whole lines.
  unsigned aoff[AL], woff[WL];
  unsigned short* adst[AL];
  unsigned short* wdst[WL];
#pragma unroll
  for (int i = 0; i < AL; ++i) {
    const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
    const int gr = m0 + r < M ? m0 + r : M - 1;
    aoff[i] = (unsigned)gr * (unsigned)K + (unsigned)e;
    adst[i] = sA + r * BT_PITCH + e;
  }
#pragma unroll
  for (int i = 0; i < WL; ++i) {
    const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
    const int gr = n0 + r < N ? n0 + r : N - 1;
    woff[i] = (unsigned)gr * (unsigned)K + (unsigned)e;
    wdst[i] = sW + r * BT_PITCH + e;
  }

  // Fragment sources: this lane's row of each tile at its half-K offset, as in rowP.
  const unsigned short* fa = sA + (wm * TM * 16 + (lane & 15)) * BT_PITCH + 4 * (lane >> 4);
  const unsigned short* fw = sW + (wn * TN * 16 + (lane & 15)) * BT_PITCH + 4 * (lane >> 4);

  v8f acc[TM][TN], tot[SPLIT ? TM : 1][SPLIT ? TN : 1];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);
  if constexpr (SPLIT) {
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) tot[i][j] = (v8f)(0.f);
  }

  r4d_u32x4 ra[AL], rw[WL];                       // registers; uint4 would sit in scratch
#pragma unroll
  for (int i = 0; i < AL; ++i) ra[i] = *(const r4d_u32x4*)(A + aoff[i]);
#pragma unroll
  for (int i = 0; i < WL; ++i) rw[i] = *(const r4d_u32x4*)(W + woff[i]);

  const int nk = K / BT_BK;
  int left = spk;                                 // k steps to the end of this split slice
  for (int kt = 0; kt < nk; ++kt) {
    __syncthreads();                              // the previous tile's fragments are all read
#pragma unroll
    for (int i = 0; i < AL; ++i) *(r4d_u32x4*)adst[i] = ra[i];
#pragma unroll
    for (int i = 0; i < WL; ++i) *(r4d_u32x4*)wdst[i] = rw[i];
    __syncthreads();
    if (kt + 1 < nk) {
      const unsigned k1 = (unsigned)(kt + 1) * BT_BK;
#pragma unroll
      for (int i = 0; i < AL; ++i) ra[i] = *(const r4d_u32x4*)(A + aoff[i] + k1);
#pragma unroll
      for (int i = 0; i < WL; ++i) rw[i] = *(const r4d_u32x4*)(W + woff[i] + k1);
    }
#pragma unroll
    for (int s = 0; s < BT_BK / 16; ++s) {
      v8bf af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) af[i] = fragP(fa + i * 16 * BT_PITCH + s * 16);
#pragma unroll
      for (int j = 0; j < TN; ++j) bf[j] = fragP(fw + j * 16 * BT_PITCH + s * 16);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = mma(af[i], bf[j], acc[i][j]);
      /* Schedule each k step's fragment reads, then its products. Otherwise the compiler spreads
       * LDS reads among the previous step's WMMAs and waits on them early; grouping is 6-9% faster
       * on the 128 x 128 tile at M 384..2048, K 2560. */
      __builtin_amdgcn_sched_group_barrier(0x100, TM + TN, 0);    /* DS reads */
      __builtin_amdgcn_sched_group_barrier(0x008, TM * TN, 0);    /* WMMA */
      if constexpr (SPLIT) {
        if (--left == 0) {
          left = spk;
#pragma unroll
          for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) {
              tot[i][j] += acc[i][j];
              acc[i][j] = (v8f)(0.f);
            }
        }
      }
    }
  }

  // The skinny kernel's store: lane l holds column (l & 15) of a tile and rows 8 (l >> 4) + e.
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
      const int mb = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float v = 0.f;
        if constexpr (SPLIT) v = tot[i][j][e];
        else v += acc[i][j][e];
        const int m = mb + e;
        if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)v;
      }
    }
}

// Tile choice. A block walks all of K alone, so its time is a floor (~40 us at K = 2560), and a
// grid too small to fill the part leaves CUs idle (M = 200, N = 512 is eight 128 x 128 blocks).
// At K = 2560 the 128 x 128 tile wins once its grid reaches 32 blocks (half the CUs) and loses by
// less than the 64 x 64 tile does below that. 64 x 128 and 128 x 64 gain only a few percent.
enum { kBtLargeGrid = 32 };

int r4d_gemm_bf16_nt_tiled_arm(int M, int N) {
  const long long g = ((N + 127ll) / 128) * ((M + 127ll) / 128);
  return g >= kBtLargeGrid ? 1 : 2;
}

// `tile`: 1 = 128 x 128, 2 = 64 x 64 (M x N).
int r4d_gemm_bf16_nt_tiled(long a, long w, long c, int M, int K, int N, int SK, int tile,
                           long stream) {
  if (M < 1 || N < 1 || K < BT_BK || K % BT_BK) return -1;
  if (SK < 1 || K % ((int64_t)SK * 16)) return -1;
  if (tile != 1 && tile != 2) return -1;
  const int spk = K / SK / 16;
  const int bt = tile == 1 ? 128 : 64;
  const int chunk = r4d_bf16_gemm_chunk_rows(M, N, K, bt, bt);
  if (!chunk) return -1;
  dim3 block(256);
  auto Wp = (const unsigned short*)w;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(TM, TN, S)                                                                          \
  hipLaunchKernelGGL((r4d_gemm_bf16_nt_tiled_kernel<TM, TN, S>), grid, block, 0, st, Ap, Wp, Cp,   \
                     mm, K, N, spk)
  const bool split = SK > 1;
  for (int64_t m0 = 0; m0 < M; m0 += chunk) {
    const int mm = (int)std::min<int64_t>(chunk, M - m0);
    dim3 grid((N + bt - 1) / bt, (mm + bt - 1) / bt);
    auto Ap = (const unsigned short*)a + m0 * K;
    auto Cp = (__hip_bfloat16*)c + m0 * N;
    if (tile == 1) {
      if (split)
        LAUNCH(4, 2, true);
      else
        LAUNCH(4, 2, false);
    } else {
      if (split)
        LAUNCH(2, 1, true);
      else
        LAUNCH(2, 1, false);
    }
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
  }
#undef LAUNCH
  return 0;
}

// ============================================================== the plugin ABI entry point
extern "C" int r4d_rad_gemm_bf16_tiled(const RadArgs* a, RadStream s) {
  GemmOps g;
  int rc = gemm_ops(a, &g, 0);
  if (rc != RAD_OK) return rc;
  if (!r4d_bf16_gemm_chunk_rows(g.M, g.N, g.K, 64, 64)) return R4D_NO(RAD_E_SHAPE);
  // The skinny row's split rule with the same arguments keeps the two rows' outputs identical.
  Cfg c;
  rc = cfg_for(a, g.K, g.M, 16, BR_FLAT, &c, (g.N + 15) / 16);
  if (rc != RAD_OK) return rc;
  int tile = (int)r4d_geti_or(a, "tile", 0);
  if (tile == 0) tile = r4d_gemm_bf16_nt_tiled_arm((int)g.M, (int)g.N);
  const int e = r4d_gemm_bf16_nt_tiled(r4d_addr(g.a), r4d_addr(g.b), r4d_addr(g.y), (int)g.M,
                                       (int)g.K, (int)g.N, c.sk, tile, (long)(uintptr_t)s);
  return e == 0 ? RAD_OK : R4D_NO(e < 0 ? RAD_E_SHAPE : RAD_E_DEVICE);
}
