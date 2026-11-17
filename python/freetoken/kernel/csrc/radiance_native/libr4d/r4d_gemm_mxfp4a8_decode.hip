// r4d_gemm_mxfp4a8_decode.hip - r4d_gemm_mxfp4a8_decode: the OCP-MXFP4 x fp8 GEMM for the decode
// band, M <= 128, gfx1201. C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with the operands, weight layout and
// fold of r4d_gemm_mxfp4a8_nt_m64 and r4d_gemm_mxfp4a8_tiled.
//
// At decode the GEMM is a weight stream and the goal is DRAM bandwidth. The skinny kernel (split-K
// within a workgroup, weight straight to registers) falls short on narrow projections. This one
// splits K across workgroups with a fused last-arriver reduction and streams the weight with
// non-temporal loads.
//
// Tile: DWN = 8 waves over BND = 128 output columns (one 16-column fragment per wave) and DTM
// 16-row M fragments, the fewest that cover M. Each K slab of DBK bytes stages A (16*DTM rows) in
// LDS, shared by the eight waves; each wave loads its own fragment's weight words for the slab
// straight to registers and unpacks them to e4m3 there (r4d_mxfp4.h). The one-fragment form
// (M <= 16) stages the weight through LDS instead.
//
// Split-K is required: N/16 fragments alone (320 waves at N=5120) do not fill the part. DKS blocks
// split the K slabs; each writes an fp32 partial [DKS][M][N], and the last to arrive (an
// agent-scope counter per column block) sums them, with no second launch.
//
// Split, slab and load hint follow a rule tuned on the 27B's and TranslateGemma's per-rank shapes
// (tunables `ks`, `bk`, `nt`; 0 selects the rule):
//   - split up to 64 rows: the smallest power of two whose grid reaches 48 blocks at M <= 16 and
//     56 above; for K >= 8192 keep doubling while a split has >= 24 128-wide slabs; never past
//     ks*M = 512, where partial traffic outweighs the gain.
//   - split past 64 rows: double only while every split keeps 16 128-wide slabs, towards a
//     48-block grid or, at K >= 8192, regardless; never past ks*M = 1024.
//   - slab: 128, or 64 where LDS per block would cost a wide grid its resident blocks (see
//     mxd_plan); past 64 rows always 128. A K not a multiple of 128 takes 64 (and a 2-way split
//     widens to 4).
//   - non-temporal weight loads for weights of 4 MiB or more.
// The split count sets a row's summation order, so a change to the rule moves the bits of the rows
// it splits differently.
//
// No state is keyed on the stream, since a recorded pass may replay on another stream. The split
// partial is the op's scratch (sized by the scratch hook, read+write, so overlapping ops get
// disjoint slices); the arrival counters are keyed on the output buffer (mxd_out_ctr), zero at
// entry and reset by the reducer. Launches writing the same output are ordered; overlapping ones
// write different outputs. The row is therefore concurrent (rad_kernel_concurrent), like
// gemm_fp8a8_nt_m16. A scratch shorter than the split needs is refused.
//
// Every thread fences its partial stores before the counter; a fence on thread 0 alone would order
// only its own stores.
//
// Two activation scale grids as in the tiled kernel: one f32 per row, or one per (row, 128 of K).
// The block form carries each row's accumulator in the current block's scale, rescaling by
// s_prev / s_next at each boundary (ratio computed once per row into LDS from scales loaded a
// boundary early). Each split's partial is converted to absolute units before it is written,
// since splits end in different blocks.
#include "r4d.h"
#include "r4d_keyed.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_mxfp4.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>

#define R4D_MXD_DWN     8       // waves per block; the N tile is DWN*16 = 128 wide
#define R4D_MXD_MTILE   16
#define R4D_MXD_MAX_TM  8       // M <= 128
#define R4D_MXD_PAD     8
#define R4D_MXD_MAX_KS  16
#define R4D_MXD_MAX_N   36864   // the counter block's width; a 27B gate_up at TP=1 is 34816

typedef int      mxd_v2i __attribute__((ext_vector_type(2)));
typedef unsigned mxd_u2  __attribute__((ext_vector_type(2)));
typedef unsigned mxd_u4  __attribute__((ext_vector_type(4)));
typedef float    mxd_f4  __attribute__((ext_vector_type(4)));

template <bool NT, typename T>
static __device__ __forceinline__ T mxd_ld(const T* p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

// ---- shared by both kernels: the scale grid at a 128 boundary, and the epilogue.
//
// The accumulator is C transposed: the WMMA takes the weight fragment as A and the activation as B
// (same register layout), so a lane's accumulator for M fragment i holds one row, m = 16i + col,
// and eight consecutive columns n0 + 16*wave + kb8 + e. A lane needs one activation scale per row,
// and writes its eight outputs as one 16-byte store (one 32-byte store of partials).

// The (row, 128) scale grid: thread t < ROWS owns row t, loads the next boundary's two scales a
// boundary early and writes s_prev / s_next to LDS for the boundary's slab (the one-fragment kernel
// at that slab, published by its staging barrier; the multi-fragment kernel a slab early into the
// other half of its double buffer, published by the slab's closing barrier). A zero next block
// keeps the carried scale; after a zero block the carried scale is looked up from the grid.
static __device__ __forceinline__ float mxd_ratio(float ns, float np, const float* __restrict__ row,
                                                  int kb) {
  return ns > 0.f ? (np > 0.f ? np : r4d_mxfp4_carried(row, kb - 2)) / ns : 1.f;
}

template <int DTM>
static __device__ __forceinline__ void mxd_rescale(v8f (&acc)[DTM], const float* sR, int col) {
#pragma unroll
  for (int i = 0; i < DTM; ++i) {
    const float r = sR[i * 16 + col];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] *= r;
  }
}

// The split's result: written to C when the accumulator covers all of K, else as an fp32 partial
// that the last block to arrive on this column block sums. `kb_end` is the block the split's last
// slab ends in (ABLK: the scale the accumulators are carried in).
template <int DTM, int DKS, bool ABLK>
static __device__ __forceinline__ void mxd_epilogue(
    const v8f (&acc)[DTM], const float* __restrict__ As, int as_rs,
    const unsigned char* __restrict__ Wref, float* __restrict__ P, unsigned* __restrict__ cnt,
    __hip_bfloat16* __restrict__ C, int M, int N, int ldc, int n0, int wave, int col, int kb8,
    int ks, int kb_end, unsigned* s_last, float* sR) {
  constexpr int NTHR = R4D_MXD_DWN * 32, BND = R4D_MXD_DWN * 16;
  constexpr int ROWS = R4D_MXD_MTILE * DTM;
  const int tid = threadIdx.x;
  const int nb = n0 + wave * 16 + kb8;   // this lane's eight columns; N % 16 == 0 keeps all or none
  // Each row's final scale: the one the split's accumulators end in (ABLK), or the row's own. The
  // K loop's closing barrier retired the last ratio reads.
  if (tid < ROWS) {
    const int m = tid < M ? tid : M - 1;
    sR[tid] = ABLK ? r4d_mxfp4_carried(As + (size_t)m * as_rs, kb_end) : As[(size_t)m * as_rs];
  }
  __syncthreads();
  float sc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i) sc[i] = sR[i * 16 + col];
  if constexpr (DKS == 1) {
    if (nb < N) {
      const mxd_u2 rb = *(const mxd_u2*)(Wref + nb);
      float rf[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) rf[e] = __int_as_float((int)((rb[e >> 2] >> (8 * (e & 3))) & 0xFF) << 23);
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const int m = i * 16 + col;
        if (m < M) {
          // rounded as (__hip_bfloat16) rounds, packed in whole registers (see bf16x2_rne_nan)
          float v[8];
#pragma unroll
          for (int e = 0; e < 8; ++e)
            v[e] = ABLK ? acc[i][e] * sc[i] * rf[e] : acc[i][e] * rf[e] * sc[i];
          mxd_u4 o;
          o.x = bf16x2_rne_nan(v[0], v[1]);
          o.y = bf16x2_rne_nan(v[2], v[3]);
          o.z = bf16x2_rne_nan(v[4], v[5]);
          o.w = bf16x2_rne_nan(v[6], v[7]);
          *(mxd_u4*)(C + (size_t)m * ldc + nb) = o;
        }
      }
    }
    return;
  }
  if (nb < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i) {
      const int m = i * 16 + col;
      if (m < M) {
        mxd_f4 lo, hi;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          lo[e] = ABLK ? acc[i][e] * sc[i] : acc[i][e];
          hi[e] = ABLK ? acc[i][e + 4] * sc[i] : acc[i][e + 4];
        }
        mxd_f4* dst = (mxd_f4*)(P + ((size_t)ks * M + m) * N + nb);
        dst[0] = lo;
        dst[1] = hi;
      }
    }
  }
  // Every thread's partial stores are ordered before the counter; then the last arriver reduces.
  __threadfence();
  __syncthreads();
  if (tid == 0)
    *s_last = __hip_atomic_fetch_add(&cnt[blockIdx.x], 1u, __ATOMIC_ACQ_REL,
                                     __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  if (*s_last != (unsigned)(DKS - 1)) return;
  if (tid == 0) cnt[blockIdx.x] = 0u;          // reset for the next launch
  __threadfence();                             // acquire, before reading the other partials
  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += NTHR) {
    const float rf = __int_as_float((int)Wref[nn] << 23);
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * ldc + nn] = (__hip_bfloat16)(sum * rf * (ABLK ? 1.f : As[(size_t)m * as_rs]));
    }
  }
}

/* Waves per SIMD for each tile shape (R4D_WAVES), by K block, M fragments and scale form: pins
 * the register budget the tile was tuned at so a compiler release cannot change occupancy (LLVM 24
 * would drop the DBK 64, three-fragment form from 16 waves to 12). Where a shape spilled at 256
 * VGPRs, the budget is the one it reaches without spilling. */
constexpr int mxd_waves(int dbk, int dtm, bool ablk) {
  if (dbk == 64)
    return dtm <= 3 ? 16 : dtm == 4 ? 12 : dtm == 5 ? 10 : dtm == 6 ? (ablk ? 9 : 10)
         : dtm == 7 ? 9 : (ablk ? 8 : 9);
  return dtm <= 3 ? 10 : dtm == 4 ? 8 : dtm == 5 ? (ablk ? 8 : 7) : dtm == 6 ? (ablk ? 8 : 6)
       : 6;
}

// The multi-fragment form (M > 16). A fragment-order weight word is the WMMA fragment's own
// register (one dword a lane a k-step), so a wave loads its tile's words for a slab straight to
// registers and folds them there, with no barrier between a wave and its weight. Only the
// activation, shared by the eight tiles, goes through LDS, double-buffered so a slab takes one
// barrier. The next slab's activation and weight are loaded while this slab computes. The fold,
// k-step and slab order, scale carry and epilogue are the one-fragment form's (below).
template <int DBK, int DKS, int DTM, bool NT, bool ABLK>
__global__ __launch_bounds__(R4D_MXD_DWN * 32) R4D_WAVES(mxd_waves(DBK, DTM, ABLK))
void r4d_gemm_mxfp4a8_decode_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, float* __restrict__ P, unsigned* __restrict__ cnt,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int as_rs, int lda, int ldws,
    int ldc) {
  constexpr int DWN = R4D_MXD_DWN, BND = DWN * 16, NTHR = DWN * 32;
  constexpr int ROWS = R4D_MXD_MTILE * DTM;
  constexpr int STR = DBK + R4D_MXD_PAD;
  constexpr int KST = DBK / 16;                       // k-steps a slab
  constexpr int A_IT = (ROWS * DBK + NTHR * 16 - 1) / (NTHR * 16);
  __shared__ unsigned char sA[2][ROWS * STR];
  __shared__ float sR[2][ROWS];  // ABLK: a boundary's ratio a row; then every row's final scale
  __shared__ unsigned s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;
  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);
  const int k_hi = (s_hi * DBK < K ? s_hi * DBK : K);

  v8f acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i) acc[i] = (v8f)(0.f);
  const float* __restrict__ asrow = As + (size_t)(tid < M ? tid : M - 1) * as_rs;
  float ns = 0.f, np = 0.f;
  if constexpr (ABLK) {
    // a split that starts mid-block meets its first boundary one slab in
    const int kb = ((s_lo * DBK) >> 7) + 1;
    if (tid < ROWS && ((s_lo * DBK) & 127) && kb * 128 < k_hi) { ns = asrow[kb]; np = asrow[kb - 1]; }
  }
  // This wave's tile and this lane's column. A tile past N (N % 128 != 0) loads the last one, and
  // the epilogue drops its columns.
  const int nt = min(n0 / 16 + wave, N / 16 - 1);
  const int ncol = nt * 16 + col;
  const int ref = (int)Wref[ncol];
  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  const int ksteps_g = K / 16;

  unsigned rw[KST];
  int re[KST / 2];
  mxd_u4 ra[A_IT];
  auto load = [&](int s) {
    const int k0 = s * DBK;
#pragma unroll
    for (int j = 0; j < A_IT; ++j) {
      const int idx = j * NTHR * 16 + tid * 16;
      if (idx < ROWS * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;              // clamp, never predicate
        ra[j] = *(const mxd_u4*)(A + (size_t)rc * lda + k0 + c);
      }
    }
#pragma unroll
    for (int t = 0; t < KST; ++t)
      rw[t] = mxd_ld<NT>(&Wp[((size_t)nt * ksteps_g + k0 / 16 + t) * 32 + lane]);
#pragma unroll
    for (int q = 0; q < KST / 2; ++q)
      re[q] = (int)Ws[(size_t)(k0 / R4D_MXFP4_GROUP + q) * ldws + ncol];
  };
  auto store_a = [&](int buf) {
#pragma unroll
    for (int j = 0; j < A_IT; ++j) {
      const int idx = j * NTHR * 16 + tid * 16;
      if (idx < ROWS * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        *(mxd_u4*)(&sA[buf][r * STR + c]) = ra[j];
      }
    }
  };

  if (s_lo < s_hi) {
    load(s_lo);
    store_a(0);                  // the first slab never rescales (s > s_lo below)
  }
  __syncthreads();
  for (int s = s_lo; s < s_hi; ++s) {
    const int buf = (s - s_lo) & 1;
    const int k0 = s * DBK;
    const bool bnd = ABLK && s > s_lo && (k0 & 127) == 0;
    mxd_v2i wf[KST];
#pragma unroll
    for (int t = 0; t < KST; ++t) {
      unsigned lo, hi;
      r4d_mxfp4_unpack8(rw[t], r4d_mxfp4_fold_d(ref, re[t >> 1]), &lo, &hi);
      wf[t][0] = (int)lo; wf[t][1] = (int)hi;
    }
    if (s + 1 < s_hi) load(s + 1);           // the next slab's loads fly across this slab's WMMAs
    if constexpr (ABLK) {
      if (bnd) mxd_rescale(acc, sR[buf], col);
      if (tid < ROWS && (k0 & 127) == 0 && k0 + 128 < k_hi) {
        ns = asrow[(k0 >> 7) + 1];
        np = asrow[k0 >> 7];
      }
    }
#pragma unroll
    for (int step = 0; step < KST; ++step) {
      const int kk = step * 16 + kb8;
      mxd_v2i af[DTM];
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char* pa = &sA[buf][(i * 16 + col) * STR + kk];
        af[i][0] = *(const int*)pa; af[i][1] = *(const int*)(pa + 4);
      }
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wf[step], af[i], acc[i]);
    }
    // The next slab's activation into the other buffer, and its boundary ratio (published by the
    // barrier below; a wave still rescaling this slab reads the other half).
    if (s + 1 < s_hi) {
      store_a(buf ^ 1);
      if constexpr (ABLK) {
        const int k1 = (s + 1) * DBK;
        if ((k1 & 127) == 0 && tid < ROWS) sR[buf ^ 1][tid] = mxd_ratio(ns, np, asrow, k1 >> 7);
      }
    }
    __syncthreads();
  }
  const int kb_end = s_hi > s_lo ? (k_hi - 1) >> 7 : 0;
  mxd_epilogue<DTM, DKS, ABLK>(acc, As, as_rs, Wref, P, cnt, C, M, N, ldc, n0, wave, col, kb8, ks,
                               kb_end, &s_last, sR[0]);
}

// The one-fragment form (M <= 16), every single-sequence verify step. Same tile, fold and split as
// above, with the weight staged through LDS, plus: the next slab's A, codes and exponents are
// loaded into registers while this slab computes, keeping loads in flight across barriers; a thread
// takes four adjacent slots (one 16 B load); and each row's reference exponent is read once.
// Bit-identical to the kernel above.
template <int DBK, int DKS, bool NT, bool ABLK>
__global__ __launch_bounds__(R4D_MXD_DWN * 32) R4D_WAVES(DBK == 64 ? 16 : 12)
void r4d_gemm_mxfp4a8_decode1_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, float* __restrict__ P, unsigned* __restrict__ cnt,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int as_rs, int lda, int ldws,
    int ldc) {
  constexpr int DTM = 1, VEC = 4;
  constexpr int DWN = R4D_MXD_DWN, BND = DWN * 16, NTHR = DWN * 32;
  constexpr int ROWS = R4D_MXD_MTILE * DTM;
  constexpr int STR = DBK + R4D_MXD_PAD;
  constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
  constexpr int W_IT = NTILES_T * KSTEPS_T * 32 / (NTHR * VEC);
  constexpr int A_IT = (ROWS * DBK + NTHR * 16 - 1) / (NTHR * 16);
  __shared__ unsigned char sA[ROWS * STR];
  __shared__ unsigned char sW[BND * STR];
  __shared__ float sR[ROWS];     // ABLK: a boundary's ratio a row; then every row's final scale
  __shared__ unsigned s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;
  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);
  const int k_hi = (s_hi * DBK < K ? s_hi * DBK : K);

  v8f acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i) acc[i] = (v8f)(0.f);
  const float* __restrict__ asrow = As + (size_t)(tid < M ? tid : M - 1) * as_rs;
  float ns = 0.f, np = 0.f;
  if constexpr (ABLK) {
    const int kb = ((s_lo * DBK) >> 7) + 1;
    if (tid < ROWS && ((s_lo * DBK) & 127) && kb * 128 < k_hi) { ns = asrow[kb]; np = asrow[kb - 1]; }
  }
  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  const int ksteps_g = K / 16;

  // per-thread slot geometry, fixed across slabs
  int s_r[W_IT], s_kst[W_IT], s_kloc[W_IT], s_gc[W_IT], s_lanec[W_IT];
  int s_ref[W_IT][VEC];
#pragma unroll
  for (int it = 0; it < W_IT; ++it) {
    const int sl = it * NTHR * VEC + tid * VEC;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
    const int r = ntl * 16 + (lane_ & 15);
    const int gn = n0 + r;
    const int gc = gn < N - VEC ? gn : N - VEC;
    s_r[it] = r; s_kst[it] = kst; s_kloc[it] = kst * 16 + (lane_ >> 4) * 8;
    s_gc[it] = gc; s_lanec[it] = (lane_ & 16) | (gc & 15);
#pragma unroll
    for (int q = 0; q < VEC; ++q) s_ref[it][q] = (int)Wref[gc + q];
  }
  typedef unsigned wv_t __attribute__((ext_vector_type(VEC)));
  wv_t rw[W_IT];
  int rd[W_IT][VEC];
  mxd_u4 ra[A_IT];

  auto load = [&](int s) {
    const int k0 = s * DBK;
#pragma unroll
    for (int j = 0; j < A_IT; ++j) {
      const int idx = j * NTHR * 16 + tid * 16;
      if (idx < ROWS * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;
        ra[j] = *(const mxd_u4*)(A + (size_t)rc * lda + k0 + c);
      }
    }
    const int kstep0 = k0 / 16;
#pragma unroll
    for (int it = 0; it < W_IT; ++it) {
      const size_t slot =
          ((size_t)(s_gc[it] >> 4) * ksteps_g + kstep0 + s_kst[it]) * 32 + s_lanec[it];
      rw[it] = mxd_ld<NT>((const wv_t*)&Wp[slot]);
      const int blk = (k0 + s_kst[it] * 16) / R4D_MXFP4_GROUP;
#pragma unroll
      for (int q = 0; q < VEC; ++q) rd[it][q] = (int)Ws[(size_t)blk * ldws + s_gc[it] + q];
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int j = 0; j < A_IT; ++j) {
      const int idx = j * NTHR * 16 + tid * 16;
      if (idx < ROWS * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        *(mxd_u4*)(&sA[r * STR + c]) = ra[j];
      }
    }
#pragma unroll
    for (int it = 0; it < W_IT; ++it) {
#pragma unroll
      for (int q = 0; q < VEC; ++q) {
        unsigned int lo, hi;
        r4d_mxfp4_unpack8(rw[it][q], r4d_mxfp4_fold_d(s_ref[it][q], rd[it][q]), &lo, &hi);
        *(mxd_u2*)(&sW[(s_r[it] + q) * STR + s_kloc[it]]) = mxd_u2{lo, hi};
      }
    }
  };

  if (s_lo < s_hi) load(s_lo);
  for (int s = s_lo; s < s_hi; ++s) {
    const int k0 = s * DBK;
    const bool bnd = ABLK && s > s_lo && (k0 & 127) == 0;
    if constexpr (ABLK) {
      if (bnd && tid < ROWS) sR[tid] = mxd_ratio(ns, np, asrow, k0 >> 7);
    }
    store();
    __syncthreads();
    if (s + 1 < s_hi) load(s + 1);           // the next slab's loads fly across this slab's WMMAs
    if constexpr (ABLK) {
      if (bnd) mxd_rescale(acc, sR, col);
      if (tid < ROWS && (k0 & 127) == 0 && k0 + 128 < k_hi) {
        ns = asrow[(k0 >> 7) + 1];
        np = asrow[k0 >> 7];
      }
    }
#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      mxd_v2i af[DTM], wf;
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char* pa = &sA[(i * 16 + col) * STR + kk];
        af[i][0] = *(const int*)pa; af[i][1] = *(const int*)(pa + 4);
      }
      const unsigned char* pw = &sW[(wave * 16 + col) * STR + kk];
      wf[0] = *(const int*)pw; wf[1] = *(const int*)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wf, af[i], acc[i]);
    }
    __syncthreads();
  }
  const int kb_end = s_hi > s_lo ? (k_hi - 1) >> 7 : 0;
  mxd_epilogue<DTM, DKS, ABLK>(acc, As, as_rs, Wref, P, cnt, C, M, N, ldc, n0, wave, col, kb8, ks,
                               kb_end, &s_last, sR);
}

namespace {
// The arrival counters, one block per output buffer (see the header). They cannot live in the
// scratch, which other ops overwrite, because they must be zero at entry. One allocation per
// (device, output), kept zero by the reducer and freed only with the output's memory (a recorded
// pass replays the pointer; r4d_keyed.h), as gemm_fp8a8_nt_m16's split_ctr.
unsigned* mxd_out_ctr(const void* out) {
  return r4d::keyed_block(r4d::Keyed::MxDecode, nullptr, out,
                          (size_t)(R4D_MXD_MAX_N / (R4D_MXD_DWN * 16)) * sizeof(unsigned));
}

// The split, slab and load hint for a launch, from M, the shape and the tunables (0 = the rule; see
// the header). Shared by the launch and the scratch hook so the sized partial is the one written.
struct MxdPlan { int ks, bk; bool nt; };
MxdPlan mxd_plan(int M, int K, int N, int KS, int BK, int NT) {
  const int nblk = (N + R4D_MXD_DWN * 16 - 1) / (R4D_MXD_DWN * 16);
  const int tm = (M + R4D_MXD_MTILE - 1) / R4D_MXD_MTILE;
  MxdPlan p;
  int ks = 1;
  if (tm > 4) {
    // Past 64 rows (see the header): split only while every split keeps 16 128-wide slabs, towards
    // a 48-block grid, or at a deep K regardless.
    const int slabs = K / 128;
    while (ks < R4D_MXD_MAX_KS && (nblk * ks < 48 || K >= 8192) && slabs / (2 * ks) >= 16 &&
           2 * ks * M <= 1024)
      ks <<= 1;
  } else {
    const int fill = M <= 16 ? 48 : 56;
    while (ks < R4D_MXD_MAX_KS && nblk * ks < fill && 2 * ks * M <= 512) ks <<= 1;
    if (K >= 8192)                     // a deep K: a split of 24 or more slabs still waits on them
      while (ks < R4D_MXD_MAX_KS && (K / 128) / ks >= 24 && 2 * ks * M <= 512) ks <<= 1;
  }
  p.ks = KS > 0 ? KS : ks;
  // 64-wide slabs where LDS per block would cost resident blocks: DTM 4 unsplit on 110+ column
  // blocks, by the split the launch runs (a pin or a tuned point included); past 64 rows the
  // 128-wide slab always.
  const bool slab64 = tm == 4 && p.ks == 1 && nblk >= 110;
  p.bk = BK > 0 ? BK : (slab64 ? 64 : 128);
  p.nt = NT == 1 || (NT == 0 && (long)N * K / 2 >= (4l << 20));
  if (K % p.bk) {                      // a slab may not run past K
    p.bk = 64;
    if (p.ks == 2) p.ks = 4;
  }
  return p;
}

long mxd_partial_bytes(int M, int N, int ks) {
  return ks > 1 ? (long)ks * M * N * (long)sizeof(float) : 0;
}

void mxd_launch(long a, long ascale, long wq, long ws, long wref, long c, int M, int K, int N,
                int KS, int BK, int NT, int as_group, int as_rs, long stream, long scratch,
                long scratch_bytes, int lda, int ldws, int ldc) {
  if (M < 1 || M > R4D_MXD_MTILE * R4D_MXD_MAX_TM)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: M must be 1..128");
  if (N < 16 || N % 16 || N > R4D_MXD_MAX_N)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: N must be a multiple of 16 up to 36864");
  if (K < 64 || K % 64)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: K must be a multiple of 64");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: the activation scale is per row or per "
                             "128 of K");
  const MxdPlan pl = mxd_plan(M, K, N, KS, BK, NT);
  const int ks = pl.ks, bk = pl.bk;
  if (ks != 1 && ks != 2 && ks != 4 && ks != 8 && ks != 16)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: KS must be 1, 2, 4, 8 or 16");
  if (bk != 64 && bk != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: BK must be 64 or 128");
  if (scratch_bytes < mxd_partial_bytes(M, N, ks) || (ks > 1 && !scratch))
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: the op's scratch is shorter than the split "
                             "partial; the scratch hook sizes it");
  float* P = (float*)scratch;
  unsigned* cnt = mxd_out_ctr((const void*)c);
  if (!cnt) throw std::runtime_error("r4d gemm_mxfp4a8_decode: split counter alloc failed");
  const bool ablk = as_group == 128, nt = pl.nt;
  const int nblk = (N + R4D_MXD_DWN * 16 - 1) / (R4D_MXD_DWN * 16);
  const int tm = (M + R4D_MXD_MTILE - 1) / R4D_MXD_MTILE;
  hipStream_t st = (hipStream_t)stream;
  dim3 grid(nblk, 1, ks), block(R4D_MXD_DWN * 32);
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
#define MXD_L(BK_, KS_, TM_, NT_, AB_)                                                         \
  hipLaunchKernelGGL((r4d_gemm_mxfp4a8_decode_kernel<BK_, KS_, TM_, NT_, AB_>), grid, block, 0, \
                     st, Ap, Sp, Wp, Zp, Rp, P, cnt, Cp, M, K, N, as_rs, lda, ldws, ldc)
#define MXD_L1(BK_, KS_, NT_, AB_)                                                              \
  hipLaunchKernelGGL((r4d_gemm_mxfp4a8_decode1_kernel<BK_, KS_, NT_, AB_>), grid, block, 0, st, \
                     Ap, Sp, Wp, Zp, Rp, P, cnt, Cp, M, K, N, as_rs, lda, ldws, ldc)
#define MXD_AB(BK_, KS_, TM_, NT_) \
  { if (ablk) MXD_L(BK_, KS_, TM_, NT_, true); else MXD_L(BK_, KS_, TM_, NT_, false); }
#define MXD_NT(BK_, KS_, TM_) \
  { if (nt) MXD_AB(BK_, KS_, TM_, true) else MXD_AB(BK_, KS_, TM_, false) }
#define MXD_AB1(BK_, KS_, NT_) \
  { if (ablk) MXD_L1(BK_, KS_, NT_, true); else MXD_L1(BK_, KS_, NT_, false); }
#define MXD_NT1(BK_, KS_) { if (nt) MXD_AB1(BK_, KS_, true) else MXD_AB1(BK_, KS_, false) }
#define MXD_TM(BK_, KS_)                                                                       \
  { switch (tm) { case 1: MXD_NT1(BK_, KS_) break; case 2: MXD_NT(BK_, KS_, 2) break;          \
                  case 3: MXD_NT(BK_, KS_, 3) break; case 4: MXD_NT(BK_, KS_, 4) break;        \
                  case 5: MXD_NT(BK_, KS_, 5) break; case 6: MXD_NT(BK_, KS_, 6) break;        \
                  case 7: MXD_NT(BK_, KS_, 7) break; default: MXD_NT(BK_, KS_, 8) break; } }
#define MXD_KS(BK_)                                                                            \
  { switch (ks) { case 1: MXD_TM(BK_, 1) break; case 2: MXD_TM(BK_, 2) break;                  \
                  case 4: MXD_TM(BK_, 4) break; case 8: MXD_TM(BK_, 8) break;                  \
                  default: MXD_TM(BK_, 16) break; } }
  if (bk == 64) MXD_KS(64) else MXD_KS(128)
#undef MXD_KS
#undef MXD_TM
#undef MXD_NT1
#undef MXD_AB1
#undef MXD_NT
#undef MXD_AB
#undef MXD_L1
#undef MXD_L
}
}  // namespace

extern "C" int r4d_gemm_mxfp4a8_decode_max_m(void) { return R4D_MXD_MTILE * R4D_MXD_MAX_TM; }

// ============================================================== the plugin ABI entry point
extern "C" int r4d_rad_gemm_mxfp4a8_decode(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (g.N < 16 || g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale || !g.bref) return R4D_NO(RAD_E_INVAL);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    if (g.K % 64) return R4D_NO(RAD_E_SHAPE);
    if (g.M > r4d_gemm_mxfp4a8_decode_max_m() || g.N > R4D_MXD_MAX_N) return R4D_NO(RAD_E_SHAPE);
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs, mp.n);
    if (rc != RAD_OK) return rc;
    R4dMxDecCfg t;
    r4d_cfg_mxfp4_decode(a, &t);
    const long long kb = (g.K + 127) / 128;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        R4D_TRY("gemm_mxfp4a8_decode",
            mxd_launch(r4d_addr(g.a) + p * g.K, r4d_addr(g.ascale) + (long)(p * kb * 4),
                       r4d_addr(g.b) + n0 * g.K / 2, r4d_addr(g.bscale) + n0,
                       r4d_addr(g.bref) + n0, r4d_addr(g.y) + n0 * 2,
                       (int)g.M, (int)g.K, (int)np, t.ks, t.bk, t.nt, as_group, as_rs,
                       (long)(uintptr_t)s, (long)(uintptr_t)a->scratch, (long)a->scratch_bytes,
                       (int)(mp.lda ? mp.lda : g.K), (int)g.N, (int)g.N));
    }
    return RAD_OK;
}

/* The op's scratch size. A narrower M can split wider than the band's widest M, so the size is the
 * widest split over the band times the widest M, an upper bound for every launch. Parts of a
 * partitioned weight split at their own N and run in turn sharing the scratch, so the bound is the
 * largest part's. */
extern "C" int64_t r4d_rad_gemm_mxfp4a8_decode_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), N = r4d_geti_or(a, "N", 0);
    const long long K = r4d_geti_or(a, "K", 0);
    if (M < 1 || N < 16 || K < 64 || M > r4d_gemm_mxfp4a8_decode_max_m()) return 0;
    R4dMxDecCfg t;
    r4d_cfg_mxfp4_decode(a, &t);
    long long bound[4];
    mx_part_bounds(N, r4d_geti_or(a, "pb1", 0), r4d_geti_or(a, "pb2", 0), bound);
    int64_t worst = 0;
    for (int i = 0; i < 3; ++i) {
        const long long np = bound[i + 1] - bound[i];
        if (np < 16) continue;
        int ks = 1;
        for (int m = 1; m <= (int)M; ++m) {
            const int k = mxd_plan(m, (int)K, (int)np, t.ks, t.bk, t.nt).ks;
            if (k > ks) ks = k;
        }
        const int64_t b = (int64_t)mxd_partial_bytes((int)M, (int)np, ks);
        if (b > worst) worst = b;
    }
    return worst;
}
