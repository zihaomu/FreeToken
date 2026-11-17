// r4d_gemm_mxfp4a8_nt_m64.hip - r4d_gemm_mxfp4a8_nt_m64: OCP-MXFP4 weight, fp8 activation skinny
// GEMM for M up to 64, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, A quantised per row to
// e4m3 and W stored as OCP micro-scaling fp4: an e2m1 element per nibble and one E8M0 exponent
// per 32 elements along K.
//
// Unlike r4d_gemm_w4a8_nt_m64, whose int4 code the library derives, this reads a checkpoint
// format: MXFP4 ships its own per-32 E8M0 exponents, and re-deriving int4 would quantise twice. The
// fp8 WMMA runs at the iu8 rate on this part (f16 207 TF/s; fp8 412; iu8 407).
//
// The per-32 exponent is folded into the element. With a per-row reference Wref[n] =
// max_k E8M0[n][k] (computed offline), d = Wref[n] - E8M0[n][blk] is folded into the e4m3 value:
// the e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} scaled by 2^-d are e4m3 numbers, so a 16-entry
// table indexed by d replaces the per-element multiply with a v_perm_b32. One factor per row,
// 2^(Wref[n]-127), is applied in the epilogue with the activation scale.
//
// The table extends into e4m3 subnormals: stopping at 2^-6 would flush 6.7% of one checkpoint's
// weights (per-channel errors up to 18.7%). Down to 2^-9 every d <= 8 is exact, which relies on the
// gfx12 fp8 WMMA honouring e4m3 subnormals (exact for every m*2^-9, m = 1..7). d >= 13 is zero.
//
// Unpack (r4d_mxfp4.h): four v_perm_b32 per eight weights, no multiply. The scale lookup is once
// per fragment: eight consecutive k at an offset that is a multiple of 8 lie in one 32-block.
//
// Split-K is within the workgroup, reduced in LDS: N/16 tiles alone do not fill the part (320
// waves at N=5120), and a cross-workgroup split would need an HBM partial buffer and a counter.
// The kernel needs no scratch, so it is graph-replay safe.
//
// The weight is pre-permuted into fragment order (as for w4a16/w4a8). In checkpoint order the
// sixteen lanes of a half-wave would read rows K/2 bytes apart. The layout is one 32-lane slot per
// (n-tile, k-step):
//
//     Wp[((n/16) * (K/16) + (k/16)) * 32 + lane]      one uint32 = 4 packed bytes = 8 elements
//
// where lane's row is 16*(n/16) + (lane&15) and its eight elements start at k + 8*(lane>>4); a wave
// reads 128 contiguous bytes per step. Taking W as [N/16][16][K/16][8] bytes, slot l of tile
// (nt, ks) is W[nt][l&15][ks][4*(l>>4) .. +4]. permute_w() in mxfp4_layout.py is the reference.
// With the permute this LDS-free kernel beats an LDS-staged one on every shape; without it, it
// loses on every shape. The E8M0 scales stay in [K/32][N] order (a half-wave reads sixteen
// consecutive bytes).
//
// Two activation scale grids, as in r4d_gemm_mxfp4a8_tiled.hip: one f32 per row (ABLK=false) or
// one per (row, 128 of K) (ABLK=true). With ABLK each wave carries its partial sums in units of the
// current block's scale, rescaling by s_prev / s_next at each boundary. Split-K segments end in
// different blocks, so each partial is multiplied by its last scale before the LDS reduction, and
// the epilogue applies only the weight's reference factor.
//
// Benchmarking note: a weight of tens of MB streamed in a loop stays Infinity-Cache resident, so
// such a bandwidth figure can exceed the DRAM roofline.
//
// A is not staged in LDS: a lane's A fragment is eight consecutive bytes of one row. Rows past M are
// clamped, not masked, keeping the inner loop branch-free.
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_mxfp4.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_GEMM_MXFP4_MAX_M 64
#define R4D_GEMM_MXFP4_GROUP R4D_MXFP4_GROUP
#define WAVE32 32

typedef int      v2i    __attribute__((ext_vector_type(2)));
typedef unsigned uint2v __attribute__((ext_vector_type(2)));

// Eight e2m1 codes (four packed bytes) -> the B fragment: eight e4m3 bytes in k order with the
// block's exponent difference folded in (r4d_mxfp4.h).
__device__ __forceinline__ v2i r4d_mxfp4_frag8(unsigned int wv, int d) {
  unsigned int lo, hi;
  r4d_mxfp4_unpack8(wv, d, &lo, &hi);
  v2i out;
  out[0] = (int)lo;
  out[1] = (int)hi;
  return out;
}

template <int MT, int NPW, bool ABLK>
__global__ void r4d_gemm_mxfp4a8_nt_m64_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK, int as_rs, int lda, int ldws, int ldc) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  // Column tiles this wave owns, clamped to the last real tile: grid.x rounds N/16 up to a multiple
  // of WV*NPW, and the weight is indexed by tile, so an unclamped index would fault. The epilogue
  // drops these columns.
  const int ntmax = N / 16 - 1;
  int wtile[NPW], wrow[NPW];
  unsigned char wref[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    wtile[j] = t < ntmax ? t : ntmax;
    wrow[j] = wtile[j] * 16 + nlane;
    wref[j] = Wref[wrow[j]];
  }

  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  const int ksteps = K / 16;

  // ABLK: the scale each accumulator row of this lane is carried in (row (mt0+i)*16 + mhalf + e).
  float scur[ABLK ? MT : 1][8];
  if constexpr (ABLK) {
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        int r = (mt0 + i) * 16 + mhalf + e;
        r = r < M - 1 ? r : M - 1;
        const float s = As[(size_t)r * as_rs + (kbase >> 7)];
        scur[i][e] = s > 0.f ? s : 1.f;
      }
  }

  for (int k = 0; k < Ks; k += 16) {
    const int kg = kbase + k;                       // fragment base, both lane halves
    if constexpr (ABLK) {
      if (k && (kg & 127) == 0) {                   // a new 128-wide activation block
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++) {
            int r = (mt0 + i) * 16 + mhalf + e;
            r = r < M - 1 ? r : M - 1;
            float s = As[(size_t)r * as_rs + (kg >> 7)];
            s = s > 0.f ? s : scur[i][e];
            const float q = scur[i][e] / s;
            scur[i][e] = s;
#pragma unroll
            for (int j = 0; j < NPW; j++) acc[j][i][e] *= q;
          }
      }
    }
    const int kstep = kg / 16;
    const int blk = kg / R4D_GEMM_MXFP4_GROUP;      // one E8M0 block covers this whole step
    v2i af[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + nlane;
      r = r < M - 1 ? r : M - 1;
      const unsigned int* pa = (const unsigned int*)(A + (size_t)r * lda + kg + mhalf);
      af[i][0] = (int)pa[0];
      af[i][1] = (int)pa[1];
    }
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int n = wrow[j];
      // Fragment-order weight: lane l's four bytes for this (n-tile, k-step) are at slot l.
      const unsigned int wv = Wp[((size_t)wtile[j] * ksteps + kstep) * WAVE32 + lane];
      const v2i wf = r4d_mxfp4_frag8(wv, r4d_mxfp4_fold_d(wref[j], Ws[(size_t)blk * ldws + n]));
#pragma unroll
      for (int i = 0; i < MT; i++)
        acc[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[j][i]);
    }
  }

  // Reduce the SK partial sums in LDS, one M tile at a time so the buffer stays WV*NPW*SK*256.
  const int ncols = WV * NPW;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++)
        red[slot + lane * 8 + e] = ABLK ? acc[j][i][e] * scur[i][e] : acc[j][i][e];
    }
    __syncthreads();
    // One pass over the block's 256-element tiles; every thread takes a stride.
    for (int t = tid; t < ncols * 256; t += blockDim.x) {
      const int col = t >> 8, off = t & 255;
      // off = lane*8 + e. A lane's N index is lane&15 and its eight elements are M rows
      // 8*(lane>>4)+e; using the whole lane id for N would write the upper half-wave into the
      // next column tile.
      const int n = (blockIdx.x * ncols + col) * 16 + ((off >> 3) & 15);
      const int m = (mt0 + i) * 16 + ((off >> 7) << 3) + (off & 7);
      if (n >= N || m >= M) continue;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      C[(size_t)m * ldc + n] =
          (__hip_bfloat16)(v * __int_as_float((int)Wref[n] << 23) * (ABLK ? 1.f : As[m]));
    }
  }
}

extern "C" int r4d_gemm_mxfp4a8_nt_m64_max_m(void) { return R4D_GEMM_MXFP4_MAX_M; }
extern "C" int r4d_gemm_mxfp4a8_nt_m64_group(void) { return R4D_GEMM_MXFP4_GROUP; }

extern "C" void r4d_gemm_mxfp4a8_nt_m64(long a, long ascale, long wq, long ws, long wref, long c,
                                        int M, int K, int N, int WV, int SK, int MB, int NPW,
                                        int as_group, int as_rs, long stream,
                                        int lda, int ldws, int ldc) {
  // A part of a partitioned weight (ParoQuant) reads windows of larger planes: activation rows of
  // lda bytes, exponent rows of ldws, output rows of ldc. 0 means the plane's own width.
  if (!lda) lda = K;
  if (!ldws) ldws = N;
  if (!ldc) ldc = N;
  if (M < 1 || M > R4D_GEMM_MXFP4_MAX_M)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: M must be 1..64");
  if (N % 16) throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_MXFP4_GROUP))
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: K must be divisible by SK*32");
  if (WV * SK * WAVE32 > 1024)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: NPW must be 1, 2, 4 or 8");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: the activation scale is per row or per 128 of K");
  const bool ablk = as_group == 128;
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: WV*NPW*SK too large");
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MTv, NPv)                                                                          \
  do {                                                                                            \
    if (ablk) hipLaunchKernelGGL((r4d_gemm_mxfp4a8_nt_m64_kernel<MTv, NPv, true>), grid, block,   \
                                 smem, st, Ap, Sp, Wp, Zp, Rp, Cp, M, K, N, WV, SK, as_rs, lda, ldws, ldc);   \
    else      hipLaunchKernelGGL((r4d_gemm_mxfp4a8_nt_m64_kernel<MTv, NPv, false>), grid, block,  \
                                 smem, st, Ap, Sp, Wp, Zp, Rp, Cp, M, K, N, WV, SK, as_rs,        \
                                 lda, ldws, ldc);                                                 \
  } while (0)
#define LAUNCH_MB(NPv) switch (MB) { \
    case 1: LAUNCH(1, NPv); break; \
    case 2: LAUNCH(2, NPv); break; \
    case 3: LAUNCH(3, NPv); break; \
    case 4: LAUNCH(4, NPv); break; \
    default: throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: MB unsupported"); }
  switch (NPW) {
    case 1: LAUNCH_MB(1); break;
    case 2: LAUNCH_MB(2); break;
    case 4: LAUNCH_MB(4); break;
    case 8: LAUNCH_MB(8); break;
    default: throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: NPW unsupported");
  }
#undef LAUNCH_MB
#undef LAUNCH
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
extern "C" int r4d_rad_gemm_mxfp4a8_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.M > r4d_gemm_mxfp4a8_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    /* The per-row reference exponent arrives as gemm_nt_q's trailing optional operand (RadArgs
     * cannot carry a weight aux stream); the fold needs it. */
    if (!g.bref) return R4D_NO(RAD_E_INVAL);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    const int group = r4d_gemm_mxfp4a8_nt_m64_group();
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs, mp.n);
    if (rc != RAD_OK) return rc;
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_FLAT, &c);
    if (rc != RAD_OK) return rc;
    const long long kb = (g.K + 127) / 128;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        R4D_TRY("gemm_mxfp4a8_nt_m64",
            r4d_gemm_mxfp4a8_nt_m64(r4d_addr(g.a) + p * g.K,
                                    r4d_addr(g.ascale) + (long)(p * kb * 4),
                                    r4d_addr(g.b) + n0 * g.K / 2, r4d_addr(g.bscale) + n0,
                                    r4d_addr(g.bref) + n0, r4d_addr(g.y) + n0 * 2,
                                    (int)g.M, (int)g.K, (int)np, c.wv, c.sk, c.mb, c.npw,
                                    as_group, as_rs, (long)(uintptr_t)s,
                                    (int)(mp.lda ? mp.lda : g.K), (int)g.N, (int)g.N));
    }
    return RAD_OK;
}
