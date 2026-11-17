// r4d_gemm_w8a8_nt_m64.hip - r4d_gemm_w8a8_nt_m64: 8-bit-weight, 8-bit-activation skinny GEMM for
// M up to a prefill chunk, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, A quantised per row and
// W per (row, 128-wide group).
//
// Serves the linears an INT4 model promotes to eight bits (every down_proj and the attention
// o_proj, where the 4-bit grid costs the most KL) in one launch that reads exactly what the 4-bit
// GEMM reads: the same permuted int8 activation from the fused rms+Hadamard+quantise producer, the
// same per-row scale, and a weight in the same fragment order at twice the radix.
//
// It is r4d_gemm_w4a8_nt_m64 without the unpack: the eight weight bytes are the B fragment. The A
// fragment read, split-K reduction, per-group scale fold and per-row activation scale at the store
// are the same.
//
// gemm_w8a16_nt_m64 uses the f16 WMMA (207 TF/s vs 407 for iu8), which is fine for lm_head at M=8
// (DRAM bound) but caps these sites at M=72.
//
// Weight layout: per n-tile t (16 output rows) and k-block kb (KPB=32 columns), 32 lanes hold
// 16 bytes each, 512 bytes covering 16 rows x 32 k (two WMMA k steps). Byte j of lane l is
//
//     W[t*16 + (l & 15)][kb*32 + 16*(j >> 3) + 8*((j & 7) >> 2) + 4*(l >> 4) + (j & 3)]
//
// the iu8 16x16x16 fragment order the A side uses, so one permutation serves both operands. A
// group is GROUP/KPB blocks; its f16 scale is read once per group per n-plane.
//
// The scale plane is f16 only: a symmetric grid has no zero point, and the unused half of a
// (scale, zero) dword would be 1.5% extra traffic on a DRAM-bound kernel.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"
#include "r4d_chain.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

// Not structural (grid.y tiles M, rows are clamped); a sanity bound only.
#define R4D_GEMM_W8A8_MAX_M 32768
#define R4D_GEMM_W8A8_KPB   32       // K per packed block: 2 k steps, 16 bytes per lane
#define WAVE32 32

typedef int      v8i    __attribute__((ext_vector_type(8)));
typedef float    v4f    __attribute__((ext_vector_type(4)));
typedef int      v2i    __attribute__((ext_vector_type(2)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));
typedef unsigned uint2v __attribute__((ext_vector_type(2)));

typedef __attribute__((address_space(1))) const uint4v*        r4d_gw8;
typedef __attribute__((address_space(1))) const unsigned short* r4d_gs2;
typedef __attribute__((address_space(1))) const uint2v*        r4d_ga2;

// The A fragment, as r4d_gemm_w4a8_nt_m64 reads it.
__device__ __forceinline__ v2i fragA8w(r4d_ga2 p, int u) {
  const uint2v w = p[2 * u];
  v2i r;
  r[0] = (int)w[0];
  r[1] = (int)w[1];
  return r;
}

__device__ __forceinline__ v8i mmai8(v2i a, v2i b, v8i c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}

// __launch_bounds__ is a register budget and depends on MT*NPW. A block of T threads gives
// 1536/(T/128) VGPRs per wave (1024 threads: 192; 512 or fewer: the 256 cap). `acc` and `g_acc` are
// MT*NPW*8 VGPRs each, so under a flat 1024 bound every wide tile spills. A wide tile has enough
// independent WMMA to hide latency without more waves; bounding it at 256 threads gives it 256
// VGPRs. The bound is a hard launch limit, so the host checks WV*SK against the same expression.
#define R4D_W8A8_BOUNDS(MT, NPW) ((MT) * (NPW) >= 8 ? 256 : ((MT) * (NPW) >= 4 ? 512 : 1024))

// The body; the plain kernel and its chained form (r4d_chain.h) wrap it below. Its early returns
// sit after its last barrier, so every block still reaches the wrapper's add.
template <int MT, int NPW, bool NT>
__device__ __forceinline__ void r4d_gemm_w8a8_nt_m64_body(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned short* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / R4D_GEMM_W8A8_GROUP;
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    // Rows clamped into 0..M-1, never masked; the store discards rows past M. A is int8, so the
    // row stride is K bytes.
    r4d_ga2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + (lane & 15);
      r = r < M - 1 ? r : M - 1;
      const size_t off = (size_t)r * K + kbase + 8 * (lane >> 4);
      pa[i] = (r4d_ga2)(const uint2v*)(A + off);
    }

    const int kpb = K / R4D_GEMM_W8A8_KPB;
    const int nsz = K / R4D_GEMM_W8A8_GROUP;
    r4d_gw8 pw[NPW];
    r4d_gs2 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_gw8)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W8A8_KPB) * WAVE32 + lane;
      ps[j] = (r4d_gs2)(const unsigned short*)(Ws) +
              (size_t)(t * nsz + kbase / R4D_GEMM_W8A8_GROUP) * 16 + (lane & 15);
    }

    const int bpg = R4D_GEMM_W8A8_GROUP / R4D_GEMM_W8A8_KPB;

    for (int g = 0; g < groups; g++) {
      float sc[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++)
        sc[j] = __half2float(__ushort_as_half(ps[j][(size_t)g * 16]));

      v8i g_acc[NPW][MT];
      // The whole group's activation, issued before any of it is consumed. Loading each fragment
      // just before its WMMA gives one s_wait_loadcnt per load and no memory-level parallelism; a
      // lane's A bytes for consecutive k steps are 16 bytes apart, so they cannot merge into one
      // b128 and only the issue order can change.
      v2i afg[2 * (R4D_GEMM_W8A8_GROUP / R4D_GEMM_W8A8_KPB)][MT];
#pragma unroll
      for (int u = 0; u < 2 * bpg; u++)
#pragma unroll
        for (int i = 0; i < MT; i++)
          afg[u][i] = fragA8w(pa[i], u);
#pragma unroll
      for (int b = 0; b < bpg; b++) {
        uint4v wv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const size_t o = (size_t)(g * bpg + b) * WAVE32;
          wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        }
        // Two WMMA k steps per 16-byte weight load: dwords (0,1) then (2,3).
#pragma unroll
        for (int s = 0; s < 2; s++) {
          const int u = b * 2 + s;
          v2i afrag[MT];
#pragma unroll
          for (int i = 0; i < MT; i++) afrag[i] = afg[u][i];
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            // The eight weight bytes are the fragment.
            v2i bfrag;
            bfrag[0] = (int)wv[j][2 * s];
            bfrag[1] = (int)wv[j][2 * s + 1];
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = (u == 0) ? mmai8(afrag[i], bfrag, (v8i)(0))
                                     : mmai8(afrag[i], bfrag, g_acc[j][i]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++)
            acc[j][i][e] = fmaf((float)g_acc[j][i][e], sc[j], acc[j][i][e]);
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += R4D_GEMM_W8A8_GROUP / 8;
    }
  }

  // Split-K reduction and store, one row tile at a time, applying the per-row activation scale.
  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;

  // At SK == 1 there is nothing to reduce: store directly and skip the two barriers per row tile.
  // (m, n) is the LDS path's index arithmetic with off = e*32 + lane.
  if (SK == 1) {
    if (n0 >= N) return;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int n = (nt0 + j) * 16 + (lane & 15);
      if (n >= N) continue;
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int mb = (mt0 + i) * 16 + 8 * (lane >> 4);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int m = mb + e;
          if (m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(acc[j][i][e] * Ascale[m]);
        }
      }
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++) red[slot + e * WAVE32 + lane] = acc[j][i][e];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(v * Ascale[m]);
    }
  }
}
template <int MT, int NPW, bool NT>
__global__ __launch_bounds__(R4D_W8A8_BOUNDS(MT, NPW)) void r4d_gemm_w8a8_nt_m64_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned short* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
    r4d_gemm_w8a8_nt_m64_body<MT, NPW, NT>(A, Ascale, Wq, Ws, C, M, K, N, WV, SK);
}
template <int MT, int NPW, bool NT>
__global__ __launch_bounds__(R4D_W8A8_BOUNDS(MT, NPW)) void r4dchain_gemm_w8a8_nt_m64(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned short* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
    r4d_chain_wait();
    r4d_gemm_w8a8_nt_m64_body<MT, NPW, NT>(A, Ascale, Wq, Ws, C, M, K, N, WV, SK);
    r4d_chain_done();
}

/* `chain`: the chained form (the plugin entry, where the row resolved with `chain`). */
static void gemm_w8a8_m64_impl(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                               int WV, int SK, int MB, int NPW, int NT, long stream, bool chain);

void r4d_gemm_w8a8_nt_m64(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int WV, int SK, int MB, int NPW, int NT, long stream) {
  gemm_w8a8_m64_impl(a, ascale, wq, ws, c, M, K, N, WV, SK, MB, NPW, NT, stream, false);
}

static void gemm_w8a8_m64_impl(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                               int WV, int SK, int MB, int NPW, int NT, long stream, bool chain) {
  if (M < 1 || M > R4D_GEMM_W8A8_MAX_M)
    throw std::runtime_error("r4d gemm_w8a8_nt_m64: M out of range");
  if (N % 16) throw std::runtime_error("r4d gemm_w8a8_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_W8A8_GROUP))
    throw std::runtime_error("r4d gemm_w8a8_nt_m64: K must be divisible by SK*group");
  if (WV * SK * WAVE32 > R4D_W8A8_BOUNDS(MB, NPW))
    throw std::runtime_error("r4d gemm_w8a8_nt_m64: WV*SK*32 exceeds this tile's launch bound");
  // cfg_for (r4d_gemm_args.h) caps MB at 4.
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w8a8_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_w8a8_nt_m64: NPW must be 1, 2, 4 or 8");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w8a8_nt_m64: WV*NPW*SK too large");
  auto Ap = (const signed char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned*)wq;
  auto Zp = (const unsigned short*)ws;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH8(MT, NP, TT) hipLaunchKernelGGL((chain ? r4dchain_gemm_w8a8_nt_m64<MT, NP, TT>     \
                                                   : r4d_gemm_w8a8_nt_m64_kernel<MT, NP, TT>), \
    grid, block, smem, st, Ap, Sp, Wp, Zp, Cp, M, K, N, WV, SK)
// Only tiles with MT*NPW <= 8 are built: a tile costs about 16*MT*NPW + 40 VGPRs against the 256
// cap.
#define LAUNCH8_MB4(NP, TT) switch (MB) { \
    case 1: LAUNCH8(1, NP, TT); break; \
    case 2: LAUNCH8(2, NP, TT); break; \
    case 3: LAUNCH8(3, NP, TT); break; \
    case 4: LAUNCH8(4, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a8_nt_m64: MB unsupported"); }
#define LAUNCH8_MB2(NP, TT) switch (MB) { \
    case 1: LAUNCH8(1, NP, TT); break; \
    case 2: LAUNCH8(2, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a8_nt_m64: MB*NPW > 8 would spill"); }
#define LAUNCH8_MB1(NP, TT) switch (MB) { \
    case 1: LAUNCH8(1, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a8_nt_m64: MB*NPW > 8 would spill"); }
#define LAUNCH8_NPW(TT) switch (NPW) { \
    case 1: LAUNCH8_MB4(1, TT); break; \
    case 2: LAUNCH8_MB4(2, TT); break; \
    case 4: LAUNCH8_MB2(4, TT); break; \
    case 8: LAUNCH8_MB1(8, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a8_nt_m64: NPW must be 1, 2, 4 or 8"); }
  if (NT) { LAUNCH8_NPW(true); } else { LAUNCH8_NPW(false); }
#undef LAUNCH8_NPW
#undef LAUNCH8_MB1
#undef LAUNCH8_MB2
#undef LAUNCH8_MB4
#undef LAUNCH8
}

int r4d_gemm_w8a8_nt_m64_max_m(void) { return R4D_GEMM_W8A8_MAX_M; }
int r4d_gemm_w8a8_nt_m64_group(void) { return R4D_GEMM_W8A8_GROUP; }
int r4d_gemm_w8a8_nt_m64_kpb(void) { return R4D_GEMM_W8A8_KPB; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
extern "C" int r4d_rad_gemm_w8a8_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.M > r4d_gemm_w8a8_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    const int group = r4d_gemm_w8a8_nt_m64_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_MBNPW, &c);
    if (rc != RAD_OK) return rc;
    /* the engine's chained launch (RadKernelChainFn), at decode heights only */
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && g.M <= 64;
    R4D_TRY("gemm_w8a8_nt_m64",
        gemm_w8a8_m64_impl(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                           r4d_addr(g.bscale), r4d_addr(g.y),
                           (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw, c.nt,
                           (long)(uintptr_t)s, chain));
    return RAD_OK;
}

