// r4d_gemm_w2a8_nt.hip - r4d_gemm_w2a8_nt: 2-bit-weight, 8-bit-activation skinny GEMM, gfx1201.
// C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with A quantised per row and W asymmetric per (row, group).
//
// Used for the DFlash2 draft head, `x[M,5120] @ W[124160,5120]^T` at two bits, once per decode
// step. Its 178.8 MB weight gives a memory floor near 280 us. A bf16 dequant form runs on the f16
// WMMA (207 TF/s) and is compute bound at M = 64 (~815 us); v_wmma_i32_16x16x16_iu8 at 407 TF/s
// puts the compute floor near 200 us, under the memory floor, across the M range.
//
// Structure and activation as r4d_gemm_w4a8_nt_m64: 512-byte packed block (32 lanes x 16 bytes),
// permuted int8 activation from r4d_quant_act_i8_asum, [K/group][Mpad] group-major activation sums,
// split-K reduction, per-row scale at the store. A packed block covers 128 K (64 for 4-bit).
//
// The unpack puts the code in the byte's high two bits; for q in 0..3 the byte 64*q is the
// unsigned int8 of 64*q:
//
//     t0 = (w << 6) & 0xC0C0C0C0     bytes <- fields 0, 4,  8, 12
//     t1 = (w << 4) & 0xC0C0C0C0     bytes <- fields 1, 5,  9, 13
//     t2 = (w << 2) & 0xC0C0C0C0     bytes <- fields 2, 6, 10, 14
//     t3 =  w       & 0xC0C0C0C0     bytes <- fields 3, 7, 11, 15
//
// Seven instructions per dword (two 16x16x16 fragments); the factor 64 is removed at the store.
// Fragment 2d+0 is (t0, t1) and 2d+1 is (t2, t3), so element e of fragment h reads field
// 4*(e&3) + (e>>2) + 2h of dword d, as the packer writes it.
//
// Unsigned because four levels are too few for a symmetric grid: the quantiser is asymmetric per
// (row, group of 128), w ~= sc*(q - zp). The WMMA reads B unsigned and the zero point is removed
// as sc*(<a,q> - zp*sum a) with the group's activation sum from Asum, as in
// r4d_gemm_w4a8_asym_nt_m64.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_GEMM_W2A8_KPB   128      // K per packed block: 8 k steps, 16 bytes per lane
#define WAVE32 32

typedef int      v8i    __attribute__((ext_vector_type(8)));
typedef float    v4f    __attribute__((ext_vector_type(4)));
typedef int      v2i    __attribute__((ext_vector_type(2)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));
typedef unsigned uint2v __attribute__((ext_vector_type(2)));

typedef __attribute__((address_space(1))) const uint4v*        r4d_w2_gw4;
typedef __attribute__((address_space(1))) const unsigned*      r4d_w2_gu1;
typedef __attribute__((address_space(1))) const uint2v*        r4d_w2_ga2;

__device__ __forceinline__ v2i w2_fragA(r4d_w2_ga2 p, int u) {
  const uint2v w = p[2 * u];
  v2i r;
  r[0] = (int)w[0];
  r[1] = (int)w[1];
  return r;
}

// One dword of sixteen 2-bit fields -> the two 16x16x16 B fragments it holds.
__device__ __forceinline__ void w2_dequant(unsigned w, v2i& f0, v2i& f1) {
  const unsigned M2 = 0xC0C0C0C0u;
  f0[0] = (int)((w << 6) & M2);
  f0[1] = (int)((w << 4) & M2);
  f1[0] = (int)((w << 2) & M2);
  f1[1] = (int)(w & M2);
}

__device__ __forceinline__ v8i w2_mmai(v2i a, v2i b, v8i c) {
  // B unsigned: the code is offset binary and the zero point is corrected with the activation sum.
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, false, b, c, false);
}

// __launch_bounds__ depends on MT as well as NPW. Accumulators are 8*MT*NPW VGPRs per array plus
// 8*MT for the activation sums; a bound of 1024 budgets 1536/32 = 48 VGPRs per wave, and every MT=4
// config then spills badly. The bound must be what the config can run. r4d_gemm_w4a8_nt_m64 has
// the same rule on NPW alone.
#define R4D_W2A8_BOUNDS(MT, NPW) \
    ((MT) >= 4 ? 256 : (((MT) >= 2 || (NPW) >= 4) ? 512 : 1024))

#define R4D_GEMM_W2A8_NT_PARAMS                                                                 \
    const signed char* __restrict__ A, const float* __restrict__ Ascale,                        \
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,                           \
    const float* __restrict__ Asum, __hip_bfloat16* __restrict__ C, int M, int K, int N,        \
    int WV, int SK
template <int MT, int NPW, bool NT>
__device__ __forceinline__ void r4d_gemm_w2a8_nt_body(R4D_GEMM_W2A8_NT_PARAMS) {
  extern __shared__ float redw2[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / R4D_GEMM_W2A8_GROUP;
  const int kbase = ksplit * Ks;
  const int Mpad = ((M + 15) / 16) * 16;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    r4d_w2_ga2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + (lane & 15);
      r = r < M - 1 ? r : M - 1;
      pa[i] = (r4d_w2_ga2)(const uint2v*)(A + (size_t)r * K + kbase + 8 * (lane >> 4));
    }

    const int kpb = K / R4D_GEMM_W2A8_KPB;
    const int nsz = K / R4D_GEMM_W2A8_GROUP;
    r4d_w2_gw4 pw[NPW];
    r4d_w2_gu1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_w2_gw4)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W2A8_KPB) * WAVE32 + lane;
      ps[j] = (r4d_w2_gu1)(const unsigned*)(Ws) +
              (size_t)(t * nsz + kbase / R4D_GEMM_W2A8_GROUP) * 16 + (lane & 15);
    }

    // No next-group weight prefetch: four more VGPRs per column tile on a kernel at 214 costs a wave
    // per SIMD, which loses more latency cover than the prefetch gains (likewise in w4a8_nt_m64).
    for (int g = 0; g < groups; g++) {
      // Two f32 FMAs, sc*g_acc and (-sc*z64)*asum, rather than fmaf(g_acc - z64*asum, sc, acc),
      // which compiles to v_mul + v_sub + v_fma_mix per element. This saves an instruction and
      // pairs as VOPD; the scale is converted to f32 once per group.
      float sc[NPW], scz[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const unsigned sz = ps[j][(size_t)g * 16];
        sc[j] = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
        const float nz = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
        scz[j] = -sc[j] * (64.f * (-nz - 1024.f));
      }
      // Element e of tile i at this lane is output row (mt0+i)*16 + 8*(lane>>4) + e, so a lane's
      // eight rows are contiguous at a fixed group; hence Asum is [G][Mpad].
      float asum[MT][8];
      {
        const size_t gi = (size_t)(kbase / R4D_GEMM_W2A8_GROUP + g) * Mpad;
#pragma unroll
        for (int i = 0; i < MT; i++) {
          // Clamped: MB may exceed the row tiles, and an unclamped read could fault.
          int rb = (mt0 + i) * 16 + 8 * (lane >> 4);
          rb = rb < Mpad - 8 ? rb : Mpad - 8;
          const v4f* pv = (const v4f*)(Asum + gi + rb);
          const v4f v0 = pv[0], v1 = pv[1];
#pragma unroll
          for (int e = 0; e < 4; e++) {
            asum[i][e] = v0[e];
            asum[i][e + 4] = v1[e];
          }
        }
      }

      uint4v wv[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const size_t o = (size_t)g * WAVE32;
        wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
      }

      v8i g_acc[NPW][MT];
#pragma unroll
      for (int d = 0; d < 4; d++) {
        v2i bf[NPW][2];
#pragma unroll
        for (int j = 0; j < NPW; j++) w2_dequant(wv[j][d], bf[j][0], bf[j][1]);
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int u = 2 * d + h;
          v2i afrag[MT];
#pragma unroll
          for (int i = 0; i < MT; i++) afrag[i] = w2_fragA(pa[i], u);
#pragma unroll
          for (int j = 0; j < NPW; j++)
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = (u == 0) ? w2_mmai(afrag[i], bf[j][h], (v8i)(0))
                                     : w2_mmai(afrag[i], bf[j][h], g_acc[j][i]);
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++) {
            float a = fmaf((float)g_acc[j][i][e], sc[j], acc[j][i][e]);
            acc[j][i][e] = fmaf(asum[i][e], scz[j], a);
          }
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += R4D_GEMM_W2A8_GROUP / 8;
    }
  }

  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++) redw2[slot + e * WAVE32 + lane] = acc[j][i][e];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += redw2[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);
      // 1/64 undoes the shift that put the 2-bit code in the byte's high half.
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(v * Ascale[m] * 0.015625f);
    }
  }
}

// The kernel around the body: CH `r4dchain` is the chained form (r4d_chain.h), taken at decode
// heights.
template <int MT, int NPW, bool NT, class CH>
__global__ __launch_bounds__(R4D_W2A8_BOUNDS(MT, NPW))
void r4d_gemm_w2a8_nt_kernel(R4D_GEMM_W2A8_NT_PARAMS) {
  if (CH::on) r4d_chain_wait();
  r4d_gemm_w2a8_nt_body<MT, NPW, NT>(A, Ascale, Wq, Ws, Asum, C, M, K, N, WV, SK);
  if (CH::on) r4d_chain_done();
}
#undef R4D_GEMM_W2A8_NT_PARAMS

static void gemm_w2a8_nt_emit(long a, long ascale, long wq, long ws, long asum, long c,
                              int M, int K, int N, int WV, int SK, int MB, int NPW, int NT,
                              long stream, bool chain) {
  if (M < 1) throw std::runtime_error("r4d gemm_w2a8_nt: M must be >= 1");
  if (N % 16) throw std::runtime_error("r4d gemm_w2a8_nt: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_W2A8_GROUP))
    throw std::runtime_error("r4d gemm_w2a8_nt: K must be divisible by SK*group");
  if (!asum) throw std::runtime_error("r4d gemm_w2a8_nt: asum is required");
  // __launch_bounds__ is a hard maximum: a wider block fails to launch, surfacing as
  // hipErrorLaunchFailure at the next sync.
  if (WV * SK * WAVE32 > R4D_W2A8_BOUNDS(MB, NPW))
    throw std::runtime_error("r4d gemm_w2a8_nt: block wider than this MB/NPW's launch bound");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w2a8_nt: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4)
    throw std::runtime_error("r4d gemm_w2a8_nt: NPW must be 1, 2 or 4");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w2a8_nt: WV*NPW*SK too large");
  auto Ap = (const signed char*)a;
  auto Asp = (const float*)ascale;
  auto Wp = (const unsigned*)wq;
  auto Sp = (const unsigned*)ws;
  auto Sump = (const float*)asum;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LW2C(MT, NP, TT, CH_)                                                                    \
  hipLaunchKernelGGL((r4d_gemm_w2a8_nt_kernel<MT, NP, TT, CH_>), grid, block, smem, st, Ap, Asp,  \
                     Wp, Sp, Sump, Cp, M, K, N, WV, SK)
#define LW2(MT, NP, TT)                                                                          \
  do { if (chain) LW2C(MT, NP, TT, r4dchain); else LW2C(MT, NP, TT, R4dUnchained); } while (0)
#define LW2_MB(NP, TT) switch (MB) { \
    case 1: LW2(1, NP, TT); break; \
    case 2: LW2(2, NP, TT); break; \
    case 3: LW2(3, NP, TT); break; \
    case 4: LW2(4, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w2a8_nt: MB unsupported"); }
  if (NPW == 1) {
    if (NT) { LW2_MB(1, true); } else { LW2_MB(1, false); }
  } else if (NPW == 2) {
    if (NT) { LW2_MB(2, true); } else { LW2_MB(2, false); }
  } else {
    if (NT) { LW2_MB(4, true); } else { LW2_MB(4, false); }
  }
#undef LW2_MB
#undef LW2
#undef LW2C
}

void r4d_gemm_w2a8_nt(long a, long ascale, long wq, long ws, long asum, long c,
                      int M, int K, int N, int WV, int SK, int MB, int NPW, int NT, long stream) {
  gemm_w2a8_nt_emit(a, ascale, wq, ws, asum, c, M, K, N, WV, SK, MB, NPW, NT, stream, false);
}

int r4d_gemm_w2a8_nt_group(void) { return R4D_GEMM_W2A8_GROUP; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
extern "C" int r4d_rad_gemm_w2a8(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    /* The 2-bit grid is always asymmetric, so the group sums are required. */
    if (!g.asum) return R4D_NO(RAD_E_INVAL);
    const int group = r4d_gemm_w2a8_nt_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_W2A8, &c);
    if (rc != RAD_OK) return rc;
    if (c.mb < 1 || c.mb > 4) return R4D_NO(RAD_E_SHAPE);
    if (c.npw != 1 && c.npw != 2 && c.npw != 4) return R4D_NO(RAD_E_SHAPE);
    R4D_TRY("gemm_w2a8_nt",
        gemm_w2a8_nt_emit(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale),
                          r4d_addr(g.asum), r4d_addr(g.y),
                          (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw, c.nt,
                          (long)(uintptr_t)s, r4d_geti_or(a, "chain", 0) != 0 && g.M <= 64));
    return RAD_OK;
}

