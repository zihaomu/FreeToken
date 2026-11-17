// r4d_gemm_w8a16_nt_m64.hip - r4d_gemm_w8a16_nt_m64: 8-bit-weight skinny GEMM for M up to 64,
// gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f16 A, bf16 C, fp32 accumulate.
//
// Used for lm_head, quantised to eight bits rather than four because its error lands on the
// sampled token rather than on a residual later layers can absorb.
//
// A skinny kernel holds every row of M in registers, so the weight is read once regardless of M.
// General int8 GEMMs pad M <= 16 to two 16-row macro tiles and read the weight twice. The scale is
// applied to the in-register accumulator and stored as bf16, with no separate epilogue launch.
//
// Structure matches r4d_gemm_w4a16_nt_m64: fragment order, 512 B packed block (32 lanes x 16 B),
// (scale, zero) dword per (row, group), split-K reduction and clamped tiles. Only the bits per
// element differ, so a packed block covers 32 K instead of 64.
//
// Dequant: `0x6400 | b` is exactly the f16 for 1024 + b for b in 0..255, so four v_perm_b32 turn
// two dwords into a 16x16x16 fragment, and one v_pk_add_f16 per pair adds -(1024 + zero). The
// subtraction is exact (integers below 2048), so the group scale is applied to the f32 accumulator.
//
// A is f16, not bf16, because bf16's eight significant bits cannot hold 1024 + b. The alternative
// (raw codes plus a per-group activation sum, as in r4d_gemm_w4a8_asym_nt_m64) gains nothing here:
// the adds hide under the weight stream. The caller's bf16 -> f16 cast is exact for a post-norm
// hidden state (O(10)).
//
// Quantiser: asymmetric, per output channel, per group of R4D_GEMM_W8_GROUP contiguous K:
// w ~= scale * (q - zero), q in 0..255, integer zero. One dword per (row, group): the f16 scale low,
// the f16 of -(1024 + zero) high. The group costs 3.1% more bytes than a per-channel scale and
// halves the KL divergence against the unquantised layer.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_GEMM_W8_MAX_M 64
#define R4D_GEMM_W8_KPB   32         // K per packed block: 2 k steps, 16 bytes per lane
#define WAVE32 32

typedef _Float16 v8h  __attribute__((ext_vector_type(8)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));

typedef __attribute__((address_space(1))) const uint4v*             r4d_gw4;
typedef __attribute__((address_space(1))) const unsigned long long* r4d_gu2;
typedef __attribute__((address_space(1))) const unsigned*           r4d_gu1;

__device__ __forceinline__ v8h fragA8(r4d_gu2 p, int u) {
  union { unsigned long long w[2]; v8h f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  return x.f;
}

// Two dwords of four bytes -> one 16x16x16 f16 B fragment, minus the zero point. The packer puts
// fragment elements 0..3 in the first dword and 4..7 in the second, so each v_perm_b32 pairs two
// bytes with the high byte 0x64 (the f16 exponent for 1024). nz holds -(1024 + zero) in both halves.
__device__ __forceinline__ v8h dequant8(unsigned w0, unsigned w1, __half2 nz) {
  const unsigned K64 = 0x64646464u;
  union { unsigned u[4]; __half2 h2[4]; v8h f; } x;
  x.u[0] = __builtin_amdgcn_perm(K64, w0, 0x05010400u);   // bytes 0, 1 -> elements 0, 1
  x.u[1] = __builtin_amdgcn_perm(K64, w0, 0x05030402u);   // bytes 2, 3 -> elements 2, 3
  x.u[2] = __builtin_amdgcn_perm(K64, w1, 0x05010400u);   // bytes 4, 5 -> elements 4, 5
  x.u[3] = __builtin_amdgcn_perm(K64, w1, 0x05030402u);   // bytes 6, 7 -> elements 6, 7
#pragma unroll
  for (int i = 0; i < 4; i++) x.h2[i] = __hadd2(x.h2[i], nz);
  return x.f;
}

__device__ __forceinline__ v8f mmah8(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

template <int MT, int NPW, bool NT>
__global__ __launch_bounds__(1024) void r4d_gemm_w8a16_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned* __restrict__ Wq,
    const unsigned* __restrict__ Wsz, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  extern __shared__ float red8[];       // WV*NPW*SK*256 floats: a 16x16 tile per (column, split)
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / R4D_GEMM_W8_GROUP;
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    unsigned aoff[MT];
#pragma unroll
    for (int i = 0; i < MT; i++)
      aoff[i] = rowGo((unsigned)K, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;
    r4d_gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (r4d_gu2)(const unsigned long long*)(A + aoff[i]);

    const int kpb = K / R4D_GEMM_W8_KPB;                  // packed blocks in a full row of tiles
    const int nsz = K / R4D_GEMM_W8_GROUP;                // (scale, zero) dwords per row of tiles
    r4d_gw4 pw[NPW];
    r4d_gu1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_gw4)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W8_KPB) * WAVE32 + lane;
      ps[j] = (r4d_gu1)(const unsigned*)(Wsz) +
              (size_t)(t * nsz + kbase / R4D_GEMM_W8_GROUP) * 16 + (lane & 15);
    }

    const int bpg = R4D_GEMM_W8_GROUP / R4D_GEMM_W8_KPB;  // packed blocks per group

    for (int g = 0; g < groups; g++) {
      float sc[NPW];
      __half2 nz[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const unsigned sz = ps[j][(size_t)g * 16];
        sc[j] = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
        nz[j] = __half2half2(__ushort_as_half((unsigned short)(sz >> 16)));
      }

      v8f g_acc[NPW][MT];
#pragma unroll
      for (int b = 0; b < bpg; b++) {
        uint4v wv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const size_t o = (size_t)(g * bpg + b) * WAVE32;
          wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        }
#pragma unroll
        for (int s = 0; s < 2; s++) {
          const int u = b * 2 + s;
          v8h afrag[MT];
#pragma unroll
          for (int i = 0; i < MT; i++) afrag[i] = fragA8(pa[i], u);
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const v8h bfrag = dequant8(wv[j][2 * s], wv[j][2 * s + 1], nz[j]);
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = u == 0 ? mmah8(afrag[i], bfrag, (v8f)(0.f))
                                   : mmah8(afrag[i], bfrag, g_acc[j][i]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++) acc[j][i][e] = fmaf(g_acc[j][i][e], sc[j], acc[j][i][e]);
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += (R4D_GEMM_W8_GROUP / 16) * 4;
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
      for (int e = 0; e < 8; e++) red8[slot + e * WAVE32 + lane] = acc[j][i][e];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red8[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)v;
    }
  }
}

void r4d_gemm_w8a16_nt_m64(long a, long wq, long wsz, long c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, long stream) {
  // Any M is correct (grid.y covers row tiles past MT; rows past M are clamped and masked). 64 is
  // the band this shape suits, since each extra row-tile group re-reads the weight; max_m()
  // reports it.
  if (M < 1) throw std::runtime_error("r4d gemm_w8a16_nt_m64: M must be >= 1");
  if (N % 16) throw std::runtime_error("r4d gemm_w8a16_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_W8_GROUP))
    throw std::runtime_error("r4d gemm_w8a16_nt_m64: K must be divisible by SK*128");
  if (WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_w8a16_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w8a16_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4)
    throw std::runtime_error("r4d gemm_w8a16_nt_m64: NPW must be 1, 2 or 4");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w8a16_nt_m64: WV*NPW*SK too large");
  auto Ap = (const unsigned short*)a;
  auto Wp = (const unsigned*)wq;
  auto Sp = (const unsigned*)wsz;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH8(MT, NP, TT) hipLaunchKernelGGL((r4d_gemm_w8a16_nt_m64_kernel<MT, NP, TT>), grid, \
                                               block, smem, st, Ap, Wp, Sp, Cp, M, K, N, WV, SK)
#define LAUNCH8_MB(NP, TT) switch (MB) { \
    case 1: LAUNCH8(1, NP, TT); break; \
    case 2: LAUNCH8(2, NP, TT); break; \
    case 3: LAUNCH8(3, NP, TT); break; \
    case 4: LAUNCH8(4, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a16_nt_m64: MB unsupported"); }
  if (NPW == 1) {
    if (NT) { LAUNCH8_MB(1, true); } else { LAUNCH8_MB(1, false); }
  } else if (NPW == 2) {
    if (NT) { LAUNCH8_MB(2, true); } else { LAUNCH8_MB(2, false); }
  } else {
    if (NT) { LAUNCH8_MB(4, true); } else { LAUNCH8_MB(4, false); }
  }
#undef LAUNCH8_MB
#undef LAUNCH8
}

int r4d_gemm_w8a16_nt_m64_max_m(void) { return R4D_GEMM_W8_MAX_M; }
int r4d_gemm_w8a16_nt_m64_group(void) { return R4D_GEMM_W8_GROUP; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
extern "C" int r4d_rad_gemm_w8a16_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    /* Any other activation dtype would read plausible but wrong numbers. */
    if (g.a->dtype != RAD_F16) return R4D_NO(RAD_E_DTYPE);
    if (g.M > r4d_gemm_w8a16_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    const int group = r4d_gemm_w8a16_nt_m64_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_FLAT, &c);
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_w8a16_nt_m64",
        r4d_gemm_w8a16_nt_m64(r4d_addr(g.a), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
                              (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw, c.nt,
                              (long)(uintptr_t)s));
    return RAD_OK;
}

