// r4d_gemm_w4a16_nt_m64.hip - r4d_gemm_w4a16_nt_m64: 4-bit-weight skinny GEMM for M up to 64,
// gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f16 A, bf16 C, fp32 accumulate.
//
// The decomposition of r4d_gemm_bf16_nt_m64 with the weight at 4 bits. M is a speculative decode
// batch, so the weight dominates the cost (at M=8 on the drafter's MLP: 82 KiB of activation
// against 44 MiB of weight).
//
// f16, not bf16: `0x6400 | n` is exactly the f16 for 1024 + n, so four v_perm_b32 turn a dword of
// eight nibbles into a 16x16x16 fragment, and the zero point is subtracted with v_pk_add_f16.
// gfx1201 has no packed bf16 arithmetic, so in bf16 that subtract would be sixteen instructions
// instead of four. The WMMA inputs are exact; the group scale is applied to the f32 accumulator.
//
// The cost is the slowest matrix instruction on this part: v_wmma_f32_16x16x16_f16 at 207 TF/s
// (iu8 and fp8 at 407-412, iu4 at 830). Below M=32 the kernel is at the DRAM roofline (620-648 GB/s)
// and the arithmetic hides; at M=64 r4d_gemm_w4a8_nt_m64 runs the same packed weight through the
// int8 form. This kernel owns the small-M band.
//
// The scale is not folded into the fragment as fma(1024 + n, s, -(1024 + z) * s): the addend is
// not representable in f16 and cancels ~1024*s against a result of ~4*s, an error of ~12%. The
// subtraction must happen on the integer grid and the scale after the cancellation.
//
// A fragment covers sixteen rows whatever M is, and every column tile re-reads A: one column tile
// per wave moves four times as many activation bytes (from cache) as weight bytes. NPW (column
// tiles per wave) reuses one activation fragment on NPW weight fragments, with no LDS or barrier,
// dropping the ratio to 4*MT/NPW. WV alone does not help: a block's waves share only through cache.
//
// Layout: Wq is pre-permuted offline into fragment order, so a lane's 16 bytes are contiguous and a
// wave's weight load for four k steps is one 512-byte global_load_b128, with no LDS and no mask:
//
//     idx = lane%16 , k = 8*(e>>2) + 4*(lane>>4) + (e&3)      (r4d_gdn_wmma.h)
//
// fragment element e of lane l is W[n0 + (l&15)][16s + 8*(e>>2) + 4*(l>>4) + (e&3)], stored in
// nibble 2e for e<4 and nibble 2(e-4)+1 for e>=4, the order the unpack below produces. The packer
// in radiance_w4.py builds this layout.
//
// Quantiser: per output channel, per group of R4D_GEMM_W4_GROUP contiguous K: w ~= scale * (q - zero)
// with q in 0..15 and an integer zero, which keeps the subtract exact. One dword per (row, group):
// the f16 scale low and the f16 of -(1024 + zero) high, broadcast into a v_pk_add_f16.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_GEMM_W4_MAX_M 64
#define R4D_GEMM_W4_KPB   64         // K per packed block: 4 k steps, 16 bytes per lane
#define WAVE32 32

typedef _Float16 v8h  __attribute__((ext_vector_type(8)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));

// NT reads the weight non-temporally so the re-read activation keeps the last level. It wins at
// M=8 and M=16 and loses from M=24 up on the longest-K shapes, so it is per call; radiance_w4.py
// carries the value per M band. A benchmark must use a pooled working set: one weight fits the
// 64 MiB last level, so looping over a single buffer hides the effect.

// NPW is a template parameter (it sizes the accumulators) chosen per M: at M=8 NPW>1 only costs
// blocks; at M=64 NPW=4 recovers the activation re-read. Only 1 and 4 are instantiated; anything
// else is rejected.

// Address-space-qualified so loads stay global_load, not flat_load (on gfx12 a flat load also
// increments dscnt). See r4d_gemm_bf16_nt_m64.hip for why the element type is a scalar.
typedef __attribute__((address_space(1))) const uint4v*          r4d_gw4;
typedef __attribute__((address_space(1))) const unsigned long long* r4d_gu2;
typedef __attribute__((address_space(1))) const unsigned*        r4d_gu1;

__device__ __forceinline__ v8h fragA(r4d_gu2 p, int u) {
  union { unsigned long long w[2]; v8h f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  return x.f;
}

// One dword of eight nibbles -> one 16x16x16 f16 B fragment, minus the zero point. t0/t1 split the
// even and odd nibbles into bytes; each v_perm_b32 pairs two of them with the high byte 0x64 (the
// f16 exponent for 1024). nz holds -(1024 + zero) in both halves.

// Do not fold the group scale into the B fragment. A v_pk_mul_f16 on the integer q is exact and
// saves registers (192 -> 135 VGPRs at M=64), but it is slower on most shapes: it lengthens the
// dependent chain between unpack and WMMA, and this kernel is stall-bound, not register-bound.

__device__ __forceinline__ v8h dequant(unsigned w, __half2 nz) {
  const unsigned K64 = 0x64646464u;
  // The stored nibble is two's complement, which r4d_gemm_w4a8_nt_m64 reads as a signed high
  // nibble. This kernel needs offset binary (1024 + n minus 1024 + 8); the two differ in bit 3, so
  // one XOR per dword converts and the zero point is 8 for every group. The conversion lives here,
  // where the unpack is negligible, rather than in the int8 kernel.
  const unsigned t0 = (w & 0x0F0F0F0Fu) ^ 0x08080808u;
  const unsigned t1 = ((w >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
  union { unsigned u[4]; __half2 h2[4]; v8h f; } x;
  x.u[0] = __builtin_amdgcn_perm(K64, t0, 0x05010400u);   // nibbles 0, 2
  x.u[1] = __builtin_amdgcn_perm(K64, t0, 0x05030402u);   // nibbles 4, 6
  x.u[2] = __builtin_amdgcn_perm(K64, t1, 0x05010400u);   // nibbles 1, 3
  x.u[3] = __builtin_amdgcn_perm(K64, t1, 0x05030402u);   // nibbles 5, 7
#pragma unroll
  for (int i = 0; i < 4; i++) x.h2[i] = __hadd2(x.h2[i], nz);
  return x.f;
}

__device__ __forceinline__ v8f mmah(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// MT = row tiles per block, as in the bf16 kernel.
template <int MT, int NPW, bool NT>
__global__ __launch_bounds__(1024) void r4d_gemm_w4a16_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned* __restrict__ Wq,
    const unsigned* __restrict__ Wsz, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  extern __shared__ float red[];        // WV*NPW*SK*256 floats: a 16x16 tile per (column, split)
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / R4D_GEMM_W4_GROUP;              // groups this split owns
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    // Activation rows are clamped, not masked; results past M are never stored.
    unsigned aoff[MT];
#pragma unroll
    for (int i = 0; i < MT; i++)
      aoff[i] = rowGo((unsigned)K, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;
    r4d_gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (r4d_gu2)(const unsigned long long*)(A + aoff[i]);

    // Weight: one uint4 per lane per four k steps, already permuted, so the k loop is
    // global_load_b128 at a constant stride.
    const int kpb = K / R4D_GEMM_W4_KPB;                  // packed blocks in a full row of tiles
    const int nsz = K / R4D_GEMM_W4_GROUP;                // (scale, zero) dwords per row of tiles
    r4d_gw4 pw[NPW];
    r4d_gu1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      // Column tiles past N are clamped to the last real tile; the store discards them.
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_gw4)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W4_KPB) * WAVE32 + lane;
      // Quantizer parameters: 16 distinct dwords per group, read by the 16 lanes that own the rows.
      ps[j] = (r4d_gu1)(const unsigned*)(Wsz) +
              (size_t)(t * nsz + kbase / R4D_GEMM_W4_GROUP) * 16 + (lane & 15);
    }

    const int bpg = R4D_GEMM_W4_GROUP / R4D_GEMM_W4_KPB;  // packed blocks per group

    for (int g = 0; g < groups; g++) {
      float sc[NPW];
      __half2 nz[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const unsigned sz = ps[j][(size_t)g * 16];
        sc[j] = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
        nz[j] = __half2half2(__ushort_as_half((unsigned short)(sz >> 16)));
      }

      // The first WMMA of a group takes an inline zero for C, so no clear is needed.
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
        for (int s = 0; s < 4; s++) {
          const int u = b * 4 + s;
          v8h afrag[MT];
#pragma unroll
          for (int i = 0; i < MT; i++) afrag[i] = fragA(pa[i], u);   // loaded once, spent NPW times
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const v8h bfrag = dequant(wv[j][s], nz[j]);
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = u == 0 ? mmah(afrag[i], bfrag, (v8f)(0.f))
                                   : mmah(afrag[i], bfrag, g_acc[j][i]);
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
      for (int i = 0; i < MT; i++) pa[i] += (R4D_GEMM_W4_GROUP / 16) * 4;
    }
  }

  // Split-K reduction and store as in r4d_gemm_bf16_nt_m64: one row tile at a time, fixed split
  // order.
  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;
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
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)v;
    }
  }
}

void r4d_gemm_w4a16_nt_m64(long a, long wq, long wsz, long c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, long stream) {
  if (M < 1 || M > R4D_GEMM_W4_MAX_M)
    throw std::runtime_error("r4d gemm_w4a16_nt_m64: M must be 1..64");
  if (N % 16) throw std::runtime_error("r4d gemm_w4a16_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_W4_GROUP))
    throw std::runtime_error("r4d gemm_w4a16_nt_m64: K must be divisible by SK*128");
  if (WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_w4a16_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w4a16_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 4) throw std::runtime_error("r4d gemm_w4a16_nt_m64: NPW must be 1 or 4");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w4a16_nt_m64: WV*NPW*SK too large");
  auto Ap = (const unsigned short*)a;
  auto Wp = (const unsigned*)wq;
  auto Sp = (const unsigned*)wsz;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT, NP, TT) hipLaunchKernelGGL((r4d_gemm_w4a16_nt_m64_kernel<MT, NP, TT>), grid, \
                                              block, smem, st, Ap, Wp, Sp, Cp, M, K, N, WV, SK)
#define LAUNCH_MB(NP, TT) switch (MB) { \
    case 1: LAUNCH(1, NP, TT); break; \
    case 2: LAUNCH(2, NP, TT); break; \
    case 3: LAUNCH(3, NP, TT); break; \
    case 4: LAUNCH(4, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w4a16_nt_m64: MB unsupported"); }
  if (NPW == 1) {
    if (NT) { LAUNCH_MB(1, true); } else { LAUNCH_MB(1, false); }
  } else {
    if (NT) { LAUNCH_MB(4, true); } else { LAUNCH_MB(4, false); }
  }
#undef LAUNCH_MB
#undef LAUNCH
}

int r4d_gemm_w4a16_nt_m64_max_m(void) { return R4D_GEMM_W4_MAX_M; }
int r4d_gemm_w4a16_nt_m64_group(void) { return R4D_GEMM_W4_GROUP; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
/* ------------------------------------------------------------------ f16-activation grids */

extern "C" int r4d_rad_gemm_w4a16_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    /* Any other activation dtype would read plausible but wrong numbers. */
    if (g.a->dtype != RAD_F16) return R4D_NO(RAD_E_DTYPE);
    if (g.M > r4d_gemm_w4a16_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    const int group = r4d_gemm_w4a16_nt_m64_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_FLAT, &c);
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_w4a16_nt_m64",
        r4d_gemm_w4a16_nt_m64(r4d_addr(g.a), r4d_addr(g.b), r4d_addr(g.bscale), r4d_addr(g.y),
                              (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw, c.nt,
                              (long)(uintptr_t)s));
    return RAD_OK;
}

