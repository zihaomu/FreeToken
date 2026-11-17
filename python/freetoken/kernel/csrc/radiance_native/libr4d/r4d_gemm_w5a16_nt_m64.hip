// r4d_gemm_w5a16_nt_m64.hip - r4d_gemm_w5a16_nt_m64: 5-bit-weight skinny GEMM for M up to 64,
// gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f16 A, bf16 C, fp32 accumulate.
//
// r4d_gemm_w4a16_nt_m64 with a fifth bit per weight, for ParoQuant's int5 (asymmetric: an f16 scale
// and a 5-bit integer zero per 128 of K, with the rotation applied to the activation beforehand).
// That kernel's header applies; this one covers only what the fifth bit changes.
//
// The weight blob is two planes. The low four bits of each code are in w4's fragment order (offset
// binary: the nibble is q & 15, no sign flip). The fifth bits follow as 32-bit words, one per
// (n-tile, 64-wide k block, lane):
//
//     hi[(t * (K/64) + kb) * 32 + lane]     byte s = k step s of the block, bit e = element e
//
// where element e is the one nibble 2e / 2(e-4)+1 of dword s holds. A lane's 16 bytes of nibbles and
// 4 bytes of fifth bits cover the same 32 weights; the second load is one dword per block.
//
// The unpack adds two multiplies to w4's. The even elements' fifth bits are the byte's low nibble,
// the odd elements' its high one. n * 0x00204081 moves bit j of a 4-bit mask to bit 8j (no other
// term lands on a multiple of eight below bit 32); masking with 0x01010101 and shifting by 4 puts
// it at the 16s place. The code is then 0..31, 1024 + q is exact in f16, and the integer zero
// (0..31, stored as -(1024 + zero)) subtracts exactly.
//
// The dispatcher runs M past 64 as chunks of 64 rows, re-reading the weight per chunk (near the f16
// compute bound at a 2048-row prefill chunk anyway), and partitioned weights (ParoQuant's gate|up:
// `pb1`/`pb2`, see r4d_gemm_args.h) as one launch per part.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#define R4D_GEMM_W4_MAX_M 64
#define R4D_GEMM_W4_KPB   64         // K per packed block: 4 k steps, 16 bytes per lane
#define WAVE32 32

typedef _Float16 v8h  __attribute__((ext_vector_type(8)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));

// NT and NPW: see r4d_gemm_w4a16_nt_m64.hip. Only NPW 1 and 4 are instantiated.

// Address-space-qualified so loads stay global_load, not flat_load (on gfx12 a flat load also
// increments dscnt). See r4d_gemm_bf16_nt_m64.hip for why the element type is a scalar.
typedef __attribute__((address_space(1))) const uint4v*          r4d_gw4;
typedef __attribute__((address_space(1))) const unsigned long long* r4d_gu2;
typedef __attribute__((address_space(1))) const unsigned*        r4d_gu1;

__device__ __forceinline__ v8h fragA(r4d_gu2 p, int u, int bf) {
  union { unsigned long long w[2]; uint16_t h[8]; v8h f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  if (!bf) return x.f;
  // Same narrowing as narrow_f16_kernel: bf16 -> f32 -> f16, nearest even.
  v8h f;
#pragma unroll
  for (int i = 0; i < 8; i++)
    f[i] = (_Float16)__builtin_bit_cast(float, (uint32_t)x.h[i] << 16);
  return f;
}

// Do not fold the group scale into the B fragment; see r4d_gemm_w4a16_nt_m64.hip.

// One dword of eight nibbles plus the byte of their fifth bits -> one 16x16x16 f16 B fragment,
// minus the zero point. Bits 0..3 of `hb` are the even nibbles' fifth bits, bits 4..7 the odd
// nibbles'. Each v_perm_b32 pairs two code bytes with the high byte 0x64 (1024 + code); nz holds
// -(1024 + zero) in both halves. Codes are offset binary (0..31, or 0..15 without the fifth bit).
__device__ __forceinline__ unsigned spread16(unsigned n) {      // 4 bits -> bit 4 of each byte
  return ((n * 0x00204081u) & 0x01010101u) << 4;
}
__device__ __forceinline__ v8h dequant5(unsigned w, unsigned hb, __half2 nz) {
  const unsigned K64 = 0x64646464u;
  const unsigned t0 = (w & 0x0F0F0F0Fu) | spread16(hb & 15u);
  const unsigned t1 = ((w >> 4) & 0x0F0F0F0Fu) | spread16(hb >> 4);
  union { unsigned u[4]; __half2 h2[4]; v8h f; } x;
  x.u[0] = __builtin_amdgcn_perm(K64, t0, 0x05010400u);   // elements 0, 1
  x.u[1] = __builtin_amdgcn_perm(K64, t0, 0x05030402u);   // elements 2, 3
  x.u[2] = __builtin_amdgcn_perm(K64, t1, 0x05010400u);   // elements 4, 5
  x.u[3] = __builtin_amdgcn_perm(K64, t1, 0x05030402u);   // elements 6, 7
#pragma unroll
  for (int i = 0; i < 4; i++) x.h2[i] = __hadd2(x.h2[i], nz);
  return x.f;
}

__device__ __forceinline__ v8f mmah(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// MT = row tiles per block, as in the bf16 kernel. HI false is the 4-bit asymmetric grid (z-lab's
// plain int4), with no fifth-bit plane.
template <int MT, int NPW, bool NT, bool HI, bool ONE = false>
__global__ __launch_bounds__(1024) void r4d_gemm_w5a16_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned* __restrict__ Wq,
    const unsigned* __restrict__ Whi, const unsigned* __restrict__ Wsz,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int lda, int ldc,
    int bf) {
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
      aoff[i] = rowGo((unsigned)lda, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;
    r4d_gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (r4d_gu2)(const unsigned long long*)(A + aoff[i]);

    // Weight: one uint4 per lane per four k steps, already permuted, so the k loop is
    // global_load_b128 at a constant stride.
    const int kpb = K / R4D_GEMM_W4_KPB;                  // packed blocks in a full row of tiles
    const int nsz = K / R4D_GEMM_W4_GROUP;                // (scale, zero) dwords per row of tiles
    r4d_gw4 pw[NPW];
    r4d_gu1 ph[NPW];
    r4d_gu1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      // Column tiles past N are clamped to the last real tile; the store discards them.
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_gw4)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W4_KPB) * WAVE32 + lane;
      // Fifth-bit plane: one dword per lane per packed block (unused when !HI).
      ph[j] = (r4d_gu1)(const unsigned*)(HI ? Whi : Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W4_KPB) * WAVE32 + lane;
      // Quantiser parameters: 16 dwords per group, read by the 16 lanes that own the rows.
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
        unsigned hv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const size_t o = (size_t)(g * bpg + b) * WAVE32;
          wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
          hv[j] = HI ? (NT ? __builtin_nontemporal_load(&ph[j][o]) : ph[j][o]) : 0u;
        }
#pragma unroll
        for (int s = 0; s < 4; s++) {
          const int u = b * 4 + s;
          v8h afrag[MT];
#pragma unroll
          for (int i = 0; i < MT; i++) afrag[i] = fragA(pa[i], u, bf);   // loaded once, spent NPW times
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const v8h bfrag = dequant5(wv[j][s], (hv[j] >> (8 * s)) & 0xFFu, nz[j]);
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
          for (int e = 0; e < (ONE ? 1 : 8); e++)
          acc[j][i][e] = fmaf(g_acc[j][i][e], sc[j], acc[j][i][e]);
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += (R4D_GEMM_W4_GROUP / 16) * 4;
    }
  }

  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;
  if constexpr (ONE) {
    // M=1: only element 0 of lanes 0..15 belongs to the live row. Keep the same WMMA,
    // per-group FMA and split order, but do not scale/reduce the other fifteen rows.
    static_assert(MT == 1, "the one-row epilogue needs one row tile");
    if (SK == 1) {
      if (lane < 16) {
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const int n = (nt0 + j) * 16 + lane;
          if (n < N) C[n] = (__hip_bfloat16)(0.f + acc[j][0][0]);
        }
      }
      return;
    }
    if (lane < 16) {
#pragma unroll
      for (int j = 0; j < NPW; j++)
        red[((col_local * NPW + j) * SK + ksplit) * 16 + lane] = acc[j][0][0];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 16; idx += nthreads) {
      const int col = idx / 16, n = (blockIdx.x * ncols + col) * 16 + idx % 16;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 16 + idx % 16];
      if (n < N) C[n] = (__hip_bfloat16)v;
    }
    return;
  }

  // Split-K reduction and store as in r4d_gemm_bf16_nt_m64: one row tile at a time, fixed split
  // order.
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
      if (n < N && m < M) C[(size_t)m * ldc + n] = (__hip_bfloat16)v;
    }
  }
}

void r4d_gemm_w5a16_nt_m64(long a, long wq, long whi, long wsz, long c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, long stream, int lda, int ldc,
                           bool hi5, bool bf) {
  if (!lda) lda = K;
  if (!ldc) ldc = N;
  if (M < 1 || M > R4D_GEMM_W4_MAX_M)
    throw std::runtime_error("r4d gemm_w5a16_nt_m64: M must be 1..64");
  if (N % 16) throw std::runtime_error("r4d gemm_w5a16_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_W4_GROUP))
    throw std::runtime_error("r4d gemm_w5a16_nt_m64: K must be divisible by SK*128");
  if (WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_w5a16_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w5a16_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 4) throw std::runtime_error("r4d gemm_w5a16_nt_m64: NPW must be 1 or 4");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  // Keep the same accepted tuning space as the multi-row form. The single-row specialization
  // uses 1/16 of its reduction storage (none without a split).
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w5a16_nt_m64: WV*NPW*SK too large");
  const bool one = M == 1 && MB == 1;
  if (one) smem = SK == 1 ? 0 : smem / 16;
  auto Ap = (const unsigned short*)a;
  auto Wp = (const unsigned*)wq;
  auto Hp = (const unsigned*)whi;
  auto Sp = (const unsigned*)wsz;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT, NP, TT) do { \
    if (hi5) hipLaunchKernelGGL((r4d_gemm_w5a16_nt_m64_kernel<MT, NP, TT, true>), grid, block, smem, \
                                st, Ap, Wp, Hp, Sp, Cp, M, K, N, WV, SK, lda, ldc, bf); \
    else     hipLaunchKernelGGL((r4d_gemm_w5a16_nt_m64_kernel<MT, NP, TT, false>), grid, block, smem, \
                                st, Ap, Wp, Hp, Sp, Cp, M, K, N, WV, SK, lda, ldc, bf); } while (0)
#define LAUNCH_ONE(NP, TT) do { \
    if (hi5) hipLaunchKernelGGL((r4d_gemm_w5a16_nt_m64_kernel<1, NP, TT, true, true>), grid, block, smem, \
                                st, Ap, Wp, Hp, Sp, Cp, M, K, N, WV, SK, lda, ldc, bf); \
    else     hipLaunchKernelGGL((r4d_gemm_w5a16_nt_m64_kernel<1, NP, TT, false, true>), grid, block, smem, \
                                st, Ap, Wp, Hp, Sp, Cp, M, K, N, WV, SK, lda, ldc, bf); } while (0)
#define LAUNCH_MB(NP, TT) switch (MB) { \
    case 1: if (one) { LAUNCH_ONE(NP, TT); } else { LAUNCH(1, NP, TT); } break; \
    case 2: LAUNCH(2, NP, TT); break; \
    case 3: LAUNCH(3, NP, TT); break; \
    case 4: LAUNCH(4, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w5a16_nt_m64: MB unsupported"); }
  if (NPW == 1) {
    if (NT) { LAUNCH_MB(1, true); } else { LAUNCH_MB(1, false); }
  } else {
    if (NT) { LAUNCH_MB(4, true); } else { LAUNCH_MB(4, false); }
  }
#undef LAUNCH_MB
#undef LAUNCH_ONE
#undef LAUNCH
}

int r4d_gemm_w5a16_nt_m64_max_m(void) { return R4D_GEMM_W4_MAX_M; }
int r4d_gemm_w5a16_nt_m64_group(void) { return R4D_GEMM_W4_GROUP; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
/* ------------------------------------------------------------------ f16-activation grids */

static int w5_dispatch(const RadArgs* a, RadStream s, bool hi5) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    /* BF16 is narrowed on load with the cast kernel's rounding, so a separate cast is not required. */
    const bool bf = g.a->dtype == RAD_BF16;
    if (g.a->dtype != RAD_F16 && !bf) return R4D_NO(RAD_E_DTYPE);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    if (g.K % R4D_GEMM_W4_GROUP) return R4D_NO(RAD_E_SHAPE);
    const int group = r4d_gemm_w5a16_nt_m64_group();
    // Rows past 64 run as chunks of 64, each a pass over the weight.
    const long long mc = std::min<long long>(g.M, r4d_gemm_w5a16_nt_m64_max_m());
    Cfg c;
    rc = cfg_for(a, g.K, mc, group, BR_FLAT, &c);
    if (rc != RAD_OK) return rc;
    /* cfg_for pins MB = 1 in the decode band, but here the dequantise is the cost: with one block per
     * row tile, each weight fragment is unpacked once per tile. Cover every row tile in one block
     * unless the tuner pinned MB; the split is unchanged, so the output bits are too. */
    R4dGemmCfg tuned;
    r4d_cfg_gemm(a, &tuned);
    if (tuned.mb <= 0) c.mb = std::min<long long>(4, (mc + 15) / 16);
    const long long Ntot = g.N, K = g.K, lda = mp.lda ? mp.lda : g.K;
    const long long nsz = K / R4D_GEMM_W4_GROUP;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        for (long long m0 = 0; m0 < g.M; m0 += mc) {
            const long long mm = std::min<long long>(mc, g.M - m0);
            R4D_TRY("gemm_w5a16_nt_m64",
                r4d_gemm_w5a16_nt_m64(
                    r4d_addr(g.a) + (m0 * lda + p * K) * 2,
                    r4d_addr(g.b) + n0 * K / 2,                              /* the nibble plane */
                    hi5 ? r4d_addr(g.b) + Ntot * K / 2 + n0 * K / 8 : 0L,    /* the fifth-bit plane */
                    r4d_addr(g.bscale) + (n0 / 16) * nsz * 16 * 4,
                    r4d_addr(g.y) + (m0 * Ntot + n0) * 2,
                    (int)mm, (int)K, (int)np, c.wv, c.sk, c.mb, c.npw, c.nt, (long)(uintptr_t)s,
                    (int)lda, (int)Ntot, hi5, bf));
        }
    }
    return RAD_OK;
}

extern "C" int r4d_rad_gemm_w5a16_m64(const RadArgs* a, RadStream s) { return w5_dispatch(a, s, true); }
/* ...and the plain 4-bit asymmetric grid (z-lab's int4), whose blob is the nibble plane alone. */
extern "C" int r4d_rad_gemm_w4a16asym_m64(const RadArgs* a, RadStream s) { return w5_dispatch(a, s, false); }
