// r4d_gemm_w8a8_tiled.hip - r4d_gemm_w8a8_tiled: the W8A8 GEMM for prefill batches.
//
// r4d_gemm_w4a8_tiled with the weight radix doubled: the inner loop, double buffering, XOR swizzle
// and magic-bias epilogue are that kernel's, and staging copies the weight's bytes to LDS in
// `[tile][lane*8 + u*2]` order instead of unpacking nibbles.
//
// At prefill the promotion to eight bits is nearly free: the kernel is matrix-bound and the WMMA
// count does not depend on the radix, while twice the weight and LDS bytes are amortised over BM
// rows. (A skinny kernel would re-read the weight M/(16*MB) times.)
//
// The packed layout is r4d_gemm_w8a8_nt_m64's: a lane owns 16 bytes of a 32-column block, staged
// with coalesced global_load_b128 to `lane*8 + kh*4` dwords of the tile's LDS run. One weight feeds
// both kernels. The scale plane is f16 only: a symmetric 8-bit grid has no zero point.
//
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_TILED8_KPB 32   // packed block: 32 columns, 16 bytes a lane
// BK = 32, not the 4-bit kernel's 64: at BK = 64 the 64x256 tile needs 40960 B of LDS (one
// workgroup per CU); BK = 32 needs 20480 B (three), worth several percent on down_proj at every
// prefill M despite twice the barriers per scale group.
#define R4D_TILED8_BK  32

typedef int      v2i8_t __attribute__((ext_vector_type(2)));
typedef int      v8i8_t __attribute__((ext_vector_type(8)));
typedef float    v8f8_t __attribute__((ext_vector_type(8)));
typedef unsigned u4v8_t __attribute__((ext_vector_type(4)));

// WM x WN waves, each owning TM x TN WMMA tiles. BM = WM*TM*16, BN = WN*TN*16. UNR k blocks are
// unrolled per scale group. The accumulator carries the magic bias of r4d_gemm_w4a8_tiled.hip:
// for |x| < 2^22, float(x) == __uint_as_float(0x4B400000 + x) - 12582912.0f.
//
// `Asum` is always null (an 8-bit grid here is symmetric). The slot keeps the 4-bit kernel's
// argument layout: dropping it changes the generated code of the tuned tiles.
template <int WM, int WN, int TM, int TN, int UNR>
__global__ __launch_bounds__(WM * WN * 32, 8) void r4d_gemm_w8a8_tiled_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    const float* __restrict__ Asum, __hip_bfloat16* __restrict__ C, int M, int K, int N) {
  constexpr int BK = R4D_TILED8_BK;
  constexpr int GRP = R4D_GEMM_W8A8_GROUP;
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int NTILE = BN / 16;
  constexpr int AW = BM * BK / 4;                   // A dwords per buffer
  constexpr int WW = NTILE * BK * 4;                // W dwords per buffer: NTILE*16 rows x BK bytes

  __shared__ unsigned lds_a[2][AW];
  __shared__ unsigned lds_w[2][WW];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int bx = blockIdx.x;
  const int m0 = blockIdx.y * BM, n0 = bx * BN;
  const int kpb = K / R4D_TILED8_KPB, nsz = K / GRP;
  constexpr int NT_ = WM * WN * 32;

  v8f8_t acc[TN][TM];
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++) acc[j][i] = (v8f8_t)(0.f);
  v8i8_t g[TN][TM];
  float csum[TN];
#pragma unroll
  for (int j = 0; j < TN; j++) csum[j] = 0.f;
  unsigned scv[TN];

  // XOR swizzle of the A rows instead of padding, separating rows r >> 1 (see
  // r4d_gemm_w4a8_tiled.hip). The XOR must stay inside a row of BK/4 = 8 dwords, so the mask is
  // derived from BK and steps by 2 for ds_load_b64 alignment.
#define ASW_M2 ((BK / 8) - 1)
#define ASW(r, c) ((r) * (BK / 4) + ((c) ^ ((((r) >> 1) & ASW_M2) << 1)))

  // Prefetch into registers, compute off the other buffer, then commit to LDS, so the load wait
  // lands after the WMMA work (a prefetch that writes LDS directly would stall the compute; >2x).
  // Both stagings may be ragged (a predicate on the last pass), so the tile need not divide the
  // workgroup width (e.g. BM = 80, 112; at BK = 32 the NTILE*32 weight chunks).
  constexpr int APT = (AW + NT_ - 1) / NT_;
  constexpr bool APAD = (AW % NT_) != 0;
  unsigned ra[APT];
  constexpr int WWP = NTILE * BK;                   // 16-byte weight chunks per buffer
  constexpr int WPTU = (WWP + NT_ - 1) / NT_;
  constexpr bool WPAD = (WWP % NT_) != 0;
  u4v8_t rwu[WPTU];
  // Global indices into A are 32-bit: a 64-bit address costs four extra instructions per load.
  auto issue = [&](int kk) {
    // Sixteen bytes per thread, copied unchanged. A lane owns 16 bytes of a 32-column packed
    // block; at BK = 64 `kh` would pick one of two blocks 512 bytes apart. Coalesced across the
    // wave.
#pragma unroll
    for (int x = 0; x < WPTU; x++) {
      const int t = tid + x * NT_;
      if (WPAD && t >= WWP) continue;
      const int tl = t / BK, rem = t % BK, kh = rem / 32, ln = rem % 32;
      rwu[x] = *((const u4v8_t*)Wq +
                 (size_t)(((n0 / 16 + tl) * kpb + kk / R4D_TILED8_KPB + kh) * 32 + ln));
    }
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t / (BK / 4), c = t % (BK / 4);
      if (APAD && t >= AW) continue;
      const int mm = m0 + r < M ? m0 + r : M - 1;
      ra[x] = *(const unsigned*)(A + (unsigned)(mm * K + kk + c * 4));
    }
  };
  auto commit = [&](int buf) {
#pragma unroll
    for (int x = 0; x < WPTU; x++) {
      const int t = tid + x * NT_;
      if (WPAD && t >= WWP) continue;
      const int tl = t / BK, rem = t % BK, kh = rem / 32, ln = rem % 32;
      // [tile][lane][u][2] dwords: a lane's 16 bytes are k steps 2*kh and 2*kh+1, four
      // consecutive dwords at `ln*8 + kh*4`, one ds_store_b128.
      *(u4v8_t*)(lds_w[buf] + (size_t)tl * (BK * 4) + ln * (BK / 8) + kh * 4) = rwu[x];
    }
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t / (BK / 4), c = t % (BK / 4);
      if (APAD && t >= AW) continue;
      lds_a[buf][ASW(r, c)] = ra[x];
    }
  };

  issue(0);
  commit(0);
  __syncthreads();

  // Unroll the K loop over the group (UNR = GRP/BK) so the group-boundary tests are compile-time per
  // copy; a dynamic branch around the epilogue stops the scheduler interleaving it with the WMMA.
  // Worth several percent.
  for (int kg = 0, it = 0; kg < K; kg += BK * UNR) {
#pragma unroll
    for (int h = 0; h < UNR; h++, it++) {
    const int k0 = kg + h * BK;
    const int buf = it & 1;
    if (k0 + BK < K) issue(k0 + BK);                // loads in flight across the WMMA work below

    const bool first_half = (k0 % GRP) == 0;
    // Fetch the group scale at the start of the group, so the waves do not all stall on it at the
    // fold; keep the raw bits and convert at the group close. 32-bit index: the table has at most
    // (N/16)*(K/GRP)*16 elements.
    if ((k0 % GRP) == 0) {
#pragma unroll
      for (int j = 0; j < TN; j++)
        scv[j] = (unsigned)((const unsigned short*)Ws)[(unsigned)(((n0 / 16 + wn * TN + j) * nsz +
                                                                   k0 / GRP) * 16 + (lane & 15))];
    }
#pragma unroll
    for (int u = 0; u < BK / 16; u++) {
      v2i8_t af[TM];
#pragma unroll
      for (int i = 0; i < TM; i++) {
        const int row = (wm * TM + i) * 16 + (lane & 15);
        const unsigned* pa = lds_a[buf] + ASW(row, u * 4 + 2 * (lane >> 4));
        af[i][0] = (int)pa[0];
        af[i][1] = (int)pa[1];
      }
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const v2i8_t bf = *(const v2i8_t*)(lds_w[buf] + (size_t)(wn * TN + j) * (BK * 4)
                                           + lane * (BK / 8) + u * 2);
#pragma unroll
        for (int i = 0; i < TM; i++)
          g[j][i] = (first_half && u == 0)
                        ? __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                              true, af[i], true, bf, (v8i8_t)(0x4B400000), false)
                        : __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                              true, af[i], true, bf, g[j][i], false);
      }
    }

    if ((k0 + BK) % GRP == 0) {                     // the group closed: fold it into the float acc
      float sc[TN];
#pragma unroll
      for (int j = 0; j < TN; j++)
        sc[j] = __half2float(__ushort_as_half((unsigned short)(scv[j] & 0xFFFFu)));
#pragma unroll
      for (int j = 0; j < TN; j++) csum[j] += sc[j];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++)
#pragma unroll
          for (int e = 0; e < 8; e++)
            acc[j][i][e] = fmaf(__uint_as_float((unsigned)g[j][i][e]), sc[j], acc[j][i][e]);
    }
    if (k0 + BK < K) commit(buf ^ 1);               // the wait for those loads lands here
    __syncthreads();
    }
  }
#undef ASW
#undef ASW_M2

#pragma unroll
  for (int j = 0; j < TN; j++) {
    const float d = -12582912.0f * csum[j];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[j][i][e] += d;
  }

  // The flat predicated store, as in the 4-bit kernel.
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
        const int m = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4) + e;
        if (n < N && m < M)
          // No 1/16 here, unlike the 4-bit kernel: an 8-bit code is the byte itself, with no
          // high-nibble factor to undo.
          C[(size_t)m * N + n] = (__hip_bfloat16)(acc[j][i][e] * Ascale[m]);
      }
}

template <int WM, int WN, int TM, int TN, int UNR>
static void r4d_tiled8_launch(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                              long stream) {
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  // An 8-bit weight doubles the staged bytes.
  static_assert(2 * (BM * R4D_TILED8_BK / 4 + (BN / 16) * R4D_TILED8_BK * 4) * sizeof(unsigned) <=
                    64 * 1024, "this tile needs more than 64 KB of LDS");
  if (N % BN)
    throw std::runtime_error("r4d gemm_w8a8_tiled: N is not a multiple of this variant's block");
  dim3 grid(N / BN, (M + BM - 1) / BM), block(WM * WN * 32);
  hipLaunchKernelGGL((r4d_gemm_w8a8_tiled_kernel<WM, WN, TM, TN, UNR>), grid, block, 0,
                     (hipStream_t)stream, (const signed char*)a, (const float*)ascale,
                     (const unsigned*)wq, (const unsigned*)ws, nullptr, (__hip_bfloat16*)c, M, K, N);
}

void r4d_gemm_w8a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                         long stream, int variant) {
  if (N % 256) throw std::runtime_error("r4d gemm_w8a8_tiled: N must be a multiple of 256");
  if (K % R4D_GEMM_W8A8_GROUP)
    throw std::runtime_error("r4d gemm_w8a8_tiled: K must be a multiple of the scale group");
  if (M < 1) throw std::runtime_error("r4d gemm_w8a8_tiled: M must be positive");
  // The unroll covers exactly one scale group (GRP/BK steps); see r4d_gemm_w4a8_tiled.hip. The K
  // check above makes every K a whole number of them.
  static_assert(R4D_GEMM_W8A8_GROUP % R4D_TILED8_BK == 0, "the scale group must be whole k steps");
  constexpr int UNRG = R4D_GEMM_W8A8_GROUP / R4D_TILED8_BK;
#define R4D_TILED8_LAUNCH(...) \
  r4d_tiled8_launch<__VA_ARGS__>(a, ascale, wq, ws, c, M, K, N, stream)
  // The arms tile_bmbn in r4d_rows.cpp lists, with the (BM, BN) it records for each.
  switch (variant) {
    case 6:   R4D_TILED8_LAUNCH(2, 8, 2, 2, UNRG); break;   // BM 64,  BN 256
    case 10:  R4D_TILED8_LAUNCH(2, 8, 3, 2, UNRG); break;   // BM 96,  BN 256
    case 12:  R4D_TILED8_LAUNCH(2, 8, 3, 1, UNRG); break;   // BM 96,  BN 128
    case 57:  R4D_TILED8_LAUNCH(2, 8, 5, 1, UNRG); break;   // BM 160, BN 128
    case 59:  R4D_TILED8_LAUNCH(2, 8, 4, 1, UNRG); break;   // BM 128, BN 128
    case 114: R4D_TILED8_LAUNCH(1, 8, 5, 1, UNRG); break;   // BM 80,  BN 128
    case 124: R4D_TILED8_LAUNCH(1, 8, 4, 1, UNRG); break;   // BM 64,  BN 128
    case 125: R4D_TILED8_LAUNCH(1, 8, 7, 1, UNRG); break;   // BM 112, BN 128
    case 126: R4D_TILED8_LAUNCH(1, 8, 10, 1, UNRG); break;  // BM 160, BN 128
    case 128: R4D_TILED8_LAUNCH(1, 8, 2, 1, UNRG); break;   // BM 32,  BN 128
    case 129: R4D_TILED8_LAUNCH(1, 8, 3, 1, UNRG); break;   // BM 48,  BN 128
    case 138: R4D_TILED8_LAUNCH(1, 16, 3, 1, UNRG); break;  // BM 48,  BN 256
    case 139: R4D_TILED8_LAUNCH(1, 16, 7, 1, UNRG); break;  // BM 112, BN 256
    case 146: R4D_TILED8_LAUNCH(1, 8, 9, 1, UNRG); break;   // BM 144, BN 128
    default:  R4D_TILED8_LAUNCH(2, 8, 4, 2, UNRG); break;   // BM 128, BN 256
  }
#undef R4D_TILED8_LAUNCH
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
extern "C" int r4d_rad_gemm_w8a8_tiled(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    if (is_asym(a)) return R4D_NO(RAD_E_UNSUPPORTED);
    if (g.N % 256) return R4D_NO(RAD_E_SHAPE);
    if (g.K % R4D_GEMM_W8A8_GROUP) return R4D_NO(RAD_E_SHAPE);
    R4dTileCfg tv;
    r4d_cfg_tiled(a, &tv);
    const R4dTileCfg* t = &tv;
    rc = tile_ok(t, g.N);
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_w8a8_tiled",
        r4d_gemm_w8a8_tiled(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale),
                            r4d_addr(g.y), (int)g.M, (int)g.K, (int)g.N, (long)(uintptr_t)s,
                            t->variant));
    return RAD_OK;
}
