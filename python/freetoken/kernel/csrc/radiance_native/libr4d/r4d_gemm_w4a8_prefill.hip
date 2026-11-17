// W4A8 GEMM for prefill: the packed weight and rotated int8 activation of r4d_gemm_w4a8_tiled, for
// the compute-bound regime. At decode this GEMM is a weight stream at the DRAM ceiling; at M = 4096
// (730 GFLOP against 44.6 MB) every non-matrix instruction is overhead.
//
// Ceiling: v_wmma_i32_16x16x16_iu8 occupies its SIMD for 8 cycles on gfx1201 (f16/bf16: 16), an
// int8 peak of ~375 TOPS. There is no wider iu8 form (v_wmma_i32_16x16x32_iu8 does not exist), and
// iu4 needs both operands 4-bit. The budget is 8 cycles per matrix instruction plus one per other
// VALU instruction. The group fold is the largest non-matrix item, and as v_fma_mix_f32 (VOP3P,
// never dual-issued) it costs more than as v_fmac_f32, which pairs as v_dual_fmac_f32, the only
// VOPD form this loop can use (no v_dual_and_b32 or v_dual_lshlrev_b32). So the f16 scale is
// materialised and folded with plain FMACs.
//
// A fragment reads: the tiled kernel stores A k-major and issues one ds_load_b64 per (row tile,
// k step). Here a row's 64 bytes are half-major, [row][lane half][k step][2], so one ds_load_b128
// fetches a lane's two consecutive k steps (TM*2 reads per 64-k block instead of TM*4). Only the LDS
// copy is permuted; a lane still consumes the k values the weight names.
//
// A 4-dword XOR swizzle keeps it conflict-free. A row is 16 dwords across 32 banks, so the row
// index contributes only its low bit; the mask must separate the rest and permute whole 4-dword
// blocks so a ds_load_b128 never straddles two. ((r >> 1) & 3) << 2 gives lanes 0..7 start banks
// {0,16,4,20,8,24,12,28}, all 32 banks once.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_PF_BK 64

typedef int      v2i_t __attribute__((ext_vector_type(2)));
typedef int      v8i_t __attribute__((ext_vector_type(8)));
typedef float    v8f_t __attribute__((ext_vector_type(8)));
typedef unsigned u4v_t __attribute__((ext_vector_type(4)));
typedef float    v4f_t __attribute__((ext_vector_type(4)));

// The unpacked byte is 16*nibble, read signed; the 16 is removed at the store. Three instructions
// for two dwords is the floor: gfx1201 has no v_lshl_and_b32 and no nibble permute.
__device__ __forceinline__ v2i_t pf_dq8(unsigned w) {
  v2i_t r;
  r[0] = (int)((w << 4) & 0xF0F0F0F0u);
  r[1] = (int)(w & 0xF0F0F0F0u);
  return r;
}

// OPT bits
//   1  grid swizzle: walk GSW m-tiles per n-tile so resident workgroups share A row blocks and
//      W column blocks
//   2  fold with plain v_fmac_f32 (VOPD) rather than v_fma_mix_f32
//   8  read A fragments with one ds_load_b64 per k step (the tiled kernel's shape)
// 128  load a row tile's eight Ascale values as two b128: a lane's rows
//      (m0 + i*16 + 8*half + e, e = 0..7) are contiguous
// 256  unpredicated stores for blocks wholly inside the matrix (all but the last row block)
// 512  non-temporal stores for C
template <int WM, int WN, int TM, int TN, int OPT, int GSW>
__global__ __launch_bounds__(WM * WN * 32, 8) void r4d_gemm_w4a8_prefill_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N) {
  constexpr int BK = R4D_PF_BK;
  constexpr int GRP = R4D_GEMM_W4A8_GROUP;
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int NTILE = BN / 16;
  constexpr int AD = BM * 16;                        // A dwords per buffer (BM rows x 16)
  constexpr int WD = NTILE * 128;                    // W dwords per buffer
  constexpr int NT_ = WM * WN * 32;
  constexpr int UNR = GRP / BK;                      // k blocks per scale group
  constexpr bool VOPD = (OPT & 2) != 0;
  constexpr bool NARROWA = (OPT & 8) != 0;
  constexpr bool ASCVEC = (OPT & 128) != 0;
  constexpr bool STFAST = (OPT & 256) != 0;
  constexpr bool STNT = (OPT & 512) != 0;

  // Both stagings move 16 bytes per thread per pass, predicated on the last pass so the tile need
  // not divide the thread count (exact at 512 threads only for BM = 128, BN = 256).
  constexpr int AU = AD / 4, WU = WD / 4;
  constexpr int APT = (AU + NT_ - 1) / NT_, WPT = (WU + NT_ - 1) / NT_;
  constexpr bool APAD = (AU % NT_) != 0, WPAD = (WU % NT_) != 0;

  __shared__ unsigned lds_a[2][AD];
  __shared__ unsigned lds_w[2][WD];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int half = lane >> 4;                        // which 8-byte half of the 16-k step

  int bx = blockIdx.x, by = blockIdx.y;
  if (OPT & 1) {                                     // square-ish walk over the grid
    const int nt = gridDim.x, mt = gridDim.y;
    const int lin = by * nt + bx, per = GSW * nt;
    const int grp = lin / per, rem = lin - grp * per;
    const int gm = (mt - grp * GSW) < GSW ? (mt - grp * GSW) : GSW;
    by = grp * GSW + rem % gm;
    bx = rem / gm;
  }
  const int m0 = by * BM, n0 = bx * BN;
  const int kpb = K / BK, nsz = K / GRP;

  // A LDS position of (row, lane half, k step): half-major within the row, 4-dword XOR swizzle.
#define PF_APOS(r, h, u) ((r) * 16 + ((((h) * 8 + (u) * 2)) ^ ((((r) >> 1) & 3) << 2)))

  v8f_t acc[TN][TM];
  v8i_t g[TN][TM];
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++) acc[j][i] = (v8f_t)(0.f);
  float csum[TN];
#pragma unroll
  for (int j = 0; j < TN; j++) csum[j] = 0.f;
  unsigned scv[TN];

  u4v_t ra[APT], rw[WPT];
  auto issue = [&](int kk) {
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t >> 2, q = t & 3;
      if (APAD && t >= AU) continue;
      const int mm = m0 + r < M ? m0 + r : M - 1;
      ra[x] = *(const u4v_t*)(A + (unsigned)(mm * K + kk + q * 16));
    }
#pragma unroll
    for (int x = 0; x < WPT; x++) {
      const int t = tid + x * NT_, tl = t >> 5, o4 = t & 31;
      if (WPAD && t >= WU) continue;
      rw[x] = *(const u4v_t*)(Wq + (unsigned)(((n0 / 16 + tl) * kpb + kk / BK) * 128 + o4 * 4));
    }
  };
  auto commit = [&](int buf) {
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t >> 2, q = t & 3;
      if (APAD && t >= AU) continue;
      // the 16 bytes are k = 16q .. 16q+15: dwords 0,1 are the low half, 2,3 the high half
      v2i_t lo, hi;
      lo[0] = (int)ra[x][0]; lo[1] = (int)ra[x][1];
      hi[0] = (int)ra[x][2]; hi[1] = (int)ra[x][3];
      *(v2i_t*)(lds_a[buf] + PF_APOS(r, 0, q)) = lo;
      *(v2i_t*)(lds_a[buf] + PF_APOS(r, 1, q)) = hi;
    }
#pragma unroll
    for (int x = 0; x < WPT; x++) {
      const int t = tid + x * NT_;
      if (WPAD && t >= WU) continue;
      *(u4v_t*)(lds_w[buf] + t * 4) = rw[x];
    }
  };
  auto bar = [&]() {
    // The release fence makes commit's LDS stores land before the signal. Unlike
    // __syncthreads(), s_barrier_signal emits no s_wait_dscnt, so without the fence other
    // waves would read the previous k block's LDS contents.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
  };

  issue(0);
  commit(0);
  bar();

  for (int kg = 0; kg < K; kg += BK * UNR) {
#pragma unroll
    for (int h = 0; h < UNR; h++) {
      const int k0 = kg + h * BK;
      const int buf = h & 1;
      if (k0 + BK < K) issue(k0 + BK);
      if ((k0 % GRP) == 0) {                         // the group scale, one k block early
#pragma unroll
        for (int j = 0; j < TN; j++)
          scv[j] = Ws[(unsigned)(((n0 / 16 + wn * TN + j) * nsz + k0 / GRP) * 16 + (lane & 15))];
      }

      // ---- fragments for the whole 64-k block, up front ----
      v2i_t af[TM][BK / 16];
#pragma unroll
      for (int i = 0; i < TM; i++) {
        const int row = (wm * TM + i) * 16 + (lane & 15);
        if (NARROWA) {
#pragma unroll
          for (int u = 0; u < BK / 16; u++) {
            const unsigned* p = lds_a[buf] + PF_APOS(row, half, u);
            af[i][u][0] = (int)p[0];
            af[i][u][1] = (int)p[1];
          }
        } else {
#pragma unroll
          for (int q = 0; q < BK / 32; q++) {        // one b128 = two k steps
            const u4v_t v =
                *(const u4v_t*)(lds_a[buf] + PF_APOS(row, half, 2 * q));
            af[i][2 * q][0] = (int)v[0];
            af[i][2 * q][1] = (int)v[1];
            af[i][2 * q + 1][0] = (int)v[2];
            af[i][2 * q + 1][1] = (int)v[3];
          }
        }
      }
      u4v_t wv[TN];
#pragma unroll
      for (int j = 0; j < TN; j++)
        wv[j] = *(const u4v_t*)(lds_w[buf] + (size_t)(wn * TN + j) * 128 + lane * 4);

      // ---- matrix work ----
      // A group's first WMMA seeds C with 0x4B400000 (the float 1.5 * 2^23), so the int32 sum's
      // bits read as a float are 12582912 + sum while |sum| < 2^22. The fold below multiplies that
      // float by the scale, and the bias 12582912 * sum(scale) is removed after the K loop.
#pragma unroll
      for (int u = 0; u < BK / 16; u++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const v2i_t bf = pf_dq8(wv[j][u]);
#pragma unroll
          for (int i = 0; i < TM; i++)
            g[j][i] = ((k0 % GRP) == 0 && u == 0)
                          ? __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                true, af[i][u], true, bf, (v8i_t)(0x4B400000), false)
                          : __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                true, af[i][u], true, bf, g[j][i], false);
        }
      }

      if ((k0 + BK) % GRP == 0) {                    // fold the closed group into the f32 acc
        float sc[TN];
#pragma unroll
        for (int j = 0; j < TN; j++) {
          sc[j] = __half2float(__ushort_as_half((unsigned short)(scv[j] & 0xFFFFu)));
          // Materialise the convert; otherwise it contracts into v_fma_mix_f32 (VOP3P, never
          // dual-issued).
          if (VOPD) asm("" : "+v"(sc[j]));
          csum[j] += sc[j];
        }
#pragma unroll
        for (int j = 0; j < TN; j++)
#pragma unroll
          for (int i = 0; i < TM; i++)
#pragma unroll
            for (int e = 0; e < 8; e++)
              acc[j][i][e] = fmaf(__uint_as_float((unsigned)g[j][i][e]), sc[j], acc[j][i][e]);
      }

      if (k0 + BK < K) commit(buf ^ 1);
      bar();
    }
  }
#undef PF_APOS

#pragma unroll
  for (int j = 0; j < TN; j++) {
    const float d = -12582912.0f * csum[j];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[j][i][e] += d;
  }

  const bool inside = STFAST && (m0 + BM <= M);
#pragma unroll
  for (int i = 0; i < TM; i++) {
    float as[8];
    const int mb = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
    if (ASCVEC) {
      // The vector read Ascale[mc .. mc+7] stays inside [0, M) only when M >= 8; a short M, or a
      // lane whose rows were clamped, takes the clamped scalar read instead.
      const int mc = mb < M - 8 ? mb : (M > 8 ? M - 8 : 0);
      if (M >= 8) {
        const v4f_t* pv = (const v4f_t*)(Ascale + mc);
        const v4f_t s0 = pv[0], s1 = pv[1];
#pragma unroll
        for (int e = 0; e < 4; e++) { as[e] = s0[e]; as[e + 4] = s1[e]; }
      }
      if (M < 8 || mb != mc) {
#pragma unroll
        for (int e = 0; e < 8; e++) as[e] = Ascale[mb + e < M ? mb + e : M - 1];
      }
    }
#pragma unroll
    for (int j = 0; j < TN; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
        const int m = mb + e;
        if (inside || (n < N && m < M)) {
          // The scale read stays under the guard: past the last full tile `m >= M`, and the row it
          // would read belongs to the next tile or to nothing at all.
          const __hip_bfloat16 v =
              (__hip_bfloat16)(acc[j][i][e] * (ASCVEC ? as[e] : Ascale[m]) * 0.0625f);
          // the non-temporal builtin does not take a __hip_bfloat16*, so store its bits
          if (STNT)
            __builtin_nontemporal_store(*(const unsigned short*)&v,
                                        (unsigned short*)(C + (size_t)m * N + n));
          else
            C[(size_t)m * N + n] = v;
        }
      }
  }
}

template <int WM, int WN, int TM, int TN, int OPT, int GSW>
static void pf_launch(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                      long stream) {
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  if (N % BN) throw std::runtime_error("r4d gemm_w4a8_prefill: N must be a multiple of BN");
  if (K % R4D_GEMM_W4A8_GROUP) throw std::runtime_error("r4d gemm_w4a8_prefill: K % group");
  dim3 grid(N / BN, (M + BM - 1) / BM);
  hipLaunchKernelGGL((r4d_gemm_w4a8_prefill_kernel<WM, WN, TM, TN, OPT, GSW>), grid,
                     dim3(WM * WN * 32), 0, (hipStream_t)stream, (const signed char*)a,
                     (const float*)ascale, (const unsigned*)wq, (const unsigned*)ws,
                     (__hip_bfloat16*)c, M, K, N);
}

void r4d_gemm_w4a8_prefill(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                           long stream, int variant) {
#define PF(...) pf_launch<__VA_ARGS__>(a, ascale, wq, ws, c, M, K, N, stream)
  // The arms tile_bmbn in r4d_rows.cpp lists; 8, 9, 10, 13 and 14 change the tile, the rest are
  // BM 128 x BN 256 with other OPT bits.
  switch (variant) {
    case 2:  PF(2, 8, 4, 2, 8, 1); break;                 // b64 A, no VOPD fold
    case 3:  PF(2, 8, 4, 2, 10, 1); break;                // b64 A, VOPD fold
    case 5:  PF(2, 8, 4, 2, 3, 4); break;                 // + grid swizzle, GSW 4
    case 6:  PF(2, 8, 4, 2, 3, 8); break;                 // + grid swizzle, GSW 8
    case 8:  PF(2, 8, 2, 2, 2, 1); break;                 // BM 64,  BN 256
    case 9:  PF(2, 8, 3, 2, 2, 1); break;                 // BM 96,  BN 256
    case 10: PF(2, 8, 4, 1, 2, 1); break;                 // BM 128, BN 128
    case 13: PF(1, 8, 4, 2, 2, 1); break;                 // BM 64,  BN 256, 256 threads
    case 14: PF(2, 8, 8, 1, 2, 1); break;                 // BM 256, BN 128
    case 27: PF(2, 8, 4, 2, 2 | 128, 1); break;           // vectorised Ascale
    case 28: PF(2, 8, 4, 2, 8 | 128, 1); break;           // vectorised Ascale, b64 A
    case 29: PF(2, 8, 4, 2, 8 | 128 | 256, 1); break;     // + unpredicated store
    case 30: PF(2, 8, 4, 2, 8 | 128 | 256 | 512, 1); break;  // + non-temporal
    default: PF(2, 8, 4, 2, 2, 1); break;                 // b128 A + VOPD fold
  }
#undef PF
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
extern "C" int r4d_rad_gemm_w4a8_prefill(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    if (is_asym(a)) return R4D_NO(RAD_E_UNSUPPORTED);   /* no asymmetric prefill form */
    if (g.K % R4D_GEMM_W4A8_GROUP) return R4D_NO(RAD_E_SHAPE);
    R4dTileCfg tv;
    r4d_cfg_prefill(a, &tv);
    const R4dTileCfg* t = &tv;
    rc = tile_ok(t, g.N);
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_w4a8_prefill",
        r4d_gemm_w4a8_prefill(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                              r4d_addr(g.bscale), r4d_addr(g.y),
                              (int)g.M, (int)g.K, (int)g.N, (long)(uintptr_t)s, t->variant));
    return RAD_OK;
}

