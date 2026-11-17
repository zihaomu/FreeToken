// r4d_gemm_w4a8_nt_m64.hip - r4d_gemm_w4a8_nt_m64: 4-bit-weight, 8-bit-activation skinny GEMM for
// M up to 64, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with A quantised per row.
//
// Complements r4d_gemm_w4a16_nt_m64, which is at the DRAM roofline at M=8 but capped by the f16
// matrix instruction at M=64 (72-78 TF/s against a 207 TF/s issue ceiling; removing the unpack
// barely helps). Every 8-bit form runs at twice the f16 rate:
//
//     v_wmma_f32_16x16x16_f16       207 TF/s      5.95 cyc/instr
//     v_wmma_f32_16x16x16_fp8_fp8   412           2.99
//     v_wmma_i32_16x16x16_iu8       407           3.03
//     v_wmma_i32_16x16x32_iu4       830           2.97
//
// The unpack is three instructions with no zero point. The weight is a 4-bit two's complement code
// placed in the high half of its byte: for q in -8..7 the byte 16*q is the signed int8 of 16*q:
//
//     lo = (w << 4) & 0xF0F0F0F0     bytes <- nibbles 0,2,4,6      (elements 0..3)
//     hi =  w       & 0xF0F0F0F0     bytes <- nibbles 1,3,5,7      (elements 4..7)
//
// Two v_and_b32 and one v_lshlrev_b32 per eight weights (eleven on the f16 route); the factor 16 is
// removed at the store. A symmetric grid needs no per-group row sum of A for a zero point.
//
// The weight is shared byte for byte with r4d_gemm_w4a16_nt_m64. Its f16 unpack pairs nibble 2b
// into element b and 2b+1 into element 4+b, the order the two masks above produce, and with a
// constant zero point 8 it computes the same n-8. Only the low half (the scale) of the
// (scale, -(1024+zero)) dword is read here. Small M stays on the f16 kernel, which needs no
// activation-quantiser launch.
//
// The activation is shuffled within each k step. Fragment element e of lane l is
// A[row][16s + 8*(e>>2) + 4*(l>>4) + (e&3)], two dwords eight bytes apart in a row-major A. The
// quantiser writes each step's sixteen bytes as [half 0's eight][half 1's eight], so a fragment is
// one global_load_b64 (32 loads per group instead of 64).
//
// Do not lay A out as two planes ([2][M][K/2]): it allows global_load_b128 but is slower, because
// one instruction then spans two distant regions and touches twice the cache lines. The shuffle
// keeps a wave within the same sixteen rows.
//
// The per-row activation scale is constant over n and groups, so it is applied once at the store.
// The loop keeps one v_cvt_f32_i32 per accumulator element per group. A wider group would suit this
// kernel alone; 128 keeps enough legal split-K counts for the f16 kernel at M=8 (one artifact
// feeds every W4 kernel).
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"
#include "r4d_chain.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

// M is not structural (grid.y tiles it, rows are clamped); this is a sanity bound wide enough for a
// prefill chunk and the decode band (e.g. 8 sequences x 9 verify rows = 72), not a tuning band.
#define R4D_GEMM_W4A8_MAX_M 8192
// R4D_GEMM_W4A8_GROUP comes from r4d_common.h; see the note there.
#define R4D_GEMM_W4A8_KPB   64       // K per packed block: 4 k steps, 16 bytes per lane
#define WAVE32 32

typedef int      v8i   __attribute__((ext_vector_type(8)));
typedef float    v4f   __attribute__((ext_vector_type(4)));
typedef int      v2i   __attribute__((ext_vector_type(2)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));

typedef __attribute__((address_space(1))) const uint4v*   r4d_gw4;
typedef __attribute__((address_space(1))) const unsigned* r4d_gh1;
typedef unsigned uint2v __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(1))) const uint2v*   r4d_ga2;

// A fragments from the pre-permuted plane: eight contiguous bytes per k step.
__device__ __forceinline__ v2i fragA8(r4d_ga2 p, int u) {
  const uint2v w = p[2 * u];
  v2i r;
  r[0] = (int)w[0];
  r[1] = (int)w[1];
  return r;
}

// The whole dequant: two ands and a shift.
__device__ __forceinline__ v2i dequant8(unsigned w) {
  v2i r;
  r[0] = (int)((w << 4) & 0xF0F0F0F0u);
  r[1] = (int)(w & 0xF0F0F0F0u);
  return r;
}

// BSIGNED=false reads the unpacked byte as 16*nibble (0..240) instead of 16*twos(nibble)
// (-128..112), which makes an asymmetric grid expressible: 16*nibble is affine in the nibble, so
// sum_k a_k (nibble_k - z) = sum_k a_k nibble_k - z sum_k a_k.
template <bool BSIGNED>
__device__ __forceinline__ v8i mmai(v2i a, v2i b, v8i c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, BSIGNED, b, c, false);
}

// __launch_bounds__ depends on NPW. The split-K buffer (WV*NPW*SK KiB) must fit 64 KiB of LDS, so
// threads = 32*WV*SK <= 32*64/NPW. A flat 1024 would budget 32 waves per workgroup and make NPW=4
// and NPW=8 spill, though they can never run more than 16 or 8 waves. The bound is a hard launch
// limit, so the host checks WV*SK against the same expression.
#define R4D_W4A8_BOUNDS(NPW) ((NPW) >= 8 ? 256 : ((NPW) >= 4 ? 512 : 1024))

// CHAIN brackets the body with the chained launch's wait and add (r4d_chain.h); every block runs
// to the end. The two kernels below are the plain form and the chained one (r4dchain*).
template <int MT, int NPW, bool NT, bool ASYM, bool CHAIN>
__device__ __forceinline__ void r4d_gemm_w4a8_nt_m64_body(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    const float* __restrict__ Asum, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  if (CHAIN) r4d_chain_wait();
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / R4D_GEMM_W4A8_GROUP;
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  // The asymmetric activation sums are loaded at the top of each group, hidden behind the WMMA
  // block. Loading them later (shorter live range) or staging them in LDS were both slower.

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

    const int kpb = K / R4D_GEMM_W4A8_KPB;
    const int nsz = K / R4D_GEMM_W4A8_GROUP;
    r4d_gw4 pw[NPW];
    r4d_gh1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_gw4)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W4A8_KPB) * WAVE32 + lane;
      ps[j] = (r4d_gh1)(const unsigned*)(Ws) +
              (size_t)(t * nsz + kbase / R4D_GEMM_W4A8_GROUP) * 16 + (lane & 15);
    }

    const int bpg = R4D_GEMM_W4A8_GROUP / R4D_GEMM_W4A8_KPB;

    for (int g = 0; g < groups; g++) {
      float sc[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++)
        sc[j] = __half2float(__ushort_as_half(
            (unsigned short)(ps[j][(size_t)g * 16] & 0xFFFFu)));

      // The zero point is the dword's high half, the f16 of -(1024 + zero) (exact: f16 has unit
      // steps between 1024 and 2048). The symmetric grid always stores 8 there.
      float z16[NPW];
      float asum[MT][8];
      if (ASYM) {
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const float nz = __half2float(__ushort_as_half(
              (unsigned short)(ps[j][(size_t)g * 16] >> 16)));
          z16[j] = 16.f * (-nz - 1024.f);
        }
        // Element e of tile i at this lane is output row (mt0+i)*16 + 8*(lane>>4) + e, so a lane's
        // eight rows are contiguous at a fixed group; hence Asum is [G][Mpad]. Two v4f loads need a
        // 32-byte aligned base, which Mpad (a multiple of 16) provides; it also keeps the last
        // tile's rb+7 in bounds. These loads are most of the asymmetric grid's extra cost.
        const int Mpad = ((M + 15) / 16) * 16;
        const size_t gi = (size_t)(kbase / R4D_GEMM_W4A8_GROUP + g) * Mpad;
#pragma unroll
        for (int i = 0; i < MT; i++) {
          // Clamped: MB may exceed the number of M tiles (MB=4 with M=9 is legal), and an unclamped
          // Asum read could fault. The clamped base stays a multiple of 8, keeping v4f alignment.
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

      v8i g_acc[NPW][MT];
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
          v2i afrag[MT];
#pragma unroll
          for (int i = 0; i < MT; i++) afrag[i] = fragA8(pa[i], u);
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const v2i bfrag = dequant8(wv[j][s]);
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = (u == 0) ? mmai<!ASYM>(afrag[i], bfrag, (v8i)(0))
                                     : mmai<!ASYM>(afrag[i], bfrag, g_acc[j][i]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++) {
            float t = (float)g_acc[j][i][e];
            if (ASYM) t -= z16[j] * asum[i][e];
            acc[j][i][e] = fmaf(t, sc[j], acc[j][i][e]);
          }
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += R4D_GEMM_W4A8_GROUP / 8;
    }
  }

  // Split-K reduction and store, one row tile at a time, applying the per-row activation scale.
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
      // 0.0625 undoes the 16 from placing the nibble in the byte's high half.
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(v * Ascale[m] * 0.0625f);
    }
  }
  if (CHAIN) r4d_chain_done();
}

#define R4D_W4A8_PARAMS const signed char* __restrict__ A, const float* __restrict__ Ascale, \
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,                        \
    const float* __restrict__ Asum, __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK
template <int MT, int NPW, bool NT, bool ASYM>
__global__ __launch_bounds__(R4D_W4A8_BOUNDS(NPW)) void r4d_gemm_w4a8_nt_m64_kernel(R4D_W4A8_PARAMS) {
  r4d_gemm_w4a8_nt_m64_body<MT, NPW, NT, ASYM, false>(A, Ascale, Wq, Ws, Asum, C, M, K, N, WV, SK);
}
template <int MT, int NPW, bool NT, bool ASYM>
__global__ __launch_bounds__(R4D_W4A8_BOUNDS(NPW)) void r4dchain_gemm_w4a8_nt_m64(R4D_W4A8_PARAMS) {
  r4d_gemm_w4a8_nt_m64_body<MT, NPW, NT, ASYM, true>(A, Ascale, Wq, Ws, Asum, C, M, K, N, WV, SK);
}
#undef R4D_W4A8_PARAMS

static void r4d_gemm_w4a8_impl(long a, long ascale, long wq, long ws, long asum, long c,
                               int M, int K, int N, int WV, int SK, int MB, int NPW, int NT,
                               bool asym, long stream, bool chain = false) {
  if (M < 1 || M > R4D_GEMM_W4A8_MAX_M)
    throw std::runtime_error("r4d gemm_w4a8_nt_m64: M out of range");
  if (N % 16) throw std::runtime_error("r4d gemm_w4a8_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_W4A8_GROUP))
    throw std::runtime_error("r4d gemm_w4a8_nt_m64: K must be divisible by SK*group");
  if (WV * SK * WAVE32 > R4D_W4A8_BOUNDS(NPW))
    throw std::runtime_error("r4d gemm_w4a8_nt_m64: WV*SK*32 exceeds this tile's launch bound");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w4a8_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_w4a8_nt_m64: NPW must be 1, 2, 4 or 8");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w4a8_nt_m64: WV*NPW*SK too large");
  auto Ap = (const signed char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned*)wq;
  auto Zp = (const unsigned*)ws;
  auto Up = (const float*)asum;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT, NP, TT, AS) do { if (chain) \
    hipLaunchKernelGGL((r4dchain_gemm_w4a8_nt_m64<MT, NP, TT, AS>), \
    grid, block, smem, st, Ap, Sp, Wp, Zp, Up, Cp, M, K, N, WV, SK); else \
    hipLaunchKernelGGL((r4d_gemm_w4a8_nt_m64_kernel<MT, NP, TT, AS>), \
    grid, block, smem, st, Ap, Sp, Wp, Zp, Up, Cp, M, K, N, WV, SK); } while (0)
#define LAUNCH_MB(NP, TT, AS) switch (MB) { \
    case 1: LAUNCH(1, NP, TT, AS); break; \
    case 2: LAUNCH(2, NP, TT, AS); break; \
    case 3: LAUNCH(3, NP, TT, AS); break; \
    case 4: LAUNCH(4, NP, TT, AS); break; \
    default: throw std::runtime_error("r4d gemm_w4a8_nt_m64: MB unsupported"); }
#define LAUNCH_NPW(TT, AS) switch (NPW) { \
    case 1: LAUNCH_MB(1, TT, AS); break; \
    case 2: LAUNCH_MB(2, TT, AS); break; \
    case 4: LAUNCH_MB(4, TT, AS); break; \
    case 8: LAUNCH_MB(8, TT, AS); break; \
    default: throw std::runtime_error("r4d gemm_w4a8_nt_m64: NPW must be 1, 2, 4 or 8"); }
  if (asym) {
    if (NT) { LAUNCH_NPW(true, true); } else { LAUNCH_NPW(false, true); }
  } else {
    if (NT) { LAUNCH_NPW(true, false); } else { LAUNCH_NPW(false, false); }
  }
#undef LAUNCH_NPW
#undef LAUNCH_MB
#undef LAUNCH
}

void r4d_gemm_w4a8_nt_m64(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int WV, int SK, int MB, int NPW, int NT, long stream) {
  r4d_gemm_w4a8_impl(a, ascale, wq, ws, 0, c, M, K, N, WV, SK, MB, NPW, NT, false, stream);
}

// Asymmetric grid: the nibble is read unsigned and the zero point is corrected with the per-row
// per-group activation sums from r4d_quant_act_i8_asum. `asum` is [K/group][Mpad] float,
// Mpad = ceil(M/16)*16 (see the kernel). Allocate it zeroed.
void r4d_gemm_w4a8_asym_nt_m64(long a, long ascale, long wq, long ws, long asum, long c,
                               int M, int K, int N, int WV, int SK, int MB, int NPW, int NT,
                               long stream) {
  if (!asum) throw std::runtime_error("r4d gemm_w4a8_asym_nt_m64: asum is required");
  r4d_gemm_w4a8_impl(a, ascale, wq, ws, asum, c, M, K, N, WV, SK, MB, NPW, NT, true, stream);
}

int r4d_gemm_w4a8_nt_m64_max_m(void) { return R4D_GEMM_W4A8_MAX_M; }
int r4d_gemm_w4a8_nt_m64_group(void) { return R4D_GEMM_W4A8_GROUP; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
/* ------------------------------------------------------------------ int8-activation grids */

/* The multi-user verify heights (33 to 64 rows: five to eight users of a DFlash2 verify, 8 rows a
 * user). The law's point there covers every row tile in one block at one N tile a wave (mb = all,
 * npw = 1), which re-reads the activation once a 16-column tile and holds MT accumulators a wave;
 * on 4x R9700 (one card, the 31B's per-card trunk shapes, every point byte-identical to the law's)
 * it ran a layer's four GEMMs in 259 us at 64 rows against 157 us here:
 *
 *                 q|k|v 3584x5376   o_proj 5376x2048   gate_up 10752x5376   down 5376x5376
 *   40 rows  law  46.4              27.4               67.1                 36.4
 *            here 22.0 (mb1 npw2)   21.1 (mb1 npw2)    58.8 (mb1 npw4)      30.8 (mb1 npw2)
 *   48 rows  law  31.4              30.4               71.9                 42.4
 *            here 22.2              21.2               58.9                 31.6
 *   56 rows  law  56.8              35.0               94.2                 62.4
 *            here 24.0 (mb1 npw2)   22.3 (mb2 npw1)    69.0 (mb2 npw4)      43.0 (mb1 npw2)
 *   64 rows  law  58.6              37.3               97.6                 65.2
 *            here 24.7              22.7               69.6                 44.0
 *
 * Neither axis moves a byte: the split stays the law's (a point whose wider block cannot hold it
 * is not taken, since a smaller split would regroup the K partials), and every output element
 * keeps its accumulation order. Only where neither `mb` nor `npw` was tuned. */
static void multi_user_point(const RadArgs* a, long long M, long long N, long long K, Cfg* c) {
    if (M <= 32 || M > 64) return;
    R4dGemmCfg cv;
    r4d_cfg_gemm(a, &cv);
    if (cv.mb > 0 || cv.npw > 1) return;
    int mb, npw;
    if (M <= 48)          { mb = 1; npw = N >= 8192 ? 4 : 2; }
    else if (N >= 8192)   { mb = 2; npw = 4; }
    else if (K <= 2048)   { mb = 2; npw = 1; }
    else                  { mb = 1; npw = 2; }
    if (c->wv * c->sk * WAVE32 > block_bound(BR_NPW, mb, npw)) return;
    if ((size_t)c->wv * npw * c->sk * 256 * sizeof(float) > 64 * 1024) return;
    c->mb = mb;
    c->npw = npw;
}

/* The asymmetric grid is selected by `dtype`, not by the presence of a_sum (see is_asym). */

extern "C" int r4d_rad_gemm_w4a8_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.M > r4d_gemm_w4a8_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    const int group = r4d_gemm_w4a8_nt_m64_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_NPW, &c);
    if (rc != RAD_OK) return rc;
    multi_user_point(a, g.M, g.N, g.K, &c);

    /* the engine's chained launch (RadKernelChainFn), at decode heights only */
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && g.M <= 64;
    if (is_asym(a)) {
        if (!g.asum) return R4D_NO(RAD_E_INVAL);      /* the entry point throws "asum is required" */
        R4D_TRY("gemm_w4a8_asym_nt_m64",
            r4d_gemm_w4a8_impl(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                               r4d_addr(g.bscale), r4d_addr(g.asum), r4d_addr(g.y),
                               (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw,
                               c.nt, true, (long)(uintptr_t)s, chain));
        return RAD_OK;
    }
    R4D_TRY("gemm_w4a8_nt_m64",
        r4d_gemm_w4a8_impl(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                           r4d_addr(g.bscale), 0, r4d_addr(g.y),
                           (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw, c.nt,
                           false, (long)(uintptr_t)s, chain));
    return RAD_OK;
}

