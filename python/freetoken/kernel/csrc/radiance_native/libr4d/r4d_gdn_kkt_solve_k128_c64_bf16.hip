// r4d_gdn_kkt_solve_k128_c64_bf16.hip -- the gated-delta-net chunk preamble in one kernel:
//
//     A = (I + strict_lower(diag(beta) . K K^T . e^{g_i - g_j}))^{-1}     per chunk of 64
//
// Replaces FLA's chunk_scaled_dot_kkt_fwd + solve_tril, whose 64x64 fp32 intermediate per
// (chunk, head) round-trips through HBM. Here the gram stays in LDS and only the bf16 inverse is
// stored. Also:
//
//   * The gram is computed once per k head, not per v head: K K^T does not depend on beta or g,
//     so one workgroup owns a (chunk, k head) and applies each v head's scales to it.
//   * The gram is computed as its upper-triangle tiles. K K^T is symmetric, and read transposed
//     into A's strictly lower half each accumulator holds eight consecutive columns of one row:
//     one 16-byte LDS store instead of eight 2-byte ones.
//
// The decay is applied per element as e^{g_i-g_j}, not split about a per-chunk reference c: each
// half of e^{g_i-c} * e^{c-g_j} can overflow fp32 on its own. Here the exponent is always <= 0
// (g decreases along the chunk and only i > j is kept), and the eight exps per tile are cheap in
// a memory-bound kernel.
//
// The inverse is blocked 16 -> 32 -> 64. The four diagonal blocks use fp32 forward substitution
// (one wave each, one lane per column), cheaper than a WMMA Neumann series at this size and exact
// in fp32, as in FLA. The 32-merges and the 64-merge are WMMA, using
//
//     [[P, 0], [R, Q]]^-1 = [[P^-1, 0], [-Q^-1 R P^-1, Q^-1]]
//
// mma(X, Y) computes X . Y^T from two k-contiguous operands, so a left operand is row-major and a
// right operand is stored transposed. An accumulator writes its transpose contiguously, so the
// intermediates are staged transposed; the final inverse is assembled row-major for the HBM store.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_common.h"   // lds_barrier
#include "r4d_gdn_args.h"
#include "r4d_gdn_wmma.h"

#define BT   64                 // chunk
#define KD  128                 // head_k
#define NBK   4                 // 16x16 blocks per chunk edge
#define AP   68                 // row pitch of the [BT][BT] buffers, in shorts (136 B): 8-byte
                                // aligned for the paired stores, and 16 rows still cover every
                                // bank
#define NW    4
#define NTHR (NW * 32)
#define NTILE 10                // upper-triangular tiles of a 4x4 tiling
#define MAXT  ((NTILE + NW - 1) / NW)

// LDS limits occupancy, and this kernel is memory-bound, so buffers are minimised. No transposed
// inverse is kept (fragT rebuilds the three right operands that need it). The inverse is formed
// in place in A, since each stage reads the block it overwrites before writing it, with barriers
// between stages:
//   diagonal p    reads A[p][p], writes it after the substitution has finished reading
//   32-merge      reads A10 and the two diagonal inverses, writes A10
//   64-merge      reads R = A[32:,:32] into TMP everywhere, barrier, then writes it
// One 64x64 buffer plus a 32x32 scratch: 11 KB instead of 31 KB.
struct KkSmem {
  unsigned short A[BT][AP];     // A on the way in, its inverse on the way out
  unsigned short TMP[32][36];   // one 32x32 intermediate, transposed
  float gb[BT], bt[BT];         // the chunk's gate cumsum and beta, per token
};

// (mt, nt) of upper-triangular tile ti, computed with wave-uniform arithmetic instead of a
// __constant__ table load.
__device__ __forceinline__ void kkt_tile(int ti, int* mt, int* nt) {
  int m = 0, n = ti;
  while (n >= NBK - m) { n -= (NBK - m); ++m; }
  *mt = m; *nt = n + m;
}

// No occupancy hint: at HG_RATIO=3 the kernel uses 190 VGPRs (8 waves per SIMD; LDS would allow
// 10), and forcing 10 waves spills 148 bytes per lane and is ~57% slower.
template <int HG_RATIO>
__global__
__launch_bounds__(NTHR)
void r4d_gdn_kkt_solve_kernel(const unsigned short* __restrict__ k,
                              const float* __restrict__ beta,
                              const float* __restrict__ g,
                              unsigned short* __restrict__ Ao,
                              const int* __restrict__ cu,
                              int N, int H, int Hg)
{
  __shared__ KkSmem s;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int lo = lane & 15, hi = lane >> 4;
  const int hg = blockIdx.y;

  // Resolve the flat chunk id to (sequence, chunk). The grid is an upper bound on the chunk
  // count, so surplus workgroups exit here; this saves the host a device-to-host sync to size
  // the grid exactly.
  int n = -1, t0 = 0, bos = 0, rows = 0;
  {
    int acc = 0, want = blockIdx.x;
    for (int i = 0; i < N; ++i) {
      const int b = cu[i], e = cu[i + 1];
      const int nc = (e - b + BT - 1) / BT;
      if (want < acc + nc) { n = i; bos = b; t0 = (want - acc) * BT;
                             rows = min(BT, e - b - t0); break; }
      acc += nc;
    }
    if (n < 0) return;
  }

  // ---- zero A once: the parts no store reaches (the upper tiles) are the same for every head.
  // 16-byte stores (the buffer is 8704 bytes, 16-byte aligned).
  for (int i = tid; i < BT * AP / 8; i += NTHR) ((uint4*)&s.A[0][0])[i] = make_uint4(0, 0, 0, 0);

  // ---- the gram, once for all HG_RATIO v heads of this k head -----------------------------
  const unsigned short* kp = k + ((size_t)(bos + t0) * Hg + hg) * KD;
  const unsigned krow = (unsigned)(Hg * KD);
  v8f P[MAXT];
#pragma unroll
  for (int t = 0; t < MAXT; ++t) P[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int t = 0; t < MAXT; ++t) {
    const int ti = warp + t * NW;
    if (ti < NTILE) {
      int mt, nt; kkt_tile(ti, &mt, &nt);
      // One 32-bit offset per operand row, hoisted out of the k loop (see rowGo/fragGo).
      const unsigned om = rowGo(krow, mt * 16, rows - 1, lane);
      const unsigned on = rowGo(krow, nt * 16, rows - 1, lane);
#pragma unroll
      for (int sk = 0; sk < KD / 16; ++sk)
        P[t] = mma(fragGo(kp, om + sk * 16), fragGo(kp, on + sk * 16), P[t]);
    }
  }

  for (int hh = 0; hh < HG_RATIO; ++hh) {
    const int hv = hg * HG_RATIO + hh;

    // ---- per-head gate and beta for this chunk
    if (tid < BT) {
      const int t = min(tid, rows - 1);
      s.gb[tid] = g[(size_t)(bos + t0 + t) * H + hv];
      s.bt[tid] = beta[(size_t)(bos + t0 + t) * H + hv];
    }
    lds_barrier();

    // ---- scale, mask and place the gram as the strictly lower half of A ---------------------
    // Tile (mt, nt) holds P[m][n]; the element written is A[n][m], in the lower triangle when
    // n > m. n is fixed per lane and m runs over the eight accumulator elements: one row of A,
    // one 16-byte store.
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
      const int ti = warp + t * NW;
      if (ti < NTILE) {
        int mt, nt; kkt_tile(ti, &mt, &nt);
        const int i = nt * 16 + lo;                      // row of A
        const int j0 = mt * 16 + 8 * hi;                 // first column of A
        const bool live = (i < rows);
        const float bi = live ? s.bt[i] : 0.0f;
        const float gi = live ? s.gb[i] : 0.0f;
        float vv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int j = j0 + e;
          // For a live row, i > j gives g_i <= g_j, so the exponent is never positive. Padding
          // rows and columns must be masked to -inf: with gi = 0 the exponent -gb[j] is large and
          // positive, 0 * inf = NaN, and the WMMA merges in the blocked inverse would spread it
          // into live rows.
          const float d = (live && i > j && j < rows) ? (gi - s.gb[j]) : -INFINITY;
          vv[e] = bi * __expf(d) * P[t][e];
        }
        *(uint2*)(&s.A[i][j0])     = f2bf4(vv[0], vv[1], vv[2], vv[3]);
        *(uint2*)(&s.A[i][j0 + 4]) = f2bf4(vv[4], vv[5], vv[6], vv[7]);
      }
    }
    lds_barrier();

    // ---- 1. the four diagonal blocks, by fp32 forward substitution --------------------------
    // D = (I + N)^-1 solves D = I - N D row by row. Lane l owns column l of D, so the dot product
    // over j < i is in 16 registers and N[i][j] is a broadcast LDS read.
    for (int p = warp; p < NBK; p += NW) {
      const int b0 = p * 16;
      float d[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j < i) acc += bf2f(s.A[b0 + i][b0 + j]) * d[j];
        d[i] = ((lo == i) ? 1.0f : 0.0f) - acc;
      }
      if (lane < 16) {
#pragma unroll
        for (int i = 0; i < 16; ++i) s.A[b0 + i][b0 + lo] = f2bf(d[i]);
      }
    }
    lds_barrier();

    // ---- 2. the two 32-merges: M10 = -D1 A10 D0 --------------------------------------------
    for (int blk = warp; blk < 2; blk += NW) {
      const int r0 = blk * 32;                            // 32-block origin
      // X = A10 . D0     (left A10 row-major, right D0 transposed = MT)
      v8f X = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
      X = mma(frag(&s.A[0][0], AP, r0 + 16, r0, lane),
              fragT(&s.A[0][0], AP, r0, r0, lane), X);
      // Stage X transposed for the next product; each of the two merges owns 16 rows of TMP.
      *(uint2*)(&s.TMP[blk * 16 + lo][8 * hi])     = f2bf4(X[0], X[1], X[2], X[3]);
      *(uint2*)(&s.TMP[blk * 16 + lo][8 * hi + 4]) = f2bf4(X[4], X[5], X[6], X[7]);
      // fragT reads A column-wise below, so the merge result needs only the row-major copy
      __builtin_amdgcn_wave_barrier();
      // M10 = -D1 . X    (left D1 row-major, in place in A; right X transposed, in TMP)
      v8f Y = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
      Y = mma(frag(&s.A[0][0], AP, r0 + 16, r0 + 16, lane),
              frag(&s.TMP[0][0], 36, blk * 16, 0, lane), Y);
      const int i = r0 + 16 + 8 * hi;                     // rows of M10
      const int j = r0 + lo;                              // column of M10
#pragma unroll
      for (int e = 0; e < 8; ++e) s.A[i + e][j] = f2bf(-Y[e]);
    }
    lds_barrier();

    // ---- 3. the 64-merge: M[32:,:32] = -Q^-1 (R P^-1) --------------------------------------
    // Y = R . P^-1, R = A[32:,:32] row-major, P^-1 read transposed out of A[:32,:32] by fragT
    for (int t = warp; t < 4; t += NW) {
      const int mt = t >> 1, nt = t & 1;
      v8f Y = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int sk = 0; sk < 2; ++sk)
        Y = mma(frag(&s.A[0][0], AP, 32 + mt * 16, sk * 16, lane),
                fragT(&s.A[0][0], AP, nt * 16, sk * 16, lane), Y);
      *(uint2*)(&s.TMP[nt * 16 + lo][mt * 16 + 8 * hi])     = f2bf4(Y[0], Y[1], Y[2], Y[3]);
      *(uint2*)(&s.TMP[nt * 16 + lo][mt * 16 + 8 * hi + 4]) = f2bf4(Y[4], Y[5], Y[6], Y[7]);
    }
    lds_barrier();                  // every wave has read R into TMP; now it may be overwritten
    // Z = -Q^-1 . Y, Q^-1 = A[32:,32:] row-major, Y transposed = TMP
    for (int t = warp; t < 4; t += NW) {
      const int mt = t >> 1, nt = t & 1;
      v8f Z = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int sk = 0; sk < 2; ++sk)
        Z = mma(frag(&s.A[0][0], AP, 32 + mt * 16, 32 + sk * 16, lane),
                frag(&s.TMP[0][0], 36, nt * 16, sk * 16, lane), Z);
      const int i = 32 + mt * 16 + 8 * hi;
      const int j = nt * 16 + lo;
#pragma unroll
      for (int e = 0; e < 8; ++e) s.A[i + e][j] = f2bf(-Z[e]);
    }
    lds_barrier();

    // ---- store: one row of the inverse is 64 bf16, contiguous in HBM -----------------------
    {
      unsigned short* op = Ao + ((size_t)(bos + t0) * H + hv) * BT;
      const size_t orow = (size_t)H * BT;
      for (int r = tid / 8; r < rows; r += NTHR / 8) {
        const int c8 = (tid & 7) * 8;
        *(uint4*)(op + (size_t)r * orow + c8) = *(const uint4*)(&s.A[r][c8]);
      }
    }
    if (hh + 1 < HG_RATIO) lds_barrier();
  }
}

extern "C" int r4d_gdn_kkt_solve_k128_c64_bf16(const void* k, const void* beta, const void* g,
                                               void* A, const void* cu, int N, int T, int H,
                                               int Hg, int K, int bt, void* stream)
{
  if (K != KD || bt != BT) return -1;
  if (H % Hg) return -2;
  const int nchunk_ub = (T + BT - 1) / BT + N;   // sum of per-sequence chunk counts, bounded
  dim3 grid(nchunk_ub, Hg), blk(NTHR);
#define KK_LAUNCH(R)                                                                          \
  hipLaunchKernelGGL((r4d_gdn_kkt_solve_kernel<R>), grid, blk, 0, (hipStream_t)stream,        \
                     (const unsigned short*)k, (const float*)beta, (const float*)g,           \
                     (unsigned short*)A, (const int*)cu, N, H, Hg)
  switch (H / Hg) {
    case 1: KK_LAUNCH(1); break;
    case 2: KK_LAUNCH(2); break;
    case 3: KK_LAUNCH(3); break;
    case 4: KK_LAUNCH(4); break;
    case 6: KK_LAUNCH(6); break;
    case 8: KK_LAUNCH(8); break;
    default: return -3;
  }
#undef KK_LAUNCH
  return (int)hipGetLastError();
}

// ============================================================== the plugin ABI entry points
// Shared operand helpers: r4d_gdn_args.h.
/* ------------------------------------------------------------------ kkt solve */

enum { KS_K = 0, KS_BETA, KS_G, KS_CU, KS_A };

extern "C" int r4d_rad_gdn_kkt_solve(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* k  = r4d_opt(a, KS_K);
    const RadTensor* bt = r4d_opt(a, KS_BETA);
    const RadTensor* g  = r4d_opt(a, KS_G);
    const RadTensor* cu = r4d_opt(a, KS_CU);
    const RadTensor* A  = r4d_opt(a, KS_A);
    if (!k || !bt || !g || !cu || !A) return R4D_NO(RAD_E_INVAL);
    if (k->rank != 3 || bt->rank != 2 || g->rank != 2 || A->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(k) || !r4d_contig(bt) || !r4d_contig(g) || !r4d_contig(A))
        return R4D_NO(RAD_E_STRIDE);
    /* bf16 k and A (the output, written as bf16); fp32 beta and gate. */
    R4D_DT(k, RAD_BF16); R4D_DT(A, RAD_BF16); R4D_DT(bt, RAD_F32); R4D_DT(g, RAD_F32);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;

    const long long T  = k->shape[0];
    const long long Hg = k->shape[1];
    const long long K  = k->shape[2];
    const long long H  = bt->shape[1];
    const long long chunk = A->shape[2];
    if (g->shape[0] != T || g->shape[1] != H || bt->shape[0] != T) return R4D_NO(RAD_E_SHAPE);
    if (A->shape[0] != T || A->shape[1] != H) return R4D_NO(RAD_E_SHAPE);
    if (H == 0 || Hg == 0 || H % Hg) return R4D_NO(RAD_E_SHAPE);

    return R4D_RAW(r4d_gdn_kkt_solve_k128_c64_bf16(k->data, bt->data, g->data, A->data,
                                                   cu->data, N, (int)T, (int)H, (int)Hg, (int)K,
                                                   (int)chunk, s));
}

