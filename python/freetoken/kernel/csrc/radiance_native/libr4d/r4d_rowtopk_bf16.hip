// r4d_rowtopk_bf16.hip - r4d_rowtopk_bf16: exact per-row top-R of a wide bf16 matrix.
//
// Purpose: a quantised draft head produces coarse logits over a vocabulary shard, and only the
// best few dozen are rescored exactly. A general top-k costs more than that saves: torch.topk over
// a 15520-wide array takes 132 us at M=64 in three kernels, and several times that over a full
// 124160-wide row.
//
// Two stages, because scanning and selecting want opposite shapes. Reading the matrix wants
// thousands of waves in flight; selection is a chain of dependent cross-lane steps (~300 cycles a
// round), too costly to run thousands of times. On a 15.9 MB matrix with a 25 us roofline:
//
//     256-thread blocks, R=32 rounds, two barriers a round     75.6 us   (chain, plus barriers)
//     one wave per chunk, R=32 rounds, no barrier              68.5 us   (chain, or too few waves)
//
// So stage 1 runs many waves with a short chain (one wave per (row, chunk), keeping the chunk's
// top-RC), and stage 2 runs one wave per row over the nch*RC survivors with the full R-round chain.
//
//   * A lane keeps its own top-PT in registers, sorted. Insertion is guarded by one compare
//     against the tail, so most elements cost one instruction; only a placing element pays the
//     branch-free shift. No LDS candidate array.
//   * A round only removes a list head: the winner is the head of exactly one lane's list, so that
//     lane shifts its own registers. Nothing is re-read per round.
//   * One wave per block, so a round is five xor-butterfly steps and no barrier.
//
// The key is one 64-bit integer, so a round is a plain max. The high 32 bits are the float's
// order-preserving image (invert all bits if negative, set the sign bit if not), the low 32 are
// ~index, so a larger key is a larger value and, on a tie, the lower column (a dense top-k's tie
// rule). The index also makes every key distinct, so a round identifies its winner by equality.
//
// PT bounds how many relevant candidates one lane's slice can hold:
//
//   * Stage 1 gives a lane every 32nd eight-column group, and PT must cover a whole vector load:
//     eight bf16 arrive at once, so PT=4 can lose a candidate to a single load (and does disagree
//     with torch.topk on smooth data). PT=8 places one load in full; beyond that a chunk would need
//     nine of its own top-RC in one of 32 slices.
//   * Stage 2's slice is not random: stage 1 writes chunk-major, so at stride 32 with RC=32 lane 0
//     receives every chunk's maximum, exactly where the row's top-R concentrates. Stage 2 therefore
//     uses PT = 32 >= R: one lane can hold the entire answer, so it is exact for any distribution.
//
// RC is the only possible approximation, and the caller's choice. A chunk hands on its top-RC, so
// the row's top-R survives unless more than RC of it falls in one chunk. RC=R is exact for any
// data; a smaller RC shortens stage 1's chain at a risk that falls fast with the chunk count (at
// R=32, RC=8 and 64 chunks it takes nine of the 32 in one sixty-fourth of the row). A caller that
// rescores exactly then loses a candidate, not correctness.
#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_fp8_wmma.h"
#include "r4d_fp8_lm.h"
#include "r4d_chain.h"
#include "r4d_iqr.h"
#include <hip/hip_bf16.h>
#include <cstdint>

#define R4D_TOPK_WAVE  32
#define R4D_TOPK_PT2   32      // stage 2: >= R, so a lane can hold the whole answer

typedef unsigned u4v_tk __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) const u4v_tk* r4d_tk_gv4;

#define R4D_TOPK_WPB 8        // independent waves per block: one chunk each, no barrier between
#define R4D_TOPK_UNROLL 4     // vector loads issued before any of them is consumed

__device__ __forceinline__ unsigned long long tk_key(unsigned short bf, unsigned idx) {
  unsigned u = (unsigned)bf << 16;
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((unsigned long long)u << 32) | (unsigned)(0xFFFFFFFFu - idx);
}

__device__ __forceinline__ float tk_val(unsigned long long k) {
  unsigned u = (unsigned)(k >> 32);
  u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
  return __uint_as_float(u);
}

__device__ __forceinline__ int tk_idx(unsigned long long k) {
  return (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
}

// The larger of this lane's key and its partner's at lane ^ N, read through DPP or permlanex16
// (see r4d_xor_lane).
template <int N>
__device__ __forceinline__ unsigned long long tk_xor_max(unsigned long long m) {
  const unsigned long long p = r4d_xor_lane_u64<N>(m);
  return p > m ? p : m;
}

// The insert's only guard is its own key compare. A float guard against the tail's value costs
// ~18% at M=64 in stage 1 (a branch breaks the unrolled straight-line body), and it is wrong with
// NaN: the key orders a positive NaN above +inf and a negative one below -inf, while a float
// compare is false, so once a NaN is the tail every later candidate would be dropped. Every stage
// and the merge rank on the key alone, so they agree on one order.
template <int PT>
__device__ __forceinline__ void tk_insert(unsigned long long* t, unsigned long long k) {
  if (k > t[PT - 1]) {
    unsigned long long c = k;
#pragma unroll
    for (int q = 0; q < PT; q++) {
      const unsigned long long hi = t[q] > c ? t[q] : c;
      c = t[q] > c ? c : t[q];
      t[q] = hi;
    }
  }
}

// One reduction round over the wave: every lane gets the winner and its owner drops its head. No
// LDS and no barrier.
template <int PT>
__device__ __forceinline__ unsigned long long tk_round(unsigned long long* t) {
  unsigned long long m = t[0];
  m = tk_xor_max<16>(m);
  m = tk_xor_max<8>(m);
  m = tk_xor_max<4>(m);
  m = tk_xor_max<2>(m);
  m = tk_xor_max<1>(m);
  if (m && t[0] == m) {
#pragma unroll
    for (int q = 0; q < PT - 1; q++) t[q] = t[q + 1];
    t[PT - 1] = 0ull;
  }
  return m;
}

// The body; the plain kernel and its chained form (r4d_chain.h) wrap it below. A wave past the
// chunks returns early; there is no barrier in it.
template <int PT>
__device__ __forceinline__ void r4d_rowtopk_stage1_body(
    const unsigned short* __restrict__ Y, unsigned long long* __restrict__ P,
    int N, int RC, int NCH, int CH) {
  // WPB independent waves per block, each on its own chunk, sharing nothing. Grouping them only
  // matters because 32-thread blocks exhaust the CU's workgroup slots before its wave slots.
  const int m = blockIdx.y;
  const int c = blockIdx.x * R4D_TOPK_WPB + (int)(threadIdx.x >> 5);
  if (c >= NCH) return;
  const int lane = threadIdx.x & 31;
  const int c0 = c * CH;
  int c1 = c0 + CH;
  if (c1 > N) c1 = N;

  unsigned long long t[PT];
#pragma unroll
  for (int q = 0; q < PT; q++) t[q] = 0ull;

  const unsigned short* row = Y + (size_t)m * N;
  // Eight bf16 per lane per load, so the wave moves 512 contiguous bytes per issue. UNROLL loads
  // are issued before the first is consumed, since the insert chain would otherwise put a memory
  // latency between consecutive loads.
  const int v0 = c0 >> 3, v1 = c1 >> 3;
  const int step = R4D_TOPK_WAVE * R4D_TOPK_UNROLL;
  int v = v0 + lane;
  for (; v + step - R4D_TOPK_WAVE < v1; v += step) {
    u4v_tk p[R4D_TOPK_UNROLL];
#pragma unroll
    for (int u = 0; u < R4D_TOPK_UNROLL; u++)
      p[u] = *(r4d_tk_gv4)(row + (size_t)(v + u * R4D_TOPK_WAVE) * 8);
#pragma unroll
    for (int u = 0; u < R4D_TOPK_UNROLL; u++) {
      const unsigned base = (unsigned)((v + u * R4D_TOPK_WAVE) * 8);
#pragma unroll
      for (int e = 0; e < 4; e++) {
        tk_insert<PT>(t, tk_key((unsigned short)(p[u][e] & 0xFFFFu), base + 2 * e));
        tk_insert<PT>(t, tk_key((unsigned short)(p[u][e] >> 16), base + 2 * e + 1));
      }
    }
  }
  for (; v < v1; v += R4D_TOPK_WAVE) {
    const u4v_tk p = *(r4d_tk_gv4)(row + (size_t)v * 8);
    const unsigned base = (unsigned)(v * 8);
#pragma unroll
    for (int e = 0; e < 4; e++) {
      tk_insert<PT>(t, tk_key((unsigned short)(p[e] & 0xFFFFu), base + 2 * e));
      tk_insert<PT>(t, tk_key((unsigned short)(p[e] >> 16), base + 2 * e + 1));
    }
  }
  for (int i = (v1 << 3) + lane; i < c1; i += R4D_TOPK_WAVE) {
    tk_insert<PT>(t, tk_key(row[i], (unsigned)i));
  }

  unsigned long long* out = P + ((size_t)m * NCH + c) * RC;
  for (int r = 0; r < RC; r++) {
    const unsigned long long w = tk_round<PT>(t);
    if (lane == 0) out[r] = w;
  }
}
template <int PT>
__global__ __launch_bounds__(R4D_TOPK_WAVE * R4D_TOPK_WPB) void r4d_rowtopk_stage1_kernel(
    const unsigned short* __restrict__ Y, unsigned long long* __restrict__ P,
    int N, int RC, int NCH, int CH) {
    r4d_rowtopk_stage1_body<PT>(Y, P, N, RC, NCH, CH);
}
template <int PT>
__global__ __launch_bounds__(R4D_TOPK_WAVE * R4D_TOPK_WPB) void r4dchain_rowtopk_stage1(
    const unsigned short* __restrict__ Y, unsigned long long* __restrict__ P,
    int N, int RC, int NCH, int CH) {
    r4d_chain_wait();
    r4d_rowtopk_stage1_body<PT>(Y, P, N, RC, NCH, CH);
    r4d_chain_done();
}

// `off` is the first global column this rank's plane stands for, and `OP` the optional (id, value)
// pair plane an all-gather moves: eight bytes an entry, an int32 id in the first float slot, the
// layout core/sample/sampler.h holds as buf_pairs_. Both are inert at off = 0 with OP null, which
// is the single-rank launch bit for bit.
//
// An empty slot is not token `off`. A chunk that ran out of candidates leaves a zero key; OI gets
// column 0 for it, without the offset, so a pad cannot look like this rank's first real token. The
// pair plane writes -1, the sentinel the merge drops (not -INFINITY, a valid masked logit).
// The body; the plain kernel and its chained form (r4d_chain.h) wrap it below. PT is a lane's
// list, at least R (see the header): 32 serves every R, and a launch for R <= 8 or R <= 16 takes
// the shorter list, whose insert and whose round's shift are that much shorter. The output is the
// same for any PT >= R.
template <int PT>
__device__ __forceinline__ void r4d_rowtopk_stage2_body(
    const unsigned long long* __restrict__ P, float* __restrict__ OV, int* __restrict__ OI,
    float* __restrict__ OP, int off, int R, int n) {
  const int m = blockIdx.x;
  const int lane = threadIdx.x;

  unsigned long long t[PT];
#pragma unroll
  for (int q = 0; q < PT; q++) t[q] = 0ull;

  // The key decides, as in stage 1 (see tk_insert). A NaN a chunk handed on displaced a real
  // candidate, so dropping it here can leave the row short of R; the pad filling the gap is what
  // the merge reports as id -1. An empty slot's key is 0 and never places.
  // UNROLL keys are loaded before the first is placed, as in stage 1: one load a pass would put a
  // memory latency between consecutive inserts. A lane places its keys in the same order.
  const unsigned long long* p = P + (size_t)m * n;
  int i = lane;
  for (; i + (R4D_TOPK_UNROLL - 1) * R4D_TOPK_WAVE < n; i += R4D_TOPK_UNROLL * R4D_TOPK_WAVE) {
    unsigned long long k[R4D_TOPK_UNROLL];
#pragma unroll
    for (int u = 0; u < R4D_TOPK_UNROLL; u++) k[u] = p[i + u * R4D_TOPK_WAVE];
#pragma unroll
    for (int u = 0; u < R4D_TOPK_UNROLL; u++) tk_insert<PT>(t, k[u]);
  }
  for (; i < n; i += R4D_TOPK_WAVE) tk_insert<PT>(t, p[i]);

  for (int r = 0; r < R; r++) {
    const unsigned long long w = tk_round<PT>(t);
    if (lane == 0) {
      // A zero key is an empty slot, not a real column.
      const float v = w ? tk_val(w) : -INFINITY;
      OV[(size_t)m * R + r] = v;
      OI[(size_t)m * R + r] = w ? (int)tk_idx(w) + off : 0;
      if (OP) {
        float* const q = OP + ((size_t)m * R + r) * 2;
        const int id = w ? (int)tk_idx(w) + off : -1;
        __builtin_memcpy(q, &id, sizeof(int));
        q[1] = v;
      }
    }
  }
}
template <int PT>
__global__ __launch_bounds__(R4D_TOPK_WAVE) void r4d_rowtopk_stage2_kernel(
    const unsigned long long* __restrict__ P, float* __restrict__ OV, int* __restrict__ OI,
    float* __restrict__ OP, int off, int R, int n) {
    r4d_rowtopk_stage2_body<PT>(P, OV, OI, OP, off, R, n);
}
template <int PT>
__global__ __launch_bounds__(R4D_TOPK_WAVE) void r4dchain_rowtopk_stage2(
    const unsigned long long* __restrict__ P, float* __restrict__ OV, int* __restrict__ OI,
    float* __restrict__ OP, int off, int R, int n) {
    r4d_chain_wait();
    r4d_rowtopk_stage2_body<PT>(P, OV, OI, OP, off, R, n);
    r4d_chain_done();
}

// Chunks are a multiple of eight columns so stage 1 stays on the vector load; the last one is
// clamped to N rather than padded (a padded tail would need a mask per element). The caller sizes
// `partial` from r4d_rowtopk_bf16_chunks, so the rounding lives here only.
static int rowtopk_chunk_width(int N, int NCH) {
  const int CH = ((N + NCH - 1) / NCH + 7) & ~7;
  return CH < 8 ? 8 : CH;
}

int r4d_rowtopk_bf16_chunks(int N, int NCH) {
  const int CH = rowtopk_chunk_width(N, NCH);
  return (N + CH - 1) / CH;
}

// The top-R over one rank's slice of a wider plane: `off` is the slice's first global column, and
// `outp` (optional, 0 for none) is the M*R (int32 id, float value) pair plane an all-gather moves
// so ranks can agree on a global top-R without gathering the plane.
static int rowtopk_bf16_sharded_emit(long y, long partial, long outv, long outi, long outp,
                                     int M, int N, int R, int RC, int NCH, int PT1, int off,
                                     RadStream stream, bool chain = false) {
  int emit_rc = RAD_OK;
  // R <= PT2 is what makes stage 2 exact for any candidates; a real limit, so checked, not
  // clamped.
  if (M < 1 || N < 1 || N % 8 || R < 1 || R > R4D_TOPK_PT2 || RC < 1 ||
      RC > PT1 * R4D_TOPK_WAVE || NCH < 1)
    return R4D_NO(RAD_E_SHAPE);
  const int CH  = rowtopk_chunk_width(N, NCH);
  const int nch = (N + CH - 1) / CH;
#define R4D_TK_S1(PT) emit_rc = R4D_EMIT(stream, (chain ? r4dchain_rowtopk_stage1<PT>           \
                                                       : r4d_rowtopk_stage1_kernel<PT>),         \
                        dim3((nch + R4D_TOPK_WPB - 1) / R4D_TOPK_WPB, M),                        \
                        dim3(R4D_TOPK_WAVE * R4D_TOPK_WPB), 0,                                   \
                        (const unsigned short*)y, (unsigned long long*)partial, N, RC, nch, CH)
  switch (PT1) {
    case 2: R4D_TK_S1(2); break;
    case 4: R4D_TK_S1(4); break;
    case 8: R4D_TK_S1(8); break;
    case 16: R4D_TK_S1(16); break;
    default: return R4D_NO(RAD_E_INVAL);
  }
#undef R4D_TK_S1
  if (emit_rc != RAD_OK) return emit_rc;
#define R4D_TK_S2(PT) R4D_EMIT(stream, (chain ? r4dchain_rowtopk_stage2<PT>                     \
                                               : r4d_rowtopk_stage2_kernel<PT>), dim3(M),         \
                               dim3(R4D_TOPK_WAVE), 0,                                             \
                               (const unsigned long long*)partial, (float*)outv, (int*)outi,      \
                               (float*)outp, off, R, nch * RC)
  const int rc2 = R <= 8 ? R4D_TK_S2(8) : R <= 16 ? R4D_TK_S2(16) : R4D_TK_S2(R4D_TOPK_PT2);
#undef R4D_TK_S2
  return rc2;
}

// ============================================================ the vocab-parallel merge
//
// The other half of a sharded top-R. Each rank ran the two stages over its own slice and wrote an
// [M, R, 2] pair plane (int32 global id, float value); an all-gather concatenated those into
// [M, W*R, 2] with rank r's at [r*R, (r+1)*R). This picks the R largest of the W*R and writes them
// in stage 2's (idx, val) form.
//
// The key is the same 64-bit integer, built from the float's bits, so a round is a plain max and
// ties break on the lower id, which makes this equal to a top-R over the unsharded plane. A pad
// carries id -1 and is skipped (a sentinel, since -INFINITY is a legitimate masked logit).
//
// One wave per row over W*R entries (32 at world 2, R 16); its cost is its launch.
__device__ __forceinline__ unsigned long long tk_key_f32(float v, unsigned id) {
  unsigned u = __float_as_uint(v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((unsigned long long)u << 32) | (unsigned)(0xFFFFFFFFu - id);
}

// The body; the plain kernel and its chained form (r4d_chain.h) wrap it below.
__device__ __forceinline__ void r4d_rowtopk_merge_body(
    const float* __restrict__ G, float* __restrict__ OV, int* __restrict__ OI, int R, int n) {
  const int m = blockIdx.x;
  const int lane = threadIdx.x;

  unsigned long long t[R4D_TOPK_PT2];
#pragma unroll
  for (int q = 0; q < R4D_TOPK_PT2; q++) t[q] = 0ull;

  const float* const row = G + (size_t)m * n * 2;
  for (int i = lane; i < n; i += R4D_TOPK_WAVE) {
    int id;
    __builtin_memcpy(&id, row + (size_t)i * 2, sizeof(int));
    if (id < 0) continue;
    tk_insert<R4D_TOPK_PT2>(t, tk_key_f32(row[(size_t)i * 2 + 1], (unsigned)id));
  }

  for (int r = 0; r < R; r++) {
    const unsigned long long w = tk_round<R4D_TOPK_PT2>(t);
    if (lane == 0) {
      OV[(size_t)m * R + r] = w ? tk_val(w) : -INFINITY;
      OI[(size_t)m * R + r] = w ? tk_idx(w) : -1;
    }
  }
}
__global__ __launch_bounds__(R4D_TOPK_WAVE) void r4d_rowtopk_merge_kernel(
    const float* __restrict__ G, float* __restrict__ OV, int* __restrict__ OI, int R, int n) {
    r4d_rowtopk_merge_body(G, OV, OI, R, n);
}
__global__ __launch_bounds__(R4D_TOPK_WAVE) void r4dchain_rowtopk_merge(
    const float* __restrict__ G, float* __restrict__ OV, int* __restrict__ OI, int R, int n) {
    r4d_chain_wait();
    r4d_rowtopk_merge_body(G, OV, OI, R, n);
    r4d_chain_done();
}

// ============================================================== logit_rerank
//
// The exact second half of a coarse top-R: the coarse head proposes R candidates per row, and this
// scores only those against the full-precision head, then re-sorts on the exact values. A draft
// round reads R weight rows instead of the whole head (40 KB a row against 318 MB). On
// Qwen3.8-Flash-Next the 2-bit head's acceptance is 4 to 8 points below the exact head's, and this
// closes that gap.
//
// One block per row, waves spread over the candidates. A wave walks K with a lane stride and folds
// with the same five xor-butterfly steps, so the reduction order is a fixed function of the shape.
// Wave 0 then runs the ordinary top-R chain over the R scores (one lane a candidate, R <= PT2),
// so ties break by tk_key_f32's rule, as in row_topk and row_topk_merge; ties within and between
// shards resolve alike, which keeps the merged answer exact.
//
// A candidate outside this rank's rows is a pad. `voff` is the shard's first column and ids are
// global, as row_topk reports them; under vocab-parallel drafting a rank only gets its own ids, and
// the guard is for a caller that mixes them.
//
// I8 reads the int8 head logits_gemm_i8 stores (r4d.i8lm.r128): the same row (K codes, then the
// row's bf16 scale per 128 columns in its tail) with int8 codes instead of E4M3. The drafter
// rescores against whichever head the trunk's logits GEMM reads, so an int8 head needs this form.
#define R4D_RERANK_WPB 8       // waves a block: 256 threads, and R <= 32 so a wave takes <= 4
#define R4D_RERANK_U   5       // 512-column steps a lane has in flight: all of K = 2560

/* TB > 0: an IQR head (r4d_iqr.h's lm form), TB-bit codes then the row's scales, the table of 2^TB
 * int8 entries after the last row. */
#define R4D_LOGIT_RERANK_PARAMS                                                                 \
    const unsigned short* __restrict__ X, const int* __restrict__ IN,                           \
    const unsigned char* __restrict__ W, int* __restrict__ OI, float* __restrict__ OV,          \
    float* __restrict__ OP, int R, long K, long voff, long nv, long w_ld
template <bool I8, uint32_t TB = 0>
__device__ __forceinline__ void r4d_logit_rerank_body(R4D_LOGIT_RERANK_PARAMS) {
  __shared__ float s_sc[R4D_TOPK_PT2];
  __shared__ int   s_id[R4D_TOPK_PT2];

  const int  m    = blockIdx.x;
  const int  lane = threadIdx.x & (R4D_TOPK_WAVE - 1);
  const int  wv   = threadIdx.x / R4D_TOPK_WAVE;
  const unsigned short* const xr = X + (size_t)m * K;

  for (int r = wv; r < R; r += R4D_RERANK_WPB) {
    const int  gid = IN[(size_t)m * R + r];
    const long loc = (long)gid - voff;
    const bool own = gid >= 0 && loc >= 0 && loc < nv;
    float acc = 0.0f;
    if (own) {
      /* The stored row, as logits_gemm_fp8 reads it (r4d_fp8_lm.h): K E4M3 codes then K/128 bf16
       * block scales in the row's tail. A lane owns a strided subset of columns and applies each
       * column's block scale locally, so the fold is a plain sum whose order depends only on the
       * lane count. */
      const unsigned char* const wr = W + (size_t)loc * w_ld;
      const unsigned short* const sr = (const unsigned short*)(wr + (TB ? K / 8 * TB : K));
      [[maybe_unused]] const int8_t* const tab =
          (const int8_t*)(W + ((size_t)nv * w_ld + 255) / 256 * 256);
      /* At K a multiple of 512 a lane takes sixteen consecutive columns of every 512, all in one
       * scale block: their products sum from zero in column order and the block's scale multiplies
       * that partial sum, the same arithmetic for every code form (an IQR head's scores are the
       * int8 head's to the bit). Byte codes on aligned planes load whole: R4D_RERANK_U steps are
       * in flight before the first is summed, where a column at a time is 80 loads in series at
       * K = 2560, each waiting for the last. */
      const bool ord = (K & 511) == 0;
      bool vec = false;
      if constexpr (TB == 0) vec = ord && ((size_t)X & 15) == 0 && ((size_t)W & 7) == 0;
      if constexpr (TB == 0) {
        for (long k0 = 16 * lane; vec && k0 < K; k0 += 512 * R4D_RERANK_U) {
          uint2 c[R4D_RERANK_U][2];
          uint4 xv[R4D_RERANK_U][2];
          uint32_t sv[R4D_RERANK_U];
#pragma unroll
          for (int u = 0; u < R4D_RERANK_U; u++) {
            const long k = k0 + 512 * u;
            if (k < K) {
              const uint2* const pc = (const uint2*)(wr + k);
              const uint4* const px = (const uint4*)(xr + k);
              c[u][0] = pc[0]; c[u][1] = pc[1];
              xv[u][0] = px[0]; xv[u][1] = px[1];
              sv[u] = sr[k / 128];
            }
          }
#pragma unroll
          for (int u = 0; u < R4D_RERANK_U; u++) {
            if (k0 + 512 * u < K) {
              const uint32_t cw[4] = { c[u][0].x, c[u][0].y, c[u][1].x, c[u][1].y };
              const uint32_t xw[8] = { xv[u][0].x, xv[u][0].y, xv[u][0].z, xv[u][0].w,
                                       xv[u][1].x, xv[u][1].y, xv[u][1].z, xv[u][1].w };
              float part = 0.0f;
#pragma unroll
              for (int d = 0; d < 4; d++) {
                float f[4];
                if constexpr (I8) {
#pragma unroll
                  for (int q = 0; q < 4; q++) f[q] = (float)((int)(cw[d] << (24 - 8 * q)) >> 24);
                } else {
                  const v2f lo = r4d_fp8::e4m3_pk<false>(cw[d]), hi = r4d_fp8::e4m3_pk<true>(cw[d]);
                  f[0] = lo.x; f[1] = lo.y; f[2] = hi.x; f[3] = hi.y;
                }
                part = fmaf(f[0], __uint_as_float(xw[2 * d] << 16), part);
                part = fmaf(f[1], __uint_as_float(xw[2 * d] & 0xffff0000u), part);
                part = fmaf(f[2], __uint_as_float(xw[2 * d + 1] << 16), part);
                part = fmaf(f[3], __uint_as_float(xw[2 * d + 1] & 0xffff0000u), part);
              }
              acc = fmaf(part, r4d_fp8::bscale(sv[u]), acc);
            }
          }
        }
      }
      for (long k0 = 16 * lane; ord && !vec && k0 < K; k0 += 512) {
        float part = 0.0f;
#pragma unroll
        for (int q = 0; q < 16; q++) {
          const long k = k0 + q;
          float c;
          if constexpr (TB != 0) {
            const long bit = k * (long)TB;
            const uint32_t wv = (uint32_t)wr[bit >> 3] | (uint32_t)wr[(bit >> 3) + 1] << 8;
            c = (float)tab[(wv >> (bit & 7)) & ((1u << TB) - 1u)];
          } else {
            c = I8 ? (float)(int8_t)wr[k] : r4d_fp8::e4m3((uint32_t)wr[k]);
          }
          part = fmaf(c, bf16_to_f32(xr[k]), part);
        }
        acc = fmaf(part, r4d_fp8::bscale(sr[k0 / 128]), acc);
      }
      for (long k = ord ? K : lane; k < K; k += R4D_TOPK_WAVE) {
        float c;
        if constexpr (TB != 0) {
          const long bit = k * (long)TB;
          const uint32_t wv = (uint32_t)wr[bit >> 3] | (uint32_t)wr[(bit >> 3) + 1] << 8;
          c = (float)tab[(wv >> (bit & 7)) & ((1u << TB) - 1u)];
        } else {
          c = I8 ? (float)(int8_t)wr[k] : r4d_fp8::e4m3((uint32_t)wr[k]);
        }
        acc = fmaf(c * r4d_fp8::bscale(sr[k / 128]), bf16_to_f32(xr[k]), acc);
      }
      acc = r4d_wave_sum(acc);
    }
    if (lane == 0) { s_sc[r] = own ? acc : -INFINITY; s_id[r] = own ? gid : -1; }
  }
  __syncthreads();
  if (wv != 0) return;

  unsigned long long t[R4D_TOPK_PT2];
#pragma unroll
  for (int q = 0; q < R4D_TOPK_PT2; q++) t[q] = 0ull;
  if (lane < R && s_id[lane] >= 0)
    tk_insert<R4D_TOPK_PT2>(t, tk_key_f32(s_sc[lane], (unsigned)s_id[lane]));

  for (int r = 0; r < R; r++) {
    const unsigned long long w = tk_round<R4D_TOPK_PT2>(t);
    if (lane == 0) {
      const float v = w ? tk_val(w) : -INFINITY;
      const int   i = w ? tk_idx(w) : -1;
      OV[(size_t)m * R + r] = v;
      OI[(size_t)m * R + r] = i;
      if (OP) {
        float* const p = OP + ((size_t)m * R + r) * 2;
        __builtin_memcpy(p, &i, sizeof i);
        p[1] = v;
      }
    }
  }
}

// The kernel around the body: CH `r4dchain` is the chained form (r4d_chain.h), taken at decode
// heights.
template <bool I8, uint32_t TB, class CH>
__global__ __launch_bounds__(R4D_TOPK_WAVE * R4D_RERANK_WPB)
void r4d_logit_rerank_kernel(R4D_LOGIT_RERANK_PARAMS) {
  if (CH::on) r4d_chain_wait();
  r4d_logit_rerank_body<I8, TB>(X, IN, W, OI, OV, OP, R, K, voff, nv, w_ld);
  if (CH::on) r4d_chain_done();
}
#undef R4D_LOGIT_RERANK_PARAMS

static int logit_rerank_emit(long x, long idx_in, long head, long outi, long outv, long outp,
                             int M, int R, long K, long voff, long nv, RadStream stream,
                             bool i8, uint32_t tb = 0, bool chain = false) {
  r4d_iqr::Form tf;
  tf.bits = tb;
  const long w_ld = tb ? (long)r4d_iqr::lm_row(tf, K) : r4d_fp8::lm_row_bytes(K);
  // R <= PT2 makes the re-sort exact for any scores (one lane holds the whole answer), so it is
  // checked, not clamped.
  if (M < 1 || R < 1 || K < 1 || nv < 1 || R > R4D_TOPK_PT2) return R4D_NO(RAD_E_SHAPE);
#define R4D_RERANK_CH(I8, TB, CH_)                                                              \
  R4D_EMIT(stream, (r4d_logit_rerank_kernel<I8, TB, CH_>), dim3(M),                              \
           dim3(R4D_TOPK_WAVE * R4D_RERANK_WPB), 0, (const unsigned short*)x,                    \
           (const int*)idx_in, (const unsigned char*)head, (int*)outi, (float*)outv,             \
           (float*)outp, R, K, voff, nv, w_ld)
#define R4D_RERANK_EMIT(I8, TB)                                                                 \
  (chain ? R4D_RERANK_CH(I8, TB, r4dchain) : R4D_RERANK_CH(I8, TB, R4dUnchained))
  const int rc = tb == 2u ? R4D_RERANK_EMIT(true, 2) : tb == 3u ? R4D_RERANK_EMIT(true, 3)
               : tb == 4u ? R4D_RERANK_EMIT(true, 4) : tb == 5u ? R4D_RERANK_EMIT(true, 5)
               : tb == 6u ? R4D_RERANK_EMIT(true, 6)
               : i8 ? R4D_RERANK_EMIT(true, 0) : R4D_RERANK_EMIT(false, 0);
#undef R4D_RERANK_EMIT
#undef R4D_RERANK_CH
  return rc;
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h. See libr4d/r4d_args.h on why there
// is no adapter layer.
/* ------------------------------------------------------------------ row top-k */

enum { T_X = 0, T_IDX, T_VAL, T_PAIRS };

/* Also the scratch sizer's reader, which runs at declare with no tensors and no constraint check, so
 * it repeats the row's R <= 32. */
static int topk_params(const RadArgs* a,
                       const RadTensor** x, const RadTensor** idx, const RadTensor** val,
                       long long* M, long long* N, long long* R, int* RC, int* NCH, int* PT1) {
    *x   = r4d_opt(a, T_X);
    *idx = r4d_opt(a, T_IDX);
    *val = r4d_opt(a, T_VAL);
    /* The scratch sizer runs at declare, with no tensors, through this same function. `M` and `N`
     * are declared parameters of this op for that reason (spec §2.1) and serve as the fallback;
     * when the tensor is present it wins, since a band ceiling is only an upper bound. */
    if (*x) {
        if ((*x)->rank != 2 || !r4d_contig(*x)) return R4D_NO(RAD_E_STRIDE);
        *M = (*x)->shape[0];
        *N = (*x)->shape[1];
    } else {
        *M = r4d_geti_or(a, "M", 0);
        *N = r4d_geti_or(a, "N", 0);
    }
    *R = r4d_geti_or(a, "R", 0);
    if (*M < 1 || *N < 1) return R4D_NO(RAD_E_SHAPE);
    if (*N % 8) return R4D_NO(RAD_E_SHAPE);
    if (*R < 1 || *R > 32) return R4D_NO(RAD_E_SHAPE);

    R4dTopkCfg tk;
    r4d_cfg_topk(a, &tk);
    const R4dTopkCfg* c = &tk;
    *PT1 = c->pt1;
    if (*PT1 != 2 && *PT1 != 4 && *PT1 != 8 && *PT1 != 16) return R4D_NO(RAD_E_INVAL);
    *RC  = c->rc > 0 ? c->rc : (int)*R;
    if (*RC < (int)*R) return R4D_NO(RAD_E_INVAL);          /* a chunk must keep at least R */
    if (*RC > *PT1 * 32) return R4D_NO(RAD_E_SHAPE);
    /* NCH is a requested chunk count. 0 means one chunk per 2048 columns, at least one, the shape
     * the MoE router this kernel serves runs. */
    *NCH = c->nch > 0 ? c->nch : (int)((*N + 2047) / 2048);
    if (*NCH < 1) *NCH = 1;
    return RAD_OK;
}

extern "C" int64_t r4d_rad_rowtopk_scratch(const RadArgs* a) {
    const RadTensor *x, *idx, *val;
    long long M, N, R;
    int RC, NCH, PT1;
    int rc = topk_params(a, &x, &idx, &val, &M, &N, &R, &RC, &NCH, &PT1);
    if (rc != RAD_OK) return rc;
    /* The chunk width is rounded inside libr4d, so the count is asked for, not recomputed. */
    const int nch = r4d_rowtopk_bf16_chunks((int)N, NCH);
    return (int64_t)M * nch * RC * (int64_t)sizeof(unsigned long long);
}

extern "C" int r4d_rad_rowtopk(const RadArgs* a, RadStream stream) {
    const RadTensor *x, *idx, *val;
    long long M, N, R;
    int RC, NCH, PT1;
    int rc = topk_params(a, &x, &idx, &val, &M, &N, &R, &RC, &NCH, &PT1);
    if (rc != RAD_OK) return rc;
    if (!x || !idx || !val) return R4D_NO(RAD_E_INVAL);
    if (!r4d_contig(idx) || !r4d_contig(val)) return R4D_NO(RAD_E_STRIDE);

    const int nch = r4d_rowtopk_bf16_chunks((int)N, NCH);
    const int64_t need = (int64_t)M * nch * RC * (int64_t)sizeof(unsigned long long);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);

    /* The vocab-parallel form; both halves optional. `vocab_off` shifts the reported columns to
     * global ids, and `pairs` is the [M, R, 2] plane an all_gather moves. Absent and zero is the
     * single-rank launch, bit for bit. */
    const RadTensor* pairs = r4d_opt(a, T_PAIRS);
    const long long off = r4d_geti_or(a, "vocab_off", 0);
    if (off < 0) return R4D_NO(RAD_E_SHAPE);
    if (pairs) {
        if (pairs->dtype != RAD_F32 || !r4d_contig(pairs)) return R4D_NO(RAD_E_STRIDE);
        if (r4d_numel(pairs) < M * R * 2) return R4D_NO(RAD_E_SHAPE);
    }

    /* The chained forms at decode heights, where the row resolved with `chain`. */
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    return rowtopk_bf16_sharded_emit(r4d_addr(x), (long)(uintptr_t)a->scratch, r4d_addr(val),
                                     r4d_addr(idx), pairs ? r4d_addr(pairs) : 0,
                                     (int)M, (int)N, (int)R, RC, NCH, PT1, (int)off, stream, ch);
}

/* ------------------------------------------------------------------ row_topk_merge
 *
 * The R largest of the all-gathered per-rank candidate sets. No variants and no scratch: one wave
 * per row over world_size * R pairs, so its cost is its launch; it only has to be exact. */
extern "C" int r4d_rad_rowtopk_merge(const RadArgs* a, RadStream s) {
    const RadTensor* g   = r4d_opt(a, 0);
    const RadTensor* idx = r4d_opt(a, 1);
    const RadTensor* val = r4d_opt(a, 2);
    if (!g || !idx || !val) return R4D_NO(RAD_E_INVAL);
    if (g->dtype != RAD_F32 || !r4d_contig(g)) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(idx) || !r4d_contig(val)) return R4D_NO(RAD_E_STRIDE);

    /* M comes from the operand: the `M` parameter is a band ceiling, and the gathered plane is
     * narrowed to the step's rows, so sizing the checks by the parameter would refuse real calls.
     * The parameter is the fallback for a sizing query made before there are tensors. */
    const long long M = (g->rank >= 1) ? g->shape[0] : r4d_geti_or(a, "M", 0);
    const long long R = r4d_geti_or(a, "R", 0);
    const long long W = r4d_geti_or(a, "world_size", 0);
    if (M < 1 || R < 1 || W < 1) return R4D_NO(RAD_E_SHAPE);
    if (R > 32) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(g) < M * W * R * 2) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(idx) < M * R || r4d_numel(val) < M * R) return R4D_NO(RAD_E_SHAPE);

    /* R <= PT2 makes the merge exact for any candidates (one lane can hold the whole answer). */
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    return R4D_EMIT(s, (ch ? r4dchain_rowtopk_merge : r4d_rowtopk_merge_kernel), dim3((unsigned)M),
                    dim3(R4D_TOPK_WAVE), 0, (const float*)g->data, (float*)val->data,
                    (int*)idx->data, (int)R, (int)(W * R));
}

/* `M` and the candidate count come from the operands, as in rowtopk_merge: `M` is a band ceiling.
 * `n_vocab` is this rank's rows of the head, so the ownership guard is a shard test. */
static int rad_logit_rerank(const RadArgs* a, RadStream stream, bool i8, bool iqr = false) {
    const RadTensor* x   = r4d_opt(a, 0);
    const RadTensor* w   = r4d_opt(a, 1);
    const RadTensor* in  = r4d_opt(a, 2);
    const RadTensor* idx = r4d_opt(a, 3);
    const RadTensor* val = r4d_opt(a, 4);
    const RadTensor* pr  = r4d_opt(a, 5);
    if (!x || !in || !w || !idx || !val) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    /* The declared dtype arrives, which for this weight is the trunk's bf16; the E4M3 (or int8)
     * plane is the stored form. Its stored width cannot be checked here and follows the layout
     * hook's rule; the engine's layout-tag check guarantees agreement by refusing a container laid
     * out for another row. So the head is accepted at either spelling and read as r4d.fp8lm.b128,
     * or r4d.i8lm.r128 for the int8 form. */
    uint32_t tb = 0;
    if (iqr) {
        if (!r4d_iqr::is_weight_dtype(w->dtype) || !r4d_iqr::lm_built(w->dtype & r4d_iqr::kWordMask))
            return R4D_NO(RAD_E_DTYPE);
        tb = r4d_iqr::form(w->dtype & r4d_iqr::kWordMask).bits;
    } else if (w->dtype != (i8 ? RAD_I8 : RAD_F8E4M3) && w->dtype != RAD_BF16) {
        return R4D_NO(RAD_E_DTYPE);
    }
    if (in->dtype != RAD_I32 || idx->dtype != RAD_I32 || val->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (pr && pr->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(x) || !r4d_contig(in) || !r4d_contig(w) || !r4d_contig(idx) ||
        !r4d_contig(val) || (pr && !r4d_contig(pr))) return R4D_NO(RAD_E_STRIDE);

    const long long K    = r4d_geti_or(a, "n_embd", 0);
    const long long NV   = r4d_geti_or(a, "n_vocab", 0);
    const long long R    = r4d_geti_or(a, "R", 0);
    const long long voff = r4d_geti_or(a, "vocab_off", 0);
    if (K < 1 || NV < 1 || R < 1) return R4D_NO(RAD_E_SHAPE);
    if (R > 32) return R4D_NO(RAD_E_SHAPE);
    const long long M = (x->rank >= 1) ? x->shape[0] : r4d_geti_or(a, "M", 0);
    if (M < 1) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(x) < M * K || r4d_numel(in) < M * R) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(idx) < M * R || r4d_numel(val) < M * R) return R4D_NO(RAD_E_SHAPE);
    if (pr && r4d_numel(pr) < M * R * 2) return R4D_NO(RAD_E_SHAPE);
    if (K % 128) return R4D_NO(RAD_E_SHAPE);   /* the scale block; lm_row_bytes assumes it */

    return logit_rerank_emit(r4d_addr(x), r4d_addr(in), r4d_addr(w), r4d_addr(idx), r4d_addr(val),
                             pr ? r4d_addr(pr) : 0, (int)M, (int)R, (long)K, (long)voff, (long)NV,
                             stream, i8, tb, r4d_geti_or(a, "chain", 0) != 0 && M <= 64);
}
extern "C" int r4d_rad_logit_rerank_i8(const RadArgs* a, RadStream stream) {
    return rad_logit_rerank(a, stream, true);
}
/* An IQR head's rows (r4d_iqr.h's lm form), as logits_gemm_iqr reads them. */
extern "C" int r4d_rad_logit_rerank_iqr(const RadArgs* a, RadStream stream) {
    return rad_logit_rerank(a, stream, true, true);
}

