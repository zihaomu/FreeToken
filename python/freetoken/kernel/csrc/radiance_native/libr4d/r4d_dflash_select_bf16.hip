// DFlash2's candidate selector: edge scores and the path walk, one workgroup per sequence.
//
// A block-diffusion drafter proposes every position of a block in one backbone pass, so position
// l+1 cannot condition on the choice at l. The selector restores that dependence: each position
// keeps the draft head's top K candidates, a low-rank bilinear form scores each (predecessor,
// candidate) edge between consecutive positions, and a greedy walk picks one token per position.
//
//   score[s, l, c] = unary[s, l, c]
//                  + sum_r  pred[pid(s,l)][r] * hp[s,l][r] * succ[cand[s,l,c]][r]
//
//   pid(s, 0) = anchor[s * anchor_stride]           the token the target just committed
//   pid(s, l) = cand[s, l-1, chosen(s, l-1)]       the walk's own previous pick
//
// `hp` is optional; absent means ones, giving the bigram form score = unary + <pred[pid],
// succ[cand]>. That is DSpark's Markov head (a rank-256 learned bigram bias
// markov_w2 @ markov_w1[prev] on the draft logits, prev = anchor at position 0, then the last
// pick). The reference runs it as a full-vocabulary [rank, n_vocab] GEMM per position, serially;
// here it costs only the candidate rows of two codebooks. Restricting to the base head's top K
// affects only acceptance, since drafts are verified.
//
// The walk is serial in l, which is why this is a kernel rather than a batched GEMM plus argmax.
// The reference (vLLM's _score_edges einsum) materialises [steps, K, K] scores and reads one row
// per step, since the predecessor is only known after the previous step. Here the predecessor's
// codebook row is fetched after the previous argmax, inside one workgroup.
//
// The arithmetic is tiny (steps * K * R multiply-adds per sequence, 28K at 7/16/256), so this is a
// latency kernel: one workgroup per sequence, no global synchronisation.
#include <hip/hip_runtime.h>
#include "r4d_gdn_args.h"
#include "r4d_dflash_conv_body.h"   // r4d_bf2f, r4d_u16
#include "r4d_common.h"   // r4d_wave_sum
#include "r4d_chain.h"

#define DSL_THREADS 256
#define DSL_WAVE    32
#define DSL_WAVES   (DSL_THREADS / DSL_WAVE)
#define DSL_MAX_K   32              // the entry point rejects more; ssc is sized for it
#define DSL_MAX_R   512             // and sa for this

// cand   [rows, cand_ld] i32   global token ids, row = s * steps + l
// unary  [rows, cand_ld] f32   the draft head's logit for each candidate (row_topk's value plane)
// hp     [rows, hp_ld]   bf16  hidden_projection(x) for the same row
// anchor [M * anchor_stride] i32  the last committed token of each sequence, at stride s
// pred   [vocab, R]      bf16  predecessor codebook
// succ   [vocab, R]      bf16  successor codebook
// out    [M, out_ld]     i32   the chosen token per step
//
// Ids outside [0, V) do occur (row_topk_merge reports an unfilled slot as -1, and step 0's
// predecessor is whatever the anchor plane holds). As in ref_dflash_select, an out-of-range
// predecessor contributes nothing and an out-of-range candidate scores its unary term alone.
// The body; the plain kernel and its chained form (r4d_chain.h) wrap it below.
__device__ __forceinline__ void r4d_dflash_select_body(
    const int* __restrict__ cand, const float* __restrict__ unary,
    const r4d_u16* __restrict__ hp, const int* __restrict__ anchor,
    const r4d_u16* __restrict__ pred, const r4d_u16* __restrict__ succ,
    int* __restrict__ out, int steps, int K, int R, int V,
    int cand_ld, int hp_ld, int out_ld, int anchor_stride)
{
  const int s    = blockIdx.x;
  const int tid  = threadIdx.x;
  const int lane = tid & (DSL_WAVE - 1);
  const int wave = tid / DSL_WAVE;

  // a[r] = pred[pid][r] * hp[r]: the candidate-independent half, computed once per step.
  __shared__ float sa[DSL_MAX_R];
  __shared__ float ssc[DSL_MAX_K];
  __shared__ int   spid;            // the token the previous step chose: this step's predecessor

  // Candidates per wave, at most: c = wave, wave + DSL_WAVES, ...
  constexpr int CPW = (DSL_MAX_K + DSL_WAVES - 1) / DSL_WAVES;

  for (int l = 0; l < steps; ++l) {
    const long row = (long)s * steps + l;
    // Candidate ids and unary scores do not depend on the predecessor, so load them first,
    // overlapping the codebook row fetch. A wave past K re-reads the last candidate and scores
    // nothing.
    int   cid[CPW];
    float un[CPW];
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
      const int c = wave + i * DSL_WAVES;
      const long at = row * cand_ld + (c < K ? c : K - 1);
      cid[i] = cand[at];
      un[i]  = unary[at];
    }
    // Step 0's predecessor is the committed anchor; later steps use the previous pick, left in
    // LDS by thread 0 before the barrier that ended that step.
    const int pid = (l == 0) ? anchor[(long)s * anchor_stride] : spid;
    const bool pid_ok = pid >= 0 && pid < V;

    const r4d_u16* __restrict__ prow = pred + (long)(pid_ok ? pid : 0) * R;
    /* hp absent means ones: the predecessor's row is used as is, without reading a plane. */
    if (!pid_ok) {
      for (int r = tid; r < R; r += DSL_THREADS) sa[r] = 0.0f;
    } else if (hp) {
      const r4d_u16* __restrict__ hrow = hp + row * hp_ld;
      for (int r = tid; r < R; r += DSL_THREADS)
        sa[r] = r4d_bf2f(prow[r]) * r4d_bf2f(hrow[r]);
    } else {
      for (int r = tid; r < R; r += DSL_THREADS)
        sa[r] = r4d_bf2f(prow[r]);
    }
    __syncthreads();

    // One wave per candidate. R is a multiple of the wave size, so the reduction is a barrier-free
    // xor butterfly. An out-of-range candidate reads row 0 and drops the sum, keeping the loads
    // out of a branch.
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
      const int c = wave + i * DSL_WAVES;
      if (c < K) {
        const bool ok = cid[i] >= 0 && cid[i] < V;
        const r4d_u16* __restrict__ srow = succ + (long)(ok ? cid[i] : 0) * R;
        float acc = 0.0f;
#pragma unroll 8
        for (int r = lane; r < R; r += DSL_WAVE) acc += sa[r] * r4d_bf2f(srow[r]);
        acc = r4d_wave_sum(ok ? acc : 0.0f);
        if (lane == 0) ssc[c] = acc + un[i];
      }
    }
    __syncthreads();

    // K <= 32 and the walk is serial, so one thread does the argmax. Ties go to the lower index
    // (row_topk's rule, the more likely candidate). Seeded at -inf, as in ref, so a NaN in the
    // first slot cannot win by position.
    if (tid == 0) {
      int   best = 0;
      float bv   = -INFINITY;
      for (int c = 0; c < K; ++c)
        if (ssc[c] > bv) { bv = ssc[c]; best = c; }
      const int tok = cand[row * cand_ld + best];
      spid = tok;
      out[(long)s * out_ld + l] = tok;
    }
    // The next step reads spid and overwrites sa, which other threads may still be reading.
    __syncthreads();
  }
}
__global__ __launch_bounds__(DSL_THREADS) void r4d_dflash_select_kernel(
    const int* __restrict__ cand, const float* __restrict__ unary,
    const r4d_u16* __restrict__ hp, const int* __restrict__ anchor,
    const r4d_u16* __restrict__ pred, const r4d_u16* __restrict__ succ,
    int* __restrict__ out, int steps, int K, int R, int V,
    int cand_ld, int hp_ld, int out_ld, int anchor_stride) {
    r4d_dflash_select_body(cand, unary, hp, anchor, pred, succ, out, steps, K, R, V, cand_ld, hp_ld,
                           out_ld, anchor_stride);
}
__global__ __launch_bounds__(DSL_THREADS) void r4dchain_dflash_select(
    const int* __restrict__ cand, const float* __restrict__ unary,
    const r4d_u16* __restrict__ hp, const int* __restrict__ anchor,
    const r4d_u16* __restrict__ pred, const r4d_u16* __restrict__ succ,
    int* __restrict__ out, int steps, int K, int R, int V,
    int cand_ld, int hp_ld, int out_ld, int anchor_stride) {
    r4d_chain_wait();
    r4d_dflash_select_body(cand, unary, hp, anchor, pred, succ, out, steps, K, R, V, cand_ld, hp_ld,
                           out_ld, anchor_stride);
    r4d_chain_done();
}

/* `chain`: the chained form (the plugin entry, where the row resolved with `chain`). */
static int dflash_select_emit(
    const void* cand, const void* unary, const void* hp, const void* anchor,
    const void* pred, const void* succ, void* out,
    int M, int steps, int K, int R, int V, int cand_ld, int hp_ld, int out_ld,
    int anchor_stride, void* stream, bool chain);

extern "C" int r4d_dflash_select_bf16(
    const void* cand, const void* unary, const void* hp, const void* anchor,
    const void* pred, const void* succ, void* out,
    int M, int steps, int K, int R, int V, int cand_ld, int hp_ld, int out_ld,
    int anchor_stride, void* stream)
{
  return dflash_select_emit(cand, unary, hp, anchor, pred, succ, out, M, steps, K, R, V, cand_ld,
                            hp_ld, out_ld, anchor_stride, stream, false);
}

static int dflash_select_emit(
    const void* cand, const void* unary, const void* hp, const void* anchor,
    const void* pred, const void* succ, void* out,
    int M, int steps, int K, int R, int V, int cand_ld, int hp_ld, int out_ld,
    int anchor_stride, void* stream, bool chain)
{
  if (M <= 0 || steps <= 0) return 0;
  if (K < 1 || K > DSL_MAX_K) return -1;
  if (R < DSL_WAVE || R > DSL_MAX_R || (R % DSL_WAVE)) return -1;
  if (V < 1) return -1;
  if (cand_ld < K || out_ld < steps || anchor_stride < 1) return -1;
  /* hp may be null (the bigram form); then hp_ld is unused. */
  if (hp && hp_ld < R) return -1;

  hipLaunchKernelGGL((chain ? r4dchain_dflash_select : r4d_dflash_select_kernel), dim3(M),
                     dim3(DSL_THREADS), 0, (hipStream_t)stream,
                     (const int*)cand, (const float*)unary, (const r4d_u16*)hp,
                     (const int*)anchor, (const r4d_u16*)pred, (const r4d_u16*)succ,
                     (int*)out, steps, K, R, V, cand_ld, hp_ld, out_ld, anchor_stride);
  return (int)hipGetLastError();
}

// ============================================================== the plugin ABI entry points
// Shared operand helpers: r4d_gdn_args.h.
/* ------------------------------------------------------------------ dflash select */

enum { DS_CAND = 0, DS_UNARY, DS_HP, DS_ANCHOR, DS_PRED, DS_SUCC, DS_OUT };

/* A rank-2 [rows, width] operand with an explicit row pitch. `width` is checked, not derived:
 * three planes are read at the same row index, and a wrong-width plane could still have a
 * plausible row count. */
static long long ds_plane(const RadTensor* t, uint32_t dtype, long long width, long long rows) {
    if (!t || !t->data || t->rank != 2) return -1;
    if (t->dtype != dtype) return -1;
    if (t->shape[1] < width || t->shape[0] < rows) return -1;
    const long long pitch = r4d_row_pitch(t);
    return (pitch >= width) ? pitch : -1;
}

extern "C" int r4d_rad_dflash_select(const RadArgs* a, RadStream s) {
    const RadTensor* cand = r4d_opt(a, DS_CAND);
    const RadTensor* un   = r4d_opt(a, DS_UNARY);
    const RadTensor* hp   = r4d_opt(a, DS_HP);
    const RadTensor* anc  = r4d_opt(a, DS_ANCHOR);
    const RadTensor* pr   = r4d_opt(a, DS_PRED);
    const RadTensor* sc   = r4d_opt(a, DS_SUCC);
    const RadTensor* out  = r4d_opt(a, DS_OUT);
    /* Only hp may be absent. */
    if (!cand || !un || !anc || !pr || !sc || !out) return R4D_NO(RAD_E_INVAL);

    const long long steps = r4d_geti_or(a, "steps", 0);
    const long long K     = r4d_geti_or(a, "top_k", 0);
    const long long R     = r4d_geti_or(a, "rank", 0);
    const long long V     = r4d_geti_or(a, "n_vocab", 0);
    /* M comes from the output plane (one row per sequence), not the parameter: RadArgs carries the
     * band's bound (max_seqs) for M, as for q_len in fill_paged. The parameter would demand
     * max_seqs * steps candidate rows on a one-sequence step. */
    if (out->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = out->shape[0];
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || V <= 0) return R4D_NO(RAD_E_INVAL);

    /* The input planes are read at row s*steps+l and must hold M*steps rows; a short plane would
     * silently draft from another request's candidates. */
    const long long rows    = M * steps;
    const long long cand_ld = ds_plane(cand, RAD_I32,  K, rows);
    const long long un_ld   = ds_plane(un,   RAD_F32,  K, rows);
    const long long out_ld  = ds_plane(out,  RAD_I32,  steps, M);
    /* Absent hp may arrive as a missing operand or as an empty one with null data (operands are
     * positional); both mean the bigram form. */
    const bool     has_hp = hp && hp->data;
    const long long hp_ld = has_hp ? ds_plane(hp, RAD_BF16, R, rows) : 0;
    if (cand_ld < 0 || un_ld < 0 || hp_ld < 0 || out_ld < 0) return R4D_NO(RAD_E_SHAPE);
    /* The kernel indexes cand and unary with one pitch. */
    if (cand_ld != un_ld) return R4D_NO(RAD_E_STRIDE);

    /* The codebooks are gathered by global token id, and candidates arrive merged across ranks, so
     * both must be replicated over the whole vocabulary; a row-sharded one would fail silently. */
    if (pr->dtype != RAD_BF16 || sc->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(pr) || !r4d_contig(sc)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(pr) < V * R || r4d_numel(sc) < V * R) return R4D_NO(RAD_E_SHAPE);

    /* anchor is usually the step's token_ids (`block` ids per sequence, anchor first), hence the
     * stride; a packed [M] array is stride 1. */
    const long long astr = r4d_geti_or(a, "anchor_stride", 1);
    if (anc->dtype != RAD_I32 && anc->dtype != RAD_U32) return R4D_NO(RAD_E_DTYPE);
    if (astr < 1 || !r4d_contig(anc)) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(anc) < (M - 1) * astr + 1) return R4D_NO(RAD_E_SHAPE);

    /* The chained form at decode heights, where the row resolved with `chain`. */
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    return R4D_RAW(dflash_select_emit(cand->data, un->data, has_hp ? hp->data : nullptr,
                                      anc->data, pr->data, sc->data,
                                      out->data, (int)M, (int)steps, (int)K, (int)R, (int)V,
                                      (int)cand_ld, (int)hp_ld, (int)out_ld, (int)astr, s, ch));
}
