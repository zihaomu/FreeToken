// DFlash2's candidate lattice grown best-first into a tree (rad_drafter_tree), one workgroup per
// sequence. docs/OPS.md, `dflash_tree`, is the contract; libref's ref_dflash_tree the reference.
//
// The selector (r4d_dflash_select_bf16.hip) walks one path: at each position the candidate with the
// best edge score after the previous pick. A tree keeps more than one way forward where the walk is
// unsure. Node 0 is the anchor. Expanding a node at lattice position l with token t scores every
// candidate c at l + 1 exactly as the walk scores it after t,
//
//   score[c] = unary[l+1, c] + sum_r pred[t][r] * hp[l+1][r] * succ[cand[l+1, c]][r]
//
// and pushes each valid candidate as a child with log probability parent + log_softmax(score/temp)
// over the valid candidates. Each round takes the untaken child with the largest log probability
// (ties to the one pushed first), until `nodes` are taken; the taken nodes, in order, are the
// proposal, each with its parent's node index. A parent is always taken before its child.
//
// The same latency-kernel shape as the walk: `nodes` expansions of K dot products of R each, and a
// reduction over a frontier of at most nodes * K entries per round, all in LDS.
#include <hip/hip_runtime.h>
#include "r4d_gdn_args.h"
#include "r4d_dflash_conv_body.h"   // r4d_bf2f, r4d_u16
#include "r4d_common.h"   // r4d_wave_sum
#include "r4d_chain.h"

#define DTR_THREADS 256
#define DTR_WAVE    32
#define DTR_WAVES   (DTR_THREADS / DTR_WAVE)
#define DTR_MAX_K   32              // the entry point rejects more
#define DTR_MAX_R   512
#define DTR_MAX_N   63              // nodes and steps: the node table and the frontier below
#define DTR_FRONT   (DTR_MAX_N * DTR_MAX_K)

// cand   [rows, cand_ld] i32   row = s * steps + l
// unary  [rows, cand_ld] f32
// hp     [rows, hp_ld]   bf16  (null: ones, the bigram form)
// anchor [M * anchor_stride] i32
// pred, succ [vocab, R] bf16
// tok, par   [M, out_ld] i32   the taken nodes' tokens and parents, nodes a row
__device__ __forceinline__ void r4d_dflash_tree_body(
    const int* __restrict__ cand, const float* __restrict__ unary,
    const r4d_u16* __restrict__ hp, const int* __restrict__ anchor,
    const r4d_u16* __restrict__ pred, const r4d_u16* __restrict__ succ,
    int* __restrict__ tok_out, int* __restrict__ par_out, int steps, int K, int R, int V, int N,
    float inv_temp, int cand_ld, int hp_ld, int out_ld, int anchor_stride)
{
  const int s    = blockIdx.x;
  const int tid  = threadIdx.x;
  const int lane = tid & (DTR_WAVE - 1);
  const int wave = tid / DTR_WAVE;

  __shared__ float sa[DTR_MAX_R];
  __shared__ float ssc[DTR_MAX_K];
  // the frontier: every child pushed so far, in push order; a taken one's lp becomes -inf
  __shared__ float f_lp[DTR_FRONT];
  __shared__ int   f_tok[DTR_FRONT];
  __shared__ short f_par[DTR_FRONT];
  __shared__ short f_l[DTR_FRONT];
  __shared__ int   n_front;
  // the taken nodes, node 0 the anchor
  __shared__ int   n_tok[DTR_MAX_N + 1];
  __shared__ int   n_l[DTR_MAX_N + 1];
  __shared__ float n_lp[DTR_MAX_N + 1];
  // the round's reduction
  __shared__ float r_lp[DTR_WAVES];
  __shared__ int   r_ix[DTR_WAVES];
  __shared__ int   take;

  constexpr int CPW = (DTR_MAX_K + DTR_WAVES - 1) / DTR_WAVES;

  if (tid == 0) {
    n_tok[0] = anchor[(long)s * anchor_stride];
    n_l[0]   = -1;
    n_lp[0]  = 0.0f;
    n_front  = 0;
  }
  __syncthreads();

  // Push node n's children: every valid candidate at its next position. Uniform across the block.
  auto expand = [&](int n) {
    const int l = n_l[n] + 1;
    if (l >= steps) return;
    const long row = (long)s * steps + l;
    const int pid = n_tok[n];
    const bool pid_ok = pid >= 0 && pid < V;
    const r4d_u16* __restrict__ prow = pred + (long)(pid_ok ? pid : 0) * R;
    if (!pid_ok) {
      for (int r = tid; r < R; r += DTR_THREADS) sa[r] = 0.0f;
    } else if (hp) {
      const r4d_u16* __restrict__ hrow = hp + row * hp_ld;
      for (int r = tid; r < R; r += DTR_THREADS) sa[r] = r4d_bf2f(prow[r]) * r4d_bf2f(hrow[r]);
    } else {
      for (int r = tid; r < R; r += DTR_THREADS) sa[r] = r4d_bf2f(prow[r]);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
      const int c = wave + i * DTR_WAVES;
      if (c < K) {
        const long at = row * cand_ld + c;
        const int cid = cand[at];
        const bool ok = cid >= 0 && cid < V;
        const r4d_u16* __restrict__ srow = succ + (long)(ok ? cid : 0) * R;
        float acc = 0.0f;
#pragma unroll 8
        for (int r = lane; r < R; r += DTR_WAVE) acc += sa[r] * r4d_bf2f(srow[r]);
        acc = r4d_wave_sum(ok ? acc : 0.0f);
        if (lane == 0) ssc[c] = ok ? (acc + unary[at]) * inv_temp : -INFINITY;
      }
    }
    __syncthreads();
    // log_softmax over the valid candidates and the push, in c order, by one thread (K <= 32)
    if (tid == 0) {
      float mx = -INFINITY;
      for (int c = 0; c < K; ++c) mx = fmaxf(mx, ssc[c]);
      if (mx > -INFINITY) {
        float sum = 0.0f;
        for (int c = 0; c < K; ++c) if (ssc[c] > -INFINITY) sum += expf(ssc[c] - mx);
        const float lse = mx + logf(sum);
        int f = n_front;
        for (int c = 0; c < K && f < DTR_FRONT; ++c) {
          if (!(ssc[c] > -INFINITY)) continue;
          f_lp[f]  = n_lp[n] + (ssc[c] - lse);
          f_tok[f] = cand[row * cand_ld + c];
          f_par[f] = (short)n;
          f_l[f]   = (short)l;
          ++f;
        }
        n_front = f;
      }
    }
    __syncthreads();
  };

  expand(0);
  for (int j = 1; j <= N; ++j) {
    // the round: the largest untaken log probability, the earliest pushed among equals
    float best = -INFINITY;
    int   bix  = -1;
    const int nf = n_front;
    for (int f = tid; f < nf; f += DTR_THREADS) {
      const float v = f_lp[f];
      if (v > best) { best = v; bix = f; }      // ascending f: the first of equals stays
    }
    for (int d = DTR_WAVE / 2; d > 0; d >>= 1) {
      const float ov = __shfl_xor(best, d);
      const int   oi = __shfl_xor(bix, d);
      if (ov > best || (ov == best && oi >= 0 && (bix < 0 || oi < bix))) { best = ov; bix = oi; }
    }
    if (lane == 0) { r_lp[wave] = best; r_ix[wave] = bix; }
    __syncthreads();
    if (tid == 0) {
      float b = -INFINITY;
      int   bi = -1;
      for (int w = 0; w < DTR_WAVES; ++w)
        if (r_lp[w] > b || (r_lp[w] == b && r_ix[w] >= 0 && (bi < 0 || r_ix[w] < bi))) {
          b = r_lp[w]; bi = r_ix[w];
        }
      int t, p, l;
      float lp;
      if (bi < 0 || !(b > -INFINITY)) {
        // a dry frontier: token 0 under the last node, never expanded
        t = 0; p = j - 1; l = steps; lp = -INFINITY;
      } else {
        t = f_tok[bi]; p = f_par[bi]; l = f_l[bi]; lp = f_lp[bi];
        f_lp[bi] = -INFINITY;
      }
      n_tok[j] = t; n_l[j] = l; n_lp[j] = lp;
      tok_out[(long)s * out_ld + (j - 1)] = t;
      par_out[(long)s * out_ld + (j - 1)] = p;
      take = (bi >= 0 && b > -INFINITY) ? 1 : 0;
    }
    __syncthreads();
    if (take && j < N) expand(j);
  }
}

__global__ __launch_bounds__(DTR_THREADS) void r4d_dflash_tree_kernel(
    const int* __restrict__ cand, const float* __restrict__ unary,
    const r4d_u16* __restrict__ hp, const int* __restrict__ anchor,
    const r4d_u16* __restrict__ pred, const r4d_u16* __restrict__ succ,
    int* __restrict__ tok, int* __restrict__ par, int steps, int K, int R, int V, int N,
    float inv_temp, int cand_ld, int hp_ld, int out_ld, int anchor_stride) {
  r4d_dflash_tree_body(cand, unary, hp, anchor, pred, succ, tok, par, steps, K, R, V, N, inv_temp,
                       cand_ld, hp_ld, out_ld, anchor_stride);
}
__global__ __launch_bounds__(DTR_THREADS) void r4dchain_dflash_tree(
    const int* __restrict__ cand, const float* __restrict__ unary,
    const r4d_u16* __restrict__ hp, const int* __restrict__ anchor,
    const r4d_u16* __restrict__ pred, const r4d_u16* __restrict__ succ,
    int* __restrict__ tok, int* __restrict__ par, int steps, int K, int R, int V, int N,
    float inv_temp, int cand_ld, int hp_ld, int out_ld, int anchor_stride) {
  r4d_chain_wait();
  r4d_dflash_tree_body(cand, unary, hp, anchor, pred, succ, tok, par, steps, K, R, V, N, inv_temp,
                       cand_ld, hp_ld, out_ld, anchor_stride);
  r4d_chain_done();
}

// ============================================================== the plugin ABI entry point
enum { DT_CAND = 0, DT_UNARY, DT_HP, DT_ANCHOR, DT_PRED, DT_SUCC, DT_TOK, DT_PAR };

static long long dt_plane(const RadTensor* t, uint32_t dtype, long long width, long long rows) {
    if (!t || !t->data || t->rank != 2) return -1;
    if (t->dtype != dtype) return -1;
    if (t->shape[1] < width || t->shape[0] < rows) return -1;
    const long long pitch = r4d_row_pitch(t);
    return (pitch >= width) ? pitch : -1;
}

extern "C" int r4d_rad_dflash_tree(const RadArgs* a, RadStream st) {
    const RadTensor* cand = r4d_opt(a, DT_CAND);
    const RadTensor* un   = r4d_opt(a, DT_UNARY);
    const RadTensor* hp   = r4d_opt(a, DT_HP);
    const RadTensor* anc  = r4d_opt(a, DT_ANCHOR);
    const RadTensor* pr   = r4d_opt(a, DT_PRED);
    const RadTensor* sc   = r4d_opt(a, DT_SUCC);
    const RadTensor* tok  = r4d_opt(a, DT_TOK);
    const RadTensor* par  = r4d_opt(a, DT_PAR);
    if (!cand || !un || !anc || !pr || !sc || !tok || !par) return R4D_NO(RAD_E_INVAL);

    const long long steps = r4d_geti_or(a, "steps", 0);
    const long long K     = r4d_geti_or(a, "top_k", 0);
    const long long R     = r4d_geti_or(a, "rank", 0);
    const long long V     = r4d_geti_or(a, "n_vocab", 0);
    const long long N     = r4d_geti_or(a, "nodes", steps);
    const double    temp  = rad_args_getf_or(a, "temp", 1.0);
    /* M from the output plane, as dflash_select: the parameter is the band's bound. */
    if (tok->rank != 2 || par->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = tok->shape[0];
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || V <= 0 || N <= 0) return R4D_NO(RAD_E_INVAL);
    if (K > DTR_MAX_K || R < DTR_WAVE || R > DTR_MAX_R || (R % DTR_WAVE) || N > DTR_MAX_N ||
        steps > DTR_MAX_N || !(temp > 0.0))
        return R4D_NO(RAD_E_SHAPE);

    const long long rows    = M * steps;
    const long long cand_ld = dt_plane(cand, RAD_I32, K, rows);
    const long long un_ld   = dt_plane(un,   RAD_F32, K, rows);
    const long long out_ld  = dt_plane(tok,  RAD_I32, N, M);
    const long long par_ld  = dt_plane(par,  RAD_I32, N, M);
    const bool      has_hp  = hp && hp->data;
    const long long hp_ld   = has_hp ? dt_plane(hp, RAD_BF16, R, rows) : 0;
    if (cand_ld < 0 || un_ld < 0 || hp_ld < 0 || out_ld < 0 || par_ld < 0) return R4D_NO(RAD_E_SHAPE);
    if (cand_ld != un_ld || out_ld != par_ld) return R4D_NO(RAD_E_STRIDE);
    if (pr->dtype != RAD_BF16 || sc->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(pr) || !r4d_contig(sc)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(pr) < V * R || r4d_numel(sc) < V * R) return R4D_NO(RAD_E_SHAPE);
    const long long astr = r4d_geti_or(a, "anchor_stride", 1);
    if (anc->dtype != RAD_I32 && anc->dtype != RAD_U32) return R4D_NO(RAD_E_DTYPE);
    if (astr < 1 || !r4d_contig(anc)) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(anc) < (M - 1) * astr + 1) return R4D_NO(RAD_E_SHAPE);

    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    hipLaunchKernelGGL((ch ? r4dchain_dflash_tree : r4d_dflash_tree_kernel), dim3((unsigned)M),
                       dim3(DTR_THREADS), 0, (hipStream_t)st,
                       (const int*)cand->data, (const float*)un->data,
                       has_hp ? (const r4d_u16*)hp->data : nullptr, (const int*)anc->data,
                       (const r4d_u16*)pr->data, (const r4d_u16*)sc->data,
                       (int*)tok->data, (int*)par->data, (int)steps, (int)K, (int)R, (int)V, (int)N,
                       (float)(1.0 / temp), (int)cand_ld, (int)hp_ld, (int)out_ld, (int)astr);
    return R4D_RAW((int)hipGetLastError());
}
