// r4d_gdn_conv_w4_h128_bf16.hip -- the gated-delta-net depthwise causal convolution and the work fused around
// it, one entry point per serving phase.
//
// PREFILL (r4d_gdn_conv_prep): everything the layer does between the qkv projection and the
// chunked scan, in one kernel:
//
//     conv(x)        depthwise causal convolution, width 4, silu, varlen, with the conv-state
//                    cache read at a sequence's first chunk and rewritten from its last tokens
//     split          the packed [q | k | v] channel range into three contiguous tensors
//     l2norm         over each 128-dim q and k head
//     gate           g = -exp(A_log) . softplus(a + dt_bias), beta = sigmoid(b)
//     cumsum         g summed along the chunk, which is what the scan actually consumes
//
// Replaces causal_conv1d_fn + fused_post_conv_prep + chunk_local_cumsum. The unfused pair writes
// the whole conv output to HBM and reads it straight back (34 MB each way per layer per rank at
// this model's shape); here it stays in registers.
//
// Layout: a workgroup is HPB waves, one wave per 128-dim head, four dims per lane, so a head's
// norm is a barrier-free wave reduction and each lane does one 8-byte access. Channel groups are
// counted in heads so a group never straddles the q/k/v boundaries (head multiples at every TP).
//
// Tokens are blocked by the 64-token chunk because the cumsum is per chunk: it becomes a wave
// prefix over 32 lanes of two tokens each. Each block reads three tokens of halo (4.7% of the
// traffic), cheaper than a barrier around a shared staging buffer.
//
// The conv output is rounded to bf16 before the norm and the store, as in the unfused pair, so
// this kernel is a bit-exact drop-in for it.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_gdn_args.h"
#include "r4d_gdn_wmma.h"
#include "r4d_common.h"
#include "r4d_gdn_conv_tap.h"

#define CP_HD    128            // dims per head, q k and v alike
#define CP_BT     64            // token block = the scan's chunk, so the cumsum stays wave-local
#define CP_W       R4D_GDN_CONV_W   // conv width
#define CP_ST   (CP_W - 1)      // conv state depth
// Tokens the decode update preloads into registers. 16 is ~13% slower than 8 (register pressure
// and predicated-load overhead outweigh the hidden round trips).
//
// The xb[] preload is indexed with a compile-time t: a register array under a runtime index
// compiles to a select tree (about a third of the kernel's instructions at HPB=1). Both readers
// unroll over CU_MAXT with a wave-uniform `t < slen` guard, so every index is static and the guard
// is a scalar branch. The preload is issued before the history load: the history address depends
// on `slot` (a cache_idx load) and its values are consumed immediately, so otherwise the two
// memory round trips serialise, with no other wave on the SIMD to hide them.
#define CU_MAXT 8
#define CP_DPL     4            // dims per lane (CP_HD / 32)

// cp_silu is in r4d_gdn_conv_tap.h, shared with the recurrent update's folded convolution.

// softplus, in FLA's two-sided form with the same large-x cutoff
__device__ __forceinline__ float cp_softplus(float x, float thr) {
  if (x > thr) return x;
  return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x)))
                    : __logf(1.0f + __expf(x));
}

// a and b are read where they lie: a chunk() of the ba projection, bf16 with row stride 2*H (or
// fp32), avoiding a .float().contiguous() copy and two launches per layer.
template <int ABF16>
__device__ __forceinline__ float cp_ab(const void* p, size_t i) {
  return ABF16 ? bf2f(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}

// `has_init` says whether the slot holds a history; its element width (int32 or byte) is given by
// the caller. The engine passes RadBatch::ctx_lens (int32); read as bytes, a ctx_len of 512 would
// look like 0 and the conv window would silently start from zeros.
__device__ __forceinline__ bool r4d_has_init(const void* p, int i32, int n) {
  if (!p) return false;
  return i32 ? ((const int*)p)[n] != 0 : ((const unsigned char*)p)[n] != 0;
}

template <int HPB, int ABF16, typename WT>
__global__ __launch_bounds__(HPB * 32)
void r4d_gdn_conv_prep_kernel(const unsigned short* __restrict__ x, long xpitch,
                              const WT* __restrict__ wgt,
                              const unsigned short* __restrict__ bias,
                              unsigned short* __restrict__ cstate,
                              long cs_seq, long cs_dim, long cs_tok,
                              const int* __restrict__ cache_idx, long ci_stride,
                              const void* __restrict__ has_init, int hi_i32,
                              const void* __restrict__ av, const void* __restrict__ bv,
                              long ab_stride,
                              const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                              unsigned short* __restrict__ q, unsigned short* __restrict__ k,
                              unsigned short* __restrict__ v,
                              float* __restrict__ go, float* __restrict__ beta,
                              const int* __restrict__ cu, int N, int H, int Hg, float sp_thr)
{
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // flat chunk id -> (sequence, chunk); the grid is an upper bound (see
  // r4d_gdn_kkt_solve_k128_c64_bf16.hip)
  int n = -1, t0 = 0, bos = 0, rows = 0, slen = 0;
  {
    int acc = 0, want = blockIdx.x;
    for (int i = 0; i < N; ++i) {
      const int b = cu[i], e = cu[i + 1];
      const int nc = (e - b + CP_BT - 1) / CP_BT;
      if (want < acc + nc) { n = i; bos = b; slen = e - b; t0 = (want - acc) * CP_BT;
                             rows = min(CP_BT, slen - t0); break; }
      acc += nc;
    }
    if (n < 0) return;
  }
  const bool first = (t0 == 0);

  // ---- the gating half, on one workgroup per chunk ----------------------------------------
  // g and beta depend only on a and b; computed by the y == 0 workgroups instead of a launch.
  if (blockIdx.y == 0) {
    for (int hv = warp; hv < H; hv += HPB) {
      const float alog = __expf(A_log[hv]), dtb = dt_bias[hv];
      const int i0 = 2 * lane, i1 = i0 + 1;
      float g0 = 0.0f, g1 = 0.0f;
      if (i0 < rows)
        g0 = -alog * cp_softplus(cp_ab<ABF16>(av, (size_t)(bos + t0 + i0) * ab_stride + hv)
                                 + dtb, sp_thr);
      if (i1 < rows)
        g1 = -alog * cp_softplus(cp_ab<ABF16>(av, (size_t)(bos + t0 + i1) * ab_stride + hv)
                                 + dtb, sp_thr);
      // inclusive wave scan of the per-lane pair sums, made exclusive by subtracting the pair
      float p = g0 + g1;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float o = __shfl_up(p, d, 32);
        if (lane >= d) p += o;
      }
      const float pre = p - (g0 + g1);            // sum of all lanes before this one
      if (i0 < rows) {
        go[(size_t)(bos + t0 + i0) * H + hv] = pre + g0;
        beta[(size_t)(bos + t0 + i0) * H + hv] = 1.0f / (1.0f + __expf(
            -cp_ab<ABF16>(bv, (size_t)(bos + t0 + i0) * ab_stride + hv)));
      }
      if (i1 < rows) {
        go[(size_t)(bos + t0 + i1) * H + hv] = pre + g0 + g1;
        beta[(size_t)(bos + t0 + i1) * H + hv] = 1.0f / (1.0f + __expf(
            -cp_ab<ABF16>(bv, (size_t)(bos + t0 + i1) * ab_stride + hv)));
      }
    }
  }

  // ---- the convolution half ----------------------------------------------------------------
  const int unit = blockIdx.y * HPB + warp;       // global 128-dim head unit
  const int d0 = unit * CP_HD + lane * CP_DPL;    // this lane's first channel
  const unsigned short* xp = x + (size_t)(bos + t0) * xpitch + d0;

  float wt[CP_W][CP_DPL], bs[CP_DPL];
#pragma unroll
  for (int c = 0; c < CP_DPL; ++c) {
#pragma unroll
    for (int i = 0; i < CP_W; ++i) wt[i][c] = cp_tap(wgt[(size_t)(d0 + c) * CP_W + i]);
    bs[c] = bias ? bf2f(bias[d0 + c]) : 0.0f;
  }

  // win[i] holds x[t - (W-1) + i]; seeded from the previous tokens, or from the conv state cache
  // at the sequence's first chunk
  float win[CP_W][CP_DPL];
  {
    const bool init = first && r4d_has_init(has_init, hi_i32, n);
    const unsigned short* sp = init ? (cstate + (size_t)cache_idx[(size_t)n * ci_stride] * cs_seq
                                       + (size_t)d0 * cs_dim) : nullptr;
#pragma unroll
    for (int i = 0; i < CP_ST; ++i) {              // i = 0,1,2  ->  x[t0-3+i]
      const int src = t0 - CP_ST + i;
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) {
        float val = 0.0f;
        if (src >= 0) val = bf2f(x[(size_t)(bos + src) * xpitch + d0 + c]);
        else if (init) val = bf2f(sp[(size_t)c * cs_dim + (size_t)(CP_ST + src) * cs_tok]);
        win[i][c] = val;
      }
    }
  }

  // ---- write the new conv state, from x, at the sequence's first chunk ---------------------
  // The cache is updated in place, so the read above and this write must be in the same workgroup
  // for the same channels. That is why the first chunk writes the state even though its tokens
  // come from the last chunk.
  if (first && cache_idx) {
    unsigned short* sp = cstate + (size_t)cache_idx[(size_t)n * ci_stride] * cs_seq
                       + (size_t)d0 * cs_dim;
    const bool init = r4d_has_init(has_init, hi_i32, n);
    // Channel-major: the cache is [dim][slot] with slot contiguous, so a lane's CP_DPL*CP_ST
    // shorts are one contiguous run. Slot-major order would make every store a strided scalar.
#pragma unroll
    for (int c = 0; c < CP_DPL; ++c) {
#pragma unroll
      for (int i = 0; i < CP_ST; ++i) {
        const int src = slen - CP_ST + i;
        unsigned short val = 0;
        if (src >= 0) val = x[(size_t)(bos + src) * xpitch + d0 + c];
        else if (init) val = sp[(size_t)c * cs_dim + (size_t)(CP_ST + src) * cs_tok];
        sp[(size_t)c * cs_dim + (size_t)i * cs_tok] = val;
      }
    }
  }

  const int hq = unit, hk = unit - Hg, hv = unit - 2 * Hg;
  const int region = (unit < Hg) ? 0 : ((unit < 2 * Hg) ? 1 : 2);

  for (int t = 0; t < rows; ++t) {
    float o[CP_DPL];
    {
      const unsigned short* xr = xp + (size_t)t * xpitch;
      const ushort4 raw = *(const ushort4*)xr;
      win[CP_W - 1][0] = bf2f(raw.x); win[CP_W - 1][1] = bf2f(raw.y);
      win[CP_W - 1][2] = bf2f(raw.z); win[CP_W - 1][3] = bf2f(raw.w);
    }
#pragma unroll
    for (int c = 0; c < CP_DPL; ++c) {
      float acc = bs[c];
#pragma unroll
      for (int i = 0; i < CP_W; ++i) acc += wt[i][c] * win[i][c];
      // round as the unfused pair does on the way to HBM
      o[c] = bf2f(f2bf(cp_silu(acc)));
    }

    if (region < 2) {
      float ss = 0.0f;
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) ss += o[c] * o[c];
      ss += r4d_xor_lane<16>(ss); ss += r4d_xor_lane<8>(ss); ss += r4d_xor_lane<4>(ss);
      ss += r4d_xor_lane<2>(ss);  ss += r4d_xor_lane<1>(ss);
      const float inv = 1.0f / sqrtf(ss + 1e-6f);   // L2NORM_EPS, as in the kernel it replaces
      unsigned short* dst = (region ? k : q)
                          + ((size_t)(bos + t0 + t) * Hg + (region ? hk : hq)) * CP_HD
                          + lane * CP_DPL;
      *(ushort4*)dst = make_ushort4(f2bf(o[0] * inv), f2bf(o[1] * inv),
                                    f2bf(o[2] * inv), f2bf(o[3] * inv));
    } else {
      unsigned short* dst = v + ((size_t)(bos + t0 + t) * H + hv) * CP_HD + lane * CP_DPL;
      *(ushort4*)dst = make_ushort4(f2bf(o[0]), f2bf(o[1]), f2bf(o[2]), f2bf(o[3]));
    }

#pragma unroll
    for (int i = 0; i < CP_W - 1; ++i)
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) win[i][c] = win[i + 1][c];
  }
}

template <typename WT>
static int cp_prep_launch(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, const void* cache_idx, long ci_stride,
    const void* has_init, int hi_i32,
    const void* a, const void* b, long ab_stride, int ab_is_bf16, const void* A_log,
    const void* dt_bias, void* q, void* k, void* v, void* g, void* beta, const void* cu,
    int N, int T, int H, int Hg, int K, int V, int width, float softplus_thr, void* stream)
{
  if (K != CP_HD || V != CP_HD || width != CP_W) return -1;
  const int units = 2 * Hg + H;
  int hpb = 8;
  while (hpb > 1 && (Hg % hpb || H % hpb)) hpb >>= 1;
  const int nchunk_ub = (T + CP_BT - 1) / CP_BT + N;
  dim3 grid(nchunk_ub, units / hpb), blk(hpb * 32);
#define CP_LAUNCH(P, AB)                                                                      \
  hipLaunchKernelGGL((r4d_gdn_conv_prep_kernel<P, AB, WT>), grid, blk, 0, (hipStream_t)stream, \
                     (const unsigned short*)x, xpitch, (const WT*)wgt,                        \
                     (const unsigned short*)bias, (unsigned short*)cstate,                    \
                     cs_seq, cs_dim, cs_tok, (const int*)cache_idx, ci_stride,                \
                     has_init, hi_i32, a, b, ab_stride,                                     \
                     (const float*)A_log, (const float*)dt_bias, (unsigned short*)q,          \
                     (unsigned short*)k, (unsigned short*)v, (float*)g, (float*)beta,         \
                     (const int*)cu, N, H, Hg, softplus_thr)
#define CP_SWITCH(AB)                                                                         \
  switch (hpb) {                                                                              \
    case 8: CP_LAUNCH(8, AB); break;                                                          \
    case 4: CP_LAUNCH(4, AB); break;                                                          \
    case 2: CP_LAUNCH(2, AB); break;                                                          \
    default: CP_LAUNCH(1, AB); break;                                                         \
  }
  if (ab_is_bf16) { CP_SWITCH(1); } else { CP_SWITCH(0); }
#undef CP_SWITCH
#undef CP_LAUNCH
  return (int)hipGetLastError();
}

extern "C" int r4d_gdn_conv_prep_w4_h128_bf16(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, const void* cache_idx, long ci_stride,
    const void* has_init, int hi_i32,
    const void* a, const void* b, long ab_stride, int ab_is_bf16, const void* A_log,
    const void* dt_bias, void* q, void* k, void* v, void* g, void* beta, const void* cu,
    int N, int T, int H, int Hg, int K, int V, int width, float softplus_thr, void* stream)
{
  return cp_prep_launch<unsigned short>(x, xpitch, wgt, bias, cstate, cs_seq, cs_dim, cs_tok,
                                        cache_idx, ci_stride, has_init, hi_i32, a, b, ab_stride,
                                        ab_is_bf16, A_log, dt_bias, q, k, v, g, beta, cu, N, T, H,
                                        Hg, K, V, width, softplus_thr, stream);
}

extern "C" int r4d_gdn_conv_prep_w4_h128_bf16_f32wgt(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, const void* cache_idx, long ci_stride,
    const void* has_init, int hi_i32,
    const void* a, const void* b, long ab_stride, int ab_is_bf16, const void* A_log,
    const void* dt_bias, void* q, void* k, void* v, void* g, void* beta, const void* cu,
    int N, int T, int H, int Hg, int K, int V, int width, float softplus_thr, void* stream)
{
  return cp_prep_launch<float>(x, xpitch, wgt, bias, cstate, cs_seq, cs_dim, cs_tok, cache_idx,
                               ci_stride, has_init, hi_i32, a, b, ab_stride, ab_is_bf16, A_log,
                               dt_bias, q, k, v, g, beta, cu, N, T, H, Hg, K, V, width,
                               softplus_thr, stream);
}

// ---------------------------------------------------------------------------------------------
// Decode-time convolution. Same filter, but a step's tokens are a speculative window and the
// state cache is a rolling buffer of width-1 + num_spec entries: a step reads its window starting
// at the slot the last accepted token left, and rewrites the buffer shifted by one. Writes q, k
// and v in their own layouts (no torch.cat for the recurrent kernel).
// ---------------------------------------------------------------------------------------------
template <int HPB, typename WT>
__global__ __launch_bounds__(HPB * 32)
void r4d_gdn_conv_update_kernel(const unsigned short* __restrict__ x, long xpitch,
                                const WT* __restrict__ wgt,
                                const unsigned short* __restrict__ bias,
                                unsigned short* __restrict__ cstate,
                                long cs_seq, long cs_dim, long cs_tok, int state_len_max,
                                const int* __restrict__ cache_idx, long ci_stride,
                                const int* __restrict__ naccept,
                                unsigned short* __restrict__ q, unsigned short* __restrict__ k,
                                unsigned short* __restrict__ v,
                                const int* __restrict__ cu, int H, int Hg, int maxq)
{
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n = blockIdx.x;
  const int bos = cu[n], slen = cu[n + 1] - bos;
  if (slen <= 0) return;
  const int slot = cache_idx[(size_t)n * ci_stride];
  // A negative slot is padding; 0 is a real slot. (vLLM reserves 0 as NULL_BLOCK_ID, but this
  // engine's state slots are 0-based, as in libref/ref_gdn.cpp.)
  if (slot < 0) return;

  const int off = naccept ? (naccept[n] - 1) : 0;
  // cache depth this step uses: as in FLA, the bound shrinks by the tokens this sequence lacks
  const int slen_eff = state_len_max - (maxq - slen);

  const int unit = blockIdx.y * HPB + warp;
  const int d0 = unit * CP_HD + lane * CP_DPL;

  // A channel's weights are CP_W contiguous elements: one vector load per channel.
  float wt[CP_W][CP_DPL], bs[CP_DPL];
#pragma unroll
  for (int c = 0; c < CP_DPL; ++c) {
    const cp_taps<WT> w4 = *(const cp_taps<WT>*)(wgt + (size_t)(d0 + c) * CP_W);
    const WT* ws = (const WT*)&w4;
#pragma unroll
    for (int i = 0; i < CP_W; ++i) wt[i][c] = cp_tap(ws[i]);
    bs[c] = bias ? bf2f(bias[d0 + c]) : 0.0f;
  }

  // This kernel's time is address arithmetic. size_t indexing gives each of the ~40 cache stores
  // and every input tap a 64-bit multiply. The strides are small, so offsets are 32-bit ints off
  // base pointers that carry the 64-bit part, and loops walk their pointers.
  const int csd = (int)cs_dim, cst = (int)cs_tok, xpi = (int)xpitch;
  unsigned short* const sp = cstate + (size_t)slot * cs_seq + (size_t)d0 * cs_dim;
  const unsigned short* const xrow = x + (size_t)bos * xpitch + d0;
  // Every input load of the step is issued before any is waited on; the cache rewrite below
  // reuses the same values.
  ushort4 xb[CU_MAXT];
  const bool xpre = (slen <= CU_MAXT);
#pragma unroll
  for (int t = 0; t < CU_MAXT; ++t)
    if (xpre && t < slen) xb[t] = *(const ushort4*)(xrow + t * xpi);
  float win[CP_W][CP_DPL];
  unsigned short hist[CP_ST][CP_DPL];             // the raw history, for the shifted rewrite
  {
    const unsigned short* hp = sp + off * cst;
#pragma unroll
    for (int i = 0; i < CP_ST; ++i)
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) {
        const unsigned short raw = hp[c * csd + i * cst];
        hist[i][c] = raw;
        win[i][c] = bf2f(raw);
      }
  }

  const int hq = unit, hk = unit - Hg, hv = unit - 2 * Hg;
  const int region = (unit < Hg) ? 0 : ((unit < 2 * Hg) ? 1 : 2);

  // Output pointer walked by a fixed stride per token, avoiding a 64-bit multiply each.
  unsigned short* op = (region == 2)
      ? (v + ((size_t)bos * H + hv) * CP_HD + lane * CP_DPL)
      : ((region ? k : q) + ((size_t)bos * Hg + (region ? hk : hq)) * CP_HD + lane * CP_DPL);
  const int ostep = (region == 2) ? H * CP_HD : Hg * CP_HD;

#define CU_TOKEN_BODY(RAW)                                                       \
    {                                                                            \
      const ushort4 raw_ = (RAW);                                                \
      win[CP_W - 1][0] = bf2f(raw_.x); win[CP_W - 1][1] = bf2f(raw_.y);           \
      win[CP_W - 1][2] = bf2f(raw_.z); win[CP_W - 1][3] = bf2f(raw_.w);           \
      unsigned short out[CP_DPL];                                                \
      _Pragma("unroll")                                                          \
      for (int c = 0; c < CP_DPL; ++c) {                                         \
        float acc = bs[c];                                                       \
        _Pragma("unroll")                                                        \
        for (int i = 0; i < CP_W; ++i) acc += wt[i][c] * win[i][c];              \
        out[c] = f2bf(cp_silu(acc));                                             \
      }                                                                          \
      *(ushort4*)op = make_ushort4(out[0], out[1], out[2], out[3]);               \
      op += ostep;                                                               \
      _Pragma("unroll")                                                          \
      for (int i = 0; i < CP_W - 1; ++i)                                         \
        _Pragma("unroll")                                                        \
        for (int c = 0; c < CP_DPL; ++c) win[i][c] = win[i + 1][c];              \
    }
  if (xpre) {
    // A guard, not a break: a data-dependent exit would leave the loop rolled and t a runtime
    // index again.
#pragma unroll
    for (int t = 0; t < CU_MAXT; ++t) {
      if (t < slen) CU_TOKEN_BODY(xb[t])          // static index: a register read, not a tree
    }
  } else {
    for (int t = 0; t < slen; ++t) CU_TOKEN_BODY(*(const ushort4*)(xrow + t * xpi))
  }

  // Rewrite the cache shifted by one: the tail of the old history (already in registers, read
  // before any write, so the in-place update is safe), then this step's tokens. Channel-major, as
  // in prefill: one contiguous run per lane instead of strided 2-byte stores.
  // The split between the two sources is a loop bound, not a per-store test. `nh` is CP_W-2 on the
  // shipped path but computed, so a caller that shrinks the window differently stays correct.
  const int VAL = CP_W - 2;
  const int nh = slen_eff - slen < 0 ? 0 : (slen_eff - slen);
#pragma unroll
  for (int c = 0; c < CP_DPL; ++c) {
    unsigned short* dst = sp + c * csd;
    // History half, unrolled to hist's bound (i + 1 < CP_ST).
#pragma unroll
    for (int i = 0; i + 1 < CP_ST; ++i)
      if (i < nh && i < slen_eff) dst[i * cst] = hist[i + 1][c];
    // Token half, walked by token so the xb index is the loop counter.
    if (xpre) {
#pragma unroll
      for (int t = 0; t < CU_MAXT; ++t) {
        const int i = t + VAL;
        if (i >= nh && i < slen_eff && t < slen) {
          // Not `((const unsigned short*)&xb[t])[c]`: taking an element's address forces the
          // whole array to scratch. With t and c compile-time, this ternary is a register read.
          const ushort4 vv = xb[t];
          dst[i * cst] = (c == 0) ? vv.x : (c == 1) ? vv.y : (c == 2) ? vv.z : vv.w;
        }
      }
    }
    // Remainder: slots whose token is outside [0, slen), or the whole range when slen exceeds the
    // preload. Empty on the shipped path.
    for (int i = nh; i < slen_eff; ++i) {
      const int ti = i - VAL;
      if (!xpre || ti < 0 || ti >= slen) dst[i * cst] = xrow[ti * xpi + c];
    }
  }
#undef CU_TOKEN_BODY
}

template <typename WT>
static int cu_update_launch(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
    long ci_stride, const void* num_accepted, void* q, void* k, void* v, const void* cu,
    int N, int H, int Hg, int K, int V, int width, int max_query_len, void* stream)
{
  if (K != CP_HD || V != CP_HD || width != CP_W) return -1;
  const int units = 2 * Hg + H;
  int hpb = 8;
  while (hpb > 1 && (Hg % hpb || H % hpb)) hpb >>= 1;
  // A decode step has few sequences, so narrow the channel group until the grid covers the
  // device (8 sequences at 8 heads per block would be 40 workgroups for 64 CUs).
  while (hpb > 1 && (long)N * (units / hpb) < 256) hpb >>= 1;
  dim3 grid(N, units / hpb), blk(hpb * 32);
#define CU_LAUNCH(P)                                                                          \
  hipLaunchKernelGGL((r4d_gdn_conv_update_kernel<P, WT>), grid, blk, 0, (hipStream_t)stream,  \
                     (const unsigned short*)x, xpitch, (const WT*)wgt,                        \
                     (const unsigned short*)bias, (unsigned short*)cstate, cs_seq, cs_dim,    \
                     cs_tok, state_len_max, (const int*)cache_idx, ci_stride,                 \
                     (const int*)num_accepted, (unsigned short*)q, (unsigned short*)k,        \
                     (unsigned short*)v, (const int*)cu, H, Hg, max_query_len)
  switch (hpb) {
    case 8: CU_LAUNCH(8); break;
    case 4: CU_LAUNCH(4); break;
    case 2: CU_LAUNCH(2); break;
    default: CU_LAUNCH(1); break;
  }
#undef CU_LAUNCH
  return (int)hipGetLastError();
}

extern "C" int r4d_gdn_conv_update_w4_h128_bf16(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
    long ci_stride, const void* num_accepted, void* q, void* k, void* v, const void* cu,
    int N, int H, int Hg, int K, int V, int width, int max_query_len, void* stream)
{
  return cu_update_launch<unsigned short>(x, xpitch, wgt, bias, cstate, cs_seq, cs_dim, cs_tok,
                                          state_len_max, cache_idx, ci_stride, num_accepted, q,
                                          k, v, cu, N, H, Hg, K, V, width, max_query_len, stream);
}

extern "C" int r4d_gdn_conv_update_w4_h128_bf16_f32wgt(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
    long ci_stride, const void* num_accepted, void* q, void* k, void* v, const void* cu,
    int N, int H, int Hg, int K, int V, int width, int max_query_len, void* stream)
{
  return cu_update_launch<float>(x, xpitch, wgt, bias, cstate, cs_seq, cs_dim, cs_tok,
                                 state_len_max, cache_idx, ci_stride, num_accepted, q, k, v, cu,
                                 N, H, Hg, K, V, width, max_query_len, stream);
}

// ============================================================== the plugin ABI entry points
// Shared operand helpers: r4d_gdn_args.h. Operand enums use an OPD_ prefix because CP_W and CP_ST
// are already kernel constants in this unit.
/* ------------------------------------------------------------------ conv prep */


/* Inputs first, outputs last -- docs/OPS.md's resolved table. */
enum { OPD_CP_X = 0, OPD_CP_W, OPD_CP_B, OPD_CP_ALOG, OPD_CP_DTB, OPD_CP_CSTATE, OPD_CP_CU,
       OPD_CP_A, OPD_CP_BGATE, OPD_CP_CACHE_IDX, OPD_CP_HAS_INIT,
       OPD_CP_Q, OPD_CP_K, OPD_CP_V, OPD_CP_G, OPD_CP_BETA };

extern "C" int r4d_rad_gdn_conv_prep(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* x    = r4d_opt(a, OPD_CP_X);
    const RadTensor* w    = r4d_opt(a, OPD_CP_W);
    const RadTensor* b    = r4d_opt(a, OPD_CP_B);
    const RadTensor* alog = r4d_opt(a, OPD_CP_ALOG);
    const RadTensor* dtb  = r4d_opt(a, OPD_CP_DTB);
    const RadTensor* cst  = r4d_opt(a, OPD_CP_CSTATE);
    const RadTensor* cu   = r4d_opt(a, OPD_CP_CU);
    const RadTensor* q    = r4d_opt(a, OPD_CP_Q);
    const RadTensor* k    = r4d_opt(a, OPD_CP_K);
    const RadTensor* v    = r4d_opt(a, OPD_CP_V);
    const RadTensor* g    = r4d_opt(a, OPD_CP_G);
    const RadTensor* beta = r4d_opt(a, OPD_CP_BETA);
    const RadTensor* ta   = r4d_opt(a, OPD_CP_A);
    const RadTensor* tb   = r4d_opt(a, OPD_CP_BGATE);
    const RadTensor* cidx = r4d_opt(a, OPD_CP_CACHE_IDX);
    const RadTensor* hin  = r4d_opt(a, OPD_CP_HAS_INIT);

    if (!x || !w || !alog || !dtb || !cst || !cu || !q || !k || !v || !g || !beta)
        return R4D_NO(RAD_E_INVAL);
    /* Optional in the schema (so docs/OPS.md's shorter list stays a positional prefix) but
     * required here: the kernel dereferences all three. */
    if (!ta || !tb || !cidx) return R4D_NO(RAD_E_INVAL);

    if (x->rank < 2 || q->rank != 3 || k->rank != 3 || v->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (g->rank != 2 || beta->rank != 2 || cst->rank != 3 || ta->rank != 2) return R4D_NO(RAD_E_SHAPE);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;

    /* x may be a padded view (the layer's qkvz split), so its row pitch is an argument; inner
     * axes must be dense. */
    const long long xpitch = r4d_row_pitch(x);
    if (xpitch < 0) return R4D_NO(RAD_E_STRIDE);
    const long long ab_stride = r4d_row_pitch(ta);
    if (ab_stride < 0) return R4D_NO(RAD_E_STRIDE);
    if (r4d_row_pitch(tb) != ab_stride) return R4D_NO(RAD_E_STRIDE);

    if (!r4d_contig(q) || !r4d_contig(k) || !r4d_contig(v) || !r4d_contig(g) ||
        !r4d_contig(beta) || !r4d_contig(w) || !r4d_contig(alog) || !r4d_contig(dtb))
        return R4D_NO(RAD_E_STRIDE);
    if (cidx->rank < 1) return R4D_NO(RAD_E_SHAPE);

    const long long T  = q->shape[0];
    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    if (k->shape[0] != T || k->shape[1] != Hg || k->shape[2] != K) return R4D_NO(RAD_E_SHAPE);
    if (v->shape[0] != T) return R4D_NO(RAD_E_SHAPE);
    if (g->shape[0] != T || g->shape[1] != H) return R4D_NO(RAD_E_SHAPE);
    if (beta->shape[0] != T || beta->shape[1] != H) return R4D_NO(RAD_E_SHAPE);
    if (H == 0 || Hg == 0 || H % Hg) return R4D_NO(RAD_E_SHAPE);

    const int width = (int)r4d_geti_or(a, "conv_width", 4);
    const float thr = (float)r4d_getf(a, "softplus_thr", 20.0);
    /* The l2 norm epsilon is compiled in (1e-6f); refuse any other value. */
    if ((float)r4d_getf(a, "l2_eps", 1e-6) != 1e-6f) return R4D_NO(RAD_E_UNSUPPORTED);
    /* a and b may be bf16 or f32 (a flag, not two entry points). */
    const int ab_is_bf16 = (ta->dtype == RAD_BF16) ? 1 : 0;
    if (tb->dtype != ta->dtype) return R4D_NO(RAD_E_DTYPE);
    if (!ab_is_bf16 && ta->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);

    /* bf16 x, conv state and q/k/v; bf16 or f32 weights (r4d_gdn_conv_tap.h); fp32 A_log,
     * dt_bias, g and beta. */
    R4D_DT(x, RAD_BF16); R4D_DT(cst, RAD_BF16);
    R4D_DT(q, RAD_BF16); R4D_DT(k, RAD_BF16); R4D_DT(v, RAD_BF16);
    R4D_DT(alog, RAD_F32); R4D_DT(dtb, RAD_F32); R4D_DT(g, RAD_F32); R4D_DT(beta, RAD_F32);
    if (w->dtype != RAD_BF16 && w->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    const auto prep = w->dtype == RAD_F32 ? r4d_gdn_conv_prep_w4_h128_bf16_f32wgt
                                          : r4d_gdn_conv_prep_w4_h128_bf16;

    return R4D_RAW(prep(
        x->data, (long)xpitch, w->data, b ? b->data : nullptr, cst->data,
        (long)cst->stride[0], (long)cst->stride[1], (long)cst->stride[2],
        cidx->data, (long)cidx->stride[0], hin ? hin->data : nullptr,
        /* has_init element width (see r4d_has_init) */
        hin && (hin->dtype == RAD_I32 || hin->dtype == RAD_U32),
        ta->data, tb->data, (long)ab_stride, ab_is_bf16,
        alog->data, dtb->data, q->data, k->data, v->data, g->data, beta->data,
        cu->data, N, (int)T, (int)H, (int)Hg, (int)K, (int)V, width, thr, s));
}

/* ------------------------------------------------------------------ conv update */

enum { CUP_X = 0, CUP_W, CUP_B, CUP_CSTATE, CUP_SIDX, CUP_NACC, CUP_CU, CUP_Q, CUP_K, CUP_V };

extern "C" int r4d_rad_gdn_conv_update(const RadArgs* a, RadStream s) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* x    = r4d_opt(a, CUP_X);
    const RadTensor* w    = r4d_opt(a, CUP_W);
    const RadTensor* b    = r4d_opt(a, CUP_B);
    const RadTensor* cst  = r4d_opt(a, CUP_CSTATE);
    const RadTensor* sidx = r4d_opt(a, CUP_SIDX);
    const RadTensor* nacc = r4d_opt(a, CUP_NACC);
    const RadTensor* cu   = r4d_opt(a, CUP_CU);
    const RadTensor* q    = r4d_opt(a, CUP_Q);
    const RadTensor* k    = r4d_opt(a, CUP_K);
    const RadTensor* v    = r4d_opt(a, CUP_V);
    if (!x || !w || !cst || !sidx || !cu || !q || !k || !v) return R4D_NO(RAD_E_INVAL);
    if (cst->rank != 3 || q->rank != 3 || k->rank != 3 || v->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (sidx->rank < 1) return R4D_NO(RAD_E_SHAPE);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;

    const long long xpitch = r4d_row_pitch(x);
    if (xpitch < 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(q) || !r4d_contig(k) || !r4d_contig(v) || !r4d_contig(w))
        return R4D_NO(RAD_E_STRIDE);

    const long long T  = q->shape[0];
    const long long Hg = q->shape[1];
    const long long K  = q->shape[2];
    const long long H  = v->shape[1];
    const long long V  = v->shape[2];
    if (H == 0 || Hg == 0 || H % Hg) return R4D_NO(RAD_E_SHAPE);
    if (v->shape[0] != T || k->shape[0] != T) return R4D_NO(RAD_E_SHAPE);

    /* bf16 x, conv state and q/k/v, bf16 or f32 weights, as in the prefill conv. */
    R4D_DT(x, RAD_BF16); R4D_DT(cst, RAD_BF16);
    R4D_DT(q, RAD_BF16); R4D_DT(k, RAD_BF16); R4D_DT(v, RAD_BF16);
    if (w->dtype != RAD_BF16 && w->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    const auto update = w->dtype == RAD_F32 ? r4d_gdn_conv_update_w4_h128_bf16_f32wgt
                                            : r4d_gdn_conv_update_w4_h128_bf16;

    const int width = (int)r4d_geti_or(a, "conv_width", 4);
    /* state_len_max and max_query_len describe one window, so only one is free. The state cache
     * is a rolling window read at the slot the last accepted token left, so a speculative
     * rejection is a change of read offset, not a recompute (spec 10). The kernel splits it into
     * `width - 2` history entries and the step's tokens at state_len_max - max_query_len, so that
     * difference must be width - 2; max_query_len follows from the cache depth.
     *
     * Do not derive max_query_len from the step: the pool is sized for the deployment's deepest
     * draft, and a non-drafting step would misplace the split, so the convolution would stop
     * seeing the latest tokens and the model would loop on one phrase. */
    const long long state_len_max = cst->shape[2];
    const long long mql = state_len_max - (width - 2);
    if (width < 2 || mql < 1) return R4D_NO(RAD_E_SHAPE);

    return R4D_RAW(update(
        x->data, (long)xpitch, w->data, b ? b->data : nullptr, cst->data,
        (long)cst->stride[0], (long)cst->stride[1], (long)cst->stride[2],
        (int)state_len_max, sidx->data, (long)sidx->stride[0],
        nacc ? nacc->data : nullptr, q->data, k->data, v->data, cu->data,
        N, (int)H, (int)Hg, (int)K, (int)V, width, (int)mql, s));
}

