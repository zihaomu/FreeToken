// Sampling on the device: the vLLM chain for one position in one workgroup (temperature, softmax,
// top_k, top_p, min_p, renormalise, draw), plus the second draw and accept probability that a
// greedy-draft rejection-sampling verify needs.
//
// The head leaves a [vocab][M] plane in VRAM, row-split across ranks. A host sampler would need
// every band copied back and a softmax and nucleus cut over 248K entries per position on the
// dispatch thread; on the device it costs one gather of the other ranks' bands.
//
// Three outputs: accepting a draft token when it equals the target's argmax is only correct at
// temperature 0. With a sampling target the rule is rejection sampling, and a greedy draft is a
// point mass, so it is accepted with probability p(x) under the target's filtered distribution. On
// rejection the token comes from the residual, which for a point mass is the same distribution
// with the proposal removed and renormalised. One pass produces all three; the host then walks the
// positions with its own uniforms. Acceptance is lower than greedy-against-greedy, which is the
// cost of emitting the target's true distribution.
//
// The order is vLLM's; reordering changes which tokens survive. Temperature 0 is a real path, not
// special-cased by the caller, so it uses the argmax kernel's tie-break (lowest id wins) and a
// temperature-0 request matches a greedy run.
//
// One block owns one position and walks the vocabulary several times; after the first pass the
// plane is L2-resident, which beats staging at this vocabulary size. The top_k and top_p cuts are
// four radix passes each over 256 buckets of a monotone key, giving the k-th largest probability
// and the nucleus threshold exactly without sorting.
//
// The kernel reads the logits through explicit row and column strides, so nothing is
// model-specific. The entry point takes flat scalars and returns a negative code instead of
// throwing, for the reason given in r4d_gemm_fp8a8.hip. `f32` in the name is the logits dtype.

#include "r4d.h"
#include "r4d_fp8_args.h"

#include <hip/hip_runtime.h>
#include <cmath>

namespace {

typedef R4DSpecDraw SpecDraw;


constexpr uint32_t kThr = 256;

// A monotone map from float to uint32, so a radix pass over the bits orders the values. Written for
// the general case although probabilities are non-negative.
__device__ __forceinline__ uint32_t mono(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Everything the block shares, declared once at kernel scope.
struct Shared {
  float    red[kThr];       // the general-purpose reduction buffer
  uint32_t red_i[kThr];     // ...and the ids that ride along with it
  float    mass[256];       // radix buckets by probability mass (top_p)
  uint32_t cnt[256];        // ...and by count (top_k)
  uint32_t prefix;          // the threshold key being narrowed, high byte first
  uint32_t slice;           // which thread's slice an inverse-CDF draw landed in
  float    slice_base;      // the cumulative mass before that slice
  uint32_t pick;
  float    keep_total;
  float    excl_total;
  float    p_query;
};

__device__ __forceinline__ float reduce_max(Shared& sh, float v) {
  sh.red[threadIdx.x] = v;
  __syncthreads();
  for (uint32_t k = kThr / 2; k; k >>= 1) {
    if (threadIdx.x < k) sh.red[threadIdx.x] = fmaxf(sh.red[threadIdx.x], sh.red[threadIdx.x + k]);
    __syncthreads();
  }
  const float r = sh.red[0];
  __syncthreads();
  return r;
}

__device__ __forceinline__ float reduce_sum(Shared& sh, float v) {
  sh.red[threadIdx.x] = v;
  __syncthreads();
  for (uint32_t k = kThr / 2; k; k >>= 1) {
    if (threadIdx.x < k) sh.red[threadIdx.x] += sh.red[threadIdx.x + k];
    __syncthreads();
  }
  const float r = sh.red[0];
  __syncthreads();
  return r;
}

// ---- the kernel ---------------------------------------------------------------------------------
//
// See the file header for the chain order and the multi-pass structure.
__global__ __launch_bounds__(kThr) void sample_kernel(
    const float* __restrict__ logits, uint32_t vocab, uint32_t row_stride, uint32_t col_stride,
    const uint32_t* __restrict__ query, const float* __restrict__ u, float temperature,
    float top_p, uint32_t top_k, float min_p, SpecDraw* __restrict__ out) {
  __shared__ Shared sh;

  const uint32_t pos = blockIdx.x, tid = threadIdx.x;
  const float* const col = logits + (size_t)pos * col_stride;
  const uint32_t q = query ? query[pos] : 0xFFFFFFFFu;

  auto lg = [&](uint32_t v) { return col[(size_t)v * row_stride]; };

  const float m = reduce_max(sh, [&] {
    float best = -INFINITY;
    for (uint32_t v = tid; v < vocab; v += kThr) best = fmaxf(best, lg(v));
    return best;
  }());

  // ---- temperature 0 is greedy, and stays a real path, with the argmax kernel's tie-break
  //      (lowest id wins) so it matches a greedy run.
  if (!(temperature > 0.0f)) {
    float bv = -INFINITY;
    uint32_t bi = 0xFFFFFFFFu;
    for (uint32_t v = tid; v < vocab; v += kThr) {
      const float x = lg(v);
      if (x > bv || (x == bv && v < bi)) { bv = x; bi = v; }
    }
    sh.red[tid] = bv;
    sh.red_i[tid] = bi;
    __syncthreads();
    for (uint32_t k = kThr / 2; k; k >>= 1) {
      if (tid < k) {
        const float ov = sh.red[tid + k];
        const uint32_t oi = sh.red_i[tid + k];
        if (ov > sh.red[tid] || (ov == sh.red[tid] && oi < sh.red_i[tid])) {
          sh.red[tid] = ov; sh.red_i[tid] = oi;
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      // An all-NaN row leaves the reduction at its seed, which is not a token; emit token 0.
      const uint32_t g = sh.red_i[0] < vocab ? sh.red_i[0] : 0u;
      out[pos].tok = g;
      out[pos].tok_excl = g;
      out[pos].p_query = (q == g) ? 1.0f : 0.0f;
    }
    return;
  }

  const float inv_t = 1.0f / temperature;
  float zs = 0.0f;
  for (uint32_t v = tid; v < vocab; v += kThr) zs += __expf((lg(v) - m) * inv_t);
  const float z = reduce_sum(sh, zs);
  const float inv_z = z > 0.0f ? 1.0f / z : 0.0f;
  auto prob = [&](uint32_t v) { return __expf((lg(v) - m) * inv_t) * inv_z; };

  // ---- top_k, as an exact threshold on the monotone key ----------------------------------------
  //
  // Four radix passes narrowing one byte at a time, counting tokens: the k-th largest probability
  // exactly, without sorting.
  uint32_t k_key = 0u;
  if (top_k > 0 && top_k < vocab) {
    if (tid == 0) sh.prefix = 0u;
    __syncthreads();
    uint32_t remaining = top_k;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (tid < 256) sh.cnt[tid] = 0u;
      __syncthreads();
      const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
      for (uint32_t v = tid; v < vocab; v += kThr) {
        const uint32_t key = mono(prob(v));
        if ((key & mask) == (sh.prefix & mask)) atomicAdd(&sh.cnt[(key >> shift) & 0xFFu], 1u);
      }
      __syncthreads();
      if (tid == 0) {
        uint32_t acc = 0u, b = 0u;
        for (int i = 255; i >= 0; --i) {
          if (acc + sh.cnt[i] >= remaining) { b = (uint32_t)i; break; }
          acc += sh.cnt[i];
        }
        remaining -= acc;
        sh.prefix |= b << shift;
      }
      __syncthreads();
    }
    k_key = sh.prefix;
  }

  // ---- top_p, the same passes accumulating MASS ------------------------------------------------
  //
  // Descend the buckets summing probability until the nucleus target is reached; the crossing
  // bucket holds the threshold and the next pass narrows inside it. Ties at the threshold are all
  // kept, which can overshoot top_p by the tie's mass, rather than inventing an order among equal
  // probabilities.
  uint32_t p_key = 0u;
  if (top_p < 1.0f) {
    if (tid == 0) sh.prefix = 0u;
    __syncthreads();
    float target = top_p;
    for (int shift = 24; shift >= 0; shift -= 8) {
      if (tid < 256) sh.mass[tid] = 0.0f;
      __syncthreads();
      const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
      for (uint32_t v = tid; v < vocab; v += kThr) {
        const float p = prob(v);
        const uint32_t key = mono(p);
        if ((key & mask) == (sh.prefix & mask) && (k_key == 0u || key >= k_key))
          atomicAdd(&sh.mass[(key >> shift) & 0xFFu], p);
      }
      __syncthreads();
      if (tid == 0) {
        float acc = 0.0f;
        uint32_t b = 0u;
        for (int i = 255; i >= 0; --i) {
          if (acc + sh.mass[i] >= target) { b = (uint32_t)i; break; }
          acc += sh.mass[i];
        }
        target -= acc;
        sh.prefix |= b << shift;
      }
      __syncthreads();
    }
    p_key = sh.prefix;
  }

  // min_p is a floor relative to the mode, and the mode's probability is exp(0)/Z.
  const float floor_p = min_p > 0.0f ? min_p * inv_z : 0.0f;

  auto keep = [&](float p) {
    const uint32_t key = mono(p);
    if (k_key && key < k_key) return false;
    if (p_key && key < p_key) return false;
    return !(p < floor_p);
  };

  // ---- the surviving mass, with and without the proposal ----------------------------------------
  float kt = 0.0f, ke = 0.0f, pq = 0.0f;
  for (uint32_t v = tid; v < vocab; v += kThr) {
    const float p = prob(v);
    if (!keep(p)) continue;
    kt += p;
    if (v == q) pq += p; else ke += p;
  }
  const float keep_total = reduce_sum(sh, kt);
  const float excl_total = reduce_sum(sh, ke);
  const float p_query = reduce_sum(sh, pq);
  if (tid == 0) {
    sh.keep_total = keep_total;
    sh.excl_total = excl_total;
    sh.p_query = p_query;
  }
  __syncthreads();

  // ---- inverse CDF, in two stages ---------------------------------------------------------------
  //
  // Contiguous slices, unlike the strided ones elsewhere here, for correctness: any fixed order
  // samples correctly, but only ascending token id makes the result a function of the uniform
  // alone and not of the block size, so a host reference or another kThr picks the same token.
  //
  // Each thread sums its own range, thread 0 finds the range the draw lands in, and that thread
  // rescans it. Exact, no sort. Every thread reaches both calls, so the barriers are uniform.
  //
  // Only a range with mass, and a token with mass inside it, can be chosen. An empty range
  // satisfies `acc + 0 >= want` when the draw is 0 or float drift leaves the walk short, and
  // rescanning it would emit 0xFFFFFFFF. So empty ranges are skipped, the fall-through is the last
  // range with mass, and within it the last token with mass; `total > 0` guarantees one exists.
  const uint32_t chunk = (vocab + kThr - 1) / kThr;
  const uint32_t lo = tid * chunk;
  const uint32_t hi = lo + chunk < vocab ? lo + chunk : vocab;

  auto draw = [&](float uu, float total, uint32_t excl) -> uint32_t {
    if (!(total > 0.0f)) return 0u;
    const float want = uu * total;
    float mine = 0.0f;
    for (uint32_t v = lo; v < hi; ++v) {
      const float p = prob(v);
      if (!keep(p) || v == excl) continue;
      mine += p;
    }
    sh.red[tid] = mine;
    __syncthreads();
    if (tid == 0) {
      float acc = 0.0f, last_base = 0.0f;
      uint32_t who = kThr, last_who = 0;
      for (uint32_t k = 0; k < kThr; ++k) {
        if (!(sh.red[k] > 0.0f)) continue;
        last_who = k;
        last_base = acc;
        if (acc + sh.red[k] >= want) { who = k; break; }
        acc += sh.red[k];
      }
      if (who == kThr) { who = last_who; acc = last_base; }
      sh.slice = who;
      sh.slice_base = acc;
      sh.pick = 0xFFFFFFFFu;
    }
    __syncthreads();
    if (tid == sh.slice) {
      float acc = sh.slice_base;
      uint32_t last = 0xFFFFFFFFu;
      for (uint32_t v = lo; v < hi; ++v) {
        const float p = prob(v);
        if (!keep(p) || v == excl || !(p > 0.0f)) continue;
        last = v;
        acc += p;
        if (acc >= want) { sh.pick = v; break; }
      }
      // Float drift can leave the walk just short at the top of the range; the last token with
      // mass is the answer there.
      if (sh.pick == 0xFFFFFFFFu) sh.pick = last;
    }
    __syncthreads();
    const uint32_t r = sh.pick < vocab ? sh.pick : 0u;
    __syncthreads();
    return r;
  };

  const uint32_t t1 = draw(u[2 * pos + 0], sh.keep_total, 0xFFFFFFFFu);
  const uint32_t t2 = draw(u[2 * pos + 1], sh.excl_total, q);

  if (tid == 0) {
    out[pos].tok = t1;
    out[pos].tok_excl = t2;
    out[pos].p_query = sh.keep_total > 0.0f ? sh.p_query / sh.keep_total : 0.0f;
  }
}

}  // namespace

// `logits` is the head's plane, addressed as `logits[pos*col_stride + v*row_stride]`, so either
// head layout is read without a transpose.
//
// `query[i]` is the draft's proposal for position i, or 0xFFFFFFFF where there is none (the last
// position of a verify block); the pointer may be 0. `u` holds two uniforms per position in [0,1),
// for `tok` and `tok_excl`; the accept draw stays on the host, which owns the seed.
//
// `n_vocab` is the full vocabulary: a vocab-parallel caller gathers the other bands first.
//
// Returns 0, -4 for a failed launch, or another negative code for arguments it refuses.
extern "C" int r4d_sample_chain_f32(long logits, long query, long u, long out,
                                    int M, int n_vocab, int row_stride, int col_stride,
                                    float temperature, float top_p, int top_k, float min_p,
                                    long stream) {
    if (!logits || !out || !u) return -1;
    if (M <= 0 || n_vocab <= 0) return -2;
    if (top_k < 0) return -3;
    if (!row_stride) row_stride = 1;
    if (!col_stride) col_stride = n_vocab;
    hipLaunchKernelGGL(sample_kernel, dim3((uint32_t)M), dim3(kThr), 0, (hipStream_t)stream,
                       (const float*)logits, (uint32_t)n_vocab, (uint32_t)row_stride,
                       (uint32_t)col_stride, (const uint32_t*)query, (const float*)u,
                       temperature, top_p, (uint32_t)top_k, min_p, (SpecDraw*)out);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h. See libr4d/r4d_args.h on why there is no adapter
// layer.
/* ================================================================== the device sampler */

extern "C" int r4d_rad_sample_chain(const RadArgs* a, RadStream s) {
    const RadTensor* lg = r4d_opt(a, SC_LOGITS);
    const RadTensor* qy = r4d_opt(a, SC_QUERY);
    const RadTensor* u  = r4d_opt(a, SC_U);
    const RadTensor* dr = r4d_opt(a, SC_DRAW);
    if (!lg || !u || !dr) return R4D_NO(RAD_E_INVAL);
    if (lg->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = lg->shape[0], V = lg->shape[1];
    if (M < 1 || V < 1) return R4D_NO(RAD_E_SHAPE);
    /* Two uniforms per position, one for each draw; a caller supplying M would read past its own
     * buffer. */
    if (r4d_numel(u) < 2 * M) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(dr) < M) return R4D_NO(RAD_E_SHAPE);
    if (qy && r4d_numel(qy) < M) return R4D_NO(RAD_E_SHAPE);
    /* The head writes [M][V] or [V][M], and the kernel reads both, so the strides come from the
     * tensor. */
    const long long rs = lg->stride[1], cs = lg->stride[0];
    if (rs < 1 || cs < 1) return R4D_NO(RAD_E_STRIDE);

    const double temp = r4d_getf(a, "temp", 1.0);
    const double topp = r4d_getf(a, "p", 1.0);
    const double minp = r4d_getf(a, "min_p", 0.0);
    const long long topk = r4d_geti_or(a, "k", 0);
    if (topk < 0) return R4D_NO(RAD_E_SHAPE);

    const int e = r4d_sample_chain_f32(r4d_addr(lg), r4d_addr(qy), r4d_addr(u), r4d_addr(dr),
                                       (int)M, (int)V, (int)rs, (int)cs,
                                       (float)temp, (float)topp, (int)topk, (float)minp,
                                       (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : R4D_NO(e == -4 ? RAD_E_DEVICE : RAD_E_SHAPE);
}
