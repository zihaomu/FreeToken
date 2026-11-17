// r4d_quant_act_i8.hip - r4d_quant_act_i8: per-row symmetric int8 quantisation of a bf16 activation,
// written in the order r4d_gemm_w4a8_nt_m64's A fragment wants.
//
// One row per workgroup: absmax over K, then a second pass writing the bytes.
//
// Byte order: fragment element e of lane l is A[row][16s + 8*(e>>2) + 4*(l>>4) + (e&3)], two dwords
// eight bytes apart in a row-major activation. Each k step's sixteen bytes are written as
// [half 0's eight][half 1's eight], so the GEMM reads a fragment with one global_load_b64. Rows are
// not permuted; splitting A into two planes is slower (see the note in the GEMM).
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_chain.h"
#include <hip/hip_bf16.h>
#include <cstdint>

/* CHAIN: the chained form's wait and add (r4d_chain.h) around the same body; no block returns
 * early. */
template <int THR, bool CHAIN = false>
__device__ __forceinline__ void r4d_quant_act_i8_body(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    int K) {
  if (CHAIN) r4d_chain_wait();
  const int row = blockIdx.x;
  const __hip_bfloat16* a = A + (size_t)row * K;
  signed char* q = Q + (size_t)row * K;

  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += THR) {
    const float v = __bfloat162float(a[i]);
    m = fmaxf(m, fabsf(v));
  }
  __shared__ float red[THR / 32];
  m = r4d_wave_max(m);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < THR / 32; i++) m = fmaxf(m, red[i]);
  const float scale = fmaxf(m, 1e-8f) / 127.f;
  const float inv = 1.f / scale;
  if (threadIdx.x == 0) S[row] = scale;

  // Byte j of a k step comes from lane-half j>>3 and element j&7, i.e. source 8*((j&7)>>2) +
  // 4*(j>>3) + (j&3) inside the step.
  for (int i = threadIdx.x; i < K; i += THR) {
    const int s = i >> 4, j = i & 15;
    const int src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const float v = __bfloat162float(a[src]) * inv;
    const int t = __float2int_rn(v);
    q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
  if (CHAIN) r4d_chain_done();
}

template <int THR>
__global__ __launch_bounds__(THR) void r4d_quant_act_i8_kernel(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    int K) {
  r4d_quant_act_i8_body<THR, false>(A, Q, S, K);
}
template <int THR>
__global__ __launch_bounds__(THR) void r4dchain_quant_act_i8(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    int K) {
  r4d_quant_act_i8_body<THR, true>(A, Q, S, K);
}

// ---------------------------------------------------------------------------------------------
// Same quantisation, plus the per-row per-group sum of the int8 codes for an asymmetric weight
// grid: sum_k a_k (nibble_k - z_g) = sum_k a_k nibble_k - z_g * sum_k a_k, and the second term is
// computed once here instead of N times in the GEMM. Output is [K/group][Mpad], group-major with
// Mpad = ceil(M/16)*16, because a lane's eight accumulator elements are eight contiguous rows at
// one group, read as two aligned v4f. The caller zeroes it; pad rows are never written.
//
// The fragment permutation stays within a 16-element step and a group is whole steps, so group
// membership is unchanged.
#define R4D_QAI8_MAXG 256

#define R4D_QUANT_ACT_I8_ASUM_PARAMS                                                            \
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q,                          \
    float* __restrict__ S, float* __restrict__ Asum, int K, int group, int Mpad
template <int THR>
__device__ __forceinline__ void r4d_quant_act_i8_asum_body(R4D_QUANT_ACT_I8_ASUM_PARAMS) {
  const int row = blockIdx.x;
  const __hip_bfloat16* a = A + (size_t)row * K;
  signed char* q = Q + (size_t)row * K;
  const int G = K / group;

  __shared__ int gs[R4D_QAI8_MAXG];
  for (int g = threadIdx.x; g < G; g += THR) gs[g] = 0;

  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += THR) {
    const float v = __bfloat162float(a[i]);
    m = fmaxf(m, fabsf(v));
  }
  __shared__ float red[THR / 32];
  m = r4d_wave_max(m);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < THR / 32; i++) m = fmaxf(m, red[i]);
  const float scale = fmaxf(m, 1e-8f) / 127.f;
  const float inv = 1.f / scale;
  if (threadIdx.x == 0) S[row] = scale;

  for (int i = threadIdx.x; i < K; i += THR) {
    const int st = i >> 4, j = i & 15;
    const int src = (st << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const int t = __float2int_rn(__bfloat162float(a[src]) * inv);
    const int c = t < -128 ? -128 : (t > 127 ? 127 : t);
    q[i] = (signed char)c;
    atomicAdd(&gs[src / group], c);
  }
  __syncthreads();
  for (int g = threadIdx.x; g < G; g += THR)
    Asum[(size_t)g * Mpad + row] = (float)gs[g];
}

// The kernel around the body: CH `r4dchain` is the chained form (r4d_chain.h), taken at decode
// heights.
template <int THR, class CH>
__global__ __launch_bounds__(THR)
void r4d_quant_act_i8_asum_kernel(R4D_QUANT_ACT_I8_ASUM_PARAMS) {
  if (CH::on) r4d_chain_wait();
  r4d_quant_act_i8_asum_body<THR>(A, Q, S, Asum, K, group, Mpad);
  if (CH::on) r4d_chain_done();
}
#undef R4D_QUANT_ACT_I8_ASUM_PARAMS

// The plugin ABI entry point (RadLaunchFn); a refused shape is a status code, not a throw. See
// libr4d/r4d_args.h.
//
// The optional fourth operand (asum) selects the form: the row sums are a property of the call,
// so this is one row with a branch rather than two rows a constraint cannot tell apart.
enum { QA_X = 0, QA_Q, QA_S, QA_ASUM };

extern "C" int r4d_rad_quant_act_i8(const RadArgs* args, RadStream stream) {
  int emit_rc = RAD_OK;
  const RadTensor* x  = r4d_opt(args, QA_X);
  const RadTensor* q  = r4d_opt(args, QA_Q);
  const RadTensor* sc = r4d_opt(args, QA_S);
  const RadTensor* as = r4d_opt(args, QA_ASUM);
  if (!x || !q || !sc) return R4D_NO(RAD_E_INVAL);
  if (!r4d_contig(x) || !r4d_contig(q) || !r4d_contig(sc)) return R4D_NO(RAD_E_STRIDE);

  long long M = 0, K = 0;
  const int rc = r4d_rows_of(x, &M, &K);
  if (rc != RAD_OK) return R4D_NO(rc);
  if (K % 16) return R4D_NO(RAD_E_SHAPE);          // a k step is sixteen bytes of one row
  if (r4d_numel(q) != M * K) return R4D_NO(RAD_E_SHAPE);

  const int thr = r4d_act_threads((int)M, (int)K);   // the workgroup width, from M

  if (!as) {
    /* The chained form where the row resolved with `chain`, at decode heights. */
    const bool chain = r4d_geti_or(args, "chain", 0) != 0 && M <= 64;
#define R4D_QAI8_LAUNCH(T) emit_rc = chain                                                       \
    ? R4D_EMIT(stream, (r4dchain_quant_act_i8<T>), dim3((int)M), dim3(T), 0,                     \
               (const __hip_bfloat16*)r4d_p(x), (signed char*)r4d_wp(q), (float*)r4d_wp(sc),       \
               (int)K)                                                                           \
    : R4D_EMIT(stream, (r4d_quant_act_i8_kernel<T>), dim3((int)M), dim3(T), 0,                   \
               (const __hip_bfloat16*)r4d_p(x), (signed char*)r4d_wp(q), (float*)r4d_wp(sc),       \
               (int)K)
    switch (thr) {
      case 128:  R4D_QAI8_LAUNCH(128);  break;
      case 256:  R4D_QAI8_LAUNCH(256);  break;
      case 512:  R4D_QAI8_LAUNCH(512);  break;
      case 1024: R4D_QAI8_LAUNCH(1024); break;
      default:   return R4D_NO(RAD_E_INVAL);       // r4d_act_threads answers only these four
    }
#undef R4D_QAI8_LAUNCH
    return emit_rc;
  }

  if (!r4d_contig(as)) return R4D_NO(RAD_E_STRIDE);
  const long long group = r4d_geti_or(args, "group", R4D_W4_GROUP);
  if (group < 1 || K % group) return R4D_NO(RAD_E_SHAPE);
  if (K / group > R4D_QAI8_MAXG) return R4D_NO(RAD_E_SHAPE);   // the LDS accumulator's bound
  // asum is [K/group][Mpad], Mpad = ceil(M/16)*16, because the asymmetric GEMM reads whole
  // 16-row tiles. Sizing it from M alone overruns.
  const long long mpad = ((M + 15) / 16) * 16;
  if (r4d_numel(as) < (K / group) * mpad) return R4D_NO(RAD_E_SCRATCH);

  const bool chain = r4d_geti_or(args, "chain", 0) != 0 && M <= 64;
#define R4D_QAI8A_CH(T, CH_) R4D_EMIT(stream, (r4d_quant_act_i8_asum_kernel<T, CH_>),            \
    dim3((int)M), dim3(T), 0, (const __hip_bfloat16*)r4d_p(x), (signed char*)r4d_wp(q),          \
    (float*)r4d_wp(sc), (float*)r4d_wp(as), (int)K, (int)group, (int)mpad)
#define R4D_QAI8A_LAUNCH(T) emit_rc = chain ? R4D_QAI8A_CH(T, r4dchain)                          \
                                            : R4D_QAI8A_CH(T, R4dUnchained)
  switch (thr) {
    case 128:  R4D_QAI8A_LAUNCH(128);  break;
    case 256:  R4D_QAI8A_LAUNCH(256);  break;
    case 512:  R4D_QAI8A_LAUNCH(512);  break;
    case 1024: R4D_QAI8A_LAUNCH(1024); break;
    default:   return R4D_NO(RAD_E_INVAL);
  }
#undef R4D_QAI8A_LAUNCH
#undef R4D_QAI8A_CH
  return emit_rc;
}
