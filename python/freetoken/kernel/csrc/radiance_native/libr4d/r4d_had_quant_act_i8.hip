// r4d_had_quant_act_i8.hip - r4d_had_quant_act_i8: a block Hadamard along K, then per-row symmetric
// int8 quantisation, written in the order r4d_gemm_w4a8_nt_m64's A fragment wants.
//
// Why the rotation: plain per-token int8 activations are unusable on these models (KL against bf16
// an order of magnitude worse than per-token e4m3); a block Hadamard first makes int8 several times
// better than e4m3. A rotation fixes range, not precision: int8 spends its grid covering a few
// outlier channels, and mixing each value with 127 others removes them. It does not help e4m3
// (marginally worse), whose four exponent bits already carry the outliers. It must happen before
// the absmax, since the benefit is a smaller row maximum, so it belongs in the quantiser.
//
// The seven butterfly stages (~14 ops per element) ride on a launch the quantiser pays anyway. The
// remaining cost is one launch per rotated site per layer, which fusing into the producing RMSNorm
// or SwiGLU removes.
//
// Width: 128, 256 and 512 are built; 1024 is not, for sharding reasons. A rotation block cannot
// straddle a tensor-parallel boundary, so the width must divide the per-rank K, and a down_proj of
// 17408 at TP=2 is 8704 = 2^9 * 17. The weight error barely moves from 128 to 1024, so the width
// is picked on the activation's terms alone.
//
// The transform is in registers (r4d_fwht.h): one wave owns a `had`-wide block, and the only LDS
// traffic left is one staging write so the row absmax can see the whole row.
//
// Normalisation is folded into the scale. The stages compute the unnormalised transform (sqrt(had)
// times the orthonormal one); quantising with inv = 127/absmax_unnorm gives exactly the normalised
// codes, and S[row] = absmax_unnorm / (sqrt(had) * 127) undoes both.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_fwht.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

template <int HAD, int THR>
__global__ __launch_bounds__(THR) void r4d_had_quant_act_i8_kernel(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    int K, float rnorm) {
  constexpr int VPL = HAD / 32;
  extern __shared__ float lds[];                 // K floats of transformed row, then the reduction
  float* red = lds + K;
  const int row = blockIdx.x;
  const __hip_bfloat16* a = A + (size_t)row * K;
  signed char* q = Q + (size_t)row * K;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  constexpr int NW = THR / 32;

  float m = 0.f;
  for (int blk = wave; blk < K / HAD; blk += NW) {
    const int base = blk * HAD + lane * VPL;
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; j++) v[j] = __bfloat162float(a[base + j]);
    r4d_fwht_wave<HAD>(v, lane);
#pragma unroll
    for (int j = 0; j < VPL; j++) {
      lds[base + j] = v[j];
      m = fmaxf(m, fabsf(v[j]));
    }
  }
  m = r4d_wave_max(m);
  if (lane == 0) red[wave] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < NW; i++) m = fmaxf(m, red[i]);
  m = fmaxf(m, 1e-8f);
  const float inv = 127.f / m;
  if (threadIdx.x == 0) S[row] = m * rnorm / 127.f;

  // Same fragment permutation as r4d_quant_act_i8: byte j of a k step comes from lane-half j>>3 and
  // element j&7, i.e. source 8*((j&7)>>2) + 4*(j>>3) + (j&3) inside the step.
  for (int i = threadIdx.x; i < K; i += THR) {
    const int s = i >> 4, j = i & 15;
    const int src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const int t = __float2int_rn(lds[src] * inv);
    q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
}

// The launch, as the plugin ABI's RadLaunchFn. See the note at the head of r4d_quant_act_i8.hip on
// the argument list and the status return.
//
// The width set belongs to this file, not the row: the rows only bound `had <= 512`, and the kernel
// compiles exactly 128, 256 and 512, so a legal-looking 64 is refused here.
enum { HQ_X = 0, HQ_Q, HQ_S };

extern "C" int r4d_rad_had_quant_act_i8(const RadArgs* args, RadStream stream) {
  const RadTensor* x  = r4d_opt(args, HQ_X);
  const RadTensor* q  = r4d_opt(args, HQ_Q);
  const RadTensor* sc = r4d_opt(args, HQ_S);
  if (!x || !q || !sc) return R4D_NO(RAD_E_INVAL);
  if (!r4d_contig(x) || !r4d_contig(q) || !r4d_contig(sc)) return R4D_NO(RAD_E_STRIDE);

  long long M = 0, K = 0;
  const int rc = r4d_rows_of(x, &M, &K);
  if (rc != RAD_OK) return R4D_NO(rc);
  if (K % 16) return R4D_NO(RAD_E_SHAPE);

  const long long had = r4d_geti_or(args, "group", 0);
  if (had != 128 && had != 256 && had != 512) return R4D_NO(RAD_E_SHAPE);
  // A rotation block cannot straddle a tensor-parallel shard, so it must divide the per-rank K
  // (down_proj's 8704 admits at most 512).
  if (K % had) return R4D_NO(RAD_E_SHAPE);

  const int thr = r4d_act_threads((int)M, (int)K);
  const size_t shmem = (size_t)K * sizeof(float) + (thr / 32) * sizeof(float);
  if (shmem > 65536) return R4D_NO(RAD_E_SCRATCH);   // K too large to stage in LDS
  const float rn = 1.f / sqrtf((float)had);
#define R4D_HQ_LAUNCH(H, T) hipLaunchKernelGGL((r4d_had_quant_act_i8_kernel<H, T>), dim3((int)M), \
    dim3(T), shmem, (hipStream_t)stream, (const __hip_bfloat16*)r4d_p(x), \
    (signed char*)r4d_wp(q), (float*)r4d_wp(sc), (int)K, rn)
#define R4D_HQ_T(H) switch (thr) { \
    case 256: R4D_HQ_LAUNCH(H, 256); break; \
    case 512: R4D_HQ_LAUNCH(H, 512); break; \
    case 1024: R4D_HQ_LAUNCH(H, 1024); break; \
    default: return R4D_NO(RAD_E_INVAL); }
  switch (had) {
    case 128: R4D_HQ_T(128); break;
    case 256: R4D_HQ_T(256); break;
    default:  R4D_HQ_T(512); break;
  }
#undef R4D_HQ_T
#undef R4D_HQ_LAUNCH
  return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}
