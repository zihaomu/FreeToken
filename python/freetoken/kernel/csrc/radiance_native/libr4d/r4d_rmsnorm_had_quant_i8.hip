// r4d_rmsnorm_had_quant_i8.hip - r4d_rmsnorm_had_quant_i8: the residual add, the RMS norm, the
// block Hadamard and the int8 activation quantiser, in one pass over the row.
//
// W4A8 needs a rotated int8 activation in front of every quantised linear: four per layer, 256 per
// decode step at 64 layers. As separate kernels that is 0.4-1.0 ms of launch overhead per step
// (the arithmetic is ~14 ops per element), and vLLM's fp8 path already fuses its quantiser into the
// norm. Fused, the rotation and quantiser ride on a pass that already has the row in registers.
//
// Outputs: the int8 row in r4d_gemm_w4a8_nt_m64's A-fragment byte order, and optionally the bf16
// unrotated norm output for a GDN block's in_proj_a / in_proj_b, which are in
// modules_to_not_convert and read the same post-norm tensor (null skips the write). They keep
// their original weights; rotating them offline would round W_a into bf16 for no gain.
//
// `wadd` is added to every gain: Qwen3.5/Qwen3-Next norms are GemmaRMSNorm, `x_hat * (1 + w)` with
// a zero-centred gamma, which vLLM forms as `weight.float() + 1.0` in fp32. Rounding `1 + w` to
// bf16 first loses most of w (a bf16 ULP near 1 is 2^-7, so 1.0031 becomes 1.0000), a systematic
// 0.3% per-channel error far larger than the int8 quantiser's. Added here after the bf16 load, in
// fp32, it matches vLLM bit for bit. Pass 0.0f for a plain (Llama-style) RMS norm.
//
// The residual is updated in place as vLLM's fused_add_rms_norm does: residual += x, then the norm
// of the sum. A null residual is the non-additive form.
//
// The transform does not wait for the norm: the FWHT is linear and `inv_rms` a per-row scalar, so
// FWHT(x * Wg * inv_rms) = inv_rms * FWHT(x * Wg). A positive scalar cancels out of the int8 codes
// (round(value * 127 / absmax(value))), so inv_rms only enters S[row], like 1/sqrt(had). The sum of
// squares and the row max reduce behind the same barrier.
//
// The row goes through LDS once: staging it across the reduction and rewriting it costs an LDS
// write, read and rewrite per element (3.5x the plain quantiser at M=4096, against 1.4x for
// gated_had_quant_i8). Each wave keeps its blocks in registers across the reduction (~34 floats per
// lane at the widest per-rank K); LDS holds only the permuted store's source.
//
// Re-reading the residual in a second pass is not an alternative: it is stored as bf16, and
// normalising from the bf16-rounded sum gives thousands of +/-1 code errors and a 1e-3 scale error
// (against 5e-7). Storing bf16 is correct, since the next layer reads it; normalising from it is
// not.
//
// As in r4d_had_quant_act_i8, the butterfly computes the unnormalised transform and the row scale
// carries the 1/sqrt(had).
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_fwht.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

// Register staging bound: blocks-per-wave * VPL floats per lane, ~K/256 at NW=8, so 72 covers
// every per-rank K up to 18432 at every compiled width.
#define R4D_RHQ_MAXVALS 72

template <int HAD, int THR, bool WOUT>
__global__ __launch_bounds__(THR) void r4d_rmsnorm_had_quant_i8_kernel(
    const __hip_bfloat16* __restrict__ X, __hip_bfloat16* __restrict__ RES,
    const __hip_bfloat16* __restrict__ Wg, __hip_bfloat16* __restrict__ OUTB,
    signed char* __restrict__ Q, float* __restrict__ S, int K, float eps, float wadd,
    float rnorm) {
  constexpr int VPL = HAD / 32, NW = THR / 32;
  constexpr int MAXB = R4D_RHQ_MAXVALS / VPL;
  extern __shared__ float lds[];
  float* red = lds + K;
  const int row = blockIdx.x;
  const __hip_bfloat16* x = X + (size_t)row * K;
  __hip_bfloat16* res = RES ? RES + (size_t)row * K : nullptr;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int nblk = K / HAD;

  // Kept only for the bf16 consumer: the unrotated output needs inv_rms, so it cannot be written
  // before the reduction.
  float u[WOUT ? MAXB : 1][VPL];

  float ss = 0.f, m = 0.f;
  int nb = 0;
  for (int blk = wave; blk < nblk; blk += NW, nb++) {
    const int base = blk * HAD + lane * VPL;
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; j++) {
      float t = __bfloat162float(x[base + j]);
      if (res) {
        t += __bfloat162float(res[base + j]);
        res[base + j] = __float2bfloat16(t);
      }
      ss += t * t;
      v[j] = t * (__bfloat162float(Wg[base + j]) + wadd);
      if (WOUT) u[nb][j] = v[j];
    }
    r4d_fwht_wave<HAD>(v, lane);
#pragma unroll
    for (int j = 0; j < VPL; j++) {
      lds[base + j] = v[j];
      m = fmaxf(m, fabsf(v[j]));
    }
  }

  // one barrier for both reductions
  ss = r4d_wave_sum(ss);
  m = r4d_wave_max(m);
  if (lane == 0) { red[wave] = ss; red[NW + wave] = m; }
  __syncthreads();
  ss = 0.f; m = 0.f;
#pragma unroll
  for (int i = 0; i < NW; i++) { ss += red[i]; m = fmaxf(m, red[NW + i]); }
  const float inv_rms = rsqrtf(ss / (float)K + eps);
  m = fmaxf(m, 1e-8f);
  const float inv = 127.f / m;
  if (threadIdx.x == 0) S[row] = m * inv_rms * rnorm / 127.f;

  if (WOUT) {
    nb = 0;
    for (int blk = wave; blk < nblk; blk += NW, nb++) {
      const int base = blk * HAD + lane * VPL;
#pragma unroll
      for (int j = 0; j < VPL; j++)
        OUTB[(size_t)row * K + base + j] = __float2bfloat16(u[nb][j] * inv_rms);
    }
  }

  signed char* q = Q + (size_t)row * K;
  for (int i = threadIdx.x; i < K; i += THR) {
    const int s = i >> 4, j = i & 15;
    const int sr = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const int t = __float2int_rn(lds[sr] * inv);
    q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
}

// The launch, as the plugin ABI's RadLaunchFn. See the note at the head of r4d_quant_act_i8.hip.
//
// The residual and the bf16 echo are optional operands that change the computed chain: with a
// residual it is `add`, `rmsnorm`, quantiser; without, the last two. So the fusion declaration
// (RadFuseFn) takes the present-operand mask, since the chain length depends on an operand.
enum { RN_X = 0, RN_RESID, RN_W, RN_Q, RN_S, RN_OUTBF16 };

extern "C" int r4d_rad_rmsnorm_had_quant_i8(const RadArgs* args, RadStream stream) {
  const RadTensor* x  = r4d_opt(args, RN_X);
  const RadTensor* rs = r4d_opt(args, RN_RESID);
  const RadTensor* w  = r4d_opt(args, RN_W);
  const RadTensor* q  = r4d_opt(args, RN_Q);
  const RadTensor* sc = r4d_opt(args, RN_S);
  const RadTensor* ob = r4d_opt(args, RN_OUTBF16);
  if (!x || !w || !q || !sc) return R4D_NO(RAD_E_INVAL);
  if (!r4d_contig(x) || !r4d_contig(q) || !r4d_contig(sc) || !r4d_contig(w))
    return R4D_NO(RAD_E_STRIDE);
  if (rs && !r4d_contig(rs)) return R4D_NO(RAD_E_STRIDE);
  if (ob && !r4d_contig(ob)) return R4D_NO(RAD_E_STRIDE);

  long long M = 0, K = 0;
  int rc = r4d_rows_of(x, &M, &K);
  if (rc != RAD_OK) return R4D_NO(rc);
  if (K % 16) return R4D_NO(RAD_E_SHAPE);
  if (r4d_numel(w) != K) return R4D_NO(RAD_E_SHAPE);

  const long long had = r4d_geti_or(args, "group", 0);
  if (had != 128 && had != 256 && had != 512) return R4D_NO(RAD_E_SHAPE);
  // A rotation block cannot straddle a tensor-parallel shard: down_proj is 8704 per rank, so 1024
  // is unavailable.
  if (K % had) return R4D_NO(RAD_E_SHAPE);

  const float eps = (float)r4d_getf(args, "eps", 1e-6);
  // GemmaRMSNorm's `1 +`, added in fp32 after the load (see the file header); 0 = plain RMS.
  const float wadd = (float)r4d_getf(args, "wadd", 0.0);

  const int thr = r4d_act_threads((int)M, (int)K);
  if ((K / had + thr / 32 - 1) / (thr / 32) * (had / 32) > R4D_RHQ_MAXVALS)
    return R4D_NO(RAD_E_SHAPE);                    // K too large for the register staging here
  const size_t shmem = (size_t)K * sizeof(float) + 2 * (thr / 32) * sizeof(float);
  if (shmem > 65536) return R4D_NO(RAD_E_SCRATCH); // K too large to stage in LDS
  const float rn = 1.f / sqrtf((float)had);
#define R4D_RHQ_LAUNCH(H, T, W) hipLaunchKernelGGL((r4d_rmsnorm_had_quant_i8_kernel<H, T, W>), \
    dim3((int)M), dim3(T), shmem, (hipStream_t)stream, (const __hip_bfloat16*)r4d_p(x), \
    (__hip_bfloat16*)r4d_wp(rs), (const __hip_bfloat16*)r4d_p(w), (__hip_bfloat16*)r4d_wp(ob), \
    (signed char*)r4d_wp(q), (float*)r4d_wp(sc), (int)K, eps, wadd, rn)
#define R4D_RHQ_T(H, W) switch (thr) { \
    case 256: R4D_RHQ_LAUNCH(H, 256, W); break; \
    case 512: R4D_RHQ_LAUNCH(H, 512, W); break; \
    case 1024: R4D_RHQ_LAUNCH(H, 1024, W); break; \
    default: return R4D_NO(RAD_E_INVAL); }
#define R4D_RHQ_W(W) switch (had) { \
    case 128: R4D_RHQ_T(128, W); break; \
    case 256: R4D_RHQ_T(256, W); break; \
    default:  R4D_RHQ_T(512, W); break; }
  if (ob) { R4D_RHQ_W(true); } else { R4D_RHQ_W(false); }
#undef R4D_RHQ_W
#undef R4D_RHQ_T
#undef R4D_RHQ_LAUNCH
  return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}
