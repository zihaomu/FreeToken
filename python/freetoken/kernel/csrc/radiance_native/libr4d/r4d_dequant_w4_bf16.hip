// r4d_dequant_w4_bf16.hip - r4d_dequant_w4_bf16: the packed 4-bit weight back to a plain row-major
// bf16 [N, K].
//
// Used at prefill. W4A8 gains nothing there (iu8 and fp8 matrix rates are within a few percent and
// the shape is compute-bound), and the skinny W4A8 GEMM re-reads the whole weight slice once per
// 64 output rows, so its rate is flat in M. Dequantising one layer into scratch and running the
// tuned bf16 GEMM costs one extra write and read of the bf16 weight, a few percent of the GEMM.
//
// Two encodings share these bytes and differ by bit 3, so the caller must say which. The
// symmetric grid stores a two's complement code, value (n^8)-8, so the int8 GEMM can read the
// high-nibble byte signed and get 16*q. The asymmetric grid stores an unsigned nibble minus a
// per-group zero, corrected in the GEMM by an activation row-sum. Mixing them up agrees for n<8
// and is off by 16 above.
//
// Nearly lossless: a 4-bit code times an f16 scale fits bf16's 8-bit mantissa to within a ULP.
//
// Fragment order: radiance_w4.pack lays a dword out as [t][kb][lh][r][s] with eight nibbles along
// i, where t = n/16, r = n%16, kb = k/64, s = (k%64)/16, and the k inside the step is
// 4*lh + _KOFF[i] with _KOFF = {0,8,1,9,2,10,3,11}. One thread per dword scatters its eight
// nibbles. The scale dword at [t][g][r] holds the f16 scale low and -(1024+zero) high, so one
// routine serves the symmetric grid (zero == 8) and the asymmetric one.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_DQ_KPB     64
#define R4D_DQ_THREADS 256

__constant__ int r4d_dq_koff[8] = {0, 8, 1, 9, 2, 10, 3, 11};

__global__ __launch_bounds__(R4D_DQ_THREADS) void r4d_dequant_w4_bf16_kernel(
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    __hip_bfloat16* __restrict__ Out, int N, int K, int group, int twos) {
  const long total = (long)N * K / 8;                 // one thread per packed dword
  const int nkb = K / R4D_DQ_KPB, ng = K / group;
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total;
       idx += (long)gridDim.x * blockDim.x) {
    // [t][kb][lh][r][s]
    const int s  = (int)(idx & 3);
    const int r  = (int)((idx >> 2) & 15);
    const int lh = (int)((idx >> 6) & 1);
    const int kb = (int)((idx >> 7) % nkb);
    const int t  = (int)((idx >> 7) / nkb);
    const int row_n = t * 16 + r;
    const unsigned w = Wq[idx];

    const int kbase = kb * R4D_DQ_KPB + s * 16 + 4 * lh;
    const unsigned sz = Ws[(size_t)(t * ng + kbase / group) * 16 + r];
    const float sc = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
    const float nz = __half2float(__ushort_as_half((unsigned short)(sz >> 16)));
    const float zero = -nz - 1024.f;                  // exact for zero in 0..15

#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int n = (int)((w >> (4 * i)) & 0xF);
      const float q = twos ? (float)((n ^ 8) - 8) : (float)n;
      // Position (lh, i) holds kk = 4*lh + _KOFF[i]: lh=0 covers {0,8,1,9,2,10,3,11}, lh=1
      // the other eight.
      const int k = kbase + r4d_dq_koff[i];
      Out[(size_t)row_n * K + k] = __float2bfloat16((q - (twos ? 0.f : zero)) * sc);
    }
  }
}

void r4d_dequant_w4_bf16(long wq, long ws, long out, int N, int K, int group, int twos,
                         long stream) {
  if (N % 16) throw std::runtime_error("r4d dequant_w4_bf16: N must be a multiple of 16");
  if (K % R4D_DQ_KPB) throw std::runtime_error("r4d dequant_w4_bf16: K must be a multiple of 64");
  if (group < 1 || K % group) throw std::runtime_error(
      "r4d dequant_w4_bf16: group must divide K");
  const long total = (long)N * K / 8;
  const int blocks = (int)((total + R4D_DQ_THREADS - 1) / R4D_DQ_THREADS);
  hipLaunchKernelGGL(r4d_dequant_w4_bf16_kernel, dim3(blocks < 65535 ? blocks : 65535),
                     dim3(R4D_DQ_THREADS), 0, (hipStream_t)stream, (const unsigned*)wq,
                     (const unsigned*)ws, (__hip_bfloat16*)out, N, K, group, twos);
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
/* ------------------------------------------------------------------ dequant */

enum { D_WQ = 0, D_WS, D_Y };

extern "C" int r4d_rad_dequant_w4(const RadArgs* a, RadStream s) {
    const RadTensor* wq = r4d_opt(a, D_WQ);
    const RadTensor* ws = r4d_opt(a, D_WS);
    const RadTensor* y  = r4d_opt(a, D_Y);
    if (!wq || !ws || !y) return R4D_NO(RAD_E_INVAL);
    if (y->rank != 2 || !r4d_contig(y)) return R4D_NO(RAD_E_STRIDE);

    const long long N = y->shape[0];
    const long long K = y->shape[1];
    const long long group = r4d_geti_or(a, "group", R4D_W4_GROUP);
    const long long twos  = r4d_geti_or(a, "twos", 1);
    if (N % 16 || K % 64) return R4D_NO(RAD_E_SHAPE);
    if (group < 1 || K % group) return R4D_NO(RAD_E_SHAPE);
    if (twos != 0 && twos != 1) return R4D_NO(RAD_E_INVAL);

    R4D_TRY("dequant_w4_bf16",
        r4d_dequant_w4_bf16(r4d_addr(wq), r4d_addr(ws), r4d_addr(y),
                            (int)N, (int)K, (int)group, (int)twos, (long)(uintptr_t)s));
    return RAD_OK;
}

