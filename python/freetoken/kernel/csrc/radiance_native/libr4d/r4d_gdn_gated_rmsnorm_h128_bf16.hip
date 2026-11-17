// r4d_gdn_gated_rmsnorm_h128_bf16.hip -- the gated RMS norm a gated-delta-net layer applies to its own
// output before the output projection:
//
//     out[r][d] = x[r][d] . rsqrt(mean_d(x[r][:]^2) + eps) . w[d] . act(z[r][d])
//
// one row per (token, head), 128 wide. Replaces FLA's `rmsnorm_fn` (mamba/ops/layernorm_gated).
//
// A separate kernel because the chunked scan's workgroup owns only 64 of a row's 128 v channels
// (BV=64; 128 would leave 24 workgroups on 64 CUs). The decode recurrent update owns whole rows and
// fuses this into its epilogue (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip).
//
// One wave per row, four channels per lane: a barrier-free five-shuffle reduction, and one 8-byte
// access per lane for each load and the store.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_gdn_args.h"
#include "r4d_gdn_wmma.h"
#include "r4d_common.h"   // r4d_wave_sum

#define GN_D    128             // channels per row
#define GN_DPL    4             // channels per lane
#define GN_NW     8             // rows in flight per workgroup

__device__ __forceinline__ float gn_act(float z, int kind) {
  const float s = 1.0f / (1.0f + __expf(-z));
  return kind ? s : z * s;      // 1 = sigmoid, 0 = silu / swish
}

template <int ACT>
__global__ __launch_bounds__(GN_NW * 32)
void r4d_gdn_gated_rmsnorm_kernel(const unsigned short* __restrict__ x,
                                  const unsigned short* __restrict__ z,
                                  const float* __restrict__ w,
                                  unsigned short* __restrict__ o,
                                  long rows, long xrow, long zrow, long orow,
                                  long zheads, long ztok, float eps)
{
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const long r = (long)blockIdx.x * GN_NW + warp;
  if (r >= rows) return;
  const int d0 = lane * GN_DPL;
  // The gate may be a column view of a fused [q|k|v|z] projection: within a token the `zheads`
  // gate rows are `zrow` apart and the next token is `ztok` away. zheads <= 0 is the flat case.
  const long zoff = zheads > 0 ? (r / zheads) * ztok + (r % zheads) * zrow : r * zrow;

  const ushort4 xv = *(const ushort4*)(x + r * xrow + d0);
  float v[GN_DPL] = {bf2f(xv.x), bf2f(xv.y), bf2f(xv.z), bf2f(xv.w)};

  float ss = 0.0f;
#pragma unroll
  for (int c = 0; c < GN_DPL; ++c) ss += v[c] * v[c];
  ss = r4d_wave_sum(ss);
  const float scale = __frsqrt_rn(ss * (1.0f / GN_D) + eps);

  const ushort4 zv = *(const ushort4*)(z + zoff + d0);
  const float zz[GN_DPL] = {bf2f(zv.x), bf2f(zv.y), bf2f(zv.z), bf2f(zv.w)};
  const float4 ww = *(const float4*)(w + d0);
  const float wv[GN_DPL] = {ww.x, ww.y, ww.z, ww.w};

  unsigned short out[GN_DPL];
#pragma unroll
  for (int c = 0; c < GN_DPL; ++c)
    out[c] = f2bf(v[c] * scale * wv[c] * gn_act(zz[c], ACT));
  *(ushort4*)(o + r * orow + d0) = make_ushort4(out[0], out[1], out[2], out[3]);
}

extern "C" int r4d_gdn_gated_rmsnorm_h128_bf16(const void* x, const void* z, const void* w,
                                               void* o, long rows, long xrow, long zrow,
                                               long orow, long zheads, long ztok, int width,
                                               float eps, int act, void* stream)
{
  if (width != GN_D) return -1;
  dim3 grid((unsigned)((rows + GN_NW - 1) / GN_NW)), blk(GN_NW * 32);
  if (act)
    hipLaunchKernelGGL((r4d_gdn_gated_rmsnorm_kernel<1>), grid, blk, 0, (hipStream_t)stream,
                       (const unsigned short*)x, (const unsigned short*)z, (const float*)w,
                       (unsigned short*)o, rows, xrow, zrow, orow, zheads, ztok, eps);
  else
    hipLaunchKernelGGL((r4d_gdn_gated_rmsnorm_kernel<0>), grid, blk, 0, (hipStream_t)stream,
                       (const unsigned short*)x, (const unsigned short*)z, (const float*)w,
                       (unsigned short*)o, rows, xrow, zrow, orow, zheads, ztok, eps);
  return (int)hipGetLastError();
}

// ============================================================== the plugin ABI entry points
// Shared operand helpers: r4d_gdn_args.h.
/* ------------------------------------------------------------------ gated rms norm */

enum { GN_X = 0, GN_Z, GN_W, GN_O };

extern "C" int r4d_rad_gdn_gated_rmsnorm(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, GN_X);
    const RadTensor* z = r4d_opt(a, GN_Z);
    const RadTensor* w = r4d_opt(a, GN_W);
    const RadTensor* o = r4d_opt(a, GN_O);
    /* bf16 x, z, out; fp32 weight. */
    R4D_DT(x, RAD_BF16); R4D_DT(z, RAD_BF16); R4D_DT(o, RAD_BF16); R4D_DT(w, RAD_F32);
    if (!x || !z || !w || !o) return R4D_NO(RAD_E_INVAL);
    if (x->rank < 2 || z->rank < 2 || o->rank < 2) return R4D_NO(RAD_E_SHAPE);

    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);

    /* One row is one (token, head). x and out are [tokens, heads, channels] (rank 3, as the chunk
     * scan requires) while the gate may be the flat [tokens, heads*channels] projection. So the
     * width comes from `channels` (or x's last axis), the row count from x, and each operand's
     * pitch between consecutive (token, head) rows is read per its shape. */
    const long long width = r4d_geti_or(a, "channels", x->shape[x->rank - 1]);
    if (width <= 0 || r4d_numel(x) % width) return R4D_NO(RAD_E_SHAPE);
    const long long rows = r4d_numel(x) / width;
    if (r4d_numel(z) < rows * width || r4d_numel(o) < rows * width) return R4D_NO(RAD_E_SHAPE);

    const long long xrow = gn_row_pitch(x, width);
    const long long orow = gn_row_pitch(o, width);
    if (xrow < 0 || orow < 0) return R4D_NO(RAD_E_STRIDE);

    /* The gate may also be a column slice of a wider [q|k|v|z] row (fp8 delta net): heads are
     * `width` apart within a token and tokens are the slice's stride apart, so the kernel also
     * takes the head count. zheads stays 0 for the flat layouts. */
    long long zrow = gn_row_pitch(z, width), zheads = 0, ztok = 0;
    if (zrow < 0 && z->rank == 2 && z->stride[1] == 1 && z->shape[1] % width == 0) {
        zheads = z->shape[1] / width;
        ztok   = z->stride[0];
        zrow   = width;
        /* The gate's head count must divide the row count and match x's head axis if it has one;
         * a mismatch would silently interleave rows. */
        if (zheads <= 0 || rows % zheads) return R4D_NO(RAD_E_SHAPE);
        if (x->rank >= 3 && x->shape[x->rank - 2] != zheads) return R4D_NO(RAD_E_SHAPE);
    }
    if (zrow < 0) return R4D_NO(RAD_E_STRIDE);

    const float eps = (float)r4d_getf(a, "eps", 1e-6);
    /* "silu" (0) or "sigmoid" (1). */
    const int   act = r4d_act_code(a, "act", nullptr, -1);
    if (act != 0 && act != 1) return R4D_NO(RAD_E_INVAL);

    return R4D_RAW(r4d_gdn_gated_rmsnorm_h128_bf16(x->data, z->data, w->data, o->data,
                                                   (long)rows, (long)xrow, (long)zrow,
                                                   (long)orow, (long)zheads, (long)ztok,
                                                   (int)width, eps, act, s));
}

