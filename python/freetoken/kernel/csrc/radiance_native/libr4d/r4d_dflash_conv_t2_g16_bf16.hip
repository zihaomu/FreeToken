// DFlash2 grouped dynamic depthwise convolution, fused.
//
// Device math is in r4d_dflash_conv_body.h (HIP-header-free for standalone ISA review).
#include <hip/hip_runtime.h>
#include "r4d_gdn_args.h"
#include "r4d_dflash_conv_body.h"
#include "r4d_chain.h"

#define DFC_THREADS 128
#define DFC_VEC     8      // bf16 lanes per thread: one dwordx4 in and one out

template <int TAPS, int GROUP>
__global__ __launch_bounds__(DFC_THREADS) void r4d_dflash_conv_kernel(
    const r4d_u16* __restrict__ x, const r4d_u16* __restrict__ delta,
    const r4d_u16* __restrict__ base, r4d_u16* __restrict__ out,
    int H, int dpitch, int NG, int vpr, int blockmask)
{
  const int hv = blockIdx.x * DFC_THREADS + threadIdx.x;
  r4d_dflash_conv_body<TAPS, GROUP>(x, delta, base, out, H, dpitch, NG, vpr,
                                    blockmask, blockIdx.y, hv);
}

// The chained form (r4d_chain.h): the body returns for lanes past the row, the block still adds.
template <int TAPS, int GROUP>
__global__ __launch_bounds__(DFC_THREADS) void r4dchain_dflash_conv(
    const r4d_u16* __restrict__ x, const r4d_u16* __restrict__ delta,
    const r4d_u16* __restrict__ base, r4d_u16* __restrict__ out,
    int H, int dpitch, int NG, int vpr, int blockmask)
{
  r4d_chain_wait();
  const int hv = blockIdx.x * DFC_THREADS + threadIdx.x;
  r4d_dflash_conv_body<TAPS, GROUP>(x, delta, base, out, H, dpitch, NG, vpr,
                                    blockmask, blockIdx.y, hv);
  r4d_chain_done();
}

// x     [T, H]        bf16, row pitch H
// delta [T, taps, NG] bf16, row pitch dpitch: a slice of the [T, 2, taps, NG] kernel projection,
//                     so dpitch is 2*taps*NG. The packed pitch (taps*NG) would silently read the
//                     other half's coefficients.
// base  [taps, H]     bf16, contiguous
// out   [T, H]        bf16. Must not alias x: tap 1 of row t reads input row t-1.
static int dflash_conv_launch(const void* x, const void* delta, const void* base, void* out,
                              int T, int H, int dpitch, int NG, int taps, int group,
                              int block_size, void* stream, bool chain);
extern "C" int r4d_dflash_conv_t2_g16_bf16(
    const void* x, const void* delta, const void* base, void* out,
    int T, int H, int dpitch, int NG, int taps, int group, int block_size, void* stream)
{
  return dflash_conv_launch(x, delta, base, out, T, H, dpitch, NG, taps, group, block_size,
                            stream, false);
}
/* `chain`: the chained form at decode heights (r4d_rad_dflash_conv passes the row's parameter). */
static int dflash_conv_launch(const void* x, const void* delta, const void* base, void* out,
                              int T, int H, int dpitch, int NG, int taps, int group,
                              int block_size, void* stream, bool chain)
{
  if (taps != 2 || group != 16) return -1;
  if (H % (group * DFC_VEC) || NG != H / group) return -1;
  if (block_size < 1 || (block_size & (block_size - 1))) return -1;  // body masks: pow2 only
  if (T <= 0) return 0;

  const int vpr = H / DFC_VEC;
  dim3 grid((vpr + DFC_THREADS - 1) / DFC_THREADS, T), blk(DFC_THREADS);
  hipLaunchKernelGGL((chain && T <= 64 ? r4dchain_dflash_conv<2, 16>
                                         : r4d_dflash_conv_kernel<2, 16>),
                     grid, blk, 0, (hipStream_t)stream,
                     (const r4d_u16*)x, (const r4d_u16*)delta, (const r4d_u16*)base,
                     (r4d_u16*)out, H, dpitch, NG, vpr, block_size - 1);
  return (int)hipGetLastError();
}

// ============================================================== the plugin ABI entry points
// Shared operand helpers: r4d_gdn_args.h.
/* ------------------------------------------------------------------ dflash conv */

enum { DC_X = 0, DC_DELTA, DC_BASE, DC_OUT };

extern "C" int r4d_rad_dflash_conv(const RadArgs* a, RadStream s) {
    const RadTensor* x  = r4d_opt(a, DC_X);
    const RadTensor* d  = r4d_opt(a, DC_DELTA);
    const RadTensor* bs = r4d_opt(a, DC_BASE);
    const RadTensor* o  = r4d_opt(a, DC_OUT);
    if (!x || !d || !bs || !o) return R4D_NO(RAD_E_INVAL);
    if (x->rank != 2 || o->rank != 2 || d->rank < 2) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(x) || !r4d_contig(o) || !r4d_contig(bs)) return R4D_NO(RAD_E_STRIDE);

    /* out must not alias x (the kernel reads x while writing out); nothing downstream checks it. */
    if (x->data == o->data) return R4D_NO(RAD_E_INVAL);

    const long long T = x->shape[0];
    const long long H = x->shape[1];
    if (o->shape[0] != T || o->shape[1] != H) return R4D_NO(RAD_E_SHAPE);

    /* delta is a slice of the [T, 2, taps, NG] projection: its row pitch (from stride[0]) is
     * 2*taps*NG, not the slice width. */
    const long long dpitch = r4d_row_pitch(d);
    if (dpitch < 0) return R4D_NO(RAD_E_STRIDE);

    const int taps  = (int)r4d_geti_or(a, "taps", 2);
    const int group = (int)r4d_geti_or(a, "group_size", 16);
    const int NG    = (int)r4d_geti_or(a, "NG", 0);
    const int bsz   = (int)r4d_geti_or(a, "block_size", 0);
    if (NG <= 0 || bsz <= 0) return R4D_NO(RAD_E_INVAL);
    /* The kernel masks with block_size-1, so block_size must be a power of two; a row constraint
     * cannot express that, so it is checked here. */
    if (bsz & (bsz - 1)) return R4D_NO(RAD_E_INVAL);
    if (dpitch != 2LL * taps * NG) return R4D_NO(RAD_E_STRIDE);

    return R4D_RAW(dflash_conv_launch(x->data, d->data, bs->data, o->data, (int)T,
                                      (int)H, (int)dpitch, NG, taps, group, bsz, s,
                                      r4d_geti_or(a, "chain", 0) != 0));
}

