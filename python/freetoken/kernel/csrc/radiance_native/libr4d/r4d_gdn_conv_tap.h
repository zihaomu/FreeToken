// r4d_gdn_conv_tap.h -- the arithmetic of one gated-delta-net convolution output, shared by the
// standalone conv kernels (r4d_gdn_conv_w4_h128_bf16.hip) and the recurrent update that folds the
// decode convolution into its prologue (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip).
// A single definition keeps the fold byte-identical to the standalone path: the reciprocal form
// below changes the last bit of some bf16 outputs.
#pragma once
#include "r4d_gdn_wmma.h"

#define R4D_GDN_CONV_W 4        // conv width; the state holds R4D_GDN_CONV_W - 1 taps of history

// The weight is bf16 from a checkpoint and f32 from a GGUF, which keeps ssm_conv1d unquantised.
// cp_taps is a channel's taps as one vector load; cp_tap widens one tap to f32, exactly for
// either type.
template <typename WT> struct cp_vec4;
template <> struct cp_vec4<unsigned short> { using T = ushort4; };
template <> struct cp_vec4<float>          { using T = float4; };
template <typename WT> using cp_taps = typename cp_vec4<WT>::T;
static_assert(R4D_GDN_CONV_W == 4, "cp_taps holds a channel's taps in one four-wide vector");

__device__ __forceinline__ float cp_tap(unsigned short w) { return bf2f(w); }
__device__ __forceinline__ float cp_tap(float w) { return w; }

// The result is rounded to bf16 immediately, so an IEEE-exact divide buys nothing: v_rcp_f32 is
// 1 ULP (~1e-7 relative) against bf16's 2^-8 quantum. The exact form is ~10 dependent instructions
// (v_div_scale, v_rcp, Newton, v_div_fmas, v_div_fixup); this is two. The conv update runs four
// waves on five workgroups, so its time is its instruction count.
__device__ __forceinline__ float cp_silu(float z) {
  return z * __builtin_amdgcn_rcpf(1.0f + __expf(-z));
}
