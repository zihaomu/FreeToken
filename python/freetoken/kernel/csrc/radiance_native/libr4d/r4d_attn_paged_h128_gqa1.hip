/* r4d_attn_paged_h128_gqa1.hip -- head_dim 128 with ONE query head per KV head: plain MHA.
 *
 * Everything else in this library pages attention through a grouped ratio (4, 6, 8 or 12 queries
 * per KV head). The Unlimited-OCR decoder has 10 query heads over 10 KV heads at 128 wide, so the
 * ratio is 1 and none of the existing legs matches it; the engine reports every band of that
 * geometry as having no kernel rather than serving it on the host.
 *
 * The device code is the generic body (r4d_attn_paged_gqa8.hip, which is parameterised by
 * R4D_ATTN_GQA rather than fixed at 8) instantiated at ratio 1: the same templated prefill and
 * split-KV decode kernels, with GQA=1 substituted. No new kernel, no new masking path.
 *
 * This unit ships BOTH legs, unlike the h128/gqa4 drafter unit: a decoder runs prefill as well as
 * decode, and the caller-side fallback the drafter relies on is an order of magnitude slower.
 * The prefill option set is the h128 set (G8_P_OPT_128), which is the one compiled and measured
 * for this head width. */
#define R4D_ATTN_GQA  1
#define R4D_ATTN_H128 1
#define R4D_ATTN_H256 0

#include "r4d_attn_paged_gqa8.hip"
