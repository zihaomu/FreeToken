/* avx_isa.h -- CPU feature detection, and the single list of ops the plugin is built around.
 *
 * Why dispatch. The plugin is a shared object the engine dlopens (spec.md §2) and runs on machines
 * it was not built on, so -march=native is out: an AVX-512 build SIGILLs on older CPUs. A baseline
 * build gives up most of the machine (an AVX-512 rmsnorm moves 64 bytes a cycle, SSE2 16). So the
 * kernels are compiled once per ISA level into separate objects with their own -m flags, and one
 * baseline translation unit (avx_dispatch.cpp) picks between them at load. Nothing outside the
 * per-level objects may contain an instruction above the baseline: the dispatcher, registry and
 * schemas are built without -m flags, since a stray intrinsic there would fault before dispatch.
 *
 * Levels:
 *   AVX_LEVEL_SCALAR (0)  x86-64 baseline. The compiler uses SSE2 for scalar float, but nothing is
 *                         hand-vectorised. Always selectable; every other level is checked
 *                         against it.
 *   AVX_LEVEL_AVX    (1)  AVX. 256-bit float only: Sandy/Ivy Bridge have no 256-bit integer ALU
 *                         and no FMA, so the bf16<->f32 conversions in avx_vec.h split into two
 *                         128-bit halves and every a*b+c is a separate mul and add.
 *   AVX_LEVEL_AVX2   (2)  AVX2 + FMA + F16C. 256-bit integer, vpermd/vpgatherdd, hardware f16.
 *                         Nearly every x86 machine since 2013.
 *   AVX_LEVEL_AVX512 (3)  AVX-512 F/BW/DQ/VL. Besides the width: 32 vector registers instead of 16
 *                         (room for the GEMM microkernel's 12x2 accumulator block) and write
 *                         masks, which remove the scalar tail loops.
 *
 * AVX-512 is checked as the conjunction F && BW && DQ && VL: Knights Landing has F without the
 * others. __builtin_cpu_supports is used instead of raw CPUID because it also checks XGETBV; a CPU
 * may enumerate AVX-512 while the OS has not enabled ZMM state, and using it then faults.
 *
 * AVX_OP_LIST is the only place the op set is written. It generates the table struct, the four
 * table declarations, the registry's baseline thunks and the harness's op enumeration. An op
 * missing a definition at any level is a link error, never a null table entry. (libref uses the
 * same technique for its shape hooks, libref/ref_ops.h.)
 */
#ifndef RAD_AVX_ISA_H
#define RAD_AVX_ISA_H

#include "rad_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    AVX_LEVEL_SCALAR = 0,
    AVX_LEVEL_AVX    = 1,
    AVX_LEVEL_AVX2   = 2,
    AVX_LEVEL_AVX512 = 3,
    AVX_N_LEVELS     = 4
};

/* The highest level this CPU can execute, computed once. Safe to call before rad_plugin_open. */
int         avx_isa_detect(void);
/* The level currently selected. Defaults to avx_isa_detect(). */
int         avx_isa_level(void);
const char* avx_isa_name(int level);

/* Force a level; for the checker only. An exported symbol rather than an environment variable, so
 * a deployment cannot change what a kernel computes by accident and the engine never reaches it.
 * Returns the level selected, min(requested, detected): a level the CPU cannot execute would
 * SIGILL. */
int         avx_isa_force(int level);

/* Run every parallel region with `threads` team threads whatever its work threshold, or restore
 * the thresholds with 0; for the checker only, like avx_isa_force. Returns the team size now
 * forced, 0 when the plugin was built without OpenMP. */
int         avx_par_force(int threads);

/* ================================================================== the op list
 * X(name) -- `name` is both the op name in the conventional vocabulary (docs/OPS.md) and the
 * suffix of every implementation symbol: avx_<name>_sc / _v1 / _v2 / _v3. */
#define AVX_OP_LIST(X)                                                                            \
    /* ---- elementwise and norms */                                                              \
    X(rmsnorm)              X(rmsnorm_p)            X(rmsnorm_add)          X(hc_enter)           \
    X(hc_read)              X(hc_write)             X(hc_project)           X(mtp_enter)          \
    X(layernorm)            X(mhc_pre)              X(mhc_post)             X(mhc_contract)       \
    X(kda_update)           X(mhc_pre_partial)      X(mhc_pre_finish)                             \
    X(swiglu_clamped)       X(ssd_update)                                                         \
    X(swiglu_step)                                                                                \
    X(router_sigmoid_topk)                                                                        \
    X(router_sigmoid_group_topk)                                                                  \
    X(dsa_pool)                                                                                   \
    X(dsa_query)                                                                                  \
    X(gemm_nt_f32)                                                                                \
    X(mla_project)                                                                                \
    X(dsa_score)                                                                                  \
    X(mla_attn)                                                                                   \
    X(mla_attn_paged)                                                                             \
    X(add)                  X(grid_embed)           X(grid_embed_aa)        X(causal_conv1d)      \
    X(grid_embed_cpe)       X(grid_embed_pad)       X(window_order)                               \
    X(lili_cand)            X(lili_attn)            X(lili_factor)          X(lili_walk)          \
    X(mul)                  X(silu_mul)             X(gelu_mul)             X(gelu)               \
    X(silu)                 X(sigmoid)              X(logit_softcap)        X(gather_rows)        \
    X(scatter_rows)         X(ngram_ids)            X(ple_conv_bias)                              \
    X(ple_gate)             X(ple_conv)             X(cast)                 X(softmax)            \
    X(scale_rows)           X(scale)                X(fill)                 X(act_mul)            \
    X(seg_mean_bf16_sum) \
    X(seg_mean)             X(mul_weight)           X(pool_rows)            X(pool_cols)          \
    X(point_fourier)                                                                              \
    X(seg_softmax_pool)                                                                           \
    X(relu)                 X(clamp)                X(conv2d)               X(conv1d_dw)          \
    X(rel_sinusoid)         X(seg_sinusoid)         X(conv2d_unfold)                              \
    X(position_scale)                                                                            \
    X(embed_attn)           X(ministral_rope)        X(pixtral_rope)         X(pixtral_merge)      \
    /* ---- quantisation and rotation */                                                          \
    X(quant_act_i8)         X(quant_act_i8g)        X(had_quant_act_i8)     X(quant_act_fp8)      \
    X(gram_accum)           X(rmsnorm_had_quant_i8) X(gated_had_quant_i8) \
    X(dequant)              X(quant_cols_fp8)       X(had_quant_act_i8g)    X(dequant_w4)         \
    X(rmsnorm_quant_fp8)    X(gated_quant_fp8)      \
    /* ---- gemm and the vocabulary edges */                                                      \
    X(gemm_nt)              X(gemm_nt_bias)         X(gemm_nt_q)            X(gemm_nt_q_bias)     \
    X(gemm_nt_q_gated)      X(embed_lookup)         X(embed_lookup_q)       X(logits_gemm)        \
    X(lora_shrink)          X(lora_expand)          X(lora_expand_mm)                             \
    /* ---- attention */                                                                          \
    X(attn_paged)           X(attn_paged_sink)      X(attn_dense)           X(attn_cross) X(attn_na3d) X(table_add) X(mix_rows) X(chan_stats) X(kv_store) X(kv_store_streamed) X(kv_resolve)             X(rope)               \
    X(rope_table)           X(qk_norm_rope)         X(qsa_block_key)        X(attn_local_rel)     \
    X(qsa_score)            X(qsa_select)           X(qsa_tail_store)       X(qsa_work)           \
    X(sam_attn)             \
    /* ---- mixture of experts */                                                                 \
    X(router_topk)          X(moe_scatter)          X(moe_gemm)             X(moe_gemm_q)         \
    X(moe_gather)           X(row_topk)             X(row_topk_merge)       X(logit_rerank)     \
    X(moe_gemm_q_mod4)      X(pair_argmax)          \
    X(router_topk_scatter)  X(router_gemm_topk_scatter)                                          \
    /* ---- gated delta net */                                                                    \
    X(gdn_conv_prep)        X(gdn_conv_update)      X(gdn_kkt_solve)        X(gdn_chunk_scan)     \
    X(gdn_recurrent_update) X(gdn_conv_recurrent_update)                X(gdn_gated_rmsnorm)  \
    /* ---- collectives */                                                                        \
    X(all_reduce)           X(all_gather)                                                         \
    /* ---- image generation */                                                                   \
    X(ln_mod)               X(gate_add)             X(timestep_embed)     \
    X(gather_rows2)         X(lane_gather)          X(lane_store)           X(randn_lanes)        \
    X(flow_euler)           X(chan_affine)          X(conv2d_nhwc)          X(l2norm_chan)        \
    X(d2s_add)              X(s2d_avg_add)          X(to_rgba8)             X(gelu_tanh)          \
    X(scatter_add_rows)     X(attn_prefix_self)     X(kd_input)             X(kd_step)            \
    X(gelu_quick)           X(im2col_nhwc)                                                        \
    /* ---- video generation */                                                                   \
    X(rms_mod)              X(flow_x0_step)         X(patch_unfold)       \
    X(vtile_gather)         X(vtile_stitch)         X(grid_pos3)            X(vsa_select)         \
    X(attn_vsa)             X(conv3d_ndhwc)         X(group_norm)           X(tconv_dw)           \
    X(resize_bilinear)      X(patch_fold)           X(flow_renoise)         X(flow_res_multistep) \
    X(cond_aug)             X(vtile_plan)           X(vtile_crop)           X(vtile_join)         \
    /* ---- LTX-2 */                                                                              \
    X(rope_split)           X(head_gate)            X(ltx_guide)            X(pixel_norm)         \
    X(d2s3d)                X(s2d3d)                X(pix_patch)                                  \
    X(stft_logmel)          X(bwe_up)               X(tile_grid)            X(tile_crop)          \
    X(tile_stitch)          X(vol_index)                                                          \
    /* ---- audio generation */                                                                   \
    X(conv1d)               X(conv_t1d)             X(snake)                X(cfg_topk_sample)    \
    X(rvq_embed)            X(frame_mix_store)      X(end_status)           X(frame_store)        \
    X(resample_nearest)     X(fourier_embed)        X(win_input)            X(win_prep)           \
    X(win_finish)           X(chan_split)           X(interleave2)          X(iota)               \
    X(cu_repeat)            X(snake_aa)             X(int_affine)           X(sub_i32)            \
    X(ras_pick)             X(vq_encode)            X(vq_euclidean)         X(elu)                \
    /* ---- sampling, and the drafter's token walk */                                             \
    X(sample_penalties)     X(sample_dry)           X(sample_temp)          X(sample_topk)        \
    X(sample_topp)          X(sample_minp)          X(sample_typical)       X(sample_xtc)         \
    X(sample_merge_topk)    X(sample_mask)          X(sample_pick)          X(sample_argmax)      \
    X(sample_ngram_ban)     X(sample_logprobs)      X(dflash_select)        X(dflash_conv)          \
    X(dflash_tree)          X(attn_paged_tree)      X(attn_paged_anchor_window)                   \
    X(dflash_markov_greedy) X(dflash_markov_greedy_f32)

/* One ISA level's table: one member per op, in AVX_OP_LIST order. A missing definition is a link
 * error from that level's avx_table.cpp object. */
typedef struct AvxKernelTable {
#define AVX_DECL_MEMBER(nm) int (*nm)(const RadArgs*, RadStream);
    AVX_OP_LIST(AVX_DECL_MEMBER)
#undef AVX_DECL_MEMBER
} AvxKernelTable;

/* One per level, defined in the level's own object. The dispatcher picks one. */
extern const AvxKernelTable avx_table_sc;
extern const AvxKernelTable avx_table_v1;
extern const AvxKernelTable avx_table_v2;
extern const AvxKernelTable avx_table_v3;

/* The table for a level, or the scalar table if that level was not built. */
const AvxKernelTable* avx_table_for(int level);
/* The live table the thunks call. Repointed by avx_isa_force. */
const AvxKernelTable* avx_table_live(void);

#ifdef __cplusplus
}
#endif
#endif /* RAD_AVX_ISA_H */
