# libr4d serves RDNA4 only, declared here and nowhere else. Its kernels rely on wave32, the gfx12
# WMMA fragment layouts and gfx12-only instructions such as global_load_tr_b128; built for other
# targets they would compute garbage.
#
# A build targeting other cards skips libr4d. The engine then has no device kernel library for that
# card and refuses ops only a reference library implements, unless
# --debug-accept-reference-kernels is passed (spec §17). The supported fix is a kernel library for
# that card.
#
# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S libr4d -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`, naming the HIP compiler as for
# the tree); see cmake/RadAddPlugin.cmake.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libr4d LANGUAGES CXX HIP)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_plugin_gpu_targets(R4D_GPU_TARGETS "^gfx12")
if(NOT R4D_GPU_TARGETS)
  message(STATUS
    "libr4d: RDNA4 (gfx12xx) only, and this build targets ${RAD_GPU_TARGETS} -- not built")
  return()
endif()

# R4D_UNITS is the definitive list of libr4d's device translation units, as "unit|extra flags".
# The GDN chunk scan needs -mcumode. The quantised GEMMs' group sizes are source constants, not
# build options: the host packer reads r4d_args.h and the units their own defaults, which must
# agree (r4d_gemm_args.h static_asserts the w4 and w8 groups).
#
# Every unit is compiled -ffp-contract=off. The quantised all-reduce requires it: the two ranks sum
# two dequantised products per element in opposite order, and an FMA on either side makes them
# differ by ~1 ULP, breaking the replicated-state invariant. Elsewhere it keeps results and the
# costs quoted in kernel comments reproducible.

# Kernels and libr4d's declaration (row table, layouts) share one directory. There is no launch
# adapter: every entry point takes `(const RadArgs*, RadStream)` and sits beside its kernel. The
# plugin links rad_abi only, like one written out of tree.

set(R4D_UNITS
  "r4d_attn_paged_h256_gqa6|"
  "r4d_attn_paged_h128_gqa4|"
  # Gemma 4 dense sliding window: head_dim 256, 2 queries per KV head, windowed decode and
  # prefill.
  "r4d_attn_paged_h256_gqa2|"
  # Gemma 3 sliding window: head_dim 128, 2 queries per KV head, windowed decode and prefill.
  "r4d_attn_paged_h128_gqa2w|"
  # One query head per KV head at 128 wide: plain MHA, the Unlimited-OCR decoder (10 over 10).
  # Both legs, prefill and decode; see the file's header for why it is not a drafter-shaped unit.
  "r4d_attn_paged_h128_gqa1|"
  # Eight queries per KV head, one body instantiated at head_dim 256 (the routed 35B-A3B) and 128
  # (MiniCPM5-2B).
  "r4d_attn_paged_gqa8|"
  # The gqa12, gqa4 and gqa6_bs4 units set the ratio and include the parameterised body above.
  # gqa12, head_dim 256: Qwen3.8-Flash-Next's QSA layers (24 query heads over 2).
  "r4d_attn_paged_h256_gqa12|"
  # gqa4, head_dim 256: Qwen3.5-0.8B (8 over 2).
  "r4d_attn_paged_h256_gqa4|"
  # gqa6, head_dim 256, 4-token page: the same QSA layers at four ranks, each KV head replicated
  # across a rank pair.
  "r4d_attn_paged_h256_gqa6_bs4|"
  # head_dim 128/256 and 512 at 1, 2, 4 or 8 queries per KV head with any window: Gemma 4's sliding
  # (h256/gqa2, window 1024) and full (h512, gqa 8 at TP1/TP2) layers and Unlimited-OCR's h128 MHA
  # decoder. One kernel per geometry covers every q_len and reads q at any row pitch
  # (attn_paged_qrow).
  "r4d_attn_paged_window|"
  # Two queries per KV head at head_dim 128 only (Qwen3-1.7B's 16 over 8, Qwen3-ASR's decoder):
  # the gqa8 body with the ratio set.
  "r4d_attn_paged_h128_gqa2|"
  # Sixteen queries per KV head at head_dim 128 (Nemotron-H's 32 over 2 at one or two ranks).
  "r4d_attn_paged_h128_gqa16|"
  # Seven queries per KV head at head_dim 128: Nanbeige4.3-33B (56 over 8). Seven divides neither
  # 384 nor 64, so the unit runs 21 warps and caps a decode workgroup at 63 rows; see its header.
  "r4d_attn_paged_h128_gqa7|"
  "r4d_attn_vit_h72_bf16|"
  # An encoder's dense attention (bidirectional, GQA in the workgroup, a two-sided window) and
  # the small ops around it: act_mul, seg_mean, qk_norm_rope with a v norm, the f32 widening.
  "r4d_attn_dense_bf16|"
  # Unlimited-OCR's SAM tower: windowed attention with decomposed relative position.
  "r4d_sam_attn_bf16|"
  "r4d_encoder_bf16|"
  "r4d_nemotron_bf16|"
  # The image generator: attention over a paged prefix plus the pass's own rows, the causal
  # head_dim-128 dense form and the 1152-wide decoder head; plus the sampler, adaptive norm,
  # implicit-GEMM convolution and pixel ops.
  "r4d_attn_gen|"
  "r4d_gen|"
  # The audio generator: a vocoder's 1-D convolutions and snake activation, the guided top-k
  # sampler for a frame's codes, the frame and window ops, and a depth decoder's short attention.
  "r4d_audio|"
  # The video generator: a diffusion transformer's modulated RMS norm and sampler step, the video
  # decoder's patch unfold, tiles, stitch and rotary positions, and the soundtrack decoder's
  # alias-free SnakeBeta.
  "r4d_video|"
  "r4d_ltx|"
  # A video latent upscaler: the 3x3x3 convolution as an implicit GEMM over voxel rows, the group
  # norm, the temporal depthwise convolution, the per-frame bilinear resize, the patch fold and the
  # re-noise.
  "r4d_upscaler|"
  # Video sparse attention: the tile selection a distilled video transformer was trained with and
  # the block-sparse attention over it.
  "r4d_vsa|"
  # Vision tower ops: layer norm, the learned-grid position embedding, and the biased GEMM with
  # activation and residual fused into its store.
  "r4d_vit_bf16|"
  # Muse Glimmer's vision-tower coordinate ops: grid_embed_pad, window_order.
  "r4d_vit_muse|"
  # The same biased GEMM over f32 operands (a diffusion transformer's patch projections and heads,
  # a video decoder's 1x1 projections), on the f32 FMA tile r4d_audio's f32 convolutions share.
  "r4d_gemm_f32_nt_bias|"
  "r4d_gdn_chunk_scan_k128_v128_c64_bf16|-mcumode"
  "r4d_gdn_conv_w4_h128_bf16|"
  "r4d_gdn_kkt_solve_k128_c64_bf16|"
  "r4d_gdn_recurrent_update_k128_v128_bf16_fp32state|"
  "r4d_gdn_gated_rmsnorm_h128_bf16|"
  # The hyper-connection gated residual (in place of pre-norm plus residual add). hc_read is two
  # launches behind one entry point: the mix depends on every column of the low-rank waist, so it
  # needs a grid-wide barrier.
  "r4d_hc_bf16|"
  "r4d_mhc|"
  "r4d_glm|"
  "r4d_dsa|"
  "r4d_mla|"
  "r4d_kda|"
  # Mamba-2's selective state update (Nemotron-H), the KDA kernel's register layout.
  "r4d_ssd|"
  "r4d_ngram_ids|"
  "r4d_ple_gate_bf16|"
  "r4d_ple_conv_bf16|"
  # The all-reduce kernels share one instance (peer scratch, IPC-equivalent pointers, per-block
  # flags, device sequence counters) set up by one rendezvous across ranks, so all entry points
  # live in r4d_ar_entry. Spreading them across the kernel units would give each its own copy.
  "r4d_ar_entry|"
  "r4d_ar_gather_2rank|"
  "r4d_ar_gather_Nrank|"
  "r4d_ar_oneshot_2rank_exact|"
  "r4d_ar_oneshot_2rank_wht6|"
  "r4d_ar_oneshot_Nrank_exact|"
  "r4d_ar_twoshot_Nrank_exact|"
  "r4d_ar_twoshot_Nrank_ti8|"
  "r4d_gemm_bf16_nt_m16|"
  "r4d_gemm_bf16_nt_m64|"
  "r4d_gemm_bf16_nt_tiled|"
  "r4d_gemm_w4a16_nt_m64|"
  "r4d_gemm_w5a16_nt_m64|"
  "r4d_gemm_wpa8i8_tiled|"
  "r4d_gemm_w8a16_nt_m64|"
  "r4d_gemm_w2a8_nt|"
  "r4d_gemm_w4a8_nt_m64|"
  "r4d_gemm_w8a8_nt_m64|"
  "r4d_gemm_w8a8_tiled|"
  "r4d_gemm_w4a8_tiled|"
  "r4d_gemm_w4a8_prefill|"
  "r4d_gemm_mxfp4a8_nt_m64|"
  "r4d_gemm_mxfp4a8_tiled|"
  "r4d_gemm_mxfp4a8_decode|"
  "r4d_gemm_mxfp4a8_gemv|"
  # Block-scaled FP8 (the format Qwen3.8-27B-FP8 ships in). The 128x128 block is fixed by the
  # checkpoint, and the entry points reject a mismatch.
  "r4d_gemm_fp8a16_nt_m1|"
  "r4d_gemm_fp8a8|"
  "r4d_gemm_iqr|"
  "r4d_gemm_iqr_p1|"
  "r4d_gemm_iqr_p2|"
  "r4d_gemm_iqr_p3|"
  "r4d_moe_iqr|"
  "r4d_iqr_embed|"
  "r4d_logits_vocab|"
  "r4d_quant_act_fp8|"
  "r4d_fused_quant_fp8|"
  "r4d_gram_accum_fp8|"
  "r4d_qsa_block_key_bf16|"
  "r4d_qsa_score_bf16|"
  "r4d_qsa_select_bf16|"
  "r4d_kv_resolve|"
  "r4d_qsa_tail_store_bf16|"
  "r4d_qsa_work_bf16|"
  "r4d_sample_chain_f32|"
  "r4d_model_bf16|"
  "r4d_sample_stages_f32|"
  # The windowed no-repeat-ngram ban: one block a row over the row's history.
  "r4d_sample_ngram_f32|"
  "r4d_quant_act_i8|"
  "r4d_rmsnorm_quant_i8|"
  "r4d_qkv_norm_rope|"
  # A batch's context lengths less a runtime trim (the 31B's per-slice prefill attention).
  "r4d_kv_len_trim|"
  # A weight read into the on-die cache beside a collective (the prefetch op).
  "r4d_prefetch|"
  "r4d_gelu_mul_quant_i8|"
  "r4d_had_quant_act_i8|"
  "r4d_rmsnorm_had_quant_i8|"
  "r4d_ar_ln_had_quant_i8|"
  "r4d_gated_had_quant_i8|"
  "r4d_qk_norm_rope_gate|"
  "r4d_gdn_gated_norm_had_quant_i8|"
  "r4d_dequant_w4_bf16|"
  "r4d_rowtopk_bf16|"
  # The vocab-sharded draft head's cross-rank top-1, one id a row.
  "r4d_pair_argmax_f32|"
  # LiLiCorr's correlator: candidates, lattice attention, factor input, walk.
  "r4d_lili|"
  # Gemma 4's sandwich-norm joins (norm_add_norm, norm_add_norm_route, join_add_scale) and
  # expert_scale, each one kernel for a chain of small ones.
  "r4d_norm_join_bf16|"
  "r4d_expert_scale_bf16|"
  # Runtime LoRA's two halves (lora_shrink, lora_expand), per-row adapter slot: scalar and tiled rows.
  "r4d_lora_bf16|"
  "r4d_dflash_conv_t2_g16_bf16|"
  "r4d_dflash_select_bf16|"
  "r4d_dflash_markov_bf16|"
  "r4d_dflash_tree_bf16|"
  # Routed MoE: router, scatter, the grouped block-scaled fp8 GEMM and gather, in one unit because
  # the other three only feed the GEMM.
  "r4d_moe|"
  # Blocks a row keeps for an output buffer, freed with the memory they serve (r4d_keyed.h).
  "r4d_keyed|"
)

set(R4D_SRC "")
foreach(u ${R4D_UNITS})
  string(REPLACE "|" ";" parts "${u}")
  list(GET parts 0 unit)
  list(LENGTH parts n)
  set(extra "")
  if(n GREATER 1)
    list(GET parts 1 extra)
  endif()
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${unit}.hip")
  if(NOT EXISTS "${src}")
    message(FATAL_ERROR "libr4d unit ${unit}.hip is missing from ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if(extra)
    set_source_files_properties("${src}" PROPERTIES COMPILE_OPTIONS "${extra}")
  endif()
  list(APPEND R4D_SRC "${src}")
endforeach()

# The host sources are declaration only: the row table, weight layouts and relayouts, operand
# descriptions and fusion claims. They are pure host C++ that includes neither r4d.h nor
# hip_runtime.h, so these tables compile and can be inspected without ROCm.
set(R4D_DECLARATION r4d_rows.cpp r4d_layout.cpp r4d_fp8_layout.cpp r4d_iqr_layout.cpp r4d_shapes.cpp
                    r4d_fuse.cpp r4d_hsimd.cpp)
# The relayouts' hot loops, once per CPU level (r4d_hsimd.h); r4d_hsimd.cpp picks one at runtime.
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/RadSimdLevels.cmake)
rad_simd_level_objects(r4d_hsimd ${CMAKE_CURRENT_SOURCE_DIR}/r4d_hsimd_kernels.cpp R4D_HSIMD_OBJS
                       DEFINE R4D_HSIMD_LEVEL INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND R4D_DECLARATION ${R4D_HSIMD_OBJS})

rad_add_plugin(libr4d kernels
  SOURCES     ${R4D_DECLARATION}
  HIP_SOURCES ${R4D_SRC}
  GPU_TARGETS ${R4D_GPU_TARGETS}
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR})

# Catalogue dump: prints every row and, on a card, calls each launch with a minimal valid input, to
# catch transcription errors in the constraint table. It dlopens the plugin so an undefined symbol
# in the .so is caught too.
add_executable(r4d_selftest ${CMAKE_CURRENT_SOURCE_DIR}/r4d_selftest.cpp)
# Links the HIP runtime to allocate the device buffers it feeds the kernels.
target_link_libraries(r4d_selftest PRIVATE radiance::abi ${CMAKE_DL_LIBS} ${RAD_HIP_RUNTIME})
target_include_directories(r4d_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(r4d_selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# Installed beside the tools, so the image can run the rows' own card checks without a build tree:
#   r4d_selftest /opt/radiance/share/radiance/kernels/libr4d.so [--case <row>]
# The collective cases need every card of the box in the container.
install(TARGETS r4d_selftest RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# C++17: at C++20 clang tightens the address-space check on
# __builtin_amdgcn_global_load_tr_b128_v8i16 in r4d_common.h. Kernel cost figures are for C++17.
set_target_properties(libr4d PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
