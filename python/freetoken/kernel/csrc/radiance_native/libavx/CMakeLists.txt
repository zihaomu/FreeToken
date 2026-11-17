# libavx -- the conventional op vocabulary in hand-written x86 SIMD, dispatched by CPU feature.
#
# This build is what makes runtime dispatch safe:
#
#   The kernels are compiled four times, once per ISA level, into OBJECT libraries with different
#   -m flags and AVX_LEVEL=0..3. avx_vec.h wraps each level in its own inline namespace; without
#   that the linker would keep one definition (e.g. of avx::row_to_f32) for all four tables, and
#   the scalar path could execute AVX-512 and fault.
#
#   The dispatcher and registry are compiled at baseline with no -m flags, because they decide
#   which levels may run. This depends on the flags, not the source: with -mavx512f the compiler
#   may auto-vectorise code that has no intrinsics.
#
#   No -march=native: the .so is copied to other machines and dlopened by an engine that does not
#   know its target, so it would SIGILL on older CPUs before main.
#
# The root CMakeLists adds this directory on every x86-64 build; these are the production host
# kernels (e.g. the n-gram embedding gather when placed on the host). It also builds standalone
# against an installed radiance, like any third-party plugin:
# `cmake -S libavx -B build-avx -DCMAKE_PREFIX_PATH=<radiance prefix>`. That needs no ROCm, and
# the checker's oracle is then the installed reference library.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libavx LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

# The harnesses go beside the plugin in the build's $RADIANCE_HOME; the loader ignores non-.so files.
set(AVX_OUT_DIR ${RAD_PLUGIN_OUT}/kernels)

# ============================ the four ISA levels ============================
# Every kernel translation unit. The baseline sources belong to the plugin target below.
set(AVX_KERNEL_SRC
  avx_support.cpp
  avx_table.cpp
  avx_elementwise.cpp
  avx_norm.cpp
  avx_hc.cpp
  avx_glm.cpp
  avx_dsa.cpp
  avx_mla.cpp
  avx_quant.cpp
  avx_gemm.cpp
  avx_gemm_tile.cpp
  avx_planes.cpp
  avx_lora.cpp
  avx_attn.cpp
  avx_kvstream.cpp
  avx_pixtral.cpp
  avx_sam_attn.cpp
  avx_qsa.cpp
  avx_moe.cpp
  avx_gdn.cpp
  avx_ssd.cpp
  avx_sample.cpp
  avx_misc.cpp
  avx_gen.cpp
  avx_gen_attn.cpp
  avx_video.cpp
  avx_ltx.cpp
  avx_vsa.cpp
  avx_audio.cpp
  avx_lili.cpp
)

# The flag sets, written out because each states which instructions the object may contain.
#
#   0 scalar  x86-64 baseline (SSE2).
#   1 avx     -mavx only. F16C and FMA are separate CPUID bits (Ivy Bridge, Haswell); requiring them
#             would fault on the Sandy Bridge parts this level exists for.
#   2 avx2    -mavx2 -mfma -mf16c, which ship together.
#   3 avx512  F + BW + DQ + VL (the conjunction avx_dispatch.cpp checks) plus the AVX2 set. No
#             avx512bf16 / avx512vnni: many AVX-512 parts lack them (avx_vec.h converts bf16 with
#             integer ops instead of vcvtneps2bf16).
set(AVX_FLAGS_0 "")
set(AVX_FLAGS_1 -mavx)
set(AVX_FLAGS_2 -mavx2 -mfma -mf16c)
set(AVX_FLAGS_3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx2 -mfma -mf16c)

find_package(OpenMP QUIET)

foreach(lvl RANGE 0 3)
  add_library(avx_lvl${lvl} OBJECT ${AVX_KERNEL_SRC})
  target_compile_definitions(avx_lvl${lvl} PRIVATE AVX_LEVEL=${lvl})
  target_link_libraries(avx_lvl${lvl} PRIVATE radiance::abi)
  target_include_directories(avx_lvl${lvl} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(avx_lvl${lvl} PRIVATE ${AVX_FLAGS_${lvl}}
    # Contraction is allowed, unlike libref (the oracle, which must not contract). libavx is checked
    # against it to a tolerance, and vectorised reductions are not bit-exact anyway, so FMA is kept.
    -ffp-contract=fast
    # No -ffast-math: it assumes no NaN/Inf and no signed zero, and these ops rely on them (a masked
    # logit is -inf; an attention row with no valid keys carries a NaN through the online softmax).
    # The checker compares the finiteness pattern before any value.
    -fno-math-errno)
  set_property(TARGET avx_lvl${lvl} PROPERTY POSITION_INDEPENDENT_CODE ON)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(avx_lvl${lvl} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()

# ============================ the plugin ============================
# Baseline sources. avx_ngram.cpp is also here because it holds the embedding gather's row cache
# and reader thread, which must exist once per process, not once per level.
set(AVX_BASELINE_SRC avx_dispatch.cpp avx_registry.cpp avx_ngram.cpp)

rad_add_plugin(libavx kernels SOURCES ${AVX_BASELINE_SRC} INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})

foreach(lvl RANGE 0 3)
  target_link_libraries(libavx PRIVATE avx_lvl${lvl})
endforeach()
find_package(Threads REQUIRED)
target_link_libraries(libavx PRIVATE Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(libavx PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "libavx: no OpenMP -- the plugin will run single-threaded")
endif()

# ============================ the harnesses ============================
# The harnesses dlopen both plugins by path and link neither, so they exercise the exact .so the
# engine would load, in either build or against a plugin built elsewhere.
add_executable(rad-avx-check avx_check.cpp avx_check_hand.cpp avx_check_rows.cpp avx_cases.cpp)
target_include_directories(rad-avx-check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rad-avx-check PRIVATE radiance::abi ${CMAKE_DL_LIBS} Threads::Threads)
# In the source tree the checker also decodes weights with the core's own decoder, an oracle
# independent of both plugins (avx_check_hand.cpp). One source file, not the core library.
if(TARGET rad_core)
  target_sources(rad-avx-check PRIVATE ${CMAKE_SOURCE_DIR}/core/format/encoding.cpp)
  target_include_directories(rad-avx-check PRIVATE ${CMAKE_SOURCE_DIR}/core)
  target_compile_definitions(rad-avx-check PRIVATE AVX_CHECK_CORE_DECODER=1)
endif()
set_target_properties(rad-avx-check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${AVX_OUT_DIR})

add_executable(rad-avx-bench avx_bench.cpp avx_cases.cpp)
target_include_directories(rad-avx-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rad-avx-bench PRIVATE radiance::abi ${CMAKE_DL_LIBS})
set_target_properties(rad-avx-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${AVX_OUT_DIR})
if(OpenMP_CXX_FOUND)
  target_link_libraries(rad-avx-bench PRIVATE OpenMP::OpenMP_CXX)
endif()
