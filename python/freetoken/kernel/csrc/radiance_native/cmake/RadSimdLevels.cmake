# rad_simd_level_objects(<prefix> <source> <out-var> [DEFINE <macro>] [INCLUDES <dir>...])
#
# A host source compiled once per CPU level, as the core's util/simd_kernels.cpp is
# (core/CMakeLists.txt): object libraries <prefix>_lvl0 .. _lvl4, each with <macro>=<level> and the
# level's -m flags, position-independent so a plugin module can take them. <out-var> receives the
# objects for target_sources. The levels: 0 baseline, 1 SSE4.1/4.2, 2 AVX, 3 AVX2 + FMA + F16C,
# 4 AVX-512 F/BW/DQ/VL. Off x86-64 only level 0 is built; the source's dispatcher then has only the
# scalar table. No other object may carry these flags: the dispatcher picks a level at runtime.
function(rad_simd_level_objects prefix src out)
  cmake_parse_arguments(A "" "DEFINE" "INCLUDES" ${ARGN})
  if(NOT A_DEFINE)
    set(A_DEFINE RAD_SIMD_LEVEL)
  endif()
  set(flags_0 "")
  set(flags_1 -msse4.1 -msse4.2 -mpopcnt)
  set(flags_2 -mavx -mpopcnt)
  set(flags_3 -mavx2 -mfma -mf16c -mpopcnt)
  set(flags_4 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx2 -mfma -mf16c -mpopcnt)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(levels 0 1 2 3 4)
  else()
    set(levels 0)
  endif()
  set(objs "")
  foreach(lvl ${levels})
    add_library(${prefix}_lvl${lvl} OBJECT ${src})
    target_compile_definitions(${prefix}_lvl${lvl} PRIVATE ${A_DEFINE}=${lvl})
    if(A_INCLUDES)
      target_include_directories(${prefix}_lvl${lvl} PRIVATE ${A_INCLUDES})
    endif()
    target_compile_options(${prefix}_lvl${lvl} PRIVATE ${flags_${lvl}})
    set_property(TARGET ${prefix}_lvl${lvl} PROPERTY POSITION_INDEPENDENT_CODE ON)
    list(APPEND objs $<TARGET_OBJECTS:${prefix}_lvl${lvl}>)
  endforeach()
  set(${out} ${objs} PARENT_SCOPE)
endfunction()
