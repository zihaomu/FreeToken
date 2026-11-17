# spechub -- the linear-MTP coupling kernels.
#
# Two ops, `mtp_couple_propose` and `mtp_couple_verify`, each with a device row and a host reference
# row. The arithmetic is abi/rad_spechub.h's, shared with the engine. The device launches include
# libr4d/r4d_emit.h by its path in this tree, so the device form builds from this tree's sources
# (in the tree's build or standalone against an install); a host-only build needs abi/ alone.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(spechub LANGUAGES CXX)
  find_package(radiance REQUIRED)
  # The device rows are built where the installed radiance has a device build; HIP is enabled
  # after project(), from the package's record of that build.
  if(RAD_HIP_FOUND)
    enable_language(HIP)
  endif()
  rad_plugin_defaults()
endif()

set(SPECHUB_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/spechub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ref_spechub.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/exports.cpp)

if(NOT RAD_HIP_FOUND)
  message(STATUS "spechub: this radiance has no device build -- only the host reference is built")
  rad_add_plugin(spechub kernels SOURCES ${SPECHUB_COMMON})
else()
  # The owning project must enable HIP itself; enable_language() is project scope. The top-level tree
  # does it with -DRAD_WITH_HIP=ON, and a standalone build must do the same after project().
  get_property(spechub_langs GLOBAL PROPERTY ENABLED_LANGUAGES)
  if(NOT "HIP" IN_LIST spechub_langs)
    message(FATAL_ERROR "spechub: HIP_SOURCES need `enable_language(HIP)` in this project (set "
                        "-DRAD_WITH_HIP=ON, or enable it after project()).")
  endif()
  rad_add_plugin(spechub kernels SOURCES ${SPECHUB_COMMON}
                  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/spechub.hip)
  target_compile_definitions(spechub PRIVATE RAD_SPECHUB_WITH_DEVICE=1)
endif()

# -ffp-contract=off for correctness: an FMA fused into the ordered FP64 mass sums would move the
# token off the host reference.
target_compile_options(spechub PRIVATE -ffp-contract=off)
