# libref -- every op in the conventional vocabulary, naively and generically (spec §17).
#
# Links radiance::abi and nothing else: abi/ is the whole plugin contract, so the plugin stays
# substitutable. Shared helpers (dtype conversions, parameter readers) come inline from the ABI
# headers.
# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S libref -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`); see cmake/RadAddPlugin.cmake.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libref LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

file(GLOB REF_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

rad_add_plugin(libref kernels SOURCES ${REF_SRC})

# -ffp-contract=off is a correctness flag: rad-kbench compares kernels against this reference to a
# tolerance, so `a*b + c` must round twice and give the same result under any compiler
# (libref/ref_common.h).
target_compile_options(libref PRIVATE -ffp-contract=off)

# Optional OpenMP over the outer dimension only. One thread owns whole output elements and there is
# no reduction() clause, so results are bit-identical at any thread count.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(libref PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "libref: no OpenMP -- the reference will run single-threaded")
endif()
