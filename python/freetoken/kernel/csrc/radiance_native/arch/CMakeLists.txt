# Architecture plugins: one directory and one .so each, linking only rad_abi, never the core.
#
# qwen35 covers the Qwen3.5, Qwen3.6 and Qwen3.8 dense models, which all report that id; qwen35moe
# is the same graph with a routed FFN (docs/ARCHITECTURES.md). There is one plugin per
# (architecture, quantisation) pair, which is the selection key (spec §2.4): qwen35_bf16 and
# qwen35_fp8 both claim "qwen35" and are told apart by the container's quantisation descriptor
# (rad_arch_quant in abi/rad_builder.h).
#
# Plugins are discovered, not listed, so the build is exactly what is in the tree. The glob is
# sorted because file(GLOB) returns directory order, and the link order sets the plugin load order,
# which decides which architecture wins a tie.
file(GLOB RAD_ARCH_DIRS CONFIGURE_DEPENDS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
list(SORT RAD_ARCH_DIRS)

# In-tree plugins use the installed include spelling. arch/common/ installs as
# <include>/radiance/arch/, so plugins write `#include <arch/rad_arch.h>`; in this tree that
# resolves through a build-tree directory whose arch/ links to common/.
set(RAD_ARCH_INCLUDE_ROOT ${CMAKE_BINARY_DIR}/rad_include)
file(MAKE_DIRECTORY ${RAD_ARCH_INCLUDE_ROOT})
if(NOT EXISTS ${RAD_ARCH_INCLUDE_ROOT}/arch)
  file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/common ${RAD_ARCH_INCLUDE_ROOT}/arch SYMBOLIC)
endif()
target_include_directories(rad_abi INTERFACE $<BUILD_INTERFACE:${RAD_ARCH_INCLUDE_ROOT}>)

foreach(d ${RAD_ARCH_DIRS})
  if(NOT IS_DIRECTORY ${d})
    continue()
  endif()
  get_filename_component(a ${d} NAME)
  # common/ is the shared header library, not a plugin.
  if(a STREQUAL "common")
    continue()
  endif()
  if(EXISTS ${d}/CMakeLists.txt)
    # Each plugin's own build rules, the same ones it uses out of tree.
    add_subdirectory(${a})
  else()
    # Reported at configure time rather than skipped silently: either a plugin is missing its build
    # file or the directory does not belong here.
    message(STATUS
      "arch: '${a}' has no CMakeLists.txt -- not an architecture plugin, skipped")
  endif()
endforeach()

# arch/common/ ships: abi/rad_builder.h directs C++ plugin authors to the grp_*() helpers in
# arch/common/rad_arch.h instead of the rad_group_*() compound-literal macros (a GNU extension in
# C++).
#
# Installed flat into arch/. These headers include each other by bare basename, which resolves
# relative to the including file in both trees, and rad_arch.h's "rad_builder.h" resolves against
# the include root that radiance::abi provides.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/common/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/radiance/arch
        FILES_MATCHING PATTERN "*.h")
