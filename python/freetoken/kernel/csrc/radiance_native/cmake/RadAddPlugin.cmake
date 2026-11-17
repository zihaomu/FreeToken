# rad_add_plugin / rad_hipify -- the single definition of how sources become a loadable radiance
# plugin, shared by the in-tree and out-of-tree builds.
#
# The top-level CMakeLists.txt includes it while building the ABI target; the installed
# radiance-config.cmake includes it after importing that target from radianceTargets.cmake. Keep
# one copy: a drifted duplicate would emit plugins where the loader does not scan, with no build
# error.
#
# What an includer must set first:
#   radiance::abi      the ABI interface target -- an in-tree ALIAS, or the imported target
#   RAD_PLUGIN_OUT     the build-tree $RADIANCE_HOME; a plugin lands in ${RAD_PLUGIN_OUT}/<kind>
#   RAD_HIP_FOUND      whether device sources can be compiled at all
#   RAD_GPU_TARGETS    the offload architectures this build targets. Any gfx name; a library that
#                      serves only some of them narrows the list with rad_plugin_gpu_targets()
#   RAD_HIP_RUNTIME    the amdhip64 library a device plugin links
# The top-level file sets all five as it detects them. radiance-config.cmake sets them from what the
# install recorded, so an out-of-tree plugin uses the engine's runtime; its GPU targets default to
# the engine's and can be overridden with -DRAD_GPU_TARGETS.
#
# Optional:
#   RAD_PLUGIN_INSTALL_HOME  the $RADIANCE_HOME `cmake --install` puts a plugin under. Unset in
#                      the tree (plugins install beside the engine under the same prefix);
#                      radiance-config.cmake sets it to the found installation's home.
#   RAD_HIP_VOPD_REPEATABLE  ON when the HIP compiler's VOPD pairing is repeatable; unset is OFF,
#                      which turns the pairing off on LLVM 24 (rad_hip_repeatable, below).

# For CMAKE_INSTALL_DATADIR. Included here because a consumer may not have; without it plugins
# would install to <prefix>/radiance/<kind>, where the loader never looks.
include(GNUInstallDirs)

# The tree's defaults for a plugin built standalone: C++20 and an optimised build type when none was
# given. Every plugin directory, in tree or not, opens with:
#
#   if(NOT COMMAND rad_add_plugin)          # standalone, not inside radiance's tree
#     cmake_minimum_required(VERSION 3.21)
#     project(mylib LANGUAGES CXX)          # CXX HIP for device code
#     find_package(radiance REQUIRED)
#     rad_plugin_defaults()
#   endif()
macro(rad_plugin_defaults)
  if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
  endif()
endmacro()

# CMake leaves two HIP rule variables empty. Without CMAKE_DEPFILE_FLAGS_HIP, Ninja records no
# header as a dependency of a .hip object (`ninja -t deps` shows #deps 0), so editing a header
# relinks a plugin around stale objects, which can leave engine and plugin disagreeing on a
# signature. Without CMAKE_INCLUDE_SYSTEM_FLAG_HIP, SYSTEM include directories reach .hip sources as
# `-I` and their warnings become ours. A macro, so the variables land in the caller's scope; values
# already set are kept.
macro(rad_hip_rule_fixes)
  if(NOT CMAKE_DEPFILE_FLAGS_HIP)
    set(CMAKE_DEPFILE_FLAGS_HIP "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
  endif()
  if(NOT CMAKE_INCLUDE_SYSTEM_FLAG_HIP)
    set(CMAKE_INCLUDE_SYSTEM_FLAG_HIP "-isystem ")
  endif()
endmacro()

# Build a kernel, architecture or quantiser plugin. `kind` is the $RADIANCE_HOME directory the engine
# scans for it; an installed plugin must use one of the three, or it would never be loaded. Test
# plugins (never installed, see RAD_PLUGIN_INSTALL) use their own directories.
function(rad_add_plugin name kind)
  cmake_parse_arguments(P "" "" "SOURCES;HIP_SOURCES;LIBS;INCLUDES;GPU_TARGETS" ${ARGN})
  if((NOT DEFINED RAD_PLUGIN_INSTALL OR RAD_PLUGIN_INSTALL) AND
     NOT kind MATCHES "^(kernels|architectures|quantizers)$")
    message(FATAL_ERROR
      "rad_add_plugin(${name} ${kind}): the kind is the directory the engine loads the plugin "
      "from, and it is one of kernels, architectures or quantizers.")
  endif()
  set(srcs ${P_SOURCES})
  if(P_HIP_SOURCES AND NOT RAD_HIP_FOUND)
    # Warn: a kernel library without its device sources still installs, with rows that have no
    # kernels behind them.
    message(WARNING
      "rad_add_plugin(${name}) was given HIP_SOURCES but this radiance has no device build "
      "(RAD_HIP_FOUND is OFF), so they are left out. A radiance configured without ROCm, or on a "
      "machine with no card and no -DRAD_GPU_TARGETS, has none.")
  endif()
  if(RAD_HIP_FOUND AND P_HIP_SOURCES)
    # The owning project must enable HIP itself; enable_language() is project scope and changes the
    # consumer's compiler set. Without it CMake fails with errors that do not name the cause, so
    # check here.
    get_property(_rad_langs GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT "HIP" IN_LIST _rad_langs)
      message(FATAL_ERROR
        "rad_add_plugin(${name}) was given HIP_SOURCES but the HIP language is not enabled in this "
        "project. Add `enable_language(HIP)` after project(); the project must enable it itself.")
    endif()
    set_source_files_properties(${P_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
    # Each device source gets a compilation unit ID from the plugin name and its relative path.
    # clang's default hashes absolute paths and the command line, and the resulting
    # __hip_cuid_<hash> symbol has variable length, which shifts read-only data and every
    # PC-relative address into it. A location-independent ID keeps code objects identical wherever
    # the plugin is built, while still unique per source.
    foreach(src ${P_HIP_SOURCES})
      get_filename_component(_rad_abs "${src}" ABSOLUTE)
      file(RELATIVE_PATH _rad_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_rad_abs}")
      string(SHA256 _rad_cuid "${name}/${_rad_rel}")
      string(SUBSTRING "${_rad_cuid}" 0 16 _rad_cuid)
      set_property(SOURCE "${src}" APPEND PROPERTY COMPILE_OPTIONS "-cuid=${_rad_cuid}")
    endforeach()
    list(APPEND srcs ${P_HIP_SOURCES})
  endif()
  add_library(${name} MODULE ${srcs})
  target_link_libraries(${name} PRIVATE radiance::abi ${P_LIBS})
  target_include_directories(${name} PRIVATE ${P_INCLUDES})
  # The build tree is a working $RADIANCE_HOME. Create the kind directory at configure time: the
  # engine scans it at startup, and a missing one fails the scan rather than the lookup.
  file(MAKE_DIRECTORY ${RAD_PLUGIN_OUT}/${kind})
  set_target_properties(${name} PROPERTIES
    PREFIX "" OUTPUT_NAME "${name}"
    LIBRARY_OUTPUT_DIRECTORY ${RAD_PLUGIN_OUT}/${kind})
  if(RAD_HIP_FOUND AND P_HIP_SOURCES)
    rad_hipify(${name} ${P_GPU_TARGETS})
  endif()

  # The install tree has the same shape, under RAD_DEFAULT_HOME (<prefix>/share/radiance).
  # RAD_PLUGIN_INSTALL=OFF opts out; its only in-tree user is core/plugin/testplugin (set in
  # core/CMakeLists.txt), whose plugins must never reach a real installation.
  if(NOT DEFINED RAD_PLUGIN_INSTALL)
    set(RAD_PLUGIN_INSTALL ON)
  endif()
  if(RAD_PLUGIN_INSTALL)
    if(RAD_PLUGIN_INSTALL_HOME)
      set(_rad_dest "${RAD_PLUGIN_INSTALL_HOME}/${kind}")
    else()
      set(_rad_dest "${CMAKE_INSTALL_DATADIR}/radiance/${kind}")
    endif()
    install(TARGETS ${name} LIBRARY DESTINATION "${_rad_dest}")
  endif()
endfunction()

# Repeatable device code on LLVM 24. ROCm 10.1's clang (AMD clang 24) is not deterministic: the
# post-RA VOPD pairing pass (GCNVOPDUtils.cpp) caches each load's transitive load dependencies, a
# fusion adds edges behind that cache, and which entries are cached follows the iteration order of
# a pointer-keyed set. So two compiles can pair differently, giving different .text and speed.
# LLVM 22 (ROCm 7.2) has no such cache and is repeatable.
#
# The docker image patches libLLVM to drop the cache after every fusion
# (docker/llvm-vopd-pairing.patch) and sets RAD_HIP_VOPD_REPEATABLE=ON. Any other LLVM 24 gets VOPD
# disabled, which costs 1.4 to 3.5% on the fp8 prefill tiles.
function(rad_hip_repeatable tgt)
  if(CMAKE_HIP_COMPILER_ID STREQUAL "Clang" AND CMAKE_HIP_COMPILER_VERSION VERSION_GREATER_EQUAL 24
     AND NOT RAD_HIP_VOPD_REPEATABLE)
    target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-mllvm=-amdgpu-enable-vopd=false>)
  endif()
endfunction()

# Set a target's HIP architectures and device flags. -ffp-contract=off is mandatory for any
# quantised cross-rank op: two ranks sum the dequantised products in opposite order, and an FMA on
# either side makes them differ by ~1 ULP, breaking the replicated-state invariant.
function(rad_hipify tgt)
  # Extra arguments narrow the architectures for this target; default is RAD_GPU_TARGETS.
  set(arches "${ARGN}")
  if(NOT arches)
    set(arches "${RAD_GPU_TARGETS}")
  endif()
  set_target_properties(${tgt} PROPERTIES HIP_ARCHITECTURES "${arches}")
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
  # Device optimisation level, set on the target because CMake's CMAKE_HIP_FLAGS_<CONFIG> is,
  # depending on version, empty (-O0 with asserts) or host-style '-O2 -g'. These come after the
  # toolchain flags, so they win. Non-Debug levels end in -g0 because a later -O does not undo -g,
  # and device debug info slows compiles, inhibits scheduling and changes code objects. NDEBUG is
  # required: a device assert is a hostcall, and the engine's queues have no hostcall buffer
  # (docs/PLUGIN.md). RAD_HIP_FLAGS_<CONFIG> overrides, as in the tree.
  foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    if(RAD_HIP_FLAGS_${cfg})
      set(lvl "${RAD_HIP_FLAGS_${cfg}}")
    elseif(cfg STREQUAL "DEBUG")
      set(lvl "-O0 -g")
    elseif(cfg STREQUAL "MINSIZEREL")
      set(lvl "-Os -DNDEBUG -g0")
    else()
      set(lvl "-O3 -DNDEBUG -g0")
    endif()
    separate_arguments(lvl)
    target_compile_options(${tgt} PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:${cfg}>>:${lvl}>")
  endforeach()
  # Uncompressed fat binary: the engine extracts the code object itself (core/device/aql_code.cpp)
  # and rejects a compressed one at first launch. This overrides any toolchain setting.
  target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--no-offload-compress>)
  target_link_options(${tgt} PRIVATE $<$<LINK_LANGUAGE:HIP>:--no-offload-compress>)
  target_link_libraries(${tgt} PRIVATE ${RAD_HIP_RUNTIME})
  rad_hip_repeatable(${tgt})
endfunction()

# Narrow RAD_GPU_TARGETS to the architectures one kernel library serves, given as regular
# expressions ("^gfx12" for a generation, "^gfx1201$" for a chip). The result may be empty; the
# plugin should then skip itself with a message rather than fail the configure.
function(rad_plugin_gpu_targets out)
  set(keep "")
  foreach(t ${RAD_GPU_TARGETS})
    foreach(pat ${ARGN})
      if(t MATCHES "${pat}")
        list(APPEND keep ${t})
        break()
      endif()
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES keep)
  set(${out} "${keep}" PARENT_SCOPE)
endfunction()
