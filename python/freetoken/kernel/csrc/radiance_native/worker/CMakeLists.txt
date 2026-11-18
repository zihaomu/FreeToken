add_executable(freetoken-native-worker worker_main.cpp)
target_link_libraries(freetoken-native-worker PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rad_core>")
set_target_properties(freetoken-native-worker PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin ENABLE_EXPORTS ON)
install(TARGETS freetoken-native-worker RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(freetoken-native-reference reference_main.cpp)
target_link_libraries(freetoken-native-reference PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rad_core>")
set_target_properties(freetoken-native-reference PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin ENABLE_EXPORTS ON)

set(FT_EXECUTOR_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../native_executor)
add_executable(freetoken-executor-bridge ${FT_EXECUTOR_SOURCE}/worker_main.cpp ${FT_EXECUTOR_SOURCE}/qwen_plan.cpp)
target_include_directories(freetoken-executor-bridge PRIVATE ${FT_EXECUTOR_SOURCE})
target_link_libraries(freetoken-executor-bridge PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rad_core>")
set_target_properties(freetoken-executor-bridge PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin ENABLE_EXPORTS ON)

add_executable(freetoken-batch-worker ${FT_EXECUTOR_SOURCE}/worker_main.cpp ${FT_EXECUTOR_SOURCE}/qwen_plan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../native_executor/batch_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../native_executor/state_storage.cpp)
target_sources(freetoken-batch-worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../native_executor/residency.cpp)
target_include_directories(freetoken-batch-worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../native_executor)
target_compile_definitions(freetoken-batch-worker PRIVATE FREETOKEN_EXPLICIT_BATCHES=1)
target_link_libraries(freetoken-batch-worker PRIVATE rad_core)
target_compile_options(freetoken-batch-worker PRIVATE -ffunction-sections -fdata-sections)
target_link_options(freetoken-batch-worker PRIVATE
  "LINKER:--gc-sections" "LINKER:--export-dynamic-symbol=rad_*"
  "LINKER:-Map,${CMAKE_BINARY_DIR}/freetoken-batch-worker.map")
set_target_properties(freetoken-batch-worker PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin ENABLE_EXPORTS OFF)

# Plugins are dlopen dependencies, so linking rad_core alone does not rebuild them.
foreach(plugin libr4d qwen4exp_fp8 libavx libquant spechub)
  if(TARGET ${plugin})
    add_dependencies(freetoken-native-worker ${plugin})
    add_dependencies(freetoken-native-reference ${plugin})
    add_dependencies(freetoken-executor-bridge ${plugin})
    if(NOT plugin STREQUAL "qwen4exp_fp8")
      add_dependencies(freetoken-batch-worker ${plugin})
    endif()
  endif()
endforeach()
