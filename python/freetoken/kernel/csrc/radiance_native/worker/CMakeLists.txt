add_executable(freetoken-native-worker worker_main.cpp)
target_link_libraries(freetoken-native-worker PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rad_core>")
set_target_properties(freetoken-native-worker PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin ENABLE_EXPORTS ON)
install(TARGETS freetoken-native-worker RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(freetoken-native-reference reference_main.cpp)
target_link_libraries(freetoken-native-reference PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rad_core>")
set_target_properties(freetoken-native-reference PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin ENABLE_EXPORTS ON)

# Plugins are dlopen dependencies, so linking rad_core alone does not rebuild them.
foreach(plugin libr4d qwen4exp_fp8 libavx libquant spechub)
  if(TARGET ${plugin})
    add_dependencies(freetoken-native-worker ${plugin})
    add_dependencies(freetoken-native-reference ${plugin})
  endif()
endforeach()
