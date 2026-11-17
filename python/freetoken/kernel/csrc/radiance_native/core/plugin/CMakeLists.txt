# The plugin layer's sources are compiled into rad_core by core/CMakeLists.txt's glob. This file
# only adds the test plugins, which must be separate .so files so the loader can dlopen them.
add_subdirectory(testplugin)
