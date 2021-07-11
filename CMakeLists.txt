cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GPUTI LANGUAGES CXX CUDA)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(GPUTI_TOPLEVEL_PROJECT ON)
else()
  set(GPUTI_TOPLEVEL_PROJECT OFF)
endif()

### Configuration
set(GPUTI_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Warnings)
include(UseColors)
include(GPUTI_Utils)

include(${PROJECT_NAME}_Dependencies)

### the library
add_subdirectory(src)
prepend_current_path(GPUTI_SOURCES)
gupti_copy_headers(${GPUTI_SOURCES})
gputi_set_source_group(${GPUTI_SOURCES})
add_library(gputi STATIC ${GPUTI_SOURCES})
target_include_directories(gputi PUBLIC ${PROJECT_BINARY_DIR}/include)
# target_link_libraries(gputi PRIVATE warnings::all)
target_compile_definitions(gputi PUBLIC NOMINMAX)

# Request that gputi be built with -std=c++11
# As this is a public compile feature anything that links to gputi
# will also build with -std=c++11
target_compile_features(gputi PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the gputi
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( gputi
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

# executable
add_executable(gputi_bin app/main.cu)
set_target_properties(gputi_bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gputi_bin PRIVATE gputi)
                  
                  
if(APPLE)
# We need to add the default path to the driver (libcuda.dylib) as an rpath,
# so that the static cuda runtime can find it at runtime.
set_property(TARGET gputi_bin PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()



###############################################################################