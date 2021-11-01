set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(GPUTI LANGUAGES CXX CUDA)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(GPUTI_TOPLEVEL_PROJECT ON)
else()
  set(GPUTI_TOPLEVEL_PROJECT OFF)
endif()
OPTION(GPUTI_WITH_BENCHMARK          "run benchmark for testing"   ON)


### Configuration
set(GPUTI_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Warnings)
# include(UseColors)
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

 target_compile_options(gputi PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
 --generate-line-info
 --use_fast_math
 --relocatable-device-code=true
 # --ptxas-options=-v
 #--maxrregcount=40
 >)  
# executable 2
if (GPUTI_WITH_BENCHMARK)
# set(Boost_USE_MULTITHREADED TRUE)
# find_package(Boost COMPONENTS filesystem REQUIRED)


add_executable(test_bin app/test.cu)
set_target_properties(test_bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_bin PRIVATE gputi)
# target_link_libraries(test_bin PUBLIC Boost::filesystem)
# target_compile_features(test_bin PUBLIC cxx_std_14)
# gmp
find_package(GMP)
IF(NOT ${GMP_FOUND})
        MESSAGE(FATAL_ERROR "Cannot find GMP")
ENDIF()
target_include_directories(test_bin PUBLIC ${GMP_INCLUDE_DIRS})
target_link_libraries(test_bin PUBLIC ${GMP_LIBRARIES})
target_compile_options(test_bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       --ptxas-options=-v
                      #--maxrregcount=40
                       >)                  
if(APPLE)
# We need to add the default path to the driver (libcuda.dylib) as an rpath,
# so that the static cuda runtime can find it at runtime.
set_property( test_bin PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

endif()
###############################################################################