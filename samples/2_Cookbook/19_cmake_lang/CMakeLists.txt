# The test must be with CMake 3.18 or above which supports LINK_LANGUAGE that is necessary for
# Fortran build.
cmake_minimum_required(VERSION 3.18.0)

project(cmake_lang_test CXX Fortran)

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
  endif()
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

find_package(hip REQUIRED)

add_executable(test_fortran TestFortran.F90)
add_executable(test_cpp MatrixTranspose.cpp)

target_link_libraries(test_cpp PUBLIC hip::device)

target_include_directories(test_cpp PRIVATE ../../common)

# Assuming to build a C/C++-to-Fortran library binding.
target_link_libraries(test_fortran PUBLIC hip::device)
