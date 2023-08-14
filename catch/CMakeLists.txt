cmake_minimum_required(VERSION 3.16.8)

# to skip the simple compiler test
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

project(hiptests)

# flag to generate standalone exe per src file.
message(STATUS "STANDALONE_TESTS : ${STANDALONE_TESTS}")

# Check if platform is set
if(NOT HIP_PLATFORM STREQUAL "amd" AND NOT HIP_PLATFORM STREQUAL "nvidia")
    message(FATAL_ERROR "Unexpected HIP_PLATFORM: " ${HIP_PLATFORM})
endif()

if(WIN32)
    set(EXT ".bat")
endif()

if(HIP_PLATFORM STREQUAL "amd")
   if(UNIX AND DEFINED ROCM_PATH)
      # Read -DROCM_PATH and set CXX_FLAGS for amd platform only
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --rocm-path=${ROCM_PATH}")
   endif()

   if(DEFINED HIP_PATH)
      # Read -DHIP_PATH and set CXX_FLAGS for amd platform only
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --hip-path=${HIP_PATH}")
   endif()
endif()

# Read -DHIP_PATH
# If not set read env{HIP_PATH} only on Windows
if(WIN32)
    if(NOT DEFINED HIP_PATH)
        if(DEFINED ENV{HIP_PATH})
            set(HIP_PATH $ENV{HIP_PATH} CACHE STRING "HIP Path")
        endif()
    endif()
endif()

if(NOT DEFINED HIP_PATH)
   if(DEFINED ROCM_PATH)
      set(HIP_PATH ${ROCM_PATH})
   else()
      set(HIP_PATH "/opt/rocm")
   endif()
endif()
message(STATUS "HIP_PATH: ${HIP_PATH}")

set(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/hipcc${EXT}")
set(CMAKE_C_COMPILER "${HIP_PATH}/bin/hipcc${EXT}")
set(HIPCONFIG_EXECUTABLE "${HIP_PATH}/bin/hipconfig${EXT}")
execute_process(COMMAND ${HIPCONFIG_EXECUTABLE} --version
                OUTPUT_VARIABLE HIP_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# enforce c++17
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17")

string(REPLACE "." ";" VERSION_LIST ${HIP_VERSION})
list(GET VERSION_LIST 0 HIP_VERSION_MAJOR)
list(GET VERSION_LIST 1 HIP_VERSION_MINOR)
list(GET VERSION_LIST 2 HIP_VERSION_PATCH_GITHASH)
string(REPLACE "-" ";" VERSION_LIST ${HIP_VERSION_PATCH_GITHASH})
list(GET VERSION_LIST 0 HIP_VERSION_PATCH)

if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
   set(HIP_PACKAGING_VERSION_PATCH ${HIP_VERSION_PATCH}.$ENV{ROCM_LIBPATCH_VERSION})
else()
   set(HIP_PACKAGING_VERSION_PATCH ${HIP_VERSION_PATCH}-${HIP_VERSION_GITHASH})
endif()

if(NOT DEFINED CATCH2_PATH)
    if(DEFINED ENV{CATCH2_PATH})
        set(CATCH2_PATH $ENV{CATCH2_PATH} CACHE STRING "Catch2 Path")
    else()
        set(CATCH2_PATH "${CMAKE_CURRENT_LIST_DIR}/external/Catch2")
    endif()
endif()
message(STATUS "Catch2 Path: ${CATCH2_PATH}")

# Set JSON Parser path
if(NOT DEFINED JSON_PARSER)
    if(DEFINED ENV{JSON_PARSER})
        set(JSON_PARSER $ENV{JSON_PARSER} CACHE STRING "JSON Parser Path")
    else()
        set(JSON_PARSER "${CMAKE_CURRENT_LIST_DIR}/external/picojson")
    endif()
endif()

message(STATUS "Searching Catch2 in: ${CMAKE_CURRENT_LIST_DIR}/external")
find_package(Catch2 REQUIRED
    PATHS
        ${CMAKE_CURRENT_LIST_DIR}/external
    PATH_SUFFIXES
    Catch2/cmake/Catch2
)
include(Catch)
include(CTest)

# path used for generating the *_include.cmake file
set(CATCH2_INCLUDE ${CATCH2_PATH}/cmake/Catch2/catch_include.cmake.in)

include_directories(
    ${CATCH2_PATH}
    "./include"
    "./kernels"
    ${HIP_PATH}/include
    ${JSON_PARSER}
)

option(RTC_TESTING "Run tests using HIP RTC to compile the kernels" OFF)
if (RTC_TESTING)
    add_definitions(-DRTC_TESTING=ON)
endif()
add_definitions(-DKERNELS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")

set(CATCH_BUILD_DIR catch_tests)
file(COPY ./hipTestMain/config DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/hipTestMain)
file(COPY ./external/Catch2/cmake/Catch2/CatchAddTests.cmake
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/script)
file(COPY ./external/Catch2/cmake/Catch2/catch_include.cmake
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/script)
set(ADD_SCRIPT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/script/CatchAddTests.cmake)
set(CATCH_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/script/catch_include.cmake)



if (WIN32)
  configure_file(catchProp_in_rc.in ${CMAKE_CURRENT_BINARY_DIR}/catchProp.rc @ONLY)
  cmake_path(SET LLVM_RC_PATH "${HIP_PATH}/../lc/bin/llvm-rc.exe")
  cmake_path(SET LLVM_RC_PATH NORMALIZE "${LLVM_RC_PATH}")

  # generates the .res files to be used by executables to populate the properties
  # expects LC folder with clang, llvm-rc to be present one level up of HIP
  execute_process(COMMAND ${LLVM_RC_PATH} ${CMAKE_CURRENT_BINARY_DIR}/catchProp.rc
                  OUTPUT_VARIABLE RC_OUTPUT)
  set(PROP_RC ${CMAKE_CURRENT_BINARY_DIR})
  # When args to linker exceeds max chars.
  # msbuild writes args to a rsp file.
  # This is used to reference the obj file correctly
  SET(CMAKE_C_RESPONSE_FILE_LINK_FLAG "")
  SET(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "")
endif()

if(HIP_PLATFORM STREQUAL "amd")
    add_compile_options(-Wall -Wextra -Werror -Wno-deprecated)
endif()

cmake_policy(PUSH)
if(POLICY CMP0037)
    cmake_policy(SET CMP0037 OLD)
endif()

# Turn off CMAKE_HIP_ARCHITECTURES Feature if cmake version is 3.21+
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0)
    set(CMAKE_HIP_ARCHITECTURES OFF)
endif()
message(STATUS "CMAKE HIP ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")

# Note to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx900  --offload-arch=gfx906"
# having space at the start/end of OFFLOAD_ARCH_STR can cause build failures
# Identify the GPU Targets.
# This is done due to limitation of rocm_agent_enumerator
# While building test parallelly, rocm_agent_enumerator can fail and give out an empty target
# That results in hipcc building the test for gfx803 (the default target)
# preference to pass arch -
# OFFLOAD_ARCH_STR
# rocm_agent_enumerator
if(NOT DEFINED OFFLOAD_ARCH_STR
   AND EXISTS "${ROCM_PATH}/bin/rocm_agent_enumerator"
   AND HIP_PLATFORM STREQUAL "amd" AND UNIX)
    execute_process(COMMAND ${ROCM_PATH}/bin/rocm_agent_enumerator
         OUTPUT_VARIABLE HIP_GPU_ARCH
         RESULT_VARIABLE ROCM_AGENT_ENUM_RESULT)
    # Trim out gfx000
    string(REPLACE "gfx000\n" "" HIP_GPU_ARCH ${HIP_GPU_ARCH})
    if (NOT HIP_GPU_ARCH STREQUAL "")
        string(LENGTH ${HIP_GPU_ARCH} HIP_GPU_ARCH_LEN)
        # If string has more gfx target except gfx000
        if(${HIP_GPU_ARCH_LEN} GREATER_EQUAL 1)
            string(REGEX REPLACE "\n" ";" HIP_GPU_ARCH_LIST "${HIP_GPU_ARCH}")
            set(OFFLOAD_ARCH_STR "")
            foreach(_hip_gpu_arch ${HIP_GPU_ARCH_LIST})
                set(OFFLOAD_ARCH_STR "--offload-arch=${_hip_gpu_arch} ${OFFLOAD_ARCH_STR}")
            endforeach()
        endif()
    else()
        message(STATUS "ROCm Agent Enumurator found no valid architectures")
    endif()
elseif(DEFINED OFFLOAD_ARCH_STR)
    string(REPLACE "--offload-arch=" "" HIP_GPU_ARCH_LIST ${OFFLOAD_ARCH_STR})
endif()

if(DEFINED OFFLOAD_ARCH_STR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OFFLOAD_ARCH_STR} ")
endif()
message(STATUS "Using offload arch string: ${OFFLOAD_ARCH_STR}")

find_package(Git)
# get hip-tests commit short hash
execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE git_result
  OUTPUT_VARIABLE git_output
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(git_result EQUAL 0)
  set(HIP_TESTS_GITHASH ${git_output})
endif()


# prints the catch info to a file
string(TIMESTAMP _timestamp UTC)
set(_autogen "# Auto-generated by cmake on ${_timestamp} UTC\n")
set(_catchInfo ${_autogen} "HIP_VERSION=${HIP_VERSION}\n")
set(_catchInfo ${_catchInfo} "HIP_PLATFORM=${HIP_PLATFORM}\n")
set(_catchInfo ${_catchInfo} "HIP_TESTS_GITHASH=${HIP_TESTS_GITHASH}\n")
set(_catchInfo ${_catchInfo} "ARCHS=${HIP_GPU_ARCH_LIST}\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/catchInfo.txt ${_catchInfo})
# allows user to run ctest from catch_tests level
set(_subdirs ${_autogen} "subdirs(..)\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${CATCH_BUILD_DIR}/CTestTestfile.cmake ${_subdirs})

# Enable device lambda on nvidia platforms
if(HIP_PLATFORM STREQUAL "nvidia")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --extended-lambda")
endif()

# Disable CXX extensions (gnu++11 etc)
set(CMAKE_CXX_EXTENSIONS OFF)

add_custom_target(build_tests)


# Tests folder
add_subdirectory(unit ${CATCH_BUILD_DIR}/unit)
add_subdirectory(ABM ${CATCH_BUILD_DIR}/ABM)
add_subdirectory(kernels ${CATCH_BUILD_DIR}/kernels)
add_subdirectory(hipTestMain ${CATCH_BUILD_DIR}/hipTestMain)
add_subdirectory(stress ${CATCH_BUILD_DIR}/stress)
add_subdirectory(TypeQualifiers ${CATCH_BUILD_DIR}/TypeQualifiers)
if(UNIX)
   add_subdirectory(multiproc ${CATCH_BUILD_DIR}/multiproc)
endif()
add_subdirectory(performance ${CATCH_BUILD_DIR}/performance)

cmake_policy(POP)

# packaging the tests
# make package_test to generate packages for test
set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/packages/)
add_subdirectory(packaging)
if(UNIX)
add_custom_target(package_test COMMAND ${CMAKE_COMMAND} .
    COMMAND rm -rf *.deb *.rpm *.tar.gz
    COMMAND make package
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
file(TO_NATIVE_PATH ${PROJECT_BINARY_DIR} CATCH_BINARY_DIR)
add_custom_target(package_test COMMAND ${CMAKE_COMMAND} .
    COMMAND cpack
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
