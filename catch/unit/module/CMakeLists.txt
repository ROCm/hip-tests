# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Common Tests - Test independent of all platforms
set(TEST_SRC
    hip_module_common.cc
    hipModuleGetGlobal.cc
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code
                   COMMAND ${CMAKE_CXX_COMPILER} --genco --std=c++17 ${CMAKE_CURRENT_SOURCE_DIR}/get_global_test_module.cc -o get_global_test_module.code 
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/get_global_test_module.cc)   
add_custom_target(get_global_test_module ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/get_global_test_module.code)

if(HIP_PLATFORM MATCHES "amd")
set(TEST_SRC
    ${TEST_SRC}
    hipExtModuleLaunchKernel.cc
)

# Note to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx900  --offload-arch=gfx906"
# having space at the start/end of OFFLOAD_ARCH_STR can cause build failures

add_custom_target(copyKernel.code
                  COMMAND ${CMAKE_CXX_COMPILER} -mcode-object-version=5 --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copyKernel.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})

add_custom_target(copyKernel.s
                  COMMAND ${CMAKE_CXX_COMPILER} -mcode-object-version=5 -S ${CMAKE_CURRENT_SOURCE_DIR}/copyKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copyKernel.s
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})

if(UNIX)
set(TEST_SRC
    ${TEST_SRC}
    hipKerArgOptimization.cc)

add_custom_target(copiousArgKernel.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})

add_custom_target(copiousArgKernel0.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  -mllvm -amdgpu-kernarg-preload-count=0
                  ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel0.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})

add_custom_target(copiousArgKernel1.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  -mllvm -amdgpu-kernarg-preload-count=1
                  ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel1.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})

add_custom_target(copiousArgKernel2.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  -mllvm -amdgpu-kernarg-preload-count=2
                  ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel2.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})

add_custom_target(copiousArgKernel3.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  -mllvm -amdgpu-kernarg-preload-count=3
                  ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel3.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})

add_custom_target(copiousArgKernel16.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  -mllvm -amdgpu-kernarg-preload-count=16
                  ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel16.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})

add_custom_target(copiousArgKernel17.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  -mllvm -amdgpu-kernarg-preload-count=17
                  ${CMAKE_CURRENT_SOURCE_DIR}/copiousArgKernel.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/module/copiousArgKernel17.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})
endif()
endif()

if(HIP_PLATFORM MATCHES "amd")
    set(RTCLIB "hiprtc")
else()
    set(RTCLIB "nvrtc")
endif()
hip_add_exe_to_target(NAME ModuleTest
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests
  LINKER_LIBS ${RTCLIB}
  COMMON_SHARED_SRC ${COMMON_SHARED_SRC})

add_dependencies(ModuleTest get_global_test_module)

if(HIP_PLATFORM MATCHES "amd")
add_dependencies(build_tests copyKernel.code copyKernel.s)
if(UNIX)
add_dependencies(build_tests copiousArgKernel.code copiousArgKernel0.code copiousArgKernel1.code copiousArgKernel2.code
copiousArgKernel3.code copiousArgKernel16.code copiousArgKernel17.code)
endif()
endif()
