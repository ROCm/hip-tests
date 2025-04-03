# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
    hipModuleLoad.cc
    hipModuleLoadData.cc
    hipModuleLoadDataEx.cc
    hipModuleUnload.cc
    hipModuleGetFunction.cc
    hipModuleLaunchKernel.cc
    hipModuleGetGlobal.cc
    hipModuleGetTexRef.cc
    hipModuleLaunchCooperativeKernel.cc
    hipModuleLaunchCooperativeKernelMultiDevice.cc
    hipFuncGetAttribute.cc
    hipGetFuncBySymbol.cc
    hipDrvLaunchKernelEx.cc
)

function(add_custom_module target module_name source_file_name flags)
  set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${module_name})
  set(source_file ${CMAKE_CURRENT_SOURCE_DIR}/${source_file_name})

  # Note to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx900
  # --offload-arch=gfx906" having space at the start/end of OFFLOAD_ARCH_STR can
  # cause build failures
  add_custom_target(
    ${module_name}
    COMMAND
      ${CMAKE_CXX_COMPILER} ${flags} ${source_file} -o ${output_file} -I${HIP_PATH}/include/
      -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --hip-path=${HIP_PATH})

  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${output_file})
  add_dependencies(${target} ${module_name})
endfunction()

add_custom_module(build_tests get_function_module.code get_function_module.cc "--genco;${OFFLOAD_ARCH_STR}")
add_custom_module(build_tests launch_kernel_module.code launch_kernel_module.cc "--genco;${OFFLOAD_ARCH_STR}")
add_custom_module(build_tests get_global_test_module.code get_global_test_module.cc "--genco;${OFFLOAD_ARCH_STR}")
add_custom_module(build_tests get_tex_ref_module.code get_tex_ref_module.cc "--genco;${OFFLOAD_ARCH_STR}")
add_custom_module(build_tests empty_module.code empty_module.cc "--genco;${OFFLOAD_ARCH_STR}")
add_custom_module(build_tests coopKernel.code coopKernel.cpp "--genco";${OFFLOAD_ARCH_STR})

set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_SOURCE_DIR}/not_a_module.txt)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_SOURCE_DIR}/empty_file.txt)

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} hipExtModuleLaunchKernel.cc
               hipHccModuleLaunchKernel.cc hipGetProcAddressModuleApis.cc)

  set(OFFLOAD_ARCH_GENERIC_STR "--offload-arch=gfx9-generic --offload-arch=gfx9-4-generic:sramecc+:xnack- --offload-arch=gfx9-4-generic:sramecc-:xnack- --offload-arch=gfx9-4-generic:xnack+ --offload-arch=gfx10-1-generic --offload-arch=gfx10-3-generic --offload-arch=gfx11-generic --offload-arch=gfx12-generic")

  add_custom_module(build_tests addKernel.code addKernel.cc
                    "-mcode-object-version=5 --genco;${OFFLOAD_ARCH_STR}")
  add_custom_module(build_tests copyKernelCompressed.code copyKernel.cc
                    "-mcode-object-version=5 --genco;--offload-compress;${OFFLOAD_ARCH_STR}")
  add_custom_module(build_tests copyKernel.code copyKernel.cc
                    "-mcode-object-version=5 --genco;${OFFLOAD_ARCH_STR}")
  add_custom_module(build_tests copyKernel.s copyKernel.cc
                    "-mcode-object-version=5;-S;")
  add_custom_module(build_tests copyKernelGenericTarget.code copyKernel.cc
                    "--genco -mcode-object-version=6;" ${OFFLOAD_ARCH_GENERIC_STR})
  add_custom_module(build_tests copyKernelGenericTargetCompressed.code copyKernel.cc
                    "--genco -mcode-object-version=6;--offload-compress" ${OFFLOAD_ARCH_GENERIC_STR})

  if(UNIX)
    set(TEST_SRC ${TEST_SRC} hipKerArgOptimization.cc)

    add_custom_module(build_tests copiousArgKernel.code copiousArgKernel.cc
      "--genco;${OFFLOAD_ARCH_STR}")
    add_custom_module(build_tests copiousArgKernel0.code copiousArgKernel.cc
      "-mllvm -amdgpu-kernarg-preload-count=0 --genco;${OFFLOAD_ARCH_STR}"
    )
    add_custom_module(
      build_tests copiousArgKernel1.code copiousArgKernel.cc
      "-mllvm -amdgpu-kernarg-preload-count=1 --genco;${OFFLOAD_ARCH_STR}"
    )
    add_custom_module(build_tests copiousArgKernel2.code copiousArgKernel.cc
      "-mllvm -amdgpu-kernarg-preload-count=2 --genco;${OFFLOAD_ARCH_STR}"
    )
    add_custom_module(build_tests copiousArgKernel3.code copiousArgKernel.cc
      "-mllvm -amdgpu-kernarg-preload-count=3 --genco;${OFFLOAD_ARCH_STR}"
    )
    add_custom_module(build_tests copiousArgKernel16.code copiousArgKernel.cc
      "-mllvm -amdgpu-kernarg-preload-count=16 --genco;${OFFLOAD_ARCH_STR}"
    )
    add_custom_module(build_tests copiousArgKernel17.code copiousArgKernel.cc
      "-mllvm -amdgpu-kernarg-preload-count=17 --genco;${OFFLOAD_ARCH_STR}"
    )
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
  COMMON_SHARED_SRC ${COMMON_SHARED_SRC}
  COMPILE_OPTIONS -std=c++17)

if(HIP_PLATFORM MATCHES "amd")
if(UNIX)
add_dependencies(build_tests copiousArgKernel.code copiousArgKernel0.code copiousArgKernel1.code copiousArgKernel2.code
copiousArgKernel3.code copiousArgKernel16.code copiousArgKernel17.code)
endif()
endif()

add_executable(hipGetFuncBySymbol_exe EXCLUDE_FROM_ALL hipGetFuncBySymbol_exe.cc)
add_dependencies(build_tests hipGetFuncBySymbol_exe)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipGetFuncBySymbol_exe)

# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipFuncSetAttribute.cc
  hipFuncGetAttributes.cc
  hipFuncSetSharedMemConfig.cc
  hipManagedKeyword.cc
  hipModule.cc
  hipModuleLoadMultProcessOnMultGPU.cc
)
set(AMD_TEST_SRC
    hipExtLaunchKernelGGL.cc
    hipExtLaunchMultiKernelMultiDevice.cc
)

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
endif()

hip_add_exe_to_target(NAME module
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

hip_add_exe_to_target(NAME managedKernel
                      TEST_SRC ${LINUX_TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${CMAKE_DL_LIBS})

hip_add_exe_to_target(NAME VcpyKernel
                      TEST_SRC ${LINUX_TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${CMAKE_DL_LIBS})

hip_add_exe_to_target(NAME matmul
                      TEST_SRC ${LINUX_TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${CMAKE_DL_LIBS})

add_custom_module(module managed_kernel.code managed_kernel.cpp "--genco;${OFFLOAD_ARCH_STR}")
add_custom_module(module vcpy_kernel.code vcpy_kernel.cpp "--genco;${OFFLOAD_ARCH_STR}")
add_custom_module(module matmul.code matmul.cpp "--genco;${OFFLOAD_ARCH_STR}")
add_custom_module(module kernel_composite_test.code kernel_composite_test.cpp "--genco;${OFFLOAD_ARCH_STR}")

add_executable(testhipModuleLoadUnloadFunc_exe EXCLUDE_FROM_ALL testhipModuleLoadUnloadFunc_exe.cc)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS testhipModuleLoadUnloadFunc_exe)

hip_add_exe_to_target(NAME compositeKernel
                      TEST_SRC ${LINUX_TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${CMAKE_DL_LIBS})

add_dependencies(module testhipModuleLoadUnloadFunc_exe)
