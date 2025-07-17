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

function(add_custom_module target module_name source_file_name flags type)
  set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${module_name}.${type})
  set(source_file ${CMAKE_CURRENT_SOURCE_DIR}/${source_file_name})

  add_custom_command(
    OUTPUT ${output_file}
    COMMAND ${CMAKE_CXX_COMPILER} ${flags} ${source_file} -o ${output_file}
            -I${HIP_PATH}/include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --hip-path=${HIP_PATH}
    DEPENDS ${source_file}
  )

  set(custom_target_name ${module_name}_${type})
  add_custom_target(${custom_target_name} ALL DEPENDS ${output_file})
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${output_file})
  add_dependencies(${target} ${custom_target_name})
endfunction()

function(add_custom_file target module_name source_file_name flags)
  set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${module_name}.txt)
  set(source_file ${CMAKE_CURRENT_SOURCE_DIR}/${source_file_name})

  add_custom_command(
    OUTPUT ${output_file}
    COMMAND ${CMAKE_COMMAND} -E copy ${source_file} ${output_file}
    DEPENDS ${source_file}
  )

  add_custom_target(${module_name} ALL DEPENDS ${output_file})
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${output_file})
  add_dependencies(${target} ${module_name})
endfunction()

add_custom_module(build_tests get_function_module get_function_module.cc "--genco;${OFFLOAD_ARCH_STR}" "code")
add_custom_module(build_tests launch_kernel_module launch_kernel_module.cc "--genco;${OFFLOAD_ARCH_STR}" "code")
add_custom_module(build_tests get_global_test_module get_global_test_module.cc "--genco;${OFFLOAD_ARCH_STR}" "code")
add_custom_module(build_tests get_tex_ref_module get_tex_ref_module.cc "--genco;${OFFLOAD_ARCH_STR}" "code")
add_custom_module(build_tests empty_module empty_module.cc "--genco;${OFFLOAD_ARCH_STR}" "code")
add_custom_module(build_tests coopKernel coopKernel.cpp "--genco;${OFFLOAD_ARCH_STR}" "code")

add_custom_file(build_tests empty_file empty_file.txt "")
add_custom_file(build_tests not_a_module not_a_module.txt "")

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} hipExtModuleLaunchKernel.cc
               hipHccModuleLaunchKernel.cc hipGetProcAddressModuleApis.cc)

  set(OFFLOAD_ARCH_GENERIC_STR
    "--offload-arch=gfx9-generic --offload-arch=gfx9-4-generic:sramecc+:xnack- \
     --offload-arch=gfx9-4-generic:sramecc-:xnack- --offload-arch=gfx9-4-generic:xnack+ \
     --offload-arch=gfx10-1-generic --offload-arch=gfx10-3-generic --offload-arch=gfx11-generic \
     --offload-arch=gfx12-generic")

  add_custom_module(build_tests addKernel addKernel.cc
                    "-mcode-object-version=5;--genco;${OFFLOAD_ARCH_STR};" "code")
  add_custom_module(build_tests copyKernelCompressed copyKernel.cc
                    "-mcode-object-version=5;--genco;--offload-compress;${OFFLOAD_ARCH_STR};" "code")
  add_custom_module(build_tests copyKernel copyKernel.cc
                    "-mcode-object-version=5;--genco;${OFFLOAD_ARCH_STR};" "code")
  add_custom_module(build_tests copyKernel copyKernel.cc
                    "-mcode-object-version=5;-S;" "s")
  add_custom_module(build_tests copyKernelGenericTarget copyKernel.cc
                    "-mcode-object-version=6;--genco;${OFFLOAD_ARCH_GENERIC_STR};" "code")
  add_custom_module(build_tests copyKernelGenericTargetCompressed copyKernel.cc
                    "-mcode-object-version=6;--offload-compress;--genco;${OFFLOAD_ARCH_GENERIC_STR}" "code")

  if(UNIX)
    set(TEST_SRC ${TEST_SRC} hipKerArgOptimization.cc)

    add_custom_module(build_tests copiousArgKernel copiousArgKernel.cc
      "--genco;${OFFLOAD_ARCH_STR}" "code")
    add_custom_module(build_tests copiousArgKernel0 copiousArgKernel.cc
      "-mllvm;-amdgpu-kernarg-preload-count=0;--genco;${OFFLOAD_ARCH_STR}" "code"
    )
    add_custom_module(
      build_tests copiousArgKernel1 copiousArgKernel.cc
      "-mllvm;-amdgpu-kernarg-preload-count=1;--genco;${OFFLOAD_ARCH_STR}" "code"
    )
    add_custom_module(build_tests copiousArgKernel2 copiousArgKernel.cc
      "-mllvm;-amdgpu-kernarg-preload-count=2;--genco;${OFFLOAD_ARCH_STR}" "code"
    )
    add_custom_module(build_tests copiousArgKernel3 copiousArgKernel.cc
      "-mllvm;-amdgpu-kernarg-preload-count=3;--genco;${OFFLOAD_ARCH_STR}" "code"
    )
    add_custom_module(build_tests copiousArgKernel16 copiousArgKernel.cc
      "-mllvm;-amdgpu-kernarg-preload-count=16;--genco;${OFFLOAD_ARCH_STR}" "code"
    )
    add_custom_module(build_tests copiousArgKernel17 copiousArgKernel.cc
      "-mllvm;-amdgpu-kernarg-preload-count=17;--genco;${OFFLOAD_ARCH_STR}" "code"
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

add_custom_module(module managed_kernel managed_kernel.cpp "--genco;${OFFLOAD_ARCH_STR}" "code")
add_custom_module(module vcpy_kernel vcpy_kernel.cpp "--genco;${OFFLOAD_ARCH_STR}" "code")
add_custom_module(module matmul matmul.cpp "--genco;${OFFLOAD_ARCH_STR}" "code")
add_custom_module(module kernel_composite_test kernel_composite_test.cpp "--genco;${OFFLOAD_ARCH_STR}" "code")

add_executable(testhipModuleLoadUnloadFunc_exe EXCLUDE_FROM_ALL testhipModuleLoadUnloadFunc_exe.cc)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS testhipModuleLoadUnloadFunc_exe)

hip_add_exe_to_target(NAME compositeKernel
                      TEST_SRC ${LINUX_TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS ${CMAKE_DL_LIBS})

add_dependencies(module testhipModuleLoadUnloadFunc_exe)
