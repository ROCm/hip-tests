# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipModuleNegative.cc
    hipModuleGetGlobal.cc
    hipModuleOccupancyMaxPotentialActiveBlockSize.cc
    hipModuleLoadMultiThreaded.cc
    hipModuleLoadDataMultThreaded.cc
    hipModuleLoadDataMultThreadOnMultGPU.cc
    hipModuleLoadUnloadStress.cc
    hipModuleTexture2dDrv.cc
)

add_custom_target(vcpy_kernel.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR} ${CMAKE_CURRENT_SOURCE_DIR}/vcpy_kernel.cc -o ${CMAKE_CURRENT_BINARY_DIR}/../module/vcpy_kernel.code -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_custom_target(global_kernel.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR} ${CMAKE_CURRENT_SOURCE_DIR}/global_kernel.cc -o ${CMAKE_CURRENT_BINARY_DIR}/../module/global_kernel.code -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_custom_target(empty_kernel.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR} ${CMAKE_CURRENT_SOURCE_DIR}/empty_kernel.cc -o ${CMAKE_CURRENT_BINARY_DIR}/../module/empty_kernel.code -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_custom_target(kernel_composite_test.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR} ${CMAKE_CURRENT_SOURCE_DIR}/kernel_composite_test.cc -o ${CMAKE_CURRENT_BINARY_DIR}/../module/kernel_composite_test.code -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

add_custom_target(tex2d_kernel.code COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR} ${CMAKE_CURRENT_SOURCE_DIR}/tex2d_kernel.cc -o ${CMAKE_CURRENT_BINARY_DIR}/../module/tex2d_kernel.code -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

if (UNIX)
add_custom_target(opencl_add.cc COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/opencl_add.cc ${CMAKE_CURRENT_BINARY_DIR})
endif()

# only for AMD
if(HIP_PLATFORM MATCHES "amd")
  set(AMD_SRC
      hipOpenCLCOTest.cc
  )
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
endif()

hip_add_exe_to_target(NAME ModuleTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

if (UNIX)
add_dependencies(build_tests vcpy_kernel.code global_kernel.code empty_kernel.code kernel_composite_test.code tex2d_kernel.code opencl_add.cc)
else()
add_dependencies(build_tests vcpy_kernel.code global_kernel.code empty_kernel.code kernel_composite_test.code tex2d_kernel.code)
endif()
