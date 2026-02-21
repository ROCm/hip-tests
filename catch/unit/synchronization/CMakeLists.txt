# Common Tests - Test independent of all platforms
set(TEST_SRC
    copy_coherency.cc
)

if(NOT HIP_PLATFORM_IS_SPIRV)
add_custom_target(memcpyInt.hsaco COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/memcpyIntDevice.cpp -o
                  ${CMAKE_CURRENT_BINARY_DIR}/../synchronization/memcpyInt.hsaco -I
                  ${HIP_PATH}/include -I
                  ${CMAKE_CURRENT_SOURCE_DIR}/../../include -L
                  ${HIP_PATH}/${CMAKE_INSTALL_LIBDIR}/../../include --rocm-path=${ROCM_PATH})
endif()

# only for AMD (not SPIR-V)
if(HIP_PLATFORM MATCHES "amd" AND NOT HIP_PLATFORM_IS_SPIRV)
  set(AMD_SRC
    cache_coherency_cpu_gpu.cc
    cache_coherency_gpu_gpu.cc
  )
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
endif()

hip_add_exe_to_target(NAME synchronizationTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++14)

if(NOT HIP_PLATFORM_IS_SPIRV)
  add_dependencies(synchronizationTests memcpyInt.hsaco)
endif()
