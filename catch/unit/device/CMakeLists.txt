# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipChooseDevice.cc
    hipDeviceComputeCapability.cc
    hipDeviceGetByPCIBusId.cc
    hipDeviceGetLimit_old.cc
    hipDeviceGetName.cc
    hipDeviceGetPCIBusId.cc
    hipDeviceSetGetCacheConfig.cc
    hipDeviceSynchronize.cc
    hipDeviceTotalMem.cc
    hipGetDeviceAttribute.cc
    hipGetDeviceCount.cc
    hipGetDeviceProperties.cc
    hipRuntimeGetVersion.cc
    hipGetSetDeviceFlags.cc
    hipSetGetDevice.cc
    hipDeviceGetUuid.cc
    hipDeviceAPUCheck.cc
    hipDeviceGetDefaultMemPool.cc
    hipDeviceCanAccessPeer.cc
    hipDeviceEnableDisablePeerAccess.cc
    hipExtGetLinkTypeAndHopCount.cc
    hipDeviceSetLimit_old.cc
    hipDeviceSetGetLimit.cc
    hipDeviceSetGetSharedMemConfig.cc
    hipDeviceReset.cc
    hipDeviceSetGetMemPool.cc
    hipInit.cc
    hipDriverGetVersion.cc
)

if(HIP_PLATFORM MATCHES "amd" AND BUILD_SHARED_LIBS)
  set(AMD_SRC
      hipGetProcAddressDevMgmt.cc)
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
endif()

if(UNIX)
    set(TEST_SRC ${TEST_SRC}
                 hipIpcOpenMemHandle.cc
                 hipIpcGetMemHandle.cc
                 hipIpcCloseMemHandle.cc
       )
endif()

set_source_files_properties(hipGetDeviceCount.cc PROPERTIES COMPILE_FLAGS -std=c++17)
set_source_files_properties(hipDeviceGetUuid.cc PROPERTIES COMPILE_FLAGS -std=c++17)

add_executable(getDeviceCount EXCLUDE_FROM_ALL getDeviceCount_exe.cc)

hip_add_exe_to_target(NAME DeviceTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++17)

add_dependencies(build_tests getDeviceCount)
