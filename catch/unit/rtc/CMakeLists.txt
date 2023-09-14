# Common Tests - Test independent of all platforms
set(TEST_SRC
    saxpy.cc
    warpsize.cc
    hipRtcFunctional.cc
    hipStreamCaptureRtc.cc
    includepath.cc
)

# AMD only tests
set(AMD_TEST_SRC
    customOptions.cc
    hipRtcBfloat16.cc
    linker.cc
    shfl.cc
)

add_custom_target(copyRtcHeaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/saxpy.h
  ${CMAKE_CURRENT_BINARY_DIR}/saxpy.h
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/headers
  ${CMAKE_CURRENT_BINARY_DIR}/headers)

if(UNIX)
   set(AMD_TEST_SRC ${TEST_SRC}
       RtcFunctions.cpp
       RtcUtility.cpp)
endif()

if(HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME RTC
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
  hip_add_exe_to_target(NAME RTC
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc)
endif()

add_dependencies(build_tests copyRtcHeaders)
