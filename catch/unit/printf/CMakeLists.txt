# AMD Tests
set(TEST_SRC
    printfFlags.cc
    printfLength.cc
    printfSpecifiers.cc
    printfFlagsNonHost.cc
    printfSpecifiersNonHost.cc
    printfHost.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
  set(LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(LINKER_LIBS hiprtc)
endif()

if(UNIX)
  set(AMD_TEST_SRC
      printfNonHost.cc)
endif()

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
endif()

if(HIP_PLATFORM MATCHES "amd")
    set_source_files_properties(printfFlagsNonHost.cc PROPERTIES COMPILE_OPTIONS "-mprintf-kind=buffered")
    set_source_files_properties(printfSpecifiersNonHost.cc PROPERTIES COMPILE_OPTIONS "-mprintf-kind=buffered")
    set_source_files_properties(printfNonHost.cc PROPERTIES COMPILE_OPTIONS "-mprintf-kind=buffered")
endif()

if(HIP_PLATFORM MATCHES "amd")
  hip_add_exe_to_target(NAME PrintfTest
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    LINKER_LIBS ${LINKER_LIBS}
                    PROPERTY CXX_STANDARD 17)
elseif (HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME PrintfTest
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    LINKER_LIBS ${LINKER_LIBS}
                    COMPILE_OPTIONS -std=c++17)
endif()

# This test fails in PSDB
#add_test(NAME Unit_Printf_Negative
#         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
#         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
#         printf_negative_kernels.cc 11)

# Standalone exes
add_executable(printfFlags_exe EXCLUDE_FROM_ALL printfFlags_exe.cc)
add_executable(printfLength_exe EXCLUDE_FROM_ALL printfLength_exe.cc)
add_executable(printfSpecifiers_exe EXCLUDE_FROM_ALL printfSpecifiers_exe.cc)
add_executable(printfFlagsNonHost_exe EXCLUDE_FROM_ALL printfFlagsNonHost_exe.cc)
add_executable(printfSpecifiersNonHost_exe EXCLUDE_FROM_ALL printfSpecifiersNonHost_exe.cc)

add_dependencies(build_tests printfFlags_exe)
add_dependencies(build_tests printfLength_exe)
add_dependencies(build_tests printfSpecifiers_exe)
add_dependencies(build_tests printfFlagsNonHost_exe)
add_dependencies(build_tests printfSpecifiersNonHost_exe)

