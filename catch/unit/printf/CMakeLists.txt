# AMD Tests
set(TEST_SRC
    printf_flags.cc
    printf_specifiers.cc
    printf_length.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
  set(LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(LINKER_LIBS hiprtc)
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

add_test(NAME Unit_Device_printf_Negative
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         printf_negative_kernels.cc 11)

# Standalone exes
add_executable(printf_flags_exe EXCLUDE_FROM_ALL printf_flags_exe.cc)
add_executable(printf_specifiers_exe EXCLUDE_FROM_ALL printf_specifiers_exe.cc)
add_executable(printf_length_exe EXCLUDE_FROM_ALL printf_length_exe.cc)

add_dependencies(build_tests printf_flags_exe)
add_dependencies(build_tests printf_specifiers_exe)
add_dependencies(build_tests printf_length_exe)