# Common Tests - Test independent of all platforms
set(TEST_SRC
    error_handling_common.cc
    hipGetErrorName.cc
    hipGetErrorString.cc
    hipDrvGetErrorName.cc
    hipDrvGetErrorString.cc
    hipGetLastError.cc
    hipPeekAtLastError.cc
)

hip_add_exe_to_target(NAME ErrorHandlingTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS "-std=c++17;-Wno-switch")