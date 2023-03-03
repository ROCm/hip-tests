# Common Tests - Test independent of all platforms
set(TEST_SRC
  warp_shfl_up.cc
  warp_shfl_down.cc
)

hip_add_exe_to_target(NAME WarpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
