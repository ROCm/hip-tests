# Common Tests - Test independent of all platforms
set(TEST_SRC
  warp_shfl_xor.cc
  warp_shfl.cc
  warp_shfl_up.cc
  warp_shfl_down.cc
  warp_ballot.cc
  warp_any.cc
  warp_all.cc
)

hip_add_exe_to_target(NAME WarpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
