# Common Tests - Test independent of all platforms
set(TEST_SRC
  warp_ballot.cc
  warp_any.cc
  warp_all.cc
)

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC}
    warp_shfl.cc
    warp_shfl_xor.cc
    warp_shfl_up.cc
    warp_shfl_down.cc
    hipShflUpDownTest.cc
    hipShflTests.cc
    # FIXME: The tests for sync intrinsics are temporarily disabled on CUDA
    # because they depend on __match_any_sync, which is not available on older
    # NVIDIA devices. (SWDEV-453145)
    hipMatchSyncAllTests.cc
    hipMatchSyncAnyTests.cc
    hipShflSyncDownTests.cc
    hipShflSyncUpTests.cc
    hipShflSyncXorTests.cc
    hipShflSyncTests.cc
    hipVoteSyncTests.cc
  )
endif()

hip_add_exe_to_target(NAME WarpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
