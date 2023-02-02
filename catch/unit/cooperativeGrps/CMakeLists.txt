# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipCGThreadBlockType.cc
  hipCGMultiGridGroupType.cc
  grid_group.cc
  hipCGThreadBlockTileType.cc
  hipCGThreadBlockTileTypeShfl.cc
  hipLaunchCooperativeKernelMultiDevice.cc
  hipCGCoalescedGroupType.cc
  hipCGCoalescedGroupTypeShfl_down.cc
  hipCGCoalescedGroupTypeShfl_up.cc
  hipCGCoalescedGroupTypeShfl.cc
)
if(HIP_PLATFORM STREQUAL "nvidia")
  set_source_files_properties(hipCGMultiGridGroupType.cc PROPERTIES COMPILE_FLAGS "-D_CG_ABI_EXPERIMENTAL -rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
  set_source_files_properties(hipLaunchCooperativeKernelMultiDevice.cc PROPERTIES COMPILE_FLAGS "-D_CG_ABI_EXPERIMENTAL -rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
  hip_add_exe_to_target(NAME coopGrpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS "-rdc=true -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80, -gencode arch=compute_86,code=sm_86, -gencode=arch=compute_86,code=compute_86")
else()
  hip_add_exe_to_target(NAME coopGrpTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
endif()
