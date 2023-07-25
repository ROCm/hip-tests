# Common Tests - Test independent of all platforms
set(TEST_SRC
    floatMath.cc
    anyAll.cc
    ballot.cc
    clz.cc
    ffs.cc
    funnelshift.cc
    brev.cc
    popc.cc
    ldg.cc
    threadfence_system.cc
    syncthreadsand.cc
    syncthreadscount.cc
    syncthreadsor.cc
    deviceAllocation.cc
    Atomic_func.cc
    hfloat2half.cc
    DoublePrecisionIntrinsics.cc
    DoublePrecisionMathDevice.cc
    DoublePrecisionMathHost.cc
    FloatMathPrecise.cc
    IntegerIntrinsics.cc
    SinglePrecisionIntrinsics.cc
    SinglePrecisionMathDevice.cc
    SinglePrecisionMathHost.cc
    SimpleAtomicsTest.cc
    hipTestAtomicAdd.cc
    hipStdComplex.cc
    hipTestClock.cc
)

# AMD only tests
set(AMD_TEST_SRC
    unsafeAtomicAdd.cc
    mbcnt.cc
    bitExtract.cc
    bitInsert.cc
    floatTM.cc
    hipMathFunctions.cc
    hmax_hmin.cc
    bfloat16.cc
)
set(AMD_ARCH_SPEC_TEST_SRC
    AtomicAdd_Coherent_withunsafeflag.cc
    AtomicAdd_Coherent_withoutflag.cc
    AtomicAdd_Coherent_withnoUnsafeflag.cc
    AtomicAdd_NonCoherent_withoutflag.cc
    AtomicAdd_NonCoherent_withnoUnsafeflag.cc
    AtomicAdd_NonCoherent_withunsafeflag.cc
    BuiltIns_fmax.cc
    BuiltIns_fmin.cc
    BuiltIns_fadd.cc
    unsafeAtomicAdd_RTC.cc
    unsafeAtomicAdd_Coherent_withunsafeflag.cc
    unsafeAtomicAdd_Coherent_withoutflag.cc
    unsafeAtomicAdd_Coherent_withnounsafeflag.cc
    unsafeAtomicAdd_NonCoherent_withoutflag.cc
    unsafeAtomicAdd_NonCoherent_withnounsafeflag.cc
    unsafeAtomicAdd_NonCoherent_withunsafeflag.cc
)

# Note to pass arch use format like -DOFFLOAD_ARCH_STR="--offload-arch=gfx900  --offload-arch=gfx906"
# having space at the start/end of OFFLOAD_ARCH_STR can cause build failures
add_custom_target(kerDevAllocMultCO.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/kerDevAllocMultCO.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/deviceLib/kerDevAllocMultCO.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})
add_custom_target(kerDevWriteMultCO.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/kerDevWriteMultCO.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/deviceLib/kerDevWriteMultCO.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})
add_custom_target(kerDevFreeMultCO.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/kerDevFreeMultCO.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/deviceLib/kerDevFreeMultCO.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})
add_custom_target(kerDevAllocSingleKer.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/kerDevAllocSingleKer.cc
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/deviceLib/kerDevAllocSingleKer.code
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include --rocm-path=${ROCM_PATH})

# Accepted archs to compile this cmake file
set(ACCEPTED_OFFLOAD_ARCHS gfx90a gfx940 gfx941 gfx942)
function(CheckAcceptedArchs OFFLOAD_ARCH_STR_LOCAL)
  set(ARCH_CHECK -1 PARENT_SCOPE)
  string(REGEX MATCHALL "--offload-arch=gfx[0-9a-z]+" OFFLOAD_ARCH_LIST ${OFFLOAD_ARCH_STR_LOCAL})
  foreach(OFFLOAD_ARCH IN LISTS OFFLOAD_ARCH_LIST)
    string(REGEX MATCHALL "--offload-arch=(gfx[0-9a-z]+)" matches ${OFFLOAD_ARCH})
    if (CMAKE_MATCH_COUNT EQUAL 1)
      if (CMAKE_MATCH_1 IN_LIST ACCEPTED_OFFLOAD_ARCHS)
        set(ARCH_CHECK 1 PARENT_SCOPE)
      endif()         # CMAKE_MATCH_1
    endif()        # CMAKE_MATCH_COUNT
  endforeach()   # OFFLOAD_ARCH_LIST
endfunction()  # CheckAcceptedArchs

if(HIP_PLATFORM MATCHES "amd")
    if (DEFINED OFFLOAD_ARCH_STR)
      CheckAcceptedArchs(${OFFLOAD_ARCH_STR})
    elseif(DEFINED $ENV{HCC_AMDGPU_TARGET})
      CheckAcceptedArchs($ENV{HCC_AMDGPU_TARGET})
    else()
      set(ARCH_CHECK -1)
    endif()
    set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
    set_source_files_properties(floatTM.cc PROPERTIES COMPILE_FLAGS -std=c++17)
if(${ARCH_CHECK} GREATER_EQUAL 0)
	set(TEST_SRC ${TEST_SRC} ${AMD_ARCH_SPEC_TEST_SRC})
    set_source_files_properties(AtomicAdd_Coherent_withunsafeflag.cc PROPERTIES COMPILE_OPTIONS "-munsafe-fp-atomics")
    set_source_files_properties(AtomicAdd_NonCoherent_withunsafeflag.cc PROPERTIES COMPILE_OPTIONS "-munsafe-fp-atomics")
    set_source_files_properties(AtomicAdd_Coherent_withnoUnsafeflag.cc PROPERTIES COMPILE_OPTIONS "-mno-unsafe-fp-atomics")
    set_source_files_properties(AtomicAdd_NonCoherent_withnoUnsafeflag.cc PROPERTIES COMPILE_OPTIONS "-mno-unsafe-fp-atomics")
    set_source_files_properties(unsafeAtomicAdd_Coherent_withunsafeflag.cc PROPERTIES COMPILE_OPTIONS "-munsafe-fp-atomics")
    set_source_files_properties(unsafeAtomicAdd_NonCoherent_withunsafeflag.cc PROPERTIES COMPILE_OPTIONS "-munsafe-fp-atomics")
    set_source_files_properties(unsafeAtomicAdd_Coherent_withnounsafeflag.cc PROPERTIES COMPILE_OPTIONS "-mno-unsafe-fp-atomics")
    set_source_files_properties(unsafeAtomicAdd_NonCoherent_withnounsafeflag.cc PROPERTIES COMPILE_OPTIONS "-mno-unsafe-fp-atomics")
    set_source_files_properties(hipMathFunctions.cc PROPERTIES COMPILE_FLAGS "-Xclang -fallow-half-arguments-and-returns")
    file(GLOB AtomicAdd_files *AtomicAdd_*_*.cc)
    set_property(SOURCE ${AtomicAdd_files} PROPERTY COMPILE_FLAGS --save-temps)
    file(GLOB unsafeAtomicAdd_files *unsafeAtomicAdd_*_*.cc)
    set_property(SOURCE ${unsafeAtomicAdd_files} PROPERTY COMPILE_FLAGS --save-temps)
endif()
    hip_add_exe_to_target(NAME UnitDeviceTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc)
elseif(HIP_PLATFORM MATCHES "nvidia")
    hip_add_exe_to_target(NAME UnitDeviceTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS --Wno-deprecated-declarations)
endif()

add_dependencies(build_tests kerDevAllocMultCO.code kerDevWriteMultCO.code kerDevFreeMultCO.code kerDevAllocSingleKer.code)
