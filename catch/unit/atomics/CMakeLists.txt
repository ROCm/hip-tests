# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC
      atomicAnd.cc
      atomicAnd_system.cc
      atomicOr.cc
      atomicOr_system.cc
      atomicXor.cc
      atomicXor_system.cc
      atomicMin.cc
      atomicMin_system.cc
      atomicMax.cc
      atomicMax_system.cc
      safeAtomicMin.cc
      unsafeAtomicMin.cc
      safeAtomicMax.cc
      unsafeAtomicMax.cc
      __hip_atomic_fetch_min.cc
      __hip_atomic_fetch_max.cc
      atomic_builtins.cc
      acquire_release.cc
      sequential_consistency.cc
      atomicAdd.cc
      atomicAdd_system.cc
      unsafeAtomicAdd.cc
      safeAtomicAdd.cc
      atomicSub.cc
      atomicSub_system.cc
      atomicCAS.cc
      atomicCAS_system.cc
      __hip_atomic_fetch_add.cc
      __hip_atomic_compare_exchange_strong.cc
      atomicExch.cc
      atomicExch_system.cc
      __hip_atomic_fetch_and.cc
      __hip_atomic_fetch_or.cc
      __hip_atomic_fetch_xor.cc
      __hip_atomic_exchange.cc
    )

  #atomicInc & atomicDec tests are disabled on MI300X due to SWDEV-440688
  set(NOT_FOR_MI300X_TEST
      atomicInc.cc
      atomicDec.cc
  )
  set(MI300X_TARGET gfx941)
  function(CheckRejectedArchs OFFLOAD_ARCH_STR_LOCAL)
    set(ARCH_CHECK -1 PARENT_SCOPE)
    string(REGEX MATCHALL "--offload-arch=gfx[0-9a-z]+" OFFLOAD_ARCH_LIST ${OFFLOAD_ARCH_STR_LOCAL})
    foreach(OFFLOAD_ARCH IN LISTS OFFLOAD_ARCH_LIST)
      string(REGEX MATCHALL "--offload-arch=(gfx[0-9a-z]+)" matches ${OFFLOAD_ARCH})
      if (CMAKE_MATCH_COUNT EQUAL 1)
        if (CMAKE_MATCH_1 IN_LIST MI300X_TARGET)
          set(ARCH_CHECK 1 PARENT_SCOPE)
        endif()         # CMAKE_MATCH_1
      endif()        # CMAKE_MATCH_COUNT
    endforeach()   # OFFLOAD_ARCH_LIST
  endfunction()  # CheckAcceptedArchs

  if (DEFINED OFFLOAD_ARCH_STR)
    CheckRejectedArchs(${OFFLOAD_ARCH_STR})
  elseif(DEFINED $ENV{HCC_AMDGPU_TARGET})
    CheckRejectedArchs($ENV{HCC_AMDGPU_TARGET})
  else()
    set(ARCH_CHECK -1)
  endif()
  if(${ARCH_CHECK} EQUAL -1)
    message(STATUS "Adding test: ${NOT_FOR_MI300X_TEST}")
    set(TEST_SRC ${TEST_SRC} ${NOT_FOR_MI300X_TEST})
  else()
    message(STATUS "Removing test: ${NOT_FOR_MI300X_TEST}")
  endif()


  hip_add_exe_to_target(NAME AtomicsTest
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests
                        LINKER_LIBS hiprtc)
  set(EXPECTED_ERRORS 48)

  # Below tests fail in PSDB
  #add_test(NAME Unit_atomicAnd_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomicAnd_negative_kernels.cc ${EXPECTED_ERRORS})
  #
  #add_test(NAME Unit_atomicOr_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomicOr_negative_kernels.cc ${EXPECTED_ERRORS})
  #
  #add_test(NAME Unit_atomicXor_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomicXor_negative_kernels.cc ${EXPECTED_ERRORS})
  #
  #add_test(NAME Unit_atomicMin_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomicMin_negative_kernels.cc ${EXPECTED_ERRORS})
  #
  #add_test(NAME Unit_atomicMax_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomicMax_negative_kernels.cc ${EXPECTED_ERRORS})
  #add_test(NAME Unit_AtomicBuiltins_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomic_builtins_kernels.cc 60 27) # Should be 35 warnings, see EXSWHTEC-309
  #add_test(NAME Unit_atomicAdd_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomicAdd_negative_kernels.cc 48)
  #add_test(NAME Unit_atomicSub_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomicSub_negative_kernels.cc 48)
  #add_test(NAME Unit_atomicInc_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomicInc_negative_kernels.cc 8)
  #
  #add_test(NAME Unit_atomicDec_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomicDec_negative_kernels.cc 8)
  #
  #add_test(NAME Unit_atomicCAS_Negative_Parameters
  #        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #        ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #        atomicCAS_negative_kernels.cc 48)
  #
  # SWDEV-435667: Below 2 tests failed in stress test on 01/12/23
  #add_test(NAME Unit_atomicExch_Negative_Parameters
  #         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #         atomicExch_negative_kernels.cc 40)
  #
  #add_test(NAME Unit_atomicExch_system_Negative_Parameters
  #         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
  #         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
  #         atomicExch_system_negative_kernels.cc 40)
endif()