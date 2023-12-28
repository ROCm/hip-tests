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
    atomicExch.cc
    atomicExch_system.cc
    __hip_atomic_fetch_and.cc
    __hip_atomic_fetch_or.cc
    __hip_atomic_fetch_xor.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
    set_source_files_properties(atomicExch_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicAnd_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicOr_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicXor_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicMin_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicMax_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set_source_files_properties(atomicExch_system.cc PROPERTIES COMPILE_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    hip_add_exe_to_target(NAME AtomicsTest
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS "nvrtc -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
    set(EXPECTED_ERRORS 42) # EXSWHTEC-278
elseif(HIP_PLATFORM MATCHES "amd")
    hip_add_exe_to_target(NAME AtomicsTest
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS hiprtc)
    set(EXPECTED_ERRORS 48)
endif()

add_test(NAME Unit_atomicAnd_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicAnd_negative_kernels.cc ${EXPECTED_ERRORS})

add_test(NAME Unit_atomicOr_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicOr_negative_kernels.cc ${EXPECTED_ERRORS})

add_test(NAME Unit_atomicXor_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicXor_negative_kernels.cc ${EXPECTED_ERRORS})

add_test(NAME Unit_atomicMin_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicMin_negative_kernels.cc ${EXPECTED_ERRORS})

add_test(NAME Unit_atomicMax_Negative_Parameters
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../compileAndCaptureOutput.py
         ${CMAKE_CURRENT_SOURCE_DIR} ${HIP_PLATFORM} ${HIP_PATH}
         atomicMax_negative_kernels.cc ${EXPECTED_ERRORS})

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
