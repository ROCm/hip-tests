# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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
    hipStreamWaitEvent.cc
    hipStreamGetFlags.cc
    hipStreamGetPriority.cc
    hipExtStreamCreateWithCUMask.cc
    hipExtStreamGetCUMask.cc
    hipStreamAddCallback.cc
    hipStreamWaitValue.cc
    hipStreamWriteValue.cc
    hipMallocAsync.cc
    hipFreeAsync.cc
    hipMemPoolCreate.cc
    hipMemPoolDestroy.cc
    hipMemPoolTrimTo.cc
    hipMemPoolSetAttribute.cc
    hipMemPoolGetAttribute.cc
    hipMemPoolSetAccess.cc
    hipMallocFromPoolAsync.cc
    hipMemPoolExportToShareableHandle.cc
    hipMemPoolImportFromShareableHandle.cc
    hipMemPoolExportPointer.cc
    hipMemPoolImportPointer.cc
    hipStreamBasic.cc
)
else()
set(TEST_SRC
    hipStreamWaitEvent.cc
    hipStreamGetFlags.cc
    hipStreamGetPriority.cc
    hipStreamAddCallback.cc
    hipStreamWaitValue.cc
    hipStreamWriteValue.cc
    hipMallocAsync.cc
    hipFreeAsync.cc
    hipStreamBasic.cc
)
endif()

hip_add_exe_to_target(NAME StreamPerformance
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++17)
