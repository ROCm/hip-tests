# Copyright (c) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Common Tests - Test independent of all platforms
set(COMMON_SHARED_SRC DriverContext.cc)

# FIXME: cjatin
# This is a temp hack to pass it on windows since it has a char limit for
# command line. Eventually we need to use rsp files with ninja on windows.
# But this exposes several problems with hipcc and build system on windows.
# While those are fixed, this hack should allow us to pass windows CI.
set(TEST_SRC
    memset.cc
    malloc.cc
    hipMemcpy2DToArray.cc
    hipMemcpy2DToArray_old.cc
    hipMemcpy2DToArrayAsync.cc
    hipMemcpy2DToArrayAsync_old.cc
    hipMemcpy3D.cc
    hipMemcpy3D_old.cc
    hipMemcpy3DAsync.cc
    hipMemcpy3DAsync_old.cc
    hipMemcpyParam2D.cc
    hipMemcpyParam2DAsync.cc
    hipMemcpy2D.cc
    hipMemcpy2DAsync.cc
    hipMemcpy2DFromArray.cc
    hipMemcpy2DFromArray_old.cc
    hipMemcpy2DFromArrayAsync.cc
    hipMemcpy2DFromArrayAsync_old.cc
    hipMemcpyAtoH.cc
    hipMemcpyAtoH_old.cc
    hipMemcpyHtoA.cc
    hipMemcpyHtoA_old.cc
    hipMemcpyAllApiNegative.cc
    hipMemcpy_MultiThread.cc
    hipHostRegister.cc
    hipHostUnregister.cc
    hipHostGetFlags.cc
    hipHostGetDevicePointer.cc
    hipMallocManaged_MultiScenario.cc
    hipMemsetNegative.cc
    hipMemset.cc
    hipMemset3D.cc
    hipMemset2D.cc
    hipHostMallocTests.cc
    hipMemset3DFunctional.cc
    hipMemset3DRegressMultiThread.cc
    hipMallocManagedFlagsTst.cc
    hipMemPrefetchAsyncExtTsts.cc
    hipMemAdviseMmap.cc
    hipMallocManaged.cc
    hipMemRangeGetAttribute.cc
    hipMemRangeGetAttribute_old.cc
    hipMemcpyFromSymbol.cc
    hipPtrGetAttribute.cc
    hipMemPoolApi.cc
    hipMemcpyPeer.cc
    hipMemcpyPeer_old.cc
    hipMemcpyPeerAsync.cc
    hipMemcpyPeerAsync_old.cc
    hipMemcpyWithStream_old.cc
    hipMemcpyWithStream.cc
    hipMemcpyWithStreamMultiThread.cc
    hipMemsetAsyncAndKernel.cc
    hipMemset2DAsyncMultiThreadAndKernel.cc
    hipMallocConcurrency.cc
    hipMemcpyDtoD.cc
    hipMemcpyDtoDAsync.cc
    hipHostMalloc.cc
    hipMemcpy_old.cc
    hipMemcpy_derivatives.cc
    hipMemcpyAsync.cc
    hipMemcpyAsync_old.cc
    hipMemcpyAsync_derivatives.cc
    hipMemsetFunctional.cc
    hipMalloc.cc
    hipMallocPitch.cc
    hipMallocArray.cc
    hipMalloc3D.cc
    hipMalloc3DArray.cc
    hipArrayCreate.cc
    hipArray3DCreate.cc
    hipDrvMemcpy3D.cc
    hipDrvMemcpy3D_old.cc
    hipDrvMemcpy3DAsync.cc
    hipDrvMemcpy3DAsync_old.cc
    hipPointerGetAttribute.cc
    hipDrvPtrGetAttributes.cc
    hipMemPrefetchAsync.cc
    hipMemGetInfo.cc
    hipFree.cc
    hipMemcpySync.cc
    hipMemsetSync.cc
    hipMemsetAsync.cc
    hipMemAdvise_old.cc
    hipMemAdvise.cc
    hipMemRangeGetAttributes.cc
    hipFreeAsync.cc
    hipMallocAsync.cc
    hipStreamAttachMemAsync.cc
    hipMemRangeGetAttributes_old.cc
    hipMemGetAddressRange.cc)

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
      ${TEST_SRC}
      hipMemPtrGetInfo.cc
      hipPointerGetAttributes.cc
      hipMemCoherencyTst.cc
      hipExtMallocWithFlags.cc
      hipMallocMngdMultiThread.cc
      hipArray.cc
      hipMemVmm.cc)
else()
  set(TEST_SRC ${TEST_SRC} hipGetSymbolSizeAddress.cc)
endif()

hip_add_exe_to_target(NAME MemoryTest
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests COMMON_SHARED_SRC ${COMMON_SHARED_SRC})

if(HIP_PLATFORM MATCHES "amd")
  set_source_files_properties(hipHostRegister.cc PROPERTIES COMPILE_FLAGS -std=c++17)
  add_executable(hipHostRegisterPerf EXCLUDE_FROM_ALL hipHostRegister_exe.cc)
endif()
