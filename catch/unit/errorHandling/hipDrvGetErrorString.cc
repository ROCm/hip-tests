/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include "errorEnumerators.h"

// Local Function to return the error string.

static const char *ErrorString(hipError_t enumerator) {
  switch (enumerator) {
    case hipSuccess:
      return "no error";
    case hipErrorInvalidValue:
      return "invalid argument";
    case hipErrorOutOfMemory:
      return "out of memory";
    case hipErrorNotInitialized:
      return "initialization error";
    case hipErrorDeinitialized:
      return "driver shutting down";
    case hipErrorProfilerDisabled:
      return "profiler disabled while using external profiling tool";
    case hipErrorProfilerNotInitialized:
    #if HT_AMD
      return "profiler is not initialized";
    #elif HT_NVIDIA
      return "profiler not initialized: call cudaProfilerInitialize()";
    #endif
    case hipErrorProfilerAlreadyStarted:
      return "profiler already started";
    case hipErrorProfilerAlreadyStopped:
      return "profiler already stopped";
    #if HT_AMD
    case hipErrorInvalidConfiguration:
      return "invalid configuration argument";
    #elif HT_NVIDIA
      return "unknown error";
    #endif
    #if HT_AMD
    case hipErrorInvalidPitchValue:
      return "invalid pitch argument";
    #elif HT_NVIDIA
      return "unknown error";
    #endif
    #if HT_AMD
    case hipErrorInvalidSymbol:
      return "invalid device symbol";
    #elif HT_NVIDIA
      return "unknown error";
    #endif
    #if HT_AMD
    case hipErrorInvalidDevicePointer:
      return "invalid device pointer";
    #elif HT_NVIDIA
      return "unknown error";
    #endif
    #if HT_AMD
    case hipErrorInvalidMemcpyDirection:
      return "invalid copy direction for memcpy";
    #elif HT_NVIDIA
      return "unknown error";
    #endif
    #if HT_AMD
    case hipErrorInsufficientDriver:
      return "driver version is insufficient for runtime version";
    #elif HT_NVIDIA
      return "unknown error";
    #endif
    #if HT_AMD
    case hipErrorMissingConfiguration:
      return "__global__ function call is not configured";
    #elif HT_NVIDIA
      return "unknown error";
    #endif
    #if HT_AMD
    case hipErrorPriorLaunchFailure:
      return "unspecified launch failure in prior launch";
    #elif HT_NVIDIA
      return "unknown error";
    #endif
    #if HT_AMD
    case hipErrorInvalidDeviceFunction:
      return "invalid device function";
    #elif HT_NVIDIA
      return "unknown error";
    #endif
    case hipErrorNoDevice:
    #if HT_AMD
      return "no ROCm-capable device is detected";
    #elif HT_NVIDIA
      return "no CUDA-capable device is detected";
    #endif
    case hipErrorInvalidDevice:
      return "invalid device ordinal";
    case hipErrorInvalidImage:
      return "device kernel image is invalid";
    case hipErrorInvalidContext:
      return "invalid device context";
    case hipErrorContextAlreadyCurrent:
    #if HT_AMD
      return "context is already current context";
    #elif HT_NVIDIA
      return "context already current";
    #endif
    case hipErrorMapFailed:
      return "mapping of buffer object failed";
    case hipErrorUnmapFailed:
      return "unmapping of buffer object failed";
    case hipErrorArrayIsMapped:
      return "array is mapped";
    case hipErrorAlreadyMapped:
      return "resource already mapped";
    case hipErrorNoBinaryForGpu:
      return "no kernel image is available for execution on the device";
    case hipErrorAlreadyAcquired:
      return "resource already acquired";
    case hipErrorNotMapped:
      return "resource not mapped";
    case hipErrorNotMappedAsArray:
      return "resource not mapped as array";
    case hipErrorNotMappedAsPointer:
      return "resource not mapped as pointer";
    case hipErrorECCNotCorrectable:
      return "uncorrectable ECC error encountered";
    case hipErrorUnsupportedLimit:
      return "limit is not supported on this architecture";
    case hipErrorContextAlreadyInUse:
      return "exclusive-thread device already in use by a different thread";
    case hipErrorPeerAccessUnsupported:
      return "peer access is not supported between these two devices";
    case hipErrorInvalidKernelFile:
    #if HT_AMD
      return "invalid kernel file";
    #elif HT_NVIDIA
      return "a PTX JIT compilation failed";
    #endif
    case hipErrorInvalidGraphicsContext:
      return "invalid OpenGL or DirectX context";
    case hipErrorInvalidSource:
      return "device kernel image is invalid";
    case hipErrorFileNotFound:
      return "file not found";
    case hipErrorSharedObjectSymbolNotFound:
      return "shared object symbol not found";
    case hipErrorSharedObjectInitFailed:
      return "shared object initialization failed";
    case hipErrorOperatingSystem:
      return "OS call failed or operation not supported on this OS";
    case hipErrorInvalidHandle:
      return "invalid resource handle";
    case hipErrorIllegalState:
      return "the operation cannot be performed in the present state";
    case hipErrorNotFound:
      return "named symbol not found";
    case hipErrorNotReady:
      return "device not ready";
    case hipErrorIllegalAddress:
      return "an illegal memory access was encountered";
    case hipErrorLaunchOutOfResources:
      return "too many resources requested for launch";
    case hipErrorLaunchTimeOut:
      return "the launch timed out and was terminated";
    case hipErrorPeerAccessAlreadyEnabled:
      return "peer access is already enabled";
    case hipErrorPeerAccessNotEnabled:
      return "peer access has not been enabled";
    case hipErrorSetOnActiveProcess:
      return "cannot set while device is active in this process";
    case hipErrorContextIsDestroyed:
      return "context is destroyed";
    case hipErrorAssert:
      return "device-side assert triggered";
    case hipErrorHostMemoryAlreadyRegistered:
      return "part or all of the requested memory range is already mapped";
    case hipErrorHostMemoryNotRegistered:
      return "pointer does not correspond to a registered memory region";
    case hipErrorLaunchFailure:
      return "unspecified launch failure";
    case hipErrorCooperativeLaunchTooLarge:
      return "too many blocks in cooperative launch";
    case hipErrorNotSupported:
      return "operation not supported";
    case hipErrorStreamCaptureUnsupported:
      return "operation not permitted when stream is capturing";
    case hipErrorStreamCaptureInvalidated:
      return "operation failed due to a previous error during capture";
    case hipErrorStreamCaptureMerge:
      return "operation would result in a merge of separate capture sequences";
    case hipErrorStreamCaptureUnmatched:
      return "capture was not ended in the same stream as it began";
    case hipErrorStreamCaptureUnjoined:
      return "capturing stream has unjoined work";
    case hipErrorStreamCaptureIsolation:
      return "dependency created on uncaptured work in another stream";
    case hipErrorStreamCaptureImplicit:
      return "operation would make the legacy stream depend on a capturing blocking stream";  //NOLINT
    case hipErrorCapturedEvent:
      return "operation not permitted on an event last recorded in a capturing stream";  //NOLINT
    case hipErrorStreamCaptureWrongThread:
      return "attempt to terminate a thread-local capture sequence from another thread";  //NOLINT
    case hipErrorGraphExecUpdateFailure:
      return "the graph update was not performed because it included changes which violated constraints specific to instantiated graph update";  //NOLINT
    case hipErrorRuntimeMemory:
      return "runtime memory call returned error";
    case hipErrorRuntimeOther:
      return "runtime call other than memory returned error";
    case hipErrorUnknown:
    default:
    #if HT_AMD
      return "unknown error";
    #elif HT_NVIDIA
      return "unknown error";
    #endif
  }
}

// Test case to verify the returned error string is
// same as generated error string.

TEST_CASE("Unit_hipDrvGetErrorString_Functional") {
  const char* error_string = nullptr;
  const auto enumerator =
      GENERATE(from_range(std::begin(kErrorEnumerators),
                           std::end(kErrorEnumerators)));
  hipError_t error_ret = hipDrvGetErrorString(enumerator, &error_string);
  REQUIRE(error_string != nullptr);
  REQUIRE(strcmp(error_string, ErrorString(enumerator)) == 0);
  REQUIRE(error_ret == hipSuccess);
}

// Negative test cases.

TEST_CASE("Unit_hipDrvGetErrorString_Negative") {
  const char* error_string = nullptr;
  SECTION("pass unknown value to hipError") {
    REQUIRE((hipDrvGetErrorString(static_cast<hipError_t>(-1), &error_string))
                                  == hipErrorInvalidValue);
  }
  #if HT_AMD
  SECTION("pass nullptr to error string") {
     REQUIRE((hipDrvGetErrorString(static_cast<hipError_t>(0), nullptr))
                                   == hipErrorInvalidValue);
  }
  #endif
}
