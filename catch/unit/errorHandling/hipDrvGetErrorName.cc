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

// Local Function to return the error code in string

static const char *ErrorName(hipError_t enumerator) {
  switch (enumerator) {
    #if HT_AMD
    case hipSuccess:
        return "hipSuccess";
    case hipErrorInvalidValue:
        return "hipErrorInvalidValue";
    case hipErrorOutOfMemory:
        return "hipErrorOutOfMemory";
    case hipErrorNotInitialized:
        return "hipErrorNotInitialized";
    case hipErrorDeinitialized:
        return "hipErrorDeinitialized";
    case hipErrorProfilerDisabled:
        return "hipErrorProfilerDisabled";
    case hipErrorProfilerNotInitialized:
        return "hipErrorProfilerNotInitialized";
    case hipErrorProfilerAlreadyStarted:
        return "hipErrorProfilerAlreadyStarted";
    case hipErrorProfilerAlreadyStopped:
        return "hipErrorProfilerAlreadyStopped";
    case hipErrorInvalidConfiguration:
        return "hipErrorInvalidConfiguration";
    case hipErrorInvalidSymbol:
        return "hipErrorInvalidSymbol";
    case hipErrorInvalidDevicePointer:
        return "hipErrorInvalidDevicePointer";
    case hipErrorInvalidMemcpyDirection:
        return "hipErrorInvalidMemcpyDirection";
    case hipErrorInsufficientDriver:
        return "hipErrorInsufficientDriver";
    case hipErrorMissingConfiguration:
        return "hipErrorMissingConfiguration";
    case hipErrorPriorLaunchFailure:
        return "hipErrorPriorLaunchFailure";
    case hipErrorInvalidDeviceFunction:
        return "hipErrorInvalidDeviceFunction";
    case hipErrorNoDevice:
        return "hipErrorNoDevice";
    case hipErrorInvalidDevice:
        return "hipErrorInvalidDevice";
    case hipErrorInvalidPitchValue:
        return "hipErrorInvalidPitchValue";
    case hipErrorInvalidImage:
        return "hipErrorInvalidImage";
    case hipErrorInvalidContext:
        return "hipErrorInvalidContext";
    case hipErrorContextAlreadyCurrent:
        return "hipErrorContextAlreadyCurrent";
    case hipErrorMapFailed:
        return "hipErrorMapFailed";
    case hipErrorUnmapFailed:
        return "hipErrorUnmapFailed";
    case hipErrorArrayIsMapped:
        return "hipErrorArrayIsMapped";
    case hipErrorAlreadyMapped:
        return "hipErrorAlreadyMapped";
    case hipErrorNoBinaryForGpu:
        return "hipErrorNoBinaryForGpu";
    case hipErrorAlreadyAcquired:
        return "hipErrorAlreadyAcquired";
    case hipErrorNotMapped:
        return "hipErrorNotMapped";
    case hipErrorNotMappedAsArray:
        return "hipErrorNotMappedAsArray";
    case hipErrorNotMappedAsPointer:
        return "hipErrorNotMappedAsPointer";
    case hipErrorECCNotCorrectable:
        return "hipErrorECCNotCorrectable";
    case hipErrorUnsupportedLimit:
        return "hipErrorUnsupportedLimit";
    case hipErrorContextAlreadyInUse:
        return "hipErrorContextAlreadyInUse";
    case hipErrorPeerAccessUnsupported:
        return "hipErrorPeerAccessUnsupported";
    case hipErrorInvalidKernelFile:
        return "hipErrorInvalidKernelFile";
    case hipErrorInvalidGraphicsContext:
        return "hipErrorInvalidGraphicsContext";
    case hipErrorInvalidSource:
        return "hipErrorInvalidSource";
    case hipErrorFileNotFound:
        return "hipErrorFileNotFound";
    case hipErrorSharedObjectSymbolNotFound:
        return "hipErrorSharedObjectSymbolNotFound";
    case hipErrorSharedObjectInitFailed:
        return "hipErrorSharedObjectInitFailed";
    case hipErrorOperatingSystem:
        return "hipErrorOperatingSystem";
    case hipErrorInvalidHandle:
        return "hipErrorInvalidHandle";
    case hipErrorIllegalState:
        return "hipErrorIllegalState";
    case hipErrorNotFound:
        return "hipErrorNotFound";
    case hipErrorNotReady:
        return "hipErrorNotReady";
    case hipErrorIllegalAddress:
        return "hipErrorIllegalAddress";
    case hipErrorLaunchOutOfResources:
        return "hipErrorLaunchOutOfResources";
    case hipErrorLaunchTimeOut:
        return "hipErrorLaunchTimeOut";
    case hipErrorPeerAccessAlreadyEnabled:
        return "hipErrorPeerAccessAlreadyEnabled";
    case hipErrorPeerAccessNotEnabled:
        return "hipErrorPeerAccessNotEnabled";
    case hipErrorSetOnActiveProcess:
        return "hipErrorSetOnActiveProcess";
    case hipErrorContextIsDestroyed:
        return "hipErrorContextIsDestroyed";
    case hipErrorAssert:
        return "hipErrorAssert";
    case hipErrorHostMemoryAlreadyRegistered:
        return "hipErrorHostMemoryAlreadyRegistered";
    case hipErrorHostMemoryNotRegistered:
        return "hipErrorHostMemoryNotRegistered";
    case hipErrorLaunchFailure:
        return "hipErrorLaunchFailure";
    case hipErrorNotSupported:
        return "hipErrorNotSupported";
    case hipErrorUnknown:
        return "hipErrorUnknown";
    case hipErrorRuntimeMemory:
        return "hipErrorRuntimeMemory";
    case hipErrorRuntimeOther:
        return "hipErrorRuntimeOther";
    case hipErrorCooperativeLaunchTooLarge:
        return "hipErrorCooperativeLaunchTooLarge";
    case hipErrorStreamCaptureUnsupported:
        return "hipErrorStreamCaptureUnsupported";
    case hipErrorStreamCaptureInvalidated:
        return "hipErrorStreamCaptureInvalidated";
    case hipErrorStreamCaptureMerge:
        return "hipErrorStreamCaptureMerge";
    case hipErrorStreamCaptureUnmatched:
        return "hipErrorStreamCaptureUnmatched";
    case hipErrorStreamCaptureUnjoined:
        return "hipErrorStreamCaptureUnjoined";
    case hipErrorStreamCaptureIsolation:
        return "hipErrorStreamCaptureIsolation";
    case hipErrorStreamCaptureImplicit:
        return "hipErrorStreamCaptureImplicit";
    case hipErrorCapturedEvent:
        return "hipErrorCapturedEvent";
    case hipErrorStreamCaptureWrongThread:
        return "hipErrorStreamCaptureWrongThread";
    case hipErrorGraphExecUpdateFailure:
        return "hipErrorGraphExecUpdateFailure";
    case hipErrorTbd:
        return "hipErrorTbd";
    default:
        return "hipErrorUnknown";
    #endif
    #if HT_NVIDIA
    case hipSuccess:
        return "CUDA_SUCCESS";
    case hipErrorInvalidValue:
        return "CUDA_ERROR_INVALID_VALUE";
    case hipErrorOutOfMemory:
        return "CUDA_ERROR_OUT_OF_MEMORY";
    case hipErrorNotInitialized:
        return "CUDA_ERROR_NOT_INITIALIZED";
    case hipErrorDeinitialized:
        return "CUDA_ERROR_DEINITIALIZED";
    case hipErrorProfilerDisabled:
        return "CUDA_ERROR_PROFILER_DISABLED";
    case hipErrorProfilerNotInitialized:
        return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    case hipErrorProfilerAlreadyStarted:
        return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    case hipErrorProfilerAlreadyStopped:
        return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    case hipErrorInvalidConfiguration:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorInvalidSymbol:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorInvalidDevicePointer:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorInvalidMemcpyDirection:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorInsufficientDriver:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorMissingConfiguration:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorPriorLaunchFailure:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorInvalidDeviceFunction:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorNoDevice:
        return "CUDA_ERROR_NO_DEVICE";
    case hipErrorInvalidDevice:
        return "CUDA_ERROR_INVALID_DEVICE";
    case hipErrorInvalidPitchValue:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorInvalidImage:
        return "CUDA_ERROR_INVALID_IMAGE";
    case hipErrorInvalidContext:
        return "CUDA_ERROR_INVALID_CONTEXT";
    case hipErrorContextAlreadyCurrent:
        return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    case hipErrorMapFailed:
        return "CUDA_ERROR_MAP_FAILED";
    case hipErrorUnmapFailed:
        return "CUDA_ERROR_UNMAP_FAILED";
    case hipErrorArrayIsMapped:
        return "CUDA_ERROR_ARRAY_IS_MAPPED";
    case hipErrorAlreadyMapped:
        return "CUDA_ERROR_ALREADY_MAPPED";
    case hipErrorNoBinaryForGpu:
        return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    case hipErrorAlreadyAcquired:
        return "CUDA_ERROR_ALREADY_ACQUIRED";
    case hipErrorNotMapped:
        return "CUDA_ERROR_NOT_MAPPED";
    case hipErrorNotMappedAsArray:
        return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
    case hipErrorNotMappedAsPointer:
        return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
    case hipErrorECCNotCorrectable:
        return "CUDA_ERROR_ECC_UNCORRECTABLE";
    case hipErrorUnsupportedLimit:
        return "CUDA_ERROR_UNSUPPORTED_LIMIT";
    case hipErrorContextAlreadyInUse:
        return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
    case hipErrorPeerAccessUnsupported:
        return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
    case hipErrorInvalidKernelFile:
        return "CUDA_ERROR_INVALID_PTX";
    case hipErrorInvalidGraphicsContext:
        return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
    case hipErrorInvalidSource:
        return "CUDA_ERROR_INVALID_SOURCE";
    case hipErrorFileNotFound:
        return "CUDA_ERROR_FILE_NOT_FOUND";
    case hipErrorSharedObjectSymbolNotFound:
        return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    case hipErrorSharedObjectInitFailed:
        return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
    case hipErrorOperatingSystem:
        return "CUDA_ERROR_OPERATING_SYSTEM";
    case hipErrorInvalidHandle:
        return "CUDA_ERROR_INVALID_HANDLE";
    case hipErrorIllegalState:
        return "CUDA_ERROR_ILLEGAL_STATE";
    case hipErrorNotFound:
        return "CUDA_ERROR_NOT_FOUND";
    case hipErrorNotReady:
        return "CUDA_ERROR_NOT_READY";
    case hipErrorIllegalAddress:
        return "CUDA_ERROR_ILLEGAL_ADDRESS";
    case hipErrorLaunchOutOfResources:
        return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    case hipErrorLaunchTimeOut:
        return "CUDA_ERROR_LAUNCH_TIMEOUT";
    case hipErrorPeerAccessAlreadyEnabled:
        return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    case hipErrorPeerAccessNotEnabled:
        return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    case hipErrorSetOnActiveProcess:
        return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    case hipErrorContextIsDestroyed:
        return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
    case hipErrorAssert:
        return "CUDA_ERROR_ASSERT";
    case hipErrorHostMemoryAlreadyRegistered:
        return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    case hipErrorHostMemoryNotRegistered:
        return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
    case hipErrorLaunchFailure:
        return "CUDA_ERROR_LAUNCH_FAILED";
    case hipErrorNotSupported:
        return "CUDA_ERROR_NOT_SUPPORTED";
    case hipErrorUnknown:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorRuntimeMemory:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorRuntimeOther:
        return "CUDA_ERROR_UNKNOWN";
    case hipErrorCooperativeLaunchTooLarge:
        return "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";
    case hipErrorStreamCaptureUnsupported:
        return "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED";
    case hipErrorStreamCaptureInvalidated:
        return "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";
    case hipErrorStreamCaptureMerge:
        return "CUDA_ERROR_STREAM_CAPTURE_MERGE";
    case hipErrorStreamCaptureUnmatched:
        return "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED";
    case hipErrorStreamCaptureUnjoined:
        return "CUDA_ERROR_STREAM_CAPTURE_UNJOINED";
    case hipErrorStreamCaptureIsolation:
        return "CUDA_ERROR_STREAM_CAPTURE_ISOLATION";
    case hipErrorStreamCaptureImplicit:
        return "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT";
    case hipErrorCapturedEvent:
        return "CUDA_ERROR_CAPTURED_EVENT";
    case hipErrorStreamCaptureWrongThread:
        return "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD";
    case hipErrorGraphExecUpdateFailure:
        return "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE";
    default:
        return "CUDA_ERROR_UNKNOWN";
    #endif
    }
}

// Functional test case
// Test case to verify the returned error name is same as generated error name.

TEST_CASE("Unit_hipDrvGetErrorName_Functional") {
  const char* error_string = nullptr;
  hipError_t error_ret;
  const auto enumerator =
      GENERATE(from_range(std::begin(kErrorEnumerators),
                           std::end(kErrorEnumerators)));
  error_ret = hipDrvGetErrorName(enumerator, &error_string);
  REQUIRE(error_string != nullptr);
  REQUIRE(strcmp(error_string, ErrorName(enumerator)) == 0);
  REQUIRE(error_ret == hipSuccess);
}

// Negative test cases.

TEST_CASE("Unit_hipDrvGetErrorName_Negative") {
  const char* error_string = nullptr;
  SECTION("pass unknown value to hipError") {
    REQUIRE((hipDrvGetErrorName(static_cast<hipError_t>(-1), &error_string))
                                  == hipErrorInvalidValue);
  }
  #if HT_AMD
  SECTION("pass nullptr to error string") {
    REQUIRE((hipDrvGetErrorString(static_cast<hipError_t>(0), nullptr))
                                   == hipErrorInvalidValue);
  }
  #endif
}
