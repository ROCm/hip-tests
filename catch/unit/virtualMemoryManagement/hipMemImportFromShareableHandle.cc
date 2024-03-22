/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipMemExportToShareableHandle hipMemExportToShareableHandle
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t *handle,
 *                                             void *osHandle,
 *                                             hipMemAllocationHandleType shHandleType)` -
 * 	Imports an allocation from a requested shareable handle type.
 */

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

/**
 * Test Description
 * ------------------------
 *    - Basic sanity test.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemImportFromShareableHandle_Positive_Basic") {
  HIP_CHECK(hipFree(0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, granularity * 2, &prop, 0));

  void* shareable_handle = nullptr;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));

  hipMemGenericAllocationHandle_t imported_handle;
  HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle, shareable_handle,
                                            hipMemHandleTypePosixFileDescriptor));

  HIP_CHECK(hipMemRelease(handle));
}

/**
 * Test Description
 * ------------------------
 *    - Basic multiprocess sanity test.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemImportFromShareableHandle_Positive_MultiProc") {
  int fd[2];
  REQUIRE(pipe(fd) == 0);

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);

    void* shareable_handle = nullptr;
    REQUIRE(read(fd[0], &shareable_handle, sizeof(shareable_handle)) >= 0);
    REQUIRE(close(fd[0]) == 0);

    REQUIRE(shareable_handle != nullptr);

    HIP_CHECK(hipFree(0));

    hipMemGenericAllocationHandle_t imported_handle;
    HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle, shareable_handle,
                                              hipMemHandleTypePosixFileDescriptor));

    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);

    HIP_CHECK(hipFree(0));

    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);

    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;

    size_t granularity;
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, granularity * 2, &prop, 0));

    void* shareable_handle = nullptr;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                            hipMemHandleTypePosixFileDescriptor, 0));

    REQUIRE(write(fd[1], &shareable_handle, sizeof(shareable_handle)) >= 0);
    REQUIRE(close(fd[1]) == 0);

    REQUIRE(wait(NULL) >= 0);

    HIP_CHECK(hipMemRelease(handle));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemImportFromShareableHandle_Negative_Parameters") {
  HIP_CHECK(hipFree(0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, granularity * 2, &prop, 0));

  void* shareable_handle = nullptr;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));

  hipMemGenericAllocationHandle_t imported_handle;

#if HT_AMD
  SECTION("handle == nullptr") {
    HIP_CHECK_ERROR(hipMemImportFromShareableHandle(nullptr, shareable_handle,
                                                    hipMemHandleTypePosixFileDescriptor),
                    hipErrorInvalidValue);
  }
#endif

  SECTION("shareableHandle == nullptr") {
    HIP_CHECK_ERROR(hipMemImportFromShareableHandle(&imported_handle, nullptr,
                                                    hipMemHandleTypePosixFileDescriptor),
                    hipErrorInvalidValue);
  }

  SECTION("invalid handleType") {
    HIP_CHECK_ERROR(
        hipMemImportFromShareableHandle(&imported_handle, shareable_handle, hipMemHandleTypeWin32),
        hipErrorNotSupported);
  }

  HIP_CHECK(hipMemRelease(handle));
}

/**
* End doxygen group VirtualMemoryManagementTest.
* @}
*/
