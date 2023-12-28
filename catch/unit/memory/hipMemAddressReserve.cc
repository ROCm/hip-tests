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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipMemAddressReserve hipMemAddressReserve
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemAddressReserve (void** ptr,
 *                                   size_t size,
 *                                   size_t alignment,
 *                                   void* addr,
 *                                   unsigned long long flags)` -
 * Reserves an address range.
 */

#include <hip_test_common.hh>
#include "hip_vmm_common.hh"

#define DATA_SIZE (1 << 13)

/**
 * Test Description
 * ------------------------
 *    - Verify if reserved address returned by hipMemAddressReserve
 * for different alignment values are correctly aligned.
 * ------------------------
 *    - catch\unit\memory\hipMemAddressReserve.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemAddressReserve_AlignmentTest") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  constexpr int initializer = 0;
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device)
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
  hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem =
  ((granularity + buffer_size - 1) / granularity) * granularity;
  // Allocate virtual address range
  hipDeviceptr_t ptrA;
  size_t alignmnt = 1;
  hipMemGenericAllocationHandle_t handle;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  // Allocate host memory and intialize data
  std::vector<int> A_h(N), B_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
  }
  // check for address alignment fron 2 to 1024
  for (int iter = 0; iter < 12; iter++) {
    alignmnt = alignmnt * 2;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, alignmnt, 0, 0));
    REQUIRE((reinterpret_cast<size_t>(ptrA) % alignmnt) == 0);
    std::fill(B_h.begin(), B_h.end(), initializer);
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(ptrA, A_h.data(), buffer_size));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), ptrA, buffer_size));
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), A_h.data()));
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
  }
  HIP_CHECK(hipMemRelease(handle));
}

/**
 * Test Description
 * ------------------------
 *    - Negative Tests
 * ------------------------
 *    - catch\unit\memory\hipMemAddressReserve.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemAddressReserve_Negative") {
  size_t granularity = 0;
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device)
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
  hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t size_mem =
  ((granularity + buffer_size - 1) / granularity) * granularity;
  // Allocate virtual address range
  hipDeviceptr_t ptrA;
  SECTION("Nullptr to ptr") {
    REQUIRE(hipMemAddressReserve(nullptr, size_mem, 0, 0, 0) ==
            hipErrorInvalidValue);
  }

  SECTION("pass size as 0") {
    REQUIRE(hipMemAddressReserve(&ptrA, 0, 0, 0, 0) ==
            hipErrorMemoryAllocation);
  }
#if HT_NVIDIA
  SECTION("pass non power of two for alignment") {
    REQUIRE(hipMemAddressReserve(&ptrA, size_mem, 3, 0, 0) ==
            hipErrorMemoryAllocation);
  }
#endif
  SECTION("pass size as non multiple of host page size") {
    REQUIRE(hipMemAddressReserve(&ptrA, (size_mem - 1), 0, 0, 0) ==
            hipErrorMemoryAllocation);
  }
}
