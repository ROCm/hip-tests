/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <utils.hh>
#include "hip_vmm_common.hh"

/**
 * Test Description
 * ------------------------
 *  - This test will get a handle for a malloced ptr and vmm ptr to be used to reopen in another
 *  - process.
 * Test source
 * ------------------------
 *  - unit/virtualMemoryManagement/hipMemGetHandleForAddressRangeBasic.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

constexpr size_t kNumElems = 64;
constexpr size_t kNumElemsSize = (kNumElems * sizeof(int));

TEST_CASE("GetHandleForMallocedPtr_Basic") {
 int* dptr = nullptr;
  HIP_CHECK(hipMalloc(&dptr, kNumElemsSize));
  HIP_CHECK(hipMemset(dptr, 0x00, kNumElemsSize));

  int fd = -1;
  HIP_CHECK(hipMemGetHandleForAddressRange(&fd, dptr, kNumElemsSize,
                                           hipMemRangeHandleTypeDmaBufFd, 0));

  HIP_CHECK(hipFree(dptr));
}

size_t GetGranularity(hipDevice_t device) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices
  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  assert(granularity > 0);
  return granularity;
}

hipMemGenericAllocationHandle_t GetPhysicalMemory(hipDevice_t device, size_t size_mem) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;  // Current Devices

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  return handle;
}
 
TEST_CASE("GetHandleForMemAddressReservedPtr_Basic") {
  hipDevice_t device;
  constexpr int kDeviceId = 0;
  HIP_CHECK(hipDeviceGet(&device, kDeviceId));
  checkVMMSupported(device);

  size_t granularity = GetGranularity(device);
  if (granularity <= 0) {
    std::cout<<"Invalid Granularity"<<std::endl;
    return;
  }

  size_t size_mem = ((granularity + kNumElemsSize - 1) / granularity) * granularity;
  hipDeviceptr_t ptrA;
  HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, granularity, 0, 0));

  hipMemGenericAllocationHandle_t handle = GetPhysicalMemory(device, size_mem);
 
  HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));

  int fd = 0;
  HIP_CHECK(hipMemGetHandleForAddressRange(&fd, ptrA, kNumElemsSize,
                                            hipMemRangeHandleTypeDmaBufFd, 0));
}