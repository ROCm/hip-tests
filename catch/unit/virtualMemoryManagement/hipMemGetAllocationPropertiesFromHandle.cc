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
 * @addtogroup hipMemGetAllocationPropertiesFromHandle hipMemGetAllocationPropertiesFromHandle
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
 *                                                     hipMemGenericAllocationHandle_t handle)` -
 * Retrieve the property structure of the given handle.
 */

#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

#define DATA_SIZE (1 << 13)

/**
 * Test Description
 * ------------------------
 *    - Functional test to verify the values of hipMemAllocationProp properties.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemGetAllocationPropertiesFromHandle.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemGetAllocationPropertiesFromHandle_functional") {
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);
  hipMemGenericAllocationHandle_t handle;
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  // create a temp prop structure.
  hipMemAllocationProp prop_temp = {};
  size_t granularity = 0;
  int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t mem_size = ((granularity + buffer_size - 1) / granularity) * granularity;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, mem_size, &prop, 0));
  // verify properties has been retrived from handle
  HIP_CHECK(hipMemGetAllocationPropertiesFromHandle(&prop_temp, handle));
  REQUIRE(prop_temp.type == prop.type);
  REQUIRE(prop_temp.location.type == prop.location.type);
  REQUIRE(prop_temp.location.id == prop.location.id);
  HIP_CHECK(hipMemRelease(handle));
}

/**
 * Test Description
 * ------------------------
 *    - Negative Tests.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemGetAllocationPropertiesFromHandle.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemGetAllocationPropertiesFromHandle_Negative") {
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);
  hipMemGenericAllocationHandle_t handle;
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  // create a temp prop structure.
  hipMemAllocationProp prop_temp = {};
  size_t granularity = 0;
  int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  size_t mem_size = ((granularity + buffer_size - 1) / granularity) * granularity;
  // Allocate physical memory
  HIP_CHECK(hipMemCreate(&handle, mem_size, &prop, 0));

  SECTION("Nullptr as prop") {
    REQUIRE(hipMemGetAllocationPropertiesFromHandle(nullptr, handle) == hipErrorInvalidValue);
  }

  SECTION("null handle") {
    prop.location.type = hipMemLocationTypeInvalid;
    REQUIRE(hipMemGetAllocationPropertiesFromHandle(&prop_temp, nullptr) == hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemRelease(handle));
}
