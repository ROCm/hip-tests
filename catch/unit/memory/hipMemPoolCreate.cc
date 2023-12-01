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
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <hip_test_common.hh>

TEST_CASE("Unit_hipMemPoolCreate_Negative_Parameter") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  hipMemPoolProps pool_props;
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.handleTypes = hipMemHandleTypeNone;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.location.id = 0;
  pool_props.win32SecurityAttributes = nullptr;
  memset(pool_props.reserved, 0, sizeof(pool_props.reserved));

  hipMemPool_t mem_pool = nullptr;

  SECTION("Passing nullptr to mem_pool") {
    HIP_CHECK_ERROR(hipMemPoolCreate(nullptr, &pool_props), hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to props") {
    HIP_CHECK_ERROR(hipMemPoolCreate(&mem_pool, nullptr), hipErrorInvalidValue);
  }

  SECTION("Passing invalid props alloc type") {
    pool_props.allocType = hipMemAllocationTypeInvalid;
    HIP_CHECK_ERROR(hipMemPoolCreate(&mem_pool, &pool_props), hipErrorInvalidValue);
    pool_props.allocType = hipMemAllocationTypePinned;
  }

  SECTION("Passing invalid props location type") {
    pool_props.location.type = hipMemLocationTypeInvalid;
    HIP_CHECK_ERROR(hipMemPoolCreate(&mem_pool, &pool_props), hipErrorInvalidValue);
    pool_props.location.type = hipMemLocationTypeDevice;
  }

  SECTION("Passing invalid props location id") {
    pool_props.location.id = num_dev;
    HIP_CHECK_ERROR(hipMemPoolCreate(&mem_pool, &pool_props), hipErrorInvalidValue);
    pool_props.location.id = 0;
  }
}
