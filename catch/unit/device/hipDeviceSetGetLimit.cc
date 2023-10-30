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

/**
 * @addtogroup hipDeviceSetLimit hipDeviceSetLimit
 * @{
 * @ingroup DeviceTest
 * `hipDeviceSetLimit(enum hipLimit_t limit, size_t value)` -
 * Set Resource limits of current device.
 */

void DeviceSetLimitTest(hipLimit_t limit) {
  size_t old_val;
  HIP_CHECK(hipDeviceGetLimit(&old_val, limit));
  REQUIRE(old_val != 0);

  HIP_CHECK(hipDeviceSetLimit(limit, old_val + 8));

  size_t new_val;
  HIP_CHECK(hipDeviceGetLimit(&new_val, limit));
  REQUIRE(new_val == old_val + 8);
}

TEST_CASE("Unit_hipDeviceSetLimit_Positive_StackSize") { DeviceSetLimitTest(hipLimitStackSize); }

#if HT_NVIDIA

TEST_CASE("Unit_hipDeviceSetLimit_Positive_PrintfFifoSize") {
  DeviceSetLimitTest(hipLimitPrintfFifoSize);
}

__global__ void PrintfKernel() {
  while (true) printf("");
}

TEST_CASE("Unit_hipDeviceSetLimit_Negative_PrintfFifoSize") {
  PrintfKernel<<<1, 1>>>();
  HIP_CHECK_ERROR(hipDeviceSetLimit(hipLimitPrintfFIfoSize, 1024), hipErrorInvalidValue);
}

TEST_CASE("Unit_hipDeviceSetLimit_Positive_MallocHeapSize") {
  DeviceSetLimitTest(hipLimitMallocHeapSize);
}

__global__ void MallocKernel() {
  while (true) free(malloc(1));
}

TEST_CASE("Unit_hipDeviceSetLimit_Negative_MallocHeapSize") {
  MallocKernel<<<1, 1>>>();
  HIP_CHECK_ERROR(hipDeviceSetLimit(hipLimitMallocHeapSize, 1024), hipErrorInvalidValue);
}

TEST_CASE("Unit_hipDeviceSetLimit_Positive_MaxL2FetchGranularity") {
  DeviceSetLimitTest(hipLimitMaxL2FetchGranularity);
}

TEST_CASE("Unit_hipDeviceSetLimit_Positive_PersistingL2CacheSize") {
  DeviceSetLimitTest(hipLimitPersistingL2CacheSize);
}

#endif

TEST_CASE("Unit_hipDeviceSetLimit_Negative_Parameters") {
  HIP_CHECK_ERROR(hipDeviceSetLimit(static_cast<hipLimit_t>(-1), 1024), hipErrorUnsupportedLimit);
}

/**
 * End doxygen group hipDeviceSetLimit.
 * @}
 */

/**
 * @addtogroup hipDeviceGetLimit hipDeviceGetLimit
 * @{
 * @ingroup DeviceTest
 * `hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit)` -
 * Get Resource limits of current device.
 */

TEST_CASE("Unit_hipDeviceGetLimit_Negative_Parameters") {
  SECTION("nullptr") {
    HIP_CHECK_ERROR(hipDeviceGetLimit(nullptr, hipLimitStackSize), hipErrorInvalidValue);
  }

  SECTION("unsupported limit") {
    size_t val;
    HIP_CHECK_ERROR(hipDeviceGetLimit(&val, static_cast<hipLimit_t>(-1)), hipErrorUnsupportedLimit);
  }
}