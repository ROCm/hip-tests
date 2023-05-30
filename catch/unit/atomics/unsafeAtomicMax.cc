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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "min_max_common.hh"
#include <hip_test_common.hh>

/**
 * @addtogroup unsafeAtomicMax unsafeAtomicMax
 * @{
 * @ingroup AtomicsTest
 * `unsafeAtomicMax(TestType* address, TestType* val)` -
 * calculates maximum between address and val, returns old value.
 */

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on the same address.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_unsafeAtomicMax_Positive_SameAddress", "", float, double) {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
        1, sizeof(TestType));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on adjacent addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_unsafeAtomicMax_Positive_Adjacent_Addresses", "", float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
        warp_size, sizeof(TestType));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on the scaterred addresses.
 *  - Uses only one device and launches one kernel.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_unsafeAtomicMax_Positive_Scattered_Addresses", "", float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    MinMax::SingleDeviceSingleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
        warp_size, cache_line_size);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on the same address.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_unsafeAtomicMax_Positive_Multi_Kernel_Same_Address", "", float, double) {
  for (auto current = 0; current < cmd_options.iterations; ++current) {
    MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
        2, 1, sizeof(TestType));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on adjacent addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_unsafeAtomicMax_Positive_Multi_Kernel_Adjacent_Addresses", "", float,
                   double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
        2, warp_size, sizeof(TestType));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs unsafeAtomicMax from multiple threads on the scaterred addresses.
 *  - Uses only one device and launches multiple kernels.
 * Test source
 * ------------------------
 *  - unit/atomics/unsafeAtomicMax.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_unsafeAtomicMax_Positive_Multi_Kernel_Scattered_Addresses", "", float,
                   double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  for (auto current = 0; current < cmd_options.iterations; ++current) {
    MinMax::SingleDeviceMultipleKernelTest<TestType, MinMax::AtomicOperation::kUnsafeMax>(
        2, warp_size, cache_line_size);
  }
}
