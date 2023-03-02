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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "atomic_exch_common.hh"

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Same_Address_Compile_Time", "", int, unsigned int,
                   unsigned long long, float) {
  AtomicExchSameAddressTest<TestType, AtomicScopes::device>();
}


TEMPLATE_TEST_CASE("Unit_atomicExch_Positive", "", int, unsigned int, unsigned long long, float) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  SECTION("Same address") {
    AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::device>(1, sizeof(TestType));
  }

  SECTION("Adjacent addresses") {
    AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::device>(warp_size,
                                                                           sizeof(TestType));
  }

  SECTION("Scattered addresses") {
    AtomicExchSingleDeviceSingleKernelTest<TestType, AtomicScopes::device>(warp_size,
                                                                           cache_line_size);
  }
}


TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Multi_Kernel", "", int, unsigned int,
                   unsigned long long, float) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  SECTION("Same address") {
    AtomicExchSingleDeviceMultipleKernelTest<TestType, AtomicScopes::device>(2, 1,
                                                                             sizeof(TestType));
  }

  SECTION("Adjacent addresses") {
    AtomicExchSingleDeviceMultipleKernelTest<TestType, AtomicScopes::device>(2, warp_size,
                                                                             sizeof(TestType));
  }

  SECTION("Scattered addresses") {
    AtomicExchSingleDeviceMultipleKernelTest<TestType, AtomicScopes::device>(2, warp_size,
                                                                             cache_line_size);
  }
}
