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

#include "bitwise_common.hh"
#include <hip_test_common.hh>

TEMPLATE_TEST_CASE("Unit_atomicXor_system_Positive_Peer_GPUs_Same_Address", "", int, unsigned int,
                   unsigned long long) {
  Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kXorSystem>(2, 2, 1,
                                                                          sizeof(TestType));
}

TEMPLATE_TEST_CASE("Unit_atomicXor_system_Positive_Peer_GPUs_Adjacent_Addresses", "", int,
                   unsigned int, unsigned long long) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kXorSystem>(2, 2, warp_size,
                                                                          sizeof(TestType));
}

TEMPLATE_TEST_CASE("Unit_atomicXor_system_Positive_Peer_GPUs_Scattered_Addresses", "", int,
                   unsigned int, unsigned long long) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  Bitwise::MultipleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kXorSystem>(2, 2, warp_size,
                                                                          cache_line_size);
}
