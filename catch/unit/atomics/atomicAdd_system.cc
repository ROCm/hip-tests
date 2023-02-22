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

#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>

template <typename T> __global__ void AtomicAddSystem(T* const addr, const T val) {
  atomicAdd_system(addr, val);
}

template <typename T> void AtomicAddCPU(T* const addr, const T val, int n) {
  for (int i = 0; i < n; ++i) __sync_fetch_and_add(addr, val);
}

TEMPLATE_TEST_CASE("Unit_atomicAdd_system_Positive_CPU", "", int, unsigned int,
                   unsigned long long) {
  const auto allocation_type = GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMallocManaged,
                                        LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 3.125 : 3;

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  memset(alloc.host_ptr(), 0, kSize);

  int num_blocks = 3, num_threads = 128;
  HipTest::launchKernel(AtomicAddSystem<TestType>, num_blocks, num_threads, 0, nullptr, alloc.ptr(),
                        kValue);
  AtomicAddCPU(alloc.host_ptr(), kValue, num_blocks * num_threads);

  HIP_CHECK(hipDeviceSynchronize());

  auto res = *alloc.host_ptr();

  const auto expected_res = num_blocks * num_threads * kValue * 2;
  REQUIRE(res == expected_res);
}

TEMPLATE_TEST_CASE("Unit_atomicAdd_system_Positive_PeerGPU", "", int, unsigned int,
                   unsigned long long, float, double) {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Two or more GPUs are required");
    return;
  }

  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 3.125 : 3;

  HIP_CHECK(hipSetDevice(0));

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemset(alloc.ptr(), 0, kSize));

  int num_blocks = 3, num_threads = 128;
  for (int i = 0; i < device_count; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HipTest::launchKernel(AtomicAddSystem<TestType>, num_blocks, num_threads, 0, nullptr,
                          alloc.ptr(), kValue);
  }

  for (int i = 0; i < device_count; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipSetDevice(0));

  TestType res;
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  const auto expected_res = num_blocks * num_threads * kValue * device_count;
  REQUIRE(res == expected_res);
}