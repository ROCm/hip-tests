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

template <typename T> __global__ void AtomicAndSystem(T* const addr, const T val) {
  atomicAnd_system(addr, val);
}

template <typename T> void AtomicAndCPU(T* const addr, const T val, int n) {
  for (int i = 0; i < n; ++i) __sync_fetch_and_and(addr, val);
}

TEMPLATE_TEST_CASE("Unit_atomicAnd_system_Positive_CPU", "", int, unsigned int, unsigned long,
                   unsigned long long) {
  const auto allocation_type = GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMallocManaged,
                                        LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kMask = 0xAAAA;
  const TestType kInitValue = std::numeric_limits<TestType>::max();

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemcpy(alloc.ptr(), &kInitValue, kSize, hipMemcpyHostToDevice));

  int num_blocks = 3, num_threads = 128;
  HipTest::launchKernel(AtomicAndSystem<TestType>, num_blocks, num_threads, 0, nullptr, alloc.ptr(),
                        kMask);
  AtomicAndCPU(alloc.host_ptr(), kMask, num_blocks * num_threads);

  HIP_CHECK(hipDeviceSynchronize());

  auto res = *alloc.host_ptr();

  const auto expected_res = kInitValue & kMask;
  REQUIRE(res == expected_res);
}

TEMPLATE_TEST_CASE("Unit_atomicAnd_system_Positive_PeerGPU", "", int, unsigned int, unsigned long,
                   unsigned long long) {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Two or more GPUs are required");
    return;
  }

  // Enable peer access between all pairs of devices to avoid Memory access fault by GPU node.
  for (int i = 0; i < device_count; ++i) {
    HIP_CHECK(hipSetDevice(i));
    for (int j = 0; j < device_count; ++j) {
      if (i == j) continue;
      int can_access_peer = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, i, j));
      if (!can_access_peer) {
        INFO("Peer access cannot be enabled between devices " << i << " " << j);
        REQUIRE(can_access_peer);
      }
      HIP_CHECK(hipDeviceEnablePeerAccess(j, 0));
    }
  }

  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kMask = 0xAAAA;
  const TestType kInitValue = std::numeric_limits<TestType>::max();

  HIP_CHECK(hipSetDevice(0));

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemcpy(alloc.ptr(), &kInitValue, kSize, hipMemcpyHostToDevice));

  int num_blocks = 3, num_threads = 128;
  for (int i = 0; i < device_count; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HipTest::launchKernel(AtomicAndSystem<TestType>, num_blocks, num_threads, 0, nullptr,
                          alloc.ptr(), kMask);
  }

  for (int i = 0; i < device_count; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipSetDevice(0));

  TestType res;
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  const auto expected_res = kInitValue & kMask;
  REQUIRE(res == expected_res);
}
