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

#include <hip_test_common.hh>
#include <performance_common.hh>
#include <resource_guards.hh>

class MemcpyBenchmark : public Benchmark<MemcpyBenchmark> {
 public:
  void operator()(LinearAllocs dst_allocation_type, LinearAllocs src_allocation_type, size_t size,
                  hipMemcpyKind kind, bool enable_peer_access) {
    if (kind != hipMemcpyDeviceToDevice) {
      LinearAllocGuard<int> src_allocation(src_allocation_type, size);
      LinearAllocGuard<int> dst_allocation(dst_allocation_type, size);

      TIMED_SECTION(TIMER_TYPE_EVENT) {
        HIP_CHECK(hipMemcpy(dst_allocation.ptr(), src_allocation.ptr(), size, kind));
      }
    } else {
      int src_device = 0;
      int dst_device = 1;

      if (enable_peer_access) {
        int can_access_peer = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
        if (!can_access_peer) {
          INFO("Peer access cannot be enabled between devices " << src_device << " and " << dst_device);
          REQUIRE(can_access_peer);
        }
        HIP_CHECK(hipDeviceEnablePeerAccess(dst_device, 0));
      }
      LinearAllocGuard<int> src_allocation(LinearAllocs::hipMalloc, size);
      HIP_CHECK(hipSetDevice(dst_device));
      LinearAllocGuard<int> dst_allocation(LinearAllocs::hipMalloc, size);

      HIP_CHECK(hipSetDevice(src_device));
      TIMED_SECTION(TIMER_TYPE_EVENT) {
        HIP_CHECK(hipMemcpy(dst_allocation.ptr(), src_allocation.ptr(), size, kind));
      }
    }
  }
};

static void RunBenchmark(LinearAllocs dst_allocation_type, LinearAllocs src_allocation_type, size_t size,
    hipMemcpyKind kind, bool enable_peer_access=false) {
  MemcpyBenchmark benchmark;
  benchmark.Configure(1000, 100, true);
  auto time = benchmark.Run(dst_allocation_type, src_allocation_type, size, kind, enable_peer_access);
  std::cout << time << " ms" << std::endl;
}

TEST_CASE("Performance_hipMemcpy_DeviceToHost") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);

  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToHost);
}

TEST_CASE("Performance_hipMemcpy_HostToDevice") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const auto dst_allocation_type = LinearAllocs::hipMalloc;

  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyHostToDevice);
}

TEST_CASE("Performance_hipMemcpy_HostToHost") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const auto dst_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);

  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyHostToHost);
}

TEST_CASE("Performance_hipMemcpy_DeviceToDevice_EnablePeerAccess") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = LinearAllocs::hipMalloc;

  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToDevice, true);
}

TEST_CASE("Performance_hipMemcpy_DeviceToDevice_DisablePeerAccess") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = LinearAllocs::hipMalloc;

  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToDevice);
}
