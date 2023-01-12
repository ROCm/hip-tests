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

class MemcpyDtoDAsyncBenchmark : public Benchmark<MemcpyDtoDAsyncBenchmark> {
 public:
  void operator()(bool enable_peer_access, size_t size) {
    if (HipTest::getDeviceCount() < 2) {
      HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
      return;
    }

    const StreamGuard stream_guard(Streams::created);
    const hipStream_t stream = stream_guard.stream();

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
      HIP_CHECK(hipMemcpyDtoDAsync(dst_allocation.ptr(), src_allocation.ptr(), size, stream))
    }

    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(bool enable_peer_access, size_t size) {
  MemcpyDtoDAsyncBenchmark benchmark;
  benchmark.Configure(100, 1000, true);
  auto time = benchmark.Run(enable_peer_access, size);
  std::cout << time << " ms" << std::endl;
}

TEST_CASE("Performance_hipMemcpyDtoDAsync_PeerAccessEnabled") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);

  RunBenchmark(true, allocation_size);
}

TEST_CASE("Performance_hipMemcpyDtoDAsync_PeerAccessDisabled") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);

  RunBenchmark(false, allocation_size);
}
