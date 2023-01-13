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

class Memcpy2DFromArrayAsyncBenchmark : public Benchmark<Memcpy2DFromArrayAsyncBenchmark> {
 public:
  void operator()(size_t width, size_t height, hipMemcpyKind kind, bool enable_peer_access){
    const StreamGuard stream_guard(Streams::created);
    const hipStream_t stream = stream_guard.stream();

    if (kind == hipMemcpyHostToDevice) {
      size_t allocation_size = width * height * sizeof(int);
      LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, allocation_size);
      ArrayAllocGuard<int> array_allocation(make_hipExtent(width, height, 0), hipArrayDefault);

      TIMED_SECTION(kTimerTypeEvent, stream) {
        HIP_CHECK(hipMemcpy2DFromArrayAsync(host_allocation.ptr(), width * sizeof(int),
                  array_allocation.ptr(), 0, 0, width * sizeof(int), height,
                  hipMemcpyHostToDevice, stream));
      }
      HIP_CHECK(hipStreamSynchronize(stream));
    } else {
      // hipMemcpyDeviceToDevice
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
      LinearAllocGuard2D<int> device_allocation(width, height);
      HIP_CHECK(hipSetDevice(dst_device));
      ArrayAllocGuard<int> array_allocation(make_hipExtent(width, height, 0), hipArrayDefault);

      HIP_CHECK(hipSetDevice(src_device));
      TIMED_SECTION(kTimerTypeEvent, stream) {
        HIP_CHECK(hipMemcpy2DFromArrayAsync(device_allocation.ptr(), device_allocation.pitch(),
                  array_allocation.ptr(), 0, 0, device_allocation.width(),
                  device_allocation.height(), hipMemcpyHostToDevice, stream));
      }
      HIP_CHECK(hipStreamSynchronize(stream));
    }
  }
};

static void RunBenchmark(size_t width, size_t height, hipMemcpyKind kind, bool enable_peer_access=false) {
  Memcpy2DFromArrayAsyncBenchmark benchmark;
  std::stringstream section_name{};
  section_name << "size(" << width << ", " << height << ")";
  benchmark.AddSectionName(section_name.str());
  benchmark.Configure(1000, 100, true);
  benchmark.Run(width, height, kind, enable_peer_access);
}

TEST_CASE("Performance_hipMemcpy2DFromArrayAsync_HostToDevice") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyHostToDevice);
}

TEST_CASE("Performance_hipMemcpy2DFromArrayAsync_DeviceToDevice_DisablePeerAccess") {
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToDevice);
}

TEST_CASE("Performance_hipMemcpy2DFromArrayAsync_DeviceToDevice_EnablePeerAccess") {
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToDevice, true);
}
