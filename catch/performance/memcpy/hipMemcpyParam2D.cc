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

#include <performance_common.hh>
#include "memcpy_performance_common.hh"

class MemcpyParam2DBenchmark : public Benchmark<MemcpyParam2DBenchmark> {
 public:
  void operator()(size_t width, size_t height, hipMemcpyKind kind, bool enable_peer_access) {
    if (kind == hipMemcpyDeviceToHost) {
      LinearAllocGuard2D<int> device_allocation(width, height);
      LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, device_allocation.width() * height);
      hip_Memcpy2D params = CreateMemcpy2DParam(host_allocation.ptr(), device_allocation.width(),
                                                device_allocation.ptr(), device_allocation.pitch(),
                                                device_allocation.width(), device_allocation.height(),
                                                kind);
      TIMED_SECTION(kTimerTypeEvent) {
        HIP_CHECK(hipMemcpyParam2D(&params));
      }
    } else if (kind == hipMemcpyHostToDevice) {
      LinearAllocGuard2D<int> device_allocation(width, height);
      LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, device_allocation.width() * height);
      hip_Memcpy2D params = CreateMemcpy2DParam(device_allocation.ptr(), device_allocation.pitch(),
                                                host_allocation.ptr(), device_allocation.width(),
                                                device_allocation.width(), device_allocation.height(),
                                                kind);
      TIMED_SECTION(kTimerTypeEvent) {
        HIP_CHECK(hipMemcpyParam2D(&params));
      }
    } else if (kind == hipMemcpyHostToHost) {
      LinearAllocGuard<int> src_allocation(LinearAllocs::hipHostMalloc, width * sizeof(int) * height);
      LinearAllocGuard<int> dst_allocation(LinearAllocs::hipHostMalloc, width * sizeof(int) * height);
      hip_Memcpy2D params = CreateMemcpy2DParam(dst_allocation.ptr(), width * sizeof(int),
                                                src_allocation.ptr(), width * sizeof(int),
                                                width * sizeof(int), height, kind);
      TIMED_SECTION(kTimerTypeEvent) {
        HIP_CHECK(hipMemcpyParam2D(&params));
      }
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
      } else {
        dst_device = 0;
      }
      LinearAllocGuard2D<int> src_allocation(width, height);
      HIP_CHECK(hipSetDevice(dst_device));
      LinearAllocGuard2D<int> dst_allocation(width, height);

      HIP_CHECK(hipSetDevice(src_device));
      hip_Memcpy2D params = CreateMemcpy2DParam(dst_allocation.ptr(), dst_allocation.pitch(),
                                                src_allocation.ptr(), src_allocation.pitch(),
                                                dst_allocation.width(), dst_allocation.height(),
                                                kind);
      TIMED_SECTION(kTimerTypeEvent) {
        HIP_CHECK(hipMemcpyParam2D(&params));
      }
    }
  }
};

static void RunBenchmark(size_t width, size_t height, hipMemcpyKind kind,
                         bool enable_peer_access=false) {
  MemcpyParam2DBenchmark benchmark;
  std::stringstream section_name{};
  section_name << "size(" << width << ", " << height << ")";
  benchmark.AddSectionName(section_name.str());
  benchmark.Configure(1000, 100, true);
  benchmark.Run(width, height, kind, enable_peer_access);
}

#if HT_NVIDIA
TEST_CASE("Performance_hipMemcpyParam2D_DeviceToHost") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToHost);
}
#endif

TEST_CASE("Performance_hipMemcpyParam2D_HostToDevice") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyHostToDevice);
}

#if HT_NVIDIA
TEST_CASE("Performance_hipMemcpyParam2D_HostToHost") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyHostToHost);
}
#endif

TEST_CASE("Performance_hipMemcpyParam2D_DeviceToDevice_DisablePeerAccess") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToDevice);
}

TEST_CASE("Performance_hipMemcpyParam2D_DeviceToDevice_EnablePeerAccess") {
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(width, 32, hipMemcpyDeviceToDevice, true);
}
