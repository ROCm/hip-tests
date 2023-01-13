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

static hipMemcpy3DParms CreateMemcpy3DParam(hipPitchedPtr dst_ptr, hipPos dst_pos, hipPitchedPtr src_ptr,
                                            hipPos src_pos, hipExtent extent, hipMemcpyKind kind) {
  hipMemcpy3DParms params = {0};
  params.dstPtr = dst_ptr;
  params.dstPos = dst_pos;
  params.srcPtr = src_ptr;
  params.srcPos = src_pos;
  params.extent = extent;
  params.kind = kind;
  return params;
}

class Memcpy3DBenchmark : public Benchmark<Memcpy3DBenchmark> {
 public:
  void operator()(const hipExtent extent, hipMemcpyKind kind, bool enable_peer_access) {
    if (kind == hipMemcpyDeviceToHost) {
      LinearAllocGuard3D<int> device_allocation(extent);
      LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, device_allocation.width() * 
                                            device_allocation.height() * device_allocation.depth());
      hipMemcpy3DParms params = CreateMemcpy3DParam(make_hipPitchedPtr(host_allocation.ptr(), device_allocation.width(), 
                                                    device_allocation.width(), device_allocation.height()),
                                                    make_hipPos(0, 0, 0), device_allocation.pitched_ptr(),
                                                    make_hipPos(0, 0, 0),
                                                    device_allocation.extent(), kind);
      TIMED_SECTION(TIMER_TYPE_EVENT) {
        HIP_CHECK(hipMemcpy3D(&params));
      }
    } else if (kind == hipMemcpyHostToDevice) {
      LinearAllocGuard3D<int> device_allocation(extent);
      const size_t host_pitch = device_allocation.pitch();
      LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, device_allocation.pitch() * 
                                            device_allocation.height() * device_allocation.depth());
      hipMemcpy3DParms params = CreateMemcpy3DParam(device_allocation.pitched_ptr(), make_hipPos(0, 0, 0),
                                                    make_hipPitchedPtr(host_allocation.ptr(), device_allocation.pitch(),
                                                                       device_allocation.width(), device_allocation.height()),
                                                    make_hipPos(0, 0, 0), device_allocation.extent(), kind);
      TIMED_SECTION(TIMER_TYPE_EVENT) {
        HIP_CHECK(hipMemcpy3D(&params));
      }
    } else if (kind == hipMemcpyHostToHost) {
      LinearAllocGuard3D<int> device_allocation(extent);
      LinearAllocGuard<int> src_allocation(LinearAllocs::hipHostMalloc, extent.width * 
                                           extent.height * extent.depth);
      LinearAllocGuard<int> dst_allocation(LinearAllocs::hipHostMalloc, extent.width * 
                                           extent.height * extent.depth);
      hipMemcpy3DParms params = CreateMemcpy3DParam(make_hipPitchedPtr(dst_allocation.ptr(), extent.width, extent.width, extent.height),
                                                    make_hipPos(0, 0, 0),
                                                    make_hipPitchedPtr(src_allocation.ptr(), extent.width, extent.width, extent.height),
                                                    make_hipPos(0, 0, 0), extent, kind);
      TIMED_SECTION(TIMER_TYPE_EVENT) {
        HIP_CHECK(hipMemcpy3D(&params));
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
      }
      LinearAllocGuard3D<int> src_allocation(extent);
      HIP_CHECK(hipSetDevice(dst_device));
      LinearAllocGuard3D<int> dst_allocation(extent);

      HIP_CHECK(hipSetDevice(src_device));
      hipMemcpy3DParms params = CreateMemcpy3DParam(dst_allocation.pitched_ptr(),
                                                    make_hipPos(0, 0, 0),
                                                    src_allocation.pitched_ptr(),
                                                    make_hipPos(0, 0, 0), dst_allocation.extent(), kind);
      TIMED_SECTION(TIMER_TYPE_EVENT) {
        HIP_CHECK(hipMemcpy3D(&params));
      }
    }
  }
};

static void RunBenchmark(const hipExtent extent, hipMemcpyKind kind, bool enable_peer_access=false) {
  Memcpy3DBenchmark benchmark;
  std::stringstream section_name{};
  section_name << "extent(" << extent.width << ", " << extent.height << ", " << extent.depth << ")";
  benchmark.AddSectionName(section_name.str());
  benchmark.Configure(1000, 100, true);
  benchmark.Run(extent, kind, enable_peer_access);
}

TEST_CASE("Performance_hipMemcpy3D_DeviceToHost") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyDeviceToHost);
}

TEST_CASE("Performance_hipMemcpy3D_HostToDevice") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyHostToDevice);
}

TEST_CASE("Performance_hipMemcpy3D_HostToHost") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyHostToHost);
}

TEST_CASE("Performance_hipMemcpy3D_DeviceToDevice_DisablePeerAccess") {
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyDeviceToDevice);
}

TEST_CASE("Performance_hipMemcpy3D_DeviceToDevice_EnablePeerAccess") {
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyDeviceToDevice, true);
}
