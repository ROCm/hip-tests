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

/**
 * @addtogroup memcpy memcpy
 * @{
 * @ingroup PerformanceTest
 */

class Memcpy3DAsyncBenchmark : public Benchmark<Memcpy3DAsyncBenchmark> {
 public:
  void operator()(const hipExtent extent, hipMemcpyKind kind, bool enable_peer_access) {
    const StreamGuard stream_guard(Streams::created);
    const hipStream_t stream = stream_guard.stream();

    if (kind == hipMemcpyDeviceToHost) {
      LinearAllocGuard3D<int> device_allocation(extent);
      LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, device_allocation.width() * 
                                            device_allocation.height() * device_allocation.depth());
      hipMemcpy3DParms params = CreateMemcpy3DParam(make_hipPitchedPtr(host_allocation.ptr(),
                                                                       device_allocation.width(),
                                                                       device_allocation.width(),
                                                                       device_allocation.height()),
                                                    make_hipPos(0, 0, 0), device_allocation.pitched_ptr(),
                                                    make_hipPos(0, 0, 0),
                                                    device_allocation.extent(), kind);
      TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
        HIP_CHECK(hipMemcpy3D(&params));
      }
      HIP_CHECK(hipStreamSynchronize(stream));
    } else if (kind == hipMemcpyHostToDevice) {
      LinearAllocGuard3D<int> device_allocation(extent);
      LinearAllocGuard<int> host_allocation(LinearAllocs::hipHostMalloc, device_allocation.pitch() * 
                                            device_allocation.height() * device_allocation.depth());
      hipMemcpy3DParms params = CreateMemcpy3DParam(device_allocation.pitched_ptr(), make_hipPos(0, 0, 0),
                                                    make_hipPitchedPtr(host_allocation.ptr(),
                                                                       device_allocation.pitch(),
                                                                       device_allocation.width(),
                                                                       device_allocation.height()),
                                                    make_hipPos(0, 0, 0), device_allocation.extent(), kind);
      TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
        HIP_CHECK(hipMemcpy3D(&params));
      }
      HIP_CHECK(hipStreamSynchronize(stream));
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
      TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
        HIP_CHECK(hipMemcpy3D(&params));
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
      } else {
        dst_device = 0;
      }
      LinearAllocGuard3D<int> src_allocation(extent);
      HIP_CHECK(hipSetDevice(dst_device));
      LinearAllocGuard3D<int> dst_allocation(extent);

      HIP_CHECK(hipSetDevice(src_device));
      hipMemcpy3DParms params = CreateMemcpy3DParam(dst_allocation.pitched_ptr(),
                                                    make_hipPos(0, 0, 0),
                                                    src_allocation.pitched_ptr(),
                                                    make_hipPos(0, 0, 0), dst_allocation.extent(), kind);
      TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
        HIP_CHECK(hipMemcpy3D(&params));
      }
      HIP_CHECK(hipStreamSynchronize(stream));
    }
  }
};

static void RunBenchmark(const hipExtent extent, hipMemcpyKind kind, bool enable_peer_access=false) {
  Memcpy3DAsyncBenchmark benchmark;
  std::stringstream section_name{};
  section_name << "extent(" << extent.width << ", " << extent.height << ", " << extent.depth << ")";
  benchmark.AddSectionName(section_name.str());
  benchmark.Configure(1000, 100, true);
  benchmark.Run(extent, kind, enable_peer_access);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Device to Host:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 *  - unit/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy3DAsync_DeviceToHost") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyDeviceToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Host to Device:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 *  - unit/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy3DAsync_HostToDevice") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyHostToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Host to Host:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 *  - unit/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy3DAsync_HostToHost") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyHostToHost);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Device to Device with peer access disabled:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 *  - unit/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy3DAsync_DeviceToDevice_DisablePeerAccess") {
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyDeviceToDevice);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpy3DAsync` from Device to Device with peer access enabled:
 *    -# Allocation size
 *      - Small: 4 KB x 16 B x 4 B
 *      - Medium: 4 MB x 16 B x 4 B
 *      - Large: 16 MB x 16 B x 4 B
 * Test source
 * ------------------------
 *  - unit/memcpy/hipMemcpy3DAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpy3DAsync_DeviceToDevice_EnablePeerAccess") {
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto width = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(make_hipExtent(width, 16, 4), hipMemcpyDeviceToDevice, true);
}
