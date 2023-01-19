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

class MemcpyDtoDBenchmark : public Benchmark<MemcpyDtoDBenchmark> {
 public:
  void operator()(size_t size, bool enable_peer_access) {
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

    LinearAllocGuard<int> src_allocation(LinearAllocs::hipMalloc, size);
    HIP_CHECK(hipSetDevice(dst_device));
    LinearAllocGuard<int> dst_allocation(LinearAllocs::hipMalloc, size);

    HIP_CHECK(hipSetDevice(src_device));
    TIMED_SECTION(kTimerTypeEvent) {
      HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst_allocation.ptr()),
                              reinterpret_cast<hipDeviceptr_t>(src_allocation.ptr()),
                              size));
    }
  }
};

static void RunBenchmark(size_t size, bool enable_peer_access=false) {
  MemcpyDtoDBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.Run(size, enable_peer_access);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyDtoD` from Device to Device with peer access enabled:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: device malloc
 * Test source
 * ------------------------
 *  - unit/memcpy/hipMemcpyDtoD.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - Device supports Peer-to-Peer access
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyDtoD_PeerAccessEnabled") {
  if (HipTest::getDeviceCount() < 2) {
    HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.");
    return;
  }
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(allocation_size, true);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyDtoD` from Device to Device with peer access disabled:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: device malloc
 * Test source
 * ------------------------
 *  - unit/memcpy/hipMemcpyDtoD.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyDtoD_PeerAccessDisabled") {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(allocation_size);
}
