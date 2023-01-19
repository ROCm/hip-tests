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

class MemcpyDtoHAsyncBenchmark : public Benchmark<MemcpyDtoHAsyncBenchmark> {
 public:
  void operator()(LinearAllocs host_allocation_type, LinearAllocs device_allocation_type, size_t size) {
    const StreamGuard stream_guard(Streams::created);
    const hipStream_t stream = stream_guard.stream();

    LinearAllocGuard<int> device_allocation(device_allocation_type, size);
    LinearAllocGuard<int> host_allocation(host_allocation_type, size);

    TIMED_SECTION_STREAM(kTimerTypeEvent, stream) {
      HIP_CHECK(hipMemcpyDtoHAsync(host_allocation.ptr(),
                                   reinterpret_cast<hipDeviceptr_t>(device_allocation.ptr()),
                                   size, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(LinearAllocs host_allocation_type, LinearAllocs device_allocation_type, size_t size) {
  MemcpyDtoHAsyncBenchmark benchmark;
  std::stringstream section_name{};
  section_name << "size(" << size << ")";
  section_name << "/" << GetAllocationSectionName(host_allocation_type);
  benchmark.AddSectionName(section_name.str());
  benchmark.Run(host_allocation_type, device_allocation_type, size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemcpyDtoHAsync` from Device to Host:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - Source: device malloc
 *      - Destination: host pinned and pageable
 * Test source
 * ------------------------
 *  - unit/memcpy/hipMemcpyDtoHAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemcpyDtoHAsync") {
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto device_allocation_type = LinearAllocs::hipMalloc;
  const auto host_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  RunBenchmark(host_allocation_type, device_allocation_type, allocation_size);
}
