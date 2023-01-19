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

/**
 * @addtogroup memset memset
 * @{
 * @ingroup PerformanceTest
 */

class MemsetD8AsyncBenchmark : public Benchmark<MemsetD8AsyncBenchmark> {
 public:
  void operator()(LinearAllocs allocation_type, size_t size) {
    LinearAllocGuard<void> dst(allocation_type, size);
    StreamGuard stream(Streams::created);

    TIMED_SECTION_STREAM(kTimerTypeEvent, stream.stream()) {
      HIP_CHECK(hipMemsetD8Async(dst.ptr(), 17, size, stream.stream()));
    }
  }
};

static void RunBenchmark(LinearAllocs allocation_type, size_t size) {
  MemsetD8AsyncBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(size));
  benchmark.AddSectionName(GetAllocationSectionName(allocation_type));
  benchmark.Run(allocation_type, size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipMemsetD8Async`:
 *    -# Allocation size
 *      - Small: 4 KB
 *      - Medium: 4 MB
 *      - Large: 16 MB
 *    -# Allocation type
 *      - device
 *      - host
 *      - managed
 * Test source
 * ------------------------
 *  - performance/memset/hipMemsetD8Async.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipMemsetD8Async") {
  const auto size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto allocation_type = GENERATE(LinearAllocs::hipMalloc, LinearAllocs::hipHostMalloc,
                                        LinearAllocs::hipMallocManaged);
  RunBenchmark(allocation_type, size);
}
