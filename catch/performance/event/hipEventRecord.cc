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
 * @addtogroup event event
 * @{
 * @ingroup PerformanceTest
 */

class HipEventRecordBenchmark : public Benchmark<HipEventRecordBenchmark> {
 public:
  void operator()(hipStream_t stream) {
    hipEvent_t event;
    HIP_CHECK(hipEventCreate(&event));

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventRecord(event, stream)); }

    HIP_CHECK(hipEventDestroy(event));
  }
};


static void RunBenchmark(hipStream_t stream) {
  HipEventRecordBenchmark benchmark;
  if (stream == NULL) {
    benchmark.AddSectionName("Default stream");
  } else {
    benchmark.AddSectionName("Created stream");
  }
  benchmark.Run(stream);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventRecord`
 *    -# Executed both on
 *      - default stream
 *      - created stream
 * Test source
 * ------------------------
 *  - performance/event/hipEventRecord.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipEventRecord") {
  RunBenchmark(NULL);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  RunBenchmark(stream);
  HIP_CHECK(hipStreamDestroy(stream));
}
