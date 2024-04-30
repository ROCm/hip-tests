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
 * Contains performance tests for all hipEvent related HIP APIs.
 */

class HipEventCreateBenchmark : public Benchmark<HipEventCreateBenchmark> {
 public:
  void operator()() {
    hipEvent_t event;

    TIMED_SECTION(kTimerTypeCpu) { HIP_CHECK(hipEventCreate(&event)); }

    HIP_CHECK(hipEventDestroy(event));
  }
};

/**
 * Test Description
 * ------------------------
 *  - Executes `hipEventCreate`
 * Test source
 * ------------------------
 *  - performance/event/hipEventCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipEventCreate") {
  HipEventCreateBenchmark benchmark;
  benchmark.Run();
}

/**
* End doxygen group PerformanceTest.
* @}
*/
