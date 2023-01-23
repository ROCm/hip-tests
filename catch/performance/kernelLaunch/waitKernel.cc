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

#include "kernel_launch_perf_common.hh"

/**
 * @addtogroup kernelLaunch kernel launch
 * @{
 * @ingroup PerformanceTest
 */

class WaitKernelLaunchBenchmark : public Benchmark<WaitKernelLaunchBenchmark> {
 public:
  void operator()(int cycles) {
    TIMED_SECTION(kTimerTypeCpu) {
      waitKernel<<<1, 1>>>(cycles);
      HIP_CHECK(hipDeviceSynchronize());
    }
  }
};

static void RunBenchmark(int cycles, float wait_time_in_ms) {
  WaitKernelLaunchBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(wait_time_in_ms));
  auto result = benchmark.Run(cycles);
  std::cout << "\tDriver overhead: " << std::get<0>(result) - wait_time_in_ms << " ms" << std::endl;
}

/**
 * Test Description
 * ------------------------
 *  - Calls wait kernel with triple chevron annotation:
 *    -# Waiting for number of device ticks:
 *      - Small: 10 * wall clock rate
 *      - Medium: 50 * wall clock rate
 *      - Large: 100 * wall clock rate
 * Test source
 * ------------------------
 *  - performance/kernelLaunch/waitKernel.cc
 * Test requirements
 * ------------------------
 *  - Device supports wall clock rate
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_WaitKernel") {
  int wall_clock_rate{0}; //in kilohertz
  HIP_CHECK(hipDeviceGetAttribute(&wall_clock_rate, hipDeviceAttributeWallClockRate, 0));
  if (!wall_clock_rate) {
    HipTest::HIP_SKIP_TEST("hipDeviceAttributeWallClockRate has not been supported. Skipping.");
    return;
  }
  int cycles = GENERATE_COPY(10 * wall_clock_rate, 50 * wall_clock_rate, 100 * wall_clock_rate);
  float miliseconds = 1.f * cycles / wall_clock_rate;
  RunBenchmark(cycles, miliseconds);
}
