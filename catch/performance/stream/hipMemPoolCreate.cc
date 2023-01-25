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

#include "stream_performance_common.hh"

class MemPoolCreateBenchmark : public Benchmark<MemPoolCreateBenchmark> {
 public:
  void operator()() {
    hipMemPool_t mem_pool{nullptr};

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));
    }

    REQUIRE(mem_pool != nullptr);
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark() {
  MemPoolCreateBenchmark benchmark;
  benchmark.Run();
}

TEST_CASE("Performance_hipMemPoolCreate") {
  if (!AreMemPoolsSupported(0)) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeMemoryPoolsSupported "
                           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  RunBenchmark();
}
