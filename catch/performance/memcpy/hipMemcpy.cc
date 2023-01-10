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
#include <utils.hh>

class MemcpyBenchmark : public Benchmark<MemcpyBenchmark> {
 public:
  void operator()(LinearAllocs dst_allocation_type, LinearAllocs src_allocation_type, size_t size, hipMemcpyKind kind) {
    LinearAllocGuard<int> src_allocation(src_allocation_type, size);
    LinearAllocGuard<int> dst_allocation(dst_allocation_type, size);

    TIMED_SECTION(TIMER_TYPE_EVENT) {
      HIP_CHECK(hipMemcpy(dst_allocation.ptr(), src_allocation.ptr(), size, kind));
    }
  }
};

static void RunBenchmark(LinearAllocs dst_allocation_type, LinearAllocs src_allocation_type, size_t size, hipMemcpyKind kind) {
  MemcpyBenchmark benchmark;
  benchmark.Configure(1000, 100);
  auto time = benchmark.Run(dst_allocation_type, src_allocation_type, size, kind);
  std::cout << time << " ms" << std::endl;
}

TEST_CASE("Performance_hipMemcpy_DeviceToHost") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);

  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToHost);
}

TEST_CASE("Performance_hipMemcpy_HostToDevice") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const auto dst_allocation_type = LinearAllocs::hipMalloc;

  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyHostToDevice);
}

TEST_CASE("Performance_hipMemcpy_HostToHost") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);
  const auto dst_allocation_type = GENERATE(LinearAllocs::malloc, LinearAllocs::hipHostMalloc);

  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyHostToHost);
}

TEST_CASE("Performance_hipMemcpy_DeviceToDevice") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  const auto allocation_size = GENERATE(4_KB, 4_MB, 16_MB);
  const auto src_allocation_type = LinearAllocs::hipMalloc;
  const auto dst_allocation_type = LinearAllocs::hipMalloc;

  RunBenchmark(dst_allocation_type, src_allocation_type, allocation_size, hipMemcpyDeviceToDevice);
}
