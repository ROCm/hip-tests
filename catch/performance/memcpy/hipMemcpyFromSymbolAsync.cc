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

__device__ int devSymbol[1_MB];

class MemcpyFromSymbolAsyncBenchmark : public Benchmark<MemcpyFromSymbolAsyncBenchmark> {
 public:
  void operator()(const void* source, void* result, size_t size=1, size_t offset=0) {
    const StreamGuard stream_guard(Streams::created);
    const hipStream_t stream = stream_guard.stream();

    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(devSymbol), source, size, offset,
              hipMemcpyHostToDevice, stream));
    TIMED_SECTION(TIMER_TYPE_EVENT) {
      HIP_CHECK(hipMemcpyFromSymbolAsync(result, HIP_SYMBOL(devSymbol), size, offset,
                hipMemcpyDeviceToHost, stream));
    }

    HIP_CHECK(hipStreamSynchronize(stream));
  }
};

static void RunBenchmark(const void* source, void* result, size_t size=1, size_t offset=0) {
  MemcpyFromSymbolAsyncBenchmark benchmark;
  benchmark.Configure(100, 1000);
  auto time = benchmark.Run(source, result, size, offset);
  std::cout << time << " ms" << std::endl;
}

TEST_CASE("Performance_hipMemcpyFromSymbolAsync_SingularValue") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  int set{42};
  int result{0};
  RunBenchmark(&set, &result);
}

TEST_CASE("Performance_hipMemcpyFromSymbolAsync_ArrayValue") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  size_t size = GENERATE(1_KB, 4_KB, 512_KB);
  int array[size];
  std::fill_n(array, size, 42);
  int result[size];
  std::fill_n(result, size, 0);

  RunBenchmark(array, result, sizeof(int) * size);
}

TEST_CASE("Performance_hipMemcpyFromSymbolAsync_WithOffset") {
  std::cout << Catch::getResultCapture().getCurrentTestName() << std::endl;
  size_t size = GENERATE(1_KB, 4_KB, 512_KB);
  int array[size];
  std::fill_n(array, size, 42);
  int result[size];
  std::fill_n(result, size, 0);

  size_t offset = GENERATE_REF(0, size / 2);
  RunBenchmark(array + offset, result + offset, sizeof(int) * (size - offset), offset * sizeof(int));
}
