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

/**
 * @addtogroup stream stream
 * @{
 * @ingroup PerformanceTest
 */

class StreamWriteValue32Benchmark : public Benchmark<StreamWriteValue32Benchmark> {
 public:
  void operator()(const size_t array_size) {
    const StreamGuard stream_guard{Streams::created};
    const hipStream_t stream = stream_guard.stream();
    uint32_t* value_ptr;
    uint32_t value{0};
    HIP_CHECK(hipMalloc(&value_ptr, sizeof(uint32_t) * array_size));
    HIP_CHECK(hipMemset(value_ptr, value, sizeof(uint32_t) * array_size));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipStreamWriteValue32(stream, value_ptr, value, 0));
    }
    HIP_CHECK(hipFree(value_ptr));
  }
};

class StreamWriteValue64Benchmark : public Benchmark<StreamWriteValue64Benchmark> {
 public:
  void operator()(const size_t array_size) {
    const StreamGuard stream_guard{Streams::created};
    const hipStream_t stream = stream_guard.stream();
    uint64_t* value_ptr;
    uint64_t value{0};
    HIP_CHECK(hipMalloc(&value_ptr, sizeof(uint64_t) * array_size));
    HIP_CHECK(hipMemset(value_ptr, value, sizeof(uint64_t) * array_size));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipStreamWriteValue64(stream, value_ptr, value, 0));
    }
    HIP_CHECK(hipFree(value_ptr));
  }
};

template <typename WriteValueBenchmark>
static void RunBenchmark(const size_t array_size) {
  WriteValueBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(array_size));
  benchmark.Run(array_size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamWriteValue32`:
 *    -# Allocation size:
 *      - 4 KB
 *      - 4 MB
 *      - 16 MB
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamWriteValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipStreamWriteValue32") {
  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark<StreamWriteValue32Benchmark>(array_size);
}

/**
 * Test Description
 * ------------------------
 *  - Executes `hipStreamWriteValue64`:
 *    -# Allocation size:
 *      - 4 KB
 *      - 4 MB
 *      - 16 MB
 * Test source
 * ------------------------
 *  - performance/stream/hipStreamWriteValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipStreamWriteValue64") {
  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark<StreamWriteValue64Benchmark>(array_size);
}
