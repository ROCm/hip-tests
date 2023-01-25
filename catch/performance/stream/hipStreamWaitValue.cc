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

class StreamWaitValue32Benchmark : public Benchmark<StreamWaitValue32Benchmark> {
 public:
  void operator()(const size_t array_size, unsigned int flag) {
    const StreamGuard stream_guard{Streams::created};
    const hipStream_t stream = stream_guard.stream();
    uint32_t* value_ptr;
    uint32_t value{0};
    if (flag == hipStreamWaitValueAnd) {
      value = 1;
    }
    HIP_CHECK(hipMalloc(&value_ptr, sizeof(uint32_t) * array_size));
    HIP_CHECK(hipMemset(value_ptr, value, sizeof(uint32_t) * array_size));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipStreamWaitValue32(stream, value_ptr, value, flag));
    }
    HIP_CHECK(hipFree(value_ptr));
  }
};

class StreamWaitValue64Benchmark : public Benchmark<StreamWaitValue64Benchmark> {
 public:
  void operator()(const size_t array_size, unsigned int flag) {
    const StreamGuard stream_guard{Streams::created};
    const hipStream_t stream = stream_guard.stream();
    uint64_t* value_ptr;
    uint64_t value{0};
    if (flag == hipStreamWaitValueAnd) {
      value = 1;
    }
    HIP_CHECK(hipMalloc(&value_ptr, sizeof(uint64_t) * array_size));
    HIP_CHECK(hipMemset(value_ptr, value, sizeof(uint64_t) * array_size));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipStreamWaitValue64(stream, value_ptr, value, flag));
    }
    HIP_CHECK(hipFree(value_ptr));
  }
};

template <typename WaitValueBenchmark>
static void RunBenchmark(const size_t array_size, unsigned int flag) {
  WaitValueBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(array_size));
  benchmark.AddSectionName(GetFlagWaitSectionName(flag));
  benchmark.Run(array_size, flag);
}

TEST_CASE("Performance_hipStreamWaitValue32") {
  if (!IsStreamWaitValueSupported(0)) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeCanUseStreamWaitValue "
                           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  unsigned int flag = GENERATE(hipStreamWaitValueGte, hipStreamWaitValueEq,
                               hipStreamWaitValueAnd, hipStreamWaitValueNor);
  RunBenchmark<StreamWaitValue32Benchmark>(array_size, flag);
}

TEST_CASE("Performance_hipStreamWaitValue64") {
  if (!IsStreamWaitValueSupported(0)) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeCanUseStreamWaitValue "
                           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  unsigned int flag = GENERATE(hipStreamWaitValueGte, hipStreamWaitValueEq,
                               hipStreamWaitValueAnd, hipStreamWaitValueNor);
  RunBenchmark<StreamWaitValue64Benchmark>(array_size, flag);
}
