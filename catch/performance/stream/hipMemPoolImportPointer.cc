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

class MemPoolImportPointerBenchmark : public Benchmark<MemPoolImportPointerBenchmark> {
 public:
  void operator()(const size_t array_size) {
    float* device_ptr{nullptr};
    float* device_ptr_import{nullptr};
    hipMemPool_t mem_pool{nullptr};
    hipMemPoolPtrExportData exp_data;

    hipMemPoolProps props = CreateMemPoolProps(0, hipMemHandleTypePosixFileDescriptor);
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &props));
    HIP_CHECK(hipMallocFromPoolAsync(&device_ptr, array_size * sizeof(float), mem_pool, nullptr));
    HIP_CHECK(hipStreamSynchronize(nullptr));
    HIP_CHECK(hipMemPoolExportPointer(&exp_data, device_ptr));

    TIMED_SECTION(kTimerTypeCpu) {
      HIP_CHECK(hipMemPoolImportPointer(reinterpret_cast<void**>(device_ptr_import), mem_pool, &exp_data));
    }

    HIP_CHECK(hipFree(device_ptr));
    HIP_CHECK(hipFree(device_ptr_import));
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark(const size_t array_size) {
  MemPoolImportPointerBenchmark benchmark;
  benchmark.AddSectionName(std::to_string(array_size));
  benchmark.Run(array_size);
}

TEST_CASE("Performance_hipMemPoolImportPointer") {
  if (!AreMemPoolsSupported(0)) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeMemoryPoolsSupported "
                           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  size_t array_size = GENERATE(4_KB, 4_MB, 16_MB);
  RunBenchmark(array_size);
}
