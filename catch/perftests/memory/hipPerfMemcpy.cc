/*
 Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

/**
* @addtogroup hipMemcpy hipMemcpy
* @{
* @ingroup perfMemoryTest
* `hipMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind)` -
* Copies data between host and device.
*/

#include <hip_test_common.hh>

#define NUM_SIZE 8
#define NUM_ITER 0x40000

class hipPerfMemcpy {
 private:
  unsigned int numBuffers_;
  size_t totalSizes_[NUM_SIZE];
  void setHostBuffer(int *A, int val, size_t size);
 public:
  hipPerfMemcpy();
  ~hipPerfMemcpy() {}
  bool run(unsigned int numTests);
};

hipPerfMemcpy::hipPerfMemcpy() : numBuffers_(0) {
  for (int i = 0; i < NUM_SIZE; i++) {
    totalSizes_[i] = 1 << (i + 6);
  }
}

void hipPerfMemcpy::setHostBuffer(int *A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (int i = 0; i < len; i++) {
    A[i] = val;
  }
}

bool hipPerfMemcpy::run(unsigned int numTests) {
  int *A, *Ad;
  A = new int[totalSizes_[numTests]];
  setHostBuffer(A, 1, totalSizes_[numTests]);
  HIP_CHECK(hipMalloc(&Ad, totalSizes_[numTests]));

  // measure performance based on host time
  auto all_start = std::chrono::steady_clock::now();

  for (int j = 0; j < NUM_ITER; j++) {
    HIP_CHECK(hipMemcpy(Ad, A, totalSizes_[numTests], hipMemcpyHostToDevice));
  }

  HIP_CHECK(hipDeviceSynchronize());

  auto all_end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> diff = all_end - all_start;

  INFO("hipPerfMemcpy[" << numTests << "] " << "Host to Device copy took "
       << diff.count() / NUM_ITER << " sec for memory size of " <<
       totalSizes_[numTests] << " Bytes.");

  delete [] A;
  HIP_CHECK(hipFree(Ad));

  return true;
}

/**
* Test Description
* ------------------------
*  - Verify hipPerfMemcpy status.
* Test source
* ------------------------
*  - perftests/memory/hipPerfMemcpy.cc
* Test requirements
* ------------------------
*  - HIP_VERSION >= 5.6
*/

TEST_CASE("Perf_hipPerfMemcpy_test") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices <= 0) {
    SUCCEED("Skipped testcase hipPerfMemcpy as there is no device to test.");
  } else {
    int deviceId = 0;
    HIP_CHECK(hipSetDevice(deviceId));
    hipDeviceProp_t props = {0};
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    INFO("info: running on bus " << "0x" << props.pciBusID << " " <<
         props.name << " with " << props.multiProcessorCount << " CUs "
         << " and device id: " << deviceId);

    hipPerfMemcpy hipPerfMemcpy;
    for (auto testCase = 0; testCase < NUM_SIZE; testCase++) {
      REQUIRE(true == hipPerfMemcpy.run(testCase));
    }
  }
}
