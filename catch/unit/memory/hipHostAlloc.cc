
/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

static __global__ void write_integer(int* memory, int value) {
  if (memory) {
    *memory = value;
  }
}

int get_flags() {
  return GENERATE(hipHostMallocDefault,
                  hipHostMallocPortable,
                  hipHostMallocMapped,
                  hipHostMallocWriteCombined,
                  hipHostMallocPortable | hipHostMallocMapped,
                  hipHostMallocPortable | hipHostMallocWriteCombined,
                  hipHostMallocMapped | hipHostMallocWriteCombined,
                  hipHostMallocPortable | hipHostMallocMapped | hipHostMallocWriteCombined);
}

TEST_CASE("Unit_hipHostAlloc_Positive") {
  int* host_memory = nullptr;
  int flags = get_flags();

  HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&host_memory), sizeof(int), flags));

  REQUIRE(host_memory != nullptr);

  HIP_CHECK(hipFreeHost(host_memory));
}

TEST_CASE("Unit_hipHostAlloc_DataValidation") {
  int validation_number = 10;
  int* host_memory = nullptr;
  int* device_memory = nullptr;
  hipEvent_t event = nullptr;
  int flags = get_flags();

  HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&host_memory), sizeof(int), flags));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_memory), host_memory, 0));

  write_integer<<<1, 1>>>(device_memory, validation_number);

  SECTION("device sync") {
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("event sync") {
    HIP_CHECK(hipEventCreateWithFlags(&event, 0));
    HIP_CHECK(hipEventRecord(event, nullptr));
    HIP_CHECK(hipEventSynchronize(event));
  }

  SECTION("stream sync") {
    HIP_CHECK(hipStreamSynchronize(nullptr));
  }

  REQUIRE(*host_memory == validation_number);

  if (event != nullptr) {
    HIP_CHECK(hipEventDestroy(event));
  }

  HIP_CHECK(hipFreeHost(host_memory));
}

TEST_CASE("Unit_hipHostAlloc_Negative") {
  int* host_memory = nullptr;
  int flags = get_flags();

  SECTION("host memory is nullptr") {
    HIP_CHECK_ERROR(hipHostAlloc(nullptr, sizeof(int), flags), hipErrorInvalidValue);
  }

  SECTION("size is negative") {
    HIP_CHECK_ERROR(hipHostAlloc(reinterpret_cast<void**>(&host_memory), -1, flags),
                    hipErrorOutOfMemory);
  }

  SECTION("flag is out of range") {
    unsigned int flag = 999;
    HIP_CHECK_ERROR(hipHostAlloc(reinterpret_cast<void**>(&host_memory), sizeof(int), flag),
                    hipErrorInvalidValue);
  }
}
