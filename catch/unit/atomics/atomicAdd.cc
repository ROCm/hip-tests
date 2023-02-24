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

#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include "negative_kernels_rtc.hh"

template <typename T, bool shared = false> __global__ void AtomicAdd(T* const addr, const T val) {
  extern __shared__ char shmem[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  T* ptr = addr;

  if constexpr (shared) {
    ptr = reinterpret_cast<T*>(shmem);
    if (tid == 0) ptr[0] = addr[0];
    __syncthreads();
  }

  atomicAdd(ptr, val);

  if constexpr (shared) {
    __syncthreads();
    if (tid == 0) addr[0] = ptr[0];
  }
}

template <typename T, bool shared = false>
__global__ void AtomicAddMultiDest(T* const addr, const T val, const int n) {
  extern __shared__ char shmem[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  T* ptr = addr;

  if constexpr (shared) {
    ptr = reinterpret_cast<T*>(shmem);
    if (tid < n) ptr[tid] = addr[tid];
    __syncthreads();
  }

  atomicAdd(ptr + tid % n, val);

  if constexpr (shared) {
    __syncthreads();
    if (tid < n) addr[tid] = ptr[tid];
  }
}

TEMPLATE_TEST_CASE("Unit_atomicAdd_Positive_SameAddress", "", int, unsigned int, unsigned long,
                   unsigned long long, float, double) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 3.125 : 3;

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemset(alloc.ptr(), 0, kSize));

  int num_blocks, num_threads;

  SECTION("device memory") {
    num_blocks = 3, num_threads = 128;
    HipTest::launchKernel(AtomicAdd<TestType, false>, num_blocks, num_threads, 0, nullptr,
                          alloc.ptr(), kValue);
  }

  SECTION("shared memory") {
    num_blocks = 1, num_threads = 256;
    HipTest::launchKernel(AtomicAdd<TestType, true>, num_blocks, num_threads, kSize, nullptr,
                          alloc.ptr(), kValue);
  }

  TestType res;
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  const auto expected_res = num_blocks * num_threads * kValue;
  REQUIRE(res == expected_res);
}

TEMPLATE_TEST_CASE("Unit_atomicAdd_Positive_DifferentAddressSameWarp", "", int, unsigned int,
                   unsigned long, unsigned long long, float, double) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  int warp_size;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  const auto kSize = sizeof(TestType) * warp_size;
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 3.125 : 3;

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemset(alloc.ptr(), 0, kSize));

  int num_blocks, num_threads;

  SECTION("device memory") {
    num_blocks = 3, num_threads = 128;
    HipTest::launchKernel(AtomicAddMultiDest<TestType, false>, num_blocks, num_threads, 0, nullptr,
                          alloc.ptr(), kValue, warp_size);
  }

  SECTION("shared memory") {
    num_blocks = 1, num_threads = 256;
    HipTest::launchKernel(AtomicAddMultiDest<TestType, true>, num_blocks, num_threads, kSize,
                          nullptr, alloc.ptr(), kValue, warp_size);
  }

  TestType res[warp_size];
  HIP_CHECK(hipMemcpy(res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  const auto expected_res = num_blocks * num_threads / warp_size * kValue;
  for (int i = 0; i < warp_size; ++i) REQUIRE(res[i] == expected_res);
}

TEMPLATE_TEST_CASE("Unit_atomicAdd_Positive_MultiKernel", "", int, unsigned int, unsigned long,
                   unsigned long long, float, double) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 3.125 : 3;

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemset(alloc.ptr(), 0, kSize));

  StreamGuard stream1(Streams::created);
  StreamGuard stream2(Streams::created);

  int num_blocks = 3, num_threads = 128;
  HipTest::launchKernel(AtomicAdd<TestType, false>, num_blocks, num_threads, 0, stream1.stream(),
                        alloc.ptr(), kValue);
  HipTest::launchKernel(AtomicAdd<TestType, false>, num_blocks, num_threads, 0, stream2.stream(),
                        alloc.ptr(), kValue);

  TestType res;
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  const auto expected_res = num_blocks * num_threads * kValue * 2;
  REQUIRE(res == expected_res);
}

void ControlFlowTestRef(int* array, int n, int num_blocks, int num_threads) {
  for (int idx = 0; idx < num_blocks * num_threads; ++idx) {
    int val = idx % num_threads;
    bool condition1 = (val % 2 == 0);
    bool condition2 = (val % 4 == 0);
    bool condition3 = (val % 8 == 0);

    if (condition1 && condition2) {
      array[0] += val;
    } else if (condition3) {
      int temp = array[idx % n];
      array[idx % n] += val * 2;
      if (temp > 10) {
        array[idx % n] -= val;
      }
    } else {
      for (int i = 0; i < val; ++i) {
        array[idx % n] += i;
      }
    }
  }
}

__device__ void ControlFlowTest(int* array, int n, int idx) {
  int val = threadIdx.x;
  bool condition1 = (val % 2 == 0);
  bool condition2 = (val % 4 == 0);
  bool condition3 = (val % 8 == 0);

  if (condition1 && condition2) {
    atomicAdd(&array[0], val);
  } else if (condition3) {
    int temp = atomicAdd(&array[idx % n], val * 2);
    if (temp > 10) {
      atomicAdd(&array[idx % n], -val);
    }
  } else {
    for (int i = 0; i < val; ++i) {
      atomicAdd(&array[idx % n], i);
    }
  }
}

__global__ void ControlFlowTestKernel(int* array, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ControlFlowTest(array, n, idx);
}

TEST_CASE("Unit_atomicAdd_Positive_ControlFlow") {
  constexpr auto N = 2;
  constexpr auto kSize = sizeof(int) * N;

  LinearAllocGuard<int> alloc(LinearAllocs::hipMalloc, kSize);
  HIP_CHECK(hipMemset(alloc.ptr(), 0, kSize));

  int num_blocks = 2, num_threads = 64;
  HipTest::launchKernel(ControlFlowTestKernel, num_blocks, num_threads, 0, nullptr, alloc.ptr(), N);

  int res[N];
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  int expected_res[N];
  memset(expected_res, 0, kSize);

  ControlFlowTestRef(expected_res, N, num_blocks, num_threads);

  for (int i = 0; i < N; ++i) {
    REQUIRE(res[i] == expected_res[i]);
  }
}

TEST_CASE("Unit_atomicAdd_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source =
    GENERATE(kAtomicAdd_int, kAtomicAdd_uint, kAtomicAdd_ulong,
             kAtomicAdd_ulonglong, kAtomicAdd_float, kAtomicAdd_double);
  HIPRTC_CHECK(hiprtcCreateProgram(&program, program_source, "atomicAdd_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};
  // Please check the content of negative_kernels_rtc.hh
  int expected_error_count{8};
  std::string error_message{"error:"};

  size_t n_pos = log.find(error_message, 0);
  while(n_pos != std::string::npos) {
    ++error_count;
    n_pos = log.find(error_message, n_pos + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_error_count);
}
