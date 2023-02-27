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
#include "atomicMax_negative_kernels_rtc.hh"

template <typename T, bool shared = false>
__global__ void AtomicMax(T* const addr, const T val) {
  extern __shared__ char shmem[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  T* ptr = addr;

  if constexpr (shared) {
    ptr = reinterpret_cast<T*>(shmem);
    if (tid == 0) ptr[0] = addr[0];
    __syncthreads();
  }

  atomicMax(ptr, val + tid);

  if constexpr (shared) {
    __syncthreads();
    if(tid == 0) addr[0] = ptr[0];
  }
}

template <typename T, bool shared = false>
__global__ void AtomicMaxMultiDest(T* const addr, const T val, const int n) {
  extern __shared__ char shmem[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  T* ptr = addr;

  if constexpr (shared) {
    ptr = reinterpret_cast<T*>(shmem);
    if (tid < n) ptr[tid] = addr[tid];
    __syncthreads();
  }

  atomicMax(ptr + tid % n , val + tid % n);

  if constexpr (shared) {
    __syncthreads();
    if (tid < n) addr[tid] = ptr[tid];
  }
}

TEMPLATE_TEST_CASE("Unit_atomicMax_Positive_SameAddress", "", int, unsigned int,
                   unsigned long long) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 5.5f : 5;
  const TestType kInitValue = GENERATE(kValue - 2, kValue + 2);

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemset(alloc.ptr(), 0, kSize));
  HIP_CHECK(hipMemset(alloc.ptr(), kInitValue, 1));

  int num_blocks, num_threads;

  SECTION("device memory") {
    num_blocks = 3, num_threads = 128;
    HipTest::launchKernel(AtomicMax<TestType, false>, num_blocks, num_threads, 0, nullptr,
                          alloc.ptr(), kValue + num_blocks * num_threads - 1);
  }

  SECTION("shared memory") {
    num_blocks = 1, num_threads = 256;
    HipTest::launchKernel(AtomicMax<TestType, true>, num_blocks, num_threads, kSize, nullptr,
                          alloc.ptr(), kValue);
  }

  TestType res;
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  REQUIRE(res == kValue + num_blocks * num_threads - 1);
}

TEMPLATE_TEST_CASE("Unit_atomicMax_Positive_DifferentAddressSameWarp", "", int, unsigned int,
                   unsigned long long) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  int warp_size;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  const auto kSize = sizeof(TestType) * warp_size;
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 5.5f : 5;
  const TestType kInitValue = GENERATE_REF(warp_size / 2 - 2, warp_size / 2 + 2);

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);
  TestType src[warp_size];
  for (int i = 0; i < warp_size; ++i) {
    src[i] = kInitValue;
  }
  HIP_CHECK(hipMemcpy(alloc.ptr(), src, kSize, hipMemcpyHostToDevice));

  int num_blocks, num_threads;

  SECTION("device memory") {
    num_blocks = 3, num_threads = 128;
    HipTest::launchKernel(AtomicMinMultiDest<TestType, false>, num_blocks, num_threads, 0, nullptr,
                          alloc.ptr(), kValue + warp_size - 1, warp_size);
  }

  SECTION("shared memory") {
    num_blocks = 1, num_threads = 256;
    HipTest::launchKernel(AtomicMaxMultiDest<TestType, true>, num_blocks, num_threads, kSize, nullptr,
                          alloc.ptr(), kValue, warp_size);
  }

  TestType res[warp_size];
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  for (int i = 0; i < warp_size; ++i) {
    const auto expected_res = std::max(kInitValue, kValue + i);
    REQUIRE(res[i] == expected_res);
  }
}

TEMPLATE_TEST_CASE("Unit_atomicMax_Positive_MultiKernel", "", int, unsigned int,
                   unsigned long long) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kValue = std::is_floating_point_v<TestType> ? 5.5f : 5;
  const TestType kInitValue = GENERATE(kValue - 2, kValue + 2);

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemset(alloc.ptr(), 0, kSize));
  HIP_CHECK(hipMemset(alloc.ptr(), kInitValue, 1));

  StreamGuard stream1(Streams::created);
  StreamGuard stream2(Streams::created);

  int num_blocks = 3, num_threads = 128;
  HipTest::launchKernel(AtomicMax<TestType, false>, num_blocks, num_threads, 0, stream1.stream(),
                        alloc.ptr(), kValue);
  HipTest::launchKernel(AtomicMax<TestType, false>, num_blocks, num_threads, 0, stream2.stream(),
                        alloc.ptr(), kValue);

  TestType res;
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  REQUIRE(res == kValue + num_blocks * num_threads - 1);
}

void ControlFlowTestRefMax(int* array, int n, int num_blocks, int num_threads) {
  for (int idx = 0; idx < num_blocks * num_threads; ++idx) {
    int val = idx % num_threads;
    bool condition1 = (val % 2 == 0);
    bool condition2 = (val % 4 == 0);
    bool condition3 = (val % 8 == 0);

    if (condition1 && condition2) {
      array[0] = std::max(val, array[0]);
    } else if (condition3) {
      array[idx % n] = std::max(array[idx % n], val / 2);
      if (array[idx % n] > 10) {
        array[idx % n] = std::max(array[idx % n], 30);
      }
    } else {
      for (int i = 0; i < val; ++i) {
        array[idx % n] = std::max(array[idx % n], val - i);
      }
    }
  }
}

__device__ void ControlFlowTestMax(int* array, int n, int idx) {
  int val = threadIdx.x;
  bool condition1 = (val % 2 == 0);
  bool condition2 = (val % 4 == 0);
  bool condition3 = (val % 8 == 0);

  if (condition1 && condition2) {
    atomicMax(&array[0], val);
  } else if (condition3) {
    atomicMax(&array[idx % n], val / 2);
    if (array[idx % n] > 10) {
      atomicMax(&array[idx % n], 30);
    }
  } else {
    for (int i = 0; i < val; ++i) {
      atomicMax(&array[idx % n], val - i);
    }
  }
}

__global__ void ControlFlowTestKernelMax(int* array, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ControlFlowTestMax(array, n, idx);
}

TEST_CASE("Unit_atomicMax_Positive_ControlFlow") {
  constexpr auto N = 30;
  constexpr auto kSize = sizeof(int) * N;

  LinearAllocGuard<int> alloc(LinearAllocs::hipMalloc, kSize);
  int expected_res[N];
  for (int i = 0; i < N; ++i) {
    expected_res[i] = 23;
  }
  HIP_CHECK(hipMemcpy(alloc.ptr(), expected_res, kSize, hipMemcpyHostToDevice));

  int num_blocks = 2, num_threads = 64;
  HipTest::launchKernel(ControlFlowTestKernelMax, num_blocks, num_threads, 0, nullptr, alloc.ptr(), N);

  int res[N];
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  ControlFlowTestRefMax(expected_res, N, num_blocks, num_threads);

  for (int i = 0; i < N; ++i) {
    REQUIRE(res[i] == expected_res[i]);
  }
}

TEST_CASE("Unit_atomicMax_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source =
    GENERATE(kAtomicMax_int, kAtomicMax_uint, kAtomicMax_ulong,
             kAtomicMax_ulonglong, kAtomicMax_float, kAtomicMax_double);
  HIPRTC_CHECK(hiprtcCreateProgram(&program, program_source, "atomicMax_negative.cc", 0, nullptr, nullptr));
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
