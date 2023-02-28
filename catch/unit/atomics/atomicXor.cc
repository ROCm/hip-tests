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
#include "atomicXor_negative_kernels_rtc.hh"

template <typename T, bool shared = false>
__global__ void AtomicXor(T* const addr, const T val) {
  extern __shared__ char shmem[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  T* ptr = addr;

  if constexpr (shared) {
    ptr = reinterpret_cast<T*>(shmem);
    if (tid == 0) ptr[0] = addr[0];
    __syncthreads();
  }

  atomicXor(ptr, val);

  if constexpr (shared) {
    __syncthreads();
    if(tid == 0) addr[0] = ptr[0];
  }
}

template <typename T, bool shared = false>
__global__ void AtomicXorMultiDest(T* const addr, const T val, const int n) {
  extern __shared__ char shmem[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  T* ptr = addr;

  if constexpr (shared) {
    ptr = reinterpret_cast<T*>(shmem);
    if (tid < n) ptr[tid] = addr[tid];
    __syncthreads();
  }

  atomicXor(ptr + tid % n , val);

  if constexpr (shared) {
    __syncthreads();
    if (tid < n) addr[tid] = ptr[tid];
  }
}

TEMPLATE_TEST_CASE("Unit_atomicXor_Positive_SameAddress", "", int, unsigned int,
                   unsigned long, unsigned long long) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kMask = 0xAAAA;
  const TestType kInitValue = 0x4545;

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemcpy(alloc.ptr(), &kInitValue, kSize, hipMemcpyHostToDevice));

  int num_blocks, num_threads;

  SECTION("device memory") {
    num_blocks = 3, num_threads = 127;
    HipTest::launchKernel(AtomicXor<TestType, false>, num_blocks, num_threads, 0, nullptr,
                          alloc.ptr(), kMask);
  }

  SECTION("shared memory") {
    num_blocks = 1, num_threads = 255;
    HipTest::launchKernel(AtomicXor<TestType, true>, num_blocks, num_threads, kSize, nullptr,
                          alloc.ptr(), kMask);
  }

  TestType res;
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  auto expected_res = kInitValue ^ kMask;
  for (int i = 0; i < num_blocks * num_threads - 1; ++i) expected_res = expected_res ^ kMask;
  std::cout << kInitValue << ", " << res << ", " << expected_res << std::endl;
  REQUIRE(res == expected_res);
}

TEMPLATE_TEST_CASE("Unit_atomicXor_Positive_DifferentAddressSameWarp", "", int, unsigned int,
                   unsigned long, unsigned long long) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  int warp_size;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  const auto kSize = sizeof(TestType) * warp_size;
  constexpr TestType kMask = 0xAAAA;
  const TestType kInitValue = 0x4545;

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);
  TestType src[warp_size];
  for (int i = 0; i < warp_size; ++i) {
    src[i] = kInitValue;
  }
  HIP_CHECK(hipMemcpy(alloc.ptr(), src, kSize, hipMemcpyHostToDevice));

  int num_blocks, num_threads;

  SECTION("device memory") {
    num_blocks = 3, num_threads = 127;
    HipTest::launchKernel(AtomicXorMultiDest<TestType, false>, num_blocks, num_threads, 0, nullptr,
                          alloc.ptr(), kMask, warp_size);
  }

  SECTION("shared memory") {
    num_blocks = 1, num_threads = 255;
    HipTest::launchKernel(AtomicXorMultiDest<TestType, true>, num_blocks, num_threads, kSize, nullptr,
                          alloc.ptr(), kMask, warp_size);
  }

  TestType res[warp_size];
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  for (int i = 0; i < warp_size; ++i) {
    auto expected_res = kInitValue ^ kMask;
    for (int i = 0; i < num_blocks * num_threads - 1; ++i) expected_res = expected_res ^ kMask;
    REQUIRE(res[i] == expected_res);
  }
}

TEMPLATE_TEST_CASE("Unit_atomicXor_Positive_MultiKernel", "", int, unsigned int,
                   unsigned long, unsigned long long) {
  const auto allocation_type =
      GENERATE(LinearAllocs::hipHostMalloc, LinearAllocs::hipMalloc, LinearAllocs::hipMallocManaged,
               LinearAllocs::mallocAndRegister);

  constexpr auto kSize = sizeof(TestType);
  constexpr TestType kMask = 0xAAAA;
  const TestType kInitValue = 0x4545;

  LinearAllocGuard<TestType> alloc(allocation_type, kSize);

  HIP_CHECK(hipMemcpy(alloc.ptr(), &kInitValue, kSize, hipMemcpyHostToDevice));

  StreamGuard stream1(Streams::created);
  StreamGuard stream2(Streams::created);
  StreamGuard stream3(Streams::created);

  int num_blocks = 3, num_threads = 127;
  HipTest::launchKernel(AtomicXor<TestType, false>, num_blocks, num_threads, 0, stream1.stream(),
                        alloc.ptr(), kMask);
  HipTest::launchKernel(AtomicXor<TestType, false>, num_blocks, num_threads, 0, stream2.stream(),
                        alloc.ptr(), kMask);
  HipTest::launchKernel(AtomicXor<TestType, false>, num_blocks, num_threads, 0, stream3.stream(),
                        alloc.ptr(), kMask);

  TestType res;
  HIP_CHECK(hipMemcpy(&res, alloc.ptr(), kSize, hipMemcpyDeviceToHost));

  auto expected_res = kInitValue ^ kMask ;
  for (int i = 0; i < 3 * num_blocks * num_threads - 1; ++i) expected_res = expected_res ^ kMask;
  REQUIRE(res == expected_res);
}

TEST_CASE("Unit_atomicXor_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source =
    GENERATE(kAtomicXor_int, kAtomicXor_uint, kAtomicXor_ulong, kAtomicXor_ulonglong);
  HIPRTC_CHECK(hiprtcCreateProgram(&program, program_source, "atomicXor_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};
  // Please check the content of negative_kernels_rtc.hh
  int expected_error_count{10};
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
