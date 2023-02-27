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
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T, bool use_shared_mem>
__global__ void atomicExchKernel(T* const global_mem, T* const old_vals) {
  __shared__ T shared_mem;

  const auto tid = cg::this_grid().thread_rank();

  T* const mem = use_shared_mem ? &shared_mem : global_mem;

  if constexpr (use_shared_mem) {
    if (tid == 0) mem[0] = global_mem[0];
    __syncthreads();
  }

  old_vals[tid] = atomicExch(mem, static_cast<T>(tid + 1));

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid == 0) global_mem[0] = mem[0];
  }
}

template <typename TestType, bool use_shared_mem>
void AtomicExchSameAddressTest(const dim3 blocks, const dim3 threads,
                               const LinearAllocs alloc_type) {
  LinearAllocGuard<TestType> mem_dev(alloc_type, sizeof(TestType));

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType);
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(thread_count + 1);


  HIP_CHECK(hipMemset(mem_dev.ptr(), 0, sizeof(TestType)));
  atomicExchKernel<TestType, use_shared_mem>
      <<<blocks, threads>>>(mem_dev.ptr(), old_vals_dev.ptr());
  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(old_vals.data() + thread_count, mem_dev.ptr(), sizeof(TestType),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  std::sort(old_vals.begin(), old_vals.end());
  for (auto i = 0u; i < old_vals.size(); ++i) {
    REQUIRE(i == old_vals[i]);
  }
}

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Basic_Same_Address", "", int, unsigned int,
                   unsigned long long, float) {
  const auto threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(31));

  SECTION("Global memory") {
    const auto blocks = GENERATE(dim3(20));
    using LA = LinearAllocs;
    const auto allocation_type =
        GENERATE(LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister);
    AtomicExchSameAddressTest<TestType, false>(blocks, threads, allocation_type);
  }

  SECTION("Shared memory") {
    const auto blocks = GENERATE(dim3(1));
    AtomicExchSameAddressTest<TestType, true>(blocks, threads, LinearAllocs::hipMalloc);
  }
}

template <typename T, bool use_shared_mem>
__global__ void atomicExchMultiDestKernel(T* const global_mem, T* const old_vals,
                                          const unsigned int width) {
  extern __shared__ uint8_t shared_mem[];

  const auto tid = cg::this_grid().thread_rank();

  T* const mem = use_shared_mem ? reinterpret_cast<T*>(shared_mem) : global_mem;

  if constexpr (use_shared_mem) {
    if (tid < width) mem[tid] = global_mem[tid];
    __syncthreads();
  }

  old_vals[tid] = atomicExch(mem + tid % width, static_cast<T>(tid + width));

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid < width) global_mem[tid] = mem[tid];
  }
}

template <typename TestType, bool use_shared_mem>
void AtomicExchMultiDestTest(const dim3 blocks, const dim3 threads, const LinearAllocs alloc_type,
                             const unsigned int width) {
  const auto mem_alloc_size = width * sizeof(TestType);
  LinearAllocGuard<TestType> mem_dev(alloc_type, mem_alloc_size);

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType);
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(thread_count + width);
  std::iota(old_vals.begin(), old_vals.begin() + width, 0);

  HIP_CHECK(hipMemcpy(mem_dev.ptr(), old_vals.data(), mem_alloc_size, hipMemcpyHostToDevice));
  const auto shared_mem_size = use_shared_mem ? mem_alloc_size : 0u;
  atomicExchMultiDestKernel<TestType, use_shared_mem>
      <<<blocks, threads, shared_mem_size>>>(mem_dev.ptr(), old_vals_dev.ptr(), width);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(old_vals.data() + thread_count, mem_dev.ptr(), mem_alloc_size,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Every thread will exchange its grid-wide linear id into a target location within mem_dev,
  // receiving back the value previously present therein. This previous value is written to
  // old_vals_dev.
  // old_vals_dev will not contain values that the final scheduled warp exchanged into mem_dev, but
  // mem_dev obviously will.
  // Given that mem_dev initially contains values in the range [0, width) and that the maximum value
  // the final thread shall write is thread_count + width - 1, presuming correct operation of
  // atomicExch, the union of mem_dev and old_vals_dev shall contain values in the range
  //[0, thread_count + width)
  std::sort(old_vals.begin(), old_vals.end());
  for (auto i = 0u; i < old_vals.size(); ++i) {
    REQUIRE(i == old_vals[i]);
  }
}

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Basic_Different_Address_Same_Warp", "", int,
                   unsigned int, unsigned long long, float) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  const auto threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(31));

  SECTION("Global memory") {
    const auto blocks = GENERATE(dim3(40));
    using LA = LinearAllocs;
    const auto allocation_type =
        GENERATE(LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister);
    AtomicExchMultiDestTest<TestType, false>(blocks, threads, LinearAllocs::hipMalloc, warp_size);
  }

  SECTION("Shared memory") {
    const auto blocks = GENERATE(dim3(1));
    AtomicExchMultiDestTest<TestType, true>(blocks, threads, LinearAllocs::hipMalloc, warp_size);
  }
}