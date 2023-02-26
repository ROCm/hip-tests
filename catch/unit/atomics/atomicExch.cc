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

  old_vals[tid] = atomicExch(mem, static_cast<T>(tid) + 1);

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid == 0) global_mem[0] = mem[0];
  }
}

template <typename TestType, bool use_shared_mem>
void AtomicExchSameAddressTest(const dim3 blocks, const dim3 threads) {
  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_size = thread_count;
  const auto old_vals_alloc_size = old_vals_size * sizeof(TestType);
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(old_vals_size);

  LinearAllocGuard<TestType> mem_dev(LinearAllocs::hipMalloc, sizeof(TestType));
  TestType mem;

  HIP_CHECK(hipMemset(mem_dev.ptr(), 0, sizeof(TestType)));
  HIP_CHECK(hipMemset(old_vals_dev.ptr(), 0, old_vals_alloc_size));
  atomicExchKernel<TestType, use_shared_mem>
      <<<blocks, threads>>>(mem_dev.ptr(), old_vals_dev.ptr());
  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&mem, mem_dev.ptr(), sizeof(TestType), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(0 < mem);
  REQUIRE(old_vals_size >= mem);

  std::sort(old_vals.begin(), old_vals.end());
  if (old_vals.back() == old_vals.size() - 1) {
    for (auto i = 0u; i < old_vals_size; ++i) {
      REQUIRE(i == old_vals[i]);
    }
  } else if (old_vals.back() == old_vals.size()) {
    bool skipped = false;
    for (auto i = 0u; i < old_vals.size(); ++i) {
      if (!skipped) {
        skipped = !skipped & old_vals[i] == i + 1;
        REQUIRE(skipped | old_vals[i] == i);
      } else {
        REQUIRE(old_vals[i] == i + 1);
      }
    }
  } else {
    INFO("Largest old value should be equal to num_threads or num_threads - 1");
    REQUIRE(false);
  }
}

TEMPLATE_TEST_CASE("Unit_atomicExch_Positive_Basic_Same_Address", "", int, unsigned int,
                   unsigned long long, float) {
  const auto threads = GENERATE(dim3(1024));

  SECTION("Global memory") {
    const auto blocks = GENERATE(dim3(20));
    AtomicExchSameAddressTest<TestType, false>(blocks, threads);
  }

  SECTION("Shared memory") {
    const auto blocks = GENERATE(dim3(1));
    AtomicExchSameAddressTest<TestType, true>(blocks, threads);
  }
}