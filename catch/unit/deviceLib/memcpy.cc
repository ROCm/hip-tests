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
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include <hip/hip_cooperative_groups.h>

template <typename T> using kernel_sig = void (*)(T*, T*, const size_t);

template <typename T>
__global__ void memcpy_at_once_kernel(T* dst, T* src, const size_t alloc_size) {
  memcpy(dst, src, alloc_size);
}

template <typename T>
__global__ void memcpy_one_by_one_kernel(T* dst, T* src, const size_t N) {
  const auto tid = cooperative_groups::this_grid().thread_rank();
  const auto stride = cooperative_groups::this_grid().size();

  for (auto i = tid; i < N; i += stride) {
    memcpy(dst + tid, src + tid, sizeof(T));
  }
}

template <typename T>
void DeviceMemcpyCommon(kernel_sig<T> memcpy_kernel) {
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto element_count = allocation_size / sizeof(T);

  LinearAllocGuard<T> input(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> src_allocation(LinearAllocs::hipMalloc, allocation_size);
  LinearAllocGuard<T> result(LinearAllocs::hipHostMalloc, allocation_size);
  LinearAllocGuard<T> dst_allocation(LinearAllocs::hipMalloc, allocation_size);

  for (auto i = 0; i < element_count; i++) {
    input.host_ptr()[i] = static_cast<T>(i);
  }

  HIP_CHECK(
      hipMemcpy(src_allocation.ptr(), input.host_ptr(), allocation_size, hipMemcpyHostToDevice));

  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;

  if (memcpy_kernel == memcpy_at_once_kernel<T>) {
    memcpy_at_once_kernel<T><<<1, 1>>>(dst_allocation.ptr(), src_allocation.ptr(), allocation_size);
  } else {
    memcpy_one_by_one_kernel<T><<<thread_count, block_count>>>(dst_allocation.ptr(), src_allocation.ptr(), element_count);
  }

  HIP_CHECK(
      hipMemcpy(result.host_ptr(), dst_allocation.ptr(), allocation_size, hipMemcpyDeviceToHost));

  ArrayMismatch(input.host_ptr(), result.host_ptr(), element_count);
}

TEMPLATE_TEST_CASE("Unit_Device_memcpy_Positive", "", int, unsigned int, long, unsigned long,
                   long long, unsigned long long, float, double) {

  SECTION("Memcpy whole buffer in one thread") {
    DeviceMemcpyCommon<TestType>(memcpy_at_once_kernel);
  }
  SECTION("Memcpy buffer in multiple threads/blocks") {
    DeviceMemcpyCommon<TestType>(memcpy_one_by_one_kernel);
  }
}
