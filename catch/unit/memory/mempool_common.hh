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
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */
#pragma once

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

namespace {
constexpr auto wait_ms = 500;
}  // anonymous namespace

template <typename T> __global__ void kernel_500ms(T* host_res, int clk_rate) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  host_res[tid] = tid + 1;
  __threadfence_system();
  // expecting that the data is getting flushed to host here!
  uint64_t start = clock64() / clk_rate, cur;
  if (clk_rate > 1) {
    do {
      cur = clock64() / clk_rate - start;
    } while (cur < wait_ms);
  } else {
    do {
      cur = clock64() / start;
    } while (cur < wait_ms);
  }
}

template <typename T> __global__ void kernel_500ms_gfx11(T* host_res, int clk_rate) {
#if HT_AMD
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  host_res[tid] = tid + 1;
  __threadfence_system();
  // expecting that the data is getting flushed to host here!
  uint64_t start = wall_clock64() / clk_rate, cur;
  if (clk_rate > 1) {
    do {
      cur = wall_clock64() / clk_rate - start;
    } while (cur < wait_ms);
  } else {
    do {
      cur = wall_clock64() / start;
    } while (cur < wait_ms);
  }
#endif
}

template <typename F> void MallocMemPoolAsync_OneAlloc(F malloc_func, const MemPools mempool_type) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);
  MemPoolGuard mempool(mempool_type, device_id);

  int* alloc_mem;
  StreamGuard stream(Streams::created);

  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem), allocation_size, mempool.mempool(),
                        stream.stream()));

  int blocks = 16;
  int clk_rate;
  hipMemPoolAttr attr;
  if (IsGfx11()) {
    HIP_CHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeWallClockRate, 0));
    kernel_500ms_gfx11<<<32, blocks, 0, stream.stream()>>>(alloc_mem, clk_rate);
  } else {
    HIP_CHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeClockRate, 0));

    kernel_500ms<<<32, blocks, 0, stream.stream()>>>(alloc_mem, clk_rate);
  }

  const auto element_count = allocation_size / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 17;
  VectorSet<<<block_count, thread_count, 0, stream.stream()>>>(alloc_mem, expected_value,
                                                               element_count);

  HIP_CHECK(hipMemcpyAsync(host_alloc.host_ptr(), alloc_mem, allocation_size, hipMemcpyDeviceToHost,
                           stream.stream()));

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem), stream.stream()));

  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_before_sync));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_sync));
  // Sync must release memory to OS
  REQUIRE(res_after_sync <= res_before_sync);

  std::uint64_t used_mem = 10;
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  REQUIRE(0 == used_mem);

  ArrayFindIfNot(host_alloc.host_ptr(), expected_value, element_count);
}

template <typename F>
void MallocMemPoolAsync_TwoAllocs(F malloc_func, const MemPools mempool_type) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);
  MemPoolGuard mempool(mempool_type, device_id);

  int* alloc_mem1;
  int* alloc_mem2;
  StreamGuard stream(Streams::created);

  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem1), allocation_size, mempool.mempool(),
                        stream.stream()));
  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem2), allocation_size, mempool.mempool(),
                        stream.stream()));

  int blocks = 16;
  int clk_rate;
  hipMemPoolAttr attr;
  if (IsGfx11()) {
    HIP_CHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeWallClockRate, 0));
    kernel_500ms_gfx11<<<32, blocks, 0, stream.stream()>>>(alloc_mem1, clk_rate);
  } else {
    HIP_CHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeClockRate, 0));

    kernel_500ms<<<32, blocks, 0, stream.stream()>>>(alloc_mem1, clk_rate);
  }

  const auto element_count = allocation_size / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 17;
  VectorSet<<<block_count, thread_count, 0, stream.stream()>>>(alloc_mem1, expected_value,
                                                               element_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpyAsync(alloc_mem2, alloc_mem1, allocation_size, hipMemcpyDeviceToDevice,
                           stream.stream()));

  HIP_CHECK(hipMemcpyAsync(host_alloc.host_ptr(), alloc_mem2, allocation_size,
                           hipMemcpyDeviceToHost, stream.stream()));

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream.stream()));

  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_before_sync));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_sync));
  // Sync must release memory to OS
  REQUIRE(res_after_sync <= res_before_sync);

  std::uint64_t used_mem = 0;
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  // Make sure the current usage query works - just second buffer is left
  REQUIRE(allocation_size == used_mem);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  // Make sure the high watermark usage works - both buffers must be reported
  REQUIRE((2 * allocation_size) == used_mem);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  // Make sure the current usage query works - none of the buffers are used
  REQUIRE(0 == used_mem);

  ArrayFindIfNot(host_alloc.host_ptr(), expected_value, element_count);
}

template <typename F> void MallocMemPoolAsync_Reuse(F malloc_func, const MemPools mempool_type) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  MemPoolGuard mempool(mempool_type, device_id);

  int *alloc_mem1, *alloc_mem2, *alloc_mem3;
  StreamGuard stream(Streams::created);

  size_t allocation_size1 = kPageSize * kPageSize * 2;
  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem1), allocation_size1, mempool.mempool(),
                        stream.stream()));

  size_t allocation_size2 = kPageSize;
  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem3), allocation_size2, mempool.mempool(),
                        stream.stream()));

  int blocks = 2;
  int clk_rate;

  if (IsGfx11()) {
    HIP_CHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeWallClockRate, 0));
    kernel_500ms_gfx11<<<32, blocks, 0, stream.stream()>>>(alloc_mem1, clk_rate);
  } else {
    HIP_CHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeClockRate, 0));

    kernel_500ms<<<32, blocks, 0, stream.stream()>>>(alloc_mem1, clk_rate);
  }

  hipMemPoolAttr attr;
  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream.stream()));

  HIP_CHECK(malloc_func(reinterpret_cast<void**>(&alloc_mem2), allocation_size1, mempool.mempool(),
                        stream.stream()));
  // Runtime must reuse the pointer
  REQUIRE(alloc_mem1 == alloc_mem2);

  // Make a sync before the second kernel launch to make sure memory B isn't gone
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  // Second kernel launch with new memory
  if (IsGfx11()) {
    kernel_500ms_gfx11<<<32, blocks, 0, stream.stream()>>>(alloc_mem2, clk_rate);
  } else {
    kernel_500ms<<<32, blocks, 0, stream.stream()>>>(alloc_mem2, clk_rate);
  }

  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  attr = hipMemPoolAttrUsedMemCurrent;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the current usage reports the both buffers
  REQUIRE((allocation_size1 + allocation_size2) == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE((allocation_size1 + allocation_size2) == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream.stream()));
  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the current usage reports just one buffer, because the above free doesn't hold memory
  REQUIRE(allocation_size2 == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem3), stream.stream()));
}
