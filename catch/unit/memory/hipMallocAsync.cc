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
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include <limits>

TEST_CASE("Unit_hipMallocAsync_Basic_OneAlloc") {
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
  MemPoolGuard mempool(MemPools::dev_default, device_id);

  int* alloc_mem;
  StreamGuard stream(Streams::created);

  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&alloc_mem), allocation_size, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  size_t used_mem = 0;
  hipMemPoolAttr attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  REQUIRE(allocation_size == used_mem);

  const auto element_count = allocation_size / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 17;
  VectorSet<<<block_count, thread_count, 0, stream.stream()>>>(alloc_mem, expected_value,
                                                               element_count);
  HIP_CHECK(hipGetLastError());

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

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  REQUIRE(0 == used_mem);

  ArrayFindIfNot(host_alloc.host_ptr(), expected_value, element_count);
}

TEST_CASE("Unit_hipMallocAsync_Basic_TwoAllocs") {
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
  MemPoolGuard mempool(MemPools::dev_default, device_id);

  int* alloc_mem1;
  int* alloc_mem2;
  StreamGuard stream(Streams::created);

  HIP_CHECK(
      hipMallocAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size, stream.stream()));
  HIP_CHECK(
      hipMallocAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t used_mem = 0;
  hipMemPoolAttr attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &used_mem));
  // Make sure the current usage query works - both allocations are reported
  REQUIRE((2 * allocation_size) == used_mem);

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

TEST_CASE("Unit_hipMallocAsync_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  int* p = nullptr;
  size_t max_size = std::numeric_limits<size_t>::max();
  size_t alloc_size = 1024;
  MemPoolGuard mempool(MemPools::dev_default, device_id);
  StreamGuard stream(Streams::created);

  SECTION("dev_ptr is nullptr") {
    HIP_CHECK_ERROR(hipMallocAsync(nullptr, alloc_size, stream.stream()), hipErrorInvalidValue);
  }

  SECTION("invalid stream handle") {
    HIP_CHECK_ERROR(
        hipMallocAsync(reinterpret_cast<void**>(&p), alloc_size, reinterpret_cast<hipStream_t>(-1)),
        hipErrorInvalidHandle);
  }

  SECTION("Size is max size_t") {
    HIP_CHECK_ERROR(hipMallocAsync(reinterpret_cast<void**>(&p), max_size, stream.stream()),
                    hipErrorOutOfMemory);
  }
}
