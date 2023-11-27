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

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>


#include <limits>

static inline hipMemPool_t CreateMemPool(const int device) {
  hipMemPoolProps kPoolProps;
  kPoolProps.allocType = hipMemAllocationTypePinned;
  kPoolProps.handleTypes = hipMemHandleTypeNone;
  kPoolProps.location.type = hipMemLocationTypeDevice;
  kPoolProps.location.id = device;
  kPoolProps.win32SecurityAttributes = nullptr;
  memset(kPoolProps.reserved, 0, sizeof(kPoolProps.reserved));

  hipMemPool_t mem_pool;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));

  return mem_pool;
}

TEST_CASE("Unit_hipMallocFromPoolAsync_Basic_OneAlloc") {
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
  hipMemPool_t mem_pool = CreateMemPool(device_id);
  hipMemAccessDesc desc_list = {{hipMemLocationTypeDevice, 0}, hipMemAccessFlagsProtReadWrite};
  int count = 1;
  HIP_CHECK(hipMemPoolSetAccess(mem_pool, &desc_list, count));

  int* alloc_mem;
  StreamGuard stream(Streams::created);

  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem), allocation_size, mem_pool,
                                   stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  size_t used_mem = 0;
  hipMemPoolAttr attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &used_mem));
  REQUIRE(allocation_size == used_mem);

  const auto element_count = allocation_size / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 17;
  VectorSet<<<block_count, thread_count, 0>>>(alloc_mem, expected_value, element_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(host_alloc.host_ptr(), alloc_mem, allocation_size, hipMemcpyDeviceToHost));

  ArrayFindIfNot(host_alloc.host_ptr(), expected_value, element_count);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem), stream.stream()));

  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_before_sync));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_after_sync));
  // Sync must release memory to OS
  REQUIRE(res_after_sync <= res_before_sync);

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &used_mem));
  REQUIRE(0 == used_mem);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

TEST_CASE("Unit_hipMallocFromPoolAsync_Basic_TwoAllocs") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  const auto allocation_size1 = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto allocation_size2 = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);

  LinearAllocGuard<int> host_alloc1(LinearAllocs::hipHostMalloc, allocation_size1);
  LinearAllocGuard<int> host_alloc2(LinearAllocs::hipHostMalloc, allocation_size2);

  hipMemPool_t mem_pool = CreateMemPool(device_id);
  hipMemAccessDesc desc_list = {{hipMemLocationTypeDevice, 0}, hipMemAccessFlagsProtReadWrite};
  int count = 1;
  HIP_CHECK(hipMemPoolSetAccess(mem_pool, &desc_list, count));

  int* alloc_mem1;
  int* alloc_mem2;
  StreamGuard stream(Streams::created);

  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size1,
                                   mem_pool, stream.stream()));
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size2,
                                   mem_pool, stream.stream()));

  size_t used_mem = 0;
  hipMemPoolAttr attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &used_mem));
  // Make sure the current usage query works - both allocations are reported
  REQUIRE((allocation_size1 + allocation_size2) == used_mem);

  const auto element_count1 = allocation_size1 / sizeof(int);
  const auto element_count2 = allocation_size2 / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count1 = element_count1 / thread_count + 1;
  const auto block_count2 = element_count2 / thread_count + 1;
  constexpr int expected_value = 17;
  VectorSet<<<block_count1, thread_count, 0>>>(alloc_mem1, expected_value, element_count1);
  VectorSet<<<block_count2, thread_count, 0>>>(alloc_mem2, expected_value + 1, element_count2);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(host_alloc1.host_ptr(), alloc_mem1, allocation_size1, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(host_alloc2.host_ptr(), alloc_mem2, allocation_size2, hipMemcpyDeviceToHost));

  ArrayFindIfNot(host_alloc1.host_ptr(), expected_value, element_count1);
  ArrayFindIfNot(host_alloc2.host_ptr(), expected_value + 1, element_count2);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream.stream()));

  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_before_sync));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &res_after_sync));
  // Sync must release memory to OS
  REQUIRE(res_after_sync <= res_before_sync);

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &used_mem));
  // Make sure the current usage query works - just second buffer is left
  REQUIRE(allocation_size2 == used_mem);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &used_mem));
  // Make sure the high watermark usage works - both buffers must be reported
  REQUIRE((allocation_size1 + allocation_size2) == used_mem);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mem_pool, attr, &used_mem));
  // Make sure the current usage query works - none of the buffers are used
  REQUIRE(0 == used_mem);

  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}

TEST_CASE("Unit_hipMallocFromPoolAsync_Negative") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  void* p = nullptr;
  size_t max_size = std::numeric_limits<size_t>::max();
  size_t alloc_size = 1024;
  hipMemPool_t mem_pool = CreateMemPool(device_id);
  StreamGuard stream(Streams::created);

  SECTION("dev_ptr is nullptr") {
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(nullptr, alloc_size, mem_pool, stream.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Mempool not created") {
    hipMemPool_t dummy_mem_pool = nullptr;
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(static_cast<void**>(&p), alloc_size, dummy_mem_pool,
                                           stream.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("invalid stream handle") {
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(static_cast<void**>(&p), alloc_size, mem_pool,
                                           reinterpret_cast<hipStream_t>(-1)),
                    hipErrorInvalidHandle);
  }

  SECTION("Size is max size_t") {
    HIP_CHECK_ERROR(
        hipMallocFromPoolAsync(static_cast<void**>(&p), max_size, mem_pool, stream.stream()),
        hipErrorOutOfMemory);
  }

  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipMemPoolDestroy(mem_pool));
}
