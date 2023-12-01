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


#include "mempool_common.hh"

#include <resource_guards.hh>
#include <utils.hh>

TEST_CASE("Unit_hipMemPoolTrimTo_Negative_Parameter") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  size_t trim_size = 1024;

  SECTION("Passing nullptr to mem_pool") {
    HIP_CHECK_ERROR(hipMemPoolTrimTo(nullptr, trim_size), hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipMemPoolTrimTo_Positive_Basic") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  const size_t allocation_size1 = kPageSize * kPageSize * 2;
  const size_t allocation_size2 = kPageSize / 2;
  MemPoolGuard mempool(MemPools::created, device_id);

  int* alloc_mem1;
  int* alloc_mem2;
  StreamGuard stream(Streams::created);

  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size1,
                                   mempool.mempool(), stream.stream()));
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size2,
                                   mempool.mempool(), stream.stream()));

  int blocks = 2;
  int clk_rate;
  if (IsGfx11()) {
    HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeWallClockRate, 0));
    kernel_500ms_gfx11<<<32, blocks, 0, stream.stream()>>>(alloc_mem1, clk_rate);
  } else {
    HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));

    kernel_500ms<<<32, blocks, 0, stream.stream()>>>(alloc_mem1, clk_rate);
  }

  hipMemPoolAttr attr;
  attr = hipMemPoolAttrReleaseThreshold;
  // The pool must hold 128MB
  std::uint64_t threshold = 128 * 1024 * 1024;
  HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &threshold));

  // Not a real free, since kernel isn't done
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream.stream()));

  // Get reserved memory before trim
  attr = hipMemPoolAttrReservedMemCurrent;
  std::uint64_t res_before_trim = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_before_trim));

  size_t min_bytes_to_hold = allocation_size2;
  HIP_CHECK(hipMemPoolTrimTo(mempool.mempool(), min_bytes_to_hold));

  std::uint64_t res_after_trim = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_trim));
  // Trim must be a nop because execution isn't done
  REQUIRE(res_before_trim == res_after_trim);

  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t res_after_sync = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_sync));
  // Since hipMemPoolAttrReleaseThreshold is 128 MB sync does nothing to the freed memory
  REQUIRE(res_after_trim == res_after_sync);

  HIP_CHECK(hipMemPoolTrimTo(mempool.mempool(), min_bytes_to_hold));

  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &res_after_trim));
  // Validate memory after real trim. The pool must hold less memory than before
  REQUIRE(res_after_trim < res_after_sync);

  attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value64 = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the threshold query works
  REQUIRE(threshold == value64);

  attr = hipMemPoolAttrUsedMemCurrent;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the current usage query works - just small buffer left
  REQUIRE(allocation_size2 == value64);

  attr = hipMemPoolAttrUsedMemHigh;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  // Make sure the high watermark usage works - the both buffers must be reported
  REQUIRE((allocation_size1 + allocation_size2) == value64);

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream.stream()));
}
