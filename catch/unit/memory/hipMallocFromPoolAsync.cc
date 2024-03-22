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

#include <limits>

/**
 * @addtogroup hipMallocFromPoolAsync hipMallocFromPoolAsync
 * @{
 * @ingroup StreamOTest
 * `hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream)`
 * - Allocates memory from a specified pool with stream ordered semantics
 */


/**
 * Test Description
 * ------------------------
 *  - Basic test to verify proper allocation and stream ordering of hipMallocFromPoolAsync when one
 * memory allocation is performed.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMallocFromPoolAsync_Basic_OneAlloc") {
  MallocMemPoolAsync_OneAlloc(
      [](void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
        return hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream);
      },
      MemPools::created);
}


/**
 * Test Description
 * ------------------------
 *  - Basic test to verify proper allocation and stream ordering of hipMallocFromPoolAsync when two
 * memory allocations are performed.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMallocFromPoolAsync_Basic_TwoAllocs") {
  MallocMemPoolAsync_TwoAllocs(
      [](void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
        return hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream);
      },
      MemPools::created);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that memory allocated with hipMallocFromPoolAsync can be properly reused.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMallocFromPoolAsync_Basic_Reuse") {
  MallocMemPoolAsync_Reuse(
      [](void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream) {
        return hipMallocFromPoolAsync(dev_ptr, size, mem_pool, stream);
      },
      MemPools::created);
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMallocFromPoolAsync behavior with invalid arguments:
 *    -# Nullptr dev_ptr
 *    -# Nullptr mem_pool
 *    -# Invalid stream handle
 *    -# Size is max size_t
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMallocFromPoolAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMallocFromPoolAsync_Negative_Parameters") {
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
  MemPoolGuard mempool(MemPools::created, device_id);
  StreamGuard stream(Streams::created);

  SECTION("dev_ptr is nullptr") {
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(nullptr, alloc_size, mempool.mempool(), stream.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Mempool not created") {
    hipMemPool_t dummy_mem_pool = nullptr;
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(static_cast<void**>(&p), alloc_size, dummy_mem_pool,
                                           stream.stream()),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid stream handle") {
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(static_cast<void**>(&p), alloc_size, mempool.mempool(),
                                           reinterpret_cast<hipStream_t>(-1)),
                    hipErrorInvalidHandle);
  }

  SECTION("Size is max size_t") {
    HIP_CHECK_ERROR(hipMallocFromPoolAsync(static_cast<void**>(&p), max_size, mempool.mempool(),
                                           stream.stream()),
                    hipErrorOutOfMemory);
  }
}

/**
* End doxygen group StreamOTest.
* @}
*/
