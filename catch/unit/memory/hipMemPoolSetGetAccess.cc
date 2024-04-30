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

/**
 * @addtogroup hipMemPoolSetAccess hipMemPoolSetAccess
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list, size_t count)`
 * - Controls visibility of the specified pool between devices
 */

__global__ void copyP2PAndScale(int* dst, const int* src, size_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    // scale & store src vector.
    dst[idx] = 2 * src[idx];
  }
}

static void MemPoolSetGetAccess(const MemPools mempool_type, int src_device, int dst_device,
                                hipMemAccessFlags access_flags) {
  MemPoolGuard mempool(mempool_type, src_device);

  hipMemAccessDesc desc;
  memset(&desc, 0, sizeof(hipMemAccessDesc));
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = dst_device;
  desc.flags = access_flags;
  HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), &desc, 1));

  hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
  HIP_CHECK(hipMemPoolGetAccess(&flags, mempool.mempool(), &desc.location));
  REQUIRE(flags == access_flags);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify hipMemPoolSetAccess/hipMemPoolGetAccess on a single device.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolSetGetAccess_Positive_Basic") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));

  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, device));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  const auto mempool_type = GENERATE(MemPools::dev_default, MemPools::created);

  MemPoolSetGetAccess(mempool_type, device, device, hipMemAccessFlagsProtReadWrite);
}

int CheckP2PMemPoolSupport(int src_device, int dst_device) {
  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, src_device));
  if (mem_pool_support) {
    HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported,
                                    dst_device));
  }
  return mem_pool_support;
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify hipMemPoolSetAccess/hipMemPoolGetAccess on multiple devices.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolSetGetAccess_Positive_MultipleGPU") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }
  const auto src_device = GENERATE(range(0, HipTest::getDeviceCount()));
  const auto dst_device = GENERATE(range(0, HipTest::getDeviceCount()));
  INFO("Src device: " << src_device << ", Dst device: " << dst_device);

  int mem_pool_support = CheckP2PMemPoolSupport(src_device, dst_device);
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  const auto mempool_type = GENERATE(MemPools::dev_default, MemPools::created);
  const auto access_flag = GENERATE(hipMemAccessFlagsProtNone, hipMemAccessFlagsProtRead,
                                    hipMemAccessFlagsProtReadWrite);

  int can_access_peer = 0;
  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    MemPoolSetGetAccess(mempool_type, src_device, dst_device, access_flag);
  }
}

void MemPoolSetGetAccess_P2P(const MemPools mempool_type) {
  const auto src_device = GENERATE(range(0, HipTest::getDeviceCount()));
  const auto dst_device = GENERATE(range(0, HipTest::getDeviceCount()));
  INFO("Src device: " << src_device << ", Dst device: " << dst_device);

  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);

  int mem_pool_support = CheckP2PMemPoolSupport(src_device, dst_device);
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  int *alloc_mem1, *alloc_mem2;
  int can_access_peer = 0;
  HIP_CHECK(hipSetDevice(src_device));
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, src_device, dst_device));
  if (can_access_peer) {
    hipEvent_t waitOnStream1;

    LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, allocation_size);
    HIP_CHECK(hipEventCreate(&waitOnStream1))
    StreamGuard stream1(Streams::withFlags, hipStreamNonBlocking);
    // Get/create mempool for src_device
    MemPoolGuard mempool(mempool_type, src_device);

    // Allocate memory in a stream from the pool set above
    if (mempool_type == MemPools::dev_default) {
      HIP_CHECK(
          hipMallocAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size, stream1.stream()));
    } else {
      HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                       mempool.mempool(), stream1.stream()));
    }

    const auto element_count = allocation_size / sizeof(int);
    constexpr auto thread_count = 1024;
    const auto block_count = element_count / thread_count + 1;
    constexpr int expected_value = 15;
    VectorSet<<<block_count, thread_count, 0, stream1.stream()>>>(alloc_mem1, expected_value,
                                                                  element_count);
    HIP_CHECK(hipEventRecord(waitOnStream1, stream1.stream()));

    HIP_CHECK(hipSetDevice(dst_device));
    StreamGuard stream2(Streams::withFlags, hipStreamNonBlocking);

    // Allocate memory in dst device
    HIP_CHECK(
        hipMallocAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size, stream2.stream()));

    // Setup peer mappings for dst device
    hipMemAccessDesc desc;
    memset(&desc, 0, sizeof(hipMemAccessDesc));
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = dst_device;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolSetAccess(mempool.mempool(), &desc, 1));

    hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
    HIP_CHECK(hipMemPoolGetAccess(&flags, mempool.mempool(), &desc.location));
    REQUIRE(flags == hipMemAccessFlagsProtReadWrite);

    HIP_CHECK(hipStreamWaitEvent(stream2.stream(), waitOnStream1, 0));
    copyP2PAndScale<<<block_count, thread_count, 0, stream2.stream()>>>(alloc_mem2, alloc_mem1,
                                                                        element_count);

    HIP_CHECK(hipMemcpyAsync(host_alloc.host_ptr(), alloc_mem2, allocation_size,
                             hipMemcpyDeviceToHost, stream2.stream()));
    HIP_CHECK(hipFreeAsync(alloc_mem1, stream2.stream()));
    HIP_CHECK(hipFreeAsync(alloc_mem2, stream2.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    ArrayFindIfNot(host_alloc.host_ptr(), 2 * expected_value, element_count);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify peer-to-peer access of stream ordered memory with hipMemPoolSetAccess.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolSetGetAccess_Positive_P2P") {
  const auto device_count = HipTest::getDeviceCount();
  if (device_count < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }

  SECTION("Default MemPool") { MemPoolSetGetAccess_P2P(MemPools::dev_default); }

  SECTION("Created MemPool") { MemPoolSetGetAccess_P2P(MemPools::created); }
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolSetAccess behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *    -# Desc is nullptr and count is > 0
 *    -# Count > num_device
 *    -# Invalid desc location type
 *    -# Invalid desc location id
 *    -# Revoking access to own memory pool
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolSetAccess_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  MemPoolGuard mempool(MemPools::dev_default, device_id);

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  hipMemAccessDesc desc;
  memset(&desc, 0, sizeof(hipMemAccessDesc));
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = device_id;
  desc.flags = hipMemAccessFlagsProtReadWrite;

  SECTION("Mempool is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolSetAccess(nullptr, &desc, 1), hipErrorInvalidValue);
  }
#if HT_AMD
  SECTION("Desc is nullptr and count is > 0") {
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), nullptr, 1), hipErrorInvalidValue);
  }
#endif
  SECTION("Count > num_device") {
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, (num_dev + 1)),
                    hipErrorInvalidDevice);
  }

  SECTION("Passing invalid desc location type") {
    desc.location.type = hipMemLocationTypeInvalid;
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, 1), hipErrorInvalidValue);
    desc.location.type = hipMemLocationTypeDevice;
  }

  SECTION("Passing invalid desc location id") {
    desc.location.id = num_dev;
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, 1), hipErrorInvalidDevice);
    desc.location.id = device_id;
  }

  SECTION("Revoking access to own memory pool") {
    desc.flags = hipMemAccessFlagsProtNone;
    HIP_CHECK_ERROR(hipMemPoolSetAccess(mempool.mempool(), &desc, 1), hipErrorInvalidDevice);
    desc.flags = hipMemAccessFlagsProtReadWrite;
  }
}

/**
 * End doxygen group hipMemPoolSetAccess.
 * @}
 */

/**
 * @addtogroup hipMemPoolGetAccess hipMemPoolGetAccess
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool, hipMemLocation* location)`
 * - Returns the accessibility of a pool from a device
 */

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolGetAccess behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *    -# Flags is nullptr
 *    -# Invalid location type
 *    -# Invalid location id
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAccess.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolGetAccess_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  MemPoolGuard mempool(MemPools::dev_default, device_id);

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  hipMemAccessFlags flags = hipMemAccessFlagsProtNone;
  hipMemLocation location = {hipMemLocationTypeDevice, device_id};

  SECTION("Mempool is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolGetAccess(&flags, nullptr, &location), hipErrorInvalidValue);
  }
#if HT_AMD
  SECTION("Flags is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolGetAccess(nullptr, mempool.mempool(), &location),
                    hipErrorInvalidValue);
  }
#endif
  SECTION("Passing invalid location type") {
    location.type = hipMemLocationTypeInvalid;
    HIP_CHECK_ERROR(hipMemPoolGetAccess(&flags, mempool.mempool(), &location),
                    hipErrorInvalidValue);
    location.type = hipMemLocationTypeDevice;
  }

  SECTION("Passing invalid location id") {
    location.id = num_dev;
    HIP_CHECK_ERROR(hipMemPoolGetAccess(&flags, mempool.mempool(), &location),
                    hipErrorInvalidValue);
    location.id = device_id;
  }
}
