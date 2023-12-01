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

/**
 * @addtogroup hipMemPoolSetAttribute hipMemPoolSetAttribute
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value)`
 * - Sets attributes of a memory pool
 */

template <typename T>
static void MemPoolSetGetAttribute(const hipMemPool_t mempool, const hipMemPoolAttr attr,
                                   T& set_value) {
  T get_value = 100;
  HIP_CHECK(hipMemPoolSetAttribute(mempool, attr, &set_value));
  HIP_CHECK(hipMemPoolGetAttribute(mempool, attr, &get_value));
  REQUIRE(get_value == set_value);
}


/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that default attribute values are correct.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolSetGetAttribute_Positive_Default") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));

  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, device));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  const auto mempool_type = GENERATE(MemPools::dev_default, MemPools::created);
  MemPoolGuard mempool(mempool_type, device);

  const auto attr_type =
      GENERATE(hipMemPoolReuseFollowEventDependencies, hipMemPoolReuseAllowOpportunistic,
               hipMemPoolReuseAllowInternalDependencies);

  // Check default value
  int def_value = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr_type, &def_value));
  REQUIRE(def_value == 1);

  // Check if attribute can be disabled
  int set_value = 0;
  MemPoolSetGetAttribute(mempool.mempool(), attr_type, set_value);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify hipMemPoolSetAttribute/hipMemPoolGetAttribute functionality.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolSetGetAttribute_Positive_MemBasic") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));

  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, device));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  const auto mempool_type = GENERATE(MemPools::dev_default, MemPools::created);
  MemPoolGuard mempool(mempool_type, device);

  // Check hipMemPoolAttrReleaseThreshold default value
  hipMemPoolAttr attr = hipMemPoolAttrReleaseThreshold;
  std::uint64_t value64 = 100;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), attr, &value64));
  REQUIRE(value64 == 0);

  // Check setting hipMemPoolAttrReleaseThreshold to a value
  std::uint64_t set_value64 = kPageSize;
  MemPoolSetGetAttribute(mempool.mempool(), hipMemPoolAttrReleaseThreshold, set_value64);

  // Check reset of hipMemPoolAttrReservedMemHigh and hipMemPoolAttrUsedMemHigh
  set_value64 = 0;
  MemPoolSetGetAttribute(mempool.mempool(), hipMemPoolAttrReservedMemHigh, set_value64);
  MemPoolSetGetAttribute(mempool.mempool(), hipMemPoolAttrUsedMemHigh, set_value64);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify correct behavior of the Opportunistic attribute.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolSetAttribute_Opportunistic") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  MemPoolGuard mempool(MemPools::created, device_id);

  hipMemPoolAttr attr;
  int blocks = 2;
  int clk_rate;
  if (IsGfx11()) {
    HIPCHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeWallClockRate, 0));
  } else {
    HIPCHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeClockRate, 0));
  }

  int *alloc_mem1, *alloc_mem2, *alloc_mem3;

  // Create 2 async non-blocking streams
  StreamGuard stream1(Streams::withFlags, hipStreamNonBlocking);
  StreamGuard stream2(Streams::withFlags, hipStreamNonBlocking);

  size_t allocation_size = kPageSize;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem3), allocation_size,
                                   mempool.mempool(), stream1.stream()));
  int value = 0;

  SECTION("Disallow Opportunistic - No Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    // Disable all default pool states
    attr = hipMemPoolReuseFollowEventDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));
    attr = hipMemPoolReuseAllowOpportunistic;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));
    attr = hipMemPoolReuseAllowInternalDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel for 500 ms in the first stream
    if (IsGfx11()) {
      kernel_500ms_gfx11<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    } else {
      kernel_500ms<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    }

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));

    // Sleep for 1 second GPU should be idle by now
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream2.stream()));
    // Without Opportunistic state runtime must allocate another buffer
    REQUIRE(alloc_mem1 != alloc_mem2);

    // Run kernel with the new memory in the second stream
    if (IsGfx11()) {
      kernel_500ms_gfx11<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, clk_rate);
    } else {
      kernel_500ms<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, clk_rate);
    }

    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream2.stream()));
  }

  SECTION("Disallow Opportunistic - Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    // Disable all default pool states
    attr = hipMemPoolReuseFollowEventDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));
    attr = hipMemPoolReuseAllowOpportunistic;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));
    attr = hipMemPoolReuseAllowInternalDependencies;
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel for 500 ms in the first stream
    if (IsGfx11()) {
      kernel_500ms_gfx11<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    } else {
      kernel_500ms<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    }

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));

    // Sleep for 1 second GPU should be idle by now
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream1.stream()));
    // Without Opportunistic state runtime must allocate another buffer
    REQUIRE(alloc_mem1 == alloc_mem2);

    // Run kernel with the new memory in the second stream
    if (IsGfx11()) {
      kernel_500ms_gfx11<<<32, blocks, 0, stream1.stream()>>>(alloc_mem2, clk_rate);
    } else {
      kernel_500ms<<<32, blocks, 0, stream1.stream()>>>(alloc_mem2, clk_rate);
    }

    HIP_CHECK(hipStreamSynchronize(stream1.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream1.stream()));
  }

  SECTION("Allow Opportunistic - Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    value = 1;
    attr = hipMemPoolReuseAllowOpportunistic;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel for 500 ms in the first stream
    if (IsGfx11()) {
      HIP_CHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeWallClockRate, 0));
      kernel_500ms_gfx11<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    } else {
      HIP_CHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeClockRate, 0));
      kernel_500ms<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    }

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));

    // Sleep for 1 second GPU should be idle by now
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream2.stream()));
    // With Opportunistic state runtime will reuse freed buffer A
    REQUIRE(alloc_mem1 == alloc_mem2);

    // Run kernel with the new memory in the second stream
    if (IsGfx11()) {
      kernel_500ms_gfx11<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, clk_rate);
    } else {
      kernel_500ms<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, clk_rate);
    }

    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream2.stream()));
  }

  SECTION("Allow Opportunistic - No Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    value = 1;
    attr = hipMemPoolReuseAllowOpportunistic;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel for 500 ms in the first stream

    if (IsGfx11()) {
      kernel_500ms_gfx11<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    } else {
      kernel_500ms<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    }

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream2.stream()));
    // With Opportunistic state runtime can't reuse freed buffer A, because it's still busy with the
    // kernel
    REQUIRE(alloc_mem1 != alloc_mem2);

    // Run kernel with the new memory in the second stream
    if (IsGfx11()) {
      kernel_500ms_gfx11<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, clk_rate);
    } else {
      kernel_500ms<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, clk_rate);
    }

    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream2.stream()));
  }

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem3), stream1.stream()));
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify correct behavior of the EventDependencies attribute.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolSetAttribute_EventDependencies") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  MemPoolGuard mempool(MemPools::created, device_id);

  hipMemPoolAttr attr;
  int blocks = 2;
  int clk_rate;
  if (IsGfx11()) {
    HIPCHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeWallClockRate, 0));
  } else {
    HIPCHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeClockRate, 0));
  }

  int *alloc_mem1, *alloc_mem2, *alloc_mem3;

  // Create 2 async non-blocking streams
  StreamGuard stream1(Streams::withFlags, hipStreamNonBlocking);
  StreamGuard stream2(Streams::withFlags, hipStreamNonBlocking);

  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  size_t allocation_size = kPageSize;
  HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem3), allocation_size,
                                   mempool.mempool(), stream1.stream()));
  int value = 0;

  SECTION("Allow Event Dependencies - Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    value = 1;
    attr = hipMemPoolReuseFollowEventDependencies;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel for 500 ms in the first stream
    if (IsGfx11()) {
      HIP_CHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeWallClockRate, 0));
      kernel_500ms_gfx11<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    } else {
      HIP_CHECK(hipDeviceGetAttribute(&clk_rate, hipDeviceAttributeClockRate, 0));
      kernel_500ms<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    }

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));

    HIP_CHECK(hipEventRecord(event, stream1.stream()));
    HIP_CHECK(hipStreamWaitEvent(stream2.stream(), event, 0));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream2.stream()));
    // With Opportunistic state runtime will reuse freed buffer A
    REQUIRE(alloc_mem1 == alloc_mem2);

    // Run kernel with the new memory in the second stream
    if (IsGfx11()) {
      kernel_500ms_gfx11<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, clk_rate);
    } else {
      kernel_500ms<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, clk_rate);
    }

    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream2.stream()));
  }

  SECTION("Disallow Event Dependencies - No Reuse") {
    allocation_size = kPageSize * kPageSize * 2;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem1), allocation_size,
                                     mempool.mempool(), stream1.stream()));

    value = 0;
    attr = hipMemPoolReuseFollowEventDependencies;
    // Enable Opportunistic
    HIP_CHECK(hipMemPoolSetAttribute(mempool.mempool(), attr, &value));

    // Run kernel for 500 ms in the first stream

    if (IsGfx11()) {
      kernel_500ms_gfx11<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    } else {
      kernel_500ms<<<32, blocks, 0, stream1.stream()>>>(alloc_mem1, clk_rate);
    }

    // Not a real free, since kernel isn't done
    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem1), stream1.stream()));

    HIP_CHECK(hipEventRecord(event, stream1.stream()));
    HIP_CHECK(hipStreamWaitEvent(stream2.stream(), event, 0));

    // Allocate memory for the second stream
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&alloc_mem2), allocation_size,
                                     mempool.mempool(), stream2.stream()));
    // With Opportunistic state runtime can't reuse freed buffer A, because it's still busy with the
    // kernel
    REQUIRE(alloc_mem1 != alloc_mem2);

    // Run kernel with the new memory in the second stream
    if (IsGfx11()) {
      kernel_500ms_gfx11<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, clk_rate);
    } else {
      kernel_500ms<<<32, blocks, 0, stream2.stream()>>>(alloc_mem2, clk_rate);
    }

    HIP_CHECK(hipStreamSynchronize(stream1.stream()));
    HIP_CHECK(hipStreamSynchronize(stream2.stream()));

    HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem2), stream2.stream()));
  }

  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(alloc_mem3), stream1.stream()));
  HIP_CHECK(hipEventDestroy(event));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolSetAttribute behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *    -# Attribute value is not valid
 *    -# Nullptr value
 *    -# hipMemPoolAttrReservedMemHigh set to non-zero
 *    -# IhipMemPoolAttrUsedMemHigh set to non-zero
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolSetAttribute_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  MemPoolGuard mempool(MemPools::dev_default, device_id);

  hipMemPoolAttr attr = hipMemPoolReuseFollowEventDependencies;
  int set_value = 0;
  std::uint64_t set_value64 = 0;

  SECTION("Mempool is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolSetAttribute(nullptr, attr, &set_value), hipErrorInvalidValue);
  }

  SECTION("Attribute value is not valid") {
    HIP_CHECK_ERROR(
        hipMemPoolSetAttribute(mempool.mempool(), static_cast<hipMemPoolAttr>(0x9), &set_value),
        hipErrorInvalidValue);
  }
#if HT_AMD
  SECTION("Set values is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolSetAttribute(mempool.mempool(), attr, nullptr), hipErrorInvalidValue);
  }
#endif

  SECTION("Set hipMemPoolAttrReservedMemHigh to non-zero") {
    hipMemPoolAttr attr = hipMemPoolAttrReservedMemHigh;
    set_value64 = 1;
    HIP_CHECK_ERROR((hipMemPoolSetAttribute(mempool.mempool(), attr, &set_value64)),
                    hipErrorInvalidValue);
  }

  SECTION("Set hipMemPoolAttrUsedMemHigh to non-zero") {
    hipMemPoolAttr attr = hipMemPoolAttrUsedMemHigh;
    set_value64 = 1;
    HIP_CHECK_ERROR((hipMemPoolSetAttribute(mempool.mempool(), attr, &set_value64)),
                    hipErrorInvalidValue);
  }
}

/**
 * End doxygen group hipMemPoolSetAttribute.
 * @}
 */

/**
 * @addtogroup hipMemPoolGetAttribute hipMemPoolGetAttribute
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value)`
 * - 	Gets attributes of a memory pool
 */

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipMemPoolGetAttribute behavior with invalid arguments:
 *    -# Nullptr mem_pool
 *    -# Attribute value is not valid
 *    -# Nullptr value
 *
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolSetGetAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipMemPoolGetAttribute_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  MemPoolGuard mempool(MemPools::dev_default, device_id);


  hipMemPoolAttr attr = hipMemPoolReuseFollowEventDependencies;
  int get_value = 0;

  SECTION("Mempool is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolGetAttribute(nullptr, attr, &get_value), hipErrorInvalidValue);
  }

  SECTION("Attribute value is not valid") {
    HIP_CHECK_ERROR(
        hipMemPoolGetAttribute(mempool.mempool(), static_cast<hipMemPoolAttr>(0x9), &get_value),
        hipErrorInvalidValue);
  }

  SECTION("Get values is nullptr") {
    HIP_CHECK_ERROR(hipMemPoolGetAttribute(mempool.mempool(), attr, nullptr), hipErrorInvalidValue);
  }
}
