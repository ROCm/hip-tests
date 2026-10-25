/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "mempool_common.hh"

#include <resource_guards.hh>
#include <utils.hh>

#include <chrono>
#include <thread>

/**
 * @addtogroup hipMemPoolSetAttribute hipMemPoolSetAttribute
 * @{
 * @ingroup StreamOTest
 * `hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value)`
 * - Sets attributes of a memory pool
 */

static __global__ void TouchPagesKernel(int* data, size_t count) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t j = i; j < count; j += stride) data[j] = 0xABCD;
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify that a release threshold pinned to UINT64_MAX is honored when
 *    the pool's free-pressure path fires:
 *    -# pin hipMemPoolAttrReleaseThreshold to UINT64_MAX (readback confirms);
 *    -# allocate and touch 3 x 64 MiB, so the pool reserves 192 MiB;
 *    -# free A and B, let their free markers retire on the idle stream, then free
 *       C - at this point the pool's pressure path (freed 128 MiB > busy 64 MiB)
 *       runs its release logic;
 *    -# the pinned threshold must hold reserved memory at 192 MiB.
 *  - On affected runtimes the reserved bytes drop to ~64 MiB, i.e. the pinned
 *    threshold is silently ignored on the release paths.
 * Test source
 * ------------------------
 *  - /unit/memory/hipMemPoolReleaseThreshold.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemPoolAttrReleaseThreshold_Positive_PinnedUnderFreePressure) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));
  checkMempoolSupported(device_id)

  MemPoolGuard mempool(MemPools::dev_default, device_id);
  StreamGuard stream(Streams::created);

  constexpr size_t kAllocSize = 64ull << 20;
  constexpr int kNumAllocs = 3;

  // Pin the release threshold: the pool must not release any reserved memory
  std::uint64_t threshold = ~0ull;
  HIP_CHECK(
      hipMemPoolSetAttribute(mempool.mempool(), hipMemPoolAttrReleaseThreshold, &threshold));
  std::uint64_t readback = 0;
  HIP_CHECK(
      hipMemPoolGetAttribute(mempool.mempool(), hipMemPoolAttrReleaseThreshold, &readback));
  REQUIRE(readback == threshold);

  int* mem[kNumAllocs];
  for (int i = 0; i < kNumAllocs; ++i) {
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&mem[i]), kAllocSize,
                                     mempool.mempool(), stream.stream()));
    TouchPagesKernel<<<64, 256, 0, stream.stream()>>>(mem[i], kAllocSize / sizeof(int));
  }
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t reserved_before = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), hipMemPoolAttrReservedMemCurrent,
                                   &reserved_before));
  REQUIRE(reserved_before >= kNumAllocs * kAllocSize);

  // Free A and B, let their free markers retire on the now-idle stream, then free C.
  // The pool's free-pressure path (freed 128 MiB > busy 64 MiB) fires here and runs
  // the release logic; the pinned threshold must still be honored.
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(mem[0]), stream.stream()));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(mem[1]), stream.stream()));
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(mem[2]), stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  std::uint64_t reserved_after = 0;
  HIP_CHECK(hipMemPoolGetAttribute(mempool.mempool(), hipMemPoolAttrReservedMemCurrent,
                                   &reserved_after));
  REQUIRE(reserved_after >= kNumAllocs * kAllocSize);
}

/**
 * End doxygen group hipMemPoolSetAttribute.
 * @}
 */
