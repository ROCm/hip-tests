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

#pragma once

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <resource_guards.hh>

namespace cg = cooperative_groups;

enum class AtomicOp { kAdd = 0, kAddSystem, kSub, kSubSystem, kInc, kDec, kUnsafeAdd, kSafeAdd };

constexpr auto kIntegerTestValue = 7;
constexpr auto kFloatingPointTestValue = 3.125;

template <typename TestType, AtomicOp op> TestType GetTestValue() {
  if constexpr (op == AtomicOp::kInc || op == AtomicOp::kDec) {
    return 1;
  }

  return std::is_floating_point_v<TestType> ? kFloatingPointTestValue : kIntegerTestValue;
}

template <typename TestType, AtomicOp op> __device__ TestType GetTestValue() {
  if constexpr (op == AtomicOp::kInc || op == AtomicOp::kDec) {
    return cg::this_grid().size();
  }

  return std::is_floating_point_v<TestType> ? kFloatingPointTestValue : kIntegerTestValue;
}

static std::string to_string(const LinearAllocs allocation_type) {
  switch (allocation_type) {
    case LinearAllocs::malloc:
      return "host pageable";
    case LinearAllocs::mallocAndRegister:
      return "mapped";
    case LinearAllocs::hipHostMalloc:
      return "host pinned";
    case LinearAllocs::hipMalloc:
      return "device malloc";
    case LinearAllocs::hipMallocManaged:
      return "managed";
    default:
      return "unknown alloc type";
  }
}

template <typename TestType, AtomicOp op>
__device__ void PerformAtomicOperation(TestType* const mem, TestType* const old_vals,
                                       const unsigned int tid) {
  TestType val = GetTestValue<TestType, op>();

  if constexpr (op == AtomicOp::kAdd) {
    old_vals[tid] = atomicAdd(mem, val);
  } else if constexpr (op == AtomicOp::kAddSystem) {
    old_vals[tid] = atomicAdd_system(mem, val);
  } else if constexpr (op == AtomicOp::kSub) {
    old_vals[tid] = atomicSub(mem, val);
  } else if constexpr (op == AtomicOp::kSubSystem) {
    old_vals[tid] = atomicSub_system(mem, val);
  } else if constexpr (op == AtomicOp::kInc) {
    old_vals[tid] = atomicInc(mem, val);
  } else if constexpr (op == AtomicOp::kDec) {
    old_vals[tid] = atomicDec(mem, val);
  } else if constexpr (op == AtomicOp::kUnsafeAdd) {
    old_vals[tid] = unsafeAtomicAdd(mem, val);
  } else if constexpr (op == AtomicOp::kSafeAdd) {
    old_vals[tid] = safeAtomicAdd(mem, val);
  }
}

template <typename TestType, AtomicOp op, bool use_shared_mem>
__global__ void AtomicTestKernel(TestType* const global_mem, TestType* const old_vals) {
  __shared__ TestType shared_mem;

  const auto tid = cg::this_grid().thread_rank();

  TestType* const mem = use_shared_mem ? &shared_mem : global_mem;

  if constexpr (use_shared_mem) {
    if (tid == 0) mem[0] = global_mem[0];
    __syncthreads();
  }

  PerformAtomicOperation<TestType, op>(mem, old_vals, tid);

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid == 0) global_mem[0] = mem[0];
  }
}

template <typename TestType>
__device__ TestType* PitchedOffset(TestType* const ptr, const unsigned int pitch,
                                   const unsigned int idx) {
  const auto byte_ptr = reinterpret_cast<uint8_t*>(ptr);
  return reinterpret_cast<TestType*>(byte_ptr + idx * pitch);
}

template <typename TestType, AtomicOp op>
__device__ void PerformPitchedAtomicOperation(TestType* const mem, TestType* const old_vals,
                                              const unsigned int width, const unsigned int pitch,
                                              const unsigned int tid) {
  TestType val = GetTestValue<TestType, op>();

  if constexpr (op == AtomicOp::kAdd) {
    old_vals[tid] = atomicAdd(PitchedOffset(mem, pitch, tid % width), val);
  } else if constexpr (op == AtomicOp::kAddSystem) {
    old_vals[tid] = atomicAdd_system(PitchedOffset(mem, pitch, tid % width), val);
  } else if constexpr (op == AtomicOp::kSub) {
    old_vals[tid] = atomicSub(PitchedOffset(mem, pitch, tid % width), val);
  } else if constexpr (op == AtomicOp::kSubSystem) {
    old_vals[tid] = atomicSub_system(PitchedOffset(mem, pitch, tid % width), val);
  } else if constexpr (op == AtomicOp::kInc) {
    old_vals[tid] = atomicInc(PitchedOffset(mem, pitch, tid % width), val);
  } else if constexpr (op == AtomicOp::kDec) {
    old_vals[tid] = atomicDec(PitchedOffset(mem, pitch, tid % width), val);
  } else if constexpr (op == AtomicOp::kUnsafeAdd) {
    old_vals[tid] = unsafeAtomicAdd(PitchedOffset(mem, pitch, tid % width), val);
  } else if constexpr (op == AtomicOp::kSafeAdd) {
    old_vals[tid] = safeAtomicAdd(PitchedOffset(mem, pitch, tid % width), val);
  }
}

template <typename TestType, AtomicOp op, bool use_shared_mem>
__global__ void MultiDestAtomicTestKernel(TestType* const global_mem, TestType* const old_vals,
                                          const unsigned int width, const unsigned pitch) {
  extern __shared__ uint8_t shared_mem[];

  const auto tid = cg::this_grid().thread_rank();

  TestType* const mem = use_shared_mem ? reinterpret_cast<TestType*>(shared_mem) : global_mem;

  if constexpr (use_shared_mem) {
    if (tid < width) {
      const auto target = PitchedOffset(mem, pitch, tid);
      *target = *PitchedOffset(global_mem, pitch, tid);
    };
    __syncthreads();
  }

  PerformPitchedAtomicOperation<TestType, op>(mem, old_vals, width, pitch, tid);

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid < width) {
      const auto target = PitchedOffset(global_mem, pitch, tid);
      *target = *PitchedOffset(mem, pitch, tid);
    };
  }
}

template <typename TestType, AtomicOp op>
void InitializeMemory(TestType* const mem, const unsigned int width, const unsigned int pitch,
                      const unsigned int thread_count) {
  TestType val = GetTestValue<TestType, op>();

  if constexpr (op == AtomicOp::kAdd || op == AtomicOp::kAddSystem || op == AtomicOp::kInc) {
    HIP_CHECK(hipMemset(mem, 0, width * pitch));
  } else if constexpr (op == AtomicOp::kSub || op == AtomicOp::kSubSystem || op == AtomicOp::kDec) {
    HIP_CHECK(hipMemset2D(mem, pitch, thread_count / width * val, sizeof(TestType), width));
  }
}

template <typename TestType, AtomicOp op>
void VerifyResult(TestType* const result, const unsigned int width,
                  const unsigned int thread_count) {
  TestType val = GetTestValue<TestType, op>();

  if constexpr (op == AtomicOp::kAdd || op == AtomicOp::kAddSystem || op == AtomicOp::kInc) {
    for (auto i = 0u; i < width; ++i) REQUIRE(thread_count / width * val == result[i]);
  } else if constexpr (op == AtomicOp::kSub || op == AtomicOp::kSubSystem || op == AtomicOp::kDec) {
    for (auto i = 0u; i < width; ++i) REQUIRE(0 == result[i]);
  }
}

template <typename TestType, AtomicOp op>
void VerifyOldValues(std::vector<TestType>& old_vals, const unsigned int width) {
  TestType val = GetTestValue<TestType, op>();

  std::sort(old_vals.begin(), old_vals.end());
  for (auto i = 0u, j = 0u; i < old_vals.size(); ++i, j += (i % width == 0)) {
    REQUIRE(j * val == old_vals[i]);
  }
}

template <typename TestType, AtomicOp op, bool use_shared_mem>
void SameAddressTestImpl(const dim3 blocks, const dim3 threads, const LinearAllocs alloc_type) {
  const unsigned int flags =
      alloc_type == LinearAllocs::mallocAndRegister ? hipHostRegisterMapped : 0u;
  LinearAllocGuard<TestType> mem_dev(alloc_type, sizeof(TestType), flags);
  TestType result;

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType);
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(thread_count);

  TestType* ptr = alloc_type == LinearAllocs::hipMalloc ? mem_dev.ptr() : mem_dev.host_ptr();
  InitializeMemory<TestType, op>(ptr, 1, sizeof(TestType), thread_count);

  AtomicTestKernel<TestType, op, use_shared_mem>
      <<<blocks, threads>>>(mem_dev.ptr(), old_vals_dev.ptr());
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(&result, ptr, sizeof(TestType), hipMemcpyDeviceToHost));

  VerifyResult<TestType, op>(&result, 1, thread_count);

  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));

  VerifyOldValues<TestType, op>(old_vals, 1);
}

template <typename TestType, AtomicOp op> void SameAddressTest() {
  const auto threads = GENERATE(dim3(1024));

  SECTION("Global memory") {
    constexpr auto blocks = dim3(3);
    using LA = LinearAllocs;
    const auto alloc_type =
        GENERATE(LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister);
    SameAddressTestImpl<TestType, op, false>(blocks, threads, alloc_type);
  }

  SECTION("Shared memory") {
    constexpr auto blocks = dim3(1);
    SameAddressTestImpl<TestType, op, true>(blocks, threads, LinearAllocs::hipMalloc);
  }
}

template <typename TestType, AtomicOp op, bool use_shared_mem>
void MultiDestWithScatterTestImpl(const dim3 blocks, const dim3 threads,
                                  const LinearAllocs alloc_type, const unsigned int width,
                                  const unsigned int pitch) {
  const auto mem_alloc_size = width * pitch;
  const unsigned int flags =
      alloc_type == LinearAllocs::mallocAndRegister ? hipHostRegisterMapped : 0u;
  LinearAllocGuard<TestType> mem_dev(alloc_type, mem_alloc_size, flags);
  std::vector<TestType> result(width);

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType);
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(thread_count);

  TestType* ptr = alloc_type == LinearAllocs::hipMalloc ? mem_dev.ptr() : mem_dev.host_ptr();
  InitializeMemory<TestType, op>(ptr, width, pitch, thread_count);

  const auto shared_mem_size = use_shared_mem ? mem_alloc_size : 0u;
  MultiDestAtomicTestKernel<TestType, op, use_shared_mem>
      <<<blocks, threads, shared_mem_size>>>(mem_dev.ptr(), old_vals_dev.ptr(), width, pitch);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy2D(result.data(), sizeof(TestType), ptr, pitch, sizeof(TestType), width,
                        hipMemcpyDeviceToHost));

  VerifyResult<TestType, op>(result.data(), width, thread_count);

  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));

  VerifyOldValues<TestType, op>(old_vals, width);
}

template <typename TestType, AtomicOp op>
void MultiDestWithScatterTest(const unsigned int width, const unsigned int pitch) {
  const auto threads = GENERATE(dim3(1024));

  SECTION("Global memory") {
    constexpr auto blocks = dim3(3);
    using LA = LinearAllocs;
    for (const auto alloc_type :
         {LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
      DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
        MultiDestWithScatterTestImpl<TestType, op, false>(blocks, threads, alloc_type, width,
                                                          pitch);
      }
    }
  }

  SECTION("Shared memory") {
    constexpr auto blocks = dim3(1);
    MultiDestWithScatterTestImpl<TestType, op, true>(blocks, threads, LinearAllocs::hipMalloc,
                                                     width, pitch);
  }
}

template <typename TestType, AtomicOp op>
void MultiKernelTestImpl(const dim3 blocks, const dim3 threads, const LinearAllocs alloc_type) {
  const unsigned int flags =
      alloc_type == LinearAllocs::mallocAndRegister ? hipHostRegisterMapped : 0u;
  LinearAllocGuard<TestType> mem_dev(alloc_type, sizeof(TestType), flags);
  TestType result;

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType) * 2;
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(thread_count * 2);

  TestType* ptr = alloc_type == LinearAllocs::hipMalloc ? mem_dev.ptr() : mem_dev.host_ptr();
  InitializeMemory<TestType, op>(ptr, 1, sizeof(TestType), thread_count * 2);

  StreamGuard stream1(Streams::created);
  StreamGuard stream2(Streams::created);

  AtomicTestKernel<TestType, op, false>
      <<<blocks, threads, 0, stream1.stream()>>>(mem_dev.ptr(), old_vals_dev.ptr());
  HIP_CHECK(hipGetLastError());

  AtomicTestKernel<TestType, op, false>
      <<<blocks, threads, 0, stream2.stream()>>>(mem_dev.ptr(), old_vals_dev.ptr() + thread_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipStreamSynchronize(stream1.stream()));
  HIP_CHECK(hipStreamSynchronize(stream2.stream()));

  HIP_CHECK(hipMemcpy(&result, ptr, sizeof(TestType), hipMemcpyDeviceToHost));

  VerifyResult<TestType, op>(ptr, 1, thread_count * 2);

  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));

  VerifyOldValues<TestType, op>(old_vals, 1);
}

template <typename TestType, AtomicOp op> void MultiKernelTest() {
  const auto threads = GENERATE(dim3(1024));

  constexpr auto blocks = dim3(3);
  using LA = LinearAllocs;
  for (const auto alloc_type :
       {LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      MultiKernelTestImpl<TestType, op>(blocks, threads, alloc_type);
    }
  }
}

template <typename TestType, AtomicOp op>
void PerformAtomicOperationHost(TestType* const mem, TestType* old_vals, const unsigned int tid) {
  TestType val = GetTestValue<TestType, op>();

  if constexpr (op == AtomicOp::kAddSystem) {
    old_vals[tid] = __atomic_fetch_add(mem, val, __ATOMIC_RELAXED);
  } else if constexpr (op == AtomicOp::kSubSystem) {
    old_vals[tid] = __atomic_fetch_sub(mem, val, __ATOMIC_RELAXED);
  }
}

template <typename TestType, AtomicOp op>
void AtomicTestHost(const unsigned int thread_count, TestType* const mem,
                    TestType* const old_vals) {
  for (auto tid = 0; tid < thread_count; ++tid) {
    PerformAtomicOperationHost<TestType, op>(mem, old_vals, tid + thread_count);
  }
}

template <typename TestType, AtomicOp op>
void HostCoherencyTestImpl(const dim3 blocks, const dim3 threads, const LinearAllocs alloc_type) {
  const unsigned int flags =
      alloc_type == LinearAllocs::mallocAndRegister ? hipHostRegisterMapped : 0u;
  LinearAllocGuard<TestType> mem_dev(alloc_type, sizeof(TestType), flags);

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType);
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(thread_count * 2);

  InitializeMemory<TestType, op>(mem_dev.host_ptr(), 1, sizeof(TestType), thread_count * 2);

  AtomicTestKernel<TestType, op, false><<<blocks, threads>>>(mem_dev.ptr(), old_vals_dev.ptr());
  HIP_CHECK(hipGetLastError());

  AtomicTestHost<TestType, op>(thread_count, mem_dev.host_ptr(), old_vals.data());
  HIP_CHECK(hipDeviceSynchronize());

  VerifyResult<TestType, op>(mem_dev.host_ptr(), 1, thread_count * 2);

  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));

  VerifyOldValues<TestType, op>(old_vals, 1);
}

template <typename TestType, AtomicOp op> void HostCoherencyTest() {
  const auto threads = GENERATE(dim3(1024));

  constexpr auto blocks = dim3(2);
  using LA = LinearAllocs;
  for (const auto alloc_type : {LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      HostCoherencyTestImpl<TestType, op>(blocks, threads, alloc_type);
    }
  }
}

template <typename TestType, AtomicOp op>
void PeerDeviceCoherencyTestImpl(const dim3 blocks, const dim3 threads,
                                 const LinearAllocs alloc_type) {
  HIP_CHECK(hipSetDevice(0));

  const unsigned int flags =
      alloc_type == LinearAllocs::mallocAndRegister ? hipHostRegisterMapped : 0u;
  LinearAllocGuard<TestType> mem_dev(alloc_type, sizeof(TestType), flags);

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType);
  std::vector<TestType> old_vals(thread_count * 2);

  InitializeMemory<TestType, op>(mem_dev.host_ptr(), 1, sizeof(TestType), thread_count * 2);

  HIP_CHECK(hipSetDevice(0));
  LinearAllocGuard<TestType> old_vals_dev0(LinearAllocs::hipMalloc, old_vals_alloc_size);
  AtomicTestKernel<TestType, op, false><<<blocks, threads>>>(mem_dev.ptr(), old_vals_dev0.ptr());
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipSetDevice(1));
  LinearAllocGuard<TestType> old_vals_dev1(LinearAllocs::hipMalloc, old_vals_alloc_size);
  AtomicTestKernel<TestType, op, false><<<blocks, threads>>>(mem_dev.ptr(), old_vals_dev1.ptr());
  HIP_CHECK(hipGetLastError());

  for (auto i = 0u; i < 2; ++i) {
    HIP_CHECK(hipSetDevice(i));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipSetDevice(0));

  VerifyResult<TestType, op>(mem_dev.host_ptr(), 1, thread_count * 2);

  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev0.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));

  HIP_CHECK(hipMemcpy(old_vals.data() + thread_count, old_vals_dev1.ptr(), old_vals_alloc_size,
                      hipMemcpyDeviceToHost));

  VerifyOldValues<TestType, op>(old_vals, 1);
}

template <typename TestType, AtomicOp op> void PeerDeviceCoherencyTest() {
  const auto threads = GENERATE(dim3(1024));

  constexpr auto blocks = dim3(2);
  using LA = LinearAllocs;
  for (const auto alloc_type : {LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      PeerDeviceCoherencyTestImpl<TestType, op>(blocks, threads, alloc_type);
    }
  }
}
