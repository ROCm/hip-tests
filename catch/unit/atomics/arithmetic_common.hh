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

#pragma once

#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <resource_guards.hh>

namespace cg = cooperative_groups;

enum class AtomicOperation {
  kAdd = 0,
  kAddSystem,
  kSub,
  kSubSystem,
  kInc,
  kDec,
  kUnsafeAdd,
  kSafeAdd,
  kCASAdd,
  kCASAddSystem,
  kBuiltinAdd,
  kBuiltinCAS
};

constexpr auto kIntegerTestValue = 7;
constexpr auto kFloatingPointTestValue = 3.125;
constexpr auto kIncDecWraparoundValue = 1023;

template <typename TestType, AtomicOperation operation>
__host__ __device__ TestType GetTestValue() {
  if constexpr (operation == AtomicOperation::kInc || operation == AtomicOperation::kDec) {
    return kIncDecWraparoundValue;
  }

  return std::is_floating_point_v<TestType> ? kFloatingPointTestValue : kIntegerTestValue;
}

template <typename TestType> __device__ TestType CASAtomicAdd(TestType* address, TestType val) {
  TestType old = *address, assumed;

  do {
    assumed = old;
    old = atomicCAS(address, assumed, val + assumed);
  } while (assumed != old);

  return old;
}

template <typename TestType>
__device__ TestType CASAtomicAddSystem(TestType* address, TestType val) {
  TestType old = *address, assumed;

  do {
    assumed = old;
    old = atomicCAS_system(address, assumed, val + assumed);
  } while (assumed != old);

  return old;
}

template <typename TestType, int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
__device__ TestType BuiltinCASAtomicAdd(TestType* address, TestType val) {
  TestType old = *address, assumed;

  const auto builtin_cas = [](TestType* address, TestType assumed, TestType val) {
    __hip_atomic_compare_exchange_strong(address, &assumed, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                         memory_scope);
    return assumed;
  };

  do {
    assumed = old;
    old = builtin_cas(address, assumed, val + assumed);
  } while (assumed != old);

  return old;
}

template <typename TestType, AtomicOperation operation, int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
__device__ TestType PerformAtomicOperation(TestType* const mem) {
  const auto val = GetTestValue<TestType, operation>();

  if constexpr (operation == AtomicOperation::kAdd) {
    return atomicAdd(mem, val);
  } else if constexpr (operation == AtomicOperation::kAddSystem) {
    return atomicAdd_system(mem, val);
  } else if constexpr (operation == AtomicOperation::kSub) {
    return atomicSub(mem, val);
  } else if constexpr (operation == AtomicOperation::kSubSystem) {
    return atomicSub_system(mem, val);
  } else if constexpr (operation == AtomicOperation::kInc) {
    return atomicInc(mem, val);
  } else if constexpr (operation == AtomicOperation::kDec) {
    return atomicDec(mem, val);
  } else if constexpr (operation == AtomicOperation::kUnsafeAdd) {
    return unsafeAtomicAdd(mem, val);
  } else if constexpr (operation == AtomicOperation::kSafeAdd) {
    return safeAtomicAdd(mem, val);
  } else if constexpr (operation == AtomicOperation::kCASAdd) {
    return CASAtomicAdd(mem, val);
  } else if constexpr (operation == AtomicOperation::kCASAddSystem) {
    return CASAtomicAddSystem(mem, val);
  } else if constexpr (operation == AtomicOperation::kBuiltinAdd) {
    return __hip_atomic_fetch_add(mem, val, __ATOMIC_RELAXED, memory_scope);
  } else if constexpr (operation == AtomicOperation::kBuiltinCAS) {
    return BuiltinCASAtomicAdd<TestType, memory_scope>(mem, val);
  }
}

template <typename TestType, AtomicOperation operation, bool use_shared_mem,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
__global__ void TestKernel(TestType* const global_mem, TestType* const old_vals) {
  __shared__ TestType shared_mem;

  const auto tid = cg::this_grid().thread_rank();

  TestType* const mem = use_shared_mem ? &shared_mem : global_mem;

  if constexpr (use_shared_mem) {
    if (tid == 0) mem[0] = global_mem[0];
    __syncthreads();
  }

  old_vals[tid] = PerformAtomicOperation<TestType, operation, memory_scope>(mem);

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid == 0) global_mem[0] = mem[0];
  }
}

template <typename TestType>
__host__ __device__ TestType* PitchedOffset(TestType* const ptr, const unsigned int pitch,
                                            const unsigned int idx) {
  const auto byte_ptr = reinterpret_cast<uint8_t*>(ptr);
  return reinterpret_cast<TestType*>(byte_ptr + idx * pitch);
}

__device__ void GenerateMemoryTraffic(uint8_t* const begin_addr, uint8_t* const end_addr) {
  for (volatile uint8_t* addr = begin_addr; addr != end_addr; ++addr) {
    uint8_t val = *addr;
    val ^= 0xAB;
    *addr = val;
  }
}

template <typename TestType, AtomicOperation operation, bool use_shared_mem,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
__global__ void TestKernel(TestType* const global_mem, TestType* const old_vals,
                           const unsigned int width, const unsigned int pitch) {
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

  const auto n = cooperative_groups::this_grid().size() - width;

  TestType* atomic_addr = PitchedOffset(mem, pitch, tid % width);

  if (tid < n) {
    old_vals[tid] = PerformAtomicOperation<TestType, operation, memory_scope>(
        PitchedOffset(mem, pitch, tid % width));
  } else {
    uint8_t* const begin_addr = reinterpret_cast<uint8_t*>(atomic_addr + 1);
    uint8_t* const end_addr = reinterpret_cast<uint8_t*>(atomic_addr) + pitch;
    GenerateMemoryTraffic(begin_addr, end_addr);
  }

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid < width) {
      const auto target = PitchedOffset(global_mem, pitch, tid);
      *target = *PitchedOffset(mem, pitch, tid);
    };
  }
}

struct TestParams {
  auto ThreadCount() const {
    return blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  }

  auto HostIterationsPerThread() const {
    return std::max(num_devices * kernel_count * ThreadCount() / 20, width);
  }

  dim3 blocks;
  dim3 threads;
  unsigned int num_devices = 1u;
  unsigned int kernel_count = 1u;
  unsigned int width = 1u;
  unsigned int pitch = 0u;
  unsigned int host_thread_count = 0u;
  LinearAllocs alloc_type;
};

template <typename TestType, AtomicOperation operation>
std::tuple<std::vector<TestType>, std::vector<TestType>> TestKernelHostRef(const TestParams& p) {
  const auto val = GetTestValue<TestType, operation>();

  const auto total_thread_count = p.num_devices * p.kernel_count * p.ThreadCount() +
      p.host_thread_count * p.HostIterationsPerThread();

  std::vector<TestType> res_vals(p.width);
  std::vector<TestType> old_vals;
  old_vals.reserve(total_thread_count);

  auto perform_op = [&](unsigned id) {
    auto& res = res_vals[id % p.width];
    old_vals.push_back(res);

    if constexpr (operation == AtomicOperation::kAdd || operation == AtomicOperation::kAddSystem ||
                  operation == AtomicOperation::kUnsafeAdd ||
                  operation == AtomicOperation::kSafeAdd || operation == AtomicOperation::kCASAdd ||
                  operation == AtomicOperation::kCASAddSystem ||
                  operation == AtomicOperation::kBuiltinAdd ||
                  operation == AtomicOperation::kBuiltinCAS) {
      res = res + val;
    } else if constexpr (operation == AtomicOperation::kSub ||
                         operation == AtomicOperation::kSubSystem) {
      res = res - val;
    } else if constexpr (operation == AtomicOperation::kInc) {
      res = (res >= val) ? 0 : res + 1;
    } else if constexpr (operation == AtomicOperation::kDec) {
      res = ((res == 0) || (res > val)) ? val : res - 1;
    }
  };

  for (auto i = 0u; i < p.num_devices; ++i) {
    for (auto j = 0u; j < p.kernel_count; ++j) {
      for (auto tid = 0u; tid < p.ThreadCount() - p.width; ++tid) {
        perform_op(tid);
      }
    }
  }

  for (auto i = 0u; i < p.host_thread_count; ++i) {
    for (auto j = 0u; j < p.HostIterationsPerThread(); ++j) {
      perform_op(j);
    }
  }

  return {res_vals, old_vals};
}

template <typename TestType, AtomicOperation operation>
void Verify(const TestParams& p, std::vector<TestType>& res_vals, std::vector<TestType>& old_vals) {
  auto [expected_res_vals, expected_old_vals] = TestKernelHostRef<TestType, operation>(p);

  for (auto i = 0u; i < res_vals.size(); ++i) {
    INFO("Results index: " << i);
    REQUIRE(expected_res_vals[i] == res_vals[i]);
  }

  std::sort(begin(old_vals), end(old_vals));
  std::sort(begin(expected_old_vals), end(expected_old_vals));
  for (auto i = 0u; i < old_vals.size(); ++i) {
    INFO("Old values index: " << i);
    REQUIRE(expected_old_vals[i] == old_vals[i]);
  }
}

template <typename TestType, AtomicOperation operation, bool use_shared_mem,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
void LaunchKernel(const TestParams& p, hipStream_t stream, TestType* const mem_ptr,
                  TestType* const old_vals) {
  const auto shared_mem_size = use_shared_mem ? p.width * p.pitch : 0u;
  if (p.width == 1 && p.pitch == sizeof(TestType))
    TestKernel<TestType, operation, use_shared_mem, memory_scope>
        <<<p.blocks, p.threads, shared_mem_size, stream>>>(mem_ptr, old_vals);
  else
    TestKernel<TestType, operation, use_shared_mem, memory_scope>
        <<<p.blocks, p.threads, shared_mem_size, stream>>>(mem_ptr, old_vals, p.width, p.pitch);
}

template <typename TestType, AtomicOperation operation>
void HostAtomicOperation(const unsigned int iterations, TestType* mem, TestType* const old_vals,
                         const unsigned int width, const unsigned pitch, TestType base_val) {
  const auto val = GetTestValue<TestType, operation>();

  for (auto i = 0u; i < iterations; ++i) {
    if constexpr (operation == AtomicOperation::kAddSystem ||
                  operation == AtomicOperation::kCASAddSystem ||
                  operation == AtomicOperation::kBuiltinAdd ||
                  operation == AtomicOperation::kBuiltinCAS) {
      old_vals[i] = __atomic_fetch_add(PitchedOffset(mem, pitch, i % width), val, __ATOMIC_RELAXED);
    } else if constexpr (operation == AtomicOperation::kSubSystem) {
      old_vals[i] = __atomic_fetch_sub(PitchedOffset(mem, pitch, i % width), val, __ATOMIC_RELAXED);
    }
  }
}

template <typename TestType, AtomicOperation operation>
void PerformHostAtomicOperation(const TestParams& p, TestType* mem, TestType* const old_vals) {
  if (p.host_thread_count == 0) {
    return;
  }

  const auto host_base_val = p.num_devices * p.kernel_count * p.ThreadCount();

  std::vector<std::thread> threads;
  for (auto i = 0u; i < p.host_thread_count; ++i) {
    const auto iterations = p.HostIterationsPerThread();
    const auto thread_base_val = host_base_val + i * iterations;
    threads.push_back(std::thread(HostAtomicOperation<TestType, operation>, iterations, mem,
                                  old_vals + thread_base_val, p.width, p.pitch, thread_base_val));
  }

  for (auto& th : threads) {
    th.join();
  }
}

template <typename TestType, AtomicOperation operation, bool use_shared_mem,
          int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
void TestCore(const TestParams& p) {
  const unsigned int flags =
      p.alloc_type == LinearAllocs::mallocAndRegister ? hipHostRegisterMapped : 0u;

  const auto old_vals_alloc_size = p.kernel_count * p.ThreadCount() * sizeof(TestType);
  std::vector<LinearAllocGuard<TestType>> old_vals_devs;
  std::vector<StreamGuard> streams;
  for (auto i = 0; i < p.num_devices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    old_vals_devs.emplace_back(LinearAllocs::hipMalloc, old_vals_alloc_size);
    for (auto j = 0; j < p.kernel_count; ++j) {
      streams.emplace_back(Streams::created);
    }
  }

  const auto mem_alloc_size = p.width * p.pitch;
  LinearAllocGuard<TestType> mem_dev(p.alloc_type, mem_alloc_size, flags);

  std::vector<TestType> old_vals(p.num_devices * p.kernel_count * p.ThreadCount() +
                                 p.host_thread_count * p.HostIterationsPerThread());
  std::vector<TestType> res_vals(p.width);

  TestType* const mem_ptr =
      p.alloc_type == LinearAllocs::hipMalloc ? mem_dev.ptr() : mem_dev.host_ptr();

  HIP_CHECK(hipMemset(mem_ptr, 0, mem_alloc_size));

  for (auto i = 0u; i < p.num_devices; ++i) {
    for (auto j = 0u; j < p.kernel_count; ++j) {
      const auto& stream = streams[i * p.kernel_count + j].stream();
      const auto old_vals = old_vals_devs[i].ptr() + j * p.ThreadCount();
      LaunchKernel<TestType, operation, use_shared_mem, memory_scope>(p, stream, mem_dev.ptr(),
                                                                      old_vals);
    }
  }

  PerformHostAtomicOperation<TestType, operation>(p, mem_dev.host_ptr(), old_vals.data());

  for (auto i = 0u; i < p.num_devices; ++i) {
    const auto device_offset = i * p.kernel_count * p.ThreadCount();
    HIP_CHECK(hipMemcpy(old_vals.data() + device_offset, old_vals_devs[i].ptr(),
                        old_vals_alloc_size, hipMemcpyDeviceToHost));
  }
  HIP_CHECK(hipMemcpy2D(res_vals.data(), sizeof(TestType), mem_ptr, p.pitch, sizeof(TestType),
                        p.width, hipMemcpyDeviceToHost));

  Verify<TestType, operation>(p, res_vals, old_vals);
}

inline dim3 GenerateThreadDimensions() { return GENERATE(dim3(16), dim3(1024)); }

inline dim3 GenerateBlockDimensions() {
  int sm_count = 0;
  HIP_CHECK(hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, 0));
  return GENERATE_COPY(dim3(sm_count), dim3(sm_count + sm_count / 2));
}

template <typename TestType, AtomicOperation operation, int memory_scope = __HIP_MEMORY_SCOPE_AGENT>
void SingleDeviceSingleKernelTest(const unsigned int width, const unsigned int pitch) {
  TestParams params;
  params.num_devices = 1;
  params.kernel_count = 1;
  if constexpr ((operation == AtomicOperation::kBuiltinAdd ||
                 operation == AtomicOperation::kBuiltinCAS) &&
                memory_scope == __HIP_MEMORY_SCOPE_SINGLETHREAD) {
    params.threads = 1;
  } else if constexpr ((operation == AtomicOperation::kBuiltinAdd ||
                        operation == AtomicOperation::kBuiltinCAS) &&
                       memory_scope == __HIP_MEMORY_SCOPE_WAVEFRONT) {
    int warp_size = 0;
    HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
    params.threads = dim3(warp_size);
  } else {
    params.threads = GenerateThreadDimensions();
  }
  params.width = width;
  params.pitch = pitch;

  SECTION("Global memory") {
    if constexpr ((operation == AtomicOperation::kBuiltinAdd ||
                   operation == AtomicOperation::kBuiltinCAS) &&
                  (memory_scope == __HIP_MEMORY_SCOPE_SINGLETHREAD ||
                   memory_scope == __HIP_MEMORY_SCOPE_WAVEFRONT ||
                   memory_scope == __HIP_MEMORY_SCOPE_WORKGROUP)) {
      params.blocks = dim3(1);
    } else {
      params.blocks = GenerateBlockDimensions();
    }
    using LA = LinearAllocs;
    for (const auto alloc_type :
         {LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
      params.alloc_type = alloc_type;
      DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
        TestCore<TestType, operation, false, memory_scope>(params);
      }
    }
  }

  SECTION("Shared memory") {
    params.blocks = dim3(1);
    params.alloc_type = LinearAllocs::hipMalloc;
    TestCore<TestType, operation, true, memory_scope>(params);
  }
}

template <typename TestType, AtomicOperation operation>
void SingleDeviceMultipleKernelTest(const unsigned int kernel_count, const unsigned int width,
                                    const unsigned int pitch) {
  int concurrent_kernels = 0;
  HIP_CHECK(hipDeviceGetAttribute(&concurrent_kernels, hipDeviceAttributeConcurrentKernels, 0));
  if (!concurrent_kernels) {
    HipTest::HIP_SKIP_TEST("Test requires support for concurrent kernel execution");
    return;
  }

  TestParams params;
  params.num_devices = 1;
  params.kernel_count = kernel_count;
  params.blocks = GenerateBlockDimensions();
  params.threads = GenerateThreadDimensions();
  params.width = width;
  params.pitch = pitch;

  using LA = LinearAllocs;
  for (const auto alloc_type :
       {LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
    params.alloc_type = alloc_type;
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      TestCore<TestType, operation, false>(params);
    }
  }
}

template <typename TestType, AtomicOperation operation>
void MultipleDeviceMultipleKernelAndHostTest(const unsigned int num_devices,
                                             const unsigned int kernel_count,
                                             const unsigned int width, const unsigned int pitch,
                                             const unsigned int host_thread_count = 0u) {
  if (num_devices > 1) {
    if (HipTest::getDeviceCount() < num_devices) {
      std::string msg = std::to_string(num_devices) + " devices are required";
      HipTest::HIP_SKIP_TEST(msg.c_str());
      return;
    }
  }

  if (kernel_count > 1) {
    for (auto i = 0u; i < num_devices; ++i) {
      int concurrent_kernels = 0;
      HIP_CHECK(hipDeviceGetAttribute(&concurrent_kernels, hipDeviceAttributeConcurrentKernels, i));
      if (!concurrent_kernels) {
        HipTest::HIP_SKIP_TEST("Test requires support for concurrent kernel execution");
        return;
      }
    }
  }

  TestParams params;
  params.num_devices = num_devices;
  params.kernel_count = kernel_count;
  params.blocks = GenerateBlockDimensions();
  params.threads = GenerateThreadDimensions();
  params.width = width;
  params.pitch = pitch;
  params.host_thread_count = host_thread_count;

  using LA = LinearAllocs;
  for (const auto alloc_type : {LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
    params.alloc_type = alloc_type;
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      TestCore<TestType, operation, false, __HIP_MEMORY_SCOPE_SYSTEM>(params);
    }
  }
}