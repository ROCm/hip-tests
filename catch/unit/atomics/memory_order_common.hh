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

#include <hip_test_common.hh>
#include <resource_guards.hh>

enum class BuiltinAtomicOperation {
  kLoadStore = 0,
  kExchange,
  kCompareExchangeStrong,
  kCompareExchangeWeak,
  kAdd,
  kAnd,
  kOr,
  kXor,
  kMin,
  kMax
};

template <BuiltinAtomicOperation operation, int memory_order, int memory_scope>
__device__ void SetFlag(int* const flag) {
  if constexpr (operation == BuiltinAtomicOperation::kLoadStore) {
    static_assert(memory_order != __ATOMIC_ACQ_REL);
    __hip_atomic_store(flag, 1, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kExchange) {
    __hip_atomic_exchange(flag, 1, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kCompareExchangeStrong) {
    int compare = 0;
    __hip_atomic_compare_exchange_strong(flag, &compare, 1, memory_order, __ATOMIC_RELAXED,
                                         memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kCompareExchangeWeak) {
    int compare = 0;
    while (!__hip_atomic_compare_exchange_weak(flag, &compare, 1, memory_order, __ATOMIC_RELAXED,
                                               memory_scope))
      compare = 0;
  } else if constexpr (operation == BuiltinAtomicOperation::kAdd) {
    __hip_atomic_fetch_add(flag, 1, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kAnd) {
    __hip_atomic_fetch_and(flag, 0x0, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kOr) {
    __hip_atomic_fetch_or(flag, 0x1, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kXor) {
    __hip_atomic_fetch_xor(flag, 0x1, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kMin) {
    __hip_atomic_fetch_min(flag, -1, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kMax) {
    __hip_atomic_fetch_max(flag, 1, memory_order, memory_scope);
  }
}

template <BuiltinAtomicOperation operation, int memory_order, int memory_scope>
__device__ int FetchFlag(int* const flag) {
  if constexpr (operation == BuiltinAtomicOperation::kLoadStore) {
    static_assert(memory_order != __ATOMIC_ACQ_REL);
    return __hip_atomic_load(flag, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kExchange) {
    return __hip_atomic_exchange(flag, 0, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kCompareExchangeStrong) {
    int compare = 1;
    __hip_atomic_compare_exchange_strong(
        flag, &compare, 1, memory_order,
        memory_order == __ATOMIC_ACQ_REL ? __ATOMIC_ACQUIRE : memory_order, memory_scope);
    return compare;
  } else if constexpr (operation == BuiltinAtomicOperation::kCompareExchangeWeak) {
    int compare = 1;
    __hip_atomic_compare_exchange_weak(
        flag, &compare, 1, memory_order,
        memory_order == __ATOMIC_ACQ_REL ? __ATOMIC_ACQUIRE : memory_order, memory_scope);
    return compare;
  } else if constexpr (operation == BuiltinAtomicOperation::kAdd) {
    return __hip_atomic_fetch_add(flag, 0, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kAnd) {
    return !__hip_atomic_fetch_and(flag, 0x1, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kOr) {
    return __hip_atomic_fetch_or(flag, 0x0, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kXor) {
    return __hip_atomic_fetch_xor(flag, 0x0, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kMin) {
    return __hip_atomic_fetch_min(flag, 0, memory_order, memory_scope);
  } else if constexpr (operation == BuiltinAtomicOperation::kMax) {
    return __hip_atomic_fetch_max(flag, 0, memory_order, memory_scope);
  }
}

namespace AcquireRelease {

constexpr auto kTestValue = 42;

template <BuiltinAtomicOperation operation, int memory_order, int memory_scope>
__host__ __device__ void Producer(int* const flag, int* const data) {
  constexpr int actual_memory_order =
      memory_order == __ATOMIC_ACQUIRE ? __ATOMIC_RELEASE : memory_order;

  data[0] = kTestValue;

#ifdef __HIP_DEVICE_COMPILE__
  SetFlag<operation, actual_memory_order, memory_scope>(flag);
#else
  __atomic_store_n(flag, 1, __ATOMIC_RELEASE);
#endif
}

template <BuiltinAtomicOperation operation, int memory_order, int memory_scope>
__host__ __device__ void Consumer(int* const flag, int* const data, int* const ret) {
  constexpr int actual_memory_order =
      memory_order == __ATOMIC_RELEASE ? __ATOMIC_ACQUIRE : memory_order;

#ifdef __HIP_DEVICE_COMPILE__
  while (!FetchFlag<operation, memory_order, memory_scope>(flag))
    ;
#else
  while (!__atomic_load_n(flag, __ATOMIC_ACQUIRE))
    ;
#endif

  ret[0] = data[0];
}

template <BuiltinAtomicOperation operation, int memory_order, int memory_scope>
__global__ void TestKernel(int* const flag, int* data, int* const ret) {
  __shared__ int shared_mem;

  if (data == nullptr) data = &shared_mem;

  if (blockIdx.x == 0 && threadIdx.x == 0) *flag = 0;
  __syncthreads();

  bool producer = false, consumer = false;

  if constexpr (memory_scope == __HIP_MEMORY_SCOPE_WAVEFRONT) {
    producer = blockIdx.x == 0 && threadIdx.x == 0;
    consumer = blockIdx.x == 0 && threadIdx.x == 1;
  } else if constexpr (memory_scope == __HIP_MEMORY_SCOPE_WORKGROUP) {
    producer = blockIdx.x == 0 && threadIdx.x == 0;
    consumer = blockIdx.x == 0 && threadIdx.x == warpSize;
  } else if constexpr (memory_scope == __HIP_MEMORY_SCOPE_AGENT) {
    producer = blockIdx.x == 0 && threadIdx.x == 0;
    consumer = blockIdx.x == 1 && threadIdx.x == 0;
  }

  if (producer) {
    Producer<operation, memory_order, memory_scope>(flag, data);
    return;
  }

  if (consumer) {
    Consumer<operation, memory_order, memory_scope>(flag, data, ret);
    return;
  }
}

template <BuiltinAtomicOperation operation, bool use_acq_rel, int memory_scope>
__global__ void ProducerKernel(int* const flag, int* const data) {
  if (!(blockIdx.x == 0 && threadIdx.x == 0)) {
    return;
  }

  Producer<operation, use_acq_rel, memory_scope>(flag, data);
}

template <BuiltinAtomicOperation operation, bool use_acq_rel, int memory_scope>
__global__ void ConsumerKernel(int* const flag, int* const data, int* const ret) {
  if (!(blockIdx.x == 0 && threadIdx.x == 0)) {
    return;
  }

  Consumer<operation, use_acq_rel, memory_scope>(flag, data, ret);
}

template <BuiltinAtomicOperation operation, int memory_order, int memory_scope>
void SingleDeviceSingleKernelTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  int blocks = 1, threads = 1;
  if (memory_scope == __HIP_MEMORY_SCOPE_WAVEFRONT) {
    blocks = 1;
    threads = 2;
  } else if (memory_scope == __HIP_MEMORY_SCOPE_WORKGROUP) {
    blocks = 1;
    threads = warp_size * 2;
  } else if (memory_scope == __HIP_MEMORY_SCOPE_AGENT) {
    blocks = 2;
    threads = 1;
  }

  LinearAllocGuard<int> flag(LinearAllocs::hipMalloc, sizeof(int));
  LinearAllocGuard<int> ret(LinearAllocs::hipMallocManaged, sizeof(int));

  SECTION("Global memory") {
    const auto alloc_type = GENERATE(LinearAllocs::hipMalloc, LinearAllocs::hipHostMalloc,
                                     LinearAllocs::hipMallocManaged);
    LinearAllocGuard<int> data(alloc_type, sizeof(int));
    TestKernel<operation, memory_order, memory_scope>
        <<<blocks, threads>>>(flag.ptr(), data.ptr(), ret.ptr());
  }

  if (memory_scope != __HIP_MEMORY_SCOPE_AGENT && memory_scope != __HIP_MEMORY_SCOPE_SYSTEM) {
    SECTION("Shared memory") {
      TestKernel<operation, memory_order, memory_scope>
          <<<blocks, threads>>>(flag.ptr(), nullptr, ret.ptr());
    }
  }

  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(ret.ptr()[0] == kTestValue);
}

} /* namespace AcquireRelease */

namespace SequentialConsistency {

template <BuiltinAtomicOperation operation, int memory_scope>
__host__ __device__ void Producer(int* const flag) {
#ifdef __HIP_DEVICE_COMPILE__
  SetFlag<operation, __ATOMIC_SEQ_CST, memory_scope>(flag);
#else
  __atomic_store_n(flag, 1, __ATOMIC_SEQ_CST);
#endif
}

template <BuiltinAtomicOperation operation, int memory_scope>
__host__ __device__ void Consumer(int* const flag1, int* const flag2, int* const counter) {
#ifdef __HIP_DEVICE_COMPILE__
  while (!FetchFlag<operation, __ATOMIC_SEQ_CST, memory_scope>(flag1))
    ;
  if (FetchFlag<operation, __ATOMIC_SEQ_CST, memory_scope>(flag2)) {
    __hip_atomic_fetch_add(counter, 1, __ATOMIC_SEQ_CST, memory_scope);
  }
#else
  while (!__atomic_load_n(flag1, __ATOMIC_SEQ_CST))
    ;
  if (__atomic_load_n(flag2, __ATOMIC_SEQ_CST)) {
    __atomic_fetch_add(counter, 1, __ATOMIC_SEQ_CST);
  }
#endif
}

template <BuiltinAtomicOperation operation, int memory_scope>
__global__ void TestKernel(int* flag1, int* flag2, int* const counter) {
  __shared__ int shared_mem[2];

  if (flag1 == nullptr) flag1 = &shared_mem[0];
  if (flag2 == nullptr) flag2 = &shared_mem[1];

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *flag1 = 0;
    *flag2 = 0;
  }
  __syncthreads();

  bool producer1 = false, producer2 = false, consumer1 = false, consumer2 = false;

  if constexpr (memory_scope == __HIP_MEMORY_SCOPE_WAVEFRONT) {
    producer1 = blockIdx.x == 0 && threadIdx.x == 0;
    consumer1 = blockIdx.x == 0 && threadIdx.x == 1;
    producer2 = blockIdx.x == 0 && threadIdx.x == 2;
    consumer2 = blockIdx.x == 0 && threadIdx.x == 3;
  } else if constexpr (memory_scope == __HIP_MEMORY_SCOPE_WORKGROUP) {
    producer1 = blockIdx.x == 0 && threadIdx.x == 0;
    consumer1 = blockIdx.x == 0 && threadIdx.x == warpSize;
    producer2 = blockIdx.x == 0 && threadIdx.x == warpSize * 2;
    consumer2 = blockIdx.x == 0 && threadIdx.x == warpSize * 3;
  } else if constexpr (memory_scope == __HIP_MEMORY_SCOPE_AGENT) {
    producer1 = blockIdx.x == 0 && threadIdx.x == 0;
    consumer1 = blockIdx.x == 1 && threadIdx.x == 0;
    producer2 = blockIdx.x == 2 && threadIdx.x == 0;
    consumer2 = blockIdx.x == 3 && threadIdx.x == 0;
  }

  if (producer1) {
    Producer<operation, memory_scope>(flag1);
    return;
  }

  if (consumer1) {
    Consumer<operation, memory_scope>(flag1, flag2, counter);
    return;
  }

  if (producer2) {
    Producer<operation, memory_scope>(flag2);
    return;
  }

  if (consumer2) {
    Consumer<operation, memory_scope>(flag2, flag1, counter);
    return;
  }
}

template <BuiltinAtomicOperation operation, int memory_scope>
__global__ void ProducerKernel(int* const flag) {
  if (!(blockIdx.x == 0 && threadIdx.x == 0)) {
    return;
  }

  Producer<operation, memory_scope>(flag);
}

template <BuiltinAtomicOperation operation, int memory_scope>
__global__ void ConsumerKernel(int* const flag1, int* const flag2, int* const counter) {
  if (!(blockIdx.x == 0 && threadIdx.x == 0)) {
    return;
  }

  Consumer<operation, memory_scope>(flag1, flag2, counter);
}

template <BuiltinAtomicOperation operation, int memory_scope> void SingleDeviceSingleKernelTest() {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  int blocks = 1, threads = 1;
  if (memory_scope == __HIP_MEMORY_SCOPE_WAVEFRONT) {
    blocks = 1;
    threads = 4;
  } else if (memory_scope == __HIP_MEMORY_SCOPE_WORKGROUP) {
    blocks = 1;
    threads = warp_size * 4;
  } else if (memory_scope == __HIP_MEMORY_SCOPE_AGENT) {
    blocks = 4;
    threads = 1;
  }

  LinearAllocGuard<int> counter(LinearAllocs::hipMallocManaged, sizeof(int));

  SECTION("Global memory") {
    const auto alloc_type = GENERATE(LinearAllocs::hipMalloc);
    LinearAllocGuard<int> flag1(alloc_type, sizeof(int));
    LinearAllocGuard<int> flag2(alloc_type, sizeof(int));
    TestKernel<operation, memory_scope>
        <<<blocks, threads>>>(flag1.ptr(), flag2.ptr(), counter.ptr());
  }

  if (memory_scope != __HIP_MEMORY_SCOPE_AGENT && memory_scope != __HIP_MEMORY_SCOPE_SYSTEM) {
    SECTION("Shared memory") {
      TestKernel<operation, memory_scope><<<blocks, threads>>>(nullptr, nullptr, counter.ptr());
    }
  }

  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(counter.ptr()[0] != 0);
}

}  // namespace SequentialConsistency