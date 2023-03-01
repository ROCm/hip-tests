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
#include <hip/hip_cooperative_groups.h>

enum class AtomicScopes { device, system };

template <typename T, AtomicScopes scope> __device__ T perform_atomic_exch(T* address, T val) {
  if constexpr (scope == AtomicScopes::device) {
    return atomicExch(address, val);
  } else if (scope == AtomicScopes::system) {
    return atomicExch_system(address, val);
  }
}

template <typename T, bool use_shared_mem, AtomicScopes scope>
__global__ void atomic_exch_kernel_compile_time(T* const global_mem, T* const old_vals) {
  __shared__ T shared_mem;

  const auto tid = cooperative_groups::this_grid().thread_rank();

  T* const mem = use_shared_mem ? &shared_mem : global_mem;

  if constexpr (use_shared_mem) {
    if (tid == 0) mem[0] = global_mem[0];
    __syncthreads();
  }

  old_vals[tid] = perform_atomic_exch<T, scope>(mem, static_cast<T>(tid + 1));

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid == 0) global_mem[0] = mem[0];
  }
}

template <typename T>
__device__ T* pitched_offset(T* const ptr, const unsigned int pitch, const unsigned int idx) {
  const auto byte_ptr = reinterpret_cast<uint8_t*>(ptr);
  return reinterpret_cast<T*>(byte_ptr + idx * pitch);
}

template <typename T, bool use_shared_mem, AtomicScopes scope>
__global__ void atomic_exch_kernel(T* const global_mem, T* const old_vals, const unsigned int width,
                                   const unsigned pitch, const T base_val = 0) {
  extern __shared__ uint8_t shared_mem[];

  const auto tid = cooperative_groups::this_grid().thread_rank();

  T* const mem = use_shared_mem ? reinterpret_cast<T*>(shared_mem) : global_mem;

  if constexpr (use_shared_mem) {
    if (tid < width) {
      const auto target = pitched_offset(mem, pitch, tid);
      *target = *pitched_offset(global_mem, pitch, tid);
    };
    __syncthreads();
  }

  old_vals[tid] = perform_atomic_exch<T, scope>(pitched_offset(mem, pitch, tid % width),
                                                base_val + static_cast<T>(tid + width));

  if constexpr (use_shared_mem) {
    __syncthreads();
    if (tid < width) {
      const auto target = pitched_offset(global_mem, pitch, tid);
      *target = *pitched_offset(mem, pitch, tid);
    };
  }
}


template <typename TestType, bool use_shared_mem, AtomicScopes scope>
void AtomicExchSameAddress(const dim3 blocks, const dim3 threads, const LinearAllocs alloc_type) {
  LinearAllocGuard<TestType> mem_dev(alloc_type, sizeof(TestType));

  const auto thread_count = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
  const auto old_vals_alloc_size = thread_count * sizeof(TestType);
  LinearAllocGuard<TestType> old_vals_dev(LinearAllocs::hipMalloc, old_vals_alloc_size);
  std::vector<TestType> old_vals(thread_count + 1);


  HIP_CHECK(hipMemset(mem_dev.ptr(), 0, sizeof(TestType)));
  atomic_exch_kernel_compile_time<TestType, use_shared_mem, scope>
      <<<blocks, threads>>>(mem_dev.ptr(), old_vals_dev.ptr());
  HIP_CHECK(
      hipMemcpy(old_vals.data(), old_vals_dev.ptr(), old_vals_alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(old_vals.data() + thread_count, mem_dev.ptr(), sizeof(TestType),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  // Every thread will exchange its grid-wide linear id into a target location within mem_dev,
  // receiving back the value previously present therein. This previous value is written to
  // old_vals_dev.
  // old_vals_dev will not contain values that the final scheduled warp exchanged into mem_dev, but
  // mem_dev obviously will.
  // Given that mem_dev initially contains values in the range [0, width) and that the maximum value
  // the final thread shall write is thread_count + width - 1, presuming correct operation of
  // atomicExch, the union of mem_dev and old_vals_dev shall contain values in the range
  //[0, thread_count + width)
  std::sort(old_vals.begin(), old_vals.end());
  for (auto i = 0u; i < old_vals.size(); ++i) {
    REQUIRE(i == old_vals[i]);
  }
}

template <typename TestType, AtomicScopes scope> void AtomicExchSameAddressTest() {
  const auto threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(31));

  SECTION("Global memory") {
    const auto blocks = GENERATE(dim3(20));
    using LA = LinearAllocs;
    const auto allocation_type =
        GENERATE(LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister);
    AtomicExchSameAddress<TestType, false, AtomicScopes::device>(blocks, threads, allocation_type);
  }

  SECTION("Shared memory") {
    const auto blocks = dim3(1);
    AtomicExchSameAddress<TestType, true, AtomicScopes::device>(blocks, threads,
                                                                LinearAllocs::hipMalloc);
  }
}

struct AtomicExchParams {
  dim3 blocks;
  dim3 threads;
  unsigned int num_devices;
  unsigned int kernel_count;
  unsigned int width;
  unsigned int pitch;
  LinearAllocs alloc_type;
};

template <typename TestType, bool use_shared_mem, AtomicScopes scope>
void AtomicExch(const AtomicExchParams& p) {
  const auto thread_count =
      p.blocks.x * p.blocks.y * p.blocks.z * p.threads.x * p.threads.y * p.threads.z;

  const auto old_vals_alloc_size = p.kernel_count * thread_count * sizeof(TestType);
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
  LinearAllocGuard<TestType> mem_dev(p.alloc_type, mem_alloc_size);

  std::vector<TestType> old_vals(p.num_devices * p.kernel_count * thread_count + p.width);
  std::iota(old_vals.begin(), old_vals.begin() + p.width, 0);

  HIP_CHECK(hipMemcpy2D(mem_dev.ptr(), p.pitch, old_vals.data(), sizeof(TestType), sizeof(TestType),
                        p.width, hipMemcpyHostToDevice));

  const auto shared_mem_size = use_shared_mem ? mem_alloc_size : 0u;
  for (auto i = 0u; i < p.num_devices; ++i) {
    const auto device_offset = i * p.kernel_count * thread_count;
    for (auto j = 0u; j < p.kernel_count; ++j) {
      const auto& stream = streams[i * p.kernel_count + j].stream();
      const auto kern_offset = j * thread_count;
      const auto old_vals = old_vals_devs[i].ptr() + kern_offset;
      atomic_exch_kernel<TestType, use_shared_mem, scope>
          <<<p.blocks, p.threads, shared_mem_size, stream>>>(mem_dev.ptr(), old_vals, p.width,
                                                             p.pitch, device_offset + kern_offset);
    }
  }

  for (auto i = 0u; i < p.num_devices; ++i) {
    const auto device_offset = i * p.kernel_count * thread_count;
    HIP_CHECK(hipMemcpy(old_vals.data() + device_offset, old_vals_devs[i].ptr(),
                        old_vals_alloc_size, hipMemcpyDeviceToHost));
  }
  HIP_CHECK(hipMemcpy2D(old_vals.data() + p.num_devices * p.kernel_count * thread_count,
                        sizeof(TestType), mem_dev.ptr(), p.pitch, sizeof(TestType), p.width,
                        hipMemcpyDeviceToHost));

  std::sort(old_vals.begin(), old_vals.end());
  for (auto i = 0u; i < old_vals.size(); ++i) {
    REQUIRE(i == old_vals[i]);
  }
}

template <typename TestType, AtomicScopes scope>
void AtomicExchSingleDeviceSingleKernelTest(const unsigned int width, const unsigned int pitch) {
  AtomicExchParams params;
  params.num_devices = 1;
  params.kernel_count = 1;
  params.threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(31));
  params.width = width;
  params.pitch = pitch;

  SECTION("Global memory") {
    params.blocks = GENERATE(dim3(40));
    using LA = LinearAllocs;
    for (const auto alloc_type :
         {LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
      params.alloc_type = alloc_type;
      DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
        AtomicExch<TestType, false, scope>(params);
      }
    }
  }

  SECTION("Shared memory") {
    params.blocks = dim3(1);
    params.alloc_type = LinearAllocs::hipMalloc;
    AtomicExch<TestType, true, scope>(params);
  }
}

template <typename TestType, AtomicScopes scope>
void AtomicExchSingleDeviceMultipleKernelTest(const unsigned int kernel_count,
                                              const unsigned int width, const unsigned int pitch) {
  int concurrent_kernels = 0;
  HIP_CHECK(hipDeviceGetAttribute(&concurrent_kernels, hipDeviceAttributeConcurrentKernels, 0));
  if (!concurrent_kernels) {
    HipTest::HIP_SKIP_TEST("Test requires support for concurrent kernel execution");
    return;
  }

  AtomicExchParams params;
  params.num_devices = 1;
  params.kernel_count = kernel_count;
  params.blocks = GENERATE(dim3(40));
  params.threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(31));
  params.width = width;
  params.pitch = pitch;

  using LA = LinearAllocs;
  for (const auto alloc_type :
       {LA::hipMalloc, LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
    params.alloc_type = alloc_type;
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      AtomicExch<TestType, false, scope>(params);
    }
  }
}

template <typename TestType>
void AtomicExchMultipleDeviceMultipleKernelTest(const unsigned int num_devices,
                                                const unsigned int kernel_count,
                                                const unsigned int width,
                                                const unsigned int pitch) {
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

  AtomicExchParams params;
  params.num_devices = num_devices;
  params.kernel_count = kernel_count;
  params.blocks = GENERATE(dim3(40));
  params.threads = GENERATE(dim3(1024), dim3(1023), dim3(511), dim3(17), dim3(31));
  params.width = width;
  params.pitch = pitch;

  using LA = LinearAllocs;
  for (const auto alloc_type : {LA::hipHostMalloc, LA::hipMallocManaged, LA::mallocAndRegister}) {
    params.alloc_type = alloc_type;
    DYNAMIC_SECTION("Allocation type: " << to_string(alloc_type)) {
      AtomicExch<TestType, false, AtomicScopes::system>(params);
    }
  }
}
