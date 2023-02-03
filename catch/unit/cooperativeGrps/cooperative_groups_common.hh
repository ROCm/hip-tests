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
#include <hip/hip_cooperative_groups.h>

namespace {
#if HT_AMD
constexpr size_t kWarpSize = 64;
#else
constexpr size_t kWarpSize = 32;
#endif
}  // namespace

#define ASSERT_EQUAL(lhs, rhs) HIP_ASSERT(lhs == rhs)
#define ASSERT_LE(lhs, rhs) HIPASSERT(lhs <= rhs)
#define ASSERT_GE(lhs, rhs) HIPASSERT(lhs >= rhs)

constexpr int MaxGPUs = 8;

template <typename T> void compareResults(T* cpu, T* gpu, int size) {
  for (unsigned int i = 0; i < size / sizeof(T); i++) {
    if (cpu[i] != gpu[i]) {
      INFO("Results do not match at index " << i);
      REQUIRE(cpu[i] == gpu[i]);
    }
  }
}

// Search if the sum exists in the expected results array
template <typename T> void verifyResults(T* hPtr, T* dPtr, int size) {
  int i = 0, j = 0;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      if (hPtr[i] == dPtr[j]) {
        break;
      }
    }
    if (j == size) {
      INFO("Result verification failed!");
      REQUIRE(j != size);
    }
  }
}

inline bool operator==(const dim3& l, const dim3& r) {
  return l.x == r.x && l.y == r.y && l.z == r.z;
}

inline bool operator!=(const dim3& l, const dim3& r) { return !(l == r); }

template <typename T, typename F>
static inline void ArrayAllOf(const T* arr, uint32_t count, F value_gen) {
  for (auto i = 0u; i < count; ++i) {
    const std::optional<T> expected_val = value_gen(i);
    if (!expected_val.has_value()) continue;
    // Using require on every iteration leads to a noticeable performance loss on large arrays,
    // even when the require passes.
    if (arr[i] != expected_val.value()) {
      INFO("Mismatch at index: " << i);
      REQUIRE(arr[i] == expected_val.value());
    }
  }
}

__device__ inline unsigned int thread_rank_in_grid() {
  const auto block_size = blockDim.x * blockDim.y * blockDim.z;
  const auto block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto thread_rank_in_block =
      (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  return block_rank_in_grid * block_size + thread_rank_in_block;
}

static __device__ void busy_wait(unsigned long long wait_period) {
  unsigned long long time_diff = 0;
  unsigned long long last_clock = clock64();
  while (time_diff < wait_period) {
    unsigned long long cur_clock = clock64();
    if (cur_clock > last_clock) {
      time_diff += (cur_clock - last_clock);
    }
    last_clock = cur_clock;
  }
}

inline dim3 GenerateThreadDimensions() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const auto multipliers = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
                            1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5};
  return GENERATE_COPY(
      dim3(1, 1, 1), dim3(props.maxThreadsDim[0], 1, 1), dim3(1, props.maxThreadsDim[1], 1),
      dim3(1, 1, props.maxThreadsDim[2]),
      map([max = props.maxThreadsDim[0]](
              double i) { return dim3(std::min(static_cast<int>(i * kWarpSize), max), 1, 1); },
          values(multipliers)),
      map([max = props.maxThreadsDim[1]](
              double i) { return dim3(1, std::min(static_cast<int>(i * kWarpSize), max), 1); },
          values(multipliers)),
      map([max = props.maxThreadsDim[2]](
              double i) { return dim3(1, 1, std::min(static_cast<int>(i * kWarpSize), max)); },
          values(multipliers)),
      dim3(16, 8, 8), dim3(32, 32, 1), dim3(64, 8, 2), dim3(16, 16, 3), dim3(kWarpSize - 1, 3, 3),
      dim3(kWarpSize + 1, 3, 3));
}

inline dim3 GenerateBlockDimensions() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const auto multipliers = {0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9, 2.0, 3.0, 4.0};
  return GENERATE_COPY(dim3(1, 1, 1),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(static_cast<int>(i * sm), 1, 1); },
                           values(multipliers)),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(1, static_cast<int>(i * sm), 1); },
                           values(multipliers)),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(1, 1, static_cast<int>(i * sm)); },
                           values(multipliers)),
                       dim3(5, 5, 5));
}

inline dim3 GenerateThreadDimensionsForShuffle() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const auto multipliers = {0.5, 0.9, 1.0, 1.5, 2.0};
  return GENERATE_COPY(
      dim3(1, 1, 1), dim3(props.maxThreadsDim[0], 1, 1), dim3(1, props.maxThreadsDim[1], 1),
      dim3(1, 1, props.maxThreadsDim[2]),
      map([max = props.maxThreadsDim[0]](
              double i) { return dim3(std::min(static_cast<int>(i * kWarpSize), max), 1, 1); },
          values(multipliers)),
      map([max = props.maxThreadsDim[1]](
              double i) { return dim3(1, std::min(static_cast<int>(i * kWarpSize), max), 1); },
          values(multipliers)),
      map([max = props.maxThreadsDim[2]](
              double i) { return dim3(1, 1, std::min(static_cast<int>(i * kWarpSize), max)); },
          values(multipliers)),
      dim3(16, 8, 8), dim3(32, 32, 1), dim3(64, 8, 2), dim3(16, 16, 3), dim3(kWarpSize - 1, 3, 3),
      dim3(kWarpSize + 1, 3, 3));
}

inline dim3 GenerateBlockDimensionsForShuffle() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  const auto multipliers = {0.5, 1.0};
  return GENERATE_COPY(dim3(1, 1, 1),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(static_cast<int>(i * sm), 1, 1); },
                           values(multipliers)),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(1, static_cast<int>(i * sm), 1); },
                           values(multipliers)),
                       map([sm = props.multiProcessorCount](
                               double i) { return dim3(1, 1, static_cast<int>(i * sm)); },
                           values(multipliers)),
                       dim3(5, 5, 5));
}

template <class T> bool CheckDimensions(unsigned int device, T kernel, dim3 blocks, dim3 threads) {
  hipDeviceProp_t props;
  int max_blocks_per_sm = 0;
  int num_sm = 0;
  HIP_CHECK(hipSetDevice(device));
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel,
                                                         threads.x * threads.y * threads.z, 0));

  HIP_CHECK(hipGetDeviceProperties(&props, device));
  num_sm = props.multiProcessorCount;

  if ((blocks.x * blocks.y * blocks.z) > max_blocks_per_sm * num_sm) {
    return false;
  }

  return true;
}