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
}