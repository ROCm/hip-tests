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
#include <hip/hip_fp16.h>

static __device__ bool deactivate_thread(const uint64_t* const active_masks) {
  const auto warp =
      cooperative_groups::tiled_partition(cooperative_groups::this_thread_block(), warpSize);
  const auto block = cooperative_groups::this_thread_block();
  const auto warps_per_block = (block.size() + warpSize - 1) / warpSize;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warpSize;

  return !(active_masks[idx] & (static_cast<uint64_t>(1) << warp.thread_rank()));
}

static inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(std::random_device{}());
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

template <typename T> static inline T GenerateRandomReal(const T min, const T max) {
  std::uniform_real_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

inline int generate_width(int warp_size) {
  int exponent = 0;
  while (warp_size >>= 1) {
    ++exponent;
  }

  return GENERATE_COPY(map([](int e) { return 1 << e; }, range(1, exponent + 1)));
}

inline uint64_t get_active_mask(unsigned int warp_id, unsigned int warp_size) {
  uint64_t active_mask = 0;
  switch (warp_id % 5) {
    case 0:  // even threads in the warp
      active_mask = 0xAAAAAAAAAAAAAAAA;
      break;
    case 1:  // odd threads in the warp
      active_mask = 0x5555555555555555;
      break;
    case 2:  // first half of the warp
      for (int i = 0; i < warp_size / 2; i++) {
        active_mask = active_mask | (static_cast<uint64_t>(1) << i);
      }
      break;
    case 3:  // second half of the warp
      for (int i = warp_size / 2; i < warp_size; i++) {
        active_mask = active_mask | (static_cast<uint64_t>(1) << i);
      }
      break;
    case 4:  // all threads
      active_mask = 0xFFFFFFFFFFFFFFFF;
      break;
  }
  return active_mask;
}
