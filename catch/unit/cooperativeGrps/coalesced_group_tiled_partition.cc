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

#include "cooperative_groups_common.hh"
#include "cpu_grid.h"

#include <bitset>
#include <optional>
#include <resource_guards.hh>
#include <utils.hh>

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

/**
 * @addtogroup coalesced_group_tile coalesced_group_tile
 * @{
 * @ingroup CooperativeGroupsTest
 * Contains unit tests for partitioning of coalesced groups into tiled partitions
 */

namespace cg = cooperative_groups;

template <size_t warp_size> __device__ bool deactivate_thread(uint64_t* active_masks) {
  const auto warp = cg::tiled_partition<warp_size>(cg::this_thread_block());
  const auto block = cg::this_thread_block();
  const auto warps_per_block = (block.size() + warp_size - 1) / warp_size;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warp.size();

  return !(active_masks[idx] & (1u << warp.thread_rank()));
}

template <size_t warp_size>
__global__ void coalesced_group_tiled_partition_size_getter(uint64_t* active_masks,
                                                            unsigned int tile_size,
                                                            unsigned int* sizes) {
  if (deactivate_thread<warp_size>(active_masks)) {
    return;
  }
  sizes[thread_rank_in_grid()] = cg::tiled_partition(cg::coalesced_threads(), tile_size).size();
}

template <size_t warp_size>
__global__ void coalesced_group_tiled_partition_thread_rank_getter(uint64_t* active_masks,
                                                                   unsigned int tile_size,
                                                                   unsigned int* sizes) {
  if (deactivate_thread<warp_size>(active_masks)) {
    return;
  }

  sizes[thread_rank_in_grid()] =
      cg::tiled_partition(cg::coalesced_threads(), tile_size).thread_rank();
}

static inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(11);
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

template <size_t warp_size> static auto coalesce_threads(const uint64_t mask) {
  std::tuple<std::array<unsigned int, warp_size>, unsigned int> res;
  auto& [threads, count] = res;

  count = 0u;
  for (auto i = 0u; i < warp_size; ++i) {
    if (mask & (1u << i)) {
      threads[count++] = i;
    }
  }

  return res;
}


template <typename T, size_t warp_size>
__global__ void coalesced_group_tiled_partition_shfl_up(uint64_t* active_masks, T* const out,
                                                        const unsigned int tile_size,
                                                        const unsigned int delta) {
  if (deactivate_thread<warp_size>(active_masks)) {
    return;
  }
  const auto warp = cg::tiled_partition<warp_size>(cg::this_thread_block());
  T var = static_cast<T>(warp.thread_rank());

  const auto tile = cg::tiled_partition(cg::coalesced_threads(), tile_size);
  out[thread_rank_in_grid()] = tile.shfl_up(var, delta);
}


template <typename T> static void CoalescedGroupTiledPartitonShflUpTestImpl() {
  const auto tile_size = GENERATE(2u, 4u, 8u, 16u, 32u);
  INFO("Tile size: " << tile_size);
  auto blocks = GenerateBlockDimensionsForShuffle();
  auto threads = GenerateThreadDimensionsForShuffle();
  INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
  INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
  const auto delta = GENERATE_COPY(range(0u, tile_size));
  INFO("Delta: " << delta);
  CPUGrid grid(blocks, threads);

  const auto alloc_size = grid.thread_count_ * sizeof(T);
  LinearAllocGuard<T> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
  LinearAllocGuard<T> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

  const auto warps_in_block = (grid.threads_in_block_count_ + kWarpSize - 1) / kWarpSize;
  const auto warps_in_grid = warps_in_block * grid.block_count_;
  LinearAllocGuard<uint64_t> active_masks_dev(LinearAllocs::hipMalloc,
                                              warps_in_grid * sizeof(uint64_t));
  LinearAllocGuard<uint64_t> active_masks(LinearAllocs::hipHostMalloc,
                                          warps_in_grid * sizeof(uint64_t));

  std::generate(active_masks.ptr(), active_masks.ptr() + warps_in_grid,
                [] { return GenerateRandomInteger(0u, std::numeric_limits<uint32_t>().max()); });
  HIP_CHECK(hipMemcpy(active_masks_dev.ptr(), active_masks.ptr(), warps_in_grid * sizeof(uint64_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemsetAsync(uint_arr_dev.ptr(), 0, alloc_size));
  coalesced_group_tiled_partition_shfl_up<T, kWarpSize>
      <<<blocks, threads>>>(active_masks_dev.ptr(), uint_arr_dev.ptr(), tile_size, delta);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  const auto tail = warps_in_block * kWarpSize - grid.threads_in_block_count_;

  for (auto i = 0u; i < warps_in_grid; ++i) {
    auto current_warp_mask = active_masks.ptr()[i];
    const auto shift_amount =
        (tail + 32 * TestContext::get().isNvidia()) * !((i + 1) % warps_in_block);
    current_warp_mask = (current_warp_mask << shift_amount) >> shift_amount;

    const auto [active_threads, active_thread_count] =
        coalesce_threads<kWarpSize>(current_warp_mask);

    // Step tile-sized window over active threads
    for (auto t = 0u; t < active_thread_count; t += tile_size) {
      const auto window_start = t + delta;
      const auto window_end = t + tile_size;
      // Iterate through window
      for (auto k = window_start; k < window_end && k < active_thread_count; ++k) {
        const auto tails = tail * (i / warps_in_block) * (i >= warps_in_block);
        const auto global_thread_idx = i * kWarpSize + active_threads[k] - tails;
        const auto expected_val = active_threads[k - delta];
        const auto actual_val = uint_arr.ptr()[global_thread_idx];
        INFO("global index: " << global_thread_idx);
        if (actual_val != expected_val) {
          REQUIRE(actual_val == expected_val);
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE("Unit_Coalesced_Group_Tiled_Partition_Shfl_Up_Positive_Basic", "", uint8_t,
                   uint16_t, uint32_t) {
  CoalescedGroupTiledPartitonShflUpTestImpl<TestType>();
}
