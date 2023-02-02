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

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#include <bitset>
#include <array>

#include <resource_guards.hh>

#include "cooperative_groups_common.hh"
#include "cpu_grid.h"

/**
 * @addtogroup thread_block_tile thread_block_tile
 * @{
 * @ingroup CooperativeGroupsTest
 * Contains unit tests for all thread_block_tile APIs and dynamic block partitioning
 */

namespace cg = cooperative_groups;

template <bool dynamic, unsigned int tile_size>
__global__ void thread_block_partition_size_getter(unsigned int* sizes) {
  const auto group = cg::this_thread_block();
  if constexpr (dynamic) {
    sizes[thread_rank_in_grid()] = cg::tiled_partition(group, tile_size).size();
  } else {
    sizes[thread_rank_in_grid()] = cg::tiled_partition<tile_size>(group).size();
  }
}

template <bool dynamic, unsigned int tile_size>
__global__ void thread_block_partition_thread_rank_getter(unsigned int* thread_ranks) {
  const auto group = cg::this_thread_block();
  if constexpr (dynamic) {
    thread_ranks[thread_rank_in_grid()] = cg::tiled_partition(group, tile_size).thread_rank();
  } else {
    thread_ranks[thread_rank_in_grid()] = cg::tiled_partition<tile_size>(group).thread_rank();
  }
}

static dim3 GenerateThreadDimensions() {
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

static dim3 GenerateBlockDimensions() {
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

template <bool dynamic, size_t tile_size> void BlockPartitionGettersBasicTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto blocks = GenerateBlockDimensions();
    auto threads = GenerateThreadDimensions();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(unsigned int);
    LinearAllocGuard<unsigned int> uint_arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<unsigned int> uint_arr(LinearAllocs::hipHostMalloc, alloc_size);

    thread_block_partition_size_getter<dynamic, tile_size><<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    thread_block_partition_thread_rank_getter<dynamic, tile_size>
        <<<blocks, threads>>>(uint_arr_dev.ptr());
    HIP_CHECK(hipGetLastError());

    ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [&grid](unsigned int i) {
      if constexpr (!dynamic) {
        return tile_size;
      }

      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();

      const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
      return tile_size - tail * (rank_in_block >= (partitions_in_block - 1) * tile_size);
    });

    HIP_CHECK(hipMemcpy(uint_arr.ptr(), uint_arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ArrayAllOf(uint_arr.ptr(), grid.thread_count_, [&grid](unsigned int i) {
      return grid.thread_rank_in_block(i).value() % tile_size;
    });
  }
}

template <bool dynamic, size_t... tile_sizes> void BlockPartitionGettersBasicTest() {
  static_cast<void>((BlockPartitionGettersBasicTestImpl<dynamic, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Creates tiled partitions for each of the valid sizes{2, 4, 8, 16, 32, 64(if AMD)} and writes
 * the return values of size and thread_rank member functions to an output array that is validated
 * on the host side. Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Thread_Block_Tile_Getters_Positive_Basic") {
  BlockPartitionGettersBasicTest<false, 2, 4, 8, 16, 32>();
#if HT_AMD
  BlockPartitionGettersBasicTest<false, 64>();
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Creates tiled partitions for each of the valid sizes{2, 4, 8, 16, 32, 64(if AMD)} via the
 * dynamic tiled partition api and writes the return values of size and thread_rank member functions
 * to an output array that is validated on host. Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Thread_Block_Tile_Dynamic_Getters_Positive_Basic") {
  BlockPartitionGettersBasicTest<true, 2, 4, 8, 16, 32>();
#if HT_AMD
  BlockPartitionGettersBasicTest<true, 64>();
#endif
}

static dim3 GenerateThreadDimensionsForShuffle() {
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

static dim3 GenerateBlockDimensionsForShuffle() {
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

template <typename T, size_t tile_size>
__global__ void block_tile_shfl_up(T* const out, const unsigned int delta) {
  const auto partition = cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl_up(var, delta);
}

template <typename T, size_t tile_size> void BlockTileShflUpTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto blocks = GenerateBlockDimensionsForShuffle();
    auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    auto delta = GENERATE(range(static_cast<size_t>(0), tile_size));
    INFO("Delta: " << delta);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    block_tile_shfl_up<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), delta);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ArrayAllOf(arr.ptr(), grid.thread_count_, [delta, &grid](unsigned int i) -> std::optional<T> {
      const int rank_in_partition = grid.thread_rank_in_block(i).value() % tile_size;
      const int target = rank_in_partition - delta;
      return target < 0 ? rank_in_partition : target;
    });
  }
}

template <typename T, size_t... tile_sizes> void BlockTileShflUpTest() {
  static_cast<void>((BlockTileShflUpTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle up behavior of thread block tiles of all valid sizes{2, 4, 8, 16, 32,
 * 64(if AMD)} for delta values of [0, tile size). The test is run for all overloads of shfl_up.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Thread_Block_Tile_Shfl_Up_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  BlockTileShflUpTest<TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
  BlockTileShflUpTest<TestType, 64>();
#endif
}


template <typename T, size_t tile_size>
__global__ void block_tile_shfl_down(T* const out, const unsigned int delta) {
  const auto partition = cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl_down(var, delta);
}

template <typename T, size_t tile_size> void BlockTileShflDownTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto blocks = GenerateBlockDimensionsForShuffle();
    auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    auto delta = GENERATE(range(static_cast<size_t>(0), tile_size));
    INFO("Delta: " << delta);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    block_tile_shfl_down<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), delta);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    ArrayAllOf(arr.ptr(), grid.thread_count_, [delta, &grid](unsigned int i) -> std::optional<T> {
      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();
      const auto rank_in_group = rank_in_block % tile_size;
      const auto target = rank_in_group + delta;
      if (rank_in_block < (partitions_in_block - 1) * tile_size) {
        return target < tile_size ? target : rank_in_group;
      } else {
        // If the number of threads in a block is not an integer multiple of tile_size, the
        // final(tail end) tile will contain inactive threads.
        // Shuffling from an inactive thread returns an undefined value, accordingly threads that
        // shuffle from one must be skipped
        const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
        return target < tile_size - tail ? std::optional(target) : std::nullopt;
      }
    });
  }
}

template <typename T, size_t... tile_sizes> void BlockTileShflDownTest() {
  static_cast<void>((BlockTileShflDownTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle down behavior of thread block tiles of all valid sizes{2, 4, 8, 16,
 * 32, 64(if AMD)} for delta values of [0, tile size). The test is run for all overloads of
 * shfl_down.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Thread_Block_Tile_Shfl_Down_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  BlockTileShflDownTest<TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
  BlockTileShflDownTest<TestType, 64>();
#endif
}


template <typename T, size_t tile_size>
__global__ void block_tile_shfl_xor(T* const out, const unsigned mask) {
  const auto partition = cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl_xor(var, mask);
}

template <typename T, size_t tile_size> void BlockTileShflXORTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto blocks = GenerateBlockDimensionsForShuffle();
    auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    const auto mask = GENERATE(range(static_cast<size_t>(0), tile_size));
    INFO("Mask: 0x" << std::hex << mask);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    block_tile_shfl_xor<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), mask);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    const auto f = [mask, &grid](unsigned int i) -> std::optional<T> {
      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();
      const int rank_in_partition = rank_in_block % tile_size;
      const auto target = rank_in_partition ^ mask;
      if (rank_in_block < (partitions_in_block - 1) * tile_size) {
        return target;
      }
      const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
      return target < tile_size - tail ? std::optional(target) : std::nullopt;
    };
    ArrayAllOf(arr.ptr(), grid.thread_count_, f);
  }
}

template <typename T, size_t... tile_sizes> void BlockTileShflXORTest() {
  static_cast<void>((BlockTileShflXORTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle xor behavior of thread block tiles of all valid sizes{2, 4, 8, 16, 32,
 * 64(if AMD)} for mask values of [0, tile size). The test is run for all overloads of shfl_xor.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Thread_Block_Tile_Shfl_XOR_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  BlockTileShflXORTest<TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
  BlockTileShflXORTest<TestType, 64>();
#endif
}

template <typename T, size_t tile_size>
__global__ void block_tile_shfl(T* const out, uint8_t* target_lanes) {
  const auto partition = cg::tiled_partition<tile_size>(cg::this_thread_block());
  T var = static_cast<T>(partition.thread_rank());
  out[thread_rank_in_grid()] = partition.shfl(var, target_lanes[partition.thread_rank()]);
}

static inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(11);
  return mt;
}

template <typename T> static inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

template <typename T, size_t tile_size> void BlockTileShflTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    auto blocks = GenerateBlockDimensionsForShuffle();
    auto threads = GenerateThreadDimensionsForShuffle();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    auto run_repetitions = GENERATE(range(0, 5));
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);

    LinearAllocGuard<uint8_t> target_lanes_dev(LinearAllocs::hipMalloc,
                                               tile_size * sizeof(uint8_t));
    LinearAllocGuard<uint8_t> target_lanes(LinearAllocs::hipHostMalloc,
                                           tile_size * sizeof(uint8_t));
    std::generate(target_lanes.ptr(), target_lanes.ptr() + tile_size,
                  [] { return GenerateRandomInteger(0, static_cast<int>(2 * tile_size)); });

    HIP_CHECK(hipMemcpy(target_lanes_dev.ptr(), target_lanes.ptr(), tile_size * sizeof(uint8_t),
                        hipMemcpyHostToDevice));
    block_tile_shfl<T, tile_size><<<blocks, threads>>>(arr_dev.ptr(), target_lanes_dev.ptr());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    const auto f = [&target_lanes, &grid](unsigned int i) -> std::optional<T> {
      const auto partitions_in_block = (grid.threads_in_block_count_ + tile_size - 1) / tile_size;
      const auto rank_in_block = grid.thread_rank_in_block(i).value();
      const int rank_in_partition = rank_in_block % tile_size;
      const auto target = target_lanes.ptr()[rank_in_partition] % tile_size;
      if (rank_in_block < (partitions_in_block - 1) * tile_size) {
        return target;
      }
      const auto tail = partitions_in_block * tile_size - grid.threads_in_block_count_;
      return target < tile_size - tail ? std::optional(target) : std::nullopt;
    };
    ArrayAllOf(arr.ptr(), grid.thread_count_, f);
  }
}

template <typename T, size_t... tile_sizes> void BlockTileShflTest() {
  static_cast<void>((BlockTileShflTestImpl<T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Validates the shuffle behavior of thread block tiles of all valid sizes{2, 4, 8, 16, 32,
 * 64(if AMD)} for generated shuffle target lanes. The test is run for all overloads of shfl. Test
 * source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Thread_Block_Tile_Shfl_Positive_Basic", "", int, unsigned int, long,
                   unsigned long, long long, unsigned long long, float, double) {
  BlockTileShflTest<TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
  BlockTileShflTest<TestType, 64>();
#endif
}


template <bool use_global, size_t tile_size, typename T>
__global__ void block_tile_sync_check(T* global_data, unsigned int* wait_modifiers) {
  extern __shared__ uint8_t shared_data[];
  T* const data = use_global ? global_data : reinterpret_cast<T*>(shared_data);
  const auto tid = cg::this_grid().thread_rank();
  const auto block = cg::this_thread_block();
  const auto partition = cg::tiled_partition<tile_size>(cg::this_thread_block());

  const auto data_idx = [&block](unsigned int i) { return use_global ? i : (i % block.size()); };

  const auto partitions_in_block = (block.size() + partition.size() - 1) / partition.size();
  const auto partition_rank = block.thread_rank() / partition.size();
  const auto tail = partitions_in_block * partition.size() - block.size();
  const auto window_size = partition.size() - tail * (partition_rank == partitions_in_block - 1);

  const auto block_base_idx = tid / block.size() * block.size();
  const auto tile_base_idx = block_base_idx + partition_rank * partition.size();

  const auto wait_modifier = wait_modifiers[tid];
  busy_wait(wait_modifier);
  data[data_idx(tid)] = partition.thread_rank();
  partition.sync();
  bool valid = true;
  for (auto i = 0; i < window_size; ++i) {
    const auto expected = (partition.thread_rank() + i) % window_size;

    if (!(valid &= (data[data_idx(tile_base_idx + expected)] == expected))) {
      break;
    }
  }
  partition.sync();
  data[data_idx(tid)] = valid;
  if constexpr (!use_global) {
    global_data[tid] = data[data_idx(tid)];
  }
}

template <bool global_memory, typename T, size_t tile_size> void BlockTileSyncTestImpl() {
  DYNAMIC_SECTION("Tile size: " << tile_size) {
    const auto randomized_run_count = GENERATE(range(0, 1));
    auto blocks = GenerateBlockDimensions();
    auto threads = GenerateThreadDimensions();
    INFO("Grid dimensions: x " << blocks.x << ", y " << blocks.y << ", z " << blocks.z);
    INFO("Block dimensions: x " << threads.x << ", y " << threads.y << ", z " << threads.z);
    CPUGrid grid(blocks, threads);

    const auto alloc_size = grid.thread_count_ * sizeof(T);
    const auto alloc_size_per_block = alloc_size / grid.block_count_;
    int max_shared_mem_per_block = 0;
    HIP_CHECK(hipDeviceGetAttribute(&max_shared_mem_per_block,
                                    hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
    if (!global_memory && (max_shared_mem_per_block < alloc_size_per_block)) {
      return;
    }

    LinearAllocGuard<T> arr_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<T> arr(LinearAllocs::hipHostMalloc, alloc_size);
    LinearAllocGuard<unsigned int> wait_modifiers_dev(LinearAllocs::hipMalloc,
                                                      grid.thread_count_ * sizeof(unsigned int));
    LinearAllocGuard<unsigned int> wait_modifiers(LinearAllocs::hipHostMalloc,
                                                  grid.thread_count_ * sizeof(unsigned int));
    if (randomized_run_count != 0) {
      std::generate(wait_modifiers.ptr(), wait_modifiers.ptr() + grid.thread_count_,
                    [] { return GenerateRandomInteger(0u, 1500u); });
    } else {
      std::fill_n(wait_modifiers.ptr(), grid.thread_count_, 0u);
    }

    const auto shared_memory_size = global_memory ? 0u : alloc_size_per_block;
    HIP_CHECK(hipMemcpy(wait_modifiers_dev.ptr(), wait_modifiers.ptr(),
                        grid.thread_count_ * sizeof(unsigned int), hipMemcpyHostToDevice));

    block_tile_sync_check<global_memory, tile_size>
        <<<blocks, threads, shared_memory_size>>>(arr_dev.ptr(), wait_modifiers_dev.ptr());
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(arr.ptr(), arr_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(
        std::all_of(arr.ptr(), arr.ptr() + grid.thread_count_, [](unsigned int e) { return e; }));
  }
}

template <bool global_memory, typename T, size_t... tile_sizes> void BlockTileSyncTest() {
  static_cast<void>((BlockTileSyncTestImpl<global_memory, T, tile_sizes>(), ...));
}

/**
 * Test Description
 * ------------------------
 *    - Launches a kernel wherein blocks are divided into tiled partitions(size of 2, 4, 8, 16, 32,
 * 64 if AMD) and every thread writes its intra-tile rank into an array slot determined by its
 * grid-wide linear index. The array is either in global or dynamic shared memory based on a compile
 * time switch, and the test is run for arrays of 1, 2, and 4 byte elements. Before the write each
 * thread executes a busy wait loop for a random amount of clock cycles, the amount being read from
 * an input array. After the write a tile-wide sync is performed and each thread validates that it
 * can read the expected values that other threads within the same tile have written to their
 * respective array slots. Each thread begins the validation from a given offset from its own index.
 * For the first run of the test, all the offsets are zero, so memory reads should be coalesced as
 * adjacent threads read from adjacent memory locations. On subsequent runs the offsets are
 * randomized for each thread, leading to non-coalesced reads and cache thrashing.
 * Test source
 * ------------------------
 *    - unit/cooperativeGrps/thread_block_tile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Thread_Block_Tile_Sync_Positive_Basic", "", uint8_t, uint16_t, uint32_t) {
  SECTION("Global memory") {
    BlockTileSyncTest<true, TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
    BlockTileSyncTest<true, TestType, 64>();
#endif
  }
  SECTION("Shared memory") {
    BlockTileSyncTest<false, TestType, 2, 4, 8, 16, 32>();
#if HT_AMD
    BlockTileSyncTest<true, TestType, 64>();
#endif
  }
}
