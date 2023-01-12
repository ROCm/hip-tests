/*
Copyright (c) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
// Test Description:
/* This test implements sum reduction kernel, first with each threads own rank
   as input and comparing the sum with expected sum output derieved from n(n-1)/2
   formula.
   This sample tests functionality of intrinsics provided by thread_block_tile type,
   shfl_down and shfl_xor.
*/

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#include "cooperative_groups_common.hh"

namespace cg = cooperative_groups;

static constexpr size_t kWaveSize = 32;

__device__ int reduction_kernel_shfl_down(cg::coalesced_group const& g, int val) {
  int sz = g.size();

  for (int i = sz / 2; i > 0; i >>= 1) {
    val += g.shfl_down(val, i);
  }

  // Choose the 0'th indexed thread that holds the reduction value to return
  if (g.thread_rank() == 0) {
    return val;
  }
  // Rest of the threads return no useful values
  else {
    return -1;
  }
}

__global__ void kernel_shfl_down (int * dPtr, int *dResults, int lane_delta, int cg_sizes) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id % cg_sizes == 0) {
    cg::coalesced_group const& g = cg::coalesced_threads();
    int rank = g.thread_rank();
    int val = dPtr[rank];
    dResults[rank] = g.shfl_down(val, lane_delta);
    return;
  }
}

__global__ void kernel_cg_partition_shfl_down(int* result, unsigned int tile_size, int cg_sizes) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id % cg_sizes == 0) {
    cg::coalesced_group  thread_block_CG_ty = cg::coalesced_threads();
    int input, output_sum, expected_sum;

    // Choose a leader thread to print the results
    if (thread_block_CG_ty.thread_rank() == 0) {
      printf(" Creating %d groups, of tile size %d threads:\n\n",
             (int)thread_block_CG_ty.size() / tile_size, tile_size);
    }

    thread_block_CG_ty.sync();

    cg::coalesced_group tiled_part = cg::tiled_partition(thread_block_CG_ty, tile_size);
    int threadRank = tiled_part.thread_rank();

    input = tiled_part.thread_rank();

    // (n-1)(n)/2
    expected_sum = ((tile_size - 1) * tile_size / 2);

    output_sum = reduction_kernel_shfl_down(tiled_part, input);

    if (tiled_part.thread_rank() == 0) {
      int parent_rank = thread_block_CG_ty.thread_rank();
      printf(
          "   Sum of all ranks %d..%d in this tiled_part group using shfl_down is %d (expected "
          "%d)\n",
          parent_rank, parent_rank + static_cast<int>(tiled_part.size()) - 1, output_sum, expected_sum);
      result[thread_block_CG_ty.thread_rank() / (tile_size)] = output_sum;
    }
    return;
  }
}

static void test_group_partition(unsigned int tile_size) {
  int block_size = 1;
  int threads_per_blk = kWaveSize;

  unsigned int cg_size = GENERATE(1, 2, 4);

  int num_tiles = ((block_size * threads_per_blk) / cg_size) / tile_size;
  int expected_sum = ((tile_size - 1) * tile_size / 2);
  int* expected_result = new int[num_tiles];

  // num_tiles = 0 when partitioning is possible. The below statement is to avoid
  // out-of-bounds error and still evaluate failure case.
  //num_tiles = (num_tiles == 0) ? 1 : num_tiles;
  if (num_tiles == 0) return;

  for (int i = 0; i < num_tiles; i++) {
    expected_result[i] = expected_sum;
  }

  int* result_dev = NULL;
  int* result_host = NULL;

  HIP_CHECK(hipHostMalloc(&result_host, num_tiles * sizeof(int), hipHostMallocDefault));
  memset(result_host, 0, num_tiles * sizeof(int));

  HIP_CHECK(hipMalloc(&result_dev, num_tiles * sizeof(int)));


  // Launch Kernel
  hipLaunchKernelGGL(kernel_cg_partition_shfl_down, block_size, threads_per_blk,
                    threads_per_blk * sizeof(int), 0, result_dev, tile_size, cg_size);
  HIP_CHECK(hipGetLastError()); 
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(result_host, result_dev, sizeof(int) * num_tiles, hipMemcpyDeviceToHost));

  verifyResults(expected_result, result_host, num_tiles);

  // Free all allocated memory on host and device
  HIP_CHECK(hipFree(result_dev));
  HIP_CHECK(hipHostFree(result_host));
  delete[] expected_result;
}

static void test_shfl_down() {

  unsigned int cg_size = GENERATE(1, 2, 4);

  int block_size = 1;
  int threads_per_blk = kWaveSize;

  int total_threads = block_size * threads_per_blk;
  int group_size = total_threads / cg_size;
  int group_size_in_bytes = group_size * sizeof(int);

  int* data_host = NULL;
  int* data_dev = NULL;
  int* result_dev = NULL;
  int lane_delta = rand() % group_size;
  INFO("Testing coalesced_groups shfl_down with lane_delta" << lane_delta << " and group size " << threads_per_blk << "\n");

  HIP_CHECK(hipHostMalloc(&data_host, group_size_in_bytes));
  // Fill up the array
  for (int i = 0; i < group_size; i++) {
    data_host[i] = rand() % 1000;
  }

  int* expected_result = (int*)malloc(group_size_in_bytes);
  for (int i = 0; i < group_size; i++) {
    expected_result[i] = (i + lane_delta >= group_size) ? data_host[i] : data_host[i + lane_delta];
  }

  HIP_CHECK(hipMalloc(&data_dev, group_size_in_bytes));
  HIP_CHECK(hipMalloc(&result_dev, group_size_in_bytes));

  HIP_CHECK(hipMemcpy(data_dev, data_host, group_size_in_bytes, hipMemcpyHostToDevice));
  // Launch Kernel
  hipLaunchKernelGGL(kernel_shfl_down, block_size, threads_per_blk,
                    threads_per_blk * sizeof(int), 0, data_dev, result_dev, lane_delta, cg_size);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(data_host, result_dev, group_size_in_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  compareResults(expected_result, data_host, group_size_in_bytes);

  HIP_CHECK(hipHostFree(data_host));
  HIP_CHECK(hipFree(data_dev));
  HIP_CHECK(hipFree(result_dev));
  free(expected_result);
}

TEST_CASE("Unit_hipCGCoalescedGroupTypeShfl_down") {
  // Use default device for validating the test
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  // Test shfl_down with random group sizes
  SECTION("Testing coalesced_groups shfl_down") {  
    for (int i = 0; i < 1; i++) {
      test_shfl_down();
    }
  }

  SECTION("Testing coalesced_groups partitioning and shfl_down") {
    unsigned int tile_size = GENERATE(2, 4, 8, 16, 32);
    test_group_partition(tile_size);
  }
}