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
// Test Description:
/* This test implements prefix sum(scan) kernel, first with each threads own rank
   as input and comparing the sum with expected serial summation output on CPU.

   This sample tests functionality of intrinsics provided by coalesced_group,
   shfl_up.
*/
#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#include "cooperative_groups_common.hh"

using namespace cooperative_groups;

static constexpr size_t kWaveSize = 32;

__device__ int prefix_sum_kernel(coalesced_group const& g, int val) {
  int sz = g.size();
  for (int i = 1; i < sz; i <<= 1) {
    int temp = g.shfl_up(val, i);

    if (g.thread_rank() >= i) {
      val += temp;
    }
  }
  return val;
}

__global__ void kernel_shfl_up (int * dPtr, int *dResults, int lane_delta, int cg_sizes) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id % cg_sizes == 0) {
    coalesced_group g = coalesced_threads();
    int rank = g.thread_rank();
    int val = dPtr[rank];
    dResults[rank] = g.shfl_up(val, lane_delta);
  return;
  }

}

__global__ void kernel_cg_partition_shfl_up(int* result, unsigned int tile_size, int cg_sizes) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id % cg_sizes == 0) {
    coalesced_group thread_block_CG_ty = coalesced_threads();
    int input, output_sum;

    // we pass its own thread rank as inputs
    input = thread_block_CG_ty.thread_rank();

    // Choose a leader thread to print the results
    if (thread_block_CG_ty.thread_rank() == 0) {
      printf(" Creating %d groups, of tile size %d threads:\n\n",
             (int)thread_block_CG_ty.size() / tile_size, tile_size);
    }

    thread_block_CG_ty.sync();

    coalesced_group tiled_part = tiled_partition(thread_block_CG_ty, tile_size);

    input = tiled_part.thread_rank();

    output_sum = prefix_sum_kernel(tiled_part, input);

    // Update the result array with the corresponsing prefix sum
    result[thread_block_CG_ty.thread_rank()] = output_sum;
    return;
  }
}

void serial_scan(int* ptr, int size) {
  // Fill up the array
  for (int i = 0; i < size; i++) {
    ptr[i] = i;
  }

  int acc = 0;
  for (int i = 0; i < size; i++) {
    acc = acc + ptr[i];
    ptr[i] = acc;
  }
}

static void test_group_partition(unsigned tile_size) {
  int block_size = 1;
  int threads_per_blk = kWaveSize;

  int* result_host = NULL;
  int* result_dev = NULL;
  int* cpu_prefix_sum = NULL;

  unsigned int cg_size = GENERATE(1, 2, 4);

  int data_size = block_size * threads_per_blk * sizeof(int);
  int num_tiles = ((block_size * threads_per_blk) / cg_size) / tile_size;

  if (num_tiles == 0) return;

  HIP_CHECK(hipHostMalloc(&result_host, data_size));
  HIP_CHECK(hipMalloc(&result_dev, data_size));

  // Launch Kernel
  hipLaunchKernelGGL(kernel_cg_partition_shfl_up, block_size, threads_per_blk,
                    threads_per_blk * sizeof(int), 0, result_dev, tile_size, cg_size);
  HIP_CHECK(hipGetLastError()); 
  HIP_CHECK(hipMemcpy(result_host, result_dev, data_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  cpu_prefix_sum = new int[tile_size];
  serial_scan(cpu_prefix_sum, tile_size);

  compareResults(cpu_prefix_sum, result_host, tile_size * sizeof(int));

  delete[] cpu_prefix_sum;
  HIP_CHECK(hipHostFree(result_host));
  HIP_CHECK(hipFree(result_dev));
}

static void test_shfl_up() {

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
    expected_result[i] = (i <= (lane_delta - 1)) ?  data_host[i] : data_host[i - lane_delta];
  }

  HIP_CHECK(hipMalloc(&data_dev, group_size_in_bytes));
  HIP_CHECK(hipMalloc(&result_dev, group_size_in_bytes));

  HIP_CHECK(hipMemcpy(data_dev, data_host, group_size_in_bytes, hipMemcpyHostToDevice));
  // Launch Kernel
  hipLaunchKernelGGL(kernel_shfl_up, block_size, threads_per_blk,
                    threads_per_blk * sizeof(int), 0, data_dev, result_dev, lane_delta, cg_size);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(data_host, result_dev, group_size_in_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  compareResults(data_host, expected_result, group_size_in_bytes);

  HIP_CHECK(hipHostFree(data_host));
  HIP_CHECK(hipFree(data_dev));
  HIP_CHECK(hipFree(result_dev));
  free(expected_result);
}

TEST_CASE("Unit_hipCGCoalescedGroupTypeShfl_up") {
  // Use default device for validating the test
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  SECTION("Testing coalesced_groups shfl_up") {
    for (int i = 0; i < 100; i++) {
      test_shfl_up();
    }
  }

  SECTION("Testing coalesced_groups partitioning and shfl_up") {
    unsigned int tile_size = GENERATE(2, 4, 8, 16, 32);
    test_group_partition(tile_size);
  }
}