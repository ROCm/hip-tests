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
#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <cstdlib>

namespace cg = cooperative_groups;

/* Parallel reduce kernel.
 *
 * Step complexity: O(log n)
 * Work complexity: O(n)
 *
 * Note: This kernel works only with power of 2 input arrays.
 */
__device__ int reduction_kernel(cg::thread_group g, int* x, int val) {
  int lane = g.thread_rank();
  int sz = g.size();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    // use lds to store the temporary result
    x[lane] = val;
    // Ensure all the stores are completed.
    g.sync();

    if (lane < i) {
      val += x[lane + i];
    }
    // It must work on one tiled thread group at a time,
    // and it must make sure all memory operations are
    // completed before moving to the next stride.
    // sync() here just does that.
    g.sync();
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

template <unsigned int tile_size>
__global__ void kernel_cg_group_partition_static(int* result, bool is_global_mem, int* global_mem) {
  cg::thread_block thread_block_CG_ty = cg::this_thread_block();

  int* workspace = NULL;

  if (is_global_mem) {
    workspace = global_mem;
  } else {
    // Declare a shared memory
    extern __shared__ int sharedMem[];
    workspace = sharedMem;
  }

  int input, output_sum;

  // we pass its own thread rank as inputs
  input = thread_block_CG_ty.thread_rank();

  output_sum = reduction_kernel(thread_block_CG_ty, workspace, input);

  // Choose a leader thread to print the results
  if (thread_block_CG_ty.thread_rank() == 0) {
    printf("\n\n\n Sum of all ranks 0..%d in threadBlockCooperativeGroup is %d\n\n",
           (int)thread_block_CG_ty.size() - 1, output_sum);
    printf(" Creating %d groups, of tile size %d threads:\n\n",
           (int)thread_block_CG_ty.size() / tile_size, tile_size);
  }

  thread_block_CG_ty.sync();

  cg::thread_block_tile<tile_size> tiled_part = cg::tiled_partition<tile_size>(thread_block_CG_ty);

  // This offset allows each group to have its own unique area in the workspace array
  int workspace_offset = thread_block_CG_ty.thread_rank() - tiled_part.thread_rank();

  output_sum = reduction_kernel(tiled_part, workspace + workspace_offset, input);

  if (thread_block_CG_ty.thread_rank() == 0) {
    printf("   Sum of all ranks %d..%d in this tiled_part group is %d.\n",
        input, input + tiled_part.size() - 1, output_sum);
    result[input / (tile_size)] = output_sum;
  }
  return;
}

__global__ void kernel_cg_group_partition_dynamic(unsigned int tile_size, int* result,
                                                  bool isGlobalMem, int* globalMem) {
  cg::thread_block threadBlockCGTy = cg::this_thread_block();
  int threadBlockGroupSize = threadBlockCGTy.size();

  int* workspace = NULL;

  if (isGlobalMem) {
    workspace = globalMem;
  } else {
    // Declare a shared memory
    extern __shared__ int sharedMem[];
    workspace = sharedMem;
  }

  int input, outputSum;

  // input to reduction, for each thread, is its' rank in the group
  input = threadBlockCGTy.thread_rank();

  outputSum = reduction_kernel(threadBlockCGTy, workspace, input);

  if (threadBlockCGTy.thread_rank() == 0) {
    printf("\n\n\n Sum of all ranks 0..%d in threadBlockCooperativeGroup is %d\n\n",
           (int)threadBlockCGTy.size() - 1, outputSum);
    printf(" Creating %d groups, of tile size %d threads:\n\n",
           (int)threadBlockCGTy.size() / tile_size, tile_size);
  }

  threadBlockCGTy.sync();

  cg::thread_group tiledPartition = cg::tiled_partition(threadBlockCGTy, tile_size);

  // This offset allows each group to have its own unique area in the workspace array
  int workspaceOffset = threadBlockCGTy.thread_rank() - tiledPartition.thread_rank();

  outputSum = reduction_kernel(tiledPartition, workspace + workspaceOffset, input);

  if (tiledPartition.thread_rank() == 0) {
    printf("   Sum of all ranks %d..%d in this tiledPartition group is %d.\n",
        input, input + static_cast<int>(tiledPartition.size()) - 1, outputSum);
    result[input / (tile_size)] = outputSum;
  }
  return;
}

// Search if the sum exists in the expected results array
void verifyResults(int* hPtr, int* dPtr, int size) {
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

template <typename F>
static void common_group_partition(F kernel_func, unsigned int tile_size, void** params, size_t num_params, bool use_global_mem) {
  int block_size = 1;
  int threads_per_blk = 64;

  int num_tiles = (block_size * threads_per_blk) / tile_size;

  // Build an array of expected reduction sum output on the host
  // based on the sum of their respective thread ranks for verification.
  // eg: parent group has 64threads.
  // child thread ranks: 0-15, 16-31, 32-47, 48-63
  // expected sum:       120,   376,  632,  888
  int* expected_sum = new int[num_tiles];
  int temp = 0, sum = 0;

  for (int i = 1; i <= num_tiles; i++) {
    sum = temp;
    temp = (((tile_size * i) - 1) * (tile_size * i)) / 2;
    expected_sum[i-1] = temp - sum;
  }

  int* result_dev = NULL;
  HIP_CHECK(hipMalloc((void**)&result_dev, num_tiles * sizeof(int)));

  int* global_mem = NULL;
  if (use_global_mem) {
    HIP_CHECK(hipMalloc((void**)&global_mem, threads_per_blk * sizeof(int)));
  }

  int* result_host = NULL;
  HIP_CHECK(hipHostMalloc(&result_host, num_tiles * sizeof(int), hipHostMallocDefault));
  memset(result_host, 0, num_tiles * sizeof(int));

  params[num_params + 0] = &result_dev;
  params[num_params + 1] = &use_global_mem;
  params[num_params + 2] = &global_mem;

  if (use_global_mem) {
    // Launch Kernel
    HIP_CHECK(hipLaunchCooperativeKernel(kernel_func, block_size, threads_per_blk, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  } else {
    // Launch Kernel
    HIP_CHECK(hipLaunchCooperativeKernel(kernel_func, block_size, threads_per_blk, params, threads_per_blk * sizeof(int), 0));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipMemcpy(result_host, result_dev, num_tiles * sizeof(int), hipMemcpyDeviceToHost));

  verifyResults(expected_sum, result_host, num_tiles);

  // Free all allocated memory on host and device
  HIP_CHECK(hipFree(result_dev));
  HIP_CHECK(hipHostFree(result_host));
  if (use_global_mem) {
    HIP_CHECK(hipFree(global_mem));
  }
  delete[] expected_sum;
}

template <unsigned int tile_size> static void test_group_partition(bool use_global_mem) {
  void* params[3];
  size_t num_params = 0;
  common_group_partition(kernel_cg_group_partition_static<tile_size>, tile_size, params, num_params, use_global_mem);
}

static void test_group_partition(unsigned int tile_size, bool use_global_mem) {
  void* params[4];
  params[0] = &tile_size;
  size_t num_params = 1;
  common_group_partition(kernel_cg_group_partition_dynamic, tile_size, params, num_params, use_global_mem);
}

TEST_CASE("Unit_hipCGTiledGroupType") {
  // Use default device for validating the test
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  bool use_global_mem = GENERATE(true, false);

  SECTION("Static tile partition") {
    test_group_partition<2>(use_global_mem);
    test_group_partition<4>(use_global_mem);
    test_group_partition<8>(use_global_mem);
    test_group_partition<16>(use_global_mem);
    test_group_partition<32>(use_global_mem);
  }

  SECTION("Dynamic tile partition") {
    unsigned int tile_size = GENERATE(2, 4, 8, 16, 32);
    test_group_partition(tile_size, use_global_mem);
  }
}