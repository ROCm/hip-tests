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
   formula. The second part, partitions this parent group into child subgroups
   a.k.a tiles using using tiled_partition() collective operation. This can be called
   with a static tile size, passed in templated non-type variable-tiled_partition<tileSz>,
   or in runtime as tiled_partition(thread_group parent, tileSz). This test covers both these
   cases.
   This test tests functionality of cg group partitioning, (static and dynamic) and its respective
   API's size(), thread_rank(), and sync().
*/

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <stdio.h>
#include <vector>

#include "cooperative_groups_common.hh"

namespace cg = cooperative_groups;

static constexpr size_t kWaveSize = 32;

/* Test coalesced group's functionality.
 *
 */

/* Parallel reduce kernel.
 *
 * Step complexity: O(log n)
 * Work complexity: O(n)
 *
 * Note: This kernel works only with power of 2 input arrays.
 */
__device__ int reduction_kernel(cg::coalesced_group g, int* x, int val) {
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

__device__ int atomicAggInc(int *ptr) {
   cg::coalesced_group g = cg::coalesced_threads();
   int prev;
   // elect the first active thread to perform atomic add
   if (g.thread_rank() == 0) {
     prev = atomicAdd(ptr, g.size());
   }
   // broadcast previous value within the warp
   // and add each active thread’s rank to it
   prev = g.thread_rank() + g.shfl(prev, 0);
   return prev;
}

__global__ void filter_arr(int *dst, int *nres, const int *src, int n) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = id; i < n; i += gridDim.x * blockDim.x) {
    if (src[i] > 0) dst[atomicAggInc(nres)] = src[i];
  }
}

__global__ void kernel_cg_coalesced_group_partition(unsigned int tile_size, int* result,
                                                  bool is_global_mem, int* global_mem, int cg_sizes, int num_tiles) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id % cg_sizes == 0) {
    cg::coalesced_group thread_block_CG_ty = cg::coalesced_threads();

    int* workspace = NULL;

    if (is_global_mem) {
      workspace = global_mem;
    } else {
      // Declare a shared memory
      extern __shared__ int sharedMem[];
      workspace = sharedMem;
    }

    int input, output_sum;

    // input to reduction, for each thread, is its' rank in the group
    input = thread_block_CG_ty.thread_rank();

    output_sum = reduction_kernel(thread_block_CG_ty, workspace, input);

    if (thread_block_CG_ty.thread_rank() == 0) {
      printf(" Sum of all ranks 0..%d in coalesced_group is %d\n\n",
             (int)thread_block_CG_ty.size() - 1, output_sum);
      printf(" Creating %d groups, of tile size %d threads:\n\n",
             (int)thread_block_CG_ty.size() / tile_size, tile_size);
    }

    thread_block_CG_ty.sync();

    cg::coalesced_group tiled_part = cg::tiled_partition(thread_block_CG_ty, tile_size);

    // This offset allows each group to have its own unique area in the workspace array
    int workspace_offset = thread_block_CG_ty.thread_rank() - tiled_part.thread_rank();

    output_sum = reduction_kernel(tiled_part, workspace + workspace_offset, input);

    if (tiled_part.thread_rank() == 0) {
      printf(
          "   Sum of all ranks %d..%d in this tiled_part group is %d.\n",
          input, input + static_cast<int>(tiled_part.size()) - 1, output_sum);
      result[input / (tile_size)] = output_sum;
    }
    return;
  }
}

__global__ void kernel_coalesced_active_groups(unsigned int* active_size_dev) {
  cg::thread_block thread_block_CG_ty = cg::this_thread_block();

  // input to reduction, for each thread, is its' rank in the group
  int input = thread_block_CG_ty.thread_rank();

  if (thread_block_CG_ty.thread_rank() == 0) {
    printf("Creating odd and even set of active thread groups based on branch divergence\n\n");
  }

  thread_block_CG_ty.sync();

  // Group all active odd threads
  if (thread_block_CG_ty.thread_rank() % 2) {
    cg::coalesced_group active_odd = cg::coalesced_threads();

    if (active_odd.thread_rank() == 0) {
      active_size_dev[0] = active_odd.size();
      printf("ODD: Size of odd set of active threads is %d."
             " Corresponding parent thread_rank is %d.\n\n",
               active_odd.size(), thread_block_CG_ty.thread_rank());
    }
  }
  else { // Group all active even threads
    cg::coalesced_group active_even = cg::coalesced_threads();

    if (active_even.thread_rank() == 0) {
      active_size_dev[1] = active_even.size();
      printf("EVEN: Size of even set of active threads is %d."
             " Corresponding parent thread_rank is %d.\n\n",
               active_even.size(), thread_block_CG_ty.thread_rank());
    }
  }
  return;
}

static void test_active_threads_grouping() {
  int block_size = 1;
  int threads_per_blk = kWaveSize;

  unsigned int* active_size_dev;
  HIP_CHECK(hipMalloc(&active_size_dev, sizeof(unsigned int) * 2));
  unsigned int* active_size_host;
  active_size_host = reinterpret_cast<unsigned int *>(malloc(sizeof(unsigned int) * 2));

  // Launch Kernel
  hipLaunchKernelGGL(kernel_coalesced_active_groups, block_size, threads_per_blk, 0, 0, active_size_dev);
  HIP_CHECK(hipGetLastError()); 

  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(active_size_host, active_size_dev, 2* sizeof(unsigned int), hipMemcpyDeviceToHost));
  REQUIRE(active_size_host[0] == (threads_per_blk / 2));
  REQUIRE(active_size_host[1] == (threads_per_blk / 2));
}

static void test_group_partition(unsigned int tile_size, bool use_global_mem) {
  int block_size = 1;
  int threads_per_blk = kWaveSize;

  unsigned int cg_size = GENERATE(1, 2, 4);

  int num_tiles = ((block_size * threads_per_blk) / cg_size) / tile_size;

  // num_tiles = 0 when partitioning is possible. The below statement is to avoid
  // out-of-bounds error and still evaluate failure case.
  if (num_tiles == 0) return;
  // Build an array of expected reduction sum output on the host
  // based on the sum of their respective thread ranks to use for verification
  int* expected_sum = new int[num_tiles + 1];
  int temp = 0, sum = 0;
  for (int i = 1; i <= num_tiles; i++) {
    sum = temp;
    temp = (((tile_size * i) - 1) * (tile_size * i)) / 2;
    expected_sum[i-1] = temp - sum;
  }

  int* result_dev = NULL;
  HIP_CHECK(hipMalloc(&result_dev, sizeof(int) * num_tiles));

  int* global_mem = NULL;
  if (use_global_mem) {
    HIP_CHECK(hipMalloc((void**)&global_mem, threads_per_blk * sizeof(int)));
  }

  int* result_host = NULL;
  HIP_CHECK(hipHostMalloc(&result_host, num_tiles * sizeof(int), hipHostMallocDefault));
  memset(result_host, 0, num_tiles * sizeof(int));

  // Launch Kernel
  if (use_global_mem) {
    hipLaunchKernelGGL(kernel_cg_coalesced_group_partition, block_size, threads_per_blk, 0, 0, tile_size,
                       result_dev, use_global_mem, global_mem, cg_size, num_tiles);
    HIP_CHECK(hipGetLastError()); 

    HIP_CHECK(hipDeviceSynchronize());
  } else {
    hipLaunchKernelGGL(kernel_cg_coalesced_group_partition, block_size, threads_per_blk,
                      threads_per_blk * sizeof(int), 0, tile_size, result_dev, use_global_mem, global_mem, cg_size, num_tiles);
    HIP_CHECK(hipGetLastError()); 

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

TEST_CASE("Unit_hipCGCoalescedGroupType_WarpAggregatedAtomics") {
  // Use default device for validating the test
  constexpr unsigned int num_elems = 10000000;
  constexpr size_t num_threads_per_blk = 512;

  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  int *data_to_filter, *filtered_data, nres = 0;
  int *d_data_to_filter, *d_filtered_data, *d_nres;

  int num_of_buckets = 5;

  data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * num_elems));

  // Generate input data.
  for (int i = 0; i < num_elems; i++) {
    data_to_filter[i] = rand() % num_of_buckets;
  }

  HIP_CHECK(hipMalloc(&d_data_to_filter, sizeof(int) * num_elems));
  HIP_CHECK(hipMalloc(&d_filtered_data, sizeof(int) * num_elems));
  HIP_CHECK(hipMalloc(&d_nres, sizeof(int)));

  HIP_CHECK(hipMemcpy(d_data_to_filter, data_to_filter,
                             sizeof(int) * num_elems, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_nres, 0, sizeof(int)));

  dim3 dimBlock(num_threads_per_blk, 1, 1);
  dim3 dimGrid((num_elems / num_threads_per_blk) + 1, 1, 1);

  filter_arr<<<dimGrid, dimBlock>>>(d_filtered_data, d_nres, d_data_to_filter,
                                    num_elems);


  HIP_CHECK(hipMemcpy(&nres, d_nres, sizeof(int), hipMemcpyDeviceToHost));

  filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * nres));

  HIP_CHECK(hipMemcpy(filtered_data, d_filtered_data, sizeof(int) * nres,
                             hipMemcpyDeviceToHost));

  int *host_filtered_data =
      reinterpret_cast<int *>(malloc(sizeof(int) * num_elems));

  // Generate host output with host filtering code.
  int host_flt_count = 0;
  for (int i = 0; i < num_elems; i++) {
    if (data_to_filter[i] > 0) {
      host_filtered_data[host_flt_count++] = data_to_filter[i];
    }
  }

  REQUIRE(host_flt_count == nres);

  HIP_CHECK(hipFree(d_data_to_filter));
  HIP_CHECK(hipFree(d_filtered_data));
  HIP_CHECK(hipFree(d_nres));
}

TEST_CASE("Unit_hipCGCoalescedGroupType_Partitioning") {
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
  unsigned int tile_size = GENERATE(2, 4, 8, 16, 32);

  test_group_partition(tile_size, use_global_mem);
}

TEST_CASE("Unit_hipCGCoalescedGroupType_ActiveThreadsGrouping") {
  // Use default device for validating the test
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  test_active_threads_grouping();
}