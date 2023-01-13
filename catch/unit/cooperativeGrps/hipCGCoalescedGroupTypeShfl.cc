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

#include "cooperative_groups_common.hh"

namespace cg = cooperative_groups;

static constexpr size_t kWaveSize = 32;

/* Test coalesced group's functionality.
 *
 */

__global__ void kernel_shfl (int * dPtr, int *dResults, int srcLane, int cg_sizes) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id % cg_sizes == 0) {
    cg::coalesced_group const& g = cg::coalesced_threads();
    int rank = g.thread_rank();
    int val = dPtr[rank];
    dResults[rank] = g.shfl(val, srcLane);
    return;
  }
}

__global__ void kernel_shfl_any_to_any (int *randVal, int *dsrcArr, int *dResults, int cg_sizes) {

 int id = threadIdx.x + blockIdx.x * blockDim.x;

 if (id % cg_sizes == 0) {
    cg::coalesced_group const& g = cg::coalesced_threads();
    int rank = g.thread_rank();
    int val = randVal[rank];
    dResults[rank] = g.shfl(val, dsrcArr[rank]);
    return;
  }

}

static void test_shfl_any_to_any() {

  unsigned int cg_size = GENERATE(1, 2, 4);

  int block_size = 1;
  int threads_per_blk = kWaveSize;

  int total_threads = block_size * threads_per_blk;
  int group_size = (total_threads + cg_size - 1) / cg_size;
  int group_size_in_bytes = group_size * sizeof(int);

  int* data_host = NULL;
  int* data_dev = NULL;
  int* src_dev = NULL;
  int* result_dev = NULL;

  int* src_host = (int*)malloc(group_size_in_bytes);
  int* src_cpu_host = (int*)malloc(group_size_in_bytes);

  int data_size = block_size * threads_per_blk * sizeof(int);

  HIP_CHECK(hipHostMalloc(&data_host, data_size));
  // Fill up the array
  for (int i = 0; i < threads_per_blk; i++) {
    data_host[i] = rand() % 1000;
  }

  // Fill up the random array
  for (int i = 0; i < group_size; i++) {
    src_host[i] = rand() % 1000;
    src_cpu_host[i] = src_host[i] % group_size;
  }

  /* Fill cpu results array so that we can verify with gpu computation */
  int* expected_result = (int*)malloc(group_size_in_bytes);
  for(int i = 0; i < group_size; i++) {
    expected_result[i] = data_host[src_cpu_host[i]];
  }

  HIP_CHECK(hipMalloc(&data_dev, group_size_in_bytes));
  HIP_CHECK(hipMalloc(&result_dev, group_size_in_bytes));

  HIP_CHECK(hipMalloc(&src_dev, group_size_in_bytes));
  HIP_CHECK(hipMemcpy(src_dev, src_host, group_size_in_bytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipMemcpy(data_dev, data_host, group_size_in_bytes, hipMemcpyHostToDevice));
  // Launch Kernel
  hipLaunchKernelGGL(kernel_shfl_any_to_any, block_size, threads_per_blk,
                    threads_per_blk * sizeof(int), 0 , data_dev, src_dev, result_dev, cg_size);
  HIP_CHECK(hipGetLastError()); 
  HIP_CHECK(hipMemcpy(data_host, result_dev, group_size_in_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());
  
  compareResults(expected_result, data_host, group_size_in_bytes);
  
  HIP_CHECK(hipHostFree(data_host));
  HIP_CHECK(hipFree(data_dev));
  HIP_CHECK(hipFree(result_dev));
  free(src_host);
  free(src_cpu_host);
  free(expected_result);
}

static void test_shfl_broadcast() {

  unsigned int cg_size = GENERATE(1, 2, 4);

  int block_size = 1;
  int threads_per_blk = kWaveSize;

  int total_threads = block_size * threads_per_blk;
  int group_size = (total_threads + cg_size - 1) / cg_size;
  int group_size_in_bytes = group_size * sizeof(int);

  int* data_host = NULL;
  int* data_dev = NULL;
  int* result_dev = NULL;
  int src_lane = rand() % 1000;
  int src_lane_cpu = 0;
  INFO("Testing coalesced_groups shfl with src_lane " << src_lane << " and group size " << cg_size << "\n");

  int data_size = block_size * threads_per_blk * sizeof(int);

  HIP_CHECK(hipHostMalloc(&data_host, group_size_in_bytes));
  // Fill up the array
  for (int i = 0; i < group_size; i++) {
    data_host[i] = rand() % 1000;
  }
  /* Fill cpu results array so that we can verify with gpu computation */
  src_lane_cpu = data_host[src_lane % group_size];

  int* expected_result = (int*)malloc(sizeof(int) * group_size);
  for (int i = 0; i < group_size; i++) {
    expected_result[i] = src_lane_cpu;
  }

  HIP_CHECK(hipMalloc(&data_dev, group_size_in_bytes));
  HIP_CHECK(hipMalloc(&result_dev, group_size_in_bytes));

  HIP_CHECK(hipMemcpy(data_dev, data_host, group_size_in_bytes, hipMemcpyHostToDevice));
  // Launch Kernel
  hipLaunchKernelGGL(kernel_shfl, block_size, threads_per_blk,
                    threads_per_blk * sizeof(int), 0, data_dev, result_dev, src_lane, cg_size);
  HIP_CHECK(hipGetLastError()); 
  HIP_CHECK(hipMemcpy(data_host, result_dev, group_size_in_bytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  compareResults(expected_result, data_host, group_size_in_bytes);

  HIP_CHECK(hipHostFree(data_host));
  HIP_CHECK(hipFree(data_dev));
  HIP_CHECK(hipFree(result_dev));
  free(expected_result);
}

TEST_CASE("Unit_hipCGCoalescedGroupTypeShfl") {
  // Use default device for validating the test
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  SECTION("Testing shfl collective as a broadcast") {
    for (int i = 0; i < 1; i++) {
      test_shfl_broadcast();
    }
  }

  SECTION("Testing shfl operations any-to-any member lanes") {
    for (int i = 0; i < 100; i++) {
      test_shfl_any_to_any();
    }
  }
}