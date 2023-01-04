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

template <unsigned int tileSz>
__global__ void kernel_cg_group_partition_static(int* result, bool isGlobalMem, int* globalMem) {
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

  // we pass its own thread rank as inputs
  input = threadBlockCGTy.thread_rank();

  outputSum = reduction_kernel(threadBlockCGTy, workspace, input);

  // Choose a leader thread to print the results
  if (threadBlockCGTy.thread_rank() == 0) {
    printf("\n\n\n Sum of all ranks 0..%d in threadBlockCooperativeGroup is %d\n\n",
           (int)threadBlockCGTy.size() - 1, outputSum);
    printf(" Creating %d groups, of tile size %d threads:\n\n",
           (int)threadBlockCGTy.size() / tileSz, tileSz);
  }

  threadBlockCGTy.sync();

  cg::thread_block_tile<tileSz> tiledPartition = cg::tiled_partition<tileSz>(threadBlockCGTy);

  // This offset allows each group to have its own unique area in the workspace array
  int workspaceOffset = threadBlockCGTy.thread_rank() - tiledPartition.thread_rank();

  outputSum = reduction_kernel(tiledPartition, workspace + workspaceOffset, input);

  if (tiledPartition.thread_rank() == 0) {
    printf("   Sum of all ranks %d..%d in this tiledPartition group is %d.\n",
        input, input + tiledPartition.size() - 1, outputSum);
    result[input / (tileSz)] = outputSum;
  }
  return;
}

__global__ void kernel_cg_group_partition_dynamic(unsigned int tileSz, int* result,
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
           (int)threadBlockCGTy.size() / tileSz, tileSz);
  }

  threadBlockCGTy.sync();

  cg::thread_group tiledPartition = cg::tiled_partition(threadBlockCGTy, tileSz);

  // This offset allows each group to have its own unique area in the workspace array
  int workspaceOffset = threadBlockCGTy.thread_rank() - tiledPartition.thread_rank();

  outputSum = reduction_kernel(tiledPartition, workspace + workspaceOffset, input);

  if (tiledPartition.thread_rank() == 0) {
    printf("   Sum of all ranks %d..%d in this tiledPartition group is %d.\n",
        input, input + static_cast<int>(tiledPartition.size()) - 1, outputSum);
    result[input / (tileSz)] = outputSum;
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
static void common_group_partition(F kernel_func, unsigned int tileSz, void** params, size_t numParams, bool useGlobalMem) {
  int blockSize = 1;
  int threadsPerBlock = 64;

  int numTiles = (blockSize * threadsPerBlock) / tileSz;

  // Build an array of expected reduction sum output on the host
  // based on the sum of their respective thread ranks for verification.
  // eg: parent group has 64threads.
  // child thread ranks: 0-15, 16-31, 32-47, 48-63
  // expected sum:       120,   376,  632,  888
  int* expectedSum = new int[numTiles];
  int temp = 0, sum = 0;

  for (int i = 1; i <= numTiles; i++) {
    sum = temp;
    temp = (((tileSz * i) - 1) * (tileSz * i)) / 2;
    expectedSum[i-1] = temp - sum;
  }

  int* dResult = NULL;
  HIP_CHECK(hipMalloc((void**)&dResult, numTiles * sizeof(int)));

  int* globalMem = NULL;
  if (useGlobalMem) {
    HIP_CHECK(hipMalloc((void**)&globalMem, threadsPerBlock * sizeof(int)));
  }

  int* hResult = NULL;
  HIP_CHECK(hipHostMalloc(&hResult, numTiles * sizeof(int), hipHostMallocDefault));
  memset(hResult, 0, numTiles * sizeof(int));

  params[numParams + 0] = &dResult;
  params[numParams + 1] = &useGlobalMem;
  params[numParams + 2] = &globalMem;

  if (useGlobalMem) {
    // Launch Kernel
    HIP_CHECK(hipLaunchCooperativeKernel(kernel_func, blockSize, threadsPerBlock, params, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
  } else {
    // Launch Kernel
    HIP_CHECK(hipLaunchCooperativeKernel(kernel_func, blockSize, threadsPerBlock, params, threadsPerBlock * sizeof(int), 0));
    HIP_CHECK(hipDeviceSynchronize());
  }

  HIP_CHECK(hipMemcpy(hResult, dResult, numTiles * sizeof(int), hipMemcpyDeviceToHost));

  verifyResults(expectedSum, hResult, numTiles);

  // Free all allocated memory on host and device
  HIP_CHECK(hipFree(dResult));
  HIP_CHECK(hipHostFree(hResult));
  if (useGlobalMem) {
    HIP_CHECK(hipFree(globalMem));
  }
  delete[] expectedSum;
}

template <unsigned int tileSz> static void test_group_partition(bool useGlobalMem) {
  void* params[3];
  size_t numParams = 0;
  common_group_partition(kernel_cg_group_partition_static<tileSz>, tileSz, params, numParams, useGlobalMem);
}

static void test_group_partition(unsigned int tileSz, bool useGlobalMem) {
  void* params[4];
  params[0] = &tileSz;
  size_t numParams = 1;
  common_group_partition(kernel_cg_group_partition_dynamic, tileSz, params, numParams, useGlobalMem);
}

TEST_CASE("Unit_hipCGTiledGroupType") {
  // Use default device for validating the test
  int deviceId;
  hipDeviceProp_t deviceProperties;
  HIP_CHECK(hipGetDevice(&deviceId));
  HIP_CHECK(hipGetDeviceProperties(&deviceProperties, deviceId));

  if (!deviceProperties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  bool useGlobalMem = GENERATE(true, false);

  SECTION("Static tile partition") {
    test_group_partition<2>(useGlobalMem);
    test_group_partition<4>(useGlobalMem);
    test_group_partition<8>(useGlobalMem);
    test_group_partition<16>(useGlobalMem);
    test_group_partition<32>(useGlobalMem);
  }

  SECTION("Dynamic tile partition") {
    unsigned int tileSz = GENERATE(2, 4, 8, 16, 32);
    test_group_partition(tileSz, useGlobalMem);
  }
}