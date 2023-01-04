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

#include "cooperative_groups_common.hh"

namespace cg = cooperative_groups;

enum class TiledGroupShflTests { shflDown, shflXor, shflUp };

template <unsigned int tileSz>
__device__ int reduction_kernel_shfl_down(cg::thread_block_tile<tileSz> const& g, volatile int val) {
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

template <unsigned int tileSz>
__device__ int reduction_kernel_shfl_xor(cg::thread_block_tile<tileSz> const& g, int val) {
  int sz = g.size();

  for (int i = sz / 2; i > 0; i >>= 1) {
    val += g.shfl_xor(val, i);
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
__device__ int prefix_sum_kernel(cg::thread_block_tile<tileSz> const& g, volatile int val) {
  int sz = g.size();
#pragma unroll
  for (int i = 1; i < sz; i <<= 1) {
    int temp = g.shfl_up(val, i);

    if (g.thread_rank() >= i) {
      val += temp;
    }
  }
  return val;
}

template <unsigned int tileSz>
static __global__ void kernel_cg_group_partition_static(int* result, TiledGroupShflTests shflTest) {
  cg::thread_block threadBlockCGTy = cg::this_thread_block();
  int threadBlockGroupSize = threadBlockCGTy.size();
  int input, outputSum;

  // Choose a leader thread to print the results
  if (threadBlockCGTy.thread_rank() == 0) {
    printf(" Creating %d groups, of tile size %d threads:\n\n",
           (int)threadBlockCGTy.size() / tileSz, tileSz);
  }

  threadBlockCGTy.sync();

  cg::thread_block_tile<tileSz> tiledPartition = cg::tiled_partition<tileSz>(threadBlockCGTy);
  int threadRank = tiledPartition.thread_rank();

  input = tiledPartition.thread_rank();

  switch(shflTest) {
    case(TiledGroupShflTests::shflDown):
      outputSum = reduction_kernel_shfl_down(tiledPartition, input);
      break;
    case(TiledGroupShflTests::shflXor):
      outputSum = reduction_kernel_shfl_xor(tiledPartition, input);
      break;
    case(TiledGroupShflTests::shflUp):
      outputSum = prefix_sum_kernel(tiledPartition, input);
      result[threadBlockCGTy.thread_rank()] = outputSum;
  }

  if (tiledPartition.thread_rank() == 0 && shflTest != TiledGroupShflTests::shflUp) {
    printf(
        "   Sum of all ranks 0..%d in this tiledPartition group is %d\n",
        tiledPartition.size() - 1, outputSum);
    result[threadBlockCGTy.thread_rank() / (tileSz)] = outputSum;
  }
}

static void expected_result(int* expectedResult, int tileSz, int size, TiledGroupShflTests shflTest) {
  switch(shflTest) {
    case(TiledGroupShflTests::shflDown):
    case(TiledGroupShflTests::shflXor):
    {
      int expectedSum = ((tileSz - 1) * tileSz / 2);
      for (int i = 0; i < size; i++) {
        expectedResult[i] = expectedSum;
      }
      break;
    }
    case(TiledGroupShflTests::shflUp):
    {
      for (int i = 0; i < size / tileSz; i++) {
        int acc = 0;
        for (int j = 0; j < tileSz; j++) {
          acc += j;
          expectedResult[i * tileSz + j] = acc;
        }
      }
      break;
    }
  }  
}

template <unsigned int tileSz> 
static void test_group_partition(TiledGroupShflTests shflTest) {
  int blockSize = 1;
  int threadsPerBlock = 64;

  int numElem = (blockSize * threadsPerBlock) / tileSz;
  if (shflTest == TiledGroupShflTests::shflUp) {
    numElem = blockSize * threadsPerBlock;
  }   

  int* expectedResult = new int[numElem];

  int* resultDev = NULL;
  int* resultHost = NULL;

  HIP_CHECK(hipHostMalloc(&resultHost, numElem * sizeof(int), hipHostMallocDefault));
  memset(resultHost, 0, numElem * sizeof(int));

  HIP_CHECK(hipMalloc(&resultDev, numElem * sizeof(int)));

  // Launch Kernel
  hipLaunchKernelGGL(kernel_cg_group_partition_static<tileSz>, blockSize, threadsPerBlock,
                    threadsPerBlock * sizeof(int), 0, resultDev, shflTest);
  HIP_CHECK(hipDeviceSynchronize());


  HIP_CHECK(hipMemcpy(resultHost, resultDev, sizeof(int) * numElem, hipMemcpyDeviceToHost));

  expected_result(expectedResult, tileSz, numElem, shflTest); 
  compareResults(expectedResult, resultHost, numElem * sizeof(int));

  // Free all allocated memory on host and device
  HIP_CHECK(hipFree(resultDev));
  HIP_CHECK(hipHostFree(resultHost));
  delete[] expectedResult;

}

TEST_CASE("Unit_hipCGTiledGroupType_Shfl") {
  // Use default device for validating the test
  int device;
  hipDeviceProp_t deviceProperties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&deviceProperties, device));

  if (!deviceProperties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  TiledGroupShflTests shflTest = GENERATE(TiledGroupShflTests::shflDown, TiledGroupShflTests::shflXor, TiledGroupShflTests::shflUp);
  test_group_partition<2>(shflTest);
  test_group_partition<4>(shflTest);
  test_group_partition<8>(shflTest);
  test_group_partition<16>(shflTest);
  test_group_partition<32>(shflTest);
}