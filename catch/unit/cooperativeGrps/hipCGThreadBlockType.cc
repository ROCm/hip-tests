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

static __global__
void kernel_cg_thread_block_type(int *sizeTestD,
                                 int *thdRankTestD,
                                 int *syncTestD,
                                 dim3 *groupIndexTestD,
                                 dim3 *thdIndexTestD)
{
  cg::thread_block tb = cg::this_thread_block();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
  // Test size
  sizeTestD[gIdx] = tb.size();

  // Test thread_rank
  thdRankTestD[gIdx] = tb.thread_rank();

  // Test sync
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  tb.sync();
  syncTestD[gIdx] = sm[1] * sm[0];

  // Test group_index
  groupIndexTestD[gIdx] = tb.group_index();

  // Test thread_index
  thdIndexTestD[gIdx] = tb.thread_index();
}

static __global__
void kernel_cg_thread_block_type_via_base_type(int *sizeTestD,
                                               int *thdRankTestD,
                                               int *syncTestD)
{
  cg::thread_group tg = cg::this_thread_block();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test size
  sizeTestD[gIdx] = tg.size();

  // Test thread_rank
  thdRankTestD[gIdx] = tg.thread_rank();

  // Test sync
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  tg.sync();
  syncTestD[gIdx] = sm[1] * sm[0];
}

static __global__
void kernel_cg_thread_block_type_via_public_api(int *sizeTestD,
                                                int *thdRankTestD,
                                                int *syncTestD)
{
  cg::thread_block tb = cg::this_thread_block();
  int gIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Test group_size api
  sizeTestD[gIdx] = cg::group_size(tb);

  // Test thread_rank api
  thdRankTestD[gIdx] = cg::thread_rank(tb);

  // Test sync api
  __shared__ int sm[2];
  if (threadIdx.x == 0)
    sm[0] = 10;
  else if (threadIdx.x == 1)
    sm[1] = 20;
  sync(tb);
  syncTestD[gIdx] = sm[1] * sm[0];
}

template <typename F>
static void test_cg_thread_block_type(F kernel_func, int blockSize, bool specific_api_test)
{
  int nBytes = sizeof(int) * 2 * blockSize;
  int nDim3Bytes = sizeof(dim3) * 2 * blockSize;
  int *sizeTestD, *sizeTestH;
  int *thdRankTestD, *thdRankTestH;
  int *syncTestD, *syncTestH;
  dim3 *groupIndexTestD, *groupIndexTestH;
  dim3 *thdIndexTestD, *thdIndexTestH;

  // Allocate device memory
  HIP_CHECK(hipMalloc(&sizeTestD, nBytes));
  HIP_CHECK(hipMalloc(&thdRankTestD, nBytes));
  HIP_CHECK(hipMalloc(&syncTestD, nBytes));

  // Allocate host memory
  HIP_CHECK(hipHostMalloc(&sizeTestH, nBytes));
  HIP_CHECK(hipHostMalloc(&thdRankTestH, nBytes));
  HIP_CHECK(hipHostMalloc(&syncTestH, nBytes));

  if (specific_api_test) {
    HIP_CHECK(hipMalloc(&groupIndexTestD, nDim3Bytes));
    HIP_CHECK(hipMalloc(&thdIndexTestD, nDim3Bytes));
    HIP_CHECK(hipHostMalloc(&groupIndexTestH, nDim3Bytes));
    HIP_CHECK(hipHostMalloc(&thdIndexTestH, nDim3Bytes));
  }

  // Launch Kernel
  void *params[5];
  params[0] = &sizeTestD;
  params[1] = &thdRankTestD;
  params[2] = &syncTestD;
  if (specific_api_test) {
    params[3] = &groupIndexTestD;
    params[4] = &thdIndexTestD;
  }
  HIP_CHECK(hipLaunchCooperativeKernel(kernel_func, 2, blockSize, params, 0, 0));

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(sizeTestH, sizeTestD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(thdRankTestH, thdRankTestD, nBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(syncTestH, syncTestD, nBytes, hipMemcpyDeviceToHost));
  if (specific_api_test) {
    HIP_CHECK(hipMemcpy(groupIndexTestH, groupIndexTestD, nDim3Bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(thdIndexTestH, thdIndexTestD, nDim3Bytes, hipMemcpyDeviceToHost));
  }

  // Validate results for both blocks together
  for (int i = 0; i < 2 * blockSize; ++i) {
    ASSERT_EQUAL(sizeTestH[i], blockSize);
    ASSERT_EQUAL(thdRankTestH[i], i % blockSize);
    ASSERT_EQUAL(syncTestH[i], 200);
    if (specific_api_test) {
      ASSERT_EQUAL(groupIndexTestH[i].x, (uint) i / blockSize);
      ASSERT_EQUAL(groupIndexTestH[i].y, 0);
      ASSERT_EQUAL(groupIndexTestH[i].z, 0);
      ASSERT_EQUAL(thdIndexTestH[i].x, (uint) i % blockSize);
      ASSERT_EQUAL(thdIndexTestH[i].y, 0);
      ASSERT_EQUAL(thdIndexTestH[i].z, 0);
    }
  }

  // Free device memory
  HIP_CHECK(hipFree(sizeTestD));
  HIP_CHECK(hipFree(thdRankTestD));
  HIP_CHECK(hipFree(syncTestD));

  //Free host memory
  HIP_CHECK(hipHostFree(sizeTestH));
  HIP_CHECK(hipHostFree(thdRankTestH));
  HIP_CHECK(hipHostFree(syncTestH));

  if (specific_api_test) {
    HIP_CHECK(hipFree(groupIndexTestD));
    HIP_CHECK(hipFree(thdIndexTestD));
    HIP_CHECK(hipHostFree(groupIndexTestH));
    HIP_CHECK(hipHostFree(thdIndexTestH));
  }
}


TEST_CASE("Unit_hipCGThreadBlockType") {
  using namespace std::placeholders;
  // Use default device for validating the test
  int deviceId;
  hipDeviceProp_t deviceProperties;
  HIP_CHECK(hipGetDevice(&deviceId));
  HIP_CHECK(hipGetDeviceProperties(&deviceProperties, deviceId));

  if (!deviceProperties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  void* (*kernel_func) (void);
  bool specific_api_test = false;

  SECTION("Default thread block API test") {
    kernel_func = reinterpret_cast<void*(*)()>(kernel_cg_thread_block_type);
    specific_api_test = true;
  }

  SECTION("Base type thread block API test") {
    kernel_func = reinterpret_cast<void*(*)()>(kernel_cg_thread_block_type_via_base_type);
  }

  SECTION("Public API thread block test") {
    kernel_func = reinterpret_cast<void*(*)()>(kernel_cg_thread_block_type_via_public_api);
  }

  // Test for blockSizes in powers of 2
  int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
  for (int blockSize = 2; blockSize <= maxThreadsPerBlock; blockSize = blockSize*2) {
    test_cg_thread_block_type(kernel_func, blockSize, specific_api_test);
  }

  // Test for random blockSizes, but the sequence is the same every execution
  srand(0);
  for (int i = 0; i < 10; i++) {
    // Test fails for only 1 thread per block
    test_cg_thread_block_type(kernel_func, max(2, rand() % maxThreadsPerBlock), specific_api_test);
  }
}