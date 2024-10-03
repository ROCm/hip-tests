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

// Test Description:
/* This test implements simple tests to verfy intrinsic shuffle functions.*/

#include <hip_test_common.hh>
#include <stdio.h>
#include <vector>

/* Test basic shfl functionality.
 *
 */
void __global__ shfl_test(int *presult, int delta, int width) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int val = __shfl(id, id + delta, width);
  presult[id] = val;
}

void __global__ shfl_up_test(int *presult, int delta, int width) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int val = __shfl_up(id, delta, width);
  presult[id] = val;
}

void __global__ shfl_down_test(int *presult, int delta, int width) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int val = __shfl_down(id, delta, width);
  presult[id] = val;
}

TEST_CASE("Unit_shfl_function") {

  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0))
  unsigned int wave_size = devProp.warpSize;
  size_t pow2 = static_cast<int>(log2(wave_size));
  std::vector<size_t> width;
  std::vector<size_t> delta;
  int blockSize = 1;
  int threadsPerBlock = wave_size;
  int totalThreads = blockSize * threadsPerBlock;
  int* dResults = NULL;
  int* hPtr = NULL;
  bool testpassed = true;

  for (int i = 0; i < pow2; i++) {
    width.push_back(pow(2, i));
    delta.push_back(pow(2, i) + 1);
  }

  HIPCHECK(hipHostMalloc(&hPtr, sizeof(int) * totalThreads));
  HIPCHECK(hipMalloc(&dResults, sizeof(int) * totalThreads));
  SECTION("test shfl function") {
    for (auto w : width) {
      for (auto d : delta) {
        hipLaunchKernelGGL(shfl_test, blockSize, threadsPerBlock,
                          threadsPerBlock * sizeof(int), 0, dResults, d, w);
        hipDeviceSynchronize();
        HIPCHECK(hipMemcpy(hPtr, dResults, totalThreads * sizeof(int), hipMemcpyDeviceToHost));

        for (size_t tid = 0; tid < wave_size; ++tid) {
          auto src = ((tid + d) % w) + (tid / w) * w;
          if (hPtr[tid] != src) {
            testpassed = false;
          }
        }
      }
    }
    REQUIRE(testpassed == true);
  }
  SECTION("test shfl up function") {
    for (auto w : width) {
      for (auto d : delta) {
        hipLaunchKernelGGL(shfl_up_test, blockSize, threadsPerBlock,
                          threadsPerBlock * sizeof(int), 0, dResults, d, w);
        HIPCHECK(hipDeviceSynchronize());
        HIPCHECK(hipMemcpy(hPtr, dResults, totalThreads * sizeof(int), hipMemcpyDeviceToHost));
        for (size_t tid = d; tid < wave_size; ++tid) {
          auto src = ((tid - d) < ((tid / w)*w)) ? tid : (tid - d);
          if (hPtr[tid] != src) {
            testpassed = false;
          }
        }
      }
    }
    REQUIRE(testpassed == true);
  }
  SECTION("test shfl down function") {
    for (auto w : width) {
      for (auto d : delta) {
        hipLaunchKernelGGL(shfl_down_test, blockSize, threadsPerBlock,
                          threadsPerBlock * sizeof(int), 0, dResults, d, w);
        HIPCHECK(hipDeviceSynchronize());
        HIPCHECK(hipMemcpy(hPtr, dResults, totalThreads * sizeof(int), hipMemcpyDeviceToHost));
        for (size_t tid = 0; tid < wave_size; ++tid) {
          auto src = (((tid % w) + d) >= w) ? tid : tid + d;
          if (hPtr[tid] != src) {
            testpassed = false;
          }
        }
      }
    }
    REQUIRE(testpassed == true);
  }
  HIPCHECK(hipFree(dResults));
}
