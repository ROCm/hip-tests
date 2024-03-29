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
/*
Testcase Scenarios :
Unit_hipOccupancyMaxPotentialBlockSize_Positive_RangeValidation - Test correct execution of
hipOccupancyMaxPotentialBlockSize for diffrent parameter values
Unit_hipOccupancyMaxPotentialBlockSize_Positive_TemplateInvocation - Test correct execution of
hipOccupancyMaxPotentialBlockSize template for diffrent parameter values
Unit_hipOccupancyMaxPotentialBlockSize_Negative_Parameters - Test unsuccessful execution of
hipOccupancyMaxPotentialBlockSize api when parameters are invalid
*/
#include "occupancy_common.hh"

static __global__ void f1(float* a) { *a = 1.0; }

template <typename T> static __global__ void f2(T* a) { *a = 1; }

TEST_CASE("Unit_hipOccupancyMaxPotentialBlockSize_Negative_Parameters") {
  // Common negative tests
  MaxPotentialBlockSizeNegative([](int* gridSize, int* blockSize) {
    return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f1, 0, 0);
  });

#if HT_AMD
#if 0 // EXSWHTEC-219
  SECTION("Kernel function is NULL") {
    int blockSize = 0;
    int gridSize = 0;
    // nvcc doesnt support kernelfunc(NULL) for api
    HIP_CHECK_ERROR(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, NULL, 0, 0),
                    hipErrorInvalidDeviceFunction);
  }
#endif
#endif
}

TEST_CASE("Unit_hipOccupancyMaxPotentialBlockSize_Positive_RangeValidation") {
  hipDeviceProp_t devProp;

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0, blockSizeLimit = 0") {
    MaxPotentialBlockSize(
        [](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f1, 0, 0);
        },
        devProp.maxThreadsPerBlock);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock") {
    MaxPotentialBlockSize(
        [devProp](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize(
              gridSize, blockSize, f1, devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock);
        },
        devProp.maxThreadsPerBlock);
  }
}

TEST_CASE("Unit_hipOccupancyMaxPotentialBlockSize_Positive_TemplateInvocation") {
  hipDeviceProp_t devProp;

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0, blockSizeLimit = 0") {
    MaxPotentialBlockSize(
        [](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize<void (*)(int*)>(gridSize, blockSize, f2, 0, 0);
        },
        devProp.maxThreadsPerBlock);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock, blockSizeLimit = maxThreadsPerBlock") {
    MaxPotentialBlockSize(
        [devProp](int* gridSize, int* blockSize) {
          return hipOccupancyMaxPotentialBlockSize<void (*)(int*)>(
              gridSize, blockSize, f2, devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock);
        },
        devProp.maxThreadsPerBlock);
  }
}
