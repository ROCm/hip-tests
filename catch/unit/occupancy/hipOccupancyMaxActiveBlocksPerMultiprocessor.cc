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

#include "occupancy_common.hh"

/**
 * @addtogroup hipOccupancyMaxActiveBlocksPerMultiprocessor
 * hipOccupancyMaxActiveBlocksPerMultiprocessor
 * @{
 * @ingroup OccupancyTest
 * `hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
 * int blockSize, size_t dynSharedMemPerBlk)` -
 * Returns occupancy for a device function.
 */

static __global__ void f1(float* a) { *a = 1.0; }

template <typename T> static __global__ void f2(T* a) { *a = 1; }

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the grid size is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When block size is 0
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pointer to the function is `nullptr`
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidDeviceFunction`
 * Test source
 * ------------------------
 *  - unit/occupancy/hipOccupancyMaxActiveBlocksPerMultiprocessor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_Negative_Parameters") {
  int numBlocks = 0;
  int blockSize = 0;
  int gridSize = 0;

  // Get potential blocksize
  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0));

  // Common negative tests
  MaxActiveBlocksPerMultiprocessorNegative(
      [](int* numBlocks, int blockSize, size_t dynSharedMemPerBlk) {
        return hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f1, blockSize,
                                                            dynSharedMemPerBlk);
      },
      blockSize);

  SECTION("Kernel function is NULL") {
    HIP_CHECK_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, NULL, blockSize, 0),
                    hipErrorInvalidDeviceFunction);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Check if grid size and block size are within valid range using basic kernel functions:
 *    -# When `dynSharedMemPerBlk = 0`
 *      - Expected output: return `hipSuccess`
 *    -# When `dynSharedMemPerBlk = sharedMemPerBlock`
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/occupancy/hipOccupancyMaxActiveBlocksPerMultiprocessor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_Positive_RangeValidation") {
  hipDeviceProp_t devProp;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0") {
    // Get potential blocksize
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, 0, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize](int* numBlocks) {
          return hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f1, blockSize, 0);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }
  SECTION("dynSharedMemPerBlk = sharedMemPerBlock") {
    // Get potential blocksize
    HIP_CHECK(
        hipOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, f1, devProp.sharedMemPerBlock, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize, devProp](int* numBlocks) {
          return hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f1, blockSize,
                                                              devProp.sharedMemPerBlock);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Check is number of blocks is greater than 0 when API is invoked with a template:
 *    -# When `dynSharedMemPerBlk = 0`
 *      - Expected output: return `hipSuccess`
 *    -# When `dynSharedMemPerBlk = sharedMemPerBlock`
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/occupancy/hipOccupancyMaxActiveBlocksPerMultiprocessor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipOccupancyMaxActiveBlocksPerMultiprocessor_Positive_TemplateInvocation") {
  hipDeviceProp_t devProp;
  int blockSize = 0;
  int gridSize = 0;

  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

  SECTION("dynSharedMemPerBlk = 0") {
    // Get potential blocksize
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize<void (*)(int*)>(&gridSize, &blockSize, f2, 0, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize](int* numBlocks) {
          return hipOccupancyMaxActiveBlocksPerMultiprocessor<void (*)(int*)>(numBlocks, f2,
                                                                              blockSize, 0);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }

  SECTION("dynSharedMemPerBlk = sharedMemPerBlock") {
    // Get potential blocksize
    HIP_CHECK(hipOccupancyMaxPotentialBlockSize<void (*)(int*)>(&gridSize, &blockSize, f2,
                                                                devProp.sharedMemPerBlock, 0));

    MaxActiveBlocksPerMultiprocessor(
        [blockSize, devProp](int* numBlocks) {
          return hipOccupancyMaxActiveBlocksPerMultiprocessor<void (*)(int*)>(
              numBlocks, f2, blockSize, devProp.sharedMemPerBlock);
        },
        blockSize, devProp.maxThreadsPerMultiProcessor);
  }
}

/**
 * End doxygen group hipOccupancyMaxActiveBlocksPerMultiprocessor.
 * @}
 */

/**
 * @addtogroup hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 * hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
 * @{
 * @ingroup OccupancyTest
 * `hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* f,
 * int blockSize, size_t dynSharedMemPerBlk, unsigned int flags __dparm(hipOccupancyDefault))` -
 * Returns occupancy for a device function.
 * @warning Flags ignored currently, skipped tests implementation.
 */
