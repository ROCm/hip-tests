/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <utils.hh>

const int N = 1024 * 1024 * 1024;
const size_t Nbytes = N * sizeof(int);

/**
 * Local Function to fill the array with given value
 */
void fillArr(int *arr, int size, int value) {
  for ( int i = 0; i < size; i++ ) {
    arr[i] = value;
  }
}

/**
 * Local Function to validate the array with given reference value
 */
bool validateArr(int *arr, int size, int refValue) {
  for ( int i = 0; i < size; i++ ) {
    if ( arr[i] != refValue ) {
      return false;
    }
  }
  return true;
}

 /**
 * Test Description
 * ------------------------
 *  - This test, tests the following sceanrio with per thread compiler option:-
 *  - Do Memory copy Asynchronously H2D in Legacy stream
 *  - and do Memory copy Asynchronously D2H in Legacy stream.
 *  - The task 2 should wait till the execution of task 1.
 *  - Even it is compiled with stream per thread, since one stream for both
 *  - these should be in sync.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_compiler_options.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_WithSptCompilerOption") {
  int *hostArrSrc = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostArrSrc != nullptr);
  fillArr(hostArrSrc, N, 1);

  int *devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, Nbytes));
  REQUIRE(devArr != nullptr);

  int *hostArrDst = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostArrDst != nullptr);
  fillArr(hostArrDst, N, 3);

  HIP_CHECK(hipMemcpyAsync(devArr, hostArrSrc, Nbytes,
                           hipMemcpyHostToDevice, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(hostArrDst, devArr, Nbytes,
                           hipMemcpyDeviceToHost, hipStreamLegacy));

  REQUIRE(validateArr(hostArrDst, N, 1) == true);

  free(hostArrSrc);
  free(hostArrDst);
  HIP_CHECK(hipFree(devArr));
}

/*
 * Local helper function to copy data from host to device
 */
void copyHostToDevice(int* hostArr, int* devArr) {
  HIP_CHECK(hipMemcpyAsync(devArr, hostArr, Nbytes,
                           hipMemcpyHostToDevice,
                           hipStreamLegacy));
}

/*
 * Local helper function to copy data from device to host
 */
void copyDeviceToHost(int* devArr, int* hostArr) {
  HIP_CHECK(hipMemcpyAsync(hostArr, devArr, Nbytes,
                           hipMemcpyDeviceToHost,
                           hipStreamLegacy));
}

/**
 * Test Description
 * ------------------------
 *  - This test, tests the following sceanrio with per thread compiler option:-
 *  - Launch two threads,
 *  -  In thread  1  : H -> D Copy
 *  -  In thread  2  : D -> H Copy
 *  - These two threads should run concurrently since each has its own stream
 *  - and the final result will be inconsistent which is in hostArrDst.
 *  - This test ensures, even with per thread compiler option, operations will
 *  - run without any crash/conflcits.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamLegacy_compiler_options.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipStreamLegacy_TwoThreadsDiffOperationWithSptCompOption") {
  int *hostArrSrc = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostArrSrc != nullptr);
  fillArr(hostArrSrc, N, 50);

  int *devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, Nbytes));
  REQUIRE(devArr != nullptr);

  int *hostArrDst = reinterpret_cast<int *>(malloc(Nbytes));
  REQUIRE(hostArrDst != nullptr);
  fillArr(hostArrDst, N, 52);

  ::std::thread H2D_Thread(copyHostToDevice, hostArrSrc, devArr);
  ::std::thread D2H_Thread(copyDeviceToHost, devArr, hostArrDst);

  H2D_Thread.join();
  D2H_Thread.join();

  free(hostArrSrc);
  free(hostArrDst);
  HIP_CHECK(hipFree(devArr));
}
