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

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipStreamCreate hipStreamCreate
 * @{
 * @ingroup StreamTest
 * `hipError_t hipStreamCreate(hipStream_t* stream)` -
 * Create a new asynchronous stream.
 */

/**
 * Test Description
 * ------------------------
 *    - Test case to verify hipStreamCreate performance by recording below sets of time.
 * create 4 set of streams and record that time taken as b1
 * create another 4 set of streams before destroying earlier streams and record time taken as b2
 * destroy streams and then create 4 streams again and record time taken as b3
 * and verify if the condition b1 > b2 > b3  is satisfied.

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamCreatePerformance.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamCreate_Performance") {
  HIP_CHECK(hipSetDevice(0)); // just to initialise HIP runtime.
  // create stream
  hipStream_t streamb1[4];
  // record time for batch1 stream creation
  auto Startb1 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreate(&streamb1[k]));
  }
  auto Stopb1 = std::chrono::high_resolution_clock::now();
  double performb1 = std::chrono::duration<double, std::micro>(Stopb1 - Startb1).count(); // NOLINT
  printf("Stream create performance for batch1 is %lf\n", performb1);

  // record time for batch2 stream creation before
  // destroying already created streams
  hipStream_t streamb2[4];
  auto Startb2 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreate(&streamb2[k]));
  }
  auto Stopb2 = std::chrono::high_resolution_clock::now();
  double performb2 = std::chrono::duration<double, std::micro>(Stopb2 - Startb2).count(); // NOLINT
  printf("Stream create performance for batch2 is %lf\n", performb2);
  // destroy batch 1 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb1[k]));
  }
  // destroy batch 2 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb2[k]));
  }
  // record time for batch3 stream creation after stream destroy
  hipStream_t streamb3[4];
  auto Startb3 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
       HIP_CHECK(hipStreamCreate(&streamb3[k]));
  }
  auto Stopb3 = std::chrono::high_resolution_clock::now();
  double performb3 = std::chrono::duration<double, std::micro>(Stopb3 - Startb3).count(); // NOLINT
  printf("Stream create performance for batch3 is %lf\n", performb3);

  // destroy streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb3[k]));
  }
  REQUIRE(performb1 > performb2);
  REQUIRE(performb2 > performb3);
}

/**
 * End doxygen group hipStreamCreate.
 * @}
 */
 
/**
 * @addtogroup hipStreamCreateWithFlags hipStreamCreateWithFlags
 * @{
 * @ingroup StreamTest
 * `hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags)` -
 * Create a new asynchronous stream.
 */

/**
 * Test Description
 * ------------------------
 *    - Test case to verify hipStreamCreateWithFlags performance with 
 * hipStreamNonBlocking flagby recording below sets of time.
 * create 4 set of streams and record that time taken as b1
 * create another 4 set of streams before destroying earlier streams
 * and record time taken as b2
 * destroy streams and then create 4 streams again and record time taken as b3
 * and verify if the condition b1 > b2 > b3  is satisfied.

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamCreatePerformance.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */
 
TEST_CASE("Unit_hipStreamCreate_WithFlagsPerformance_Nonblocking") {
  HIP_CHECK(hipSetDevice(0)); // just to initialise HIP runtime.
  // create stream
  hipStream_t streamb1[4];
  // record time for batch1 stream creation
  auto Startb1 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithFlags(&streamb1[k], hipStreamNonBlocking));
  }
  auto Stopb1 = std::chrono::high_resolution_clock::now();
  double performb1 = std::chrono::duration<double, std::micro>(Stopb1 - Startb1).count(); // NOLINT
  printf("Stream create performance for batch1 is %lf\n", performb1);

  // record time for batch2 stream creation before
  // destroying already created streams
  hipStream_t streamb2[4];
  auto Startb2 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithFlags(&streamb2[k], hipStreamNonBlocking));
  }
  auto Stopb2 = std::chrono::high_resolution_clock::now();
  double performb2 = std::chrono::duration<double, std::micro>(Stopb2 - Startb2).count(); // NOLINT
  printf("Stream create performance for batch2 is %lf\n", performb2);
  // destroy batch 1 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb1[k]));
  }
  // destroy batch 2 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb2[k]));
  }
  // record time for batch3 stream creation after stream destroy
  hipStream_t streamb3[4];
  auto Startb3 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
       HIP_CHECK(hipStreamCreateWithFlags(&streamb3[k], hipStreamNonBlocking));
  }
  auto Stopb3 = std::chrono::high_resolution_clock::now();
  double performb3 = std::chrono::duration<double, std::micro>(Stopb3 - Startb3).count(); // NOLINT
  printf("Stream create performance for batch3 is %lf\n", performb3);

  // destroy streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb3[k]));
  }
  REQUIRE(performb1 > performb2);
  REQUIRE(performb2 > performb3);
}

/**
 * Test Description
 * ------------------------
 *    - Test case to verify hipStreamCreateWithFlags performance
 * with hipStreamDefault flagby recording below sets of time.
 * create 4 set of streams and record that time taken as b1
 * create another 4 set of streams before destroying earlier streams
 * and record time taken as b2
 * destroy streams and then create 4 streams again and record time taken as b3
 * and verify if the condition b1 > b2 > b3  is satisfied.

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamCreatePerformance.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamCreate_WithFlagsPerformance_Default") {
  HIP_CHECK(hipSetDevice(0)); // just to initialise HIP runtime.
  // create stream
  hipStream_t streamb1[4];
  // record time for batch1 stream creation
  auto Startb1 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithFlags(&streamb1[k], hipStreamDefault));
  }
  auto Stopb1 = std::chrono::high_resolution_clock::now();
  double performb1 = std::chrono::duration<double, std::micro>(Stopb1 - Startb1).count(); // NOLINT
  printf("Stream create performance for batch1 is %lf\n", performb1);

  // record time for batch2 stream creation before
  // destroying already created streams
  hipStream_t streamb2[4];
  auto Startb2 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithFlags(&streamb2[k], hipStreamDefault));
  }
  auto Stopb2 = std::chrono::high_resolution_clock::now();
  double performb2 = std::chrono::duration<double, std::micro>(Stopb2 - Startb2).count(); // NOLINT
  printf("Stream create performance for batch2 is %lf\n", performb2);
  // destroy batch 1 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb1[k]));
  }
  // destroy batch 2 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb2[k]));
  }
  // record time for batch3 stream creation after stream destroy
  hipStream_t streamb3[4];
  auto Startb3 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
       HIP_CHECK(hipStreamCreateWithFlags(&streamb3[k], hipStreamDefault));
  }
  auto Stopb3 = std::chrono::high_resolution_clock::now();
  double performb3 = std::chrono::duration<double, std::micro>(Stopb3 - Startb3).count(); // NOLINT
  printf("Stream create performance for batch3 is %lf\n", performb3);

  // destroy streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb3[k]));
  }
  REQUIRE(performb1 > performb2);
  REQUIRE(performb2 > performb3);
}
/**
 * End doxygen group hipStreamCreateWithFlags.
 * @}
 */

/**
 * @addtogroup hipStreamCreateWithPriority hipStreamCreateWithPriority
 * @{
 * @ingroup StreamTest
 * `hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority)` -
 * Create a new asynchronous stream.
 */

/**
 * Test Description
 * ------------------------
 *    - Test case to verify hipStreamCreateWithPriority performance 
 * with hipStreamNonBlocking flag along with low priority
 * by recording below sets of time.
 * create 4 set of streams and record that time taken as b1
 * create another 4 set of streams before destroying earlier streams
 * and record time taken as b2
 * destroy streams and then create 4 streams again and record time taken as b3
 * and verify if the condition b1 > b2 > b3  is satisfied.

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamCreatePerformance.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamCreate_WithPriorityPerformance_Nonblocking_low") {
  int priority_low, priority_high;
  HIP_CHECK(hipSetDevice(0)); // just to initialise HIP runtime.
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  // create stream
  hipStream_t streamb1[4];
  // record time for batch1 stream creation
  auto Startb1 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithPriority(&streamb1[k], hipStreamNonBlocking, priority_low));
  }
  auto Stopb1 = std::chrono::high_resolution_clock::now();
  double performb1 = std::chrono::duration<double, std::micro>(Stopb1 - Startb1).count(); // NOLINT
  printf("Stream create performance for batch1 is %lf\n", performb1);

  // record time for batch2 stream creation before
  // destroying already created streams
  hipStream_t streamb2[4];
  auto Startb2 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithPriority(&streamb2[k], hipStreamNonBlocking, priority_low));
  }
  auto Stopb2 = std::chrono::high_resolution_clock::now();
  double performb2 = std::chrono::duration<double, std::micro>(Stopb2 - Startb2).count(); // NOLINT
  printf("Stream create performance for batch2 is %lf\n", performb2);
  // destroy batch 1 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb1[k]));
  }
  // destroy batch 2 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb2[k]));
  }
  // record time for batch3 stream creation after stream destroy
  hipStream_t streamb3[4];
  auto Startb3 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
       HIP_CHECK(hipStreamCreateWithPriority(&streamb3[k], hipStreamNonBlocking, priority_low));
  }
  auto Stopb3 = std::chrono::high_resolution_clock::now();
  double performb3 = std::chrono::duration<double, std::micro>(Stopb3 - Startb3).count(); // NOLINT
  printf("Stream create performance for batch3 is %lf\n", performb3);

  // destroy streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb3[k]));
  }
  REQUIRE(performb1 > performb2);
  REQUIRE(performb2 > performb3);
}

/**
 * Test Description
 * ------------------------
 *    - Test case to verify hipStreamCreateWithPriority performance 
 * with hipStreamNonBlocking flag along with high priority
 * by recording below sets of time.
 * create 4 set of streams and record that time taken as b1
 * create another 4 set of streams before destroying earlier streams
 * and record time taken as b2
 * destroy streams and then create 4 streams again and record time taken as b3
 * and verify if the condition b1 > b2 > b3  is satisfied.

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamCreatePerformance.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamCreate_WithPriorityPerformance_Nonblocking_high") {
  int priority_low, priority_high;
  HIP_CHECK(hipSetDevice(0)); // just to initialise HIP runtime.
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  // create stream
  hipStream_t streamb1[4];
  // record time for batch1 stream creation
  auto Startb1 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithPriority(&streamb1[k], hipStreamNonBlocking, priority_high));
  }
  auto Stopb1 = std::chrono::high_resolution_clock::now();
  double performb1 = std::chrono::duration<double, std::micro>(Stopb1 - Startb1).count(); // NOLINT
  printf("Stream create performance for batch1 is %lf\n", performb1);

  // record time for batch2 stream creation before
  // destroying already created streams
  hipStream_t streamb2[4];
  auto Startb2 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithPriority(&streamb2[k], hipStreamNonBlocking, priority_high));
  }
  auto Stopb2 = std::chrono::high_resolution_clock::now();
  double performb2 = std::chrono::duration<double, std::micro>(Stopb2 - Startb2).count(); // NOLINT
  printf("Stream create performance for batch2 is %lf\n", performb2);
  // destroy batch 1 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb1[k]));
  }
  // destroy batch 2 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb2[k]));
  }
  // record time for batch3 stream creation after stream destroy
  hipStream_t streamb3[4];
  auto Startb3 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
       HIP_CHECK(hipStreamCreateWithPriority(&streamb3[k], hipStreamNonBlocking, priority_high));
  }
  auto Stopb3 = std::chrono::high_resolution_clock::now();
  double performb3 = std::chrono::duration<double, std::micro>(Stopb3 - Startb3).count(); // NOLINT
  printf("Stream create performance for batch3 is %lf\n", performb3);

  // destroy streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb3[k]));
  }
  REQUIRE(performb1 > performb2);
  REQUIRE(performb2 > performb3);
}

/**
 * Test Description
 * ------------------------
 *    - Test case to verify hipStreamCreateWithPriority performance
 * with hipStreamDefault flag along with low priority by recording below sets of time.
 * create 4 set of streams and record that time taken as b1
 * create another 4 set of streams before destroying earlier streams
 * and record time taken as b2
 * destroy streams and then create 4 streams again and record time taken as b3
 * and verify if the condition b1 > b2 > b3  is satisfied.

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamCreatePerformance.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamCreate_WithPriorityPerformance_Default_low") {
  int priority_low, priority_high;
  HIP_CHECK(hipSetDevice(0)); // just to initialise HIP runtime.
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  // create stream
  hipStream_t streamb1[4];
  // record time for batch1 stream creation
  auto Startb1 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithPriority(&streamb1[k], hipStreamDefault, priority_low));
  }
  auto Stopb1 = std::chrono::high_resolution_clock::now();
  double performb1 = std::chrono::duration<double, std::micro>(Stopb1 - Startb1).count(); // NOLINT
  printf("Stream create performance for batch1 is %lf\n", performb1);

  // record time for batch2 stream creation before
  // destroying already created streams
  hipStream_t streamb2[4];
  auto Startb2 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithPriority(&streamb2[k], hipStreamDefault, priority_low));
  }
  auto Stopb2 = std::chrono::high_resolution_clock::now();
  double performb2 = std::chrono::duration<double, std::micro>(Stopb2 - Startb2).count(); // NOLINT
  printf("Stream create performance for batch2 is %lf\n", performb2);
  // destroy batch 1 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb1[k]));
  }
  // destroy batch 2 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb2[k]));
  }
  // record time for batch3 stream creation after stream destroy
  hipStream_t streamb3[4];
  auto Startb3 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
       HIP_CHECK(hipStreamCreateWithPriority(&streamb3[k], hipStreamDefault, priority_low));
  }
  auto Stopb3 = std::chrono::high_resolution_clock::now();
  double performb3 = std::chrono::duration<double, std::micro>(Stopb3 - Startb3).count(); // NOLINT
  printf("Stream create performance for batch3 is %lf\n", performb3);

  // destroy streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb3[k]));
  }
  REQUIRE(performb1 > performb2);
  REQUIRE(performb2 > performb3);
}

/**
 * Test Description
 * ------------------------
 *    - Test case to verify hipStreamCreateWithPriority performance
 * with hipStreamDefault flag along with high priority by recording below sets of time.
 * create 4 set of streams and record that time taken as b1
 * create another 4 set of streams before destroying earlier streams
 * and record time taken as b2
 * destroy streams and then create 4 streams again and record time taken as b3
 * and verify if the condition b1 > b2 > b3  is satisfied.

 * Test source
 * ------------------------
 *    - catch/unit/stream/hipStreamCreatePerformance.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipStreamCreate_WithPriorityPerformance_Default_high") {
  int priority_low = 1, priority_high = 2;
  HIP_CHECK(hipSetDevice(0)); // just to initialise HIP runtime.
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  // create stream
  hipStream_t streamb1[4];
  // record time for batch1 stream creation
  auto Startb1 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithPriority(&streamb1[k], hipStreamDefault, priority_high));
  }
  auto Stopb1 = std::chrono::high_resolution_clock::now();
  double performb1 = std::chrono::duration<double, std::micro>(Stopb1 - Startb1).count(); // NOLINT
  printf("Stream create performance for batch1 is %lf\n", performb1);

  // record time for batch2 stream creation before
  // destroying already created streams
  hipStream_t streamb2[4];
  auto Startb2 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamCreateWithPriority(&streamb2[k], hipStreamDefault, priority_high));
  }
  auto Stopb2 = std::chrono::high_resolution_clock::now();
  double performb2 = std::chrono::duration<double, std::micro>(Stopb2 - Startb2).count(); // NOLINT
  printf("Stream create performance for batch2 is %lf\n", performb2);
  // destroy batch 1 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb1[k]));
  }
  // destroy batch 2 streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb2[k]));
  }
  // record time for batch3 stream creation after stream destroy
  hipStream_t streamb3[4];
  auto Startb3 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 4; k++) {
       HIP_CHECK(hipStreamCreateWithPriority(&streamb3[k], hipStreamDefault, priority_high));
  }
  auto Stopb3 = std::chrono::high_resolution_clock::now();
  double performb3 = std::chrono::duration<double, std::micro>(Stopb3 - Startb3).count(); // NOLINT
  printf("Stream create performance for batch3 is %lf\n", performb3);

  // destroy streams
  for (int k = 0; k < 4; k++) {
    HIP_CHECK(hipStreamDestroy(streamb3[k]));
  }
  REQUIRE(performb1 > performb2);
  REQUIRE(performb2 > performb3);
}
/**
 * End doxygen group hipStreamCreateWithPriority.
 * @}
 */
