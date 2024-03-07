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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <utils.hh>
#include "streamCommon.hh"

/**
 * @addtogroup hipStreamSynchronize hipStreamSynchronize
 * @{
 * @ingroup StreamTest
 * `hipStreamSynchronize(hipStream_t stream)` -
 * Wait for all commands in stream to complete.
 */

namespace hipStreamSynchronizeTest {

/**
 * Test Description
 * ------------------------
 *  - Synchronize an empty stream.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamSynchronize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamSynchronize_EmptyStream") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

#if !HT_NVIDIA
// Test removed for Nvidia devices because it returns unexpected error.

/**
 * Test Description
 * ------------------------
 *  - Synchronize an uninitialized stream
 *    - Expected output: return `hipErrorContextIsDestroyed`
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamSynchronize.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamSynchronize_UninitializedStream") {
  hipStream_t stream{reinterpret_cast<hipStream_t>(0xFFFF)};
  HIP_CHECK_ERROR(hipStreamSynchronize(stream), hipErrorContextIsDestroyed);
}
#endif

#if HT_AMD /* Disabled because frequency based wait is timing out on nvidia platforms */

/**
 * Test Description
 * ------------------------
 *  - Check that all work executing in a stream is finished after synchronization.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamSynchronize.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamSynchronize_FinishWork") {
  const hipStream_t explicitStream = reinterpret_cast<hipStream_t>(-1);
  hipStream_t stream = GENERATE_COPY(explicitStream, hip::nullStream, hip::streamPerThread);

  const bool isExplicitStream = stream == explicitStream;
  if (isExplicitStream) {
    HIP_CHECK(hipStreamCreate(&stream));
  }

  LaunchDelayKernel(std::chrono::milliseconds(500), stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamQuery(stream));

  if (isExplicitStream) {
    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Check that synchronizing the nullStream implicitly synchronizes all executing streams.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamSynchronize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamSynchronize_NullStreamSynchronization") {
  int totalStreams = 10;

  std::vector<hipStream_t> streams{};

  for (int i = 0; i < totalStreams; ++i) {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    streams.push_back(stream);
  }

  for (int i = 0; i < totalStreams; ++i) {
    LaunchDelayKernel(std::chrono::milliseconds(1000), streams[i]);
  }

  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK_ERROR(hipStreamQuery(streams[i]), hipErrorNotReady);
  }

  HIP_CHECK(hipStreamSynchronize(hip::nullStream));
  HIP_CHECK(hipStreamQuery(hip::nullStream));

  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK(hipStreamQuery(streams[i]));
  }

  for (int i = 0; i < totalStreams; ++i) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Check that synchronizing one stream does not synchronize other streams.
 *  - Check that submiting work to the nullStream does not affect synchronization of other streams.
 *  - Check that querying the nullStream does not affect synchronization of other streams.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamSynchronize.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (NVIDIA)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamSynchronize_SynchronizeStreamAndQueryNullStream") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-22");
#else

  hipStream_t stream1;
  hipStream_t stream2;

  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));

  LaunchDelayKernel(std::chrono::milliseconds(500), stream1);
  LaunchDelayKernel(std::chrono::milliseconds(2000), stream2);

  SECTION("Do not use NullStream") {}
  SECTION("Submit Kernel to NullStream") {
    hip::stream::empty_kernel<<<1, 1, 0, hip::nullStream> > >();
  }
  SECTION("Query NullStream") {
    HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);
  }

  HIP_CHECK_ERROR(hipStreamQuery(stream1), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(stream2), hipErrorNotReady);


  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipStreamQuery(stream1));
  HIP_CHECK_ERROR(hipStreamQuery(stream2), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);

  HIP_CHECK(hipStreamSynchronize(stream2));
  HIP_CHECK(hipStreamQuery(stream2));

  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Check that synchronizing the null stream also synchronizes the
 *    per thread special stream.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamSynchronize.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamSynchronize_NullStreamAndStreamPerThread") {
  LaunchDelayKernel(std::chrono::milliseconds(500), hip::streamPerThread);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(hip::streamPerThread), hipErrorNotReady);
  LaunchDelayKernel(std::chrono::milliseconds(500), hip::nullStream);
  HIP_CHECK(hipStreamSynchronize(hip::nullStream))
  HIP_CHECK_ERROR(hipStreamQuery(hip::streamPerThread), hipSuccess);
  HIP_CHECK_ERROR(hipStreamQuery(hip::nullStream), hipSuccess);
}
#endif
}  // namespace hipStreamSynchronizeTest