/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "streamCommon.hh"

/**
 * @addtogroup hipStreamCreate hipStreamCreate
 * @{
 * @ingroup StreamTest
 * `hipStreamCreate(hipStream_t* stream)` -
 * Create an asynchronous stream.
 */

/**
 * Test Description
 * ------------------------
 *  - Create valid stream and check its flags and priority correctness.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamCreate_default") {
  int id = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(id));

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);         // Check if stream has a valid ptr
  REQUIRE(hip::checkStream(stream));  // check its flags and priority
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Validate handling of invalid arguments:
 *    -# When output pointer to the stream is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamCreate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamCreate_Negative") {
  REQUIRE(hipErrorInvalidValue == hipStreamCreate(nullptr));
}
