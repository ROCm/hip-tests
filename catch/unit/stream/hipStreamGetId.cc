/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
1) Negative tests for hipStreamGetId.
2) Basic Positive scenario for hipStreamGetId
*/

#include <hip_test_common.hh>

/**
 *  @brief Pass uninitialized stream and id as nullptr to check if the API behaves as expected.
 */
TEST_CASE("Unit_hipStreamGetId_Negative") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  SECTION("Null Pointer") {
    HIP_CHECK_ERROR(hipStreamGetId(stream, nullptr), hipErrorInvalidValue);
  }
}

/**
 *  @brief Pass null stream, legacy stream and streamperthread, check the API behaves as expected.
 *  Also, check the stream id generated is not same for any two streams.
 */
TEST_CASE("Unit_hipStreamGetId_Basic") {
  hipStream_t stream1, stream2;
  unsigned long long id1, id2, id3, id4, id5;
  SECTION("Unique Stream Id") {
    HIP_CHECK(hipStreamCreate(&stream1));
    HIP_CHECK(hipStreamCreate(&stream2));
    HIP_CHECK(hipStreamGetId(stream1, &id1));
    HIP_CHECK(hipStreamGetId(stream2, &id2));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream2));
    REQUIRE(id1 != id2);
  }
  SECTION("Null and legacy stream") {
    HIP_CHECK(hipStreamGetId(nullptr, &id3));
    HIP_CHECK(hipStreamGetId(hipStreamLegacy, &id4));
    REQUIRE(id3 == id4);
  }
  SECTION("StreamPerThread") {
    HIP_CHECK(hipStreamGetId(hipStreamPerThread, &id5));
    REQUIRE(id5);
  }
}
