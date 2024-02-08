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

#include <hip_test_common.hh>

<<<<<<< HEAD
/**
 * @addtogroup hipFreeAsync hipFreeAsync
 * @{
 * @ingroup StreamOTest
 * `hipFreeAsync(void* dev_ptr, hipStream_t stream)` -
 * Frees memory with stream ordered semantics.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemPoolApi_Default
 *  - @ref Unit_hipMemPoolApi_Basic
 */

/**
 * Test Description
 * ------------------------
 *  - Verifies handling of invalid arguments:
 *    -# When the device allocation pointer is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When the stream is invalid
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/memory/hipFreeAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
=======
>>>>>>> origin/develop
TEST_CASE("Unit_hipFreeAsync_negative") {
  HIP_CHECK(hipSetDevice(0));
  void* p = nullptr;
  hipStream_t stream{nullptr};
<<<<<<< HEAD
  hipStreamCreate(&stream);
=======
  HIP_CHECK(hipStreamCreate(&stream));
>>>>>>> origin/develop

  SECTION("dev_ptr is nullptr") { REQUIRE(hipFreeAsync(nullptr, stream) != hipSuccess); }

  SECTION("invalid stream handle") {
    HIP_CHECK(hipMallocAsync(static_cast<void**>(&p), 100, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    hipError_t error = hipFreeAsync(p, reinterpret_cast<hipStream_t>(-1));
    HIP_CHECK(hipFreeAsync(p, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(error != hipSuccess);
  }

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}
