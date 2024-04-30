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

#include <hip_test_common.hh>

/**
 * @addtogroup hipFreeMipmappedArray hipFreeMipmappedArray
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test for `hipFreeMipmappedArray`.
 * Test source
 * ------------------------
 *    - unit/texture/hipFreeMipmappedArray.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipFreeMipmappedArray_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT;

#ifdef __linux__
    HipTest::HIP_SKIP_TEST("Mipmap APIs are not supported on Linux");
    return;
#endif //__linux__

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipFreeMipmappedArray(nullptr), hipErrorInvalidValue);
  }

  SECTION("double free") {
    hipMipmappedArray_t array;
    hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
    hipExtent extent = make_hipExtent(4, 4, 6);
    unsigned int levels = 4;

    HIP_CHECK(hipMallocMipmappedArray(&array, &desc, extent, levels, 0));

    HIP_CHECK(hipFreeMipmappedArray(array));
    HIP_CHECK_ERROR(hipFreeMipmappedArray(array), hipErrorContextIsDestroyed);
  }
}

/**
* End doxygen group TextureTest.
* @}
*/
