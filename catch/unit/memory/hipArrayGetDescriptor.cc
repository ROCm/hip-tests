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

/**
 * @addtogroup hipArrayGetDescriptor hipArrayGetDescriptor
 * @{
 * @ingroup MemoryTest
 * `hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor, hipArray* array)` -
 * Gets a 1D or 2D array descriptor.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

/**
 * Test Description
 * ------------------------
 *  - Basic sanity test for `hipArrayGetDescriptor`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetDescriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipArrayGetDescriptor_Positive_Basic") {
  HIP_ARRAY_DESCRIPTOR expected_desc{};
  using vec_info = vector_info<float>;
  expected_desc.Format = vec_info::format;
  expected_desc.NumChannels = vec_info::size;
  expected_desc.Width = 1024 / sizeof(float);
  expected_desc.Height = 4;

  hipArray_t ptr;
  HIP_CHECK(hipArrayCreate(&ptr, &expected_desc));

  HIP_ARRAY_DESCRIPTOR desc;
  HIP_CHECK(hipArrayGetDescriptor(&desc, ptr));

  REQUIRE(desc.Format == expected_desc.Format);
  REQUIRE(desc.NumChannels == expected_desc.NumChannels);
  REQUIRE(desc.Width == expected_desc.Width);
  REQUIRE(desc.Height == expected_desc.Height);

  HIP_CHECK(hipArrayDestroy(ptr));
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipArrayGetDescriptor`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetDescriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipArrayGetDescriptor_Negative_Parameters") {
  HIP_ARRAY_DESCRIPTOR expected_desc{};
  using vec_info = vector_info<float>;
  expected_desc.Format = vec_info::format;
  expected_desc.NumChannels = vec_info::size;
  expected_desc.Width = 1024 / sizeof(float);
  expected_desc.Height = 4;

  hipArray_t ptr;
  HIP_CHECK(hipArrayCreate(&ptr, &expected_desc));

  HIP_ARRAY_DESCRIPTOR desc;

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipArrayGetDescriptor(nullptr, ptr), hipErrorInvalidValue);
  }

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipArrayGetDescriptor(&desc, nullptr), hipErrorInvalidHandle);
  }

  SECTION("array is freed") {
    HIP_CHECK(hipArrayDestroy(ptr));
    HIP_CHECK_ERROR(hipArrayGetDescriptor(&desc, ptr), hipErrorInvalidHandle);
  }

  static_cast<void>(hipArrayDestroy(ptr));
}