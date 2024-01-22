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
#include <resource_guards.hh>

__global__ void __brev_kernel(unsigned int* y, unsigned int x) { y[0] = __brev(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__brev(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___brev_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  __brev_kernel<<<1, 1>>>(y.ptr(), 0xAAAAAAAA);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0x55555555);
}

__global__ void __brevll_kernel(unsigned long long int* y, unsigned long long int x) {
  y[0] = __brevll(x);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__brevll(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___brevll_Sanity_Positive") {
  LinearAllocGuard<unsigned long long int> y(LinearAllocs::hipMallocManaged,
                                             sizeof(unsigned long long int));

  __brevll_kernel<<<1, 1>>>(y.ptr(), 0xAAAAAAAAAAAAAAAA);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0x5555555555555555);
}

template <typename T> __global__ void __clz_kernel(T* y, T x) { y[0] = __clz(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__clz(x)`. Run for `int` and `unsigned int` overloads.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device___clz_Sanity_Positive", "", int, unsigned int) {
  LinearAllocGuard<TestType> y(LinearAllocs::hipMallocManaged, sizeof(TestType));

  __clz_kernel<<<1, 1>>>(y.ptr(), static_cast<TestType>(0));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 32);

  TestType x = 1;
  for (int i = 0; i < 32; ++i) {
    __clz_kernel<<<1, 1>>>(y.ptr(), x << i);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == 31 - i);
  }
}

template <typename T> __global__ void __clzll_kernel(T* y, T x) { y[0] = __clzll(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__clzll(x)`. Run for `long long int` and `unsigned long long int`
 * overloads.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device___clzll_Sanity_Positive", "", long long int,
                   unsigned long long int) {
  LinearAllocGuard<TestType> y(LinearAllocs::hipMallocManaged, sizeof(TestType));

  __clzll_kernel<<<1, 1>>>(y.ptr(), static_cast<TestType>(0));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 64);

  TestType x = 1;
  for (int i = 0; i < 64; ++i) {
    __clzll_kernel<<<1, 1>>>(y.ptr(), x << i);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == 63 - i);
  }
}

template <typename T> __global__ void __ffs_kernel(T* y, T x) { y[0] = __ffs(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__ffs(x)`. Run for `int` and `unsigned int` overloads.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device___ffs_Sanity_Positive", "", int, unsigned int) {
  LinearAllocGuard<TestType> y(LinearAllocs::hipMallocManaged, sizeof(TestType));

  __ffs_kernel<<<1, 1>>>(y.ptr(), static_cast<TestType>(0));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0);

  TestType x = 1;
  for (int i = 0; i < 32; ++i) {
    __ffs_kernel<<<1, 1>>>(y.ptr(), x << i);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == i + 1);
  }
}

template <typename T> __global__ void __ffsll_kernel(T* y, T x) { y[0] = __ffsll(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__ffsll(x)`. Run for `long long int` and `unsigned long long int`
 * overloads.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device___ffsll_Sanity_Positive", "", long long int,
                   unsigned long long int) {
  LinearAllocGuard<TestType> y(LinearAllocs::hipMallocManaged, sizeof(TestType));

  __ffsll_kernel<<<1, 1>>>(y.ptr(), static_cast<TestType>(0));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0);

  TestType x = 1;
  for (int i = 0; i < 64; ++i) {
    __ffsll_kernel<<<1, 1>>>(y.ptr(), x << i);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == i + 1);
  }
}

__global__ void __popc_kernel(unsigned int* y, unsigned int x) { y[0] = __popc(x); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__popc(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___popc_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  __popc_kernel<<<1, 1>>>(y.ptr(), 0);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0);

  unsigned int x = 0;
  for (int i = 0; i < 32; ++i) {
    __popc_kernel<<<1, 1>>>(y.ptr(), x |= (1u << i));
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == i + 1);
  }
}

__global__ void __popcll_kernel(unsigned long long int* y, unsigned long long int x) {
  y[0] = __popcll(x);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__popcll(x)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___popcll_Sanity_Positive") {
  LinearAllocGuard<unsigned long long int> y(LinearAllocs::hipMallocManaged,
                                             sizeof(unsigned long long int));

  __popcll_kernel<<<1, 1>>>(y.ptr(), 0);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == 0);

  unsigned long long int x = 0;
  for (int i = 0; i < 64; ++i) {
    __popcll_kernel<<<1, 1>>>(y.ptr(), x |= (1ull << i));
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(y.ptr()[0] == i + 1);
  }
}

__global__ void __mul24_kernel(int* y, int x1, int x2) { y[0] = __mul24(x1, x2); }

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__mul24(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___mul24_Sanity_Positive") {
  LinearAllocGuard<int> y(LinearAllocs::hipMallocManaged, sizeof(int));

  int x1 = GENERATE(0, -42, 42, 0xFFFFFFFF);
  int x2 = GENERATE(0, -42, 42, 0xFFFFFFFF);

  __mul24_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == x1 * x2);
}

__global__ void __umul24_kernel(unsigned int* y, unsigned int x1, unsigned int x2) {
  y[0] = __umul24(x1, x2);
}

/**
 * Test Description
 * ------------------------
 *    - Sanity test for `__umul24(x,y)`.
 *
 * Test source
 * ------------------------
 *    - unit/math/integer_intrinsics.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device___umul24_Sanity_Positive") {
  LinearAllocGuard<unsigned int> y(LinearAllocs::hipMallocManaged, sizeof(unsigned int));

  unsigned int x1 = GENERATE(0, 42, 0xFFFFFF);
  unsigned int x2 = GENERATE(0, 42, 0xFFFFFF);

  __umul24_kernel<<<1, 1>>>(y.ptr(), x1, x2);
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(y.ptr()[0] == x1 * x2);
}