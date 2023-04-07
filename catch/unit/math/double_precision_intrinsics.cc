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

#include "unary_common.hh"
#include "binary_common.hh"

#define INTRINSIC_UNARY_DOUBLE_KERNEL_DEF(func_name)                                               \
  __global__ void func_name##_kernel(double* const ys, const size_t num_xs, double* const xs) {    \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(xs[i]);                                                                    \
    }                                                                                              \
  }

#define INTRINSIC_BINARY_DOUBLE_KERNEL_DEF(func_name)                                              \
  __global__ void func_name##_kernel(double* const ys, const size_t num_xs, double* const x1s,     \
                                     double* const x2s) {                                          \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(x1s[i], x2s[i]);                                                           \
    }                                                                                              \
  }

#define INTRINSIC_UNARY_DOUBLE_TEST_DEF(kern_name, ref_func, ulp)                                  \
  INTRINSIC_UNARY_DOUBLE_KERNEL_DEF(kern_name)                                                     \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - double") {                             \
    long double (*ref)(long double) = ref_func;                                                    \
    UnaryDoublePrecisionTest(kern_name##_kernel, ref, ULPValidatorBuilderFactory<double>(ulp));    \
  }

#define INTRINSIC_BINARY_DOUBLE_TEST_DEF(kern_name, ref_func, ulp)                                 \
  INTRINSIC_BINARY_DOUBLE_KERNEL_DEF(kern_name)                                                    \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - double") {                             \
    long double (*ref)(long double, long double) = ref_func;                                       \
    BinaryFloatingPointTest(kern_name##_kernel, ref, ULPValidatorBuilderFactory<double>(ulp));     \
  }

template <typename T> T add(T x, T y) { return x + y; }
INTRINSIC_BINARY_DOUBLE_TEST_DEF(__dadd_rn, add<long double>, 0);

template <typename T> T sub(T x, T y) { return x + y; }
INTRINSIC_BINARY_DOUBLE_TEST_DEF(__dsub_rn, sub<long double>, 0);

template <typename T> T mul(T x, T y) { return x * y; }
INTRINSIC_BINARY_DOUBLE_TEST_DEF(__dmul_rn, mul<long double>, 0);

template <typename T> T div(T x, T y) { return x * y; }
INTRINSIC_BINARY_DOUBLE_TEST_DEF(__ddiv_rn, div<long double>, 0);

INTRINSIC_UNARY_DOUBLE_TEST_DEF(__dsqrt_rn, std::sqrt, 0);