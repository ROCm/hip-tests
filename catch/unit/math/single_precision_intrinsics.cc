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

__device__ float sincosf_wrapper1(float x) {
  float sin, cos;
  __sincosf(x, &sin, &cos);
  return sin;
}

__device__ float sincosf_wrapper2(float x) {
  float sin, cos;
  __sincosf(x, &sin, &cos);
  return cos;
}

#define INTRINSIC_UNARY_FLOAT_KERNEL_DEF(func_name)                                                \
  __global__ void func_name##_kernel(float* const ys, const size_t num_xs, float* const xs) {      \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(xs[i]);                                                                    \
    }                                                                                              \
  }


#define INTRINSIC_BINARY_FLOAT_KERNEL_DEF(func_name)                                               \
  __global__ void func_name##_kernel(float* const ys, const size_t num_xs, float* const x1s,       \
                                     float* const x2s) {                                           \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(x1s[i], x2s[i]);                                                           \
    }                                                                                              \
  }

#define INTRINSIC_UNARY_FLOAT_TEST_DEF(kern_name, ref_func, ulp)                                   \
  INTRINSIC_UNARY_FLOAT_KERNEL_DEF(kern_name)                                                      \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - float") {                              \
    double (*ref)(double) = ref_func;                                                              \
    UnarySinglePrecisionTest(kern_name##_kernel, ref, ULPValidatorBuilderFactory<float>(ulp));     \
  }

#define INTRINSIC_BINARY_FLOAT_TEST_DEF(kern_name, ref_func, ulp)                                  \
  INTRINSIC_BINARY_FLOAT_KERNEL_DEF(kern_name)                                                     \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - float") {                              \
    double (*ref)(double, double) = ref_func;                                                      \
    BinaryFloatingPointTest(kern_name##_kernel, ref, ULPValidatorBuilderFactory<float>(ulp));      \
  }

INTRINSIC_UNARY_FLOAT_TEST_DEF(__fsqrt_rn, std::sqrt, 0);
INTRINSIC_UNARY_FLOAT_TEST_DEF(__expf, std::exp, 1);
INTRINSIC_UNARY_FLOAT_TEST_DEF(__exp10f, exp10, 1);
INTRINSIC_UNARY_FLOAT_TEST_DEF(__logf, std::log, 1);
INTRINSIC_UNARY_FLOAT_TEST_DEF(__log2f, std::log2, 1);
INTRINSIC_UNARY_FLOAT_TEST_DEF(__log10f, std::log10, 1);
INTRINSIC_UNARY_FLOAT_TEST_DEF(__sinf, std::sin, 2);
INTRINSIC_UNARY_FLOAT_TEST_DEF(sincosf_wrapper1, std::sin, 2);
INTRINSIC_UNARY_FLOAT_TEST_DEF(__cosf, std::cos, 2);
INTRINSIC_UNARY_FLOAT_TEST_DEF(sincosf_wrapper2, std::sin, 2);
INTRINSIC_UNARY_FLOAT_TEST_DEF(__tanf, std::tan, 2);

template <typename T> T add(T x, T y) { return x + y; }
INTRINSIC_BINARY_FLOAT_TEST_DEF(__fadd_rn, add<double>, 0);

template <typename T> T sub(T x, T y) { return x + y; }
INTRINSIC_BINARY_FLOAT_TEST_DEF(__fsub_rn, sub<double>, 0);

template <typename T> T mul(T x, T y) { return x * y; }
INTRINSIC_BINARY_FLOAT_TEST_DEF(__fmul_rn, mul<double>, 0);

template <typename T> T div(T x, T y) { return x * y; }
INTRINSIC_BINARY_FLOAT_TEST_DEF(__fdiv_rn, div<double>, 0);

INTRINSIC_BINARY_FLOAT_TEST_DEF(__fdividef, div<double>, 0);
INTRINSIC_BINARY_FLOAT_TEST_DEF(__powf, std::pow, 2);