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

#include "unary_common.hh"
#include "math_log_negative_kernels_rtc.hh"

#define MATH_ILOGB_KERNEL_DEF(func_name)                                                           \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(int* const ys, const size_t num_xs, T* const xs) {            \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[i] = func_name##f(xs[i]);                                                               \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[i] = func_name(xs[i]);                                                                  \
      }                                                                                            \
    }                                                                                              \
  }

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(log, 1, 1)
TEST_CASE("Unit_Device_log_logf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLog); }

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(log2, 1, 1)
TEST_CASE("Unit_Device_log2_log2f_Negative_RTC") { NegativeTestRTCWrapper<4>(kLog2); }

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(log10, 2, 1)
TEST_CASE("Unit_Device_log10_log10f_Negative_RTC") { NegativeTestRTCWrapper<4>(kLog10); }

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(log1p, 1, 1)
TEST_CASE("Unit_Device_log1p_log1pf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLog1p); }

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(logb, 0, 0)
TEST_CASE("Unit_Device_logb_logbf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLogb); }

MATH_ILOGB_KERNEL_DEF(ilogb)

TEST_CASE("Unit_Device_ilogbf_Accuracy_Positive") {
  int (*ref)(double) = std::ilogb;
  UnarySinglePrecisionTest(ilogb_kernel<float>, ref, EqValidatorBuilderFactor<int>());
}

TEST_CASE("Unit_Device_ilogb_Accuracy_Positive") {
  int (*ref)(long double) = std::ilogb;
  UnaryDoublePrecisionTest(ilogb_kernel<double>, ref, EqValidatorBuilderFactor<int>());
}

TEST_CASE("Unit_Device_ilogb_ilogbf_Negative_RTC") { NegativeTestRTCWrapper<4>(kIlogb); }
