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
#include "binary_common.hh"

#include <boost/math/special_functions.hpp>

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(sin, 2, 2)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(cos, 2, 2)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(tan, 4, 2)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(asin, 2, 2)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(acos, 2, 2)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(atan, 2, 2)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(sinh, 3, 2)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(cosh, 2, 1)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(tanh, 2, 1)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(asinh, 3, 2)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(acosh, 4, 2)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(atanh, 3, 2)

MATH_UNARY_WITHIN_ULP_TEST_DEF(sinpi, boost::math::sin_pi, 2, 2);

MATH_UNARY_WITHIN_ULP_TEST_DEF(cospi, boost::math::cos_pi, 2, 2);


MATH_BINARY_KERNEL_DEF(atan2)

TEMPLATE_TEST_CASE("Unit_Device_atan2_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  RT (*ref)(RT, RT) = std::atan2;
  const auto ulp = std::is_same_v<float, TestType> ? 3 : 2;

  BinaryFloatingPointTest<TestType>(atan2_kernel<TestType>, ref,
                                    ULPValidatorBuilderFactory<TestType>(2));
}


template <typename T>
__global__ void sincos_kernel(std::pair<T, T>* const ys, const size_t num_xs, T* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    if constexpr (std::is_same_v<float, T>) {
      sincosf(xs[i], &ys[i].first, &ys[i].second);
    } else if constexpr (std::is_same_v<double, T>) {
      sincos(xs[i], &ys[i].first, &ys[i].second);
    }
  }
}

template <typename T> std::pair<T, T> sincos(T x) { return {std::sin(x), std::cos(x)}; }

TEST_CASE("Unit_Device_sincos_Accuracy_Positive - float") {
  SECTION("Brute force") {
    UnarySinglePrecisionBruteForceTest(
        sincos_kernel<float>, sincos<double>,
        PairValidatorBuilderFactory<float>(ULPValidatorBuilderFactory<float>(2)));
  }
}

TEST_CASE("Unit_Device_sincos_Accuracy_Positive - double") {
  const auto validator_builder =
      PairValidatorBuilderFactory<float>(ULPValidatorBuilderFactory<float>(2));

  SECTION("Special values") {
    UnaryDoublePrecisionSpecialValuesTest(sincos_kernel<double>, sincos<long double>,
                                          validator_builder);
  }

  SECTION("Brute force") {
    UnaryDoublePrecisionBruteForceTest(sincos_kernel<double>, sincos<long double>,
                                       validator_builder);
  }
}


template <typename T>
__global__ void sincospi_kernel(std::pair<T, T>* const ys, const size_t num_xs, T* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    if constexpr (std::is_same_v<float, T>) {
      sincospif(xs[i], &ys[i].first, &ys[i].second);
    } else if constexpr (std::is_same_v<double, T>) {
      sincospi(xs[i], &ys[i].first, &ys[i].second);
    }
  }
}

template <typename T> std::pair<T, T> sincospi(T x) {
  return {boost::math::sin_pi(x), boost::math::cos_pi(x)};
}

TEST_CASE("Unit_Device_sincospi_Accuracy_Positive - float") {
  SECTION("Brute force") {
    UnarySinglePrecisionBruteForceTest(
        sincospi_kernel<float>, sincospi<double>,
        PairValidatorBuilderFactory<float>(ULPValidatorBuilderFactory<float>(2)));
  }
}

TEST_CASE("Unit_Device_sincospi_Accuracy_Positive - double") {
  const auto validator_builder =
      PairValidatorBuilderFactory<float>(ULPValidatorBuilderFactory<float>(2));

  SECTION("Special values") {
    UnaryDoublePrecisionSpecialValuesTest(sincospi_kernel<double>, sincospi<long double>,
                                          validator_builder);
  }

  SECTION("Brute force") {
    UnaryDoublePrecisionBruteForceTest(sincospi_kernel<double>, sincospi<long double>,
                                       validator_builder);
  }
}