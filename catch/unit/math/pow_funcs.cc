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
#include "pow_common.hh"

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(exp, 2, 1)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(exp2, 2, 1)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(expm1, 1, 1)


MATH_UNARY_KERNEL_DEF(exp10)

TEST_CASE("Unit_Device_exp10f_Accuracy_Positive") {
  auto exp10_ref = [](double arg) -> double { return std::pow(10, arg); };
  double (*ref)(double) = exp10_ref;
  UnarySinglePrecisionTest(exp10_kernel<float>, ref,
                           ULPValidatorBuilderFactory<float>(2));
}

TEST_CASE("Unit_Device_exp10_Accuracy_Positive") {
  auto exp10_ref = [](long double arg) -> long double { return std::pow(10, arg); };
  long double (*ref)(long double) = exp10_ref;
  UnaryDoublePrecisionTest(exp10_kernel<double>, ref,
                           ULPValidatorBuilderFactory<double>(1));
}

MATH_BINARY_KERNEL_DEF(pow)

TEMPLATE_TEST_CASE("Unit_Device_pow_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto pow_ref = [](RT arg1, RT arg2) -> RT {
    if (std::isinf(arg1) && arg2 < 0)
      return 0; 
    return std::pow(arg1, arg2);
  };
  RT (*ref)(RT, RT) = pow_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 4 : 2;
  BinaryFloatingPointTest(pow_kernel<TestType>, ref,
                                     ULPValidatorBuilderFactory<TestType>(ulp));
}

MATH_POW_INT_KERNEL_DEF(ldexp)

TEMPLATE_TEST_CASE("Unit_Device_ldexp_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  RT (*ref)(RT, int) = std::ldexp;
  const auto ulp = std::is_same_v<float, TestType> ? 4 : 2;
  PowIntFloatingPointTest(ldexp_kernel<TestType,int>, ref,
                                     ULPValidatorBuilderFactory<TestType>(0));
}

MATH_POW_INT_KERNEL_DEF(powi)

TEMPLATE_TEST_CASE("Unit_Device_powi_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto pow_ref = [](RT arg1, int arg2) -> RT {
    if (std::isinf(arg1) && arg2 < 0)
      return 0; 
    return std::pow(arg1, static_cast<RT>(arg2));
  };
  RT (*ref)(RT, int) = pow_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 4 : 2;
  PowIntFloatingPointTest(powi_kernel<TestType,int>, ref,
                                     ULPValidatorBuilderFactory<TestType>(ulp));
}

MATH_POW_INT_KERNEL_DEF(scalbn)

TEMPLATE_TEST_CASE("Unit_Device_scalbn_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  RT (*ref)(RT, int) = std::scalbn;
  const auto ulp = std::is_same_v<float, TestType> ? 4 : 2;
  PowIntFloatingPointTest(scalbn_kernel<TestType,int>, ref,
                                     ULPValidatorBuilderFactory<TestType>(0));
}

MATH_POW_INT_KERNEL_DEF(scalbln)

TEMPLATE_TEST_CASE("Unit_Device_scalbln_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  RT (*ref)(RT, long int) = std::scalbln;
  const auto ulp = std::is_same_v<float, TestType> ? 4 : 2;
  PowIntFloatingPointTest(scalbln_kernel<TestType,long int>, ref,
                                     ULPValidatorBuilderFactory<TestType>(0));
}


template <typename T>
__global__ void frexp_kernel(std::pair<T, int>* const ys, const size_t num_xs, T* const xs) {
  const auto tid = cg::this_grid().thread_rank();
  const auto stride = cg::this_grid().size();

  for (auto i = tid; i < num_xs; i += stride) {
    if constexpr (std::is_same_v<float, T>) {
      ys[i].first = frexpf(xs[i], &ys[i].second);
    } else if constexpr (std::is_same_v<double, T>) {
       ys[i].first = frexp(xs[i], &ys[i].second);
    }
  }
}

template <typename T> std::pair<T, int> frexp_ref(T arg) {
  int exp_v;
  T res = std::frexp(arg, &exp_v);
  return {res, exp_v};
}

TEST_CASE("Unit_Device_frexpf_Accuracy_Positive") {
  UnarySinglePrecisionTest(
      frexp_kernel<float>, frexp_ref<double>,
      PairValidatorBuilderFactory<float, int>(ULPValidatorBuilderFactory<float>(0), EqValidatorBuilderFactory<int>()));
}

TEST_CASE("Unit_Device_frexp_Accuracy_Positive") {
  UnaryDoublePrecisionTest(
      frexp_kernel<double>, frexp_ref<long double>,
      PairValidatorBuilderFactory<double, int>(ULPValidatorBuilderFactory<double>(0), EqValidatorBuilderFactory<int>()));
}