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
#include "math_pow_common.hh"
#include "math_special_values.hh"

#include <hip/hip_cooperative_groups.h>
#include <math.h>
#include <cmath>

namespace cg = cooperative_groups;

MATH_SINGLE_ARG_KERNEL_DEF(exp)

TEMPLATE_TEST_CASE("Unit_Math_exp_Positive", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T) = exp;
  int64_t ulps = std::is_same_v<TestType, float> ? 2 : 1;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  MathTest(ULPValidator{ulps}, 1u, special_vals.size, exp_kernel<TestType>, ref, special_vals.size,
           special_vals.data);
}

MATH_SINGLE_ARG_KERNEL_DEF(exp2)

TEMPLATE_TEST_CASE("Unit_Math_exp2_Positive", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T) = exp2;
  int64_t ulps = std::is_same_v<TestType, float> ? 2 : 1;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  MathTest(ULPValidator{ulps}, 1u, special_vals.size, exp2_kernel<TestType>, ref, special_vals.size,
           special_vals.data);
}

MATH_SINGLE_ARG_KERNEL_DEF(exp10)

TEMPLATE_TEST_CASE("Unit_Math_exp10_Positive", "", float, double) {
  using T = RefType_t<TestType>;
  auto exp10_ref = [](T arg) -> T { return pow(10, arg); };
  T (*ref)(T) = exp10_ref;
  int64_t ulps = std::is_same_v<TestType, float> ? 2 : 1;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  MathTest(ULPValidator{ulps}, 1u, special_vals.size, exp10_kernel<TestType>, ref, special_vals.size,
           special_vals.data);
}

MATH_SINGLE_ARG_KERNEL_DEF(expm1)

TEMPLATE_TEST_CASE("Unit_Math_expm1_Positive", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T) = expm1;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  MathTest(ULPValidator{1}, 1u, special_vals.size, expm1_kernel<TestType>, ref, special_vals.size,
           special_vals.data);
}

MATH_DOUBLE_ARG_KERNEL_DEF(pow)

TEMPLATE_TEST_CASE("Unit_Math_pow_Positive", "", float, double) {
  using T = RefType_t<TestType>;
  TestType (*ref)(TestType, TestType) = pow;
  int64_t ulps = std::is_same_v<TestType, float> ? 4 : 2;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  TestType arg1[special_vals.size];
  for (int i = 0; i < special_vals.size; i++) {
    std::fill_n(arg1, special_vals.size, special_vals.data[i]);
    MathTest(ULPValidator{ulps}, 1u, special_vals.size, pow_kernel<TestType>, ref, special_vals.size,
            arg1, special_vals.data);
  }
}

MATH_POW_DOUBLE_INT_ARG_KERNEL_DEF(ldexp)

TEMPLATE_TEST_CASE("Unit_Math_ldexp_Positive", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T, int) = ldexp;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  const auto& special_vals_int = std::get<SpecialVals<int>>(kSpecialValRegistry);
  int arg2[special_vals.size];
  for (int i = 0; i < special_vals_int.size; i++) {
    std::fill_n(arg2, special_vals.size, special_vals_int.data[i]);
    PowIntTest(ULPValidator{0}, 1u, special_vals.size, ldexp_kernel<TestType>, ref, special_vals.size,
            special_vals.data, arg2);
  }
}

MATH_POW_DOUBLE_INT_ARG_KERNEL_DEF(powi)

TEMPLATE_TEST_CASE("Unit_Math_powi_Positive", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T, T) = pow;
  int64_t ulps = std::is_same_v<TestType, float> ? 4 : 2;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  const auto& special_vals_int = std::get<SpecialVals<int>>(kSpecialValRegistry);
  int arg2[special_vals.size];
  for (int i = 0; i < special_vals_int.size; i++) {
    std::fill_n(arg2, special_vals.size, special_vals_int.data[i]);
    PowIntTest(ULPValidator{ulps}, 1u, special_vals.size, powi_kernel<TestType>, ref, special_vals.size,
            special_vals.data, arg2);
  }
}

MATH_POW_DOUBLE_INT_ARG_KERNEL_DEF(scalbn)

TEMPLATE_TEST_CASE("Unit_Math_scalbn_Positive", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T, int) = scalbn;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  const auto& special_vals_int = std::get<SpecialVals<int>>(kSpecialValRegistry);
  int arg2[special_vals.size];
  for (int i = 0; i < special_vals_int.size; i++) {
    std::fill_n(arg2, special_vals.size, special_vals_int.data[i]);
    PowIntTest(ULPValidator{0}, 1u, special_vals.size, scalbn_kernel<TestType>, ref, special_vals.size,
            special_vals.data, arg2);
  }
}

MATH_POW_DOUBLE_INT_ARG_KERNEL_DEF(scalbln)

TEMPLATE_TEST_CASE("Unit_Math_scalbln_Positive", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T, long int) = scalbln;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  const auto& special_vals_long = std::get<SpecialVals<int>>(kSpecialValRegistry);
  long int arg2[special_vals.size];
  for (int i = 0; i < special_vals_long.size; i++) {
    std::fill_n(arg2, special_vals.size, special_vals_long.data[i]);
    PowIntTest(ULPValidator{0}, 1u, special_vals.size, scalbln_kernel<TestType>, ref, special_vals.size,
            special_vals.data, arg2);
  }
}

MATH_POW_FREXP_ARG_KERNEL_DEF(frexp)

TEMPLATE_TEST_CASE("Unit_Math_frexp_Positive", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T, int*) = frexp;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  int arg2[special_vals.size];
  for (int i = 0; i < special_vals.size; i++) {
    PowFrexpTest(0, 1u, special_vals.size, frexp_kernel<TestType>, ref, special_vals.size,
            special_vals.data, arg2);
  }
}