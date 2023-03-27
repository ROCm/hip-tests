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
#include "math_common.hh"
#include "math_special_values.hh"

#include <hip/hip_cooperative_groups.h>
#include <math.h>
#include <cmath>

namespace cg = cooperative_groups;

MATH_SINGLE_ARG_KERNEL_DEF(sqrt)

TEMPLATE_TEST_CASE("Unit_Math_sqrt", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T) = sqrt;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  MathTest(ULPValidator{1}, 1u, special_vals.size, sqrt_kernel<TestType>, ref, special_vals.size,
           special_vals.data);
}

MATH_SINGLE_ARG_KERNEL_DEF(rsqrt)

template <typename T> T rsqrt_ref(T arg) {
  return 1. / sqrt(arg);
}

TEMPLATE_TEST_CASE("Unit_Math_rsqrt", "", float, double) {
  using T = RefType_t<TestType>;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  MathTest(ULPValidator{2}, 1u, special_vals.size, rsqrt_kernel<TestType>, rsqrt_ref<T>, special_vals.size,
           special_vals.data);
}

MATH_SINGLE_ARG_KERNEL_DEF(cbrt)

TEMPLATE_TEST_CASE("Unit_Math_cbrt", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T) = cbrt;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  MathTest(ULPValidator{1}, 1u, special_vals.size, cbrt_kernel<TestType>, ref, special_vals.size,
           special_vals.data);
}

MATH_SINGLE_ARG_KERNEL_DEF(rcbrt)

template <typename T> T rcbrt_ref(T arg) {
  return 1. / cbrt(arg);
}

TEMPLATE_TEST_CASE("Unit_Math_rcbrt", "", float, double) {
  using T = RefType_t<TestType>;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  MathTest(ULPValidator{1}, 1u, special_vals.size, rcbrt_kernel<TestType>, rcbrt_ref<T>, special_vals.size,
           special_vals.data);
}

MATH_DOUBLE_ARG_KERNEL_DEF(hypot)

TEMPLATE_TEST_CASE("Unit_Math_hypot", "", float, double) {
  using T = RefType_t<TestType>;
  T (*ref)(T, T) = hypot;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  TestType arg1[special_vals.size];
  for (int i = 0; i < special_vals.size; i++) {
    std::fill_n(arg1, special_vals.size, special_vals.data[i]);
    MathTest(ULPValidator{3}, 1u, special_vals.size, hypot_kernel<TestType>, ref, special_vals.size,
            arg1, special_vals.data);
  }
}

MATH_DOUBLE_ARG_KERNEL_DEF(rhypot)

template <typename T> T rhypot_ref(T arg1, T arg2) {
  return 1. / hypot(arg1, arg2);
}

TEMPLATE_TEST_CASE("Unit_Math_rhypot", "", float, double) {
  using T = RefType_t<TestType>;
  const auto& special_vals = std::get<SpecialVals<TestType>>(kSpecialValRegistry);
  TestType arg1[special_vals.size];
  for (int i = 0; i < special_vals.size; i++) {
    std::fill_n(arg1, special_vals.size, special_vals.data[i]);
    MathTest(ULPValidator{2}, 1u, special_vals.size, rhypot_kernel<TestType>, rhypot_ref<T>, special_vals.size,
            arg1, special_vals.data);
  }
}