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
  BinaryBruteForceTest<TestType, RT>(atan2_kernel<TestType>, ref,
                                     ULPValidatorBuilderFactory<TestType>(ulp));
}
