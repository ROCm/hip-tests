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

MATH_BINARY_KERNEL_DEF(fmod)

TEMPLATE_TEST_CASE("Unit_Device_fmod_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  RT (*ref)(RT, RT) = std::fmod;
  BinaryBruteForceTest<TestType, RT>(fmod_kernel<TestType>, ref,
                                     ULPValidatorBuilderFactory<TestType>(0));
}


MATH_BINARY_KERNEL_DEF(remainder)

TEMPLATE_TEST_CASE("Unit_Device_remainder_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  RT (*ref)(RT, RT) = std::remainder;
  BinaryBruteForceTest<TestType, RT>(remainder_kernel<TestType>, ref,
                                     ULPValidatorBuilderFactory<TestType>(0));
}

MATH_BINARY_KERNEL_DEF(fdim)

TEMPLATE_TEST_CASE("Unit_Device_fdim_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  RT (*ref)(RT, RT) = std::fdim;
  BinaryBruteForceTest<TestType, RT>(fdim_kernel<TestType>, ref,
                                     ULPValidatorBuilderFactory<TestType>(0));
}


MATH_UNARY_KERNEL_DEF(trunc)

TEST_CASE("Unit_Device_truncf_Accuracy_Positive") {
  UnarySinglePrecisionTest(trunc_kernel<float>, std::trunc, ULPValidatorBuilderFactory<float>(0));
}

TEST_CASE("Unit_Device_trunc_Accuracy_Positive") {
  UnaryDoublePrecisionTest(trunc_kernel<double>, std::trunc, ULPValidatorBuilderFactory<double>(0));
}


MATH_UNARY_KERNEL_DEF(round)

TEST_CASE("Unit_Device_roundf_Accuracy_Positive") {
  UnarySinglePrecisionTest(round_kernel<float>, std::round, ULPValidatorBuilderFactory<float>(0));
}

TEST_CASE("Unit_Device_round_Accuracy_Positive") {
  UnaryDoublePrecisionTest(round_kernel<double>, std::round, ULPValidatorBuilderFactory<double>(0));
}


MATH_UNARY_KERNEL_DEF(rint)

TEST_CASE("Unit_Device_rintf_Accuracy_Positive") {
  UnarySinglePrecisionTest(rint_kernel<float>, std::rint, ULPValidatorBuilderFactory<float>(0));
}

TEST_CASE("Unit_Device_rint_Accuracy_Positive") {
  UnaryDoublePrecisionTest(rint_kernel<double>, std::rint, ULPValidatorBuilderFactory<double>(0));
}


MATH_UNARY_KERNEL_DEF(nearbyint)

TEST_CASE("Unit_Device_nearbyintf_Accuracy_Positive") {
  UnarySinglePrecisionTest(nearbyint_kernel<float>, std::nearbyint,
                           ULPValidatorBuilderFactory<float>(0));
}

TEST_CASE("Unit_Device_nearbyint_Accuracy_Positive") {
  UnaryDoublePrecisionTest(nearbyint_kernel<double>, std::nearbyint,
                           ULPValidatorBuilderFactory<double>(0));
}


MATH_UNARY_KERNEL_DEF(ceil)

TEST_CASE("Unit_Device_ceilf_Accuracy_Positive") {
  UnarySinglePrecisionTest(ceil_kernel<float>, std::ceil, ULPValidatorBuilderFactory<float>(0));
}

TEST_CASE("Unit_Device_ceil_Accuracy_Positive") {
  UnaryDoublePrecisionTest(ceil_kernel<double>, std::ceil, ULPValidatorBuilderFactory<double>(0));
}


MATH_UNARY_KERNEL_DEF(floor)

TEST_CASE("Unit_Device_floorf_Accuracy_Positive") {
  UnarySinglePrecisionTest(floor_kernel<float>, std::floor, ULPValidatorBuilderFactory<float>(0));
}

TEST_CASE("Unit_Device_floor_Accuracy_Positive") {
  UnaryDoublePrecisionTest(floor_kernel<double>, std::floor, ULPValidatorBuilderFactory<double>(0));
}
