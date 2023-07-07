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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "complex_function_common.hh"
#include "complex_cast_negative_kernels_rtc.hh"
#include "complex_make_negative_kernels_rtc.hh"
#include "complex_negative_kernels_1Arg_rtc.hh"
#include "complex_negative_kernels_2Arg_rtc.hh"
#include "complex_negative_kernels_3Arg_rtc.hh"

/**
 * Test Description
 * ------------------------
 *    - Test that checks unary complex functions on device for reduced set of input values. The
 * results are compared against manually calculated ones:
 *    -# hipConj, hipConjf
 *    -# hipCreal, hipCrealf
 *    -# hipCimag, hipCimagf
 *    -# hipCabs, hipCabsf
 *    -# hipCsqabs, hipCsqabsf
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_Complex_Unary_Device_Sanity_Positive", "", hipFloatComplex,
                   hipDoubleComplex) {
  decltype(TestType().x) input_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input_i = GENERATE(-4.75, 0, 1.75);

  TestType input_val = MakeComplexType<TestType>(input_r, input_i);
  for (const auto function :
       {ComplexFunction::kConj, ComplexFunction::kReal, ComplexFunction::kImag,
        ComplexFunction::kAbs, ComplexFunction::kSqabs}) {
    DYNAMIC_SECTION("function: " << to_string(function)) {
      ComplexFunctionUnaryDeviceTest(function, input_val);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test that checks unary complex functions on host for reduced set of input values. The
 * results are compared against manually calculated ones:
 *    -# hipConj, hipConjf
 *    -# hipCreal, hipCrealf
 *    -# hipCimag, hipCimagf
 *    -# hipCabs, hipCabsf
 *    -# hipCsqabs, hipCsqabsf
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_Complex_Unary_Host_Sanity_Positive", "", hipFloatComplex,
                   hipDoubleComplex) {
  decltype(TestType().x) input_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input_i = GENERATE(-4.75, 0, 1.75);

  TestType input_val = MakeComplexType<TestType>(input_r, input_i);
  for (const auto function :
       {ComplexFunction::kConj, ComplexFunction::kReal, ComplexFunction::kImag,
        ComplexFunction::kAbs, ComplexFunction::kSqabs}) {
    DYNAMIC_SECTION("function: " << to_string(function)) {
      ComplexFunctionUnaryHostTest(function, input_val);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for unary complex functions.
 *
 * Test source
 * ------------------------
 *    - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_Complex_Unary_Negative_Parameters_RTC") {
  ComplexTypeRTCWrapper<28>(kComplexConj);
  ComplexTypeRTCWrapper<24>(kComplexReal);
  ComplexTypeRTCWrapper<24>(kComplexImag);
  ComplexTypeRTCWrapper<24>(kComplexAbs);
  ComplexTypeRTCWrapper<24>(kComplexSqabs);
}

/**
 * Test Description
 * ------------------------
 *    - Test that checks binary complex functions on device for reduced set of input values. The
 * results are compared against manually calculated ones:
 *    -# hipCadd, hipCaddf
 *    -# hipCsub, hipCsubf
 *    -# hipCmul, hipCmulf
 *    -# hipCdiv, hipCdivf
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_Complex_Binary_Device_Sanity_Positive", "", hipFloatComplex,
                   hipDoubleComplex) {
  decltype(TestType().x) input1_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input1_i = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input2_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input2_i = GENERATE(-4.75, 0, 1.75);

  TestType input_val1 = MakeComplexType<TestType>(input1_r, input1_i);
  TestType input_val2 = MakeComplexType<TestType>(input2_r, input2_i);
  for (const auto function : {ComplexFunction::kAdd, ComplexFunction::kSub, ComplexFunction::kMul,
                              ComplexFunction::kDiv}) {
    DYNAMIC_SECTION("function: " << to_string(function)) {
      ComplexFunctionBinaryDeviceTest(function, input_val1, input_val2);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test that checks binary complex functions on host for reduced set of input values. The
 * results are compared against manually calculated ones:
 *    -# hipCadd, hipCaddf
 *    -# hipCsub, hipCsubf
 *    -# hipCmul, hipCmulf
 *    -# hipCdiv, hipCdivf
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_Complex_Binary_Host_Sanity_Positive", "", hipFloatComplex,
                   hipDoubleComplex) {
  decltype(TestType().x) input1_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input1_i = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input2_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input2_i = GENERATE(-4.75, 0, 1.75);

  TestType input_val1 = MakeComplexType<TestType>(input1_r, input1_i);
  TestType input_val2 = MakeComplexType<TestType>(input2_r, input2_i);
  for (const auto function : {ComplexFunction::kAdd, ComplexFunction::kSub, ComplexFunction::kMul,
                              ComplexFunction::kDiv}) {
    DYNAMIC_SECTION("function: " << to_string(function)) {
      ComplexFunctionBinaryHostTest(function, input_val1, input_val2);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for binary complex functions.
 *
 * Test source
 * ------------------------
 *    - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_Complex_Binary_Negative_Parameters_RTC") {
  ComplexTypeRTCWrapper<44>(kComplexAdd);
  ComplexTypeRTCWrapper<44>(kComplexSub);
  ComplexTypeRTCWrapper<44>(kComplexMul);
  ComplexTypeRTCWrapper<44>(kComplexDiv);
}

/**
 * Test Description
 * ------------------------
 *    - Test that checks hipCfma/hipCfmaf complex functions on device for reduced set of input
 * values. The results are compared against manually calculated ones.
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_Complex_hipCfma_Device_Sanity_Positive", "", hipFloatComplex,
                   hipDoubleComplex) {
  decltype(TestType().x) input1_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input1_i = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input2_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input2_i = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input3_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input3_i = GENERATE(-4.75, 0, 1.75);

  TestType input_val1 = MakeComplexType<TestType>(input1_r, input1_i);
  TestType input_val2 = MakeComplexType<TestType>(input2_r, input2_i);
  TestType input_val3 = MakeComplexType<TestType>(input3_r, input3_i);

  ComplexFunctionTernaryDeviceTest(ComplexFunction::kFma, input_val1, input_val2, input_val3);
}

/**
 * Test Description
 * ------------------------
 *    - Test that checks hipCfma/hipCfmaf complex functions on host for reduced set of input
 * values. The results are compared against manually calculated ones.
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_Complex_hipCfma_Host_Sanity_Positive", "", hipFloatComplex,
                   hipDoubleComplex) {
  decltype(TestType().x) input1_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input1_i = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input2_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input2_i = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input3_r = GENERATE(-4.75, 0, 1.75);
  decltype(TestType().x) input3_i = GENERATE(-4.75, 0, 1.75);

  TestType input_val1 = MakeComplexType<TestType>(input1_r, input1_i);
  TestType input_val2 = MakeComplexType<TestType>(input2_r, input2_i);
  TestType input_val3 = MakeComplexType<TestType>(input3_r, input3_i);

  ComplexFunctionTernaryHostTest(ComplexFunction::kFma, input_val1, input_val2, input_val3);
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for hipCfma/hipCfmaf complex function.
 *
 * Test source
 * ------------------------
 *    - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_Complex_hipCfma_Negative_Parameters_RTC") {
  ComplexTypeRTCWrapper<60>(kComplexFma);
}

/**
 * Test Description
 * ------------------------
 *    - Test that checks make_hipFloatComplex/make_hipDoubleComplex functions on device for reduced
 * set of input values. The results are compared against manually calculated ones.
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_make_Complex_Device_Positive", "", hipFloatComplex,
                   hipDoubleComplex) {
  decltype(TestType().x) input_r = GENERATE(-0.25, 0, 0.25);
  decltype(TestType().x) input_i = GENERATE(-1.75, 0, 1.75);

  LinearAllocGuard<TestType> result_d(LinearAllocs::hipMalloc, sizeof(TestType));
  LinearAllocGuard<TestType> result_h(LinearAllocs::hipHostMalloc, sizeof(TestType));

  MakeComplexTypeKernel<TestType><<<1, 1>>>(result_d.ptr(), input_r, input_i);
  HIP_CHECK(hipMemcpy(result_h.ptr(), result_d.ptr(), sizeof(TestType), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(result_h.ptr()[0].x == input_r);
  REQUIRE(result_h.ptr()[0].y == input_i);
}

/**
 * Test Description
 * ------------------------
 *    - Test that checks make_hipFloatComplex/make_hipDoubleComplex functions on device for reduced
 * set of input values. The results are compared against manually calculated ones.
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_make_Complex_Host_Positive", "", hipFloatComplex,
                   hipDoubleComplex) {
  decltype(TestType().x) input_r = GENERATE(-0.25, 0, 0.25);
  decltype(TestType().x) input_i = GENERATE(-1.75, 0, 1.75);

  TestType result = MakeComplexType<TestType>(input_r, input_i);

  REQUIRE(result.x == input_r);
  REQUIRE(result.y == input_i);
}

#if HT_AMD  // EXSWHTEC-321
/**
 * Test Description
 * ------------------------
 *    - Test that checks make_hipComplex functions on device for reduced set of input values. The
 * results are compared against manually calculated ones.
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_make_hipComplex_Device_Positive") {
  float input_r = GENERATE(-0.25, 0, 0.25);
  float input_i = GENERATE(-1.75, 0, 1.75);

  LinearAllocGuard<hipComplex> result_d(LinearAllocs::hipMalloc, sizeof(hipComplex));
  LinearAllocGuard<hipComplex> result_h(LinearAllocs::hipHostMalloc, sizeof(hipComplex));

  MakeHipComplexTypeKernel<<<1, 1>>>(result_d.ptr(), input_r, input_i);
  HIP_CHECK(hipMemcpy(result_h.ptr(), result_d.ptr(), sizeof(hipComplex), hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(result_h.ptr()[0].x == input_r);
  REQUIRE(result_h.ptr()[0].y == input_i);
}

/**
 * Test Description
 * ------------------------
 *    - Test that checks make_hipComplex functions on host for reduced set of input values. The
 * results are compared against manually calculated ones.
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_make_hipComplex_Host_Positive") {
  float input_r = GENERATE(-0.25, 0, 0.25);
  float input_i = GENERATE(-1.75, 0, 1.75);

  hipComplex result = make_hipComplex(input_r, input_i);

  REQUIRE(result.x == input_r);
  REQUIRE(result.y == input_i);
}
#endif

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for make complex functions.
 *
 * Test source
 * ------------------------
 *    - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_make_Complex_Negative_Parameters_RTC") {
  ComplexTypeRTCWrapper<18>(kMakeHipComplex);
  ComplexTypeRTCWrapper<18>(kMakeHipFloatComplex);
  ComplexTypeRTCWrapper<18>(kMakeHipDoubleComplex);
}

/**
 * Test Description
 * ------------------------
 *    - Test that checks hipComplexDoubleToFloat/hipComplexFloatToDouble functions on device for
 * reduced set of input values. The results are compared against manually calculated ones.
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_Complex_Cast_Device_Sanity_Positive", "", hipFloatComplex,
                   hipDoubleComplex) {
  decltype(TestType().x) input_r = GENERATE(-0.25, 0, 0.25);
  decltype(TestType().x) input_i = GENERATE(-1.75, 0, 1.75);
  TestType input = MakeComplexType<TestType>(input_r, input_i);

  LinearAllocGuard<CastType_t<TestType>> result_d{LinearAllocs::hipMalloc,
                                                  sizeof(CastType_t<TestType>)};
  LinearAllocGuard<CastType_t<TestType>> result_h{LinearAllocs::hipHostMalloc,
                                                  sizeof(CastType_t<TestType>)};

  CastComplexTypeKernel<<<1, 1>>>(result_d.ptr(), input);
  HIP_CHECK(hipMemcpy(result_h.ptr(), result_d.ptr(), sizeof(CastType_t<TestType>),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  REQUIRE(result_h.ptr()[0].x == static_cast<decltype(CastType_t<TestType>().x)>(input_r));
  REQUIRE(result_h.ptr()[0].y == static_cast<decltype(CastType_t<TestType>().x)>(input_i));
}

/**
 * Test Description
 * ------------------------
 *    - Test that checks hipComplexDoubleToFloat/hipComplexFloatToDouble functions on host for
 * reduced set of input values. The results are compared against manually calculated ones.
 *
 * Test source
 * ------------------------
 *  - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_Device_Complex_Cast_Host_Sanity_Positive", "", hipFloatComplex,
                   hipDoubleComplex) {
  decltype(TestType().x) input_r = GENERATE(-0.25, 0, 0.25);
  decltype(TestType().x) input_i = GENERATE(-1.75, 0, 1.75);
  TestType input = MakeComplexType<TestType>(input_r, input_i);

  CastType_t<TestType> result = CastComplexType<CastType_t<TestType>>(input);

  REQUIRE(result.x == static_cast<decltype(CastType_t<TestType>().x)>(input_r));
  REQUIRE(result.y == static_cast<decltype(CastType_t<TestType>().x)>(input_i));
}

/**
 * Test Description
 * ------------------------
 *    - RTCs kernels that pass argument of invalid type for complex cast functions.
 *
 * Test source
 * ------------------------
 *    - unit/complex/complex.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Device_Complex_Cast_Negative_Parameters_RTC") {
  ComplexTypeRTCWrapper<14>(kComplexDoubleToFloat);
  ComplexTypeRTCWrapper<14>(kComplexFloatToDouble);
}
