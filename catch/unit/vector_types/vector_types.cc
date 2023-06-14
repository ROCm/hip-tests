/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "vector_operations_common.hh"
#if HT_AMD
#include "negative_negate_unsigned_rtc.hh"
#include "negative_bitwise_float_double_rtc.hh"
#include "negative_calculate_assign_with_value_rtc.hh"
#endif

TEMPLATE_TEST_CASE("Unit_make_vector_SanityCheck_Basic_Host", "", char1, uchar1, char2, uchar2,
                   char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2, short3, ushort3,
                   short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4, uint4, long1,
                   ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1, ulonglong1,
                   longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4, float1,
                   float2, float3, float4, double1, double2, double3, double4) {
  auto value = GetTestValue<decltype(TestType().x)>(0);
  TestType vector = MakeVectorTypeHost<TestType>(value);
  SanityCheck(vector, value);
}

TEMPLATE_TEST_CASE("Unit_make_vector_SanityCheck_Basic_Device", "", char1, uchar1, char2, uchar2,
                   char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2, short3, ushort3,
                   short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4, uint4, long1,
                   ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1, ulonglong1,
                   longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4, float1,
                   float2, float3, float4, double1, double2, double3, double4) {
  auto value = GetTestValue<decltype(TestType().x)>(0);
  TestType vector = MakeVectorTypeDevice<TestType>(value);
  SanityCheck(vector, value);
}

#if HT_AMD
TEMPLATE_TEST_CASE("Unit_VectorAndVectorOperations_SanityCheck_Basic_Host", "", char1, uchar1,
                   char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2,
                   short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4,
                   uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1,
                   ulonglong1, longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4,
                   float1, float2, float3, float4, double1, double2, double3, double4) {
  auto value1 = GetTestValue<decltype(TestType().x)>(0);
  auto value2 = GetTestValue<decltype(TestType().x)>(1);

  for (const auto operation : {VectorOperation::kIncrementPrefix,
                               VectorOperation::kIncrementPostfix,
                               VectorOperation::kDecrementPrefix,
                               VectorOperation::kDecrementPostfix,
                               VectorOperation::kAddAssign,
                               VectorOperation::kSubtractAssign,
                               VectorOperation::kMultiplyAssign,
                               VectorOperation::kDivideAssign,
                               VectorOperation::kNegate,
                               VectorOperation::kBitwiseNot,
                               VectorOperation::kModuloAssign,
                               VectorOperation::kBitwiseXorAssign,
                               VectorOperation::kBitwiseOrAssign,
                               VectorOperation::kBitwiseAndAssign,
                               VectorOperation::kRightShiftAssign,
                               VectorOperation::kLeftShiftAssign,
                               VectorOperation::kAdd,
                               VectorOperation::kSubtract,
                               VectorOperation::kMultiply,
                               VectorOperation::kDivide,
                               VectorOperation::kEqual,
                               VectorOperation::kNotEqual,
                               VectorOperation::kModulo,
                               VectorOperation::kBitwiseXor,
                               VectorOperation::kBitwiseOr,
                               VectorOperation::kBitwiseAnd,
                               VectorOperation::kRightShift,
                               VectorOperation::kLeftShift}) {
    DYNAMIC_SECTION("operation: " << to_string(operation)) {
      TestType vector = PerformVectorOperationHost<TestType>(operation, value1, value2);
      SanityCheck(operation, vector, value1, value2);
    }
  }
}

TEMPLATE_TEST_CASE("Unit_VectorAndValueTypeOperations_SanityCheck_Basic_Host", "", char1, uchar1,
                   char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2,
                   short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4,
                   uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1,
                   ulonglong1, longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4,
                   float1, float2, float3, float4, double1, double2, double3, double4) {
  auto value1 = GetTestValue<decltype(TestType().x)>(0);
  auto value2 = GetTestValue<decltype(TestType().x)>(1);

  for (const auto operation :
       {VectorOperation::kAddAssign, VectorOperation::kSubtractAssign,
        VectorOperation::kMultiplyAssign, VectorOperation::kDivideAssign, VectorOperation::kAdd,
        VectorOperation::kSubtract, VectorOperation::kMultiply, VectorOperation::kDivide,
        VectorOperation::kEqual, VectorOperation::kNotEqual, VectorOperation::kModulo,
        VectorOperation::kBitwiseXor, VectorOperation::kBitwiseOr, VectorOperation::kBitwiseAnd,
        VectorOperation::kRightShift, VectorOperation::kLeftShift}) {
    DYNAMIC_SECTION("operation: " << to_string(operation)) {
      TestType vector = PerformVectorOperationHost<TestType, false>(operation, value1, value2);
      SanityCheck(operation, vector, value1, value2);
    }
  }
}

TEMPLATE_TEST_CASE("Unit_VectorAndVectorOperations_SanityCheck_Basic_Device", "", char1, uchar1,
                   char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2,
                   short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4,
                   uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1,
                   ulonglong1, longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4,
                   float1, float2, float3, float4, double1, double2, double3, double4) {
  auto value1 = GetTestValue<decltype(TestType().x)>(0);
  auto value2 = GetTestValue<decltype(TestType().x)>(1);

  for (const auto operation : {VectorOperation::kIncrementPrefix,
                               VectorOperation::kIncrementPostfix,
                               VectorOperation::kDecrementPrefix,
                               VectorOperation::kDecrementPostfix,
                               VectorOperation::kAddAssign,
                               VectorOperation::kSubtractAssign,
                               VectorOperation::kMultiplyAssign,
                               VectorOperation::kDivideAssign,
                               VectorOperation::kNegate,
                               VectorOperation::kBitwiseNot,
                               VectorOperation::kModuloAssign,
                               VectorOperation::kBitwiseXorAssign,
                               VectorOperation::kBitwiseOrAssign,
                               VectorOperation::kBitwiseAndAssign,
                               VectorOperation::kRightShiftAssign,
                               VectorOperation::kLeftShiftAssign,
                               VectorOperation::kAdd,
                               VectorOperation::kSubtract,
                               VectorOperation::kMultiply,
                               VectorOperation::kDivide,
                               VectorOperation::kEqual,
                               VectorOperation::kNotEqual,
                               VectorOperation::kModulo,
                               VectorOperation::kBitwiseXor,
                               VectorOperation::kBitwiseOr,
                               VectorOperation::kBitwiseAnd,
                               VectorOperation::kRightShift,
                               VectorOperation::kLeftShift}) {
    DYNAMIC_SECTION("operation: " << to_string(operation)) {
      TestType vector = PerformVectorOperationDevice<TestType>(operation, value1, value2);
      SanityCheck(operation, vector, value1, value2);
    }
  }
}

TEMPLATE_TEST_CASE("Unit_VectorAndValueTypeOperations_SanityCheck_Basic_Device", "", char1, uchar1,
                   char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2,
                   short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4,
                   uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1,
                   ulonglong1, longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4,
                   float1, float2, float3, float4, double1, double2, double3, double4) {
  auto value1 = GetTestValue<decltype(TestType().x)>(0);
  auto value2 = GetTestValue<decltype(TestType().x)>(1);

  for (const auto operation :
       {VectorOperation::kAddAssign, VectorOperation::kSubtractAssign,
        VectorOperation::kMultiplyAssign, VectorOperation::kDivideAssign, VectorOperation::kAdd,
        VectorOperation::kSubtract, VectorOperation::kMultiply, VectorOperation::kDivide,
        VectorOperation::kEqual, VectorOperation::kNotEqual, VectorOperation::kModulo,
        VectorOperation::kBitwiseXor, VectorOperation::kBitwiseOr, VectorOperation::kBitwiseAnd,
        VectorOperation::kRightShift, VectorOperation::kLeftShift}) {
    DYNAMIC_SECTION("operation: " << to_string(operation)) {
      TestType vector = PerformVectorOperationDevice<TestType, false>(operation, value1, value2);
      SanityCheck(operation, vector, value1, value2);
    }
  }
}

template <int expected_errors_num> void VectorTypesRTCWrapper(const char* program_source) {
  hiprtcProgram program{};
  HIPRTC_CHECK(hiprtcCreateProgram(&program, program_source, "vector_types_kernels.cc", 0, nullptr,
                                   nullptr));

#if HT_AMD
  std::string args = std::string("-ferror-limit=100");
  const char* options[] = {args.c_str()};
  hiprtcResult result{hiprtcCompileProgram(program, 1, options)};
#else
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};
#endif

  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};

  std::string error_message{"error:"};

  size_t npos_e = log.find(error_message, 0);
  while (npos_e != std::string::npos) {
    ++error_count;
    npos_e = log.find(error_message, npos_e + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_errors_num);
}

TEST_CASE("Unit_VectorOperators_Negative_Parameters_RTC") {
  VectorTypesRTCWrapper<8>(kNegateUnsignedChar);
  VectorTypesRTCWrapper<8>(kNegateUnsignedShort);
  VectorTypesRTCWrapper<8>(kNegateUnsignedInt);
  VectorTypesRTCWrapper<8>(kNegateUnsignedLong);
  VectorTypesRTCWrapper<8>(kNegateUnsignedLongLong);
  VectorTypesRTCWrapper<96>(kBitwiseFloat);
  VectorTypesRTCWrapper<96>(kBitwiseDouble);
  VectorTypesRTCWrapper<96>(kCalculateAssignChar);
  VectorTypesRTCWrapper<96>(kCalculateAssignShort);
  VectorTypesRTCWrapper<96>(kCalculateAssignInt);
  VectorTypesRTCWrapper<96>(kCalculateAssignLong);
  VectorTypesRTCWrapper<96>(kCalculateAssignLongLong);
}
#endif  // HT_AMD
