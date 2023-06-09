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

TEMPLATE_TEST_CASE("Unit_make_vector_SanityCheck_Basic_Host", "", char1, uchar1, char2, uchar2,
                   char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2, short3, ushort3,
                   short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4, uint4, long1,
                   ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1, ulonglong1,
                   longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4, float1,
                   float2, float3, float4, double1, double2, double3, double4) {
  auto value = GetTestValue<typename TestType::value_type>(0);
  TestType vector = MakeVectorTypeHost<TestType>(value);
  SanityCheck(vector, value);
}

TEMPLATE_TEST_CASE("Unit_make_vector_SanityCheck_Basic_Device", "", char1, uchar1, char2, uchar2,
                   char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2, short3, ushort3,
                   short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4, uint4, long1,
                   ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1, ulonglong1,
                   longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4, float1,
                   float2, float3, float4, double1, double2, double3, double4) {
  auto value = GetTestValue<typename TestType::value_type>(0);
  TestType vector = MakeVectorTypeDevice<TestType>(value);
  SanityCheck(vector, value);
}

TEMPLATE_TEST_CASE("Unit_vector_and_vector_operations_SanityCheck_Basic_Host", "", char1, uchar1,
                   char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2,
                   short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4,
                   uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1,
                   ulonglong1, longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4,
                   float1, float2, float3, float4, double1, double2, double3, double4) {
  auto value1 = GetTestValue<typename TestType::value_type>(0);
  auto value2 = GetTestValue<typename TestType::value_type>(1);

  using VO = VectorOperation;
  for (const auto operation : {VO::kIncrementPrefix,
                               VO::kIncrementPostfix,
                               VO::kDecrementPrefix,
                               VO::kDecrementPostfix,
                               VO::kAddAssign,
                               VO::kSubtractAssign,
                               VO::kMultiplyAssign,
                               VO::kDivideAssign,
                               VO::kNegate,
                               VO::kBitwiseNot,
                               VO::kModuloAssign,
                               VO::kBitwiseXorAssign,
                               VO::kBitwiseOrAssign,
                               VO::kBitwiseAndAssign,
                               VO::kRightShiftAssign,
                               VO::kLeftShiftAssign,
                               VO::kAdd,
                               VO::kSubtract,
                               VO::kMultiply,
                               VO::kDivide,
                               VO::kEqual,
                               VO::kNotEqual,
                               VO::kModulo,
                               VO::kBitwiseXor,
                               VO::kBitwiseOr,
                               VO::kBitwiseAnd,
                               VO::kRightShift,
                               VO::kLeftShift}) {
    DYNAMIC_SECTION("operation: " << to_string(operation)) {
      TestType vector = PerformVectorsOperationHost<TestType>(operation, value1, value2);
      SanityCheck(operation, vector, value1, value2);
    }
  }
}

TEMPLATE_TEST_CASE("Unit_vector_and_value_type_operations_SanityCheck_Basic_Host", "", char1,
                   uchar1, char2, uchar2, char3, uchar3, char4, uchar4, short1, ushort1, short2,
                   ushort2, short3, ushort3, short4, ushort4, int1, uint1, int2, uint2, int3, uint3,
                   int4, uint4, long1, ulong1, long2, ulong2, long3, ulong3, long4, ulong4,
                   longlong1, ulonglong1, longlong2, ulonglong2, longlong3, ulonglong3, longlong4,
                   ulonglong4, float1, float2, float3, float4, double1, double2, double3, double4) {
  auto value1 = GetTestValue<typename TestType::value_type>(0);
  auto value2 = GetTestValue<typename TestType::value_type>(1);

  using VO = VectorOperation;
  for (const auto operation :
       {VO::kAddAssign, VO::kSubtractAssign, VO::kMultiplyAssign, VO::kDivideAssign, VO::kAdd,
        VO::kSubtract, VO::kMultiply, VO::kDivide, VO::kEqual, VO::kNotEqual, VO::kModulo,
        VO::kBitwiseXor, VO::kBitwiseOr, VO::kBitwiseAnd, VO::kRightShift, VO::kLeftShift}) {
    DYNAMIC_SECTION("operation: " << to_string(operation)) {
      TestType vector = PerformVectorAndValueOperationHost<TestType>(operation, value1, value2);
      SanityCheck(operation, vector, value1, value2);
    }
  }
}
