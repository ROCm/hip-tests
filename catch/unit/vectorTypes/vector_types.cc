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

#include "vector_types_common.hh"

__global__ void MakeKernel() {
  uint2 b_2 = make_uint2(42, 43);
  printf("Kernel log: %u, %u\n", b_2.x, b_2.y);

  float4 f_4 = make_float4(2.5f, 1.5f, 3.5f, 4.5f);
  printf("Kernel log: %.2f, %.2f, %.2f, %.2f\n", f_4.x, f_4.y, f_4.z, f_4.w);
}

TEST_CASE("Unit_make_int_Basic") {
  int1 a_1 = make_int1(42);
  std::cout << "value: [" << a_1.x << "]"
            << ", alignment: " << std::alignment_of_v<decltype(a_1)> << std::endl;

  int2 a_2 = make_int2(42, 43);
  std::cout << "value: [" << a_2.x << ", " << a_2.y << "]"
            << ", alignment: " << std::alignment_of_v<decltype(a_2)> << std::endl;
  std::cout << "------------------------------------------------------" << std::endl;

  MakeKernel<<<1, 1, 0, 0>>>();
  HIP_CHECK(hipDeviceSynchronize());
}

TEMPLATE_TEST_CASE("Unit_make_vector_SanityCheck_Basic_Host", "", char1, uchar1, char2, uchar2,
                   char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2, short3, ushort3,
                   short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4, uint4, long1,
                   ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1, ulonglong1,
                   longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4, float1,
                   float2, float3, float4, double1, double2, double3, double4) {
  auto value = static_cast<typename TestType::value_type>(42);
  TestType vector = MakeVectorTypeHost<TestType>(value);
  std::cout << "alignment: " << std::alignment_of_v<TestType> << ", size: " << sizeof(TestType);

  size_t dimension = sizeof(TestType) / sizeof(typename TestType::value_type);
  switch (dimension) {
    case 1:
      std::cout << ", dimension 1" << std::endl;
      break;
    case 2:
      std::cout << ", dimension 2" << std::endl;
      break;
    case 3:
      std::cout << ", dimension 3" << std::endl;
      break;
    case 4:
      std::cout << ", dimension 4" << std::endl;
  }

  SanityCheck(vector, value);
}

TEMPLATE_TEST_CASE("Unit_make_vector_SanityCheck_Basic_Device", "", char1, uchar1, char2, uchar2,
                   char3, uchar3, char4, uchar4, short1, ushort1, short2, ushort2, short3, ushort3,
                   short4, ushort4, int1, uint1, int2, uint2, int3, uint3, int4, uint4, long1,
                   ulong1, long2, ulong2, long3, ulong3, long4, ulong4, longlong1, ulonglong1,
                   longlong2, ulonglong2, longlong3, ulonglong3, longlong4, ulonglong4, float1,
                   float2, float3, float4, double1, double2, double3, double4) {
  auto value = static_cast<typename TestType::value_type>(42);
  TestType vector = MakeVectorTypeDevice<TestType>(value);

  SanityCheck(vector, value);
}
