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

#include <hip_test_common.hh>

template <typename T>
typename std::enable_if<sizeof(T) / sizeof(typename T::value_type) == 1>::type SanityCheck(
    T vector, typename T::value_type expected_value) {
  std::cout << "SanityCheck: 1D" << std::endl;
  REQUIRE(vector.x == expected_value);
}

template <typename T>
typename std::enable_if<sizeof(T) / sizeof(typename T::value_type) == 2>::type SanityCheck(
    T vector, typename T::value_type expected_value) {
  std::cout << "SanityCheck: 2D" << std::endl;
  REQUIRE(vector.x == expected_value);
  REQUIRE(vector.y == expected_value);
}

template <typename T>
typename std::enable_if<sizeof(T) / sizeof(typename T::value_type) == 3>::type SanityCheck(
    T vector, typename T::value_type expected_value) {
  std::cout << "SanityCheck: 3D" << std::endl;
  REQUIRE(vector.x == expected_value);
  REQUIRE(vector.y == expected_value);
  REQUIRE(vector.z == expected_value);
}

template <typename T>
typename std::enable_if<sizeof(T) / sizeof(typename T::value_type) == 4>::type SanityCheck(
    T vector, typename T::value_type expected_value) {
  std::cout << "SanityCheck : 4D " << std::endl;
  REQUIRE(vector.x == expected_value);
  REQUIRE(vector.y == expected_value);
  REQUIRE(vector.z == expected_value);
  REQUIRE(vector.w == expected_value);
}

template <typename T> T MakeVectorTypeHost(typename T::value_type value) {
  T vector{};

  if constexpr (std::is_same_v<T, char1>) {
    vector = make_char1(value);
  } else if constexpr (std::is_same_v<T, uchar1>) {
    vector = make_uchar1(value);
  } else if constexpr (std::is_same_v<T, char2>) {
    vector = make_char2(value, value);
  } else if constexpr (std::is_same_v<T, uchar2>) {
    vector = make_uchar2(value, value);
  } else if constexpr (std::is_same_v<T, char3>) {
    vector = make_char3(value, value, value);
  } else if constexpr (std::is_same_v<T, uchar3>) {
    vector = make_uchar3(value, value, value);
  } else if constexpr (std::is_same_v<T, char4>) {
    vector = make_char4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, uchar4>) {
    vector = make_uchar4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, short1>) {
    vector = make_short1(value);
  } else if constexpr (std::is_same_v<T, ushort1>) {
    vector = make_ushort1(value);
  } else if constexpr (std::is_same_v<T, short2>) {
    vector = make_short2(value, value);
  } else if constexpr (std::is_same_v<T, ushort2>) {
    vector = make_ushort2(value, value);
  } else if constexpr (std::is_same_v<T, short3>) {
    vector = make_short3(value, value, value);
  } else if constexpr (std::is_same_v<T, ushort3>) {
    vector = make_ushort3(value, value, value);
  } else if constexpr (std::is_same_v<T, short4>) {
    vector = make_short4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, ushort4>) {
    vector = make_ushort4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, int1>) {
    vector = make_int1(value);
  } else if constexpr (std::is_same_v<T, uint1>) {
    vector = make_uint1(value);
  } else if constexpr (std::is_same_v<T, int2>) {
    vector = make_int2(value, value);
  } else if constexpr (std::is_same_v<T, uint2>) {
    vector = make_uint2(value, value);
  } else if constexpr (std::is_same_v<T, int3>) {
    vector = make_int3(value, value, value);
  } else if constexpr (std::is_same_v<T, uint3>) {
    vector = make_uint3(value, value, value);
  } else if constexpr (std::is_same_v<T, int4>) {
    vector = make_int4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, uint4>) {
    vector = make_uint4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, long1>) {
    vector = make_long1(value);
  } else if constexpr (std::is_same_v<T, ulong1>) {
    vector = make_ulong1(value);
  } else if constexpr (std::is_same_v<T, long2>) {
    vector = make_long2(value, value);
  } else if constexpr (std::is_same_v<T, ulong2>) {
    vector = make_ulong2(value, value);
  } else if constexpr (std::is_same_v<T, long3>) {
    vector = make_long3(value, value, value);
  } else if constexpr (std::is_same_v<T, ulong3>) {
    vector = make_ulong3(value, value, value);
  } else if constexpr (std::is_same_v<T, long4>) {
    vector = make_long4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, ulong4>) {
    vector = make_ulong4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, longlong1>) {
    vector = make_longlong1(value);
  } else if constexpr (std::is_same_v<T, ulonglong1>) {
    vector = make_ulonglong1(value);
  } else if constexpr (std::is_same_v<T, longlong2>) {
    vector = make_longlong2(value, value);
  } else if constexpr (std::is_same_v<T, ulonglong2>) {
    vector = make_ulonglong2(value, value);
  } else if constexpr (std::is_same_v<T, longlong3>) {
    vector = make_longlong3(value, value, value);
  } else if constexpr (std::is_same_v<T, ulonglong3>) {
    vector = make_ulonglong3(value, value, value);
  } else if constexpr (std::is_same_v<T, longlong4>) {
    vector = make_longlong4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, ulonglong4>) {
    vector = make_ulonglong4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, float1>) {
    vector = make_float1(value);
  } else if constexpr (std::is_same_v<T, float2>) {
    vector = make_float2(value, value);
  } else if constexpr (std::is_same_v<T, float3>) {
    vector = make_float3(value, value, value);
  } else if constexpr (std::is_same_v<T, float4>) {
    vector = make_float4(value, value, value, value);
  } else if constexpr (std::is_same_v<T, double1>) {
    vector = make_double1(value);
  } else if constexpr (std::is_same_v<T, double2>) {
    vector = make_double2(value, value);
  } else if constexpr (std::is_same_v<T, double3>) {
    vector = make_double3(value, value, value);
  } else if constexpr (std::is_same_v<T, double4>) {
    vector = make_double4(value, value, value, value);
  }

  return vector;
}
