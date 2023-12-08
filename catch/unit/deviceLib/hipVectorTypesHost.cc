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
#include <hip/hip_vector_types.h>

#include <string>
#include <sstream>
#include <type_traits>

using namespace std; // NOLINT

template<typename V,
    enable_if_t<!is_integral<decltype(declval<V>().x)>{}>* = nullptr>
bool integer_unary_tests(V&, V&) {
  return true;
}

template<typename V,
    enable_if_t<!is_integral<decltype(declval<V>().x)>{}>* = nullptr>
bool integer_binary_tests(V&, V&, V&...) {
  return true;
}

template<typename V,
    enable_if_t<is_integral<decltype(declval<V>().x)>{}>* = nullptr>
bool integer_unary_tests(V f1, V f2) {
  f1 %= f2;
  if (f1 != V{0}) return false;
  f1 &= f2;
  if (f1 != V{0}) return false;
  f1 |= f2;
  if (f1 != V{1}) return false;
  f1 ^= f2;
  if (f1 != V{0}) return false;
  f1 = V{1};
  f1 <<= f2;
  if (f1 != V{2}) return false;
  f1 >>= f2;
  if (f1 != V{1}) return false;
  f2 = ~f1;
  return f2 == V{~1};
}

template<typename V,
  enable_if_t<is_integral<decltype(declval<V>().x)>{}>* = nullptr>
bool integer_binary_tests(V f1, V f2, V f3) {
  f3 = f1 % f2;
  if (f3 != V{0}) return false;
  f1 = f3 & f2;
  if (f1 != V{0}) return false;
  f2 = f1 ^ f3;
  if (f2 != V{0}) return false;
  f1 = V{1};
  f2 = V{2};
  f3 = f1 << f2;
  if (f3 != V{4}) return false;
  f2 = f3 >> f1;
  return f2 == V{2};
}

template<typename V>
bool constructor_tests() {
  if (is_constructible<V, unsigned char>{} &&
     is_constructible<V, signed char>{} &&
     is_constructible<V, uint32_t>{} &&
     is_constructible<V, int32_t>{} &&
     is_constructible<V, unsigned int>{} &&
     is_constructible<V, signed int>{} &&
     is_constructible<V, uint64_t>{} &&
     is_constructible<V, int64_t>{} &&
     is_constructible<V, uint64_t>{} &&
     is_constructible<V, int64_t>{} &&
     is_constructible<V, float>{} &&
     is_constructible<V, double>{}) {
    return true;
  }
}

template<typename V>
bool TestVectorType() {
  constexpr V v1{1};
  constexpr V v2{2};
  constexpr V v3{3};
  constexpr V v4{4};

  V f1{1};
  V f2{1};
  V f3 = f1 + f2;
  if (f3 != v2) return false;
  f2 = f3 - f1;
  if (f2 != v1) return false;
  f1 = f2 * f3;
  if (f1 != v2) return false;
  f2 = f1 / f3;
  if (f2 != v1) return false;
  if (!integer_binary_tests(f1, f2, f3)) return false;

  f1 = V{2};
  f2 = V{1};
  f1 += f2;
  if (f1 != v3) return false;
  f1 -= f2;
  if (f1 != v2) return false;
  f1 *= f2;
  if (f1 != v2) return false;
  f1 /= f2;
  if (f1 != v2) return false;
  if (!integer_unary_tests(f1, f2)) return false;

  f1 = v2;
  f2 = f1++;
  if (f1 != v3) return false;
  if (f2 != v2) return false;
  f2 = f1--;
  if (f2 != v3) return false;
  if (f1 != v2) return false;
  f2 = ++f1;
  if (f1 != v3) return false;
  if (f2 != v3) return false;
  f2 = --f1;
  if (f1 != v2) return false;
  if (f2 != v2) return false;

  if (!constructor_tests<V>()) return false;

  f1 = v3;
  f2 = v4;
  f3 = v3;
  if (f1 == f2) return false;
  if (!(f1 != f2)) return false;

  using T = typename V::value_type;

  const T& x = f1.x;
  T& y = f2.x;
  const volatile T& z = f3.x;
  volatile T& w = f2.x;

  if (x != T{3}) return false;
  if (y != T{4}) return false;
  if (z != T{3}) return false;
  if (w != T{4}) return false;

  stringstream str;
  str << f1.x;
  str >> f2.x;

  if (f1.x != f2.x) return false;

  return true;
}

template<typename... Ts, enable_if_t<sizeof...(Ts) == 0>* = nullptr>
bool TestVectorTypes() {
  return true;
}

template<typename T, typename... Ts>
bool TestVectorTypes() {
  if (!TestVectorType<T>()) return false;
  return TestVectorTypes<Ts...>();
}

bool CheckVectorTypes() {
  return TestVectorTypes<
         char1, char2, char3, char4,
         uchar1, uchar2, uchar3, uchar4,
         short1, short2, short3, short4,
         ushort1, ushort2, ushort3, ushort4,
         int1, int2, int3, int4,
         uint1, uint2, uint3, uint4,
         long1, long2, long3, long4,
         ulong1, ulong2, ulong3, ulong4,
         longlong1, longlong2, longlong3, longlong4,
         ulonglong1, ulonglong2, ulonglong3, ulonglong4,
         float1, float2, float3, float4,
         double1, double2, double3, double4>();
}
TEST_CASE("Unit_hipVectorTypes_test_on_host") {
  REQUIRE(sizeof(float1) == 4);
  REQUIRE(sizeof(float2) >= 8);
  REQUIRE(sizeof(float3) == 12);
  REQUIRE(sizeof(float4) >= 16);

  bool result = false;
  result = CheckVectorTypes();
  REQUIRE(result == true);
}
