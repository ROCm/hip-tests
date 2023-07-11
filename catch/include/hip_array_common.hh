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

#pragma once

#include <hip_test_common.hh>

template <class T, size_t N, hipArray_Format Format> struct type_and_size_and_format {
  using type = T;
  static constexpr size_t size = N;
  static constexpr hipArray_Format format = Format;
};

// Create a map of type to scalar type, vector size and scalar type format enum.
// This is useful for creating simpler function that depend on the vector size.
template <typename T> struct vector_info;
template <>
struct vector_info<int> : type_and_size_and_format<int, 1, HIP_AD_FORMAT_SIGNED_INT32> {};
template <> struct vector_info<float> : type_and_size_and_format<float, 1, HIP_AD_FORMAT_FLOAT> {};
template <>
struct vector_info<short> : type_and_size_and_format<short, 1, HIP_AD_FORMAT_SIGNED_INT16> {};
template <>
struct vector_info<char> : type_and_size_and_format<char, 1, HIP_AD_FORMAT_SIGNED_INT8> {};
template <>
struct vector_info<unsigned int>
    : type_and_size_and_format<unsigned int, 1, HIP_AD_FORMAT_UNSIGNED_INT32> {};
template <>
struct vector_info<unsigned short>
    : type_and_size_and_format<unsigned short, 1, HIP_AD_FORMAT_UNSIGNED_INT16> {};
template <>
struct vector_info<unsigned char>
    : type_and_size_and_format<unsigned char, 1, HIP_AD_FORMAT_UNSIGNED_INT8> {};

template <>
struct vector_info<int2> : type_and_size_and_format<int, 2, HIP_AD_FORMAT_SIGNED_INT32> {};
template <> struct vector_info<float2> : type_and_size_and_format<float, 2, HIP_AD_FORMAT_FLOAT> {};
template <>
struct vector_info<short2> : type_and_size_and_format<short, 2, HIP_AD_FORMAT_SIGNED_INT16> {};
template <>
struct vector_info<char2> : type_and_size_and_format<char, 2, HIP_AD_FORMAT_SIGNED_INT8> {};
template <>
struct vector_info<uint2>
    : type_and_size_and_format<unsigned int, 2, HIP_AD_FORMAT_UNSIGNED_INT32> {};
template <>
struct vector_info<ushort2>
    : type_and_size_and_format<unsigned short, 2, HIP_AD_FORMAT_UNSIGNED_INT16> {};
template <>
struct vector_info<uchar2>
    : type_and_size_and_format<unsigned char, 2, HIP_AD_FORMAT_UNSIGNED_INT8> {};

template <>
struct vector_info<int4> : type_and_size_and_format<int, 4, HIP_AD_FORMAT_SIGNED_INT32> {};
template <> struct vector_info<float4> : type_and_size_and_format<float, 4, HIP_AD_FORMAT_FLOAT> {};
template <>
struct vector_info<short4> : type_and_size_and_format<short, 4, HIP_AD_FORMAT_SIGNED_INT16> {};
template <>
struct vector_info<char4> : type_and_size_and_format<char, 4, HIP_AD_FORMAT_SIGNED_INT8> {};
template <>
struct vector_info<uint4>
    : type_and_size_and_format<unsigned int, 4, HIP_AD_FORMAT_UNSIGNED_INT32> {};
template <>
struct vector_info<ushort4>
    : type_and_size_and_format<unsigned short, 4, HIP_AD_FORMAT_UNSIGNED_INT16> {};
template <>
struct vector_info<uchar4>
    : type_and_size_and_format<unsigned char, 4, HIP_AD_FORMAT_UNSIGNED_INT8> {};

template <
  typename T,
  typename std::enable_if<std::is_scalar<T>::value == false>::type* = nullptr>
static inline __host__ __device__ constexpr int rank() {
  return sizeof(T) / sizeof(decltype(T::x));
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 1>::type* = nullptr>
static inline bool isEqual(const T &val0, const T &val1) {
  return val0.x == val1.x;
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 2>::type* = nullptr>
static inline bool isEqual(const T &val0, const T &val1) {
  return val0.x == val1.x &&
         val0.y == val1.y;
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 4>::type* = nullptr>
static inline bool isEqual(const T &val0, const T &val1) {
  return val0.x == val1.x &&
         val0.y == val1.y &&
         val0.z == val1.z &&
         val0.w == val1.w;
}

template<
  typename T,
  typename std::enable_if<std::is_scalar<T>::value>::type* = nullptr>
static inline bool isEqual(const T &val0, const T &val1) {
  return val0 == val1;
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 1>::type* = nullptr>
const std::string getString(const T& t)
{
  std::ostringstream os;
  os<< "(" << t.x << ")";
  return os.str();
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 2>::type* = nullptr>
const std::string getString(const T& t)
{
  std::ostringstream os;
  os<< "(" << t.x << ", " << t.y << ")";
  return os.str();
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 3>::type* = nullptr>
const std::string getString(const T& t)
{
  std::ostringstream os;
  os<< "(" << t.x << ", " << t.y << ", " << t.z << ")";
  return os.str();
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 4>::type* = nullptr>
const std::string getString(const T& t)
{
  std::ostringstream os;
  os<< "(" << t.x << ", " << t.y << ", " << t.z << ", " << t.w << ")";
  return os.str();
}

template<
  typename T,
  typename std::enable_if<std::is_scalar<T>::value>::type* = nullptr>
std::string getString(const T& t)
{
  std::ostringstream os;
  os << t;
  return os.str();
}

template<typename T>
static inline T getRandom() {
  double r = 0;
  if (std::is_signed<T>::value) {
    r = (std::rand() - RAND_MAX / 2.0) / (RAND_MAX / 2.0 + 1.);
  } else {
    r = std::rand() / (RAND_MAX + 1.);
  }
  return static_cast<T>(std::numeric_limits < T > ::max() * r);
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 1>::type* = nullptr>
static inline void initVal(T &val) {
  val.x = getRandom<decltype(T::x)>();
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 2>::type* = nullptr>
static inline void initVal(T &val) {
  val.x = getRandom<decltype(T::x)>();
  val.y = getRandom<decltype(T::x)>();
}

template<
  typename T,
  typename std::enable_if<rank<T>() == 4>::type* = nullptr>
static inline void initVal(T &val) {
  val.x = getRandom<decltype(T::x)>();
  val.y = getRandom<decltype(T::x)>();
  val.z = getRandom<decltype(T::x)>();
  val.w = getRandom<decltype(T::x)>();
}

template<
  typename T,
  typename std::enable_if<std::is_scalar<T>::value>::type* = nullptr>
static inline void initVal(T &val) {
  val = getRandom<T>();
}
