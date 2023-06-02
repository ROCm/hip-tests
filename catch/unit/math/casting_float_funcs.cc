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

#include "casting_common.hh"
#include "casting_float_negative_kernels_rtc.hh"

#define CAST_FLOAT2INT_TEST_DEF(T, kern_name, round_dir)                                           \
  CAST_KERNEL_DEF(kern_name, T, float)                                                             \
  CAST_RND_RINT_REF_DEF(kern_name, T, float, round_dir)                                            \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),         \
                                  std::numeric_limits<float>::lowest(),                            \
                                  std::numeric_limits<float>::max());                              \
  }

#define CAST_FLOAT2INT_RZ_TEST_DEF(T, kern_name)                                                   \
  CAST_KERNEL_DEF(kern_name, T, float)                                                             \
  CAST_RND_RZ_REF_DEF(kern_name, T, float)                                                         \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),         \
                                  std::numeric_limits<float>::lowest(),                            \
                                  std::numeric_limits<float>::max());                              \
  }

CAST_FLOAT2INT_TEST_DEF(int, float2int_rd, FE_DOWNWARD)
CAST_FLOAT2INT_TEST_DEF(int, float2int_rn, FE_TONEAREST)
CAST_FLOAT2INT_TEST_DEF(int, float2int_ru, FE_UPWARD)
CAST_FLOAT2INT_TEST_DEF(int, float2int_rz, FE_TOWARDZERO)

TEST_CASE("Unit_Device_float2int_Negative_RTC") { NegativeTestRTCWrapper<12>(kFloat2Int); }

CAST_FLOAT2INT_TEST_DEF(unsigned int, float2uint_rd, FE_DOWNWARD)
CAST_FLOAT2INT_TEST_DEF(unsigned int, float2uint_rn, FE_TONEAREST)
CAST_FLOAT2INT_TEST_DEF(unsigned int, float2uint_ru, FE_UPWARD)
CAST_FLOAT2INT_RZ_TEST_DEF(unsigned int, float2uint_rz)

TEST_CASE("Unit_Device_float2uint_Negative_RTC") { NegativeTestRTCWrapper<12>(kFloat2Uint); }

#define CAST_FLOAT2LL_TEST_DEF(T, kern_name, round_dir)                                            \
  CAST_KERNEL_DEF(kern_name, T, float)                                                             \
  CAST_RND_RINT_REF_DEF(kern_name, T, float, round_dir)                                            \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),         \
                                  static_cast<float>(std::numeric_limits<T>::min()),               \
                                  static_cast<float>(std::numeric_limits<T>::max()));              \
  }

#define CAST_FLOAT2LL_RZ_TEST_DEF(T, kern_name)                                                    \
  CAST_KERNEL_DEF(kern_name, T, float)                                                             \
  CAST_RND_RZ_REF_DEF(kern_name, T, float)                                                         \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(float) = kern_name##_ref;                                                             \
    UnarySinglePrecisionRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),         \
                                  static_cast<float>(std::numeric_limits<T>::min()),               \
                                  static_cast<float>(std::numeric_limits<T>::max()));              \
  }

CAST_FLOAT2LL_TEST_DEF(long long int, float2ll_rd, FE_DOWNWARD)
CAST_FLOAT2LL_TEST_DEF(long long int, float2ll_rn, FE_TONEAREST)
CAST_FLOAT2LL_TEST_DEF(long long int, float2ll_ru, FE_UPWARD)
CAST_FLOAT2LL_RZ_TEST_DEF(long long int, float2ll_rz)

TEST_CASE("Unit_Device_float2ll_Negative_RTC") { NegativeTestRTCWrapper<12>(kFloat2LL); }

CAST_FLOAT2LL_TEST_DEF(unsigned long long int, float2ull_rd, FE_DOWNWARD)
CAST_FLOAT2LL_TEST_DEF(unsigned long long int, float2ull_rn, FE_TONEAREST)
CAST_FLOAT2LL_TEST_DEF(unsigned long long int, float2ull_ru, FE_UPWARD)
CAST_FLOAT2LL_RZ_TEST_DEF(unsigned long long int, float2ull_rz)

TEST_CASE("Unit_Device_float2ull_Negative_RTC") { NegativeTestRTCWrapper<12>(kFloat2ULL); }

CAST_KERNEL_DEF(float_as_int, int, float)

TEST_CASE("Unit_Device_float_as_int_Positive") {
  int (*ref)(float) = type2_as_type1_ref<int, float>;
  UnarySinglePrecisionTest(float_as_int_kernel, ref, EqValidatorBuilderFactory<int>());
}

TEST_CASE("Unit_Device_float_as_int_Negative_RTC") { NegativeTestRTCWrapper<3>(kFloatAsInt); }

CAST_KERNEL_DEF(float_as_uint, unsigned int, float)

TEST_CASE("Unit_Device_float_as_uint_Positive") {
  unsigned int (*ref)(float) = type2_as_type1_ref<unsigned int, float>;
  UnarySinglePrecisionTest(float_as_uint_kernel, ref, EqValidatorBuilderFactory<unsigned int>());
}

TEST_CASE("Unit_Device_float_as_uint_Negative_RTC") { NegativeTestRTCWrapper<3>(kFloatAsUint); }
