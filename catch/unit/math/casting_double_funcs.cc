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
#include "casting_double_negative_kernels_rtc.hh"

#define CAST_DOUBLE2INT_TEST_DEF(T, kern_name, round_dir)                                          \
  CAST_KERNEL_DEF(kern_name, T, double)                                                            \
  CAST_RND_RINT_REF_DEF(kern_name, T, double, round_dir)                                           \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(double) = kern_name##_ref;                                                            \
    CastDoublePrecisionTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>());              \
  }

#define CAST_DOUBLE2INT_RZ_TEST_DEF(T, kern_name)                                                  \
  CAST_KERNEL_DEF(kern_name, T, double)                                                            \
  CAST_RND_RZ_REF_DEF(kern_name, T, double)                                                        \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(double) = kern_name##_ref;                                                            \
    CastDoublePrecisionTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>());              \
  }

CAST_DOUBLE2INT_TEST_DEF(int, double2int_rd, FE_DOWNWARD)
CAST_DOUBLE2INT_TEST_DEF(int, double2int_rn, FE_TONEAREST)
CAST_DOUBLE2INT_TEST_DEF(int, double2int_ru, FE_UPWARD)
CAST_DOUBLE2INT_RZ_TEST_DEF(int, double2int_rz)

TEST_CASE("Unit_Device_double2int_Negative_RTC") { NegativeTestRTCWrapper<12>(kDouble2Int); }

CAST_DOUBLE2INT_TEST_DEF(unsigned int, double2uint_rd, FE_DOWNWARD)
CAST_DOUBLE2INT_TEST_DEF(unsigned int, double2uint_rn, FE_TONEAREST)
CAST_DOUBLE2INT_TEST_DEF(unsigned int, double2uint_ru, FE_UPWARD)
CAST_DOUBLE2INT_RZ_TEST_DEF(unsigned int, double2uint_rz)

TEST_CASE("Unit_Device_double2uint_Negative_RTC") { NegativeTestRTCWrapper<12>(kDouble2Uint); }

#define CAST_DOUBLE2LL_TEST_DEF(T, kern_name, round_dir)                                           \
  CAST_KERNEL_DEF(kern_name, T, double)                                                            \
  CAST_RND_RINT_REF_DEF(kern_name, T, double, round_dir)                                           \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(double) = kern_name##_ref;                                                            \
    UnaryDoublePrecisionBruteForceTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),    \
                                       static_cast<double>(std::numeric_limits<T>::min()),         \
                                       static_cast<double>(std::numeric_limits<T>::max()));        \
  }

#define CAST_DOUBLE2LL_RZ_TEST_DEF(T, kern_name)                                                   \
  CAST_KERNEL_DEF(kern_name, T, double)                                                            \
  CAST_RND_RZ_REF_DEF(kern_name, T, double)                                                        \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    T (*ref)(double) = kern_name##_ref;                                                            \
    UnaryDoublePrecisionBruteForceTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>(),    \
                                       static_cast<double>(std::numeric_limits<T>::min()),         \
                                       static_cast<double>(std::numeric_limits<T>::max()));        \
  }

CAST_DOUBLE2LL_TEST_DEF(long long int, double2ll_rd, FE_DOWNWARD)
CAST_DOUBLE2LL_TEST_DEF(long long int, double2ll_rn, FE_TONEAREST)
CAST_DOUBLE2LL_TEST_DEF(long long int, double2ll_ru, FE_UPWARD)
CAST_DOUBLE2LL_RZ_TEST_DEF(long long int, double2ll_rz)

TEST_CASE("Unit_Device_double2ll_Negative_RTC") { NegativeTestRTCWrapper<12>(kDouble2LL); }

CAST_DOUBLE2LL_TEST_DEF(unsigned long long int, double2ull_rd, FE_DOWNWARD)
CAST_DOUBLE2LL_TEST_DEF(unsigned long long int, double2ull_rn, FE_TONEAREST)
CAST_DOUBLE2LL_TEST_DEF(unsigned long long int, double2ull_ru, FE_UPWARD)
CAST_DOUBLE2LL_RZ_TEST_DEF(unsigned long long int, double2ull_rz)

TEST_CASE("Unit_Device_double2ull_Negative_RTC") { NegativeTestRTCWrapper<12>(kDouble2ULL); }

#define CAST_DOUBLE2FLOAT_TEST_DEF(kern_name, round_dir)                                           \
  CAST_KERNEL_DEF(kern_name, float, double)                                                        \
  CAST_RND_REF_DEF(kern_name, float, double, round_dir)                                            \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    float (*ref)(double) = kern_name##_ref;                                                        \
    CastDoublePrecisionTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<float>());          \
  }

#define CAST_DOUBLE2FLOAT_RN_TEST_DEF(kern_name)                                                   \
  CAST_KERNEL_DEF(kern_name, float, double)                                                        \
  CAST_REF_DEF(kern_name, float, double)                                                           \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Positive") {                                               \
    float (*ref)(double) = kern_name##_ref;                                                        \
    CastDoublePrecisionTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<float>());          \
  }

CAST_DOUBLE2FLOAT_TEST_DEF(double2float_rd, FE_DOWNWARD)
CAST_DOUBLE2FLOAT_RN_TEST_DEF(double2float_rn)
CAST_DOUBLE2FLOAT_TEST_DEF(double2float_ru, FE_UPWARD)
CAST_DOUBLE2FLOAT_TEST_DEF(double2float_rz, FE_TOWARDZERO)

TEST_CASE("Unit_Device_double2float_Negative_RTC") { NegativeTestRTCWrapper<12>(kDouble2Float); }

CAST_KERNEL_DEF(double2hiint, int, double)

int double2hiint_ref(double arg) {
  int tmp[2];
  memcpy(tmp, &arg, sizeof(tmp));
  return tmp[1];
}

TEST_CASE("Unit_Device_double2hiint_Positive") {
  int (*ref)(double) = double2hiint_ref;
  CastDoublePrecisionTest(double2hiint_kernel, ref, EqValidatorBuilderFactory<int>());
}

TEST_CASE("Unit_Device_double2hiint_Negative_RTC") { NegativeTestRTCWrapper<3>(kDouble2Hiint); }

CAST_KERNEL_DEF(double2loint, int, double)

int double2loint_ref(double arg) {
  int tmp[2];
  memcpy(tmp, &arg, sizeof(tmp));
  return tmp[0];
}

TEST_CASE("Unit_Device_double2loint_Positive") {
  int (*ref)(double) = double2loint_ref;
  CastDoublePrecisionTest(double2loint_kernel, ref, EqValidatorBuilderFactory<int>());
}

TEST_CASE("Unit_Device_double2loint_Negative_RTC") { NegativeTestRTCWrapper<3>(kDouble2Loint); }

CAST_KERNEL_DEF(double_as_longlong, long long int, double)

TEST_CASE("Unit_Device_double_as_longlong_Positive") {
  long long int (*ref)(double) = type2_as_type1_ref<long long int, double>;
  CastDoublePrecisionTest(double_as_longlong_kernel, ref,
                          EqValidatorBuilderFactory<long long int>());
}

TEST_CASE("Unit_Device_double_as_longlong_Negative_RTC") {
  NegativeTestRTCWrapper<3>(kDoubleAsLonglong);
}