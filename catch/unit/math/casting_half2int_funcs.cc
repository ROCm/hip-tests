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

#include "half_precision_common.hh"
#include "casting_common.hh"

/**
 * @addtogroup HalfPrecisionComparison HalfPrecisionComparison
 * @{
 * @ingroup MathTest
 */

#define CAST_HALF2INT_RN_TEST_DEF(T, kern_name)                                                    \
  CAST_KERNEL_DEF(kern_name, T, Float16)                                                           \
  CAST_F2I_RZ_REF_DEF(kern_name, T, Float16)                                                       \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive") {                                      \
    T (*ref)(Float16) = kern_name##_ref;                                                           \
    CastUnaryHalfPrecisionTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<T>());           \
  }

CAST_HALF2INT_RN_TEST_DEF(int, half2int_rn)
CAST_HALF2INT_RN_TEST_DEF(int, half2int_rz)
CAST_HALF2INT_RN_TEST_DEF(int, half2int_rd)
CAST_HALF2INT_RN_TEST_DEF(int, half2int_ru)

CAST_HALF2INT_RN_TEST_DEF(unsigned int, half2uint_rn)
CAST_HALF2INT_RN_TEST_DEF(unsigned int, half2uint_rz)
CAST_HALF2INT_RN_TEST_DEF(unsigned int, half2uint_rd)
CAST_HALF2INT_RN_TEST_DEF(unsigned int, half2uint_ru)

CAST_HALF2INT_RN_TEST_DEF(short, half2short_rn)
CAST_HALF2INT_RN_TEST_DEF(short, half2short_rz)
CAST_HALF2INT_RN_TEST_DEF(short, half2short_rd)
CAST_HALF2INT_RN_TEST_DEF(short, half2short_ru)

CAST_HALF2INT_RN_TEST_DEF(unsigned short, half2ushort_rn)
CAST_HALF2INT_RN_TEST_DEF(unsigned short, half2ushort_rz)
CAST_HALF2INT_RN_TEST_DEF(unsigned short, half2ushort_rd)
CAST_HALF2INT_RN_TEST_DEF(unsigned short, half2ushort_ru)

CAST_HALF2INT_RN_TEST_DEF(long long, half2ll_rn)
CAST_HALF2INT_RN_TEST_DEF(long long, half2ll_rz)
CAST_HALF2INT_RN_TEST_DEF(long long, half2ll_rd)
CAST_HALF2INT_RN_TEST_DEF(long long, half2ll_ru)

CAST_HALF2INT_RN_TEST_DEF(unsigned long long, half2ull_rn)
CAST_HALF2INT_RN_TEST_DEF(unsigned long long, half2ull_rz)
CAST_HALF2INT_RN_TEST_DEF(unsigned long long, half2ull_rd)
CAST_HALF2INT_RN_TEST_DEF(unsigned long long, half2ull_ru)

CAST_KERNEL_DEF(half_as_short, short, Float16)

TEST_CASE("Unit_Device_half_as_short_Accuracy__Positive") {
  short (*ref)(Float16) = type2_as_type1_ref<short, Float16>;
  CastUnaryHalfPrecisionTest(half_as_short_kernel, ref, EqValidatorBuilderFactory<short>());
}

CAST_KERNEL_DEF(half_as_ushort, unsigned short, Float16)

TEST_CASE("Unit_Device_half_as_ushort_Accuracy__Positive") {
  unsigned short (*ref)(Float16) = type2_as_type1_ref<unsigned short, Float16>;
  CastUnaryHalfPrecisionTest(half_as_ushort_kernel, ref,
                             EqValidatorBuilderFactory<unsigned short>());
}