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

#define CAST_INT2HALF_RN_TEST_DEF(T, kern_name)                                                    \
  CAST_KERNEL_DEF(kern_name, Float16, T)                                                           \
  CAST_REF_DEF(kern_name, Float16, T)                                                              \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive") {                                      \
    Float16 (*ref)(T) = kern_name##_ref;                                                           \
    CastIntRangeTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<Float16>());               \
  }

CAST_INT2HALF_RN_TEST_DEF(int, int2half_rn)
CAST_INT2HALF_RN_TEST_DEF(int, int2half_rz)
CAST_INT2HALF_RN_TEST_DEF(int, int2half_rd)
CAST_INT2HALF_RN_TEST_DEF(int, int2half_ru)

CAST_INT2HALF_RN_TEST_DEF(unsigned int, uint2half_rn)
CAST_INT2HALF_RN_TEST_DEF(unsigned int, uint2half_rz)
CAST_INT2HALF_RN_TEST_DEF(unsigned int, uint2half_rd)
CAST_INT2HALF_RN_TEST_DEF(unsigned int, uint2half_ru)

CAST_INT2HALF_RN_TEST_DEF(short, short2half_rn)
CAST_INT2HALF_RN_TEST_DEF(short, short2half_rz)
CAST_INT2HALF_RN_TEST_DEF(short, short2half_rd)
CAST_INT2HALF_RN_TEST_DEF(short, short2half_ru)

CAST_INT2HALF_RN_TEST_DEF(unsigned short, ushort2half_rn)
CAST_INT2HALF_RN_TEST_DEF(unsigned short, ushort2half_rz)
CAST_INT2HALF_RN_TEST_DEF(unsigned short, ushort2half_rd)
CAST_INT2HALF_RN_TEST_DEF(unsigned short, ushort2half_ru)

#define CAST_LL2HALF_TEST_DEF(T, kern_name)                                                        \
  CAST_KERNEL_DEF(kern_name, Float16, T)                                                           \
  CAST_REF_DEF(kern_name, Float16, T)                                                              \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive") {                                      \
    Float16 (*ref)(T) = kern_name##_ref;                                                           \
    CastIntBruteForceTest(kern_name##_kernel, ref, EqValidatorBuilderFactory<Float16>());          \
  }

CAST_LL2HALF_TEST_DEF(long long, ll2half_rn)
CAST_LL2HALF_TEST_DEF(long long, ll2half_rz)
CAST_LL2HALF_TEST_DEF(long long, ll2half_rd)
CAST_LL2HALF_TEST_DEF(long long, ll2half_ru)

CAST_LL2HALF_TEST_DEF(unsigned long long, ull2half_rn)
CAST_LL2HALF_TEST_DEF(unsigned long long, ull2half_rz)
CAST_LL2HALF_TEST_DEF(unsigned long long, ull2half_rd)
CAST_LL2HALF_TEST_DEF(unsigned long long, ull2half_ru)

CAST_KERNEL_DEF(short_as_half, Float16, short)

TEST_CASE("Unit_Device_short_as_half_Accuracy_Positive") {
  Float16 (*ref)(short) = type2_as_type1_ref<Float16, short>;
  CastIntBruteForceTest(short_as_half_kernel, ref, EqValidatorBuilderFactory<Float16>());
}

CAST_KERNEL_DEF(ushort_as_half, Float16, unsigned short)

TEST_CASE("Unit_Device_ushort_as_half_Accuracy_Positive") {
  Float16 (*ref)(unsigned short) = type2_as_type1_ref<Float16, unsigned short>;
  CastIntBruteForceTest(ushort_as_half_kernel, ref, EqValidatorBuilderFactory<Float16>());
}