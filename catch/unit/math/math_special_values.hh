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

#pragma once

#include <array>
#include <limits>

/*-----------------------------------------------------------------------------
   HEX_FLT, HEXT_DBL, HEX_LDBL -- Create hex floating point literal of type
   float, double, long double respectively. Arguments:

      sm    -- sign of number,
      int   -- integer part of mantissa (without `0x' prefix),
      fract -- fractional part of mantissa (without decimal point and `L' or
            `LL' suffixes),
      se    -- sign of exponent,
      exp   -- absolute value of (binary) exponent.

   Example:

      double yhi = HEX_DBL(+, 1, 5555555555555, -, 2); // 0x1.5555555555555p-2

   Note:

      We have to pass signs as separate arguments because gcc pass negative
   integer values (e. g. `-2') into a macro as two separate tokens, so
   `HEX_FLT(1, 0, -2)' produces result `0x1.0p- 2' (note a space between minus
   and two) which is not a correct floating point literal.
-----------------------------------------------------------------------------*/
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
// If compiler does not support hex floating point literals:
#define HEX_FLT(sm, int, fract, se, exp)                                                           \
  sm ldexpf((float)(0x##int##fract##UL),                                                           \
            se exp + ilogbf((float)0x##int) - ilogbf((float)(0x##int##fract##UL)))
#define HEX_DBL(sm, int, fract, se, exp)                                                           \
  sm ldexp((double)(0x##int##fract##ULL),                                                          \
           se exp + ilogb((double)0x##int) - ilogb((double)(0x##int##fract##ULL)))
#define HEX_LDBL(sm, int, fract, se, exp)                                                          \
  sm ldexpl((long double)(0x##int##fract##ULL),                                                    \
            se exp + ilogbl((long double)0x##int) - ilogbl((long double)(0x##int##fract##ULL)))
#else
// If compiler supports hex floating point literals: just concatenate all the
// parts into a literal.
#define HEX_FLT(sm, int, fract, se, exp) sm 0x##int##.##fract##p##se##exp##F
#define HEX_DBL(sm, int, fract, se, exp) sm 0x##int##.##fract##p##se##exp
#define HEX_LDBL(sm, int, fract, se, exp) sm 0x##int##.##fract##p##se##exp##L
#endif

inline constexpr std::array kSpecialValuesDouble{
    -std::numeric_limits<double>::quiet_NaN(),
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::max(),
    HEX_DBL(-, 1, 0000000000001, +, 64),
    HEX_DBL(-, 1, 0, +, 64),
    HEX_DBL(-, 1, fffffffffffff, +, 63),
    HEX_DBL(-, 1, 0000000000001, +, 63),
    HEX_DBL(-, 1, 0, +, 63),
    HEX_DBL(-, 1, fffffffffffff, +, 62),
    HEX_DBL(-, 1, 000002, +, 32),
    HEX_DBL(-, 1, 0, +, 32),
    HEX_DBL(-, 1, fffffffffffff, +, 31),
    HEX_DBL(-, 1, 0000000000001, +, 31),
    HEX_DBL(-, 1, 0, +, 31),
    HEX_DBL(-, 1, fffffffffffff, +, 30),
    -1000.0,
    -100.0,
    -4.0,
    -3.5,
    -3.0,
    HEX_DBL(-, 1, 8000000000001, +, 1),
    -2.5,
    HEX_DBL(-, 1, 7ffffffffffff, +, 1),
    -2.0,
    HEX_DBL(-, 1, 8000000000001, +, 0),
    -1.5,
    HEX_DBL(-, 1, 7ffffffffffff, +, 0),
    HEX_DBL(-, 1, 0000000000001, +, 0),
    -1.0,
    HEX_DBL(-, 1, fffffffffffff, -, 1),
    HEX_DBL(-, 1, 0000000000001, -, 1),
    -0.5,
    HEX_DBL(-, 1, fffffffffffff, -, 2),
    HEX_DBL(-, 1, 0000000000001, -, 2),
    -0.25,
    HEX_DBL(-, 1, fffffffffffff, -, 3),
    HEX_DBL(-, 1, 0000000000001, -, 1022),
    -std::numeric_limits<double>::min(),
    HEX_DBL(-, 0, fffffffffffff, -, 1022),
    HEX_DBL(-, 0, 0000000000fff, -, 1022),
    HEX_DBL(-, 0, 00000000000fe, -, 1022),
    HEX_DBL(-, 0, 000000000000e, -, 1022),
    HEX_DBL(-, 0, 000000000000c, -, 1022),
    HEX_DBL(-, 0, 000000000000a, -, 1022),
    HEX_DBL(-, 0, 0000000000008, -, 1022),
    HEX_DBL(-, 0, 0000000000007, -, 1022),
    HEX_DBL(-, 0, 0000000000006, -, 1022),
    HEX_DBL(-, 0, 0000000000005, -, 1022),
    HEX_DBL(-, 0, 0000000000004, -, 1022),
    HEX_DBL(-, 0, 0000000000003, -, 1022),
    HEX_DBL(-, 0, 0000000000002, -, 1022),
    HEX_DBL(-, 0, 0000000000001, -, 1022),
    -0.0,

    std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::max(),
    HEX_DBL(+, 1, 0000000000001, +, 64),
    HEX_DBL(+, 1, 0, +, 64),
    HEX_DBL(+, 1, fffffffffffff, +, 63),
    HEX_DBL(+, 1, 0000000000001, +, 63),
    HEX_DBL(+, 1, 0, +, 63),
    HEX_DBL(+, 1, fffffffffffff, +, 62),
    HEX_DBL(+, 1, 000002, +, 32),
    HEX_DBL(+, 1, 0, +, 32),
    HEX_DBL(+, 1, fffffffffffff, +, 31),
    HEX_DBL(+, 1, 0000000000001, +, 31),
    HEX_DBL(+, 1, 0, +, 31),
    HEX_DBL(+, 1, fffffffffffff, +, 30),
    +1000.0,
    +100.0,
    +4.0,
    +3.5,
    +3.0,
    HEX_DBL(+, 1, 8000000000001, +, 1),
    +2.5,
    HEX_DBL(+, 1, 7ffffffffffff, +, 1),
    +2.0,
    HEX_DBL(+, 1, 8000000000001, +, 0),
    +1.5,
    HEX_DBL(+, 1, 7ffffffffffff, +, 0),
    HEX_DBL(+, 1, 0000000000001, +, 0),
    +1.0,
    HEX_DBL(+, 1, fffffffffffff, -, 1),
    HEX_DBL(+, 1, 0000000000001, -, 1),
    +0.5,
    HEX_DBL(+, 1, fffffffffffff, -, 2),
    HEX_DBL(+, 1, 0000000000001, -, 2),
    +0.25,
    HEX_DBL(+, 1, fffffffffffff, -, 3),
    HEX_DBL(+, 1, 0000000000001, -, 1022),
    +std::numeric_limits<double>::min(),
    HEX_DBL(+, 0, fffffffffffff, -, 1022),
    HEX_DBL(+, 0, 0000000000fff, -, 1022),
    HEX_DBL(+, 0, 00000000000fe, -, 1022),
    HEX_DBL(+, 0, 000000000000e, -, 1022),
    HEX_DBL(+, 0, 000000000000c, -, 1022),
    HEX_DBL(+, 0, 000000000000a, -, 1022),
    HEX_DBL(+, 0, 0000000000008, -, 1022),
    HEX_DBL(+, 0, 0000000000007, -, 1022),
    HEX_DBL(+, 0, 0000000000006, -, 1022),
    HEX_DBL(+, 0, 0000000000005, -, 1022),
    HEX_DBL(+, 0, 0000000000004, -, 1022),
    HEX_DBL(+, 0, 0000000000003, -, 1022),
    HEX_DBL(+, 0, 0000000000002, -, 1022),
    HEX_DBL(+, 0, 0000000000001, -, 1022),
    +0.0,
};

inline constexpr std::array kSpecialValuesFloat{
    -std::numeric_limits<float>::quiet_NaN(),
    -std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::max(),
    HEX_FLT(-, 1, 000002, +, 64),
    HEX_FLT(-, 1, 0, +, 64),
    HEX_FLT(-, 1, fffffe, +, 63),
    HEX_FLT(-, 1, 000002, +, 63),
    HEX_FLT(-, 1, 0, +, 63),
    HEX_FLT(-, 1, fffffe, +, 62),
    HEX_FLT(-, 1, 000002, +, 32),
    HEX_FLT(-, 1, 0, +, 32),
    HEX_FLT(-, 1, fffffe, +, 31),
    HEX_FLT(-, 1, 000002, +, 31),
    HEX_FLT(-, 1, 0, +, 31),
    HEX_FLT(-, 1, fffffe, +, 30),
    -1000.f,
    -100.f,
    -4.0f,
    -3.5f,
    -3.0f,
    HEX_FLT(-, 1, 800002, +, 1),
    -2.5f,
    HEX_FLT(-, 1, 7ffffe, +, 1),
    -2.0f,
    HEX_FLT(-, 1, 800002, +, 0),
    -1.5f,
    HEX_FLT(-, 1, 7ffffe, +, 0),
    HEX_FLT(-, 1, 000002, +, 0),
    -1.0f,
    HEX_FLT(-, 1, fffffe, -, 1),
    HEX_FLT(-, 1, 000002, -, 1),
    -0.5f,
    HEX_FLT(-, 1, fffffe, -, 2),
    HEX_FLT(-, 1, 000002, -, 2),
    -0.25f,
    HEX_FLT(-, 1, fffffe, -, 3),
    HEX_FLT(-, 1, 000002, -, 126),
    -std::numeric_limits<float>::min(),
    HEX_FLT(-, 0, fffffe, -, 126),
    HEX_FLT(-, 0, 000ffe, -, 126),
    HEX_FLT(-, 0, 0000fe, -, 126),
    HEX_FLT(-, 0, 00000e, -, 126),
    HEX_FLT(-, 0, 00000c, -, 126),
    HEX_FLT(-, 0, 00000a, -, 126),
    HEX_FLT(-, 0, 000008, -, 126),
    HEX_FLT(-, 0, 000006, -, 126),
    HEX_FLT(-, 0, 000004, -, 126),
    HEX_FLT(-, 0, 000002, -, 126),
    -0.0f,

    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::max(),
    HEX_FLT(+, 1, 000002, +, 64),
    HEX_FLT(+, 1, 0, +, 64),
    HEX_FLT(+, 1, fffffe, +, 63),
    HEX_FLT(+, 1, 000002, +, 63),
    HEX_FLT(+, 1, 0, +, 63),
    HEX_FLT(+, 1, fffffe, +, 62),
    HEX_FLT(+, 1, 000002, +, 32),
    HEX_FLT(+, 1, 0, +, 32),
    HEX_FLT(+, 1, fffffe, +, 31),
    HEX_FLT(+, 1, 000002, +, 31),
    HEX_FLT(+, 1, 0, +, 31),
    HEX_FLT(+, 1, fffffe, +, 30),
    +1000.f,
    +100.f,
    +4.0f,
    +3.5f,
    +3.0f,
    HEX_FLT(+, 1, 800002, +, 1),
    2.5f,
    HEX_FLT(+, 1, 7ffffe, +, 1),
    +2.0f,
    HEX_FLT(+, 1, 800002, +, 0),
    1.5f,
    HEX_FLT(+, 1, 7ffffe, +, 0),
    HEX_FLT(+, 1, 000002, +, 0),
    +1.0f,
    HEX_FLT(+, 1, fffffe, -, 1),
    HEX_FLT(+, 1, 000002, -, 1),
    +0.5f,
    HEX_FLT(+, 1, fffffe, -, 2),
    HEX_FLT(+, 1, 000002, -, 2),
    +0.25f,
    HEX_FLT(+, 1, fffffe, -, 3),
    HEX_FLT(+, 1, 000002, -, 126),
    +std::numeric_limits<float>::min(),
    HEX_FLT(+, 0, fffffe, -, 126),
    HEX_FLT(+, 0, 000ffe, -, 126),
    HEX_FLT(+, 0, 0000fe, -, 126),
    HEX_FLT(+, 0, 00000e, -, 126),
    HEX_FLT(+, 0, 00000c, -, 126),
    HEX_FLT(+, 0, 00000a, -, 126),
    HEX_FLT(+, 0, 000008, -, 126),
    HEX_FLT(+, 0, 000006, -, 126),
    HEX_FLT(+, 0, 000004, -, 126),
    HEX_FLT(+, 0, 000002, -, 126),
    +0.0f,
};

template <typename T> struct SpecialVals {
  const T* const data;
  const size_t size;
};

inline constexpr auto kSpecialValRegistry =
    std::make_tuple(SpecialVals<float>{kSpecialValuesFloat.data(), kSpecialValuesFloat.size()},
                    SpecialVals<double>{kSpecialValuesDouble.data(), kSpecialValuesDouble.size()});
