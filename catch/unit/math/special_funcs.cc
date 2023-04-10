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

#include "unary_common.hh"
#include "binary_common.hh"
#include "special_common.hh"
#include "math_special_func_kernels_rtc.hh"

#include <boost/math/special_functions.hpp>

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(erf, 2, 2)
TEST_CASE("Unit_Device_erf_erff_Negative_RTC") { NegativeTestRTCWrapper<4>(kErf); }

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(erfc, 4, 5)
TEST_CASE("Unit_Device_erfc_erfcf_Negative_RTC") { NegativeTestRTCWrapper<4>(kErfc); }

MATH_UNARY_KERNEL_DEF(tgamma)

TEST_CASE("Unit_Device_tgammaf_Accuracy_Limited_Positive") {
  double (*ref)(double) = std::tgamma;
  UnarySinglePrecisionRangeTest(tgamma_kernel<float>, ref,
                                        ULPValidatorBuilderFactory<float>(5), 0, 171);
}

TEST_CASE("Unit_Device_tgamma_Accuracy_Limited_Positive") {
  long double (*ref)(long double) = std::tgamma;
  UnaryDoublePrecisionTest(tgamma_kernel<double>, ref,
                                   ULPValidatorBuilderFactory<double>(10));
}

TEST_CASE("Unit_Device_tgamma_tgammaf_Negative_RTC") { NegativeTestRTCWrapper<4>(kTgamma); }

MATH_UNARY_KERNEL_DEF(lgamma)

TEST_CASE("Unit_Device_lgammaf_Accuracy_Limited_Positive") {
  double (*ref)(double) = std::lgamma;
  UnarySinglePrecisionRangeTest(lgamma_kernel<float>, ref,
                                        ULPValidatorBuilderFactory<float>(6), -2, 100);
}

TEST_CASE("Unit_Device_lgamma_Accuracy_Limited_Positive") {
  long double (*ref)(long double) = std::lgamma;
  UnaryDoublePrecisionBruteForceTest(lgamma_kernel<double>, ref,
                                             ULPValidatorBuilderFactory<double>(4), -2, 100);
}

TEST_CASE("Unit_Device_lgamma_lgammaf_Negative_RTC") { NegativeTestRTCWrapper<4>(kLgamma); }

MATH_UNARY_KERNEL_DEF(erfcinv)

TEST_CASE("Unit_Device_erfcinvf_Accuracy_Positive") {
  auto erfcinv_ref = [](double arg) -> double {
    if (arg == 0)
      return std::numeric_limits<double>::infinity();
    else if (arg == 2)
      return -std::numeric_limits<double>::infinity();
    else if (arg < 0 || arg > 2)
      return std::numeric_limits<double>::quiet_NaN();
    return boost::math::erfc_inv(arg);
  };
  double (*ref)(double) = erfcinv_ref;
  UnarySinglePrecisionTest(erfcinv_kernel<float>, ref, ULPValidatorBuilderFactory<float>(4));
}

TEST_CASE("Unit_Device_erfcinv_Accuracy_Positive") {
  auto erfcinv_ref = [](long double arg) -> long double {
    if (arg == 0)
      return std::numeric_limits<long double>::infinity();
    else if (arg == 2)
      return -std::numeric_limits<long double>::infinity();
    else if (arg < 0 || arg > 2)
      return std::numeric_limits<long double>::quiet_NaN();
    return boost::math::erfc_inv(arg);
  };
  long double (*ref)(long double) = erfcinv_ref;
  UnaryDoublePrecisionTest(erfcinv_kernel<double>, ref, ULPValidatorBuilderFactory<double>(6));
}

TEST_CASE("Unit_Device_erfcinv_erfcinvf_Negative_RTC") { NegativeTestRTCWrapper<4>(kErfcinv); }

MATH_UNARY_KERNEL_DEF(erfinv)

TEST_CASE("Unit_Device_erfinvf_Accuracy_Positive") {
  auto erfinv_ref = [](double arg) -> double {
    if (arg == 0) return 0;
    if (arg == 1)
      return std::numeric_limits<double>::infinity();
    else if (arg == -1)
      return -std::numeric_limits<double>::infinity();
    else if (arg < -1 || arg > 1)
      return std::numeric_limits<double>::quiet_NaN();
    return boost::math::erf_inv(arg);
  };
  double (*ref)(double) = erfinv_ref;
  UnarySinglePrecisionTest(erfinv_kernel<float>, ref, ULPValidatorBuilderFactory<float>(2));
}

TEST_CASE("Unit_Device_erfinv_Accuracy_Positive") {
  auto erfinv_ref = [](long double arg) -> long double {
    if (arg == 0) return 0;
    if (arg == 1)
      return std::numeric_limits<long double>::infinity();
    else if (arg == -1)
      return -std::numeric_limits<long double>::infinity();
    else if (arg < -1 || arg > 1)
      return std::numeric_limits<long double>::quiet_NaN();
    return boost::math::erf_inv(arg);
  };
  long double (*ref)(long double) = erfinv_ref;
  UnaryDoublePrecisionTest(erfinv_kernel<double>, ref, ULPValidatorBuilderFactory<double>(5));
}

TEST_CASE("Unit_Device_erfinv_erfinvf_Negative_RTC") { NegativeTestRTCWrapper<4>(kErfinv); }

MATH_UNARY_KERNEL_DEF(normcdf)

TEST_CASE("Unit_Device_normcdff_Accuracy_Positive") {
  auto normcdf_ref = [](double arg) -> double { return std::erfc(-arg / std::sqrt(2)) / 2; };
  double (*ref)(double) = normcdf_ref;
  UnarySinglePrecisionTest(normcdf_kernel<float>, ref, ULPValidatorBuilderFactory<float>(5));
}

TEST_CASE("Unit_Device_normcdf_Accuracy_Positive") {
  auto normcdf_ref = [](long double arg) -> long double {
    return std::erfc(-arg / std::sqrt(2.L)) / 2;
  };
  long double (*ref)(long double) = normcdf_ref;
  UnaryDoublePrecisionTest(normcdf_kernel<double>, ref, ULPValidatorBuilderFactory<double>(5));
}

TEST_CASE("Unit_Device_normcdf_normcdff_Negative_RTC") { NegativeTestRTCWrapper<4>(kNormcdf); }

MATH_UNARY_KERNEL_DEF(normcdfinv)

TEST_CASE("Unit_Device_normcdfinvf_Sanity_Positive") {
  constexpr std::array<float, 9> input{0.f, 0.1f, 0.25f, 0.4f, 0.5f, 0.6f, 0.75f, 0.9f, 1.f};
  constexpr std::array<float, 9> reference{-std::numeric_limits<float>::infinity(),
                                           -1.28155160f,
                                           -0.674489737f,
                                           -0.253347069f,
                                           0,
                                           0.253347158f,
                                           0.674489737f,
                                           1.28155148f,
                                           std::numeric_limits<float>::infinity()};
  SpecialSimpleTest<float>(normcdfinv_kernel<float>, ULPValidatorBuilderFactory<float>(5),
                           input.data(), reference.data(), input.size());
}

TEST_CASE("Unit_Device_normcdfinv_Sanity_Positive") {
  constexpr std::array<double, 9> input{0., 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.};
  constexpr std::array<double, 9> reference{-std::numeric_limits<float>::infinity(),
                                            -1.2815515655446004,
                                            -0.67448975019608159,
                                            -0.25334710313579972,
                                            0,
                                            0.25334710313579972,
                                            0.67448975019608159,
                                            1.2815515655446006,
                                            std::numeric_limits<float>::infinity()};
  SpecialSimpleTest<double>(normcdfinv_kernel<double>, ULPValidatorBuilderFactory<double>(5),
                            input.data(), reference.data(), input.size());
}

TEST_CASE("Unit_Device_normcdfinv_normcdfinvf_Negative_RTC") {
  NegativeTestRTCWrapper<4>(kNormcdfinv);
}

MATH_UNARY_KERNEL_DEF(erfcx)

TEST_CASE("Unit_Device_erfcxf_Sanity_Positive") {
  constexpr std::array<float, 11> input{-std::numeric_limits<float>::infinity(),
                                        -1000.f,
                                        -100.f,
                                        -5.f,
                                        -0.5f,
                                        0.,
                                        0.75f,
                                        15.f,
                                        200.f,
                                        500.f,
                                        std::numeric_limits<float>::infinity()};
  constexpr std::array<float, 11> reference{std::numeric_limits<float>::infinity(),
                                            std::numeric_limits<float>::infinity(),
                                            std::numeric_limits<float>::infinity(),
                                            1.44009806e11f,
                                            1.95236027f,
                                            1.0f,
                                            5.06937683e-1f,
                                            3.75296101e-2f,
                                            2.82091252e-3f,
                                            1.12837693e-3f,
                                            0.f};
  SpecialSimpleTest<float>(erfcx_kernel<float>, ULPValidatorBuilderFactory<float>(5), input.data(),
                           reference.data(), input.size());
}

TEST_CASE("Unit_Device_erfcx_Sanity_Positive") {
  constexpr std::array<double, 11> input{
      -std::numeric_limits<double>::infinity(), -1000., -100., -5., -0.5, 0., 0.75, 15., 200., 500.,
      std::numeric_limits<double>::infinity()};
  constexpr std::array<double, 11> reference{std::numeric_limits<double>::infinity(),
                                             std::numeric_limits<double>::infinity(),
                                             std::numeric_limits<double>::infinity(),
                                             1.4400979867466104e11,
                                             1.9523604891825568,
                                             1.0,
                                             5.0693765029314475e-1,
                                             3.7529606388505762e-2,
                                             2.8209126572120466e-3,
                                             1.1283769103507188e-3,
                                             0.};
  SpecialSimpleTest<double>(erfcx_kernel<double>, ULPValidatorBuilderFactory<double>(5),
                            input.data(), reference.data(), input.size());
}

TEST_CASE("Unit_Device_erfcx_erfcxf_Negative_RTC") { NegativeTestRTCWrapper<4>(kErfcx); }

MATH_UNARY_KERNEL_DEF(cyl_bessel_i0)

TEST_CASE("Unit_Device_cyl_bessel_i0f_Accuracy_Limited_Positive") {
  auto cyl_bessel_i0_ref = [](double arg) -> double { return std::cyl_bessel_i(0, arg); };
  double (*ref)(double) = cyl_bessel_i0_ref;
  UnarySinglePrecisionRangeTest(cyl_bessel_i0_kernel<float>, ref,
                                ULPValidatorBuilderFactory<float>(6), 0, 1000);
}

TEST_CASE("Unit_Device_cyl_bessel_i0_Accuracy_Limited_Positive") {
  auto cyl_bessel_i0_ref = [](long double arg) -> long double { return std::cyl_bessel_i(0, arg); };
  long double (*ref)(long double) = cyl_bessel_i0_ref;
  UnaryDoublePrecisionBruteForceTest(cyl_bessel_i0_kernel<double>, ref,
                                             ULPValidatorBuilderFactory<double>(6), 0, 1000);
}


TEST_CASE("Unit_Device_cyl_bessel_i0_cyl_bessel_i0f_Negative_RTC") {
  NegativeTestRTCWrapper<4>(kCylBesselI0);
}

MATH_UNARY_KERNEL_DEF(cyl_bessel_i1)

TEST_CASE("Unit_Device_cyl_bessel_i1f_Accuracy_Limited_Positive") {
  auto cyl_bessel_i1_ref = [](double arg) -> double { return std::cyl_bessel_i(1, arg); };
  double (*ref)(double) = cyl_bessel_i1_ref;
  UnarySinglePrecisionRangeTest(cyl_bessel_i1_kernel<float>, ref,
                                ULPValidatorBuilderFactory<float>(6), 0, 1000);
}

TEST_CASE("Unit_Device_cyl_bessel_i1_Accuracy_Limited_Positive") {
  auto cyl_bessel_i1_ref = [](long double arg) -> long double { return std::cyl_bessel_i(1, arg); };
  long double (*ref)(long double) = cyl_bessel_i1_ref;
  UnaryDoublePrecisionBruteForceTest(cyl_bessel_i1_kernel<double>, ref,
                                     ULPValidatorBuilderFactory<double>(6), 0, 1000);
}

TEST_CASE("Unit_Device_cyl_bessel_i1_cyl_bessel_i1f_Negative_RTC") {
  NegativeTestRTCWrapper<4>(kCylBesselI1);
}

MATH_UNARY_KERNEL_DEF(y0)

TEST_CASE("Unit_Device_y0f_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(double) = y0;
#elif _WIN64
  double (*ref)(double) = _y0;
#endif
  UnarySinglePrecisionRangeTest(y0_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f,
                                8.f);
  UnarySinglePrecisionRangeTest(y0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022),
                                8.f, std::numeric_limits<float>::max());
}

TEST_CASE("Unit_Device_y0_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(long double) = y0l;
#elif _WIN64
  long double (*ref)(long double) = _y0l;
#endif
  UnaryDoublePrecisionBruteForceTest(y0_kernel<double>, ref,
                                     AbsValidatorBuilderFactory<float>(5.e-12), -8.,
                                     std::numeric_limits<double>::max());
}


TEST_CASE("Unit_Device_y0_y0f_Negative_RTC") { NegativeTestRTCWrapper<4>(kY0); }

MATH_UNARY_KERNEL_DEF(y1)

TEST_CASE("Unit_Device_y1f_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(double) = y1;
#elif _WIN64
  double (*ref)(double) = _y1;
#endif
  UnarySinglePrecisionRangeTest(y1_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f,
                                8.f);
  UnarySinglePrecisionRangeTest(y1_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022),
                                8.f, std::numeric_limits<float>::max());
}

TEST_CASE("Unit_Device_y1_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(long double) = y1l;
#elif _WIN64
  long double (*ref)(long double) = _y1l;
#endif
  UnaryDoublePrecisionBruteForceTest(y1_kernel<double>, ref,
                                     AbsValidatorBuilderFactory<float>(5.e-12), -8.,
                                     std::numeric_limits<double>::max());
}

TEST_CASE("Unit_Device_y1_y1f_Negative_RTC") { NegativeTestRTCWrapper<4>(kY1); }

MATH_BESSEL_N_KERNEL_DEF(yn)

TEST_CASE("Unit_Device_ynf_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(int, double) = yn;
#elif _WIN64
  double (*ref)(int, double) = _yn;
#endif
  int n = GENERATE(5, 25, 120);
  BesselSinglePrecisionRangeTest(yn_kernel, ref, AbsValidatorBuilderFactory<float>(0.0000022), n, n,
                                 std::numeric_limits<float>::max());
}

TEST_CASE("Unit_Device_yn_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(int, long double) = ynl;
#elif _WIN64
  long double (*ref)(int, long double) = _ynl;
#endif
  int n = GENERATE(5, 25, 120);
  BesselFloatingPointBruteForceTest(yn_kernel<double>, ref,
                                    AbsValidatorBuilderFactory<double>(5.e-12), n, 1.5 * n,
                                    std::numeric_limits<double>::max());
}

TEST_CASE("Unit_Device_yn_ynf_Negative_RTC") { NegativeTestRTCWrapper<8>(kYn); }

MATH_UNARY_KERNEL_DEF(j0)

TEST_CASE("Unit_Device_j0f_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(double) = j0;
#elif _WIN64
  double (*ref)(double) = _j0;
#endif
  UnarySinglePrecisionRangeTest(j0_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f,
                                8.f);
  UnarySinglePrecisionRangeTest(j0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022),
                                8.f, std::numeric_limits<float>::max());
}

TEST_CASE("Unit_Device_j0_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(long double) = j0l;
#elif _WIN64
  long double (*ref)(long double) = _j0l;
#endif
  UnaryDoublePrecisionBruteForceTest(
      j0_kernel<double>, ref, AbsValidatorBuilderFactory<float>(5.e-12),
      std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max());
}

TEST_CASE("Unit_Device_j0_j0f_Negative_RTC") { NegativeTestRTCWrapper<4>(kJ0); }

MATH_UNARY_KERNEL_DEF(j1)

TEST_CASE("Unit_Device_j1f_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(double) = j1;
#elif _WIN64
  double (*ref)(double) = _j1;
#endif
  UnarySinglePrecisionRangeTest(j1_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f,
                                8.f);
  UnarySinglePrecisionRangeTest(j1_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022),
                                8.f, std::numeric_limits<float>::max());
}

TEST_CASE("Unit_Device_j1_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(long double) = j1l;
#elif _WIN64
  long double (*ref)(long double) = _j1l;
#endif
  UnaryDoublePrecisionBruteForceTest(
      j1_kernel<double>, ref, AbsValidatorBuilderFactory<float>(5.e-12),
      std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max());
}
TEST_CASE("Unit_Device_j1_j1f_Negative_RTC") { NegativeTestRTCWrapper<4>(kJ1); }

MATH_BESSEL_N_KERNEL_DEF(jn)

TEST_CASE("Unit_Device_jnf_Accuracy_Limited_Positive") {
#ifdef __unix__
  double (*ref)(int, double) = jn;
#elif _WIN64
  double (*ref)(int, double) = _jn;
#endif
  int n = GENERATE(5, 25, 120);
  BesselSinglePrecisionRangeTest(jn_kernel<float>, ref,
                                 AbsValidatorBuilderFactory<float>(0.0000022), n, n,
                                 std::numeric_limits<float>::max());
}

TEST_CASE("Unit_Device_jn_Accuracy_Limited_Positive") {
#ifdef __unix__
  long double (*ref)(int, long double) = jnl;
#elif _WIN64
  long double (*ref)(int, long double) = _jnl;
#endif
  int n = GENERATE(5, 25, 120);
  BesselFloatingPointBruteForceTest(jn_kernel<double>, ref,
                                    AbsValidatorBuilderFactory<double>(5.e-12), n, 1.5 * n,
                                    std::numeric_limits<double>::max());
}

TEST_CASE("Unit_Device_jn_jnf_Negative_RTC") { NegativeTestRTCWrapper<8>(kJn); }
