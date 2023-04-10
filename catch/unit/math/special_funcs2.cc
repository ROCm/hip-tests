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

#include <boost/math/special_functions.hpp>

#define MATH_BESSEL_N_KERNEL_DEF(func_name)                                                        \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, int* n, T* const xs) {      \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[i] = func_name##f(n[i], xs[i]);                                                         \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[i] = func_name(n[i], xs[i]);                                                            \
      }                                                                                            \
    }                                                                                              \
  }


template <typename RT = RefType_t<float>, typename F, typename RF, typename ValidatorBuilder>
void UnarySinglePrecisionSpecialValuesTest(F kernel, RF ref_func,
                                           const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values = std::get<SpecialVals<float>>(kSpecialValRegistry);

  MathTest math_test(kernel, values.size);
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func, values.size,
                                values.data);
}

template <typename T, typename F, typename ValidatorBuilder>
void SpecialSimpleTest(F kernel, const ValidatorBuilder& validator_builder,
                       const T* xs, const T* ref, size_t num_args) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);

  LinearAllocGuard<T> xs_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  LinearAllocGuard<T> ys{LinearAllocs::hipHostMalloc, num_args * sizeof(T)};
  LinearAllocGuard<T> ys_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};

  HIP_CHECK(hipMemcpy(xs_dev.ptr(), xs, num_args * sizeof(T), hipMemcpyHostToDevice));

  kernel<<<grid_size, block_size>>>(ys_dev.ptr(), num_args, xs_dev.ptr());
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(ys.ptr(), ys_dev.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));
  for (auto i = 0u; i <  num_args; ++i) {
    const auto actual_val = ys.ptr()[i];
    const auto ref_val = ref[i];
    const auto validator = validator_builder(ref_val);

    //if (actual_val == ref_val) {
    if (!validator.match(actual_val)) {
      std::stringstream ss;
      ss << "Input value(s): " << std::scientific << std::setprecision(std::numeric_limits<T>::max_digits10 - 1);
      ss << xs[i] << " " << actual_val <<" " << ref_val<<"\n";
      std::cout<< ss.str();
      //printf("NINA: %lf %lf %lf\n", xs[i], actual_val, ref_val);
      REQUIRE(false);
    }
  }
}

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(erf, 2, 2)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(erfc, 4, 5)

MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(tgamma, 5, 10)

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
  UnarySinglePrecisionTest(erfcinv_kernel<float>, ref,
                           ULPValidatorBuilderFactory<float>(4));
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
  UnaryDoublePrecisionTest(erfcinv_kernel<double>, ref,
                           ULPValidatorBuilderFactory<double>(6));
}

MATH_UNARY_KERNEL_DEF(erfinv)

TEST_CASE("Unit_Device_erfinvf_Accuracy_Positive") {
  auto erfinv_ref = [](double arg) -> double {
    if (arg == 0)
      return 0;
    if (arg == 1)
      return std::numeric_limits<double>::infinity();
    else if (arg == -1)
      return -std::numeric_limits<double>::infinity();
    else if (arg < -1 || arg > 1)
      return std::numeric_limits<double>::quiet_NaN();
    return boost::math::erf_inv(arg);
  };
  double (*ref)(double) = erfinv_ref;
  UnarySinglePrecisionTest(erfinv_kernel<float>, ref,
                           ULPValidatorBuilderFactory<float>(2));
}

TEST_CASE("Unit_Device_erfinv_Accuracy_Positive") {
  auto erfinv_ref = [](long double arg) -> long double {
    if (arg == 0)
      return 0;
    if (arg == 1)
      return std::numeric_limits<long double>::infinity();
    else if (arg == -1)
      return -std::numeric_limits<long double>::infinity();
    else if (arg < -1 || arg > 1)
      return std::numeric_limits<long double>::quiet_NaN();
    return boost::math::erf_inv(arg);
  };
  long double (*ref)(long double) = erfinv_ref;
  UnaryDoublePrecisionTest(erfinv_kernel<double>, ref,
                           ULPValidatorBuilderFactory<double>(5));
}

MATH_UNARY_KERNEL_DEF(normcdf)

TEST_CASE("Unit_Device_normcdff_Accuracy_Positive") {
  auto normcdf_ref = [](double arg) -> double { return std::erfc(-arg / std::sqrt(2)) / 2; };
  double (*ref)(double) = normcdf_ref;
  UnarySinglePrecisionTest(normcdf_kernel<float>, ref,
                           ULPValidatorBuilderFactory<float>(5));
}

TEST_CASE("Unit_Device_normcdf_Accuracy_Positive") {
  auto normcdf_ref = [](long double arg) -> long double { return std::erfc(-arg / std::sqrt(2.L)) / 2; };
  long double (*ref)(long double) = normcdf_ref;
  UnaryDoublePrecisionTest(normcdf_kernel<double>, ref,
                           ULPValidatorBuilderFactory<double>(5));
}

MATH_UNARY_KERNEL_DEF(normcdfinv)

TEST_CASE("Unit_Device_normcdfinvf_Accuracy_Positive") {
  constexpr std::array<float, 9> input  {0.f, 0.1f, 0.25f, 0.4f, 0.5f, 0.6f, 0.75f, 0.9f, 1.f};
  constexpr std::array<float, 9> reference {-std::numeric_limits<float>::infinity(), -1.28155160f, -0.674489737f, -0.253347069f, 0, 0.253347158f, 0.674489737f, 1.28155148f, std::numeric_limits<float>::infinity()};
  SpecialSimpleTest<float>(normcdfinv_kernel<float>, ULPValidatorBuilderFactory<float>(5), input.data(), reference.data(), input.size());
}

TEST_CASE("Unit_Device_normcdfinv_Accuracy_Positive") {
  constexpr std::array<double, 9> input  {0., 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.};
  constexpr std::array<double, 9> reference {-std::numeric_limits<float>::infinity(), -1.2815515655446004, -0.67448975019608159, -0.25334710313579972, 0, 0.25334710313579972, 0.67448975019608159, 1.2815515655446006, std::numeric_limits<float>::infinity()};
  SpecialSimpleTest<double>(normcdfinv_kernel<double>, ULPValidatorBuilderFactory<double>(5), input.data(), reference.data(), input.size());
}

MATH_UNARY_KERNEL_DEF(erfcx)

TEST_CASE("Unit_Device_erfcxf_Accuracy_Positive") {
 /**auto erfcx_ref = [](double arg) -> double {
    if (arg == -std::numeric_limits<double>::infinity())
      return std::numeric_limits<double>::infinity();
    else if (arg == std::numeric_limits<double>::infinity())
      return 0;
    else if (std::isnan(arg))
      return std::numeric_limits<double>::quiet_NaN();
    return std::exp(arg*arg) * std::erfc(arg);
  };
  double (*ref)(double) = erfcx_ref;
  UnarySinglePrecisionSpecialValuesTest(erfcx_kernel<float>, ref,
                           ULPValidatorBuilderFactory<float>(4));*/
  constexpr std::array<float, 11> input  {-std::numeric_limits<float>::infinity(), 1000.f, 100.f, 5.f, 0.5f, 0., 0.75f, 15.f, 200.f, 500.f, std::numeric_limits<float>::infinity()};
  constexpr std::array<float, 11> reference {std::numeric_limits<float>::infinity(), 5.64189337e-4f, 5.64161362e-3f, 1.10704638e-1f, 6.15690351e-1f, 1.0f, 5.06937683e-1f, 3.75296101e-2f, 2.82091252e-3f, 1.12837693e-3f, 0.f};
  SpecialSimpleTest<float>(erfcx_kernel<float>, ULPValidatorBuilderFactory<float>(5), input.data(), reference.data(), input.size());
}

TEST_CASE("Unit_Device_erfcx_Accuracy_Positive") {
/*  auto erfcx_ref = [](long double arg) -> long double {
    if (arg == -std::numeric_limits<long double>::infinity())
      return std::numeric_limits<long double>::infinity();
    else if (arg == std::numeric_limits<long double>::infinity())
      return 0;
    else if (std::isnan(arg))
      return std::numeric_limits<long double>::quiet_NaN();
    return std::exp(arg*arg) * std::erfc(arg);
  };
  long double (*ref)(long double) = erfcx_ref;
  UnaryDoublePrecisionTest(erfcx_kernel<double>, ref,
                           ULPValidatorBuilderFactory<double>(5));*/
  constexpr std::array<double, 11> input  {-std::numeric_limits<double>::infinity(), 1000., 100., 5., 0.5, 0., 0.75, 15., 200., 500., std::numeric_limits<double>::infinity()};
  constexpr std::array<double, 11> reference {std::numeric_limits<double>::infinity(), 5.6418930145338774e-4, 5.6416137829894330e-3, 1.1070463773306863e-1, 6.1569034419292590e-1, 1.0, 5.0693765029314475e-1, 3.7529606388505762e-2, 2.8209126572120466e-3, 1.1283769103507188e-3, 0.};
  SpecialSimpleTest<double>(erfcx_kernel<double>, ULPValidatorBuilderFactory<double>(5), input.data(), reference.data(), input.size());
}

template <typename T, typename RT = RefType_t<T>, typename F, typename RF, typename ValidatorBuilder>
void BesselSpecialValuesTest(F kernel, RF ref_func,
                             const ValidatorBuilder& validator_builder,
                             const float a = std::numeric_limits<T>::lowest(),
                             const float b = std::numeric_limits<T>::max()) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values = std::get<SpecialVals<T>>(kSpecialValRegistry);
  std::vector<T> range_values;
  for (uint64_t i = 0; i < values.size; i++) {
    //if (isnan(values.data[i]) || isinf(values.data[i])) range_values.push_back(values.data[i]);
    if (values.data[i] >= a && values.data[i] <= b) range_values.push_back(values.data[i]);
  }

  MathTest math_test(kernel, range_values.size());
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func, range_values.size(),
                                &range_values[0]);
}

template <typename T, typename RT = RefType_t<T>, typename F, typename RF, typename ValidatorBuilder>
void BesselNSpecialValuesTest(F kernel, RF ref_func,
                              const ValidatorBuilder& validator_builder, int n,
                              const float a = std::numeric_limits<T>::lowest(),
                              const float b = std::numeric_limits<T>::max()) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values = std::get<SpecialVals<T>>(kSpecialValRegistry);
  std::vector<T> range_values;
  for (uint64_t i = 0; i < values.size; i++) {
    if (isnan(values.data[i]) || isinf(values.data[i])) range_values.push_back(values.data[i]);
    else if (values.data[i] >= a && values.data[i] <= b) range_values.push_back(values.data[i]);
  }
  LinearAllocGuard<int> ns{LinearAllocs::hipHostMalloc, range_values.size() * sizeof(int)};
  std::fill_n(ns.ptr(), range_values.size(), n);

  MathTest math_test(kernel, range_values.size());
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func, range_values.size(),
                                ns.ptr(), &range_values[0]);
}

MATH_UNARY_KERNEL_DEF(cyl_bessel_i0)

TEST_CASE("Unit_Device_cyl_bessel_i0f_Accuracy_Positive") {
  auto cyl_bessel_i0_ref = [](double arg) -> double {
    return std::cyl_bessel_i(0, arg);
  };
  double (*ref)(double) = cyl_bessel_i0_ref;
  UnarySinglePrecisionRangeTest(cyl_bessel_i0_kernel<float>, ref, ULPValidatorBuilderFactory<float>(6), 0, 100000);
  //BesselSpecialValuesTest<float, double>(cyl_bessel_i0_kernel<float>, ref, ULPValidatorBuilderFactory<float>(6), 0);
}

TEST_CASE("Unit_Device_cyl_bessel_i0_Accuracy_Positive") {
  auto cyl_bessel_i0_ref = [](long double arg) -> long double {
    return std::cyl_bessel_i(0, arg);
  };
  long double (*ref)(long double) = cyl_bessel_i0_ref;
  UnaryDoublePrecisionBruteForceTest<long double>(cyl_bessel_i0_kernel<double>, ref, ULPValidatorBuilderFactory<double>(6), 0, 100000);
  //BesselSpecialValuesTest<double, long double>(cyl_bessel_i0_kernel<double>, ref, ULPValidatorBuilderFactory<double>(6), 0);
}

MATH_UNARY_KERNEL_DEF(cyl_bessel_i1)

TEST_CASE("Unit_Device_cyl_bessel_i1f_Accuracy_Positive") {
  auto cyl_bessel_i1_ref = [](double arg) -> double {
    return std::cyl_bessel_i(1, arg);
  };
  double (*ref)(double) = cyl_bessel_i1_ref;
  UnarySinglePrecisionRangeTest(cyl_bessel_i1_kernel<float>, ref, ULPValidatorBuilderFactory<float>(6), 0, 15000);
  //BesselSpecialValuesTest<float, double>(cyl_bessel_i1_kernel<float>, ref, ULPValidatorBuilderFactory<float>(6), 0);
}

TEST_CASE("Unit_Device_cyl_bessel_i1_Accuracy_Positive") {
  auto cyl_bessel_i1_ref = [](long double arg) -> long double {
    return std::cyl_bessel_i(0, arg);
  };
  long double (*ref)(long double) = cyl_bessel_i1_ref;
  UnaryDoublePrecisionBruteForceTest(cyl_bessel_i1_kernel<double>, ref, ULPValidatorBuilderFactory<double>(6), 0, 15000);
  //BesselSpecialValuesTest<double, long double>(cyl_bessel_i0_kernel<double>, ref, ULPValidatorBuilderFactory<double>(6), 0);
}

MATH_UNARY_KERNEL_DEF(y0)

TEST_CASE("Unit_Device_y0f_Accuracy_Positive") {
  double (*ref)(double) = y0;
  //BesselSpecialValuesTest<float, double>(y0_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f, 8.f);
  UnarySinglePrecisionRangeTest(y0_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f, 8.f);
  UnarySinglePrecisionRangeTest(y0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022), 8.f, std::numeric_limits<float>::max());
  //UnarySinglePrecisionBruteForceTest<double>(y0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022));
  //BesselSpecialValuesTest<float, double>(y0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022), std::numeric_limits<float>::lowest());
  //BesselSpecialValuesTest<float, double>(y0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022), 8.f);
}

TEST_CASE("Unit_Device_y0_Accuracy_Positive") {
  long double (*ref)(long double) = y0l;
  UnaryDoublePrecisionBruteForceTest(y0_kernel<double>, ref, ULPValidatorBuilderFactory<double>(9), -8., 8);
  UnaryDoublePrecisionBruteForceTest(y0_kernel<double>, ref, AbsValidatorBuilderFactory<double>(0.0000022), 8., std::numeric_limits<double>::max());
  //UnaryDoublePrecisionBruteForceTest<long double>(y0_kernel<double>, ref, AbsValidatorBuilderFactory<double>(2.e-16), -8., 8.);
  //UnaryDoublePrecisionBruteForceTest<long double>(y0_kernel<double>, ref, AbsValidatorBuilderFactory<double>(0.0000022));
  //BesselSpecialValuesTest<double, long double>(y0_kernel<double>, ref, ULPValidatorBuilderFactory<double>(7), -8., 8.);
  //BesselSpecialValuesTest<double, long double>(y0_kernel<double>, ref, AbsValidatorBuilderFactory<double>(0.0000022), std::numeric_limits<double>::lowest(), -8.);
  //BesselSpecialValuesTest<double, long double>(y0_kernel<double>, ref, AbsValidatorBuilderFactory<double>(0.0000022), 8.);
}

MATH_UNARY_KERNEL_DEF(y1)

TEST_CASE("Unit_Device_y1f_Accuracy_Positive") {
  double (*ref)(double) = y1;
  BesselSpecialValuesTest<float, double>(y1_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f, 8.f);
  BesselSpecialValuesTest<float, double>(y1_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022), std::numeric_limits<float>::lowest(), -8.f);
  BesselSpecialValuesTest<float, double>(y1_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022), 8.f);
}

TEST_CASE("Unit_Device_y1_Accuracy_Positive") {
  long double (*ref)(long double) = y1l;
  BesselSpecialValuesTest<double, long double>(y1_kernel<double>, ref, ULPValidatorBuilderFactory<double>(7), -8., 8.);
  BesselSpecialValuesTest<double, long double>(y1_kernel<double>, ref, AbsValidatorBuilderFactory<double>(0.0000022), std::numeric_limits<double>::lowest(), -8.);
  BesselSpecialValuesTest<double, long double>(y1_kernel<double>, ref, AbsValidatorBuilderFactory<double>(0.0000022), 8.);
}

MATH_BESSEL_N_KERNEL_DEF(yn)

TEST_CASE("Unit_Device_ynf_Accuracy_Positive") {
  double (*ref)(int, double) = yn;
  int n = 5;
  BesselNSpecialValuesTest<float, double>(yn_kernel<float>, ref, ULPValidatorBuilderFactory<float>(std::ceil(2 + 2.5 * n)), n, -n, n);
  BesselNSpecialValuesTest<float, double>(yn_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022), n, std::numeric_limits<float>::lowest(), -n);
  BesselNSpecialValuesTest<float, double>(yn_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022), n, n);
}

TEST_CASE("Unit_Device_yn_Accuracy_Positive") {
  long double (*ref)(int, long double) = ynl;
  int n = 5;
  BesselNSpecialValuesTest<double, long double>(yn_kernel<double>, ref, ULPValidatorBuilderFactory<double>(std::ceil(2 + 2.5 * n)), n, -n, n);
  BesselNSpecialValuesTest<double, long double>(yn_kernel<double>, ref, AbsValidatorBuilderFactory<double>(0.0000022), n, std::numeric_limits<double>::lowest(), -n);
  BesselNSpecialValuesTest<double, long double>(yn_kernel<double>, ref, AbsValidatorBuilderFactory<double>(0.0000022), n, n);
}

MATH_UNARY_KERNEL_DEF(j0)

TEST_CASE("Unit_Device_j0f_Accuracy_Positive") {
  double (*ref)(double) = j0;
  BesselSpecialValuesTest<float, double>(j0_kernel<float>, ref, ULPValidatorBuilderFactory<float>(9), -8.f, 8.f);
  BesselSpecialValuesTest<float, double>(j0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022), std::numeric_limits<float>::lowest(), -8.f);
  BesselSpecialValuesTest<float, double>(j0_kernel<float>, ref, AbsValidatorBuilderFactory<float>(0.0000022), 8.f);
}
/*
TEST_CASE("Unit_Device_y1_Accuracy_Positive") {
  long double (*ref)(long double) = y1l;
  BesselSinglePrecisionSpecialValuesTest<double, long double>(y1_kernel<double>, ref, ULPValidatorBuilderFactory<double>(7), -8., 8.);
  BesselSinglePrecisionSpecialValuesTest<double, long double>(y1_kernel<double>, ref, AbsValidatorBuilderFactory<double>(0.0000022), std::numeric_limits<double>::lowest(), -8.);
  BesselSinglePrecisionSpecialValuesTest<double, long double>(y1_kernel<double>, ref, AbsValidatorBuilderFactory<double>(0.0000022), 8.);
}*/