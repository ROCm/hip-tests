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
#include "ternary_common.hh"
#include "quaternary_common.hh"

#define MATH_NORM_KERNEL_DEF(func_name)                                                            \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, int dim, T* const x1s) {                         \
                                                                                                   \
    if constexpr (std::is_same_v<float, T>) {                                                      \
      *ys = func_name##f(dim, x1s);                                                                \
    } else if constexpr (std::is_same_v<double, T>) {                                              \
      *ys = func_name(dim, x1s);                                                                   \
    }                                                                                              \
  }

MATH_UNARY_KERNEL_DEF(sqrt)

TEST_CASE("Unit_Device_sqrtf_Accuracy_Positive") {
  float (*ref)(float) = std::sqrt;
  UnarySinglePrecisionTest(sqrt_kernel<float>, ref,
                           ULPValidatorBuilderFactory<float>(1));
}

TEST_CASE("Unit_Device_sqrt_Accuracy_Positive") {
  double (*ref)(double) = std::sqrt;
  UnaryDoublePrecisionTest<double>(sqrt_kernel<double>, ref,
                           ULPValidatorBuilderFactory<double>(0));
}

MATH_UNARY_KERNEL_DEF(rsqrt)

TEST_CASE("Unit_Device_rsqrtf_Accuracy_Positive") {
  auto rsqrt_ref = [](double arg) -> double { return 1. / std::sqrt(arg); };
  double (*ref)(double) = rsqrt_ref;
  UnarySinglePrecisionTest(rsqrt_kernel<float>, ref,
                           ULPValidatorBuilderFactory<float>(2));
}

TEST_CASE("Unit_Device_rsqrt_Accuracy_Positive") {
  auto rsqrt_ref = [](long double arg) -> long double { return 1.L / std::sqrt(arg); };
  long double (*ref)(long double) = rsqrt_ref;
  UnaryDoublePrecisionTest(rsqrt_kernel<double>, ref,
                           ULPValidatorBuilderFactory<double>(1));
}

MATH_UNARY_WITHIN_ULP_TEST_DEF(cbrt, std::cbrt, 1, 1) 

MATH_UNARY_KERNEL_DEF(rcbrt)

TEST_CASE("Unit_Device_rcbrtf_Accuracy_Positive") {
  auto rcbrt_ref = [](double arg) -> double { return 1. / std::cbrt(arg); };
  double (*ref)(double) = rcbrt_ref;
  UnarySinglePrecisionTest(rcbrt_kernel<float>, ref,
                           ULPValidatorBuilderFactory<float>(1));
}

TEST_CASE("Unit_Device_rcbrt_Accuracy_Positive") {
  auto rcbrt_ref = [](long double arg) -> long double { return 1. / std::cbrt(arg); };
  long double (*ref)(long double) = rcbrt_ref;
  UnaryDoublePrecisionTest(rcbrt_kernel<double>, ref,
                           ULPValidatorBuilderFactory<double>(1));
}

MATH_BINARY_WITHIN_ULP_TEST_DEF(hypot, std::hypot, 3, 2)

MATH_BINARY_KERNEL_DEF(rhypot)

TEMPLATE_TEST_CASE("Unit_Device_rhypot_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto rhypot_ref = [](RT arg1, RT arg2) -> RT { return 1. / std::hypot(arg1, arg2); };
  RT (*ref)(RT, RT) = rhypot_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 2 : 1;
  BinaryFloatingPointTest(rhypot_kernel<TestType>, ref,
                                     ULPValidatorBuilderFactory<TestType>(ulp));
}

MATH_TERNARY_KERNEL_DEF(norm3d)

TEMPLATE_TEST_CASE("Unit_Device_norm3d_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto norm3d_ref = [](RT arg1, RT arg2, RT arg3) -> RT {
    if (std::isinf(arg1) || std::isinf(arg2) || std::isinf(arg3)) {
      return std::numeric_limits<RT>::infinity();
    }
    return std::sqrt(arg1 * arg1 + arg2 * arg2 + arg3 * arg3);
  };
  RT (*ref)(RT, RT, RT) = norm3d_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 3 : 2;
  TernaryFloatingPointTest(norm3d_kernel<TestType>, ref,
                                     ULPValidatorBuilderFactory<TestType>(ulp));
}

MATH_TERNARY_KERNEL_DEF(rnorm3d)

TEMPLATE_TEST_CASE("Unit_Device_rnorm3d_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto rnorm3d_ref = [](RT arg1, RT arg2, RT arg3) -> RT {
    if (std::isinf(arg1) || std::isinf(arg2) || std::isinf(arg3)) {
      return 0;
    }
    return 1. / std::sqrt(arg1 * arg1 + arg2 * arg2 + arg3 * arg3);
  };
  RT (*ref)(RT, RT, RT) = rnorm3d_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 2 : 1;
  TernaryFloatingPointTest(rnorm3d_kernel<TestType>, ref,
                                     ULPValidatorBuilderFactory<TestType>(ulp));
}

MATH_QUATERNARY_KERNEL_DEF(norm4d)

TEMPLATE_TEST_CASE("Unit_Device_norm4d_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto norm4d_ref = [](RT arg1, RT arg2, RT arg3, RT arg4) -> RT {
    if (std::isinf(arg1) || std::isinf(arg2) || std::isinf(arg3) || std::isinf(arg4)) {
      return std::numeric_limits<RT>::infinity();
    }
    return std::sqrt(arg1 * arg1 + arg2 * arg2 + arg3 * arg3 + arg4 * arg4);
  };
  RT (*ref)(RT, RT, RT, RT) = norm4d_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 3 : 2;
  QuaternaryFloatingPointTest(norm4d_kernel<TestType>, ref,
                                     ULPValidatorBuilderFactory<TestType>(ulp));
}

MATH_QUATERNARY_KERNEL_DEF(rnorm4d)

TEMPLATE_TEST_CASE("Unit_Device_rnorm4d_Accuracy_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto rnorm4d_ref = [](RT arg1, RT arg2, RT arg3, RT arg4) -> RT {
    if (std::isinf(arg1) || std::isinf(arg2) || std::isinf(arg3) || std::isinf(arg4)) {
      return 0;
    }
    return 1. / std::sqrt(arg1 * arg1 + arg2 * arg2 + arg3 * arg3 + arg4 * arg4);
  };
  RT (*ref)(RT, RT, RT, RT) = rnorm4d_ref;
  const auto ulp = std::is_same_v<float, TestType> ? 2 : 1;
  QuaternaryFloatingPointTest(rnorm4d_kernel<TestType>, ref,
                                     ULPValidatorBuilderFactory<TestType>(ulp));
}

template <typename T, typename RT = RefType_t<T>, typename F, typename RF,
          typename ValidatorBuilder>
void NormSimpleTest(F kernel, RF ref_func, const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto max_dim = 10000;

  LinearAllocGuard<T> x{LinearAllocs::hipHostMalloc, max_dim * sizeof(T)};
  LinearAllocGuard<T> x_dev{LinearAllocs::hipMalloc, max_dim * sizeof(T)};
  LinearAllocGuard<T> y{LinearAllocs::hipHostMalloc, sizeof(T)};
  LinearAllocGuard<T> y_dev{LinearAllocs::hipMalloc, sizeof(T)};

  std::fill_n(x.ptr(), max_dim, 1);
  HIP_CHECK(hipMemcpy(x_dev.ptr(), x.ptr(), max_dim * sizeof(T), hipMemcpyHostToDevice));

  for (uint64_t i = 1u; i < max_dim; i ++) {
    kernel<<<grid_size, block_size>>>(y_dev.ptr(), i, x_dev.ptr());
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(y.ptr(), y_dev.ptr(), sizeof(T), hipMemcpyDeviceToHost));
    const auto actual_val = *y.ptr();
    const auto ref_val = static_cast<T>(ref_func(i, x.ptr()));
    const auto validator = validator_builder(ref_val);

    if (!validator.match(actual_val)) {
      std::stringstream ss;
      ss << std::scientific << std::setprecision(std::numeric_limits<T>::max_digits10 - 1);
      ss << "Validation fails for dim: " << i << " " << actual_val << " " << ref_val;
      INFO(ss.str());
      REQUIRE(false);
    }
  }
}

MATH_NORM_KERNEL_DEF(norm)

TEMPLATE_TEST_CASE("Unit_Device_norm_Accuracy_Limited_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto norm_ref = [](int dim, TestType* args) -> RT {
    RT sum = 0;
    for (int i = 0; i < dim; i++)
    {
      if (std::isinf(args[i]))
        return std::numeric_limits<RT>::infinity();
      sum += static_cast<RT>(args[i]) * static_cast<RT>(args[i]);
    }
    return std::sqrt(sum);
  };
  RT (*ref)(int, TestType*) = norm_ref;
  const auto validator_builder = ULPValidatorBuilderFactory<TestType>(10);

  NormSimpleTest<TestType, RT>(norm_kernel<TestType>, ref, validator_builder);
}

MATH_NORM_KERNEL_DEF(rnorm)

TEMPLATE_TEST_CASE("Unit_Device_rnorm_Accuracy_Limited_Positive", "", float, double) {
  using RT = RefType_t<TestType>;
  auto rnorm_ref = [](int dim, TestType* args) -> RT {
    RT sum = 0;
    for (int i = 0; i < dim; i++)
    {
      if (std::isinf(args[i]))
        return std::numeric_limits<RT>::infinity();
      sum += static_cast<RT>(args[i]) * static_cast<RT>(args[i]);
    }
    return 1. / std::sqrt(sum);
  };
  RT (*ref)(int, TestType*) = rnorm_ref;
  const auto validator_builder = ULPValidatorBuilderFactory<TestType>(10);

  NormSimpleTest<TestType, RT>(rnorm_kernel<TestType>, ref, validator_builder);
}