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

#include "math_common.hh"
#include "math_special_values.hh"

#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#define MATH_UNARY_KERNEL_DEF(func_name)                                                           \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const xs) {              \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[i] = func_name##f(xs[i]);                                                               \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[i] = func_name(xs[i]);                                                                  \
      }                                                                                            \
    }                                                                                              \
  }

template <typename RT = RefType_t<float>, typename F, typename RF, typename ValidatorBuilder>
void UnarySinglePrecisionBruteForceTest(F kernel, RF ref_func,
                                        const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  // Divide by two on account of an output device array being allocated beside the input array.
  // Possible point of optimization: reuse the input array to store the outputs
  uint64_t stop = std::numeric_limits<uint32_t>::max() + 1ul;
  const auto max_batch_size =
      std::min(GetMaxAllowedDeviceMemoryUsage() / (sizeof(float) * 2), stop);
  LinearAllocGuard<float> values{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(float)};

  MathTest<float, RT, 1> math_test(max_batch_size);

  auto batch_size = max_batch_size;
  uint32_t val = 0u;
  const auto num_threads = thread_pool.thread_count();

  for (uint64_t v = 0u; v < stop;) {
    batch_size = std::min<uint64_t>(max_batch_size, stop - v);

    const auto min_sub_batch_size = batch_size / num_threads;
    const auto tail = batch_size % num_threads;

    auto base_idx = 0u;
    for (auto i = 0u; i < num_threads; ++i) {
      const auto sub_batch_size = min_sub_batch_size + (i < tail);

      thread_pool.Post([=, &values] {
        auto t = v;
        uint32_t val;
        for (auto j = 0u; j < sub_batch_size; ++j) {
          val = static_cast<uint32_t>(t++);
          values.ptr()[base_idx + j] = *reinterpret_cast<float*>(&val);
        }
      });

      v += sub_batch_size;
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, kernel, ref_func, batch_size,
                  values.ptr());
  }
}

template <typename RT = RefType_t<double>, typename F, typename RF, typename ValidatorBuilder>
void UnaryDoublePrecisionBruteForceTest(F kernel, RF ref_func,
                                        const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const uint64_t num_iterations = GetTestIterationCount();
  const auto max_batch_size =
      std::min(GetMaxAllowedDeviceMemoryUsage() / (sizeof(double) * 2), num_iterations);
  LinearAllocGuard<double> values{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(double)};

  MathTest<double, RT, 1> math_test(max_batch_size);

  auto batch_size = max_batch_size;
  const auto num_threads = thread_pool.thread_count();
  for (uint64_t i = 0ul; i < num_iterations; i += batch_size) {
    batch_size = std::min<uint64_t>(max_batch_size, num_iterations - i);

    const auto min_sub_batch_size = batch_size / num_threads;
    const auto tail = batch_size % num_threads;

    auto base_idx = 0u;
    for (auto i = 0u; i < num_threads; ++i) {
      const auto sub_batch_size = min_sub_batch_size + (i < tail);
      thread_pool.Post([=, &values] {
        const auto generator = [] {
          static thread_local std::mt19937 rng(std::random_device{}());
          std::uniform_real_distribution<double> unif_dist(std::numeric_limits<double>::lowest(),
                                                           std::numeric_limits<double>::max());
          return unif_dist(rng);
        };
        std::generate(values.ptr() + base_idx, values.ptr() + base_idx + sub_batch_size, generator);
      });
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, kernel, ref_func, batch_size,
                  values.ptr());
  }
}

template <typename RT = RefType_t<double>, typename F, typename RF, typename ValidatorBuilder>
void UnaryDoublePrecisionSpecialValuesTest(F kernel, RF ref_func,
                                           const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values = std::get<SpecialVals<double>>(kSpecialValRegistry);

  MathTest<double, RT, 1> math_test(values.size);
  math_test.template Run<false>(validator_builder, grid_size, block_size, kernel, ref_func,
                                values.size, values.data);
}

template <typename RT = double, typename ValidatorBuilder>
void UnarySinglePrecisionTest(void (*kernel)(float* const, const size_t, float* const),
                              RT (*ref)(RT), const ValidatorBuilder& validator_builder) {
  SECTION("Brute force") { UnarySinglePrecisionBruteForceTest<RT>(kernel, ref, validator_builder); }
}

template <typename RT = long double, typename ValidatorBuilder>
void UnaryDoublePrecisionTest(void (*kernel)(double* const, const size_t, double* const),
                              RT (*ref)(RT), const ValidatorBuilder& validator_builder) {
  SECTION("Special values") {
    UnaryDoublePrecisionSpecialValuesTest<RT>(kernel, ref, validator_builder);
  }

  SECTION("Brute force") { UnaryDoublePrecisionBruteForceTest<RT>(kernel, ref, validator_builder); }
}

#define MATH_UNARY_WITHIN_ULP_TEST_DEF(kern_name, ref_func, sp_ulp, dp_ulp)                        \
  MATH_UNARY_KERNEL_DEF(kern_name)                                                                 \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - float") {                              \
    UnarySinglePrecisionTest(kern_name##_kernel<float>, ref_func,                                  \
                             ULPValidatorBuilderFactory<float>(sp_ulp));                           \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - double") {                             \
    UnaryDoublePrecisionTest(kern_name##_kernel<double>, ref_func,                                 \
                             ULPValidatorBuilderFactory<double>(dp_ulp));                          \
  }

#define MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(func_name, sp_ulp, dp_ulp)                          \
  MATH_UNARY_WITHIN_ULP_TEST_DEF(func_name, std::func_name, sp_ulp, dp_ulp)