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

#include <hip/hip_cooperative_groups.h>

#include <hip_test_common.hh>
#include <resource_guards.hh>

namespace cg = cooperative_groups;

#define MATH_SINGLE_ARG_KERNEL_DEF(func_name)                                                      \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const xs) {              \
    const auto tid = cg::this_grid().thread_rank();                                                \
                                                                                                   \
    if (tid < num_xs) {                                                                            \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[tid] = func_name##f(xs[tid]);                                                           \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[tid] = func_name(xs[tid]);                                                              \
      }                                                                                            \
    }                                                                                              \
  }

#define MATH_DOUBLE_ARG_KERNEL_DEF(func_name)                                                      \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s,               \
                                     T* const x2s) {                                               \
    const auto tid = cg::this_grid().thread_rank();                                                \
                                                                                                   \
    if (tid < num_xs) {                                                                            \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[tid] = func_name##f(x1s[tid], x2s[tid]);                                                \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[tid] = func_name(x1s[tid], x2s[tid]);                                                   \
      }                                                                                            \
    }                                                                                              \
  }

#define MATH_TRIPLE_ARG_KERNEL_DEF(func_name)                                                      \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s, T* const x2s, \
                                     T* const x3s) {                                               \
    const auto tid = cg::this_grid().thread_rank();                                                \
                                                                                                   \
    if (tid < num_xs) {                                                                            \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[tid] = func_name##f(x1s[tid], x2s[tid], x3s[tid]);                                      \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[tid] = func_name(x1s[tid], x2s[tid], x3s[tid]);                                         \
      }                                                                                            \
    }                                                                                              \
  }

#define MATH_QUADRUPLE_ARG_KERNEL_DEF(func_name)                                                   \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s, T* const x2s, \
                                     T* const x3s, T* const x4s) {                                 \
    const auto tid = cg::this_grid().thread_rank();                                                \
                                                                                                   \
    if (tid < num_xs) {                                                                            \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[tid] = func_name##f(x1s[tid], x2s[tid], x3s[tid], x4s[tid]);                            \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[tid] = func_name(x1s[tid], x2s[tid], x3s[tid], x4s[tid]);                               \
      }                                                                                            \
    }                                                                                              \
  }

template <bool parallel = true, typename T, typename RT, typename ValidatorBuilder, typename... Ts,
          typename... RTs, size_t... I>
void MathTestImpl(const ValidatorBuilder& validator_builder, const size_t grid_dim,
                  const size_t block_dim, void (*const kernel)(T*, const size_t, Ts*...),
                  RT (*const ref_func)(RTs...), const size_t num_args, std::index_sequence<I...> is,
                  const Ts*... xss) {
  struct LAWrapper {
    LAWrapper(const size_t size, const T* const init_vals)
        : la_{LinearAllocs::hipMalloc, size, 0u} {
      HIP_CHECK(hipMemcpy(ptr(), init_vals, size, hipMemcpyHostToDevice));
    }

    T* ptr() { return la_.ptr(); }

    LinearAllocGuard<T> la_;
  };

  std::array xss_dev{LAWrapper(num_args * sizeof(T), xss)...};

  LinearAllocGuard<T> y_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  kernel<<<grid_dim, block_dim>>>(y_dev.ptr(), num_args, xss_dev[I].ptr()...);
  HIP_CHECK(hipGetLastError());

  std::vector<T> y(num_args);
  HIP_CHECK(hipMemcpy(y.data(), y_dev.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));

  if constexpr (!parallel) {
    for (auto i = 0u; i < num_args; ++i) {
      REQUIRE_THAT(y[i], validator_builder(static_cast<T>(ref_func(static_cast<RT>(xss[i])...))));
    }
    return;
  }

  std::atomic<bool> fail_flag{false};
  std::mutex ss_mtx;
  std::stringstream ss;

  const auto tf = [&, validator_builder](size_t iters, size_t base_idx) mutable {
    for (auto i = 0; i < iters; ++i) {
      if (fail_flag.load(std::memory_order_relaxed)) {
        return;
      }

      const auto actual_val = y[base_idx + i];
      const auto ref_val = static_cast<T>(ref_func(static_cast<RT>(xss[base_idx + i])...));
      const auto validator = validator_builder(ref_val);
      if (!validator.match(actual_val)) {
        fail_flag.store(true, std::memory_order_relaxed);
        // Several threads might have passed the first check, but failed validation. On the chance
        // of this happening, access to the string stream must be serialized.
        {
          std::lock_guard{ss_mtx};
          ss << std::to_string(actual_val) << ' ' << validator.describe() << '\n';
        }
        return;
      }
    }
  };

  // This will be replaced by a proper thread-pool implementation
  std::vector<std::thread> threads;
  const auto core_count = std::thread::hardware_concurrency();
  const auto chunk_size = num_args / core_count;
  const auto tail = num_args % core_count;
  auto base_idx = 0u;
  for (auto i = 0u; i < core_count; ++i) {
    const auto iters = i < tail ? chunk_size + 1 : chunk_size;
    threads.emplace_back(tf, iters, base_idx);
    base_idx += iters;
  }

  for (auto& t : threads) {
    t.join();
  }

  INFO(ss.str());
  REQUIRE(!fail_flag);
}

template <bool parallel = true, typename T, typename RT, typename ValidatorBuilder, typename... Ts,
          typename... RTs>
void MathTest(const ValidatorBuilder& validator_builder, const size_t grid_dims,
              const size_t block_dims, void (*const kernel)(T*, const size_t, Ts*...),
              RT (*const ref_func)(RTs...), const size_t num_args, const Ts*... xss) {
  MathTestImpl<parallel>(validator_builder, grid_dims, block_dims, kernel, ref_func, num_args,
                         std::index_sequence_for<Ts...>{}, xss...);
}

template <typename T, typename Matcher> class ValidatorBase : public Catch::MatcherBase<T> {
 public:
  template <typename... Ts>
  ValidatorBase(T target, Ts&&... args) : matcher_{std::forward<Ts>(args)...}, target_{target} {}

  bool match(const T& val) const override {
    if (std::isnan(target_)) {
      return std::isnan(val);
    }

    return matcher_.match(val);
  }

  virtual std::string describe() const override {
    if (std::isnan(target_)) {
      return "is not NaN";
    }

    return matcher_.describe();
  }

 private:
  Matcher matcher_;
  T target_;
  bool nan = false;
};

template <typename T> auto ULPValidatorGenerator(int64_t ulps) {
  return [=](T target) {
    return ValidatorBase<T, Catch::Matchers::Floating::WithinUlpsMatcher>{
        target, Catch::WithinULP(target, ulps)};
  };
};

template <typename T> auto AbsValidatorBuilderFactory(double margin) {
  return [=](T target) {
    return ValidatorBase<T, Catch::Matchers::Floating::WithinAbsMatcher>{
        target, Catch::WithinAbs(target, margin)};
  };
}

template <typename T> auto RelValidatorBuilderFactory(T margin) {
  return [=](T target) {
    return ValidatorBase<T, Catch::Matchers::Floating::WithinRelMatcher>{
        target, Catch::WithinRel(target, margin)};
  };
}

template <typename T> struct RefType {};

template <> struct RefType<float> { using type = double; };

template <> struct RefType<double> { using type = long double; };

template <typename T> using RefType_t = typename RefType<T>::type;
