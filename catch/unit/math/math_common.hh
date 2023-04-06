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

#include <hip_test_common.hh>
#include <resource_guards.hh>

#include <hip/hip_cooperative_groups.h>

#include "thread_pool.hh"

namespace cg = cooperative_groups;

#define MATH_TERNARY_KERNEL_DEF(func_name)                                                         \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s, T* const x2s, \
                                     T* const x3s) {                                               \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[i] = func_name##f(x1s[i], x2s[i], x3s[i]);                                              \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[i] = func_name(x1s[i], x2s[i], x3s[i]);                                                 \
      }                                                                                            \
    }                                                                                              \
  }

#define MATH_QUATERNARY_KERNEL_DEF(func_name)                                                      \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s, T* const x2s, \
                                     T* const x3s, T* const x4s) {                                 \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[i] = func_name##f(x1s[i], x2s[i], x3s[i], x4s[i]);                                      \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[i] = func_name(x1s[i], x2s[i], x3s[i], x4s[i]);                                         \
      }                                                                                            \
    }                                                                                              \
  }


template <typename T, typename RT, size_t N> class MathTest {
 public:
  MathTest(const size_t max_num_args)
      : xss_dev_(CreateArray(max_num_args * sizeof(T))),
        y_dev_{LinearAllocs::hipMalloc, max_num_args * sizeof(T)},
        y_{LinearAllocs::hipHostMalloc, max_num_args * sizeof(T)} {}


  template <bool parallel = true, typename ValidatorBuilder, typename... Ts, typename... RTs>
  void Run(const ValidatorBuilder& validator_builder, const size_t grid_dims,
           const size_t block_dims, void (*const kernel)(T*, const size_t, Ts*...),
           RT (*const ref_func)(RTs...), const size_t num_args, const Ts*... xss) {
    fail_flag_.store(false);
    error_info_.clear();
    RunImpl<parallel>(validator_builder, grid_dims, block_dims, kernel, ref_func, num_args,
                      std::index_sequence_for<Ts...>{}, xss...);
  }

 private:
  std::array<LinearAllocGuard<T>, N> xss_dev_;
  LinearAllocGuard<T> y_dev_;
  LinearAllocGuard<T> y_;
  std::atomic<bool> fail_flag_{false};
  std::mutex mtx_;
  std::string error_info_;

  template <bool parallel, typename ValidatorBuilder, typename... Ts, typename... RTs, size_t... I>
  void RunImpl(const ValidatorBuilder& validator_builder, const size_t grid_dim,
               const size_t block_dim, void (*const kernel)(T*, const size_t, Ts*...),
               RT (*const ref_func)(RTs...), const size_t num_args, std::index_sequence<I...> is,
               const Ts*... xss) {
    const std::array<const T*, N> xss_arr{xss...};

    auto f = [&, this](int i) {
      HIP_CHECK(
          hipMemcpy(xss_dev_[i].ptr(), xss_arr[i], num_args * sizeof(T), hipMemcpyHostToDevice));
    };

    ((f(I)), ...);

    kernel<<<grid_dim, block_dim>>>(y_dev_.ptr(), num_args, xss_dev_[I].ptr()...);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(y_.ptr(), y_dev_.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipStreamSynchronize(nullptr));

    if constexpr (!parallel) {
      for (auto i = 0u; i < num_args; ++i) {
        const auto actual_val = y_.ptr()[i];
        const auto ref_val = static_cast<T>(ref_func(static_cast<RT>(xss[i])...));
        const auto validator = validator_builder(ref_val);

        if (!validator.match(actual_val)) {
          const auto log = MakeLogMessage(actual_val, xss[i]...) + validator.describe() + "\n";
          INFO(log);
          REQUIRE(false);
        }
      }

      return;
    }

    const auto task = [&, this](size_t iters, size_t base_idx) {
      for (auto i = 0u; i < iters; ++i) {
        if (fail_flag_.load(std::memory_order_relaxed)) return;

        const auto actual_val = y_.ptr()[base_idx + i];
        const auto ref_val = static_cast<T>(ref_func(static_cast<RT>(xss[base_idx + i])...));
        const auto validator = validator_builder(ref_val);

        if (!validator.match(actual_val)) {
          fail_flag_.store(true, std::memory_order_relaxed);
          // Several threads might have passed the first check, but failed validation. On the chance
          // of this happening, access to the string stream must be serialized.
          const auto log =
              MakeLogMessage(actual_val, xss[base_idx + i]...) + validator.describe() + "\n";
          {
            std::lock_guard lg{mtx_};
            error_info_ += log;
          }
          return;
        }
      }
    };

    const auto task_count = thread_pool.thread_count();
    const auto chunk_size = num_args / task_count;
    const auto tail = num_args % task_count;

    auto base_idx = 0u;
    for (auto i = 0u; i < task_count; ++i) {
      const auto iters = chunk_size + (i < tail);
      thread_pool.Post([=, &task] { task(iters, base_idx); });
      base_idx += iters;
    }

    thread_pool.Wait();

    INFO(error_info_);
    REQUIRE(!fail_flag_);
  }

  template <typename... Args> std::string MakeLogMessage(T actual_val, Args... args) {
    std::stringstream ss;
    ss << "Input value(s): " << std::scientific
       << std::setprecision(std::numeric_limits<T>::max_digits10 - 1);
    ((ss << " " << args), ...) << "\n" << actual_val << " ";

    return ss.str();
  }

  template <std::size_t... Is>
  constexpr std::array<LinearAllocGuard<T>, N> CreateArrayImpl(std::size_t size,
                                                               std::index_sequence<Is...>) {
    return {{(static_cast<void>(Is), LinearAllocGuard<T>{LinearAllocs::hipMalloc, size})...}};
  }

  constexpr std::array<LinearAllocGuard<T>, N> CreateArray(std::size_t size) {
    return CreateArrayImpl(size, std::make_index_sequence<N>());
  }
};

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

template <typename T> auto ULPValidatorBuilderFactory(int64_t ulps) {
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

template <typename F> auto GetOccupancyMaxPotentialBlockSize(F kernel) {
  int grid_size = 0, block_size = 0;
  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, 0, 0));
  return std::make_tuple(grid_size, block_size);
}

inline size_t GetMaxAllowedDeviceMemoryUsage() {
  // TODO - Add setting of allowed memory from the command line
  // If the cmd option is set, return that, otherwise return 80% of available
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  return props.totalGlobalMem * 0.8;
}

inline uint64_t GetTestIterationCount() {
  // TODO - Add setting of iteration count from the command line
  return std::numeric_limits<uint32_t>::max() + 1ul;
}
