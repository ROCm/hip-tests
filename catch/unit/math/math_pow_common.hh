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

#include <hip_test_common.hh>
#include <resource_guards.hh>

#define MATH_POW_DOUBLE_INT_ARG_KERNEL_DEF(func_name)                                              \
  template <typename T, typename I>                                                                \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s,               \
                                     I* const x2s) {                                               \
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

#define MATH_POW_FREXP_ARG_KERNEL_DEF(func_name)                                                   \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s,               \
                                     int* const x2s) {                                             \
    const auto tid = cg::this_grid().thread_rank();                                                \
                                                                                                   \
    if (tid < num_xs) {                                                                            \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[tid] = func_name##f(x1s[tid], &x2s[tid]);                                               \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[tid] = func_name(x1s[tid], &x2s[tid]);                                                  \
      }                                                                                            \
    }                                                                                              \
  }

template <typename T, typename I, typename RT, typename RI, typename Validator>
void PowIntTest(Validator validator, const size_t grid_dim, const size_t block_dim,
              void (*const kernel)(T*, const size_t, T*, I*), RT (*const ref_func)(RT, RI),
              const size_t num_args, const T* x1s, const I* x2s) {
  LinearAllocGuard<T> x1s_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  HIP_CHECK(hipMemcpy(x1s_dev.ptr(), x1s, num_args * sizeof(T), hipMemcpyHostToDevice));
  LinearAllocGuard<I> x2s_dev{LinearAllocs::hipMalloc, num_args * sizeof(I)};
  HIP_CHECK(hipMemcpy(x2s_dev.ptr(), x2s, num_args * sizeof(I), hipMemcpyHostToDevice));

  LinearAllocGuard<T> y_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  kernel<<<grid_dim, block_dim>>>(y_dev.ptr(), num_args, x1s_dev.ptr(), x2s_dev.ptr());
  HIP_CHECK(hipGetLastError());

  std::vector<T> y(num_args);
  HIP_CHECK(hipMemcpy(y.data(), y_dev.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));

  for (auto i = 0u; i < num_args; ++i) {
    validator(y[i], static_cast<T>(ref_func(static_cast<RT>(x1s[i]), static_cast<RI>(x2s[i]))));
  }
}

template <typename T, typename RT>
void PowFrexpTest(const int64_t ulps, const size_t grid_dim, const size_t block_dim,
              void (*const kernel)(T*, const size_t, T*, int*), RT (*const ref_func)(RT, int*),
              const size_t num_args, const T* x1s, int* x2s) {
  LinearAllocGuard<T> x1s_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  HIP_CHECK(hipMemcpy(x1s_dev.ptr(), x1s, num_args * sizeof(T), hipMemcpyHostToDevice));
  LinearAllocGuard<int> x2s_dev{LinearAllocs::hipMalloc, num_args * sizeof(int)};

  LinearAllocGuard<T> y_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  kernel<<<grid_dim, block_dim>>>(y_dev.ptr(), num_args, x1s_dev.ptr(), x2s_dev.ptr());
  HIP_CHECK(hipGetLastError());

  std::vector<T> y1(num_args);
  std::vector<int> y2(num_args);
  HIP_CHECK(hipMemcpy(y1.data(), y_dev.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(y2.data(), x2s_dev.ptr(), num_args * sizeof(int), hipMemcpyDeviceToHost));

  for (auto i = 0u; i < num_args; ++i) {
    T ref_val = static_cast<T>(ref_func(static_cast<RT>(x1s[i]), &x2s[i]));
    if (std::isnan(ref_val)) {
      REQUIRE(std::isnan(y1[i]));
    } else if (std::isinf(ref_val)) {
      REQUIRE(std::isinf(y1[i]));
    } else {
      REQUIRE_THAT(y1[i], Catch::WithinULP(ref_val, ulps));
      REQUIRE(y2[i] == x2s[i]);
    }
  }
}