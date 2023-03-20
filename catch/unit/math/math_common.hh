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

template <typename T, typename RT, typename V>
void MathTest(void (*kernel)(T*, size_t, T*), RT (*ref_func)(RT), size_t num_args, T* xs,
              V validator, size_t grid_dim, size_t block_dim) {
  LinearAllocGuard<T> xs_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  HIP_CHECK(hipMemcpy(xs_dev.ptr(), xs, num_args * sizeof(T), hipMemcpyHostToDevice));

  LinearAllocGuard<T> ys_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  kernel<<<grid_dim, block_dim>>>(ys_dev.ptr(), num_args, xs_dev.ptr());
  HIP_CHECK(hipGetLastError());

  std::vector<T> ys(num_args);
  HIP_CHECK(hipMemcpy(ys.data(), ys_dev.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));

  for (auto i = 0u; i < num_args; ++i) {
    validator(ys[i], static_cast<T>(ref_func(static_cast<RT>(xs[i]))));
  }
}

template <typename T, typename RT, typename V>
void MathTest(void (*kernel)(T*, size_t, T*, T*), RT (*ref_func)(RT, RT), size_t num_args, T* x1s,
              T* x2s, V validator, size_t grid_dim, size_t block_dim) {
  LinearAllocGuard<T> x1s_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  HIP_CHECK(hipMemcpy(x1s_dev.ptr(), x1s, num_args * sizeof(T), hipMemcpyHostToDevice));
  LinearAllocGuard<T> x2s_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  HIP_CHECK(hipMemcpy(x2s_dev.ptr(), x2s, num_args * sizeof(T), hipMemcpyHostToDevice));

  LinearAllocGuard<T> ys_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  kernel<<<grid_dim, block_dim>>>(ys_dev.ptr(), num_args, x1s_dev.ptr(), x2s_dev.ptr());
  HIP_CHECK(hipGetLastError());

  std::vector<T> ys(num_args);
  HIP_CHECK(hipMemcpy(ys.data(), ys_dev.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));

  for (auto i = 0u; i < num_args; ++i) {
    validator(ys[i], static_cast<T>(ref_func(static_cast<RT>(x1s[i]), static_cast<RT>(x2s[i]))));
  }
}


struct ULPValidator {
  template <typename T> void operator()(const T actual_val, const T ref_val) const {
    REQUIRE_THAT(actual_val, Catch::WithinULP(ref_val, ulps));
  }

  const int64_t ulps;
};

struct AbsValidator {
  template <typename T> void operator()(const T actual_val, const T ref_val) const {
    REQUIRE_THAT(actual_val, Catch::WithinAbs(ref_val, margin));
  }

  const double margin;
};

template <typename T> struct RelValidator {
  void operator()(const T actual_val, const T ref_val) const {
    REQUIRE_THAT(actual_val, Catch::WithinRel(ref_val, margin));
  }

  const T margin;
};

struct EqValidator {
  template <typename T> void operator()(const T actual_val, const T ref_val) const {
    REQUIRE(actual_val == ref_val);
  }
};