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
#define HEX_FLT(sm, int, fract, se, exp)                                       \
    sm ldexpf((float)(0x##int##fract##UL),                                     \
              se exp + ilogbf((float)0x##int)                                  \
                  - ilogbf((float)(0x##int##fract##UL)))
#define HEX_DBL(sm, int, fract, se, exp)                                       \
    sm ldexp((double)(0x##int##fract##ULL),                                    \
             se exp + ilogb((double)0x##int)                                   \
                 - ilogb((double)(0x##int##fract##ULL)))
#define HEX_LDBL(sm, int, fract, se, exp)                                      \
    sm ldexpl((long double)(0x##int##fract##ULL),                              \
              se exp + ilogbl((long double)0x##int)                            \
                  - ilogbl((long double)(0x##int##fract##ULL)))
#else
// If compiler supports hex floating point literals: just concatenate all the
// parts into a literal.
#define HEX_FLT(sm, int, fract, se, exp) sm 0x##int##.##fract##p##se##exp##F
#define HEX_DBL(sm, int, fract, se, exp) sm 0x##int##.##fract##p##se##exp
#define HEX_LDBL(sm, int, fract, se, exp) sm 0x##int##.##fract##p##se##exp##L
#endif

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
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s,               \
                                     T* const x2s, T* const x3s) {                                               \
    const auto tid = cg::this_grid().thread_rank();                                                \
                                                                                                   \
    if (tid < num_xs) {                                                                            \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[tid] = func_name##f(x1s[tid], x2s[tid], x3s[tid]);                                                \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[tid] = func_name(x1s[tid], x2s[tid], x3s[tid]);                                                   \
      }                                                                                            \
    }                                                                                              \
  }

#define MATH_QUADRUPLE_ARG_KERNEL_DEF(func_name)                                                      \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s,               \
                                     T* const x2s, T* const x3s, T* const x4s) {                                               \
    const auto tid = cg::this_grid().thread_rank();                                                \
                                                                                                   \
    if (tid < num_xs) {                                                                            \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[tid] = func_name##f(x1s[tid], x2s[tid], x3s[tid], x4s[tid]);                                                \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[tid] = func_name(x1s[tid], x2s[tid], x3s[tid], x4s[tid]);                                                   \
      }                                                                                            \
    }                                                                                              \
  }

// Extern declaration of SpecialValues<Double/Float> arrays defined in math_special_values
extern const double specialValuesDouble[];
extern const size_t specialValuesDoubleCount;
extern const float specialValuesFloat[];
extern const size_t specialValuesFloatCount;

template <typename T, typename RT, typename Validator, typename... Ts, typename... RTs, size_t... I>
void MathTestImpl(Validator validator, const size_t grid_dim, const size_t block_dim,
                  void (*const kernel)(T*, const size_t, Ts*...), RT (*const ref_func)(RTs...),
                  const size_t num_args, std::index_sequence<I...> is, Ts*... xss) {
  struct LAWrapper {
    LAWrapper(const size_t size, T* const init_vals) : la_{LinearAllocs::hipMalloc, size, 0u} {
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

  for (auto i = 0u; i < num_args; ++i) {
    validator(y[i], static_cast<T>(ref_func(static_cast<RT>(xss[i])...)));
  }
}

template <typename T, typename RT, typename Validator, typename... Ts, typename... RTs>
void MathTest(Validator validator, const size_t grid_dims, const size_t block_dims,
              void (*const kernel)(T*, const size_t, Ts*...), RT (*const ref_func)(RTs...),
              const size_t num_args, Ts*... xss) {
  MathTestImpl(validator, grid_dims, block_dims, kernel, ref_func, num_args,
               std::index_sequence_for<Ts...>{}, xss...);
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
