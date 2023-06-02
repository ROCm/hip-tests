/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define NEGATIVE_KERNELS_SHELL_ONE_ARG(T1, T2, func_name)                                          \
  __global__ void func_name##_kernel_v1(T1* result, T2* x) { *result = __##func_name(x); }         \
  __global__ void func_name##_kernel_v2(T1* result, Dummy x) { *result = __##func_name(x); }       \
  __global__ void func_name##_kernel_v3(Dummy* result, T2 x) { *result = __##func_name(x); }

#define NEGATIVE_KERNELS_SHELL_TWO_ARGS(T1, T2, func_name)                                         \
  __global__ void func_name##_kernel_v1(T1* result, T2* x, T2 y) {                                 \
    *result = __##func_name(x, y);                                                                 \
  }                                                                                                \
  __global__ void func_name##_kernel_v2(T1* result, T2 x, T2* y) {                                 \
    *result = __##func_name(x, y);                                                                 \
  }                                                                                                \
  __global__ void func_name##_kernel_v3(T1* result, Dummy x, T2 y) {                               \
    *result = __##func_name(x, y);                                                                 \
  }                                                                                                \
  __global__ void func_name##_kernel_v4(T1* result, T2 x, Dummy y) {                               \
    *result = __##func_name(x, y);                                                                 \
  }                                                                                                \
  __global__ void func_name##_kernel_v5(Dummy* result, T2 x, T2 y) {                               \
    *result = __##func_name(x, y);                                                                 \
  }

NEGATIVE_KERNELS_SHELL_ONE_ARG(float, int, int2float_rd)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, int, int2float_rn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, int, int2float_ru)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, int, int2float_rz)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, unsigned int, uint2float_rd)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, unsigned int, uint2float_rn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, unsigned int, uint2float_ru)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, unsigned int, uint2float_rz)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, long long int, ll2float_rd)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, long long int, ll2float_rn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, long long int, ll2float_ru)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, long long int, ll2float_rz)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, unsigned long long int, ull2float_rd)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, unsigned long long int, ull2float_rn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, unsigned long long int, ull2float_ru)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, unsigned long long int, ull2float_rz)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, int, int2double_rn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, unsigned int, uint2double_rn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, long long int, ll2double_rd)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, long long int, ll2double_rn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, long long int, ll2double_ru)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, long long int, ll2double_rz)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, unsigned long long int, ull2double_rd)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, unsigned long long int, ull2double_rn)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, unsigned long long int, ull2double_ru)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, unsigned long long int, ull2double_rz)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, int, int_as_float)
NEGATIVE_KERNELS_SHELL_ONE_ARG(float, unsigned int, uint_as_float)
NEGATIVE_KERNELS_SHELL_ONE_ARG(double, long long int, longlong_as_double)
NEGATIVE_KERNELS_SHELL_TWO_ARGS(double, int, hiloint2double)