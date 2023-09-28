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

#include <hip_test_common.hh>
#include <hip/hip_fp16.h>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define NEGATIVE_KERNELS_SHELL(T, func_name)                                                       \
  __global__ void func_name##_kernel_v1(T* result, __half* x) { *result = __##func_name(x); }      \
  __global__ void func_name##_kernel_v2(T* result, Dummy x) { *result = __##func_name(x); }        \
  __global__ void func_name##_kernel_v3(Dummy* result, __half x) { *result = __##func_name(x); }

NEGATIVE_KERNELS_SHELL(int, half2int_rn)
NEGATIVE_KERNELS_SHELL(int, half2int_rz)
NEGATIVE_KERNELS_SHELL(int, half2int_rd)
NEGATIVE_KERNELS_SHELL(int, half2int_ru)
NEGATIVE_KERNELS_SHELL(unsigned int, half2uint_rn)
NEGATIVE_KERNELS_SHELL(unsigned int, half2uint_rz)
NEGATIVE_KERNELS_SHELL(unsigned int, half2uint_rd)
NEGATIVE_KERNELS_SHELL(unsigned int, half2uint_ru)
NEGATIVE_KERNELS_SHELL(short, half2short_rn)
NEGATIVE_KERNELS_SHELL(short, half2short_rz)
NEGATIVE_KERNELS_SHELL(short, half2short_rd)
NEGATIVE_KERNELS_SHELL(short, half2short_ru)
NEGATIVE_KERNELS_SHELL(short, half_as_short)
NEGATIVE_KERNELS_SHELL(unsigned short, half2ushort_rn)
NEGATIVE_KERNELS_SHELL(unsigned short, half2ushort_rz)
NEGATIVE_KERNELS_SHELL(unsigned short, half2ushort_rd)
NEGATIVE_KERNELS_SHELL(unsigned short, half2ushort_ru)
NEGATIVE_KERNELS_SHELL(unsigned short, half_as_ushort)
NEGATIVE_KERNELS_SHELL(long long, half2ll_rn)
NEGATIVE_KERNELS_SHELL(long long, half2ll_rz)
NEGATIVE_KERNELS_SHELL(long long, half2ll_rd)
NEGATIVE_KERNELS_SHELL(long long, half2ll_ru)
NEGATIVE_KERNELS_SHELL(unsigned long long, half2ull_rn)
NEGATIVE_KERNELS_SHELL(unsigned long long, half2ull_rz)
NEGATIVE_KERNELS_SHELL(unsigned long long, half2ull_rd)
NEGATIVE_KERNELS_SHELL(unsigned long long, half2ull_ru)