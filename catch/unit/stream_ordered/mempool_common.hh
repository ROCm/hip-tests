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
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */
#pragma once

#include <hip_test_common.hh>

namespace {
constexpr hipMemPoolProps kPoolProps = {
    hipMemAllocationTypePinned, hipMemHandleTypeNone, {hipMemLocationTypeDevice, 0}, nullptr, {0}};

constexpr auto wait_ms = 500;
}  // anonymous namespace


template <typename T>
__global__ void kernel_500ms(T* host_res, int clk_rate) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  host_res[tid] = tid + 1;
  __threadfence_system();
  // expecting that the data is getting flushed to host here!
  uint64_t start = clock64()/clk_rate, cur;
  if (clk_rate > 1) {
    do { cur = clock64()/clk_rate-start;}while (cur < wait_ms);
  } else {
    do { cur = clock64()/start;}while (cur < wait_ms);
  }
}

template <typename T>
__global__ void kernel_500ms_gfx11(T* host_res, int clk_rate) {
#if HT_AMD
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  host_res[tid] = tid + 1;
  __threadfence_system();
  // expecting that the data is getting flushed to host here!
  uint64_t start = wall_clock64()/clk_rate, cur;
  if (clk_rate > 1) {
    do { cur = wall_clock64()/clk_rate-start;}while (cur < wait_ms);
  } else {
    do { cur = wall_clock64()/start;}while (cur < wait_ms);
  }
#endif
}
