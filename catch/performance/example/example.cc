/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <performance_common.hh>
#include <resource_guards.hh>

TEST_CASE("Performance_Example_Event") {
  LinearAllocGuard<void> dst(LinearAllocs::hipMalloc, 4_MB);

  auto user_code = [](void* dst) { HIP_CHECK(hipMemset(dst, 42, 4_MB)); };
  EventBenchmark benchmark(std::move(user_code));

  benchmark.Configure(1000, 100);  // 1000 iterations, 100 warmup iterations
  std::cout << benchmark.Run(dst.ptr()) << " ms" << std::endl;
}

TEST_CASE("Performance_Example_Cpu") {
  LinearAllocGuard<void> dst(LinearAllocs::hipMalloc, 4_MB);

  auto user_code = [](void* dst) { HIP_CHECK(hipMemset(dst, 42, 4_MB)); };
  CpuBenchmark benchmark(std::move(user_code));

  benchmark.Configure(1000, 100);  // 1000 iterations, 100 warmup iterations
  std::cout << benchmark.Run(dst.ptr()) << " ms" << std::endl;
}