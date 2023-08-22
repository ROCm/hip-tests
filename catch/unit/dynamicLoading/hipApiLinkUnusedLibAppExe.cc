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
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <iostream>
#include <vector>
#include <random>

#include <hip/hip_runtime.h>

constexpr int arSize = 1024;
constexpr int arBytes = arSize * sizeof(int);

__global__ void kerSquare(int *inout_a) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  inout_a[id] = inout_a[id] * inout_a[id];
}

int main() {
  int testPassed = 0;  // 0 = passed, -1 = failed
  int *a_h = new int[arSize];
  if (a_h == nullptr) {
    testPassed = -1;
  }
  // Inititialize data
  for (size_t i = 0; i < arSize; i++) {
    a_h[i] = i;
  }
  int *a_d;
  if (hipSuccess != hipMalloc(&a_d, arBytes)) {
    testPassed = -1;
  }
  constexpr int blksize = 128;
  constexpr int grdsize = arSize / blksize;
  if (hipSuccess != hipMemcpy(a_d, a_h, arSize, hipMemcpyHostToDevice)) {
    testPassed = -1;
  }
  kerSquare<<<grdsize, blksize>>>(a_d);
  // kernel invocation should return error hipErrorSharedObjectInitFailed
  if (hipErrorSharedObjectInitFailed != hipGetLastError()) {
    testPassed = -1;
  }
  if (hipSuccess != hipFree(a_d)) {
    testPassed = -1;
  }
  delete[] a_h;
  // std::cout prints the results back to output stream for parent
  // process to read.
  std::cout << testPassed << std::endl;
  return testPassed;
}
