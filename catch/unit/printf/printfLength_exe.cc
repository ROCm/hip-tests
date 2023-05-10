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

#include <hip/hip_runtime.h>
#include <cwchar>

__global__ void test_kernel() {
  const char* N = nullptr;
  printf("%hhd %hhi\n", char(-42), char(-42));
  printf("%hd %hi\n", short(-42), short(-42));
  printf("%ld %li\n", -42l, -42l);
  printf("%lld %lli\n", -42ll, -42ll);
  printf("%jd %ji\n", -42l, -42l);
  printf("%zd %zi\n", -42l, -42l);
  printf("%td %ti\n", (ptrdiff_t)N, (ptrdiff_t)N);
  printf("%hhu %hho\n", uchar(42), uchar(42));
  printf("%hu %ho\n", ushort(42), ushort(42));
  printf("%lu %lo\n", 42l, 42l);
  printf("%llu %llo\n", 42ll, 42ll);
  printf("%ju %jo\n", 42l, 42l);
  printf("%zu %zo\n", 42l, 42l);
  printf("%tu %to\n", (ptrdiff_t)N, (ptrdiff_t)N);
  printf("%lc\n", 'x');
}

int main() {
  test_kernel<<<1, 1>>>();
  static_cast<void>(hipDeviceSynchronize());
}