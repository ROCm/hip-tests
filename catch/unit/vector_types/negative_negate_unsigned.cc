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

#define NEGATE_UNSIGNED_VECTOR_FUNCTIONS(type)                                                     \
  __global__ void NegateDevice(type* vector_dev_ptr) {                                             \
    type vector_dev = *vector_dev_ptr;                                                             \
    vector_dev = -vector_dev;                                                                      \
  }                                                                                                \
  void NegateHost(type& vector_host) { vector_host = -vector_host; }

NEGATE_UNSIGNED_VECTOR_FUNCTIONS(uchar1)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(uchar2)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(uchar3)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(uchar4)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ushort1)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ushort2)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ushort3)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ushort4)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(uint1)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(uint2)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(uint3)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(uint4)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ulong1)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ulong2)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ulong3)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ulong4)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ulonglong1)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ulonglong2)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ulonglong3)
NEGATE_UNSIGNED_VECTOR_FUNCTIONS(ulonglong4)
