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

#define NEGATE_ASSIGN_VECTOR_FUNCTIONS(type)                                                       \
  __global__ void CalculateAssignDevice(type* vector_dev_ptr, typename type::value_type value) {   \
    type vector_dev = *vector_dev_ptr;                                                             \
    vector_dev %= value;                                                                           \
    vector_dev ^= value;                                                                           \
    vector_dev |= value;                                                                           \
    vector_dev &= value;                                                                           \
    vector_dev >>= value;                                                                          \
    vector_dev <<= value;                                                                          \
  }                                                                                                \
  void CalculateAssignHost(type& vector_host, typename type::value_type value) {                   \
    vector_host %= value;                                                                          \
    vector_host ^= value;                                                                          \
    vector_host |= value;                                                                          \
    vector_host &= value;                                                                          \
    vector_host >>= value;                                                                         \
    vector_host <<= value;                                                                         \
  }

NEGATE_ASSIGN_VECTOR_FUNCTIONS(uchar2)
NEGATE_ASSIGN_VECTOR_FUNCTIONS(ushort2)
NEGATE_ASSIGN_VECTOR_FUNCTIONS(uint2)
NEGATE_ASSIGN_VECTOR_FUNCTIONS(ulong2)
NEGATE_ASSIGN_VECTOR_FUNCTIONS(ulonglong2)
