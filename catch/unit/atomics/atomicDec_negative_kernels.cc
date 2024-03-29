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

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

/* unsigned int atomicDec(unsigned int* address, unsigned int val) */
__global__ void atomicDec_uint_v1(unsigned int* address, unsigned int* result) {
  *result = atomicDec(&address, 1234);
}

__global__ void atomicDec_uint_v2(unsigned int* address, unsigned int* result) {
  *result = atomicDec(address, address);
}

__global__ void atomicDec_uint_v3(unsigned int* address, unsigned int* result) {
  *result = atomicDec(1234, 1234);
}

__global__ void atomicDec_uint_v4(Dummy* address, unsigned int* result) {
  *result = atomicDec(address, 1234);
}

__global__ void atomicDec_uint_v5(char* address, unsigned int* result) {
  *result = atomicDec(address, 1234);
}

__global__ void atomicDec_uint_v6(short* address, unsigned int* result) {
  *result = atomicDec(address, 1234);
}

__global__ void atomicDec_uint_v7(long* address, unsigned int* result) {
  *result = atomicDec(address, 1234);
}

__global__ void atomicDec_uint_v8(long long* address, unsigned int* result) {
  *result = atomicDec(address, 1234);
}