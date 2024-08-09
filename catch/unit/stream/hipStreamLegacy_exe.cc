/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip/hip_runtime.h"

const int N = 2 * 1024 * 1024;
const size_t Nbytes = N * sizeof(int);

#define HIP_CHECK(error)                                   \
{                                                          \
  hipError_t localError = error;                           \
  if ((localError != hipSuccess) &&                        \
      (localError != hipErrorPeerAccessAlreadyEnabled)) {  \
        printf("error: '%s'(%d) from %s at %s:%d\n",       \
               hipGetErrorString(localError),              \
               localError, #error, __FUNCTION__, __LINE__);\
  exit(0);                                                 \
  }                                                        \
}

/**
 * Used hipStreamLegacy in child process and copies H2H, H2D, D2D, D2H
 */
int main() {
  int elementVal = 3456;

  int *hostArr1 = reinterpret_cast<int *>(malloc(Nbytes));
  if (hostArr1 == nullptr) {
    return -1;
  }

  for ( int i = 0; i < N; i++ ) {
    hostArr1[i] = elementVal;
  }

  int *hostArr2 = reinterpret_cast<int *>(malloc(Nbytes));
  if (hostArr2 == nullptr) {
    return -1;
  }

  int *devArr1 = nullptr;
  HIP_CHECK(hipMalloc(&devArr1, Nbytes));
  if (devArr1 == nullptr) {
    return -1;
  }

  int *devArr2 = nullptr;
  HIP_CHECK(hipMalloc(&devArr2, Nbytes));
  if (devArr2 == nullptr) {
    return -1;
  }

  int *hostArr3 = reinterpret_cast<int *>(malloc(Nbytes));
  if (hostArr3 == nullptr) {
    return -1;
  }

  HIP_CHECK(hipMemcpyAsync(hostArr2, hostArr1, Nbytes,
                           hipMemcpyHostToHost, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(devArr1, hostArr2, Nbytes,
                           hipMemcpyHostToDevice, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(devArr2, devArr1, Nbytes,
                           hipMemcpyDeviceToDevice, hipStreamLegacy));
  HIP_CHECK(hipMemcpyAsync(hostArr3, devArr2, Nbytes,
                           hipMemcpyDeviceToHost, hipStreamLegacy));

  for (int i = 0; i < N; i++) {
    if (hostArr3[i] != elementVal) {
      return -1;
    }
  }

  free(hostArr1);
  free(hostArr2);
  HIP_CHECK(hipFree(devArr1));
  HIP_CHECK(hipFree(devArr2));
  free(hostArr3);

  return 0;
}
