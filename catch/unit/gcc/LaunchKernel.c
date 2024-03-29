
/* Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <stdio.h>
#include <hip/hip_runtime.h>
#include "LaunchKernel.h"
#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"

#define HIPCHECK(error)                                                                      \
  {                                                                                          \
    hipError_t localError = error;                                                           \
    if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {    \
      printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, hipGetErrorString(localError),  \
               localError, #error, __FILE__, __LINE__, KNRM);                                \
      printf("API returned error code.\n");                                                  \
    }                                                                                        \
  }

bool LaunchKernelArg()
{
  dim3 blocks 	    = {1,1,1};
  dim3 threads      = {1,1,1};

  HIPCHECK(hipLaunchKernel(getKernelFunc(mykernel), blocks, threads, NULL, 0, 0));

  return true;
}

bool LaunchKernelArg1()
{
  int A = 0;
  int *A_d = NULL;
  dim3 blocks       = {1,1,1};
  dim3 threads      = {1,1,1};

  // Allocate Device memory
  HIPCHECK(hipMalloc((void**)&A_d, sizeof(int)));

  void* Args[]={&A_d};

  HIPCHECK(hipLaunchKernel(getKernelFunc(mykernel1), blocks, threads, Args, 0, 0));

  // Get the result back to host memory
  HIPCHECK(hipMemcpy(&A, A_d, sizeof(int), hipMemcpyDeviceToHost));

  HIPCHECK(hipFree(A_d));

  if(A != 333)
	  return false;

  return true;
}

bool LaunchKernelArg2()
{
  int A = 0;
  int B = 123;
  int *A_d = NULL;
  int *B_d = NULL;

  dim3 blocks       = {1,1,1};
  dim3 threads      = {1,1,1};

  // Allocate Device memory
  HIPCHECK(hipMalloc((void**)&A_d, sizeof(int)));

  HIPCHECK(hipMalloc((void**)&B_d, sizeof(int)));

  // Copy data from host memory to device memory
  HIPCHECK(hipMemcpy(B_d, &B, sizeof(int), hipMemcpyHostToDevice));

  void* Args[]={&A_d, &B_d};
  HIPCHECK(hipLaunchKernel(getKernelFunc(mykernel2), blocks, threads, Args,0,0));

  // Get the result back to host memory
  HIPCHECK(hipMemcpy(&A, A_d, sizeof(int), hipMemcpyDeviceToHost));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(B_d));

  if(A != 123)
    return false;

  return true;
}

bool LaunchKernelArg3()
{
  int A = 321;
  int B = 123;
  int C = 0;
  int *A_d = NULL;
  int *B_d = NULL;
  int *C_d = NULL;

  dim3 blocks       = {1,1,1};
  dim3 threads      = {1,1,1};

  // Allocate Device memory
  HIPCHECK(hipMalloc((void**)&A_d, sizeof(int)));

  HIPCHECK(hipMalloc((void**)&B_d, sizeof(int)));

  HIPCHECK(hipMalloc((void**)&C_d, sizeof(int)));

  // Copy data from host memory to device memory
  HIPCHECK(hipMemcpy(A_d, &A, sizeof(int), hipMemcpyHostToDevice));

  HIPCHECK(hipMemcpy(B_d, &B, sizeof(int), hipMemcpyHostToDevice));

  void* Args[]={&A_d, &B_d, &C_d};
  HIPCHECK(hipLaunchKernel(getKernelFunc(mykernel3), blocks, threads, Args,0,0));

  // Get the result back to host memory
  HIPCHECK(hipMemcpy(&C, C_d, sizeof(int), hipMemcpyDeviceToHost));

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(B_d));
  HIPCHECK(hipFree(C_d));

  if(C != 444)
    return false;

  return true;
}

bool LaunchKernelArg4()
{
  int A = 0;
  int *A_d = NULL;
  dim3 blocks       = {1,1,1};
  dim3 threads      = {1,1,1};

  // Allocate Device memory
  HIPCHECK(hipMalloc((void**)&A_d, sizeof(int)));

  char c = 1;
  short s = 10;
  int i = 100;
  struct things t = {2,20,200};

  void* Args[]={&A_d, &c, &s, &i, &t};
  HIPCHECK(hipLaunchKernel(getKernelFunc(mykernel4), blocks, threads, Args, 0, 0));

  // Get the result back to host memory
  HIPCHECK(hipMemcpy(&A, A_d, sizeof(int), hipMemcpyDeviceToHost));

  HIPCHECK(hipFree(A_d));

  if (A != (c + s + i + t.c + t.s + t.i))
	  return false;

  return true;
}

#ifdef __cplusplus
extern "C"
{
#endif
int launchKernel()
{
  printf("Calling LaunchKernel.c file\n");
  if( LaunchKernelArg()  &&
      LaunchKernelArg1() &&
      LaunchKernelArg2() &&
      LaunchKernelArg3() &&
      LaunchKernelArg4())
    {
      printf("PASSED!\n");
      return 1;
    }
  else
    printf("FAILED\n");
    return 0;
}
#ifdef __cplusplus
}
#endif
