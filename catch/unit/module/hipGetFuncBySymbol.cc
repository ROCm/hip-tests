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
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

#define LEN 64
#define SIZE LEN * sizeof(float)

extern "C" __global__ void bit_extract_kernel(uint32_t* C_d, const uint32_t*
                                              A_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < N; i += stride) {
#if HT_AMD
    C_d[i] = __bitextract_u32(A_d[i], 8, 4);
#else  /* defined __HIP_PLATFORM_NVIDIA__ or other path */
    C_d[i] = ((A_d[i] & 0xf00) >> 8);
#endif
  }
}

/**
* @addtogroup hipGetFuncBySymbol hipModuleLaunchKernel
* @{
* @ingroup KernelTest
* `hipError_t hipGetFuncBySymbol(hipFunction_t*, const void*)` -
* function with kernelname will be fetched when pointer to the kernel function is passed.
* `hipError_t hipModuleLaunchKernel(hipFunction_t, unsigned int,
*             unsigned int, unsigned int, unsigned int, unsigned int,
*             unsigned int, unsigned int, hipStream_t, void**, void**)` -
* launches Kernel with launch parameters and shared memory on stream with arguments passed.
*/

/**
 * Test Description
 * ------------------------
 * - Test is to get function ptr (hipFunction_t) using hipGetFuncBySymbol and launch
 *   bit_extract kernel. Verify the output.

 * Test source
 * ------------------------
 * - catch/unit/module/hipGetFuncBySymbol.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */

TEST_CASE("Unit_hipGetFuncBySymbol") {
  uint32_t *A_d, *C_d;
  uint32_t *A_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(uint32_t);

  hipDevice_t device;
  HIPCHECK(hipGetDevice(&device));

  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, device));
  A_h = reinterpret_cast<uint32_t*>(malloc(Nbytes));
  REQUIRE(A_h != NULL);
  C_h = reinterpret_cast<uint32_t*>(malloc(Nbytes));
  REQUIRE(C_h != NULL);

  for (size_t i = 0; i < N; i++) {
    A_h[i] = i;
  }

  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&C_d), Nbytes));

  HIPCHECK(hipMemcpyHtoD((hipDeviceptr_t)(A_d), A_h, Nbytes));

  struct {
    void* _Cd;
    void* _Ad;
    size_t _N;
  } args;
  args._Cd = reinterpret_cast<void**> (C_d);
  args._Ad = reinterpret_cast<void**> (A_d);
  args._N = static_cast<size_t> (N);
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  hipFunction_t Function;
  HIPCHECK(hipGetFuncBySymbol(&Function, reinterpret_cast<void*>(bit_extract_kernel)));

  HIPCHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL,
                                     reinterpret_cast<void**>(&config)));

  HIPCHECK(hipMemcpyDtoH(C_h, (hipDeviceptr_t)(C_d), Nbytes));

  for (size_t i = 0; i < N; i++) {
    unsigned Agold = ((A_h[i] & 0xf00) >> 8);
    REQUIRE(C_h[i] == Agold);
  }

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));

  free(A_h);
  free(C_h);
}
