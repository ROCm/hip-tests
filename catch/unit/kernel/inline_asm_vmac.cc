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
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

#define NUM 1024
#define THREADS_PER_BLOCK_X 4

// Device (Kernel) function, it must be void
__global__ void vmac_asm(float* out, float* in, float a) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  asm volatile("v_mac_f32_e32 %0, %2, %3" : "=v"(out[i]) :
               "0"(out[i]), "v"(a), "v"(in[i]));
}

// CPU implementation of saxpy
void addCPUReference(float* output, float* input, float a) {
  for (unsigned int j = 0; j < NUM; j++) {
    output[j] = a * input[j] + output[j];
  }
}
/**
* @addtogroup hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 * - Test case to check inline asm vmac instruction via kernel call.

 * Test source
 * ------------------------
 * - catch/unit/kernel/inline_asm_vmac.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_kernel_inline_asm_vmac_Functional") {
  float* VectorA;
  float* ResultVector;
  float* VectorB;

  float* gpuVector;
  float* gpuResultVector;

  const float a = 10.0f;
  int i;
  int errors;

  VectorA = reinterpret_cast<float*>(malloc(NUM * sizeof(float)));
  ResultVector = reinterpret_cast<float*>(malloc(NUM * sizeof(float)));
  VectorB = reinterpret_cast<float*>(malloc(NUM * sizeof(float)));

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    VectorA[i] = static_cast<float>(i * 10.0f);
    VectorB[i] = static_cast<float>(i * 30.0f);
  }

  // allocate the memory on the device side
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpuVector),
                      NUM * sizeof(float)));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&gpuResultVector),
                      NUM * sizeof(float)));

  // Memory transfer from host to device
  HIP_CHECK(hipMemcpy(gpuVector, VectorA, NUM * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(gpuResultVector, VectorB, NUM * sizeof(float),
                      hipMemcpyHostToDevice));

  // Lauching kernel from host
  hipLaunchKernelGGL(vmac_asm, dim3(NUM / THREADS_PER_BLOCK_X),
                    dim3(THREADS_PER_BLOCK_X), 0, 0,
                    gpuResultVector, gpuVector, a);

  // Memory transfer from device to host
  HIP_CHECK(hipMemcpy(ResultVector, gpuResultVector, NUM * sizeof(float),
                      hipMemcpyDeviceToHost));

  // CPU Result computation
  addCPUReference(VectorB, VectorA, a);

  // verify the results
  errors = 0;
  double eps = 1.0E-3;
  for (i = 0; i < NUM; i++) {
    if (std::abs(ResultVector[i] - VectorB[i]) > eps) {
      errors++;
    }
  }
  if (errors != 0) {
    REQUIRE(false);
  } else {
    REQUIRE(true);
  }

  // free the resources on device side
  HIP_CHECK(hipFree(gpuVector));
  HIP_CHECK(hipFree(gpuResultVector));
  HIP_CHECK(hipDeviceReset());

  // free the resources on host side
  free(VectorA);
  free(ResultVector);
  free(VectorB);
}
