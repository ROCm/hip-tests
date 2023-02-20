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

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <fstream>
#include <vector>

#define fileName "global_kernel.code"
#define LEN 64
#define SIZE LEN * sizeof(float)
#define ARRAY_SIZE 16

struct {
  void* _Ad;
  void* _Bd;
} args;

TEST_CASE("Unit_hipModuleGetGlobal") {
  float *A, *B;
  float *Ad, *Bd;
  A = new float[LEN];
  B = new float[LEN];

  for (uint32_t i = 0; i < LEN; i++) {
      A[i] = i * 1.0f;
      B[i] = 0.0f;
  }

  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  #if HT_NVIDIA
  HIP_CHECK(hipCtxCreate(&context, 0, device));
  #endif
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));

  HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(Ad), A, SIZE));
  HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t)(Bd), B, SIZE));
  hipModule_t Module;
  HIP_CHECK(hipModuleLoad(&Module, fileName));

  float myDeviceGlobal_h = 42.0;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize,
                               Module, "myDeviceGlobal"));
  HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(deviceGlobal), &myDeviceGlobal_h,
                                                        deviceGlobalSize));

  float myDeviceGlobalArray_h[ARRAY_SIZE];
  hipDeviceptr_t myDeviceGlobalArray;
  size_t myDeviceGlobalArraySize;

  HIP_CHECK(hipModuleGetGlobal(reinterpret_cast<hipDeviceptr_t*>(&myDeviceGlobalArray), //NOLINT
           &myDeviceGlobalArraySize, Module, "myDeviceGlobalArray"));

  for (int i = 0; i < ARRAY_SIZE; i++) {
    myDeviceGlobalArray_h[i] = i * 1000.0f;
    HIP_CHECK(hipMemcpyHtoD(hipDeviceptr_t(myDeviceGlobalArray),
               &myDeviceGlobalArray_h, myDeviceGlobalArraySize));
  }

  args._Ad = reinterpret_cast<void*>(Ad);
  args._Bd = reinterpret_cast<void*>(Bd);

  size_t size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  SECTION("Running test for hello world kernel") {
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "hello_world"));
    HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL,
                                    reinterpret_cast<void**>(&config)));

    HIP_CHECK(hipMemcpyDtoH(B, hipDeviceptr_t(Bd), SIZE));

    int mismatchCount = 0;
    for (uint32_t i = 0; i < LEN; i++) {
      if (A[i] != B[i]) {
        mismatchCount++;
      if (mismatchCount >= 10) {
        break;
      }
    }
    }
  REQUIRE(mismatchCount == 0);
  }

  SECTION("running test for tests_globals kernel") {
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module,
                                   "test_globals"));
    HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL,
                                    reinterpret_cast<void**>(&config)));
    HIP_CHECK(hipMemcpyDtoH(B, hipDeviceptr_t(Bd), SIZE));

    int mismatchCount = 0;
    for (uint32_t i = 0; i < LEN; i++) {
      float expected;
      expected = A[i] + myDeviceGlobal_h + + myDeviceGlobalArray_h[i % 16];
      if (expected != B[i]) {
        mismatchCount++;
      if (mismatchCount >= 10) {
        break;
      }
    }
    }
  REQUIRE(mismatchCount == 0);
  }

  HIP_CHECK(hipModuleUnload(Module));
  #if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
  #endif
}
