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

#define fileName "vcpy_kernel.code"
#define kernel_name "hello_world"

TEST_CASE("Unit_hipModuleOccupancyMaxPotentialActiveBlockSize") {
  int gridSize = 0;
  int blockSize = 0;
  int numBlock = 0;
  HIP_CHECK(hipInit(0));

  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  #if HT_NVIDIA
  HIP_CHECK(hipCtxCreate(&context, 0, device));
  #endif
  hipModule_t Module;
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                                    Function, 0, 0));
  assert(gridSize != 0 && blockSize != 0);
  HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock,
                                                 Function, blockSize, 0));
  assert(numBlock != 0);
  HIP_CHECK(hipModuleUnload(Module));
  #if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
  #endif
}
