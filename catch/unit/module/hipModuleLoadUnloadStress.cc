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
#include <cstddef>
#include <vector>

#define TEST_ITERATIONS 1000
#define CODEOBJ_FILE "kernel_composite_test.code"

/**
 * Internal Function
 */
static std::vector<char> load_file() {
  std::ifstream file(CODEOBJ_FILE, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    printf("Info:could not open code object '%s'\n", CODEOBJ_FILE);
  }
  file.close();
  return buffer;
}
/**
 * Validates no memory leakage for hipModuleLoad
 */
static void testhipModuleLoadUnloadStress() {
  for (int count = 0; count < TEST_ITERATIONS; count++) {
    hipModule_t Module;
    HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "testWeightedCopy"));
    HIP_CHECK(hipModuleUnload(Module));
  }
}
/**
 * Validates no memory leakage for hipModuleLoadData
 */
static void testhipModuleLoadDataUnloadStress() {
  auto buffer = load_file();
  for (int count = 0; count < TEST_ITERATIONS; count++) {
    hipModule_t Module;
    HIP_CHECK(hipModuleLoadData(&Module, &buffer[0]));
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "testWeightedCopy"));
    HIP_CHECK(hipModuleUnload(Module));
  }
}
/**
 * Validates no memory leakage for hipModuleLoadDataEx
 */
static void testhipModuleLoadDataExUnloadStress() {
  auto buffer = load_file();
  for (int count = 0; count < TEST_ITERATIONS; count++) {
    hipModule_t Module;
    HIP_CHECK(hipModuleLoadDataEx(&Module, &buffer[0], 0,
                                nullptr, nullptr));
    hipFunction_t Function;
    HIP_CHECK(hipModuleGetFunction(&Function, Module, "testWeightedCopy"));
    HIP_CHECK(hipModuleUnload(Module));
  }
}

TEST_CASE("Unit_hipModuleLoadUnloadStress") {
#if HT_NVIDIA
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));
#endif
  SECTION("running hipModuleLoadUnloadStress") {
    testhipModuleLoadUnloadStress();
  }
  SECTION("running hipModuleLoadDataUnloadStress") {
    testhipModuleLoadDataUnloadStress();
  }
  SECTION("running hipModuleLoadDataExUnloadStress") {
    testhipModuleLoadDataExUnloadStress();
  }
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
}
