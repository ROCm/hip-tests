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

#include <signal.h>
#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <ctime>
#include <fstream>
#include <cstddef>
#include <vector>


#define FILENAME_NONEXST "sample_nonexst.code"
#define FILENAME_EMPTY "emptyfile.code"
#define FILENAME_RAND "rand_file.code"
#define RANDOMFILE_LEN 2048
#define CODEOBJ_FILE "vcpy_kernel.code"
#define KERNEL_NAME "hello_world"
#define KERNEL_NAME_NONEXST "xyz"
#define CODEOBJ_GLOBAL "global_kernel.code"
#define DEVGLOB_VAR_NONEXIST "xyz"
#define DEVGLOB_VAR "myDeviceGlobal"
/**
 * Internal Function
 */
static std::vector<char> load_file(const char* filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    printf("Info:could not open code object '%s'\n", filename);
  }
  file.close();
  return buffer;
}

/**
 * Internal Function
 */
void createRandomFile(const char* filename) {
  std::ofstream outfile(filename, std::ios::binary);
  char buf[RANDOMFILE_LEN];
  unsigned int seed = 1;
  for (int i = 0; i < RANDOMFILE_LEN; i++) {
    buf[i] = HipTest::RAND_R(&seed) % 256;
  }
  outfile.write(buf, RANDOMFILE_LEN);
  outfile.close();
}

/**
 * Validates negative scenarios for hipModuleLoad
 * module = nullptr
 */
bool testhipModuleLoadNeg1() {
  bool TestPassed = false;
  hipError_t ret;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoad(nullptr, CODEOBJ_FILE);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoad
 * fname = nullptr
 */
bool testhipModuleLoadNeg2() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoad(&Module, nullptr);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}
/**
 * Validates negative scenarios for hipModuleLoad
 * fname = empty file
 */
bool testhipModuleLoadNeg3() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  // Create an empty
  std::fstream fs;
  fs.open(FILENAME_EMPTY, std::ios::out);
  fs.close();
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoad(&Module, FILENAME_EMPTY);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  remove(FILENAME_EMPTY);
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoad
 * fname = ramdom file
 */
bool testhipModuleLoadNeg4() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  // Create a binary file with random numbers
  createRandomFile(FILENAME_RAND);
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoad(&Module, FILENAME_RAND);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  remove(FILENAME_RAND);
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoad
 * fname = non existent file
 */
bool testhipModuleLoadNeg5() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoad(&Module, FILENAME_NONEXST);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoad
 * fname = empty string ""
 */
bool testhipModuleLoadNeg6() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoad(&Module, "");
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoadData
 * module = nullptr
 */
bool testhipModuleLoadDataNeg1() {
  bool TestPassed = false;
  hipError_t ret;
  auto buffer = load_file(CODEOBJ_FILE);
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoadData(nullptr, &buffer[0]);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoadData
 * image = nullptr
 */
bool testhipModuleLoadDataNeg2() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoadData(&Module, nullptr);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoadData
 * image = ramdom file
 */
bool testhipModuleLoadDataNeg3() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  // Create a binary file with random numbers
  createRandomFile(FILENAME_RAND);
  // Open the code object file and copy it in a buffer
  auto buffer = load_file(FILENAME_RAND);
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoadData(&Module, &buffer[0]);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  remove(FILENAME_RAND);
  return TestPassed;
}
/**
 * Validates negative scenarios for hipModuleLoadDataEx
 * module = nullptr
 */
bool testhipModuleLoadDataExNeg1() {
  bool TestPassed = false;
  hipError_t ret;
  // Open the code object file and copy it in a buffer
  auto buffer = load_file(CODEOBJ_FILE);
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoadDataEx(nullptr, &buffer[0], 0, nullptr, nullptr);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoadDataEx
 * image = nullptr
 */
bool testhipModuleLoadDataExNeg2() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoadDataEx(&Module, nullptr, 0, nullptr, nullptr);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleLoadDataEx
 * image = ramdom file
 */
bool testhipModuleLoadDataExNeg3() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  // Create a binary file with random numbers
  createRandomFile(FILENAME_RAND);
  // Open the code object file and copy it in a buffer
  auto buffer = load_file(FILENAME_RAND);
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleLoadDataEx(&Module, &buffer[0], 0, nullptr, nullptr);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  remove(FILENAME_RAND);
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * Function = nullptr
 */
bool testhipModuleGetFunctionNeg1() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  ret = hipModuleGetFunction(nullptr, Module, KERNEL_NAME);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
  HIP_CHECK(hipModuleUnload(Module));
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * Module is uninitialized
 */
bool testhipModuleGetFunctionNeg2() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module = nullptr;
  hipFunction_t Function;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleGetFunction(&Function, Module, KERNEL_NAME);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * kname = non existing function
 */
bool testhipModuleGetFunctionNeg3() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipFunction_t Function;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  ret = hipModuleGetFunction(&Function, Module, KERNEL_NAME_NONEXST);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
  HIP_CHECK(hipModuleUnload(Module));
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * kname = nullptr
 */
bool testhipModuleGetFunctionNeg4() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipFunction_t Function;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  ret = hipModuleGetFunction(&Function, Module, nullptr);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
  HIP_CHECK(hipModuleUnload(Module));
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * Module = Unloaded Module
 */
bool testhipModuleGetFunctionNeg5() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipFunction_t Function;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIP_CHECK(hipModuleUnload(Module));
  ret = hipModuleGetFunction(&Function, Module, KERNEL_NAME);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetFunction
 * kname = Empty String ""
 */
bool testhipModuleGetFunctionNeg6() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipFunction_t Function;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  ret = hipModuleGetFunction(&Function, Module, "");
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
  HIP_CHECK(hipModuleUnload(Module));
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * dptr = nullptr
 */
bool testhipModuleGetGlobalNeg1() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  size_t deviceGlobalSize;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  ret = hipModuleGetGlobal(nullptr, &deviceGlobalSize, Module, DEVGLOB_VAR);
  REQUIRE(ret == hipSuccess);
  TestPassed = true;
  HIP_CHECK(hipModuleUnload(Module));
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * bytes = nullptr
 */
bool testhipModuleGetGlobalNeg2() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  ret = hipModuleGetGlobal(&deviceGlobal, nullptr, Module, DEVGLOB_VAR);
  REQUIRE(ret == hipSuccess);
  TestPassed = true;
  HIP_CHECK(hipModuleUnload(Module));
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * name = nullptr
 */
bool testhipModuleGetGlobalNeg3() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  ret = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, nullptr);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
  HIP_CHECK(hipModuleUnload(Module));
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * name = wrong name
 */
bool testhipModuleGetGlobalNeg4() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  ret = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module,
                           DEVGLOB_VAR_NONEXIST);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
  HIP_CHECK(hipModuleUnload(Module));
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * name = Empty String ""
 */
bool testhipModuleGetGlobalNeg5() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  ret = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, "");
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
  HIP_CHECK(hipModuleUnload(Module));
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * Module = Unloaded Module
 */
bool testhipModuleGetGlobalNeg6() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_GLOBAL));
  HIP_CHECK(hipModuleUnload(Module));
  ret = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module,
                           DEVGLOB_VAR);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetGlobal
 * Module = Uninitialized Module
 */
bool testhipModuleGetGlobalNeg7() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module = nullptr;
  hipDeviceptr_t deviceGlobal;
  size_t deviceGlobalSize;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  ret = hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize,
                           Module, DEVGLOB_VAR);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleUnload
 * 1. Unload an uninitialized module
 * 2. Unload an unloaded module
 */
bool testhipModuleLoadNeg7() {
  bool TestPassed = false;
  hipError_t ret;
  hipModule_t Module = nullptr;
#if HT_NVIDIA
  hipCtx_t context;
  initHipCtx(&context);
#endif
  // test case 1
  SECTION("No obj file") {
  ret = hipModuleUnload(Module);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
  }
  // test case 2
  SECTION("CODEOBJ file") {
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIP_CHECK(hipModuleUnload(Module));
  ret = hipModuleUnload(Module);
  REQUIRE(ret != hipSuccess);
  TestPassed = true;
  }
#if HT_NVIDIA
  HIP_CHECK(hipCtxDestroy(context));
#endif
  return TestPassed;
}

TEST_CASE("Unit_hipModuleNegative") {
  bool TestPassed = true;

  SECTION("test running for testhipModuleLoadNeg1") {
    REQUIRE(TestPassed == testhipModuleLoadNeg1());
  }
  SECTION("test running for testhipModuleLoadNeg2") {
    REQUIRE(TestPassed == testhipModuleLoadNeg2());
  }
  SECTION("test running for testhipModuleLoadNeg3") {
    REQUIRE(TestPassed == testhipModuleLoadNeg3());
  }
  SECTION("test running for testhipModuleLoadNeg4") {
    REQUIRE(TestPassed == testhipModuleLoadNeg4());
  }
  SECTION("test running for testhipModuleLoadNeg5") {
    REQUIRE(TestPassed == testhipModuleLoadNeg5());
  }
  SECTION("test running for testhipModuleLoadNeg6") {
    REQUIRE(TestPassed == testhipModuleLoadNeg6());
  }
  SECTION("test running for testhipModuleLoadDataNeg1") {
    REQUIRE(TestPassed == testhipModuleLoadDataNeg1());
  }
  SECTION("test running for testhipModuleLoadDataNeg2") {
    REQUIRE(TestPassed == testhipModuleLoadDataNeg2());
  }
  SECTION("test running for testhipModuleLoadDataNeg3") {
    REQUIRE(TestPassed == testhipModuleLoadDataNeg3());
  }
  SECTION("test running for testhipModuleLoadDataExNeg1") {
    REQUIRE(TestPassed == testhipModuleLoadDataExNeg1());
  }
  SECTION("test running for testhipModuleLoadDataExNeg2") {
    REQUIRE(TestPassed == testhipModuleLoadDataExNeg2());
  }
  SECTION("test running for testhipModuleLoadDataExNeg3") {
    REQUIRE(TestPassed == testhipModuleLoadDataExNeg3());
  }
  SECTION("test running for testhipModuleGetFunctionNeg1") {
    REQUIRE(TestPassed == testhipModuleGetFunctionNeg1());
  }
  SECTION("test running for testhipModuleGetFunctionNeg2") {
    REQUIRE(TestPassed == testhipModuleGetFunctionNeg2());
  }
  SECTION("test running for testhipModuleGetFunctionNeg3") {
    REQUIRE(TestPassed == testhipModuleGetFunctionNeg3());
  }
  SECTION("test running for testhipModuleGetFunctionNeg4") {
    REQUIRE(TestPassed == testhipModuleGetFunctionNeg4());
  }
  #if HT_AMD
  SECTION("test running for testhipModuleGetFunctionNeg5") {
    REQUIRE(TestPassed == testhipModuleGetFunctionNeg5());
  }
  #endif
  SECTION("test running for testhipModuleGetFunctionNeg6") {
    REQUIRE(TestPassed == testhipModuleGetFunctionNeg6());
  }
  SECTION("test running for testhipModuleGetGlobalNeg1") {
    REQUIRE(TestPassed == testhipModuleGetGlobalNeg1());
  }
  SECTION("test running for testhipModuleGetGlobalNeg2") {
    REQUIRE(TestPassed == testhipModuleGetGlobalNeg2());
  }
  SECTION("test running for testhipModuleGetGlobalNeg3") {
    REQUIRE(TestPassed == testhipModuleGetGlobalNeg3());
  }
  SECTION("test running for testhipModuleGetGlobalNeg4") {
    REQUIRE(TestPassed == testhipModuleGetGlobalNeg4());
  }
  SECTION("test running for testhipModuleGetGlobalNeg5") {
    REQUIRE(TestPassed == testhipModuleGetGlobalNeg5());
  }
  #if HT_AMD
  SECTION("test running for testhipModuleGetGlobalNeg6") {
    REQUIRE(TestPassed == testhipModuleGetGlobalNeg6());
  }
  SECTION("test running for testhipModuleGetGlobalNeg7") {
    REQUIRE(TestPassed == testhipModuleGetGlobalNeg7());
  }
  SECTION("test running for testhipModuleLoadNeg7") {
    REQUIRE(TestPassed == testhipModuleLoadNeg7());
  }
  #endif
}
