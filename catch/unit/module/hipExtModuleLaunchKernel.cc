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

/**
 * @addtogroup hipExtModuleLaunchKernel hipExtModuleLaunchKernel
 * @{
 * @ingroup KernelTest
 * `hipError_t hipExtModuleLaunchKernel	(hipFunction_t  f,
 *                                      uint32_t  globalWorkSizeX,
 *                                      uint32_t  globalWorkSizeY,
 *                                      uint32_t  globalWorkSizeZ,
 *                                      uint32_t  localWorkSizeX,
 *                                      uint32_t  localWorkSizeY,
 *                                      uint32_t  localWorkSizeZ,
 *                                      size_t  sharedMemBytes,
 *                                      hipStream_t  hStream,
 *                                      void **  kernelParams,
 *                                      void **  extra,
 *                                      hipEvent_t  startEvent = nullptr,
 *                                      hipEvent_t 	stopEvent = nullptr,
 *                                      uint32_t  flags = 0
 *                                      )` -
 * Launches kernel with parameters and shared memory on stream with arguments
 * passed to kernel params or extra arguments.
 */

#include <hip_test_common.hh>
 
#include <iostream>
#include <fstream>
#include "hip/hip_ext.h"
#include <regex>  // NOLINT

#include "hip_module_launch_kernel_common.hh"

static constexpr auto totalWorkGroups{1024};
static constexpr auto localWorkSize{512};
static constexpr auto lastWorkSizeEven{256};
static constexpr auto lastWorkSizeOdd{257};

#define fileName "copyKernel.code"
#define kernel_name "copy_ker"

/**
  Local Function to search a string in file.
*/
static bool searchRegExpr(const std::regex& expr, const char* filename) {
  std::ifstream assemblyfile(filename, std::ifstream::binary);
  REQUIRE(true == assemblyfile.is_open());

  // Copy the contents of the file to buffer
  assemblyfile.seekg(0, assemblyfile.end);
  int len = assemblyfile.tellg();
  assemblyfile.seekg(0, assemblyfile.beg);
  char* fbuf = new char[len + 1];
  assemblyfile.read(fbuf, len);
  fbuf[len] = '\0';

  // Search for uniform_work_group_size
  std::smatch pattern;
  std::string assemblyStr = fbuf;
  bool bfound = std::regex_search(assemblyStr, pattern, expr);
  delete[] fbuf;
  assemblyfile.close();
  return bfound;
}
/**
 * Test Description
 * ------------------------
 *    - Parse the Code Object Assembly file copyKernel.s and verify
 * if .uniform_work_group_size metadata is available.
 * ------------------------
 *    - catch\unit\module\hipExtModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipExtModuleLaunchKernel_CheckCodeObjAttr") {
  // Open copyKernel.s and read the file
  const std::regex regexp("uniform_work_group_size\\s*:\\s*[0-1]");
  REQUIRE(true == searchRegExpr(regexp, "copyKernel.s"));
}

/**
 * Test Description
 * ------------------------
 *    - Precondition: .uniform_work_group_size = 1. Which means uniform workgroup
 *      is enforced.
 *    - Create a buffer of size globalWorkSizeX = (non multiple of localWorkSizeX)
 * and launch a kernel to perform some operations on it. If uniform work grouping
 * is enforced then hipExtModuleLaunchKernel returns hipErrorInvalidValue.
 * Test source
 * ------------------------
 *    - catch\unit\module\hipExtModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipExtModuleLaunchKernel_NonUniformWorkGroup") {
  // first check if uniform_work_group_size = 1.
  const std::regex regexp("uniform_work_group_size\\s*:\\s*1");
  if (false == searchRegExpr(regexp, "copyKernel.s")) {
    HipTest::HIP_SKIP_TEST("uniform_work_group_size != 1. Skipping test ...");
    return;
  }
  REQUIRE(true == searchRegExpr(regexp, "copyKernel.s"));
  auto isEven = GENERATE(0, 1);
  // Calculate size
  auto lastWorkSize = isEven ? lastWorkSizeEven : lastWorkSizeOdd;
  size_t arraylength = (totalWorkGroups - 1) * localWorkSize + lastWorkSize;
  size_t sizeBytes{arraylength * sizeof(int)};
  // Get module and function from module
  hipModule_t Module;
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  // Allocate resources
  int* A = new int[arraylength];
  REQUIRE(A != nullptr);
  int* B = new int[arraylength];
  REQUIRE(B != nullptr);
  // Inititialize data
  for (size_t i = 0; i < arraylength; i++) {
    A[i] = i;
  }
  int *Ad, *Bd;
  HIP_CHECK(hipMalloc(&Ad, sizeBytes));
  HIP_CHECK(hipMalloc(&Bd, sizeBytes));
  struct {
    void* _Ad;
    void* _Bd;
    size_t buffersize;
  } args;

  args._Ad = Ad;
  args._Bd = Bd;
  args.buffersize = arraylength;
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  // Memcpy from A to Ad
  HIP_CHECK(hipMemcpy(Ad, A, sizeBytes, hipMemcpyDefault));
  REQUIRE(hipErrorInvalidValue ==
          hipExtModuleLaunchKernel(Function, arraylength, 1, 1, localWorkSize, 1, 1, 0, 0, NULL,
                                   reinterpret_cast<void**>(&config), 0));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  delete[] A;
  delete[] B;
  HIP_CHECK(hipModuleUnload(Module));
}

/**
 * Test Description
 * ------------------------
 *    - Create a buffer of size globalWorkSizeX = (multiple of localWorkSizeX)
 * and launch a kernel to perform some operations on it. Verify the output. This
 * operation should be allowed for both uniform_work_group_size = true/false.
 *
 * Test source
 * ------------------------
 *    - catch\unit\module\hipExtModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipExtModuleLaunchKernel_UniformWorkGroup") {
  size_t arraylength = totalWorkGroups * localWorkSize;
  size_t sizeBytes{arraylength * sizeof(int)};
  // Get module and function from module
  hipModule_t Module;
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  // Allocate resources
  int* A = new int[arraylength];
  REQUIRE(A != nullptr);
  int* B = new int[arraylength];
  REQUIRE(B != nullptr);
  // Inititialize data
  for (size_t i = 0; i < arraylength; i++) {
    A[i] = i;
  }
  int *Ad, *Bd;
  HIP_CHECK(hipMalloc(&Ad, sizeBytes));
  HIP_CHECK(hipMalloc(&Bd, sizeBytes));
  struct {
    void* _Ad;
    void* _Bd;
    size_t buffersize;
  } args;

  args._Ad = Ad;
  args._Bd = Bd;
  args.buffersize = arraylength;
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  // Memcpy from A to Ad
  HIP_CHECK(hipMemcpy(Ad, A, sizeBytes, hipMemcpyDefault));
  HIP_CHECK(hipExtModuleLaunchKernel(Function, arraylength, 1, 1, localWorkSize, 1, 1, 0, 0, NULL,
                                     reinterpret_cast<void**>(&config), 0));
  // Memcpy results back to host
  HIP_CHECK(hipMemcpy(B, Bd, sizeBytes, hipMemcpyDefault));
  HIP_CHECK(hipDeviceSynchronize());
  // Verify results
  for (size_t i = 0; i < arraylength; i++) {
    REQUIRE(B[i] == i);
  }
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  delete[] A;
  delete[] B;
  HIP_CHECK(hipModuleUnload(Module));
}

TEST_CASE("Unit_hipExtModuleLaunchKernel_Positive_Basic") {
  ModuleLaunchKernelPositiveBasic<hipExtModuleLaunchKernel>();

  SECTION("Timed kernel launch with events") {
    hipEvent_t start_event = nullptr, stop_event = nullptr;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));
    const auto kernel = GetKernel(mg.module(), "Delay");
    int clock_rate = 0;
    HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeClockRate, 0));
    uint32_t interval = 100;
    uint32_t ticks_per_second = clock_rate;
    void* kernel_params[2] = {&interval, &ticks_per_second};
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_params, nullptr,
                                       start_event, stop_event));
    HIP_CHECK(hipDeviceSynchronize());
    auto elapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed, start_event, stop_event));
    REQUIRE(static_cast<uint32_t>(elapsed) >= interval);
  }
}

TEST_CASE("Unit_hipExtModuleLaunchKernel_Positive_Parameters") {
  ModuleLaunchKernelPositiveParameters<hipExtModuleLaunchKernel>();

  SECTION("Pass only start event") {
    hipEvent_t start_event = nullptr;
    HIP_CHECK(hipEventCreate(&start_event));
    const auto kernel = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr,
                                       start_event, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventQuery(start_event));
  }

  SECTION("Pass only stop event") {
    hipEvent_t stop_event = nullptr;
    HIP_CHECK(hipEventCreate(&stop_event));
    const auto kernel = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr,
                                       nullptr, stop_event));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventQuery(stop_event));
  }
}

TEST_CASE("Unit_hipExtModuleLaunchKernel_Negative_Parameters") {
  ModuleLaunchKernelNegativeParameters<hipExtModuleLaunchKernel>();
}

/**
* End doxygen group KernelTest.
* @}
*/
