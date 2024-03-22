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
 * @addtogroup hipModuleLaunchCooperativeKernelMultiDevice
 * hipModuleLaunchCooperativeKernelMultiDevice
 * @{
 * @ingroup ModuleTest
 * `hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList, unsigned
 * int numDevices, unsigned int flags)` -
 * Launches kernels on multiple devices where thread blocks can cooperate and synchronize as they
 * execute.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include "hip_module_launch_kernel_common.hh"

/**
 * Test Description
 * ------------------------
 *  - Tests `hipModuleLaunchCooperativeKernel` for a cooperative kernel with no parameters.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchCooperativeKernelMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
TEST_CASE("Unit_hipModuleLaunchCooperativeKernelMultiDevice_Positive_Basic") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipFunction_t f = GetKernel(mg.module(), "CoopKernel");

  const auto device_count = HipTest::getDeviceCount();

  std::vector<hipFunctionLaunchParams> params_list(device_count);

  int device = 0;
  for (auto& params : params_list) {
    params.function = f;
    params.gridDimX = 1;
    params.gridDimY = 1;
    params.gridDimZ = 1;
    params.blockDimX = 1;
    params.blockDimY = 1;
    params.blockDimZ = 1;
    params.kernelParams = nullptr;
    params.sharedMemBytes = 0;
    HIP_CHECK(hipSetDevice(device++));
    HIP_CHECK(hipStreamCreate(&params.hStream));
  }

  HIP_CHECK(hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u));

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamSynchronize(params.hStream));
  }

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamDestroy(params.hStream));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipModuleLaunchCooperativeKernelMultiDevice`.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchCooperativeKernelMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
TEST_CASE("Unit_hipModuleLaunchCooperativeKernelMultiDevice_Negative_Parameters") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipFunction_t f = GetKernel(mg.module(), "CoopKernel");

  const auto device_count = HipTest::getDeviceCount();

  std::vector<hipFunctionLaunchParams> params_list(device_count);

  int device = 0;
  for (auto& params : params_list) {
    params.function = f;
    params.gridDimX = 1;
    params.gridDimY = 1;
    params.gridDimZ = 1;
    params.blockDimX = 1;
    params.blockDimY = 1;
    params.blockDimZ = 1;
    params.kernelParams = nullptr;
    params.sharedMemBytes = 0;
    HIP_CHECK(hipSetDevice(device++));
    HIP_CHECK(hipStreamCreate(&params.hStream));
  }

  SECTION("launchParamsList == nullptr") {
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernelMultiDevice(nullptr, device_count, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("numDevices == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), 0, 0u),
                    hipErrorInvalidValue);
  }

  SECTION("numDevices > device count") {
    HIP_CHECK_ERROR(
        hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count + 1, 0u),
        hipErrorInvalidValue);
  }

  SECTION("invalid flags") {
    HIP_CHECK_ERROR(
        hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 999),
        hipErrorInvalidValue);
  }

  if (device_count > 1) {
    SECTION("launchParamsList.func doesn't match across all devices") {
      params_list[1].function = GetKernel(mg.module(), "NOPKernel");
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidValue);
    }

    SECTION("launchParamsList.gridDim doesn't match across all kernels") {
      params_list[1].gridDimX = 2;
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidValue);
    }

    SECTION("launchParamsList.blockDim doesn't match across all kernels") {
      params_list[1].blockDimX = 2;
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidValue);
    }

    SECTION("launchParamsList.sharedMem doesn't match across all kernels") {
      params_list[1].sharedMemBytes = 1024;
      HIP_CHECK_ERROR(
          hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), device_count, 0u),
          hipErrorInvalidValue);
    }
  }

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamDestroy(params.hStream));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Tries running `hipModuleLaunchCooperativeKernelMultiDevice` with multiple kernels on the same
 * device.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleLaunchCooperativeKernelMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.5
 */
TEST_CASE("Unit_hipModuleLaunchCooperativeKernelMultiDevice_Negative_MultiKernelSameDevice") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  hipFunction_t f = GetKernel(mg.module(), "CoopKernel");

  HIP_CHECK(hipSetDevice(0));

  std::vector<hipFunctionLaunchParams> params_list(2);

  for (auto& params : params_list) {
    params.function = f;
    params.gridDimX = 1;
    params.gridDimY = 1;
    params.gridDimZ = 1;
    params.blockDimX = 1;
    params.blockDimY = 1;
    params.blockDimZ = 1;
    params.kernelParams = nullptr;
    params.sharedMemBytes = 0;
    HIP_CHECK(hipStreamCreate(&params.hStream));
  }

  HIP_CHECK_ERROR(hipModuleLaunchCooperativeKernelMultiDevice(params_list.data(), 2, 0u),
                  hipErrorInvalidValue);

  for (const auto params : params_list) {
    HIP_CHECK(hipStreamDestroy(params.hStream));
  }
}

/**
* End doxygen group ModuleTest.
* @}
*/
