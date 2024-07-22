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
#include "printf_common.h"  // NOLINT

__global__ void test_kernel_width() {
  printf("%16d\n", 42);
  printf("%.8d\n", 42);
  printf("%16.5d\n", -42);
  printf("%.8x\n", 0x42);
  printf("%.8o\n", 042);
  printf("%16.8e\n", 12345.67891);
  printf("%16.8f\n", -12345.67891);
  printf("%16.8g\n", 12345.67891);
  printf("%8.4e\n", -12345.67891);
  printf("%8.4f\n", 12345.67891);
  printf("%8.4g\n", 12345.67891);
  printf("%4.2f\n", 12345.67891);
  printf("%.1f\n", 12345.67891);
  printf("%.5s\n", "helloxyz");
}
/**
* @addtogroup printf printf
* @{
* @ingroup PrintfTest
* `int printf()` -
* Method to print the content on output device.
*/
/**
* Test Description
* ------------------------
* - Test case to verify the floating point details via printf API
* Test source
* ------------------------
* - catch/unit/printf/hipPrintfWidthPrecision.cc
* Test requirements
* ------------------------
* - HIP_VERSION >= 6.2
*/
TEST_CASE("Unit_Printf_PrintfWidthPrecision") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic,
                                  hipDeviceAttributeHostNativeAtomicSupported,
                                  0));
  if (!pcieAtomic) {
    HipTest::HIP_SKIP_TEST("Device doesn't support pcie atomic, Skipped");
    return;
  }
  std::string reference(R"here(              42
00000042
          -00042
00000042
00000042
  1.23456789e+04
 -12345.67891000
       12345.679
-1.2346e+04
12345.6789
1.235e+04
12345.68
12345.7
hello
)here");
  CaptureStream captured(stdout);
  hipLaunchKernelGGL(test_kernel_width, dim3(1), dim3(1), 0, 0);
  HIP_CHECK(hipStreamSynchronize(0));
  auto CapturedData = captured.getCapturedData();
  std::string device_output = captured.gulp(CapturedData);
  REQUIRE(device_output == reference);
}

/**
* End doxygen group PrintfTest.
* @}
*/
