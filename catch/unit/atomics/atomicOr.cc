/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include "atomicOr_negative_kernels_rtc.hh"
#include "bitwise_common.hh"
#include <hip_test_common.hh>

TEMPLATE_TEST_CASE("Unit_atomicOr_Positive_SameAddress", "", int, unsigned int, unsigned long long) {
  Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(1, sizeof(TestType));
}

TEMPLATE_TEST_CASE("Unit_atomicOr_Positive_Adjacent_Addresses", "", int, unsigned int, unsigned long long) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(warp_size, sizeof(TestType));
}

TEMPLATE_TEST_CASE("Unit_atomicOr_Positive_Scattered_Addresses", "", int, unsigned int, unsigned long long) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  Bitwise::SingleDeviceSingleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(warp_size, cache_line_size);
}

TEMPLATE_TEST_CASE("Unit_atomicOr_Positive_Multi_Kernel_Same_Address", "", int, unsigned int, unsigned long long) {
  Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(2, 1, sizeof(TestType));
}

TEMPLATE_TEST_CASE("Unit_atomicOr_Positive_Multi_Kernel_Adjacent_Addresses", "", int, unsigned int, unsigned long long) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));

  Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(2, warp_size, sizeof(TestType));
}

TEMPLATE_TEST_CASE("Unit_atomicOr_Positive_Multi_Kernel_Scattered_Addresses", "", int, unsigned int, unsigned long long) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  Bitwise::SingleDeviceMultipleKernelTest<TestType, Bitwise::AtomicOperation::kOr>(2, warp_size, cache_line_size);
}

TEST_CASE("Unit_atomicOr_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source =
    GENERATE(kAtomicOr_int, kAtomicOr_uint, kAtomicOr_ulong, kAtomicOr_ulonglong);
  HIPRTC_CHECK(hiprtcCreateProgram(&program, program_source, "atomicOr_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};
  // Please check the content of negative_kernels_rtc.hh
  int expected_error_count{10};
  std::string error_message{"error:"};

  size_t n_pos = log.find(error_message, 0);
  while(n_pos != std::string::npos) {
    ++error_count;
    n_pos = log.find(error_message, n_pos + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_error_count);
}
