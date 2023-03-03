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

#include "atomicExch_common.hh"
#include "atomicExch_system_negative_kernels_rtc.hh"

TEMPLATE_TEST_CASE("Unit_atomicExch_system_Positive_Peer_GPUs", "", int, unsigned int,
                   unsigned long long, float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  SECTION("Same address") {
    AtomicExchMultipleDeviceMultipleKernelAndHostTest<TestType>(2, 2, 1, sizeof(TestType));
  }

  SECTION("Adjacent addresses") {
    AtomicExchMultipleDeviceMultipleKernelAndHostTest<TestType>(2, 2, warp_size, sizeof(TestType));
  }

  SECTION("Scattered addresses") {
    AtomicExchMultipleDeviceMultipleKernelAndHostTest<TestType>(2, 2, warp_size, cache_line_size);
  }
}

TEMPLATE_TEST_CASE("Unit_atomicExch_system_Positive_Host_And_GPU", "", int, unsigned int,
                   unsigned long long, float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  SECTION("Same address") {
    AtomicExchMultipleDeviceMultipleKernelAndHostTest<TestType>(1, 1, 1, sizeof(TestType), 4);
  }

  SECTION("Adjacent addresses") {
    AtomicExchMultipleDeviceMultipleKernelAndHostTest<TestType>(1, 1, warp_size, sizeof(TestType),
                                                                4);
  }

  SECTION("Scattered addresses") {
    AtomicExchMultipleDeviceMultipleKernelAndHostTest<TestType>(1, 1, warp_size, cache_line_size,
                                                                4);
  }
}

TEMPLATE_TEST_CASE("Unit_atomicExch_system_Positive_Host_And_Peer_GPUs", "", int, unsigned int,
                   unsigned long long, float, double) {
  int warp_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0));
  const auto cache_line_size = 128u;

  SECTION("Same address") {
    AtomicExchMultipleDeviceMultipleKernelAndHostTest<TestType>(2, 2, 1, sizeof(TestType), 4);
  }

  SECTION("Adjacent addresses") {
    AtomicExchMultipleDeviceMultipleKernelAndHostTest<TestType>(2, 2, warp_size, sizeof(TestType),
                                                                4);
  }

  SECTION("Scattered addresses") {
    AtomicExchMultipleDeviceMultipleKernelAndHostTest<TestType>(2, 2, warp_size, cache_line_size,
                                                                4);
  }
}

TEST_CASE("Unit_atomicExch_system_Negative_Parameters_RTC") {
  hiprtcProgram program{};

  const auto program_source =
      GENERATE(kAtomicExchSystemInt, kAtomicExchSystemUnsignedInt, kAtomicExchSystemULL,
               kAtomicExchSystemFloat, kAtomicExchSystemDouble);
  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "atomicExch_negative.cc", 0, nullptr, nullptr));
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};

  int expected_error_count{8};
  std::string error_message{"error:"};

  size_t n_pos = log.find(error_message, 0);
  while (n_pos != std::string::npos) {
    ++error_count;
    n_pos = log.find(error_message, n_pos + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_error_count);
}
