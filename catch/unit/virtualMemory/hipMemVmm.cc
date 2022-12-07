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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <resource_guards.hh>

TEST_CASE("Unit_hipMemVmm_OneToOne_Basic") {
  int vmm = 0;
  HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  INFO("hipDeviceAttributeVirtualMemoryManagementSupported: " << vmm);

  if (vmm == 0) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }

  size_t size = 4 * 1024;
  VirtualMemoryGuard virtual_memory{size};

  hipDeviceptr_t device_memory_ptr = reinterpret_cast<hipDeviceptr_t>(virtual_memory.virtual_memory_ptr);
  HIP_CHECK(hipMemsetD32(device_memory_ptr, 0xDEADBEAF, size/4));
  std::vector<unsigned int> values(size/4);
  HIP_CHECK(hipMemcpy(&values[0], virtual_memory.virtual_memory_ptr, size, hipMemcpyDeviceToHost));

  for (const auto& value: values) {
    REQUIRE(value == 0xDEADBEAF);
  }
}

TEST_CASE("Unit_hipMemVmm_OneToN_Basic") {
  int vmm = 0;
  HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  INFO("hipDeviceAttributeVirtualMemoryManagementSupported: " << vmm);

  if (vmm == 0) {
    HipTest::HIP_SKIP_TEST("GPU 0 doesn't support hipDeviceAttributeVirtualMemoryManagement "
           "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }

  size_t size = 4 * 1024;
  VirtualMemoryGuard virtual_memory_A{size};
  VirtualMemoryGuard virtual_memory_B{size, 0, &virtual_memory_A.handle};

  hipDeviceptr_t device_memory_ptr = reinterpret_cast<hipDeviceptr_t>(virtual_memory_A.virtual_memory_ptr);
  HIP_CHECK(hipMemsetD32(device_memory_ptr, 0xDEADBEAF, size/4));
  std::vector<unsigned int> values(size/4);
  HIP_CHECK(hipMemcpy(&values[0], virtual_memory_B.virtual_memory_ptr, size, hipMemcpyDeviceToHost));

  for (const auto& value: values) {
    REQUIRE(value == 0xDEADBEAF);
  }
}