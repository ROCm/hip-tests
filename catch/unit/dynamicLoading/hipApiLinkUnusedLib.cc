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
#include <hip_test_process.hh>
#include <map>
/**
* @addtogroup hipLaunchKernelGGL hipLaunchKernelGGL
* @{
* @ingroup DynamicLoading
* Launches Kernel with launch parameters and shared memory on stream with arguments passed.
*/

/**
 * Test Description
 * ------------------------
 * - This test links a HIP Application with an unused shared object built
 * using a different architecture. Launch a kernel built with current GPU
 * architecture. Launching the kernel should return error.
 * Test source
 * ------------------------
 * - catch/unit/dynamicLoading/hipApiLinkUnusedLib.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipApiLinkUnusedLibs") {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::string curArch = prop.gcnArchName;
  std::string difArch = "";
  // Build the application for current GPU architecture
  // and build the unrelated library for a different GPU.
  std::map<std::string, std::string> arch = {{"gfx9", "gfx1010"},
                                            {"gfx10", "gfx1100"},
                                            {"gfx11", "gfx900"}};

  for (auto& elem : arch) {
    if (std::string::npos != curArch.find(elem.first)) {
      difArch = elem.second;
    }
  }
  if (difArch == "") {
    HipTest::HIP_SKIP_TEST("offload-arch not Found. Skipping Test ...");
    return;
  }
  std::string cmd1 = std::string("hipcc vecadd.cc --offload-arch=")
                     + difArch;
  std::string cmd2 = " -fPIC -c -o vecadd.o";
  cmd1 = cmd1 + cmd2;
  std::system(cmd1.data());
  cmd1 = std::string("hipcc vecadd.o -shared --offload-arch=") + difArch;
  cmd2 = " -o vecadd.so";
  cmd1 = cmd1 + cmd2;
  std::system(cmd1.data());
  cmd1 = std::string(
  "hipcc hipApiLinkUnusedLibAppExe.cc -c -o hipApiLinkUnusedLibAppExe.o");
  std::system(cmd1.data());
  cmd1 = std::string(
  "hipcc hipApiLinkUnusedLibAppExe.o -o hipApiLinkUnusedLibAppExe");
  cmd2 = std::string(" -Wl,-rpath,. vecadd.so");
  cmd1 = cmd1 + cmd2;
  std::system(cmd1.data());
  hip::SpawnProc proc("hipApiLinkUnusedLibAppExe", true);
  REQUIRE(proc.run() == 0);
  int result = std::stof(proc.getOutput());
  INFO("result: " << result);
  REQUIRE(result == 0);
}
