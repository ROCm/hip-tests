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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#define N   1024

#ifdef __linux__
#include <unistd.h>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <string>

static void deleteFile(const char* fName) {
  if ( remove(fName) != 0 ) {
    INFO("Error in deleting file -" << fName);
  } else {
    INFO("Successfully deleted file -" << fName);
  }
}

static bool checkFileExists(const char* fName) {
  return (access(fName, F_OK) != -1);
}

static int countSubstr(std::string input_str, std::string substr) {
  int count = 0;
  std::string::size_type srch_pos = 0, cur_pos = 0;
  while ((cur_pos = input_str.find(substr, srch_pos)) != std::string::npos) {
    count++;
    srch_pos = (cur_pos + substr.length());
  }
  return count;
}

static bool validateDotFile(const char* fName,
                           std::map<std::string, int> *graphData) {
  bool isTestPassed = true;
  std::ifstream infile(fName);
  std::stringstream buffer;
  buffer << infile.rdbuf();
  std::map<std::string, int>::iterator it;
  for (it = (*graphData).begin(); it != (*graphData).end(); it++) {
    if (it->second != countSubstr(buffer.str(), it->first)) {
      isTestPassed = false;
      break;
    }
  }
  return isTestPassed;
}

static void hipGraphDebugDotPrint_Functional(const char* fName,
                                             unsigned int flag = 0) {
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipStream_t stream;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, kNodeAdd, memsetNode;
  hipKernelNodeParams kNodeParams{};
  int *A_d, *B_d, *C_d, *mem_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipMalloc(&mem_d, Nbytes));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h,
                                   Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h,
                                   Nbytes, hipMemcpyHostToDevice));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kNodeAdd, graph, nullptr, 0, &kNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kNodeAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kNodeAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNodeAdd, &memcpy_C, 1));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(mem_d);
  memsetParams.value = 7;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                  &memsetParams));

  std::map<std::string, int> graphData;
  graphData["->"] = 3;     //  number of edges
  graphData["MEMCPY"] = 3;
  graphData["HtoD"] = 2;
  graphData["DtoH"] = 1;
  graphData["MEMSET"] = 1;
  if ( flag == hipGraphDebugDotFlagsVerbose ) graphData["KERNEL"] = 1;

  HIP_CHECK(hipGraphDebugDotPrint(graph, fName, flag));
  REQUIRE(true == checkFileExists(fName));
  REQUIRE(true == validateDotFile(fName, &graphData));
  deleteFile(fName);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  HIP_CHECK(hipFree(mem_d));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/* Functional Test for API - hipGraphDebugDotPrint
   Call hipGraphDebugDotPrint and provice path where to write the DOT file.
   Verify that DOT file get created or not for each flag passed. */

TEST_CASE("Unit_hipGraphDebugDotPrint_Functional") {
  SECTION("Call with hipGraphDebugDotFlagsVerbose flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncVerbose.dot",
                                     hipGraphDebugDotFlagsVerbose);
  }
  SECTION("Call with hipGraphDebugDotFlagsKernelNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncKernelParams.dot",
                                     hipGraphDebugDotFlagsKernelNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsMemcpyNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncMemcpy.dot",
                                     hipGraphDebugDotFlagsMemcpyNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsMemsetNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncMemset.dot",
                                     hipGraphDebugDotFlagsMemsetNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsHostNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncHost.dot",
                                     hipGraphDebugDotFlagsHostNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsEventNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncEvent.dot",
                                     hipGraphDebugDotFlagsEventNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsExtSemasSignalNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncExtSemasSignal.dot",
                          hipGraphDebugDotFlagsExtSemasSignalNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsExtSemasWaitNodeParams flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncExtSemasWait.dot",
                          hipGraphDebugDotFlagsExtSemasWaitNodeParams);
  }
  SECTION("Call with hipGraphDebugDotFlagsKernelNodeAttributes flag") {
    hipGraphDebugDotPrint_Functional("./graphDotFileFuncKernelNodeAttr.dot",
                          hipGraphDebugDotFlagsKernelNodeAttributes);
  }
}
#endif  //  __linux__

/**
* Negative Test for API - hipGraphDebugDotPrint Argument Check
1) Pass graph as nullptr.
2) Pass graph as uninitialize structure
3) Pass path for dot file to store as nullptr
4) Pass path for dot file to store as empth path
5) Pass flag as hipGraphDebugDotFlags MIN - 1
6) Pass flag as hipGraphDebugDotFlags MAX + 1
7) Pass flag as INT_MAX
*/

#define DOT_FILE_PATH_NEG   "./graphDotFileNeg.dot"

TEST_CASE("Unit_hipGraphDebugDotPrint_Argument_Check") {
  hipGraph_t graph;
  hipError_t ret;

  HIP_CHECK(hipGraphCreate(&graph, 0));

  SECTION("Pass graph as nullptr") {
    ret = hipGraphDebugDotPrint(nullptr, DOT_FILE_PATH_NEG, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass graph as uninitialize structure") {
    hipGraph_t graphT{};
    ret = hipGraphDebugDotPrint(graphT, DOT_FILE_PATH_NEG, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass path for dot file to store as nullptr") {
    ret = hipGraphDebugDotPrint(graph, nullptr, 0);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass path for dot file to store as empth path") {
    ret = hipGraphDebugDotPrint(graph, "", 0);
    REQUIRE(hipErrorOperatingSystem == ret);
  }
  SECTION("Pass flag as hipGraphDebugDotFlags MIN - 1") {
    ret = hipGraphDebugDotPrint(graph, DOT_FILE_PATH_NEG,
                                hipGraphDebugDotFlagsVerbose-1);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass flag as hipGraphDebugDotFlags MAX + 1") {
    ret = hipGraphDebugDotPrint(graph, DOT_FILE_PATH_NEG,
                                hipGraphDebugDotFlagsHandles+1);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass flag as INT_MAX") {
    ret = hipGraphDebugDotPrint(graph, DOT_FILE_PATH_NEG, INT_MAX);
    REQUIRE(hipSuccess == ret);
  }
  HIP_CHECK(hipGraphDestroy(graph));
}

