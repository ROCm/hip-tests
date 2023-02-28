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

/**
Negative Testcase Scenarios for api hipGraphAddMemsetNode :
1) Pass pGraphNode as nullptr and check if api returns error.
2) Pass pGraphNode as un-initialize object and check.
3) Pass Graph as nullptr and check if api returns error.
4) Pass Graph as empty object(skipping graph creation), api should return error code.
5) Pass pDependencies as nullptr, api should return success.
6) Pass numDependencies is max(size_t) and pDependencies is not valid ptr, api expected to return error code.
7) Pass pDependencies is nullptr, but numDependencies is non-zero, api expected to return error.
8) Pass pMemsetParams as nullptr and check if api returns error code.
9) Pass pMemsetParams as un-initialize object and check if api returns error code.
10) Pass hipMemsetParams::dst as nullptr should return error code.
11) Pass hipMemsetParams::element size other than 1, 2, or 4 and check api should return error code.
12) Pass hipMemsetParams::height as zero and check api should return error code.
Functional Scenarios for api hipGraphAddMemsetNode :
1. Allocate a 2D array using hipMallocPitch. Initialize the allocated memory using hipGraphAddMemsetNode.
   Copy the values in device memory to host using hipGraphAddMemcpyNode. Verify the results
2. Allocate a 1D array using hipMallocPitch. Initialize the allocated memory using hipGraphAddMemsetNode.
   Copy the values in device memory to host using hipGraphAddMemcpyNode. Verify the results..
3. Allocate a 2D array using hipMalloc3D. Initialize the allocated memory using hipGraphAddMemsetNode.
   Copy the values in device memory to host using hipGraphAddMemcpyNode. Verify the results.
4. Allocate a 1D array using hipMalloc3D. Initialize the allocated memory using hipGraphAddMemsetNode.
   Copy the values in device memory to host using hipGraphAddMemcpyNode. Verify the results.
5. Allocate a 1D array using hipMalloc. Initialize the allocated memory using hipGraphAddMemsetNode.
   Copy the values in device memory to host using hipGraphAddMemcpyNode. Verify the results.
6. Allocate memory using hipMallocManaged. Initialize the allocated memory using hipGraphAddMemsetNode.
   Copy the values in device memory to host using hipGraphAddMemcpyNode. Verify the results.
*/

#include <hip_test_common.hh>
/**
 * Negative Test for API hipGraphAddMemsetNode
 */
#define SIZE 1024
static char memSetVal = 'a';
TEST_CASE("Unit_hipGraphAddMemsetNode_Negative") {
  hipError_t ret;
  hipGraph_t graph;
  hipGraphNode_t memsetNode;
  char *devData;

  HIP_CHECK(hipMalloc(&devData, 1024));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(devData);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = 1024;
  memsetParams.height = 1;

  SECTION("Pass pGraphNode as nullptr") {
    ret = hipGraphAddMemsetNode(nullptr, graph, nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pGraphNode as un-initialize object") {
    hipGraphNode_t memsetNode_1;
    ret = hipGraphAddMemsetNode(&memsetNode_1, graph,
                                nullptr, 0, &memsetParams);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass graph as nullptr") {
    ret = hipGraphAddMemsetNode(&memsetNode, nullptr,
                                nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass Graph as empty object") {
    hipGraph_t graph_1{};
    ret = hipGraphAddMemsetNode(&memsetNode, graph_1,
                                nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pDependencies as nullptr") {
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass numDependencies is max and pDependencies is not valid ptr") {
    ret = hipGraphAddMemsetNode(&memsetNode, graph,
                                nullptr, INT_MAX, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pDependencies as nullptr, but numDependencies is non-zero") {
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 9, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pMemsetParams as nullptr") {
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass pMemsetParams as un-initialize object") {
    hipMemsetParams memsetParams1;
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                &memsetParams1);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hipMemsetParams::dst as nullptr") {
    memsetParams.dst = nullptr;
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hipMemsetParams::element size other than 1, 2, or 4") {
    memsetParams.dst = reinterpret_cast<void*>(devData);
    memsetParams.elementSize = 9;
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hipMemsetParams::height as zero") {
    memsetParams.elementSize = sizeof(char);
    memsetParams.height = 0;
    ret = hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipFree(devData));
  HIP_CHECK(hipGraphDestroy(graph));
}
/*
 * Allocate a 2D array using hipMallocPitch. Initialize the allocated memory
 * using hipGraphAddMemsetNode. Copy the values in device memory to host using
 * hipGraphAddMemcpyNode. Verify the results.
*/
TEST_CASE("Unit_hipGraphAddMemsetNode_hipMallocPitch_2D") {
  size_t width = SIZE * sizeof(char), numW{SIZE},
         numH{SIZE}, pitch_A;
  char *A_d;

  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  // Host memory.
  char* A_h = new char[numW * numH];
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      *(A_h + i * numH + j) = ' ';
    }
  }
  // 2D Memory allocation hipMallocPitch
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width,
                          numH));
  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;
  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void *>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = pitch_A;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = numH;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
          &memsetParams));
  nodeDependencies.push_back(memsetNode);
  // Add MemCpy Node
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(A_d, pitch_A, numW, numH);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.extent = make_hipExtent(width, numH, 1);
  myparms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                            nodeDependencies.size(), &myparms));
  nodeDependencies.clear();
  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr,
                                nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      REQUIRE(*(A_h + i*numH + j) == memSetVal);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  delete[] A_h;
  HIP_CHECK(hipFree(A_d));
}
/*
 * Allocate a 1D array using hipMallocPitch. Initialize the allocated memory using
 * hipGraphAddMemsetNode. Copy the values in device memory to host using
 * hipGraphAddMemcpyNode. Verify the results.
*/
TEST_CASE("Unit_hipGraphAddMemsetNode_hipMallocPitch_1D") {
  size_t width = SIZE * sizeof(char), numW{SIZE}, pitch_A;
  char *A_d;

  // Initialize the host memory
  std::vector<char> A_h(numW, ' ');

  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  // 1D Memory allocation hipMallocPitch
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width,
                          1));
  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;
  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void *>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = pitch_A;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
          &memsetParams));
  nodeDependencies.push_back(memsetNode);
  // Add MemCpy Node
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(A_d, pitch_A, numW, 1);
  myparms.dstPtr = make_hipPitchedPtr(A_h.data(), width, numW, 1);
  myparms.extent = make_hipExtent(width, 1, 1);
  myparms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                            nodeDependencies.size(), &myparms));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr,
                                nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    REQUIRE(A_h[i] == memSetVal);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
}
/*
 * Allocate a 2D array using hipMalloc3D. Initialize the allocated memory using
 * hipGraphAddMemsetNode. Copy the values in device memory to host using
 * hipGraphAddMemcpyNode. Verify the results.
*/
TEST_CASE("Unit_hipGraphAddMemsetNode_hipMalloc3D_2D") {
  size_t width = SIZE * sizeof(char);
  size_t numW = SIZE, numH = SIZE;

  // Host Memory
  char* A_h = new char[numW * numH];
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      *(A_h + i * numH + j) = ' ';
    }
  }
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;

  hipPitchedPtr A_d;
  hipExtent extent3D = make_hipExtent(width, numH, 1);

  // Allocate 3D memory.
  HIPCHECK(hipMalloc3D(&A_d, extent3D));

  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;

  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = A_d.ptr;
  memsetParams.value = memSetVal;
  memsetParams.pitch = A_d.pitch;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = numH;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
          &memsetParams));
  nodeDependencies.push_back(memsetNode);

  // MemCpy params
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = A_d;
  myparms.dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  myparms.extent = make_hipExtent(width, numH, 1);
  myparms.kind = hipMemcpyDeviceToHost;

  // Add MemCpy Node
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                            nodeDependencies.size(), &myparms));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr,
                                nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      REQUIRE(*(A_h + i*numH + j) == memSetVal);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  delete[] A_h;
  HIP_CHECK(hipFree(A_d.ptr));
}
/*
 * Allocate a 1D array using hipMalloc3D. Initialize the allocated
 * memory using hipGraphAddMemsetNode. Copy the values in device
 * memory to host using hipGraphAddMemcpyNode. Verify the results.
*/
TEST_CASE("Unit_hipGraphAddMemsetNode_hipMalloc3D_1D") {
  size_t width = SIZE * sizeof(char);
  size_t numW = SIZE;

  // Initialize the host memory
  std::vector<char> A_h(numW, ' ');

  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;

  hipPitchedPtr A_d;
  hipExtent extent1D = make_hipExtent(width, 1, 1);

  // Allocate 3D memory.
  HIPCHECK(hipMalloc3D(&A_d, extent1D));

  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;

  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = A_d.ptr;
  memsetParams.value = memSetVal;
  memsetParams.pitch = A_d.pitch;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
          &memsetParams));
  nodeDependencies.push_back(memsetNode);

  // MemCpy params
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = A_d;
  myparms.dstPtr = make_hipPitchedPtr(A_h.data(), width, numW, 1);
  myparms.extent = make_hipExtent(width, 1, 1);
  myparms.kind = hipMemcpyDeviceToHost;

  // Add MemCpy Node
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                            nodeDependencies.size(), &myparms));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr,
                                nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
     REQUIRE(A_h[i] == memSetVal);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph))
  HIP_CHECK(hipFree(A_d.ptr));
}
/*
 * Allocate a 1D array using hipMalloc. Initialize the allocated memory using
 * hipGraphAddMemsetNode. Copy the values in device memory to host using
 * hipGraphAddMemcpyNode. Verify the results.
*/
TEST_CASE("Unit_hipGraphAddMemsetNode_hipMalloc_1D") {
  char *A_d;
  size_t NumW = SIZE;
  size_t Nbytes1D = SIZE * sizeof(char);

  // Initialize the host memory
  std::vector<char> A_h(NumW, ' ');

  // Allocate memory to Device pointer
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes1D));

  // Create the graph
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memsetNode, memcpyNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Add Memset node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void *>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = Nbytes1D;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = NumW;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
            &memsetParams));
  nodeDependencies.push_back(memsetNode);
  // Add MemCpy Node
  hipPitchedPtr devPitchedPtr{A_d, Nbytes1D, NumW, 0};
  hipPitchedPtr hostPitchedPtr{A_h.data(), Nbytes1D, NumW, 0};
  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = devPitchedPtr;
  myparms.dstPtr = hostPitchedPtr;
  myparms.extent = make_hipExtent(Nbytes1D, 1, 1);
  myparms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                            nodeDependencies.size(), &myparms));
  nodeDependencies.clear();
  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr,
                                nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < NumW; i++) {
     REQUIRE(A_h[i] == memSetVal);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
}

TEST_CASE("Unit_hipGraphAddMemsetNode_hipMallocManaged") {
  int managed = 0;
  HIP_CHECK(hipDeviceGetAttribute(&managed,
                                  hipDeviceAttributeManagedMemory, 0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);
  if (managed != 1) {
    WARN(
      "GPU 0 doesn't support hipDeviceAttributeManagedMemory attribute"
       "so defaulting to system memory.");
  }
  size_t Nbytes1D = SIZE * sizeof(char);
  char *A_d;
  // Initialize the host memory
  std::vector<char> A_h(SIZE, ' ');
  // Device Memory
  HIP_CHECK(hipMallocManaged(&A_d, SIZE * sizeof(char)));
  // Create the graph
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memsetNode, memcpyNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Add Memset node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void *>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = Nbytes1D;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = SIZE;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
            &memsetParams));
  nodeDependencies.push_back(memsetNode);

  // Add MemCpy Node
  hipPitchedPtr devPitchedPtr{A_d, Nbytes1D, SIZE, 1};
  hipPitchedPtr hostPitchedPtr{A_h.data(), Nbytes1D, SIZE, 1};

  hipMemcpy3DParms myparms{};
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = devPitchedPtr;
  myparms.dstPtr = hostPitchedPtr;
  myparms.extent = make_hipExtent(Nbytes1D, 1, 1);
  myparms.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                            nodeDependencies.size(), &myparms));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr,
                                nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < SIZE; i++) {
     REQUIRE(A_h[i] == memSetVal);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));
}
