/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
Testcase Scenarios :
 1) Add different kinds of nodes to graph and add dependencies to nodes.
 Verify sequence of graph execution is based on dependencies created.
 2) Negative Scenarios
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * Functional Test for adding dependencies in graph and verifying execution.
 */
TEST_CASE("Unit_hipGraphAddDependencies_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  hipMemsetParams memsetParams{};
  int memsetVal{};
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = A_d;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(int);
  memsetParams.width = N;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                                              &memsetParams));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(B_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(int);
  memsetParams.width = N;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_B, graph, nullptr, 0,
                                                              &memsetParams));

  void* kernelArgs1[] = {&C_d, &memsetVal, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func =
                       reinterpret_cast<void *>(HipTest::memsetReverse<uint32_t>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&memsetKer_C, graph, nullptr, 0,
                                                        &kernelNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h,
                                   Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h,
                                   Nbytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d,
                                   Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<uint32_t>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0,
                                                        &kernelNodeParams));

  SECTION("Create dependencies node by node") {
    // Create dependencies
    HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memset_B, &memcpyH2D_B, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kernel_vecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &memsetKer_C, &kernel_vecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_C, 1));
  }

  SECTION("Create dependencies with node lists") {
    hipGraphNode_t from_list[] = {memset_A, memset_B, memcpyH2D_A, memcpyH2D_B, memsetKer_C, kernel_vecAdd};
    hipGraphNode_t to_list[] = {memcpyH2D_A, memcpyH2D_B, kernel_vecAdd, kernel_vecAdd, kernel_vecAdd, memcpyD2H_C};
    // Create dependencies
    HIP_CHECK(hipGraphAddDependencies(graph, from_list, to_list, 6));
  }

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

TEST_CASE("Unit_hipGraphAddDependencies_Positive_Parameters") {
  constexpr size_t Nbytes = 1024;
  hipGraphNode_t memcpyH2D_A;
  hipGraphNode_t memcpyD2H_A;
  hipGraphNode_t memset_A;
  hipMemsetParams memsetParams{};
  char *A_d;
  char *A_h;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  A_h = reinterpret_cast<char*>(malloc(Nbytes));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;

  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                   &memsetParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr, 0, A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

  SECTION("numDependencies is zero, To/From are nullptr") {
    HIP_CHECK(hipGraphAddDependencies(graph, nullptr, nullptr, 0));
  }
  SECTION("numDependencies is zero, To or From are nullptr") {
    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, nullptr, 0));
    HIP_CHECK(hipGraphAddDependencies(graph, nullptr, &memcpyH2D_A, 0));
  }
  SECTION("numDependencies is zero, To/From are valid") {

    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &memcpyD2H_A, 0));
  }
  SECTION("numDependencies is zero, To/From are the same") {

    HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &memcpyH2D_A, 0));
  }
  SECTION("numDependencies < To/From length") {
    size_t numDependencies = 0;
    hipGraphNode_t from_list[] = {memset_A, memcpyH2D_A};
    hipGraphNode_t to_list[] = {memcpyH2D_A, memcpyD2H_A};
    HIP_CHECK(hipGraphAddDependencies(graph, from_list, to_list, 1));
    HIP_CHECK(hipGraphNodeGetDependencies(memcpyH2D_A, nullptr, &numDependencies));
    REQUIRE(numDependencies == 1);
    HIP_CHECK(hipGraphNodeGetDependencies(memcpyD2H_A, nullptr, &numDependencies));
    REQUIRE(numDependencies == 0);
  }

  // Destroy
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipGraphDestroy(graph));
  free(A_h);
}

/**
 * Negative Tests for hipGraphAddDependencies.
 */
TEST_CASE("Unit_hipGraphAddDependencies_Negative_Parameters") {
  // Initialize
  constexpr size_t Nbytes = 1024;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  char *A_d;
  hipGraphNode_t memset_A;
  hipMemsetParams memsetParams{};
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  hipGraphNode_t memcpyH2D_A;
  hipGraphNode_t memcpyD2H_A;
  char *A_h;
  A_h = reinterpret_cast<char*>(malloc(Nbytes));

  SECTION("Null Graph") {
    // Create dependencies
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                   &memsetParams));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0,
                                   A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipGraphAddDependencies(nullptr, &memset_A, &memcpyH2D_A, 1), hipErrorInvalidValue);
  }

  SECTION("graph is uninitialized") {
    hipGraph_t graph_uninit{};
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph_uninit, &memset_A, &memcpyH2D_A, 1), hipErrorInvalidValue);
  }

  SECTION("To or From is nullptr") {
    // Create dependencies
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                    &memsetParams));
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph, &memset_A, nullptr, 1), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph, nullptr, &memset_A, 1), hipErrorInvalidValue);
  }

  SECTION("To/From are Null Graph Node") {
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph, nullptr, nullptr, 1), hipErrorInvalidValue);
  }

  // The following tests fail on AMD.
  SECTION("From belongs to different graph") {
    hipGraph_t graph1;
    HIP_CHECK(hipGraphCreate(&graph1, 0));
    // Create dependencies
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph1, nullptr, 0,
                                    &memsetParams));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0,
                            A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1), hipErrorInvalidValue);
    HIP_CHECK(hipGraphDestroy(graph1));
  }

  SECTION("To belongs to different graph") {
    hipGraph_t graph1;
    HIP_CHECK(hipGraphCreate(&graph1, 0));
    // Create dependencies
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                &memsetParams));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph1, nullptr,
                                0, A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1), hipErrorInvalidValue);
    HIP_CHECK(hipGraphDestroy(graph1));
  }

  SECTION("From is uninitialized") {
    // Create dependencies
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr,
                                0, A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1), hipErrorInvalidValue);
  }

  SECTION("To is uninitialized") {
    // Create dependencies
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                    &memsetParams));
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1), hipErrorInvalidValue);
  }

  SECTION("Duplicate Dependencies") {
    // Create dependencies
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                          &memsetParams));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0,
                          A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1));
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1), hipErrorInvalidValue);
  }

  SECTION("Same Node Dependencies") {
    // Create dependencies
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                    &memsetParams));
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph, &memset_A, &memset_A, 1), hipErrorInvalidValue);
  }

  SECTION("numDependencies > To/From length") {
    // Create dependencies
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0,
                                   &memsetParams));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0,
                                   A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr, 0,
                                   A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
    hipGraphNode_t from_list[] = {memset_A, memcpyH2D_A};
    hipGraphNode_t to_list[] = {memcpyH2D_A, memcpyD2H_A};
    HIP_CHECK_ERROR(hipGraphAddDependencies(graph, from_list, to_list, 3), hipErrorInvalidValue);
  }

  // Destroy
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipGraphDestroy(graph));
  free(A_h);
}
