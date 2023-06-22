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
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * @addtogroup hipGraphDestroyNode hipGraphDestroyNode
 * @{
 * @ingroup GraphTest
 * `hipGraphDestroyNode(hipGraphNode_t node)` -
 * Remove a node from the graph.
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When node handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphDestroyNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphDestroyNode_Negative") {
  SECTION("Passing nullptr to graph Node") {
    REQUIRE(hipGraphDestroyNode(nullptr) == hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Creates graph.
 *  - Destroys one of the nodes successfully.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphDestroyNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphDestroyNode_BasicFunctionality") {
  char* pOutBuff_d{};
  constexpr size_t size = 1024;
  hipGraph_t graph{};
  hipGraphNode_t memsetNode{};

  HIP_CHECK(hipMalloc(&pOutBuff_d, size));
  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(pOutBuff_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = size * sizeof(char);
  memsetParams.height = 1;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
  REQUIRE(hipGraphDestroyNode(memsetNode) == hipSuccess);
  HIP_CHECK(hipFree(pOutBuff_d));
}

/**
 * Test Description
 * ------------------------
 *  - Creates graph with dependencies.
 *  - Destroys one of the dependency nodes.
 *  - Executes the graph.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphDestroyNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphDestroyNode_DestroyDependencyNode") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyH2D_B2Copies, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;
  size_t NElem{N};
  hipStream_t streamForGraph;

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B2Copies, graph, nullptr, 0, B_d, C_h, Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, B_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0, &kernelNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B2Copies, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_C, 1));

  // Destroy one of the dependency node
  HIP_CHECK(hipGraphDestroyNode(memcpyH2D_B));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, C_h, B_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Functional Test to test hipGraphDestroyNode using hipGraphGetNodes
 * and hipGraphGetEdges APIs.
 */
TEST_CASE("Unit_hipGraphDestroyNode_Complx_ChkNumOfNodesNDep") {
  hipGraph_t graph;
  hipGraphNode_t kernelnode[NUM_OF_DUMMY_NODES];
  hipKernelNodeParams kernelNodeParams[NUM_OF_DUMMY_NODES];
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create graph with no dependencies
  for (int i = 0; i < NUM_OF_DUMMY_NODES; i++) {
    void* kernelArgs[] = {nullptr};
    kernelNodeParams[i].func = reinterpret_cast<void*>(dummyKernel);
    kernelNodeParams[i].gridDim = dim3(1);
    kernelNodeParams[i].blockDim = dim3(1);
    kernelNodeParams[i].sharedMemBytes = 0;
    kernelNodeParams[i].kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams[i].extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kernelnode[i], graph, nullptr, 0, &kernelNodeParams[i]));
  }
  // Create dependencies between nodes
  for (int i = 1; i < NUM_OF_DUMMY_NODES; i++) {
    HIP_CHECK(hipGraphAddDependencies(graph, &kernelnode[i - 1], &kernelnode[i], 1));
  }
  // Start destroying nodes from 0
  size_t numOfNodes = 0, numOfDep = 0;
  for (size_t i = 0; i < (NUM_OF_DUMMY_NODES - 1); i++) {
    // destroy node i
    HIP_CHECK(hipGraphDestroyNode(kernelnode[i]));
    HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numOfNodes));
    REQUIRE(numOfNodes == (NUM_OF_DUMMY_NODES - i - 1));
    HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numOfDep));
    REQUIRE(numOfDep == (NUM_OF_DUMMY_NODES - i - 2));
  }
  HIP_CHECK(hipGraphDestroyNode(kernelnode[NUM_OF_DUMMY_NODES - 1]));
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numOfNodes));
  REQUIRE(numOfNodes == 0);
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Functional Test to test hipGraphDestroyNode using hipGraphGetNodes
 * and hipGraphGetEdges APIs on a cloned graph
 */
TEST_CASE("Unit_hipGraphDestroyNode_Complx_ChkNumOfNodesNDep_ClonedGrph") {
  hipGraph_t graph, clonedgraph;
  hipGraphNode_t kernelnode[NUM_OF_DUMMY_NODES];
  hipKernelNodeParams kernelNodeParams[NUM_OF_DUMMY_NODES];
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphCreate(&clonedgraph, 0));
  // Create graph with no dependencies
  for (int i = 0; i < NUM_OF_DUMMY_NODES; i++) {
    void* kernelArgs[] = {nullptr};
    kernelNodeParams[i].func = reinterpret_cast<void*>(dummyKernel);
    kernelNodeParams[i].gridDim = dim3(1);
    kernelNodeParams[i].blockDim = dim3(1);
    kernelNodeParams[i].sharedMemBytes = 0;
    kernelNodeParams[i].kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams[i].extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kernelnode[i], graph, nullptr, 0, &kernelNodeParams[i]));
  }
  // Create dependencies between nodes
  for (int i = 1; i < NUM_OF_DUMMY_NODES; i++) {
    HIP_CHECK(hipGraphAddDependencies(graph, &kernelnode[i - 1], &kernelnode[i], 1));
  }
  HIP_CHECK(hipGraphClone(&clonedgraph, graph));
  // Start destroying nodes from 0 and validate number of nodes in
  // cloned graph
  size_t numOfNodes = 0, numOfDep = 0;
  for (size_t i = 0; i < (NUM_OF_DUMMY_NODES - 1); i++) {
    hipGraphNode_t node;
    // destroy node i
    HIP_CHECK(hipGraphNodeFindInClone(&node, kernelnode[i], clonedgraph));
    HIP_CHECK(hipGraphDestroyNode(node));
    HIP_CHECK(hipGraphGetNodes(clonedgraph, nullptr, &numOfNodes));
    REQUIRE(numOfNodes == (NUM_OF_DUMMY_NODES - i - 1));
    HIP_CHECK(hipGraphGetEdges(clonedgraph, nullptr, nullptr, &numOfDep));
    REQUIRE(numOfDep == (NUM_OF_DUMMY_NODES - i - 2));
  }
  // Verify the number of nodes in original graph
  numOfNodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numOfNodes));
  REQUIRE(numOfNodes == NUM_OF_DUMMY_NODES);
  HIP_CHECK(hipGraphDestroy(clonedgraph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Functional Test to test hipGraphDestroyNode on child node using
 * hipGraphGetNodes and hipGraphGetEdges APIs on a cloned graph.
 */
TEST_CASE("Unit_hipGraphDestroyNode_Complx_ChkNumOfNodesNDep_ChldNode") {
  hipGraph_t graph0, graph1;
  hipGraphNode_t kernelnode[NUM_OF_DUMMY_NODES], childGraphNode;
  hipKernelNodeParams kernelNodeParams[NUM_OF_DUMMY_NODES];
  HIP_CHECK(hipGraphCreate(&graph0, 0));
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  // Create graph with no dependencies
  for (int i = 0; i < NUM_OF_DUMMY_NODES; i++) {
    void* kernelArgs[] = {nullptr};
    kernelNodeParams[i].func = reinterpret_cast<void*>(dummyKernel);
    kernelNodeParams[i].gridDim = dim3(1);
    kernelNodeParams[i].blockDim = dim3(1);
    kernelNodeParams[i].sharedMemBytes = 0;
    kernelNodeParams[i].kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams[i].extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kernelnode[i], graph0, nullptr, 0, &kernelNodeParams[i]));
  }
  // Create dependencies between nodes
  for (int i = 1; i < NUM_OF_DUMMY_NODES; i++) {
    HIP_CHECK(hipGraphAddDependencies(graph0, &kernelnode[i - 1], &kernelnode[i], 1));
  }
  // Create child node and add it to graph1
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph1, nullptr, 0, graph0));
  // delete the child node from graph1
  HIP_CHECK(hipGraphDestroyNode(childGraphNode));
  // Start destroying nodes from 0
  size_t numOfNodes = 0, numOfDep = 0;
  for (size_t i = 0; i < (NUM_OF_DUMMY_NODES - 1); i++) {
    // destroy node i
    HIP_CHECK(hipGraphDestroyNode(kernelnode[i]));
    HIP_CHECK(hipGraphGetNodes(graph0, nullptr, &numOfNodes));
    REQUIRE(numOfNodes == (NUM_OF_DUMMY_NODES - i - 1));
    HIP_CHECK(hipGraphGetEdges(graph0, nullptr, nullptr, &numOfDep));
    REQUIRE(numOfDep == (NUM_OF_DUMMY_NODES - i - 2));
  }
  HIP_CHECK(hipGraphGetNodes(graph1, nullptr, &numOfNodes));
  REQUIRE(numOfNodes == 0);
  HIP_CHECK(hipGraphDestroy(graph0));
  HIP_CHECK(hipGraphDestroy(graph1));
}
