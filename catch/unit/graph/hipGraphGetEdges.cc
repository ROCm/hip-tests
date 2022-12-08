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
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :
 1) Add nodes to graph with dependencies defined. Call api and verify number
    of edges and from/to list returned corresponds to the dependencies defined.
 2) Pass from and to as nullptr and verify the api returns number of edges.
 3) Pass numEdges lesser than actual number and verify the api returns from/to
    list with requested number of edges.
 4) Pass numEdges greater than actual number and verify the remaining entries
    in from/to list are set to null and number of edges actually returned will
    be written to numEdges.
 5) Validate numEdges when 0 or 1 node is present in graph.
 6) Negative Test Cases
    - Input graph parameter is a nullptr.
    - From node parameter is a nullptr.
    - To node parameter is a nullptr.
    - numEdges parameter is a nullptr.
    - Input graph parameter is uninitialized.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

namespace {
inline constexpr size_t kNumOfEdges = 6;
}  // anonymous namespace

/**
 * Local Function to validate number of edges.
 */
static void validate_hipGraphGetEdges_fromto(size_t numEdgesToGet,
                                        int testnum,
                                        hipGraphNode_t *nodes_from,
                                        hipGraphNode_t *nodes_to,
                                        hipGraph_t graph) {
  int numEdges = static_cast<int>(numEdgesToGet);
  hipGraphNode_t *fromnode = new hipGraphNode_t[numEdges]{};
  hipGraphNode_t *tonode = new hipGraphNode_t[numEdges]{};
  hipGraphNode_t *expected_from_nodes = nodes_from;
  hipGraphNode_t *expected_to_nodes = nodes_to;
  HIP_CHECK(hipGraphGetEdges(graph, fromnode, tonode, &numEdgesToGet));
  bool nodeFound;
  int found_count = 0;
  for (int idx_from = 0; idx_from < kNumOfEdges; idx_from++) {
    nodeFound = false;
    int idx = 0;
    for (; idx < kNumOfEdges; idx++) {
      if (expected_from_nodes[idx_from] == fromnode[idx]) {
        nodeFound = true;
        break;
      }
    }
    if (nodeFound && (tonode[idx] == expected_to_nodes[idx_from])) {
      found_count++;
    }
  }
  // Validate
  if (testnum == 0) {
    REQUIRE(found_count == kNumOfEdges);
  } else if (testnum == 1) {
    REQUIRE(found_count == numEdges);
  } else if (testnum == 2) {
    REQUIRE(found_count == kNumOfEdges);
    for (int idx = (kNumOfEdges - 1); idx > (numEdges - 1); idx++) {
      REQUIRE(fromnode[idx] == nullptr);
      REQUIRE(tonode[idx] == nullptr);
    }
  }

  delete[] tonode;
  delete[] fromnode;
}

/**
 * Scenario 1: Finctionality tests to validate hipGraphGetEdges()
 * for different number of edges.
 */
TEST_CASE("Unit_hipGraphGetEdges_Positive_Functionality") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memset_A, memset_B, memsetKer_C;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipMemsetParams memsetParams{};
  int memsetVal{};
  size_t NElem{N};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
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
                       reinterpret_cast<void *>(HipTest::memsetReverse<int>);
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
  kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0,
                                                        &kernelNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_B, &memcpyH2D_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memsetKer_C, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_C, 1));

  hipGraphNode_t nodes_from[kNumOfEdges] = {memset_A, memset_B,
  memcpyH2D_A, memcpyH2D_B, memsetKer_C, kernel_vecAdd};
  hipGraphNode_t nodes_to[kNumOfEdges] = {memcpyH2D_A, memcpyH2D_B,
  kernel_vecAdd, kernel_vecAdd, kernel_vecAdd, memcpyD2H_C};
  // Validate hipGraphGetEdges() API
  // Scenario 1
  SECTION("Validate number of edges") {
    size_t numEdges = 0;
    HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
    REQUIRE(numEdges == kNumOfEdges);
  }
  // Scenario 2
  SECTION("Validate from/to list when numEdges = num of edges") {
    validate_hipGraphGetEdges_fromto(kNumOfEdges, 0,
                                    nodes_from, nodes_to, graph);
  }
  // Scenario 3
  SECTION("Validate from/to list when numEdges = less than num of edges") {
    validate_hipGraphGetEdges_fromto(kNumOfEdges - 1, 1,
                                    nodes_from, nodes_to, graph);
  }
  // Scenario 4
  SECTION("Validate from/to list when numEdges = more than num of edges") {
    validate_hipGraphGetEdges_fromto(kNumOfEdges + 1, 2,
                                    nodes_from, nodes_to, graph);
  }
  // Scenario 5
  SECTION("Validate number of edges when zero or one node in graph") {
    size_t numEdges = 0;
    hipGraph_t graphempty;
    HIP_CHECK(hipGraphCreate(&graphempty, 0));
    HIP_CHECK(hipGraphGetEdges(graphempty, nullptr, nullptr, &numEdges));
    REQUIRE(numEdges == 0);
    // Add an empty node
    hipGraphNode_t emptyNode{};
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graphempty, nullptr, 0));
    HIP_CHECK(hipGraphGetEdges(graphempty, nullptr, nullptr, &numEdges));
    REQUIRE(numEdges == 0);
    HIP_CHECK(hipGraphDestroy(graphempty));
  }
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Begin stream capture and push operations to stream.
 * Verify nodes of created graph are matching the operations pushed.
 */
TEST_CASE("Unit_hipGraphGetEdges_Positive_CapturedStream") {
  hipGraph_t graph{nullptr};
  constexpr unsigned threadsPerBlock = 256;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr size_t N = 1024;
  size_t Nbytes = N * sizeof(int);
  size_t NElem{N};
  int memsetVal{};
  constexpr int numMemcpy[2]{2, 3}, numKernel[2]{2, 3}, numMemset[2]{2, 0};
  hipEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
  int cntMemcpy[2]{}, cntKernel[2]{}, cntMemset[2]{};
  hipStream_t stream1, stream2, stream3;
  hipGraphNodeType nodeType;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  // Create streams and events
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));
  HIP_CHECK(hipEventCreate(&memsetEvent1));
  HIP_CHECK(hipEventCreate(&memsetEvent2));
  // Begin stream capture
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIP_CHECK(hipStreamWaitEvent(stream3, forkStreamEvent, 0));
  // Add operations to stream3
  hipLaunchKernelGGL(HipTest::memsetReverse<int>,
                    dim3(blocks), dim3(threadsPerBlock), 0, stream3,
                    C_d, memsetVal, NElem);
  HIP_CHECK(hipEventRecord(memsetEvent1, stream3));
  // Add operations to stream2
  HIP_CHECK(hipMemsetAsync(B_d, 0, Nbytes, stream2));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream2));
  HIP_CHECK(hipEventRecord(memsetEvent2, stream2));
  // Add operations to stream1
  HIP_CHECK(hipMemsetAsync(A_d, 0, Nbytes, stream1));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream1, memsetEvent2, 0));
  HIP_CHECK(hipStreamWaitEvent(stream1, memsetEvent1, 0));
  hipLaunchKernelGGL(HipTest::vectorADD<int>,
                    dim3(blocks), dim3(threadsPerBlock), 0, stream1,
                    A_d, B_d, C_d, NElem);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost,
                           stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  REQUIRE(graph != nullptr);

  size_t numEdges = 0;
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  REQUIRE(numEdges == kNumOfEdges);

  int numBytes = sizeof(hipGraphNode_t) * numEdges;
  hipGraphNode_t* from_nodes = reinterpret_cast<hipGraphNode_t *>(malloc(numBytes));
  REQUIRE(from_nodes != nullptr);
  hipGraphNode_t* to_nodes = reinterpret_cast<hipGraphNode_t *>(malloc(numBytes));
  REQUIRE(to_nodes != nullptr);

  HIP_CHECK(hipGraphGetEdges(graph, from_nodes, to_nodes, &numEdges));
  for (size_t i = 0; i < 2; i++) {
    hipGraphNode_t* current_nodes = (i == 0) ? from_nodes : to_nodes;
    for (size_t j = 0; j < numEdges; j++) {
    HIP_CHECK(hipGraphNodeGetType(current_nodes[j], &nodeType));
      switch (nodeType) {
        case hipGraphNodeTypeMemcpy:
          cntMemcpy[i]++;
          break;

        case hipGraphNodeTypeKernel:
          cntKernel[i]++;
          break;

        case hipGraphNodeTypeMemset:
          cntMemset[i]++;
          break;

        default:
          INFO("Unexpected nodetype returned : " << nodeType);
          REQUIRE(false);
     }
    }
    REQUIRE(cntMemcpy[i] == numMemcpy[i]);
    REQUIRE(cntKernel[i] == numKernel[i]);
    REQUIRE(cntMemset[i] == numMemset[i]);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipEventDestroy(forkStreamEvent));
  HIP_CHECK(hipEventDestroy(memsetEvent1));
  HIP_CHECK(hipEventDestroy(memsetEvent2));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * Scenario 5: Negative Test Cases
 */
TEST_CASE("Unit_hipGraphGetEdges_Negative_Parameters") {
  hipGraph_t graph{}, graph_uninit{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t nodes_from[kNumOfEdges]{},
                nodes_to[kNumOfEdges]{};
  size_t numEdges = 0;
  SECTION("graph is nullptr") {
    HIP_CHECK_ERROR(hipGraphGetEdges(nullptr, nodes_from, nodes_to, &numEdges), hipErrorInvalidValue);
  }

  SECTION("graph is uninitialized") {
    HIP_CHECK_ERROR(hipGraphGetEdges(graph_uninit, nodes_from, nodes_to, &numEdges), hipErrorInvalidValue);
  }

  SECTION("From is nullptr") {
    HIP_CHECK_ERROR(hipGraphGetEdges(graph, nullptr, nodes_to, &numEdges), hipErrorInvalidValue);
  }

  SECTION("To is nullptr") {
    HIP_CHECK_ERROR(hipGraphGetEdges(graph, nodes_from, nullptr, &numEdges), hipErrorInvalidValue);
  }
  SECTION("numEdges is nullptr") {
    HIP_CHECK_ERROR(hipGraphGetEdges(graph, nodes_from, nodes_to, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}
