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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/**
 * @addtogroup hipGraphLaunch hipGraphLaunch
 * @{
 * @ingroup GraphTest
 * `hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream)` -
 * Launches an executable graph in a stream.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipGraph_BasicFunctional
 *  - @ref Unit_hipGraph_SimpleGraphWithKernel
 */

static void HostFunctionSetToZero(void* arg) {
  int* test_number = (int*)arg;
  (*test_number) = 0;
}

static void HostFunctionAddOne(void* arg) {
  int* test_number = (int*)arg;
  (*test_number) += 1;
}

/* create an executable graph that will set an integer pointed to by 'number' to one*/
static void CreateTestExecutableGraph(hipGraphExec_t* graph_exec, int* number) {
  hipGraph_t graph;
  hipGraphNode_t node_error;

  hipGraphNode_t node_set_zero;
  hipHostNodeParams params_set_to_zero = {HostFunctionSetToZero, number};

  hipGraphNode_t node_add_one;
  hipHostNodeParams params_set_add_one = {HostFunctionAddOne, number};

  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddHostNode(&node_set_zero, graph, nullptr, 0, &params_set_to_zero));
  HIP_CHECK(hipGraphAddHostNode(&node_add_one, graph, &node_set_zero, 1, &params_set_add_one));

  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, &node_error, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));
}

static int HipGraphLaunch_Positive_Simple(hipStream_t stream) {
  int number = 5;

  hipGraphExec_t graph_exec;
  CreateTestExecutableGraph(&graph_exec, &number);

  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(number == 1);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
}

/**
 * Test Description
 * ------------------------
 *  - Validates several basic scenarios:
 *    -# When graph is launched with a regular, created stream
 *    -# When graph is launched with a per thread stream
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphLaunch_Positive") {
  SECTION("stream as a created stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HipGraphLaunch_Positive_Simple(stream);
    HIP_CHECK(hipStreamDestroy(stream));
  }

  SECTION("with stream as hipStreamPerThread") {
    HipGraphLaunch_Positive_Simple(hipStreamPerThread);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When graph handle is `nullptr` and stream is a created stream
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When graph handle is `nullptr` and stream is per thread
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When graph handle is an empty object
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When graph handle is destroyed before calling launch
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("hipGraphLaunch_Negative_Parameters") {
  SECTION("graphExec is nullptr and stream is a created stream") {
    hipStream_t stream;
    hipError_t ret;
    HIP_CHECK(hipStreamCreate(&stream));
    ret = hipGraphLaunch(nullptr, stream);
    HIP_CHECK(hipStreamDestroy(stream));
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("graphExec is nullptr and stream is hipStreamPerThread") {
    HIP_CHECK_ERROR(hipGraphLaunch(nullptr, hipStreamPerThread), hipErrorInvalidValue);
  }

  SECTION("graphExec is an empty object") {
    hipGraphExec_t graph_exec{};
    HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, hipStreamPerThread), hipErrorInvalidValue);
  }

  SECTION("graphExec is destroyed") {
    int number = 5;
    hipGraphExec_t graph_exec;
    CreateTestExecutableGraph(&graph_exec, &number);
    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    REQUIRE(number == 1);
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, hipStreamPerThread), hipErrorInvalidValue);
  }
}

// Function to fill input data
static void fillRandInpData(int* A1_h, int* A2_h, size_t N) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < N; i++) {
    A1_h[i] = (HipTest::RAND_R(&seed) & 0xFF);
    A2_h[i] = (HipTest::RAND_R(&seed) & 0xFF);
  }
}
// Function to validate result
static void validateOutData(int* A1_h, int* A2_h, size_t N) {
  for (size_t i = 0; i < N; i++) {
    int result = (A1_h[i] * A1_h[i]);
    REQUIRE(result == A2_h[i]);
  }
}
/*
 * 1.Create a graph with multiple nodes. Create an executable graph.
 * Launch the executable graph 3 times in stream simultaneously.
 * Wait for stream. Validate the output. No issues should be observed
 * 2.Create a graph with multiple nodes. Create an executable graph.
 * Verify if an executable graph be launched on null stream.
 */
TEST_CASE("Unit_hipGraphLaunch_Functional_MultipleLaunch") {
  size_t memSize = SIZE;
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, SIZE);
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  int *A_h{nullptr}, *A_d{nullptr}, *C_d{nullptr}, *C_h{nullptr};

  HipTest::initArrays<int>(&A_d, &C_d, nullptr, &A_h, &C_h, nullptr, SIZE, false);

  hipGraphNode_t memcpyH2D, memcpyD2H, kernelNode;

  // Create memcpy H2D nodes
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D, graph, nullptr, 0, A_d, A_h, (sizeof(int) * SIZE),
                                    hipMemcpyHostToDevice));
  nodeDependencies.push_back(memcpyH2D);
  // Creating kernel node
  hipKernelNodeParams kerNodeParams;
  void* kernelArgs[] = {reinterpret_cast<void*>(&A_d), reinterpret_cast<void*>(&C_d),
                        reinterpret_cast<void*>(&memSize)};
  kerNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kerNodeParams.gridDim = dim3(blocks);
  kerNodeParams.blockDim = dim3(threadsPerBlock);
  kerNodeParams.sharedMemBytes = 0;
  kerNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kerNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                                  nodeDependencies.size(), &kerNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  // Create memcpy D2H nodes
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph, nodeDependencies.data(),
                                    nodeDependencies.size(), C_h, C_d, (sizeof(int) * SIZE),
                                    hipMemcpyDeviceToHost));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  // Execute graph
  SECTION("Multiple Graph Launch") {
    for (int iter = 0; iter < TEST_LOOP_SIZE; iter++) {
      fillRandInpData(A_h, C_h, SIZE);
      HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
      HIP_CHECK(hipStreamSynchronize(streamForGraph));
      validateOutData(A_h, C_h, SIZE);
    }
  }
  SECTION("Graph launch on Null stream") {
    for (int iter = 0; iter < TEST_LOOP_SIZE; iter++) {
      fillRandInpData(A_h, C_h, SIZE);
      HIP_CHECK(hipGraphLaunch(graphExec, 0));
      HIP_CHECK(hipStreamSynchronize(0));
      validateOutData(A_h, C_h, SIZE);
    }
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));

  // Free
  HipTest::freeArrays<int>(A_d, C_d, nullptr, A_h, C_h, nullptr, false);
}
