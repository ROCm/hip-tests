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

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

/**
 * @addtogroup hipGraphExecEventWaitNodeSetEvent hipGraphExecEventWaitNodeSetEvent
 * @{
 * @ingroup GraphTest
 * `hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec,
 * hipGraphNode_t hNode, hipEvent_t event)` -
 * Sets the event for an event record node in the given graphExec.
 */

// Kernel Functions to perform square and introduce delay in device.
static __global__ void sqr_ker_func(int* a, int* b, size_t N, int clockrate, size_t delayMs) {
  int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (tx < N) b[tx] = a[tx] * a[tx];
  uint64_t wait_t = delayMs, start = clock64() / clockrate, cur;
  do {
    cur = clock64() / clockrate - start;
  } while (cur < wait_t);
}

static __global__ void sqr_ker_func_gfx11(int* a, int* b, int clockrate) {
#if HT_AMD
  int tx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (tx < LEN) b[tx] = a[tx] * a[tx];
  uint64_t wait_t = DELAY_IN_MS, start = wall_clock64() / clockrate, cur;
  do {
    cur = wall_clock64() / clockrate - start;
  } while (cur < wait_t);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Create a graph with event record nodes as follows:
 *    -# Launch kernel.
 *    -# Record event.
 *  - Instantiate graph.
 *  - Create another graph with event record nodes as follows:
 *    -# Create wait event node on event
 *  - Instantiate the second graph.
 *  - Change the event in first graph to the second event
 *  - Change the event in second graph to the second event
 *  - Execute both graphs.
 *  - Verify the kernel results match.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecEventWaitNodeSetEvent.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecEventWaitNodeSetEvent_SetAndVerifyMemory") {
  constexpr size_t gridSize = 64;
  constexpr size_t blockSize = 256;
  constexpr size_t N = gridSize * blockSize;
  size_t memsize = N * sizeof(int);
  hipGraph_t graph1, graph2;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  // Create events
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  // Create nodes with event_start and event1_end
  hipGraphNode_t event_rec;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_rec, graph1, nullptr, 0, event1));
  int *inp_h, *inp_d, *out_h, *out_d;
  // Allocate host buffers
  inp_h = reinterpret_cast<int*>(malloc(memsize));
  REQUIRE(inp_h != nullptr);
  out_h = reinterpret_cast<int*>(malloc(memsize));
  REQUIRE(out_h != nullptr);
  // Allocate device buffers
  HIP_CHECK(hipMalloc(&inp_d, memsize));
  HIP_CHECK(hipMalloc(&out_d, memsize));
  // Initialize host buffer
  for (uint32_t i = 0; i < N; i++) {
    inp_h[i] = i;
    out_h[i] = 0;
  }
  // graph1 creation ...........
  // Create memcpy and kernel nodes for graph1
  // MemcpyH2D -> kernel1 -> event_rec
  hipGraphNode_t memcpyH2D, kernelnode1;
  hipKernelNodeParams kernelNodeParams1{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D, graph1, nullptr, 0, inp_d, inp_h, memsize,
                                    hipMemcpyHostToDevice));
  // Get device clock rate
  int clkRate = 0;
  size_t NElem{N};
  size_t delayMs{2000};
  HIPCHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
  // kernel1
  void* kernelArgs[] = {&inp_d, &out_d, reinterpret_cast<void*>(&NElem),
                        reinterpret_cast<void*>(&clkRate), reinterpret_cast<void*>(&delayMs)};
  kernelNodeParams1.func = reinterpret_cast<void*>(sqr_ker_func);
  kernelNodeParams1.gridDim = dim3(gridSize);
  kernelNodeParams1.blockDim = dim3(blockSize);
  kernelNodeParams1.sharedMemBytes = 0;
  kernelNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams1.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelnode1, graph1, nullptr, 0, &kernelNodeParams1));
  // Create dependencies for graph1
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpyH2D, &kernelnode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &kernelnode1, &event_rec, 1));
  // graph2 creation ...........
  // waitnode(event1) -> MemcpyD2H
  hipGraphNode_t event_wait_node, memcpyD2H;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph2, nullptr, 0, out_h, out_d, memsize,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddEventWaitNode(&event_wait_node, graph2, nullptr, 0, event1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &event_wait_node, &memcpyD2H, 1));
  // Instantiate graph1 and graph2
  hipStream_t streamForGraph1, streamForGraph2;
  hipGraphExec_t graphExec1, graphExec2;
  HIP_CHECK(hipStreamCreate(&streamForGraph1));
  HIP_CHECK(hipStreamCreate(&streamForGraph2));
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  // Launch graph1 and graph2
  HIP_CHECK(hipGraphLaunch(graphExec1, streamForGraph1));
  HIP_CHECK(hipGraphLaunch(graphExec2, streamForGraph2));
  // Wait for graph2 to complete
  HIP_CHECK(hipStreamSynchronize(streamForGraph2));
  // Validate output
  bool btestPassed = true;
  for (uint32_t i = 0; i < N; i++) {
    if (out_h[i] != (inp_h[i] * inp_h[i])) {
      btestPassed = false;
      break;
    }
  }
  REQUIRE(btestPassed == true);
  // hipGraphExecEventWaitNodeSetEvent() TEST
  // Change the event at event_wait_node node to event2 and
  // the event at event_rec node to event2.
  HIP_CHECK(hipGraphExecEventRecordNodeSetEvent(graphExec1, event_rec, event2));
  HIP_CHECK(hipGraphExecEventWaitNodeSetEvent(graphExec2, event_wait_node, event2));
  // Launch graph1 and graph2
  HIP_CHECK(hipGraphLaunch(graphExec1, streamForGraph1));
  HIP_CHECK(hipGraphLaunch(graphExec2, streamForGraph2));
  // Wait for graph2 to complete
  HIP_CHECK(hipStreamSynchronize(streamForGraph2));
  // Validate output
  btestPassed = true;
  for (uint32_t i = 0; i < N; i++) {
    if (out_h[i] != (inp_h[i] * inp_h[i])) {
      btestPassed = false;
      break;
    }
  }
  REQUIRE(btestPassed == true);
  // Free resources
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipStreamDestroy(streamForGraph1));
  HIP_CHECK(hipStreamDestroy(streamForGraph2));
  HIP_CHECK(hipFree(inp_d));
  HIP_CHECK(hipFree(out_d));
  free(inp_h);
  free(out_h);
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *  - Scenario to verify that event set does not
 *    impact the graph and changes only the executable graph.
 *  - Create an event wait node with first event and add it to graph.
 *  - Instantiate the graph to create an executable graph.
 *  - Change the event in the executable graph to the second event.
 *  - Verify that the event wait node still contains first event.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecEventWaitNodeSetEvent.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecEventWaitNodeSetEvent_VerifyEventNotChanged") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event1, event2, event_out;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t eventwait;
  HIP_CHECK(hipGraphAddEventWaitNode(&eventwait, graph, nullptr, 0, event1));
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphExecEventWaitNodeSetEvent(graphExec, eventwait, event2));
  HIP_CHECK(hipGraphEventWaitNodeGetEvent(eventwait, &event_out));
  // validate set event and get event are same
  REQUIRE(event1 == event_out);
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event2));
  HIP_CHECK(hipEventDestroy(event1));
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When executable graph handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node handle is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pointer to the event is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When executable graph is not initialized
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node is not initialized
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When event is not initialized
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When event wait node does not exist
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node is memset node
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When node is record wait node
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecEventWaitNodeSetEvent.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecEventWaitNodeSetEvent_Negative") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t eventrec, eventwait;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0, event1));
  HIP_CHECK(hipGraphAddEventWaitNode(&eventwait, graph, nullptr, 0, event1));
  // Create memset
  constexpr size_t Nbytes = 1024;
  char* A_d;
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

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("hGraphExec = nullptr") {
    HIP_CHECK_ERROR(hipGraphExecEventWaitNodeSetEvent(nullptr, eventwait, event2),
                    hipErrorInvalidValue);
  }

  SECTION("hNode = nullptr") {
    HIP_CHECK_ERROR(hipGraphExecEventWaitNodeSetEvent(graphExec, nullptr, event2),
                    hipErrorInvalidValue);
  }

  SECTION("event = nullptr") {
    HIP_CHECK_ERROR(hipGraphExecEventWaitNodeSetEvent(graphExec, eventwait, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("hGraphExec is uninitialized") {
    hipGraphExec_t graphExec1{};
    HIP_CHECK_ERROR(hipGraphExecEventWaitNodeSetEvent(graphExec1, eventwait, event2),
                    hipErrorInvalidValue);
  }

  SECTION("hNode is uninitialized") {
    hipGraphNode_t dummy{};
    HIP_CHECK_ERROR(hipGraphExecEventWaitNodeSetEvent(graphExec, dummy, event2),
                    hipErrorInvalidValue);
  }

  SECTION("event is uninitialized") {
    hipEvent_t event_dummy{};
    HIP_CHECK_ERROR(hipGraphExecEventWaitNodeSetEvent(graphExec, eventwait, event_dummy),
                    hipErrorInvalidValue);
  }

  SECTION("event wait node does not exist") {
    hipGraph_t graph1;
    HIP_CHECK(hipGraphCreate(&graph1, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph1, nullptr, 0, &memsetParams));
    hipGraphExec_t graphExec1;
    HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
    HIP_CHECK_ERROR(hipGraphExecEventWaitNodeSetEvent(graphExec1, eventwait, event2),
                    hipErrorInvalidValue);
    HIP_CHECK(hipGraphExecDestroy(graphExec1));
    HIP_CHECK(hipGraphDestroy(graph1));
  }

  SECTION("pass memset node as hNode") {
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0, &memsetParams));
    HIP_CHECK_ERROR(hipGraphExecEventWaitNodeSetEvent(graphExec, memset_A, event2),
                    hipErrorInvalidValue);
  }

  SECTION("pass event record node as hNode") {
    HIP_CHECK_ERROR(hipGraphExecEventWaitNodeSetEvent(graphExec, eventrec, event2),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
}
