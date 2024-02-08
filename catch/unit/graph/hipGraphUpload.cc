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

/**
 * @addtogroup hipGraphUpload hipGraphUpload
 * @{
 * @ingroup GraphTest
 * `hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream)` -
 * Uploads graphExec to the device in stream without executing it.
 * @warning No HIP version supports this API yet.
 */

#if 0

static void HostFunctionSetToZero(void *arg)
{
  int *test_number = (int *) arg;
  (*test_number) = 0;
}

static void HostFunctionAddOne(void *arg)
{
  int *test_number = (int *) arg;
  (*test_number) += 1;
}

/* create an executable graph that will set an integer pointed to by 'number' to one*/
static void CreateTestExecutableGraph(hipGraphExec_t *graph_exec, int *number)
{
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

static int HipGraphUpload_Positive_Simple(hipStream_t stream) {
  int number = 5;
  
  hipGraphExec_t graph_exec;
  CreateTestExecutableGraph(&graph_exec, &number);
  
  
  HIP_CHECK(hipGraphUpload(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(number == 5);
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(number == 1);
  
  
  HIP_CHECK(hipGraphExecDestroy(graph_exec));

}


/**
 * Test Description
 * ------------------------ 
 *  - Validates several basic scenarios:
 *    -# When graph is uploaded with a stream as a created stream
 *    -# with graph is uploaded with a stream as a per thread
 * Test source
 * ------------------------ 
 *  - unit/graph/hipGraphUpload.cc
 * Test requirements
 * ------------------------ 
 *  - No hip version supports hipGraphUpload still
 */
TEST_CASE("Unit_hipGraphUpload_Positive") {
  SECTION("stream as a created stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HipGraphUpload_Positive_Simple(stream);
    HIP_CHECK(hipStreamDestroy(stream));
  }
  
  SECTION("with stream as hipStreamPerThread") {
    HipGraphUpload_Positive_Simple(hipStreamPerThread);
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
 *    -# When graph handle is destroyed before calling upload
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------ 
 *    - unit/graph/hipGraphUpload.cc
 * Test requirements
 * ------------------------ 
 *    - No hip version supports hipGraphUpload still
 */

TEST_CASE("Unit_hipGraphUpload_Functional_With_Priority_Stream") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream1, stream2;
  int minPriority = 0, maxPriority = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
  HIP_CHECK(hipStreamCreateWithPriority(&stream1, hipStreamDefault,
                                        minPriority));
  HIP_CHECK(hipStreamCreateWithPriority(&stream2, hipStreamDefault,
                                        maxPriority));

  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(B_d, B_h, Nbytes, hipMemcpyHostToDevice, stream1));
  HipTest::vectorADD<int><<<1, 1, 0, stream1>>>(A_d, B_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphUpload(graphExec, stream2));

  HIP_CHECK(hipGraphLaunch(graphExec, stream2));
  HIP_CHECK(hipStreamSynchronize(stream2));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}

/**
* Negative Test for API - hipGraphUpload Argument Check
1) Pass graphExec node as nullptr.
2) Pass graphExec node as uninitialize object
3) Pass stream as uninitialize object
4) Graphexec is destroyed before upload
*/

TEST_CASE("Unit_hipGraphUpload_Negative_Parameters") {
  hipGraphExec_t graphExec{};
  hipError_t ret;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("Pass graphExec node as nullptr") {
    HIP_CHECK_ERROR(hipGraphUpload(nullptr, stream), hipErrorInvalidValue);
  }
  SECTION("Pass graphExec node as uninitialize object") {
    HIP_CHECK_ERROR(hipGraphUpload(graphExec, stream), hipErrorInvalidValue);
  }
  SECTION("Pass stream as uninitialize object") {
    hipStream_t stream1{};
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    ret = hipGraphUpload(graphExec, stream1);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("graphExec is destroyed"){
    hipGraphExec_t graph_exec;
    hipGraph_t graph;

    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    
    HIP_CHECK(hipGraphUpload(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK_ERROR(hipGraphUpload(graph_exec, hipStreamPerThread), hipErrorInvalidValue);
  }
  HIP_CHECK(hipStreamDestroy(stream));
}

#endif
