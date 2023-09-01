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

/**
 * @addtogroup hipGraphPerfCheck hipGraphPerfCheck
 * @{
 * @ingroup GraphTest
 * `hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
 *     const hipGraphNode_t* pDependencies, size_t numDependencies,
 *     const hipKernelNodeParams* pNodeParams)` -
 * Creates a kernel execution node and adds it to a graph.
 * Optimize HIPGraph Performance.
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#ifdef _WIN64
#define setenv(x,y,z) _putenv_s(x,y)
#endif

static constexpr int N = 1024;
static constexpr int Nbytes = N * sizeof(int);
static size_t NElem{N};
static constexpr int blocksPerCU = 6;  // to hide latency
static constexpr int threadsPerBlock = 256;

static bool verifyVectorSquare(int *A_h, int* C_h, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      INFO("VectorSquare A and C not matching at " << i);
      return false;
    }
  }
  return true;
}

/*  - Added 2 nodes of MemCpy, and multiple node if Kernel call in continous
      sequence and copy back the result and verify. */
static void checkGraphContinousKernelCall(const unsigned int kNumNode) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3;
  hipGraphNode_t kNode[kNumNode];
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i=0; i < kNumNode; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[i], graph, nullptr, 0,
                                    &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode[i-1], &kNode[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode[kNumNode-1], &memCpy3, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*  - Added multiple nodes of MemCpy, Kernel node continuously for
      2 block & copy back result in MemCpy. */
static void checkGraphContinousKernelCallIn2Blocks(
                 const unsigned int kNumNode1, const unsigned int kNumNode2) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  hipGraphNode_t memCpy1, memCpy2, memCpy3, memCpy4;
  hipGraphNode_t kNode1[kNumNode1], kNode2[kNumNode2];
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy1, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy2, graph, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph, &memCpy1, &memCpy2, 1));

  for (int i=0; i < kNumNode1; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode1[i], graph, nullptr, 0,
                                    &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy2, &kNode1[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode1[i-1], &kNode1[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy3, graph, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode1[kNumNode1-1], &memCpy3, 1));

  for (int i=0; i < kNumNode2; i++) {
    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorSUB<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode2[i], graph, nullptr, 0,
                                    &kernelNodeParams));
    if (i == 0) {
      HIP_CHECK(hipGraphAddDependencies(graph, &memCpy3, &kNode2[i], 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &kNode2[i-1], &kNode2[i], 1));
    }
  }
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memCpy4, graph, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddDependencies(graph, &kNode2[kNumNode2-1], &memCpy4, 1));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*  - Added 2 nodes of MemCpy, Kernel node whicl compute the operation & copy
      back result using MemCpy node. Call this multiple times sequentially. */
static void checkGraphMemcpyKernelMixCall(const unsigned int kNumIter) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  constexpr int kNumNode = 3;
  hipGraphNode_t node[kNumIter * kNumNode];
  hipGraphNode_t kNode[kNumIter];
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int i = 0;
  for (int iter=0; iter < kNumIter; iter++) {
    i = kNumNode * iter;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node[i], graph, nullptr, 0, A_d, A_h,
                                      Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node[i+1], graph, nullptr, 0, B_d, B_h,
                                      Nbytes, hipMemcpyHostToDevice));

    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
    kernelNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[iter], graph, nullptr, 0,
                                    &kernelNodeParams));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&node[i+2], graph, nullptr, 0, C_h, C_d,
                                      Nbytes, hipMemcpyDeviceToHost));
    if (i != 0)
      HIP_CHECK(hipGraphAddDependencies(graph, &node[i-1], &node[i], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &node[i], &node[i+1], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &node[i+1], &kNode[iter], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &kNode[iter], &node[i+2], 1));
  }

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*  - Added a nodes of MemCpy, MesSet, Kernel to do operation and copy back
      result using MemCpy node and call above operation in sequence. */
static void checkGraphMemcpyMemsetKernelMixCall(const unsigned int kNumIter) {
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  constexpr int kNumNode = 3;
  hipGraphNode_t node[kNumIter * kNumNode];
  hipGraphNode_t kNode[kNumIter];
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int pitch_M;

  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int i = 0;
  for (int iter=0; iter < kNumIter; iter++) {
    i = kNumNode * iter;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node[i], graph, nullptr, 0, A_d, A_h,
                                      Nbytes, hipMemcpyHostToDevice));

    hipMemsetParams memsetParams{};
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void *>(B_d);
    memsetParams.value = 2;
    memsetParams.pitch = pitch_M;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = N;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&node[i+1], graph, nullptr, 0,
                                    &memsetParams));

    hipKernelNodeParams kernelNodeParams{};
    void* kernelArgs[] = {&A_d, &C_d, reinterpret_cast<void *>(&NElem)};
    kernelNodeParams.func =
                     reinterpret_cast<void *>(HipTest::vector_square<int>);
    kernelNodeParams.gridDim = dim3(blocks);
    kernelNodeParams.blockDim = dim3(threadsPerBlock);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
    kernelNodeParams.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&kNode[iter], graph, nullptr, 0,
                                    &kernelNodeParams));

    HIP_CHECK(hipGraphAddMemcpyNode1D(&node[i+2], graph, nullptr, 0, C_h, C_d,
                                      Nbytes, hipMemcpyDeviceToHost));
    if (i != 0)
      HIP_CHECK(hipGraphAddDependencies(graph, &node[i-1], &node[i], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &node[i], &node[i+1], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &node[i+1], &kNode[iter], 1));
    HIP_CHECK(hipGraphAddDependencies(graph, &kNode[iter], &node[i+2], 1));
  }

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  REQUIRE(true == verifyVectorSquare(A_h, C_h, N));

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Validate hipGraph performance with doorbell set.
 *  - DEBUG_CLR_GRAPH_ENABLE_BUFFERING and DEBUG_CLR_GRAPH_MAX_AQL_BUFFER_SIZE
 *  - Added multiple nodes of MemCpy, Kernel in sequence multiple times.
 *  - Added multiple nodes of MemCpy, MesSet, Kernel in sequence.
 *  - Added multiple nodes of MemCpy, Kernel node continuously & copy back result in MemCpy.
 *  - Added multiple nodes of MemCpy, Kernel node continuously for 2 block & copy back result in MemCpy.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphPerf.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraph_Perf_Check_MemcpyKernelMixCall") {
  if ((setenv("DEBUG_CLR_GRAPH_ENABLE_BUFFERING", "true", 1)) != 0) {
    HipTest::HIP_SKIP_TEST("Unable to turn on "
             "DEBUG_CLR_GRAPH_ENABLE_BUFFERING, hence exit!");
    return;
  }
  auto BufferSz = GENERATE("25", "35", "15");
  if ((setenv("DEBUG_CLR_GRAPH_MAX_AQL_BUFFER_SIZE", BufferSz, 1)) != 0) {
    HipTest::HIP_SKIP_TEST("Unable to turn on "
             "DEBUG_CLR_GRAPH_MAX_AQL_BUFFER_SIZE, hence exit!");
    return;
  }

  constexpr int kNumIter1 = 25;
  constexpr int kNumIter2 = 30;
  constexpr int kNumKernelNode1 = 15;
  constexpr int kNumKernelNode2 = 45;

  checkGraphMemcpyKernelMixCall(kNumIter1);
  checkGraphMemcpyMemsetKernelMixCall(kNumIter2);
  checkGraphContinousKernelCall(kNumKernelNode1);
  checkGraphContinousKernelCallIn2Blocks(kNumKernelNode1, kNumKernelNode2);

  checkGraphMemcpyKernelMixCall(kNumIter2);
  checkGraphMemcpyMemsetKernelMixCall(kNumIter1);
  checkGraphContinousKernelCall(kNumKernelNode2);
  checkGraphContinousKernelCallIn2Blocks(kNumKernelNode2, kNumKernelNode1);
}
