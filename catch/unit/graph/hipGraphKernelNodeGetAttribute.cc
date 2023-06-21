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
#include <hip_test_kernels.hh>

/**
* @addtogroup hipGraphKernelNodeGetAttribute hipGraphKernelNodeGetAttribute
* @{
* @ingroup GraphTest
* `hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode,
*          hipKernelNodeAttrID attr, hipKernelNodeAttrValue* value_out )` -
* Queries node attribute.
*/

/**
* Test Description
* ------------------------
*  - Functional Test for API - hipGraphKernelNodeGetAttribute
*    1) GetKernelAttribute for ID hipKernelNodeAttributeCooperative
*    2) GetKernelAttribute for ID hipKernelNodeAttributeAccessPolicyWindow
* Test source
* ------------------------
*  - unit/graph/hipGraphKernelNodeGetAttribute.cc
* Test requirements
* ------------------------
*  - HIP_VERSION >= 5.6
*/

TEST_CASE("Unit_hipGraphKernelNodeGetAttribute_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, kernel_vecAdd;
  hipKernelNodeParams kNodeParams{};
  hipStream_t stream;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0,
                                  &kNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpy_C, 1));

  hipKernelNodeAttrValue value_out;
  memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));

  SECTION("GetKernelAttribute for hipKernelNodeAttributeCooperative") {
    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                   hipKernelNodeAttributeCooperative, &value_out));
  }
  SECTION("GetKernelAttribute for hipKernelNodeAttributeAccessPolicyWindow") {
    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                   hipKernelNodeAttributeAccessPolicyWindow, &value_out));
  }

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
* Test Description
* ------------------------
*  - Negative Test for API - hipGraphKernelNodeGetAttribute
*    1) Pass kernel node as nullptr for Get attribute api & verify
*    2) Pass KernelNodeAttrID as negative value for Get attribute api & verify
*    3) Pass KernelNodeAttrID as INT_MAX value for Get attribute api & verify
*    4) Pass KernelNodeAttrValue as nullptr for Get attribute api & verify
* Test source
* ------------------------
*  - unit/graph/hipGraphKernelNodeGetAttribute.cc
* Test requirements
* ------------------------
*  - HIP_VERSION >= 5.6
*/

TEST_CASE("Unit_hipGraphKernelNodeGetAttribute_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, kernel_vecAdd;
  hipKernelNodeParams kNodeParams{};
  hipStream_t stream;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  hipError_t ret;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h,
                                    Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d,
                                    Nbytes, hipMemcpyDeviceToHost));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};
  kNodeParams.func = reinterpret_cast<void *>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0,
                                  &kNodeParams));

  hipKernelNodeAttrValue value_out;
  memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));

  SECTION("Pass kernel node as nullptr for Get attribute api") {
    ret = hipGraphKernelNodeGetAttribute(nullptr,
                   hipKernelNodeAttributeAccessPolicyWindow, &value_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass KernelNodeAttrID as negative value for Get attribute api") {
    ret = hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                         hipKernelNodeAttrID(-1), &value_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass KernelNodeAttrID as INT_MAX value for Get attribute api") {
    ret = hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                         hipKernelNodeAttrID(INT_MAX), &value_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#if HT_AMD  // getting SIGSEGV error in Cuda Setup
  SECTION("Pass KernelNodeAttrValue as nullptr for Get attribute api") {
    ret = hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                   hipKernelNodeAttributeAccessPolicyWindow, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
#endif

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}
