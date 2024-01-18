/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "numeric"
#define XSIZE 32

/**
 * Test Description
 * ------------------------
 *  - Negative Scenario - for API hipDrvGraphAddMemcpyNode
 * 1) Pass memcpyNode as nullptr
 * 2) Pass memcpyNode as empty structure
 * 3) Pass graph as nullptr
 * 4) Pass myparams as nullptr
 * 5) Pass context as nullptr
 * 6) When numDependencies is max & pDependencies is not valid ptr
 * 7) When pDependencies is nullptr, but numDependencies is non-zero
 * 8) API expects atleast one memcpy Source pointer to be set
 * 9) API expects atleast one memcpy Destination pointer to be set
 * 10) Passing different element size for srcArray and dstArray
 * Test source
 * ------------------------
 *  - unit/graph/hipDrvGraphAddMemcpyNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
#if HT_AMD
TEST_CASE("Unit_hipDrvGraphAddMemcpyNode_Negative") {
  CHECK_IMAGE_SUPPORT

  constexpr size_t size = 1024;
  size_t numW = size * sizeof(int);
  // Host Vectors
  std::vector<int> A_h(numW);
  std::vector<int> B_h(numW);
  hipCtx_t context;
  // Initialization
  std::iota(A_h.begin(), A_h.end(), 0);
  std::fill_n(B_h.begin(), size, 0);

  hipError_t ret;
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphNode_t memcpyNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipCtxCreate(&context, 0, 0));

  HIP_MEMCPY3D myparams{};
  memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
  myparams.srcXInBytes = 0;
  myparams.srcY = 0;
  myparams.srcZ = 0;
  myparams.dstXInBytes = 0;
  myparams.dstY = 0;
  myparams.dstZ = 0;
  myparams.WidthInBytes = numW;
  myparams.Height = 1;
  myparams.Depth = 1;
  myparams.srcMemoryType = hipMemoryTypeHost;
  myparams.dstMemoryType = hipMemoryTypeHost;
  myparams.srcHost = A_h.data();
  myparams.srcPitch = numW;
  myparams.srcHeight = 1;
  myparams.dstHost = B_h.data();
  myparams.dstPitch = numW;
  myparams.dstHeight = 1;

  SECTION("Pass memcpyNode as nullptr") {
    ret = hipDrvGraphAddMemcpyNode(nullptr, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass memcpyNode as empty structure") {
    hipGraphNode_t memcpyNode_t = {};
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode_t, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass graph as nullptr") {
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, nullptr, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass myparams as nullptr") {
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, nullptr, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass context as nullptr") {
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("When numDependencies is max & pDependencies is not valid ptr") {
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   INT_MAX, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("When pDependencies is nullptr, but numDependencies is non-zero") {
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   2, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("API expects atleast one memcpy src pointer to be set") {
    memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
    myparams.srcXInBytes = 0;
    myparams.srcY = 0;
    myparams.srcZ = 0;
    myparams.dstXInBytes = 0;
    myparams.dstY = 0;
    myparams.dstZ = 0;
    myparams.WidthInBytes = numW;
    myparams.Height = 1;
    myparams.Depth = 1;
    myparams.srcMemoryType = hipMemoryTypeHost;
    myparams.dstMemoryType = hipMemoryTypeHost;
    myparams.srcPitch = numW;
    myparams.srcHeight = 1;
    myparams.dstHost = B_h.data();
    myparams.dstPitch = numW;
    myparams.dstHeight = 1;

    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("API expects atleast one memcpy dst pointer to be set") {
    memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
    myparams.srcXInBytes = 0;
    myparams.srcY = 0;
    myparams.srcZ = 0;
    myparams.dstXInBytes = 0;
    myparams.dstY = 0;
    myparams.dstZ = 0;
    myparams.WidthInBytes = numW;
    myparams.Height = 1;
    myparams.Depth = 1;
    myparams.srcMemoryType = hipMemoryTypeHost;
    myparams.dstMemoryType = hipMemoryTypeHost;
    myparams.srcHost = A_h.data();
    myparams.srcPitch = numW;
    myparams.srcHeight = 1;
    myparams.dstPitch = numW;
    myparams.dstHeight = 1;

    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass source memory type as hipMemoryTypeHost and set srcDevice") {
    memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
    myparams.srcXInBytes = 0;
    myparams.srcY = 0;
    myparams.srcZ = 0;
    myparams.dstXInBytes = 0;
    myparams.dstY = 0;
    myparams.dstZ = 0;
    myparams.WidthInBytes = numW;
    myparams.Height = 1;
    myparams.Depth = 1;
    myparams.srcMemoryType = hipMemoryTypeHost;
    myparams.dstMemoryType = hipMemoryTypeHost;
    myparams.srcDevice = hipDeviceptr_t(A_h.data());
    myparams.dstHost = B_h.data();
    myparams.srcPitch = numW;
    myparams.srcHeight = 1;
    myparams.dstPitch = numW;
    myparams.dstHeight = 1;

    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass dest memory type as hipMemoryTypeHost and set dstDevice") {
    memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
    myparams.srcXInBytes = 0;
    myparams.srcY = 0;
    myparams.srcZ = 0;
    myparams.dstXInBytes = 0;
    myparams.dstY = 0;
    myparams.dstZ = 0;
    myparams.WidthInBytes = numW;
    myparams.Height = 1;
    myparams.Depth = 1;
    myparams.srcMemoryType = hipMemoryTypeHost;
    myparams.dstMemoryType = hipMemoryTypeHost;
    myparams.srcHost = A_h.data();
    myparams.dstDevice = hipDeviceptr_t(B_h.data());
    myparams.srcPitch = numW;
    myparams.srcHeight = 1;
    myparams.dstPitch = numW;
    myparams.dstHeight = 1;

    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Passing different element size for HIP_MEMCPY3D::srcArray"
                  "and HIP_MEMCPY3D::dstArray") {
    memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
    myparams.srcXInBytes = 0;
    myparams.srcY = 0;
    myparams.srcZ = 0;
    myparams.dstXInBytes = 0;
    myparams.dstY = 0;
    myparams.dstZ = 0;
    myparams.WidthInBytes = numW;
    myparams.Height = 1;
    myparams.Depth = 1;
    myparams.srcMemoryType = hipMemoryTypeHost;
    myparams.srcHost = A_h.data();
    myparams.srcDevice = hipDeviceptr_t(A_h.data());
    myparams.srcPitch = numW * sizeof(int);
    myparams.srcHeight = 1;
    myparams.dstMemoryType = hipMemoryTypeArray;
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(int) * 8,
                                       0, 0, 0, hipChannelFormatKindSigned);
    hipArray *devArray1, *devArray2;
    HIP_CHECK(hipMalloc3DArray(&devArray1, &channelDesc,
                               make_hipExtent(10, 10, 10), hipArrayDefault));
    HIP_CHECK(hipMalloc3DArray(&devArray2, &channelDesc,
                               make_hipExtent(11, 11, 11), hipArrayDefault));
    myparams.srcArray = devArray1;
    myparams.dstArray = devArray2;
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
    HIP_CHECK(hipFreeArray(devArray1));
    HIP_CHECK(hipFreeArray(devArray2));
  }
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
 * Create two host pointers, copy the data between them by the api
 * hipDrvGraphAddMemcpyNode with data transfer kind hipMemcpyHostToHost.
 * Validate the output.
 */
static void hipDrvGraphAddMemcpyNode_test(int deviceid = 0) {
  HIP_CHECK(hipSetDevice(deviceid));

  constexpr size_t size = 1024;
  size_t numW = size * sizeof(int);
  // Host Vectors
  std::vector<int> A_h(numW);
  std::vector<int> B_h(numW);
  hipCtx_t context;
  // Initialization
  std::iota(A_h.begin(), A_h.end(), 0);
  std::fill_n(B_h.begin(), size, 0);

  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyH2H;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipCtxCreate(&context, 0, deviceid));

  HIP_MEMCPY3D myparams{};
  memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
  myparams.srcXInBytes = 0;
  myparams.srcY = 0;
  myparams.srcZ = 0;
  myparams.dstXInBytes = 0;
  myparams.dstY = 0;
  myparams.dstZ = 0;
  myparams.WidthInBytes = numW;
  myparams.Height = 1;
  myparams.Depth = 1;
  myparams.srcMemoryType = hipMemoryTypeHost;
  myparams.dstMemoryType = hipMemoryTypeHost;
  myparams.srcHost = A_h.data();
  myparams.srcPitch = numW;
  myparams.srcHeight = 1;
  myparams.dstHost = B_h.data();
  myparams.dstPitch = numW;
  myparams.dstHeight = 1;
  // Host to Host
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyH2H, graph, nullptr,
                                     0, &myparams, context));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  // Validation
  REQUIRE(memcmp(A_h.data(), B_h.data(), numW) == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Functional Scenario -
 * 1) Create two host pointers, copy the data between them by the api
 *    hipDrvGraphAddMemcpyNode with data transfer kind hipMemcpyHostToHost.
 *    Validate the output.
 * Test source
 * ------------------------
 *  - unit/graph/hipDrvGraphAddMemcpyNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipDrvGraphAddMemcpyNode_test") {
  CHECK_IMAGE_SUPPORT

  hipDrvGraphAddMemcpyNode_test();
}

/**
 * Test Description
 * ------------------------
 *  - Functional Scenario - for Multiple device case
 * 1) Create two host pointers, copy the data between them by the api
 *    hipDrvGraphAddMemcpyNode with data transfer kind hipMemcpyHostToHost.
 *    Validate the output.
 * Test source
 * ------------------------
 *  - unit/graph/hipDrvGraphAddMemcpyNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipDrvGraphAddMemcpyNode_MulitDevice") {
  CHECK_IMAGE_SUPPORT

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    for (int device=0; device < numDevices; device++) {
      hipDrvGraphAddMemcpyNode_test(device);
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}
#endif
