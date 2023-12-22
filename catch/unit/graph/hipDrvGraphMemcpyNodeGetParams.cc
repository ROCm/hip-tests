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

#include <hip_test_defgroups.hh>
#include <hip_test_common.hh>
#include <memcpy3d_tests_common.hh>

/**
 * @addtogroup hipDrvGraphMemcpyNodeGetParams hipDrvGraphMemcpyNodeGetParams
 * @{
 * @ingroup GraphTest
 * `hipDrvGraphMemcpyNodeGetParams(hipGraphNode_t hNode, HIP_MEMCPY3D* nodeParams)` -
 * 	Gets a memcpy node's parameters
 * ________________________
 * Test cases from other APIs:
 *  - @ref Unit_hipDrvGraphMemcpyNodeSetParams_Positive_Basic
 */

/**
 * Test Description
 * ------------------------
 *    - Verify API behaviour with invalid arguments:
 *      -# node is nullptr
 *      -# pNodeParams is nullptr
 *      -# node is destroyed
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphMemcpyNodeGetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphMemcpyNodeGetParams_Negative_Parameters") {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  constexpr hipExtent extent{128 * sizeof(int), 128, 8};

  LinearAllocGuard3D<int> src_alloc(extent);
  LinearAllocGuard3D<int> dst_alloc(extent);

  auto params =
      GetDrvMemcpy3DParms(dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                          make_hipPos(0, 0, 0), dst_alloc.extent(), hipMemcpyDeviceToDevice);

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeGetParams(nullptr, &params), hipErrorInvalidValue);
  }

  SECTION("pNodeParams == nullptr") {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context));
    HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeGetParams(node, nullptr), hipErrorInvalidValue);
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Node is destroyed") {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeGetParams(node, &params), hipErrorInvalidValue);
  }

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}
