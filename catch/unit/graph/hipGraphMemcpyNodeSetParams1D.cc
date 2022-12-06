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

#include <functional>

#include <hip_test_common.hh>
#include <memcpy1d_tests_common.hh>

#include "graph_tests_common.hh"

static inline hipMemcpyKind ReverseMemcpyDirection(const hipMemcpyKind direction) {
  switch (direction) {
    case hipMemcpyHostToDevice:
      return hipMemcpyDeviceToHost;
    case hipMemcpyDeviceToHost:
      return hipMemcpyHostToDevice;
    default:
      return direction;
  }
};

TEST_CASE("Unit_hipGraphMemcpyNodeSetParams1D_Positive_Basic") {
  constexpr auto f = [](void* dst, void* src, size_t count, hipMemcpyKind direction) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, src, dst, count / 2,
                                      ReverseMemcpyDirection(direction)));
    HIP_CHECK(hipGraphMemcpyNodeSetParams1D(node, dst, src, count, direction));
    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

  MemcpyWithDirectionCommonTests<false>(f);
}

TEST_CASE("Unit_hipGraphMemcpyNodeSetParams1D_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int src[2] = {}, dst[2] = {};

  hipGraphNode_t node = nullptr;
  HIP_CHECK(
      hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, src, sizeof(dst), hipMemcpyDefault));


  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams1D(nullptr, dst, src, sizeof(dst), hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  MemcpyWithDirectionCommonNegativeTests(
      std::bind(hipGraphMemcpyNodeSetParams1D, node, _1, _2, _3, _4), dst, src, sizeof(dst),
      hipMemcpyDefault);

  SECTION("count == 0") {
    HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams1D(node, dst, src, 0, hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  SECTION("count larger than dst allocation size") {
    LinearAllocGuard<int> dev_dst(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(
        hipGraphMemcpyNodeSetParams1D(node, dev_dst.ptr(), src, sizeof(src), hipMemcpyDefault),
        hipErrorInvalidValue);
  }

  SECTION("count larger than src allocation size") {
    LinearAllocGuard<int> dev_src(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(
        hipGraphMemcpyNodeSetParams1D(node, dst, dev_src.ptr(), sizeof(dst), hipMemcpyDefault),
        hipErrorInvalidValue);
  }
}