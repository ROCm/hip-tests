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

TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParams1D_Positive_Basic") {
  constexpr auto f = [](void* dst, void* src, size_t count, hipMemcpyKind direction) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t node = nullptr;
    const auto offset_src = reinterpret_cast<uint8_t*>(src) + 1;
    const auto offset_dst = reinterpret_cast<uint8_t*>(dst) + 1;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, offset_dst, offset_src, count - 1,
                                      direction));
    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(graph_exec, node, dst, src, count, direction));
    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

  MemcpyWithDirectionCommonTests<false>(f);
}

TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParams1D_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int src[2] = {}, dst[2] = {};

  hipGraphNode_t node = nullptr;
  HIP_CHECK(
      hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, src, sizeof(dst), hipMemcpyDefault));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  SECTION("pGraphExec == nullptr") {
    HIP_CHECK_ERROR(
        hipGraphExecMemcpyNodeSetParams1D(nullptr, node, dst, src, sizeof(dst), hipMemcpyDefault),
        hipErrorInvalidValue);
  }

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParams1D(graph_exec, nullptr, dst, src, sizeof(dst),
                                                      hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  MemcpyWithDirectionCommonNegativeTests(
      std::bind(hipGraphExecMemcpyNodeSetParams1D, graph_exec, node, _1, _2, _3, _4), dst, src,
      sizeof(dst), hipMemcpyDefault);

  SECTION("count == 0") {
    HIP_CHECK_ERROR(
        hipGraphExecMemcpyNodeSetParams1D(graph_exec, node, dst, src, 0, hipMemcpyDefault),
        hipErrorInvalidValue);
  }

  SECTION("count larger than dst allocation size") {
    LinearAllocGuard<int> dev_dst(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParams1D(graph_exec, node, dev_dst.ptr(), src,
                                                      sizeof(src), hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  SECTION("count larger than src allocation size") {
    LinearAllocGuard<int> dev_src(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParams1D(graph_exec, node, dst, dev_src.ptr(),
                                                      sizeof(dst), hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParams1D_Negative_Changing_Memcpy_Direction") {
  int host;
  LinearAllocGuard<int> dev(LinearAllocs::hipMalloc, sizeof(int));

  const auto [dir, src, dst] =
      GENERATE_REF(std::make_tuple(hipMemcpyHostToHost, &host, &host),
                   std::make_tuple(hipMemcpyHostToDevice, &host, dev.ptr()),
                   std::make_tuple(hipMemcpyDeviceToHost, dev.ptr(), &host),
                   std::make_tuple(hipMemcpyDeviceToDevice, dev.ptr(), dev.ptr()));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, src, sizeof(int), dir));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  const auto set_dir = GENERATE(hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
                                hipMemcpyDeviceToDevice, hipMemcpyDefault);
  if (dir == set_dir) {
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));
    return;
  }

  HIP_CHECK_ERROR(
      hipGraphExecMemcpyNodeSetParams1D(graph_exec, node, dst, src, sizeof(int), set_dir),
      hipErrorInvalidValue);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}