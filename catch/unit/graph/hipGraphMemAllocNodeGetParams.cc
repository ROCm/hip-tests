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
#include <resource_guards.hh>
#include <utils.hh>

static constexpr auto element_count{512 * 1024 * 1024};

/*
This test case verifies the negative scenarios of
hipGraphMemAllocNodeGetParams API
*/
TEST_CASE("Unit_hipGraphMemAllocNodeGetParams_Negative") {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  hipGraphNode_t alloc_node;

  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = N;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param));

  hipMemAllocNodeParams get_alloc_params;

  SECTION("Passing nullptr to graph node") {
    HIP_CHECK_ERROR(hipGraphMemAllocNodeGetParams(nullptr, &get_alloc_params),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to allocParams") {
    HIP_CHECK_ERROR(hipGraphMemAllocNodeGetParams(alloc_node, nullptr), hipErrorInvalidValue);
  }

  SECTION("Node is not an alloc node") {
    hipGraphNode_t empty_node;
    HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, nullptr, 0));
    HIP_CHECK_ERROR(hipGraphMemAllocNodeGetParams(empty_node, &get_alloc_params),
                    hipErrorInvalidValue);
  }

  SECTION("input node is uninitialized node") {
    hipGraphNode_t node_unit{};
    HIP_CHECK_ERROR(hipGraphMemAllocNodeGetParams(node_unit, &get_alloc_params),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

TEST_CASE("Unit_hipGraphMemAllocNodeGetParams_Positive") {
  constexpr size_t num_bytes = element_count * sizeof(int);

  hipGraphExec_t graph_exec;
  hipGraph_t graph;

  LinearAllocGuard<int> A_h =
      LinearAllocGuard<int>(LinearAllocs::malloc, element_count * sizeof(int));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipMemAccessDesc desc;
  memset(&desc, 0, sizeof(hipMemAccessDesc));
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = 0;
  desc.flags = hipMemAccessFlagsProtReadWrite;

  hipGraphNode_t alloc_node;
  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = num_bytes;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;
  alloc_param.accessDescs = &desc;
  alloc_param.accessDescCount = 1;

  HIP_CHECK(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param));
  REQUIRE(alloc_param.dptr != nullptr);
  int* A_d = reinterpret_cast<int*>(alloc_param.dptr);

  hipMemAllocNodeParams get_alloc_params;
  HIP_CHECK(hipGraphMemAllocNodeGetParams(alloc_node, &get_alloc_params));
  REQUIRE(memcmp(&alloc_param, &get_alloc_params, sizeof(hipMemAllocNodeParams)) == 0);

  constexpr int fill_value = 11;
  hipGraphNode_t memset_node;
  hipMemsetParams memset_params{};
  memset(&memset_params, 0, sizeof(memset_params));
  memset_params.dst = reinterpret_cast<void*>(A_d);
  memset_params.value = fill_value;
  memset_params.pitch = 0;
  memset_params.elementSize = sizeof(int);
  memset_params.width = element_count;
  memset_params.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_node, graph, &alloc_node, 1, &memset_params));

  hipGraphNode_t memcpy_node;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_node, graph, &memset_node, 1, A_h.host_ptr(), A_d,
                                    num_bytes, hipMemcpyDeviceToHost));

  hipGraphNode_t free_node;
  HIP_CHECK(hipGraphAddMemFreeNode(&free_node, graph, &memcpy_node, 1, (void*)A_d));

  // Instantiate graph
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  ArrayFindIfNot(A_h.host_ptr(), fill_value, element_count);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}
