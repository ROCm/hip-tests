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

/**
 * @addtogroup hipGraphAddMemAllocNode hipGraphAddMemAllocNode
 * @{
 * @ingroup GraphTest
 * `hipGraphAddMemAllocNode (hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t
 * *pDependencies, size_t numDependencies, hipMemAllocNodeParams *pNodeParams)` -
 * Creates a memory allocation node and adds it to a graph.
 */

static constexpr auto element_count{512 * 1024 * 1024};

__global__ void validateGPU(int* const vec, const int value, size_t N, unsigned int* mismatch) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    if (vec[idx] != value) {
      atomicAdd(mismatch, 1);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode behavior with invalid arguments:
 *    -# Null graph node
 *    -# Null graph node
 *    -# Invalid numDependencies for null list of dependencies
 *    -# Invalid numDependencies and valid list for dependencies
 *    -# Null alloc params
 *    -# Invalid poolProps alloc type
 *    -# Invalid poolProps location type
 *    -# Invalid poolProps location id
 *    -# Bytesize is max size_t
 *    -# Invalid accessDescCount
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddMemAllocNode_Negative_Params") {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  hipGraphNode_t alloc_node;
  std::vector<hipGraphNode_t> dependencies;

  HIP_CHECK(hipGraphCreate(&graph, 0));

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  hipMemAccessDesc desc;
  memset(&desc, 0, sizeof(hipMemAccessDesc));
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id = 0;
  desc.flags = hipMemAccessFlagsProtReadWrite;

  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = N;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;
  alloc_param.accessDescs = &desc;
  alloc_param.accessDescCount = 1;

  SECTION("Passing nullptr to graph node") {
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(nullptr, graph, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to graph") {
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, nullptr, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies") {
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 11, &alloc_param),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies and valid list for dependencies") {
    HIP_CHECK(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param));
    dependencies.push_back(alloc_node);
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, dependencies.data(),
                                            dependencies.size() + 1, &alloc_param),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to alloc params") {
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Passing invalid poolProps alloc type") {
    alloc_param.poolProps.allocType = hipMemAllocationTypeInvalid;
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
    alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  }

  SECTION("Passing invalid poolProps location type") {
    alloc_param.poolProps.location.type = hipMemLocationTypeInvalid;
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
    alloc_param.poolProps.location.type = hipMemLocationTypeDevice;
  }

  SECTION("Passing invalid poolProps location id") {
    alloc_param.poolProps.location.id = num_dev;
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
    alloc_param.poolProps.location.id = 0;
  }

#if HT_NVIDIA //EXSWHTEC-353
  SECTION("Passing max size_t bytesize") {
    alloc_param.bytesize = std::numeric_limits<size_t>::max();
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param),
                    hipErrorOutOfMemory);
    alloc_param.bytesize = N;
  }

  SECTION("Passing invalid accessDescCount") {
    alloc_param.accessDescCount = num_dev + 1;
    HIP_CHECK_ERROR(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param),
                    hipErrorInvalidValue);
    alloc_param.accessDescCount = 0;
  }
#endif

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode unsupported behavior:
 *    -# More than one instantiation of the graph exist at the same time
 *    -# Clone graph with mem alloc node
 *    -# Use graph with mem alloc node in a child node
 *    -# Delete edge of the graph with mem alloc node
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddMemAllocNode_Negative_NotSupported") {
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

  SECTION("More than one instantation of the graph exists") {
    hipGraphExec_t graph_exec1, graph_exec2;
    HIP_CHECK(hipGraphInstantiate(&graph_exec1, graph, nullptr, nullptr, 0));
    HIP_CHECK_ERROR(hipGraphInstantiate(&graph_exec2, graph, nullptr, nullptr, 0),
                    hipErrorNotSupported);
    HIP_CHECK(hipGraphExecDestroy(graph_exec1));
  }

#if HT_NVIDIA //EXSWHTEC-353
  SECTION("Clone graph with mem alloc node") {
    hipGraph_t cloned_graph;
    HIP_CHECK_ERROR(hipGraphClone(&cloned_graph, graph), hipErrorNotSupported);
  }

  SECTION("Use graph in a child node") {
    hipGraph_t parent_graph;
    HIP_CHECK(hipGraphCreate(&parent_graph, 0));
    hipGraphNode_t child_graph_node;
    HIP_CHECK_ERROR(hipGraphAddChildGraphNode(&child_graph_node, parent_graph, nullptr, 0, graph),
                    hipErrorNotSupported);
    HIP_CHECK(hipGraphDestroy(parent_graph));
  }

  SECTION("Delete edge of the graph") {
    hipGraphNode_t empty_node;
    HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, &alloc_node, 1));
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(graph, &alloc_node, &empty_node, 1),
                    hipErrorNotSupported);
  }
#endif

  HIP_CHECK(hipGraphDestroy(graph));
}

/* Create graph with memory nodes that copies memset data to host array */
static void createGraph(hipGraphExec_t* graph_exec, int* A_h, int fill_value,
                        int** device_alloc = nullptr) {
  constexpr size_t num_bytes = element_count * sizeof(int);

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t alloc_node;
  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = num_bytes;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param));
  REQUIRE(alloc_param.dptr != nullptr);
  int* A_d = reinterpret_cast<int*>(alloc_param.dptr);

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
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_node, graph, &memset_node, 1, A_h, A_d, num_bytes,
                                    hipMemcpyDeviceToHost));

  if (device_alloc == nullptr) {
    hipGraphNode_t free_node;
    HIP_CHECK(hipGraphAddMemFreeNode(&free_node, graph, &memcpy_node, 1, (void*)A_d));
  } else {
    *device_alloc = A_d;
  }

  // Instantiate graph
  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphDestroy(graph));
}

static void createFreeGraph(hipGraphExec_t* graph_exec, int* device_alloc) {
  hipGraph_t graph;
  hipGraphNode_t free_node;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemFreeNode(&free_node, graph, nullptr, 0, (void*)device_alloc));

  // Instantiate graph
  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode allocates memory correctly and graph behaves as
 * expected when free node is added to the same graph.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddMemAllocNode_Positive_FreeInGraph") {
  hipGraphExec_t graph_exec;

  LinearAllocGuard<int> host_alloc =
      LinearAllocGuard<int>(LinearAllocs::malloc, element_count * sizeof(int));

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr int fill_value = 11;
  createGraph(&graph_exec, host_alloc.ptr(), fill_value);
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  ArrayFindIfNot(host_alloc.host_ptr(), fill_value, element_count);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode allocates memory correctly, graph behaves as expected
 * and allocated memory can can be accessed by outside the graph before memory is freed outside the
 * stream.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddMemAllocNode_Positive_FreeOutsideStream") {
  hipGraphExec_t graph_exec;

  LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, element_count * sizeof(int));
  LinearAllocGuard<unsigned int> mismatch_count_h =
      LinearAllocGuard<unsigned int>(LinearAllocs::malloc, sizeof(unsigned int));
  LinearAllocGuard<unsigned int> mismatch_count_d =
      LinearAllocGuard<unsigned int>(LinearAllocs::hipMalloc, sizeof(unsigned int));
  HIP_CHECK(hipMemset(mismatch_count_d.ptr(), 0, sizeof(unsigned int)));
  int* dev_p;

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int fill_value = 12;

  createGraph(&graph_exec, host_alloc.ptr(), fill_value, &dev_p);
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  validateGPU<<<block_count, thread_count, 0, stream>>>(dev_p, fill_value, element_count,
                                                        mismatch_count_d.ptr());
  // Since hipFree is synchronous, the stream must synchronize before freeing dev_p
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipFree(dev_p));

  HIP_CHECK(hipMemcpy(mismatch_count_h.host_ptr(), mismatch_count_d.ptr(), sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
  REQUIRE(mismatch_count_h.host_ptr()[0] == 0);
  ArrayFindIfNot(host_alloc.host_ptr(), fill_value, element_count);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode allocates memory correctly, graph behaves as expected
 * and allocated memory can can be accessed by outside the graph before memory is freed.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddMemAllocNode_Positive_FreeOutsideGraph") {
  hipGraphExec_t graph_exec;

  LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, element_count * sizeof(int));
  LinearAllocGuard<unsigned int> mismatch_count_h =
      LinearAllocGuard<unsigned int>(LinearAllocs::malloc, sizeof(unsigned int));
  LinearAllocGuard<unsigned int> mismatch_count_d =
      LinearAllocGuard<unsigned int>(LinearAllocs::hipMalloc, sizeof(unsigned int));
  HIP_CHECK(hipMemset(mismatch_count_d.ptr(), 0, sizeof(unsigned int)));
  int* dev_p;

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int fill_value = 13;

  createGraph(&graph_exec, host_alloc.ptr(), fill_value, &dev_p);
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  validateGPU<<<block_count, thread_count, 0, stream>>>(dev_p, fill_value, element_count,
                                                        mismatch_count_d.ptr());
  HIP_CHECK(hipFreeAsync(dev_p, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(mismatch_count_h.host_ptr(), mismatch_count_d.ptr(), sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
  REQUIRE(mismatch_count_h.host_ptr()[0] == 0);
  ArrayFindIfNot(host_alloc.host_ptr(), fill_value, element_count);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemAllocNode allocates memory correctly, graph behaves as expected
 * and allocated memory can can be accessed by outside the graph before memory is freed in a
 * different graph.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemAllocNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddMemAllocNode_Positive_FreeSeparateGraph") {
  hipGraphExec_t graph_exec1, graph_exec2;

  LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, element_count * sizeof(int));
  LinearAllocGuard<unsigned int> mismatch_count_h =
      LinearAllocGuard<unsigned int>(LinearAllocs::malloc, sizeof(unsigned int));
  LinearAllocGuard<unsigned int> mismatch_count_d =
      LinearAllocGuard<unsigned int>(LinearAllocs::hipMalloc, sizeof(unsigned int));
  HIP_CHECK(hipMemset(mismatch_count_d.ptr(), 0, sizeof(unsigned int)));
  int* dev_p;

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int fill_value = 13;

  createGraph(&graph_exec1, host_alloc.ptr(), fill_value, &dev_p);
  createFreeGraph(&graph_exec2, dev_p);
  HIP_CHECK(hipGraphLaunch(graph_exec1, stream));
  validateGPU<<<block_count, thread_count, 0, stream>>>(dev_p, fill_value, element_count,
                                                        mismatch_count_d.ptr());
  HIP_CHECK(hipGraphLaunch(graph_exec2, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(mismatch_count_h.host_ptr(), mismatch_count_d.ptr(), sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
  REQUIRE(mismatch_count_h.host_ptr()[0] == 0);
  ArrayFindIfNot(host_alloc.host_ptr(), fill_value, element_count);

  HIP_CHECK(hipGraphExecDestroy(graph_exec1));
  HIP_CHECK(hipGraphExecDestroy(graph_exec2));
}

/**
* End doxygen group GraphTest.
* @}
*/
