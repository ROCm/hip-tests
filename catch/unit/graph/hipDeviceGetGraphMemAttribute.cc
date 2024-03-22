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
 * @addtogroup hipDeviceGetGraphMemAttribute hipDeviceGetGraphMemAttribute
 * @{
 * @ingroup GraphTest
 * `hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value)` -
 * Get the mem attribute for graphs.
 */

static constexpr auto element_count{64 * 1024 * 1024};


/* Create graph with memory node */
static void createGraph(hipGraphExec_t* graph_exec, int** device_alloc = nullptr) {
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

  if (device_alloc == nullptr) {
    hipGraphNode_t free_node;
    HIP_CHECK(hipGraphAddMemFreeNode(&free_node, graph, &alloc_node, 1, (void*)A_d));
  } else {
    *device_alloc = A_d;
  }

  // Instantiate graph
  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphDestroy(graph));
}

/* check if memory attributes for graphs contain expected values */
static void checkGraphMemAttribute(size_t used_mem, size_t high_mem) {
  size_t read_mem;
  hipGraphMemAttributeType attr = hipGraphMemAttrUsedMemCurrent;
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, attr, reinterpret_cast<void*>(&read_mem)));
  REQUIRE(read_mem == used_mem);

  attr = hipGraphMemAttrReservedMemCurrent;
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, attr, reinterpret_cast<void*>(&read_mem)));
  REQUIRE(read_mem == used_mem);

  attr = hipGraphMemAttrUsedMemHigh;
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, attr, reinterpret_cast<void*>(&read_mem)));
  REQUIRE(read_mem == high_mem);

  attr = hipGraphMemAttrReservedMemHigh;
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, attr, reinterpret_cast<void*>(&read_mem)));
  REQUIRE(read_mem == high_mem);
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that hipDeviceGetGraphMemAttribute return correct memory attribute values
 * when graphs with allocation nodes are launched, and after memory is freed to OS.
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGetGraphMemAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDeviceGetGraphMemAttribute_Positive_DoubleMemory") {
  hipGraphExec_t graph_exec1, graph_exec2;
  int *dev_p1, *dev_p2;

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  createGraph(&graph_exec1, &dev_p1);
  HIP_CHECK(hipGraphLaunch(graph_exec1, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  checkGraphMemAttribute(element_count * sizeof(int), element_count * sizeof(int));

  createGraph(&graph_exec2, &dev_p2);
  HIP_CHECK(hipGraphLaunch(graph_exec2, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  checkGraphMemAttribute(2 * element_count * sizeof(int), 2 * element_count * sizeof(int));

  HIP_CHECK(hipFree(dev_p1));
  HIP_CHECK(hipFree(dev_p2));

  HIP_CHECK(hipGraphExecDestroy(graph_exec1));
  HIP_CHECK(hipGraphExecDestroy(graph_exec2));
  HIP_CHECK(hipDeviceGraphMemTrim(0));
  checkGraphMemAttribute(0, 2 * element_count * sizeof(int));
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that hipDeviceGetGraphMemAttribute return correct memory attribute values
 * when graphs with allocation and free nodes are launched, and after memory is freed to OS.
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGetGraphMemAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDeviceGetGraphMemAttribute_Positive_ReuseMemory") {
  hipGraphExec_t graph_exec1, graph_exec2;

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  createGraph(&graph_exec1);
  HIP_CHECK(hipGraphLaunch(graph_exec1, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  checkGraphMemAttribute(element_count * sizeof(int), element_count * sizeof(int));

  createGraph(&graph_exec2);
  HIP_CHECK(hipGraphLaunch(graph_exec2, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  checkGraphMemAttribute(element_count * sizeof(int), element_count * sizeof(int));

  HIP_CHECK(hipGraphExecDestroy(graph_exec1));
  HIP_CHECK(hipGraphExecDestroy(graph_exec2));
  HIP_CHECK(hipDeviceGraphMemTrim(0));
  checkGraphMemAttribute(0, element_count * sizeof(int));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipDeviceGetGraphMemAttribute behavior with invalid arguments:
 *    -# Device is not valid
 *    -# Attribute value is not valid
 *    -# Get value is nullptr
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGetGraphMemAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDeviceGetGraphMemAttribute_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  hipGraphMemAttributeType attr = hipGraphMemAttrUsedMemHigh;
  size_t get_value = 0;

  SECTION("Device is not valid") {
    HIP_CHECK_ERROR(
        hipDeviceGetGraphMemAttribute(num_dev, attr, reinterpret_cast<void*>(&get_value)),
        hipErrorInvalidDevice);
  }

  SECTION("Attribute value is not valid") {
    HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(0, static_cast<hipGraphMemAttributeType>(0x7),
                                                  reinterpret_cast<void*>(&get_value)),
                    hipErrorInvalidValue);
  }

  SECTION("Get value is nullptr") {
    HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(0, attr, nullptr), hipErrorInvalidValue);
  }
}

/**
* End doxygen group GraphTest.
* @}
*/
