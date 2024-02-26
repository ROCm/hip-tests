/*
Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

// Test groups are named based on the group names from hip_api_runtime.h, with adding "Test" suffix

/**
 * @defgroup CallbackTest Callback Activity APIs
 * @{
 * This section describes tests for the callback/Activity of HIP runtime API.
 * @}
 */

/**
 * @defgroup ContextTest Context Management
 * @{
 * This section describes tests for the context management functions of HIP runtime API.
 * @warning All Context Management APIs are **deprecated** and shall not be implemented.
 * @}
 */

/**
 * @defgroup DeviceLanguageTest Device Language
 * @{
 * This section describes tests for the Device Language API.
 * @}
 */

/**
 * @defgroup DeviceTest Device Management
 * @{
 * This section describes tests for device management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup DriverTest Initialization and Version
 * @{
 * This section describes tests for the initialization and version functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup DynamicLoadingTest Kernel Loading Management
 * @{
 * This section describes the different kernel launch approaches.
 * @}
 */

/**
 * @defgroup ErrorTest Error Handling
 * @{
 * This section describes tests for the error handling functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup EventTest Event Management
 * @{
 * This section describes tests for the event management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ExecutionTest Execution Control
 * @{
 * This section describes tests for the execution control functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup GraphTest Graph Management
 * @{
 * This section describes tests for the graph management types & functions of HIP runtime API.
 * @}
 */

/**
* @defgroup KernelTest Kernel Functions Management
* @{
* This section describes the various kernel functions invocation.
* @}
*/

/**
 * @defgroup AtomicsTest Device Atomics
 * @{
 * This section describes tests for the Device Atomic APIs.
 * @}
 */

/**
 * @defgroup MemoryTest memory Management APIs
 * @{
 * This section describes the memory management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup PeerToPeerTest PeerToPeer Device Memory Access
 * @{
 * This section describes tests for the PeerToPeer device memory access functions of HIP runtime API.
 * @warning PeerToPeer support is experimental.
 * @}
 */

/**
 * @defgroup PerformanceTest Performance tests
 * @{
 * This section describes performance tests for the target API groups and use-cases.
 * @}
 */

/**
 * @defgroup MemoryTest Memory Management
 * @{
 * This section describes tests for the memory management functions of HIP runtime API.
 */

/**
 * @addtogroup hipMemset hipMemset
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemset") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemset") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemset") {}
/**
 * End doxygen group hipMemset.
 * @}
 */

/**
 * @addtogroup hipMemsetD32 hipMemsetD32
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemsetD32") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemsetD32") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemsetD32") {}
/**
 * End doxygen group hipMemsetD32.
 * @}
 */

/**
 * @addtogroup hipMemsetD16 hipMemsetD16
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemsetD16") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemsetD16") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemsetD16") {}
/**
 * End doxygen group hipMemsetD16.
 * @}
 */

/**
 * @addtogroup hipMemsetD8 hipMemsetD8
 * @{
 * @ingroup MemoryTest
 */
/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero value is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroValue_hipMemsetD8") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when small size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_SmallSize_hipMemsetD8") {}

/**
 * Test Description
 * ------------------------
 *  - Validates the case when zero size is set.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemsetFunctional.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemsetFunctional_ZeroSize_hipMemsetD8") {} /**
                                                               * End doxygen group hipMemsetD8.
                                                               * @}
                                                               */

/**
 * End doxygen group MemoryTest.
 * @}
 */

/**
 * @defgroup MemoryMTest Managed Memory
 * @{
 * This section describes tests for the managed memory management functions of HIP runtime API.
 */

/**
 * @defgroup ShflTest warp shuffle function Management
 * @{
 * This section describes the warp shuffle types & functions of HIP runtime API.
 */

/**
 * @defgroup p2pTest P2P Management
 * @{
 * This section describes the P2P management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup StreamOTest Ordered Memory Allocator
 * @{
 * This section describes the tests for Stream Ordered Memory Allocator functions of HIP runtime
 * API.
 */

/**
 * @defgroup StreamTest Stream Management
 * @{
 *This section describes the stream management types& functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup TextureTest Texture Management
 * @{
 * This section describes tests for the texture management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup VectorTypeTest Vector types
 * @{
 * This section describes tests for the Vector type functions and operators.
 * @}
 */

/**
 * @defgroup PrintfTest Printf API Management
 * @{
 * This section describes the various Printf use case Scenarios.
 * @}
 */

/**
 * @defgroup SurfaceTest Surface Management
 * @{
 * This section describes tests for the surface management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ComplexTest Complex type
 * @{
 * This section describes tests for the Complex type functions.
 * @}
 */
