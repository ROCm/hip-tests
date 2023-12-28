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
 * @defgroup AtomicsTest Device Atomics
 * @{
 * This section describes tests for the Device Atomic APIs.
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
 * @defgroup SyncthreadsTest Synchronization Functions
 * @{
 * This section describes tests for Synchronization Functions.
 * @}
 */

/**
 * @defgroup ThreadfenceTest Memory Fence Functions
 * @{
 * This section describes tests for Memory Fence Functions.
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
 * This section describes the stream management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ModuleTest Module Management
 * @{
 * This section describes the module management types & functions of HIP runtime API.
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
 * @defgroup AtomicsTest Device Atomics
 * @{
 * This section describes tests for the Device Atomic APIs.
 */

/**
 * @addtogroup atomicAdd atomicAdd
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicAdd with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicAdd_Negative_Parameters") {}
/**
 * End doxygen group atomicAdd.
 * @}
 */

/**
 * @addtogroup atomicSub atomicSub
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicSub with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicSub_Negative_Parameters") {}
/**
 * End doxygen group atomicSub.
 * @}
 */

/**
 * @addtogroup atomicInc atomicInc
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicInc with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicInc_Negative_Parameters") {}
/**
 * End doxygen group atomicInc.
 * @}
 */

/**
 * @addtogroup atomicDec atomicDec
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicDec with invalid parameters.
 *  - Compiles the source with specialized Python tool.
 *    -# Utilizes sub-process to invoke compilation of faulty source.
 *    -# Performs post-processing of compiler output and counts errors.
 * Test source
 * ------------------------
 *  - unit/atomics/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_atomicDec_Negative_Parameters") {}
/**
 * End doxygen group atomicDec.
 * @}
 */

/**
 * End doxygen group AtomicsTest.
 * @defgroup MathTest Math Device Functions
 * @{
 * This section describes tests for device math functions of HIP runtime API.
 * @}
 */

 /**
 * @defgroup MathTest Math Device Functions
 * @{
 * This section describes tests for device math functions of HIP runtime API.
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

/**
 * @defgroup DeviceLanguageTest Device Language
 * @{
 * This section describes tests for the Device Language API.
 */

/**
 * @addtogroup launch_bounds launch_bounds
 * @{
 * @ingroup DeviceLanguageTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# Compiles kernels that are not created appropriately:
 *      - Maximum number of threads is 0
 *      - Maximum number of threads is not integer value
 *      - Mimimum number of warps is not integer value
 *    -# Expected output: compiler error
 * Test source
 * ------------------------
 *  - unit/launch_bounds/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Kernel_Launch_bounds_Negative_Parameters_CompilerError") {}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# Compiles kernels that are not created appropriately:
 *      - Maximum number of threads is negative
 *      - Mimimum number of warps is negative
 *  - Validates handling of invalid arguments:
 *    -# Expected output: parse error
 * Test source
 * ------------------------
 *  - unit/launch_bounds/CMakeLists.txt
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_Kernel_Launch_bounds_Negative_Parameters_ParseError") {}

/**
 * End doxygen group launch_bounds.
 * @}
 */

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 * @defgroup VectorTypeTest Vector types
 * @{
 * This section describes tests for the Vector type functions and operators.
 */
