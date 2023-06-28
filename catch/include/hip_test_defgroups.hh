/*
Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

// Test groups are named based on the group names from hip_api_runtime.h, with adding "Test" suffix

/**
 * @defgroup CallbackTest Callback Activity APIs
 * @{
 * This section describes tests for the callback/Activity of HIP runtime API.
 * @}
 */

/**
 * @defgroup GraphTest Graph Management
 * @{
 * This section describes tests for the graph management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup DeviceTest Device Management
 * @{
 * This section describes tests for device management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup EventTest Event Management
 * @{
 * This section describes tests for the event management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup ErrorTest Error Handling
 * @{
 * This section describes tests for the error handling functions of HIP runtime API.
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
 * @defgroup ShflTest warp shuffle function Management
 * @{
 * This section describes the warp shuffle types & functions of HIP runtime API.
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
 * @defgroup StreamTest Stream Management
 * @{
 * This section describes the stream management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup AtomicsTest Device Atomics
 * @{
 * This section describes tests for the Device Atomic APIs.
 * @}
 */

 /**
 * @addtogroup atomicExch atomicExch
 * @{
 * @ingroup AtomicsTest
 */

/**
 * Test Description
 * ------------------------
 *  - Compiles atomicExch with invalid parameters.
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
TEST_CASE("Unit_atomicExch_Negative_Parameters") {}
/**
 * End doxygen group atomicExch.
 * @}
 */

/**
 * End doxygen group AtomicsTest.
 * @}
 */
