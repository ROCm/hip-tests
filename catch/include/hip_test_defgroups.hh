/*
Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
 * This section describes tests for the PeerToPeer device memory access functions of HIP runtime
 * API.
 * @warning PeerToPeer support is experimental.
 * @}
 */

/**
 * @defgroup DriverTest Initialization and Version
 * @{
 * This section describes tests for the initialization and version functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup DeviceLanguageTest Device Language
 * @{
 * This section describes tests for the Device Language API.
 * @}
 */

/**
 * @defgroup ExecutionTest Execution Control
 * @{
 * This section describes tests for the execution control functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup StreamTest Stream Management
 * @{
 * This section describes tests for the stream management functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup StreamMTest Stream Memory Operations
 * @{
 * This section describes tests for the Stream Memory Wait and Write functions of HIP runtime API.
 */

// Adding dummy Test Cases that are in the form of function macros/templates and are
// not possible to generate with Doxygen.

/**
 * @addtogroup hipStreamWaitValue32 hipStreamWaitValue32
 * @{
 * @ingroup StreamMTest
 * `hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value,
 * unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF))` -
 * Enqueues a wait command to the stream, all operations enqueued on this stream after this, will
 * not execute until the defined wait condition is true.
 */

/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_NoMask_Eq") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_NoMask_Eq") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_NoMask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_NoMask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using And (&) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_NoMask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using And (&) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_NoMask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Nor (|) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_NoMask_Nor") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Nor (|) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_NoMask_Nor") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_Mask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_Mask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_Mask_Eq_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_Mask_Eq_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_Mask_Eq_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_Mask_Eq_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using And (&) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_Blocking_Mask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using And (&) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait32_NonBlocking_Mask_And") {}
TEST_CASE("Unit_hipStreamValue_Negative_InvalidMemory") {}
/**
 * End doxygen group hipStreamWaitValue32.
 * @}
 */

/**
 * @addtogroup hipStreamWaitValue64 hipStreamWaitValue64
 * @{
 * @ingroup StreamMTest
 * `hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value,
 * unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF))` -
 * Enqueues a wait command to the stream, all operations enqueued  on this stream after this, will
 * not execute until the defined wait condition is true.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipStreamValue_Negative_InvalidFlag
 *  - @ref Unit_hipStreamValue_Negative_InvalidMemory
 *  - @ref Unit_hipStreamValue_Negative_UninitializedStream
 */

/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_NoMask_Eq") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_NoMask_Eq") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_NoMask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_NoMask_Gte") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using And (&) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_NoMask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using And (&) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_NoMask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Nor (|) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_NoMask_Nor") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Nor (|) without masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_NoMask_Nor") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_Mask_Gte_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_Mask_Gte_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_Mask_Gte_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Gte (>=) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_Mask_Gte_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_Mask_Eq_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_Mask_Eq_1") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_Mask_Eq_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using Eq (==) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_Mask_Eq_2") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs blocking wait for specified value using And (&) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_Blocking_Mask_And") {}
/**
 * Test Description
 * ------------------------
 *  - Creates valid stream.
 *  - Performs non-blocking wait for specified value using And (&) with masking.
 *  - Checks if results are valid.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamValue.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipStreamValue_Wait64_NonBlocking_Mask_And") {}
/**
 * End doxygen group hipStreamWaitValue64.
 * @}
 */

/**
 * @addtogroup hipStreamWriteValue64 hipStreamWriteValue64
 * @{
 * @ingroup StreamMTest
 * `hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags)` -
 * Enqueues a write command to the stream, write operation is performed after all earlier commands
 * on this stream have completed the execution.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipStreamValue_Write
 *  - @ref Unit_hipStreamValue_Negative_InvalidMemory
 *  - @ref Unit_hipStreamValue_Negative_UninitializedStream
 * @}
 */

/**
 * End doxygen group StreamMTest.
 * @}
 */

/**
 * @defgroup ShflTest warp shuffle function Management
 * @{
 * This section describes the warp shuffle types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup StreamOTest Ordered Memory Allocator
 * @{
 * This section describes the tests for Stream Ordered Memory Allocator functions of HIP runtime
 * API.
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
 * @defgroup DynamicLoadingTest Kernel Loading Management
 * @{
 * This section describes the different kernel launch approaches.
 * @}
 */

/**
 * @defgroup MemoryTest memory Management APIs
 * @{
 * This section describes the memory management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup KernelTest Kernel Functions Management
 * @{
 * This section describes the various kernel functions invocation.
 * @}
 */
