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

#include <hip_test_common.hh>

/**
 * @addtogroup hipKernelNameRefByPtr hipKernelNameRefByPtr
 * @{
 * @ingroup CallbackTest
 * `hipKernelNameRefByPtr(const void* hostFunction, hipStream_t stream)` -
 * returns the name of passed function pointer on desired stream
 */

__global__ void testKernel() {
    return;
}

/**
 * Test Description
 * ------------------------ 
 *    - Creates new stream and a function pointer
 *    - Verifies that valid API name is returned
 * Test source
 * ------------------------ 
 *    - unit/callback/hipKernelNameRefByPtr.cc
 * Test requirements
 * ------------------------ 
 *    - HIP_VERSION >= 5.2
 *    - Platform specific (AMD) 
 */
TEST_CASE("Unit_hipKernelNameRefByPtr_Positive_Basic") {
    hipStream_t stream{nullptr};
    const void* kernelPtr{reinterpret_cast<const void*>(&testKernel)};

    HIP_CHECK(hipStreamCreate(&stream));
    REQUIRE(hipKernelNameRefByPtr(kernelPtr, stream) != nullptr);
    HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------ 
 *    - Passes `nullptr` stream while function pointer is valid
 *    - Verifies that the returned value is not `nullptr`
 * Test source
 * ------------------------ 
 *    - unit/callback/hipKernelNameRefByPtr.cc
 * Test requirements
 * ------------------------ 
 *    - HIP_VERSION >= 5.2
 *    - Platform specific (AMD)
 */
TEST_CASE("Unit_hipKernelNameRefByPtr_Negative_StreamNullptr") {
    hipStream_t stream{nullptr};
    const void* kernelPtr{reinterpret_cast<const void*>(&testKernel)};

    REQUIRE(hipKernelNameRefByPtr(kernelPtr, stream) != nullptr);
}

/**
 * Test Description
 * ------------------------ 
 *    - Performs validation when the function pointer is `nullptr`
 *        -# When stream is `nullptr`
 *            -# Expected output: `nullptr`
 *        -# When stream is valid
 *            -# Expected output: `nullptr`
 * Test source
 * ------------------------ 
 *    - unit/callback/hipKernelNameRefByPtr.cc
 * Test requirements
 * ------------------------ 
 *    - HIP_VERSION >= 5.2
 *    - Platform specific (AMD)
 */
TEST_CASE("Unit_hipKernelNameRefByPtr_Negative_KernelNullptr") {
    hipStream_t stream{nullptr};
    const void* kernelPtr{nullptr};

    SECTION("stream is nullptr") {
        REQUIRE(hipKernelNameRefByPtr(kernelPtr, stream) == nullptr);
    }

    SECTION("stream is created") {
        HIP_CHECK(hipStreamCreate(&stream));
        REQUIRE(hipKernelNameRefByPtr(kernelPtr, stream) == nullptr);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}
