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

#include "vulkan_test.hh"

/**
 * @addtogroup hipImportExternalSemaphore hipImportExternalSemaphore
 * @{
 * @ingroup MemoryTest
 * `hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out,
 * const hipExternalSemaphoreHandleDesc* semHandleDesc)` -
 * Imports an external semaphore.
 */

constexpr bool enable_validation = false;

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When pointer to external semaphore is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When pointer to the semaphore descriptor is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When flags within descriptor are zero
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When type within descriptor is not valid (-1)
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When handle within descriptor is `nullptr`
 *      - Host specific (WINDOWS)
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/vulkan_interop/hipImportExternalSemaphore.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipImportExternalSemaphore_Vulkan_Negative_Parameters") {
  VulkanTest vkt(enable_validation);
  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  auto handle_desc = vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_BINARY);
  hipExternalSemaphore_t ext_semaphore;

  SECTION("extSem_out == nullptr") {
    HIP_CHECK_ERROR(hipImportExternalSemaphore(nullptr, &handle_desc), hipErrorInvalidValue);
  }

  SECTION("semHandleDesc == nullptr") {
    HIP_CHECK_ERROR(hipImportExternalSemaphore(&ext_semaphore, nullptr), hipErrorInvalidValue);
  }
  /*
   * CUDA doesn't specify the case
  SECTION("semHandleDesc.flags != 0") {
    handle_desc.flags = 1;
    HIP_CHECK_ERROR(hipImportExternalSemaphore(&ext_semaphore, &handle_desc), hipErrorInvalidValue);
  }
  */

  SECTION("Invalid semHandleDesc.type") {
    handle_desc.type = static_cast<hipExternalSemaphoreHandleType>(-1);
    HIP_CHECK_ERROR(hipImportExternalSemaphore(&ext_semaphore, &handle_desc), hipErrorInvalidValue);
  }

#ifdef _WIN32
  SECTION("semHandleDesc.handle == NULL") {
    handle_desc.handle.win32.handle = NULL;
    HIP_CHECK_ERROR(hipImportExternalSemaphore(&ext_semaphore, &handle_desc), hipErrorInvalidValue);
  }
#endif
}