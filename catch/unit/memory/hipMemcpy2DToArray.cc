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
#include "array_memcpy_tests_common.hh"

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup hipMemcpy2DToArray hipMemcpy2DToArray
 * @{
 * @ingroup MemoryTest
 * `hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src,
 * size_t spitch, size_t width, size_t height, hipMemcpyKind kind)` -
 * Copies data between host and device.
 */

/**
 * Test Description
 * ------------------------
 *  - Validates basic behaviour for copying 2D memory to the array
 *    between host and device.
 *  - The test is run for a various width/height sizes, host allocation types
 *    and flag combinations:
 *      -# Host to array on the device
 *      -# Host to array with default kind
 *      -# Device to array
 *        - Peer access disabled
 *        - Peer access enabled
 *        - Platform specific (NVIDIA)
 *      -# Device to array with default kind
 *        - Peer access disabled
 *        - Peer access enabled
 *        - Platform specific (NVIDIA)
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DToArray.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy2DToArray_Positive_Default") {
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;

  const auto width = GENERATE(16, 32, 48);
  const auto height = GENERATE(1, 16, 32, 48);

  SECTION("Host to Array") {
    Memcpy2DHosttoAShell<false, int>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3,
                                               width * sizeof(int), height, hipMemcpyHostToDevice),
                                     width, height);
  }

  SECTION("Host to Array with default kind") {
    Memcpy2DHosttoAShell<false, int>(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3,
                                               width * sizeof(int), height, hipMemcpyDefault),
                                     width, height);
  }
#if HT_NVIDIA  // EXSWHTEC-120
  SECTION("Device to Array") {
    SECTION("Peer access disabled") {
      Memcpy2DDevicetoAShell<false, false, int>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height,
                    hipMemcpyDeviceToDevice),
          width, height);
    }
    SECTION("Peer access enabled") {
      Memcpy2DDevicetoAShell<false, true, int>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height,
                    hipMemcpyDeviceToDevice),
          width, height);
    }
  }

  SECTION("Device to Array with default kind") {
    SECTION("Peer access disabled") {
      Memcpy2DDevicetoAShell<false, false, int>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height,
                    hipMemcpyDefault),
          width, height);
    }
    SECTION("Peer access enabled") {
      Memcpy2DDevicetoAShell<false, true, int>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), height,
                    hipMemcpyDefault),
          width, height);
    }
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates that API is asynchronous regarding to host when copying
 *    from device memory to device memory.
 *  - Validates following memcpy directions:
 *    -# Host to array
 *    -# Device to array
 *      - Platform specific (NVIDIA)
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DToArray.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy2DToArray_Positive_Synchronization_Behavior") {
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Array") {
    const auto width = GENERATE(16, 32, 48);
    const auto height = GENERATE(16, 32, 48);

    MemcpyHtoASyncBehavior(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, width * sizeof(int),
                                     width * sizeof(int), height, hipMemcpyHostToDevice),
                           width, height, true);
  }
#if HT_NVIDIA  // EXSWHTEC-214
  SECTION("Device to Array") {
    const auto width = GENERATE(16, 32, 48);
    const auto height = GENERATE(16, 32, 48);

    MemcpyDtoASyncBehavior(std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int),
                                     height, hipMemcpyDeviceToDevice),
                           width, height, false);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validate that nothing will be copied if width or height are set to zero.
 *  - Following scenarios are considered:
 *    -# When copying array to host
 *      - Heigth is 0
 *      - Width is 0
 *    -# When copying from array to device
 *      - Height is 0
 *      - Width is 0
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DToArray.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy2DToArray_Positive_ZeroWidthHeight") {
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;
  const auto width = 16;
  const auto height = 16;

  SECTION("Array to host") {
    SECTION("Height is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), 0,
                    hipMemcpyHostToDevice),
          width, height);
    }
    SECTION("Width is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, 0, height, hipMemcpyHostToDevice), width,
          height);
    }
  }
  SECTION("Array to device") {
    SECTION("Height is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, width * sizeof(int), 0,
                    hipMemcpyDeviceToDevice),
          width, height);
    }
    SECTION("Width is 0") {
      Memcpy2DToArrayZeroWidthHeight<false>(
          std::bind(hipMemcpy2DToArray, _1, 0, 0, _2, _3, 0, height, hipMemcpyDeviceToDevice),
          width, height);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When destination pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidHandle`
 *    -# When source pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When source pitch is less than width
 *      - Platform specific (NVIDIA)
 *      - Expected output: return `hipErrorInvalidPitchValue`
 *    -# When width/height increased by offset overflows
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When width/height overflows
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When memcpy direction is not valid
 *      - Expected output: return `hipErrorInvalidMemcpyDirection`
 *  - Following scenarios are repeated for:
 *    -# Host to array
 *    -# Device to array
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpy2DToArray.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpy2DToArray_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;

  const auto width = 32;
  const auto height = 32;
  const auto allocation_size = 2 * width * height * sizeof(int);

  const unsigned int flag = hipArrayDefault;

  ArrayAllocGuard<int> array_alloc(make_hipExtent(width, height, 0), flag);
  LinearAllocGuard2D<int> device_alloc(width, height);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, allocation_size);

  SECTION("Host to Array") {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(nullptr, 0, 0, host_alloc.ptr(), 2 * width * sizeof(int),
                                         width * sizeof(int), height, hipMemcpyHostToDevice),
                      hipErrorInvalidHandle);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, nullptr, 2 * width * sizeof(int),
                                         width * sizeof(int), height, hipMemcpyHostToDevice),
                      hipErrorInvalidValue);
    }
#if HT_NVIDIA  // EXSWHTEC-119
    SECTION("spitch < width") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), width * sizeof(int) - 10,
                             width * sizeof(int), height, hipMemcpyHostToDevice),
          hipErrorInvalidPitchValue);
    }
    SECTION("Offset + width/height overflows") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 1, 0, host_alloc.ptr(), 2 * width * sizeof(int),
                             width * sizeof(int), height, hipMemcpyHostToDevice),
          hipErrorInvalidValue);
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 1, host_alloc.ptr(), 2 * width * sizeof(int),
                             width * sizeof(int), height, hipMemcpyHostToDevice),
          hipErrorInvalidValue);
    }
    SECTION("Width/height overflows") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), 2 * width * sizeof(int),
                             width * sizeof(int) + 1, height, hipMemcpyHostToDevice),
          hipErrorInvalidValue);
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), 2 * width * sizeof(int),
                             width * sizeof(int), height + 1, hipMemcpyHostToDevice),
          hipErrorInvalidValue);
    }
    SECTION("Memcpy kind is invalid") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, host_alloc.ptr(), 2 * width * sizeof(int),
                             width * sizeof(int), height, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidMemcpyDirection);
    }
#endif
  }
  SECTION("Device to Array") {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(nullptr, 0, 0, device_alloc.ptr(), device_alloc.pitch(),
                                         width * sizeof(int), height, hipMemcpyDeviceToDevice),
                      hipErrorInvalidHandle);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, nullptr, device_alloc.pitch(),
                                         width * sizeof(int), height, hipMemcpyDeviceToDevice),
                      hipErrorInvalidValue);
    }
#if HT_NVIDIA  // EXSWHTEC-119
    SECTION("spitch < width") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), width * sizeof(int) - 10,
                             width * sizeof(int), height, hipMemcpyDeviceToDevice),
          hipErrorInvalidPitchValue);
    }
    SECTION("Offset + width/height overflows") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 1, 0, device_alloc.ptr(), device_alloc.pitch(),
                             width * sizeof(int), height, hipMemcpyDeviceToDevice),
          hipErrorInvalidValue);
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 1, device_alloc.ptr(), device_alloc.pitch(),
                             width * sizeof(int), height, hipMemcpyDeviceToDevice),
          hipErrorInvalidValue);
    }
    SECTION("Width/height overflows") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), device_alloc.pitch(),
                             width * sizeof(int) + 1, height, hipMemcpyDeviceToDevice),
          hipErrorInvalidValue);
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), device_alloc.pitch(),
                             width * sizeof(int), height + 1, hipMemcpyDeviceToDevice),
          hipErrorInvalidValue);
    }
    SECTION("Memcpy kind is invalid") {
      HIP_CHECK_ERROR(
          hipMemcpy2DToArray(array_alloc.ptr(), 0, 0, device_alloc.ptr(), device_alloc.pitch(),
                             width * sizeof(int), height, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidMemcpyDirection);
    }
#endif
  }
}
