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
#include <hip/hip_runtime_api.h>
#include <memcpy1d_tests_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup hipMemcpyAsync hipMemcpyAsync
 * @{
 * @ingroup MemoryTest
 * `hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
 * hipMemcpyKind kind, hipStream_t stream __dparm(0))` -
 * Copy data from src to dst asynchronously.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemcpy_MultiThread_AllAPIs
 */

/**
 * Test Description
 * ------------------------
 *  - Validate basic device to host behaviour with Device to Host and Default kind.
 *  - Following copy scenarios are considered:
 *    -# Device to Host
 *    -# Device to Host with default kind
 *    -# Host to Device
 *    -# Host to Device with default kind
 *    -# Host to Host
 *    -# Host to Host with default kind
 *    -# Device to Device
 *      - Peer access enabled
 *      - Peer access disabled
 *    -# Device to Device with default kind
 *      - Peer access enabled
 *      - Peer access disabled
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyAsync_Positive_Basic") {
  using namespace std::placeholders;
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  MemcpyWithDirectionCommonTests<true>(std::bind(hipMemcpyAsync, _1, _2, _3, _4, stream));
}

/**
 * Test Description
 * ------------------------
 *  - Validates that the API is asynchronous regarding to:
 *    -# Copying from pageable or pinned host memory to device memory
 *      - Platform specific (NVIDIA)
 *    -# Copying from device memory to pageable or pinned host memory
 *  - Validates that the API is asynchronous regarding to:
 *    -# Copying from device memory to device memory
 *    -# Copying from pageable or pinned host memory to pageable or pinned
 *       host memory
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyAsync_Positive_Synchronization_Behavior") {
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host memory to device memory") {
    MemcpyHtoDSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToDevice, nullptr),
                           true);
  }

  SECTION("Device memory to pageable host memory") {
    MemcpyDtoHPageableSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr), true);
  }

  SECTION("Device memory to pinned host memory") {
    MemcpyDtoHPinnedSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr), false);
  }

  SECTION("Device memory to device memory") {
    MemcpyDtoDSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToDevice, nullptr),
                           false);
  }

  SECTION("Device memory to device Memory No CU") {
    MemcpyDtoDSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToDeviceNoCU,
                            nullptr),false);
  }

  SECTION("Host memory to host memory") {
    MemcpyHtoHSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToHost, nullptr),
                           true);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When the destination pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the source pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When memcpy kind is not valid (-1)
 *      - Expected output: return `hipErrorInvalidMemcpyDirection`
 *    -# When stream is not valid
 *      - Expected output: return `hipErrorContextIsDestroyed`
 *  - Perform handling for following directions:
 *    -# Host to device
 *    -# Device to host
 *    -# Host to host
 *    -# Device to device
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyAsync_Negative_Parameters") {
  using namespace std::placeholders;
  constexpr auto InvalidStream = [] {
    StreamGuard sg(Streams::created);
    return sg.stream();
  };

  SECTION("Host to device") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToDevice, nullptr),
                              device_alloc.ptr(), host_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }

    SECTION("Invalid stream") {
      HIP_CHECK_ERROR(hipMemcpyAsync(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                     hipMemcpyHostToDevice, InvalidStream()),
                      hipErrorContextIsDestroyed);
    }
  }

  SECTION("Device to host") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr),
                              host_alloc.ptr(), device_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }

    SECTION("Invalid stream") {
      HIP_CHECK_ERROR(hipMemcpyAsync(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                     hipMemcpyDeviceToHost, InvalidStream()),
                      hipErrorContextIsDestroyed);
    }
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToHost, nullptr),
                              dst_alloc.ptr(), src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }

    SECTION("Invalid stream") {
      HIP_CHECK_ERROR(hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                     hipMemcpyHostToHost, InvalidStream()),
                      hipErrorContextIsDestroyed);
    }
  }

  SECTION("Device to device") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    MemcpyCommonNegativeTests(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToDevice, nullptr), dst_alloc.ptr(),
        src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(src_alloc.ptr(), dst_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidMemcpyDirection);
    }

    SECTION("Invalid stream") {
      HIP_CHECK_ERROR(hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                     hipMemcpyDeviceToDevice, InvalidStream()),
                      hipErrorContextIsDestroyed);
    }
  }
}
