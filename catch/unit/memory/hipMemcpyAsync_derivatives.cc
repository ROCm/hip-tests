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
 * @addtogroup hipMemcpyDtoHAsync hipMemcpyDtoHAsync
 * @{
 * @ingroup MemoryTest
 * `hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream)` -
 * Copy data from Device to Host asynchronously.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemcpy_MultiThread_AllAPIs
 *  - @ref Unit_hipMemcpy_Negative
 *  - @ref Unit_hipMemcpy_NullCheck
 *  - @ref Unit_hipMemcpy_HalfMemCopy
 */

static hipStream_t InvalidStream() {
  StreamGuard sg(Streams::created);
  return sg.stream();
}

/**
 * Test Description
 * ------------------------
 *  - Validates basic behaviour:
 *    -# Allocate array on host.
 *    -# Copy memory from Host to Device.
 *    -# Synchronize on stream.
 *    -# Launch kernel.
 *    -# Copy memory from Device to Host.
 *    -# Validate results.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoHAsync_Positive_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  const auto f = [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
    return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, stream);
  };
  MemcpyDeviceToHostShell<true>(f, stream_guard.stream());
}

/**
 * Test Description
 * ------------------------
 *  - Validate that API synchronizes regarding to host when copying from
 *    device to pageable and pinned host memory.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoHAsync_Positive_Synchronization_Behavior") {
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Device memory to pageable host memory") {
    MemcpyDtoHPageableSyncBehavior(
        [](void* dst, void* src, size_t count) {
          return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
        },
        true);
  }

  SECTION("Device memory to pinned host memory") {
    MemcpyDtoHPinnedSyncBehavior(
        [](void* dst, void* src, size_t count) {
          return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
        },
        false);
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
 *    -# When stream is not valid
 *      - Expected output: return `hipErrorContextIsDestroyed`
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoHAsync_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
      },
      host_alloc.ptr(), device_alloc.ptr(), kPageSize);

  SECTION("Invalid stream") {
    HIP_CHECK_ERROR(
        hipMemcpyDtoHAsync(host_alloc.ptr(), reinterpret_cast<hipDeviceptr_t>(device_alloc.ptr()),
                           kPageSize, InvalidStream()),
        hipErrorContextIsDestroyed);
  }
}

/**
 * End doxygen group hipMemcpyDtoHAsync.
 * @}
 */

/**
 * @addtogroup hipMemcpyHtoDAsync hipMemcpyHtoDAsync
 * @{
 * @ingroup MemoryTest
 * `hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src,
 * size_t sizeBytes, hipStream_t stream)` -
 * Copy data from Host to Device asynchronously.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemcpy_MultiThread_AllAPIs
 *  - @ref Unit_hipMemcpy_Negative
 *  - @ref Unit_hipMemcpy_NullCheck
 *  - @ref Unit_hipMemcpy_HalfMemCopy
 */

/**
 * Test Description
 * ------------------------
 *  - Validates basic behaviour:
 *    -# Allocate array on host.
 *    -# Copy memory from Host to Device.
 *    -# Synchronize on stream.
 *    -# Launch kernel.
 *    -# Copy memory from Device to Host.
 *    -# Validate results.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyHtoDAsync_Positive_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  const auto f = [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
    return hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, count, stream);
  };
  MemcpyHostToDeviceShell<true>(f, stream_guard.stream());
}

/**
 * Test Description
 * ------------------------
 *  - Validates that the API is asynchronous regarding to:
 *    -# Copying from pageable or pinned host memory to device memory
 *      - Platform specific (NVIDIA)
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyHtoDAsync_Positive_Synchronization_Behavior") {
  // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
  // respect to the host
#if HT_AMD
  HipTest::HIP_SKIP_TEST(
      "EXSWCPHIPT-127 - MemcpyAsync from host to device memory behavior differs on AMD and "
      "Nvidia");
  return;
#endif
  MemcpyHtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, count, nullptr);
      },
      false);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When the destination pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the source pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When stream is not valid
 *      - Expected output: return `hipErrorContextIsDestroyed`
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyHtoDAsync_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, count, nullptr);
      },
      device_alloc.ptr(), host_alloc.ptr(), kPageSize);

  SECTION("Invalid stream") {
    HIP_CHECK_ERROR(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(device_alloc.ptr()),
                                       host_alloc.ptr(), kPageSize, InvalidStream()),
                    hipErrorContextIsDestroyed);
  }
}

/**
 * End doxygen group hipMemcpyHtoDAsync.
 * @}
 */

/**
 * @addtogroup hipMemcpyDtoDAsync hipMemcpyDtoDAsync
 * @{
 * @ingroup MemoryTest
 * `hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src,
 * size_t sizeBytes, hipStream_t stream)` -
 * Copy data from Device to Device asynchronously.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipMemcpy_MultiThread_AllAPIs
 *  - @ref Unit_hipMemcpy_Negative
 *  - @ref Unit_hipMemcpy_NullCheck
 *  - @ref Unit_hipMemcpy_HalfMemCopy
 */

/**
 * Test Description
 * ------------------------
 *  - Validates basic behaviour:
 *    -# Allocate array on host.
 *    -# Allocate another array on host.
 *    -# Copy memory from Host to Device.
 *    -# Synchronize on stream.
 *    -# Validate results.
 *  - Cheks two possible scenarios:
 *    -# Peer access enabled
 *    -# Peer access disabled
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoDAsync_Positive_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  SECTION("Device to device") {
    SECTION("Peer access enabled") {
      MemcpyDeviceToDeviceShell<true, true>(
          [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
            return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                      reinterpret_cast<hipDeviceptr_t>(src), count, stream);
          });
    }
    SECTION("Peer access disabled") {
      MemcpyDeviceToDeviceShell<true, false>(
          [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
            return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                      reinterpret_cast<hipDeviceptr_t>(src), count, stream);
          });
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates that API is asynchronous regaring to host when copying from device
 *    memory to device memory.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoDAsync_Positive_Synchronization_Behavior") {
  MemcpyDtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                  reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
      },
      false);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When the destination pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the source pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When stream is not valid
 *      - Expected output: return `hipErrorContextIsDestroyed`
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAsync_derivatives.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMemcpyDtoDAsync_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                  reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
      },
      dst_alloc.ptr(), src_alloc.ptr(), kPageSize);

  SECTION("Invalid stream") {
    HIP_CHECK_ERROR(hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst_alloc.ptr()),
                                       reinterpret_cast<hipDeviceptr_t>(src_alloc.ptr()), kPageSize,
                                       InvalidStream()),
                    hipErrorContextIsDestroyed);
  }
}
