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
#include <hip_array_common.hh>
#include "hipArrayCommon.hh"
#include "DriverContext.hh"
#include <utils.hh>

/**
 * @addtogroup hipFree hipFree
 * @{
 * @ingroup MemoryTest
 * `hipFree(void* ptr)` -
 * Free memory allocated by the hcc hip memory allocation API.
 */

enum class FreeType { DevFree, ArrayFree, ArrayDestroy, HostFree };

// Amount of time kernel should wait
using namespace std::chrono_literals;
constexpr size_t numAllocs = 10;

/**
 * Test Description
 * ------------------------
 *  - Validate that memory freeing causes device synchronization.
 *  - Uses query on the null stream to check device state.
 *  - The test is run for various allocation sizes.
 * Test source
 * ------------------------
 *  - unit/memory/hipFree.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipFreeImplicitSyncDev") {
  int* devPtr{};
  size_t size_mult = GENERATE(1, 32, 64, 128, 256);
  HIP_CHECK(hipMalloc(&devPtr, sizeof(*devPtr) * size_mult));

  HipTest::BlockingContext b_context{nullptr};

  b_context.block_stream();
  REQUIRE(b_context.is_blocked());

  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  b_context.unblock_stream();

  HIP_CHECK(hipFree(devPtr));
  HIP_CHECK(hipStreamQuery(nullptr));
}

/**
 * End doxygen group hipFree.
 * @}
 */

/**
 * @addtogroup hipHostFree hipHostFree
 * @{
 * @ingroup MemoryTest
 * `hipHostFree(void* ptr)` -
 * Free memory allocated by the hcc hip host memory allocation API.
 */

/**
 * Test Description
 * ------------------------
 *  - Validate that API call causes device synchronization.
 *  - Run kernel that executes for a couple tens of ms.
 *  - Query null stream to check the synchronization state.
 * Test source
 * ------------------------
 *  - unit/memory/hipFree.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipFreeImplicitSyncHost") {
  int* hostPtr{};
  size_t size_mult = GENERATE(1, 32, 64, 128, 256);

  HIP_CHECK(hipHostMalloc(&hostPtr, sizeof(*hostPtr) * size_mult));

  HipTest::BlockingContext b_context{nullptr};

  b_context.block_stream();
  REQUIRE(b_context.is_blocked());

  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  b_context.unblock_stream();

  HIP_CHECK(hipHostFree(hostPtr));
  HIP_CHECK(hipStreamQuery(nullptr));
}

/**
 * End doxygen group hipHostFree.
 * @}
 */

/**
 * @addtogroup hipArrayDestroy hipArrayDestroy
 * @{
 * @ingroup MemoryTest
 * `hipArrayDestroy(hipArray* array)` -
 * Destroys an array.
 */

#if HT_NVIDIA
/**
 * Test Description
 * ------------------------
 *  - Validates handling when the device is busy:
 *    -# When @ref hipFreeArray is called
 *      - Expected output: return `hipSuccess`
 *    -# When @ref hipArrayDestroy is called
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/memory/hipFree.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncArray", "", char, float, float2, float4) {
  CHECK_IMAGE_SUPPORT

  using vec_info = vector_info<TestType>;
  const std::chrono::duration<uint64_t, std::milli> delay = 50ms;
  DriverContext ctx;


  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(32, 512, 1024);

  SECTION("ArrayFree") {
    hipArray_t arrayPtr{};
    hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

    HIP_CHECK(hipMallocArray(&arrayPtr, &desc, width, height, hipArrayDefault));
    LaunchDelayKernel(delay);
    // make sure device is busy
    HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
    HIP_CHECK(hipFreeArray(arrayPtr));
    HIP_CHECK(hipStreamQuery(nullptr));
  }
  SECTION("ArrayDestroy") {
    hipArray_t cuArrayPtr{};

    HIP_ARRAY_DESCRIPTOR cuDesc;
    cuDesc.Width = width;
    cuDesc.Height = height;
    cuDesc.Format = vec_info::format;
    cuDesc.NumChannels = vec_info::size;
    HIP_CHECK(hipArrayCreate(&cuArrayPtr, &cuDesc));
    LaunchDelayKernel(delay);
    // make sure device is busy
    HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
    HIP_CHECK(hipArrayDestroy(cuArrayPtr));
    HIP_CHECK(hipStreamQuery(nullptr));
  }
}
#else  // AMD
TEMPLATE_TEST_CASE("Unit_hipFreeImplicitSyncArray", "", char, float, float2, float4) {
  CHECK_IMAGE_SUPPORT

  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = GENERATE(32, 128, 256, 512, 1024);
  extent.height = GENERATE(0, 32, 128, 256, 512, 1024);
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));
  HipTest::BlockingContext b_context{nullptr};

  b_context.block_stream();
  REQUIRE(b_context.is_blocked());

  HIP_CHECK_ERROR(hipStreamQuery(nullptr), hipErrorNotReady);
  b_context.unblock_stream();

  // Second free segfaults
  SECTION("ArrayDestroy") {
    HIP_CHECK(hipArrayDestroy(arrayPtr));
    HIP_CHECK(hipStreamQuery(nullptr));
  }
  SECTION("ArrayFree") {
    HIP_CHECK(hipFreeArray(arrayPtr));
    HIP_CHECK(hipStreamQuery(nullptr));
  }
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Validate handling when array is `nullptr`:
 *    -# When @ref hipFreeArray is called:
 *      - Expected output (NVIDIA): return `hipSuccess`
 *      - Expected output (AMD): return `hipErrorInvalidValue`
 *    -# When @ref hipArrayDestroy is called:
 *      - Expected output (NVIDIA): return `hipErrorInvalidResourceHandle`
 *      - Expected output (AMD): return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/memory/hipFree.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
#if HT_NVIDIA
TEST_CASE("Unit_hipFreeNegativeArray") {
  DriverContext ctx;

  SECTION("ArrayFree") { HIP_CHECK(hipFreeArray(nullptr)); }
  SECTION("ArrayDestroy") {
    HIP_CHECK_ERROR(hipArrayDestroy(nullptr), hipErrorInvalidResourceHandle);
  }
}
#else
// Freeing a invalid pointer with array
TEST_CASE("Unit_hipFreeNegativeArray") {
  SECTION("ArrayFree") { HIP_CHECK_ERROR(hipFreeArray(nullptr), hipErrorInvalidValue); }
  SECTION("ArrayDestroy") { HIP_CHECK_ERROR(hipArrayDestroy(nullptr), hipErrorInvalidValue); }
}
#endif

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - Validates handling of following scenarios:
 *    -# When @ref hipFreeArray is called two times
 *      - Expected output: return `hipErrorContextIsDestroyed`
 *    -# When @ref hipArrayDestroy is called two times
 *      - Expected output: return `hipErrorContextIsDestroyed`
 * Test source
 * ------------------------
 *  - unit/memory/hipFree.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipFreeDoubleArray") {
  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = width;
  extent.height = height;
  hipChannelFormatDesc desc = hipCreateChannelDesc<char>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));

  SECTION("ArrayFree") {
    HIP_CHECK(hipFreeArray(arrayPtr));
    HIP_CHECK_ERROR(hipFreeArray(arrayPtr), hipErrorContextIsDestroyed);
  }
  SECTION("ArrayDestroy") {
    HIP_CHECK(hipArrayDestroy(arrayPtr));
    HIP_CHECK_ERROR(hipArrayDestroy(arrayPtr), hipErrorContextIsDestroyed);
  }
}
#else  // NVIDIA
TEST_CASE("Unit_hipFreeDoubleArrayFree") {
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-120");
  return;

  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = width;
  extent.height = height;
  hipChannelFormatDesc desc = hipCreateChannelDesc<char>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));

  HIP_CHECK(hipFreeArray(arrayPtr));
  HIP_CHECK_ERROR(hipFreeArray(arrayPtr), hipErrorContextIsDestroyed);
}

TEST_CASE("Unit_hipFreeDoubleArrayDestroy") {
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-120");
  return;
  using vec_info = vector_info<char>;

  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  DriverContext ctx{};

  hipArray_t ArrayPtr{};
  HIP_ARRAY_DESCRIPTOR cuDesc;
  cuDesc.Width = width;
  cuDesc.Height = height;
  cuDesc.Format = vec_info::format;
  cuDesc.NumChannels = vec_info::size;
  HIP_CHECK(hipArrayCreate(&ArrayPtr, &cuDesc));
  HIP_CHECK(hipArrayDestroy(ArrayPtr));
  HIP_CHECK_ERROR(hipArrayDestroy(ArrayPtr), hipErrorContextIsDestroyed);
}

#else  // AMD

TEST_CASE("Unit_hipFreeDoubleArray") {
  CHECK_IMAGE_SUPPORT

  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = width;
  extent.height = height;
  hipChannelFormatDesc desc = hipCreateChannelDesc<char>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));

  SECTION("ArrayFree") {
    HIP_CHECK(hipFreeArray(arrayPtr));
    HIP_CHECK_ERROR(hipFreeArray(arrayPtr), hipErrorContextIsDestroyed);
  }
  SECTION("ArrayDestroy") {
    HIP_CHECK(hipArrayDestroy(arrayPtr));
    HIP_CHECK_ERROR(hipArrayDestroy(arrayPtr), hipErrorContextIsDestroyed);
  }
}

#endif

#if HT_NVIDIA
/**
 * Test Description
 * ------------------------
 *  - Validates handling of multiple arrays in multiple threads:
 *    -# When @ref hipArrayDestroy is called
 *      - Expected output: return `hipSuccess`
 *    -# When @ref hipFreeArray is called
 *      - Expected output: return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/memory/hipFree.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipFreeMultiTArray", "", char, int, float2, float4) {
  using vec_info = vector_info<TestType>;

  size_t width = GENERATE(32, 128, 256, 512, 1024);
  size_t height = GENERATE(32, 128, 256, 512, 1024);
  DriverContext ctx;
  std::vector<std::thread> threads;


  SECTION("ArrayDestroy") {
    std::vector<hipArray_t> ptrs(numAllocs);
    HIP_ARRAY_DESCRIPTOR cuDesc;
    cuDesc.Width = width;
    cuDesc.Height = height;
    cuDesc.Format = vec_info::format;
    cuDesc.NumChannels = vec_info::size;
    for (auto& ptr : ptrs) {
      HIP_CHECK(hipArrayCreate(&ptr, &cuDesc));
    }


    for (auto& ptr : ptrs) {
      threads.emplace_back(([ptr] {
        HIP_CHECK_THREAD(hipArrayDestroy(ptr));
        HIP_CHECK_THREAD(hipStreamQuery(nullptr));
      }));
    }
    for (auto& t : threads) {
      t.join();
    }
    HIP_CHECK_THREAD_FINALIZE();
  }

  SECTION("ArrayFree") {
    std::vector<hipArray_t> ptrs(numAllocs);
    hipExtent extent{};
    extent.width = width;
    extent.height = height;
    hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

    for (auto& ptr : ptrs) {
      HIP_CHECK(hipMallocArray(&ptr, &desc, extent.width, extent.height, hipArrayDefault));
    }

    for (auto ptr : ptrs) {
      SECTION("ArrayFree") {
        threads.emplace_back(([ptr] {
          HIP_CHECK_THREAD(hipFreeArray(ptr));
          HIP_CHECK_THREAD(hipStreamQuery(nullptr));
        }));
      }
    }
    for (auto& t : threads) {
      t.join();
    }
    HIP_CHECK_THREAD_FINALIZE();
  }
}
#else

TEMPLATE_TEST_CASE("Unit_hipFreeMultiTArray", "", char, int, float2, float4) {
  CHECK_IMAGE_SUPPORT

  using vec_info = vector_info<TestType>;

  hipExtent extent{};
  extent.width = GENERATE(32, 128, 256, 512, 1024);
  extent.height = GENERATE(0, 32, 128, 256, 512, 1024);
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  std::vector<std::thread> threads;

  SECTION("ArrayFree") {
    std::vector<hipArray_t> ptrs(numAllocs);
    for (auto& ptr : ptrs) {
      HIP_CHECK(hipMallocArray(&ptr, &desc, extent.width, extent.height, hipArrayDefault));
      threads.emplace_back([ptr] {
        HIP_CHECK_THREAD(hipFreeArray(ptr));
        HIP_CHECK_THREAD(hipStreamQuery(nullptr));
      });
    }
  }
  SECTION("ArrayDestroy") {
    std::vector<hipArray_t> cuArrayPtrs(numAllocs);

    HIP_ARRAY_DESCRIPTOR cuDesc;
    cuDesc.Width = extent.width;
    cuDesc.Height = extent.height;
    cuDesc.Format = vec_info::format;
    cuDesc.NumChannels = vec_info::size;
    for (auto ptr : cuArrayPtrs) {
      HIP_CHECK(hipArrayCreate(&ptr, &cuDesc));

      threads.emplace_back([ptr] {
        HIP_CHECK_THREAD(hipArrayDestroy(ptr));
        HIP_CHECK_THREAD(hipStreamQuery(nullptr));
      });
    }
  }
  for (auto& t : threads) {
    t.join();
  }
  HIP_CHECK_THREAD_FINALIZE();
}

#endif
