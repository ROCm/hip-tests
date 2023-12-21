/*
   Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
   IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <utils.hh>
#include <resource_guards.hh>

TEST_CASE("Unit_hipMemPoolImportExport_Functional") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  int shareable_handle;
  hipMemPoolPtrExportData export_ptr;
  void* ptr;

  hipMemAllocationHandleType handle_type = hipMemHandleTypePosixFileDescriptor;
  HIP_CHECK(hipSetDevice(0));
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);

  hipMemPool_t mempool;
  hipMemPoolProps pool_props;
  memset(&pool_props, 0, sizeof(hipMemPoolProps));
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.location.id = 0;

  HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));

  // Allocate memory in a stream from the pool just created
  HIP_CHECK(hipMallocFromPoolAsync(&ptr, kPageSize, mempool, stream.stream()));

  HIP_CHECK(hipMemPoolExportToShareableHandle(&shareable_handle, mempool, handle_type, 0));

  memset((void*)&export_ptr, 0, sizeof(hipMemPoolPtrExportData));
  HIP_CHECK(hipMemPoolExportPointer(&export_ptr, reinterpret_cast<void*>(ptr)));

  LinearAllocGuard<int> host_ptr(LinearAllocs::hipHostMalloc, kPageSize);

  hipMemPool_t shared_mempool;
  int* shared_ptr;

  HIP_CHECK(hipMemPoolImportFromShareableHandle(
      &shared_mempool, reinterpret_cast<void*>(shareable_handle), handle_type, 0));

  hipMemAccessFlags access_flags;
  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;
  location.id = 0;
  HIP_CHECK(hipMemPoolGetAccess(&access_flags, shared_mempool, &location));
  if (access_flags != hipMemAccessFlagsProtReadWrite) {
    hipMemAccessDesc desc;
    memset(&desc, 0, sizeof(hipMemAccessDesc));
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = 0;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolSetAccess(shared_mempool, &desc, 1));
  }

  HIP_CHECK(
      hipMemPoolImportPointer(reinterpret_cast<void**>(&shared_ptr), shared_mempool, &export_ptr));

  const auto element_count = kPageSize / sizeof(int);
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  int expected_value = 12;
  VectorSet<<<block_count, thread_count, 0, stream.stream()>>>(shared_ptr, expected_value,
                                                               element_count);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  // Copy the buffer locally
  HIP_CHECK(hipMemcpyAsync(host_ptr.host_ptr(), shared_ptr, kPageSize, hipMemcpyDeviceToHost,
                           stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  // Check if the content is as expected
  ArrayFindIfNot(host_ptr.host_ptr(), expected_value, element_count);

  // Free the memory before the exporter process frees it
  HIP_CHECK(hipFreeAsync(shared_ptr, stream.stream()));

  // And wait for all the queued up work to complete
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  HIP_CHECK(hipFreeAsync(ptr, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipMemPoolDestroy(mempool));
}

TEST_CASE("Unit_hipMemPoolExportToShareableHandle_Negative_Parameters") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  hipMemPool_t mempool;
  hipMemPoolProps pool_props;
  memset(&pool_props, 0, sizeof(hipMemPoolProps));
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.location.id = 0;
  HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));

  SECTION("Invalid shareable handle") {
    HIP_CHECK_ERROR(
        hipMemPoolExportToShareableHandle(nullptr, mempool, hipMemHandleTypePosixFileDescriptor, 0),
        hipErrorInvalidValue);
  }

  SECTION("Invalid Memory Pool") {
    int share_handle;

    HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(&share_handle, nullptr,
                                                      hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid flag") {
    int share_handle;

    HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(&share_handle, mempool,
                                                      hipMemHandleTypePosixFileDescriptor, 1),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid Memory Pool properties") {
    int share_handle;
    pool_props.handleTypes = hipMemHandleTypeNone;
    hipMemPool_t mempool_none;
    HIP_CHECK(hipMemPoolCreate(&mempool_none, &pool_props));

    HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(&share_handle, mempool_none,
                                                      hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
    pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
    HIP_CHECK(hipMemPoolDestroy(mempool_none));
  }

  SECTION("Invalid Memory Handle type") {
    int share_handle;

    HIP_CHECK_ERROR(
        hipMemPoolExportToShareableHandle(&share_handle, mempool, hipMemHandleTypeNone, 0),
        hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemPoolDestroy(mempool));
}

TEST_CASE("Unit_hipMemPoolImportFromShareableHandle_Negative_Parameters") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int share_handle;
  hipMemPool_t mempool;
  hipMemPoolProps pool_props;
  memset(&pool_props, 0, sizeof(hipMemPoolProps));
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.handleTypes = hipMemHandleTypePosixFileDescriptor;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.location.id = 0;
  HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));
  HIP_CHECK(hipMemPoolExportToShareableHandle(&share_handle, mempool,
                                              hipMemHandleTypePosixFileDescriptor, 0));

  hipMemPool_t shared_mempool = nullptr;

  SECTION("Invalid shareable handle") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&shared_mempool, nullptr,
                                                        hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid Memory Pool") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(nullptr, &share_handle,
                                                        hipMemHandleTypePosixFileDescriptor, 0),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid flag") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&shared_mempool, &share_handle,
                                                        hipMemHandleTypePosixFileDescriptor, 1),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid Memory Handle type") {
    HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&shared_mempool, &share_handle,
                                                        hipMemHandleTypeNone, 0),
                    hipErrorInvalidValue);
  }
}

TEST_CASE("Unit_hipMemPoolExportPointer_Negative_Parameters") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  void* ptr;
  hipMemPoolPtrExportData export_ptr;
  hipMemAllocationHandleType handle_type = hipMemHandleTypePosixFileDescriptor;

  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  int share_handle;
  hipMemPool_t mempool;
  hipMemPoolProps pool_props;
  memset(&pool_props, 0, sizeof(hipMemPoolProps));
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.handleTypes = handle_type;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.location.id = 0;
  HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));

  HIP_CHECK(hipMallocFromPoolAsync(&ptr, kPageSize, mempool, stream.stream()));

  HIP_CHECK(hipMemPoolExportToShareableHandle(&share_handle, mempool, handle_type, 0));

  memset(&export_ptr, 0, sizeof(hipMemPoolPtrExportData));

  SECTION("Invalid exported data") {
    HIP_CHECK_ERROR(hipMemPoolExportPointer(nullptr, reinterpret_cast<void*>(ptr)),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid device pointer") {
    HIP_CHECK_ERROR(hipMemPoolExportPointer(&export_ptr, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipFreeAsync(ptr, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipMemPoolDestroy(mempool));
}

TEST_CASE("Unit_hipMemPoolImportPointer_Negative_Parameters") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
    return;
  }

  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  void* ptr;
  hipMemPoolPtrExportData export_ptr;
  hipMemAllocationHandleType handle_type = hipMemHandleTypePosixFileDescriptor;

  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  int share_handle;
  hipMemPool_t mempool;
  hipMemPoolProps pool_props;
  memset(&pool_props, 0, sizeof(hipMemPoolProps));
  pool_props.allocType = hipMemAllocationTypePinned;
  pool_props.handleTypes = handle_type;
  pool_props.location.type = hipMemLocationTypeDevice;
  pool_props.location.id = 0;
  HIP_CHECK(hipMemPoolCreate(&mempool, &pool_props));

  // Allocate memory in a stream from the pool just created
  HIP_CHECK(hipMallocFromPoolAsync(&ptr, kPageSize, mempool, stream.stream()));

  HIP_CHECK(hipMemPoolExportToShareableHandle(&share_handle, mempool, handle_type, 0));

  memset((void*)&export_ptr, 0, sizeof(hipMemPoolPtrExportData));
  HIP_CHECK(hipMemPoolExportPointer(&export_ptr, reinterpret_cast<void*>(ptr)));

  hipMemPool_t shared_mempool;
  int* shared_ptr = nullptr;

  HIP_CHECK(hipMemPoolImportFromShareableHandle(
      &shared_mempool, reinterpret_cast<void*>(share_handle), handle_type, 0));

  hipMemAccessFlags access_flags;
  hipMemLocation location;
  location.type = hipMemLocationTypeDevice;
  location.id = 0;
  HIP_CHECK(hipMemPoolGetAccess(&access_flags, shared_mempool, &location));
  if (access_flags != hipMemAccessFlagsProtReadWrite) {
    hipMemAccessDesc desc;
    memset(&desc, 0, sizeof(hipMemAccessDesc));
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = 0;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolSetAccess(shared_mempool, &desc, 1));
  }

  SECTION("Invalid device ptr") {
    HIP_CHECK_ERROR(hipMemPoolImportPointer(nullptr, shared_mempool, &export_ptr),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid Memory Pool") {
    HIP_CHECK_ERROR(
        hipMemPoolImportPointer(reinterpret_cast<void**>(&shared_ptr), nullptr, &export_ptr),
        hipErrorInvalidValue);
  }

  SECTION("Invalid exported data") {
    HIP_CHECK_ERROR(
        hipMemPoolImportPointer(reinterpret_cast<void**>(&shared_ptr), shared_mempool, nullptr),
        hipErrorInvalidValue);
  }

  HIP_CHECK(hipFreeAsync(ptr, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipMemPoolDestroy(mempool));
}
