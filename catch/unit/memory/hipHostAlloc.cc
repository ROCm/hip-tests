/*
   Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_checkers.hh>
#include <kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_context.hh>
#include <hip_test_helper.hh>

#define SYNC_EVENT 0
#define SYNC_STREAM 1
#define SYNC_DEVICE 2
#define MEMORY_PERCENT 10
#define BLOCK_SIZE 512
#define VALUE 32

/**
* @addtogroup hipHostAlloc hipHostAlloc
* @{
* @ingroup MemoryTest
* `hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags)` -
* Allocate device accessible page locked host memory
*/

static constexpr int numElements{1024 * 16};
static constexpr size_t sizeBytes{numElements * sizeof(int)};
static std::vector<std::string> syncMsg = {"event", "stream", "device"};

static __global__ void kerTestMemAccess(char *buf) {
  size_t myId = threadIdx.x + blockDim.x * blockIdx.x;
  buf[myId] = VALUE;
}

static void CheckHostPointer(int numElements, int* ptr, unsigned eventFlags,
                             int syncMethod, std::string msg) {
    std::cerr << "test: CheckHostPointer "
              << msg
              << " eventFlags = " << std::hex << eventFlags
              << ((eventFlags & hipEventReleaseToDevice) ?
                 " hipEventReleaseToDevice" : "")
              << ((eventFlags & hipEventReleaseToSystem) ?
                 " hipEventReleaseToSystem" : "")
              << " ptr=" << ptr << " syncMethod="
              << syncMsg[syncMethod] << "\n";

    hipStream_t s;
    hipEvent_t e;

    // Init:
    HIP_CHECK(hipStreamCreate(&s));
    HIP_CHECK(hipEventCreateWithFlags(&e, eventFlags))
    dim3 dimBlock(64, 1, 1);
    dim3 dimGrid(numElements / dimBlock.x, 1, 1);

    const int expected = 13;

    // Init array to know state:
    HipTest::launchKernel(Set, dimGrid, dimBlock, 0, 0x0, ptr, -42);
    HIP_CHECK(hipDeviceSynchronize());

    HipTest::launchKernel(Set, dimGrid, dimBlock, 0, s, ptr, expected);
    HIP_CHECK(hipEventRecord(e, s));

    // Host waits for event :
    switch (syncMethod) {
        case SYNC_EVENT:
            HIP_CHECK(hipEventSynchronize(e));
            break;
        case SYNC_STREAM:
            HIP_CHECK(hipStreamSynchronize(s));
            break;
        case SYNC_DEVICE:
            HIP_CHECK(hipDeviceSynchronize());
            break;
        default:
            assert(0);
    }

    for (int i = 0; i < numElements; i++) {
        if (ptr[i] != expected) {
            printf("mismatch at %d: %d != %d\n", i, ptr[i], expected);
            REQUIRE(ptr[i] == expected);
        }
    }

    HIP_CHECK(hipStreamDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase performs the basic scenario of hipHostAlloc API by:
 *    Allocates the memory using hipHostAlloc API.
 *    Launches the kernel and performs vector addition.
 *    Validates the result.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipHostAlloc_Basic") {
  static constexpr auto LEN{1024 * 1024};
  static constexpr auto SIZE{LEN * sizeof(float)};

  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  if (prop.canMapHostMemory != 1) {
    SUCCEED("Doesn't support HostPinned Memory");
  } else {
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&A_h), SIZE,
                          hipHostMallocWriteCombined | hipHostMallocMapped));
    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&B_h), SIZE,
                          hipHostMallocDefault));
    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&C_h), SIZE,
			  hipHostMallocMapped));

    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&C_d), C_h, 0));

    HipTest::setDefaultData<float>(LEN, A_h, B_h, C_h);

    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&B_d), SIZE));
    HIP_CHECK(hipMemcpy(B_d, B_h, SIZE, hipMemcpyHostToDevice));

    dim3 dimGrid(LEN / 512, 1, 1);
    dim3 dimBlock(512, 1, 1);
    HipTest::launchKernel<float>(HipTest::vectorADD<float>, dimGrid, dimBlock,
            0, 0, static_cast<const float*>(A_d),
            static_cast<const float*>(B_d), C_d, static_cast<size_t>(LEN));
    HIP_CHECK(hipMemcpy(C_h, C_d, LEN*sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HipTest::checkVectorADD<float>(A_h, B_h, C_h, numElements);

    HIP_CHECK(hipHostFree(A_h));
    HIP_CHECK(hipHostFree(B_h));
    HIP_CHECK(hipHostFree(C_h));
    HIP_CHECK(hipFree(B_d));
  }
}

/**
 * This testcase verifies the hipHostAlloc API by allocating memory
 * using default flag-
 * Launches the kernel and modifies the variable
 * using different synchronization techniquies
 * validates the result.
*/
TEST_CASE("Unit_hipHostAlloc_Default") {
  int* A = nullptr;
  HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&A), sizeBytes,
			  hipHostMallocDefault));
  const char* ptrType = "default";
  CheckHostPointer(numElements, A, 0, SYNC_DEVICE, ptrType);
  CheckHostPointer(numElements, A, 0, SYNC_STREAM, ptrType);
  CheckHostPointer(numElements, A, 0, SYNC_EVENT, ptrType);
}

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Allocating memory using hipHostMallocNonCoherent flag.
 *  This is a negative test as hipHostMallocNonCoherent
 *  flag is not supported by hipHostAlloc.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
#if HT_AMD
TEST_CASE("Unit_hipHostAlloc_Negative_NonCoherent") {
  int* A = nullptr;
  REQUIRE(hipHostAlloc(reinterpret_cast<void**>(&A), sizeBytes,
                       hipHostMallocNonCoherent) == hipErrorInvalidValue);
  REQUIRE(A == nullptr);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Allocating memory using hipHostMallocCoherent flag.
 *  This is a negative test as hipHostMallocCoherent
 *  flag is not supported by hipHostAlloc.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
#if HT_AMD
TEST_CASE("Unit_hipHostAlloc_Negative_Coherent") {
  int* A = nullptr;
  REQUIRE(hipHostAlloc(reinterpret_cast<void**>(&A), sizeBytes,
                    hipHostMallocCoherent) == hipErrorInvalidValue);
  REQUIRE(A == nullptr);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Allocating memory using hipHostMallocNumaUser flag.
 *  This is a negative test as hipHostMallocNumaUser
 *  flag is not supported by hipHostAlloc.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
#if HT_AMD
TEST_CASE("Unit_hipHostAlloc_Negative_NumaUser") {
  int* A = nullptr;
  REQUIRE(hipHostAlloc(reinterpret_cast<void**>(&A), sizeBytes,
                    hipHostMallocNumaUser) == hipErrorInvalidValue);
  REQUIRE(A == nullptr);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Allocating more memory than total GPU memory.
 *  Validate return hipSuccess.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipHostAlloc_AllocateMoreThanAvailGPUMemory") {
  char* A = nullptr;
  size_t maxGpuMem = 0, availableMem = 0;
  // Get available GPU memory and total GPU memory
  HIP_CHECK(hipMemGetInfo(&availableMem, &maxGpuMem));
  #if defined(_WIN32)
  size_t allocsize = availableMem - (256 * 1024 * 1024);
  allocsize -= allocsize * (MEMORY_PERCENT / 100.0);
  #else
  size_t allocsize = maxGpuMem + ((maxGpuMem * MEMORY_PERCENT) / 100);
  #endif
  // Get free host In bytes
  size_t hostMemFree = HipTest::getMemoryAmount() * 1024 * 1024;
  // Ensure that allocsize < hostMemFree
  if (allocsize < hostMemFree) {
	  printf("inside at line 285\n");
    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&A), allocsize, hipHostMallocDefault));
    HIP_CHECK(hipHostFree(A));
  } else {
    WARN("Skipping test as CPU memory is less than GPU memory");
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Allocating more memory than the total GPU memory.
 *  Validating memory access in a device function.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
#if HT_AMD
TEST_CASE("Unit_hipHostAlloc_AllocateUseMoreThanAvailGPUMemory") {
  char* A = nullptr;
  size_t maxGpuMem = 0, availableMem = 0;
  // Get available GPU memory and total GPU memory
  HIP_CHECK(hipMemGetInfo(&availableMem, &maxGpuMem));
  #if defined(_WIN32)
  size_t allocsize = availableMem - (256 * 1024 * 1024);
  allocsize -= allocsize * (MEMORY_PERCENT / 100.0);
  #else
  size_t allocsize = maxGpuMem + ((maxGpuMem * MEMORY_PERCENT) / 100);
  #endif
  // Get free host In bytes
  size_t hostMemFree = HipTest::getMemoryAmount() * 1024 * 1024;
  // Ensure that allocsize < hostMemFree
  if (allocsize < hostMemFree) {
    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&A), allocsize, hipHostMallocDefault));
    constexpr int sample_size = 1024;
    // memset a sample size to 0
    HIP_CHECK(hipMemset(A, 0, sample_size));
    unsigned int grid_size = allocsize/BLOCK_SIZE;
    // Check if the allocated memory can be accessed in kernels
    kerTestMemAccess<<<grid_size, BLOCK_SIZE>>>(A);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipHostFree(A));
  } else {
    WARN("Skipping test as CPU memory is less than GPU memory");
  }
}
#endif

/**
 * Test Description
 * ------------------------
 *  - This testcase verifies the hipHostAlloc API by:
 *  Test hipHostAlloc() api with ptr as nullptr and check for return value.
 *  Test hipHostAlloc() api with size as max(size_t) and check for OOM error.
 *  Pass size as zero for hipHostAlloc() api and check ptr is reset with
 *  with return value success.
 * Test source
 * ------------------------
 *  - unit/memory/hipHostAlloc.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.3
 */
TEST_CASE("Unit_hipHostAlloc_ArgValidation") {
  constexpr size_t allocSize = 1000;
  char* ptr;

  SECTION("Pass ptr as nullptr") {
    HIP_CHECK_ERROR(hipHostAlloc(static_cast<void**>(nullptr), allocSize,
                    hipHostMallocDefault), hipErrorInvalidValue);
  }

  SECTION("Size as max(size_t)") {
    HIP_CHECK_ERROR(hipHostAlloc(reinterpret_cast<void**>(&ptr),
                    (std::numeric_limits<std::size_t>::max)(),
                    hipHostMallocDefault), hipErrorMemoryAllocation);
  }

  SECTION("Pass size as zero and check ptr reset") {
    HIP_CHECK(hipHostAlloc(reinterpret_cast<void**>(&ptr), 0,
              hipHostMallocDefault));
    REQUIRE(ptr == nullptr);
  }
}


