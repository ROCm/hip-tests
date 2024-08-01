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

/*
This testfile verifies the following scenarios of hipExtHostAlloc API
1. Basic scenario of hipExtHostAlloc API
2. Negative Scenarios of hipExtHostAlloc API
3. Allocating memory using hipExtHostAlloc with Coherent flag
4. Allocating memory using hipExtHostAlloc with NonCoherent flag
5. Allocating memory using hipExtHostAlloc with default flag
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

namespace {
std::vector<std::string> syncMsg = {"event", "stream", "device"};
static constexpr int numElements{1024 * 16};
static constexpr size_t sizeBytes{numElements * sizeof(int)};

void CheckHostPointer(int numElements, int* ptr, unsigned eventFlags,
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
}
/*
This testcase performs the basic scenario of hipExtHostAlloc API
Allocates the memory using hipExtHostAlloc API
Launches the kernel and performs vector addition.
validates thes result.
*/
TEST_CASE("Unit_hipExtHostAlloc_Basic") {
  static constexpr auto LEN{1024 * 1024};
  static constexpr auto SIZE{LEN * sizeof(float)};

  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  if (prop.canMapHostMemory != 1) {
    SUCCEED("Does support HostPinned Memory");
  } else {
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    HIP_CHECK(hipExtHostAlloc(reinterpret_cast<void**>(&A_h), SIZE,
                           hipHostAllocWriteCombined | hipHostAllocMapped));
    HIP_CHECK(hipExtHostAlloc(reinterpret_cast<void**>(&B_h), SIZE,
                           hipHostAllocDefault));
    HIP_CHECK(hipExtHostAlloc(reinterpret_cast<void**>(&C_h), SIZE,
                           hipHostAllocMapped));

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
  }
}

/*
This testcase verifies the hipExtHostAlloc API by
1.Allocating memory using noncoherent flag
2. Launches the kernel and modifies the variable
   using different synchronization
   techniquies
3. validates the result.
*/
TEST_CASE("Unit_hipExtHostAlloc_NonCoherent") {
  int* A = nullptr;
  HIP_CHECK(hipExtHostAlloc(reinterpret_cast<void**>(&A),
                          sizeBytes, hipExtHostAllocNonCoherent));
  const char* ptrType = "non-coherent";
  CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                   SYNC_DEVICE, ptrType);
  CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                   SYNC_STREAM, ptrType);
  CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                   SYNC_EVENT, ptrType);
}

/*
This testcase verifies the hipExtHostAlloc API by
1.Allocating memory using coherent flag
2. Launches the kernel and modifies the variable
   using different synchronization
   techniquies
3. validates the result.
*/
TEST_CASE("Unit_hipExtHostAlloc_Coherent") {
  int* A = nullptr;
  if (hipExtHostAlloc(reinterpret_cast<void**>(&A), sizeBytes,
                    hipExtHostAllocCoherent) == hipSuccess) {
    const char* ptrType = "coherent";
    CheckHostPointer(numElements, A, hipEventReleaseToDevice,
                     SYNC_DEVICE, ptrType);
    CheckHostPointer(numElements, A, hipEventReleaseToDevice,
                     SYNC_STREAM, ptrType);
    CheckHostPointer(numElements, A, hipEventReleaseToDevice,
                     SYNC_EVENT, ptrType);

    CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                     SYNC_DEVICE, ptrType);
    CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                     SYNC_STREAM, ptrType);
    CheckHostPointer(numElements, A, hipEventReleaseToSystem,
                     SYNC_EVENT, ptrType);
  } else {
    SUCCEED("Coherence memory allocation failed. Is SVM atomic supported?");
  }
}

/*
This testcase verifies the hipExtHostAlloc API by
1.Allocating memory using default flag
2. Launches the kernel and modifies the variable
   using different synchronization
   techniquies
3. validates the result.
*/
TEST_CASE("Unit_hipExtHostAlloc_Default") {
  int* A = nullptr;
  HIP_CHECK(hipExtHostAlloc(reinterpret_cast<void**>(&A), sizeBytes));
  const char* ptrType = "default";
  CheckHostPointer(numElements, A, 0, SYNC_DEVICE, ptrType);
  CheckHostPointer(numElements, A, 0, SYNC_STREAM, ptrType);
  CheckHostPointer(numElements, A, 0, SYNC_EVENT, ptrType);
}

/**
 * Performs argument validation of hipExtHostAlloc api.
 */
TEST_CASE("Unit_hipExtHostAlloc_ArgValidation") {

  constexpr size_t allocSize = 1000;
  char* ptr;

  SECTION("Pass ptr as nullptr") {
    HIP_CHECK_ERROR(hipExtHostAlloc(static_cast<void**>(nullptr), allocSize), hipErrorInvalidValue);
  }

  SECTION("Size as max(size_t)") {
    HIP_CHECK_ERROR(hipExtHostAlloc(&ptr, (std::numeric_limits<std::size_t>::max)()),
                    hipErrorMemoryAllocation);
  }

  SECTION("Flags as max(uint)") {
    HIP_CHECK_ERROR(hipExtHostAlloc(&ptr, allocSize, (std::numeric_limits<unsigned int>::max)()),
                    hipErrorInvalidValue);
  }

  SECTION("Pass size as zero and check ptr reset") {
    HIP_CHECK(hipExtHostAlloc(&ptr, 0));
    REQUIRE(ptr == nullptr);
  }

  SECTION("Pass hipExtHostAllocCoherent and hipExtHostAllocNonCoherent simultaneously") {
    HIP_CHECK_ERROR(
        hipExtHostAlloc(&ptr, allocSize, hipExtHostAllocCoherent | hipExtHostAllocNonCoherent),
        hipErrorInvalidValue);
  }
}
