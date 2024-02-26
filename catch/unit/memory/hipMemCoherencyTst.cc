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
#include <chrono>

__global__ void CoherentTst(int* ptr) {  // ptr was set to 1
  atomicAdd_system(ptr, 1);              // now ptr is 2
  while (atomicCAS_system(ptr, 3, 4) != 3) {
    // wait till ptr is updated to 3 in host, then change it to 4
  }
}

__global__  void SquareKrnl(int *ptr) {
  // ptr value squared here
  *ptr = (*ptr) * (*ptr);
}

// The variable below will work as signal to decide pass/fail
static bool YES_COHERENT = false;

// The function tests the coherency of allocated memory
// If this test hangs, means there is issue in coherency
static void TstCoherency(int* ptr, bool hmmMem) {
  int* dptr = nullptr;
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  // storing value 1 in the memory created above
  *ptr = 1;

  if (!hmmMem) {
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dptr), ptr, 0));
    CoherentTst<<<1, 1, 0, stream>>>(dptr);
  } else {
    CoherentTst<<<1, 1, 0, stream>>>(ptr);
  }
  // To prevent Windows batch dispatching issue, run inspecting code in thread
  std::thread my_thread([ptr] {
    int d = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    while (
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start)
            .count() <= 3) {
      d = __sync_fetch_and_add(ptr, 0);  // Retrieve *ptr
      if (d == 2) break; // If kernel has updated *ptr to 2, exit
    }  // wait till ptr is updated to 2 from kernel or 3 seconds
    if (d != 2) {
      // 3 seconds should be long enough for kernel to update ptr
      fprintf(stderr, "d = %d hasn't been updated to 2 in 3s\n", d);
      return;
    }
    // increment it to 3
    __sync_fetch_and_add(ptr, 1);
  });

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  my_thread.join();
  if (*ptr == 4) {
    YES_COHERENT = true;
  }
}

#if HT_AMD
/**
 * @addtogroup hipHostMalloc hipHostMalloc
 * @{
 * @ingroup MemoryTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validates if fine grain behaviour is observed or not with memory allocated
 *    using this API.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemCoherencyTst.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipHostMalloc_CoherentTst") {
  int *Ptr = nullptr, SIZE = sizeof(int);
  bool HmmMem = false;
  YES_COHERENT = false;

  // Allocating hipHostMalloc() memory with hipHostMallocCoherent flag
  SECTION("hipHostMalloc with hipHostMallocCoherent flag") {
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocCoherent));
  }
  SECTION("hipHostMalloc with Default flag") {
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE));
  }
  SECTION("hipHostMalloc with hipHostMallocMapped flag") {
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocMapped));
  }

  TstCoherency(Ptr, HmmMem);
  HIP_CHECK(hipHostFree(Ptr));
  REQUIRE(YES_COHERENT);
}
/**
 * End doxygen group hipHostMalloc.
 * @}
 */
#endif

/**
 * @addtogroup hipMallocManaged hipMallocManaged
 * @{
 * @ingroup MemoryMTest
 */
#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - Validates if fine grain behaviour is observed or not with memory allocated
 *    using this API.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemCoherencyTst.cc
 * Test requirements
 * ------------------------
 *  - Device supports managed memory management
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMallocManaged_CoherentTst") {
  int *Ptr = nullptr, SIZE = sizeof(int), managed = 0;
  bool HmmMem = true;
  YES_COHERENT = false;

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                  0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);
  if (managed == 1) {
    // Allocating hipMallocManaged() memory
    SECTION("hipMallocManaged with hipMemAttachGlobal flag") {
      HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachGlobal));
    }
    SECTION("hipMallocManaged with hipMemAttachHost flag") {
      HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachHost));
    }
    TstCoherency(Ptr, HmmMem);
    HIP_CHECK(hipFree(Ptr));
    REQUIRE(YES_COHERENT);
  } else {
    SUCCEED("GPU 0 doesn't support ManagedMemory "
           "device attribute. Hence skipping the test with Pass result.\n");
  }
}
#endif

/* Test case description: The following test validates if memory access is fine
   with memory allocated using hipMallocManaged() and CoarseGrain Advise*/
/**
 * Test Description
 * ------------------------
 *  - Validates if memory access is fine grained with memory allocates using
 *    this API and coarse grain advice.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemCoherencyTst.cc
 * Test requirements
 * ------------------------
 *  - Device supports managed memory management
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMallocManaged_CoherentTstWthAdvise") {
  int *Ptr = nullptr, SIZE = sizeof(int), managed = 0;
  YES_COHERENT = false;

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory,
                                  0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);

  if (managed == 1) {
    // Allocating hipMallocManaged() memory
    SECTION("hipMallocManaged with hipMemAttachGlobal flag") {
      HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachGlobal));
    }
    SECTION("hipMallocManaged with hipMemAttachHost flag") {
      HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachHost));
    }
#if HT_AMD
    HIP_CHECK(hipMemAdvise(Ptr, SIZE, hipMemAdviseSetCoarseGrain, 0));
#endif
    // Initializing Ptr memory with 9
    *Ptr = 9;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    SquareKrnl<<<1, 1, 0, strm>>>(Ptr);
    HIP_CHECK(hipStreamSynchronize(strm));
    if (*Ptr == 81) {
      YES_COHERENT = true;
    }
    HIP_CHECK(hipFree(Ptr));
    HIP_CHECK(hipStreamDestroy(strm));
    REQUIRE(YES_COHERENT);
  } else {
    SUCCEED("GPU 0 doesn't support hipDeviceAttributeManagedMemory "
             "attribute. Hence skipping the test with Pass result.\n");
  }
}

/**
 * End doxygen group hipMallocManaged.
 * @}
 */

#if HT_AMD
/**
 * @addtogroup hipMalloc hipMalloc
 * @{
 * @ingroup MemoryTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validates if memory allocated using this API are of type coarse grain.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemCoherencyTst.cc
 * Test requirements
 * ------------------------
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipMalloc_CoherentTst") {
  int *Ptr = nullptr, SIZE = sizeof(int);
  uint32_t svm_attrib = 0;
  bool IfTstPassed = false;
  // Allocating hipMalloc() memory
  HIP_CHECK(hipMalloc(&Ptr, SIZE));
  HIP_CHECK(hipMemRangeGetAttribute(&svm_attrib, sizeof(svm_attrib),
        hipMemRangeAttributeCoherencyMode, Ptr, SIZE));
  if (svm_attrib == hipMemRangeCoherencyModeCoarseGrain) {
    IfTstPassed = true;
  }
  HIP_CHECK(hipFree(Ptr));
  REQUIRE(IfTstPassed);
}
/**
 * End doxygen group hipMalloc.
 * @}
 */
#endif

#if HT_AMD
/**
 * @addtogroup hipExtMallocWithFlags hipExtMallocWithFlags
 * @{
 * @ingroup MemoryTest
 */

/**
 * Test Description
 * ------------------------
 *  - Validates if fine grain behaviour is observed or not with memory allocated
 *    using this API.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemCoherencyTst.cc
 * Test requirements
 * ------------------------
 *  - Device supports managed memory management
 *    or pageable memory access
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipExtMallocWithFlags_CoherentTst") {
  int *Ptr = nullptr, SIZE = sizeof(int), InitVal = 9, Pageable = 0, managed = 0, finegrain = 0;
  bool FineGrain = true;
  YES_COHERENT = false;

  HIP_CHECK(hipDeviceGetAttribute(&Pageable,
                                 hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << Pageable);

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, 0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);
  if (managed == 1 && Pageable == 1) {
    // Allocating hipExtMallocWithFlags() memory with flags
    HIP_CHECK(hipDeviceGetAttribute(&finegrain, hipDeviceAttributeFineGrainSupport, 0));
    if (finegrain == 1) {
      SECTION("hipExtMallocWithFlags with hipDeviceMallocFinegrained flag") {
        HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&Ptr), SIZE*2,
                                        hipDeviceMallocFinegrained));
      }
    }
    SECTION("hipExtMallocWithFlags with hipDeviceMallocSignalMemory flag") {
      // for hipMallocSignalMemory flag the size of memory must be 8
      HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&Ptr), SIZE*2,
                                      hipMallocSignalMemory));
    }
    SECTION("hipExtMallocWithFlags with hipDeviceMallocDefault flag") {
      /* hipExtMallocWithFlags() with flag
      hipDeviceMallocDefault allocates CoarseGrain memory */
      FineGrain = false;
      HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&Ptr), SIZE*2,
                                      hipDeviceMallocDefault));
    }
    if (FineGrain) {
      TstCoherency(Ptr, FineGrain);
    } else {
      *Ptr = InitVal;
      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      SquareKrnl<<<1, 1, 0, strm>>>(Ptr);
      HIP_CHECK(hipStreamSynchronize(strm));
      if (*Ptr == (InitVal * InitVal)) {
        YES_COHERENT = true;
      }
    }
    HIP_CHECK(hipFree(Ptr));
    REQUIRE(YES_COHERENT);
  } else {
    SUCCEED("GPU 0 doesn't support ManagedMemory or PageableMemoryAccess"
           "device attribute. Hence skipping the test with Pass result.\n");
  }
}
/**
 * End doxygen group hipExtMallocWithFlags.
 * @}
 */
#endif
