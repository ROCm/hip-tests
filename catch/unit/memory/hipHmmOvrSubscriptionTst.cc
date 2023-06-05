/*
Copyright (c) 2021-Present Advanced Micro Devices, Inc. All rights reserved.

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

/* Test Case Description: This test case tests the working of OverSubscription
   feature which is part of HMM.*/

#include <hip_test_common.hh>

#define INIT_VAL 2.5
#define NUM_ELMS 268435456  // 268435456 * 4 = 1GB
#define ITERATIONS 10
#define ONE_GB 1024 * 1024 * 1024

// Kernel function
__global__ void Square(int n, float *x) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    x[i] = x[i] + 10;
  }
}

static void OneGBMemTest(int dev) {
  int DataMismatch = 0;
  float *HmmAG = nullptr;
  hipStream_t strm;
  HIP_CHECK(hipStreamCreate(&strm));
  // Testing hipMemAttachGlobal Flag
  HIP_CHECK(hipMallocManaged(&HmmAG, NUM_ELMS * sizeof(float),
                            hipMemAttachGlobal));

  // Initializing HmmAG memory
  for (int i = 0; i < NUM_ELMS; i++) {
    HmmAG[i] = INIT_VAL;
  }

  int blockSize = 256;
  int numBlocks = (NUM_ELMS + blockSize - 1) / blockSize;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(blockSize, 1, 1);
  HIP_CHECK(hipSetDevice(dev));
  for (int i = 0; i < ITERATIONS; ++i) {
    Square<<<dimGrid, dimBlock, 0, strm>>>(NUM_ELMS, HmmAG);
  }
  HIP_CHECK(hipStreamSynchronize(strm));
  for (int j = 0; j < NUM_ELMS; ++j) {
    if (HmmAG[j] != (INIT_VAL + ITERATIONS * 10)) {
      DataMismatch++;
      break;
    }
  }
  if (DataMismatch != 0) {
    WARN("Data Mismatch observed when kernel launched on device: " << dev);
    REQUIRE(false);
  }
  HIP_CHECK(hipFree(HmmAG));
  HIP_CHECK(hipStreamDestroy(strm));
}

TEST_CASE("Unit_HMM_OverSubscriptionTst") {
  // Checking if xnack is enabled
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  char *p = NULL;
  p = strstr(prop.gcnArchName, "xnack+");
  if (p) {
    size_t FreeMem, TotGpuMem;
    HIP_CHECK(hipMemGetInfo(&FreeMem, &TotGpuMem));
    int NumGB = (TotGpuMem/(ONE_GB));
    int TotalThreads = (NumGB + 10);
    WARN("Launching " << TotalThreads);
    WARN(" processes to test OverSubscription.");

	std::thread Thrds[NumGB];
	
	for (int k = 0; k < TotalThreads; ++k) {
		Thrds[k] = std::thread(OneGBMemTest, 0);
	}
	for (int k = 0; k < TotalThreads; ++k) {
		Thrds[k].join();
	}
  } else {
	HipTest::HIP_SKIP_TEST("GPU is not xnack enabled hence skipping the test...\n");
  }
}
