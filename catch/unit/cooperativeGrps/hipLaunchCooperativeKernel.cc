/*
Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

static constexpr size_t kBufferLen = 1024 * 1024;

__global__ void test_gws(int* buf, size_t buf_size, long* tmp_buf, long* result)
{
    extern __shared__ long tmp[];
    uint offset = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = gridDim.x  * blockDim.x;
    cg::grid_group gg = cg::this_grid();

    long sum = 0;
    for (uint i = offset; i < buf_size; i += stride) {
        sum += buf[i];
    }
    tmp[threadIdx.x] = sum;

    __syncthreads();

    if (threadIdx.x == 0) {
        sum = 0;
        for (uint i = 0; i < blockDim.x; i++) {
            sum += tmp[i];
        }
        tmp_buf[blockIdx.x] = sum;
    }

    gg.sync();

    if (offset == 0) {
        for (uint i = 1; i < gridDim.x; ++i) {
          sum += tmp_buf[i];
       }
       *result = sum;
    }
}

__global__ void test_kernel(uint32_t loops, unsigned long long *array, long long totalTicks) {
  cg::thread_block tb = cg::this_thread_block();
  unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < loops; i++) {
    long long time_diff = 0;
    long long last_clock = clock64();
    do {
      long long cur_clock = clock64();
      if (cur_clock > last_clock) {
        time_diff += (cur_clock - last_clock);
      }
      // If it rolls over, we don't know how much to add to catch up.
      // So just ignore those slipped cycles.
      last_clock = cur_clock;
    } while(time_diff < totalTicks);
    tb.sync();
    array[rank] += clock64();
  }
}

template<typename T>
static void verifyLeastCapacity(T& single_kernel_time, T& double_kernel_time, T& triple_kernel_time)
{
#if HT_AMD
  // hipLaunchCooperativeKernel() follows serialization policy on AMD devices
  // Test that the two cooperative kernels took roughly twice as long as the one
  REQUIRE(double_kernel_time.count() >= 1.8 * single_kernel_time.count());
  REQUIRE(double_kernel_time.count() <= 2.2 * single_kernel_time.count());
#else
  // hipLaunchCooperativeKernel() doesn't follow serialization policy on NV devices
  // Test that the two cooperative kernels took roughly as long as the one
  REQUIRE(double_kernel_time.count() >= 0.8 * single_kernel_time.count());
  REQUIRE(double_kernel_time.count() <= 1.2 * single_kernel_time.count());
#endif

  // Test that the three kernels together took roughly as long as the two
  // cooperative kernels.
  REQUIRE(triple_kernel_time.count() <= 1.1 * double_kernel_time.count());
}

template<typename T>
static void verifyHalfCapacity(T& single_kernel_time, T& double_kernel_time, T& triple_kernel_time)
{
  // Test that the two cooperative kernels took roughly twice as long as the one
  REQUIRE(double_kernel_time.count() >= 1.8 * single_kernel_time.count());
  REQUIRE(double_kernel_time.count() <= 2.2 * single_kernel_time.count());

  // Test that the three kernels together took roughly as long as the two
  // cooperative kernels.
  REQUIRE(triple_kernel_time.count() <= 1.1 * double_kernel_time.count());
}

template<typename T>
static void verifyFullCapacity(T& single_kernel_time, T& double_kernel_time, T& triple_kernel_time)
{
  // Test that the two cooperative kernels took roughly twice as long as the one
  REQUIRE(double_kernel_time.count() >= 1.8 * single_kernel_time.count());
  REQUIRE(double_kernel_time.count() <= 2.2 * single_kernel_time.count());

  // Test that the three kernels together took roughly 1.6 times as long as the two
  // cooperative kernels. If the first 2 kernels run very fast, the third
  // won't share much time with the second kernel.
  REQUIRE(triple_kernel_time.count() <= 1.7 * double_kernel_time.count());
}

template<typename T>
static void verify(int tests, T &single_kernel_time, T &double_kernel_time,
            T &triple_kernel_time) {
  switch (tests) {
    case 0:
      verifyLeastCapacity(single_kernel_time, double_kernel_time,
                          triple_kernel_time);
      break;
    case 1:
      verifyHalfCapacity(single_kernel_time, double_kernel_time,
                         triple_kernel_time);
      break;
    case 2:
      verifyFullCapacity(single_kernel_time, double_kernel_time,
                        triple_kernel_time);
      break;
    default:
      break;
  }
}

static void test_cooperative_streams(int dev, int p_tests) {
  hipStream_t streams[3];
  unsigned long long *dev_array[3];
  int loops = 1000;

  HIP_CHECK(hipSetDevice(dev));
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDeviceProperties(&device_properties, dev));

  // Test whether target device supports cooperative groups
  if (device_properties.cooperativeLaunch == 0) {
    std::cout << "Cooperative group support not available in device " << dev << std::endl;
    return;
  }

  // We will launch enough waves to fill up all of the GPU
  int warp_size = device_properties.warpSize;
  int num_sms = device_properties.multiProcessorCount;
  long long totalTicks = device_properties.clockRate;
  int max_blocks_per_sm = 0;
  // Calculate the device occupancy to know how many blocks can be run.
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                        test_kernel,
                                                        warp_size, 0));
  int max_active_blocks = max_blocks_per_sm * num_sms;
  int coop_blocks = 0;
  int reg_blocks  = 0;

  switch (p_tests) {
    case 0:
      // 1 block
      coop_blocks = 1;
      reg_blocks = 1;
      break;
    case 1:
      // Half capacity
      // To make sure the second kernel launched by hipLaunchCooperativeKernel
      // is invoked after the first kernel finished
      coop_blocks = max_active_blocks / 2 + 1;
      // To make sure the third kernel launched by hipLaunchKernelGGL is invoked
      // concurrently with the second kernel
      reg_blocks  = max_active_blocks - coop_blocks;
      break;
    case 2:
      // Full capacity
      coop_blocks = max_active_blocks;
      reg_blocks = max_active_blocks;
      break;
    default:
      break;
  }

  for (int i = 0; i < 3; i++) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
  }

  // Set up data to pass into the kernel

  for (int i = 0; i < 3; i++) {
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_array[i]),
                         warp_size * sizeof(long long)));
    HIP_CHECK(hipMemsetAsync(dev_array[i], 0, warp_size * sizeof(long long),
                              streams[i]));
  }

  HIP_CHECK(hipDeviceSynchronize());

  // Launch the kernels
  void *coop_params[3][3];
  for (int i = 0; i < 3; i++) {
    coop_params[i][0] = reinterpret_cast<void*>(&loops);
    coop_params[i][1] = reinterpret_cast<void*>(&dev_array[i]);
    coop_params[i][2] = reinterpret_cast<void*>(&totalTicks);
  }

  // We need exclude the the initial launching as it will need time to load code obj.
  auto single_start0 = std::chrono::system_clock::now();
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                      max_active_blocks, warp_size,
                                      coop_params[0], 0, streams[0]));
  HIP_CHECK(hipDeviceSynchronize());
  auto single_end0 = std::chrono::system_clock::now();
  
  // Launching a single cooperative kernel
  auto single_start = std::chrono::system_clock::now();
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                      max_active_blocks, warp_size,
                                      coop_params[0], 0, streams[0]));
  HIP_CHECK(hipDeviceSynchronize());
  auto single_end = std::chrono::system_clock::now();

  std::chrono::duration<double> single_kernel_time = (single_end - single_start);

  // Launching 2 cooperative kernels to different streams
  auto double_start = std::chrono::system_clock::now();
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                      coop_blocks, warp_size,
                                      coop_params[0], 0, streams[0]));
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                      coop_blocks, warp_size,
                                      coop_params[1], 0, streams[1]));

  HIP_CHECK(hipDeviceSynchronize());
  auto double_end = std::chrono::system_clock::now();

  // Launching 2 cooperative kernels and 1 normal kernel
  std::chrono::duration<double> double_kernel_time = (double_end - double_start);

  auto triple_start = std::chrono::system_clock::now();
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                      coop_blocks, warp_size,
                                        coop_params[0], 0, streams[0]));
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_kernel),
                                      coop_blocks, warp_size,
                                      coop_params[1], 0, streams[1]));
  hipLaunchKernelGGL(test_kernel, dim3(reg_blocks), dim3(warp_size),
                      0, streams[2], loops, dev_array[2], totalTicks);

  HIP_CHECK(hipDeviceSynchronize());
  auto triple_end = std::chrono::system_clock::now();
  std::chrono::duration<double> triple_kernel_time = (triple_end - triple_start);

  for (int k = 0; k < 3; ++k) {
    HIP_CHECK(hipFree(dev_array[k]));
    HIP_CHECK(hipStreamDestroy(streams[k]));
  }


  INFO("A single kernel took : " << single_kernel_time.count() << " seconds");
  INFO("Two cooperative kernels took: " << double_kernel_time.count() << " seconds");
  INFO("Two coop kernels and a third regular kernel took: " << triple_kernel_time.count() << " seconds");

  verify(p_tests, single_kernel_time, double_kernel_time, triple_kernel_time);
}

TEST_CASE("Unit_hipLaunchCooperativeKernel_Basic") {
  // Use default device for validating the test
  int device;
  int *A_h, *A_d;
  long *B_d;
  long *C_d;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Device doesn't support cooperative launch!");
    return;
  }

  size_t buffer_size = kBufferLen * sizeof(int);
  
  A_h = reinterpret_cast<int*>(malloc(buffer_size));
  for (uint32_t i = 0; i < kBufferLen; ++i) {
    A_h[i] = static_cast<int>(i);
  }

  HIP_CHECK(hipMalloc(&A_d, buffer_size));
  HIP_CHECK(hipMemcpy(A_d, A_h, buffer_size, hipMemcpyHostToDevice));
  HIP_CHECK(hipHostMalloc(&C_d, sizeof(long)));

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  dim3 dimBlock = dim3(1);
  dim3 dimGrid  = dim3(1);
  int numBlocks = 0;

  uint32_t workgroup = GENERATE(32, 64, 128, 256);

  dimBlock.x = workgroup;

  // Calculate the device occupancy to know how many blocks can be run concurrently
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
      test_gws, dimBlock.x * dimBlock.y * dimBlock.z, dimBlock.x * sizeof(long)));

  dimGrid.x = device_properties.multiProcessorCount * std::min(numBlocks, 32);
  HIP_CHECK(hipMalloc(&B_d, dimGrid.x * sizeof(long)));
 
  void *params[4];
  params[0] = (void*)&A_d;
  params[1] = (void*)&kBufferLen;
  params[2] = (void*)&B_d;
  params[3] = (void*)&C_d;

  INFO("Testing with grid size = " << dimGrid.x << " and block size = " << dimBlock.x << "\n");
  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(test_gws), dimGrid, dimBlock, params, dimBlock.x * sizeof(long), stream));

  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(*C_d == (((long)(kBufferLen) * (kBufferLen - 1)) / 2));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostFree(C_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}

TEST_CASE("Unit_hipLaunchCooperativeKernel_Streams") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  int p_tests = GENERATE(0, 1, 2);

  test_cooperative_streams(device, p_tests);
}