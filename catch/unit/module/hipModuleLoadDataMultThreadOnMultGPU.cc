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

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_helper.hh>
#include <fstream>
#include <vector>

#define LEN 64
#define SIZE LEN << 2
#define THREADS 8

#define FILENAME "vcpy_kernel.code"
#define kernel_name "hello_world"

static std::vector<char> load_file() {
  std::ifstream file(FILENAME, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    printf("Info:could not open code object '%s'\n", FILENAME);
  }
  return buffer;
}

static void run(const std::vector<char>& buffer, int device) {
  HIP_CHECK(hipSetDevice(device));
  hipModule_t Module;
  hipFunction_t Function;

  float *A, *B, *Ad, *Bd;
  A = new float[LEN];
  B = new float[LEN];

  for (uint32_t i = 0; i < LEN; i++) {
    A[i] = i * 1.0f;
    B[i] = 0.0f;
  }

  HIP_CHECK(hipMalloc(&Ad, SIZE));
  HIP_CHECK(hipMalloc(&Bd, SIZE));

  HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));

  HIP_CHECK(hipModuleLoadData(&Module, &buffer[0]));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void* _Ad;
    void* _Bd;
  } args;
  args._Ad = static_cast<void*>(Ad);
  args._Bd = static_cast<void*>(Bd);
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, stream,
            NULL, reinterpret_cast<void**>(&config)));

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipModuleUnload(Module));
  HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));

  for (uint32_t i = 0; i < LEN; i++) {
    REQUIRE(A[i] == B[i]);
  }
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  delete[] A;
  delete[] B;
}

struct joinable_thread : std::thread {
  template <class... Xs>
  joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...) {} // NOLINT

  joinable_thread& operator=(joinable_thread&& other) = default;
  joinable_thread(joinable_thread&& other)      = default;

  ~joinable_thread() {
    if (this->joinable())
      this->join();
  }
};

static void run_multi_threads(uint32_t n, const std::vector<char>& buffer) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }
  std::vector<joinable_thread> threads;

  for (int device =0; device < numDevices; ++device) {
    for (uint32_t i = 0; i < n; i++) {
      threads.emplace_back(std::thread{[&, device] {
        run(buffer, device);
      }
    });
    }
  }
}

TEST_CASE("Unit_hipModuleLoadDataMultGPUOnMultThread") {
  HIP_CHECK(hipInit(0));
  auto buffer = load_file();
  auto file_size = buffer.size() / (1024 * 1024);
  auto thread_count = HipTest::getHostThreadCount(file_size + 10, THREADS);
  if (thread_count == 0) {
    HipTest::HIP_SKIP_TEST("Skipping because thread_count is 0");
    return;
  }
  // run multi thread on multi devices
  run_multi_threads(thread_count, buffer);
}
