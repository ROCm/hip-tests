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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_helper.hh>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <functional>
#include <vector>
#include <thread>
#include <future>
#define THREADS 8
#define MAX_NUM_THREADS 512

#define WARMUP_RUN_COUNT 10
#define TIMING_RUN_COUNT 100
#define TOTAL_RUN_COUNT WARMUP_RUN_COUNT + TIMING_RUN_COUNT
#define FILENAME "empty_kernel.code"
#define kernel_name "EmptyKernel"

void hipModuleLaunchKernel_enqueue(std::atomic_int* shared, int max_threads) {
  // resources necessary for this thread
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoad(&module, FILENAME));
  HIP_CHECK(hipModuleGetFunction(&function, module, kernel_name));
  void* kernel_params = nullptr;
  while (max_threads != shared->load(std::memory_order_acquire)) {
    break;
  }

  for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
      HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream,
                                      &kernel_params, nullptr));
  }
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipStreamDestroy(stream));
}

// thread pool
struct thread_pool {
  thread_pool(int total_threads) : max_threads(total_threads) {
  }
  void start(std::function<void(std::atomic_int*, int)> f) {
    for (int i = 0; i < max_threads; ++i) {
      threads.push_back(std::async(std::launch::async, f, &shared,
                                     max_threads));
    }
  }
  void finish() {
    for (auto&&thread : threads) {
      thread.get();
    }
    threads.clear();
    shared = 0;
  }
  ~thread_pool() {
    finish();
  }
 private:
  std::atomic_int shared {0};
  std::vector<char> buffer;
  std::vector<std::future<void>> threads;
  int max_threads = 1;
};

TEST_CASE("Unit_hipModuleLoadMultiThreaded") {
  int max_threads = min(THREADS * std::thread::hardware_concurrency(),
                        MAX_NUM_THREADS);
  thread_pool task(max_threads);

  task.start(hipModuleLaunchKernel_enqueue);
  task.finish();
}
