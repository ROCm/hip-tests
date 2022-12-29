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

#pragma once

#include <hip_test_common.hh>

template <typename UserCode, typename Derived> class Benchmark {
 public:
  Benchmark(const UserCode& user_code) : user_code_{user_code} {}

  Benchmark(const Benchmark&) = delete;
  Benchmark& operator=(const Benchmark&) = delete;

  void Configure(std::optional<size_t> iterations, std::optional<size_t> warmup) {
    if (iterations) iterations_ = iterations.value();
    if (warmup) warmup_ = warmup.value();
  }

  template <typename... Args> float Run(Args... args) {
    std::vector<float> samples;
    samples.reserve(iterations_);

    for (size_t i = 0u; i < warmup_; ++i) {
      user_code_(args...);
    }

    for (size_t i = 0u; i < iterations_; ++i) {
      auto&& derived = static_cast<Derived&>(*this);

      derived.Prologue();
      user_code_(args...);
      derived.Epilogue();

      samples.push_back(derived.Sample());
    }

    return std::reduce(cbegin(samples), cend(samples)) / samples.size();
  }

 private:
  UserCode user_code_;

  size_t iterations_ = 100;
  size_t warmup_ = 10;
};

template <typename UserCode>
class EventBenchmark : public Benchmark<UserCode, EventBenchmark<UserCode>> {
 public:
  EventBenchmark(const UserCode& user_code, hipStream_t stream = nullptr)
      : Benchmark<UserCode, EventBenchmark<UserCode>>(user_code), stream_(stream) {
    HIP_CHECK(hipEventCreate(&start_));
    HIP_CHECK(hipEventCreate(&stop_));
  }

  ~EventBenchmark() {
    HIP_CHECK(hipEventDestroy(start_));
    HIP_CHECK(hipEventDestroy(stop_));
  }

 private:
  hipEvent_t start_;
  hipEvent_t stop_;
  hipStream_t stream_;

  friend class Benchmark<UserCode, EventBenchmark<UserCode>>;

  void Prologue() { HIP_CHECK(hipEventRecord(start_, stream_)); }

  void Epilogue() {
    HIP_CHECK(hipEventRecord(stop_, stream_));
    HIP_CHECK(hipEventSynchronize(stop_));
  }

  float Sample() {
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

template <typename UserCode>
class CpuBenchmark : public Benchmark<UserCode, CpuBenchmark<UserCode>> {
 public:
  CpuBenchmark(const UserCode& user_code, hipStream_t stream = nullptr)
      : Benchmark<UserCode, CpuBenchmark<UserCode>>(user_code), stream_(stream) {}

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::chrono::time_point<std::chrono::steady_clock> stop_;
  hipStream_t stream_;

  friend class Benchmark<UserCode, CpuBenchmark<UserCode>>;

  void Prologue() { start_ = std::chrono::steady_clock::now(); }

  void Epilogue() {
    HIP_CHECK(hipStreamSynchronize(stream_));
    stop_ = std::chrono::steady_clock::now();
  }

  float Sample() {
    std::chrono::duration<float, std::milli> ms = stop_ - start_;
    return ms.count();
  }
};

constexpr size_t operator"" _KB(unsigned long long int kb) { return kb << 10; }

constexpr size_t operator"" _MB(unsigned long long int mb) { return mb << 20; }

constexpr size_t operator"" _GB(unsigned long long int gb) { return gb << 30; }