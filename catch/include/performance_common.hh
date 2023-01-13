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

#include <algorithm>
#include <chrono>
#include <memory>
#include <type_traits>
#include <vector>

#include <hip_test_common.hh>

class Timer {
 public:
  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

 protected:
  Timer(float& time, hipStream_t stream) : time_(time), stream_(stream) {}

  void Record(float time) { time_ += time; }

  hipStream_t GetStream() const { return stream_; }

 private:
  float& time_;
  hipStream_t stream_;
};

class EventTimer : public Timer {
 public:
  EventTimer(float& time, hipStream_t stream = nullptr) : Timer(time, stream) {
    HIP_CHECK(hipEventCreate(&start_));
    HIP_CHECK(hipEventCreate(&stop_));
    HIP_CHECK(hipEventRecord(start_, GetStream()));
  }

  ~EventTimer() {
    hipError_t error;  // to avoid compiler warnings

    error = hipEventRecord(stop_, GetStream());
    error = hipEventSynchronize(stop_);

    float ms;
    error = hipEventElapsedTime(&ms, start_, stop_);
    Record(ms);

    error = hipEventDestroy(start_);
    error = hipEventDestroy(stop_);
  }

 private:
  hipEvent_t start_;
  hipEvent_t stop_;
};

class CpuTimer : public Timer {
 public:
  CpuTimer(float& time, hipStream_t stream = nullptr) : Timer(time, stream) {
    start_ = std::chrono::steady_clock::now();
  }

  ~CpuTimer() {
    hipError_t error;  // to avoid compiler warnings
    error = hipStreamSynchronize(GetStream());

    stop_ = std::chrono::steady_clock::now();

    std::chrono::duration<float, std::milli> ms = stop_ - start_;
    Record(ms.count());
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::chrono::time_point<std::chrono::steady_clock> stop_;
};

template <typename Derived> class Benchmark {
 public:
  Benchmark(size_t iterations = 100, size_t warmups = 10)
      : iterations_(iterations), warmups_(warmups), progress_bar_(false), display_output_(true) {
    benchmark_name_ = Catch::getResultCapture().getCurrentTestName();
  }

  Benchmark(const Benchmark&) = delete;
  Benchmark& operator=(const Benchmark&) = delete;

  void Configure(size_t iterations = 100, size_t warmups = 10, bool progress_bar = false) {
    iterations_ = iterations;
    warmups_ = warmups;
    progress_bar_ = progress_bar;
  }

  void AddSectionName(const std::string& section_name) {
    benchmark_name_ += "/" + section_name;
  }

  void SetDisplayOutput(bool display_output) {
    display_output_ = display_output;
  }

  template <typename... Args> float Run(Args&&... args) {
    if (display_output_) {
      std::cout << std::setw(110) << std::left << benchmark_name_;
    }
    auto& derived = static_cast<Derived&>(*this);

    current_ = -1;  // -1 represents warmup
    for (size_t i = 0u; i < warmups_; ++i) {
      if (progress_bar_ && display_output_) {
        std::cout << "\r" << std::setw(110) << std::left << benchmark_name_ << "\t|\t" << "warmup: [" 
            << static_cast<int>(100.f*(i+1)/warmups_) << "%]" << std::flush;
      }
      derived(args...);
    }
    time_ = .0;

    std::vector<float> samples;
    samples.reserve(iterations_);

    for (current_ = 0; current_ < iterations_; ++current_) {
      if (progress_bar_ && display_output_) {
        std::cout << "\r" << std::setw(110) << std::left << benchmark_name_ << "\t|\t" << "measurement: ["
            << static_cast<int>(100.f*(current_+1)/iterations_) << "%]" << std::flush;
      }
      derived(args...);
      samples.push_back(time_);
      time_ = .0;
    }

    float average_time = std::reduce(cbegin(samples), cend(samples)) / samples.size();

    if (display_output_) {
      std::cout << "\r" << std::setw(110) << std::left << benchmark_name_;
      std::cout << "\t|\t" << "Average time: " << average_time << " ms" << std::endl;
    }
    return average_time;
  }

 protected:
  template <bool event_based>
  using TimerType = std::conditional_t<event_based, EventTimer, CpuTimer>;

  template <bool event_based = false>
  std::unique_ptr<TimerType<event_based>> GetTimer(hipStream_t stream = nullptr) {
    return std::make_unique<TimerType<event_based>>(time_, stream);
  }

  float time() const { return time_; }

  size_t iterations() const { return iterations_; }

  size_t warmups() const { return warmups_; }

  ssize_t current() const { return current_; }

 private:
  float time_;
  size_t iterations_;
  size_t warmups_;
  ssize_t current_;
  bool display_output_;
  bool progress_bar_;
  std::string benchmark_name_;
};

constexpr bool kTimerTypeCpu = false;
constexpr bool kTimerTypeEvent = true;

#define TIMED_SECTION_STREAM(TIMER_TYPE, STREAM) if (auto _ = GetTimer<TIMER_TYPE>(STREAM); true)
#define TIMED_SECTION(TIMER_TYPE) TIMED_SECTION_STREAM(TIMER_TYPE, nullptr)

constexpr size_t operator"" _KB(unsigned long long int kb) { return kb << 10; }

constexpr size_t operator"" _MB(unsigned long long int mb) { return mb << 20; }

constexpr size_t operator"" _GB(unsigned long long int gb) { return gb << 30; }
