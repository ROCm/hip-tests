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
#include <hip/hip_runtime_api.h>

enum class LinearAllocs {
  malloc,
  mallocAndRegister,
  hipHostMalloc,
  hipMalloc,
  hipMallocManaged,
};

template <typename T> class LinearAllocGuard {
 public:
  LinearAllocGuard(const LinearAllocs allocation_type, const size_t size,
                   const unsigned int flags = 0u)
      : allocation_type_{allocation_type} {
    switch (allocation_type_) {
      case LinearAllocs::malloc:
        ptr_ = host_ptr_ = reinterpret_cast<T*>(malloc(size));
        break;
      case LinearAllocs::mallocAndRegister:
        host_ptr_ = reinterpret_cast<T*>(malloc(size));
        HIP_CHECK(hipHostRegister(host_ptr_, size, flags));
        HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&ptr_), host_ptr_, 0u));
        break;
      case LinearAllocs::hipHostMalloc:
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ptr_), size, flags));
        host_ptr_ = ptr_;
        break;
      case LinearAllocs::hipMalloc:
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr_), size));
        break;
      case LinearAllocs::hipMallocManaged:
        HIP_CHECK(hipMallocManaged(reinterpret_cast<void**>(&ptr_), size, flags ? flags : 1u));
        host_ptr_ = ptr_;
    }
  }

  LinearAllocGuard(const LinearAllocGuard&) = delete;
  LinearAllocGuard(LinearAllocGuard&&) = delete;

  ~LinearAllocGuard() {
    // No Catch macros, don't want to possibly throw in the destructor
    switch (allocation_type_) {
      case LinearAllocs::malloc:
        free(ptr_);
        break;
      case LinearAllocs::mallocAndRegister:
        // Cast to void to suppress nodiscard warnings
        static_cast<void>(hipHostUnregister(host_ptr_));
        free(host_ptr_);
        break;
      case LinearAllocs::hipHostMalloc:
        static_cast<void>(hipHostFree(ptr_));
        break;
      case LinearAllocs::hipMalloc:
      case LinearAllocs::hipMallocManaged:
        static_cast<void>(hipFree(ptr_));
    }
  }

  T* ptr() { return ptr_; };
  T* const ptr() const { return ptr_; };
  T* host_ptr() { return host_ptr_; }
  T* const host_ptr() const { return host_ptr(); }

 private:
  const LinearAllocs allocation_type_;
  T* ptr_ = nullptr;
  T* host_ptr_ = nullptr;
};

enum class Streams { nullstream, perThread, created, withFlags, withPriority };

class StreamGuard {
 public:
  StreamGuard(const Streams stream_type, unsigned int flags = hipStreamDefault, int priority = 0) : stream_type_{stream_type}, flags_{flags}, priority_{priority} {
    switch (stream_type_) {
      case Streams::nullstream:
        stream_ = nullptr;
        break;
      case Streams::perThread:
        stream_ = hipStreamPerThread;
        break;
      case Streams::created:
        HIP_CHECK(hipStreamCreate(&stream_));
        break;
      case Streams::withFlags:
        HIP_CHECK(hipStreamCreateWithFlags(&stream_, flags_));
      case Streams::withPriority:
        HIP_CHECK(hipStreamCreateWithPriority(&stream_, flags_, priority_));
    }
  }

  StreamGuard(const StreamGuard&) = delete;
  StreamGuard(StreamGuard&&) = delete;

  ~StreamGuard() {
    if (stream_type_ == Streams::created) {
      static_cast<void>(hipStreamDestroy(stream_));
    }
  }

  hipStream_t stream() const { return stream_; }

 private:
  const Streams stream_type_;
  unsigned int flags_;
  int priority_;
  hipStream_t stream_;
};

class EventsGuard {
public:
	EventsGuard(size_t N) : events_(N) {
		for (auto &e : events_) HIP_CHECK(hipEventCreate(&e));
	}

  EventsGuard(const EventsGuard&) = delete;
  EventsGuard(EventsGuard&&) = delete;

	~EventsGuard() {
		for (auto &e : events_) static_cast<void>(hipEventDestroy(e));
	}

  hipEvent_t& operator[](int index) { return events_.at(index); }

  operator hipEvent_t() const { return events_.at(0); }

  std::vector<hipEvent_t>& event_list() { return events_; }

private:
	std::vector<hipEvent_t> events_;
};

class StreamsGuard {
public:
	StreamsGuard(size_t N) : streams_(N) {
		for (auto &s : streams_) HIP_CHECK(hipStreamCreate(&s));
	}

  StreamsGuard(const StreamsGuard&) = delete;
  StreamsGuard(StreamsGuard&&) = delete;

	~StreamsGuard() {
		for (auto &s : streams_) static_cast<void>(hipStreamDestroy(s));
	}

  hipStream_t& operator[](int index) { return streams_.at(index); }

  operator hipStream_t() const { return streams_.at(0); }

  std::vector<hipStream_t>& stream_list() { return streams_; }

private:
	std::vector<hipStream_t> streams_;
};

class GraphExecGuard {
 public:
  GraphExecGuard(hipGraph_t graph) {
    HIP_CHECK(hipGraphInstantiate(&graphExec_, graph, nullptr, nullptr, 0));
  }

  GraphExecGuard(const GraphExecGuard&) = delete;
  GraphExecGuard(GraphExecGuard&&) = delete;

  ~GraphExecGuard() {
    static_cast<void>(hipGraphExecDestroy(graphExec_));
  }

  hipGraphExec_t& graphExec() { return graphExec_; }

 private:
  hipGraphExec_t graphExec_;
};

class GraphGuard {
 public:
  GraphGuard(bool explicit_create = false) {
    if ( explicit_create == true) HIP_CHECK(hipGraphCreate(&graph_, 0));
  }

  GraphGuard(const GraphGuard&) = delete;
  GraphGuard(GraphGuard&&) = delete;

  ~GraphGuard() {
    static_cast<void>(hipGraphDestroy(graph_));
  }

  hipGraph_t& graph() { return graph_; }

 private:
  hipGraph_t graph_;
};

class GraphsGuard {
 public:
  GraphsGuard(size_t N, bool explicit_create = false) : graphs_(N) {
    if ( explicit_create == true) {
      for (auto &g : graphs_) HIP_CHECK(hipGraphCreate(&g, 0));
    }
  }

  GraphsGuard(const GraphsGuard&) = delete;
  GraphsGuard(GraphsGuard&&) = delete;

  ~GraphsGuard() {
    for (auto &g : graphs_) static_cast<void>(hipGraphDestroy(g));
  }

  hipGraph_t& operator[](int index) { return graphs_.at(index); }

  operator hipGraph_t() const { return graphs_.at(0); }

  std::vector<hipGraph_t>& graph_list() { return graphs_; }

 private:
  std::vector<hipGraph_t> graphs_;
};
