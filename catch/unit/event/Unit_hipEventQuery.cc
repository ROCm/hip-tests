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

TEST_CASE("Unit_hipEventQuery_DifferentDevice") {
  hipEvent_t event1{}, event2{};
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  REQUIRE(event1 != nullptr);
  REQUIRE(event2 != nullptr);

  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreate(&stream));
  REQUIRE(stream != nullptr);

  HipTest::BlockingContext b_context1{stream};  // og context
  // Block stream
  {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventRecord(event1, stream));

    b_context1.block_stream();  // blocked stream
    REQUIRE(b_context1.is_blocked());

    HIP_CHECK(hipEventRecord(event2, stream));

    HIP_CHECK(hipEventSynchronize(event1));
    HIP_CHECK(hipEventQuery(event1));  // Should be done

    HIP_CHECK_ERROR(hipEventQuery(event2),
                    hipErrorNotReady);  // Wont be done since stream is blocked
  }

  // If other devices are available, set it
  if (HipTest::getDeviceCount() > 1) {
    HIP_CHECK(hipSetDevice(1));
  }

  // Query from same or other device depending on availability
  {
    HIP_CHECK(hipEventQuery(event1));
    HIP_CHECK_ERROR(hipEventQuery(event2), hipErrorNotReady);

    b_context1.unblock_stream();

    HIP_CHECK(hipEventSynchronize(event2));

    // Query, should be done now
    HIP_CHECK(hipEventQuery(event2));
  }

  // Query on same device if multiple devices are present
  if (HipTest::getDeviceCount() > 1) {
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventQuery(event2));
  }

  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
  HIP_CHECK(hipStreamDestroy(stream));
}
