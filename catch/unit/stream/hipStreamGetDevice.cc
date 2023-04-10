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

#include <hip_test_common.hh>

TEST_CASE("Unit_hipStreamGetDevice_Negative") {
  hipStream_t stream;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK_ERROR(hipStreamGetDevice(nullptr, nullptr), hipErrorInvalidValue);  // null stream
  HIP_CHECK_ERROR(hipStreamGetDevice(hipStreamPerThread, nullptr),
                  hipErrorInvalidValue);                                       // stream per thread
  HIP_CHECK_ERROR(hipStreamGetDevice(stream, nullptr), hipErrorInvalidValue);  // created stream
  HIP_CHECK(hipStreamDestroy(stream));
}

// Iterate over all devices, create stream on the device and match the device we get from stream
TEST_CASE("Unit_hipStreamGetDevice_Usecase") {
  int device_count = 0;

  HIP_CHECK(hipGetDeviceCount(&device_count));
  REQUIRE(device_count != 0);  // atleast 1 device

  SECTION("Null Stream") {
    CTX_CREATE();

    hipDevice_t device_from_stream, device_from_ordinal;
    HIP_CHECK(hipStreamGetDevice(nullptr, &device_from_stream));

    HIP_CHECK(hipDeviceGet(&device_from_ordinal, 0));  // default device
    REQUIRE(device_from_stream == device_from_ordinal);

    CTX_DESTROY();
  }

  SECTION("Stream Per Thread") {
    CTX_CREATE();

    hipDevice_t device_from_stream, device_from_ordinal;
    HIP_CHECK(hipStreamGetDevice(hipStreamPerThread, &device_from_stream));

    HIP_CHECK(hipDeviceGet(&device_from_ordinal, 0));  // default device
    REQUIRE(device_from_stream == device_from_ordinal);

    CTX_DESTROY();
  }

  SECTION("Created Stream") {
    for (int i = 0; i < device_count; i++) {
      HIP_CHECK(hipSetDevice(i));

      hipDevice_t device_from_stream, device_from_ordinal;
      hipStream_t stream;

      HIP_CHECK(hipStreamCreate(&stream));
      HIP_CHECK(hipStreamGetDevice(stream, &device_from_stream));

      HIP_CHECK(hipDeviceGet(&device_from_ordinal, i));
      REQUIRE(device_from_stream == device_from_ordinal);  // maybe match uuid??
    }
  }
}
