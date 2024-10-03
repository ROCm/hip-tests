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
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstring>

__global__ void Hfloat2halfops(float x, __half low_limit, __half up_limit,
                                int *TstResult, int TstToRun) {
  __half OutVal = 0;
  if (TstToRun == 1) {
    OutVal = __float2half_rd(x);
    if (!((low_limit <= OutVal) && (OutVal <= up_limit))) {
      *TstResult = 0;  // Indicates Test failed
    }
  } else if (TstToRun == 2) {
    OutVal = __float2half_ru(x);
    if (!((low_limit <= OutVal) && (OutVal <= up_limit))) {
    *TstResult = 0;  // Indicates Test failed
    }
  } else if (TstToRun == 3) {
    OutVal = __float2half_rz(x);
    if (!((low_limit <= OutVal) && (OutVal <= up_limit))) {
    *TstResult = 0;  // Indicates Test failed
    }
  }
}

#if HT_AMD
TEST_CASE("Unit_hfloat2half_Tsts_Functional") {
  int *Hptr = nullptr, *Dptr = nullptr;
  HIP_CHECK(hipHostMalloc(&Hptr, sizeof(int)));
  *Hptr = 1;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&Dptr), Hptr, 0));
  SECTION("Running test for hfloat2half_rd()") {
    float x = 4.1234;
    __half low_limit = 4.123, up_limit = 4.124;
    Hfloat2halfops<<<1, 1>>>(x, low_limit, up_limit, Dptr, 1);
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(*Hptr != 0);
  }
  *Hptr = 1;
  SECTION("Running test for hfloat2half_ru()") {
    float x = 5.6285;
    __half low_limit = 5.628, up_limit = 5.630;
    Hfloat2halfops<<<1, 1>>>(x, low_limit, up_limit, Dptr, 2);
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(*Hptr != 0);
  }
  *Hptr = 1;
  SECTION("Running test for hfloat2half_rz()") {
    float x = 3.2617;
    __half low_limit = 3.260, up_limit = 3.261;
    Hfloat2halfops<<<1, 1>>>(x, low_limit, up_limit, Dptr, 3);
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(*Hptr != 0);
  }
  HIP_CHECK(hipHostFree(Hptr));
}
#endif

#if HT_AMD
TEST_CASE("Unit_hfloat2half_Tsts_FloatasZero") {
  int *Hptr = nullptr, *Dptr = nullptr;
  HIP_CHECK(hipHostMalloc(&Hptr, sizeof(int)));
  *Hptr = 1;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&Dptr), Hptr, 0));
  float x = 0.00;
  __half low_limit = 0.00, up_limit = 0.00;
  SECTION("Running test for hfloat2half_rd()") {
    Hfloat2halfops<<<1, 1>>>(x, low_limit, up_limit, Dptr, 1);
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(*Hptr != 0);
  }
  SECTION("Running test for hfloat2half_ru()") {
    Hfloat2halfops<<<1, 1>>>(x, low_limit, up_limit, Dptr, 2);
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(*Hptr != 0);
  }
  SECTION("Running test for hfloat2half_rz()") {
    Hfloat2halfops<<<1, 1>>>(x, low_limit, up_limit, Dptr, 3);
    HIP_CHECK(hipStreamSynchronize(0));
    REQUIRE(*Hptr != 0);
  }
  HIP_CHECK(hipHostFree(Hptr));
}
#endif
