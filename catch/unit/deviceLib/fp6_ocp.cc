/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <hip/hip_fp6.h>

#include <vector>
#include <bitset>
#include <type_traits>

std::vector<__hip_fp6_storage_t> get_all_fp6_ocp_nums() {
  std::vector<__hip_fp6_storage_t> ret;
  constexpr unsigned short max_fp6_num = 0b0011'1111;
  ret.reserve(max_fp6_num + 1 );

  for (unsigned short i = 0; i <= max_fp6_num; i++) {
    ret.push_back(static_cast<__hip_fp6_storage_t>(i));
  }
  return ret;
}


template <bool is_e2m3, typename T>
__global__ void Type_to_fp6(T* f, __hip_fp6_storage_t* res, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    if constexpr (is_e2m3) {
      __hip_fp6_e2m3 tmp(f[i]);
      res[i] = tmp.__x;
    } else {
      __hip_fp6_e3m2 tmp(f[i]);
      res[i] = tmp.__x;
    }
  }
}

/* Test Description
- Generate fp6 numbers
- convert fp6 numbers to float on host
- convert float to fp6  on device
- compare fp6 result from device with original value
*/
TEMPLATE_TEST_CASE("Unit_all_fp6_ocp", "", float, double) {
  bool is_e2m3 = GENERATE(true, false);
  std::vector<TestType> f_vals;
  std::vector<__hip_fp6_storage_t> all_vals;

  SECTION("all representable numbers") {
    all_vals = get_all_fp6_ocp_nums();
    f_vals.reserve(all_vals.size());

    for (const auto& fp6 : all_vals) {
        TestType f = 0.0f;
        if ( is_e2m3) {
            __hip_fp6_e2m3 tmp;
            tmp.__x = fp6;
            f = tmp;
        } else {
            __hip_fp6_e3m2 tmp;
            tmp.__x = fp6;
            f = tmp;
        }
        f_vals.push_back(f);
    }
  }

  SECTION("Range stepped numbers") {
    constexpr TestType lhs = -30;
    constexpr TestType rhs = 30;
    constexpr TestType step = 0.1234f;

    f_vals.reserve(500);
    all_vals.reserve(500);

    for (float fval = lhs; fval <= rhs; fval += step) {
      if (is_e2m3) {
        __hip_fp6_e2m3 tmp(fval);
        all_vals.push_back(tmp.__x);
      } else {
        __hip_fp6_e3m2 tmp(fval);
        all_vals.push_back(tmp.__x);
      }
      f_vals.push_back(fval);
    }
  }

  TestType* d_f_vals;
  __hip_fp6_storage_t* d_res;

  HIP_CHECK(hipMalloc(&d_f_vals, sizeof(TestType) * f_vals.size()));
  HIP_CHECK(hipMalloc(&d_res, sizeof(__hip_fp6_storage_t) * f_vals.size()));

  HIP_CHECK(
      hipMemcpy(d_f_vals, f_vals.data(), sizeof(TestType) * f_vals.size(), hipMemcpyHostToDevice));

  auto fp6_kernel = is_e2m3 ? Type_to_fp6<true, TestType> : Type_to_fp6<false, TestType>;
  fp6_kernel<<<(f_vals.size() / 64) + 1, 64>>>(d_f_vals, d_res, f_vals.size());

  std::vector<__hip_fp6_storage_t> final_res(f_vals.size(), static_cast<__hip_fp6_storage_t>(0));

  HIP_CHECK(hipMemcpy(final_res.data(), d_res, sizeof(__hip_fp6_storage_t) * final_res.size(),
                      hipMemcpyDeviceToHost));

  for (size_t i = 0; i < final_res.size(); i++) {
    INFO("Checking: " << f_vals[i] << " for: " << (is_e2m3 ? "e2m3" : "e3m2")
                      << " original: " << (int)all_vals[i]
                      << " convert back: " << (int)final_res[i] << " Idx : " << i) ;
    TestType gpu_cvt_res = 0.0f, cpu_cvt_res = 0.0f;
    if (is_e2m3) {
      __hip_fp6_e2m3 gtmp;
      gtmp.__x = final_res[i];
      gpu_cvt_res = gtmp;
      __hip_fp6_e2m3 ctmp;
      ctmp.__x = all_vals[i];
      cpu_cvt_res = ctmp;
    } else {
      __hip_fp6_e3m2 gtmp;
      gtmp.__x = final_res[i];
      gpu_cvt_res = gtmp;
      __hip_fp6_e3m2 ctmp;
      ctmp.__x = all_vals[i];
      cpu_cvt_res = ctmp;
    }

    INFO("cpu cvt val: " << cpu_cvt_res << " gpu cvt val: " << gpu_cvt_res);
    REQUIRE(cpu_cvt_res == gpu_cvt_res);
  }

  HIP_CHECK(hipFree(d_f_vals));
  HIP_CHECK(hipFree(d_res));
}

/* Test Description
- Sanity testing for cvt x2 and x4 packed types
- convert cvt float2/float4 types back to float2/float4 on host
- compare results with original value
*/
TEMPLATE_TEST_CASE("Unit_all_fp6_ocp_vector_cvt_host", "", float, double) {
  constexpr bool is_float = std::is_same<TestType, float>::value;
  using vtype2 =  typename std::conditional<is_float, float2, double2>::type;
  using vtype4 =  typename std::conditional<is_float, float4, double4>::type;

  std::vector<vtype2> inputx2{
      {-1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, -1.0f}};
  std::vector<vtype4> inputx4{
       {-1.0f, 0.0f, 1.0f, 0.5f}, {0.0f, 1.0f, -0.5f, -1.0f}, {1.0f, 0.0f, 1.0f, -1.0f}};

  SECTION("e2m3_x2") {
    for (const auto val : inputx2) {
      __hip_fp6x2_e2m3 fp6x2(val);
      vtype2 ret = fp6x2;
      INFO("In: " << val.x << " - " << val.y);
      INFO("Out: " << ret.x << " - " << ret.y);
      REQUIRE(ret.x == val.x);
      REQUIRE(ret.y == val.y);
    }
  }
  SECTION("e2m3_x4") {
    for (const auto val : inputx4) {
      __hip_fp6x4_e2m3 fp6x4(val);
      vtype4 ret = fp6x4;
      INFO("In: " << val.x << " - " << val.y << " - " << val.z << " - " << val.w);
      INFO("Out: " << ret.x << " - " << ret.y << " - " << ret.z << " - " << ret.w);
      REQUIRE(ret.x == val.x);
      REQUIRE(ret.y == val.y);
      REQUIRE(ret.z == val.z);
      REQUIRE(ret.w == val.w);
    }
  }
  SECTION("e3m2_x2") {
    for (const auto val : inputx2) {
      __hip_fp6x2_e3m2 fp6x2(val);
      vtype2 ret = fp6x2;
      INFO("In: " << val.x << " - " << val.y);
      INFO("Out: " << ret.x << " - " << ret.y);
      REQUIRE(ret.x == val.x);
      REQUIRE(ret.y == val.y);
    }
  }
  SECTION("e3m2_x4") {
    for (const auto val : inputx4) {
      __hip_fp6x4_e3m2 fp6x4(val);
      vtype4 ret = fp6x4;
      INFO("In: " << val.x << " - " << val.y << " - " << val.z << " - " << val.w);
      INFO("Out: " << ret.x << " - " << ret.y << " - " << ret.z << " - " << ret.w);
      REQUIRE(ret.x == val.x);
      REQUIRE(ret.y == val.y);
      REQUIRE(ret.z == val.z);
      REQUIRE(ret.w == val.w);
    }
  }
}

template <bool is_e2m3, typename T> __global__ void cvt_float4_fp6x4_float4_ocp(T* in, size_t size)
{
  int i = threadIdx.x;
  if (i < size) {
    T val = in[i];
    if constexpr (is_e2m3) {
      __hip_fp6x4_e2m3 tmp(val);
      in[i] = tmp;
    } else {
      __hip_fp6x4_e3m2 tmp(val);
      in[i] = tmp;
    }
  }
}

template <bool is_e2m3, typename T> __global__ void cvt_float2_fp6x2_float2_ocp(T* in, size_t size)
{
  int i = threadIdx.x;
  if (i < size) {
    T val = in[i];
    if constexpr (is_e2m3) {
      __hip_fp6x2_e2m3 tmp(val);
      in[i] = tmp;
    } else {
      __hip_fp6x2_e3m2 tmp(val);
      in[i] = tmp;
    }
  }
}

/* Test Description
- Sanity testing for cvt x2 and x4 packed types on device
- convert cvt float2/float4 types back to float2/float4 on device
- compare results with original value
*/
TEMPLATE_TEST_CASE("Unit_all_fp6_ocp_vector_cvt_device", "", float, double) {
  constexpr bool is_float = std::is_same<TestType, float>::value;
  using vtype2 =  typename std::conditional<is_float, float2, double2>::type;
  using vtype4 =  typename std::conditional<is_float, float4, double4>::type;

  std::vector<vtype2> inputx2{
      {-1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, -1.0f}};
  std::vector<vtype4> inputx4{
       {-1.0f, 0.0f, 1.0f, 0.5f}, {0.0f, 1.0f, -0.5f, -1.0f}, {1.0f, -0.5f, 1.0f, -1.0f}};

  std::vector<vtype2> outputx2(inputx2.size());
  std::vector<vtype4> outputx4(inputx4.size());

  vtype2 *d_in2;
  vtype4 *d_in4;

  HIP_CHECK(hipMalloc(&d_in2, sizeof(vtype2) * inputx2.size()));
  HIP_CHECK(hipMalloc(&d_in4, sizeof(vtype4) * inputx4.size()));

  HIP_CHECK(hipMemcpy(d_in2, inputx2.data(), sizeof(vtype2) * inputx2.size(),
                      hipMemcpyHostToDevice));

  HIP_CHECK(hipMemcpy(d_in4, inputx4.data(), sizeof(vtype4) * inputx4.size(),
                      hipMemcpyHostToDevice));

  SECTION("e2m3_x2") {
    auto kern_fp6x2 = cvt_float2_fp6x2_float2_ocp<true, vtype2>;
    kern_fp6x2<<<1, 32>>>(d_in2, inputx2.size());
    HIP_CHECK(
      hipMemcpy(outputx2.data(), d_in2, sizeof(vtype2) * inputx2.size(), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < inputx2.size(); i++) {
      INFO("Original: " << inputx2[i].x << ", " << inputx2[i].y <<
           " Output: "  << outputx2[i].x << ", "
                        << outputx2[i].y);
      REQUIRE(inputx2[i].x == outputx2[i].x);
      REQUIRE(inputx2[i].y == outputx2[i].y);
    }
  }
  SECTION("e2m3_x4") {
    auto kern_fp6x4 = cvt_float4_fp6x4_float4_ocp<true, vtype4>;
    kern_fp6x4<<<1, 32>>>(d_in4, inputx4.size());
    HIP_CHECK(
      hipMemcpy(outputx4.data(), d_in4, sizeof(vtype4) * inputx4.size(), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < inputx4.size(); i++) {
      INFO("Original: " << inputx4[i].x << ", " << inputx4[i].y << ", " << inputx4[i].z << ", "
                        << inputx4[i].w << " Output: " << outputx4[i].x << ", " << outputx4[i].y
                        << ", "<< outputx4[i].z << ", " << outputx4[i].w);
      REQUIRE(inputx4[i].x == outputx4[i].x);
      REQUIRE(inputx4[i].y == outputx4[i].y);
      REQUIRE(inputx4[i].z == outputx4[i].z);
      REQUIRE(inputx4[i].w == outputx4[i].w);
    }
  }
  SECTION("e3m2_x2") {
    auto kern_fp6x2 = cvt_float2_fp6x2_float2_ocp<false, vtype2>;
    kern_fp6x2<<<1, 32>>>(d_in2, inputx2.size());
    HIP_CHECK(
      hipMemcpy(outputx2.data(), d_in2, sizeof(vtype2) * inputx2.size(), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < inputx2.size(); i++) {
      INFO("Original: " << inputx2[i].x << ", " << inputx2[i].y <<
           " Output: "  << outputx2[i].x << ", "
                        << outputx2[i].y);
      REQUIRE(inputx2[i].x == outputx2[i].x);
      REQUIRE(inputx2[i].y == outputx2[i].y);
    }
  }
  SECTION("e3m2_x4") {
    auto kern_fp6x4 = cvt_float4_fp6x4_float4_ocp<false, vtype4>;
    kern_fp6x4<<<1, 32>>>(d_in4, inputx4.size());
    HIP_CHECK(
      hipMemcpy(outputx4.data(), d_in4, sizeof(vtype4) * inputx4.size(), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < inputx4.size(); i++) {
      INFO("Original: " << inputx4[i].x << ", " << inputx4[i].y << ", " << inputx4[i].z << ", "
                        << inputx4[i].w << " Output: " << outputx4[i].x << ", " << outputx4[i].y
                        << ", "<< outputx4[i].z << ", " << outputx4[i].w);
      REQUIRE(inputx4[i].x == outputx4[i].x);
      REQUIRE(inputx4[i].y == outputx4[i].y);
      REQUIRE(inputx4[i].z == outputx4[i].z);
      REQUIRE(inputx4[i].w == outputx4[i].w);
    }
  }
  HIP_CHECK(hipFree(d_in2));
  HIP_CHECK(hipFree(d_in4));
}

template <bool is_e2m3, typename T> __global__ void Type_to_fp6_ocp(T* in, float *cvt1,
                                                                    float *cvt2, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    T val = in[i];
    if constexpr (is_e2m3) {
      __hip_fp6_e2m3 tmp(val);
      cvt1[i] = tmp;
      __hip_fp6_e2m3 tmp1;
      tmp1.__x = std::is_same<T, float>::value
                ? __hip_cvt_float_to_fp6(val, __HIP_E2M3, hipRoundZero)
                : __hip_cvt_double_to_fp6(val, __HIP_E2M3, hipRoundZero);
      cvt2[i] = tmp1;
   } else {
      __hip_fp6_e3m2 tmp(val);
      cvt1[i] = tmp;
      __hip_fp6_e3m2 tmp1;
      tmp1.__x = std::is_same<T, float>::value
                ? __hip_cvt_float_to_fp6(val, __HIP_E3M2, hipRoundZero)
                : __hip_cvt_double_to_fp6(val, __HIP_E3M2, hipRoundZero);
      cvt2[i] = tmp1;
    }
  }
}

/* Test Description
- Test all e2m3/e3m2 numbers on device
- copy all numbers to device and convert back to float
- compare results with original value on host
*/
TEMPLATE_TEST_CASE("Unit_fp6_ocp_full_range_device", "", float, double) {
  SECTION("e2m3_ocp") {
    std::vector<TestType> e2m3_ocp_nums = {
      0,       0.125,     0.25,
      0.375,   0.5,       0.625,
      0.75,    0.875,     1,
      1.125,   1.25,      1.375,
      1.5,     1.625,     1.75,
      1.875,   2,         2.25,
      2.5,     2.75,      3,
      3.25,    3.5,       3.75,
      4,       4.5,       5,
      5.5,     6,         6.5,
      7,       7.5,       -0,
      -0.125,  -0.25,     -0.375,
      -0.5,    -0.625,    -0.75,
      -0.875,  -1,        -1.125,
      -1.25,   -1.375,    -1.5,
      -1.625,  -1.75,     -1.875,
      -2,      -2.25,     -2.5,
      -2.75,   -3,        -3.25,
      -3.5,    -3.75,     -4,
      -4.5,    -5,        -5.5,
      -6,      -6.5,      -7,
      -7.5
    };
    size_t totalnums = e2m3_ocp_nums.size();
    TestType *fnums; HIP_CHECK(hipMalloc((void **)&fnums, totalnums * sizeof(TestType)));
    float *cvt1_dev; HIP_CHECK(hipMalloc((void **)&cvt1_dev, totalnums * sizeof(float)));
    float *cvt2_dev; HIP_CHECK(hipMalloc((void **)&cvt2_dev, totalnums * sizeof(float)));

    HIP_CHECK(hipMemcpy(fnums, e2m3_ocp_nums.data(), totalnums * sizeof(TestType),
                        hipMemcpyHostToDevice));

    auto fp6_kernel = Type_to_fp6_ocp<true, TestType>;
    fp6_kernel<<<totalnums / 32 + 1, 32>>>(fnums, cvt1_dev, cvt2_dev, totalnums);

    float *cvt1_host = (float *)malloc(sizeof(float) * totalnums);
    float *cvt2_host = (float *)malloc(sizeof(float) * totalnums);

    HIP_CHECK(hipMemcpy(cvt1_host, cvt1_dev, totalnums * sizeof(float) , hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(cvt2_host, cvt2_dev, totalnums * sizeof(float) , hipMemcpyDeviceToHost));

    HIP_CHECK(hipDeviceSynchronize());

    for (size_t idx = 0; idx < totalnums; idx++) {
      TestType orig = e2m3_ocp_nums[idx];
      float cvt1 = cvt1_host[idx];
      float cvt2 = cvt2_host[idx];

      INFO("Original: " << std::bitset<32>(*reinterpret_cast<const unsigned int*>(&orig)));
      INFO("Cvt back: " << std::bitset<32>(*reinterpret_cast<const unsigned int*>(&cvt1)));
      REQUIRE(cvt1 == Approx(orig));
      REQUIRE(cvt2 == cvt1);
    }

    HIP_CHECK(hipFree(fnums));
    HIP_CHECK(hipFree(cvt1_dev));
    HIP_CHECK(hipFree(cvt2_dev));
    free(cvt1_host);
    free(cvt2_host);
  }
  SECTION("e3m2_ocp") {
    std::vector<TestType> e3m2_ocp_nums = {
      0,      0.0625,    0.125,    0.1875,
      0.25,   0.3125,    0.375,    0.4375,
      0.5,    0.625,     0.75,     0.875,
      1,      1.25,      1.5,      1.75,
      2,      2.5,       3,        3.5,
      4,      5,         6,        7,
      8,      10,        12,       14,
      16,     20,        24,       28,
      -0,     -0.0625,   -0.125,   -0.1875,
      -0.25,  -0.3125,   -0.375,   -0.4375,
      -0.5,   -0.625,    -0.75,    -0.875,
      -1,     -1.25,     -1.5,     -1.75,
      -2,     -2.5,      -3,       -3.5,
      -4,     -5,        -6,       -7,
      -8,     -10,       -12,      -14,
      -16,    -20,       -24,      -28
    };
    size_t totalnums = e3m2_ocp_nums.size();
    TestType *fnums; HIP_CHECK(hipMalloc((void **)&fnums, totalnums * sizeof(TestType)));
    float *cvt1_dev; HIP_CHECK(hipMalloc((void **)&cvt1_dev, totalnums * sizeof(float)));
    float *cvt2_dev; HIP_CHECK(hipMalloc((void **)&cvt2_dev, totalnums * sizeof(float)));

    HIP_CHECK(hipMemcpy(fnums, e3m2_ocp_nums.data(), totalnums * sizeof(TestType),
                        hipMemcpyHostToDevice));

    auto fp6_kernel = Type_to_fp6_ocp<false, TestType>;
    fp6_kernel<<<totalnums / 32 + 1, 32>>>(fnums, cvt1_dev, cvt2_dev, totalnums);

    float *cvt1_host = (float *)malloc(sizeof(float) * totalnums);
    float *cvt2_host = (float *)malloc(sizeof(float) * totalnums);

    HIP_CHECK(hipMemcpy(cvt1_host, cvt1_dev, totalnums * sizeof(float) , hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(cvt2_host, cvt2_dev, totalnums * sizeof(float) , hipMemcpyDeviceToHost));

    HIP_CHECK(hipDeviceSynchronize());

    for (size_t idx = 0; idx < totalnums; idx++) {
      TestType orig = e3m2_ocp_nums[idx];
      float cvt1 = cvt1_host[idx];
      float cvt2 = cvt2_host[idx];

      INFO("Original: " << std::bitset<32>(*reinterpret_cast<const unsigned int*>(&orig)));
      INFO("Cvt back: " << std::bitset<32>(*reinterpret_cast<const unsigned int*>(&cvt1)));
      REQUIRE(cvt1 == Approx(orig));
      REQUIRE(cvt2 == cvt1);
    }

    HIP_CHECK(hipFree(fnums));
    HIP_CHECK(hipFree(cvt1_dev));
    HIP_CHECK(hipFree(cvt2_dev));
    free(cvt1_host);
    free(cvt2_host);
  }
}
/*
Test bfloat and half type convertions on host
*/
TEST_CASE("Unit_fp6_cvt_bfloat_half_host") {
  float f1 = 0.75f;
  float2 f2 = {1.0f, 2.0f};

  SECTION("e2m3_ocp_bfloat") {
    auto bf16_val = __float2bfloat16(f1);
    __hip_fp6_e2m3 tmp(bf16_val);
    __hip_fp6_e2m3 tmp1;
    tmp1.__x = __hip_cvt_bfloat16raw_to_fp6(bf16_val, __HIP_E2M3, hipRoundZero);
    float bf2_1 = tmp1;
    float f1_1 = tmp;
    REQUIRE(f1 == bf2_1);
    REQUIRE(f1 == f1_1);
  }
  SECTION("e2m3_ocp_bfloat2") {
    auto bf162_val = __float22bfloat162_rn(f2);
    __hip_fp6x2_e2m3 tmp(bf162_val);
    __hip_fp6x2_e2m3 tmp1;
    tmp1.__x = __hip_cvt_bfloat16raw2_to_fp6x2(bf162_val, __HIP_E2M3, hipRoundZero);
    float2 bf2_2 = tmp1;
    float2 f2_2 = tmp;
    REQUIRE(f2 == bf2_2);
    REQUIRE(f2 == f2_2);
  }
  SECTION("e3m2_ocp_bfloat") {
    auto bf16_val = __float2bfloat16(f1);
    __hip_fp6_e3m2 tmp(bf16_val);
    __hip_fp6_e3m2 tmp1;
    tmp1.__x = __hip_cvt_bfloat16raw_to_fp6(bf16_val, __HIP_E3M2, hipRoundZero);
    float bf2_1 = tmp1;
    float f1_1 = tmp;
    REQUIRE(f1 == bf2_1);
    REQUIRE(f1 == f1_1);
  }
  SECTION("e3m2_ocp_bfloat2") {
    auto bf162_val = __float22bfloat162_rn(f2);
    __hip_fp6x2_e3m2 tmp(bf162_val);
    __hip_fp6x2_e3m2 tmp1;
    tmp1.__x = __hip_cvt_bfloat16raw2_to_fp6x2(bf162_val, __HIP_E3M2, hipRoundZero);
    float2 bf2_2 = tmp1;
    float2 f2_2 = tmp;
    REQUIRE(f2 == bf2_2);
    REQUIRE(f2 == f2_2);
  }
  SECTION("e2m3_ocp_half") {
    auto half_val = __float2half(f1);
    __hip_fp6_e2m3 tmp(half_val);
    __hip_fp6_e2m3 tmp1;
    tmp1.__x = __hip_cvt_halfraw_to_fp6(half_val, __HIP_E2M3, hipRoundZero);
    float half_1 = tmp1;
    float f1_1 = tmp;
    REQUIRE(f1 == half_1);
    REQUIRE(f1 == f1_1);
  }
  SECTION("e2m3_ocp_half2") {
    auto half2_val = __float22half2_rn(f2);
    __hip_fp6x2_e2m3 tmp(half2_val);
    __hip_fp6x2_e2m3 tmp1;
    tmp1.__x = __hip_cvt_halfraw2_to_fp6x2(half2_val, __HIP_E2M3, hipRoundZero);
    float2 h2_2 = tmp1;
    float2 f2_2 = tmp;
    REQUIRE(f2 == h2_2);
    REQUIRE(f2 == f2_2);
  }
  SECTION("e3m2_ocp_half") {
    auto half_val = __float2half(f1);
    __hip_fp6_e3m2 tmp(half_val);
    __hip_fp6_e3m2 tmp1;
    tmp1.__x = __hip_cvt_halfraw_to_fp6(half_val, __HIP_E3M2, hipRoundZero);
    float half_1 = tmp1;
    float f1_1 = tmp;
    REQUIRE(f1 == half_1);
    REQUIRE(f1 == f1_1);
  }
  SECTION("e3m2_ocp_half2") {
    auto half2_val = __float22half2_rn(f2);
    __hip_fp6x2_e3m2 tmp(half2_val);
    __hip_fp6x2_e3m2 tmp1;
    tmp1.__x = __hip_cvt_halfraw2_to_fp6x2(half2_val, __HIP_E3M2, hipRoundZero);
    float2 h2_2 = tmp1;
    float2 f2_2 = tmp;
    REQUIRE(f2 == h2_2);
    REQUIRE(f2 == f2_2);
  }
}

template <bool is_e2m3, typename T> __global__ void Type_to_bfloat_half(T *in, float *cvt1,
                                                                        float *cvt2) {
  T val = in[0];
  if constexpr (is_e2m3) {
    __hip_fp6_e2m3 tmp(val);
    __hip_fp6_e2m3 tmp1;
    if constexpr (std::is_same<T, __hip_bfloat16>::value)
      tmp1.__x = __hip_cvt_bfloat16raw_to_fp6(val, __HIP_E2M3, hipRoundZero);
    else
      tmp1.__x = __hip_cvt_halfraw_to_fp6(val, __HIP_E2M3, hipRoundZero);
    *cvt1 = tmp1;
    *cvt2 = tmp;
  } else {
    __hip_fp6_e3m2 tmp(val);
    __hip_fp6_e3m2 tmp1;
    if constexpr (std::is_same<T, __hip_bfloat16>::value)
      tmp1.__x = __hip_cvt_bfloat16raw_to_fp6(val, __HIP_E3M2, hipRoundZero);
    else
      tmp1.__x = __hip_cvt_halfraw_to_fp6(val, __HIP_E3M2, hipRoundZero);
    *cvt1 = tmp1;
    *cvt2 = tmp;
  }
}

template <bool is_e2m3, typename T> __global__ void Type_to_bfloat2_half2(T *in, float2 *cvt1,
                                                                          float2 *cvt2) {
  T val = in[0];
  if constexpr (is_e2m3) {
    __hip_fp6x2_e2m3 tmp(val);
    __hip_fp6x2_e2m3 tmp1;
    if constexpr (std::is_same<T, __hip_bfloat162>::value)
      tmp1.__x = __hip_cvt_bfloat16raw2_to_fp6x2(val, __HIP_E2M3, hipRoundZero);
    else
      tmp1.__x = __hip_cvt_halfraw2_to_fp6x2(val, __HIP_E2M3, hipRoundZero);
    *cvt1 = tmp1;
    *cvt2 = tmp;
  } else {
    __hip_fp6x2_e3m2 tmp(val);
    __hip_fp6x2_e3m2 tmp1;
    if constexpr (std::is_same<T, __hip_bfloat162>::value)
      tmp1.__x = __hip_cvt_bfloat16raw2_to_fp6x2(val, __HIP_E3M2, hipRoundZero);
    else
      tmp1.__x = __hip_cvt_halfraw2_to_fp6x2(val, __HIP_E3M2, hipRoundZero);
    *cvt1 = tmp1;
    *cvt2 = tmp;
  }
}

/*
Test bfloat and half type convertions on device
*/
TEST_CASE("Unit_fp6_cvt_bfloat_half_device") {
  float f1 = 0.75f;
  float2 f2 = {1.0f, 2.0f};
  bool is_e2m3 = GENERATE(true, false);

  SECTION("fp6_ocp_bfloat") {
    auto bf16_val = __float2bfloat16(f1);
    float bf1_1, f1_1;
    __hip_bfloat16 *d_val; HIP_CHECK(hipMalloc((void **)&d_val, sizeof(__hip_bfloat16)));
    float *d_f1; HIP_CHECK(hipMalloc((void **)&d_f1, sizeof(float)));
    float *d_f2; HIP_CHECK(hipMalloc((void **)&d_f2, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_val, &bf16_val, sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    auto fp6_kernel = is_e2m3 ? Type_to_bfloat_half<true, __hip_bfloat16>
                              : Type_to_bfloat_half<false, __hip_bfloat16>;
    fp6_kernel<<<1,1>>>(d_val, d_f1, d_f2);

    HIP_CHECK(hipMemcpy(&bf1_1, d_f1, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&f1_1, d_f2, sizeof(float), hipMemcpyDeviceToHost));

    REQUIRE(f1 == bf1_1);
    REQUIRE(f1 == f1_1);

    HIP_CHECK(hipFree(d_val));
    HIP_CHECK(hipFree(d_f1));
    HIP_CHECK(hipFree(d_f2));
  }
  SECTION("fp6_ocp_bfloat2") {
    auto bf162_val = __float22bfloat162_rn(f2);
    float2 bf2_2, f2_2;

    __hip_bfloat162 *d_val; HIP_CHECK(hipMalloc((void **)&d_val, sizeof(__hip_bfloat162)));
    float2 *d_f1; HIP_CHECK(hipMalloc((void **)&d_f1, sizeof(float2)));
    float2 *d_f2; HIP_CHECK(hipMalloc((void **)&d_f2, sizeof(float2)));

    HIP_CHECK(hipMemcpy(d_val, &bf162_val, sizeof(__hip_bfloat162), hipMemcpyHostToDevice));
    auto fp6_kernel = is_e2m3 ? Type_to_bfloat2_half2<true, __hip_bfloat162>
                              : Type_to_bfloat2_half2<false, __hip_bfloat162>;
    fp6_kernel<<<1,1>>>(d_val, d_f1, d_f2);

    HIP_CHECK(hipMemcpy(&bf2_2, d_f1, sizeof(float2), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&f2_2, d_f2, sizeof(float2), hipMemcpyDeviceToHost));

    REQUIRE(f2 == bf2_2);
    REQUIRE(f2 == f2_2);
    HIP_CHECK(hipFree(d_val));
    HIP_CHECK(hipFree(d_f1));
    HIP_CHECK(hipFree(d_f2));
  }
  SECTION("fp6_ocp_half") {
    auto half_val = __float2half(f1);
    float h1_1, f1_1;
    __half *d_val; HIP_CHECK(hipMalloc((void **)&d_val, sizeof(__half)));
    float *d_f1; HIP_CHECK(hipMalloc((void **)&d_f1, sizeof(float)));
    float *d_f2; HIP_CHECK(hipMalloc((void **)&d_f2, sizeof(float)));

    HIP_CHECK(hipMemcpy(d_val, &half_val, sizeof(__half), hipMemcpyHostToDevice));
    auto fp6_kernel = is_e2m3 ? Type_to_bfloat_half<true, __half>
                              : Type_to_bfloat_half<false, __half>;
    fp6_kernel<<<1,1>>>(d_val, d_f1, d_f2);

    HIP_CHECK(hipMemcpy(&h1_1, d_f1, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&f1_1, d_f2, sizeof(float), hipMemcpyDeviceToHost));

    REQUIRE(f1 == h1_1);
    REQUIRE(f1 == f1_1);

    HIP_CHECK(hipFree(d_val));
    HIP_CHECK(hipFree(d_f1));
    HIP_CHECK(hipFree(d_f2));
  }
  SECTION("fp6_ocp_half2") {
    auto half2_val = __float22half2_rn(f2);
    float2 h2_2, f2_2;

    __half2 *d_val; HIP_CHECK(hipMalloc((void **)&d_val, sizeof(__half2)));
    float2 *d_f1; HIP_CHECK(hipMalloc((void **)&d_f1, sizeof(float2)));
    float2 *d_f2; HIP_CHECK(hipMalloc((void **)&d_f2, sizeof(float2)));

    HIP_CHECK(hipMemcpy(d_val, &half2_val, sizeof(__half2), hipMemcpyHostToDevice));
    auto fp6_kernel = is_e2m3 ? Type_to_bfloat2_half2<true, __half2>
                              : Type_to_bfloat2_half2<false, __half2>;
    fp6_kernel<<<1,1>>>(d_val, d_f1, d_f2);

    HIP_CHECK(hipMemcpy(&h2_2, d_f1, sizeof(float2), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&f2_2, d_f2, sizeof(float2), hipMemcpyDeviceToHost));

    REQUIRE(f2 == h2_2);
    REQUIRE(f2 == f2_2);
    HIP_CHECK(hipFree(d_val));
    HIP_CHECK(hipFree(d_f1));
    HIP_CHECK(hipFree(d_f2));
  }
}