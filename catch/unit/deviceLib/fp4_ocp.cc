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

#include <algorithm>

#include <hip_test_common.hh>

#include <hip/hip_fp4.h>


TEST_CASE("Unit_ocp_fp4_sanity_host") {
  SECTION("sanityx1") {
    std::vector<float> inputs{-1.0f, 0.0f, 1.0f};
    for (const auto input : inputs) {
      __hip_fp4_e2m1 fp4(input);
      float ret = fp4;
      INFO("Original: " << input << " Return: " << ret);
      REQUIRE(ret == input);
    }
  }

  SECTION("sanityx2") {
    std::vector<float2> inputs{
        {-1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, -1.0f}};
    for (const auto input : inputs) {
      __hip_fp4x2_e2m1 fp4x2(input);
      float2 ret = fp4x2;
      INFO("Original: " << input.x << ", " << input.y << " Return: " << ret.x << ", " << ret.y);
      REQUIRE(ret.x == input.x);
      REQUIRE(ret.y == input.y);
    }
  }

  SECTION("sanityx4") {
    std::vector<float4> inputs{
        {-1.0f, 0.0f, 1.0f, 0.5f}, {0.0f, 1.0f, -0.5f, -1.0f}, {1.0f, 0.0f, 1.0f, -1.0f}};
    for (const auto& input : inputs) {
      __hip_fp4x4_e2m1 fp4x4(input);
      float4 ret = fp4x4;
      INFO("Original: " << input.x << ", " << input.y << ", " << input.z << ", " << input.w
                        << " Return: " << ret.x << ", " << ret.y << ret.z << ", " << ret.w);
      REQUIRE(ret.x == input.x);
      REQUIRE(ret.y == input.y);
      REQUIRE(ret.z == input.z);
      REQUIRE(ret.w == input.w);
    }
  }
}

template <typename Lambda, typename... Type>
static __global__ void lambda_kernel_launch(Lambda l, Type... args) {
  l(args...);
}

TEST_CASE("Unit_ocp_fp4_sanity_device") {
  SECTION("sanityx1") {
    auto fp4x1_l = [] __device__(float* inputs, float* outputs, size_t size) {
      int i = threadIdx.x;
      if (i < size) {
        __hip_fp4_e2m1 fp4(inputs[i]);
        outputs[i] = fp4;
      }
    };

    std::vector<float> inputs{-1.0f, 0.0f, 1.0f};
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * inputs.size()));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * inputs.size()));

    HIP_CHECK(hipMemcpy(d_in, inputs.data(), sizeof(float) * inputs.size(), hipMemcpyHostToDevice));
    lambda_kernel_launch<<<1, 32>>>(fp4x1_l, d_in, d_out, inputs.size());
    std::vector<float> outputs(inputs.size(), 0.0f);
    HIP_CHECK(
        hipMemcpy(outputs.data(), d_out, sizeof(float) * inputs.size(), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < inputs.size(); i++) {
      INFO("Original: " << inputs[i] << " Output: " << outputs[i]);
      REQUIRE(inputs[i] == outputs[i]);
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("sanityx2") {
    auto fp4x2_l = [] __device__(float2 * inputs, float2 * outputs, size_t size) {
      int i = threadIdx.x;
      if (i < size) {
        __hip_fp4x2_e2m1 fp4(inputs[i]);
        outputs[i] = fp4;
      }
    };

    std::vector<float2> inputs{
        {-1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, -1.0f}, {1.0f, 0.0f}, {0.0f, -1.0f}};
    float2 *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float2) * inputs.size()));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float2) * inputs.size()));

    HIP_CHECK(
        hipMemcpy(d_in, inputs.data(), sizeof(float2) * inputs.size(), hipMemcpyHostToDevice));
    lambda_kernel_launch<<<1, 32>>>(fp4x2_l, d_in, d_out, inputs.size());
    std::vector<float2> outputs(inputs.size());
    HIP_CHECK(
        hipMemcpy(outputs.data(), d_out, sizeof(float2) * inputs.size(), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < inputs.size(); i++) {
      INFO("Original: " << inputs[i].x << ", " << inputs[i].y << " Output: " << outputs[i].x << ", "
                        << outputs[i].y);
      REQUIRE(inputs[i].x == outputs[i].x);
      REQUIRE(inputs[i].y == outputs[i].y);
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }

  SECTION("sanityx4") {
    auto fp4x4_l = [] __device__(float4 * inputs, float4 * outputs, size_t size) {
      int i = threadIdx.x;
      if (i < size) {
        __hip_fp4x4_e2m1 fp4(inputs[i]);
        outputs[i] = fp4;
      }
    };

    std::vector<float4> inputs{
        {-1.0f, 0.0f, 1.0f, 0.5f}, {0.0f, 1.0f, -0.5f, -1.0f}, {1.0f, 0.0f, 1.0f, -1.0f}};
    float4 *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float4) * inputs.size()));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float4) * inputs.size()));

    HIP_CHECK(
        hipMemcpy(d_in, inputs.data(), sizeof(float4) * inputs.size(), hipMemcpyHostToDevice));
    lambda_kernel_launch<<<1, 32>>>(fp4x4_l, d_in, d_out, inputs.size());
    std::vector<float4> outputs(inputs.size());
    HIP_CHECK(
        hipMemcpy(outputs.data(), d_out, sizeof(float4) * inputs.size(), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < inputs.size(); i++) {
      INFO("Original: " << inputs[i].x << ", " << inputs[i].y << ", " << inputs[i].z << ", "
                        << inputs[i].w << " Output: " << outputs[i].x << ", " << outputs[i].y
                        << ", " << outputs[i].z << ", " << outputs[i].w);
      REQUIRE(inputs[i].x == outputs[i].x);
      REQUIRE(inputs[i].y == outputs[i].y);
      REQUIRE(inputs[i].z == outputs[i].z);
      REQUIRE(inputs[i].w == outputs[i].w);
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
  }
}

TEST_CASE("Unit_ocp_fp4_full_range_host") {
  // FP4 is -6 to +6
  std::vector<float> all_fp4{-6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f,
                             0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f};
  std::vector<float> in;
  in.reserve(30);
  for (float i = -6.0f; i <= 6.0f; i += 0.5f) {
    in.push_back(i);
  }

  std::vector<float> expected{-6.0f, -6.0f, -4.0f, -4.0f, -4.0f, -4.0f, -3.0f, -2.0f,
                              -2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f,
                              2.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 6.0f, 6.0f};

  for (size_t i = 0; i < in.size(); i++) {
    __hip_fp4_e2m1 fp4(in[i]);
    float fp32 = fp4;
    INFO("Original: " << in[i] << " Output: " << fp32 << " Expected: " << expected[i]);
    REQUIRE(expected[i] == fp32);
  }
}

TEST_CASE("Unit_ocp_fp4_full_range_device") {
  std::vector<float> all_fp4{-6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f,
                             0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f};
  auto fp4x1_l = [] __device__(float* inputs, float* outputs, size_t size) {
    int i = threadIdx.x;
    if (i < size) {
      __hip_fp4_e2m1 fp4(inputs[i]);
      outputs[i] = fp4;
    }
  };

  std::vector<float> inputs;
  inputs.reserve(30);
  for (float i = -6.0f; i <= 6.0f; i += 0.5f) {
    inputs.push_back(i);
  }

  std::vector<float> expected{-6.0f, -6.0f, -4.0f, -4.0f, -4.0f, -4.0f, -3.0f, -2.0f,
                              -2.0f, -1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f,
                              2.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f, 6.0f, 6.0f};

  float *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, sizeof(float) * inputs.size()));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * inputs.size()));

  HIP_CHECK(hipMemcpy(d_in, inputs.data(), sizeof(float) * inputs.size(), hipMemcpyHostToDevice));
  lambda_kernel_launch<<<1, 32>>>(fp4x1_l, d_in, d_out, inputs.size());

  std::vector<float> outputs(inputs.size(), 0.0f);
  HIP_CHECK(hipMemcpy(outputs.data(), d_out, sizeof(float) * inputs.size(), hipMemcpyDeviceToHost));

  for (size_t i = 0; i < inputs.size(); i++) {
    INFO("Original: " << inputs[i] << " Output: " << outputs[i] << " Expected: " << expected[i]);
    REQUIRE(expected[i] == outputs[i]);
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}
/*
Test bfloat and half type convertions on host
*/
TEST_CASE("Unit_fp4_cvt_bfloat_half_host") {
  float f1 = 0.5f;
  float2 f2 = {-1.0f, 2.0f};

  SECTION("e2m1_ocp_bfloat") {
    auto bf16_val = __float2bfloat16(f1);
    __hip_fp4_e2m1 tmp(bf16_val);
    __hip_fp4_e2m1 tmp1;
    tmp1.__x = __hip_cvt_bfloat16raw_to_fp4(bf16_val, __HIP_E2M1, hipRoundZero);
    float bf2_1 = tmp1;
    float f1_1 = tmp;
    REQUIRE(f1 == bf2_1);
    REQUIRE(f1 == f1_1);
  }
  SECTION("e2m1_ocp_bfloat2") {
    auto bf162_val = __float22bfloat162_rn(f2);
    __hip_fp4x2_e2m1 tmp(bf162_val);
    __hip_fp4x2_e2m1 tmp1;
    tmp1.__x = __hip_cvt_bfloat16raw2_to_fp4x2(bf162_val, __HIP_E2M1, hipRoundZero);
    float2 bf2_2 = tmp1;
    float2 f2_2 = tmp;
    REQUIRE(f2 == bf2_2);
    REQUIRE(f2 == f2_2);
  }
  SECTION("e2m1_ocp_half") {
    auto half_val = __float2half(f1);
    __hip_fp4_e2m1 tmp(half_val);
    __hip_fp4_e2m1 tmp1;
    tmp1.__x = __hip_cvt_halfraw_to_fp4(half_val, __HIP_E2M1, hipRoundZero);
    float half_1 = tmp1;
    float f1_1 = tmp;
    REQUIRE(f1 == half_1);
    REQUIRE(f1 == f1_1);
  }
  SECTION("e2m1_ocp_half2") {
    auto half2_val = __float22half2_rn(f2);
    __hip_fp4x2_e2m1 tmp(half2_val);
    __hip_fp4x2_e2m1 tmp1;
    tmp1.__x = __hip_cvt_halfraw2_to_fp4x2(half2_val, __HIP_E2M1, hipRoundZero);
    float2 h2_2 = tmp1;
    float2 f2_2 = tmp;
    REQUIRE(f2 == h2_2);
    REQUIRE(f2 == f2_2);
  }
}

template <typename T> __global__ void Type_to_bfloat_half(T *in, float *cvt1,
                                                                        float *cvt2) {
  T val = in[0];
  __hip_fp4_e2m1 tmp(val);
  __hip_fp4_e2m1 tmp1;
  if constexpr (std::is_same<T, __hip_bfloat16>::value)
    tmp1.__x = __hip_cvt_bfloat16raw_to_fp4(val, __HIP_E2M1, hipRoundZero);
  else
    tmp1.__x = __hip_cvt_halfraw_to_fp4(val, __HIP_E2M1, hipRoundZero);
  *cvt1 = tmp1;
  *cvt2 = tmp;
}

template <typename T> __global__ void Type_to_bfloat2_half2(T *in, float2 *cvt1,
                                                                          float2 *cvt2) {
  T val = in[0];
  __hip_fp4x2_e2m1 tmp(val);
  __hip_fp4x2_e2m1 tmp1;
  if constexpr (std::is_same<T, __hip_bfloat162>::value)
    tmp1.__x = __hip_cvt_bfloat16raw2_to_fp4x2(val, __HIP_E2M1, hipRoundZero);
  else
    tmp1.__x = __hip_cvt_halfraw2_to_fp4x2(val, __HIP_E2M1, hipRoundZero);
  *cvt1 = tmp1;
  *cvt2 = tmp;
}

/*
Test bfloat and half type convertions on device
*/

TEST_CASE("Unit_fp4_cvt_bfloat_half_device") {
  float f1 = 0.5f;
  float2 f2 = {-1.0f, 2.0f};

  SECTION("fp4_ocp_bfloat") {
    auto bf16_val = __float2bfloat16(f1);
    float bf1_1, f1_1;
    __hip_bfloat16 *d_val; HIP_CHECK(hipMalloc((void **)&d_val, sizeof(__hip_bfloat16)));
    float *d_f1; HIP_CHECK(hipMalloc((void **)&d_f1, sizeof(float)));
    float *d_f2; HIP_CHECK(hipMalloc((void **)&d_f2, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_val, &bf16_val, sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    auto fp4_kernel = Type_to_bfloat_half<__hip_bfloat16>;
    fp4_kernel<<<1,1>>>(d_val, d_f1, d_f2);

    HIP_CHECK(hipMemcpy(&bf1_1, d_f1, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&f1_1, d_f2, sizeof(float), hipMemcpyDeviceToHost));

    REQUIRE(f1 == bf1_1);
    REQUIRE(f1 == f1_1);

    HIP_CHECK(hipFree(d_val));
    HIP_CHECK(hipFree(d_f1));
    HIP_CHECK(hipFree(d_f2));
  }
  SECTION("fp4_ocp_bfloat2") {
    auto bf162_val = __float22bfloat162_rn(f2);
    float2 bf2_2, f2_2;

    __hip_bfloat162 *d_val; HIP_CHECK(hipMalloc((void **)&d_val, sizeof(__hip_bfloat162)));
    float2 *d_f1; HIP_CHECK(hipMalloc((void **)&d_f1, sizeof(float2)));
    float2 *d_f2; HIP_CHECK(hipMalloc((void **)&d_f2, sizeof(float2)));

    HIP_CHECK(hipMemcpy(d_val, &bf162_val, sizeof(__hip_bfloat162), hipMemcpyHostToDevice));
    auto fp4_kernel = Type_to_bfloat2_half2<__hip_bfloat162>;
    fp4_kernel<<<1,1>>>(d_val, d_f1, d_f2);

    HIP_CHECK(hipMemcpy(&bf2_2, d_f1, sizeof(float2), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&f2_2, d_f2, sizeof(float2), hipMemcpyDeviceToHost));

    REQUIRE(f2 == bf2_2);
    REQUIRE(f2 == f2_2);
    HIP_CHECK(hipFree(d_val));
    HIP_CHECK(hipFree(d_f1));
    HIP_CHECK(hipFree(d_f2));
  }
  SECTION("fp4_ocp_half") {
    auto half_val = __float2half(f1);
    float h1_1, f1_1;
    __half *d_val; HIP_CHECK(hipMalloc((void **)&d_val, sizeof(__half)));
    float *d_f1; HIP_CHECK(hipMalloc((void **)&d_f1, sizeof(float)));
    float *d_f2; HIP_CHECK(hipMalloc((void **)&d_f2, sizeof(float)));

    HIP_CHECK(hipMemcpy(d_val, &half_val, sizeof(__half), hipMemcpyHostToDevice));
    auto fp4_kernel = Type_to_bfloat_half<__half>;
    fp4_kernel<<<1,1>>>(d_val, d_f1, d_f2);

    HIP_CHECK(hipMemcpy(&h1_1, d_f1, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&f1_1, d_f2, sizeof(float), hipMemcpyDeviceToHost));

    REQUIRE(f1 == h1_1);
    REQUIRE(f1 == f1_1);

    HIP_CHECK(hipFree(d_val));
    HIP_CHECK(hipFree(d_f1));
    HIP_CHECK(hipFree(d_f2));
  }
  SECTION("fp4_ocp_half2") {
    auto half2_val = __float22half2_rn(f2);
    float2 h2_2, f2_2;

    __half2 *d_val; HIP_CHECK(hipMalloc((void **)&d_val, sizeof(__half2)));
    float2 *d_f1; HIP_CHECK(hipMalloc((void **)&d_f1, sizeof(float2)));
    float2 *d_f2; HIP_CHECK(hipMalloc((void **)&d_f2, sizeof(float2)));

    HIP_CHECK(hipMemcpy(d_val, &half2_val, sizeof(__half2), hipMemcpyHostToDevice));
    auto fp4_kernel = Type_to_bfloat2_half2<__half2>;
    fp4_kernel<<<1,1>>>(d_val, d_f1, d_f2);

    HIP_CHECK(hipMemcpy(&h2_2, d_f1, sizeof(float2), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&f2_2, d_f2, sizeof(float2), hipMemcpyDeviceToHost));

    REQUIRE(f2 == h2_2);
    REQUIRE(f2 == f2_2);
    HIP_CHECK(hipFree(d_val));
    HIP_CHECK(hipFree(d_f1));
    HIP_CHECK(hipFree(d_f2));
  }
}
