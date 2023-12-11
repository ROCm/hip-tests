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

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <hip_test_common.hh>
#include <limits>

__global__ void fp16_arith_gpu(float* a, float* b, float* c) {
  c[0] = __half2float(__hadd(__float2half_rn(a[0]), __float2half_rn(b[0])));
  c[1] = __half2float(__hsub(__float2half_rn(a[1]), __float2half_rn(b[1])));
  c[2] = __half2float(__hmul(__float2half_rn(a[2]), __float2half_rn(b[2])));
  c[3] = __half2float(__hdiv(__float2half_rn(a[3]), __float2half_rn(b[3])));
  c[4] = __half2float(hfloor(__float2half_rn(a[4])));
  c[5] = __half2float(htrunc(__float2half_rn(a[5])));
  c[6] = __half2float(hceil(__float2half_rn(a[6])));
  c[7] = __half2float(hrint(__float2half_rn(a[7])));
  c[8] = __half2float(hsin(__float2half_rn(a[8])));
  c[9] = __half2float(hcos(__float2half_rn(a[9])));
  c[10] = __half2float(hexp(__float2half_rn(a[10])));
  c[11] = __half2float(hexp2(__float2half_rn(a[11])));
  c[12] = __half2float(hlog2(__float2half_rn(a[12])));
  c[13] = __half2float(hlog(__float2half_rn(a[13])));
  c[14] = __half2float(hlog10(__float2half_rn(a[14])));
  c[15] = __half2float(hsqrt(__float2half_rn(a[15])));
  c[16] = __half2float(__hneg(__float2half_rn(a[16])));
  c[17] = __half2float(hrcp(__float2half_rn(a[17])));
}

void fp16_arith_cpu(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>& c) {
  c[0] = a[0] + b[0];
  c[1] = a[0] - b[0];
  c[2] = a[0] * b[0];
  c[3] = a[0] / b[0];
  c[4] = std::floorf(a[4]);
  c[5] = std::truncf(a[5]);
  c[6] = std::ceilf(a[6]);
  c[7] = std::rintf(a[7]);
  c[8] = std::sinf(a[8]);
  c[9] = std::cosf(a[9]);
  c[10] = std::expf(a[10]);
  c[11] = std::exp2f(a[11]);
  c[12] = std::log2f(a[12]);
  c[13] = std::logf(a[13]);
  c[14] = std::log10f(a[14]);
  c[15] = std::sqrtf(a[15]);
  c[16] = -a[16];
  c[17] = 1.0f / a[17];
}

TEST_CASE("Unit_fp16_arith") {
  constexpr size_t num_of_ops = 18;
  constexpr size_t iters = 100;
  Catch::Generators::RandomFloatingGenerator<float> input1_gen(2.2f, 10.f);
  constexpr float input2 = 1.1f;
  for (size_t iter = 0; iter < iters; iter++) {
    auto input1 = input1_gen.get();

    std::vector<float> in1(num_of_ops, input1);
    std::vector<float> in2(num_of_ops, input2);
    float *din1, *din2, *dout;
    HIP_CHECK(hipMalloc(&dout, sizeof(float) * num_of_ops));
    HIP_CHECK(hipMalloc(&din1, sizeof(float) * num_of_ops));
    HIP_CHECK(hipMalloc(&din2, sizeof(float) * num_of_ops));

    HIP_CHECK(hipMemcpy(din1, in1.data(), sizeof(float) * in1.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(din2, in2.data(), sizeof(float) * in2.size(), hipMemcpyHostToDevice));

    fp16_arith_gpu<<<1, 1>>>(din1, din2, dout);
    std::vector<float> cpuout(num_of_ops, 0.0f);
    fp16_arith_cpu(in1, in2, cpuout);

    std::vector<float> out(num_of_ops, 0.0f);
    HIP_CHECK(hipMemcpy(out.data(), dout, sizeof(float) * out.size(), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < out.size(); i++) {
      INFO("Iter: " << i << " In1: " << in1[i] << " CPU res: " << cpuout[i]
                    << " GPU res: " << out[i]);
      REQUIRE(out[i] == Approx(cpuout[i]).epsilon(0.1));
    }
    HIP_CHECK(hipFree(dout));
    HIP_CHECK(hipFree(din1));
    HIP_CHECK(hipFree(din2));
  }
}

__device__ void fp162_arith_impl(float2* a, float2* b, float2* c) {
  c[0] = __half22float2(__hadd2(__float22half2_rn(a[0]), __float22half2_rn(b[0])));
  c[1] = __half22float2(__hsub2(__float22half2_rn(a[1]), __float22half2_rn(b[1])));
  c[2] = __half22float2(__hmul2(__float22half2_rn(a[2]), __float22half2_rn(b[2])));
  c[3] = __half22float2(__h2div(__float22half2_rn(a[3]), __float22half2_rn(b[3])));
  c[4] = __half22float2(h2floor(__float22half2_rn(a[4])));
  c[5] = __half22float2(h2trunc(__float22half2_rn(a[5])));
  c[6] = __half22float2(h2ceil(__float22half2_rn(a[6])));
  c[7] = __half22float2(h2rint(__float22half2_rn(a[7])));
  c[8] = __half22float2(h2sin(__float22half2_rn(a[8])));
  c[9] = __half22float2(h2cos(__float22half2_rn(a[9])));
  c[10] = __half22float2(h2exp(__float22half2_rn(a[10])));
  c[11] = __half22float2(h2exp2(__float22half2_rn(a[11])));
  c[12] = __half22float2(h2log2(__float22half2_rn(a[12])));
  c[13] = __half22float2(h2log(__float22half2_rn(a[13])));
  c[14] = __half22float2(h2log10(__float22half2_rn(a[14])));
  c[15] = __half22float2(h2sqrt(__float22half2_rn(a[15])));
  c[16] = __half22float2(__hneg2(__float22half2_rn(a[16])));
  c[17] = __half22float2(h2rcp(__float22half2_rn(a[17])));
}

__global__ void fp162_arith_gpu(float2* a, float2* b, float2* c) { fp162_arith_impl(a, b, c); }

void fp162_arith_cpu(std::vector<float2>& a, std::vector<float2>& b, std::vector<float2>& c) {
  c[0] = a[0] + b[0];
  c[1] = a[1] - b[1];
  c[2] = a[2] * b[2];
  c[3] = a[3] / b[3];
  c[4] = {std::floorf(a[4].x), std::floorf(a[4].y)};
  c[5] = {std::truncf(a[5].x), std::truncf(a[5].y)};
  c[6] = {std::ceilf(a[6].x), std::ceilf(a[6].y)};
  c[7] = {std::rintf(a[7].x), std::rintf(a[7].y)};
  c[8] = {std::sinf(a[8].x), std::sinf(a[8].y)};
  c[9] = {std::cosf(a[9].x), std::cosf(a[9].y)};
  c[10] = {std::expf(a[10].x), std::expf(a[10].y)};
  c[11] = {std::exp2f(a[11].x), std::exp2f(a[11].y)};
  c[12] = {std::log2f(a[12].x), std::log2f(a[12].y)};
  c[13] = {std::logf(a[13].x), std::logf(a[13].y)};
  c[14] = {std::log10f(a[14].x), std::log10f(a[14].y)};
  c[15] = {std::sqrtf(a[15].x), std::sqrtf(a[15].y)};
  c[16] = {-a[16].x, -a[16].y};
  c[17] = {1.0f / a[17].x, 1.0f / a[17].y};
}

TEST_CASE("Unit_fp162_arith") {
  constexpr size_t num_of_ops = 18;
  constexpr size_t iters = 100;
  Catch::Generators::RandomFloatingGenerator<float> input1_gen(2.2f, 10.f);
  for (size_t iter = 0; iter < iters; iter++) {
    auto input1 = input1_gen.get();
    auto input2 = input1_gen.get();

    std::vector<float2> in1(num_of_ops, float2{input1, input2});
    std::vector<float2> in2(num_of_ops, float2{input1_gen.get(), input1_gen.get()});
    float2 *din1, *din2, *dout;
    HIP_CHECK(hipMalloc(&dout, sizeof(float2) * num_of_ops));
    HIP_CHECK(hipMalloc(&din1, sizeof(float2) * num_of_ops));
    HIP_CHECK(hipMalloc(&din2, sizeof(float2) * num_of_ops));

    HIP_CHECK(hipMemcpy(din1, in1.data(), sizeof(float2) * in1.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(din2, in2.data(), sizeof(float2) * in2.size(), hipMemcpyHostToDevice));

    fp162_arith_gpu<<<1, 1>>>(din1, din2, dout);
    std::vector<float2> cpuout(num_of_ops, float2{0.0f, 0.0f});
    fp162_arith_cpu(in1, in2, cpuout);

    std::vector<float2> out(num_of_ops, float2{0.0f, 0.0f});
    HIP_CHECK(hipMemcpy(out.data(), dout, sizeof(float2) * out.size(), hipMemcpyDeviceToHost));

    for (size_t i = 0; i < out.size(); i++) {
      INFO("Iter: " << i << " In1: " << in1[i].x << " - " << in1[i].y << " CPU res: " << cpuout[i].x
                    << " - " << cpuout[i].y << " GPU res: " << out[i].x << " - " << out[i].y);
      REQUIRE(out[i].x == Approx(cpuout[i].x).epsilon(0.1));
      REQUIRE(out[i].y == Approx(cpuout[i].y).epsilon(0.1));
    }
    HIP_CHECK(hipFree(dout));
    HIP_CHECK(hipFree(din1));
    HIP_CHECK(hipFree(din2));
  }
}
