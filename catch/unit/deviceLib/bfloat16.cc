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

#include <hip_test_common.hh>
#include <hip/hip_bf16.h>

#include <cmath>
#include <memory>
#include <limits>
#include <algorithm>

// Struct used to generate floats from combination of various componenets
union float_holder {
  float fp32;
  struct parts_ {
    unsigned int fp32_mantisa : 16;  // ignored for bf16
    unsigned int bf16_mantisa : 7;   // bf16 mantisa
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
  unsigned int u32;
};

std::vector<float> getAllBF16() {
  constexpr unsigned char max_mantissa = std::numeric_limits<unsigned char>::max() >> 1;  // 7 bits
  const size_t max_bf16_num =
      2 /*sign*/ * std::pow(2, 8) /*exponent*/ * std::pow(2, 7) /*mantissa*/;

  std::vector<float> f_in;
  f_in.reserve(max_bf16_num);


  for (size_t s = 0; s <= 1; s++) {                                            // sign
    for (size_t e = 0; e <= std::numeric_limits<unsigned char>::max(); e++) {  // expo
      for (size_t m = 0; m <= max_mantissa; m++) {                             // man
        float_holder hold;
        hold.u32 = 0;  // Init - clear all bits

        hold.parts.sign = s;
        hold.parts.exponent = e;
        hold.parts.bf16_mantisa = m;

        f_in.push_back(hold.fp32);
      }
    }
  }
  return f_in;
}

enum MathOp { Add = 0, Sub, Mul, Div, LastOp = Div };

__device__ __hip_bfloat16 bf16_math(__hip_bfloat16 a, __hip_bfloat16 b, MathOp op) {
  switch (op) {
    case Add:
      return __hadd(a, b);
    case Sub:
      return __hsub(a, b);
    case Mul:
      return __hmul(a, b);
    case Div:
      return __hdiv(a, b);
  }
}

__device__ float fp32_math(float a, float b, MathOp op) {
  switch (op) {
    case Add:
      return a + b;
    case Sub:
      return a - b;
    case Mul:
      return a * b;
    case Div:
      return a / b;
  }
}

__device__ void bf16_math_op_kernel(float* a, float* b, float* c) {
  for (int i = Add; i < LastOp; i++) {
    auto op = static_cast<MathOp>(i);
    c[i] = __bfloat162float(bf16_math(__float2bfloat16(a[i]), __float2bfloat16(b[i]), op));
  }
}

__device__ void fp32_math_op_kernel(float* a, float* b, float* c) {
  for (int i = Add; i < LastOp; i++) {
    auto op = static_cast<MathOp>(i);
    c[i] = fp32_math(a[i], b[i], op);
  }
}

// c = a MathOp b in fp32
// conv_res = a MathOp b in bf16 converted back to fp32
__global__ void do_math(float* a, float* b, float* c, float* conv_res) {
  fp32_math_op_kernel(a, b, c);
  bf16_math_op_kernel(a, b, conv_res);
}

// Convert float -> bfloat16 -> float
__global__ void fp32_bf16_fp32(float* a, float* c, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    auto b = __float2bfloat16(a[i]);
    c[i] = __bfloat162float(b);
  }
}

__device__ unsigned bool_to_unsigned(bool in) { return in ? 1u : 0u; }

// Test equal compare
__global__ void bf16_is_equal(float* val, unsigned* res, size_t size) {
  auto i = threadIdx.x;
  if (i < size) {
    auto v1 = __float2bfloat16(val[i]);
    auto v2 = __float2bfloat16(val[i]);
    res[i] =
        bool_to_unsigned((__heq(v1, v2) && __hge(v1, v2) &&
                          __hle(v1, v2)));  // Equal, Greater Equal, Less Equal should all have true
  }
}

// Test other compare functions
__global__ void bf16_compare(float* val, unsigned* res, size_t size) {
  auto i = threadIdx.x;
  if (i < size) {
    __hip_bfloat16 v1 = __float2bfloat16(val[i]);
    __hip_bfloat16 v2 = __float2bfloat16(val[i]);

    v1 = __hadd(v1, v2);                          // v1 = v1 + v2
    bool r1 = (__hlt(v2, v1) && __hgt(v1, v2) &&  // v1 > v2
               __hne(v1, v2) &&                   // v1 != v2
               __heq(__hmax(v1, v2), v1) &&       // max(v1,v2) == v1
               __heq(__hmin(v1, v2), v2));        // min(v1,v2) == v2

    v1 = __hsub(v1, v2);      // Back to v1's original value
    bool r2 = __heq(v1, v2);  // v1 == v2

    v1 = __hmul(v1, v2);                  // v1 = v1 * v2, both have same values so square it
    bool r3 = __heq(v1, __hmul(v2, v2));  // v1 == (v2 * v2)

    v1 = hsqrt(v1);           // Back to v1's original value
    bool r4 = __heq(v1, v2);  // v1 == v2

    v1 = __hdiv(v1, v2);                          // v1 = v1/v2, both have same values
    bool r5 = __heq(v1, __float2bfloat16(1.0f));  // v1 == 1.0f

    // Uncomment to debug
    // printf("%u - %u - %u - %u - %u\n", bool_to_unsigned(r1), bool_to_unsigned(r2),
    //       bool_to_unsigned(r3), bool_to_unsigned(r4), bool_to_unsigned(r5));

    res[i] = bool_to_unsigned(r1 && r2 && r3 && r4 && r5);
  }
}

// Convert to bits
__global__ void bf16_conv_bits(float* val, unsigned short* res, size_t size) {
  auto i = threadIdx.x;
  if (i < size) {
    __hip_bfloat16 v1 = __float2bfloat16(val[i]);
    res[i] = *reinterpret_cast<unsigned short*>(&v1);
  }
}

__global__ void bf16_neg(float* in, float* out, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = __bfloat162float(__hneg(__float2bfloat16(in[i])));
  }
}

__global__ void bf16_to_short(float* in, short* s_res, unsigned short* u_res, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    s_res[i] = __bfloat16_as_short(__float2bfloat16(in[i]));
    u_res[i] = __bfloat16_as_ushort(__float2bfloat16(in[i]));
  }
}

__global__ void short_to_bf16(short* in, float* out, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = __bfloat162float(__short_as_bfloat16(in[i]));
  }
}

__global__ void ushort_to_bf16(unsigned short* in, float* out, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = __bfloat162float(__ushort_as_bfloat16(in[i]));
  }
}

TEST_CASE("Unit_bf16_basic") {
  auto f_in = getAllBF16();
  auto max_bf16_num = f_in.size();

  SECTION("Conversion float to bfloat16 to float") {
    constexpr size_t size = 256;

    float *d_a, *d_c;
    HIP_CHECK(hipMalloc(&d_a, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_c, sizeof(float) * size));

    auto h_a = std::make_unique<float[]>(size);
    auto h_c = std::make_unique<float[]>(size);

    for (size_t i = 0; i < size; i++) {
      h_a[i] = i + 1.25;
    }

    HIP_CHECK(hipMemcpy(d_a, h_a.get(), sizeof(float) * size, hipMemcpyHostToDevice));

    fp32_bf16_fp32<<<1, size>>>(d_a, d_c, size);

    HIP_CHECK(hipMemcpy(h_c.get(), d_c, sizeof(float) * size, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < size; i++) {
      INFO("Initial: " << h_a[i] << " - After Conv: " << h_c[i]);
      // The relative error should be less than 1/(2^7) since bfloat16 has 7 bits mantissa.
      REQUIRE((std::fabs(h_c[i] - h_a[i]) / h_a[i]) < (1.0 / 128.0f));
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_c));
  }

  SECTION("Math Op Accuracy") {
    constexpr size_t size = static_cast<size_t>(LastOp);

    float f_val1[size], f_val2[size], f_res[size], bf_res[size];
    for (size_t i = 0; i < size; i++) {
      f_val1[i] = i + 1.50;
      f_val2[i] = i + 1.25;
    }

    float *df_val1, *df_val2, *float_res, *bf16_res;
    HIP_CHECK(hipMalloc(&df_val1, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&df_val2, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&float_res, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&bf16_res, sizeof(float) * size));

    HIP_CHECK(hipMemcpy(df_val1, f_val1, sizeof(float) * size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(df_val2, f_val2, sizeof(float) * size, hipMemcpyHostToDevice));

    do_math<<<1, 1>>>(df_val1, df_val2, float_res, bf16_res);

    HIP_CHECK(hipMemcpy(f_res, float_res, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(bf_res, bf16_res, sizeof(float) * size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(df_val1));
    HIP_CHECK(hipFree(df_val2));
    HIP_CHECK(hipFree(float_res));
    HIP_CHECK(hipFree(bf16_res));

    for (size_t i = 0; i < size; i++) {
      INFO("FP res: " << f_res[i] << " - BF16 res: " << bf_res[i]);
      // The relative error should be less than 1/(2^7) since bfloat16 has 7 bits mantissa.
      REQUIRE((std::fabs(bf_res[i] - f_res[i]) / f_res[i]) < (1.0 / 128.0f));
    }
  }

  SECTION("Equal bfloat16") {
    constexpr size_t size = 5;
    float in[size] = {1.0f, 0.5f, -0.33333f, 0.0f, -0.0f}, *d_in;
    unsigned* d_res;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res, sizeof(unsigned) * size));

    HIP_CHECK(hipMemcpy(d_in, in, sizeof(float) * size, hipMemcpyHostToDevice));

    bf16_is_equal<<<1, size>>>(d_in, d_res, size);

    std::vector<unsigned> res(size, 0);
    HIP_CHECK(hipMemcpy(res.data(), d_res, sizeof(unsigned) * size, hipMemcpyDeviceToHost));

    REQUIRE(std::all_of(res.begin(), res.end(), [](unsigned n) { return n == 1; }));
  }

  SECTION("MathOp Compare") {
    constexpr size_t size = 7;
    float in[size] = {0.5f, 1.0f, 1.5f, 0.33333f, 2.5f, 3.0f, 3.5f}, *d_in;
    unsigned* d_res;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res, sizeof(unsigned) * size));

    HIP_CHECK(hipMemcpy(d_in, in, sizeof(float) * size, hipMemcpyHostToDevice));

    bf16_compare<<<1, size>>>(d_in, d_res, size);

    std::vector<unsigned> res(size, 0);
    HIP_CHECK(hipMemcpy(res.data(), d_res, sizeof(unsigned) * size, hipMemcpyDeviceToHost));

    REQUIRE(std::all_of(res.begin(), res.end(), [](unsigned n) { return n == 1; }));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_res));
  }

  SECTION("Bits equal") {
    constexpr size_t size = 5;
    float in[size] = {1.0f, 0.5f, 0.33333f, 3.38e38f, 3.40e38f}, *d_in;
    unsigned short* d_res;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_res, sizeof(unsigned short) * size));

    HIP_CHECK(hipMemcpy(d_in, in, sizeof(float) * size, hipMemcpyHostToDevice));

    bf16_conv_bits<<<1, size>>>(d_in, d_res, size);

    std::vector<unsigned short> res_cmp = {0x3f80, 0x3f00, 0x3eab, 0x7f7e, 0x7f80 /*Inf*/};
    std::vector<unsigned short> res(size, 0);
    HIP_CHECK(hipMemcpy(res.data(), d_res, sizeof(unsigned short) * size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_res));
    REQUIRE(res == res_cmp);
  }

  SECTION("Round trip equal") {
    constexpr size_t size = 7;
    float *d_in, *d_out;
    std::vector<float> in = {
        std::numeric_limits<float>::infinity(), -1.0f, -0.5f, -0.0f, 0.0f, 0.5f, 1.0f};
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * size));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * size));

    HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * size, hipMemcpyHostToDevice));

    fp32_bf16_fp32<<<1, size>>>(d_in, d_out, size);

    std::vector<float> res(size, 0.0f);

    HIP_CHECK(hipMemcpy(res.data(), d_out, sizeof(unsigned) * size, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    REQUIRE(in == res);
  }

  SECTION("Round trip subsection") {
    float *in, *out;
    HIP_CHECK(hipMalloc(&in, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMalloc(&out, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMemcpy(in, f_in.data(), sizeof(float) * max_bf16_num, hipMemcpyHostToDevice));

    fp32_bf16_fp32<<<(max_bf16_num / 256) + 1, 256>>>(in, out, max_bf16_num);  // round-trip

    std::vector<float> f_out(f_in.size(), 0.0f);
    HIP_CHECK(hipMemcpy(f_out.data(), out, sizeof(float) * max_bf16_num, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));

    REQUIRE(f_in.size() == f_out.size());  // Size should be equal
    for (size_t i = 0; i < f_in.size(); i++) {
      INFO("Initial: " << f_in[i] << " After Conversion: " << f_out[i]);
      if (std::isnan(f_in[i])) {  // NaNs can't be compared
        REQUIRE(std::isnan(f_out[i]));
      } else {
        REQUIRE(f_in[i] == f_out[i]);
      }
    }
  }

  SECTION("Conversion to short") {
    float* in;
    HIP_CHECK(hipMalloc(&in, sizeof(float) * max_bf16_num));
    short* s_res;
    HIP_CHECK(hipMalloc(&s_res, sizeof(short) * max_bf16_num));
    unsigned short* u_res;
    HIP_CHECK(hipMalloc(&u_res, sizeof(unsigned short) * max_bf16_num));

    HIP_CHECK(hipMemcpy(in, f_in.data(), sizeof(float) * max_bf16_num, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(s_res, 0, sizeof(short) * max_bf16_num));
    HIP_CHECK(hipMemset(u_res, 0, sizeof(unsigned short) * max_bf16_num));

    bf16_to_short<<<(max_bf16_num / 256) + 1, 256>>>(in, s_res, u_res, max_bf16_num);
    float* s_out;
    HIP_CHECK(hipMalloc(&s_out, sizeof(float) * max_bf16_num));
    float* u_out;
    HIP_CHECK(hipMalloc(&u_out, sizeof(float) * max_bf16_num));

    short_to_bf16<<<(max_bf16_num / 256) + 1, 256>>>(s_res, s_out, max_bf16_num);
    ushort_to_bf16<<<(max_bf16_num / 256) + 1, 256>>>(u_res, u_out, max_bf16_num);

    std::vector<float> f_res_s(max_bf16_num, 0.0f);
    std::vector<float> f_res_u(max_bf16_num, 0.0f);

    HIP_CHECK(
        hipMemcpy(f_res_s.data(), s_out, sizeof(float) * max_bf16_num, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(f_res_u.data(), u_out, sizeof(float) * max_bf16_num, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < f_in.size(); i++) {
      if (std::isnan(f_res_s[i])) {  // NaNs can't be compared
        REQUIRE(std::isnan(f_res_u[i]));
      } else {
        REQUIRE(f_res_s[i] == f_res_u[i]);
      }
    }

    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(s_res));
    HIP_CHECK(hipFree(u_res));
    HIP_CHECK(hipFree(s_out));
    HIP_CHECK(hipFree(u_out));
  }

  SECTION("Neg Subsection") {
    float *in, *out;
    HIP_CHECK(hipMalloc(&in, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMalloc(&out, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMemcpy(in, f_in.data(), sizeof(float) * max_bf16_num, hipMemcpyHostToDevice));

    bf16_neg<<<(max_bf16_num / 256) + 1, 256>>>(in, out, max_bf16_num);  // round-trip

    std::vector<float> f_out(f_in.size(), 0.0f);
    HIP_CHECK(hipMemcpy(f_out.data(), out, sizeof(float) * max_bf16_num, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));

    REQUIRE(f_in.size() == f_out.size());  // Size should be equal
    for (size_t i = 0; i < f_in.size(); i++) {
      INFO("Initial: " << f_in[i] << " After Conversion: " << f_out[i]);
      if (std::isnan(f_in[i])) {  // NaNs can't be compared
        REQUIRE(std::isnan(f_out[i]));
      } else {
        REQUIRE(f_in[i] == -f_out[i]);
      }
    }
  }
}

__global__ void bf162_eq(float* in, char* out, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    auto temp = __bfloat162bfloat162(__float2bfloat16(in[i]));
    out[i] = __hbequ2(temp, temp) ? 1 : 0;
  }
}

__global__ void bf162_neq(float* in, char* out, size_t size) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    auto val = __float2bfloat16(in[i]);
    auto other_val =
        __heq(__float2bfloat16(1.0f), val) ? __float2bfloat16(2.0f) : __float2bfloat16(1.0f);
    auto temp1 = __halves2bfloat162(val, other_val);
    auto temp2 = __halves2bfloat162(other_val, val);
    out[i] = (__hbneu2(temp1, temp2)) ? 1 : 0;
  }
}

TEST_CASE("Unit_bf162_basic") {
  auto f_in = getAllBF16();
  auto max_bf16_num = f_in.size();

  SECTION("Eq Operation") {
    float* in;
    HIP_CHECK(hipMalloc(&in, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMemcpy(in, f_in.data(), sizeof(float) * max_bf16_num, hipMemcpyHostToDevice));
    char* out;
    HIP_CHECK(hipMalloc(&out, sizeof(char) * max_bf16_num));
    bf162_eq<<<(max_bf16_num / 256) + 1, 256>>>(in, out, max_bf16_num);
    std::vector<char> result(max_bf16_num, 0);
    HIP_CHECK(hipMemcpy(result.data(), out, sizeof(char) * max_bf16_num, hipMemcpyDeviceToHost));
    // Cant use allof, incase of mismatch we need to show which value had a mismatch
    for (size_t i = 0; i < max_bf16_num; i++) {
      INFO("Comparing: " << f_in[i] << " for iter: " << i);
      REQUIRE(result[i] == 1);
    }
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));
  }

  SECTION("Neq Operation") {
    float* in;
    HIP_CHECK(hipMalloc(&in, sizeof(float) * max_bf16_num));
    HIP_CHECK(hipMemcpy(in, f_in.data(), sizeof(float) * max_bf16_num, hipMemcpyHostToDevice));
    char* out;
    HIP_CHECK(hipMalloc(&out, sizeof(char) * max_bf16_num));
    bf162_neq<<<(max_bf16_num / 256) + 1, 256>>>(in, out, max_bf16_num);
    std::vector<char> result(max_bf16_num, 0);
    HIP_CHECK(hipMemcpy(result.data(), out, sizeof(char) * max_bf16_num, hipMemcpyDeviceToHost));
    // Cant use allof, incase of mismatch we need to show which value had a mismatch
    for (size_t i = 0; i < max_bf16_num; i++) {
      INFO("Comparing: " << f_in[i] << " for iter: " << i);
      REQUIRE(result[i] == 1);
    }
    HIP_CHECK(hipFree(in));
    HIP_CHECK(hipFree(out));
  }
}
