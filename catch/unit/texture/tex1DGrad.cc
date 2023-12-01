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

#include "kernels.hh"
#include "test_fixture.hh"

/**
 * @addtogroup tex1DGrad tex1DGrad
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex1DGrad` and read mode set to `hipReadModeElementType`. The
 * test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex1DGrad.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_tex1DGrad_Positive_ReadModeElementType", "", char, unsigned char, short,
                   unsigned short, int, unsigned int, float) {
  CHECK_IMAGE_SUPPORT;

  TextureTestParams<TestType> params = {};
  params.extent = make_hipExtent(1024, 0, 0);
  params.num_subdivisions = 4;
  params.GenerateTextureDesc();

  TextureTestFixture<TestType> fixture{params};

  const auto [num_threads, num_blocks] = GetLaunchConfig(1024, params.NumItersX());
  tex1DGradKernel<vec4<TestType>><<<num_blocks, num_threads>>>(
      fixture.out_alloc_d.ptr(), params.NumItersX(), fixture.tex.object(), params.Width(),
      params.num_subdivisions, params.tex_desc.normalizedCoords, 0.5f, 0.5f);

  fixture.LoadOutput();

  for (auto i = 0u; i < params.NumItersX(); ++i) {
    float x = GetCoordinate(i, params.NumItersX(), params.Width(), params.num_subdivisions,
                            params.tex_desc.normalizedCoords);

    INFO("Index: " << i);
    INFO("Filtering mode: " << FilteringModeToString(params.tex_desc.filterMode));
    INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
    INFO("Address mode: " << AddressModeToString(params.tex_desc.addressMode[0]));
    INFO("x: " << std::fixed << std::setprecision(16) << x);

    auto ref_val = fixture.tex_h.Tex1D(x, params.tex_desc);
    REQUIRE(ref_val.x == fixture.out_alloc_h[i].x);
    REQUIRE(ref_val.y == fixture.out_alloc_h[i].y);
    REQUIRE(ref_val.z == fixture.out_alloc_h[i].z);
    REQUIRE(ref_val.w == fixture.out_alloc_h[i].w);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex1DGrad` and read mode set to `hipReadModeNormalizedFloat`.
 * The test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex1DGrad.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_tex1DGrad_Positive_ReadModeNormalizedFloat", "", char, unsigned char,
                   short, unsigned short) {
  CHECK_IMAGE_SUPPORT;

  TextureTestParams<TestType> params = {};
  params.extent = make_hipExtent(1024, 0, 0);
  params.num_subdivisions = 4;
  params.GenerateTextureDesc(hipReadModeNormalizedFloat);

  TextureTestFixture<TestType, true> fixture{params};

  const auto [num_threads, num_blocks] = GetLaunchConfig(1024, params.NumItersX());
  tex1DGradKernel<vec4<float>><<<num_blocks, num_threads>>>(
      fixture.out_alloc_d.ptr(), params.NumItersX(), fixture.tex.object(), params.Width(),
      params.num_subdivisions, params.tex_desc.normalizedCoords, 0.5f, 0.5f);

  fixture.LoadOutput();

  for (auto i = 0u; i < params.NumItersX(); ++i) {
    float x = GetCoordinate(i, params.NumItersX(), params.Width(), params.num_subdivisions,
                            params.tex_desc.normalizedCoords);

    INFO("i: " << i);
    INFO("Filtering mode: " << FilteringModeToString(params.tex_desc.filterMode));
    INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
    INFO("Address mode: " << AddressModeToString(params.tex_desc.addressMode[0]));
    INFO("Filter mode: " << FilteringModeToString(params.tex_desc.filterMode));
    INFO("x: " << std::fixed << std::setprecision(16) << x);

    auto ref_val =
        Vec4Map<TestType>(fixture.tex_h.Tex1D(x, params.tex_desc), NormalizeInteger<TestType>);
    REQUIRE(ref_val.x == fixture.out_alloc_h[i].x);
    REQUIRE(ref_val.y == fixture.out_alloc_h[i].y);
    REQUIRE(ref_val.z == fixture.out_alloc_h[i].z);
    REQUIRE(ref_val.w == fixture.out_alloc_h[i].w);
  }
}