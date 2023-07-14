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
 * @addtogroup texCubemapLayeredLod texCubemapLayeredLod
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `texCubemapLayeredLod` and read mode set to
 * `hipReadModeElementType`. The test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/texCubemapLayeredLod.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_texCubemapLayeredLod_Positive_ReadModeElementType", "", char,
                   unsigned char, short, unsigned short, int, unsigned int, float) {
  TextureTestParams<TestType> params = {0};
  params.extent = make_hipExtent(2, 2, 6);
  params.num_subdivisions = 4;
  params.layers = 1;
  params.cubemap = true;
  params.GenerateTextureDesc();

  TextureTestFixture<TestType> fixture{params};

  const auto [num_threads_x, num_blocks_x] = GetLaunchConfig(10, params.NumItersX());
  const auto [num_threads_y, num_blocks_y] = GetLaunchConfig(10, params.NumItersY());
  const auto [num_threads_z, num_blocks_z] = GetLaunchConfig(10, params.NumItersZ());

  dim3 dim_grid;
  dim_grid.x = num_blocks_x;
  dim_grid.y = num_blocks_y;
  dim_grid.z = num_blocks_z;

  dim3 dim_block;
  dim_block.x = num_threads_x;
  dim_block.y = num_threads_y;
  dim_block.z = num_threads_z;

  for (auto layer = 0u; layer < params.layers; ++layer) {
    texCubemapLayeredLodKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
        fixture.out_alloc_d.ptr(), params.NumItersX(), params.NumItersY(), params.NumItersZ(),
        fixture.tex.object(), params.Width(), params.Height(), params.Depth(),
        params.num_subdivisions, params.tex_desc.normalizedCoords, layer, 0.0);
    HIP_CHECK(hipGetLastError());

    fixture.LoadOutput();

    for (auto k = 0u; k < params.NumItersZ(); ++k) {
      for (auto j = 0u; j < params.NumItersY(); ++j) {
        for (auto i = 0u; i < params.NumItersX(); ++i) {
          float x = GetCoordinate(i, params.NumItersX(), params.Width(), params.num_subdivisions,
                                  params.tex_desc.normalizedCoords);
          float y = GetCoordinate(j, params.NumItersY(), params.Height(), params.num_subdivisions,
                                  params.tex_desc.normalizedCoords);
          float z = GetCoordinate(k, params.NumItersZ(), params.Depth(), params.num_subdivisions,
                                  params.tex_desc.normalizedCoords);

          INFO("Layer: " << layer);
          INFO("i: " << i);
          INFO("j: " << j);
          INFO("k: " << k);
          INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
          INFO("Address mode X: " << AddressModeToString(params.tex_desc.addressMode[0]));
          INFO("Address mode Y: " << AddressModeToString(params.tex_desc.addressMode[1]));
          INFO("Address mode Z: " << AddressModeToString(params.tex_desc.addressMode[2]));
          INFO("x: " << std::fixed << std::setprecision(16) << x);
          INFO("y: " << std::fixed << std::setprecision(16) << y);
          INFO("z: " << std::fixed << std::setprecision(16) << z);

          auto index = k * params.NumItersX() * params.NumItersY() + j * params.NumItersX() + i;

          const auto ref_val = fixture.tex_h.TexCubemap(x, y, z, params.tex_desc);
          REQUIRE(ref_val.x == fixture.out_alloc_h[index].x);
          REQUIRE(ref_val.y == fixture.out_alloc_h[index].y);
          REQUIRE(ref_val.z == fixture.out_alloc_h[index].z);
          REQUIRE(ref_val.w == fixture.out_alloc_h[index].w);
        }
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `texCubemapLayeredLod` and read mode set to
 * `hipReadModeNormalizedFloat`. The test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/texCubemapLayeredLod.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_texCubemapLayeredLod_Positive_ReadModeNormalizedFloat", "", char,
                   unsigned char, short, unsigned short) {
  TextureTestParams<TestType> params = {0};
  params.extent = make_hipExtent(2, 2, 6);
  params.num_subdivisions = 4;
  params.layers = 1;
  params.cubemap = true;
  params.GenerateTextureDesc(hipReadModeNormalizedFloat);

  TextureTestFixture<TestType, true> fixture{params};

  const auto [num_threads_x, num_blocks_x] = GetLaunchConfig(10, params.NumItersX());
  const auto [num_threads_y, num_blocks_y] = GetLaunchConfig(10, params.NumItersY());
  const auto [num_threads_z, num_blocks_z] = GetLaunchConfig(10, params.NumItersZ());

  dim3 dim_grid;
  dim_grid.x = num_blocks_x;
  dim_grid.y = num_blocks_y;
  dim_grid.z = num_blocks_z;

  dim3 dim_block;
  dim_block.x = num_threads_x;
  dim_block.y = num_threads_y;
  dim_block.z = num_threads_z;

  for (auto layer = 0u; layer < params.layers; ++layer) {
    texCubemapLayeredLodKernel<vec4<float>><<<dim_grid, dim_block>>>(
        fixture.out_alloc_d.ptr(), params.NumItersX(), params.NumItersY(), params.NumItersZ(),
        fixture.tex.object(), params.Width(), params.Height(), params.Depth(),
        params.num_subdivisions, params.tex_desc.normalizedCoords, layer, 0.0);
    HIP_CHECK(hipGetLastError());

    fixture.LoadOutput();

    for (auto k = 0u; k < params.NumItersZ(); ++k) {
      for (auto j = 0u; j < params.NumItersY(); ++j) {
        for (auto i = 0u; i < params.NumItersX(); ++i) {
          float x = GetCoordinate(i, params.NumItersX(), params.Width(), params.num_subdivisions,
                                  params.tex_desc.normalizedCoords);
          float y = GetCoordinate(j, params.NumItersY(), params.Height(), params.num_subdivisions,
                                  params.tex_desc.normalizedCoords);
          float z = GetCoordinate(k, params.NumItersZ(), params.Depth(), params.num_subdivisions,
                                  params.tex_desc.normalizedCoords);

          INFO("Layer: " << layer);
          INFO("i: " << i);
          INFO("j: " << j);
          INFO("k: " << k);
          INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
          INFO("Address mode X: " << AddressModeToString(params.tex_desc.addressMode[0]));
          INFO("Address mode Y: " << AddressModeToString(params.tex_desc.addressMode[1]));
          INFO("Address mode Z: " << AddressModeToString(params.tex_desc.addressMode[2]));
          INFO("x: " << std::fixed << std::setprecision(16) << x);
          INFO("y: " << std::fixed << std::setprecision(16) << y);
          INFO("z: " << std::fixed << std::setprecision(16) << z);

          auto index = k * params.NumItersX() * params.NumItersY() + j * params.NumItersX() + i;

          auto ref_val = Vec4Map<TestType>(fixture.tex_h.TexCubemap(x, y, z, params.tex_desc),
                                           NormalizeInteger<TestType>);
          REQUIRE(ref_val.x == fixture.out_alloc_h[index].x);
          REQUIRE(ref_val.y == fixture.out_alloc_h[index].y);
          REQUIRE(ref_val.z == fixture.out_alloc_h[index].z);
          REQUIRE(ref_val.w == fixture.out_alloc_h[index].w);
        }
      }
    }
  }
}