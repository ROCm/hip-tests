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
 * @addtogroup texCubemap texCubemap
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `texCubemap` and read mode set to `hipReadModeElementType`. The
 * test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/texCubemap.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_texCubemap_Positive_ReadModeElementType", "", /*char, unsigned char, short,
                   unsigned short, int, unsigned int,*/
                   float) {
  hipChannelFormatDesc channel_desc = hipCreateChannelDesc<float>();
  hipArray* array;
  HIP_CHECK(hipMalloc3DArray(&array, &channel_desc, make_hipExtent(4, 4, 6), hipArrayLayered));

  TextureTestParams<TestType> params = {0};
  params.extent = make_hipExtent(2, 2, 6);
  params.num_subdivisions = 4;
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

  texCubemapKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
      fixture.out_alloc_d.ptr(), params.NumItersX(), params.NumItersY(), params.NumItersZ(),
      fixture.tex.object(), params.Width(), params.Height(), params.Depth(),
      params.num_subdivisions, params.tex_desc.normalizedCoords);
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

        const auto ref_val = fixture.tex_h.Tex3D(x, y, z, params.tex_desc);
        REQUIRE(ref_val.x == fixture.out_alloc_h[index].x);
        REQUIRE(ref_val.y == fixture.out_alloc_h[index].y);
        REQUIRE(ref_val.z == fixture.out_alloc_h[index].z);
        REQUIRE(ref_val.w == fixture.out_alloc_h[index].w);
      }
    }
  }
}


template <typename T>
__global__ void transformKernelCubemap(T* output, hipTextureObject_t texture, int width) {
  // For more details, check
  // https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/simpleCubemapTexture/simpleCubemapTexture.cu
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  float u = ((x + 0.5f) / (float)width) * 2.f - 1.f;
  float v = ((y + 0.5f) / (float)width) * 2.f - 1.f;

  float cx, cy, cz;

  for (unsigned int face = 0; face < 6; ++face) {
    if (face == 0) {
      cx = 1;
      cy = -v;
      cz = -u;
    } else if (face == 1) {
      cx = -1;
      cy = -v;
      cz = u;
    } else if (face == 2) {
      cx = u;
      cy = 1;
      cz = v;
    } else if (face == 3) {
      cx = u;
      cy = -1;
      cz = -v;
    } else if (face == 4) {
      cx = u;
      cy = -v;
      cz = 1;
    } else {
      cx = -u;
      cy = -v;
      cz = -1;
    }

    output[face * width * width + y * width + x] = texCubemap<T>(texture, cx, cy, cz);
  }
}

TEST_CASE("BLA") {
  using TestType = unsigned short;
  const int width{4};
  const int faces{6};
  const int allocation_size{width * width * faces * sizeof(TestType)};

  LinearAllocGuard<TestType> input_h(LinearAllocs::malloc, allocation_size);
  for (int k = 0; k < faces; ++k) {
    for (int j = 0; j < width; ++j) {
      for (int i = 0; i < width; ++i) {
        auto index = width * width * k + j * width + i;
        input_h.host_ptr()[index] = index;
      }
    }
  }

  LinearAllocGuard<TestType> output_h(LinearAllocs::malloc, allocation_size);
  LinearAllocGuard<TestType> output_d(LinearAllocs::hipMalloc, allocation_size);
  ArrayAllocGuard<TestType> array_d(make_hipExtent(width, width, faces), hipArrayCubemap);
  hipMemcpy3DParms params{};
  params.srcPos = make_hipPos(0, 0, 0);
  params.dstPos = make_hipPos(0, 0, 0);
  params.srcPtr = make_hipPitchedPtr(input_h.host_ptr(), width * sizeof(TestType), width, width);
  params.dstArray = array_d.ptr();
  params.extent = make_hipExtent(width, width, faces);
  params.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&params));

  hipTextureObject_t texture{};
  hipResourceDesc res_desc{};
  res_desc.res.array.array = array_d.ptr();
  res_desc.resType = hipResourceTypeArray;
  hipTextureDesc tex_desc{};
  tex_desc.addressMode[0] = hipAddressModeBorder;
  tex_desc.addressMode[1] = hipAddressModeBorder;
  tex_desc.addressMode[2] = hipAddressModeBorder;
  tex_desc.filterMode = hipFilterModePoint;
  HIP_CHECK(hipCreateTextureObject(&texture, &res_desc, &tex_desc, nullptr));

  dim3 numThreads(4, 4);
  dim3 numBlocks((width + numThreads.x - 1) / numThreads.x,
                 (width + numThreads.y - 1) / numThreads.y);

  std::cout << "==============================================================================="
            << std::endl;

  transformKernelCubemap<<<numThreads, numBlocks, 0, 0>>>(output_d.ptr(), texture, width);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(output_h.host_ptr(), output_d.ptr(), allocation_size, hipMemcpyDeviceToHost));

  for (int k = 0; k < faces; ++k) {
    for (int j = 0; j < width; ++j) {
      for (int i = 0; i < width; ++i) {
        auto index = width * width * k + j * width + i;
        std::cout << std::setw(8) << output_h.host_ptr()[index] << "/" << input_h.host_ptr()[index];
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  HIP_CHECK(hipDestroyTextureObject(texture));
}