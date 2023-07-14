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

#pragma once

#include <hip/hip_runtime_api.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

__host__ __device__ inline float GetCoordinate(size_t iteration, size_t N, size_t dim,
                                               size_t num_subdivisions, bool normalized_coords) {
  float x = (static_cast<float>(iteration) - N / 2) / num_subdivisions;
  return normalized_coords ? x / dim : x;
}

template <typename TexelType>
__global__ void tex1DKernel(TexelType* const out, size_t N, hipTextureObject_t tex_obj,
                            size_t width, size_t num_subdivisions, bool normalized_coords) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid >= N) return;

  float x = GetCoordinate(tid, N, width, num_subdivisions, normalized_coords);
  out[tid] = tex1D<TexelType>(tex_obj, x);
}

template <typename TexelType>
__global__ void tex2DKernel(TexelType* const out, size_t N_x, size_t N_y,
                            hipTextureObject_t tex_obj, size_t width, size_t height,
                            size_t num_subdivisions, bool normalized_coords) {
  const auto tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= N_x) return;

  const auto tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= N_y) return;

  float x = GetCoordinate(tid_x, N_x, width, num_subdivisions, normalized_coords);
  float y = GetCoordinate(tid_y, N_y, height, num_subdivisions, normalized_coords);

  out[tid_y * N_x + tid_x] = tex2D<TexelType>(tex_obj, x, y);
}

template <typename TexelType>
__global__ void tex3DKernel(TexelType* const out, size_t N_x, size_t N_y, size_t N_z,
                            hipTextureObject_t tex_obj, size_t width, size_t height, size_t depth,
                            size_t num_subdivisions, bool normalized_coords) {
  const auto tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= N_x) return;

  const auto tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= N_y) return;

  const auto tid_z = blockIdx.z * blockDim.z + threadIdx.z;
  if (tid_z >= N_z) return;

  float x = GetCoordinate(tid_x, N_x, width, num_subdivisions, normalized_coords);
  float y = GetCoordinate(tid_y, N_y, height, num_subdivisions, normalized_coords);
  float z = GetCoordinate(tid_z, N_z, depth, num_subdivisions, normalized_coords);

  out[tid_z * N_x * N_y + tid_y * N_x + tid_x] = tex3D<TexelType>(tex_obj, x, y, z);
}

template <typename TexelType>
__global__ void texCubemapKernel(TexelType* const out, size_t N_x, size_t N_y, size_t N_z,
                                 hipTextureObject_t tex_obj, size_t width, size_t height,
                                 size_t depth, size_t num_subdivisions, bool normalized_coords) {
  const auto tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= N_x) return;

  const auto tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= N_y) return;

  const auto tid_z = blockIdx.z * blockDim.z + threadIdx.z;
  if (tid_z >= N_z) return;

  float x = GetCoordinate(tid_x, N_x, width, num_subdivisions, normalized_coords);
  float y = GetCoordinate(tid_y, N_y, height, num_subdivisions, normalized_coords);
  float z = GetCoordinate(tid_z, N_z, depth, num_subdivisions, normalized_coords);

  out[tid_z * N_x * N_y + tid_y * N_x + tid_x] = texCubemap<TexelType>(tex_obj, x, y, z);
}

template <typename TexelType>
__global__ void tex1DLayeredKernel(TexelType* const out, size_t N, hipTextureObject_t tex_obj,
                                   size_t width, size_t num_subdivisions, bool normalized_coords,
                                   size_t layer) {
  const auto tid = cg::this_grid().thread_rank();
  if (tid >= N) return;

  float x = GetCoordinate(tid, N, width, num_subdivisions, normalized_coords);
  out[tid] = tex1DLayered<TexelType>(tex_obj, x, layer);
}

template <typename TexelType>
__global__ void tex2DLayeredKernel(TexelType* const out, size_t N_x, size_t N_y,
                                   hipTextureObject_t tex_obj, size_t width, size_t height,
                                   size_t num_subdivisions, bool normalized_coords, size_t layer) {
  const auto tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= N_x) return;

  const auto tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= N_y) return;

  float x = GetCoordinate(tid_x, N_x, width, num_subdivisions, normalized_coords);
  float y = GetCoordinate(tid_y, N_y, height, num_subdivisions, normalized_coords);

  out[tid_y * N_x + tid_x] = tex2DLayered<TexelType>(tex_obj, x, y, layer);
}

template <typename TexelType>
__global__ void texCubemapLayeredKernel(TexelType* const out, size_t N_x, size_t N_y, size_t N_z,
                                        hipTextureObject_t tex_obj, size_t width, size_t height,
                                        size_t depth, size_t num_subdivisions,
                                        bool normalized_coords, size_t layer) {
  const auto tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= N_x) return;

  const auto tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= N_y) return;

  const auto tid_z = blockIdx.z * blockDim.z + threadIdx.z;
  if (tid_z >= N_z) return;

  float x = GetCoordinate(tid_x, N_x, width, num_subdivisions, normalized_coords);
  float y = GetCoordinate(tid_y, N_y, height, num_subdivisions, normalized_coords);
  float z = GetCoordinate(tid_z, N_z, depth, num_subdivisions, normalized_coords);

  out[tid_z * N_x * N_y + tid_y * N_x + tid_x] =
      texCubemapLayered<TexelType>(tex_obj, x, y, z, layer);
}