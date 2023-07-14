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

#include <cmath>

#include "fixed_point.hh"

template <typename TexelType> class TextureReference {
 public:
  TextureReference(TexelType* alloc, hipExtent extent, size_t layers)
      : alloc_{alloc}, extent_{extent}, layers_{layers} {}

  TexelType Tex1D(float x, const hipTextureDesc& tex_desc) const {
    return Tex1DLayered(x, 0, tex_desc);
  }

  TexelType Tex2D(float x, float y, const hipTextureDesc& tex_desc) const {
    return Tex2DLayered(x, y, 0, tex_desc);
  }

  TexelType Tex3D(float x, float y, float z, const hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * extent_.width : x;
    y = tex_desc.normalizedCoords ? y * extent_.height : y;
    z = tex_desc.normalizedCoords ? z * extent_.depth : z;
    if (tex_desc.filterMode == hipFilterModePoint) {
      return Sample(floorf(x), floorf(y), floorf(z), tex_desc.addressMode);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      return LinearFiltering(x, y, z, tex_desc.addressMode);
    } else {
      throw std::invalid_argument("Invalid hipFilterMode value");
    }
  }

  TexelType Tex1DLayered(float x, int layer, const hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * extent_.width : x;
    if (tex_desc.filterMode == hipFilterModePoint) {
      return Sample(floorf(x), layer, tex_desc.addressMode);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      return LinearFiltering(x, layer, tex_desc.addressMode);
    } else {
      throw std::invalid_argument("Invalid hipFilterMode value");
    }
  }

  TexelType Tex2DLayered(float x, float y, int layer, const hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * extent_.width : x;
    y = tex_desc.normalizedCoords ? y * extent_.height : y;
    if (tex_desc.filterMode == hipFilterModePoint) {
      return Sample(floorf(x), floorf(y), layer, tex_desc.addressMode);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      return LinearFiltering(x, y, layer, tex_desc.addressMode);
    } else {
      throw std::invalid_argument("Invalid hipFilterMode value");
    }
  }

  TexelType* ptr(size_t layer) const {
    return alloc_ + layer * extent_.width * (extent_.height ?: 1);
  }

  size_t width() const { return extent_.width; }

  size_t height() const { return extent_.height; }

  size_t depth() const { return extent_.depth; }

 private:
  TexelType* const alloc_;
  const hipExtent extent_;
  const size_t layers_;

  template <typename T> TexelType Vec4Sum(T arg) const { return Vec4Add(arg, Zero()); }

  template <typename T, typename... Ts> TexelType Vec4Sum(T arg, Ts... args) const {
    return Vec4Add(arg, Vec4Sum(args...));
  }

  TexelType Zero() const {
    TexelType ret;
    memset(&ret, 0, sizeof(ret));
    return ret;
  }

  float ApplyAddressMode(float coord, size_t dim, hipTextureAddressMode address_mode) const {
    switch (address_mode) {
      case hipAddressModeClamp:
        return ApplyClamp(coord, dim);
      case hipAddressModeBorder:
        if (CheckBorder(coord, dim)) {
          return std::numeric_limits<float>::quiet_NaN();
        }
      case hipAddressModeWrap:
        return ApplyWrap(coord, dim);
      case hipAddressModeMirror:
        return ApplyMirror(coord, dim);
      default:
        throw std::invalid_argument("Invalid hipAddressMode value");
    }
  }

  TexelType Sample(float x, int layer, const hipTextureAddressMode* address_mode) const {
    x = ApplyAddressMode(x, extent_.width, address_mode[0]);

    if (std::isnan(x)) {
      return Zero();
    }

    return ptr(layer)[static_cast<size_t>(x)];
  }

  TexelType Sample(float x, float y, int layer, const hipTextureAddressMode* address_mode) const {
    x = ApplyAddressMode(x, extent_.width, address_mode[0]);
    y = ApplyAddressMode(y, extent_.height, address_mode[1]);

    if (std::isnan(x) || std::isnan(y)) {
      return Zero();
    }

    return ptr(layer)[static_cast<size_t>(y) * extent_.width + static_cast<size_t>(x)];
  }

  TexelType Sample(float x, float y, float z, const hipTextureAddressMode* address_mode) const {
    x = ApplyAddressMode(x, extent_.width, address_mode[0]);
    y = ApplyAddressMode(y, extent_.height, address_mode[1]);
    z = ApplyAddressMode(z, extent_.depth, address_mode[2]);

    if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
      return Zero();
    }

    return ptr(0)[static_cast<size_t>(z) * extent_.width * extent_.height +
                  static_cast<size_t>(y) * extent_.width + static_cast<size_t>(x)];
  }

  TexelType LinearFiltering(float x, int layer, const hipTextureAddressMode* address_mode) const {
    const auto [i, alpha] = GetLinearFilteringParams(x);

    const auto T_i0 = Sample(i, layer, address_mode);
    const auto T_i1 = Sample(i + 1.0f, layer, address_mode);

    const auto term_i0 = Vec4Scale((1.0f - alpha), T_i0);
    const auto term_i1 = Vec4Scale(alpha, T_i1);

    return Vec4Sum(term_i0, term_i1);
  }

  TexelType LinearFiltering(float x, float y, int layer,
                            const hipTextureAddressMode* address_mode) const {
    const auto [i, alpha] = GetLinearFilteringParams(x);
    const auto [j, beta] = GetLinearFilteringParams(y);

    const auto T_i0j0 = Sample(i, j, layer, address_mode);
    const auto T_i1j0 = Sample(i + 1.0f, j, layer, address_mode);
    const auto T_i0j1 = Sample(i, j + 1.0f, layer, address_mode);
    const auto T_i1j1 = Sample(i + 1.0f, j + 1.0f, layer, address_mode);

    const auto term_i0j0 = Vec4Scale((1.0f - alpha) * (1.0f - beta), T_i0j0);
    const auto term_i1j0 = Vec4Scale(alpha * (1.0f - beta), T_i1j0);
    const auto term_i0j1 = Vec4Scale((1.0f - alpha) * beta, T_i0j1);
    const auto term_i1j1 = Vec4Scale(alpha * beta, T_i1j1);

    return Vec4Sum(term_i0j0, term_i1j0, term_i0j1, term_i1j1);
  }

  TexelType LinearFiltering(float x, float y, float z,
                            const hipTextureAddressMode* address_mode) const {
    const auto [i, alpha] = GetLinearFilteringParams(x);
    const auto [j, beta] = GetLinearFilteringParams(y);
    const auto [k, gamma] = GetLinearFilteringParams(z);

    const auto T_i0j0k0 = Sample(i, j, k, address_mode);
    const auto T_i1j0k0 = Sample(i + 1.0f, j, k, address_mode);
    const auto T_i0j1k0 = Sample(i, j + 1.0f, k, address_mode);
    const auto T_i1j1k0 = Sample(i + 1.0f, j + 1.0f, k, address_mode);
    const auto T_i0j0k1 = Sample(i, j, k + 1.0f, address_mode);
    const auto T_i1j0k1 = Sample(i + 1.0f, j, k + 1.0f, address_mode);
    const auto T_i0j1k1 = Sample(i, j + 1.0f, k + 1.0f, address_mode);
    const auto T_i1j1k1 = Sample(i + 1.0f, j + 1.0f, k + 1.0f, address_mode);

    const auto term_i0j0k0 = Vec4Scale((1.0f - alpha) * (1.0f - beta) * (1.0f - gamma), T_i0j0k0);
    const auto term_i1j0k0 = Vec4Scale(alpha * (1.0f - beta) * (1.0f - gamma), T_i1j0k0);
    const auto term_i0j1k0 = Vec4Scale((1.0f - alpha) * beta * (1.0f - gamma), T_i0j1k0);
    const auto term_i1j1k0 = Vec4Scale(alpha * beta * (1.0f - gamma), T_i1j1k0);
    const auto term_i0j0k1 = Vec4Scale((1.0f - alpha) * (1.0f - beta) * gamma, T_i0j0k1);
    const auto term_i1j0k1 = Vec4Scale(alpha * (1.0f - beta) * gamma, T_i1j0k1);
    const auto term_i0j1k1 = Vec4Scale((1.0f - alpha) * beta * gamma, T_i0j1k1);
    const auto term_i1j1k1 = Vec4Scale(alpha * beta * gamma, T_i1j1k1);

    return Vec4Sum(term_i0j0k0, term_i1j0k0, term_i0j1k0, term_i1j1k0, term_i0j0k1, term_i1j0k1,
                   term_i0j1k1, term_i1j1k1);
  }

  float ApplyClamp(float coord, size_t dim) const {
    return max(min(coord, static_cast<float>(dim - 1)), 0.0f);
  }

  bool CheckBorder(float coord, size_t dim) const { return coord > dim - 1 || coord < 0.0f; }

  float ApplyWrap(float coord, size_t dim) const {
    coord /= dim;
    coord = coord - floorf(coord);
    coord *= dim;

    return coord;
  }

  float ApplyMirror(float coord, size_t dim) const {
    coord /= dim;
    const float frac = coord - floor(coord);
    const bool is_reversing = static_cast<ssize_t>(floorf(coord)) % 2;
    coord = is_reversing ? 1.0f - frac : frac;
    coord *= dim;
    coord -= (coord == truncf(coord)) * is_reversing;

    return coord;
  }

  std::tuple<float, float> GetLinearFilteringParams(float coord) const {
    const auto coordB = coord - 0.5f;
    const auto index = floorf(coordB);
    const FixedPoint<8> coeff = coordB - index;

    return {index, coeff};
  }
};
