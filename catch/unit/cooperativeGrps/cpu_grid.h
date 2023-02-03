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

#include <optional>

#include <hip/hip_runtime_api.h>

struct CPUGrid {
  CPUGrid(const dim3 grid_dim, const dim3 block_dim)
      : grid_dim_{grid_dim},
        block_dim_{block_dim},
        block_count_{grid_dim.x * grid_dim.y * grid_dim.z},
        threads_in_block_count_{block_dim.x * block_dim.y * block_dim.z},
        thread_count_{block_count_ * threads_in_block_count_} {}

  inline std::optional<unsigned int> thread_rank_in_block(
      const unsigned int thread_rank_in_grid) const {
    if (thread_rank_in_grid > thread_count_) {
      return std::nullopt;
    }

    return thread_rank_in_grid % threads_in_block_count_;
  }

  inline std::optional<dim3> block_idx(const unsigned int thread_rank_in_grid) const {
    if (thread_rank_in_grid > thread_count_) {
      return std::nullopt;
    }

    dim3 block_idx;
    const auto block_rank_in_grid = thread_rank_in_grid / threads_in_block_count_;
    block_idx.x = block_rank_in_grid % grid_dim_.x;
    block_idx.y = (block_rank_in_grid / grid_dim_.x) % grid_dim_.y;
    block_idx.z = block_rank_in_grid / (grid_dim_.x * grid_dim_.y);

    return block_idx;
  }

  inline std::optional<dim3> thread_idx(const unsigned int thread_rank_in_grid) const {
    if (thread_rank_in_grid > thread_count_) {
      return std::nullopt;
    }

    dim3 thread_idx;
    const auto thread_rank_in_block = thread_rank_in_grid % threads_in_block_count_;
    thread_idx.x = thread_rank_in_block % block_dim_.x;
    thread_idx.y = (thread_rank_in_block / block_dim_.x) % block_dim_.y;
    thread_idx.z = thread_rank_in_block / (block_dim_.x * block_dim_.y);

    return thread_idx;
  }

  const dim3 grid_dim_;
  const dim3 block_dim_;
  const unsigned int block_count_;
  const unsigned int threads_in_block_count_;
  const unsigned int thread_count_;
};

struct CPUMultiGrid {
  CPUMultiGrid(const unsigned int num_grids, const dim3 grid_dims[], const dim3 block_dims[]) {
    thread_count_ = 0;
    grid_count_ = num_grids;
    grids_.reserve(grid_count_);
    for (int i = 0; i < grid_count_; i++) {
      grids_.emplace_back(grid_dims[i], block_dims[i]);
      thread_count_ += grids_[i].thread_count_;
    }
  }

  inline unsigned int thread0_rank_in_multi_grid(const unsigned int grid_rank) const {
    unsigned int multi_grid_thread_rank_0 = 0;
    unsigned int multi_grid_thread_count = 0;
    for (int i = 0; i <= grid_rank; i++) {
      multi_grid_thread_rank_0 = multi_grid_thread_count;
      multi_grid_thread_count += grids_[i].thread_count_;
    }
    return multi_grid_thread_rank_0;
  }

  std::vector<CPUGrid> grids_;
  unsigned int grid_count_;
  unsigned int thread_count_;
};