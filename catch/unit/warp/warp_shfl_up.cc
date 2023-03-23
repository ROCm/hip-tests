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

#include "warp_common.hh"

#include <bitset>

/**
 * @addtogroup shfl_up shfl_up
 * @{
 * @ingroup DeviceLanguageTest
 * `T __shfl_up(T var, unsigned int lane_delta, int width = warpSize)` -
 * Contains unit test for warp shfl_up function
 */

namespace cg = cooperative_groups;

template <typename T>
__global__ void shfl_up(T* const out, const uint64_t* const active_masks, const unsigned int delta,
                        const int width) {
  if (deactivate_thread(active_masks)) {
    return;
  }

  const auto grid = cg::this_grid();
  T var = static_cast<T>(grid.thread_rank() % warpSize);
  out[grid.thread_rank()] = __shfl_up(var, delta, width);
}

template <typename T> class WarpShflUp : public WarpTest<WarpShflUp<T>, T> {
 public:
  void launch_kernel(T* const arr_dev, const uint64_t* const active_masks) {
    width_ = generate_width(this->warp_size_);
    INFO("Width: " << width_);
    delta_ = GENERATE_COPY(range(0, width_));
    INFO("Delta: " << delta_);
    shfl_up<<<this->grid_.grid_dim_, this->grid_.block_dim_>>>(arr_dev, active_masks, delta_,
                                                               width_);
  }

  void validate(const T* const arr) {
    ArrayAllOf(arr, this->grid_.thread_count_, [this](unsigned int i) -> std::optional<T> {
      const auto rank_in_block = this->grid_.thread_rank_in_block(i).value();
      const auto rank_in_warp = rank_in_block % this->warp_size_;
      const auto mask_idx = this->warps_in_block_ * (i / this->grid_.threads_in_block_count_) +
          rank_in_block / this->warp_size_;
      const std::bitset<sizeof(uint64_t) * 8> active_mask(this->active_masks_[mask_idx]);

      const int target = rank_in_block % width_ - delta_;
      if (!active_mask.test(rank_in_warp) ||
          (target >= 0 && !active_mask.test(rank_in_warp - delta_))) {
        return std::nullopt;
      }

      return (target < 0 ? i : i - delta_) % this->warp_size_;
    });
  };

 private:
  unsigned int delta_;
  int width_;
};

/**
 * Test Description
 * ------------------------
 *  - Validates the warp shuffle up behavior for all valid width sizes {2, 4, 8, 16, 32,
 * 64(if supported)} for delta values of [0, width). The threads are deactivated based on the
 * passed active mask. The test is run for all overloads of shfl_up.
 * Test source
 * ------------------------
 *  - unit/warp/warp_shfl_up.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Device supports warp shuffle
 */
TEMPLATE_TEST_CASE("Unit_Warp_Shfl_Up_Positive_Basic", "", int, unsigned int, long, unsigned long,
                   long long, unsigned long long, float, double) {
  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.arch.hasWarpShuffle) {
    HipTest::HIP_SKIP_TEST("Device doesn't support Warp Shuffle!");
    return;
  }

  WarpShflUp<TestType>().run();
}