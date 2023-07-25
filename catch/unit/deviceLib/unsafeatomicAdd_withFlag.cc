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

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_features.hh>

template <typename Type>
static __global__ void unsafeAtomicAdd_Kernel(Type* ptr, Type* old_res, Type inc_val) {
  *old_res = unsafeAtomicAdd(ptr, inc_val);
}


TEMPLATE_TEST_CASE("Unit_unsafeAtomicAdd_Sanity", "", float, double) {
  hipDeviceProp_t prop;
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);
  if (CheckIfFeatSupported(CTFeatures::CT_FEATURE_FINEGRAIN_HWSUPPORT, gfxName)) {
    if (prop.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      constexpr TestType init_val = 5;
      TestType *h_ptr{nullptr}, *h_result{nullptr};
      TestType *d_ptr{nullptr}, *d_result{nullptr};
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_ptr), sizeof(TestType),
                              hipHostMallocNonCoherent));
      *h_ptr = init_val;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_result), sizeof(TestType),
                              hipHostMallocNonCoherent));
      *h_result = init_val;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&d_ptr), h_ptr, 0));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&d_result), h_result, 0));
      constexpr TestType inc_val = 10;
      hipLaunchKernelGGL(unsafeAtomicAdd_Kernel<TestType>, dim3(1), dim3(1), 0, 0, d_ptr, d_result,
                         inc_val);

      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());

      REQUIRE(*h_ptr == (init_val + inc_val));
      REQUIRE(*h_result == init_val);
      HIP_CHECK(hipHostFree(h_ptr));
      HIP_CHECK(hipHostFree(h_result));
    }
  }
}
