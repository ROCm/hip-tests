/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip_module_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

/**
 * @addtogroup hipModuleGetTexRef hipModuleGetTexRef
 * @{
 * @ingroup ModuleTest
 * `hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name)` -
 * Returns the handle of the texture reference with the name from the module.
 */

static hipModule_t GetModule() {
  HIP_CHECK(hipFree(nullptr));
  static const auto mg = ModuleGuard::LoadModule("get_tex_ref_module.code");
  return mg.module();
}

/**
 * Test Description
 * ------------------------
 *  - Get the texture reference handle from the loaded module.
 *  - Validates that texture handle is not `nullptr`.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleGetTexRef.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleGetTexRef_Positive_Basic") {
  hipTexRef tex_ref = nullptr;
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, GetModule(), "tex"));
  REQUIRE(tex_ref != nullptr);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the texture reference is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When module is not loaded and is `nullptr`
 *      - Expected output: return `hipErrorInvalidResourceHandle`
 *    -# When texture reference name is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When texture reference name is an empty string
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When texture reference name is the name of non-existing texture reference
 *      - Expected output: return `hipErrorNotFound`
 * Test source
 * ------------------------
 *  - unit/module/hipModuleGetTexRef.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipModuleGetTexRef_Negative_Parameters") {
  hipModule_t module = GetModule();
  hipTexRef tex_ref = nullptr;

  SECTION("texRef == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetTexRef(nullptr, module, "tex"), hipErrorInvalidValue);
  }

  SECTION("name == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetTexRef(&tex_ref, module, nullptr), hipErrorInvalidValue);
  }

  SECTION("name == non existent texture") {
    HIP_CHECK_ERROR(hipModuleGetTexRef(&tex_ref, module, "non_existent_texture"), hipErrorNotFound);
  }
}

TEST_CASE("Unit_hipModuleGetTexRef_Negative_Hmod_Is_Nullptr") {
  hipModule_t module = GetModule();
  hipTexRef tex_ref = nullptr;

  HIP_CHECK_ERROR(hipModuleGetTexRef(&tex_ref, nullptr, "tex"), hipErrorInvalidResourceHandle);
}

TEST_CASE("Unit_hipModuleGetTexRef_Negative_Name_Is_Empty_String") {
  hipModule_t module = GetModule();
  hipTexRef tex_ref = nullptr;

  HIP_CHECK_ERROR(hipModuleGetTexRef(&tex_ref, module, ""), hipErrorInvalidValue);
}
