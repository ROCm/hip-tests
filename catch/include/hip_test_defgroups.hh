/*
Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

// Test groups are named based on the group names from hip_api_runtime.h, with adding "Test" suffix

/**
 * @defgroup CallbackTest Callback Activity APIs
 * @{
 * This section describes tests for the callback/Activity of HIP runtime API.
 * @}
 */

/**
 * @defgroup GraphTest Graph Management
 * @{
 * This section describes the graph management types & functions of HIP runtime API.
 * @}
 */

/**
 * @defgroup VectorTypeTest Vector types
 * @{
 * This section describes tests for the Vector type functions and operators.
 */

/**
 * @addtogroup make_vector make_vector
 * @{
 * @ingroup VectorTypeTest
 */

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Negate (-) operation applied on the unsigned vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_NegateUnsigned_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Bitwise operations applied on the float vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_BitwiseFloat_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Bitwise operations applied on the double vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_BitwiseDouble_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 1D signed vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssign1D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 2D signed vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssign2D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 3D signed vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssign3D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 4D signed vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssign4D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 1D unsigned vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssignUnsigned1D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 2D unsigned vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssignUnsigned2D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 3D unsigned vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssignUnsigned3D_Negative_Parameters") {}

/**
 * Test Description
 * ------------------------
 *    - Compiles kernels and host functions
 *    - Calculate-assign operations applied on the 4D unsigned vectors
 * Test source
 * ------------------------
 *    - unit/vector_types/CMakeLists.txt
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_CalculateAssignUnsigned4D_Negative_Parameters") {}

/**
 * End doxygen group make_vector.
 * @}
 */

/**
 * End doxygen group VectorTypeTest.
 * @}
 */
