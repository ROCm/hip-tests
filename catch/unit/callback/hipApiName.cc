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

#include <hip_test_common.hh>

/**
 * @addtogroup CallbackTest Callback Activity APIs
 * @{
 * This section describes tests for the callback/Activity of HIP runtime API.
 */

/**
 * @addtogroup hipApiName hipApiName
 * @{
 * @ingroup CallbackTest
 * `hipApiName(uint32_t id)` -
 * returns the name of API with passed ID
 */

const char* UNKNOWN_API{"unknown"};
const uint32_t API_NUMBER{1024};

/**
 * Test Description
 * ------------------------ 
 *    - Acquires HIP API names and checks that they are valid
 * Test source
 * ------------------------ 
 *    - unit/callback/hipApiName.cc
 * Test requirements
 * ------------------------ 
 *    - HIP_VERSION >= 5.2
 *    - Platform specific (AMD)
 */
TEST_CASE("Unit_hipApiName_Positive_Basic") {
    std::vector<std::string> hipApiNames;

    for(uint32_t i = 0; i < API_NUMBER; ++i) {
        if(strcmp(hipApiName(i), UNKNOWN_API)) {
            hipApiNames.emplace_back(hipApiName(i));
        }
    }

    REQUIRE(!hipApiNames.empty());
}

/**
 * Test Description
 * ------------------------ 
 *    - Checks that upper and lower limit IDs are mapped to unknown APIs:
 *        -# Expected output "unknown" when the `uint32_t` upper limit is passed
 *        -# Expected output "unknown" when the `uint32_t` lower limit is passed
 * Test source
 * ------------------------ 
 *    - unit/callback/hipApiName.cc
 * Test requirements
 * ------------------------ 
 *    - HIP_VERSION >= 5.2
 *    - Platform specific (AMD)
 */
TEST_CASE("Unit_hipApiName_Negative_ReservedIds") {
    REQUIRE_THAT(hipApiName(std::numeric_limits<uint32_t>::min()), Catch::Equals(UNKNOWN_API));
    REQUIRE_THAT(hipApiName(std::numeric_limits<uint32_t>::max()), Catch::Equals(UNKNOWN_API));
}
