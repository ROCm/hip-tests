/*
Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <string.h>
#ifdef __linux__
#include <unistd.h>
#endif
#include <cstring>
#include <cstdio>


/**
 * @addtogroup hipDeviceGetUuid hipDeviceGetUuid
 * @{
 * @ingroup DriverTest
 * `hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device)` -
 * Returns an UUID for the device.[BETA]
 */

/**
 * Test Description
 * ------------------------
 *  - Check that non-empty UUID is returned for each available device.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetUuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetUuid_Positive") {
  hipDevice_t device;
  hipUUID uuid{0};
  bool uuidValid = false;

  const int deviceId = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipDeviceGet(&device, deviceId));

  // Scenario 1
  HIP_CHECK(hipDeviceGetUuid(&uuid, device));
  // Atleast one non zero value
  size_t uuidSize = sizeof(uuid.bytes) / sizeof(uuid.bytes[0]);
  for (size_t i = 0; i < uuidSize; i++) {
    if (uuid.bytes[i] != 0) {
      uuidValid = true;
      break;
    }
  }
  REQUIRE(uuidValid == true);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the UUID is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When device ordinal is negative
 *      - Expected output: do not return `hipSuccess`
 *    -# When device ordinal is out of bounds
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetUuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetUuid_Negative") {
  int numDevices = 0;
  hipDevice_t device;
  hipUUID uuid;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices > 0) {
    HIP_CHECK(hipDeviceGet(&device, 0));
    REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(nullptr, device));
    REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(&uuid, -1));
    REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(&uuid, numDevices));
  }
}
#ifdef __linux__
#if HT_AMD
#define COMMAND_LEN 256
#define BUFFER_LEN 512

/**
 * Test Description
 * ------------------------
 *  - Get the UUID from rocminfo and compare it with the value from hipDeviceGetUuid API
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetUuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */

TEST_CASE("Unit_hipDeviceGetUuid_From_RocmInfo") {
  int deviceCount = 0;
  hipDevice_t device;
  hipUUID uuid{0};
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  assert(deviceCount > 0);

  FILE *fpipe;
  char command[COMMAND_LEN] = "";
  const char *rocmInfo = "rocminfo";

  snprintf(command, COMMAND_LEN, "%s", rocmInfo);
  strncat(command, " | grep -i Uuid:", COMMAND_LEN);
  // Execute the rocminfo command and extract the UUID info
  fpipe = popen(command, "r");
  if (fpipe == nullptr) {
    printf("Unable to create command file\n");
    return;
  }
  char command_op[BUFFER_LEN];
  int j = 0;
  std::vector<std::string> output(deviceCount);  //NOLINT
  while (fgets(command_op, BUFFER_LEN, fpipe)) {
    std::string rocminfo_line(command_op);
    if ((std::string::npos != rocminfo_line.find("CPU-"))) {
      continue;
    } else if ((std::string::npos != rocminfo_line.find("GPU-"))) {
        output[j] =  rocminfo_line.substr(31, 16);
    }
    j++;
  }
  for (int dev = 0; dev < deviceCount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipDeviceGet(&device, dev));
    HIP_CHECK(hipDeviceGetUuid(&uuid, device));
    REQUIRE(output[dev] == uuid.bytes);
  }
}
#endif
#endif
/**
 * Test Description
 * ------------------------
 *  - Get the UUID from hipGetDeviceProperties and compare it with value from hipDeviceGetUuid API
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetUuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
// Guarding it against NVIDIA as this test is faling on it.
#if HT_AMD
TEST_CASE("Unit_hipDeviceGetUuid_VerifyUuidFrm_hipGetDeviceProperties") {
  int deviceCount = 0;
  hipDevice_t device;
  hipDeviceProp_t prop;
  hipUUID uuid{0};
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  assert(deviceCount > 0);
  for (int dev = 0; dev < deviceCount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipDeviceGet(&device, dev));
    HIP_CHECK(hipDeviceGetUuid(&uuid, device));
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    REQUIRE(strcmp(prop.uuid.bytes, uuid.bytes) == 0);
  }
}
#endif
