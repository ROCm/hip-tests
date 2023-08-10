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
#include <hip_array_common.hh>
#include <hip_texture_helper.hh>

template <typename T>
__global__ void
surf1DKernelR(hipSurfaceObject_t surfaceObject,
             T* outputData, int width)
{
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < width) {
    surf1Dread(outputData + x, surfaceObject, x * sizeof(T));
  }
#endif
}

template <typename T>
__global__ void
surf1DKernelW(hipSurfaceObject_t surfaceObject,
             T* inputData, int width)
{
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < width) {
    surf1Dwrite(inputData[x], surfaceObject, x * sizeof(T));
  }
#endif
}

template <typename T>
__global__ void
surf1DKernelRW(hipSurfaceObject_t surfaceObject,
             hipSurfaceObject_t outputSurfObj, int width)
{
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < width) {
    T data;
    surf1Dread(&data, surfaceObject, x * sizeof(T));
    surf1Dwrite(data, outputSurfObj, x * sizeof(T));
  }
#endif
}

template <typename T>
static void runTestR(const int width)
{
  unsigned int size = width * sizeof(T);
  T *hData = (T*) malloc (size);
  memset(hData, 0, size);
  for (int j = 0; j < width; j++)
  {
    initVal(hData[j]);
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();

  hipArray *hipArray = nullptr;
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width, 0, hipArraySurfaceLoadStore));

  HIP_CHECK(hipMemcpyToArray(hipArray, 0, 0, hData, size, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Create surface object
  hipSurfaceObject_t surfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surfaceObject, &resDesc));

  T *hOutputData = nullptr;
  HIP_CHECK(hipHostMalloc((void**)&hOutputData, size));
  memset(hOutputData, 0, size);

  dim3 dimBlock (16, 1, 1);
  dim3 dimGrid ((width + dimBlock.x - 1) / dimBlock.x, 1, 1);

  surf1DKernelR<T><<<dimGrid, dimBlock>>>(surfaceObject, hOutputData, width);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  for (int j = 0; j < width; j++) {
    if (!isEqual(hData[j], hOutputData[j])) {
      printf("Difference [ %d ]:%s ----%s\n", j,
             getString(hData[j]).c_str(), getString(hOutputData[j]).c_str());
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipDestroySurfaceObject(surfaceObject));
  HIP_CHECK(hipFreeArray(hipArray));
  free(hData);
  HIP_CHECK(hipHostFree(hOutputData));
  REQUIRE(true);
}

template <typename T>
static void runTestW(const int width)
{
  unsigned int size = width * sizeof(T);
  T *hData = nullptr;
  HIP_CHECK(hipHostMalloc((void**)&hData, size));
  memset(hData, 0, size);

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();

  hipArray *hipArray = nullptr;
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width, 0, hipArraySurfaceLoadStore));

  HIP_CHECK(hipMemcpyToArray(hipArray, 0, 0, hData, size, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Create surface object
  hipSurfaceObject_t surfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surfaceObject, &resDesc));

  for (int j = 0; j < width; j++)
  {
    initVal(hData[j]);
  }

  dim3 dimBlock (16, 1, 1);
  dim3 dimGrid ((width + dimBlock.x - 1) / dimBlock.x, 1, 1);

  surf1DKernelW<T><<<dimGrid, dimBlock>>>(surfaceObject, hData, width);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  T *hOutputData = (T*) malloc (size);
  memset(hOutputData, 0, size);
  HIP_CHECK(hipMemcpyFromArray(hOutputData, hipArray, 0, 0, size, hipMemcpyDeviceToHost));

  for (int j = 0; j < width; j++) {
    if (!isEqual(hData[j], hOutputData[j])) {
      printf("Difference [ %d ]:%s ----%s\n", j,
             getString(hData[j]).c_str(), getString(hOutputData[j]).c_str());
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipDestroySurfaceObject(surfaceObject));
  HIP_CHECK(hipFreeArray(hipArray));
  HIP_CHECK(hipHostFree(hData));
  free(hOutputData);
  REQUIRE(true);
}


template <typename T>
static void runTestRW(const int width)
{
  unsigned int size = width * sizeof(T);
  T *hData = (T*) malloc (size);
  memset(hData, 0, size);
  for (int j = 0; j < width; j++)
  {
    initVal(hData[j]);
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();

  hipArray *hipArray = nullptr, *hipOutArray = nullptr;
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width, 0, hipArraySurfaceLoadStore));

  HIP_CHECK(hipMemcpyToArray(hipArray, 0, 0, hData, size, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Create surface object
  hipSurfaceObject_t surfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surfaceObject, &resDesc));

  HIP_CHECK(hipMallocArray(&hipOutArray, &channelDesc, width, 0, hipArraySurfaceLoadStore));

  hipResourceDesc resOutDesc;
  memset(&resOutDesc, 0, sizeof(resOutDesc));
  resOutDesc.resType = hipResourceTypeArray;
  resOutDesc.res.array.array = hipOutArray;

  hipSurfaceObject_t outSurfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject (&outSurfaceObject, &resOutDesc));

  dim3 dimBlock (16, 1, 1);
  dim3 dimGrid ((width + dimBlock.x - 1) / dimBlock.x, 1, 1);

  surf1DKernelRW<T><<<dimGrid, dimBlock>>>(surfaceObject, outSurfaceObject, width);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  T *hOutputData = (T*) malloc (size);
  memset(hOutputData, 0, size);
  HIP_CHECK(hipMemcpyFromArray(hOutputData, hipOutArray, 0, 0, size, hipMemcpyDeviceToHost));

  for (int j = 0; j < width; j++) {
    if (!isEqual(hData[j], hOutputData[j])) {
      printf("Difference [ %d ]:%s ----%s\n", j,
             getString(hData[j]).c_str(), getString(hOutputData[j]).c_str());
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipDestroySurfaceObject(surfaceObject));
  HIP_CHECK(hipDestroySurfaceObject(outSurfaceObject));
  HIP_CHECK(hipFreeArray(hipArray));
  HIP_CHECK(hipFreeArray(hipOutArray));
  free(hData);
  free(hOutputData);
  REQUIRE(true);
}

TEMPLATE_TEST_CASE("Unit_hipSurfaceObj1D_type_R", "",
                   char,  uchar,  short,  ushort,  int,  uint, float,
                   char1, uchar1, short1, ushort1, int1, uint1, float1,
                   char2, uchar2, short2, ushort2, int2, uint2, float2,
                   char4, uchar4, short4, ushort4, int4, uint4, float4)
{
  CHECK_IMAGE_SUPPORT
  auto err = hipGetLastError(); // reset last err due to previous negative tests

  SECTION("Unit_hipSurfaceObj1D_type_R - 31") {
    runTestR<TestType>(31);
  }

  SECTION("Unit_hipSurfaceObj1D_type_R - 67") {
    runTestR<TestType>(67);
  }

  SECTION("Unit_hipSurfaceObj1D_type_R - 131") {
    runTestR<TestType>(131);
  }

  SECTION("Unit_hipSurfaceObj1D_type_R - 263") {
    runTestR<TestType>(263);
  }
}

TEMPLATE_TEST_CASE("Unit_hipSurfaceObj1D_type_W", "",
                   char,  uchar,  short,  ushort,  int,  uint, float,
                   char1, uchar1, short1, ushort1, int1, uint1, float1,
                   char2, uchar2, short2, ushort2, int2, uint2, float2,
                   char4, uchar4, short4, ushort4, int4, uint4, float4)
{
  CHECK_IMAGE_SUPPORT
  auto err = hipGetLastError(); // reset last err due to previous negative tests

  SECTION("Unit_hipSurfaceObj1D_type_W - 31") {
    runTestW<TestType>(31);
  }

  SECTION("Unit_hipSurfaceObj1D_type_W - 63") {
    runTestW<TestType>(63);
  }

  SECTION("Unit_hipSurfaceObj1D_type_W - 131") {
    runTestW<TestType>(131);
  }

  SECTION("Unit_hipSurfaceObj1D_type_W - 263") {
    runTestW<TestType>(263);
  }
}

TEMPLATE_TEST_CASE("Unit_hipSurfaceObj1D_type_RW", "",
                   char,  uchar,  short,  ushort,  int,  uint, float,
                   char1, uchar1, short1, ushort1, int1, uint1, float1,
                   char2, uchar2, short2, ushort2, int2, uint2, float2,
                   char4, uchar4, short4, ushort4, int4, uint4, float4)
{
  CHECK_IMAGE_SUPPORT
  auto err = hipGetLastError(); // reset last err due to previous negative tests

  SECTION("Unit_hipSurfaceObj1D_type_RW - 23") {
    runTestRW<TestType>(23);
  }

  SECTION("Unit_hipSurfaceObj1D_type_RW - 67") {
    runTestRW<TestType>(67);
  }

  SECTION("Unit_hipSurfaceObj1D_type_RW - 131") {
    runTestRW<TestType>(131);
  }

  SECTION("Unit_hipSurfaceObj1D_type_RW - 263") {
    runTestRW<TestType>(263);
  }
}
