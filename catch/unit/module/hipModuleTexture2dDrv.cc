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


#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#include <fstream>
#include <vector>
#include <type_traits>
#include <limits>
#include <atomic>


#define CODEOBJ_FILE "tex2d_kernel.code"
#define NON_EXISTING_TEX_NAME "xyz"
#define EMPTY_TEX_NAME ""
#define GLOBAL_KERNEL_VAR "deviceGlobalFloat"
#define TEX_REF "ftex"
#define WIDTH 256
#define HEIGHT 256
#define MAX_STREAMS 4
#define GRIDDIMX 16
#define GRIDDIMY 16
#define GRIDDIMZ 1
#define BLOCKDIMZ 1

#if HT_NVIDIA

#define CTX_CREATE() \
  hipCtx_t context;\
  initHipCtx(&context);

#define CTX_DESTROY() HIP_CHECK(hipCtxDestroy(context));
#define HIP_TEX_REFERENCE hipTexRef
#define HIP_ARRAY hiparray
#else
#define CTX_CREATE()
#define CTX_DESTROY()
#define HIP_TEX_REFERENCE textureReference*
#define HIP_ARRAY hipArray*
#endif

std::atomic<int> g_thTestPassed(1);

/**
 * Validates negative scenarios for hipModuleGetTexRef
 * texRef = nullptr
 */
bool testTexRefEqNullPtr() {
  bool TestPassed = false;
  hipModule_t Module;
  CTX_CREATE()
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  REQUIRE(hipSuccess != hipModuleGetTexRef(nullptr, Module, "tex"));
  TestPassed = true;
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetTexRef
 * name = nullptr
 */
bool testNameEqNullPtr() {
  bool TestPassed = false;
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  REQUIRE(hipSuccess != hipModuleGetTexRef(&texref, Module, nullptr));
  TestPassed = true;
  CTX_DESTROY()
  return TestPassed;
}
/**
 * Validates negative scenarios for hipModuleGetTexRef
 * name = Non Existing Tex Name
 */
bool testInvalidTexName() {
  bool TestPassed = false;
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  REQUIRE(hipSuccess != hipModuleGetTexRef(&texref, Module,
                                           NON_EXISTING_TEX_NAME));
  TestPassed = true;
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetTexRef
 * name = Empty Tex Name
 */
bool testEmptyTexName() {
  bool TestPassed = false;
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  REQUIRE(hipSuccess != hipModuleGetTexRef(&texref, Module, EMPTY_TEX_NAME));
  TestPassed = true;
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetTexRef
 * name = Global Kernel Variable
 */
bool testWrongTexRef() {
  bool TestPassed = false;
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  REQUIRE(hipSuccess != hipModuleGetTexRef(&texref, Module, GLOBAL_KERNEL_VAR));
  TestPassed = true;
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Validates negative scenarios for hipModuleGetTexRef
 * module = unloaded module
 */
bool testUnloadedMod() {
  bool TestPassed = false;
  hipModule_t Module;
  HIP_TEX_REFERENCE texref;
  CTX_CREATE()
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIP_CHECK(hipModuleUnload(Module));
  REQUIRE(hipSuccess != hipModuleGetTexRef(&texref, Module, TEX_REF));
  TestPassed = true;
  CTX_DESTROY()
  return TestPassed;
}
/**
 * Internal Functions
 *
 */
std::vector<char> load_file() {
  std::ifstream file(CODEOBJ_FILE, std::ios::binary | std::ios::ate);
  std::streamsize fsize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fsize);
  if (!file.read(buffer.data(), fsize)) {
    printf("Info:could not open code object '%s'\n", CODEOBJ_FILE);
  }
  return buffer;
}

template <class T> void fillTestBuffer(unsigned int width,
                                       unsigned int height,
                                       T* hData) {
  if (std::is_same<T, float>::value) {
    for (unsigned int i = 0; i < height; i++) {
      for (unsigned int j = 0; j < width; j++) {
        hData[i * width + j] = i * width + j + 0.5;
      }
    }
  } else if (std::is_same<T, int>::value) {
    for (unsigned int i = 0; i < height; i++) {
      for (unsigned int j = 0; j < width; j++) {
        hData[i * width + j] = i * width + j;
      }
    }
  } else if (std::is_same<T, short>::value) {  // cpplint asking to make int16 instead of short.
    for (unsigned int i = 0; i < height; i++) {
      for (unsigned int j = 0; j < width; j++) {
        hData[i * width + j] = (i * width + j)%
           (std::numeric_limits<short>::max());
      }
    }
  } else if (std::is_same<T, char>::value) {
    for (unsigned int i = 0; i < height; i++) {
      for (unsigned int j = 0; j < width; j++) {
        hData[i * width + j] = (i * width + j)%
           (std::numeric_limits<char>::max());
      }
    }
  }
}

void allocInitArray(unsigned int width,
                     unsigned int height,
                     hipArray_Format format,
                     HIP_ARRAY* array
                     ) {
  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = format;
  desc.NumChannels = 1;
  desc.Width = width;
  desc.Height = height;
  HIP_CHECK(hipArrayCreate(array, &desc));
}

template <class T, class T1> void copyBuffer2Array(unsigned int width,
                                                   unsigned int height,
                                                   T* hData,
                                                   T1 array
                                                   ) {
  hip_Memcpy2D copyParam;
  memset(&copyParam, 0, sizeof(copyParam));
#if HT_NVIDIA
  copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
  copyParam.dstArray = *array;
#else
  copyParam.dstMemoryType = hipMemoryTypeArray;
  copyParam.srcMemoryType = hipMemoryTypeHost;
  copyParam.dstArray = array;
#endif
  copyParam.srcHost = hData;
  copyParam.srcPitch = width * sizeof(T);
  copyParam.WidthInBytes = copyParam.srcPitch;
  copyParam.Height = height;
  HIP_CHECK(hipMemcpyParam2D(&copyParam));
}

template <class T> void assignArray2TexRef(hipArray_Format format,
                                           const char* texRefName,
                                           hipModule_t Module,
                                           T array
                                           ) {
  HIP_TEX_REFERENCE texref;
#if HT_NVIDIA
  HIP_CHECK(hipModuleGetTexRef(&texref, Module, texRefName));
  HIP_CHECK(hipTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_WRAP));
  HIP_CHECK(hipTexRefSetAddressMode(texref, 1, CU_TR_ADDRESS_MODE_WRAP));
  HIP_CHECK(hipTexRefSetFilterMode(texref, HIP_TR_FILTER_MODE_POINT));
  HIP_CHECK(hipTexRefSetFlags(texref, CU_TRSF_READ_AS_INTEGER));
  HIP_CHECK(hipTexRefSetFormat(texref, format, 1));
  HIP_CHECK(hipTexRefSetArray(texref, *array, CU_TRSA_OVERRIDE_FORMAT));
#else
  HIP_CHECK(hipModuleGetTexRef(&texref, Module, texRefName));
  HIP_CHECK(hipTexRefSetAddressMode(texref, 0, hipAddressModeWrap));
  HIP_CHECK(hipTexRefSetAddressMode(texref, 1, hipAddressModeWrap));
  HIP_CHECK(hipTexRefSetFilterMode(texref, hipFilterModePoint));
  HIP_CHECK(hipTexRefSetFlags(texref, HIP_TRSF_READ_AS_INTEGER));
  HIP_CHECK(hipTexRefSetFormat(texref, format, 1));
  HIP_CHECK(hipTexRefSetArray(texref, array, HIP_TRSA_OVERRIDE_FORMAT));
#endif
}

template <class T> bool validateOutput(unsigned int width,
                                       unsigned int height,
                                       T* hData,
                                       T* hOutputData) {
  for (unsigned int i = 0; i < height; i++) {
    for (unsigned int j = 0; j < width; j++) {
      REQUIRE(hData[i * width + j] == hOutputData[i * width + j]);
    }
  }
  return true;
}
/**
 * Validates texture type data functionality for hipModuleGetTexRef
 *
 */
template <class T> bool testTexType(hipArray_Format format,
                                    const char* texRefName,
                                    const char* kerFuncName) {
  bool TestPassed = true;
  unsigned int width = WIDTH;
  unsigned int height = HEIGHT;
  unsigned int size = width * height * sizeof(T);
  T* hData = reinterpret_cast<T*>(malloc(size));
  if (NULL == hData) {
    INFO("Info:Failed to allocate using malloc in testTexType.\n");
    return false;
  }
  CTX_CREATE()
  fillTestBuffer<T>(width, height, hData);
  // Load Kernel File and create hipArray
  hipModule_t Module;
  HIP_CHECK(hipModuleLoad(&Module, CODEOBJ_FILE));
  HIP_ARRAY array;
  allocInitArray(width, height, format, &array);
#if HT_NVIDIA
  // Copy from hData to array using hipMemcpyParam2D
  copyBuffer2Array<T, HIP_ARRAY*>(width, height, hData, &array);
  // Get tex reference from the loaded kernel file
  // Assign array to the tex reference
  assignArray2TexRef<HIP_ARRAY*>(format, texRefName, Module, &array);
#else
  // Copy from hData to array using hipMemcpyParam2D
  copyBuffer2Array<T, HIP_ARRAY>(width, height, hData, array);
  // Get tex reference from the loaded kernel file
  // Assign array to the tex reference
  assignArray2TexRef<HIP_ARRAY>(format, texRefName, Module, array);
#endif
  hipFunction_t Function;
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kerFuncName));

  T* dData = NULL;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dData), size));

  struct {
    void* _Ad;
    unsigned int _Bd;
    unsigned int _Cd;
  } args;
  args._Ad = reinterpret_cast<void*>(dData);
  args._Bd = width;
  args._Cd = height;

  size_t sizeTemp = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                    &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &sizeTemp,
                    HIP_LAUNCH_PARAM_END};

  int temp1 = width / GRIDDIMX;
  int temp2 = height / GRIDDIMY;
  HIP_CHECK(
    hipModuleLaunchKernel(Function, GRIDDIMX, GRIDDIMY, GRIDDIMZ,
                          temp1, temp2, BLOCKDIMZ, 0, 0,
                          NULL, reinterpret_cast<void**>(&config)));
  HIP_CHECK(hipDeviceSynchronize());
  T* hOutputData = reinterpret_cast<T*>(malloc(size));
  if (NULL == hOutputData) {
    printf("Failed to allocate using malloc in testTexType.\n");
    TestPassed = false;
  } else {
    memset(hOutputData, 0, size);
    HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));
    TestPassed = validateOutput<T>(width, height, hData, hOutputData);
  }
  free(hOutputData);
  HIP_CHECK(hipFree(dData));
  ARRAY_DESTROY(array)
  HIP_CHECK(hipModuleUnload(Module));
  free(hData);
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Validates texture functionality with multiple streams for hipModuleGetTexRef
 *
 */
template <class T> bool testTexMultStream(const std::vector<char>& buffer,
                                        hipArray_Format format,
                                        const char* texRefName,
                                        const char* kerFuncName,
                                        unsigned int numOfStreams) {
  bool TestPassed = true;
  unsigned int width = WIDTH;
  unsigned int height = HEIGHT;
  unsigned int size = width * height * sizeof(T);
  T* hData = reinterpret_cast<T*>(malloc(size));
  if (NULL == hData) {
    printf("Failed to allocate using malloc in testTexMultStream.\n");
    return false;
  }
  CTX_CREATE()
  fillTestBuffer<T>(width, height, hData);

  // Load Kernel File and create hipArray
  hipModule_t Module;
  HIP_CHECK(hipModuleLoadData(&Module, &buffer[0]));
  HIP_ARRAY array;
  allocInitArray(width, height, format, &array);
#if HT_NVIDIA
  // Copy from hData to array using hipMemcpyParam2D
  copyBuffer2Array<T, HIP_ARRAY*>(width, height, hData, &array);
  // Get tex reference from the loaded kernel file
  // Assign array to the tex reference
  assignArray2TexRef<HIP_ARRAY*>(format, texRefName, Module, &array);
#else
  // Copy from hData to array using hipMemcpyParam2D
  copyBuffer2Array<T, HIP_ARRAY>(width, height, hData, array);
  // Get tex reference from the loaded kernel file
  // Assign array to the tex reference
  assignArray2TexRef<HIP_ARRAY>(format, texRefName, Module, array);
#endif
  hipFunction_t Function;
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kerFuncName));

  // Create Multiple Strings
  hipStream_t streams[MAX_STREAMS]={0};
  T* dData[MAX_STREAMS] = {NULL};
  T* hOutputData[MAX_STREAMS] = {NULL};
  if (numOfStreams > MAX_STREAMS) {
    numOfStreams = MAX_STREAMS;
  }
  unsigned int totalStreamsCreated = 0;
  for (unsigned int stream_num = 0; stream_num < numOfStreams; stream_num++) {
    hOutputData[stream_num] = reinterpret_cast<T*>(malloc(size));
    REQUIRE(NULL != hOutputData[stream_num]);
    HIP_CHECK(hipStreamCreate(&streams[stream_num]));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dData[stream_num]), size));
    memset(hOutputData[stream_num], 0, size);
    struct {
      void* _Ad;
      unsigned int _Bd;
      unsigned int _Cd;
    } args;
    args._Ad = reinterpret_cast<void*>(dData[stream_num]);
    args._Bd = width;
    args._Cd = height;

    size_t sizeTemp = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &sizeTemp,
                      HIP_LAUNCH_PARAM_END};

    int temp1 = width / GRIDDIMX;
    int temp2 = height / GRIDDIMY;
    HIP_CHECK(
      hipModuleLaunchKernel(Function, GRIDDIMX, GRIDDIMY, GRIDDIMZ,
                          temp1, temp2, BLOCKDIMZ, 0, streams[stream_num],
                          NULL, reinterpret_cast<void**>(&config)));
    totalStreamsCreated++;
  }
  // Check the kernel results separately
  for (unsigned int stream_num = 0; stream_num < totalStreamsCreated;
       stream_num++) {
    HIP_CHECK(hipStreamSynchronize(streams[stream_num]));
    HIP_CHECK(hipMemcpy(hOutputData[stream_num], dData[stream_num], size,
              hipMemcpyDeviceToHost));
    TestPassed &= validateOutput<T>(width, height, hData,
                                    hOutputData[stream_num]);
  }
  for (unsigned int i = 0; i < totalStreamsCreated; i++) {
    HIP_CHECK(hipFree(dData[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
    free(hOutputData[i]);
  }
  ARRAY_DESTROY(array)
  HIP_CHECK(hipModuleUnload(Module));
  free(hData);
  CTX_DESTROY()
  return TestPassed;
}

/**
 * Internal Thread Functions
 *
 */
void launchSingleStreamMultGPU(int gpu, const std::vector<char>& buffer) {
  bool TestPassed = true;
  HIP_CHECK(hipSetDevice(gpu));
  TestPassed = testTexMultStream<float>(buffer,
                                        HIP_AD_FORMAT_FLOAT,
                                        "ftex",
                                        "tex2dKernelFloat", 1);
  g_thTestPassed &= static_cast<int>(TestPassed);
}

void launchMultStreamMultGPU(int gpu, const std::vector<char>& buffer) {
  bool TestPassed = true;
  HIP_CHECK(hipSetDevice(gpu));
  TestPassed = testTexMultStream<float>(buffer,
                                        HIP_AD_FORMAT_FLOAT,
                                        "ftex",
                                        "tex2dKernelFloat", 3);
  g_thTestPassed &= static_cast<int>(TestPassed);
}
/**
 * Validates texture functionality with Multiple Streams on multuple GPU
 * for hipModuleGetTexRef
 *
 */
bool testTexMultStreamMultGPU(int numOfGPUs,
                              const std::vector<char>& buffer) {
  bool TestPassed = true;
  std::thread T[numOfGPUs];

  for (int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu] = std::thread(launchMultStreamMultGPU, gpu, buffer);
  }
  for (int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu].join();
  }

  REQUIRE(TestPassed == g_thTestPassed);
  return TestPassed;
}
/**
 * Validates texture functionality with Single Stream on multuple GPU
 * for hipModuleGetTexRef
 *
 */
bool testTexSingleStreamMultGPU(int numOfGPUs,
                                const std::vector<char>& buffer) {
  bool TestPassed = true;
  std::thread T[numOfGPUs];

  for (int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu] = std::thread(launchSingleStreamMultGPU, gpu, buffer);
  }
  for (int gpu = 0; gpu < numOfGPUs; gpu++) {
    T[gpu].join();
  }

  REQUIRE(TestPassed == g_thTestPassed);
  return TestPassed;
}

TEST_CASE("Unit_hipModuleTexture2dDrv") {
  bool TestPassed = true;
  SECTION("testTexType is float") {
    REQUIRE(TestPassed == testTexType<float>(HIP_AD_FORMAT_FLOAT,
                          "ftex", "tex2dKernelFloat"));
  }
  SECTION("testTexType is int") {
    REQUIRE(TestPassed == testTexType<int>(HIP_AD_FORMAT_SIGNED_INT32,
                                  "itex", "tex2dKernelInt"));
  }
  SECTION("testTexType is short") {
    REQUIRE(TestPassed == testTexType<short>(HIP_AD_FORMAT_SIGNED_INT16,
                                    "stex", "tex2dKernelInt16"));
  }
  SECTION("testTexType is char")  {
    REQUIRE(TestPassed == testTexType<char>(HIP_AD_FORMAT_SIGNED_INT8,
                                   "ctex", "tex2dKernelInt8"));
  }
  SECTION("testTexMultStream is float") {
    auto buffer = load_file();
    REQUIRE(TestPassed ==testTexMultStream<float>(buffer,
                HIP_AD_FORMAT_FLOAT, "ftex", "tex2dKernelFloat", MAX_STREAMS));
  }
  #if HT_AMD
  SECTION("testTexSingleStreamMultGPU") {
    int gpu_cnt = 0;
    auto buffer = load_file();
    HIP_CHECK(hipGetDeviceCount(&gpu_cnt));
    REQUIRE(TestPassed == testTexSingleStreamMultGPU(gpu_cnt, buffer));
  }
  SECTION("testTexMultStreamMultGPU") {
    int gpu_cnt = 0;
    auto buffer = load_file();
    HIP_CHECK(hipGetDeviceCount(&gpu_cnt));
    REQUIRE(TestPassed == testTexMultStreamMultGPU(gpu_cnt, buffer));
  }
  #endif
  SECTION("testTexRefEqNullPtr") {
    REQUIRE(TestPassed == testTexRefEqNullPtr());
  }
  SECTION("testNameEqNullPtr") {
    REQUIRE(TestPassed == testNameEqNullPtr());
  }
  SECTION("testInvalidTexName") {
    REQUIRE(TestPassed == testInvalidTexName());
  }
  SECTION("testEmptyTexName") {
    REQUIRE(TestPassed == testEmptyTexName());
  }
  SECTION("testWrongTexRef") {
    REQUIRE(TestPassed == testWrongTexRef());
  }
  SECTION("testUnloadedMod") {
    REQUIRE(TestPassed == testUnloadedMod());
  }
}
