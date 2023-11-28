#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>


#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>

#pragma clang diagnostic ignored "-Wuninitialized"

static constexpr auto src{
    R"(
extern "C"
__global__
void saxpy(float a, float* x, float* y, float* out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
           out[tid] = a * x[tid] + y[tid];
    }
}
)"};

TEST_CASE("Unit_RTC_LinkerAPI_Negative") {
  SECTION("get bitcode - nullptr size and code") {
    hiprtcProgram program;
    REQUIRE(hiprtcGetBitcodeSize(program, nullptr) == HIPRTC_ERROR_INVALID_INPUT);
    REQUIRE(hiprtcGetBitcode(program, nullptr) == HIPRTC_ERROR_INVALID_INPUT);
  }

  SECTION("link create - nullptr image and input type") {
    hiprtcLinkState linkstate;
    REQUIRE(hiprtcLinkAddData(linkstate, HIPRTC_JIT_INPUT_LLVM_BITCODE, nullptr, 0, "code", 0,
                              nullptr, nullptr) == HIPRTC_ERROR_INVALID_INPUT);
    REQUIRE(hiprtcLinkAddData(linkstate, HIPRTC_JIT_INPUT_CUBIN, &linkstate, 1, "code", 0, nullptr,
                              nullptr) == HIPRTC_ERROR_INVALID_INPUT);
  }

  SECTION("link complete - ") {
    hiprtcLinkState linkstate;
    REQUIRE(hiprtcLinkComplete(linkstate, nullptr, nullptr) == HIPRTC_ERROR_INVALID_INPUT);
  }
}

TEST_CASE("Unit_RTC_LinkerAPI") {
  hiprtcProgram program;
  HIPRTC_CHECK(hiprtcCreateProgram(&program, src, "saxpy", 0, nullptr, nullptr));

  const char* options[]{"-fgpu-rdc"};
  HIPRTC_CHECK(hiprtcCompileProgram(program, 1, options));

  size_t codesize = 0;
  HIPRTC_CHECK(hiprtcGetBitcodeSize(program, &codesize));

  std::vector<char> code(codesize, '\0');
  HIPRTC_CHECK(hiprtcGetBitcode(program, &code[0]));

  const char* isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm", "-inlinehint-threshold=1"};
  std::vector<hiprtcJIT_option> jit_options = {HIPRTC_JIT_IR_TO_ISA_OPT_EXT,
                                               HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
  size_t isaoptssize = 4;
  const void* lopts[] = {(void*)isaopts, (void*)(isaoptssize)};
  hiprtcLinkState linkstate;
  HIPRTC_CHECK(hiprtcLinkCreate(jit_options.size(), jit_options.data(), (void**)lopts, &linkstate));
  HIPRTC_CHECK(hiprtcLinkAddData(linkstate, HIPRTC_JIT_INPUT_LLVM_BITCODE, code.data(), code.size(),
                                 "LinkISA", 0, nullptr, nullptr));

  void* finaldata;
  size_t finalsize = 0;
  HIPRTC_CHECK(hiprtcLinkComplete(linkstate, &finaldata, &finalsize));

  size_t n = 128 * 32;
  size_t bufferSize = n * sizeof(float);

  float *dX, *dY, *dOut;
  HIP_CHECK(hipMalloc(&dX, bufferSize));
  HIP_CHECK(hipMalloc(&dY, bufferSize));
  HIP_CHECK(hipMalloc(&dOut, bufferSize));

  hipModule_t module;
  hipFunction_t kernel;
  HIP_CHECK(hipModuleLoadData(&module, finaldata));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "saxpy"));

  float a = 5.1f;
  std::unique_ptr<float[]> hX{new float[n]};
  std::unique_ptr<float[]> hY{new float[n]};
  std::unique_ptr<float[]> hOut{new float[n]};
  for (size_t i = 0; i < n; ++i) {
    hX[i] = static_cast<float>(i);
    hY[i] = static_cast<float>(i * 2);
  }

  HIP_CHECK(hipMemcpy(dX, hX.get(), bufferSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dY, hY.get(), bufferSize, hipMemcpyHostToDevice));

  struct {
    float a_;
    float* b_;
    float* c_;
    float* d_;
    size_t e_;
  } args{a, dX, dY, dOut, n};

  auto size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  HIP_CHECK(hipModuleLaunchKernel(kernel, 32, 1, 1, 128, 1, 1, 0, nullptr, nullptr, config));

  HIP_CHECK(hipMemcpy(hOut.get(), dOut, bufferSize, hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(dX));
  HIP_CHECK(hipFree(dY));
  HIP_CHECK(hipFree(dOut));

  HIP_CHECK(hipModuleUnload(module));

  for (size_t i = 0; i < n; ++i) {
    REQUIRE(fabs(a * hX[i] + hY[i] - hOut[i]) <= fabs(hOut[i]) * 1e-6);
  }
}
