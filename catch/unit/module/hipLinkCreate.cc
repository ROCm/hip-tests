/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime_api.h>

#include <fstream>
#include <vector>
#include <string.h>
#include <unistd.h>
#include <limits.h>


#define SPIRV_FILE "addKernel.spv"
#define SPIRV_BUNDLED_FILE "addKernel-bundle.spv"
#define ARRAY_SIZE 1
#define REF_VALUE 5

#if HT_AMD
static inline bool load_co_from_file(const char *filename, std::vector<char> &co_source) {
    std::ifstream file_stream{filename, std::ios_base::in | std::ios_base::binary};
    if (!file_stream.good()) {
        return false;
    }

    file_stream.seekg(0, std::ios::end);
    std::streampos file_size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);

    // Read the file contents
    co_source.resize(file_size);
    file_stream.read(co_source.data(), file_size);

    file_stream.close();

    return true;
}

void JitLink(hipModule_t *Module, hipFunction_t *Kernel, hipLinkState_t *LinkState,
             hipJitInputType input_type, bool from_file, const char *filename) {

   const char* isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm", "-inlinehint-threshold=1"};
   std::vector<hipJitOption> jit_options = {hipJitOptionIRtoISAOptExt,
                                               hipJitOptionIRtoISAOptCountExt};
   size_t isaoptssize = 4;
   const void* lopts[] = {(void*)isaopts, (void*)(isaoptssize)};


   HIP_CHECK(hipLinkCreate(jit_options.size(), jit_options.data(), (void **)lopts, LinkState));

   if (!from_file) {
      std::vector<char> co_source;
      REQUIRE(load_co_from_file(filename, co_source) == true);
      HIP_CHECK(hipLinkAddData(*LinkState,input_type,(void *)co_source.data(),
                                co_source.size(), "LinkSPIRV1", 0, nullptr, nullptr));
   } else {
      HIP_CHECK(hipLinkAddFile(*LinkState,input_type, filename, 0, nullptr, nullptr));
   }

   void *linkOut;
   size_t linkSize = 0;
   // Complete Linker Step
   HIP_CHECK(hipLinkComplete(*LinkState, &linkOut, &linkSize));

   // Load codeobject into module
   HIP_CHECK(hipModuleLoadData(Module, linkOut));

   // Locate Kernel Entry Point
   HIP_CHECK(hipModuleGetFunction(Kernel, *Module, "addKernel"));

   // Destroy Linker invocation
   HIP_CHECK(hipLinkDestroy(*LinkState));

}


TEST_CASE("Unit_hip_linker_spirv_input") {
    size_t N = ARRAY_SIZE;
    size_t sizeBytes = N * sizeof(int);
    int *A_h = new int[sizeBytes];
    REQUIRE(A_h != nullptr);
    int *A_d; HIP_CHECK(hipMalloc(&A_d, sizeBytes));
    REQUIRE(A_d != nullptr);

    for ( int i = 0; i < N; i++) {
        A_h[i] = REF_VALUE + i;
    }
    HIP_CHECK(hipMemcpy(A_d, A_h, sizeBytes, hipMemcpyHostToDevice));

    hipModule_t Module;
    hipFunction_t Kernel;
    hipLinkState_t Linkstate;

    bool from_file = false;
    hipJitInputType input_type= hipJitInputSpirv;
    const char *filename;
    SECTION("Link Add Data with Bundled Spirv") {
        from_file = false;
        input_type = hipJitInputSpirv;
        filename = SPIRV_BUNDLED_FILE;
    }
    SECTION("Link Add Data with UnBundled Spirv") {
        from_file = false;
        input_type = hipJitInputSpirv;
        filename = SPIRV_FILE;
    }
    SECTION("Link Add File with Bundled Spirv") {
        from_file = true;
        input_type = hipJitInputSpirv;
        filename = SPIRV_BUNDLED_FILE;
    }
    SECTION("Link Add File with UnBundled Spirv") {
        from_file = true;
        input_type = hipJitInputSpirv;
        filename = SPIRV_FILE;
    }

    JitLink(&Module, &Kernel, &Linkstate, input_type, from_file,filename);
    void *args[2] = {&A_d, &N};
    HIP_CHECK(hipModuleLaunchKernel(Kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    HIP_CHECK(hipModuleUnload(Module));
    HIP_CHECK(hipMemcpy(A_h, A_d, sizeBytes, hipMemcpyDeviceToHost));
    REQUIRE(A_h[0] == REF_VALUE + 2);
    HIP_CHECK(hipFree(A_d));
    delete[] A_h;
}


TEST_CASE("Unit_hipLinkCreate_Negative") {

    hipLinkState_t linkstate;
    hipJitOption options[4];
    void *optionVals[4];
    options[0] = hipJitOptionInfoLogBuffer;
    options[1] = hipJitOptionInfoLogBufferSizeBytes;
    options[2] = hipJitOptionErrorLogBuffer;
    options[3] = hipJitOptionErrorLogBufferSizeBytes;

    SECTION("linkstate == nullptr") {
        HIP_CHECK_ERROR(hipLinkCreate(0, options, optionVals, nullptr), hipErrorInvalidValue);
    }

    SECTION("num_options != 0 ") {
        SECTION("option val pptr == nullptr") {
            HIP_CHECK_ERROR(hipLinkCreate(4, options, nullptr, &linkstate),
                                                                        hipErrorInvalidValue);
        }
        SECTION("option val ptr == nullptr") {
            HIP_CHECK_ERROR(hipLinkCreate(4, options, optionVals, &linkstate),
                                                                hipErrorInvalidConfiguration);
        }
    }

}

TEST_CASE("Unit_hipLinkAddFile_Negative") {
    hipLinkState_t linkstate;
    HIP_CHECK(hipLinkCreate(0, nullptr, nullptr, &linkstate));

    SECTION("linkstate == nullptr") {
        HIP_CHECK_ERROR(hipLinkAddFile(nullptr, hipJitInputSpirv, SPIRV_FILE, 0, nullptr, nullptr),
                                                                        hipErrorInvalidHandle);
    }
    SECTION("Jit input Cubin") {
        HIP_CHECK_ERROR(
                hipLinkAddFile(linkstate, hipJitInputCubin, SPIRV_FILE, 0, nullptr, nullptr),
                                                                        hipErrorInvalidValue);
    }
    SECTION("Jit Input PTX") {
        HIP_CHECK_ERROR(hipLinkAddFile(linkstate, hipJitInputPtx, SPIRV_FILE, 0, nullptr, nullptr),
                                                                        hipErrorInvalidValue);
    }
    SECTION("Input File not valid") {
        HIP_CHECK_ERROR(
                hipLinkAddFile(linkstate, hipJitInputSpirv, "unknown_file", 0, nullptr, nullptr),
                                                                    hipErrorInvalidConfiguration);
    }
    HIP_CHECK(hipLinkDestroy(linkstate));
}
#endif