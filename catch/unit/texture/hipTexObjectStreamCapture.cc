/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <hip_test_common.hh>

texture<float, 1, hipReadModeElementType> tex;

class TexObjectTestWrapper {
 private:
  float* host_data_;
  bool ommit_destroy_;

 public:
  hipTextureObject_t texture_object = 0;
  HIP_RESOURCE_DESC res_desc;
  HIP_TEXTURE_DESC tex_desc;
  HIP_RESOURCE_VIEW_DESC res_view_desc;
  HIP_ARRAY_DESCRIPTOR array_desc;
  hipArray_t array_member;
  size_t size; /* size in bytes*/
  int width;   /* width in elements */

  TexObjectTestWrapper(bool useResourceViewDescriptor, bool ommitDestroy = false)
      : ommit_destroy_(ommitDestroy), width(128) {
    int i;
    size = width * sizeof(float);

    host_data_ = (float*)malloc(size);
    memset(host_data_, 0, size);

    for (i = 0; i < width; i++) {
      host_data_[i] = i;
    }

    memset(&array_desc, 0, sizeof(array_desc));
    array_desc.Format = HIP_AD_FORMAT_FLOAT;
    array_desc.NumChannels = 1;
    array_desc.Width = width;
    array_desc.Height = 0;

    HIP_CHECK(hipArrayCreate(&array_member, &array_desc));
    HIP_CHECK(hipMemcpyHtoA(reinterpret_cast<hipArray_t>(array_member), 0, host_data_, size));

    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = HIP_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = array_member;
    res_desc.flags = 0;

    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.filterMode = HIP_TR_FILTER_MODE_POINT;
    tex_desc.flags = 0;

    memset(&res_view_desc, 0, sizeof(res_view_desc));

    if (useResourceViewDescriptor) {
#if HT_AMD
      res_view_desc.format = HIP_RES_VIEW_FORMAT_FLOAT_1X32;
      res_view_desc.width = size;
#else
      /* Resource View Descriptors are not supported on NVIDIA currently */
      useResourceViewDescriptor = false;
#endif
    }

    HIP_CHECK(hipTexObjectCreate(&texture_object, &res_desc, &tex_desc,
                                 useResourceViewDescriptor ? &res_view_desc : nullptr));
  }

  ~TexObjectTestWrapper() {
    if (!ommit_destroy_) {
      HIP_CHECK(hipTexObjectDestroy(texture_object));
    }
    HIP_CHECK(hipArrayDestroy(array_member));
    free(host_data_);
  }
};

/**
 * Test Description
 * ------------------------
 *    - Test Texture APIs while stream is capturing.
 * Test source
 * ------------------------
 *    - unit/texture/hipTextureStreamcapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipTexObject_Capture") {
    SECTION("hipTexObjectCreate") {
        float* host_data_;
        hipTextureObject_t texture_object = 0;
        HIP_RESOURCE_DESC res_desc;
        HIP_TEXTURE_DESC tex_desc;
        HIP_ARRAY_DESCRIPTOR array_desc;
        hipArray_t array_member;
        int width = 128;
        size_t size = width * sizeof(float);

        host_data_ = (float*)malloc(size);
        memset(host_data_, 0, size);

        memset(&array_desc, 0, sizeof(array_desc));
        array_desc.Format = HIP_AD_FORMAT_FLOAT;
        array_desc.NumChannels = 1;
        array_desc.Width = width;
        array_desc.Height = 0;

        HIP_CHECK(hipArrayCreate(&array_member, &array_desc));
        HIP_CHECK(hipMemcpyHtoA(reinterpret_cast<hipArray_t>(array_member), 0, host_data_, size));

        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = HIP_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array_member;
        res_desc.flags = 0;

        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.filterMode = HIP_TR_FILTER_MODE_POINT;
        tex_desc.flags = 0;

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexObjectCreate(&texture_object, &res_desc, &tex_desc, nullptr), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
        
        HIP_CHECK(hipTexObjectDestroy(texture_object));
        HIP_CHECK(hipArrayDestroy(array_member));
        free(host_data_);
    }
    SECTION("hipTexObjectDestroy") {
        float* host_data_;
        hipTextureObject_t texture_object = 0;
        HIP_RESOURCE_DESC res_desc;
        HIP_TEXTURE_DESC tex_desc;
        HIP_ARRAY_DESCRIPTOR array_desc;
        hipArray_t array_member;
        int width = 128;
        size_t size = width * sizeof(float);

        host_data_ = (float*)malloc(size);
        memset(host_data_, 0, size);

        memset(&array_desc, 0, sizeof(array_desc));
        array_desc.Format = HIP_AD_FORMAT_FLOAT;
        array_desc.NumChannels = 1;
        array_desc.Width = width;
        array_desc.Height = 0;

        HIP_CHECK(hipArrayCreate(&array_member, &array_desc));
        HIP_CHECK(hipMemcpyHtoA(reinterpret_cast<hipArray_t>(array_member), 0, host_data_, size));

        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = HIP_RESOURCE_TYPE_ARRAY;
        res_desc.res.array.hArray = array_member;
        res_desc.flags = 0;

        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.filterMode = HIP_TR_FILTER_MODE_POINT;
        tex_desc.flags = 0;

        HIP_CHECK(hipTexObjectCreate(&texture_object, &res_desc, &tex_desc, nullptr));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexObjectDestroy(texture_object), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipArrayDestroy(array_member));
        free(host_data_);
    }
    SECTION("hipTexObjectGetResourceDesc") {
        CHECK_IMAGE_SUPPORT;

        TexObjectTestWrapper tex_obj_wrapper(false);
        HIP_RESOURCE_DESC check_desc;
        memset(&check_desc, 0, sizeof(check_desc));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexObjectGetResourceDesc(&check_desc, tex_obj_wrapper.texture_object), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexObjectGetResourceViewDesc") {
        CHECK_IMAGE_SUPPORT;

        TexObjectTestWrapper tex_obj_wrapper(true);
        HIP_RESOURCE_VIEW_DESC check_desc;
        memset(&check_desc, 0, sizeof(check_desc));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexObjectGetResourceViewDesc(&check_desc, tex_obj_wrapper.texture_object), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexObjectGetTextureDesc") {
        CHECK_IMAGE_SUPPORT;

        TexObjectTestWrapper tex_obj_wrapper(false);
        HIP_TEXTURE_DESC check_desc;
        memset(&check_desc, 0, sizeof(check_desc));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexObjectGetTextureDesc(&check_desc, tex_obj_wrapper.texture_object), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000
    SECTION("hipTexRefGetBorderColor") {
        CHECK_IMAGE_SUPPORT
        float set_border_color[3] = {1, 2, 3};
        float get_border_color[3] = {0, 0, 0};
        hipModule_t module = nullptr;
        hipTexRef tex_ref = nullptr;

        HIP_CHECK(hipFree(nullptr));
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
        HIP_CHECK(hipTexRefSetBorderColor(tex_ref, set_border_color));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefGetBorderColor(get_border_color, tex_ref), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
    }
    SECTION("hipTexRefGetArray") {
        CHECK_IMAGE_SUPPORT
        hipArray_t array_set = nullptr;
        hipArray_t array_get = nullptr;
        hipModule_t module = nullptr;
        hipTexRef tex_ref = nullptr;
        HIP_ARRAY_DESCRIPTOR array_desc;

        array_desc.Format = HIP_AD_FORMAT_FLOAT;
        array_desc.NumChannels = 1;
        array_desc.Width = 16;
        array_desc.Height = 16;

        HIP_CHECK(hipFree(nullptr));
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
        HIP_CHECK(hipArrayCreate(&array_set, &array_desc));
        HIP_CHECK(hipTexRefSetArray(tex_ref, array_set, HIP_TRSA_OVERRIDE_FORMAT));

        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefGetArray(&array_get, tex_ref), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipArrayDestroy(array_set));
        HIP_CHECK(hipModuleUnload(module));
    }
    SECTION("hipTexRefSetAddressMode") {
        CHECK_IMAGE_SUPPORT

        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipCtxCreate(&ctx, 0, device));

        hipTexRef tex_ref = nullptr;
        hipModule_t module = nullptr;
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

        auto dim = 1;
#if HT_AMD
        auto am = hipAddressModeWrap;
#else
        auto am = HIP_TR_ADDRESS_MODE_WRAP;
#endif

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetAddressMode(tex_ref, dim, am), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
        HIP_CHECK(hipCtxDestroy(ctx));
    }
    SECTION("hipTexRefSetArray") {
        CHECK_IMAGE_SUPPORT
        hipArray_t array_set = nullptr;
        hipModule_t module = nullptr;
        hipTexRef tex_ref = nullptr;
        HIP_ARRAY_DESCRIPTOR array_desc;

        array_desc.Format = HIP_AD_FORMAT_FLOAT;
        array_desc.NumChannels = 1;
        array_desc.Width = 16;
        array_desc.Height = 16;

        HIP_CHECK(hipFree(nullptr));
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
        HIP_CHECK(hipArrayCreate(&array_set, &array_desc));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetArray(tex_ref, array_set, HIP_TRSA_OVERRIDE_FORMAT), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
        
        HIP_CHECK(hipArrayDestroy(array_set));
        HIP_CHECK(hipModuleUnload(module));

    }
    SECTION("hipTexRefSetFilterMode") {
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetFilterMode(&tex, hipFilterModeLinear), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexRefSetFlags") {
        CHECK_IMAGE_SUPPORT

        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipCtxCreate(&ctx, 0, device));

        hipTexRef tex_ref = nullptr;
        hipModule_t module = nullptr;
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

        unsigned int flags = HIP_TRSF_READ_AS_INTEGER;

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetFlags(tex_ref, flags), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
        HIP_CHECK(hipCtxDestroy(ctx));

    }
    SECTION("hipTexRefSetFormat") {
        CHECK_IMAGE_SUPPORT
        hipModule_t module = nullptr;
        hipTexRef tex_ref = nullptr;
        int num_channels = 0;

        hipArray_Format format_set = HIP_AD_FORMAT_UNSIGNED_INT32;

        HIP_CHECK(hipFree(nullptr));
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetFormat(tex_ref, format_set, num_channels), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
        
    }
    SECTION("hipTexRefGetAddress") {
        CHECK_IMAGE_SUPPORT
        hipDeviceptr_t device_ptr;
        hipModule_t module = nullptr;
        hipTexRef tex_ref = nullptr;
        float* tex_buffer = nullptr;
        size_t offset = 0, tex_size = sizeof(float);

        HIP_CHECK(hipFree(nullptr));
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
        HIP_CHECK(hipMalloc(&tex_buffer, sizeof(float)));
        HIP_CHECK(hipTexRefSetAddress(&offset, tex_ref, reinterpret_cast<hipDeviceptr_t>(tex_buffer),
                                        tex_size));
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefGetAddress(&device_ptr, tex_ref), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
        HIP_CHECK(hipFree(tex_buffer));
    }
    SECTION("hipTexRefGetAddressMode") {
        CHECK_IMAGE_SUPPORT

        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipCtxCreate(&ctx, 0, device));

        hipTexRef tex_ref = nullptr;
        hipModule_t module = nullptr;
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

        int dim = 0;
        #if HT_AMD
        hipTextureAddressMode am = hipAddressModeWrap;
        hipTextureAddressMode out_am;
        #else
        HIPaddress_mode am = HIP_TR_ADDRESS_MODE_WRAP;
        HIPaddress_mode out_am;
        #endif

        HIP_CHECK(hipTexRefSetAddressMode(tex_ref, dim, am));
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefGetAddressMode(&out_am, tex_ref, dim), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
        HIP_CHECK(hipCtxDestroy(ctx));
    }
    SECTION("hipTexRefGetFilterMode") {
        HIP_CHECK(hipTexRefSetFilterMode(&tex, hipFilterModeLinear));

        hipTextureFilterMode mode;

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefGetFilterMode(&mode, &tex), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexRefGetFlags") {
        CHECK_IMAGE_SUPPORT

        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipCtxCreate(&ctx, 0, device));

        hipTexRef tex_ref = nullptr;
        hipModule_t module = nullptr;
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

        unsigned int flags = HIP_TRSF_READ_AS_INTEGER;
        HIP_CHECK(hipTexRefSetFlags(tex_ref, flags));

        unsigned int out_flags;
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefGetFlags(&out_flags, tex_ref), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
        HIP_CHECK(hipCtxDestroy(ctx));
    }
    SECTION("hipTexRefGetFormat") {
        CHECK_IMAGE_SUPPORT
        hipModule_t module = nullptr;
        hipTexRef tex_ref = nullptr;
        int num_channels = 0;
        hipArray_Format format_get;

        hipArray_Format format_set = HIP_AD_FORMAT_UNSIGNED_INT32;

        HIP_CHECK(hipFree(nullptr));
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

        HIP_CHECK(hipTexRefSetFormat(tex_ref, format_set, num_channels));
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefGetFormat(&format_get, &num_channels, tex_ref), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
    }
    SECTION("hipTexRefGetMaxAnisotropy") {
        CHECK_IMAGE_SUPPORT

        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipCtxCreate(&ctx, 0, device));

        hipTexRef tex_ref = nullptr;
        hipModule_t module = nullptr;
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

        unsigned int max_anisotropy = 1;
        HIP_CHECK(hipTexRefSetMaxAnisotropy(tex_ref, max_anisotropy));

        int out_anisotropy;
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefGetMaxAnisotropy(&out_anisotropy, tex_ref), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
        HIP_CHECK(hipCtxDestroy(ctx));
    }
#endif
#if defined(_WIN32)
    SECTION("hipTexRefGetMipmapFilterMode") {
        HIP_CHECK(hipTexRefSetMipmapFilterMode(&tex, hipFilterModeLinear));
        hipTextureFilterMode mode;

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        // Currently returns hipErrorInvalidValue on success
        if (memcpy_err == hipSuccess){
            HIP_CHECK_ERROR(hipTexRefGetMipmapFilterMode(&mode, &tex), hipErrorInvalidValue);
        }
        else {
            HIP_CHECK_ERROR(hipTexRefGetMipmapFilterMode(&mode, &tex), memcpy_err);
        }
        
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexRefGetMipmapLevelBias") {
        HIP_CHECK(hipTexRefSetMipmapLevelBias(&tex, 1.0f));
        float bias;

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        // Currently returns hipErrorInvalidValue on success
        if (memcpy_err == hipSuccess){
            HIP_CHECK_ERROR(hipTexRefGetMipmapLevelBias(&bias, &tex), hipErrorInvalidValue);
        }
        else {
            HIP_CHECK_ERROR(hipTexRefGetMipmapLevelBias(&bias, &tex), memcpy_err);
        }
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexRefGetMipmapLevelClamp") {
        HIP_CHECK(hipTexRefSetMipmapLevelClamp(&tex, 0.0f, 5.0f));
        float minClamp, maxClamp;
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        // Currently returns hipErrorInvalidValue on success
        if (memcpy_err == hipSuccess){
            HIP_CHECK_ERROR(hipTexRefGetMipmapLevelClamp(&minClamp, &maxClamp, &tex), hipErrorInvalidValue);
        }
        else {
            HIP_CHECK_ERROR(hipTexRefGetMipmapLevelClamp(&minClamp, &maxClamp, &tex), memcpy_err);
        }
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexRefGetMipMappedArray") {
        // HIP_CHECK(hipTexRefSetMipmappedArray(&tex, (hipMipmappedArray_t)0, 0));
        hipMipmappedArray_t arr;
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        // Currently returns hipErrorInvalidValue
        if (memcpy_err == hipSuccess){
            HIP_CHECK_ERROR(hipTexRefGetMipMappedArray(&arr, &tex), hipErrorInvalidValue);
        }
        else {
            HIP_CHECK_ERROR(hipTexRefGetMipMappedArray(&arr, &tex), memcpy_err);
        }
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexRefSetAddress") {
        CHECK_IMAGE_SUPPORT
        hipModule_t module = nullptr;
        hipTexRef tex_ref = nullptr;
        float* tex_buffer = nullptr;
        size_t offset = 0, tex_size = sizeof(float);

        HIP_CHECK(hipFree(nullptr));
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
        HIP_CHECK(hipMalloc(&tex_buffer, sizeof(float)));
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetAddress(&offset, tex_ref, reinterpret_cast<hipDeviceptr_t>(tex_buffer),
                                            tex_size), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
        HIP_CHECK(hipFree(tex_buffer));
    }
    SECTION("hipTexRefSetAddress2D") {
        CHECK_IMAGE_SUPPORT

        constexpr int width = 256;
        constexpr int height = 256;

        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipCtxCreate(&ctx, 0, device));

        hipTexRef tex_ref = nullptr;
        hipModule_t module = nullptr;
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

        int size = width * height * sizeof(float);
        float* h_data = new float[size];

        hipDeviceptr_t d_data;
        size_t dest_pitch;
        HIP_CHECK(hipMemAllocPitch(&d_data, &dest_pitch, width * sizeof(float), height, sizeof(float)));

        HIP_ARRAY_DESCRIPTOR array_desc;
        array_desc.Format = HIP_AD_FORMAT_FLOAT;
        array_desc.Height = height;
        array_desc.Width = width;
        array_desc.NumChannels = 1;

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetAddress2D(tex_ref, &array_desc, d_data, dest_pitch), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        free(h_data);
        HIP_CHECK(hipFree((void*)d_data));
        HIP_CHECK(hipModuleUnload(module));
        HIP_CHECK(hipCtxDestroy(ctx));
        
    }
    SECTION("hipTexRefSetMaxAnisotropy") {
        CHECK_IMAGE_SUPPORT

        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipCtxCreate(&ctx, 0, device));

        hipTexRef tex_ref = nullptr;
        hipModule_t module = nullptr;
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

        unsigned int max_anisotropy = 1;
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetMaxAnisotropy(tex_ref, max_anisotropy), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
        HIP_CHECK(hipCtxDestroy(ctx));
    }
    SECTION("hipTexRefSetBorderColor") {
        float set_border_color[3] = {1, 2, 3};
        hipModule_t module = nullptr;
        hipTexRef tex_ref = nullptr;

        HIP_CHECK(hipFree(nullptr));
        HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
        HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetBorderColor(tex_ref, set_border_color), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipModuleUnload(module));
    }
#endif
#if defined(_WIN32)
    SECTION("hipTexRefSetMipmapFilterMode") {
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetMipmapFilterMode(&tex, hipFilterModeLinear), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexRefSetMipmapLevelBias") {
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetMipmapLevelBias(&tex, 1.0f), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexRefSetMipmapLevelClamp") {
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipTexRefSetMipmapLevelClamp(&tex, 0.0f, 5.0f), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipTexRefSetMipmappedArray") {
        hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
        hipExtent extent = make_hipExtent(16, 16, 1);
        hipMipmappedArray_t mipArr;
        HIP_CHECK(hipMallocMipmappedArray(&mipArr, &desc, extent, 1));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        // Currently returns hipErrorInvalidValue
        if (memcpy_err == hipSuccess){
            HIP_CHECK_ERROR(hipTexRefSetMipmappedArray(&tex, mipArr, 0), hipErrorInvalidValue);
        }
        else {
            HIP_CHECK_ERROR(hipTexRefSetMipmappedArray(&tex, mipArr, 0), memcpy_err);
        }
        END_CAPTURE_SYNC(memcpy_err);
    }
#endif
}
