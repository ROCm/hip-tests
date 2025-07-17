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

texture<float, hipTextureType2D, hipReadModeElementType> texRef;

class TextureObjectTestWrapper {
 private:
  float* host_data_;
  bool ommit_destroy_;

 public:
  hipTextureObject_t texture_object = 0;
  hipResourceDesc res_desc;
  hipTextureDesc tex_desc;
  hipChannelFormatDesc channel_desc;
  hipResourceViewDesc res_vew_desc;
  hipArray_t array_member;
  size_t size; /* size in bytes*/
  int width;   /* width in elements */

  TextureObjectTestWrapper(bool useResourceViewDescriptor, bool ommitDestroy = false)
      : ommit_destroy_(ommitDestroy), width(128) {
    int i;
    size = width * sizeof(float);

    host_data_ = (float*)malloc(size);
    memset(host_data_, 0, size);

    for (i = 0; i < width; i++) {
      host_data_[i] = i;
    }

    channel_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
    HIP_CHECK(hipMallocArray(&array_member, &channel_desc, width));

    HIP_CHECK(
        hipMemcpy2DToArray(array_member, 0, 0, host_data_, size, size, 1, hipMemcpyHostToDevice));

    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = hipResourceTypeArray;
    res_desc.res.array.array = array_member;

    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = hipAddressModeClamp;
    tex_desc.filterMode = hipFilterModePoint;
    tex_desc.readMode = hipReadModeElementType;
    tex_desc.normalizedCoords = false;

    memset(&res_vew_desc, 0, sizeof(res_vew_desc));

    if (useResourceViewDescriptor) {
#if HT_AMD
      res_vew_desc.format = hipResViewFormatFloat1;
      res_vew_desc.width = size;
#else
      std::cout << "Resource View Descriptors are not supported on NVIDIA currently" << std::endl;
      useResourceViewDescriptor = false;
#endif
    }

    HIP_CHECK(hipCreateTextureObject(&texture_object, &res_desc, &tex_desc,
                                     useResourceViewDescriptor ? &res_vew_desc : nullptr));
  }

  ~TextureObjectTestWrapper() {
    if (!ommit_destroy_) {
      HIP_CHECK(hipDestroyTextureObject(texture_object));
    }
    HIP_CHECK(hipFreeArray(array_member));
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
TEST_CASE("Unit_hipTexture_Capture") {
    SECTION("hipCreateTextureObject") {
        float* host_data_;
        hipTextureObject_t texture_object = 0;
        hipResourceDesc res_desc;
        hipTextureDesc tex_desc;
        hipChannelFormatDesc channel_desc;
        hipArray_t array_member;
        int width = 128;   /* width in elements */
        size_t size = width * sizeof(float);

        host_data_ = (float*)malloc(size);
        memset(host_data_, 0, size);

        channel_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
        HIP_CHECK(hipMallocArray(&array_member, &channel_desc, width));

        HIP_CHECK(
            hipMemcpy2DToArray(array_member, 0, 0, host_data_, size, size, 1, hipMemcpyHostToDevice));

        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = hipResourceTypeArray;
        res_desc.res.array.array = array_member;

        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = hipAddressModeClamp;
        tex_desc.filterMode = hipFilterModePoint;
        tex_desc.readMode = hipReadModeElementType;
        tex_desc.normalizedCoords = false;

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCreateTextureObject(&texture_object, &res_desc, &tex_desc, nullptr), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipDestroyTextureObject(texture_object));
        HIP_CHECK(hipFreeArray(array_member));
        free(host_data_);
    }
    SECTION("hipDestroyTextureObject") {
        float* host_data_;
        hipTextureObject_t texture_object = 0;
        hipResourceDesc res_desc;
        hipTextureDesc tex_desc;
        hipChannelFormatDesc channel_desc;
        hipArray_t array_member;
        int width = 128;   /* width in elements */
        size_t size = width * sizeof(float);

        host_data_ = (float*)malloc(size);
        memset(host_data_, 0, size);

        channel_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
        HIP_CHECK(hipMallocArray(&array_member, &channel_desc, width));

        HIP_CHECK(
            hipMemcpy2DToArray(array_member, 0, 0, host_data_, size, size, 1, hipMemcpyHostToDevice));

        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = hipResourceTypeArray;
        res_desc.res.array.array = array_member;

        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = hipAddressModeClamp;
        tex_desc.filterMode = hipFilterModePoint;
        tex_desc.readMode = hipReadModeElementType;
        tex_desc.normalizedCoords = false;

        HIP_CHECK(hipCreateTextureObject(&texture_object, &res_desc, &tex_desc, nullptr));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipDestroyTextureObject(texture_object), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipFreeArray(array_member));
        free(host_data_);
    }
    SECTION("hipGetChannelDesc") {
        CHECK_IMAGE_SUPPORT;

        hipChannelFormatDesc chan_test, chan_desc;
        hipArray_t hip_array;
        chan_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindSigned);
        HIP_CHECK(hipMallocArray(&hip_array, &chan_desc, 8, 8, 0));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipGetChannelDesc(&chan_test, hip_array), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipFreeArray(hip_array));
    }
    SECTION("hipGetTextureObjectResourceDesc") {
        CHECK_IMAGE_SUPPORT;

        TextureObjectTestWrapper tex_obj_wrapper(false);
        hipResourceDesc check_desc;
        memset(&check_desc, 0, sizeof(check_desc));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipGetTextureObjectResourceDesc(&check_desc, tex_obj_wrapper.texture_object), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipGetTextureObjectTextureDesc") {
        CHECK_IMAGE_SUPPORT;

        TextureObjectTestWrapper tex_obj_wrapper(false);
        hipTextureDesc check_desc;
        memset(&check_desc, 0, sizeof(check_desc));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipGetTextureObjectTextureDesc(&check_desc, tex_obj_wrapper.texture_object), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipBindTextureToMipmappedArray") {
#if (!HT_NVIDIA) || (CUDA_VERSION < CUDA_12000)
        CHECK_IMAGE_SUPPORT
#if __HIP_NO_IMAGE_SUPPORT
        HipTest::HIP_SKIP_TEST("__HIP_NO_IMAGE_SUPPORT is set");
        return;
#endif

#if defined(_WIN32)
        hipChannelFormatDesc desc = hipCreateChannelDesc<float>();

        hipMipmappedArray_t mipArr;
        hipExtent extent = make_hipExtent(16, 16, 0);
        HIP_CHECK(hipMallocMipmappedArray(&mipArr, &desc, extent, 1));
        
        texRef.addressMode[0] = hipAddressModeWrap;
        texRef.addressMode[1] = hipAddressModeWrap;
        texRef.filterMode = hipFilterModePoint;
        texRef.normalized = 1;

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, false);
        HIP_CHECK_ERROR(hipBindTextureToMipmappedArray(&texRef, mipArr, &desc), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipFreeMipmappedArray(mipArr));
#else
        SUCCEED("Mipmaps are Supported only on windows on devices with image support,"
            " skipping the test.");
#endif
#endif
    }
    SECTION("hipGetTextureReference") {
        CHECK_IMAGE_SUPPORT
        const textureReference *tex_ref = nullptr;

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipGetTextureReference(&tex_ref, &texRef), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipBindTexture") {
        CHECK_IMAGE_SUPPORT
        size_t offset = 0;
        float* tex_buf;

        HIP_CHECK(hipMalloc(&tex_buf, 512 * sizeof(float)));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipBindTexture(&offset, texRef, reinterpret_cast<void*>(tex_buf),
                                       512 * sizeof(float)), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipUnbindTexture(&texRef));
        HIP_CHECK(hipFree(tex_buf));
    }
    SECTION("hipBindTexture2D") {
        CHECK_IMAGE_SUPPORT
        float* device_ptr;
        size_t device_pitch, texture_offset;
        HIP_CHECK(hipMallocPitch(&device_ptr, &device_pitch, 12, 8));
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipBindTexture2D(&texture_offset, &texRef, device_ptr, &texRef.channelDesc,
                                         12, 8, device_pitch), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipUnbindTexture(texRef));
        HIP_CHECK(hipFree((void*)device_ptr));
    }
    SECTION("hipBindTextureToArray") {
        CHECK_IMAGE_SUPPORT

        constexpr unsigned int width = 1024;
        constexpr unsigned int height = 1;

        hipChannelFormatDesc desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindSigned);

        hipArray_t array;
        HIP_CHECK(hipMallocArray(&array, &desc, width, height));

        texRef.addressMode[0] = hipAddressModeWrap;
        texRef.filterMode = hipFilterModePoint;
        texRef.normalized = 0;

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipBindTextureToArray(texRef, array, desc), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipUnbindTexture(texRef));
        HIP_CHECK(hipFreeArray(array));        
    }
    SECTION("hipGetTextureAlignmentOffset") {
        CHECK_IMAGE_SUPPORT

        size_t offset = 0;
        size_t *tex_buf;
        hipChannelFormatDesc chanDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

        HIP_CHECK(hipMalloc(&tex_buf, 32));
        HIP_CHECK(hipBindTexture(&offset, texRef, reinterpret_cast<void*>(tex_buf), chanDesc, 32));
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipGetTextureAlignmentOffset(&offset, &texRef), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipFree(tex_buf));
        HIP_CHECK(hipUnbindTexture(texRef));
        
    }
    SECTION("hipUnbindTexture") {
        CHECK_IMAGE_SUPPORT
        size_t offset = 0;
        float* tex_buf;

        HIP_CHECK(hipMalloc(&tex_buf, 512 * sizeof(float)));

        HIP_CHECK(hipBindTexture(&offset, texRef, reinterpret_cast<void*>(tex_buf),
                                       512 * sizeof(float)));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipUnbindTexture(&texRef), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
        HIP_CHECK(hipFree(tex_buf));
    }
}
