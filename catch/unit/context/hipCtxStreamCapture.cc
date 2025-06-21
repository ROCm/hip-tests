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

/**
 * Test Description
 * ------------------------
 *    - Test Context APIs while stream is capturing.
 * Test source
 * ------------------------
 *    - unit/context/hipCtxStreamcapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipCtx_Capture") {
    SECTION("hipCtxCreate"){
        hipCtx_t context;
    
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxCreate(&context, 0, 0), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipCtxDestroy"){
        hipCtx_t context;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
    
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxDestroy(context), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipDevicePrimaryCtxRetain"){
        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipInit(0));
        HIP_CHECK(hipDeviceGet(&device, 0));
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipDevicePrimaryCtxRetain(&ctx,device), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipDevicePrimaryCtxRelease(device));
    }
    SECTION("hipDevicePrimaryCtxRelease"){
        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipInit(0));
        HIP_CHECK(hipDeviceGet(&device, 0));
        
        HIP_CHECK(hipDevicePrimaryCtxRetain(&ctx,device));
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipDevicePrimaryCtxRelease(device), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipCtxPushCurrent"){
        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipInit(0));
        HIP_CHECK(hipDeviceGet(&device, 0));
        
        HIP_CHECK(hipDevicePrimaryCtxRetain(&ctx,device));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxPushCurrent(ctx), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipDevicePrimaryCtxRelease(device));
    }
    SECTION("hipCtxPopCurrent"){
        hipCtx_t ctx;
        hipDevice_t device;

        HIP_CHECK(hipInit(0));
        HIP_CHECK(hipDeviceGet(&device, 0));
        
        HIP_CHECK(hipDevicePrimaryCtxRetain(&ctx,device));
        HIP_CHECK(hipCtxPushCurrent(ctx));

        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxPopCurrent(&ctx), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipDevicePrimaryCtxRelease(device));
    }
    SECTION("hipCtxSetCurrent"){
        hipCtx_t context;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
    
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxSetCurrent(context), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipCtxSetCurrent"){
        hipCtx_t context;
        hipCtx_t ctx;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
        HIP_CHECK(hipCtxSetCurrent(context));
    
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxGetCurrent(&ctx), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipCtxGetDevice"){
        hipCtx_t context;
        hipDevice_t dev;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
        HIP_CHECK(hipCtxSetCurrent(context));
    
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxGetDevice(&dev), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipCtxGetApiVersion"){
        hipCtx_t context;
        unsigned int apiVersion;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
    
        hipError_t memcpy_err = hipErrorNotSupported;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxGetApiVersion(context, &apiVersion), memcpy_err);
        if (memcpy_err == hipErrorNotSupported) {
            memcpy_err = hipSuccess;
        }
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipCtxGetCacheConfig"){
        hipCtx_t context;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
        hipFuncCache_t cache_config;
    
        hipError_t memcpy_err = hipErrorNotSupported;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxGetCacheConfig(&cache_config), memcpy_err);
        if (memcpy_err == hipErrorNotSupported) {
            memcpy_err = hipSuccess;
        }
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipCtxSetCacheConfig"){
        hipCtx_t context;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
    
        hipError_t memcpy_err = hipErrorNotSupported;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxSetCacheConfig(hipFuncCachePreferL1), memcpy_err);
        if (memcpy_err == hipErrorNotSupported) {
            memcpy_err = hipSuccess;
        }
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipCtxSetSharedMemConfig"){
        hipCtx_t context;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
    
        hipError_t memcpy_err = hipErrorNotSupported;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxSetSharedMemConfig(hipSharedMemBankSizeDefault), memcpy_err);
        if (memcpy_err == hipErrorNotSupported) {
            memcpy_err = hipSuccess;
        }
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipCtxGetSharedMemConfig"){
        hipCtx_t context;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
        hipSharedMemConfig mem_config;
    
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxGetSharedMemConfig(&mem_config), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipCtxSynchronize"){
        hipCtx_t context;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
    
        hipError_t memcpy_err = hipErrorNotSupported;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxSynchronize(), memcpy_err);
        if (memcpy_err == hipErrorNotSupported) {
            memcpy_err = hipSuccess;
        }
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipCtxGetFlags"){
        hipCtx_t context;
        HIP_CHECK(hipCtxCreate(&context, 0, 0));
        unsigned int flags;
    
        hipError_t memcpy_err = hipErrorNotSupported;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxGetFlags(&flags), memcpy_err);
        if (memcpy_err == hipErrorNotSupported) {
            memcpy_err = hipSuccess;
        }
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context));
    }
    SECTION("hipDevicePrimaryCtxRelease"){
        hipCtx_t ctx;
        hipDevice_t device;
        int isActive;

        HIP_CHECK(hipInit(0));
        HIP_CHECK(hipDeviceGet(&device, 0));
        
        HIP_CHECK(hipDevicePrimaryCtxRetain(&ctx,device));
        
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipDevicePrimaryCtxGetState(0, 0, &isActive), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);

        HIP_CHECK(hipDevicePrimaryCtxRelease(device));
    }
    SECTION("hipDevicePrimaryCtxSetFlags"){
        hipError_t memcpy_err = hipErrorContextAlreadyInUse;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipDevicePrimaryCtxSetFlags(0, 0), memcpy_err);
        if (memcpy_err == hipErrorContextAlreadyInUse) {
            memcpy_err = hipSuccess;
        }
        END_CAPTURE_SYNC(memcpy_err);
    }
    SECTION("hipDevicePrimaryCtxReset"){
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipDevicePrimaryCtxReset(0), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    }    
}

/**
 * Test Description
 * ------------------------
 *    - Test Ctx PeerAccess APIs while stream is capturing.
 * Test source
 * ------------------------
 *    - unit/context/hipCtxStreamcapture.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipCtxPeerAccess_Capture") {
    SECTION("hipCtxEnablePeerAccess"){
        int gpuCount = 0;
        HIP_CHECK(hipGetDeviceCount(&gpuCount));
        if (gpuCount < 1) {
            fprintf(stderr, "Need at least 1 GPU, skipped!\n");
            return;
        }
        hipCtx_t context0, context1;
        HIP_CHECK(hipCtxCreate(&context0, 0, 0));
        HIP_CHECK(hipCtxCreate(&context1, 0, 1));
        HIP_CHECK(hipCtxSetCurrent(context0));
    
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxEnablePeerAccess(context1, 0), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context0));
        HIP_CHECK(hipCtxDestroy(context1));
    }
    SECTION("hipCtxDisablePeerAccess"){
        int gpuCount = 0;
        HIP_CHECK(hipGetDeviceCount(&gpuCount));
        if (gpuCount < 1) {
            fprintf(stderr, "Need at least 1 GPU, skipped!\n");
            return;
        }
        hipCtx_t context0, context1;
        HIP_CHECK(hipCtxCreate(&context0, 0, 0));
        HIP_CHECK(hipCtxCreate(&context1, 0, 1));
        HIP_CHECK(hipCtxSetCurrent(context0));
    
        hipError_t memcpy_err = hipSuccess;
        BEGIN_CAPTURE_SYNC(memcpy_err, true);
        HIP_CHECK_ERROR(hipCtxDisablePeerAccess(context1), memcpy_err);
        END_CAPTURE_SYNC(memcpy_err);
    
        HIP_CHECK(hipCtxDestroy(context0));
        HIP_CHECK(hipCtxDestroy(context1));
    }
}
