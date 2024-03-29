/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <iostream>
#include <iomanip>
#include "hip/hip_runtime.h"
#include "hip_helper.h"

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

void printCompilerInfo() {
#ifdef __NVCC__
    printf("compiler: nvcc\n");
#endif
}

double bytesToKB(size_t s) { return (double)s / (1024.0); }
double bytesToGB(size_t s) { return (double)s / (1024.0 * 1024.0 * 1024.0); }

#define printLimit(w1, limit, units)                                                               \
    {                                                                                              \
        size_t val;                                                                                \
        cudaDeviceGetLimit(&val, limit);                                                           \
        std::cout << setw(w1) << #limit ": " << val << " " << units << std::endl;                  \
    }


void printDeviceProp(int deviceId) {
    using namespace std;
    const int w1 = 34;

    cout << left;

    cout << setw(w1)
         << "--------------------------------------------------------------------------------"
         << endl;
    cout << setw(w1) << "device#" << deviceId << endl;

    hipDeviceProp_t props = {0};
    checkHipErrors(hipGetDeviceProperties(&props, deviceId));

    cout << setw(w1) << "Name: " << props.name << endl;
    cout << setw(w1) << "pciBusID: " << props.pciBusID << endl;
    cout << setw(w1) << "pciDeviceID: " << props.pciDeviceID << endl;
    cout << setw(w1) << "pciDomainID: " << props.pciDomainID << endl;
    cout << setw(w1) << "multiProcessorCount: " << props.multiProcessorCount << endl;
    cout << setw(w1) << "maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor
         << endl;
    cout << setw(w1) << "isMultiGpuBoard: " << props.isMultiGpuBoard << endl;
    cout << setw(w1) << "clockRate: " << (float)props.clockRate / 1000.0 << " Mhz" << endl;
    cout << setw(w1) << "memoryClockRate: " << (float)props.memoryClockRate / 1000.0 << " Mhz"
         << endl;
    cout << setw(w1) << "memoryBusWidth: " << props.memoryBusWidth << endl;
    cout << setw(w1) << "totalGlobalMem: " << fixed << setprecision(2)
         << bytesToGB(props.totalGlobalMem) << " GB" << endl;
    cout << setw(w1) << "totalConstMem: " << props.totalConstMem << endl;
    cout << setw(w1) << "sharedMemPerBlock: " << (float)props.sharedMemPerBlock / 1024.0 << " KB"
         << endl;
    cout << setw(w1) << "canMapHostMemory: " << props.canMapHostMemory << endl;
    cout << setw(w1) << "regsPerBlock: " << props.regsPerBlock << endl;
    cout << setw(w1) << "warpSize: " << props.warpSize << endl;
    cout << setw(w1) << "l2CacheSize: " << props.l2CacheSize << endl;
    cout << setw(w1) << "computeMode: " << props.computeMode << endl;
    cout << setw(w1) << "maxThreadsPerBlock: " << props.maxThreadsPerBlock << endl;
    cout << setw(w1) << "maxThreadsDim.x: " << props.maxThreadsDim[0] << endl;
    cout << setw(w1) << "maxThreadsDim.y: " << props.maxThreadsDim[1] << endl;
    cout << setw(w1) << "maxThreadsDim.z: " << props.maxThreadsDim[2] << endl;
    cout << setw(w1) << "maxGridSize.x: " << props.maxGridSize[0] << endl;
    cout << setw(w1) << "maxGridSize.y: " << props.maxGridSize[1] << endl;
    cout << setw(w1) << "maxGridSize.z: " << props.maxGridSize[2] << endl;
    cout << setw(w1) << "major: " << props.major << endl;
    cout << setw(w1) << "minor: " << props.minor << endl;
    cout << setw(w1) << "concurrentKernels: " << props.concurrentKernels << endl;
    cout << setw(w1) << "cooperativeLaunch: " << props.cooperativeLaunch << endl;
    cout << setw(w1) << "cooperativeMultiDeviceLaunch: " << props.cooperativeMultiDeviceLaunch << endl;
    cout << setw(w1) << "isIntegrated: " << props.integrated << endl;
    cout << setw(w1) << "maxTexture1D: " << props.maxTexture1D << endl;
    cout << setw(w1) << "maxTexture2D.width: " << props.maxTexture2D[0] << endl;
    cout << setw(w1) << "maxTexture2D.height: " << props.maxTexture2D[1] << endl;
    cout << setw(w1) << "maxTexture3D.width: " << props.maxTexture3D[0] << endl;
    cout << setw(w1) << "maxTexture3D.height: " << props.maxTexture3D[1] << endl;
    cout << setw(w1) << "maxTexture3D.depth: " << props.maxTexture3D[2] << endl;
    cout << setw(w1) << "hostNativeAtomicSupported: " << props.hostNativeAtomicSupported << endl;
#ifdef __HIP_PLATFORM_AMD__
    cout << setw(w1) << "isLargeBar: " << props.isLargeBar << endl;
    cout << setw(w1) << "asicRevision: " << props.asicRevision << endl;
    cout << setw(w1) << "maxSharedMemoryPerMultiProcessor: " << fixed << setprecision(2)
         << bytesToKB(props.maxSharedMemoryPerMultiProcessor) << " KB" << endl;
    cout << setw(w1) << "clockInstructionRate: " << (float)props.clockInstructionRate / 1000.0
         << " Mhz" << endl;
    cout << setw(w1) << "arch.hasGlobalInt32Atomics: " << props.arch.hasGlobalInt32Atomics << endl;
    cout << setw(w1) << "arch.hasGlobalFloatAtomicExch: " << props.arch.hasGlobalFloatAtomicExch
         << endl;
    cout << setw(w1) << "arch.hasSharedInt32Atomics: " << props.arch.hasSharedInt32Atomics << endl;
    cout << setw(w1) << "arch.hasSharedFloatAtomicExch: " << props.arch.hasSharedFloatAtomicExch
         << endl;
    cout << setw(w1) << "arch.hasFloatAtomicAdd: " << props.arch.hasFloatAtomicAdd << endl;
    cout << setw(w1) << "arch.hasGlobalInt64Atomics: " << props.arch.hasGlobalInt64Atomics << endl;
    cout << setw(w1) << "arch.hasSharedInt64Atomics: " << props.arch.hasSharedInt64Atomics << endl;
    cout << setw(w1) << "arch.hasDoubles: " << props.arch.hasDoubles << endl;
    cout << setw(w1) << "arch.hasWarpVote: " << props.arch.hasWarpVote << endl;
    cout << setw(w1) << "arch.hasWarpBallot: " << props.arch.hasWarpBallot << endl;
    cout << setw(w1) << "arch.hasWarpShuffle: " << props.arch.hasWarpShuffle << endl;
    cout << setw(w1) << "arch.hasFunnelShift: " << props.arch.hasFunnelShift << endl;
    cout << setw(w1) << "arch.hasThreadFenceSystem: " << props.arch.hasThreadFenceSystem << endl;
    cout << setw(w1) << "arch.hasSyncThreadsExt: " << props.arch.hasSyncThreadsExt << endl;
    cout << setw(w1) << "arch.hasSurfaceFuncs: " << props.arch.hasSurfaceFuncs << endl;
    cout << setw(w1) << "arch.has3dGrid: " << props.arch.has3dGrid << endl;
    cout << setw(w1) << "arch.hasDynamicParallelism: " << props.arch.hasDynamicParallelism << endl;
    cout << setw(w1) << "gcnArchName: " << props.gcnArchName << endl;
#endif
    int deviceCnt;
    checkHipErrors(hipGetDeviceCount(&deviceCnt));
    cout << setw(w1) << "peers: ";
    for (int i = 0; i < deviceCnt; i++) {
        int isPeer;
        checkHipErrors(hipDeviceCanAccessPeer(&isPeer, i, deviceId));
        if (isPeer) {
            cout << "device#" << i << " ";
        }
    }
    cout << endl;
    cout << setw(w1) << "non-peers: ";
    for (int i = 0; i < deviceCnt; i++) {
        int isPeer;
        checkHipErrors(hipDeviceCanAccessPeer(&isPeer, i, deviceId));
        if (!isPeer) {
            cout << "device#" << i << " ";
        }
    }
    cout << endl;


#ifdef __HIP_PLATFORM_NVIDIA__
    // Limits:
    cout << endl;
    printLimit(w1, cudaLimitStackSize, "bytes/thread");
    printLimit(w1, cudaLimitPrintfFifoSize, "bytes/device");
    printLimit(w1, cudaLimitMallocHeapSize, "bytes/device");
    printLimit(w1, cudaLimitDevRuntimeSyncDepth, "grids");
    printLimit(w1, cudaLimitDevRuntimePendingLaunchCount, "launches");
#endif


    cout << endl;


    size_t free, total;
    checkHipErrors(hipMemGetInfo(&free, &total));

    cout << fixed << setprecision(2);
    cout << setw(w1) << "memInfo.total: " << bytesToGB(total) << " GB" << endl;
    cout << setw(w1) << "memInfo.free:  " << bytesToGB(free) << " GB (" << setprecision(0)
         << (float)free / total * 100.0 << "%)" << endl;
}

int main(int argc, char* argv[]) {
    using namespace std;

    cout << endl;

    printCompilerInfo();

    int deviceCnt;

    checkHipErrors(hipGetDeviceCount(&deviceCnt));

    for (int i = 0; i < deviceCnt; i++) {
        checkHipErrors(hipSetDevice(i));
        printDeviceProp(i);
    }

    std::cout << std::endl;
}
