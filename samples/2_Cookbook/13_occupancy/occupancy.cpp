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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime.h"
#include <iostream>
#include "hip_helper.h"
#define NUM 1000000

// Device (Kernel) function
__global__ void multiply(float* C, float* A, float* B, int N){

    int tx = blockDim.x*blockIdx.x+threadIdx.x;

    if (tx < N){
	C[tx] = A[tx] * B[tx];
    }
}
// CPU implementation
void multiplyCPU(float* C, float* A, float* B, int N){

    for(unsigned int i=0; i<N; i++){
        C[i] = A[i] * B[i];
    }
}

void launchKernel(float* C, float* A, float* B, bool manual){

     hipDeviceProp_t devProp;
     checkHipErrors(hipGetDeviceProperties(&devProp, 0));

     hipEvent_t start, stop;
     checkHipErrors(hipEventCreate(&start));
     checkHipErrors(hipEventCreate(&stop));
     float eventMs = 1.0f;
     const unsigned threadsperblock = 32;
     const unsigned blocks = (NUM/threadsperblock)+1;

     int mingridSize = 0;
     int gridSize = 0;
     int blockSize = 0;

     if (manual){
	blockSize = threadsperblock;
	gridSize  = blocks;
	std::cout << std::endl << "Manual Configuration with block size " << blockSize << std::endl;
     }
     else{
	checkHipErrors(hipOccupancyMaxPotentialBlockSize(&mingridSize, &blockSize, multiply, 0, 0));
	std::cout << std::endl << "Automatic Configuation based on hipOccupancyMaxPotentialBlockSize " << std::endl;
	std::cout << "Suggested blocksize is " << blockSize << ", Minimum gridsize is " << mingridSize << std::endl;
	gridSize = (NUM/blockSize)+1;
     }

     // Record the start event
     checkHipErrors(hipEventRecord(start, NULL));

     // Launching the Kernel from Host
     hipLaunchKernelGGL(multiply, dim3(gridSize), dim3(blockSize), 0, 0, C, A, B, NUM);

     // Record the stop event
     checkHipErrors(hipEventRecord(stop, NULL));
     checkHipErrors(hipEventSynchronize(stop));

     checkHipErrors(hipEventElapsedTime(&eventMs, start, stop));
     printf("kernel Execution time = %6.3fms\n", eventMs);

     checkHipErrors(hipEventDestroy(start));
     checkHipErrors(hipEventDestroy(stop));

     //Calculate Occupancy
     int numBlock = 0;
     checkHipErrors(hipOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock, multiply, blockSize, 0));

     if(devProp.maxThreadsPerMultiProcessor){
	std::cout << "Theoretical Occupancy is " << (double)numBlock* blockSize/devProp.maxThreadsPerMultiProcessor * 100 << "%" << std::endl;
     }
}

int main() {
     float *A, *B, *C0, *C1, *cpuC;
     float *Ad, *Bd, *C0d, *C1d;
     int errors=0;
     int i;

     // initialize the input data
     A  = (float *)malloc(NUM * sizeof(float));
     B  = (float *)malloc(NUM * sizeof(float));
     C0 = (float *)malloc(NUM * sizeof(float));
     C1 = (float *)malloc(NUM * sizeof(float));
     cpuC = (float *)malloc(NUM * sizeof(float));

     for(i=0; i< NUM; i++){
	A[i] = i;
	B[i] = i;
     }

     // allocate the memory on the device side
     checkHipErrors(hipMalloc((void**)&Ad, NUM * sizeof(float)));
     checkHipErrors(hipMalloc((void**)&Bd, NUM * sizeof(float)));
     checkHipErrors(hipMalloc((void**)&C0d, NUM * sizeof(float)));
     checkHipErrors(hipMalloc((void**)&C1d, NUM * sizeof(float)));

     // Memory transfer from host to device
     checkHipErrors(hipMemcpy(Ad,A,NUM * sizeof(float), hipMemcpyHostToDevice));
     checkHipErrors(hipMemcpy(Bd,B,NUM * sizeof(float), hipMemcpyHostToDevice));

     //Kernel launch with manual/default block size
     launchKernel(C0d, Ad, Bd, 1);

     //Kernel launch with the block size suggested by hipOccupancyMaxPotentialBlockSize
     launchKernel(C1d, Ad, Bd, 0);

     // Memory transfer from device to host
     checkHipErrors(hipMemcpy(C0,C0d, NUM * sizeof(float), hipMemcpyDeviceToHost));
     checkHipErrors(hipMemcpy(C1,C1d, NUM * sizeof(float), hipMemcpyDeviceToHost));

     // CPU computation
     multiplyCPU(cpuC, A, B, NUM);

     //verify the results
     double eps = 1.0E-6;

       for (i = 0; i < NUM; i++) {
	  if (std::abs(C0[i] - cpuC[i]) > eps) {
		  errors++;
	}
     }

     if (errors != 0){
	printf("\nManual Test FAILED: %d errors\n", errors);
	errors=0;
     } else {
	printf("\nManual Test PASSED!\n");
     }

     for (i = 0; i < NUM; i++) {
	  if (std::abs(C1[i] - cpuC[i]) > eps) {
		  errors++;
	}
     }

     if (errors != 0){
	printf("\n Automatic Test FAILED: %d errors\n", errors);
     } else {
	printf("\nAutomatic Test PASSED!\n");
     }

     checkHipErrors(hipFree(Ad));
     checkHipErrors(hipFree(Bd));
     checkHipErrors(hipFree(C0d));
     checkHipErrors(hipFree(C1d));

     free(A);
     free(B);
     free(C0);
     free(C1);
     free(cpuC);
     return 0;
}
