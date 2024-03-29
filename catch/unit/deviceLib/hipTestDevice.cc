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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip/math_functions.h>

#define N 512
#define SIZE (N * sizeof(float))

__global__ void test_sincosf(float* a, float* b, float* c) {
    int tid = threadIdx.x;
    sincosf(a[tid], b + tid, c + tid);
}

__global__ void test_sincospif(float* a, float* b, float* c) {
    int tid = threadIdx.x;
    sincospif(a[tid], b + tid, c + tid);
}

__global__ void test_fdividef(float* a, float* b, float* c) {
    int tid = threadIdx.x;
    c[tid] = fdividef(a[tid], b[tid]);
}

__global__ void test_llrintf(float* a, int64_t* b) {
    int tid = threadIdx.x;
    b[tid] = llrintf(a[tid]);
}

__global__ void test_lrintf(float* a, int64_t* b) {
    int tid = threadIdx.x;
    b[tid] = lrintf(a[tid]);
}

__global__ void test_rintf(float* a, float* b) {
    int tid = threadIdx.x;
    b[tid] = rintf(a[tid]);
}

__global__ void test_llroundf(float* a, int64_t* b) {
    int tid = threadIdx.x;
    b[tid] = llroundf(a[tid]);
}

__global__ void test_lroundf(float* a, int64_t* b) {
    int tid = threadIdx.x;
    b[tid] = lroundf(a[tid]);
}

__global__ void test_rhypotf(float* a, float* b, float* c) {
    int tid = threadIdx.x;
    c[tid] = rhypotf(a[tid], b[tid]);
}

__global__ void test_norm3df(float* a, float* b, float* c, float* d) {
    int tid = threadIdx.x;
    d[tid] = norm3df(a[tid], b[tid], c[tid]);
}

__global__ void test_norm4df(float* a, float* b, float* c, float* d, float* e) {
    int tid = threadIdx.x;
    e[tid] = norm4df(a[tid], b[tid], c[tid], d[tid]);
}

__global__ void test_normf(float* a, float* b) {
    int tid = threadIdx.x;
    b[tid] = normf(N, a);
}

__global__ void test_rnorm3df(float* a, float* b, float* c, float* d) {
    int tid = threadIdx.x;
    d[tid] = rnorm3df(a[tid], b[tid], c[tid]);
}

__global__ void test_rnorm4df(float* a, float* b, float* c, float* d,
                              float* e) {
    int tid = threadIdx.x;
    e[tid] = rnorm4df(a[tid], b[tid], c[tid], d[tid]);
}

__global__ void test_rnormf(float* a, float* b) {
    int tid = threadIdx.x;
    b[tid] = rnormf(N, a);
}

__global__ void test_erfinvf(float* a, float* b) {
    int tid = threadIdx.x;
    b[tid] = erff(erfinvf(a[tid]));
}


bool run_sincosf() {
    float *A, *Ad, *B, *C, *Bd, *Cd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_sincosf, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
    HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (B[i] == sinf(1.0f)) {
            passed = 1;
        }
    }
    passed = 0;
    for (int i = 0; i < 512; i++) {
        if (C[i] == cosf(1.0f)) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_sincospif() {
    float *A, *Ad, *B, *C, *Bd, *Cd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_sincospif, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
    HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (B[i] - sinf(3.14 * 1.0f) < 0.1) {
            passed = 1;
        }
    }
    passed = 0;
    for (int i = 0; i < 512; i++) {
        if (C[i] - cosf(3.14 * 1.0f) < 0.1) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_fdividef() {
    float *A, *Ad, *B, *C, *Bd, *Cd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_fdividef, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
    HIP_CHECK(hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (C[i] == A[i] / B[i]) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_llrintf() {
    float *A, *Ad;
    int64_t *B, *Bd;
    A = new float[N];
    B = new int64_t[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.345f;
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), N * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_llrintf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    HIP_CHECK(hipMemcpy(B, Bd, N * sizeof(int64_t), hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        int x = roundf(A[i]);
        if (B[i] == x) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_lrintf() {
    float *A, *Ad;
    int64_t *B, *Bd;
    A = new float[N];
    B = new int64_t[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.345f;
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), N * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_lrintf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    HIP_CHECK(hipMemcpy(B, Bd, N * sizeof(int64_t), hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        int x = roundf(A[i]);
        if (B[i] == x) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_rintf() {
    float *A, *Ad;
    float *B, *Bd;
    A = new float[N];
    B = new float[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.345f;
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_rintf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        float x = roundf(A[i]);
        if (B[i] == x) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_llroundf() {
    float *A, *Ad;
    int64_t *B, *Bd;
    A = new float[N];
    B = new int64_t[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.345f;
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), N * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_llroundf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    HIP_CHECK(hipMemcpy(B, Bd, N * sizeof(int64_t), hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        int x = roundf(A[i]);
        if (B[i] == x) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_lroundf() {
    float *A, *Ad;
    int64_t *B, *Bd;
    A = new float[N];
    B = new int64_t[N];
    for (int i = 0; i < N; i++) {
        A[i] = 1.345f;
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), N * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_lroundf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    HIP_CHECK(hipMemcpy(B, Bd, N * sizeof(int64_t), hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        int x = roundf(A[i]);
        if (B[i] == x) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_norm3df() {
    float *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
    }
    val = sqrtf(1.0f + 4.0f + 9.0f);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Dd), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_norm3df, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd);
    HIP_CHECK(hipMemcpy(D, Dd, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (D[i] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));
    HIP_CHECK(hipFree(Dd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_norm4df() {
    float *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd, *E, *Ed;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[N];
    E = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
        D[i] = 4.0f;
    }
    val = sqrtf(1.0f + 4.0f + 9.0f + 16.0f);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Dd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ed), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Dd, D, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_norm4df, dim3(1), dim3(N), 0, 0, Ad,
                       Bd, Cd, Dd, Ed);
    HIP_CHECK(hipMemcpy(E, Ed, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (E[i] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    delete[] E;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));
    HIP_CHECK(hipFree(Dd));
    HIP_CHECK(hipFree(Ed));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_normf() {
    float *A, *Ad, *B, *Bd;
    A = new float[N];
    B = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 0.0f;
        val += 1.0f;
    }
    val = sqrtf(val);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_normf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (B[0] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_rhypotf() {
    float *A, *Ad, *B, *Bd, *C, *Cd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    val = 1 / sqrtf(1.0f + 4.0f);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_rhypotf, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd);
    HIP_CHECK(hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (C[i] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_rnorm3df() {
    float *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
    }
    val = 1 / sqrtf(1.0f + 4.0f + 9.0f);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Dd), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_rnorm3df, dim3(1), dim3(N), 0, 0, Ad, Bd, Cd, Dd);
    HIP_CHECK(hipMemcpy(D, Dd, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (D[i] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));
    HIP_CHECK(hipFree(Dd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_rnorm4df() {
    float *A, *Ad, *B, *Bd, *C, *Cd, *D, *Dd, *E, *Ed;
    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[N];
    E = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 3.0f;
        D[i] = 4.0f;
    }
    val = 1 / sqrtf(1.0f + 4.0f + 9.0f + 16.0f);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Cd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Dd), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ed), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Cd, C, SIZE, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Dd, D, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_rnorm4df, dim3(1), dim3(N), 0, 0, Ad,
                       Bd, Cd, Dd, Ed);
    HIP_CHECK(hipMemcpy(E, Ed, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (E[i] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    delete[] E;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));
    HIP_CHECK(hipFree(Cd));
    HIP_CHECK(hipFree(Dd));
    HIP_CHECK(hipFree(Ed));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_rnormf() {
    float *A, *Ad, *B, *Bd;
    A = new float[N];
    B = new float[N];
    float val = 0.0f;
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 0.0f;
        val += 1.0f;
    }
    val = 1 / sqrtf(val);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_rnormf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (B[0] - val < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

bool run_erfinvf() {
    float *A, *Ad, *B, *Bd;
    A = new float[N];
    B = new float[N];
    for (int i = 0; i < N; i++) {
        A[i] = -0.6f;
        B[i] = 0.0f;
    }
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
    HIP_CHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(test_erfinvf, dim3(1), dim3(N), 0, 0, Ad, Bd);
    HIP_CHECK(hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost));
    int passed = 0;
    for (int i = 0; i < 512; i++) {
        if (B[i] - A[i] < 0.000001) {
            passed = 1;
        }
    }

    delete[] A;
    delete[] B;
    HIP_CHECK(hipFree(Ad));
    HIP_CHECK(hipFree(Bd));

    if (passed == 1) {
      return true;
    } else {
      return false;
    }
}

TEST_CASE("Unit_hipDeviceTrigFunc_Float") {
  bool result = false;
  result = run_sincosf() && run_sincospif() && run_fdividef() &&
           run_llrintf() && run_norm3df() && run_norm4df() &&
           run_normf() && run_rnorm3df() && run_rnorm4df() &&
           run_rnormf() && run_lroundf() && run_llroundf() &&
           run_rintf() && run_rhypotf() && run_erfinvf();
  REQUIRE(result == true);
}
