#include <hip_test_common.hh>

static __global__ void add_kernel(float* a, float* b, float* res, size_t size) {
  int i = threadIdx.x;
  if (i < size) {
    res[i] = a[i] + b[i];
  }
}

TEST_CASE("Unit_hipEventRecord_on_hipStreamLegacy") {
  constexpr size_t size = 32;
  hipEvent_t start, end;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&end));

  float *d_a, *d_b, *d_res;
  HIP_CHECK(hipMalloc(&d_a, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_b, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&d_res, sizeof(float) * size));

  std::vector<float> a, b, gpu_res;
  a.reserve(size);
  b.reserve(size);
  gpu_res.reserve(size);
  for (size_t i = 0; i < size; i++) {
    a.push_back(i + 1.1f);
    b.push_back((i + 1.1f) * 2.0f);
    gpu_res.push_back(0.0f);
  }

  HIP_CHECK(hipEventRecord(start, hipStreamLegacy));
  HIP_CHECK(
      hipMemcpyAsync(d_a, a.data(), sizeof(float) * size, hipMemcpyHostToDevice, hipStreamLegacy));
  HIP_CHECK(
      hipMemcpyAsync(d_b, b.data(), sizeof(float) * size, hipMemcpyHostToDevice, hipStreamLegacy));
  add_kernel<<<1, size, 0, hipStreamLegacy>>>(d_a, d_b, d_res, size);
  HIP_CHECK(hipMemcpyAsync(gpu_res.data(), d_res, sizeof(float) * size, hipMemcpyDeviceToHost,
                           hipStreamLegacy));
  HIP_CHECK(hipEventRecord(end, hipStreamLegacy));

  HIP_CHECK(
      hipEventSynchronize(start));  // Although that event might be complete, just do it for checks
  HIP_CHECK(hipEventSynchronize(end));

  float time = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&time, start, end));
  INFO("Time: " << time);
  CHECK(time > 0.0f);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(end));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_res));
}
