#include <cstdio>
#include <cuda_runtime.h>

namespace yolocpp_smoke {

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

cudaError_t launch_add(const float* a, const float* b, float* c, int n) {
  const int threads = 256;
  const int blocks  = (n + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(a, b, c, n);
  return cudaGetLastError();
}

}  // namespace yolocpp_smoke
