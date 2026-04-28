#include "yolocpp/core/device.hpp"

#include <cuda_runtime.h>
#include <torch/cuda.h>

namespace yolocpp {

bool cuda_available() {
  return torch::cuda::is_available();
}

std::vector<DeviceInfo> list_cuda_devices() {
  std::vector<DeviceInfo> out;
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return out;
  out.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
    DeviceInfo d;
    d.index                    = i;
    d.name                     = p.name;
    d.compute_capability_major = p.major;
    d.compute_capability_minor = p.minor;
    d.total_memory_bytes       = p.totalGlobalMem;
    d.multi_processor_count    = p.multiProcessorCount;
    out.push_back(std::move(d));
  }
  return out;
}

}  // namespace yolocpp
