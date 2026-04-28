#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace yolocpp {

struct DeviceInfo {
  int index;
  std::string name;
  int compute_capability_major;
  int compute_capability_minor;
  std::size_t total_memory_bytes;
  int multi_processor_count;
};

bool cuda_available();
std::vector<DeviceInfo> list_cuda_devices();

}  // namespace yolocpp
