#pragma once
#include <string>

namespace yolocpp {

struct BuildInfo {
  std::string yolocpp_version;
  std::string libtorch_version;
  std::string cuda_toolkit_version;
  std::string cuda_runtime_version;
  std::string cudnn_version;
  std::string tensorrt_version;
  std::string opencv_version;
};

BuildInfo build_info();

}  // namespace yolocpp
