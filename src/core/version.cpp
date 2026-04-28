#include "yolocpp/core/version.hpp"

#include <torch/version.h>
#include <ATen/Version.h>
#include <cuda_runtime.h>
#include <NvInferVersion.h>
#include <opencv2/core/version.hpp>

#include <yolocpp/config.hpp>

#include <cstdio>
#include <string>

namespace yolocpp {

namespace {
std::string format_cuda_runtime() {
  int v = 0;
  if (cudaRuntimeGetVersion(&v) != cudaSuccess) return "unknown";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%d.%d", v / 1000, (v % 1000) / 10);
  return buf;
}
}  // namespace

BuildInfo build_info() {
  BuildInfo info;
  info.yolocpp_version       = YOLOCPP_VERSION_STRING;
  info.libtorch_version      = TORCH_VERSION;
  info.cuda_toolkit_version  = YOLOCPP_CUDA_TOOLKIT_VERSION;
  info.cuda_runtime_version  = format_cuda_runtime();
#ifdef CUDNN_VERSION
  info.cudnn_version         = std::to_string(CUDNN_VERSION);
#else
  info.cudnn_version         = "(not linked at compile time)";
#endif
  info.tensorrt_version      = std::to_string(NV_TENSORRT_MAJOR) + "." +
                               std::to_string(NV_TENSORRT_MINOR) + "." +
                               std::to_string(NV_TENSORRT_PATCH) + "." +
                               std::to_string(NV_TENSORRT_BUILD);
  info.opencv_version        = CV_VERSION;
  return info;
}

}  // namespace yolocpp
