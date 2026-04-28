#pragma once
//
// Convert an .onnx file to a serialized TensorRT engine plan (.trt).
//
// Uses TRT's nvonnxparser to parse the graph and INetworkDefinition to build
// the engine on the local GPU (Blackwell sm_120 in our setup).
//

#include <string>

namespace yolocpp::serialization {

struct TrtBuildConfig {
  // Workspace / tactic memory limit (bytes).
  std::size_t workspace_bytes = 1ull << 30;     // 1 GiB
  // Optional FP16 precision (reduces engine size & latency on RTX 5090).
  bool fp16 = true;
  // Builder optimization level (0..5; higher = longer build, smaller plan).
  int  builder_opt_level = 3;
  // Static shape used for the optimization profile (single profile).
  int  imgsz      = 640;
  int  batch_min  = 1;
  int  batch_opt  = 1;
  int  batch_max  = 1;
  // ONNX input tensor name; must match what the exporter wrote.
  std::string input_name = "images";
};

void build_trt_engine(const std::string& onnx_path,
                      const std::string& engine_path,
                      const TrtBuildConfig& cfg = {});

}  // namespace yolocpp::serialization
