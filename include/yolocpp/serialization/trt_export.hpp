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
  // Optional INT8 PTQ — requires `calib_image_dir` to be set so the
  // builder can sample N images and produce per-tensor scales. On
  // Blackwell the INT8 tensor cores yield ~1.5× over FP16; calibration
  // is one-shot at engine-build time.
  bool int8 = false;
  // Directory of .jpg/.png images used to calibrate INT8 ranges
  // (typically val split, ~100-500 images is plenty). When `int8`
  // is true and this is empty, the build will throw.
  std::string calib_image_dir = "";
  // Cache file for INT8 calibration tables (so repeated builds at the
  // same shape don't re-sample). Empty = no cache (always recalibrate).
  std::string calib_cache = "";
  // TF32 tensor-core math for FP32 builds. Defaults on (TRT default).
  // Disable for v10 s/m/b/l/x: the deeper RepVGGDW (7×7 dwconv with bias)
  // stack accumulates enough TF32 mantissa loss to saturate the cls
  // outputs near-zero; v10n is shallow enough not to trigger it.
  bool tf32 = true;
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
