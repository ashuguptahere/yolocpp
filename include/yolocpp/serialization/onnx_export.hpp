#pragma once
//
// Self-contained ONNX exporter for YOLO8 detection models.
//
// Writes the ONNX protobuf wire format directly — no protobuf library, no
// Python, no TorchScript tracing. Tested against TRT 10's nvonnxparser and
// onnx-runtime parsers (see tests/test_onnx_export.cpp).
//
// The exported graph computes the *training-mode* per-level feature outputs
// concatenated and reshaped, plus a Detect-head decoder so that the single
// network output is [N, 4 + nc, A] in input pixels with sigmoided class
// scores — i.e. the same eval-mode tensor that Predictor expects. NMS is
// kept in the host runtime (out of the ONNX graph) for portability.
//

#include <string>

#include "yolocpp/models/yolo8.hpp"

namespace yolocpp::serialization {

struct OnnxExportConfig {
  int  imgsz = 640;     // square input
  bool fold_bn = true;  // fuse BN into Conv weights/bias
  // Output naming
  std::string input_name  = "images";
  std::string output_name = "output";
  // Producer info (set in ModelProto)
  std::string producer_name    = "yolocpp";
  std::string producer_version = "0.1.0";
  // Opset version. ONNX opset 17 = available since onnx-1.13 / TRT 10+.
  int  opset_version = 17;
};

// Export the given YOLO8 model to <path> as a single .onnx file.
//   - The model must be in eval() mode (BN running stats stable).
//   - Strides are read from model.stride (set during construction).
// Throws std::runtime_error on any error.
void export_yolo8_onnx(models::Yolo8Detect& model,
                        const std::string&    path,
                        const OnnxExportConfig& cfg = {});

}  // namespace yolocpp::serialization
