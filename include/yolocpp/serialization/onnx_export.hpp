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

#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo11_tasks.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo26_tasks.hpp"
#include "yolocpp/models/yolo8.hpp"
#include "yolocpp/models/yolo8_classify.hpp"
#include "yolocpp/models/yolo8_tasks.hpp"

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

// Export the given YOLO11 model to <path> as a single .onnx file. Same
// output contract as export_yolo8_onnx — single tensor [N, 4 + nc, A] in
// input pixel coords with sigmoided class scores.
void export_yolo11_onnx(models::Yolo11Detect&  model,
                         const std::string&    path,
                         const OnnxExportConfig& cfg = {});

// Export the given YOLO26 (DFL-free) model. Same output contract as
// export_yolo8_onnx / export_yolo11_onnx — single tensor [N, 4 + nc, A]
// with sigmoided class scores. Box decode in the graph uses Softplus(box)
// directly (no DFL projection).
void export_yolo26_onnx(models::Yolo26Detect&  model,
                         const std::string&    path,
                         const OnnxExportConfig& cfg = {});

// ─── Classify exporters ───────────────────────────────────────────────────
// Output: single tensor [N, nc] of pre-softmax class logits.
void export_yolo8_classify_onnx(models::Yolo8Classify&  model,
                                const std::string&     path,
                                const OnnxExportConfig& cfg = {});
void export_yolo11_classify_onnx(models::Yolo11Classify& model,
                                 const std::string&     path,
                                 const OnnxExportConfig& cfg = {});
void export_yolo26_classify_onnx(models::Yolo26Classify& model,
                                 const std::string&     path,
                                 const OnnxExportConfig& cfg = {});

// ─── Segment exporters ────────────────────────────────────────────────────
// Three outputs:
//   "output"  [N, 4 + nc, A]   xyxy + sigmoided cls (same as detect)
//   "coefs"   [N, nm, A]       mask coefficients
//   "protos"  [N, nm, h_p, w_p] mask prototypes from P3
void export_yolo8_segment_onnx(models::Yolo8Segment&  model,
                                const std::string&     path,
                                const OnnxExportConfig& cfg = {});
void export_yolo11_segment_onnx(models::Yolo11Segment& model,
                                 const std::string&     path,
                                 const OnnxExportConfig& cfg = {});
void export_yolo26_segment_onnx(models::Yolo26Segment& model,
                                 const std::string&     path,
                                 const OnnxExportConfig& cfg = {});

// ─── Pose exporters ───────────────────────────────────────────────────────
// Two outputs:
//   "output"     [N, 4 + nc, A]
//   "keypoints"  [N, num_kpts*kpt_dim, A]  decoded (xy in pixels, conf sigmoid)
// For v26 the cv4 emits an extra σ branch (`nk_sigma`) which we slice off
// inside the graph — keypoint output stays [N, num_kpts*kpt_dim, A].
void export_yolo8_pose_onnx(models::Yolo8Pose&  model,
                             const std::string& path,
                             const OnnxExportConfig& cfg = {});
void export_yolo11_pose_onnx(models::Yolo11Pose& model,
                              const std::string& path,
                              const OnnxExportConfig& cfg = {});
void export_yolo26_pose_onnx(models::Yolo26Pose& model,
                              const std::string& path,
                              const OnnxExportConfig& cfg = {});

// ─── OBB exporters ────────────────────────────────────────────────────────
// Two outputs:
//   "output" [N, 4 + nc, A]
//   "angle"  [N, A]           rotation angle in radians, ∈ [-π/4, 3π/4]
void export_yolo8_obb_onnx(models::Yolo8OBB&  model,
                            const std::string& path,
                            const OnnxExportConfig& cfg = {});
void export_yolo11_obb_onnx(models::Yolo11OBB& model,
                             const std::string& path,
                             const OnnxExportConfig& cfg = {});
void export_yolo26_obb_onnx(models::Yolo26OBB& model,
                             const std::string& path,
                             const OnnxExportConfig& cfg = {});

}  // namespace yolocpp::serialization
