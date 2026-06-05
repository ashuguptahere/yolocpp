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

#include "yolocpp/models/yolo1.hpp"
#include "yolocpp/models/yolo2.hpp"
#include "yolocpp/models/yolo10.hpp"
#include "yolocpp/models/yolo3.hpp"
#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/models/yolo5.hpp"
#include "yolocpp/models/yolo6.hpp"
#include "yolocpp/models/yolo7.hpp"
#include "yolocpp/models/yolo9.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo11_tasks.hpp"
#include "yolocpp/models/yolo12.hpp"
#include "yolocpp/models/yolo13.hpp"
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

// Export the given YOLO12 (Tian et al. — A2C2f / AAttn) model. Same
// output contract: [N, 4 + nc, A]. Graph emits the per-head interleaved
// qkv split, area-windowing reshape (when area>1), gamma-gated outer
// residual at l/x scales, and the legacy=False Detect head.
void export_yolo12_onnx(models::Yolo12Detect& model,
                         const std::string&    path,
                         const OnnxExportConfig& cfg = {});

// Export the given YOLO13 (iMoonLab — HyperACE / FullPAD) model. Same
// output contract: [N, 4 + nc, A]. Graph emits DSConv (depthwise →
// pointwise → BN-fused), V13AAttn (separate qk/v convs, k=5 pe applied
// to v), HyperACE (FuseModule + 2× C3AH with hypergraph attention +
// inner DSC3k chain), the five FullPAD_Tunnel gated residuals, and
// Detect (legacy=False).
void export_yolo13_onnx(models::Yolo13Detect& model,
                         const std::string&    path,
                         const OnnxExportConfig& cfg = {});

// Export the given YOLO3 (yolov3u — anchor-free DFL) model.
// Same output contract: [N, 4 + nc, A]. Walker emits Conv + Bottleneck +
// Upsample + Concat per the v3 yaml (29 layers); reuses emit_detect for
// the legacy=true Detect head (reg_max=16).
void export_yolo3_onnx(models::Yolo3& model,
                       const std::string&    path,
                       const OnnxExportConfig& cfg = {});

// Export the given YOLO5 (anchor-free `*u.pt`) model. Same
// output contract as export_yolo8_onnx — [N, 4 + nc, A] xyxy + sigmoided
// cls. Architecture differs from v8 in two structural pieces handled
// inside the emitter: a 6×6 stride-2 Conv stem at layer 0 and C3 blocks
// (cv1 + N×Bottleneck → cat with cv2 → cv3) instead of v8's C2f. The
// Detect head (legacy=true, anchor-free DFL) is identical to v8/v9 and
// reuses `emit_detect` directly.
void export_yolo5_onnx(models::Yolo5Detect& model,
                       const std::string&    path,
                       const OnnxExportConfig& cfg = {});

// Export the given YOLO4 (AlexeyAB Darknet) model. Output:
// [N, 4 + nc, A] xyxy + obj*cls (sigmoided) at the v4 scale_xy bias-fix
// (1.2/1.1/1.05 for P3/P4/P5) with anchors calibrated to the model's
// 608² training resolution (rescaled at runtime to actual imgsz).
// Walker emits ConvMish + ConvLeaky + CSPStage + SPPv4 + the explicit
// PANet top-down/bottom-up paths per the v4 deploy graph.
void export_yolo4_onnx(models::Yolo4& model,
                       const std::string&    path,
                       const OnnxExportConfig& cfg = {});

// Export the given YOLO6 (Meituan — EfficientRep + RepBiFPANNeck +
// EffiDeHead) model. Same output contract: [N, 4 + nc, A] xyxy +
// sigmoid'd cls. Covers all 12 published v6 variants: P5 n/s (RepBlock
// backbone + CSPSPPF), m/l (BepC3 + SimSPPF + DFL eval), the *_mbla
// scales (MBLABlock backbone+neck), and the P6 four-level variants
// (n6/s6/m6/l6 via the model->is_p6 branch).
void export_yolo6_onnx(models::Yolo6& model,
                       const std::string&    path,
                       const OnnxExportConfig& cfg = {});

// Export the given YOLO7 (WongKinYiu — ELAN + IDetect anchor head) model.
// Same output contract: [N, 4 + nc, A]. Walker emits ConvSiLU/LeakyReLU
// (per scale) + MP/SP + ReOrg + DownC + SPPCSPC + Yolo7RepConv +
// Yolo7Shortcut, then applies the IDetect 1×1 m[i] convs and decodes
// in WongKinYiu's "new coords" form: xy = (sigmoid(t)*2 - 0.5 +
// grid)*stride; wh = (sigmoid(t)*2)^2 * anchor; score = obj * cls.
// Anchors are read from the model's `anchor_grid` buffer (calibrated
// to the training resolution, rescaled to actual imgsz at runtime).
// 3-level head for base/tiny/x; 4-level for w6/e6/d6/e6e (1280²).
void export_yolo7_onnx(models::Yolo7& model,
                       const std::string&    path,
                       const OnnxExportConfig& cfg = {});

// Export the given YOLO9 (Wang/Yeh/Liao — GELAN backbone) model.
// Same output contract: [N, 4 + nc, A]. Walker emits the GELAN module
// set (RepNCSPELAN4 + ADown + AConv + ELAN1 + SPPELAN) plus CBLinear /
// CBFuse for the v9e two-pass backbone; reuses emit_detect for the
// legacy=true Detect head (reg_max=16).
void export_yolo9_onnx(models::Yolo9& model,
                       const std::string&    path,
                       const OnnxExportConfig& cfg = {});

// Export the given YOLO10 (Tsinghua MIG — NMS-free dual-head) model.
// Same output contract: [N, 4 + nc, A] xyxy + sigmoided cls. The
// graph emits the per-scale yaml walk (Conv / C2f / SCDown / C2fCIB /
// SPPF / PSA / Upsample / Concat) and reuses emit_detect_v11 for the
// one2one head (v11-style cv3 with DWConvBlock×2 + Conv2d). The
// one2many head is dropped at conversion time — not present in the
// deploy state-dict — so the graph is single-headed. Downstream
// "NMS-free" topk is left to the host runtime: v10's one2one head
// produces predictions that are already 1-per-object, so applying
// NMS at the standard IoU threshold is effectively a no-op (and is
// what our Predictor / TRT pipeline already does).
void export_yolo10_onnx(models::Yolo10& model,
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

// ─── Darknet-era exporters (#68, #69) ────────────────────────────────────
// Emit the YOLOv1 model — 24-conv backbone (no BN, leaky 0.1) + 2 FC
// layers + Darknet flat-block detection decoder. Output is the
// standard `[N, 4 + nc, A]` xyxy + class-score contract (A = S·S·B).
void export_yolo1_onnx(models::Yolo1&         model,
                        const std::string&     path,
                        const OnnxExportConfig& cfg = {});

// Emit the YOLOv2 model — Darknet-19 backbone + reorg passthrough +
// 5-anchor region head. Output is `[N, 4 + nc, A]` (A = na·H·W).
// Supports both `Full` and `Tiny` scales.
void export_yolo2_onnx(models::Yolo2&         model,
                        const std::string&     path,
                        const OnnxExportConfig& cfg = {});

}  // namespace yolocpp::serialization
