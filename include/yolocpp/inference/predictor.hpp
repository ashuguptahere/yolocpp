#pragma once
//
// End-to-end inference pipeline for YOLO8.
//   - load weights from .pt
//   - letterbox → forward → NMS → unscale → render
//

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#include "yolocpp/inference/nms.hpp"
#include "yolocpp/models/yolo1.hpp"
#include "yolocpp/models/yolo2.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo12.hpp"
#include "yolocpp/models/yolo13.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo3.hpp"
#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/models/yolo6.hpp"
#include "yolocpp/models/yolo7.hpp"
#include "yolocpp/models/yolo8.hpp"
#include "yolocpp/models/yolo9.hpp"
#include "yolocpp/models/yolo10.hpp"

namespace yolocpp::inference {

struct Detection {
  float x1, y1, x2, y2;
  float conf;
  int   cls;
};

class Predictor {
 public:
  // weights: path to an upstream `.pt` file (or our converted form)
  // imgsz:   letterbox target side (default 640)
  // device:  "cuda" or "cpu" (auto-detected if empty)
  // nc:      number of classes the model was trained on (must match weights)
  Predictor(const std::string& weights,
            int                imgsz   = 640,
            std::string        device  = "",
            int                nc      = 80,
            models::Yolo8Scale scale  = models::kYolo8n);

  // Run inference on a single image. Returns detections in original-image
  // coordinates.
  std::vector<Detection> predict(const cv::Mat& bgr,
                                 NMSConfig conf = {}) const;

  // Convenience: read image from disk, predict, and write annotated image.
  // Returns the detections.
  std::vector<Detection> predict_to_file(const std::string& in_path,
                                         const std::string& out_path,
                                         NMSConfig conf = {},
                                         const std::vector<std::string>& names = {}) const;

  models::Yolo8Detect& model() { return model_; }
  const torch::Device&  device() const { return device_; }

 private:
  models::Yolo8Detect model_;
  torch::Device        device_;
  int                  imgsz_;
};

// 80 standard COCO names (canonical YOLO ordering).
const std::vector<std::string>& coco_names();

// Render bounding boxes + class/conf labels onto `img` in-place. Each
// class gets a deterministic colour. `names` is optional — empty
// means render the integer class index. Used by both image-mode
// predict (`predict_to_file`) and the video-mode frame loop in the
// CLI (#51C2).
void draw_detections(cv::Mat& img,
                     const std::vector<Detection>& dets,
                     const std::vector<std::string>& names = {});

// Predict with a YOLO1 (Redmon 2016, 24-conv + 2-FC, no BN). Weights
// must be in our `.pt` form — produced by
// `serialization::convert_yolov1_weights(yolov1.weights, yolo1.pt)`
// from pjreddie's Darknet release. Default imgsz=448 matches the
// upstream training config.
std::vector<Detection> predict_v1_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 448,
    const std::string& device = "", int nc = 20,
    NMSConfig conf = {});

// Predict with a YOLO2 (Darknet-19 + reorg + region) model. Weights
// must be in our `.pt` form — produced by
// `serialization::convert_yolov2_weights(yolov2.weights, yolo2.pt)`.
// Default imgsz=416 matches the upstream cfg; the model is fully-
// convolutional so any multiple of 32 works. `scale=Tiny` selects
// the 9-conv compact variant.
std::vector<Detection> predict_v2_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 416,
    const std::string& device = "", int nc = 20,
    models::Yolo2Scale scale = models::Yolo2Scale::Full,
    NMSConfig conf = {});

// Predict with a YOLO3 (the anchor-free yolov3u variant — Darknet-53
// backbone + v8-style DFL Detect head). Weights are converted from
// the upstream `yolov3u.pt` via `convert_yolov3_pt` (fp16 → fp32).
std::vector<Detection> predict_v3_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    NMSConfig conf = {});

// Predict with a YOLO4 (CSPDarknet53 + SPP + PANet, anchor-based) model.
// Weights must be in our `.pt` form — produced by
// `serialization::convert_yolov4_weights(yolov4.weights, yolo4.pt)` from
// AlexeyAB's Darknet release. The default imgsz=608 matches yolov4.cfg's
// `width`/`height` (the anchors are calibrated to that size; we rescale
// them automatically when imgsz differs).
std::vector<Detection> predict_v4_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 608,
    const std::string& device = "", int nc = 80,
    NMSConfig conf = {});

// Predict with a YOLO6 (Meituan v3.0 deploy form) model. Weights must be
// in our `.pt` form — produced by `convert_yolov6_pt(yolov6s.pt, yolo6s.pt)`
// from Meituan's release. The resolver auto-runs the conversion when the
// upstream `.pt` is found in `data/` or the cache.
// v6 P6 variants (yolov6{n,s,m,l}6.pt) need `p6=true` to construct the
// 6-stage backbone + 4-level head and default to `imgsz=1280`.
std::vector<Detection> predict_v6_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    models::Yolo6Scale scale = models::kYolo6s,
    bool p6 = false,
    NMSConfig conf = {});

// Predict with a YOLO7 (WongKinYiu, anchor-based IDetect) model. Weights
// must be in our `.pt` form — produced by
// `convert_yolov7_pt(yolov7.pt → yolo7.pt)` (RepConv fusion + fp16→fp32);
// the resolver auto-runs the conversion.
std::vector<Detection> predict_v7_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    models::Yolo7Scale scale = models::Yolo7Scale::Base,
    NMSConfig conf = {});

// Predict with a YOLO9 (yolov9c, GELAN backbone + v8-style anchor-free
// Detect head) model. Weights must be in our `.pt` form — produced by
// `convert_yolov9_pt(yolov9c.pt → yolo9.pt)` (RepConv fusion + fp16→fp32);
// the resolver auto-runs the conversion.
std::vector<Detection> predict_v9_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    models::Yolo9Scale scale = models::Yolo9Scale::C,
    NMSConfig conf = {});

// Predict with a YOLO10 (Tsinghua MIG, end-to-end / NMS-free) model.
// Weights must be in our `.pt` form — produced by
// `convert_yolov10_pt(yolov10n.pt → yolo10.pt)` (one2many head dropped,
// one2one renamed to cv2/cv3, RepVGGDW fusion + fp16→fp32). The
// resolver auto-runs the conversion.
std::vector<Detection> predict_v10_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    models::Yolo10Scale scale = models::kYolo10n,
    NMSConfig conf = {});

// Predict with a YOLO5 (anchorless v5u) model. Same return shape as
// Predictor::predict_to_file but uses Yolo5Detect under the hood.
std::vector<Detection> predict_v5_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    models::Yolo8Scale scale = models::kYolo8n,
    NMSConfig conf = {});

// Predict with a YOLO11 detection model. Filename convention is `yolo11<scale>.pt`
// (no 'v'). Pass scale explicitly via Yolo11Scale, or use yolo11_scale_from_filename.
std::vector<Detection> predict_v11_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    models::Yolo11Scale scale = models::kYolo11n,
    NMSConfig conf = {});

// Predict with a YOLO26 (DFL-free) detection model. Filename convention is
// `yolo26<scale>.pt` (no 'v'). NMS is still applied at runtime — the spec'd
// NMS-free dual-head training isn't part of this predict path.
std::vector<Detection> predict_v26_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    models::Yolo26Scale scale = models::kYolo26n,
    NMSConfig conf = {});

// Predict with a YOLO12 (Tian et al., A2C2f-based) detection model.
// Filename convention `yolo12<scale>.pt`. All 5 scales available
// from the upstream v8.3.0+ asset host.
std::vector<Detection> predict_v12_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    models::Yolo12Scale scale = models::kYolo12n,
    NMSConfig conf = {});

// Predict with a YOLO13 (iMoonLab fork: HyperACE / FullPAD) detection
// model. Filename convention `yolo13<scale>.pt`; only n/s/l/x ship.
std::vector<Detection> predict_v13_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    models::Yolo13Scale scale = models::kYolo13n,
    NMSConfig conf = {});

}  // namespace yolocpp::inference
