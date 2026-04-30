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
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo8.hpp"

namespace yolocpp::inference {

struct Detection {
  float x1, y1, x2, y2;
  float conf;
  int   cls;
};

class Predictor {
 public:
  // weights: path to Ultralytics .pt file
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

// 80 standard COCO names (in Ultralytics order).
const std::vector<std::string>& coco_names();

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

}  // namespace yolocpp::inference
