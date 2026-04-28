#pragma once
//
// End-to-end inference pipeline for YOLOv8.
//   - load weights from .pt
//   - letterbox → forward → NMS → unscale → render
//

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#include "yolocpp/inference/nms.hpp"
#include "yolocpp/models/yolov8.hpp"

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
            models::YoloV8Scale scale  = models::kYoloV8n);

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

  models::YoloV8Detect& model() { return model_; }
  const torch::Device&  device() const { return device_; }

 private:
  models::YoloV8Detect model_;
  torch::Device        device_;
  int                  imgsz_;
};

// 80 standard COCO names (in Ultralytics order).
const std::vector<std::string>& coco_names();

// Predict with a YOLOv5 (anchorless v5u) model. Same return shape as
// Predictor::predict_to_file but uses YoloV5Detect under the hood.
std::vector<Detection> predict_v5_to_file(
    const std::string& weights, const std::string& in_path,
    const std::string& out_path, int imgsz = 640,
    const std::string& device = "", int nc = 80,
    models::YoloV8Scale scale = models::kYoloV8n,
    NMSConfig conf = {});

}  // namespace yolocpp::inference
