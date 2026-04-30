#pragma once
//
// Predictors for classify / segment / pose / OBB.
//
// Each owns its own task-specific model, loads weights from .pt, runs
// inference on an image, and produces a task-shaped result struct.
//

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "yolocpp/inference/predictor.hpp"   // for Detection
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo11_tasks.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo26_tasks.hpp"
#include "yolocpp/models/yolo8.hpp"
#include "yolocpp/models/yolo8_classify.hpp"
#include "yolocpp/models/yolo8_tasks.hpp"

namespace yolocpp::inference {

// ─── Classify ──────────────────────────────────────────────────────────────
struct ClassifyResult {
  std::vector<std::pair<int, float>> topk;  // (class_id, prob)
};

class ClassifyPredictor {
 public:
  // imgsz: classification typically uses 224 (Ultralytics default).
  ClassifyPredictor(const std::string& weights, int imgsz = 224,
                    std::string device = "", int nc = 1000,
                    models::Yolo8Scale scale = models::kYolo8n);

  ClassifyResult predict(const cv::Mat& bgr, int top_k = 5) const;

  models::Yolo8Classify& model() { return model_; }
  int imgsz() const { return imgsz_; }

 private:
  models::Yolo8Classify model_;
  torch::Device          device_;
  int                    imgsz_;
};

// 1000-class ImageNet name list (used for classify display).
const std::vector<std::string>& imagenet_names();

// ─── Segment ──────────────────────────────────────────────────────────────
struct SegInstance {
  Detection box;          // xyxy + conf + cls in original image coords
  cv::Mat   mask;         // CV_8U in original image dims (0/255)
};

class SegmentPredictor {
 public:
  SegmentPredictor(const std::string& weights, int imgsz = 640,
                   std::string device = "", int nc = 80,
                   models::Yolo8Scale scale = models::kYolo8n);

  std::vector<SegInstance> predict(const cv::Mat& bgr,
                                   NMSConfig conf = {}) const;
  std::vector<SegInstance> predict_to_file(const std::string& in_path,
                                           const std::string& out_path,
                                           NMSConfig conf = {},
                                           const std::vector<std::string>& names = {}) const;

  int imgsz() const { return imgsz_; }

 private:
  models::Yolo8Segment model_;
  torch::Device         device_;
  int                   imgsz_;
};

// ─── Pose ─────────────────────────────────────────────────────────────────
struct PoseInstance {
  Detection box;
  // 17 keypoints × (x, y, conf) in original image coords.
  std::vector<std::array<float, 3>> keypoints;
};

class PosePredictor {
 public:
  PosePredictor(const std::string& weights, int imgsz = 640,
                std::string device = "", int num_kpts = 17, int kpt_dim = 3,
                models::Yolo8Scale scale = models::kYolo8n);

  std::vector<PoseInstance> predict(const cv::Mat& bgr,
                                    NMSConfig conf = {}) const;
  std::vector<PoseInstance> predict_to_file(const std::string& in_path,
                                            const std::string& out_path,
                                            NMSConfig conf = {}) const;

  int imgsz() const { return imgsz_; }

 private:
  models::Yolo8Pose model_;
  torch::Device     device_;
  int               imgsz_;
  int               num_kpts_, kpt_dim_;
};

// ─── OBB ──────────────────────────────────────────────────────────────────
struct OBBInstance {
  // Center, width, height, angle (radians) in original image coords.
  float cx, cy, w, h, angle;
  float conf;
  int   cls;
};

class OBBPredictor {
 public:
  OBBPredictor(const std::string& weights, int imgsz = 1024,
               std::string device = "", int nc = 15,
               models::Yolo8Scale scale = models::kYolo8n);

  std::vector<OBBInstance> predict(const cv::Mat& bgr,
                                   NMSConfig conf = {}) const;
  std::vector<OBBInstance> predict_to_file(const std::string& in_path,
                                           const std::string& out_path,
                                           NMSConfig conf = {},
                                           const std::vector<std::string>& names = {}) const;

  int imgsz() const { return imgsz_; }

 private:
  models::Yolo8OBB model_;
  torch::Device     device_;
  int               imgsz_;
};

// 15-class DOTA names (default for OBB).
const std::vector<std::string>& dota_names();

// ─── YOLO11 task predictors ────────────────────────────────────────────────
//
// Same interfaces as the v8 versions above, but built on Yolo11Classify /
// Yolo11Segment / Yolo11Pose / Yolo11OBB. The predict bodies share helpers
// with the v8 implementations (template-dispatched on the model holder) so
// any post-processing fix benefits both.

class Yolo11ClassifyPredictor {
 public:
  Yolo11ClassifyPredictor(const std::string& weights, int imgsz = 224,
                           std::string device = "", int nc = 1000,
                           models::Yolo11Scale scale = models::kYolo11n);
  ClassifyResult predict(const cv::Mat& bgr, int top_k = 5) const;
  models::Yolo11Classify& model() { return model_; }
  int imgsz() const { return imgsz_; }
 private:
  models::Yolo11Classify model_;
  torch::Device          device_;
  int                    imgsz_;
};

class Yolo11SegmentPredictor {
 public:
  Yolo11SegmentPredictor(const std::string& weights, int imgsz = 640,
                          std::string device = "", int nc = 80,
                          models::Yolo11Scale scale = models::kYolo11n);
  std::vector<SegInstance> predict(const cv::Mat& bgr,
                                   NMSConfig conf = {}) const;
  std::vector<SegInstance> predict_to_file(const std::string& in_path,
                                           const std::string& out_path,
                                           NMSConfig conf = {},
                                           const std::vector<std::string>& names = {}) const;
  int imgsz() const { return imgsz_; }
 private:
  models::Yolo11Segment model_;
  torch::Device          device_;
  int                    imgsz_;
};

class Yolo11PosePredictor {
 public:
  Yolo11PosePredictor(const std::string& weights, int imgsz = 640,
                       std::string device = "", int num_kpts = 17, int kpt_dim = 3,
                       models::Yolo11Scale scale = models::kYolo11n);
  std::vector<PoseInstance> predict(const cv::Mat& bgr,
                                    NMSConfig conf = {}) const;
  std::vector<PoseInstance> predict_to_file(const std::string& in_path,
                                            const std::string& out_path,
                                            NMSConfig conf = {}) const;
  int imgsz() const { return imgsz_; }
 private:
  models::Yolo11Pose model_;
  torch::Device      device_;
  int                imgsz_;
  int                num_kpts_, kpt_dim_;
};

class Yolo11OBBPredictor {
 public:
  Yolo11OBBPredictor(const std::string& weights, int imgsz = 1024,
                      std::string device = "", int nc = 15,
                      models::Yolo11Scale scale = models::kYolo11n);
  std::vector<OBBInstance> predict(const cv::Mat& bgr,
                                   NMSConfig conf = {}) const;
  std::vector<OBBInstance> predict_to_file(const std::string& in_path,
                                           const std::string& out_path,
                                           NMSConfig conf = {},
                                           const std::vector<std::string>& names = {}) const;
  int imgsz() const { return imgsz_; }
 private:
  models::Yolo11OBB model_;
  torch::Device      device_;
  int                imgsz_;
};

// ─── YOLO26 task predictors ────────────────────────────────────────────────
//
// Same interface as the v8 / v11 versions; built on Yolo26Classify /
// Yolo26Segment / Yolo26Pose / Yolo26OBB. The detail::run_<task> helpers
// are template-dispatched on the model holder, so the post-processing path
// is shared.

class Yolo26ClassifyPredictor {
 public:
  Yolo26ClassifyPredictor(const std::string& weights, int imgsz = 224,
                           std::string device = "", int nc = 1000,
                           models::Yolo26Scale scale = models::kYolo26n);
  ClassifyResult predict(const cv::Mat& bgr, int top_k = 5) const;
  models::Yolo26Classify& model() { return model_; }
  int imgsz() const { return imgsz_; }
 private:
  models::Yolo26Classify model_;
  torch::Device          device_;
  int                    imgsz_;
};

class Yolo26SegmentPredictor {
 public:
  Yolo26SegmentPredictor(const std::string& weights, int imgsz = 640,
                          std::string device = "", int nc = 80,
                          models::Yolo26Scale scale = models::kYolo26n);
  std::vector<SegInstance> predict(const cv::Mat& bgr,
                                   NMSConfig conf = {}) const;
  std::vector<SegInstance> predict_to_file(const std::string& in_path,
                                           const std::string& out_path,
                                           NMSConfig conf = {},
                                           const std::vector<std::string>& names = {}) const;
  int imgsz() const { return imgsz_; }
 private:
  models::Yolo26Segment model_;
  torch::Device          device_;
  int                    imgsz_;
};

class Yolo26PosePredictor {
 public:
  Yolo26PosePredictor(const std::string& weights, int imgsz = 640,
                       std::string device = "", int num_kpts = 17, int kpt_dim = 3,
                       models::Yolo26Scale scale = models::kYolo26n);
  std::vector<PoseInstance> predict(const cv::Mat& bgr,
                                    NMSConfig conf = {}) const;
  std::vector<PoseInstance> predict_to_file(const std::string& in_path,
                                            const std::string& out_path,
                                            NMSConfig conf = {}) const;
  int imgsz() const { return imgsz_; }
 private:
  models::Yolo26Pose model_;
  torch::Device      device_;
  int                imgsz_;
  int                num_kpts_, kpt_dim_;
};

class Yolo26OBBPredictor {
 public:
  Yolo26OBBPredictor(const std::string& weights, int imgsz = 1024,
                      std::string device = "", int nc = 15,
                      models::Yolo26Scale scale = models::kYolo26n);
  std::vector<OBBInstance> predict(const cv::Mat& bgr,
                                   NMSConfig conf = {}) const;
  std::vector<OBBInstance> predict_to_file(const std::string& in_path,
                                           const std::string& out_path,
                                           NMSConfig conf = {},
                                           const std::vector<std::string>& names = {}) const;
  int imgsz() const { return imgsz_; }
 private:
  models::Yolo26OBB model_;
  torch::Device      device_;
  int                imgsz_;
};

}  // namespace yolocpp::inference
