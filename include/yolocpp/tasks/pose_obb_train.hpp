#pragma once
//
// Train + validate for pose and OBB tasks.
//
// Both reuse the YOLO directory layout. Pose labels carry keypoints after
// the bounding box; OBB labels carry an angle.
//

#include <torch/torch.h>

#include <random>
#include <string>
#include <vector>

#include "yolocpp/models/yolov8_tasks.hpp"

namespace yolocpp::tasks {

// ─── Pose ─────────────────────────────────────────────────────────────────
// Label format per line:
//   cls cx cy w h kx1 ky1 v1 kx2 ky2 v2 ... kxN kyN vN
// Coordinates are normalized; visibility v ∈ {0, 1, 2}.
struct PoseExample {
  torch::Tensor img;                    // [3, H, W]
  torch::Tensor targets;                // [N, 5] — (cls, cx, cy, w, h)  pixels
  torch::Tensor keypoints;              // [N, K, 3] — (x, y, v) pixels
  std::string   img_path;
  int           orig_w = 0, orig_h = 0;
  double        gain = 1.0, pad_x = 0.0, pad_y = 0.0;
};

class PoseDataset {
 public:
  PoseDataset(std::string root, std::string split, int imgsz, int num_kpts,
              int kpt_dim, bool augment = true);
  std::size_t size() const { return paths_.size(); }
  int  imgsz()       const { return imgsz_; }
  int  num_kpts()    const { return nk_; }
  PoseExample get(std::size_t idx, uint64_t seed = 0) const;
  struct Batch {
    torch::Tensor              imgs;         // [B, 3, H, W]
    torch::Tensor              targets;      // [M, 6] — (b, cls, cx, cy, w, h)
    torch::Tensor              keypoints;    // [M, K, 3]
    std::vector<PoseExample>   examples;
  };
  Batch sample_batch(std::size_t bsz, std::mt19937& rng) const;
 private:
  std::vector<std::string> paths_, lbl_paths_;
  int imgsz_, nk_, kdim_;
  bool augment_;
};

struct PoseTrainConfig {
  int    epochs     = 30;
  int    batch_size = 4;
  int    imgsz      = 320;
  double lr0        = 0.01;
  double lrf        = 0.01;
  double momentum   = 0.937;
  double weight_decay = 0.0005;
  int    warmup_epochs = 1;
  double kpt_gain   = 12.0;     // weight for keypoint regression loss
  double kobj_gain  = 1.0;      // weight for visibility BCE
  std::string device = "";
  std::string save_dir = "runs/pose";
  int        log_every = 10;
};

void train_pose(models::YoloV8Pose model,
                const PoseDataset& train,
                const PoseDataset* val,
                PoseTrainConfig cfg);

struct PoseValResult {
  double oks_map_50;        // mAP at OKS=0.5
  int    n_pred, n_gt, n_matched;
};

PoseValResult validate_pose(models::YoloV8Pose& model,
                            const PoseDataset& dataset,
                            torch::Device device);

// ─── OBB ──────────────────────────────────────────────────────────────────
// Label format per line:
//   cls cx cy w h angle      (angle in radians)
struct OBBLabelExample {
  torch::Tensor img;
  torch::Tensor targets;       // [N, 6] — (cls, cx, cy, w, h, angle)
  std::string   img_path;
  int           orig_w = 0, orig_h = 0;
  double        gain = 1.0, pad_x = 0.0, pad_y = 0.0;
};

class OBBDataset {
 public:
  OBBDataset(std::string root, std::string split, int imgsz,
             std::vector<std::string> names, bool augment = true);
  std::size_t size() const { return paths_.size(); }
  int  imgsz() const { return imgsz_; }
  int  num_classes() const { return (int)names_.size(); }
  OBBLabelExample get(std::size_t idx, uint64_t seed = 0) const;
  struct Batch {
    torch::Tensor                 imgs;
    torch::Tensor                 targets;       // [M, 7] (b, cls, cx, cy, w, h, angle)
    std::vector<OBBLabelExample>  examples;
  };
  Batch sample_batch(std::size_t bsz, std::mt19937& rng) const;
 private:
  std::vector<std::string> paths_, lbl_paths_, names_;
  int imgsz_;
  bool augment_;
};

struct OBBTrainConfig {
  int    epochs     = 30;
  int    batch_size = 4;
  int    imgsz      = 320;
  double lr0        = 0.01;
  double lrf        = 0.01;
  double momentum   = 0.937;
  double weight_decay = 0.0005;
  int    warmup_epochs = 1;
  double angle_gain = 1.0;
  std::string device = "";
  std::string save_dir = "runs/obb";
  int    log_every = 10;
};

void train_obb(models::YoloV8OBB model,
               const OBBDataset& train,
               const OBBDataset* val,
               OBBTrainConfig cfg);

struct OBBValResult {
  double map_50;
  int    n_pred, n_gt, n_matched;
};

OBBValResult validate_obb(models::YoloV8OBB& model,
                          const OBBDataset& dataset,
                          torch::Device device);

}  // namespace yolocpp::tasks
