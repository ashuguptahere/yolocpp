#pragma once
//
// Train + validate for instance segmentation.
//
// Dataset: same YOLO layout as detect, but each label line carries an
// arbitrary-length polygon after the box:
//     cls cx cy w h x1 y1 x2 y2 ... xN yN     (all coords normalized)
//
// Loss: detect loss (CIoU + DFL + BCE) + per-positive-anchor mask BCE.
// Validator: COCO-style mAP at IoU=0.5 using mask IoU instead of box IoU.
//

#include <torch/torch.h>

#include <random>
#include <string>
#include <vector>

#include "yolocpp/models/yolo8_tasks.hpp"

namespace yolocpp::tasks {

struct SegExample {
  torch::Tensor img;        // [3, H, W]
  torch::Tensor targets;    // [N, 5] — (cls, cx, cy, w, h) in input pixels
  torch::Tensor masks;      // [N, H, W] uint8 (0/1) at full input resolution
  int           orig_w = 0, orig_h = 0;
  double        gain = 1.0, pad_x = 0.0, pad_y = 0.0;
  std::string   img_path;
};

class SegDataset {
 public:
  SegDataset(std::string root, std::string split, int imgsz,
             std::vector<std::string> names, bool augment = true);

  std::size_t size() const { return paths_.size(); }
  int num_classes() const { return (int)names_.size(); }
  int imgsz() const { return imgsz_; }
  const std::vector<std::string>& names() const { return names_; }

  SegExample get(std::size_t idx, uint64_t seed = 0) const;

  struct Batch {
    torch::Tensor             imgs;       // [B, 3, H, W]
    torch::Tensor             targets;    // [M, 6] — (b, cls, cx, cy, w, h)
    torch::Tensor             masks;      // [M, H, W]  — full-res masks per target
    std::vector<SegExample>   examples;
  };
  Batch sample_batch(std::size_t bsz, std::mt19937& rng) const;

 private:
  std::vector<std::string> paths_;
  std::vector<std::string> lbl_paths_;
  std::vector<std::string> names_;
  int                      imgsz_;
  bool                     augment_;
};

struct SegTrainConfig {
  int    epochs        = 30;
  int    batch_size    = 4;
  int    imgsz         = 320;
  double lr0           = 0.01;
  double lrf           = 0.01;
  double momentum      = 0.937;
  double weight_decay  = 0.0005;
  int    warmup_epochs = 1;
  double mask_gain     = 7.5;       // weight for mask BCE in total loss
  std::string device   = "";
  std::string save_dir = "runs/segment";
  int        log_every = 10;
};

void train_segment(models::Yolo8Segment model,
                   const SegDataset& train,
                   const SegDataset* val,
                   SegTrainConfig cfg);

struct SegValResult {
  double map_50;     // mask mAP at IoU=0.5
  int    n_predictions;
  int    n_ground_truths;
};

SegValResult validate_segment(models::Yolo8Segment& model,
                              const SegDataset& dataset,
                              torch::Device device);

}  // namespace yolocpp::tasks
