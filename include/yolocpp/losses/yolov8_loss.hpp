#pragma once
//
// YOLOv8 detection loss.
//
//   total = box_gain * CIoU + dfl_gain * DFL + cls_gain * BCE
//
// Assignment: Task-Aligned Assigner (TAL) — picks each ground-truth's
// top-k best anchors based on score^α × iou^β.
//
// Inputs:
//   feats    : per-level [B, no, h_i, w_i] from Detect head (training mode)
//              where no = nc + 4 * reg_max
//   targets  : [M, 6] — (batch_idx, cls, cx, cy, w, h) in input pixels
//   strides  : per-level stride (e.g. {8, 16, 32})
//   imgsz    : input image size (square)
//

#include <torch/torch.h>

#include <vector>

namespace yolocpp::losses {

struct LossConfig {
  float box_gain = 7.5f;
  float cls_gain = 0.5f;
  float dfl_gain = 1.5f;
  int   reg_max  = 16;
  int   nc       = 80;
  // TAL params
  int   topk     = 10;
  float alpha    = 0.5f;
  float beta     = 6.0f;
};

struct LossOutput {
  torch::Tensor total;     // scalar
  torch::Tensor box;       // scalar (CIoU loss)
  torch::Tensor cls;       // scalar (BCE)
  torch::Tensor dfl;       // scalar (DFL)
};

class V8DetectionLoss {
 public:
  explicit V8DetectionLoss(LossConfig cfg = {});

  LossOutput operator()(const std::vector<torch::Tensor>& feats,
                        const torch::Tensor& targets,
                        const std::vector<double>& strides,
                        int imgsz) const;

  const LossConfig& config() const { return cfg_; }

 private:
  LossConfig cfg_;
};

}  // namespace yolocpp::losses
