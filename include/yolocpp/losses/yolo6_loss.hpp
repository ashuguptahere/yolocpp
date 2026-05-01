#pragma once
//
// YOLO6 detection loss — Meituan v6 (2022, release 0.4.0).
//
//   total = box_gain * (1 - SIoU) + dfl_gain * DFL + cls_gain * VFL
//
// Differences from V8DetectionLoss:
//   1. Cls loss is **Varifocal Loss (VFL)** instead of plain BCE. VFL
//      asymmetrically weights the BCE: positives use the IoU-quality
//      weight (TAL.target_scores) directly; negatives use a focal-style
//      `alpha * sigmoid(pred)^gamma * (1 - label)` weighting that
//      effectively does hard-negative mining.
//   2. Box loss is **SIoU** (Scale-aware IoU) instead of CIoU. SIoU
//      adds a center-distance + angle-aware shape cost on top of plain
//      IoU. Reference: Gevorgyan 2022, "SIoU Loss: More Powerful
//      Learning for Bounding Box Regression."
//   3. DFL is the same as v8 (only used when reg_max >= 1; v6n/s/n6/s6
//      use reg_max=0 and a direct 4-ch reg branch — the
//      `m/l/m6/l6` variants use reg_max=16 and DFL).
//
// Assignment uses the same Task-Aligned Assigner (TAL) as v8, with
// alpha=1.0 / beta=6.0 in upstream Meituan's `ComputeLoss`.
//
// This loss class targets the DFL-headed v6 variants (m/l/m6/l6).
// The direct-4-ch path used by v6n/s/n6/s6 needs a different decode
// (reg_preds outputs lrtb directly without DFL); shipping that is a
// follow-up — for the n/s scales, finetuning with this loss class is
// possible by adapting the converter to expose `reg_preds_dist` (the
// 68-ch DFL distillation target that's training-only upstream) at
// the head's reg branch.
//

#include <torch/torch.h>

#include <vector>

#include "yolocpp/losses/yolo8_loss.hpp"   // LossOutput is reused

namespace yolocpp::losses {

struct V6LossConfig {
  float box_gain = 2.5f;
  float cls_gain = 1.0f;
  float dfl_gain = 0.5f;
  int   reg_max  = 16;
  int   nc       = 80;
  // TAL params — Meituan upstream uses alpha=1.0, beta=6.0.
  int   topk     = 13;
  float alpha    = 1.0f;
  float beta     = 6.0f;
  // VFL params.
  float vfl_alpha = 0.75f;
  float vfl_gamma = 2.0f;
};

class V6DetectionLoss {
 public:
  explicit V6DetectionLoss(V6LossConfig cfg = {});

  // feats : per-level [B, 4*reg_max+nc, h_i, w_i] (DFL-headed variants).
  // targets: [M, 6] — (batch_idx, cls, cx, cy, w, h) in pixel coords.
  LossOutput operator()(const std::vector<torch::Tensor>& feats,
                        const torch::Tensor& targets,
                        const std::vector<double>& strides,
                        int imgsz) const;

  const V6LossConfig& config() const { return cfg_; }

 private:
  V6LossConfig cfg_;
};

}  // namespace yolocpp::losses
