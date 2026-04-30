#pragma once
//
// YOLO26 detection loss.
//
//   total = box_gain * CIoU + cls_gain * BCE_with_prog
//
// No DFL term — the v26 head is DFL-free and regresses (l, t, r, b)
// distances directly (after Softplus).
//
// Assignment: **STAL** (Stable Task-Aligned Learning) — a one-to-one
// variant of TAL.
//   - Same alignment metric as TAL: m = score^α · iou^β where score is the
//     anchor's predicted prob for the GT's class.
//   - Per GT, the **single best** anchor (argmax over A) is assigned (vs.
//     TAL's top-k). Each anchor can match at most one GT (max-iou tiebreak
//     across GTs).
//   - Soft target score for the chosen positive is the anchor's IoU with
//     the assigned GT (capped at 1). Negatives stay at 0.
//   - Result: one-to-one assignment → no NMS needed at inference (the
//     model produces unique-anchor-per-object predictions by training).
//
// ProgLoss: progressive cls weighting. Easy positives (high IoU) get full
// weight; hard ones are down-weighted at the start of training and ramp
// up. Implemented as `cls_w = (iou ^ progress) * align`, where
// progress ∈ [0, 1] is supplied by the trainer and slides 0 → 1 across
// training. progress=0 → focus mostly on alignment (smoother early
// gradients); progress=1 → recovers the full TAL-style soft target.
//
// Inputs:
//   feats     : per-level [B, 4 + nc, h_i, w_i] from Detect26 (training)
//   targets   : [M, 6] — (batch_idx, cls, cx, cy, w, h) in input pixels
//   strides   : per-level stride (e.g. {8, 16, 32})
//   imgsz     : input image size (square)
//   progress  : ∈ [0, 1] — set by trainer (e.g. epoch / total_epochs)
//

#include <torch/torch.h>

#include <vector>

namespace yolocpp::losses {

struct Yolo26LossConfig {
  float box_gain = 7.5f;
  float cls_gain = 0.5f;
  int   nc       = 80;
  // STAL alignment metric exponents.
  float alpha    = 0.5f;
  float beta     = 6.0f;
};

struct Yolo26LossOutput {
  torch::Tensor total;     // scalar
  torch::Tensor box;       // scalar (CIoU loss)
  torch::Tensor cls;       // scalar (BCE)
  torch::Tensor dfl;       // always 0 (kept for log-format parity with v8)
};

class Yolo26Loss {
 public:
  explicit Yolo26Loss(Yolo26LossConfig cfg = {});

  // progress ∈ [0, 1] — typically epoch / total_epochs at the call site.
  Yolo26LossOutput operator()(const std::vector<torch::Tensor>& feats,
                              const torch::Tensor& targets,
                              const std::vector<double>& strides,
                              int imgsz,
                              double progress = 1.0) const;

  const Yolo26LossConfig& config() const { return cfg_; }

 private:
  Yolo26LossConfig cfg_;
};

}  // namespace yolocpp::losses
