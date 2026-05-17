#pragma once
//
// YOLOv2 (Redmon & Farhadi 2017) `region` loss.
//
// Output is the pre-decode `[B, na·(5+nc), H, W]` from the final 1×1
// conv. For each cell (cy, cx) and anchor a we get:
//   tx, ty, tw, th, to, t_c[5..5+nc)
//
// Region decode per Darknet:
//   bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy           (in cell units)
//   bw = anchor_w · exp(tw),  bh = anchor_h · exp(th)       (in cell units)
//   confidence = sigmoid(to),  class_probs = softmax(t_c)
//
// Matching: each GT box assigns to the cell containing its center; among
// the `na` anchors of that cell, the one with the highest IoU vs the GT
// (with both boxes centered at the cell center, so it's a w/h-only IoU)
// is "responsible". This is upstream Darknet's exact rule.
//
// Loss components (Darknet's `region_layer.c` defaults):
//   - Coord (matched only):  λ_coord · MSE on (tx_pred − tx_gt,
//                              ty_pred − ty_gt, tw_pred − log(gt_w/aw),
//                              th_pred − log(gt_h/ah))
//   - Objectness (matched):  (sigmoid(to) − IoU_pred_gt)²
//   - Objectness (no-match): λ_noobj · (sigmoid(to) − 0)²
//   - Classification (matched): cross-entropy on softmax(t_c)
//

#include <torch/torch.h>

#include <vector>

#include "yolocpp/losses/yolo8_loss.hpp"  // LossOutput

namespace yolocpp::losses {

struct Yolo2LossConfig {
  int   nc           = 20;
  // Anchors in *grid-cell units* — paired (w, h), 5 pairs by default.
  std::vector<float> anchors;
  float lambda_coord = 1.0f;   // upstream's coord_scale=1
  float lambda_noobj = 1.0f;   // upstream's noobject_scale=1
  float lambda_obj   = 5.0f;   // upstream's object_scale=5
  float lambda_class = 1.0f;   // upstream's class_scale=1
};

struct Yolo2Loss {
  Yolo2LossConfig cfg;
  explicit Yolo2Loss(Yolo2LossConfig c = {}) : cfg(std::move(c)) {}

  // feats: single-element vector with `[B, na·(5+nc), H, W]`.
  // tgt:   `[N, 6]` (batch_idx, class, cx, cy, w, h) normalized to [0, 1].
  // strides, imgsz: standard trainer signature.
  LossOutput operator()(const std::vector<torch::Tensor>& feats,
                         const torch::Tensor& tgt,
                         const std::vector<double>& strides,
                         int imgsz) const;
};

}  // namespace yolocpp::losses
