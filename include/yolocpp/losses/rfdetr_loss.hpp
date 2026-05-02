#pragma once
//
// RF-DETR set-prediction loss (#65F).
//
// ─── Math ────────────────────────────────────────────────────────────────
//
// Per-image cost matrix (n_pred × n_gt):
//
//     C(i, j) = λ_cls · -p_i[c_j]
//             + λ_l1  · ||b_i - b_j||_1
//             + λ_giou· -GIoU(b_i, b_j)
//
// Hungarian matcher picks the bipartite assignment minimising sum
// of costs. The actual loss then sums:
//
//   - Focal cls loss over ALL preds (matched preds → GT class;
//     unmatched preds → no-object class), with α=0.25, γ=2.
//   - L1 + GIoU bbox loss over MATCHED preds only.
//
// Auxiliary outputs from earlier decoder layers contribute the
// same loss with the same target assignment (RF-DETR style — the
// matcher is run only on the final layer's outputs to keep targets
// consistent across layers).
//
// ─── API ─────────────────────────────────────────────────────────────────
//
// `rfdetr_set_loss(cls_logits_per_layer, bbox_unact_per_layer,
//   targets, cfg)` — `cls_logits_per_layer[l]` is `[B, Q, nc]` raw
//   logits (no sigmoid), `bbox_unact_per_layer[l]` is `[B, Q, 4]`
//   raw bbox before sigmoid (cx, cy, w, h in [0,1] image coords
//   AFTER sigmoid). `targets` is one entry per batch image: a vector
//   of `(class_id, cx, cy, w, h)` ground-truth boxes.

#include <torch/torch.h>

#include <vector>

namespace yolocpp::losses {

struct RFDetrLossCfg {
  float lambda_cls  = 2.0f;
  float lambda_l1   = 5.0f;
  float lambda_giou = 2.0f;
  float focal_alpha = 0.25f;
  float focal_gamma = 2.0f;
};

struct RFDetrTarget {
  int64_t       class_id;
  float         cx, cy, w, h;   // image-normalised xywh
};

struct RFDetrLossOutput {
  torch::Tensor total;
  torch::Tensor cls;
  torch::Tensor l1;
  torch::Tensor giou;
};

RFDetrLossOutput rfdetr_set_loss(
    const std::vector<torch::Tensor>& cls_logits_per_layer,
    const std::vector<torch::Tensor>& bbox_unact_per_layer,
    const std::vector<std::vector<RFDetrTarget>>& targets,
    const RFDetrLossCfg& cfg = {});

}  // namespace yolocpp::losses
