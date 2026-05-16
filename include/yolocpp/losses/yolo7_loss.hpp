#pragma once
//
// Anchor-based v3-style detection loss — used by YOLO4 and YOLO7.
//
// Per-anchor outputs at each scale: [B, na*(5+nc), H, W] where
//   na   = anchors per scale (3 for base/tiny/x; v7 P6 variants also 3)
//   5+nc = (tx, ty, tw, th, obj, cls...)
//
// Decoder (training-side, matches predict-time):
//   xy_pix = (sigmoid(t_xy) * scale_xy - 0.5*(scale_xy - 1) + grid) * stride
//   wh_pix = (sigmoid(t_wh) * 2)^2 * anchor_w_in_pixels       (v7 form)
//   wh_pix = exp(t_wh) * anchor_w_in_pixels                    (v4 cfg form)
//
// `scale_xy` is per-scale (v4: 1.2/1.1/1.05 for P3/P4/P5; v7: 2.0
// uniform). The wh decode is also model-specific — v7 uses the
// (sigmoid*2)^2 form; v4 uses exp(). The config carries a flag.
//
// Loss components:
//   box: 1 - CIoU(decoded_pred, gt) on positive (anchor, cell, gt) triplets
//   obj: BCE on objectness logits over ALL (b, anchor, h, w) cells,
//        with target = (1-gr) + gr*IoU.detach() at positives, 0 elsewhere.
//        Multi-scale balance weights: [4.0, 1.0, 0.4] for P3/P4/P5.
//   cls: BCE on cls logits only at positives, target = one-hot of GT class.
//
// Anchor matching: per upstream v3/v4/v7 — for each GT, find anchors at
// each scale with `max(gt_wh/anc_wh, anc_wh/gt_wh).max() < anchor_t=4`,
// then optionally expand to neighboring cells (offset_t=0.5).
//

#include <torch/torch.h>

#include <utility>
#include <vector>

#include "yolocpp/losses/yolo8_loss.hpp"   // LossOutput

namespace yolocpp::losses {

struct V7LossConfig {
  int nc = 80;
  int na = 3;
  // anchors[level][anchor_idx] = (w, h) in pixels at the calibration imgsz.
  std::vector<std::vector<std::pair<float, float>>> anchors;
  std::vector<double> strides;     // per-level stride
  std::vector<float>  scale_xy;    // per-level (v4: [1.2,1.1,1.05], v7: [2,2,2])
  // wh decode form: true → (sigmoid*2)^2 (v7); false → exp() (v4 Darknet).
  bool wh_sigmoid = true;

  // Loss gains (from v7 yaml hyp.scratch.p5.yaml).
  float box_gain   = 0.05f;
  float cls_gain   = 0.3f;
  float obj_gain   = 0.7f;
  float anchor_t   = 4.0f;
  float gr         = 1.0f;        // obj target = (1-gr) + gr * IoU.detach()
  std::vector<float> balance = {4.0f, 1.0f, 0.4f};   // per-level obj weight
  // Auto-balance the per-level `balance` based on observed per-level
  // positive counts during training. When enabled, `balance[li]` is
  // updated each step to be proportional to the EMA of per-level
  // positive count, clamped to [0.1, 10]×. Compensates for datasets
  // whose size distribution doesn't match COCO (e.g. screen-detection
  // has large objects landing exclusively at P5/P6; upstream's COCO-
  // tuned balance down-weights those levels by 16–64× — wrong
  // direction). Default ON; set false to use the static `balance`.
  bool  autobalance = true;
  // Reseed the anchor table from the training-set GT distribution via
  // K-means with IoU distance. Only helpful when the GT size
  // distribution diverges from COCO enough to drop BPR (best possible
  // recall under anchor_t=4) below ~0.98. For COCO-similar fine-tuning
  // this REGRESSES results in short training budgets because it breaks
  // the pretrained anchor-decoding alignment. Default OFF; flip on via
  // a custom `LossTraits` specialization or by mutating the cfg.
  bool  autoanchor  = false;
  // Center-prior offset (cells with center within `offset_t` of cell
  // boundary also become positives). 0 = disabled, 0.5 = standard.
  float offset_t   = 0.5f;
};

// Reseed the loss's `anchors` table from a list of GT (w, h) pixel
// dimensions via K-means with IoU distance. Returns the per-level
// anchor table that callers can also inspect; `out_cfg.anchors` gets
// updated in place. K = `out_cfg.na * out_cfg.strides.size()`.
//
// IoU-distance K-means matches upstream YOLO's "autoanchor": it picks
// centroids that maximize the recall of a fixed threshold (anchor_t=4)
// against the GT distribution, which on a custom dataset can shift
// anchor sizes by 5–10× from the COCO-tuned defaults.
std::vector<std::vector<std::pair<float, float>>>
kmeans_anchors(const std::vector<std::pair<float, float>>& gt_whs,
               V7LossConfig& out_cfg, int kmeans_iters = 30);

class V7DetectionLoss {
 public:
  explicit V7DetectionLoss(V7LossConfig cfg);

  // Re-cluster anchors from the training-set GT distribution.
  // Convenience wrapper around `kmeans_anchors()` that updates the
  // loss's internal config in place. `gt_whs` are (width, height) in
  // pixel units of the training-time image (after letterbox to imgsz).
  void recluster_anchors(const std::vector<std::pair<float, float>>& gt_whs);

  // feats : per-level [B, na*(5+nc), H_i, W_i] in stride-ascending order
  //         (P3, P4, P5). Caller (Yolo4 forward_train) reverses if needed.
  // targets: [M, 6] — (batch_idx, cls, cx, cy, w, h) in input-image pixel
  //         coords (matches our YoloDataset target convention).
  LossOutput operator()(const std::vector<torch::Tensor>& feats,
                        const torch::Tensor& targets,
                        const std::vector<double>& strides,
                        int imgsz) const;

  const V7LossConfig& config() const { return cfg_; }

 private:
  mutable V7LossConfig cfg_;
  // Per-level EMA of positive count. Initialized lazily on first call
  // (sized to feats.size()). Used by the autobalance loop in operator().
  mutable std::vector<double> ema_pos_count_;
  mutable int64_t              step_count_ = 0;
};

}  // namespace yolocpp::losses
