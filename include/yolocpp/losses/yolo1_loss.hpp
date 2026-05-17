#pragma once
//
// YOLOv1 (Redmon 2016) sum-of-squared-error loss.
//
// Output is the raw flat `[B, S·S·(B·5+nc)]` from the FC head, in
// Darknet's three-block storage order:
//   [0           .. S·S·nc       )  class probabilities (per cell)
//   [S·S·nc      .. S·S·(nc+B)   )  per-box objectness scores
//   [S·S·(nc+B)  .. S·S·(nc+B+4B))  per-box (x, y, w, h)
//
// For each GT box (b, c, cx, cy, w, h) (normalized to [0, 1]):
//   - Locate the (gj, gi) grid cell containing the center.
//   - Among the B predicted boxes in that cell, the "responsible"
//     box is the one with highest IoU vs the GT (computed in image
//     coordinates).
//   - Coordinate loss: λ_coord · [(tx - tx̂)² + (ty - tŷ)² +
//                                  (√w - √ŵ)² + (√h - √ĥ)²]
//   - Objectness loss (responsible): (1·IoU - confidence)²
//   - Objectness loss (non-responsible + all empty cells):
//       λ_noobj · confidence²
//   - Class loss (cells containing a GT): Σ_classes (1{c==c_gt} - p̂_c)²
//
// Defaults: λ_coord=5.0, λ_noobj=0.5. Matches Redmon's published
// hyperparameters.
//

#include <torch/torch.h>

#include "yolocpp/losses/yolo8_loss.hpp"   // LossOutput

namespace yolocpp::losses {

struct Yolo1LossConfig {
  int   S         = 7;
  int   B         = 2;
  int   nc        = 20;
  float lambda_coord = 5.0f;
  float lambda_noobj = 0.5f;
};

struct Yolo1Loss {
  Yolo1LossConfig cfg;
  explicit Yolo1Loss(Yolo1LossConfig c = {}) : cfg(std::move(c)) {}

  // feats: single-element vector with `[B, S·S·(B·5+nc)]`
  // tgt:   `[N_total_gt, 6]` = (batch_idx, class, cx, cy, w, h) in
  //        normalized [0, 1] coords (yolo dataset convention).
  // strides, imgsz: passed for parity with the templated trainer
  // signature; v1 ignores `strides` (single-cell-grid model) and uses
  // `imgsz` only for IoU computation in image-space.
  LossOutput operator()(const std::vector<torch::Tensor>& feats,
                         const torch::Tensor& tgt,
                         const std::vector<double>& strides,
                         int imgsz) const;
};

}  // namespace yolocpp::losses
