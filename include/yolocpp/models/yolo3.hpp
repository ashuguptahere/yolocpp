#pragma once
//
// YOLO3 architecture (canonical Darknet-53 backbone + 3-scale FPN head).
//
// Status: architecture only. Weight loading from Ultralytics' yolo3.pt is
// deferred — the legacy v3 weights use a flat key naming that doesn't map
// onto our nested ModuleList layout, and the original Darknet weights are
// in a different format altogether. Random-forward shape verification only.
//
// Reference: pjreddie's Darknet, with the 3-scale head from
// "YOLO3: An Incremental Improvement" (Redmon & Farhadi, 2018).
//

#include <torch/torch.h>

#include "yolocpp/models/yolo8.hpp"  // reuse Conv from yolo8.hpp

namespace yolocpp::models {

// ─── DarknetResidual (1×1 → 3×3 with skip) ──────────────────────────────
struct DarknetResidualImpl : torch::nn::Module {
  Conv cv1{nullptr};   // c → c/2 (1×1)
  Conv cv2{nullptr};   // c/2 → c (3×3)
  DarknetResidualImpl(int c);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DarknetResidual);

// ─── DarknetBlock (downsample 3×3 stride-2 + N residuals) ───────────────
struct DarknetBlockImpl : torch::nn::Module {
  Conv down{nullptr};
  torch::nn::ModuleList m{nullptr};
  DarknetBlockImpl(int c_in, int c_out, int n);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DarknetBlock);

// ─── Yolo3 model ──────────────────────────────────────────────────────
//
// Backbone (Darknet-53):
//   stem:  Conv 32, 3×3
//   block1: down→64,  1 residual
//   block2: down→128, 2 residuals
//   block3: down→256, 8 residuals  (P3 output)
//   block4: down→512, 8 residuals  (P4 output)
//   block5: down→1024, 4 residuals (P5 output)
//
// Head: 3-scale FPN with 3 anchors per scale at 80 classes →
//       per-scale output channels = 3 × (5 + 80) = 255.
struct Yolo3Impl : torch::nn::Module {
  int nc;
  // Backbone
  Conv             stem{nullptr};
  DarknetBlock     b1{nullptr}, b2{nullptr}, b3{nullptr}, b4{nullptr}, b5{nullptr};
  // Neck (FPN convs)
  Conv             p5_pre1{nullptr}, p5_pre2{nullptr}, p5_pre3{nullptr},
                   p5_pre4{nullptr}, p5_pre5{nullptr};
  Conv             p5_out_pre{nullptr};
  torch::nn::Conv2d p5_out{nullptr};
  Conv             p4_red{nullptr};
  Conv             p4_pre1{nullptr}, p4_pre2{nullptr}, p4_pre3{nullptr},
                   p4_pre4{nullptr}, p4_pre5{nullptr};
  Conv             p4_out_pre{nullptr};
  torch::nn::Conv2d p4_out{nullptr};
  Conv             p3_red{nullptr};
  Conv             p3_pre1{nullptr}, p3_pre2{nullptr}, p3_pre3{nullptr},
                   p3_pre4{nullptr}, p3_pre5{nullptr};
  Conv             p3_out_pre{nullptr};
  torch::nn::Conv2d p3_out{nullptr};

  explicit Yolo3Impl(int nc = 80);

  // Returns 3 raw output tensors at strides 32 / 16 / 8 (P5, P4, P3).
  // Each shape: [B, 3 * (5 + nc), H_i, W_i]
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo3);

}  // namespace yolocpp::models
