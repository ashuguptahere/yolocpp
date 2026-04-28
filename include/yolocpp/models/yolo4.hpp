#pragma once
//
// YOLO4 — Bochkovskiy et al., "YOLO4: Optimal Speed and Accuracy of Object
// Detection" (April 2020).
//
// Backbone : CSPDarknet53 (Cross-Stage-Partial Darknet53)
// Neck     : SPP + PANet (path-aggregation FPN)
// Head     : v3-style anchor-based head (3 scales × 3 anchors)
// Activation: Mish in backbone, LeakyReLU elsewhere.
//
// Status: STUB. Architecture not yet implemented. The original v4 weights
// ship in Darknet `.weights` format (not `.pt`); a separate loader is
// required if upstream weights are wanted. Issue: yolocpp#yolo4.
//

#include <torch/torch.h>

namespace yolocpp::models {

struct Yolo4Impl : torch::nn::Module {
  int nc;
  explicit Yolo4Impl(int nc = 80);
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo4);

}  // namespace yolocpp::models
