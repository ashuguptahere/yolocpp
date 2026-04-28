#pragma once
//
// YOLO10 — Tsinghua MIG, "YOLO10: Real-Time End-to-End Object Detection"
// (May 2024).
//
// Key trick: dual-head consistent label assignment. During training the
// model has both a one-to-many head (TAL, like v8) and a one-to-one head;
// at inference only the one-to-one head runs, eliminating NMS.
// Other refinements: rank-guided block design, partial self-attention.
//
// Status: STUB. Needs custom dual-head training graph plus an NMS-free
// post-process at inference. Ultralytics now ships v10 weights through
// their `yolo10*.pt` line, sharing many ops with v8. Issue: yolocpp#yolo10.
//

#include <torch/torch.h>

namespace yolocpp::models {

struct Yolo10Impl : torch::nn::Module {
  int nc;
  explicit Yolo10Impl(int nc = 80);
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo10);

}  // namespace yolocpp::models
