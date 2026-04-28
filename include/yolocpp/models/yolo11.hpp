#pragma once
//
// YOLO11 — Ultralytics official (Sept 2024). Ships as `yolo11<scale>.pt`
// (note: NO 'v' in the filename; this is the new official convention going
// forward, matching `yolo26<scale>.pt`).
//
// Backbone : refined CSP — replaces v8's C2f with C3k2 (kernel-tunable C3
//            block). Adds C2PSA (Cross-Stage-Partial with Position-Sensitive
//            Attention) at the deepest backbone stage.
// Neck     : same FPN + PAN topology as v8.
// Head     : same anchor-free DFL Detect as v8 (so the existing
//            `Detect`/DFL/loss pipeline can be reused without change).
// Tasks    : detect / segment / classify / pose / OBB — all 5 ship.
// Scales   : n / s / m / l / x.
//
// Status: STUB. Strategy is to lift the v8 Detect/loss/trainer wholesale
// and only swap the backbone+neck modules. Issue: yolocpp#yolo11.
//

#include <torch/torch.h>

namespace yolocpp::models {

struct Yolo11Impl : torch::nn::Module {
  int nc;
  explicit Yolo11Impl(int nc = 80);
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo11);

}  // namespace yolocpp::models
