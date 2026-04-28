#pragma once
//
// YOLO26 — Ultralytics official (preview / early-access, 2025). Ships as
// `yolo26<scale>.pt` (no 'v', matching the YOLO11 convention).
//
// Stated changes vs. YOLO11:
//   - **DFL-free** Detect head (the distribution-focal-loss bins are gone;
//     box regression is direct, simplifying export and TRT graph shape).
//   - **End-to-end NMS-free** inference (one-to-one assignment at runtime,
//     similar in spirit to v10's dual-head trick but baked in by default).
//   - **ProgLoss + STAL** (Stable Task-Aligned Learning) — replacement for
//     the v8-style TAL assigner with smoother gradients on small datasets.
//   - Designed with edge / mobile latency as a first-class target.
//
// Filename convention: `yolo26<scale>.pt`.
//
// Status: STUB. Will require: a new DFL-free Detect head, the
// one-to-one assigner, and a new loss module. Issue: yolocpp#yolo26.
//

#include <torch/torch.h>

namespace yolocpp::models {

struct Yolo26Impl : torch::nn::Module {
  int nc;
  explicit Yolo26Impl(int nc = 80);
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo26);

}  // namespace yolocpp::models
