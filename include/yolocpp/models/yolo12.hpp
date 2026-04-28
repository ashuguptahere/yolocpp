#pragma once
//
// YOLO12 — Tian et al. (unofficial fork), "YOLO12: Attention-Centric
// Real-Time Object Detectors" (Feb 2025).
//
// Backbone/Neck: introduces A2C2f (Area-Attention C2f) — an attention-
//                augmented C2f block that approximates global attention
//                with windowed area-attention to keep latency competitive.
// Head: same anchor-free DFL Detect as v8.
//
// Filename convention: `yolo12<scale>.pt` (the unofficial fork keeps the
// digit-only naming; we follow that throughout the codebase).
//
// Status: STUB. Issue: yolocpp#yolo12.
//

#include <torch/torch.h>

namespace yolocpp::models {

struct Yolo12Impl : torch::nn::Module {
  int nc;
  explicit Yolo12Impl(int nc = 80);
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo12);

}  // namespace yolocpp::models
