#pragma once
//
// YOLO6 — Meituan Vision, "YOLO6: A Single-Stage Object Detection Framework
// for Industrial Applications" (Sept 2022, refresh "YOLO6 v3.0" Jan 2023).
//
// Backbone : EfficientRep (RepVGG-style re-parameterizable blocks)
// Neck     : Rep-PAN (RepBlocks in a PAN topology)
// Head     : Decoupled (separate cls / reg branches), anchor-free,
//            TAL assigner, VFL/SIoU loss.
// Scales   : n / s / m / l (n6/s6/m6/l6 for higher input resolutions).
//
// Status: STUB. Upstream weights are in a custom YOLO6 `.pt` format with
// flat key naming distinct from Ultralytics' nested `.model.<i>.<sub>` —
// will need its own state_dict adapter. Issue: yolocpp#yolo6.
//

#include <torch/torch.h>

namespace yolocpp::models {

struct Yolo6Impl : torch::nn::Module {
  int nc;
  explicit Yolo6Impl(int nc = 80);
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo6);

}  // namespace yolocpp::models
