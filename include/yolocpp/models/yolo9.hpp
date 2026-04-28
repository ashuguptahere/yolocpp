#pragma once
//
// YOLO9 — Wang et al., "YOLO9: Learning What You Want to Learn Using
// Programmable Gradient Information" (Feb 2024).
//
// Backbone : GELAN (Generalized Efficient Layer Aggregation Network) — a
//            CSP-style block family with depth/bottleneck flexibility.
// Auxiliary: PGI (Programmable Gradient Information) — auxiliary reversible
//            branch used only during training, dropped at deployment.
// Head     : Same anchor-free DFL Detect as v8 (Ultralytics ships
//            `yolo9c.pt` / `yolo9e.pt` repackaged with the v8 head).
//
// Status: STUB. Strategy is to reuse the v8 Detect head and DFL pipeline,
// swapping only the backbone module list. Issue: yolocpp#yolo9.
//

#include <torch/torch.h>

namespace yolocpp::models {

struct Yolo9Impl : torch::nn::Module {
  int nc;
  explicit Yolo9Impl(int nc = 80);
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo9);

}  // namespace yolocpp::models
