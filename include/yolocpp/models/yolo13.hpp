#pragma once
//
// YOLO13 — Lei et al. (unofficial fork), "YOLO13: Real-Time Object
// Detection with Hypergraph-Enhanced Adaptive Visual Perception" (mid-2025).
//
// Key trick: HyperACE — a Hypergraph-based Adaptive Correlation Enhancement
//            module that models high-order pixel correlations beyond the
//            pairwise attention used by v12.
// Other  : FullPAD pipeline (full-resolution path-aggregation distribution),
//          DSConv-based depth-separable variants of standard convolutions
//          for inference-time efficiency.
// Head   : same anchor-free DFL Detect as v8.
//
// Filename convention: `yolo13<scale>.pt`.
//
// Status: STUB. Issue: yolocpp#yolo13.
//

#include <torch/torch.h>

namespace yolocpp::models {

struct Yolo13Impl : torch::nn::Module {
  int nc;
  explicit Yolo13Impl(int nc = 80);
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo13);

}  // namespace yolocpp::models
