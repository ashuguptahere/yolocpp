#pragma once
//
// YOLO7 — Wang et al., "YOLO7: Trainable bag-of-freebies sets new
// state-of-the-art for real-time object detectors" (July 2022).
//
// Backbone : E-ELAN (Extended Efficient Layer-Aggregation Networks)
// Neck     : SPPCSPC + PAN
// Head     : v3-style anchor-based head, IDetect (auxiliary head during
//            training, deep supervision; main head used for inference).
// Trick    : Trainable bag-of-freebies — re-param OREPA, planned re-param
//            convolutions, coarse-for-aux/fine-for-lead label assignment.
//
// Status: STUB. Upstream `.pt` files use a different state_dict layout
// than Ultralytics; needs its own loader path. Issue: yolocpp#yolo7.
//

#include <torch/torch.h>

namespace yolocpp::models {

struct Yolo7Impl : torch::nn::Module {
  int nc;
  explicit Yolo7Impl(int nc = 80);
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo7);

}  // namespace yolocpp::models
