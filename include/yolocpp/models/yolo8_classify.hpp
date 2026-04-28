#pragma once
//
// YOLO8 classification model.
//
// Architecture (different topology from detect):
//   layers 0–8 : same backbone modules as detect, but C2f depths from the
//                cls YAML — n=2,4,4,2 → after depth_multiple=0.33 → all 1.
//   layer 9    : Classify head =
//       Conv(c_in=256, c_out=1280, k=1, s=1) + AdaptiveAvgPool2d(1) +
//       Flatten + Dropout(0.0) + Linear(1280, nc).
//
// Forward returns logits [N, nc] (apply softmax for probs).

#include <torch/torch.h>

#include "yolocpp/models/yolo8.hpp"

namespace yolocpp::models {

struct ClassifyImpl : torch::nn::Module {
  Conv               conv{nullptr};            // c_in → 1280
  torch::nn::Linear  linear{nullptr};          // 1280 → nc
  ClassifyImpl(int c_in, int nc, int c_hidden = 1280);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Classify);

struct Yolo8ClassifyImpl : torch::nn::Module {
  Yolo8Scale scale;
  int         nc;
  torch::nn::ModuleList model{nullptr};

  Yolo8ClassifyImpl(Yolo8Scale s, int nc);

  // logits [N, nc]
  torch::Tensor forward(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo8Classify);

}  // namespace yolocpp::models
