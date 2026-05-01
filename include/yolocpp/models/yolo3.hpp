#pragma once
//
// YOLO3 — Redmon & Farhadi, "YOLOv3: An Incremental Improvement" (2018).
// We ship Ultralytics' anchor-free `yolov3u.pt` form: Darknet-53 backbone
// + v8-style anchor-free DFL Detect head. Predict piggybacks on the
// existing v8 pipeline.
//
// Architecture is a flat yaml-walker (ModuleList "model") matching
// Ultralytics' yolov3.yaml indices 0..28. Reuses v8's `Conv`,
// `Bottleneck`, and `DetectImpl(legacy=true)` directly, so upstream key
// names map 1:1.
//
// `serialization::convert_yolov3_pt(yolov3u.pt → yolo3.pt)` casts fp16
// → fp32 and drops `num_batches_tracked`. No fusion needed.
//

#include <torch/torch.h>

#include <string>
#include <vector>

#include "yolocpp/models/yolo8.hpp"

namespace yolocpp::models {

// v3 is a single-architecture family — there's no n/s/m/l/x scaling.
// `Yolo3Scale` is a placeholder struct so v3 conforms to the v8/v11
// trainer convention `M(scale, nc)` (the EMA in TrainerT<M> uses
// `M(model_->scale, model_->nc)` to construct a fresh copy).
struct Yolo3Scale { int dummy = 0; };
constexpr Yolo3Scale kYolo3{};

struct Yolo3Impl : torch::nn::Module {
  // Field layout matches the v8/v11/v9 trainer convention M(scale, nc).
  // v3 has no real scale axis, so `scale` is a placeholder that satisfies
  // the templated `TrainerT<Yolo3>` EMA construction.
  Yolo3Scale            scale;
  int                   nc;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  // reg_max = 16 (DFL bin count) — yolov3u uses v8's anchor-free DFL
  // Detect head (legacy=true), so V8DetectionLoss is the right loss
  // class. The default LossTraits<M> picks it up automatically.
  static constexpr int  reg_max = 16;

  Yolo3Impl(Yolo3Scale scale = kYolo3, int nc = 80);

  // Returns [B, 4 + nc, A] xyxy + sigmoid'd cls — drop-in for our NMS.
  torch::Tensor forward_eval(torch::Tensor x);

  // Multi-scale raw feature maps for the loss (each [B, 4*reg_max+nc, H_i, W_i]).
  // Strides are populated lazily on first call (mirrors forward_eval).
  std::vector<torch::Tensor> forward_train(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo3);

}  // namespace yolocpp::models
