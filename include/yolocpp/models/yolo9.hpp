#pragma once
//
// YOLO9 — Wang, Yeh & Liao, "YOLOv9: Learning What You Want to Learn Using
// Programmable Gradient Information" (Feb 2024). DEPLOY form.
//
// Backbone : GELAN (Generalized ELAN) — RepNCSPELAN4 blocks + ADown
//            downsamplers + SPPELAN at the bottom of the backbone.
// Neck     : PAN with RepNCSPELAN4 / ADown.
// Head     : v8-style Detect (anchor-free, DFL, reg_max=16). PGI's
//            auxiliary branch is training-only and dropped at deploy.
//
// We ship Ultralytics' yolov9c.yaml. Keys match upstream's flat ModuleList
// `model.<i>.<sub>` exactly. Each `RepConv` is stored at deploy as a
// single 3×3 Conv2d with bias + SiLU; `serialization::convert_yolov9_pt`
// fuses the train-form (conv1 3×3 + conv2 1×1, optional identity-BN)
// into that.
//
// We reuse v8's `Conv` (Conv2d + BN + SiLU) and `DetectImpl(legacy=true)`
// (anchor-free DFL head with cv2/cv3 Sequential lanes). Yolo9-specific
// modules: Yolo9RepConv, Yolo9RepBottleneck, RepCSP, RepNCSPELAN4, ADown,
// SPPELAN.
//

#include <torch/torch.h>

#include <string>
#include <vector>

#include "yolocpp/models/yolo8.hpp"

namespace yolocpp::models {

// ─── Yolo9RepConv (deploy form) — Conv2d(bias=true) + SiLU ───────────────
struct Yolo9RepConvImpl : torch::nn::Module {
  torch::nn::Conv2d conv{nullptr};
  Yolo9RepConvImpl(int c_in, int c_out);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Yolo9RepConv);

// ─── Yolo9RepBottleneck — RepConv(3×3) + Conv(3×3) [+ shortcut] ──────────
struct Yolo9RepBottleneckImpl : torch::nn::Module {
  Yolo9RepConv cv1{nullptr};
  Conv         cv2{nullptr};
  bool         add = true;
  Yolo9RepBottleneckImpl(int c1, int c2);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Yolo9RepBottleneck);

// ─── RepCSP — like v8's C3 but with Yolo9RepBottleneck ───────────────────
struct RepCSPImpl : torch::nn::Module {
  Conv                  cv1{nullptr};
  Conv                  cv2{nullptr};
  Conv                  cv3{nullptr};
  torch::nn::ModuleList m{nullptr};
  RepCSPImpl(int c1, int c2, int n = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(RepCSP);

// ─── RepNCSPELAN4 — GELAN block ──────────────────────────────────────────
struct RepNCSPELAN4Impl : torch::nn::Module {
  Conv                  cv1{nullptr};
  torch::nn::ModuleList cv2{nullptr};   // {RepCSP, Conv}
  torch::nn::ModuleList cv3{nullptr};   // {RepCSP, Conv}
  Conv                  cv4{nullptr};
  int                   c_split = 0;
  RepNCSPELAN4Impl(int c1, int c2, int c3, int c4, int n);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(RepNCSPELAN4);

// ─── ADown — avg-pool split + 3×3 stride-2 / maxpool 3×3 stride-2 ────────
struct ADownImpl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  int  c = 0;
  ADownImpl(int c1, int c2);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ADown);

// ─── AConv — avg_pool + 3×3 stride-2 conv (used by v9 t/s/m) ─────────────
struct AConvImpl : torch::nn::Module {
  Conv cv1{nullptr};
  AConvImpl(int c1, int c2);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(AConv);

// ─── ELAN1 — simpler 4-conv ELAN (used at layer 2 in yolov9t) ────────────
//
// Same skeleton as RepNCSPELAN4 but cv2 / cv3 are bare Conv (3×3) blocks
// instead of [RepCSP, Conv]. State_dict keys: cv1 / cv2 / cv3 / cv4 (no
// nested ModuleList).
struct ELAN1Impl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  Conv cv3{nullptr};
  Conv cv4{nullptr};
  int  c_split = 0;
  ELAN1Impl(int c1, int c2, int c3, int c4);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ELAN1);

// ─── CBLinear — 1×1 Conv with output split into N branches by channel ────
//
// Used by yolov9e. Saves all branches concatenated; downstream CBFuse
// slices to pick a specific branch. Weight shape: [sum(c2s), c1, 1, 1].
struct CBLinearImpl : torch::nn::Module {
  torch::nn::Conv2d conv{nullptr};
  std::vector<int>  c2s;
  CBLinearImpl(int c1, std::vector<int> c2s_list);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CBLinear);

// ─── CBFuse — element-wise sum of selected CBLinear branches + a current
// ───────────────────────────────────────────────────────────────────────
// tensor (the "anchor", target spatial size). For each non-last input,
// slice by `idx[i]` using the corresponding c2s list, upsample (nearest)
// to the anchor's spatial size, sum.
struct CBFuseImpl : torch::nn::Module {
  std::vector<int>              idx;
  std::vector<std::vector<int>> c2s_per_input;
  CBFuseImpl(std::vector<int> idx,
             std::vector<std::vector<int>> c2s_per_input);
  torch::Tensor forward(const std::vector<torch::Tensor>& xs);
};
TORCH_MODULE(CBFuse);

// ─── SPPELAN — SPP-ELAN (cv1 + 3 maxpools at k=5, cv5 cat) ───────────────
struct SPPELANImpl : torch::nn::Module {
  Conv                 cv1{nullptr};
  torch::nn::MaxPool2d cv2{nullptr};
  torch::nn::MaxPool2d cv3{nullptr};
  torch::nn::MaxPool2d cv4{nullptr};
  Conv                 cv5{nullptr};
  SPPELANImpl(int c1, int c2, int c3, int k = 5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SPPELAN);

// ─── Yolo9 scale (t / s / m / c / e, deploy form) ────────────────────────
enum class Yolo9Scale { T, S, M, C, E };
inline Yolo9Scale yolo9_scale_from_letter(const std::string& s) {
  if (s == "t") return Yolo9Scale::T;
  if (s == "s") return Yolo9Scale::S;
  if (s == "m") return Yolo9Scale::M;
  if (s == "e") return Yolo9Scale::E;
  return Yolo9Scale::C;
}

// ─── Yolo9Impl ───────────────────────────────────────────────────────────
struct Yolo9Impl : torch::nn::Module {
  // Layout matches the v8/v11 trainer convention `M(scale, nc)` so the
  // templated TrainerT<Yolo9> EMA construction (`M(model_->scale, model_->nc)`)
  // compiles unchanged — public `scale` and `nc` fields, scale-first ctor.
  Yolo9Scale            scale;
  int                   nc;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  // reg_max = 16 (DFL bin count) — v9 uses v8's Detect head with legacy=true,
  // which fixes reg_max to 16. Exposed so V8DetectionLoss picks up the right
  // value via the LossTraits default.
  static constexpr int  reg_max = 16;

  Yolo9Impl(Yolo9Scale scale = Yolo9Scale::C, int nc = 80);

  // Returns [B, 4 + nc, A] xyxy + sigmoid'd cls — drop-in for our NMS.
  torch::Tensor forward_eval(torch::Tensor x);

  // Multi-scale raw feature maps for the loss (each [B, 4*reg_max+nc, H_i, W_i]).
  // Strides are populated lazily on first call (mirrors forward_eval).
  std::vector<torch::Tensor> forward_train(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo9);

}  // namespace yolocpp::models
