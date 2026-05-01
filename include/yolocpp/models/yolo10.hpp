#pragma once
//
// YOLO10 — Wang et al., "YOLOv10: Real-Time End-to-End Object Detection"
// (Tsinghua MIG, May 2024). DEPLOY form.
//
// Backbone : Conv + C2f + SCDown + C2fCIB + SPPF + PSA.
// Head     : v10Detect — dual-assignment training (one2many + one2one).
//            At deploy only the one2one head is used; the lead head is
//            trained to produce one prediction per object so NMS is
//            essentially redundant. Our pipeline still calls our
//            standard NMS at default IoU=0.45 for safety.
//
// Yolo10Impl is a yaml-walker matching Ultralytics' yolov10*.yaml indices
// 0..23. All 6 scales (n / s / m / b / l / x) are wired via a per-scale
// spec table. Backbone/head topology is identical; per-position layer
// kind (`C2f` vs `C2fCIB`) varies — captured in `v10_yaml_for(scale)`.
//
// `serialization::convert_yolov10_pt(yolov10n.pt → yolo10.pt)` does the
// upstream-to-deploy conversion: drops the one2many `cv2/cv3` keys (used
// only at training), renames `one2one_cv2`→`cv2` and `one2one_cv3`→`cv3`
// so they fit our standard `Detect(legacy=false)` slots, fuses the
// RepVGGDW pair (7×7 dwconv + 3×3 dwconv → single 7×7 dwconv with bias),
// casts fp16 → fp32.
//

#include <torch/torch.h>

#include <string>
#include <vector>

#include "yolocpp/models/yolo11.hpp"   // PSAAttention
#include "yolocpp/models/yolo8.hpp"

namespace yolocpp::models {

// ─── SCDown — spatial-channel decoupled downsample ───────────────────────
struct SCDownImpl : torch::nn::Module {
  Conv   cv1{nullptr};
  DWConv cv2{nullptr};
  SCDownImpl(int c1, int c2, int k = 3, int s = 2);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SCDown);

// ─── RepVGGDW (deploy form) — single 7×7 DWConv + SiLU ───────────────────
struct RepVGGDWImpl : torch::nn::Module {
  torch::nn::Conv2d conv{nullptr};
  RepVGGDWImpl(int c);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(RepVGGDW);

// ─── CIB — Compact Inverted Block (5-module Sequential) ──────────────────
struct CIBImpl : torch::nn::Module {
  torch::nn::ModuleList cv1{nullptr};   // ["0".."4"] — matches upstream's
                                         // nn.Sequential layout.
  bool                  add  = true;
  bool                  lk   = false;
  // C2fCIB overrides the default e=0.5 to e=1.0 — the middle RepVGGDW
  // then operates on `2 * c2` channels.
  CIBImpl(int c1, int c2, bool shortcut = true, bool lk = false,
          double e = 0.5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CIB);

// ─── C2fCIB — C2f with CIB inner blocks ──────────────────────────────────
struct C2fCIBImpl : torch::nn::Module {
  Conv                  cv1{nullptr};
  Conv                  cv2{nullptr};
  torch::nn::ModuleList m{nullptr};
  int                   c_inner = 0;
  C2fCIBImpl(int c1, int c2, int n = 1, bool shortcut = false,
             bool lk = false);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(C2fCIB);

// ─── PSA — Position-Sensitive Attention block ────────────────────────────
struct PSAImpl : torch::nn::Module {
  Conv                  cv1{nullptr};
  Conv                  cv2{nullptr};
  PSAAttention          attn{nullptr};
  torch::nn::Sequential ffn{nullptr};
  int                   c_  = 0;
  PSAImpl(int c1, int c2);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PSA);

// ─── Yolo10 scale (n / s / m / b / l / x) ────────────────────────────────
//
// Channels scale as `make_divisible(min(c2_yaml, max_channels) * width, 8)`.
// Depth scales as `max(round(n_yaml * depth), 1)`.
struct Yolo10Scale {
  double depth;
  double width;
  int    max_channels;
};
constexpr Yolo10Scale kYolo10n{0.33, 0.25, 1024};
constexpr Yolo10Scale kYolo10s{0.33, 0.50, 1024};
constexpr Yolo10Scale kYolo10m{0.67, 0.75, 768};
constexpr Yolo10Scale kYolo10b{0.67, 1.00, 512};
constexpr Yolo10Scale kYolo10l{1.00, 1.00, 512};
constexpr Yolo10Scale kYolo10x{1.00, 1.25, 512};

inline Yolo10Scale yolo10_scale_from_letter(const std::string& s) {
  if (s == "n") return kYolo10n;
  if (s == "s") return kYolo10s;
  if (s == "m") return kYolo10m;
  if (s == "b") return kYolo10b;
  if (s == "l") return kYolo10l;
  if (s == "x") return kYolo10x;
  return kYolo10n;
}

// ─── Yolo10 model (deploy form, optional dual-head training) ────────────
struct Yolo10Impl : torch::nn::Module {
  // Field layout matches v8/v11 trainer convention `M(scale, nc)` so v10
  // plugs into TrainerT directly.
  Yolo10Scale           scale;
  int                   nc;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  // When true, a parallel one2many head (legacy=true v8-style cv3) is
  // built alongside the deploy one2one head and `forward_train` returns
  // a 6-tensor vector [o2m_p3, o2m_p4, o2m_p5, o2o_p3, o2o_p4, o2o_p5]
  // suitable for `losses::V10DualLoss` (consistent assignment per paper
  // §3.1: TAL with topk=10 for one2many, topk=1 for one2one). The parallel
  // head is registered as `o2m_detect` so its weight names don't collide
  // with the deploy head's `cv2`/`cv3` ModuleLists.
  // Eval (`forward_eval`) is unchanged — it always reads the deploy
  // one2one head only.
  bool                   dual_head = false;
  Detect                 o2m_detect{nullptr};

  // reg_max for the deploy one2one head's DFL projection.
  static constexpr int reg_max = 16;

  // Single-head ctor (deploy / one2one-only training).
  Yolo10Impl(Yolo10Scale scale = kYolo10n, int nc = 80);
  // Dual-head ctor — adds the legacy-form one2many head for paper-§3.1
  // consistent-assignment training.
  Yolo10Impl(Yolo10Scale scale, int nc, bool dual_head);

  // Returns [B, 4 + nc, A] xyxy + sigmoid'd cls — drop-in for our NMS.
  torch::Tensor forward_eval(torch::Tensor x);

  // Multi-scale raw feature maps for the loss (each [B, 4*reg_max+nc, H_i, W_i]).
  //   dual_head=false: returns 3 one2one tensors (P3, P4, P5).
  //   dual_head=true : returns 6 tensors {o2m P3, o2m P4, o2m P5,
  //                                       o2o P3, o2o P4, o2o P5}.
  std::vector<torch::Tensor> forward_train(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo10);

}  // namespace yolocpp::models
