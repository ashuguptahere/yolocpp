#pragma once
//
// YOLO26 — upstream official (preview / early-access, 2025). Ships as
// `yolo26<scale>.pt` (no 'v', matching the YOLO11 convention).
//
// Stated changes vs. YOLO11:
//   - **DFL-free** Detect head (the distribution-focal-loss bins are gone;
//     box regression is direct, simplifying export and TRT graph shape).
//   - **End-to-end NMS-free** inference (one-to-one assignment at runtime).
//   - **ProgLoss + STAL** — replacement for v8-style TAL assigner.
//   - Edge / mobile latency as a first-class target.
//
// Filename convention: `yolo26<scale>.pt`.
//
// Architectural status (this implementation):
//   - Backbone + neck reuse the v11 module set (Conv / C3k2 / SPPF / C2PSA)
//     with the same 24-module layout (Detect at idx 23).
//   - Detect head replaced by `Detect26Impl` — DFL-free, 4-channel direct
//     box regression, depthwise-separable cv2/cv3 (mobile-friendly).
//   - Predict / eval / ONNX export paths are wired end-to-end across all
//     5 scales × 5 tasks. TRT export goes through the ONNX route.
//   - Training uses the STAL assigner + ProgLoss via Yolo26Loss
//     (the v8 DFL-based loss is incompatible with the DFL-free head).
//

#include <torch/torch.h>

#include <string>
#include <vector>

#include "yolocpp/models/yolo11.hpp"  // reuse C3k2, C2PSA, scale helpers
#include "yolocpp/models/yolo8.hpp"   // reuse Conv, DWConvBlock, SPPF

namespace yolocpp::models {

// ─── Scale (parallels Yolo11Scale; same per-letter widths) ────────────────
struct Yolo26Scale {
  double depth_multiple;
  double width_multiple;
  int    max_channels;
};

constexpr Yolo26Scale kYolo26n{0.50, 0.25, 1024};
constexpr Yolo26Scale kYolo26s{0.50, 0.50, 1024};
constexpr Yolo26Scale kYolo26m{0.50, 1.00,  512};
constexpr Yolo26Scale kYolo26l{1.00, 1.00,  512};
constexpr Yolo26Scale kYolo26x{1.00, 1.50,  512};

Yolo26Scale yolo26_scale_from_letter(const std::string& letter);
Yolo26Scale yolo26_scale_from_filename(const std::string& path);

int scale_channels_v26(int c, const Yolo26Scale& s);
int scale_depth_v26(int n, const Yolo26Scale& s);

// ─── C2PSAf — v26's PAN-bottom block (replaces the v11 deep-head C3k2) ────
//
// CSP-shaped block whose inner module is a Sequential of (Bottleneck →
// PSABlock). Same outer topology as C3k2 (cv1 1×1 split into 2*c_inner;
// chain of n inner blocks; cv2 1×1 merge over (2 + n) * c_inner channels)
// but each m[i] is a 2-element Sequential. State-dict layout:
//   cv1.{conv,bn}.<...>
//   cv2.{conv,bn}.<...>
//   m.<i>.0.{cv1,cv2}.{conv,bn}.<...>      (Bottleneck, k=(3,3), e=0.5)
//   m.<i>.1.{attn.{qkv,proj,pe},ffn.{0,1}}.<...>   (PSABlock, attn_ratio=0.5)
//
// num_heads inside the PSABlock follows the v11 rule: max(1, c_inner / 64).
struct C2PSAfImpl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  torch::nn::ModuleList m{nullptr};   // n × Sequential(Bottleneck, PSABlock)
  int  c_inner = 0;
  C2PSAfImpl(int c1, int c2, int n = 1, double e = 0.5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(C2PSAf);

// ─── Detect26 — DFL-free detect head ──────────────────────────────────────
//
// Outputs differ in train vs eval:
//   train (forward_features): std::vector<Tensor>  length=nl,
//                              shape [N, 4 + nc, h_i, w_i] for each level i.
//   eval  (decode):            Tensor [N, 4 + nc, A]
//                              where A = sum(h_i * w_i),
//                              boxes decoded to xyxy in input pixels,
//                              cls is sigmoided.
//
// Head structure (matches the shipped yolo26<x>.pt):
//   cv2 (regression, 4 ch out — regular Conv, not depthwise):
//       Conv 3×3 → Conv 3×3 → Conv2d 1×1 (c → 4)
//   cv3 (classification, nc ch out — depthwise-separable, v11-style):
//       (DWConv 3×3 → Conv 1×1) → (DWConv 3×3 → Conv 1×1) → Conv2d 1×1 (c → nc)
//
// State-dict path layout (per detect-input level i):
//   cv2.<i>.0.{conv,bn}.<...>         (first Conv 3×3)
//   cv2.<i>.1.{conv,bn}.<...>         (second Conv 3×3)
//   cv2.<i>.2.{weight,bias}           (final Conv2d 1×1)
//   cv3.<i>.0.{0,1}.{conv,bn}.<...>   (first DWConvBlock)
//   cv3.<i>.1.{0,1}.{conv,bn}.<...>   (second DWConvBlock)
//   cv3.<i>.2.{weight,bias}           (final Conv2d 1×1)
// Note: NO `dfl.*` key — the head has no DFL projection.
struct Detect26Impl : torch::nn::Module {
  int  nc = 80;
  int  nl = 3;
  int  no = 0;                       // = 4 + nc
  std::vector<int>     ch;           // per-level input channels
  std::vector<double>  stride;       // per-level stride (set by parent)
  torch::nn::ModuleList cv2{nullptr};   // regression branches (Sequential)
  torch::nn::ModuleList cv3{nullptr};   // classification branches (Sequential)

  Detect26Impl(int nc, std::vector<int> ch);
  std::vector<torch::Tensor> forward_features(std::vector<torch::Tensor> x);
  torch::Tensor              decode(const std::vector<torch::Tensor>& feats);
  // Apply the upstream detection-prior bias to the cls head's final 1×1
  // conv and the reg head's bias. Idempotent — safe to call after a
  // partial state-dict load that left cls heads at torch defaults.
  void                        init_biases();
};
TORCH_MODULE(Detect26);

// ─── Whole detection model ────────────────────────────────────────────────
//
// model[0..9]   backbone (Conv/C3k2/SPPF) — same indices as v11
// model[10]     C2PSA
// model[11..22] head (Upsample/Concat/C3k2 — same PAN topology as v11)
// model[23]     Detect26 (DFL-free)
struct Yolo26DetectImpl : torch::nn::Module {
  Yolo26Scale scale;
  int          nc;

  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo26DetectImpl(Yolo26Scale s, int nc);

  std::vector<torch::Tensor> forward_train(torch::Tensor x);
  torch::Tensor              forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo26Detect);

// Stub legacy holder kept for ABI compatibility with the old header.
struct Yolo26Impl : torch::nn::Module {
  int nc;
  explicit Yolo26Impl(int nc = 80);
  std::vector<torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(Yolo26);

}  // namespace yolocpp::models
