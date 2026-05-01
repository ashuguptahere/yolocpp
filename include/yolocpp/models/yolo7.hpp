#pragma once
//
// YOLO7 — Wang, Bochkovskiy & Liao, "YOLO7: Trainable bag-of-freebies sets
// new state-of-the-art for real-time object detectors" (July 2022).
//
// This file ships the DEPLOY form of WongKinYiu's `yolov7.pt`:
//   * Layer-by-layer architecture is a flat ModuleList named "model"
//     mirroring upstream's yolov7.yaml indices 0..105 — so when the
//     converter writes `<key>` it lines up with `model.<i>.<sub>`
//     directly.
//   * Yolo7RepConv blocks (layers 102/103/104) are stored at deploy in their
//     re-parameterized form: a single 3×3 Conv2d(bias=true) + SiLU.
//     The upstream train form (rbr_dense + rbr_1x1, optional
//     rbr_identity) is fused by `serialization::convert_yolov7_pt`.
//   * IDetect at layer 105 is anchor-based v3-style: per-scale 1×1 conv
//     to 3*(5+nc) channels, with `anchors` and `anchor_grid` as buffers
//     loaded from upstream.
//
// All 7 variants supported: base / tiny / x / w6 / e6 / d6 / e6e.
// Train wired via V7DetectionLoss for base (anchor-based, scale_xy=2.0,
// (sigmoid*2)^2 wh decode); other scales ride the same loss config.
//

#include <torch/torch.h>

#include <string>
#include <vector>

namespace yolocpp::models {

// ─── ConvSiLU = Conv2d + BN + (SiLU | LeakyReLU(0.1)) ──────────────────
//
// v7 base / x use SiLU. v7-tiny uses LeakyReLU(0.1) — toggle via the
// thread-local `V7ActScope` (RAII) before constructing the model. The
// activation is captured at construction time per-instance.
struct ConvSiLUImpl : torch::nn::Module {
  torch::nn::Conv2d      conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  bool                   use_leaky = false;
  ConvSiLUImpl(int c_in, int c_out, int k = 1, int s = 1, int p = -1, int g = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ConvSiLU);

// ─── V7ActScope — RAII toggle for ConvSiLU activation ──────────────────
struct V7ActScope {
  bool prev;
  explicit V7ActScope(bool use_leaky);
  ~V7ActScope();
};

// ─── Yolo7Scale (base / tiny / x / w6 / e6 / d6 / e6e) ──────────────────
enum class Yolo7Scale { Base, Tiny, X, W6, E6, D6, E6e };
inline Yolo7Scale yolo7_scale_from_letter(const std::string& s) {
  if (s == "tiny" || s == "t") return Yolo7Scale::Tiny;
  if (s == "x")                return Yolo7Scale::X;
  if (s == "w6")               return Yolo7Scale::W6;
  if (s == "e6")               return Yolo7Scale::E6;
  if (s == "d6")               return Yolo7Scale::D6;
  if (s == "e6e")              return Yolo7Scale::E6e;
  return Yolo7Scale::Base;
}

// ─── MP — MaxPool2d 2×2 stride 2 (parameter-less) ────────────────────────
struct MPImpl : torch::nn::Module {
  torch::nn::MaxPool2d m{nullptr};
  MPImpl();
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MP);

// ─── SPPCSPC — yolov7's SPP-CSPC block (cv1..cv7 + 5/9/13 maxpools) ──────
struct SPPCSPCImpl : torch::nn::Module {
  ConvSiLU             cv1{nullptr}, cv2{nullptr}, cv3{nullptr}, cv4{nullptr},
                       cv5{nullptr}, cv6{nullptr}, cv7{nullptr};
  torch::nn::MaxPool2d m5{nullptr}, m9{nullptr}, m13{nullptr};
  SPPCSPCImpl(int c_in, int c_out);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SPPCSPC);

// ─── Yolo7RepConv (deploy form) — single Conv2d + SiLU ────────────────────────
struct Yolo7RepConvImpl : torch::nn::Module {
  torch::nn::Conv2d conv{nullptr};
  Yolo7RepConvImpl(int c_in, int c_out, int k = 3, int s = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Yolo7RepConv);

// ─── IDetect (deploy, anchor-based v3-style) ─────────────────────────────
//
// nl is derived from `ch.size()` at construction — 3 levels for v7-base
// /tiny/x (P3-P5), 4 levels for v7-w6/e6/d6/e6e (P3-P6).
struct IDetectImpl : torch::nn::Module {
  int  nc;
  int  na  = 3;        // anchors per scale
  int  no  = 0;        // 5 + nc
  int  nl  = 3;        // levels (3 for base/tiny/x; 4 for w6/e6/d6/e6e)
  torch::Tensor anchors;        // [nl, na, 2]
  torch::Tensor anchor_grid;    // [nl, 1, na, 1, 1, 2]
  std::vector<double> stride;
  torch::nn::ModuleList m{nullptr};

  IDetectImpl(int nc, const std::vector<int>& ch);
  torch::Tensor forward_eval(const std::vector<torch::Tensor>& feats);
};
TORCH_MODULE(IDetect);

// ─── ReOrg — 4× spatial-to-depth (2×2 pixel shuffle, used by P6 v7s) ────
//
// [B, C, H, W] → [B, 4*C, H/2, W/2]. No parameters.
struct ReOrgImpl : torch::nn::Module {
  ReOrgImpl() = default;
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ReOrg);

// ─── Shortcut — element-wise sum of two inputs (used by yolov7-e6e) ─────
//
// Each E-ELAN backbone/head stage has two parallel ELAN sub-blocks that
// take the same input; their outputs are summed element-wise via this
// `Shortcut`. Despite the paper's "shuffle/merge cardinality" framing,
// the deploy graph has no `channel_shuffle` — it's literally `x[0] + x[1]`.
struct Yolo7ShortcutImpl : torch::nn::Module {
  Yolo7ShortcutImpl() = default;
  torch::Tensor forward(const std::vector<torch::Tensor>& xs);
};
TORCH_MODULE(Yolo7Shortcut);

// ─── DownC — two-path strided downsample (used by yolov7-e6 / d6) ───────
//
//   path A: cv1 (1×1, c1→c_=c1) → cv3 (3×3 stride=k, c_→c2/2)
//   path B: maxpool stride=k → cv2 (1×1, c1→c2/2)
//   out = cat(cv3, cv2) → c2 ch.
struct DownCImpl : torch::nn::Module {
  ConvSiLU             cv1{nullptr};
  ConvSiLU             cv2{nullptr};
  ConvSiLU             cv3{nullptr};
  torch::nn::MaxPool2d mp{nullptr};
  DownCImpl(int c1, int c2, int k = 2);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DownC);

// ─── Public Spec / yaml accessor (used by ONNX exporter) ────────────────
//
// The actual yaml definitions live in src/models/yolo7.cpp. Exposing
// the `Spec` shape + `v7_yaml_for(scale)` lets external walkers
// (notably `serialization::export_yolo7_onnx`) reuse the canonical
// v7 connectivity per scale instead of duplicating 7 hardcoded yamls.
struct Yolo7Spec {
  std::vector<int> from;
  std::string      kind;
  std::vector<int> args;
};
const std::vector<Yolo7Spec>& yolo7_yaml_for(Yolo7Scale s);

// ─── Yolo7 model (yolov7-base, deploy form) ──────────────────────────────
struct Yolo7Impl : torch::nn::Module {
  // Public scale + nc fields matching trainer's EMA convention `M(scale,nc)`.
  Yolo7Scale            scale = Yolo7Scale::Base;
  int                       nc;
  torch::nn::ModuleList     model{nullptr};   // flat layers 0..105
  std::vector<double>       stride;           // [3] strides per detect scale

  Yolo7Impl(int nc = 80, Yolo7Scale s = Yolo7Scale::Base);
  Yolo7Impl(Yolo7Scale s, int nc) : Yolo7Impl(nc, s) {}

  // Returns the 3 backbone-head feature maps at strides 8/16/32 (raw,
  // pre-IDetect 1×1 conv).
  std::vector<torch::Tensor> forward_features(torch::Tensor x);

  // Training-mode forward: returns per-scale [B, na*(5+nc), H_i, W_i]
  // raw IDetect logits at strides ASCENDING (P3, P4[, P5[, P6]]) — input
  // shape contract for V7DetectionLoss.
  std::vector<torch::Tensor> forward_train(torch::Tensor x);

  // Decoded inference output: [B, 4 + nc, A] xyxy + sigmoid'd cls.
  torch::Tensor forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo7);

}  // namespace yolocpp::models
