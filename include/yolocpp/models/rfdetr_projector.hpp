#pragma once
//
// RF-DETR 1.6.5 projector neck (#65B2).
//
// Bridges the 4 ViT taps from the DINOv2 backbone (#65A2) into a
// single-level feature map at `hidden_dim` channels. Architecture is
// upstream's CSP-style C2f block (`projector.stages.0.0`) followed
// by a BatchNorm (`projector.stages.0.1`).
//
// Per-variant shapes (all 12 variants share the same projector
// scaffolding; only channel widths differ):
//
//   * nano/small/medium/base + all seg variants:
//     hidden=256, cv1 in_ch=4·384=1536, m has 3 bottlenecks at
//     hidden/2=128 channels, cv2 in_ch=256+3·128=640.
//
//   * large: hidden=384, cv1 in_ch=4·768=3072, m has 3 bottlenecks
//     at hidden/2=192, cv2 in_ch=384+3·192=960. Plus an extra
//     `stages.1` doing the same shape (refines tap features twice).
//
// ─── Parameter naming (matches upstream exactly) ─────────────────────────
//
//   stages.<s>.0.cv1.conv.weight                     [hidden, in_ch, 1, 1]
//   stages.<s>.0.cv1.bn.{weight,bias}                [hidden]
//   stages.<s>.0.cv2.conv.weight                     [hidden, fanin, 1, 1]
//   stages.<s>.0.cv2.bn.{weight,bias}                [hidden]
//   stages.<s>.0.m.<i>.cv1.conv.weight               [hidden/2, hidden/2, 3, 3]
//   stages.<s>.0.m.<i>.cv1.bn.{weight,bias}          [hidden/2]
//   stages.<s>.0.m.<i>.cv2.conv.weight               [hidden/2, hidden/2, 3, 3]
//   stages.<s>.0.m.<i>.cv2.bn.{weight,bias}          [hidden/2]
//   stages.<s>.1.{weight,bias}                       [hidden]   final BN

#include <torch/torch.h>

#include <vector>

#include "yolocpp/models/rfdetr_backbone.hpp"

namespace yolocpp::models::rfdetr {

// Conv2d + BatchNorm2d (track_running_stats=False, matching upstream's
// `get_norm` default for `layer_norm=False`) + SiLU. Submodules
// `conv` and `bn` named to match upstream.
class ConvBNImpl : public torch::nn::Module {
 public:
  ConvBNImpl(int in_ch, int out_ch, int kernel, int padding);
  torch::Tensor forward(torch::Tensor x);
  torch::nn::Conv2d      conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
};
TORCH_MODULE(ConvBN);

// Channels-last LayerNorm over a 4D NCHW tensor. Matches upstream's
// custom `LayerNorm` (permute → F.layer_norm → permute back). Used
// for `projector.stages.<i>.1` and similar.
class ChannelLastLNImpl : public torch::nn::Module {
 public:
  explicit ChannelLastLNImpl(int channels);
  torch::Tensor forward(torch::Tensor x);
  torch::Tensor weight;
  torch::Tensor bias;
 private:
  int64_t channels_;
};
TORCH_MODULE(ChannelLastLN);

// Bottleneck: two 3×3 ConvBN. No residual (RF-DETR's projector m
// blocks are "without shortcut" per upstream `Bottleneck(shortcut=False)`).
class ProjBottleneckImpl : public torch::nn::Module {
 public:
  explicit ProjBottleneckImpl(int channels);
  torch::Tensor forward(torch::Tensor x);
  ConvBN cv1{nullptr};
  ConvBN cv2{nullptr};
};
TORCH_MODULE(ProjBottleneck);

// One C2f stage: cv1 1×1 → split → m (N bottlenecks) → concat → cv2 1×1.
class ProjStage0Impl : public torch::nn::Module {
 public:
  ProjStage0Impl(int in_ch, int hidden, int n_bottlenecks);
  torch::Tensor forward(torch::Tensor x);
  ConvBN                cv1{nullptr};
  ConvBN                cv2{nullptr};
  torch::nn::ModuleList m{nullptr};
 private:
  int hidden_;
  int n_;
};
TORCH_MODULE(ProjStage0);

// Full projector: ModuleList "stages" of one or two ProjStages, each
// followed by a BatchNorm at slot `.1`. Submodule path:
// `stages.<s>.0.*` for the C2f, `stages.<s>.1` for the BN.
class ProjectorImpl : public torch::nn::Module {
 public:
  // n_stages: 1 (most variants) or 2 (rfdetr-large only).
  // tap_concat_ch: 4 × backbone_embed (ViT C × 4 taps concatenated).
  // hidden:        scale.hidden_dim (256 or 384).
  // n_bottlenecks: 3 (upstream constant).
  ProjectorImpl(int n_stages, int tap_concat_ch, int hidden,
                 int n_bottlenecks = 3);
  // Input: 4 ViT taps `[B, embed, Hg, Wg]` (already at the same grid).
  // Output: single-level `[B, hidden, Hg, Wg]`.
  torch::Tensor forward(const std::vector<torch::Tensor>& taps);
  torch::nn::ModuleList stages{nullptr};
};
TORCH_MODULE(Projector);

// `BackboneSlot` reproduces upstream's `backbone[0]` Module: it owns
// both `encoder` (DINOv2 wrapper-outer) and `projector` as siblings,
// so the dotted paths become `backbone.0.encoder.encoder.*` and
// `backbone.0.projector.stages.*`. RFDetrImpl registers a top-level
// ModuleList named "backbone" containing one of these.
class BackboneSlotImpl : public torch::nn::Module {
 public:
  BackboneSlotImpl(const Dinov2Cfg& backbone_cfg, int hidden_dim,
                    int n_proj_stages = 1, int n_bottlenecks = 3);
  // Backbone forward (returns 4 ViT taps in spatial form) → projector
  // → single-level `[B, hidden, Hg, Wg]` feature map.
  torch::Tensor forward(torch::Tensor x);
  // BackboneSlot itself counts as one of the 'encoder' levels in the
  // path (`backbone.0.encoder...`). So `encoder` here is the
  // single-wrapper, which has its own `encoder` child = Dinov2Model.
  // Total path: `backbone.0.encoder.encoder.embeddings.*` ✓
  Dinov2Wrapper encoder{nullptr};
  Projector     projector{nullptr};
};
TORCH_MODULE(BackboneSlot);

}  // namespace yolocpp::models::rfdetr
