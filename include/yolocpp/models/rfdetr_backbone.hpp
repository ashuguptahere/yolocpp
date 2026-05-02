#pragma once
//
// RF-DETR backbones (#65A).
//
// Two families — both ViT-based, structurally similar, differ only in
// width/depth/window-attention:
//
//   * DINOv2 ViT-L  — Roboflow's `rfdetr-l` weights. Patch=14, 24
//                      blocks, embed_dim=1024, 16 heads, MLP ratio=4,
//                      input 560×560. Pre-trained self-supervised on
//                      LVD-142M; full self-attention, no windowing.
//
//   * LW-DETR ViT   — `rfdetr-{n,s,b,m}`. Patch=16, depth +
//                      embed_dim per scale (see kLwDetr*Cfg below),
//                      windowed attention with 14×14 window every
//                      layer except the last, input 640×640.
//
// Both expose the same forward interface — feed a `[B, 3, H, W]` image
// (already letterboxed + normalised), get back a list of multi-scale
// feature maps `[[B, C0, H/8, W/8], [B, C1, H/16, W/16], [B, C2,
// H/32, W/32]]` for the deformable encoder (#65B) to consume.
//
// Implementation rationale: both architectures are stock transformer
// blocks (LayerNorm → MHA → Add → LayerNorm → MLP → Add). LW-DETR's
// windowed attention is implemented as a strided reshape into local
// windows, full self-attention inside each window, then unshape — no
// custom CUDA kernel needed. Reusing the same `ViTBlockImpl` for both
// families lets #65D's state-dict converter stay simple (one keymap
// per family).

#include <torch/torch.h>

#include <string>
#include <vector>

namespace yolocpp::models::rfdetr {

// Per-backbone configuration. The `RFDetrScale::backbone` string
// (set in `rfdetr.hpp`) selects one of these constants.
struct BackboneCfg {
  int  patch_size   = 14;
  int  embed_dim    = 1024;
  int  depth        = 24;
  int  num_heads    = 16;
  int  mlp_ratio_x4 = 4;        // multiplier × 4 / 4 to stay integer
  int  window_size  = 0;        // 0 = full self-attention (DINOv2)
  int  img_size     = 560;
  // Multi-scale feature map block indices (1-based in upstream;
  // 0-based here). For a 24-block ViT-L: blocks 11, 17, 23 →
  // strides 8/16/32 via patch_size=14 + a 2×2 tap downsample on the
  // earlier taps. The encoder (#65B) will own the actual stride
  // adaptation; this list just says which block outputs to capture.
  std::vector<int> tap_blocks  = {11, 17, 23};
};

extern const BackboneCfg kDinoV2LargeCfg;
extern const BackboneCfg kLwDetrTinyCfg;
extern const BackboneCfg kLwDetrSmallCfg;
extern const BackboneCfg kLwDetrBaseCfg;
extern const BackboneCfg kLwDetrMediumCfg;

const BackboneCfg& backbone_cfg_from_name(const std::string& backbone);

// ─── ViT building blocks ────────────────────────────────────────────────

// Patch-embedding: `Conv2d(3, embed_dim, k=patch, s=patch)` then flatten
// to `[B, N, embed_dim]` with a learnable cls + position embedding.
// Used by both DINOv2 and LW-DETR.
class PatchEmbedImpl : public torch::nn::Module {
 public:
  PatchEmbedImpl(int in_ch, int embed_dim, int patch_size, int img_size);
  torch::Tensor forward(torch::Tensor x);
  int           grid_size() const { return grid_size_; }

 private:
  torch::nn::Conv2d proj_{nullptr};
  int grid_size_;
};
TORCH_MODULE(PatchEmbed);

// Multi-head self-attention (optionally windowed). Standard
// Q/K/V = Linear(C, 3C); attn = softmax(QKᵀ/√d); out = Linear(C, C).
// When `window_size > 0`, input is reshaped to `[B·nW, win·win, C]`
// before attention and unshaped after (window-style attention from
// Swin / LW-DETR).
class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(int dim, int num_heads, int window_size, int grid_size);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Linear qkv_{nullptr};
  torch::nn::Linear proj_{nullptr};
  int num_heads_;
  int head_dim_;
  int window_size_;
  int grid_size_;
};
TORCH_MODULE(Attention);

// LN → MHA → residual → LN → MLP → residual. MLP = Linear(C, 4C) →
// GELU → Linear(4C, C).
class ViTBlockImpl : public torch::nn::Module {
 public:
  ViTBlockImpl(int dim, int num_heads, int mlp_ratio, int window_size,
               int grid_size);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::LayerNorm norm1_{nullptr};
  Attention            attn_{nullptr};
  torch::nn::LayerNorm norm2_{nullptr};
  torch::nn::Linear    fc1_{nullptr};
  torch::nn::Linear    fc2_{nullptr};
};
TORCH_MODULE(ViTBlock);

// Full backbone module — patch_embed → N×ViTBlock → tap-block
// captures → final LN. Returns a vector of feature maps in the order
// requested by `cfg.tap_blocks` (innermost → outermost stride).
class ViTBackboneImpl : public torch::nn::Module {
 public:
  explicit ViTBackboneImpl(const BackboneCfg& cfg);

  // Returns one `[B, C, Hi, Wi]` per tap. Reshapes from the
  // `[B, N, C]` token form via the patch grid.
  std::vector<torch::Tensor> forward_features(torch::Tensor x);

  const BackboneCfg& cfg() const { return cfg_; }

 private:
  BackboneCfg cfg_;
  PatchEmbed  patch_embed_{nullptr};
  torch::Tensor cls_token_;
  torch::Tensor pos_embed_;
  // ModuleList auto-names children "0", "1", … so the full
  // qualified path becomes "blocks.0.attn.qkv.weight" — same as
  // upstream, which is what #65D's state-dict converter relies on.
  torch::nn::ModuleList blocks_{nullptr};
  torch::nn::LayerNorm  norm_{nullptr};
};
TORCH_MODULE(ViTBackbone);

}  // namespace yolocpp::models::rfdetr
