#pragma once
//
// RF-DETR 1.6.5 backbone (#65A2) — HF-DINOv2 windowed-attention.
//
// Replaces the LW-DETR placeholder scaffold (0.32.0..0.37.0). Built
// to match the upstream HuggingFace `Dinov2WithRegistersModel`
// structure used in `rfdetr.models.backbone.dinov2_with_windowed_attn`,
// so the parameter dotted-paths line up 1-to-1 with the keys in
// `rf-detr-{nano,small,medium,base,large}.pth` and their seg
// counterparts.
//
// ─── Parameter naming (matches upstream exactly) ─────────────────────────
//
//   embeddings.cls_token                                      [1, 1, C]
//   embeddings.mask_token                                     [1, C]
//   embeddings.position_embeddings                            [1, N+1, C]
//   embeddings.patch_embeddings.projection.{weight,bias}      Conv2d(3,C,k=p,s=p)
//   encoder.layer.<i>.norm1.{weight,bias}                     LN
//   encoder.layer.<i>.attention.attention.query.{weight,bias} Linear(C,C)
//   encoder.layer.<i>.attention.attention.key.{...}           Linear(C,C)
//   encoder.layer.<i>.attention.attention.value.{...}         Linear(C,C)
//   encoder.layer.<i>.attention.output.dense.{weight,bias}    Linear(C,C)
//   encoder.layer.<i>.layer_scale1.lambda1                    [C]  (param)
//   encoder.layer.<i>.norm2.{weight,bias}                     LN
//   encoder.layer.<i>.mlp.fc1.{weight,bias}                   Linear(C,4C)
//   encoder.layer.<i>.mlp.fc2.{weight,bias}                   Linear(4C,C)
//   encoder.layer.<i>.layer_scale2.lambda1                    [C]
//   layernorm.{weight,bias}                                   LN
//
// Two extra wrapper Modules (`Dinov2Wrapper`, `Dinov2WrapperOuter`)
// reproduce the upstream `backbone.0.encoder.encoder.*` double-wrapping.
// `RFDetrImpl` registers a `ModuleList` named "backbone" whose first
// child is the outer wrapper, so the full path becomes
// `backbone.0.encoder.encoder.embeddings...` — matches upstream key
// names exactly so `rfdetr_weights::load_rfdetr_pt` binds in one
// pass.
//
// ─── Per-variant config ──────────────────────────────────────────────────
//
//   small (n/s/m/b/seg-* up to xxlarge): C=384, depth=12, heads=6,
//                                         mlp_ratio=4
//   base  (rfdetr-large only):           C=768, depth=12, heads=12,
//                                         mlp_ratio=4
//
// All variants use 12 ViT blocks. RF-DETR runs them at letterbox
// resolutions of {384..768} so the position embedding is interpolated
// at runtime to match the actual input grid size; the saved tensor
// stays at the upstream-dumped shape (typically 37×37+1=1370 patches
// for patch=14, or per-variant for patch=16/12).

#include <torch/torch.h>

#include <string>
#include <vector>

namespace yolocpp::models::rfdetr {

struct Dinov2Cfg {
  int  hidden_size = 384;
  int  num_layers  = 12;
  int  num_heads   = 6;
  int  mlp_ratio   = 4;
  int  patch_size  = 14;
  int  pretrain_grid = 37;     // size of the saved position_embeddings (P×P+1 tokens)
  bool qkv_bias    = true;
  // Indices (1-indexed in upstream config; we store 0-indexed) of
  // ViT blocks whose output is captured as a feature tap. Default
  // `[3,6,9,12]` (config) → `[2,5,8,11]` here.
  std::vector<int> tap_blocks = {2, 5, 8, 11};
  // Windowed-attention partitioning. After embeddings, the spatial
  // grid is reshaped into `num_windows × num_windows` non-overlapping
  // windows, each becoming an independent batch sample for windowed
  // attention. At blocks NOT in `window_block_indexes`, the windows
  // are merged (full self-attention across the entire grid) and
  // re-split after.
  int              num_windows = 1;
  std::vector<int> window_block_indexes;
};

extern const Dinov2Cfg kDinov2Small;   // C=384 (default)
extern const Dinov2Cfg kDinov2Base;    // C=768 — used by rfdetr-large

// Return the right per-variant config based on `RFDetrScale.upstream_id`,
// patch size, and pretrain grid (the saved `position_embeddings`'
// spatial size). Backbone embed dim is selected from the variant
// (`backbone_embed`) — overrides the family's default.
const Dinov2Cfg& dinov2_cfg_for(const std::string& upstream_id, int patch,
                                 int pretrain_grid, int backbone_embed);

// ─── Inner blocks ────────────────────────────────────────────────────────

class Dinov2PatchEmbeddingsImpl : public torch::nn::Module {
 public:
  Dinov2PatchEmbeddingsImpl(int in_ch, int hidden, int patch);
  torch::Tensor forward(torch::Tensor x);
  torch::nn::Conv2d projection{nullptr};
};
TORCH_MODULE(Dinov2PatchEmbeddings);

class Dinov2EmbeddingsImpl : public torch::nn::Module {
 public:
  Dinov2EmbeddingsImpl(const Dinov2Cfg& cfg);
  // Returns `[B*num_windows², (Hg/W)*(Wg/W)+1, C]` token sequence
  // with 2D-interpolated pos embedding interpolated to the input
  // grid, then split into `num_windows²` windows (each gets its
  // own cls token broadcast).
  torch::Tensor forward(torch::Tensor x);
  torch::Tensor cls_token;
  torch::Tensor mask_token;
  torch::Tensor position_embeddings;
  Dinov2PatchEmbeddings patch_embeddings{nullptr};
 private:
  int patch_size_;
  int pretrain_grid_;
  int num_windows_;
};
TORCH_MODULE(Dinov2Embeddings);

class Dinov2SelfAttentionImpl : public torch::nn::Module {
 public:
  Dinov2SelfAttentionImpl(int hidden, int heads, bool bias);
  torch::Tensor forward(torch::Tensor x);
  torch::nn::Linear query{nullptr};
  torch::nn::Linear key{nullptr};
  torch::nn::Linear value{nullptr};
 private:
  int heads_;
  int head_dim_;
};
TORCH_MODULE(Dinov2SelfAttention);

class Dinov2SelfOutputImpl : public torch::nn::Module {
 public:
  Dinov2SelfOutputImpl(int hidden);
  torch::Tensor forward(torch::Tensor x);
  torch::nn::Linear dense{nullptr};
};
TORCH_MODULE(Dinov2SelfOutput);

class Dinov2AttentionImpl : public torch::nn::Module {
 public:
  Dinov2AttentionImpl(int hidden, int heads, bool bias);
  torch::Tensor forward(torch::Tensor x);
  Dinov2SelfAttention attention{nullptr};
  Dinov2SelfOutput    output{nullptr};
};
TORCH_MODULE(Dinov2Attention);

class Dinov2LayerScaleImpl : public torch::nn::Module {
 public:
  Dinov2LayerScaleImpl(int hidden);
  torch::Tensor forward(torch::Tensor x);
  torch::Tensor lambda1;
};
TORCH_MODULE(Dinov2LayerScale);

class Dinov2MLPImpl : public torch::nn::Module {
 public:
  Dinov2MLPImpl(int hidden, int mlp_ratio);
  torch::Tensor forward(torch::Tensor x);
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
};
TORCH_MODULE(Dinov2MLP);

class Dinov2LayerImpl : public torch::nn::Module {
 public:
  Dinov2LayerImpl(const Dinov2Cfg& cfg);
  // `run_full_attention=true` un-windows before self-attn and
  // re-windows after (matches `WindowedDinov2WithRegistersLayer`
  // upstream when block index is NOT in `window_block_indexes`).
  torch::Tensor forward(torch::Tensor x, bool run_full_attention);
  torch::nn::LayerNorm norm1{nullptr};
  Dinov2Attention      attention{nullptr};
  Dinov2LayerScale     layer_scale1{nullptr};
  torch::nn::LayerNorm norm2{nullptr};
  Dinov2MLP            mlp{nullptr};
  Dinov2LayerScale     layer_scale2{nullptr};
 private:
  int num_windows_;
};
TORCH_MODULE(Dinov2Layer);

class Dinov2EncoderImpl : public torch::nn::Module {
 public:
  Dinov2EncoderImpl(const Dinov2Cfg& cfg);
  // Returns the per-tap features in WINDOWED token form
  // `[B*W², N_w+1, C]`. Caller is responsible for unwindowing back
  // to the full grid.
  std::vector<torch::Tensor> forward(torch::Tensor x);
  torch::nn::ModuleList layer{nullptr};
 private:
  std::vector<int>             taps_;
  std::vector<bool>            is_windowed_;   // per-block flag
};
TORCH_MODULE(Dinov2Encoder);

// `Dinov2Model` = embeddings + encoder + final layernorm.
class Dinov2ModelImpl : public torch::nn::Module {
 public:
  Dinov2ModelImpl(const Dinov2Cfg& cfg);
  // Returns the tap features in spatial form (`[B, C, Hg, Wg]`),
  // ready for the projector.
  std::vector<torch::Tensor> forward(torch::Tensor x);
  Dinov2Embeddings     embeddings{nullptr};
  Dinov2Encoder        encoder{nullptr};
  torch::nn::LayerNorm layernorm{nullptr};
};
TORCH_MODULE(Dinov2Model);

// `Dinov2Wrapper` adds the inner `encoder` namespace level so the
// effective param paths become `<parent>.encoder.embeddings.*`.
class Dinov2WrapperImpl : public torch::nn::Module {
 public:
  Dinov2WrapperImpl(const Dinov2Cfg& cfg);
  std::vector<torch::Tensor> forward(torch::Tensor x);
  Dinov2Model encoder{nullptr};   // intentional: param paths need 'encoder.encoder.*'
};
TORCH_MODULE(Dinov2Wrapper);

// `Dinov2WrapperOuter` adds another level: param paths become
// `<parent>.encoder.encoder.embeddings.*`. Combined with `RFDetrImpl`
// registering this under a `ModuleList` named "backbone" (slot 0),
// the full path matches upstream's `backbone.0.encoder.encoder.*`.
class Dinov2WrapperOuterImpl : public torch::nn::Module {
 public:
  Dinov2WrapperOuterImpl(const Dinov2Cfg& cfg);
  std::vector<torch::Tensor> forward(torch::Tensor x);
  Dinov2Wrapper encoder{nullptr};
};
TORCH_MODULE(Dinov2WrapperOuter);

// ─── Backwards-compatibility shim ────────────────────────────────────────
//
// The earlier scaffold defined `BackboneCfg` + `ViTBackbone` which
// downstream scaffolded modules (encoder, decoder, head) still
// reference. These are kept (with a smaller/dummier shape) so the
// rest of the scaffold continues to build until #65B2/C2/D2 land
// their replacements. The actual RF-DETR backbone is exposed via
// the `Dinov2*` modules above.
struct BackboneCfg {
  int  patch_size   = 14;
  int  embed_dim    = 384;
  int  depth        = 12;
  int  num_heads    = 6;
  int  mlp_ratio_x4 = 4;
  int  window_size  = 0;
  int  img_size     = 560;
  std::vector<int> tap_blocks = {2, 5, 8, 11};
};

extern const BackboneCfg kDinoV2LargeCfg;
extern const BackboneCfg kLwDetrTinyCfg;
extern const BackboneCfg kLwDetrSmallCfg;
extern const BackboneCfg kLwDetrBaseCfg;
extern const BackboneCfg kLwDetrMediumCfg;
const BackboneCfg& backbone_cfg_from_name(const std::string& backbone);

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

class ViTBackboneImpl : public torch::nn::Module {
 public:
  explicit ViTBackboneImpl(const BackboneCfg& cfg);
  std::vector<torch::Tensor> forward_features(torch::Tensor x);
  const BackboneCfg& cfg() const { return cfg_; }
 private:
  BackboneCfg cfg_;
  PatchEmbed  patch_embed_{nullptr};
  torch::Tensor cls_token_;
  torch::Tensor pos_embed_;
  torch::nn::ModuleList blocks_{nullptr};
  torch::nn::LayerNorm  norm_{nullptr};
};
TORCH_MODULE(ViTBackbone);

}  // namespace yolocpp::models::rfdetr
