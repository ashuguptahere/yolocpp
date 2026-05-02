#pragma once
//
// RF-DETR encoder (#65B).
//
// Takes the multi-scale feature maps from the ViT backbone (#65A)
// and runs them through N transformer encoder layers with
// **multi-scale deformable attention** (Zhu et al., 2020 — also
// called "MSDeformAttn"). Output is the per-token enriched feature
// sequence that the decoder + object queries (#65C) consume.
//
// ─── Key design notes ─────────────────────────────────────────────
//
// 1. **Channel projection.** Backbone taps land at the backbone's
//    embed_dim (192/384/512/768/1024). The encoder runs at
//    `RFDetrScale::hidden_dim` (192/256/256/384/512). A 1×1 Conv per
//    tap level projects the channel count.
//
// 2. **Multi-scale flatten.** All L tap maps are flattened and
//    concatenated along the token axis: `[B, ΣHi·Wi, hidden_dim]`.
//    `level_start_index` and `spatial_shapes` track per-level slices
//    so deformable-attn can sample across levels. Sine positional
//    encoding (2D, level-aware) is added.
//
// 3. **Deformable attention** — the heart of the encoder. Each
//    query learns `num_points × num_levels` 2D sampling offsets +
//    attention weights; values are gathered via bilinear `grid_sample`
//    on each level's feature map. We implement it as a portable
//    composition of standard ops (Linear → grid_sample → weighted
//    sum) — no custom CUDA kernel. ONNX export (#65I) decomposes
//    into the same op graph.
//
// 4. **Encoder layer** — standard pre-norm transformer block but
//    with deformable cross-attn instead of vanilla MHA in the
//    self-attn slot. LN → MSDeformAttn → Add → LN → FFN(MLP) → Add.
//
// Output shape: same as input flat token sequence, plus the
// spatial_shapes / level_start_index helpers (passed to the decoder
// for cross-attn).

#include <torch/torch.h>

#include <vector>

namespace yolocpp::models::rfdetr {

// Multi-scale deformable attention. `num_levels` = number of feature
// pyramid levels, `num_points` = sampling points per level.
class MSDeformAttnImpl : public torch::nn::Module {
 public:
  MSDeformAttnImpl(int dim, int num_heads, int num_levels, int num_points);

  // query / value:        [B, Lq, C] / [B, Lv, C]
  // reference_points:     [B, Lq, num_levels, 2] in [0, 1]
  // spatial_shapes:       [num_levels, 2] (Hi, Wi) on int64
  // level_start_index:    [num_levels] on int64
  torch::Tensor forward(torch::Tensor query, torch::Tensor value,
                        torch::Tensor reference_points,
                        torch::Tensor spatial_shapes,
                        torch::Tensor level_start_index);

 private:
  torch::nn::Linear sampling_offsets_{nullptr};
  torch::nn::Linear attention_weights_{nullptr};
  torch::nn::Linear value_proj_{nullptr};
  torch::nn::Linear output_proj_{nullptr};
  int dim_;
  int num_heads_;
  int num_levels_;
  int num_points_;
  int head_dim_;
};
TORCH_MODULE(MSDeformAttn);

// One encoder layer: pre-norm, MSDeformAttn → FFN.
class EncoderLayerImpl : public torch::nn::Module {
 public:
  EncoderLayerImpl(int dim, int num_heads, int num_levels, int num_points,
                    int mlp_ratio);

  torch::Tensor forward(torch::Tensor x, torch::Tensor pos,
                         torch::Tensor reference_points,
                         torch::Tensor spatial_shapes,
                         torch::Tensor level_start_index);

 private:
  torch::nn::LayerNorm norm1_{nullptr};
  MSDeformAttn         self_attn_{nullptr};
  torch::nn::LayerNorm norm2_{nullptr};
  torch::nn::Linear    fc1_{nullptr};
  torch::nn::Linear    fc2_{nullptr};
};
TORCH_MODULE(EncoderLayer);

// Full encoder. Owns the per-level 1×1 input projections, the stack
// of encoder layers, and the per-level positional embeddings.
//
// Output: enriched flat token sequence + the helpers the decoder
// needs (`spatial_shapes`, `level_start_index`) for cross-attn.
struct EncoderOutput {
  torch::Tensor memory;             // [B, ΣHi·Wi, hidden]
  torch::Tensor spatial_shapes;     // [L, 2] int64
  torch::Tensor level_start_index;  // [L] int64
  torch::Tensor reference_points;   // [B, ΣHi·Wi, L, 2] for decoder reuse
  torch::Tensor pos_embed;          // [B, ΣHi·Wi, hidden]
};

class EncoderImpl : public torch::nn::Module {
 public:
  EncoderImpl(const std::vector<int>& in_channels, int hidden_dim,
              int num_heads, int num_layers, int num_points = 4,
              int mlp_ratio = 4);

  EncoderOutput forward(const std::vector<torch::Tensor>& features);

 private:
  torch::nn::ModuleList input_proj_{nullptr};   // 1×1 convs per level
  torch::nn::ModuleList layers_{nullptr};
  int hidden_dim_;
  int num_levels_;
};
TORCH_MODULE(Encoder);

}  // namespace yolocpp::models::rfdetr
