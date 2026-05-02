#pragma once
//
// RF-DETR 1.6.5 transformer (#65C2 + #65D2).
//
// Replaces the encoder + decoder scaffold (0.33.0..0.37.0) with the
// real upstream layout — fused-QKV self-attn, single-level
// deformable cross-attn, shared cls/bbox heads with iterative
// refinement, and the two-stage encoder-output that initialises
// queries from backbone tokens.
//
// ─── Parameter naming (matches upstream exactly) ─────────────────────────
//
//   transformer.decoder.layers.<j>.self_attn.in_proj_weight        [3·C, C]
//   transformer.decoder.layers.<j>.self_attn.in_proj_bias          [3·C]
//   transformer.decoder.layers.<j>.self_attn.out_proj.{weight,bias} [C, C]
//   transformer.decoder.layers.<j>.norm1.{weight,bias}             [C]
//   transformer.decoder.layers.<j>.cross_attn.sampling_offsets.{weight,bias}
//                                       [ca_nheads × levels × points × 2, C]
//   transformer.decoder.layers.<j>.cross_attn.attention_weights.{weight,bias}
//                                       [ca_nheads × levels × points,    C]
//   transformer.decoder.layers.<j>.cross_attn.value_proj.{weight,bias}   [C, C]
//   transformer.decoder.layers.<j>.cross_attn.output_proj.{weight,bias}  [C, C]
//   transformer.decoder.layers.<j>.linear1.{weight,bias}                 [2048, C]
//   transformer.decoder.layers.<j>.linear2.{weight,bias}                 [C, 2048]
//   transformer.decoder.layers.<j>.norm2.{weight,bias}                   [C]
//   transformer.decoder.layers.<j>.norm3.{weight,bias}                   [C]
//   transformer.decoder.norm.{weight,bias}                               [C]
//
//   transformer.enc_output.<k>.{weight,bias}                              [C, C]   (k ∈ 0..12)
//   transformer.enc_output_norm.<k>.{weight,bias}                         [C]
//   transformer.enc_out_class_embed.<k>.{weight,bias}                     [91, C]
//   transformer.enc_out_bbox_embed.<k>.layers.{0,1,2}.{weight,bias}       [hidden→hidden→hidden→4]
//
// And the top-level shared heads on RFDetrImpl (siblings of
// `transformer` and `backbone`):
//
//   class_embed.{weight,bias}                                             [91, C]
//   bbox_embed.layers.{0,1,2}.{weight,bias}                               [hidden→…→4]
//   refpoint_embed.weight                                                 [Q·group_detr, 4]
//   query_feat.weight                                                     [Q·group_detr, hidden]
//
// `forward()` semantics are *not yet* implemented — only parameter
// registration. The legacy scaffold's forward keeps running until
// #65C3 / #65F2 close the loop.

#include <torch/torch.h>

#include <vector>

namespace yolocpp::models::rfdetr {

// ─── Building blocks ────────────────────────────────────────────────────

// Fused-QKV multi-head self-attention. Stores `in_proj_weight`/
// `in_proj_bias` directly as parameters (matching upstream's
// `nn.MultiheadAttention` checkpoint format) and an `out_proj`
// linear. No forward implemented — registers params only.
class FusedMHAImpl : public torch::nn::Module {
 public:
  FusedMHAImpl(int hidden, int num_heads);
  // Vanilla scaled-dot-product attention. Q=K=V (self-attn).
  torch::Tensor forward(torch::Tensor x);
  torch::Tensor      in_proj_weight;   // [3·hidden, hidden]
  torch::Tensor      in_proj_bias;     // [3·hidden]
  torch::nn::Linear  out_proj{nullptr};
 private:
  int hidden_;
  int num_heads_;
};
TORCH_MODULE(FusedMHA);

// Single-level multi-scale deformable attention (`num_levels=1`).
// Same primitive as the upstream `MSDeformableAttention` op.
class MSDeformAttn1LImpl : public torch::nn::Module {
 public:
  MSDeformAttn1LImpl(int hidden, int num_heads, int num_points);
  // Forward not implemented yet — placeholder for #65F2.
  torch::nn::Linear sampling_offsets{nullptr};   // [H·L·P·2, hidden]
  torch::nn::Linear attention_weights{nullptr};  // [H·L·P,   hidden]
  torch::nn::Linear value_proj{nullptr};
  torch::nn::Linear output_proj{nullptr};
};
TORCH_MODULE(MSDeformAttn1L);

// 3-layer MLP matching `bbox_embed.layers.{0,1,2}` upstream layout.
class RFDetrMLPImpl : public torch::nn::Module {
 public:
  RFDetrMLPImpl(int input_dim, int hidden_dim, int output_dim,
                 int num_layers);
  torch::Tensor forward(torch::Tensor x);
  torch::nn::ModuleList layers{nullptr};
};
TORCH_MODULE(RFDetrMLP);

// One decoder layer = self-attn + cross-attn + FFN, three LayerNorms.
class RFDetrDecoderLayerImpl : public torch::nn::Module {
 public:
  RFDetrDecoderLayerImpl(int hidden, int sa_nheads, int ca_nheads,
                          int dec_n_points, int ffn_dim);
  FusedMHA             self_attn{nullptr};
  torch::nn::LayerNorm norm1{nullptr};
  MSDeformAttn1L       cross_attn{nullptr};
  torch::nn::Linear    linear1{nullptr};
  torch::nn::Linear    linear2{nullptr};
  torch::nn::LayerNorm norm2{nullptr};
  torch::nn::LayerNorm norm3{nullptr};
};
TORCH_MODULE(RFDetrDecoderLayer);

class RFDetrDecoderImpl : public torch::nn::Module {
 public:
  RFDetrDecoderImpl(int n_layers, int hidden, int sa_nheads, int ca_nheads,
                     int dec_n_points, int ffn_dim);
  torch::nn::ModuleList layers{nullptr};
  torch::nn::LayerNorm  norm{nullptr};
  // Embeds the 4D refpoints into hidden-dim feature space. Two
  // Linear layers: input is `2·hidden` (sin/cos sinusoidal embed of
  // (cx, cy, w, h)) → hidden → hidden. Path:
  // `transformer.decoder.ref_point_head.layers.{0,1}.{weight,bias}`.
  RFDetrMLP ref_point_head{nullptr};
};
TORCH_MODULE(RFDetrDecoder);

// Two-stage encoder-output stack: `group_detr` per-group projections
// + LNs + cls/bbox heads. Used at inference to seed the decoder's
// initial reference points from backbone-token cls/bbox predictions.
class RFDetrEncOutputImpl : public torch::nn::Module {
 public:
  RFDetrEncOutputImpl(int group_detr, int hidden, int n_classes_with_bg);
  torch::nn::ModuleList enc_output{nullptr};         // 13 × Linear(C, C)
  torch::nn::ModuleList enc_output_norm{nullptr};    // 13 × LayerNorm(C)
  torch::nn::ModuleList enc_out_class_embed{nullptr};// 13 × Linear(C, n_classes)
  torch::nn::ModuleList enc_out_bbox_embed{nullptr}; // 13 × RFDetrMLP(C→C→C→4)
};
TORCH_MODULE(RFDetrEncOutput);

// Top-level transformer: owns the decoder + the enc-output stack.
// Acts purely as a parameter container at #65C2 (forward not wired).
class RFDetrTransformerImpl : public torch::nn::Module {
 public:
  RFDetrTransformerImpl(int hidden, int n_dec_layers, int sa_nheads,
                         int ca_nheads, int dec_n_points, int ffn_dim,
                         int group_detr, int n_classes_with_bg);
  RFDetrDecoder    decoder{nullptr};
  // Encoder-output siblings — flattened directly under
  // `transformer.*` (NOT wrapped in a sub-module) to match upstream's
  // `transformer.enc_output.<k>` paths.
  torch::nn::ModuleList enc_output{nullptr};
  torch::nn::ModuleList enc_output_norm{nullptr};
  torch::nn::ModuleList enc_out_class_embed{nullptr};
  torch::nn::ModuleList enc_out_bbox_embed{nullptr};
};
TORCH_MODULE(RFDetrTransformer);

}  // namespace yolocpp::models::rfdetr
