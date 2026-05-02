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
  // query: [B, Q, C]; reference_points: [B, Q, 4] (cxcywh in [0,1]);
  // memory: [B, L, C] flattened; spatial_h/w: scalar feature-map dims.
  torch::Tensor forward(torch::Tensor query,
                         torch::Tensor reference_points,
                         torch::Tensor memory,
                         int spatial_h, int spatial_w);
  torch::nn::Linear sampling_offsets{nullptr};   // [H·L·P·2, hidden]
  torch::nn::Linear attention_weights{nullptr};  // [H·L·P,   hidden]
  torch::nn::Linear value_proj{nullptr};
  torch::nn::Linear output_proj{nullptr};
 private:
  int hidden_;
  int num_heads_;
  int num_points_;
  int head_dim_;
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
  // Post-LN style: tgt → self-attn → cross-attn → FFN with residuals.
  torch::Tensor forward(torch::Tensor tgt, torch::Tensor query_pos,
                         torch::Tensor reference_points,
                         torch::Tensor memory, int spatial_h, int spatial_w);
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
  // tgt: [B, Q, C] initial query feats; refpoints: [B, Q, 4] cxcywh in [0,1].
  // memory: [B, L, C] flattened single-level feature; spatial_h/w: dims.
  // Returns last-layer output [B, Q, C] after final LN.
  torch::Tensor forward(torch::Tensor tgt, torch::Tensor refpoints,
                         torch::Tensor memory, int spatial_h, int spatial_w);
  torch::nn::ModuleList layers{nullptr};
  torch::nn::LayerNorm  norm{nullptr};
  RFDetrMLP             ref_point_head{nullptr};
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

// Output of the transformer at eval time.
struct TransformerOutput {
  torch::Tensor decoder_out;   // [B, Q, C] after final LN
  torch::Tensor refpoints;     // [B, Q, 4] cxcywh in [0,1]
};

class RFDetrTransformerImpl : public torch::nn::Module {
 public:
  RFDetrTransformerImpl(int hidden, int n_dec_layers, int sa_nheads,
                         int ca_nheads, int dec_n_points, int ffn_dim,
                         int group_detr, int n_classes_with_bg);
  // Two-stage encoder-output → top-K query selection → iterative
  // decoder refinement. `memory_2d`: [B, C, Hg, Wg] from projector.
  // `query_feat_first_group`: [Q, C] pre-sliced learned query feats
  // (= query_feat[:Q] from the top-level Parameter at eval).
  // Returns the last-layer query output + the final refined refpoints.
  TransformerOutput forward(torch::Tensor memory_2d,
                              torch::Tensor query_feat_first_group,
                              int num_queries);
  RFDetrDecoder    decoder{nullptr};
  torch::nn::ModuleList enc_output{nullptr};
  torch::nn::ModuleList enc_output_norm{nullptr};
  torch::nn::ModuleList enc_out_class_embed{nullptr};
  torch::nn::ModuleList enc_out_bbox_embed{nullptr};
 private:
  int hidden_;
};
TORCH_MODULE(RFDetrTransformer);

// Helpers (also useful for tests).
//
// gen_sineembed_for_position(refpoints, dim) — refpoints [B, Q, 4]
// → [B, Q, 4*2*dim/2] = [B, Q, 4*dim] sinusoidal embedding.
torch::Tensor gen_sineembed_for_position(const torch::Tensor& pos_tensor,
                                           int dim);

// Generates the dense per-token bbox proposals + masked memory for
// the two-stage encoder. Matches upstream's
// `gen_encoder_output_proposals` for a single feature level with
// `unsigmoid=False` (bbox_reparam mode).
struct EncoderProposals {
  torch::Tensor output_memory;     // [B, L, C]
  torch::Tensor output_proposals;  // [B, L, 4] cxcywh in [0,1]
};
EncoderProposals gen_encoder_output_proposals_1l(const torch::Tensor& memory,
                                                  int H, int W);

}  // namespace yolocpp::models::rfdetr
