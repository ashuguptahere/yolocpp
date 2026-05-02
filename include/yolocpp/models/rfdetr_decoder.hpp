#pragma once
//
// RF-DETR decoder + object-query head (#65C).
//
// ─── Architecture ───────────────────────────────────────────────────
//
// The decoder runs N transformer decoder layers over `num_queries`
// learnable object query embeddings. Each layer:
//
//   1. Self-attn over queries (vanilla MHA).
//   2. Cross-attn from queries to encoder memory via MSDeformAttn —
//      same primitive as the encoder, but the keys/values are the
//      enriched encoder memory and the reference points are
//      per-query learnable offsets refined layer-by-layer.
//   3. FFN.
//
// The head is a small MLP per task:
//
//   * Cls head: `Linear(hidden, nc)` — sigmoided at output.
//   * Bbox head: `MLP(hidden → hidden → hidden → 4)` predicting
//     `(cx, cy, w, h)` deltas in [0, 1] (sigmoided), refined
//     iteratively across decoder layers.
//
// At eval, only the last layer's output is kept; at train, all
// layers' outputs participate in the auxiliary Hungarian loss
// (#65F).
//
// Output of `forward_eval(memory, ...)`:
//   `[B, 4 + nc, num_queries]` — same channel order as YOLO so the
//   downstream NMS-free decoder (#65E) can stay generic.

#include <torch/torch.h>

#include <vector>

#include "yolocpp/models/rfdetr_encoder.hpp"

namespace yolocpp::models::rfdetr {

// Standard MLP used for the bbox head: 3 hidden layers.
class MLPImpl : public torch::nn::Module {
 public:
  MLPImpl(int input_dim, int hidden_dim, int output_dim, int num_layers);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::ModuleList layers_{nullptr};
};
TORCH_MODULE(MLP);

class DecoderLayerImpl : public torch::nn::Module {
 public:
  DecoderLayerImpl(int dim, int num_heads, int num_levels, int num_points,
                    int mlp_ratio);

  torch::Tensor forward(torch::Tensor query, torch::Tensor query_pos,
                         torch::Tensor reference_points,
                         torch::Tensor memory, torch::Tensor spatial_shapes,
                         torch::Tensor level_start_index);

 private:
  // Self-attn block (vanilla MHA).
  torch::nn::LayerNorm   norm1_{nullptr};
  torch::nn::Linear      sa_qkv_{nullptr};
  torch::nn::Linear      sa_proj_{nullptr};
  // Cross-attn block (deformable).
  torch::nn::LayerNorm   norm2_{nullptr};
  MSDeformAttn           cross_attn_{nullptr};
  // FFN.
  torch::nn::LayerNorm   norm3_{nullptr};
  torch::nn::Linear      fc1_{nullptr};
  torch::nn::Linear      fc2_{nullptr};
  int dim_;
  int num_heads_;
  int head_dim_;
};
TORCH_MODULE(DecoderLayer);

// Full decoder + heads. Owns the learnable object queries and
// reference points, the stack of decoder layers, and the per-layer
// bbox/cls heads.
class DetrHeadImpl : public torch::nn::Module {
 public:
  DetrHeadImpl(int hidden_dim, int num_heads, int num_layers,
                int num_queries, int nc, int num_points = 4,
                int mlp_ratio = 4);

  // memory:           [B, ΣHW, hidden]
  // spatial_shapes:   [L, 2] int64
  // level_start_index:[L] int64
  // Returns YOLO-shaped detection tensor `[B, 4+nc, Q]` with
  // sigmoided cls and `(cx, cy, w, h)` in [0, 1] image-normalised
  // coordinates. Caller scales to pixel coords.
  torch::Tensor forward_eval(torch::Tensor memory,
                              torch::Tensor spatial_shapes,
                              torch::Tensor level_start_index);

  // Returns the full per-layer (cls_logits, bbox_deltas) for use by
  // the Hungarian loss (#65F). Last layer's output mirrors
  // `forward_eval` (modulo the sigmoid).
  std::vector<std::pair<torch::Tensor, torch::Tensor>>
  forward_train(torch::Tensor memory, torch::Tensor spatial_shapes,
                torch::Tensor level_start_index);

  int num_queries() const { return num_queries_; }
  int nc()          const { return nc_; }

 private:
  int hidden_dim_;
  int num_queries_;
  int nc_;
  int num_levels_ = 0;  // set on first forward, stays consistent

  torch::Tensor          query_embed_;       // [Q, hidden]
  torch::Tensor          query_pos_;         // [Q, hidden]
  torch::Tensor          ref_points_unact_;  // [Q, 2] (logit-space)
  torch::nn::ModuleList  layers_{nullptr};
  torch::nn::ModuleList  cls_heads_{nullptr};
  torch::nn::ModuleList  bbox_heads_{nullptr};
};
TORCH_MODULE(DetrHead);

}  // namespace yolocpp::models::rfdetr
