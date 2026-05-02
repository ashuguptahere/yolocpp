// RF-DETR decoder + head (#65C).
//
// Two-pass forward: query embeddings + learnable reference points
// flow through N decoder layers; each layer outputs a refined
// (cls_logits, bbox_deltas) pair. Eval keeps only the last layer's
// output and reshapes to YOLO's `[B, 4+nc, Q]` contract. Train
// returns all layers for auxiliary Hungarian loss (#65F).

#include "yolocpp/models/rfdetr_decoder.hpp"

#include <cmath>

namespace yolocpp::models::rfdetr {

// ─── MLP ────────────────────────────────────────────────────────────────

MLPImpl::MLPImpl(int input_dim, int hidden_dim, int output_dim,
                  int num_layers) {
  layers_ = torch::nn::ModuleList();
  for (int i = 0; i < num_layers; ++i) {
    int in  = (i == 0) ? input_dim  : hidden_dim;
    int out = (i == num_layers - 1) ? output_dim : hidden_dim;
    layers_->push_back(torch::nn::Linear(in, out));
  }
  register_module("layers", layers_);
}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
  int n = static_cast<int>(layers_->size());
  for (int i = 0; i < n; ++i) {
    x = layers_[i]->as<torch::nn::LinearImpl>()->forward(x);
    if (i < n - 1) x = torch::relu(x);
  }
  return x;
}

// ─── DecoderLayer ───────────────────────────────────────────────────────

DecoderLayerImpl::DecoderLayerImpl(int dim, int num_heads, int num_levels,
                                     int num_points, int mlp_ratio)
    : dim_(dim), num_heads_(num_heads), head_dim_(dim / num_heads) {
  TORCH_CHECK(dim % num_heads == 0, "decoder dim must divide num_heads");
  norm1_       = register_module("norm1", torch::nn::LayerNorm(
                                                torch::nn::LayerNormOptions({dim})));
  sa_qkv_      = register_module("sa_qkv",
                                  torch::nn::Linear(torch::nn::LinearOptions(dim, dim * 3)
                                                       .bias(true)));
  sa_proj_     = register_module("sa_proj",  torch::nn::Linear(dim, dim));
  norm2_       = register_module("norm2", torch::nn::LayerNorm(
                                                torch::nn::LayerNormOptions({dim})));
  cross_attn_  = register_module(
      "cross_attn", MSDeformAttn(dim, num_heads, num_levels, num_points));
  norm3_       = register_module("norm3", torch::nn::LayerNorm(
                                                torch::nn::LayerNormOptions({dim})));
  fc1_         = register_module("fc1", torch::nn::Linear(dim, dim * mlp_ratio));
  fc2_         = register_module("fc2", torch::nn::Linear(dim * mlp_ratio, dim));
}

torch::Tensor DecoderLayerImpl::forward(torch::Tensor query,
                                          torch::Tensor query_pos,
                                          torch::Tensor reference_points,
                                          torch::Tensor memory,
                                          torch::Tensor spatial_shapes,
                                          torch::Tensor level_start_index) {
  // Pre-norm self-attn over queries (with positional embedding for q+k).
  auto h = norm1_->forward(query);
  auto q = h + query_pos;
  // Vanilla MHA: build qkv with the query as both k and v (so we
  // attend over the query set itself).
  auto qkv = sa_qkv_->forward(q);
  auto B = q.size(0); auto Lq = q.size(1);
  qkv = qkv.view({B, Lq, 3, num_heads_, head_dim_})
            .permute({2, 0, 3, 1, 4});                     // [3, B, H, Lq, D]
  auto Q = qkv[0], K = qkv[1], V = qkv[2];
  auto scale = 1.0 / std::sqrt(static_cast<double>(head_dim_));
  auto attn = torch::softmax(torch::matmul(Q, K.transpose(-2, -1)) * scale, -1);
  auto sa   = torch::matmul(attn, V).transpose(1, 2).contiguous()
                  .view({B, Lq, dim_});
  query     = query + sa_proj_->forward(sa);

  // Pre-norm cross-attn (queries → encoder memory) via MSDeformAttn.
  auto h2 = norm2_->forward(query) + query_pos;
  auto ca = cross_attn_->forward(h2, memory, reference_points,
                                  spatial_shapes, level_start_index);
  query   = query + ca;

  // Pre-norm FFN.
  auto y = norm3_->forward(query);
  y      = fc2_->forward(torch::gelu(fc1_->forward(y)));
  return query + y;
}

// ─── DetrHead ───────────────────────────────────────────────────────────

DetrHeadImpl::DetrHeadImpl(int hidden_dim, int num_heads, int num_layers,
                            int num_queries, int nc, int num_points,
                            int mlp_ratio)
    : hidden_dim_(hidden_dim), num_queries_(num_queries), nc_(nc) {
  // We don't know num_levels at construction time (it's an encoder
  // output). Use a default of 3 to match LW-DETR/DINOv2 backbone
  // taps; if a future config differs the cross_attn linears will
  // get rebuilt at first forward.
  num_levels_ = 3;
  query_embed_ = register_parameter(
      "query_embed", torch::randn({num_queries, hidden_dim}) * 0.02);
  query_pos_ = register_parameter(
      "query_pos",   torch::randn({num_queries, hidden_dim}) * 0.02);
  ref_points_unact_ = register_parameter(
      "ref_points_unact", torch::randn({num_queries, 2}) * 0.02);

  layers_ = torch::nn::ModuleList();
  cls_heads_ = torch::nn::ModuleList();
  bbox_heads_ = torch::nn::ModuleList();
  for (int i = 0; i < num_layers; ++i) {
    layers_->push_back(DecoderLayer(hidden_dim, num_heads, num_levels_,
                                       num_points, mlp_ratio));
    cls_heads_->push_back(torch::nn::Linear(hidden_dim, nc));
    bbox_heads_->push_back(MLP(hidden_dim, hidden_dim, 4, /*num_layers=*/3));
  }
  register_module("layers",     layers_);
  register_module("cls_heads",  cls_heads_);
  register_module("bbox_heads", bbox_heads_);
}

std::vector<std::pair<torch::Tensor, torch::Tensor>>
DetrHeadImpl::forward_train(torch::Tensor memory,
                              torch::Tensor spatial_shapes,
                              torch::Tensor level_start_index) {
  auto B = memory.size(0);
  auto query     = query_embed_.unsqueeze(0).expand({B, -1, -1}).contiguous();
  auto query_pos = query_pos_.unsqueeze(0).expand({B, -1, -1}).contiguous();
  // Reference points are scalar `(x, y)` per query in [0, 1].
  auto ref = torch::sigmoid(ref_points_unact_)
                  .unsqueeze(0)
                  .expand({B, -1, -1})
                  .unsqueeze(2)
                  .expand({B, -1, num_levels_, -1})
                  .contiguous();

  std::vector<std::pair<torch::Tensor, torch::Tensor>> outputs;
  outputs.reserve(layers_->size());
  for (int i = 0; i < static_cast<int>(layers_->size()); ++i) {
    query = layers_[i]->as<DecoderLayerImpl>()->forward(
        query, query_pos, ref, memory, spatial_shapes, level_start_index);
    auto cls_logits = cls_heads_[i]->as<torch::nn::LinearImpl>()->forward(query);
    auto bbox_unact = bbox_heads_[i]->as<MLPImpl>()->forward(query);
    outputs.emplace_back(cls_logits, bbox_unact);
  }
  return outputs;
}

torch::Tensor DetrHeadImpl::forward_eval(torch::Tensor memory,
                                          torch::Tensor spatial_shapes,
                                          torch::Tensor level_start_index) {
  auto outs = forward_train(std::move(memory), std::move(spatial_shapes),
                              std::move(level_start_index));
  // Last layer is the prediction layer. Sigmoid both cls + bbox.
  auto [cls_logits, bbox_unact] = outs.back();
  auto cls  = torch::sigmoid(cls_logits);                   // [B, Q, nc]
  auto bbox = torch::sigmoid(bbox_unact);                   // [B, Q, 4]
  // YOLO-style channel order: concat (bbox, cls) on channel dim,
  // then transpose to `[B, 4+nc, Q]`.
  auto out = torch::cat({bbox, cls}, /*dim=*/-1);            // [B, Q, 4+nc]
  return out.transpose(1, 2).contiguous();
}

}  // namespace yolocpp::models::rfdetr
