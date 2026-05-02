// RF-DETR transformer (#65C2 + #65D2). Parameter-registration
// only; forward semantics land under #65F2 once the decoder
// machinery (iterative bbox refinement, group-detr handling,
// two-stage query init) is implemented end-to-end.

#include "yolocpp/models/rfdetr_transformer.hpp"

#include <cmath>

namespace yolocpp::models::rfdetr {

// ─── FusedMHA ───────────────────────────────────────────────────────────

FusedMHAImpl::FusedMHAImpl(int hidden, int num_heads)
    : hidden_(hidden), num_heads_(num_heads) {
  TORCH_CHECK(hidden % num_heads == 0,
              "fused MHA: hidden must divide num_heads");
  in_proj_weight = register_parameter(
      "in_proj_weight", torch::zeros({3 * hidden, hidden}));
  in_proj_bias = register_parameter(
      "in_proj_bias",   torch::zeros({3 * hidden}));
  out_proj = register_module("out_proj", torch::nn::Linear(hidden, hidden));
}

torch::Tensor FusedMHAImpl::forward(torch::Tensor x) {
  // Vanilla self-attn from fused projection. Used at inference;
  // training kernel parity not required (LossTraits doesn't gate
  // on this op).
  auto B = x.size(0), N = x.size(1);
  auto qkv = torch::matmul(x, in_proj_weight.t()) + in_proj_bias;
  auto chunks = qkv.chunk(3, /*dim=*/-1);
  int hd = hidden_ / num_heads_;
  auto reshape_h = [&](torch::Tensor t) {
    return t.view({B, N, num_heads_, hd}).transpose(1, 2);
  };
  auto Q = reshape_h(chunks[0]);
  auto K = reshape_h(chunks[1]);
  auto V = reshape_h(chunks[2]);
  auto scale = 1.0 / std::sqrt(static_cast<double>(hd));
  auto attn  = torch::softmax(torch::matmul(Q, K.transpose(-2, -1)) * scale, -1);
  auto out   = torch::matmul(attn, V).transpose(1, 2).contiguous()
                  .view({B, N, hidden_});
  return out_proj->forward(out);
}

// ─── MSDeformAttn1L ─────────────────────────────────────────────────────

MSDeformAttn1LImpl::MSDeformAttn1LImpl(int hidden, int num_heads,
                                          int num_points) {
  // num_levels is fixed at 1 (RF-DETR projects to a single P4 level).
  int n_off = num_heads * /*levels=*/1 * num_points * 2;   // 64 for h=16, p=2
  int n_aw  = num_heads * /*levels=*/1 * num_points;       // 32
  sampling_offsets  = register_module("sampling_offsets",
                                         torch::nn::Linear(hidden, n_off));
  attention_weights = register_module("attention_weights",
                                         torch::nn::Linear(hidden, n_aw));
  value_proj  = register_module("value_proj",  torch::nn::Linear(hidden, hidden));
  output_proj = register_module("output_proj", torch::nn::Linear(hidden, hidden));
}

// ─── RFDetrMLP ──────────────────────────────────────────────────────────

RFDetrMLPImpl::RFDetrMLPImpl(int input_dim, int hidden_dim, int output_dim,
                                int num_layers) {
  layers = torch::nn::ModuleList();
  for (int i = 0; i < num_layers; ++i) {
    int in  = (i == 0) ? input_dim  : hidden_dim;
    int out = (i == num_layers - 1) ? output_dim : hidden_dim;
    layers->push_back(torch::nn::Linear(in, out));
  }
  register_module("layers", layers);
}

torch::Tensor RFDetrMLPImpl::forward(torch::Tensor x) {
  int n = static_cast<int>(layers->size());
  for (int i = 0; i < n; ++i) {
    x = layers[i]->as<torch::nn::LinearImpl>()->forward(x);
    if (i < n - 1) x = torch::relu(x);
  }
  return x;
}

// ─── RFDetrDecoderLayer ─────────────────────────────────────────────────

RFDetrDecoderLayerImpl::RFDetrDecoderLayerImpl(int hidden, int sa_nheads,
                                                  int ca_nheads,
                                                  int dec_n_points,
                                                  int ffn_dim) {
  self_attn = register_module("self_attn", FusedMHA(hidden, sa_nheads));
  norm1     = register_module(
      "norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})));
  cross_attn = register_module(
      "cross_attn", MSDeformAttn1L(hidden, ca_nheads, dec_n_points));
  linear1   = register_module("linear1", torch::nn::Linear(hidden, ffn_dim));
  linear2   = register_module("linear2", torch::nn::Linear(ffn_dim, hidden));
  norm2     = register_module(
      "norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})));
  norm3     = register_module(
      "norm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})));
}

// ─── RFDetrDecoder ──────────────────────────────────────────────────────

RFDetrDecoderImpl::RFDetrDecoderImpl(int n_layers, int hidden, int sa_nheads,
                                        int ca_nheads, int dec_n_points,
                                        int ffn_dim) {
  layers = torch::nn::ModuleList();
  for (int i = 0; i < n_layers; ++i) {
    layers->push_back(RFDetrDecoderLayer(
        hidden, sa_nheads, ca_nheads, dec_n_points, ffn_dim));
  }
  register_module("layers", layers);
  norm = register_module(
      "norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})));
  // 2-layer MLP: 2·hidden → hidden → hidden. The 2× input dim comes
  // from sinusoidal-embedding (4-channel refpoint × ~hidden/2 each
  // → packed to 2·hidden).
  ref_point_head = register_module(
      "ref_point_head",
      RFDetrMLP(/*input=*/2 * hidden, /*hidden=*/hidden,
                  /*output=*/hidden, /*num_layers=*/2));
}

// ─── RFDetrEncOutput ────────────────────────────────────────────────────

RFDetrEncOutputImpl::RFDetrEncOutputImpl(int group_detr, int hidden,
                                            int n_classes_with_bg) {
  enc_output           = torch::nn::ModuleList();
  enc_output_norm      = torch::nn::ModuleList();
  enc_out_class_embed  = torch::nn::ModuleList();
  enc_out_bbox_embed   = torch::nn::ModuleList();
  for (int g = 0; g < group_detr; ++g) {
    enc_output->push_back(torch::nn::Linear(hidden, hidden));
    enc_output_norm->push_back(torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({hidden})));
    enc_out_class_embed->push_back(
        torch::nn::Linear(hidden, n_classes_with_bg));
    enc_out_bbox_embed->push_back(
        RFDetrMLP(hidden, hidden, /*output_dim=*/4, /*num_layers=*/3));
  }
  // NOTE: registered under their own names so the subtree path is
  // `<parent>.enc_output.<k>.{weight,bias}` etc.
  register_module("enc_output",          enc_output);
  register_module("enc_output_norm",     enc_output_norm);
  register_module("enc_out_class_embed", enc_out_class_embed);
  register_module("enc_out_bbox_embed",  enc_out_bbox_embed);
}

// ─── RFDetrTransformer ──────────────────────────────────────────────────

RFDetrTransformerImpl::RFDetrTransformerImpl(int hidden, int n_dec_layers,
                                                int sa_nheads, int ca_nheads,
                                                int dec_n_points, int ffn_dim,
                                                int group_detr,
                                                int n_classes_with_bg) {
  decoder = register_module(
      "decoder", RFDetrDecoder(n_dec_layers, hidden, sa_nheads, ca_nheads,
                                 dec_n_points, ffn_dim));
  // Flatten enc_output siblings directly under transformer.* to
  // match upstream's `transformer.enc_output.<k>` path (RFDetrEncOutput
  // would add an extra namespace level).
  enc_output           = torch::nn::ModuleList();
  enc_output_norm      = torch::nn::ModuleList();
  enc_out_class_embed  = torch::nn::ModuleList();
  enc_out_bbox_embed   = torch::nn::ModuleList();
  for (int g = 0; g < group_detr; ++g) {
    enc_output->push_back(torch::nn::Linear(hidden, hidden));
    enc_output_norm->push_back(torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({hidden})));
    enc_out_class_embed->push_back(
        torch::nn::Linear(hidden, n_classes_with_bg));
    enc_out_bbox_embed->push_back(
        RFDetrMLP(hidden, hidden, /*output_dim=*/4, /*num_layers=*/3));
  }
  register_module("enc_output",          enc_output);
  register_module("enc_output_norm",     enc_output_norm);
  register_module("enc_out_class_embed", enc_out_class_embed);
  register_module("enc_out_bbox_embed",  enc_out_bbox_embed);
}

}  // namespace yolocpp::models::rfdetr
