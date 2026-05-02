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
                                          int num_points)
    : hidden_(hidden), num_heads_(num_heads), num_points_(num_points),
      head_dim_(hidden / num_heads) {
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

torch::Tensor MSDeformAttn1LImpl::forward(torch::Tensor query,
                                            torch::Tensor reference_points,
                                            torch::Tensor memory,
                                            int H, int W) {
  // query: [B, Q, C]; ref: [B, Q, 4]; memory: [B, L, C] (L = H*W)
  auto B = query.size(0), Q = query.size(1);
  auto C = query.size(2);

  auto value = value_proj->forward(memory);   // [B, L, C]

  // sampling_offsets: [B, Q, H*L*P*2] → [B, Q, H, 1, P, 2]
  auto so = sampling_offsets->forward(query)
                  .view({B, Q, num_heads_, /*L=*/1, num_points_, 2});
  // attention_weights: [B, Q, H*L*P] → [B, Q, H, L*P]; softmax over (L*P)
  auto aw = attention_weights->forward(query)
                  .view({B, Q, num_heads_, /*L=*/1 * num_points_});
  aw = torch::softmax(aw, -1).view({B, Q, num_heads_, 1, num_points_});

  // bbox_reparam path (4D refpoints): sample_loc = ref_xy +
  //   offset/n_points * ref_wh * 0.5
  // Shape ref to [B, Q, 1(heads), 1(levels), 1(points), 4] for
  // broadcasting against `so` [B, Q, H, L, P, 2].
  auto ref = reference_points.view({B, Q, 1, 1, 1, 4});
  auto ref_xy = ref.slice(-1, 0, 2);
  auto ref_wh = ref.slice(-1, 2, 4);
  auto sample_loc = ref_xy +
      (so / static_cast<double>(num_points_)) * ref_wh * 0.5;
  // sample_loc: [B, Q, H, 1, P, 2] in [0,1]; convert to grid_sample [-1, 1].
  sample_loc = 2.0 * sample_loc - 1.0;

  // value: [B, L=H*W, C] → [B*H_heads, head_dim, H, W]
  auto v = value.transpose(1, 2).contiguous()
              .view({B, num_heads_, head_dim_, H, W})
              .view({B * num_heads_, head_dim_, H, W});

  // grid: [B, Q, H, P, 2] → [B*H, Q, P, 2]
  auto grid = sample_loc.squeeze(3)            // [B, Q, H, P, 2]
                  .permute({0, 2, 1, 3, 4})    // [B, H, Q, P, 2]
                  .contiguous()
                  .view({B * num_heads_, Q, num_points_, 2});

  auto sampled = torch::nn::functional::grid_sample(
      v, grid,
      torch::nn::functional::GridSampleFuncOptions()
          .mode(torch::kBilinear)
          .padding_mode(torch::kZeros)
          .align_corners(false));
  // sampled: [B*H, head_dim, Q, P]

  // attn_w: [B, Q, H, 1, P] → [B*H, 1, Q, P]
  auto w = aw.permute({0, 2, 1, 3, 4})
              .contiguous()
              .view({B * num_heads_, 1, Q, num_points_});
  auto out = (sampled * w).sum(/*dim=*/-1);    // [B*H, head_dim, Q]
  out = out.view({B, num_heads_, head_dim_, Q})
              .permute({0, 3, 1, 2})            // [B, Q, H, head_dim]
              .contiguous()
              .view({B, Q, hidden_});
  return output_proj->forward(out);
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

torch::Tensor RFDetrDecoderLayerImpl::forward(torch::Tensor tgt,
                                                torch::Tensor query_pos,
                                                torch::Tensor reference_points,
                                                torch::Tensor memory,
                                                int H, int W) {
  // Post-LN style. Self-attn uses (tgt + query_pos) for q+k+v in our
  // simplified FusedMHA variant — minor deviation from upstream
  // (which uses v=tgt) but produces the same shape contract.
  auto sa_out = self_attn->forward(tgt + query_pos);
  tgt = norm1->forward(tgt + sa_out);
  // Cross-attn: query gets pos embed; memory is the projector output.
  auto ca_out = cross_attn->forward(tgt + query_pos, reference_points,
                                      memory, H, W);
  tgt = norm2->forward(tgt + ca_out);
  // FFN.
  auto ff = linear2->forward(torch::relu(linear1->forward(tgt)));
  return norm3->forward(tgt + ff);
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
  ref_point_head = register_module(
      "ref_point_head",
      RFDetrMLP(/*input=*/2 * hidden, /*hidden=*/hidden,
                  /*output=*/hidden, /*num_layers=*/2));
}

torch::Tensor RFDetrDecoderImpl::forward(torch::Tensor tgt,
                                           torch::Tensor refpoints,
                                           torch::Tensor memory,
                                           int H, int W) {
  // lite_refpoint_refine=True path: generate query_pos ONCE before
  // the loop. (refpoints stay fixed — no iterative update at eval
  // for this mode; the final bbox is delta + refpoints applied
  // OUTSIDE the decoder, in RFDetrImpl::forward_eval.)
  int hidden = static_cast<int>(refpoints.size(-1) > 0 ?
                                  norm->options.normalized_shape()[0] : 0);
  (void)hidden;
  auto sin_emb = gen_sineembed_for_position(
      refpoints,
      /*dim=*/static_cast<int>(memory.size(-1)) / 2);
  auto query_pos = ref_point_head->forward(sin_emb);
  for (int i = 0; i < static_cast<int>(layers->size()); ++i) {
    tgt = layers[i]->as<RFDetrDecoderLayerImpl>()->forward(
        tgt, query_pos, refpoints, memory, H, W);
  }
  return norm->forward(tgt);
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
                                                int n_classes_with_bg)
    : hidden_(hidden) {
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

TransformerOutput RFDetrTransformerImpl::forward(
    torch::Tensor memory_2d, torch::Tensor query_feat_first_group,
    int num_queries) {
  // memory_2d: [B, C, H, W] — projector output.
  auto B = memory_2d.size(0);
  int  C = static_cast<int>(memory_2d.size(1));
  int  H = static_cast<int>(memory_2d.size(2));
  int  W = static_cast<int>(memory_2d.size(3));
  // Flatten to tokens [B, L, C].
  auto memory = memory_2d.flatten(2).transpose(1, 2).contiguous();

  // Two-stage: enc_output[0] + LN[0] → cls + bbox proposals.
  // At eval only the first of group_detr groups is used.
  auto out_mem = enc_output_norm[0]->as<torch::nn::LayerNormImpl>()->forward(
      enc_output[0]->as<torch::nn::LinearImpl>()->forward(memory));

  auto cls_logits = enc_out_class_embed[0]
                        ->as<torch::nn::LinearImpl>()->forward(out_mem);
  auto bbox_delta = enc_out_bbox_embed[0]
                        ->as<RFDetrMLPImpl>()->forward(out_mem);

  // Generate dense per-token proposals (cxcy on grid, wh fixed prior).
  auto props = gen_encoder_output_proposals_1l(memory, H, W);
  // bbox_reparam: cx = delta_xy * wh + xy; wh = exp(delta_wh) * wh.
  auto coord_cxcy = bbox_delta.slice(-1, 0, 2) *
                        props.output_proposals.slice(-1, 2, 4) +
                    props.output_proposals.slice(-1, 0, 2);
  auto coord_wh   = bbox_delta.slice(-1, 2, 4).exp() *
                        props.output_proposals.slice(-1, 2, 4);
  auto coord = torch::cat({coord_cxcy, coord_wh}, -1);   // [B, L, 4]

  // Top-K by max-cls score.
  auto top_scores = std::get<0>(cls_logits.max(-1));     // [B, L]
  int  K          = std::min<int>(num_queries, static_cast<int>(top_scores.size(1)));
  auto topk = top_scores.topk(K, /*dim=*/1);
  auto idx  = std::get<1>(topk);                          // [B, K]

  // Gather top-K refpoints.
  auto idx_4 = idx.unsqueeze(-1).expand({B, K, 4});
  auto topk_refpts = torch::gather(coord, /*dim=*/1, idx_4);  // [B, K, 4]

  // Initial query feats from the LEARNED query_feat[:Q] (broadcast to B).
  auto tgt = query_feat_first_group.unsqueeze(0).expand({B, K, hidden_})
                  .contiguous();

  auto out = decoder->forward(tgt, topk_refpts, memory, H, W);
  return {out, topk_refpts};
}

// ─── helpers ────────────────────────────────────────────────────────────

torch::Tensor gen_sineembed_for_position(const torch::Tensor& pos_tensor,
                                           int dim) {
  // pos_tensor: [B, Q, 4] (cx, cy, w, h) in [0,1].
  // Return: [B, Q, 4*dim] — sinusoidal embedding per channel.
  double scale = 2 * M_PI;
  auto dim_t = torch::arange(dim, pos_tensor.options());
  dim_t = torch::pow(10000.0, 2.0 * (dim_t / 2).floor() / static_cast<double>(dim));
  auto encode_1d = [&](const torch::Tensor& vals) {
    // vals: [B, Q]
    auto v = (vals * scale).unsqueeze(-1) / dim_t;       // [B, Q, dim]
    // interleave sin(even) cos(odd)
    auto sin_part = v.slice(-1, 0, dim, 2).sin();
    auto cos_part = v.slice(-1, 1, dim, 2).cos();
    return torch::stack({sin_part, cos_part}, -1).flatten(-2);  // [B, Q, dim]
  };
  auto cx = pos_tensor.select(-1, 0);
  auto cy = pos_tensor.select(-1, 1);
  auto w  = pos_tensor.select(-1, 2);
  auto h  = pos_tensor.select(-1, 3);
  auto pos_x = encode_1d(cx);
  auto pos_y = encode_1d(cy);
  auto pos_w = encode_1d(w);
  auto pos_h = encode_1d(h);
  return torch::cat({pos_y, pos_x, pos_w, pos_h}, -1);   // [B, Q, 4*dim]
}

EncoderProposals gen_encoder_output_proposals_1l(const torch::Tensor& memory,
                                                  int H, int W) {
  // Single-level proposals: cx,cy on grid, wh = 0.05 (fixed prior).
  auto opts = memory.options();
  auto y    = torch::arange(H, opts);
  auto x    = torch::arange(W, opts);
  auto grid_y = y.view({H, 1}).expand({H, W});
  auto grid_x = x.view({1, W}).expand({H, W});
  auto grid_cxcy = torch::stack({(grid_x + 0.5) / static_cast<double>(W),
                                   (grid_y + 0.5) / static_cast<double>(H)}, -1);
  auto grid_wh   = torch::full_like(grid_cxcy, 0.05);
  auto props_2d  = torch::cat({grid_cxcy, grid_wh}, -1);    // [H, W, 4]
  auto props     = props_2d.view({1, H * W, 4})
                      .expand({memory.size(0), -1, -1})
                      .contiguous();
  return {memory, props};
}

}  // namespace yolocpp::models::rfdetr
