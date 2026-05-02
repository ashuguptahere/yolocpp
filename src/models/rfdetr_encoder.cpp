// RF-DETR encoder (#65B) — multi-scale deformable transformer.
//
// MSDeformAttn is implemented as a portable composition of standard
// ops (Linear projections + grid_sample + weighted sum) so it
// exports to ONNX without a custom op. The math matches Zhu et al.
// 2020 §3.1: each query's attention is a weighted sum of bilinearly
// sampled values at learnable offsets, summed over heads, levels,
// and points.

#include "yolocpp/models/rfdetr_encoder.hpp"

#include <cmath>

namespace yolocpp::models::rfdetr {

// ─── 2D sine positional embedding (level-aware) ──────────────────────────
//
// Returns `[B, ΣHi·Wi, hidden]`. Each token's encoding mixes a 2D
// sin/cos of its (y, x) position normalised to [0, 1] — same scheme
// as Detection Transformer.
static torch::Tensor sine_pos_embed_2d(
    int hidden, const std::vector<std::pair<int, int>>& shapes,
    torch::Device device) {
  TORCH_CHECK(hidden % 4 == 0, "pos embed dim must be divisible by 4");
  int half = hidden / 2;
  auto dim_t = torch::arange(half / 2, torch::dtype(torch::kFloat).device(device));
  // 2π scale + 10000^(2i/d) factor
  dim_t = torch::pow(10000.0, 2.0 * dim_t / static_cast<double>(half));

  std::vector<torch::Tensor> per_level;
  for (auto [H, W] : shapes) {
    auto y = torch::linspace(0.5 / H, 1.0 - 0.5 / H, H,
                              torch::dtype(torch::kFloat).device(device));
    auto x = torch::linspace(0.5 / W, 1.0 - 0.5 / W, W,
                              torch::dtype(torch::kFloat).device(device));
    auto yy = y.view({H, 1}).expand({H, W});
    auto xx = x.view({1, W}).expand({H, W});
    auto py = (yy.unsqueeze(-1) * 2 * M_PI) / dim_t;
    auto px = (xx.unsqueeze(-1) * 2 * M_PI) / dim_t;
    auto pe_y = torch::stack({py.sin(), py.cos()}, -1).flatten(-2);
    auto pe_x = torch::stack({px.sin(), px.cos()}, -1).flatten(-2);
    auto pe   = torch::cat({pe_y, pe_x}, -1);   // [H, W, hidden]
    per_level.push_back(pe.view({H * W, hidden}));
  }
  return torch::cat(per_level, /*dim=*/0).unsqueeze(0);  // [1, ΣHW, hidden]
}

// ─── MSDeformAttn ───────────────────────────────────────────────────────

MSDeformAttnImpl::MSDeformAttnImpl(int dim, int num_heads, int num_levels,
                                     int num_points)
    : dim_(dim), num_heads_(num_heads), num_levels_(num_levels),
      num_points_(num_points), head_dim_(dim / num_heads) {
  TORCH_CHECK(dim % num_heads == 0, "dim must divide num_heads");
  sampling_offsets_  = register_module(
      "sampling_offsets",
      torch::nn::Linear(dim, num_heads * num_levels * num_points * 2));
  attention_weights_ = register_module(
      "attention_weights",
      torch::nn::Linear(dim, num_heads * num_levels * num_points));
  value_proj_  = register_module("value_proj",  torch::nn::Linear(dim, dim));
  output_proj_ = register_module("output_proj", torch::nn::Linear(dim, dim));
}

torch::Tensor MSDeformAttnImpl::forward(torch::Tensor query,
                                          torch::Tensor value,
                                          torch::Tensor reference_points,
                                          torch::Tensor spatial_shapes,
                                          torch::Tensor level_start_index) {
  auto B  = query.size(0);
  auto Lq = query.size(1);

  // Project values: [B, Lv, C] → [B, Lv, H, D]
  auto val = value_proj_->forward(value)
                  .view({B, value.size(1), num_heads_, head_dim_});

  // Predict offsets + weights for each (head, level, point):
  //   offsets: [B, Lq, H, L, P, 2]
  //   weights: [B, Lq, H, L, P]   (softmaxed across L*P)
  auto off = sampling_offsets_->forward(query)
                  .view({B, Lq, num_heads_, num_levels_, num_points_, 2});
  auto wts = attention_weights_->forward(query)
                  .view({B, Lq, num_heads_, num_levels_ * num_points_});
  wts = torch::softmax(wts, -1).view(
      {B, Lq, num_heads_, num_levels_, num_points_});

  // For each level, build a `grid_sample` grid from the reference
  // points + offsets. `reference_points`: [B, Lq, L, 2] in [0, 1]
  // → expand to head + point dims, add normalised offset, scale to
  // [-1, 1] for grid_sample.
  auto ref = reference_points.unsqueeze(2).unsqueeze(4);
  // ref: [B, Lq, 1, L, 1, 2]; off: [B, Lq, H, L, P, 2]
  // sampling_loc in [0, 1]:
  auto offsets_normalizer =
      spatial_shapes.flip(-1).to(query.dtype());            // [L, 2] (W, H)
  // Broadcast normalizer to match offset shape; flip(-1) was wrong
  // dim — fix: spatial_shapes is (Hi, Wi); we want denom (Wi, Hi).
  offsets_normalizer = offsets_normalizer.view({1, 1, 1, num_levels_, 1, 2});
  auto sampling_loc = ref + off / offsets_normalizer;       // [B, Lq, H, L, P, 2]

  // grid_sample expects normalised [-1, 1]. Convert.
  sampling_loc = 2.0 * sampling_loc - 1.0;

  // For each level, sample. value[level] has shape [B, Hi*Wi, H, D];
  // reshape to [B*H, D, Hi, Wi] for grid_sample.
  std::vector<torch::Tensor> per_level;
  per_level.reserve(num_levels_);
  for (int lvl = 0; lvl < num_levels_; ++lvl) {
    int H_l = spatial_shapes[lvl][0].item<int64_t>();
    int W_l = spatial_shapes[lvl][1].item<int64_t>();
    int start = level_start_index[lvl].item<int64_t>();

    auto v_l = val.slice(1, start, start + H_l * W_l);    // [B, Hi*Wi, H, D]
    v_l = v_l.permute({0, 2, 3, 1}).contiguous()           // [B, H, D, Hi*Wi]
              .view({B * num_heads_, head_dim_, H_l, W_l});

    auto grid = sampling_loc.slice(3, lvl, lvl + 1).squeeze(3);
    // grid: [B, Lq, H, P, 2] → [B, H, Lq, P, 2] → [B*H, Lq, P, 2]
    grid = grid.permute({0, 2, 1, 3, 4}).contiguous()
                .view({B * num_heads_, Lq, num_points_, 2});

    auto sampled = torch::nn::functional::grid_sample(
        v_l, grid,
        torch::nn::functional::GridSampleFuncOptions()
            .mode(torch::kBilinear)
            .padding_mode(torch::kZeros)
            .align_corners(false));
    // sampled: [B*H, D, Lq, P] → [B, H, D, Lq, P] → [B, Lq, H, D, P]
    sampled = sampled.view({B, num_heads_, head_dim_, Lq, num_points_})
                  .permute({0, 3, 1, 2, 4});  // [B, Lq, H, D, P]
    per_level.push_back(sampled);
  }

  // Stack levels: [B, Lq, H, D, L, P]
  auto sampled = torch::stack(per_level, /*dim=*/4);
  // Weighted sum across (L, P) → [B, Lq, H, D]
  auto w = wts.permute({0, 1, 2, 3, 4}).unsqueeze(3);  // [B, Lq, H, 1, L, P]
  auto out = (sampled * w).sum(/*dim=*/{4, 5});         // [B, Lq, H, D]
  out = out.contiguous().view({B, Lq, dim_});
  return output_proj_->forward(out);
}

// ─── EncoderLayer ───────────────────────────────────────────────────────

EncoderLayerImpl::EncoderLayerImpl(int dim, int num_heads, int num_levels,
                                     int num_points, int mlp_ratio) {
  norm1_     = register_module("norm1", torch::nn::LayerNorm(
                                              torch::nn::LayerNormOptions({dim})));
  self_attn_ = register_module(
      "self_attn", MSDeformAttn(dim, num_heads, num_levels, num_points));
  norm2_     = register_module("norm2", torch::nn::LayerNorm(
                                              torch::nn::LayerNormOptions({dim})));
  fc1_       = register_module("fc1", torch::nn::Linear(dim, dim * mlp_ratio));
  fc2_       = register_module("fc2", torch::nn::Linear(dim * mlp_ratio, dim));
}

torch::Tensor EncoderLayerImpl::forward(torch::Tensor x, torch::Tensor pos,
                                          torch::Tensor reference_points,
                                          torch::Tensor spatial_shapes,
                                          torch::Tensor level_start_index) {
  // Pre-norm self-attn.
  auto h = norm1_->forward(x);
  auto q = h + pos;
  h      = self_attn_->forward(q, h, reference_points, spatial_shapes,
                                level_start_index);
  x      = x + h;
  // Pre-norm FFN.
  auto y = norm2_->forward(x);
  y      = fc2_->forward(torch::gelu(fc1_->forward(y)));
  return x + y;
}

// ─── Encoder ────────────────────────────────────────────────────────────

EncoderImpl::EncoderImpl(const std::vector<int>& in_channels, int hidden_dim,
                          int num_heads, int num_layers, int num_points,
                          int mlp_ratio)
    : hidden_dim_(hidden_dim),
      num_levels_(static_cast<int>(in_channels.size())) {
  input_proj_ = torch::nn::ModuleList();
  for (int c : in_channels) {
    input_proj_->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(c, hidden_dim, /*k=*/1)));
  }
  register_module("input_proj", input_proj_);

  layers_ = torch::nn::ModuleList();
  for (int i = 0; i < num_layers; ++i) {
    layers_->push_back(EncoderLayer(hidden_dim, num_heads, num_levels_,
                                       num_points, mlp_ratio));
  }
  register_module("layers", layers_);
}

EncoderOutput EncoderImpl::forward(
    const std::vector<torch::Tensor>& features) {
  TORCH_CHECK(static_cast<int>(features.size()) == num_levels_,
              "encoder: feature pyramid level count mismatch");
  auto B      = features[0].size(0);
  auto device = features[0].device();
  auto dtype  = features[0].scalar_type();

  // 1. Project + flatten each level.
  std::vector<torch::Tensor> projected;
  std::vector<std::pair<int, int>> shapes;
  std::vector<int64_t> shape_data;
  std::vector<int64_t> start_data;
  int64_t cum = 0;
  projected.reserve(num_levels_);
  shapes.reserve(num_levels_);
  for (int i = 0; i < num_levels_; ++i) {
    auto p = input_proj_[i]->as<torch::nn::Conv2dImpl>()->forward(features[i]);
    int H  = p.size(2), W = p.size(3);
    shapes.emplace_back(H, W);
    shape_data.push_back(H);
    shape_data.push_back(W);
    start_data.push_back(cum);
    cum += static_cast<int64_t>(H) * W;
    projected.push_back(p.flatten(2).transpose(1, 2));   // [B, Hi*Wi, hidden]
  }
  auto memory = torch::cat(projected, /*dim=*/1);         // [B, ΣHW, hidden]

  auto spatial_shapes = torch::tensor(
      shape_data,
      torch::dtype(torch::kInt64).device(device)).view({num_levels_, 2});
  auto level_start_index = torch::tensor(
      start_data, torch::dtype(torch::kInt64).device(device));

  // 2. Build reference points (one set, reused per layer).
  std::vector<torch::Tensor> ref_per_level;
  ref_per_level.reserve(num_levels_);
  for (auto [H, W] : shapes) {
    auto y = torch::linspace(0.5 / H, 1.0 - 0.5 / H, H,
                              torch::dtype(dtype).device(device));
    auto x = torch::linspace(0.5 / W, 1.0 - 0.5 / W, W,
                              torch::dtype(dtype).device(device));
    auto yy = y.view({H, 1}).expand({H, W});
    auto xx = x.view({1, W}).expand({H, W});
    auto pts = torch::stack({xx, yy}, -1).view({H * W, 2});  // (x, y) order
    ref_per_level.push_back(pts);
  }
  auto ref = torch::cat(ref_per_level, /*dim=*/0)
                  .unsqueeze(0)
                  .unsqueeze(2)
                  .expand({B, -1, num_levels_, -1});           // [B, ΣHW, L, 2]

  // 3. Positional embedding.
  auto pos = sine_pos_embed_2d(hidden_dim_, shapes, device).to(dtype);
  pos      = pos.expand({B, -1, -1});                          // [B, ΣHW, hidden]

  // 4. Run layers.
  for (int i = 0; i < static_cast<int>(layers_->size()); ++i) {
    memory = layers_[i]->as<EncoderLayerImpl>()->forward(
        memory, pos, ref, spatial_shapes, level_start_index);
  }

  return {memory, spatial_shapes, level_start_index, ref, pos};
}

}  // namespace yolocpp::models::rfdetr
