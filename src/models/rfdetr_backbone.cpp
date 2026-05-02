// RF-DETR 1.6.5 backbone (#65A2) — HF-DINOv2 windowed-attn implementation.
//
// Two parallel module trees live in this file:
//
//   * `Dinov2*` modules — the REAL HF-DINOv2 layout. Parameter
//     dotted-paths match upstream so `load_rfdetr_pt` binds in one
//     pass. RFDetrImpl registers a ModuleList "backbone" containing
//     a `Dinov2WrapperOuter` so the full path becomes
//     `backbone.0.encoder.encoder.embeddings.*`.
//
//   * Legacy `BackboneCfg` / `PatchEmbed` / `Attention` / `ViTBlock`
//     / `ViTBackbone` — the old scaffold. Kept building so the
//     existing #65B/C/E/F/G modules link. Dropped once #65B2..D2
//     replace those.
//
// Forward semantics (Dinov2Model):
//   1. Patch-embed `[B,3,H,W]` → token sequence `[B,N,C]`.
//   2. Concat learnable cls_token; add 2D-interpolated position
//      embedding (interpolation tracks input grid size — RF-DETR
//      runs at variable resolution).
//   3. Pass through 12 transformer blocks (layer-scaled pre-norm
//      + GELU MLP + residual).
//   4. Final layernorm.
//   5. Capture taps at `cfg.tap_blocks`; return them in spatial
//      form `[B, C, Hg, Wg]` (drop cls token).

#include "yolocpp/models/rfdetr_backbone.hpp"

#include <cmath>
#include <stdexcept>

namespace yolocpp::models::rfdetr {

// ─── Per-variant configs ────────────────────────────────────────────────

// Default window config — overridden per `upstream_id` in
// `dinov2_cfg_for`.
const Dinov2Cfg kDinov2Small{
    /*hidden=*/384, /*depth=*/12, /*heads=*/6, /*mlp=*/4,
    /*patch=*/14, /*pretrain_grid=*/37, /*qkv_bias=*/true,
    /*taps=*/{2, 5, 8, 11}, /*num_windows=*/1, /*window_blocks=*/{}};

const Dinov2Cfg kDinov2Base{
    /*hidden=*/768, /*depth=*/12, /*heads=*/12, /*mlp=*/4,
    /*patch=*/14, /*pretrain_grid=*/37, /*qkv_bias=*/true,
    /*taps=*/{2, 5, 8, 11}, /*num_windows=*/1, /*window_blocks=*/{}};

const Dinov2Cfg& dinov2_cfg_for(const std::string& upstream_id, int patch,
                                  int pretrain_grid, int backbone_embed) {
  // Variants share the "small" 12-block transformer family except
  // for "large" which uses 12 blocks at C=768. We honour the
  // explicit `backbone_embed` from the scale rather than infer it
  // from `upstream_id` so future variants slot in cleanly.
  static thread_local Dinov2Cfg cfg;
  if (upstream_id == "large") cfg = kDinov2Base;
  else                        cfg = kDinov2Small;
  cfg.hidden_size  = backbone_embed;
  cfg.num_heads    = backbone_embed / 64;   // 6 for 384, 12 for 768
  cfg.patch_size   = patch;
  cfg.pretrain_grid = pretrain_grid;
  // Per-variant windowed-attn partitioning + tap blocks (verified
  // via Python `RFDETR<X>().model.model.backbone[0].encoder.encoder.config`).
  if (upstream_id == "nano" || upstream_id == "small" ||
      upstream_id == "medium") {
    cfg.num_windows = 2;
    cfg.window_block_indexes = {0, 1, 3, 4, 6, 7, 9, 10};
    cfg.tap_blocks = {2, 5, 8, 11};
  } else if (upstream_id == "base") {
    cfg.num_windows = 4;
    cfg.window_block_indexes = {0, 1, 3, 4, 6, 7, 9, 10};
    cfg.tap_blocks = {2, 5, 8, 11};   // upstream `out_features=['stage2','stage5','stage8','stage11']` → 0-indexed
  } else if (upstream_id == "large") {
    cfg.num_windows = 2;
    cfg.window_block_indexes = {0, 1, 2, 4, 5, 7, 8, 10, 11};
    cfg.tap_blocks = {3, 6, 9, 11};   // out_features=stage3,6,9,12 → 0-indexed
  } else {
    cfg.num_windows = 2;
    cfg.window_block_indexes = {0, 1, 3, 4, 6, 7, 9, 10};
    cfg.tap_blocks = {2, 5, 8, 11};
  }
  return cfg;
}

// ─── Real HF-DINOv2 modules (params match upstream key names) ─────────

Dinov2PatchEmbeddingsImpl::Dinov2PatchEmbeddingsImpl(int in_ch, int hidden,
                                                       int patch) {
  projection = register_module(
      "projection",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, hidden, patch)
                            .stride(patch)));
}

torch::Tensor Dinov2PatchEmbeddingsImpl::forward(torch::Tensor x) {
  return projection->forward(x).flatten(2).transpose(1, 2);
}

Dinov2EmbeddingsImpl::Dinov2EmbeddingsImpl(const Dinov2Cfg& cfg)
    : patch_size_(cfg.patch_size), pretrain_grid_(cfg.pretrain_grid),
      num_windows_(cfg.num_windows) {
  cls_token  = register_parameter("cls_token",
                                    torch::zeros({1, 1, cfg.hidden_size}));
  mask_token = register_parameter("mask_token",
                                    torch::zeros({1, cfg.hidden_size}));
  // Pretrain grid + 1 cls token. Real shape interpolates at forward.
  int N = cfg.pretrain_grid * cfg.pretrain_grid + 1;
  position_embeddings = register_parameter(
      "position_embeddings", torch::zeros({1, N, cfg.hidden_size}));
  patch_embeddings = register_module(
      "patch_embeddings",
      Dinov2PatchEmbeddings(/*in_ch=*/3, cfg.hidden_size, cfg.patch_size));
}

// 2D bilinear interpolation of the saved position embedding to match
// the input's grid size. Matches the standard ViT behaviour used by
// HF DINOv2 (and copied into RF-DETR).
static torch::Tensor interpolate_pos_embed(const torch::Tensor& pe,
                                            int target_h, int target_w,
                                            int pretrain_grid) {
  // pe: [1, N+1, C] = [1, P*P+1, C]
  auto cls  = pe.slice(/*dim=*/1, 0, 1);
  auto pat  = pe.slice(1, 1).contiguous();
  int  C    = pe.size(2);
  // Reshape patches → [1, C, P, P]
  pat = pat.view({1, pretrain_grid, pretrain_grid, C}).permute({0, 3, 1, 2});
  // Upstream's call: `interpolate(..., scale_factor=(h/sqrt, w/sqrt),
  // mode='bicubic', align_corners=False)`. libtorch's bicubic
  // kernel produces ~6.3e-3 max-abs-diff vs the same call from
  // PyTorch Python on bit-identical inputs (verified via the
  // parity harness #65L slices 4-5: position_embeddings parameter
  // loads with 0 diff; patch + cls concat have 0 diff;
  // emb_step3_pos_embed shows 6.3e-3 immediately after this
  // interpolate call). Tried `size=` vs `scale_factor=`,
  // `recompute_scale_factor=false`, fp64-cast round-trip — all
  // produce identical 6.3e-3 diff. The kernel itself diverges
  // between the two PyTorch builds. This 6.3e-3 is the parity
  // FLOOR for variable-size inference (compounds to ~1.6 at the
  // 12th transformer block via residual amplification); fixed-
  // resolution inference at exactly `pretrain_grid · patch_size`
  // would skip interpolation entirely and could close the gap.
  std::vector<double> sf{
      static_cast<double>(target_h) / pretrain_grid,
      static_cast<double>(target_w) / pretrain_grid};
  pat = torch::nn::functional::interpolate(
      pat,
      torch::nn::functional::InterpolateFuncOptions()
          .scale_factor(sf)
          .mode(torch::kBicubic)
          .align_corners(false)
          .recompute_scale_factor(false));
  pat = pat.permute({0, 2, 3, 1}).contiguous().view({1, target_h * target_w, C});
  return torch::cat({cls, pat}, /*dim=*/1);
}

Dinov2EmbeddingsImpl::SubStages Dinov2EmbeddingsImpl::forward_stages(
    torch::Tensor x) {
  SubStages s;
  auto B  = x.size(0);
  auto Hg = x.size(2) / patch_size_;
  auto Wg = x.size(3) / patch_size_;
  auto C  = cls_token.size(2);
  s.patch    = patch_embeddings->forward(x);                 // [B, Hg*Wg, C]
  auto cls   = cls_token.expand({B, -1, -1});
  s.with_cls = torch::cat({cls, s.patch}, 1);
  s.pos_embed = interpolate_pos_embed(position_embeddings, Hg, Wg,
                                        pretrain_grid_).to(s.with_cls.dtype());
  s.with_pos = s.with_cls + s.pos_embed;
  if (num_windows_ <= 1) {
    s.windowed = s.with_pos;
    return s;
  }
  int W = num_windows_;
  int Hw = Hg / W, Ww = Wg / W;
  auto cls_pe = s.with_pos.slice(1, 0, 1);                   // [B, 1, C]
  auto pat    = s.with_pos.slice(1, 1).contiguous();          // [B, Hg*Wg, C]
  pat = pat.view({B, Hg, Wg, C});                             // [B, Hg, Wg, C]
  pat = pat.view({B, W, Hw, W, Ww, C});
  pat = pat.permute({0, 1, 3, 2, 4, 5}).contiguous()
            .view({B * W * W, Hw * Ww, C});
  auto cls_per_window = cls_pe.repeat({W * W, 1, 1});
  s.windowed = torch::cat({cls_per_window, pat}, 1);
  return s;
}

torch::Tensor Dinov2EmbeddingsImpl::forward(torch::Tensor x) {
  return forward_stages(std::move(x)).windowed;
}

Dinov2SelfAttentionImpl::Dinov2SelfAttentionImpl(int hidden, int heads,
                                                   bool bias)
    : heads_(heads), head_dim_(hidden / heads) {
  TORCH_CHECK(hidden % heads == 0, "dinov2 attn: hidden must divide heads");
  query = register_module("query",
                            torch::nn::Linear(torch::nn::LinearOptions(hidden, hidden).bias(bias)));
  key   = register_module("key",
                            torch::nn::Linear(torch::nn::LinearOptions(hidden, hidden).bias(bias)));
  value = register_module("value",
                            torch::nn::Linear(torch::nn::LinearOptions(hidden, hidden).bias(bias)));
}

torch::Tensor Dinov2SelfAttentionImpl::forward(torch::Tensor x) {
  auto B = x.size(0), N = x.size(1), C = x.size(2);
  auto reshape_heads = [&](torch::Tensor t) {
    return t.view({B, N, heads_, head_dim_}).transpose(1, 2);
  };
  auto Q = reshape_heads(query->forward(x));
  auto K = reshape_heads(key->forward(x));
  auto V = reshape_heads(value->forward(x));
  auto scale = 1.0 / std::sqrt(static_cast<double>(head_dim_));
  auto attn  = torch::softmax(torch::matmul(Q, K.transpose(-2, -1)) * scale, -1);
  auto out   = torch::matmul(attn, V);                        // [B, H, N, D]
  return out.transpose(1, 2).contiguous().view({B, N, C});    // [B, N, C]
}

Dinov2SelfOutputImpl::Dinov2SelfOutputImpl(int hidden) {
  dense = register_module("dense", torch::nn::Linear(hidden, hidden));
}

torch::Tensor Dinov2SelfOutputImpl::forward(torch::Tensor x) {
  return dense->forward(x);
}

Dinov2AttentionImpl::Dinov2AttentionImpl(int hidden, int heads, bool bias) {
  attention = register_module("attention",
                                Dinov2SelfAttention(hidden, heads, bias));
  output    = register_module("output", Dinov2SelfOutput(hidden));
}

torch::Tensor Dinov2AttentionImpl::forward(torch::Tensor x) {
  return output->forward(attention->forward(x));
}

Dinov2LayerScaleImpl::Dinov2LayerScaleImpl(int hidden) {
  lambda1 = register_parameter("lambda1", torch::ones({hidden}));
}

torch::Tensor Dinov2LayerScaleImpl::forward(torch::Tensor x) {
  return x * lambda1;
}

Dinov2MLPImpl::Dinov2MLPImpl(int hidden, int mlp_ratio) {
  fc1 = register_module("fc1", torch::nn::Linear(hidden, hidden * mlp_ratio));
  fc2 = register_module("fc2", torch::nn::Linear(hidden * mlp_ratio, hidden));
}

torch::Tensor Dinov2MLPImpl::forward(torch::Tensor x) {
  return fc2->forward(torch::gelu(fc1->forward(x)));
}

Dinov2LayerImpl::Dinov2LayerImpl(const Dinov2Cfg& cfg)
    : num_windows_(cfg.num_windows) {
  norm1        = register_module("norm1", torch::nn::LayerNorm(
                                                torch::nn::LayerNormOptions({cfg.hidden_size})
                                                    .eps(1e-6)));
  attention    = register_module(
      "attention", Dinov2Attention(cfg.hidden_size, cfg.num_heads, cfg.qkv_bias));
  layer_scale1 = register_module("layer_scale1", Dinov2LayerScale(cfg.hidden_size));
  norm2        = register_module("norm2", torch::nn::LayerNorm(
                                                torch::nn::LayerNormOptions({cfg.hidden_size})
                                                    .eps(1e-6)));
  mlp          = register_module("mlp", Dinov2MLP(cfg.hidden_size, cfg.mlp_ratio));
  layer_scale2 = register_module("layer_scale2", Dinov2LayerScale(cfg.hidden_size));
}

torch::Tensor Dinov2LayerImpl::forward(torch::Tensor x, bool run_full_attention) {
  // Optionally un-window for full self-attn — then re-window after.
  // The norm1 + attention happen inside the un-windowed view;
  // residual is added against the ORIGINAL (still-windowed) input.
  auto shortcut = x;
  if (run_full_attention && num_windows_ > 1) {
    int W2 = num_windows_ * num_windows_;
    auto B  = x.size(0) / W2;
    auto HW = x.size(1);
    auto C  = x.size(2);
    x = x.view({B, W2 * HW, C});
  }
  auto y = norm1->forward(x);
  y      = attention->forward(y);
  if (run_full_attention && num_windows_ > 1) {
    int W2 = num_windows_ * num_windows_;
    auto B = y.size(0);
    auto HW_full = y.size(1);
    auto C = y.size(2);
    y = y.view({B * W2, HW_full / W2, C});
  }
  y = layer_scale1->forward(y);
  x = shortcut + y;
  auto z = norm2->forward(x);
  z      = mlp->forward(z);
  z      = layer_scale2->forward(z);
  return x + z;
}

Dinov2EncoderImpl::Dinov2EncoderImpl(const Dinov2Cfg& cfg)
    : taps_(cfg.tap_blocks) {
  layer = torch::nn::ModuleList();
  is_windowed_.assign(cfg.num_layers, true);
  for (int idx : cfg.window_block_indexes) {
    if (idx >= 0 && idx < cfg.num_layers) is_windowed_[idx] = true;
  }
  // Set is_windowed_[i] = (i in window_block_indexes); default false.
  std::fill(is_windowed_.begin(), is_windowed_.end(), false);
  for (int idx : cfg.window_block_indexes) {
    if (idx >= 0 && idx < cfg.num_layers) is_windowed_[idx] = true;
  }
  for (int i = 0; i < cfg.num_layers; ++i) {
    layer->push_back(Dinov2Layer(cfg));
  }
  register_module("layer", layer);
}

std::vector<torch::Tensor> Dinov2EncoderImpl::forward(torch::Tensor x) {
  std::vector<torch::Tensor> taps;
  taps.reserve(taps_.size());
  for (int i = 0; i < static_cast<int>(layer->size()); ++i) {
    bool run_full = !is_windowed_[i];
    x = layer[i]->as<Dinov2LayerImpl>()->forward(x, run_full);
    for (int t : taps_) {
      if (t == i) {
        taps.push_back(x);
        break;
      }
    }
  }
  return taps;
}

std::vector<torch::Tensor> Dinov2EncoderImpl::forward_all_blocks(torch::Tensor x) {
  std::vector<torch::Tensor> outs;
  outs.reserve(layer->size());
  for (int i = 0; i < static_cast<int>(layer->size()); ++i) {
    bool run_full = !is_windowed_[i];
    x = layer[i]->as<Dinov2LayerImpl>()->forward(x, run_full);
    outs.push_back(x);
  }
  return outs;
}

Dinov2ModelImpl::Dinov2ModelImpl(const Dinov2Cfg& cfg) {
  embeddings = register_module("embeddings", Dinov2Embeddings(cfg));
  encoder    = register_module("encoder", Dinov2Encoder(cfg));
  layernorm  = register_module("layernorm",
                                  torch::nn::LayerNorm(
                                      torch::nn::LayerNormOptions({cfg.hidden_size})
                                          .eps(1e-6)));
}

std::vector<torch::Tensor> Dinov2ModelImpl::forward(torch::Tensor x) {
  int patch = static_cast<int>(embeddings->patch_embeddings->projection
                                    ->options.kernel_size()->at(0));
  int Hg = x.size(2) / patch;
  int Wg = x.size(3) / patch;
  auto B = x.size(0);
  int  W = embeddings->cls_token.size(0) > 0
              ? /*derived from cfg via embeddings:*/0 : 0;
  (void)W;
  // Re-derive num_windows from the embeddings module's behaviour:
  // its output may be [B, Hg*Wg+1, C] (no windows) or
  // [B*W², (Hg/W)*(Wg/W)+1, C] (windowed). We pull num_windows from
  // the layer's `is_windowed_` table indirectly — easier: figure it
  // out from the encoder layer impl.
  auto tokens = embeddings->forward(x);
  auto taps   = encoder->forward(tokens);
  if (taps.empty()) return {};

  // Apply final LN on the last tap's output (matches HF behaviour).
  taps.back() = layernorm->forward(taps.back());

  // Compute window count from token shape.
  auto C = taps[0].size(-1);
  int64_t Bw = taps[0].size(0);          // = B * num_windows²
  int64_t Hw = taps[0].size(1) - 1;       // tokens per window minus cls
  int     num_windows = static_cast<int>(std::sqrt(static_cast<double>(Bw / B)));
  int     Hpw = static_cast<int>(std::sqrt(static_cast<double>(Hw)));   // patches per window side

  std::vector<torch::Tensor> spatial;
  spatial.reserve(taps.size());
  for (auto& t : taps) {
    auto patch = t.slice(/*dim=*/1, 1).contiguous();         // [Bw, Hw, C]
    if (num_windows <= 1) {
      spatial.push_back(patch.transpose(1, 2).view({B, C, Hg, Wg}));
      continue;
    }
    // Un-window: [B*W², Hpw*Hpw, C] → [B, W, W, Hpw, Hpw, C] → [B, W, Hpw, W, Hpw, C] → [B, Hg, Wg, C]
    auto un = patch.view({B, num_windows, num_windows, Hpw, Hpw, C})
                    .permute({0, 1, 3, 2, 4, 5})
                    .contiguous()
                    .view({B, Hg, Wg, C});
    // [B, Hg, Wg, C] → [B, C, Hg, Wg]
    spatial.push_back(un.permute({0, 3, 1, 2}).contiguous());
  }
  return spatial;
}

Dinov2WrapperImpl::Dinov2WrapperImpl(const Dinov2Cfg& cfg) {
  encoder = register_module("encoder", Dinov2Model(cfg));
}

std::vector<torch::Tensor> Dinov2WrapperImpl::forward(torch::Tensor x) {
  return encoder->forward(x);
}

Dinov2WrapperOuterImpl::Dinov2WrapperOuterImpl(const Dinov2Cfg& cfg) {
  encoder = register_module("encoder", Dinov2Wrapper(cfg));
}

std::vector<torch::Tensor> Dinov2WrapperOuterImpl::forward(torch::Tensor x) {
  return encoder->forward(x);
}

// ─── Legacy scaffold (kept for #65B/C/E/F/G linkage until rewrites land) ─

const BackboneCfg kDinoV2LargeCfg{14, 1024, 24, 16, 4, 0,   560, {11, 17, 23}};
const BackboneCfg kLwDetrTinyCfg {16, 192,   6,  3, 4, 14,  640, {2, 4, 5}};
const BackboneCfg kLwDetrSmallCfg{16, 384,   8,  6, 4, 14,  640, {3, 5, 7}};
const BackboneCfg kLwDetrBaseCfg {16, 512,  10,  8, 4, 14,  640, {4, 7, 9}};
const BackboneCfg kLwDetrMediumCfg{16, 768, 12, 12, 4, 14,  640, {5, 8, 11}};

const BackboneCfg& backbone_cfg_from_name(const std::string& backbone) {
  if (backbone == "dinov2-large")    return kDinoV2LargeCfg;
  if (backbone == "lw-detr-tiny")    return kLwDetrTinyCfg;
  if (backbone == "lw-detr-small")   return kLwDetrSmallCfg;
  if (backbone == "lw-detr-base")    return kLwDetrBaseCfg;
  if (backbone == "lw-detr-medium")  return kLwDetrMediumCfg;
  throw std::runtime_error("rfdetr backbone: unknown name '" + backbone + "'");
}

PatchEmbedImpl::PatchEmbedImpl(int in_ch, int embed_dim, int patch_size,
                                int img_size)
    : grid_size_(img_size / patch_size) {
  proj_ = register_module(
      "proj",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, embed_dim, patch_size)
                            .stride(patch_size)));
}
torch::Tensor PatchEmbedImpl::forward(torch::Tensor x) {
  return proj_->forward(x).flatten(2).transpose(1, 2);
}

AttentionImpl::AttentionImpl(int dim, int num_heads, int window_size,
                              int grid_size)
    : num_heads_(num_heads), head_dim_(dim / num_heads),
      window_size_(window_size), grid_size_(grid_size) {
  qkv_  = register_module("qkv",
                          torch::nn::Linear(torch::nn::LinearOptions(dim, dim * 3)
                                              .bias(true)));
  proj_ = register_module("proj", torch::nn::Linear(dim, dim));
}
torch::Tensor AttentionImpl::forward(torch::Tensor x) {
  auto B = x.size(0), N = x.size(1), C = x.size(2);
  bool windowed = window_size_ > 0 && grid_size_ > 0 &&
                  grid_size_ % window_size_ == 0 && N == grid_size_ * grid_size_;
  if (windowed) {
    int H = grid_size_, W = grid_size_, w = window_size_;
    int nWh = H / w, nWw = W / w;
    x = x.view({B, H, W, C}).view({B, nWh, w, nWw, w, C})
            .permute({0, 1, 3, 2, 4, 5}).contiguous()
            .view({B * nWh * nWw, w * w, C});
  }
  auto N2 = x.size(1);
  auto qkv = qkv_->forward(x).view({x.size(0), N2, 3, num_heads_, head_dim_})
                  .permute({2, 0, 3, 1, 4});
  auto q = qkv[0], k = qkv[1], v = qkv[2];
  auto scale = 1.0 / std::sqrt(static_cast<double>(head_dim_));
  auto attn  = torch::softmax(torch::matmul(q, k.transpose(-2, -1)) * scale, -1);
  auto out   = torch::matmul(attn, v).transpose(1, 2).contiguous()
                  .view({x.size(0), N2, num_heads_ * head_dim_});
  out = proj_->forward(out);
  if (windowed) {
    int H = grid_size_, W = grid_size_, w = window_size_;
    int nWh = H / w, nWw = W / w;
    out = out.view({B, nWh, nWw, w, w, C}).permute({0, 1, 3, 2, 4, 5})
              .contiguous().view({B, H * W, C});
  }
  return out;
}

ViTBlockImpl::ViTBlockImpl(int dim, int num_heads, int mlp_ratio,
                            int window_size, int grid_size) {
  norm1_ = register_module("norm1", torch::nn::LayerNorm(
                                          torch::nn::LayerNormOptions({dim})));
  attn_  = register_module("attn",
                            Attention(dim, num_heads, window_size, grid_size));
  norm2_ = register_module("norm2", torch::nn::LayerNorm(
                                          torch::nn::LayerNormOptions({dim})));
  fc1_   = register_module("fc1", torch::nn::Linear(dim, dim * mlp_ratio));
  fc2_   = register_module("fc2", torch::nn::Linear(dim * mlp_ratio, dim));
}
torch::Tensor ViTBlockImpl::forward(torch::Tensor x) {
  x = x + attn_->forward(norm1_->forward(x));
  return x + fc2_->forward(torch::gelu(fc1_->forward(norm2_->forward(x))));
}

ViTBackboneImpl::ViTBackboneImpl(const BackboneCfg& cfg) : cfg_(cfg) {
  patch_embed_ = register_module(
      "patch_embed",
      PatchEmbed(/*in_ch=*/3, cfg.embed_dim, cfg.patch_size, cfg.img_size));
  int num_patches = patch_embed_->grid_size() * patch_embed_->grid_size();
  cls_token_ = register_parameter("cls_token", torch::zeros({1, 1, cfg.embed_dim}));
  pos_embed_ = register_parameter("pos_embed",
                                    torch::zeros({1, num_patches + 1, cfg.embed_dim}));
  blocks_ = torch::nn::ModuleList();
  for (int i = 0; i < cfg.depth; ++i) {
    int win = cfg.window_size;
    if (i == cfg.depth - 1) win = 0;
    blocks_->push_back(ViTBlock(cfg.embed_dim, cfg.num_heads,
                                  cfg.mlp_ratio_x4, win,
                                  patch_embed_->grid_size()));
  }
  register_module("blocks", blocks_);
  norm_ = register_module("norm", torch::nn::LayerNorm(
                                        torch::nn::LayerNormOptions({cfg.embed_dim})));
}
std::vector<torch::Tensor> ViTBackboneImpl::forward_features(torch::Tensor x) {
  auto B = x.size(0);
  x = patch_embed_->forward(x);
  auto cls = cls_token_.expand({B, -1, -1});
  x = torch::cat({cls, x}, 1) + pos_embed_;
  std::vector<torch::Tensor> taps;
  taps.reserve(cfg_.tap_blocks.size());
  for (int i = 0; i < static_cast<int>(blocks_->size()); ++i) {
    x = blocks_[i]->as<ViTBlockImpl>()->forward(x);
    for (int t : cfg_.tap_blocks) {
      if (t == i) {
        auto patch = x.slice(1, 1).contiguous();
        int  g     = patch_embed_->grid_size();
        taps.push_back(patch.transpose(1, 2).view({B, cfg_.embed_dim, g, g}));
        break;
      }
    }
  }
  (void)norm_;
  return taps;
}

}  // namespace yolocpp::models::rfdetr
