// RF-DETR backbones (#65A) — DINOv2 ViT-L + LW-DETR ViT family.
//
// See header for the overall design rationale. Both families share
// the same ViTBlock; LW-DETR adds windowed attention via a reshape.
// Parameter names are registered to match the upstream Roboflow /
// DINOv2 release layout so the #65D state-dict converter can do a
// simple string-prefix remap (`backbone.encoder.layer.<i>.` → our
// `blocks.<i>.`) without per-tensor surgery.

#include "yolocpp/models/rfdetr_backbone.hpp"

#include <stdexcept>

namespace yolocpp::models::rfdetr {

const BackboneCfg kDinoV2LargeCfg{
    /*patch=*/14, /*embed=*/1024, /*depth=*/24, /*heads=*/16,
    /*mlpx4=*/4,  /*window=*/0,    /*img=*/560,
    /*taps=*/{11, 17, 23}};

const BackboneCfg kLwDetrTinyCfg{
    /*patch=*/16, /*embed=*/192,  /*depth=*/6,  /*heads=*/3,
    /*mlpx4=*/4,  /*window=*/14,   /*img=*/640,
    /*taps=*/{2, 4, 5}};

const BackboneCfg kLwDetrSmallCfg{
    /*patch=*/16, /*embed=*/384,  /*depth=*/8,  /*heads=*/6,
    /*mlpx4=*/4,  /*window=*/14,   /*img=*/640,
    /*taps=*/{3, 5, 7}};

const BackboneCfg kLwDetrBaseCfg{
    /*patch=*/16, /*embed=*/512,  /*depth=*/10, /*heads=*/8,
    /*mlpx4=*/4,  /*window=*/14,   /*img=*/640,
    /*taps=*/{4, 7, 9}};

const BackboneCfg kLwDetrMediumCfg{
    /*patch=*/16, /*embed=*/768,  /*depth=*/12, /*heads=*/12,
    /*mlpx4=*/4,  /*window=*/14,   /*img=*/640,
    /*taps=*/{5, 8, 11}};

const BackboneCfg& backbone_cfg_from_name(const std::string& backbone) {
  if (backbone == "dinov2-large")    return kDinoV2LargeCfg;
  if (backbone == "lw-detr-tiny")    return kLwDetrTinyCfg;
  if (backbone == "lw-detr-small")   return kLwDetrSmallCfg;
  if (backbone == "lw-detr-base")    return kLwDetrBaseCfg;
  if (backbone == "lw-detr-medium")  return kLwDetrMediumCfg;
  throw std::runtime_error("rfdetr backbone: unknown name '" + backbone +
                           "' (#65A)");
}

// ─── PatchEmbed ─────────────────────────────────────────────────────────

PatchEmbedImpl::PatchEmbedImpl(int in_ch, int embed_dim, int patch_size,
                                int img_size)
    : grid_size_(img_size / patch_size) {
  proj_ = register_module(
      "proj",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, embed_dim, patch_size)
                            .stride(patch_size)));
}

torch::Tensor PatchEmbedImpl::forward(torch::Tensor x) {
  // [B, 3, H, W] → [B, C, H/p, W/p] → [B, N, C]
  x = proj_->forward(x);
  x = x.flatten(2).transpose(1, 2);
  return x;
}

// ─── Attention (with optional windowing) ────────────────────────────────

AttentionImpl::AttentionImpl(int dim, int num_heads, int window_size,
                              int grid_size)
    : num_heads_(num_heads), head_dim_(dim / num_heads),
      window_size_(window_size), grid_size_(grid_size) {
  if (dim % num_heads != 0) {
    throw std::invalid_argument("rfdetr attn: dim must divide num_heads");
  }
  qkv_  = register_module("qkv",
                          torch::nn::Linear(torch::nn::LinearOptions(dim, dim * 3)
                                               .bias(true)));
  proj_ = register_module("proj", torch::nn::Linear(dim, dim));
}

torch::Tensor AttentionImpl::forward(torch::Tensor x) {
  // Input shape: [B, N, C]. For windowed attn, partition into
  // [B·nW, win², C] before the QKV projection.
  auto B = x.size(0);
  auto N = x.size(1);
  auto C = x.size(2);

  bool windowed = window_size_ > 0 && grid_size_ > 0 &&
                  grid_size_ % window_size_ == 0 && N == grid_size_ * grid_size_;

  if (windowed) {
    int H = grid_size_, W = grid_size_, w = window_size_;
    int nWh = H / w, nWw = W / w;
    // [B, H, W, C] → [B, nWh, w, nWw, w, C] → [B·nW, w², C]
    x = x.view({B, H, W, C})
            .view({B, nWh, w, nWw, w, C})
            .permute({0, 1, 3, 2, 4, 5})
            .contiguous()
            .view({B * nWh * nWw, w * w, C});
  }

  auto N2 = x.size(1);
  auto qkv = qkv_->forward(x);  // [B', N, 3C]
  qkv = qkv.view({x.size(0), N2, 3, num_heads_, head_dim_})
            .permute({2, 0, 3, 1, 4});  // [3, B', H, N, D]
  auto q = qkv[0], k = qkv[1], v = qkv[2];

  auto scale  = 1.0 / std::sqrt(static_cast<double>(head_dim_));
  auto attn   = torch::matmul(q, k.transpose(-2, -1)) * scale;
  attn        = torch::softmax(attn, -1);
  auto out    = torch::matmul(attn, v);                  // [B', H, N, D]
  out         = out.transpose(1, 2).contiguous()         // [B', N, H, D]
                    .view({x.size(0), N2, num_heads_ * head_dim_});
  out         = proj_->forward(out);

  if (windowed) {
    int H = grid_size_, W = grid_size_, w = window_size_;
    int nWh = H / w, nWw = W / w;
    out = out.view({B, nWh, nWw, w, w, C})
              .permute({0, 1, 3, 2, 4, 5})
              .contiguous()
              .view({B, H * W, C});
  }
  return out;
}

// ─── ViTBlock ───────────────────────────────────────────────────────────

ViTBlockImpl::ViTBlockImpl(int dim, int num_heads, int mlp_ratio,
                            int window_size, int grid_size) {
  norm1_ = register_module("norm1", torch::nn::LayerNorm(
                                         torch::nn::LayerNormOptions({dim})));
  attn_  = register_module(
      "attn", Attention(dim, num_heads, window_size, grid_size));
  norm2_ = register_module("norm2", torch::nn::LayerNorm(
                                         torch::nn::LayerNormOptions({dim})));
  fc1_   = register_module("fc1", torch::nn::Linear(dim, dim * mlp_ratio));
  fc2_   = register_module("fc2", torch::nn::Linear(dim * mlp_ratio, dim));
}

torch::Tensor ViTBlockImpl::forward(torch::Tensor x) {
  x = x + attn_->forward(norm1_->forward(x));
  auto y = norm2_->forward(x);
  y      = fc2_->forward(torch::gelu(fc1_->forward(y)));
  return x + y;
}

// ─── ViTBackbone ────────────────────────────────────────────────────────

ViTBackboneImpl::ViTBackboneImpl(const BackboneCfg& cfg) : cfg_(cfg) {
  patch_embed_ = register_module(
      "patch_embed",
      PatchEmbed(/*in_ch=*/3, cfg.embed_dim, cfg.patch_size, cfg.img_size));

  int num_patches = patch_embed_->grid_size() * patch_embed_->grid_size();
  cls_token_ = register_parameter(
      "cls_token", torch::zeros({1, 1, cfg.embed_dim}));
  pos_embed_ = register_parameter(
      "pos_embed", torch::zeros({1, num_patches + 1, cfg.embed_dim}));

  blocks_ = torch::nn::ModuleList();
  for (int i = 0; i < cfg.depth; ++i) {
    // LW-DETR uses windowed attention everywhere except the last
    // block (which gets full self-attention to mix windows). DINOv2
    // (window_size=0) always uses full attention.
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
  // [B, 3, H, W] → tokens [B, N, C] (we drop the cls token after the
  // forward pass since the encoder consumes patch tokens only — the
  // cls token is kept only for state-dict shape compatibility with
  // upstream).
  auto B = x.size(0);
  x = patch_embed_->forward(x);                                   // [B, N, C]
  auto cls = cls_token_.expand({B, -1, -1});
  x = torch::cat({cls, x}, /*dim=*/1) + pos_embed_;

  std::vector<torch::Tensor> taps;
  taps.reserve(cfg_.tap_blocks.size());
  for (int i = 0; i < static_cast<int>(blocks_->size()); ++i) {
    x = blocks_[i]->as<ViTBlockImpl>()->forward(x);
    for (int t : cfg_.tap_blocks) {
      if (t == i) {
        // Drop cls, reshape to [B, C, Hg, Wg]
        auto patch = x.slice(/*dim=*/1, /*start=*/1).contiguous();
        int  g     = patch_embed_->grid_size();
        taps.push_back(patch.transpose(1, 2)
                            .view({B, cfg_.embed_dim, g, g}));
        break;
      }
    }
  }
  // Final LN is applied on the cls-included sequence in upstream;
  // we keep the parameter but don't use its output here since the
  // encoder consumes tap features directly.
  (void)norm_;
  return taps;
}

}  // namespace yolocpp::models::rfdetr
