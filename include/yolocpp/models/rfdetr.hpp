#pragma once
//
// RF-DETR — Roboflow's open-source DETR-family detector (#65).
//
// Architecture (very different from the YOLO family):
//   - Backbone: DINOv2 ViT (large variant) or LW-DETR transformer
//                backbone (smaller variants). Outputs per-stage feature
//                maps that the encoder consumes.
//   - Encoder:  multi-scale deformable transformer.
//   - Decoder:  transformer decoder with N learnable object queries.
//   - Head:     per-query (4-channel bbox + nc-channel cls). NO NMS at
//                inference — set prediction; the queries ARE the dets.
//   - Loss:     Hungarian bipartite matching + focal cls + L1 + GIoU
//                (training-time only).
//   - Segment variant: query-level mask coefficients × shared protos.
//
// **Status — scaffolded, not implemented.** This header declares the
// holder + scale enum + scale-from-letter helper so downstream
// machinery (registry, CLI filename resolution, weight converter)
// can be wired. Forward / train / export entry points throw
// `std::runtime_error("rfdetr <area> not yet implemented — see TODO
// #65X")` so the dispatch surface never silently produces wrong
// output. Each TODO sub-task in `TODO.md` (#65A..#65L) corresponds
// to one landable slice.
//
// Variants (Roboflow's released family):
//   nano   (n)   — 30M params, smallest backbone
//   small  (s)   — 60M
//   base   (b)   — 130M, default
//   medium (m)   — 250M
//   large  (l)   — 500M, DINOv2 backbone
// Plus an `rfdetr-seg-preview` mask variant.

#include <torch/torch.h>

#include <string>
#include <vector>

#include "yolocpp/models/rfdetr_backbone.hpp"
#include "yolocpp/models/rfdetr_decoder.hpp"
#include "yolocpp/models/rfdetr_encoder.hpp"

namespace yolocpp::models {

// Scale parameters for RF-DETR 1.6.5. The values below are the
// REAL per-variant hyperparameters from the upstream
// `rfdetr.detr.RFDETR<X>Config` classes — see `docs/rfdetr_arch.md`
// for the full ground-truth inventory dumped from the 12 official
// `.pth` / `.pt` files.
//
// All variants share `dinov2_windowed_small` (12 ViT blocks,
// embed=384, separate Q/K/V) — except `rfdetr-large` which uses
// `dinov2_windowed_base` (12 blocks, embed=768). Variants differ
// in: input resolution, ViT patch size, number of decoder layers,
// hidden dim (256 except large=384), and number of object queries.
struct RFDetrScale {
  int         num_queries     = 300;
  int         hidden_dim      = 256;
  int         num_dec_layers  = 3;
  int         sa_nheads       = 8;       // self-attn heads
  int         ca_nheads       = 16;      // cross-attn heads (deformable)
  int         dec_n_points    = 2;       // deformable points per head
  int         num_classes     = 90;      // upstream COCO schema (cls=91 incl. bg)
  int         group_detr      = 13;      // training-time query groups
  int         resolution      = 560;     // letterbox side
  int         patch_size      = 14;      // ViT patch size
  int         backbone_embed  = 384;     // 768 for large
  bool        is_segment      = false;
  const char* family          = "dinov2_windowed_small";
  const char* upstream_id     = "base";  // for converter routing
};

// Detect: 5 variants
constexpr RFDetrScale kRfdetrNano   {300, 256, 2, 8, 16, 2, 90, 13, 384, 16, 384, false,
                                       "dinov2_windowed_small", "nano"};
constexpr RFDetrScale kRfdetrSmall  {300, 256, 3, 8, 16, 2, 90, 13, 512, 16, 384, false,
                                       "dinov2_windowed_small", "small"};
constexpr RFDetrScale kRfdetrMedium {300, 256, 4, 8, 16, 2, 90, 13, 576, 16, 384, false,
                                       "dinov2_windowed_small", "medium"};
constexpr RFDetrScale kRfdetrBase   {300, 256, 3, 8, 16, 2, 90, 13, 560, 14, 384, false,
                                       "dinov2_windowed_small", "base"};
constexpr RFDetrScale kRfdetrLarge  {300, 384, 3, 8, 16, 2, 90, 13, 704, 16, 768, false,
                                       "dinov2_windowed_base", "large"};

// Segment: 7 variants (use the v8/v11 segment-task wrapper at the
// public-API layer; the underlying RFDetrSegment shares the detect
// stack + an extra mask head).
constexpr RFDetrScale kRfdetrSegNano    {100, 256, 4, 8, 16, 2, 90, 13, 368, 14, 384, true,
                                          "dinov2_windowed_small", "seg-nano"};
constexpr RFDetrScale kRfdetrSegSmall   {100, 256, 4, 8, 16, 2, 90, 13, 512, 16, 384, true,
                                          "dinov2_windowed_small", "seg-small"};
constexpr RFDetrScale kRfdetrSegMedium  {200, 256, 5, 8, 16, 2, 90, 13, 576, 16, 384, true,
                                          "dinov2_windowed_small", "seg-medium"};
constexpr RFDetrScale kRfdetrSegLarge   {300, 256, 5, 8, 16, 2, 90, 13, 672, 16, 384, true,
                                          "dinov2_windowed_small", "seg-large"};
constexpr RFDetrScale kRfdetrSegXLarge  {300, 256, 6, 8, 16, 2, 90, 13, 624, 12, 384, true,
                                          "dinov2_windowed_small", "seg-xlarge"};
constexpr RFDetrScale kRfdetrSegXXLarge {300, 256, 6, 8, 16, 2, 90, 13, 768, 12, 384, true,
                                          "dinov2_windowed_small", "seg-xxlarge"};
constexpr RFDetrScale kRfdetrSegPreview {200, 256, 4, 8, 16, 2, 90, 13, 432, 12, 384, true,
                                          "dinov2_windowed_small", "seg-preview"};

// Letter → scale resolver. Filename convention `rfdetr-<n|s|b|m|l>.pt`
// (matching the Roboflow release pattern). Falls back to `base` for
// unrecognised letters so behaviour stays consistent with the other
// `yolo<N>_scale_from_letter` helpers.
RFDetrScale rfdetr_scale_from_letter(const std::string& letter);

// Default `imgsz` per scale. RF-DETR's pre-trained weights expect
// 560 px input (DINOv2 patch size 14 × 40 patches per side), but
// nano/small were retrained at 640 to match the YOLO ecosystem.
int rfdetr_default_imgsz(const RFDetrScale& scale);

// ─── Detect holder ────────────────────────────────────────────────────────
//
// **Forward / train / state-dict load all THROW** until #65A..C land.
// Constructor stores the scale + nc; declares (but doesn't populate)
// the module sub-tree shape so `named_parameters()` returns
// stable-ordered names that future converter work (#65D) can match
// against.
class RFDetrImpl : public torch::nn::Module {
 public:
  explicit RFDetrImpl(RFDetrScale scale = kRfdetrBase, int nc = 80);

  // [B, 3, H, W] → [B, 4 + nc, num_queries] sigmoided. Throws.
  torch::Tensor forward_eval(torch::Tensor x);

  // Returns the per-decoder-layer query embeddings + bbox/cls heads,
  // for Hungarian-matching loss. Throws.
  std::vector<torch::Tensor> forward_train(torch::Tensor x);

  // Loads upstream Roboflow `rfdetr-<scale>.pt` keys. Throws today;
  // implementation under #65D will fuse + rename keys to match our
  // parameter layout.
  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);

  // Stride is conceptual for DETR-family models (output is per-query
  // not per-anchor); `nc` + `num_queries` are the meaningful shape
  // accessors.
  int                num_queries()  const { return scale.num_queries; }

  // Trainer template (`engine::TrainerT<M>`) reads these as plain
  // fields, matching the YOLO model convention. `stride` is fake
  // for RF-DETR (set prediction has no FPN strides) — the
  // LossTraits specialization ignores it.
  RFDetrScale         scale;
  int                 nc;
  std::vector<double> stride{1.0};

  // #65A — backbone runs end-to-end and returns multi-scale feature
  // maps. Used directly by `forward_eval` once the head (#65C) lands;
  // exposed publicly so unit tests can pin the backbone shape
  // without going through the throwing forward path.
  std::vector<torch::Tensor> forward_backbone(torch::Tensor x);

  // #65B — backbone + encoder run end-to-end. Returns the full
  // encoder output (memory + multi-scale helpers) which the decoder
  // (#65C) consumes via cross-attn.
  yolocpp::models::rfdetr::EncoderOutput forward_encoder(torch::Tensor x);

 private:
  yolocpp::models::rfdetr::ViTBackbone backbone_{nullptr};
  yolocpp::models::rfdetr::Encoder     encoder_{nullptr};
  yolocpp::models::rfdetr::DetrHead    head_{nullptr};
};

TORCH_MODULE(RFDetr);

// Segment variant. Same backbone + encoder + decoder; adds a
// per-query mask coefficient head + a shared proto module on top of
// the encoder output. Matches Yolo8Segment's high-level shape so
// the existing v8-task validator can plug in once #65K lands.
class RFDetrSegmentImpl : public torch::nn::Module {
 public:
  explicit RFDetrSegmentImpl(RFDetrScale scale = kRfdetrBase, int nc = 80);

  // Returns (per-query box+cls, mask coefficients, prototype masks).
  // All three throw today.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);

  RFDetrScale         scale;
  int                 nc;
  std::vector<double> stride{1.0};

  std::vector<torch::Tensor> forward_backbone(torch::Tensor x);

 private:
  yolocpp::models::rfdetr::ViTBackbone backbone_{nullptr};
};

TORCH_MODULE(RFDetrSegment);

}  // namespace yolocpp::models
