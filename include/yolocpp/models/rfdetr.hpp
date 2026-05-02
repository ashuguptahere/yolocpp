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

namespace yolocpp::models {

// Scale parameters for RF-DETR. Channel widths + transformer depth +
// number of queries differ per variant; the actual values get filled
// in under #65A (backbone) and #65B (transformer) — placeholder
// constants here just disambiguate the enum identity.
struct RFDetrScale {
  int  num_queries = 300;
  int  hidden_dim  = 256;
  int  num_encoder_layers = 6;
  int  num_decoder_layers = 6;
  int  num_heads   = 8;
  // Backbone family — dinov2 only at "large", lw-detr-style at the
  // smaller variants. Kept as a string until the backbone module is
  // implemented (#65A).
  const char* backbone = "lw-detr";
};

constexpr RFDetrScale kRfdetrNano  {/*Q=*/100, /*hd=*/192, 3, 3, 6, "lw-detr-tiny"};
constexpr RFDetrScale kRfdetrSmall {/*Q=*/200, /*hd=*/256, 4, 4, 8, "lw-detr-small"};
constexpr RFDetrScale kRfdetrBase  {/*Q=*/300, /*hd=*/256, 6, 6, 8, "lw-detr-base"};
constexpr RFDetrScale kRfdetrMedium{/*Q=*/300, /*hd=*/384, 6, 6, 8, "lw-detr-medium"};
constexpr RFDetrScale kRfdetrLarge {/*Q=*/300, /*hd=*/512, 6, 6, 8, "dinov2-large"};

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
  int                nc()           const { return nc_; }
  int                num_queries()  const { return scale_.num_queries; }
  const RFDetrScale& scale()        const { return scale_; }

 private:
  RFDetrScale scale_;
  int         nc_;
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

  int                nc()    const { return nc_; }
  const RFDetrScale& scale() const { return scale_; }

 private:
  RFDetrScale scale_;
  int         nc_;
};

TORCH_MODULE(RFDetrSegment);

}  // namespace yolocpp::models
