// RF-DETR scaffolding (#65). Forward / train / load_state_dict all
// THROW pending each module's implementation slice (see TODO.md
// #65A..#65L). Wiring this header now means the registry adapter,
// the CLI filename resolver, and downstream callers all compile and
// link cleanly while the actual transformer plumbing lands one
// slice at a time.

#include "yolocpp/models/rfdetr.hpp"

#include <stdexcept>

namespace yolocpp::models {

RFDetrScale rfdetr_scale_from_letter(const std::string& letter) {
  // Detect variants — single-letter or full-word matchers.
  if (letter == "n"  || letter == "nano")   return kRfdetrNano;
  if (letter == "s"  || letter == "small")  return kRfdetrSmall;
  if (letter == "m"  || letter == "medium") return kRfdetrMedium;
  if (letter == "b"  || letter == "base"
      || letter.empty())                    return kRfdetrBase;
  if (letter == "l"  || letter == "large")  return kRfdetrLarge;
  // Segment variants — `seg-<n|s|m|l|xl|xxl|preview>`.
  if (letter == "seg-n"       || letter == "seg-nano")    return kRfdetrSegNano;
  if (letter == "seg-s"       || letter == "seg-small")   return kRfdetrSegSmall;
  if (letter == "seg-m"       || letter == "seg-medium")  return kRfdetrSegMedium;
  if (letter == "seg-l"       || letter == "seg-large")   return kRfdetrSegLarge;
  if (letter == "seg-xl"      || letter == "seg-xlarge")  return kRfdetrSegXLarge;
  if (letter == "seg-xxl"     || letter == "seg-xxlarge"
      || letter == "seg-2xl"  || letter == "seg-2xlarge") return kRfdetrSegXXLarge;
  if (letter == "seg-preview")                            return kRfdetrSegPreview;
  return kRfdetrBase;  // permissive default
}

int rfdetr_default_imgsz(const RFDetrScale& scale) {
  // Per-variant resolution from upstream `rfdetr.detr.RFDETR<X>Config`
  // — captured into RFDetrScale.resolution so callers can override
  // via `--imgsz` but the default tracks the trained config.
  return scale.resolution;
}

[[noreturn]] static void unimplemented(const char* area, const char* slice_id) {
  throw std::runtime_error(
      std::string("rfdetr ") + area +
      ": not yet implemented — tracked under TODO " + slice_id +
      " (see TODO.md). The RF-DETR architecture (transformer "
      "encoder/decoder + object queries + Hungarian-matching loss) "
      "lands one slice at a time; predict / val / train / export "
      "entry points throw until each piece is wired.");
}

// ─── Detect ──────────────────────────────────────────────────────────────

RFDetrImpl::RFDetrImpl(RFDetrScale scale_in, int nc_in)
    : scale(scale_in), nc(nc_in) {
  // #65A..C landed; train integration (#65G) reads `scale`, `nc`,
  // and `stride` directly as fields (mirrors the YOLO model
  // convention).
  // The scaffold backbone/encoder/decoder modules are placeholders
  // until #65A2..F2 replace them with the real RF-DETR 1.6.5
  // architecture (see `docs/rfdetr_arch.md`). The mapping below
  // wires the placeholder modules with reasonable shapes so the
  // forward path stays runnable in the meantime — `load_from_state_dict`
  // still throws on the converter slice (#65D2) until the modules
  // are rewritten to match upstream key names.
  static constexpr const char* kPlaceholderBackbone = "lw-detr-tiny";
  const auto& bcfg =
      yolocpp::models::rfdetr::backbone_cfg_from_name(kPlaceholderBackbone);
  backbone_ = register_module(
      "backbone", yolocpp::models::rfdetr::ViTBackbone(bcfg));
  std::vector<int> in_channels(bcfg.tap_blocks.size(), bcfg.embed_dim);
  encoder_ = register_module(
      "encoder",
      yolocpp::models::rfdetr::Encoder(
          in_channels, scale.hidden_dim, scale.sa_nheads,
          /*num_layers=*/1, /*num_points=*/4));
  head_ = register_module(
      "head",
      yolocpp::models::rfdetr::DetrHead(
          scale.hidden_dim, scale.sa_nheads, scale.num_dec_layers,
          scale.num_queries, nc, /*num_points=*/4));
}

std::vector<torch::Tensor> RFDetrImpl::forward_backbone(torch::Tensor x) {
  return backbone_->forward_features(std::move(x));
}

yolocpp::models::rfdetr::EncoderOutput
RFDetrImpl::forward_encoder(torch::Tensor x) {
  auto feats = forward_backbone(std::move(x));
  return encoder_->forward(feats);
}

torch::Tensor RFDetrImpl::forward_eval(torch::Tensor x) {
  // Capture the input spatial dims before the encoder consumes it
  // — the head emits sigmoided cxcywh in [0, 1], we need to return
  // xyxy in pixel coords to match YOLO's `forward_eval` contract
  // (so `inference::nms` is drop-in compatible).
  int64_t H = x.size(2), W = x.size(3);
  auto enc = forward_encoder(std::move(x));
  auto out = head_->forward_eval(enc.memory, enc.spatial_shapes,
                                   enc.level_start_index);
  // out: [B, 4+nc, Q] cxcywh in [0,1] + sigmoided cls.
  auto cx = out.select(1, 0) * static_cast<double>(W);
  auto cy = out.select(1, 1) * static_cast<double>(H);
  auto w  = out.select(1, 2) * static_cast<double>(W);
  auto h  = out.select(1, 3) * static_cast<double>(H);
  auto x1 = cx - 0.5 * w, y1 = cy - 0.5 * h;
  auto x2 = cx + 0.5 * w, y2 = cy + 0.5 * h;
  auto xyxy = torch::stack({x1, y1, x2, y2}, /*dim=*/1);
  auto cls  = out.slice(1, 4);
  return torch::cat({xyxy, cls}, /*dim=*/1);
}

std::vector<torch::Tensor> RFDetrImpl::forward_train(torch::Tensor x) {
  auto enc = forward_encoder(std::move(x));
  auto outs = head_->forward_train(enc.memory, enc.spatial_shapes,
                                     enc.level_start_index);
  // Pack [(cls, bbox), ...] → flat vector for use by the
  // (still-pending) Hungarian loss surface (#65F).
  std::vector<torch::Tensor> flat;
  flat.reserve(outs.size() * 2);
  for (auto& [c, b] : outs) {
    flat.push_back(c);
    flat.push_back(b);
  }
  return flat;
}

int RFDetrImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& /*entries*/) {
  unimplemented("load_from_state_dict",
                "#65D (rfdetr-<scale>.pt → our .pt converter)");
}

// ─── Segment ─────────────────────────────────────────────────────────────

RFDetrSegmentImpl::RFDetrSegmentImpl(RFDetrScale scale_in, int nc_in)
    : scale(scale_in), nc(nc_in) {
  static constexpr const char* kPlaceholderBackbone = "lw-detr-tiny";
  const auto& cfg =
      yolocpp::models::rfdetr::backbone_cfg_from_name(kPlaceholderBackbone);
  backbone_ = register_module(
      "backbone", yolocpp::models::rfdetr::ViTBackbone(cfg));
}

std::vector<torch::Tensor> RFDetrSegmentImpl::forward_backbone(torch::Tensor x) {
  return backbone_->forward_features(std::move(x));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RFDetrSegmentImpl::forward_eval(torch::Tensor x) {
  (void)forward_backbone(std::move(x));
  unimplemented("seg forward_eval",
                "#65K (segment head: per-query mask coeffs + shared protos)");
}

int RFDetrSegmentImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& /*entries*/) {
  unimplemented("seg load_from_state_dict",
                "#65K (segment head + #65D converter for seg variant)");
}

}  // namespace yolocpp::models
