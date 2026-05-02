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
  if (letter == "n" || letter == "nano")   return kRfdetrNano;
  if (letter == "s" || letter == "small")  return kRfdetrSmall;
  if (letter == "b" || letter == "base"
      || letter.empty())                   return kRfdetrBase;
  if (letter == "m" || letter == "medium") return kRfdetrMedium;
  if (letter == "l" || letter == "large")  return kRfdetrLarge;
  return kRfdetrBase;  // permissive default
}

int rfdetr_default_imgsz(const RFDetrScale& scale) {
  // DINOv2-large was trained at 560 (40 × 14 patches). The smaller
  // variants were retrained at 640 to fit the YOLO-ecosystem
  // tooling (letterbox, NMS-free decoders, etc.).
  if (std::string_view(scale.backbone) == "dinov2-large") return 560;
  return 640;
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

RFDetrImpl::RFDetrImpl(RFDetrScale scale, int nc) : scale_(scale), nc_(nc) {
  // #65A landed: backbone is constructed + registered. #65B encoder,
  // #65C decoder/head, #65F loss heads still pending — `forward_eval`
  // and `forward_train` throw on the encoder boundary below.
  const auto& cfg =
      yolocpp::models::rfdetr::backbone_cfg_from_name(scale.backbone);
  backbone_ = register_module(
      "backbone", yolocpp::models::rfdetr::ViTBackbone(cfg));
}

std::vector<torch::Tensor> RFDetrImpl::forward_backbone(torch::Tensor x) {
  return backbone_->forward_features(std::move(x));
}

torch::Tensor RFDetrImpl::forward_eval(torch::Tensor x) {
  // Backbone runs (real shapes), then we hit the encoder boundary.
  (void)forward_backbone(std::move(x));
  unimplemented("forward_eval (encoder→decoder→head)",
                "#65B (encoder) + #65C (decoder/head)");
}

std::vector<torch::Tensor> RFDetrImpl::forward_train(torch::Tensor /*x*/) {
  unimplemented("forward_train", "#65F (Hungarian-matching loss surface)");
}

int RFDetrImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& /*entries*/) {
  unimplemented("load_from_state_dict",
                "#65D (rfdetr-<scale>.pt → our .pt converter)");
}

// ─── Segment ─────────────────────────────────────────────────────────────

RFDetrSegmentImpl::RFDetrSegmentImpl(RFDetrScale scale, int nc)
    : scale_(scale), nc_(nc) {
  const auto& cfg =
      yolocpp::models::rfdetr::backbone_cfg_from_name(scale.backbone);
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
