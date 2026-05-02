// RF-DETR scaffolding (#65). Forward / train / load_state_dict all
// THROW pending each module's implementation slice (see TODO.md
// #65A..#65L). Wiring this header now means the registry adapter,
// the CLI filename resolver, and downstream callers all compile and
// link cleanly while the actual transformer plumbing lands one
// slice at a time.

#include "yolocpp/models/rfdetr.hpp"

#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include "yolocpp/serialization/rfdetr_weights.hpp"

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
  // #65A2 — register the REAL backbone first under name "backbone".
  // ModuleList auto-numbers children, so slot 0 contains a
  // Dinov2WrapperOuter and the full path becomes
  // `backbone.0.encoder.encoder.embeddings.*` matching upstream key
  // names exactly so `rfdetr_weights::load_rfdetr_pt` binds in one
  // pass.
  const auto& real_cfg = yolocpp::models::rfdetr::dinov2_cfg_for(
      scale.upstream_id, scale.patch_size, scale.pretrain_grid,
      scale.backbone_embed);
  // Large variant has TWO projector stages (`stages.0` + `stages.1`);
  // every other variant has one. Detected via `upstream_id == "large"`.
  int n_proj_stages = (std::string(scale.upstream_id) == "large") ? 2 : 1;
  backbone_real_ = torch::nn::ModuleList();
  backbone_real_->push_back(
      yolocpp::models::rfdetr::BackboneSlot(real_cfg, scale.hidden_dim,
                                              n_proj_stages));
  register_module("backbone", backbone_real_);

  // #65C2/D2 — real transformer (decoder + two-stage encoder-output
  // siblings). Upstream's checkpoint stores 91 cls slots (90 COCO
  // classes + 1 background); we mirror that so the loaded weights
  // bind without slicing. The runtime user-facing nc may be smaller
  // — handled at the head reduction step in #65F2.
  transformer_ = register_module(
      "transformer",
      yolocpp::models::rfdetr::RFDetrTransformer(
          scale.hidden_dim, scale.num_dec_layers, scale.sa_nheads,
          scale.ca_nheads, scale.dec_n_points, /*ffn_dim=*/2048,
          scale.group_detr, /*n_classes_with_bg=*/scale.num_classes + 1));

  // #65C2 — shared cls/bbox heads + learnable query/refpoint
  // embeddings.
  class_embed_ = register_module(
      "class_embed",
      torch::nn::Linear(scale.hidden_dim, scale.num_classes + 1));
  bbox_embed_ = register_module(
      "bbox_embed",
      yolocpp::models::rfdetr::RFDetrMLP(
          scale.hidden_dim, scale.hidden_dim, /*output_dim=*/4,
          /*num_layers=*/3));
  // Match upstream's `refpoint_embed.weight` / `query_feat.weight`
  // paths — these are `nn.Embedding`'s `.weight` param. We register
  // each as a submodule whose sole parameter is named "weight".
  struct EmbeddingWeightImpl : torch::nn::Module {
    torch::Tensor weight;
    EmbeddingWeightImpl(int64_t num, int64_t dim) {
      weight = register_parameter("weight", torch::zeros({num, dim}));
    }
  };
  TORCH_MODULE_IMPL(EmbeddingWeight, EmbeddingWeightImpl);
  auto refpt = register_module(
      "refpoint_embed",
      std::make_shared<EmbeddingWeightImpl>(
          scale.num_queries * scale.group_detr, 4));
  auto qfeat = register_module(
      "query_feat",
      std::make_shared<EmbeddingWeightImpl>(
          scale.num_queries * scale.group_detr, scale.hidden_dim));
  refpoint_embed_ = refpt->weight;
  query_feat_     = qfeat->weight;

  // Legacy scaffold modules — placeholders so the existing forward
  // path keeps running until #65B2/C2/D2 replace them. Registered
  // under `_*_legacy` names so they NEVER collide with upstream
  // parameter names.
  static constexpr const char* kPlaceholderBackbone = "lw-detr-tiny";
  const auto& bcfg =
      yolocpp::models::rfdetr::backbone_cfg_from_name(kPlaceholderBackbone);
  backbone_ = register_module(
      "_backbone_legacy", yolocpp::models::rfdetr::ViTBackbone(bcfg));
  std::vector<int> in_channels(bcfg.tap_blocks.size(), bcfg.embed_dim);
  encoder_ = register_module(
      "_encoder_legacy",
      yolocpp::models::rfdetr::Encoder(
          in_channels, scale.hidden_dim, scale.sa_nheads,
          /*num_layers=*/1, /*num_points=*/4));
  head_ = register_module(
      "_head_legacy",
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
  // #65F2 — full real-arch forward through backbone+projector
  // → two-stage encoder-output → iterative-refinement decoder.
  int64_t H = x.size(2), W = x.size(3);
  // Backbone slot 0 = BackboneSlot (encoder + projector).
  auto& slot = *backbone_real_[0]->as<yolocpp::models::rfdetr::BackboneSlotImpl>();
  auto memory_2d = slot.forward(std::move(x));        // [B, hidden, Hg, Wg]

  // First-group query feats / refpoints: [:Q] (group_detr=1 at eval).
  int Q = scale.num_queries;
  auto qfeat_first  = query_feat_.slice(/*dim=*/0, 0, Q);
  auto rpemb_first  = refpoint_embed_.slice(/*dim=*/0, 0, Q);

  auto tf_out = transformer_->forward(memory_2d, qfeat_first, rpemb_first, Q);
  auto& out_feat = tf_out.decoder_out;                // [B, Q, hidden]
  auto& refpts   = tf_out.refpoints;                  // [B, Q, 4] cxcywh in [0,1]

  // Final cls + bbox heads.
  auto cls_logits = class_embed_->forward(out_feat);  // [B, Q, n_classes+1]
  auto bbox_delta = bbox_embed_->forward(out_feat);   // [B, Q, 4]
  // bbox_reparam refinement:
  auto cx_ = bbox_delta.slice(-1, 0, 2) * refpts.slice(-1, 2, 4) +
              refpts.slice(-1, 0, 2);
  auto wh_ = bbox_delta.slice(-1, 2, 4).exp() * refpts.slice(-1, 2, 4);
  // Convert to xyxy in pixel coords.
  auto cx = cx_.select(-1, 0) * static_cast<double>(W);
  auto cy = cx_.select(-1, 1) * static_cast<double>(H);
  auto w  = wh_.select(-1, 0) * static_cast<double>(W);
  auto h  = wh_.select(-1, 1) * static_cast<double>(H);
  auto x1 = cx - 0.5 * w, y1 = cy - 0.5 * h;
  auto x2 = cx + 0.5 * w, y2 = cy + 0.5 * h;
  auto xyxy = torch::stack({x1, y1, x2, y2}, /*dim=*/-1);   // [B, Q, 4]

  // Drop the trailing background class (slot index n_classes) and
  // sigmoid the rest to user-facing nc. RF-DETR's checkpoints store
  // `n_classes+1` slots; we expose the first `nc` (or all if user
  // passes nc=91).
  auto cls_sigmoid = torch::sigmoid(cls_logits);          // [B, Q, n_classes+1]
  int  user_nc     = std::min(nc, static_cast<int>(cls_sigmoid.size(-1)));
  auto cls_used    = cls_sigmoid.slice(-1, 0, user_nc);   // [B, Q, nc]

  // YOLO contract: [B, 4+nc, Q].
  auto out = torch::cat({xyxy, cls_used}, /*dim=*/-1);    // [B, Q, 4+nc]
  return out.transpose(1, 2).contiguous();                 // [B, 4+nc, Q]
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
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  // Bind upstream tensors onto same-named parameters/buffers on this
  // module. During the architecture rewrite (#65A2..D2) the scaffold's
  // module names DON'T match upstream, so most keys go unmatched —
  // that's expected and not an error in the transitional period.
  torch::NoGradGuard ng;
  std::unordered_map<std::string, at::Tensor*> dst;
  auto params = named_parameters(/*recurse=*/true);
  for (auto& kv : params) dst.emplace(kv.key(), &kv.value());
  auto buffers = named_buffers(/*recurse=*/true);
  for (auto& kv : buffers) dst.emplace(kv.key(), &kv.value());
  int matched = 0;
  for (const auto& [k, t] : entries) {
    auto it = dst.find(k);
    if (it == dst.end()) continue;
    if (it->second->sizes() != t.sizes()) continue;
    it->second->copy_(t.to(it->second->dtype()));
    ++matched;
  }
  return matched;
}

int RFDetrImpl::load_from_upstream_pt(const std::string& pt_path,
                                       bool strict) {
  auto rep = yolocpp::serialization::load_rfdetr_pt(pt_path, *this, strict);
  std::cout << rep.summary() << "\n";
  return rep.matched;
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
