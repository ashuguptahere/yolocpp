// RF-DETR set loss (#65F). Hungarian matching + focal cls + L1 + GIoU.

#include "yolocpp/losses/rfdetr_loss.hpp"

#include <cmath>

#include "yolocpp/losses/hungarian.hpp"

namespace yolocpp::losses {

namespace {

// (cx, cy, w, h) → (x1, y1, x2, y2). Operates on the last dim.
torch::Tensor cxcywh_to_xyxy(torch::Tensor b) {
  auto cx = b.select(-1, 0), cy = b.select(-1, 1);
  auto w  = b.select(-1, 2), h  = b.select(-1, 3);
  return torch::stack({cx - 0.5 * w, cy - 0.5 * h,
                       cx + 0.5 * w, cy + 0.5 * h}, -1);
}

// GIoU between two batched xyxy box sets `[N, 4]` × `[M, 4]` →
// `[N, M]`. Standard formula.
torch::Tensor giou_matrix(torch::Tensor a, torch::Tensor b) {
  auto N = a.size(0), M = b.size(0);
  if (N == 0 || M == 0) {
    return torch::zeros({N, M}, a.options());
  }
  auto a2 = a.unsqueeze(1).expand({N, M, 4});
  auto b2 = b.unsqueeze(0).expand({N, M, 4});

  auto lt = torch::maximum(a2.slice(-1, 0, 2), b2.slice(-1, 0, 2));
  auto rb = torch::minimum(a2.slice(-1, 2, 4), b2.slice(-1, 2, 4));
  auto wh = (rb - lt).clamp_min(0.0);
  auto inter = wh.select(-1, 0) * wh.select(-1, 1);

  auto area_a = (a2.select(-1, 2) - a2.select(-1, 0)) *
                (a2.select(-1, 3) - a2.select(-1, 1));
  auto area_b = (b2.select(-1, 2) - b2.select(-1, 0)) *
                (b2.select(-1, 3) - b2.select(-1, 1));
  auto uni = area_a + area_b - inter;
  auto iou = inter / uni.clamp_min(1e-7);

  auto enc_lt = torch::minimum(a2.slice(-1, 0, 2), b2.slice(-1, 0, 2));
  auto enc_rb = torch::maximum(a2.slice(-1, 2, 4), b2.slice(-1, 2, 4));
  auto enc_wh = (enc_rb - enc_lt).clamp_min(0.0);
  auto enc_a  = enc_wh.select(-1, 0) * enc_wh.select(-1, 1);

  return iou - (enc_a - uni) / enc_a.clamp_min(1e-7);
}

// Per-image Hungarian match. Returns paired (pred_idx, gt_idx) lists.
std::pair<std::vector<int64_t>, std::vector<int64_t>>
match_one(torch::Tensor cls_prob, torch::Tensor bbox,
          const std::vector<RFDetrTarget>& tgt, const RFDetrLossCfg& cfg) {
  // cls_prob: [Q, nc] (already sigmoided)
  // bbox:     [Q, 4]  (already sigmoided cxcywh)
  int64_t Q = cls_prob.size(0);
  int64_t G = static_cast<int64_t>(tgt.size());
  if (G == 0) return {{}, {}};

  // Cost cls: -p_i[c_j]
  std::vector<int64_t> cls_ids(G);
  std::vector<float>   tgt_xywh(G * 4);
  for (int64_t j = 0; j < G; ++j) {
    cls_ids[j] = tgt[j].class_id;
    tgt_xywh[j * 4 + 0] = tgt[j].cx;
    tgt_xywh[j * 4 + 1] = tgt[j].cy;
    tgt_xywh[j * 4 + 2] = tgt[j].w;
    tgt_xywh[j * 4 + 3] = tgt[j].h;
  }
  auto ids_t = torch::tensor(cls_ids, torch::dtype(torch::kInt64)
                                            .device(cls_prob.device()));
  auto cost_cls = -cls_prob.index_select(-1, ids_t);              // [Q, G]

  auto tgt_t = torch::from_blob(tgt_xywh.data(), {G, 4})
                  .to(bbox.device(), bbox.dtype())
                  .clone();
  auto cost_l1 = torch::cdist(bbox, tgt_t, /*p=*/1);              // [Q, G]
  auto cost_giou = -giou_matrix(cxcywh_to_xyxy(bbox),
                                  cxcywh_to_xyxy(tgt_t));         // [Q, G]

  auto cost = cfg.lambda_cls * cost_cls +
              cfg.lambda_l1  * cost_l1 +
              cfg.lambda_giou* cost_giou;
  auto cost_cpu = cost.detach().to(torch::kCPU).contiguous();

  // Hungarian with rows = preds (Q), cols = gts (G). Returns
  // assignment[g] = pred index.
  auto assign = hungarian_assign(cost_cpu.data_ptr<float>(), Q, G);
  std::vector<int64_t> pred_idx, gt_idx;
  pred_idx.reserve(G); gt_idx.reserve(G);
  for (int64_t g = 0; g < G; ++g) {
    if (assign[g] >= 0) {
      pred_idx.push_back(assign[g]);
      gt_idx.push_back(g);
    }
  }
  return {pred_idx, gt_idx};
}

// Sigmoid focal loss with α/γ. preds_logits: [N, C], targets: [N, C]
// in {0, 1}.
torch::Tensor focal_loss(torch::Tensor logits, torch::Tensor target,
                          float alpha, float gamma) {
  auto p   = torch::sigmoid(logits);
  auto ce  = torch::binary_cross_entropy_with_logits(
      logits, target, /*weight=*/{}, /*pos_weight=*/{},
      torch::Reduction::None);
  auto p_t = p * target + (1 - p) * (1 - target);
  auto loss = ce * torch::pow(1 - p_t, gamma);
  if (alpha >= 0) {
    auto a_t = alpha * target + (1 - alpha) * (1 - target);
    loss = a_t * loss;
  }
  return loss.mean(/*dim=*/1).sum();   // sum over queries, mean over classes
}

}  // namespace

RFDetrLossOutput rfdetr_set_loss(
    const std::vector<torch::Tensor>& cls_logits_per_layer,
    const std::vector<torch::Tensor>& bbox_unact_per_layer,
    const std::vector<std::vector<RFDetrTarget>>& targets,
    const RFDetrLossCfg& cfg) {
  TORCH_CHECK(!cls_logits_per_layer.empty() &&
                  cls_logits_per_layer.size() == bbox_unact_per_layer.size(),
              "rfdetr_set_loss: per-layer arity mismatch");
  auto B = cls_logits_per_layer[0].size(0);
  TORCH_CHECK(static_cast<int64_t>(targets.size()) == B,
              "rfdetr_set_loss: target batch size mismatch");

  auto opts = cls_logits_per_layer[0].options().dtype(torch::kFloat);
  torch::Tensor zero = torch::zeros({}, opts);
  auto loss_cls  = zero.clone();
  auto loss_l1   = zero.clone();
  auto loss_giou = zero.clone();

  // Run matcher only on the FINAL layer's outputs; reuse the
  // assignment for auxiliary layers (matches the upstream
  // implementation's behaviour and keeps targets stable).
  auto& cls_final  = cls_logits_per_layer.back();
  auto& bbox_final = bbox_unact_per_layer.back();
  auto cls_prob_final = torch::sigmoid(cls_final);
  auto bbox_sig_final = torch::sigmoid(bbox_final);

  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> matches;
  matches.reserve(B);
  int64_t total_gt = 0;
  for (int64_t b = 0; b < B; ++b) {
    matches.push_back(match_one(cls_prob_final[b], bbox_sig_final[b],
                                  targets[b], cfg));
    total_gt += static_cast<int64_t>(targets[b].size());
  }
  total_gt = std::max<int64_t>(total_gt, 1);

  for (size_t l = 0; l < cls_logits_per_layer.size(); ++l) {
    const auto& cls_l   = cls_logits_per_layer[l];
    const auto& bbox_l  = bbox_unact_per_layer[l];
    int64_t Q  = cls_l.size(1), nc = cls_l.size(2);
    auto cls_target = torch::zeros({B, Q, nc}, cls_l.options());
    std::vector<torch::Tensor> matched_preds, matched_targets;

    for (int64_t b = 0; b < B; ++b) {
      const auto& [pred_idx, gt_idx] = matches[b];
      for (size_t k = 0; k < pred_idx.size(); ++k) {
        int64_t p = pred_idx[k], g = gt_idx[k];
        cls_target[b][p][targets[b][g].class_id] = 1.0;
        matched_preds.push_back(torch::sigmoid(bbox_l[b][p]).unsqueeze(0));
        std::vector<float> g_xywh = {targets[b][g].cx, targets[b][g].cy,
                                      targets[b][g].w,  targets[b][g].h};
        auto gt_t = torch::from_blob(g_xywh.data(), {1, 4})
                        .to(bbox_l.device(), bbox_l.dtype()).clone();
        matched_targets.push_back(gt_t);
      }
    }

    loss_cls = loss_cls + focal_loss(cls_l.view({B * Q, nc}),
                                      cls_target.view({B * Q, nc}),
                                      cfg.focal_alpha, cfg.focal_gamma) /
                              static_cast<double>(total_gt);

    if (!matched_preds.empty()) {
      auto pred = torch::cat(matched_preds, 0);
      auto tgt  = torch::cat(matched_targets, 0);
      loss_l1   = loss_l1 + (pred - tgt).abs().sum() /
                                static_cast<double>(total_gt);
      auto giou = giou_matrix(cxcywh_to_xyxy(pred), cxcywh_to_xyxy(tgt))
                      .diagonal();
      loss_giou = loss_giou + (1.0 - giou).sum() /
                                static_cast<double>(total_gt);
    }
  }

  auto total = cfg.lambda_cls  * loss_cls +
               cfg.lambda_l1   * loss_l1 +
               cfg.lambda_giou * loss_giou;
  return {total, loss_cls, loss_l1, loss_giou};
}

}  // namespace yolocpp::losses
