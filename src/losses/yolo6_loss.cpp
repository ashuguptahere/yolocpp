#include "yolocpp/losses/yolo6_loss.hpp"

#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace yolocpp::losses {

namespace F = torch::nn::functional;

namespace {

// ─── Geometric primitives — duplicated from yolo8_loss.cpp anonymous
// namespace (Meituan-style "self-contained per-version loss" pattern;
// v26 loss does the same thing).

torch::Tensor make_anchors(const std::vector<double>& strides,
                           const std::vector<std::pair<int, int>>& sizes,
                           const torch::TensorOptions& opts,
                           torch::Tensor* stride_out = nullptr) {
  // Anchors in PIXEL units — matches v8's `make_anchors` exactly.
  // Critical for the TAL `in_gt` comparison against pixel-unit GT boxes.
  std::vector<torch::Tensor> a, s;
  for (size_t i = 0; i < strides.size(); ++i) {
    int h = sizes[i].first, w = sizes[i].second;
    double st = strides[i];
    auto sx = (torch::arange(w, opts) + 0.5) * st;
    auto sy = (torch::arange(h, opts) + 0.5) * st;
    auto gx = sx.view({1, w}).expand({h, w});
    auto gy = sy.view({h, 1}).expand({h, w});
    auto anc = torch::stack({gx, gy}, /*dim=*/-1).view({h * w, 2});
    a.push_back(anc);
    s.push_back(torch::full({h * w}, st, opts));
  }
  if (stride_out) *stride_out = torch::cat(s, 0);
  return torch::cat(a, 0);
}

torch::Tensor dfl_decode(const torch::Tensor& pred_dist, int reg_max) {
  // [B, A, 4*reg_max] → softmax over reg_max bins → expectation
  auto sizes = pred_dist.sizes();
  auto B = sizes[0], A = sizes[1];
  auto x = pred_dist.view({B, A, 4, reg_max}).softmax(-1);
  auto proj = torch::arange(reg_max, pred_dist.options());
  return (x * proj).sum(-1);   // [B, A, 4]
}

torch::Tensor dist2bbox(const torch::Tensor& dist, const torch::Tensor& anc) {
  auto lt = dist.slice(-1, 0, 2);
  auto rb = dist.slice(-1, 2, 4);
  auto x1y1 = anc - lt;
  auto x2y2 = anc + rb;
  return torch::cat({x1y1, x2y2}, -1);
}

torch::Tensor bbox2dist(const torch::Tensor& xyxy, const torch::Tensor& anc,
                        int reg_max) {
  auto lt = anc - xyxy.slice(-1, 0, 2);
  auto rb = xyxy.slice(-1, 2, 4) - anc;
  auto d  = torch::cat({lt, rb}, -1);
  return d.clamp_(0.0, double(reg_max - 1) - 0.01);
}

torch::Tensor dfl_loss(const torch::Tensor& pred_dist,
                       const torch::Tensor& tgt_dist, int reg_max) {
  auto P = pred_dist.size(0);
  auto pd = pred_dist.view({P, 4, reg_max});
  auto tl = tgt_dist.floor().to(torch::kLong).clamp_(0, reg_max - 1);
  auto tr = (tl + 1).clamp_(0, reg_max - 1);
  auto wl = tr.to(tgt_dist.dtype()) - tgt_dist;
  auto wr = 1.0 - wl;
  auto ce_l = F::cross_entropy(
                pd.reshape({-1, reg_max}),
                tl.reshape({-1}),
                F::CrossEntropyFuncOptions().reduction(torch::kNone))
                .view({P, 4});
  auto ce_r = F::cross_entropy(
                pd.reshape({-1, reg_max}),
                tr.reshape({-1}),
                F::CrossEntropyFuncOptions().reduction(torch::kNone))
                .view({P, 4});
  return (ce_l * wl + ce_r * wr).mean(-1);   // [P]
}

// ─── SIoU loss — Gevorgyan 2022 ─────────────────────────────────────────
//
// SIoU = IoU - 0.5 * (distance_cost + shape_cost)
//
// Where:
//   angle_cost     = 1 - 2*sin²(arcsin(min(|cw|,|ch|)/sigma) - π/4)
//   distance_cost  = Σᵢ (1 - exp(-(2 - angle_cost) * ρᵢ²))   over x, y
//   shape_cost     = Σᵢ (1 - exp(-Δsize_i / max_size_i))^4   over w, h
//
// `box1`, `box2`: xyxy, broadcastable.
torch::Tensor bbox_siou(const torch::Tensor& a, const torch::Tensor& b,
                        double eps = 1e-7) {
  auto ax1 = a.select(-1, 0), ay1 = a.select(-1, 1);
  auto ax2 = a.select(-1, 2), ay2 = a.select(-1, 3);
  auto bx1 = b.select(-1, 0), by1 = b.select(-1, 1);
  auto bx2 = b.select(-1, 2), by2 = b.select(-1, 3);

  // Plain IoU.
  auto inter = (torch::min(ax2, bx2) - torch::max(ax1, bx1)).clamp_min(0) *
               (torch::min(ay2, by2) - torch::max(ay1, by1)).clamp_min(0);
  auto aw = ax2 - ax1, ah = ay2 - ay1;
  auto bw = bx2 - bx1, bh = by2 - by1;
  auto union_ = aw * ah + bw * bh - inter + eps;
  auto iou    = inter / union_;

  // Enclosing-box size (cw, ch).
  auto cw = torch::max(ax2, bx2) - torch::min(ax1, bx1) + eps;
  auto ch = torch::max(ay2, by2) - torch::min(ay1, by1) + eps;

  // Center offset (target - pred), halved to match xywh-style center delta.
  auto s_cw = (bx1 + bx2 - ax1 - ax2) * 0.5;
  auto s_ch = (by1 + by2 - ay1 - ay2) * 0.5;
  auto sigma = (s_cw.pow(2) + s_ch.pow(2)).sqrt() + eps;

  // Angle cost. Branch on which axis is longer, then close-form via sin.
  auto sin_alpha_x = s_cw.abs() / sigma;
  auto sin_alpha_y = s_ch.abs() / sigma;
  double thr = std::sqrt(2.0) / 2.0;
  auto sin_alpha = torch::where(sin_alpha_x > thr, sin_alpha_y, sin_alpha_x);
  auto angle_cost = torch::cos(torch::asin(sin_alpha) * 2.0 - M_PI / 2.0);

  // Distance cost.
  auto rho_x = (s_cw / cw).pow(2);
  auto rho_y = (s_ch / ch).pow(2);
  auto gamma_t = 2.0 - angle_cost;
  auto distance_cost = 2.0 - torch::exp(-gamma_t * rho_x)
                            - torch::exp(-gamma_t * rho_y);

  // Shape cost.
  auto omega_w = (aw - bw).abs() / torch::max(aw, bw).clamp_min(eps);
  auto omega_h = (ah - bh).abs() / torch::max(ah, bh).clamp_min(eps);
  auto shape_cost = (1.0 - torch::exp(-omega_w)).pow(4) +
                    (1.0 - torch::exp(-omega_h)).pow(4);

  return iou - 0.5 * (distance_cost + shape_cost);
}

// ─── TAL (Task-Aligned Assigner) — same shape as v8's ─────────────────
//
// Inputs:
//   pd_scores : [B, A, nc] — pred class logits sigmoided.
//   pd_bboxes : [B, A, 4]  — pred xyxy in pixel coords (detached).
//   anc_pts   : [A, 2]     — anchor centers in pixels.
//   gt_labels : [B, M]
//   gt_bboxes : [B, M, 4]  — xyxy in pixels.
//   mask_gt   : [B, M, 1]  — 1.0 where this slot is a real GT.
//   topk, alpha, beta — TAL hyperparameters.
//
// Outputs (.target_labels [B, A], .target_bboxes [B, A, 4],
//          .target_scores [B, A, nc], .fg_mask [B, A]).
struct V6TalOutput {
  torch::Tensor target_labels;
  torch::Tensor target_bboxes;
  torch::Tensor target_scores;
  torch::Tensor fg_mask;
};

V6TalOutput tal_assign(const torch::Tensor& pd_scores,
                       const torch::Tensor& pd_bboxes,
                       const torch::Tensor& anc_pts,
                       const torch::Tensor& gt_labels,
                       const torch::Tensor& gt_bboxes,
                       const torch::Tensor& mask_gt, int nc,
                       int topk, double alpha, double beta) {
  // Pairwise box IoU between every (anchor, gt) — [B, M, A].
  auto B = pd_bboxes.size(0);
  auto A = pd_bboxes.size(1);
  auto M = gt_bboxes.size(1);

  auto ax1 = pd_bboxes.select(-1, 0).unsqueeze(1).expand({B, M, A});
  auto ay1 = pd_bboxes.select(-1, 1).unsqueeze(1).expand({B, M, A});
  auto ax2 = pd_bboxes.select(-1, 2).unsqueeze(1).expand({B, M, A});
  auto ay2 = pd_bboxes.select(-1, 3).unsqueeze(1).expand({B, M, A});
  auto bx1 = gt_bboxes.select(-1, 0).unsqueeze(2).expand({B, M, A});
  auto by1 = gt_bboxes.select(-1, 1).unsqueeze(2).expand({B, M, A});
  auto bx2 = gt_bboxes.select(-1, 2).unsqueeze(2).expand({B, M, A});
  auto by2 = gt_bboxes.select(-1, 3).unsqueeze(2).expand({B, M, A});

  auto inter = (torch::min(ax2, bx2) - torch::max(ax1, bx1)).clamp_min(0) *
               (torch::min(ay2, by2) - torch::max(ay1, by1)).clamp_min(0);
  auto area_a = (ax2 - ax1) * (ay2 - ay1);
  auto area_b = (bx2 - bx1) * (by2 - by1);
  auto iou = inter / (area_a + area_b - inter + 1e-9);
  iou = iou.clamp_min(0);

  // Per-(B, M, A) class-score lookup → score on the matched class.
  auto gl = gt_labels.to(torch::kLong);                       // [B, M]
  auto cls_idx = gl.unsqueeze(-1).expand({B, M, A});           // [B, M, A]
  auto pd_T = pd_scores.permute({0, 2, 1});                   // [B, nc, A]
  auto sc = pd_T.gather(1, cls_idx);                           // [B, M, A]

  // Alignment metric: score^alpha * iou^beta.
  auto align = sc.clamp_min(1e-9).pow(alpha) *
               iou.clamp_min(1e-9).pow(beta);

  // Anchor-in-gt mask (anchor center inside gt box).
  auto axc = anc_pts.select(-1, 0).view({1, 1, A});
  auto ayc = anc_pts.select(-1, 1).view({1, 1, A});
  auto in_gt = ((axc > bx1) & (axc < bx2) & (ayc > by1) & (ayc < by2))
                   .to(align.dtype());
  align = align * in_gt * mask_gt;                             // [B, M, A]

  // Top-k highest-aligned anchors per GT.
  auto vals = std::get<0>(align.topk(topk, /*dim=*/2));        // [B, M, topk]
  auto thr  = std::get<0>(vals.min(/*dim=*/2, /*keepdim=*/true));
  auto topk_mask = (align >= thr).to(align.dtype()) * (align > 0).to(align.dtype());

  // Resolve double-assignment: each anchor → its best GT.
  auto sum_per_a = topk_mask.sum(/*dim=*/1);                   // [B, A]
  auto multi = (sum_per_a > 1).unsqueeze(1).expand({B, M, A});
  auto best_gt = std::get<1>(iou.max(/*dim=*/1, /*keepdim=*/false));  // [B, A]
  auto bg_oh = F::one_hot(best_gt, M)
                   .permute({0, 2, 1}).to(align.dtype());            // [B, M, A]
  auto resolved = torch::where(multi, bg_oh, topk_mask);
  auto pos_mask = resolved * mask_gt.expand({B, M, A});             // [B, M, A]

  // Target lookup per anchor.
  auto fg_mask = (pos_mask.sum(1) > 0);                              // [B, A] bool
  auto tgt_gt = std::get<1>(pos_mask.max(1));                         // [B, A]
  auto t_lab = gt_labels.gather(1, tgt_gt);                          // [B, A]
  auto t_bb  = gt_bboxes.gather(
                  1, tgt_gt.unsqueeze(-1).expand({B, A, 4}));        // [B, A, 4]

  // Soft target scores: align normalised per-GT, * iou (quality).
  auto align_pos = align * pos_mask;
  auto align_max = std::get<0>(align_pos.max(2, true));               // [B, M, 1]
  auto iou_max   = std::get<0>((iou * pos_mask).max(2, true));        // [B, M, 1]
  auto norm_align = align_pos / (align_max + 1e-9) * iou_max;         // [B, M, A]
  auto t_score   = norm_align.sum(1);                                  // [B, A]

  // Build [B, A, nc] target scores using one-hot * t_score.
  auto t_lab_l = t_lab.to(torch::kLong);
  auto oh = F::one_hot(t_lab_l, nc).to(align.dtype());                // [B, A, nc]
  auto fg_f = fg_mask.to(align.dtype()).unsqueeze(-1);                 // [B, A, 1]
  auto tgt_scores = oh * fg_f * t_score.unsqueeze(-1);                 // [B, A, nc]

  V6TalOutput o;
  o.target_labels = t_lab;
  o.target_bboxes = t_bb;
  o.target_scores = tgt_scores;
  o.fg_mask       = fg_mask;
  return o;
}

}  // anonymous namespace

V6DetectionLoss::V6DetectionLoss(V6LossConfig cfg) : cfg_(cfg) {}

LossOutput V6DetectionLoss::operator()(
    const std::vector<torch::Tensor>& feats, const torch::Tensor& targets,
    const std::vector<double>& strides, int /*imgsz*/) const {
  TORCH_CHECK(!feats.empty(), "no feature levels");
  auto opts = feats[0].options().dtype(torch::kFloat32);

  // v6 convention: bins = reg_max + 1 (DFL distribution width). The
  // EffiDeHead's reg branch outputs 4 * bins channels (e.g. reg_max=16
  // → 68 ch). v8's loss uses bins = reg_max directly; we adjust here.
  int reg_max = cfg_.reg_max;
  int bins    = reg_max + 1;
  int reg_ch  = 4 * bins;
  int nc      = cfg_.nc;
  int B       = (int)feats[0].size(0);

  // ── 1. Concatenate flat predictions ────────────────────────────────
  std::vector<torch::Tensor> flat;
  std::vector<std::pair<int, int>> sizes;
  for (auto& f : feats) {
    sizes.emplace_back((int)f.size(2), (int)f.size(3));
    flat.push_back(f.reshape({B, f.size(1), -1}));
  }
  auto pred = torch::cat(flat, /*dim=*/2).permute({0, 2, 1});   // [B, A, no]

  auto pred_dist = pred.slice(-1, 0,      reg_ch);                // [B, A, 4*bins]
  auto pred_cls  = pred.slice(-1, reg_ch, pred.size(-1));         // [B, A, nc]

  auto anc_opts = opts.device(feats[0].device());
  torch::Tensor stride_t;
  auto anc_pts = make_anchors(strides, sizes, anc_opts, &stride_t);

  auto dist = dfl_decode(pred_dist, bins);                       // [B, A, 4]
  auto pred_xyxy = dist2bbox(dist * stride_t.unsqueeze(-1),
                             anc_pts.unsqueeze(0));              // [B, A, 4]

  // ── 2. Build padded GT tensors ─────────────────────────────────────
  std::vector<int> per_batch(B, 0);
  if (targets.size(0) > 0) {
    // Host copy once + accessor read; avoids a per-GT-box device->host sync
    // (CLAUDE.md: no per-batch host syncs).
    auto bi   = targets.select(1, 0).to(torch::kCPU).to(torch::kLong);
    auto bacc = bi.accessor<int64_t, 1>();
    for (int i = 0; i < (int)bi.size(0); ++i) {
      int b = (int)bacc[i];
      if (b >= 0 && b < B) per_batch[b]++;
    }
  }
  int Mmax = 1;
  for (int n : per_batch) Mmax = std::max(Mmax, n);

  auto cpu_opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto gt_labels_cpu = torch::zeros({B, Mmax},    cpu_opts);
  auto gt_bboxes_cpu = torch::zeros({B, Mmax, 4}, cpu_opts);
  auto mask_gt_cpu   = torch::zeros({B, Mmax, 1}, cpu_opts);

  if (targets.size(0) > 0) {
    auto t = targets.detach().to(torch::kCPU).to(torch::kFloat32);
    auto a = t.accessor<float, 2>();
    auto gl = gt_labels_cpu.accessor<float, 2>();
    auto gb = gt_bboxes_cpu.accessor<float, 3>();
    auto mg = mask_gt_cpu.accessor<float, 3>();
    std::vector<int> cur(B, 0);
    for (int i = 0; i < (int)t.size(0); ++i) {
      int b = (int)a[i][0];
      if (b < 0 || b >= B) continue;
      int j = cur[b]++;
      gl[b][j] = a[i][1];
      float cx = a[i][2], cy = a[i][3], w = a[i][4], h = a[i][5];
      gb[b][j][0] = cx - w / 2; gb[b][j][1] = cy - h / 2;
      gb[b][j][2] = cx + w / 2; gb[b][j][3] = cy + h / 2;
      mg[b][j][0] = 1.0f;
    }
  }
  auto gt_labels = gt_labels_cpu.to(feats[0].device());
  auto gt_bboxes = gt_bboxes_cpu.to(feats[0].device());
  auto mask_gt   = mask_gt_cpu.to(feats[0].device());

  // ── 3. TAL assignment ──────────────────────────────────────────────
  auto pd_scores_sig = pred_cls.detach().sigmoid();
  auto tal = tal_assign(pd_scores_sig, pred_xyxy.detach(), anc_pts,
                        gt_labels, gt_bboxes, mask_gt,
                        nc, cfg_.topk, cfg_.alpha, cfg_.beta);

  // ── 4. VFL cls loss ────────────────────────────────────────────────
  // VFL: weight = alpha * sigmoid(pred)^gamma * (1 - label) + gt_score * label
  // where label = (target_scores > 0), gt_score = target_scores quality weight.
  auto pred_sig = pred_cls.sigmoid();
  auto label    = (tal.target_scores > 0).to(tal.target_scores.dtype());
  auto vfl_w    = cfg_.vfl_alpha * pred_sig.detach().pow(cfg_.vfl_gamma) *
                  (1.0 - label) + tal.target_scores * label;
  auto bce_per  = F::binary_cross_entropy_with_logits(
                      pred_cls, tal.target_scores.to(pred_cls.dtype()),
                      F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kNone));
  auto target_scores_sum = tal.target_scores.sum().clamp_min(1.0);
  auto cls_loss = (bce_per * vfl_w).sum() / target_scores_sum;

  // ── 5. SIoU box + DFL loss on positives ────────────────────────────
  torch::Tensor box_loss = torch::zeros({}, opts);
  torch::Tensor dfl_l    = torch::zeros({}, opts);
  if (tal.fg_mask.any().item<bool>()) {
    auto fg = tal.fg_mask;
    auto fg_flat = fg.reshape(-1);
    auto pred_xyxy_flat = pred_xyxy.reshape({-1, 4});
    auto pred_dist_flat = pred_dist.reshape({-1, reg_ch});
    auto t_bboxes_flat  = tal.target_bboxes.reshape({-1, 4});
    auto t_scores_flat  = tal.target_scores.sum(-1).reshape(-1);

    auto pos_idx = torch::nonzero(fg_flat).reshape(-1);
    auto pp_xyxy = pred_xyxy_flat.index_select(0, pos_idx);
    auto pp_dist = pred_dist_flat.index_select(0, pos_idx);
    auto tt_bb   = t_bboxes_flat.index_select(0, pos_idx);
    auto w       = t_scores_flat.index_select(0, pos_idx);

    auto siou = bbox_siou(pp_xyxy, tt_bb);                 // [P]
    box_loss = ((1.0 - siou) * w).sum() / target_scores_sum;

    // DFL on the same positives.
    auto stride_flat = stride_t.repeat({B});
    auto anc_flat    = anc_pts.repeat({B, 1});
    auto pp_str      = stride_flat.index_select(0, pos_idx);
    auto pp_anc_pix  = anc_flat.index_select(0, pos_idx);
    auto pp_anc_cell = pp_anc_pix / pp_str.unsqueeze(-1);

    auto tt_xyxy_cell = tt_bb / pp_str.unsqueeze(-1);
    auto tt_dist_cell = bbox2dist(tt_xyxy_cell, pp_anc_cell, bins);
    auto dfl_per      = dfl_loss(pp_dist, tt_dist_cell, bins);
    dfl_l = (dfl_per * w).sum() / target_scores_sum;
  }

  LossOutput out;
  out.box   = box_loss * cfg_.box_gain;
  out.cls   = cls_loss * cfg_.cls_gain;
  out.dfl   = dfl_l    * cfg_.dfl_gain;
  out.total = out.box + out.cls + out.dfl;
  return out;
}

}  // namespace yolocpp::losses
