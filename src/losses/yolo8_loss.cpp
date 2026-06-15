#include "yolocpp/losses/yolo8_loss.hpp"

#include <torch/nn/functional.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace yolocpp::losses {

namespace F = torch::nn::functional;

namespace {

// xywh (cx,cy,w,h) → xyxy (x1,y1,x2,y2). Last-dim conversion.
torch::Tensor xywh2xyxy(const torch::Tensor& x) {
  auto cx = x.select(-1, 0), cy = x.select(-1, 1);
  auto w  = x.select(-1, 2), h  = x.select(-1, 3);
  auto x1 = cx - w / 2;
  auto y1 = cy - h / 2;
  auto x2 = cx + w / 2;
  auto y2 = cy + h / 2;
  return torch::stack({x1, y1, x2, y2}, /*dim=*/-1);
}

// Pairwise CIoU between two sets of xyxy boxes.
//   a: [..., 4], b: [..., 4] (broadcastable)
torch::Tensor bbox_ciou(const torch::Tensor& a, const torch::Tensor& b,
                        double eps = 1e-7) {
  auto ax1 = a.select(-1, 0), ay1 = a.select(-1, 1);
  auto ax2 = a.select(-1, 2), ay2 = a.select(-1, 3);
  auto bx1 = b.select(-1, 0), by1 = b.select(-1, 1);
  auto bx2 = b.select(-1, 2), by2 = b.select(-1, 3);

  auto inter_x1 = torch::max(ax1, bx1);
  auto inter_y1 = torch::max(ay1, by1);
  auto inter_x2 = torch::min(ax2, bx2);
  auto inter_y2 = torch::min(ay2, by2);
  auto inter = (inter_x2 - inter_x1).clamp_min(0) *
               (inter_y2 - inter_y1).clamp_min(0);
  auto aw = ax2 - ax1, ah = ay2 - ay1;
  auto bw = bx2 - bx1, bh = by2 - by1;
  auto union_ = aw * ah + bw * bh - inter + eps;
  auto iou = inter / union_;

  // enclosing box
  auto cw = torch::max(ax2, bx2) - torch::min(ax1, bx1);
  auto ch = torch::max(ay2, by2) - torch::min(ay1, by1);
  auto c2 = cw.pow(2) + ch.pow(2) + eps;

  // center distance
  auto rho2 = ((bx1 + bx2 - ax1 - ax2).pow(2) +
               (by1 + by2 - ay1 - ay2).pow(2)) / 4.0;

  auto v = (4.0 / (M_PI * M_PI)) *
           (torch::atan(bw / (bh + eps)) - torch::atan(aw / (ah + eps))).pow(2);
  torch::Tensor alpha;
  {
    torch::NoGradGuard g;
    alpha = v / (1 - iou + v + eps);
  }
  return iou - rho2 / c2 - alpha * v;
}

// Build anchor centers (in pixel coords) for the multi-scale feature maps.
//   strides: per-level stride
//   sizes:   per-level (h_i, w_i)
// Returns:
//   anchors: [A, 2]  — (x, y) cell centers in pixels
//   stride : [A]      — broadcasted stride
torch::Tensor make_anchors(const std::vector<double>& strides,
                           const std::vector<std::pair<int, int>>& sizes,
                           const torch::TensorOptions& opts,
                           torch::Tensor* stride_out = nullptr) {
  std::vector<torch::Tensor> a, s;
  for (size_t i = 0; i < strides.size(); ++i) {
    int   h = sizes[i].first;
    int   w = sizes[i].second;
    double st = strides[i];
    auto sx = (torch::arange(w, opts) + 0.5) * st;
    auto sy = (torch::arange(h, opts) + 0.5) * st;
    auto gy = sy.reshape({h, 1}).expand({h, w});
    auto gx = sx.reshape({1, w}).expand({h, w});
    a.push_back(torch::stack({gx, gy}, /*dim=*/-1).reshape({h * w, 2}));
    s.push_back(torch::full({h * w}, st, opts));
  }
  auto anc = torch::cat(a, 0);
  auto str = torch::cat(s, 0);
  if (stride_out) *stride_out = str;
  return anc;
}

// DFL decode: (l, t, r, b) distribution → expected distance per side.
//   pred_dist: [..., 4 * reg_max]
torch::Tensor dfl_decode(const torch::Tensor& pred_dist, int reg_max) {
  auto sizes = pred_dist.sizes().vec();
  // reshape last dim to (4, reg_max), softmax, dot with arange.
  std::vector<int64_t> new_sizes(sizes.begin(), sizes.end() - 1);
  new_sizes.push_back(4);
  new_sizes.push_back(reg_max);
  auto p = pred_dist.reshape(new_sizes).softmax(-1);
  auto proj = torch::arange(reg_max, pred_dist.options()).reshape({reg_max});
  return (p * proj).sum(-1);  // [..., 4]
}

// Convert distances (l, t, r, b) at anchor (cx, cy) to xyxy.
torch::Tensor dist2bbox(const torch::Tensor& dist, const torch::Tensor& anc) {
  auto lt = dist.slice(-1, 0, 2);  // l, t
  auto rb = dist.slice(-1, 2, 4);  // r, b
  auto x1y1 = anc - lt;
  auto x2y2 = anc + rb;
  return torch::cat({x1y1, x2y2}, -1);
}

// Encode xyxy boxes into (l, t, r, b) target distances at given anchors.
// Distances are clamped to [0, reg_max - 1 - eps].
torch::Tensor bbox2dist(const torch::Tensor& xyxy, const torch::Tensor& anc,
                        int reg_max) {
  auto x1y1 = xyxy.slice(-1, 0, 2);
  auto x2y2 = xyxy.slice(-1, 2, 4);
  auto lt = anc - x1y1;
  auto rb = x2y2 - anc;
  return torch::cat({lt, rb}, -1).clamp(0, reg_max - 1 - 1e-3);
}

// DFL loss between pred_dist [..., 4*reg_max] and target distances [..., 4].
// Implemented as the canonical "two-bin" cross entropy.
torch::Tensor dfl_loss(const torch::Tensor& pred_dist,
                       const torch::Tensor& target_dist, int reg_max) {
  // pred_dist: [N_pos, 4 * reg_max], reshape to [N_pos * 4, reg_max]
  auto pred = pred_dist.reshape({-1, reg_max});
  auto tl = target_dist.floor().to(torch::kLong).clamp(0, reg_max - 1);
  auto tr = (tl + 1).clamp(0, reg_max - 1);
  auto wl = tr.to(torch::kFloat32) - target_dist;
  auto wr = 1.0 - wl;

  auto loss_l = F::cross_entropy(pred, tl.reshape({-1}),
                                 F::CrossEntropyFuncOptions().reduction(torch::kNone))
                    .reshape(target_dist.sizes()) *
                wl;
  auto loss_r = F::cross_entropy(pred, tr.reshape({-1}),
                                 F::CrossEntropyFuncOptions().reduction(torch::kNone))
                    .reshape(target_dist.sizes()) *
                wr;
  return (loss_l + loss_r).mean(-1);  // [N_pos]
}

// Task-Aligned Assigner.
//
// Inputs (B = batch, A = #anchors total, nc = #classes):
//   pd_scores  [B, A, nc]   sigmoid class probs
//   pd_bboxes  [B, A, 4]    decoded predicted xyxy (pixel coords)
//   anc_pts    [A, 2]       anchor (x, y) centers
//   gt_labels  [B, M]       ground-truth class indices, padded with 0
//   gt_bboxes  [B, M, 4]    ground-truth xyxy in pixel coords
//   mask_gt    [B, M, 1]    1 where the gt is real (else 0 padding)
//
// Outputs:
//   target_labels  [B, A]            assigned class index
//   target_bboxes  [B, A, 4]         assigned gt xyxy
//   target_scores  [B, A, nc]        soft alignment scores (one-hot scaled)
//   fg_mask        [B, A]            true where anchor is positive
struct TalOutput {
  torch::Tensor target_labels;
  torch::Tensor target_bboxes;
  torch::Tensor target_scores;
  torch::Tensor fg_mask;
};

TalOutput tal_assign(const torch::Tensor& pd_scores,
                     const torch::Tensor& pd_bboxes,
                     const torch::Tensor& anc_pts,
                     const torch::Tensor& gt_labels,
                     const torch::Tensor& gt_bboxes,
                     const torch::Tensor& mask_gt,
                     int nc, int topk, float alpha, float beta) {
  using torch::indexing::Slice;
  using torch::indexing::None;

  auto B = pd_scores.size(0);
  auto A = pd_scores.size(1);
  auto M = gt_bboxes.size(1);

  // 1) Mask: anchor inside any gt box
  // anc_pts: [A, 2], gt_bboxes: [B, M, 4] → broadcast to [B, M, A]
  auto lt = gt_bboxes.slice(-1, 0, 2);  // [B, M, 2]
  auto rb = gt_bboxes.slice(-1, 2, 4);
  auto a_xy = anc_pts.unsqueeze(0).unsqueeze(0);  // [1, 1, A, 2]
  auto bbox_deltas = torch::cat({a_xy - lt.unsqueeze(2),
                                 rb.unsqueeze(2) - a_xy}, -1);  // [B, M, A, 4]
  auto in_gts = bbox_deltas.amin(-1).gt(1e-9);  // [B, M, A]

  // 2) Alignment metric: score^alpha * iou^beta where score is the
  //    predicted prob for the gt's class.
  auto gt_idx = gt_labels.to(torch::kLong).clamp(0, nc - 1);  // [B, M]
  // Gather pd_scores at gt_idx → [B, M, A]
  auto pd_s_perm = pd_scores.permute({0, 2, 1});  // [B, nc, A]
  auto idx_exp = gt_idx.unsqueeze(-1).expand({B, M, A});
  auto bbox_scores = pd_s_perm.gather(/*dim=*/1, idx_exp);  // [B, M, A]

  auto gt_b = gt_bboxes.unsqueeze(2);   // [B, M, 1, 4]
  auto pd_b = pd_bboxes.unsqueeze(1);   // [B, 1, A, 4]
  // Plain IoU is sufficient for assignment; CIoU here would be wasteful.
  auto inter_x1 = torch::max(gt_b.select(-1, 0), pd_b.select(-1, 0));
  auto inter_y1 = torch::max(gt_b.select(-1, 1), pd_b.select(-1, 1));
  auto inter_x2 = torch::min(gt_b.select(-1, 2), pd_b.select(-1, 2));
  auto inter_y2 = torch::min(gt_b.select(-1, 3), pd_b.select(-1, 3));
  auto inter = (inter_x2 - inter_x1).clamp_min(0) *
               (inter_y2 - inter_y1).clamp_min(0);
  auto aw = gt_b.select(-1, 2) - gt_b.select(-1, 0);
  auto ah = gt_b.select(-1, 3) - gt_b.select(-1, 1);
  auto bw = pd_b.select(-1, 2) - pd_b.select(-1, 0);
  auto bh = pd_b.select(-1, 3) - pd_b.select(-1, 1);
  auto iou = inter / (aw * ah + bw * bh - inter + 1e-9);  // [B, M, A]

  auto align_metric = bbox_scores.pow(alpha) * iou.pow(beta);  // [B, M, A]
  align_metric = align_metric * in_gts.to(align_metric.dtype()) *
                 mask_gt.to(align_metric.dtype());

  // 3) Top-k anchors per gt. The threshold for "in top-K" is the
  // K-th LARGEST metric, which is the SMALLEST of the top-K returned
  // by topk(). Using amax(-1) instead of amin(-1) here was a long-
  // standing bug: it picked `align_metric.max()` per GT, which made
  // the mask select only ONE positive per GT (the absolute max) —
  // effectively topk=1 regardless of the configured value. That's
  // why detect heads that share this loss (v3/v5/v8/v9/v11/v12/v13)
  // trained ~10× weaker positive signal than upstream and never
  // caught up in short fine-tune budgets, while v26's separate
  // scatter-based TAL did not have this bug. clamp_min keeps the
  // threshold above zero so anchors with metric=0 (no IoU overlap)
  // are excluded even when the K-th-largest happens to be 0.
  auto topk_metrics = std::get<0>(align_metric.topk(topk, /*dim=*/-1));
  auto topk_thresh  = topk_metrics.amin(-1, /*keepdim=*/true).clamp_min(1e-9);
  auto mask_topk    = align_metric >= topk_thresh.expand_as(align_metric) - 1e-9;
  auto mask_pos     = mask_topk * in_gts * mask_gt.gt(0);  // [B, M, A] bool

  // 4) Resolve overlaps: each anchor → at most one gt (the one with higher iou)
  auto fg_count = mask_pos.to(torch::kFloat32).sum(1);  // [B, A]
  auto multi    = fg_count > 1;
  if (multi.any().item<bool>()) {
    auto best = std::get<1>(iou.argmax(1, /*keepdim=*/true).contiguous().topk(1, /*dim=*/1));
    // Simpler: pick gt with max iou per anchor where multi-match.
    auto max_iou_idx = std::get<1>(iou.max(/*dim=*/1));  // [B, A]
    auto m = mask_pos.clone();
    auto onehot = torch::zeros_like(m);
    auto ones = torch::ones_like(max_iou_idx, torch::kFloat32).unsqueeze(1);
    onehot = onehot.scatter(1, max_iou_idx.unsqueeze(1).to(torch::kLong),
                            ones.to(onehot.dtype()));
    auto multi_3 = multi.unsqueeze(1).expand_as(m);
    mask_pos = torch::where(multi_3, onehot.to(torch::kBool), m);
  }

  // 5) Assigned target indices: which gt does each pos anchor belong to.
  auto target_gt_idx = std::get<1>(mask_pos.to(torch::kFloat32).max(/*dim=*/1));  // [B, A]
  auto fg_mask = mask_pos.any(1);  // [B, A]

  // 6) Assemble target boxes / labels / scores.
  auto batch_idx = torch::arange(B, gt_bboxes.options().dtype(torch::kLong))
                       .reshape({B, 1}).expand({B, A});
  auto gather_idx = batch_idx * M + target_gt_idx;  // flat into [B*M, ...]

  auto t_bboxes = gt_bboxes.reshape({B * M, 4}).index_select(0, gather_idx.reshape(-1))
                       .reshape({B, A, 4});
  auto t_labels = gt_labels.reshape({B * M}).index_select(0, gather_idx.reshape(-1))
                       .reshape({B, A}).to(torch::kLong);

  // soft scores: align_metric / max_align * iou_at_assigned, scaled to [0, 1].
  // align_metric MUST be re-masked by the FINAL positive mask (mask_pos) here —
  // it was only `in_gt`-masked (line ~217), so for a positive anchor the
  // amax-over-GTs below could pick a *non-assigned* GT's align value (any GT
  // whose box merely contains the anchor), inflating/corrupting its cls soft
  // target. Mirrors v26 STAL (yolo26_loss.cpp: align_pos = align * mask_pos).
  auto am_pos        = align_metric * mask_pos.to(align_metric.dtype());
  auto pos_align_max = am_pos.amax(/*dim=*/-1, /*keepdim=*/true);
  auto pos_iou_max   = (iou * mask_pos.to(iou.dtype()))
                           .amax(/*dim=*/-1, /*keepdim=*/true);
  auto norm_align    = (am_pos * pos_iou_max / (pos_align_max + 1e-9))
                           .amax(-2);  // [B, A]
  auto onehot_labels = torch::nn::functional::one_hot(t_labels, nc)
                           .to(pd_scores.dtype());
  auto t_scores = onehot_labels * norm_align.unsqueeze(-1);
  t_scores = t_scores * fg_mask.unsqueeze(-1).to(t_scores.dtype());

  TalOutput o;
  o.target_labels = t_labels;
  o.target_bboxes = t_bboxes;
  o.target_scores = t_scores;
  o.fg_mask       = fg_mask;
  return o;
}

}  // anonymous namespace

V8DetectionLoss::V8DetectionLoss(LossConfig cfg) : cfg_(cfg) {}

LossOutput V8DetectionLoss::operator()(
    const std::vector<torch::Tensor>& feats, const torch::Tensor& targets,
    const std::vector<double>& strides, int imgsz) const {
  TORCH_CHECK(!feats.empty(), "no feature levels");
  auto opts = feats[0].options().dtype(torch::kFloat32);

  int reg_max = cfg_.reg_max;
  int nc      = cfg_.nc;
  int B       = (int)feats[0].size(0);

  // ── 1. Concatenate flat predictions ─────────────────────────────────────
  // Compute the loss in fp32 even under bf16 autocast (the trainer wraps
  // forward + loss in an autocast scope, so `feats` arrive as bf16). bf16's
  // 8-bit mantissa loses precision through the DFL decode / IoU / BCE
  // reductions; casting at entry matches yolo1_loss and upstream's fp32 loss.
  // The cast is autograd-safe (grads flow back to the bf16 params) and a no-op
  // when feats are already fp32 (CPU / deterministic path).
  std::vector<torch::Tensor> flat;
  std::vector<std::pair<int, int>> sizes;
  for (auto& f0 : feats) {
    auto f = f0.to(torch::kFloat);
    sizes.emplace_back((int)f.size(2), (int)f.size(3));
    flat.push_back(f.reshape({B, f.size(1), -1}));     // [B, no, h*w]
  }
  auto pred = torch::cat(flat, /*dim=*/2)            // [B, no, A]
                  .permute({0, 2, 1});               // [B, A, no]

  auto pred_dist = pred.slice(-1, 0,             4 * reg_max);   // [B, A, 4*reg_max]
  auto pred_cls  = pred.slice(-1, 4 * reg_max,   pred.size(-1)); // [B, A, nc]

  // Anchors must live on the same device as the predictions for downstream ops.
  auto anc_opts = opts.device(feats[0].device());
  torch::Tensor stride_t;
  auto anc_pts = make_anchors(strides, sizes, anc_opts, &stride_t);  // [A, 2]
  auto A = anc_pts.size(0);

  // Decode predicted boxes from DFL
  auto dist = dfl_decode(pred_dist, reg_max);            // [B, A, 4]  in cell units
  auto pred_xyxy = dist2bbox(dist * stride_t.unsqueeze(-1),
                             anc_pts.unsqueeze(0));      // [B, A, 4] pixel coords

  // ── 2. Build per-batch padded GT tensors ───────────────────────────────
  // targets: [M, 6] (b_idx, cls, cx, cy, w, h)
  std::vector<int> per_batch(B, 0);
  if (targets.size(0) > 0) {
    // Pull the batch-index column to host ONCE, then read via accessor — a
    // per-element bi[i].item<int>() on a CUDA tensor would force one blocking
    // device->host sync per GT box (CLAUDE.md: no per-batch host syncs).
    auto bi   = targets.select(1, 0).to(torch::kCPU).to(torch::kLong);
    auto bacc = bi.accessor<int64_t, 1>();
    for (int i = 0; i < (int)bi.size(0); ++i) {
      int b = (int)bacc[i];
      if (b >= 0 && b < B) per_batch[b]++;
    }
  }
  int Mmax = 1;
  for (int n : per_batch) Mmax = std::max(Mmax, n);

  // Build padded GT on CPU (accessor needs CPU), then move to model device.
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

  // ── 3. TAL assignment ──────────────────────────────────────────────────
  auto pd_scores_sig = pred_cls.detach().sigmoid();
  TalOutput tal = tal_assign(pd_scores_sig, pred_xyxy.detach(), anc_pts,
                             gt_labels, gt_bboxes, mask_gt,
                             nc, cfg_.topk, cfg_.alpha, cfg_.beta);

  // ── 4. Cls loss: BCE with soft alignment scores, per-anchor mean ────────
  auto bce = F::binary_cross_entropy_with_logits(
      pred_cls, tal.target_scores.to(pred_cls.dtype()),
      F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kSum));
  auto target_scores_sum = tal.target_scores.sum().clamp_min(1.0);
  auto cls_loss = bce / target_scores_sum;

  // ── 5. Box / DFL loss on positive anchors ──────────────────────────────
  torch::Tensor box_loss = torch::zeros({}, opts);
  torch::Tensor dfl_l    = torch::zeros({}, opts);
  if (tal.fg_mask.any().item<bool>()) {
    auto fg = tal.fg_mask;                                   // [B, A]
    auto fg_flat = fg.reshape(-1);                              // [B*A]
    auto pred_xyxy_flat = pred_xyxy.reshape({-1, 4});           // [B*A, 4]
    auto pred_dist_flat = pred_dist.reshape({-1, 4 * reg_max});
    auto t_bboxes_flat  = tal.target_bboxes.reshape({-1, 4});
    auto t_scores_flat  = tal.target_scores.sum(-1).reshape(-1);// importance weight

    auto pos_idx = torch::nonzero(fg_flat).reshape(-1);
    auto pp_xyxy = pred_xyxy_flat.index_select(0, pos_idx);
    auto pp_dist = pred_dist_flat.index_select(0, pos_idx);
    auto tt_bb   = t_bboxes_flat.index_select(0, pos_idx);
    auto w       = t_scores_flat.index_select(0, pos_idx);

    auto ciou = bbox_ciou(pp_xyxy, tt_bb);          // [P]
    box_loss = ((1.0 - ciou) * w).sum() / target_scores_sum;

    // DFL targets in cell units
    // Map anchor positions per-positive
    auto stride_flat = stride_t.repeat({B});                  // [B*A]
    auto anc_flat    = anc_pts.repeat({B, 1});                // [B*A, 2]
    auto pp_str      = stride_flat.index_select(0, pos_idx);  // [P]
    auto pp_anc_pix  = anc_flat.index_select(0, pos_idx);     // [P, 2]
    auto pp_anc_cell = pp_anc_pix / pp_str.unsqueeze(-1);     // anchor in cell units

    auto tt_xyxy_cell = tt_bb / pp_str.unsqueeze(-1);
    auto tt_dist_cell = bbox2dist(tt_xyxy_cell, pp_anc_cell, reg_max);  // [P, 4]

    auto dfl_per = dfl_loss(pp_dist, tt_dist_cell, reg_max);  // [P]
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
