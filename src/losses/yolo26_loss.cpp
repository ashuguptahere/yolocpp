#include "yolocpp/losses/yolo26_loss.hpp"

#include <torch/nn/functional.h>

#include <algorithm>
#include <vector>

namespace yolocpp::losses {

namespace F = torch::nn::functional;

namespace {

// Pairwise CIoU between two sets of xyxy boxes (last-dim = 4, broadcastable).
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

  auto cw = torch::max(ax2, bx2) - torch::min(ax1, bx1);
  auto ch = torch::max(ay2, by2) - torch::min(ay1, by1);
  auto c2 = cw.pow(2) + ch.pow(2) + eps;

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

// Build anchor centers and a per-anchor stride tensor.
torch::Tensor make_anchors(const std::vector<double>& strides,
                           const std::vector<std::pair<int, int>>& sizes,
                           const torch::TensorOptions& opts,
                           torch::Tensor* stride_out) {
  std::vector<torch::Tensor> a, s;
  for (size_t i = 0; i < strides.size(); ++i) {
    int   h  = sizes[i].first;
    int   w  = sizes[i].second;
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

// STAL: one-to-one assignment.
//
// Inputs:
//   pd_scores  [B, A, nc]   sigmoid class probs (detached)
//   pd_bboxes  [B, A, 4]    decoded predicted xyxy in pixels (detached)
//   anc_pts    [A, 2]       anchor (x, y) centers in pixels
//   gt_labels  [B, M]       ground-truth class indices
//   gt_bboxes  [B, M, 4]    GT xyxy in pixels
//   mask_gt    [B, M, 1]    1 where the GT is real (else padding)
//
// Outputs:
//   target_labels  [B, A]      assigned class index (0 padded for negatives)
//   target_bboxes  [B, A, 4]   assigned GT xyxy
//   target_scores  [B, A, nc]  soft alignment scores (one-hot scaled by IoU)
//   fg_mask        [B, A]      true on positive anchors
//   pos_iou        [B, A]      IoU of the assigned (anchor, GT) pair (else 0)
struct StalOutput {
  torch::Tensor target_labels;
  torch::Tensor target_bboxes;
  torch::Tensor target_scores;
  torch::Tensor fg_mask;
  torch::Tensor pos_iou;
};

StalOutput stal_assign(const torch::Tensor& pd_scores,
                       const torch::Tensor& pd_bboxes,
                       const torch::Tensor& anc_pts,
                       const torch::Tensor& gt_labels,
                       const torch::Tensor& gt_bboxes,
                       const torch::Tensor& mask_gt,
                       int nc, float alpha, float beta) {
  auto B = pd_scores.size(0);
  auto A = pd_scores.size(1);
  auto M = gt_bboxes.size(1);

  // 1) Anchor inside any GT box.
  auto lt = gt_bboxes.slice(-1, 0, 2);                 // [B, M, 2]
  auto rb = gt_bboxes.slice(-1, 2, 4);
  auto a_xy = anc_pts.unsqueeze(0).unsqueeze(0);       // [1, 1, A, 2]
  auto bbox_deltas = torch::cat({a_xy - lt.unsqueeze(2),
                                  rb.unsqueeze(2) - a_xy}, -1);  // [B, M, A, 4]
  auto in_gts = bbox_deltas.amin(-1).gt(1e-9);         // [B, M, A]

  // 2) Score and IoU per (B, M, A).
  auto gt_idx = gt_labels.to(torch::kLong).clamp(0, nc - 1);  // [B, M]
  auto pd_s_perm = pd_scores.permute({0, 2, 1});       // [B, nc, A]
  auto idx_exp = gt_idx.unsqueeze(-1).expand({B, M, A});
  auto bbox_scores = pd_s_perm.gather(/*dim=*/1, idx_exp);    // [B, M, A]

  auto gt_b = gt_bboxes.unsqueeze(2);   // [B, M, 1, 4]
  auto pd_b = pd_bboxes.unsqueeze(1);   // [B, 1, A, 4]
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
  auto iou = inter / (aw * ah + bw * bh - inter + 1e-9);     // [B, M, A]

  auto align = bbox_scores.pow(alpha) * iou.pow(beta);       // [B, M, A]
  align = align * in_gts.to(align.dtype()) *
          mask_gt.to(align.dtype());

  // 3) Assignment — top-k anchors per GT (k=10 ≈ TAL).
  //
  // YOLOv26's paper recipe trains TWO heads (one2many TAL + one2one STAL)
  // and progressively shifts loss weight between them (ProgLoss). We
  // only have one head, so we use top-k=10 throughout to simulate the
  // TAL-heavy early phase. At top-1 (STAL strict), positive supervision
  // is so sparse (~3/img × 16 = 48 per batch over 8400 anchors) that
  // the model can't bootstrap from an nc=80→nc=5 cls-head reset in
  // short fine-tuning budgets (mAP stays at 0).
  //
  // Add a tiny floor on (in-GT, valid-GT) pairs so top-k always lands
  // on real in-GT anchors at cold start when raw scores are all ≈0.
  const int64_t topk = std::min<int64_t>(10, A);
  auto valid_floor = (in_gts.to(align.dtype()) *
                       mask_gt.to(align.dtype())) * 1e-12;            // [B, M, A]
  auto align_floored = align + valid_floor;
  auto topk_out  = align_floored.topk(topk, /*dim=*/-1, /*largest=*/true);
  auto topk_idx  = std::get<1>(topk_out);                             // [B, M, topk]
  auto topk_val  = std::get<0>(topk_out);                             // [B, M, topk]
  auto valid_top = (topk_val > 1e-13);                                // [B, M, topk]

  // mask_pos[b, m, a] = 1 iff a is in top-k for GT m AND valid.
  auto mask_pos_int = torch::zeros({B, M, A},
                                   align.options().dtype(torch::kInt32));
  mask_pos_int.scatter_(/*dim=*/-1, topk_idx, valid_top.to(torch::kInt32));
  auto mask_pos = mask_pos_int.to(torch::kBool) &
                   (mask_gt.to(torch::kBool));                        // [B, M, A]

  // 4) Resolve overlap: if multiple GTs picked the same anchor, give it to
  //    the one with the highest IoU.
  auto fg_count = mask_pos.to(torch::kFloat32).sum(1);       // [B, A]
  auto multi    = fg_count > 1;
  if (multi.any().item<bool>()) {
    auto iou_masked = iou * mask_pos.to(iou.dtype());        // 0 where unassigned
    auto max_iou_gt = std::get<1>(iou_masked.max(/*dim=*/1)); // [B, A] — best GT per anchor
    auto onehot = torch::zeros_like(mask_pos);
    auto ones   = torch::ones({B, 1, A},
                              mask_pos.options().dtype(torch::kBool));
    onehot = onehot.scatter(/*dim=*/1, max_iou_gt.unsqueeze(1), ones);
    auto multi_3 = multi.unsqueeze(1).expand_as(mask_pos);
    mask_pos = torch::where(multi_3, onehot & mask_pos.any(1, /*keepdim=*/true),
                            mask_pos);
  }

  // 5) Per-anchor target indices (which GT, if any).
  auto target_gt_idx = std::get<1>(mask_pos.to(torch::kFloat32).max(/*dim=*/1));  // [B, A]
  auto fg_mask = mask_pos.any(1);   // [B, A]

  // 6) Gather target boxes/labels.
  auto gather_idx = (torch::arange(B, gt_bboxes.options().dtype(torch::kLong))
                         .reshape({B, 1}).expand({B, A})) * M + target_gt_idx;  // [B, A]

  auto t_bboxes = gt_bboxes.reshape({B * M, 4})
                      .index_select(0, gather_idx.reshape(-1))
                      .reshape({B, A, 4});
  auto t_labels = gt_labels.reshape({B * M})
                      .index_select(0, gather_idx.reshape(-1))
                      .reshape({B, A}).to(torch::kLong);

  // pos_iou[b, a] = iou at the assigned (b, m=target_gt_idx[b,a], a).
  auto iou_pa = iou.gather(/*dim=*/1, target_gt_idx.unsqueeze(1)).squeeze(1);  // [B, A]
  auto pos_iou = iou_pa * fg_mask.to(iou_pa.dtype());        // 0 outside fg

  // ── TAL-style per-GT target normalization (matches upstream v8 TAL) ──
  // For each GT, scale its anchors' IoU so the best-aligned anchor
  // gets target = max(alignment) over THAT GT's positives. This makes
  // the cls target reach 1.0 for the single best anchor of each GT,
  // giving the model a strong supervision signal even when raw IoUs
  // are moderate. Without this, target = iou × onehot (~0.3–0.7), so
  // the model has no reason to push σ above 0.7 — explaining why our
  // v26 predictions stuck at <0.01 confidence in 3-epoch fine-tuning.
  auto align_pos = align * mask_pos.to(align.dtype());           // [B, M, A]
  auto iou_pos   = iou   * mask_pos.to(iou.dtype());             // [B, M, A]
  auto max_align_per_gt = align_pos.amax(-1, /*keepdim=*/true);  // [B, M, 1]
  auto max_iou_per_gt   = iou_pos.amax(-1, /*keepdim=*/true);    // [B, M, 1]
  // norm_align[b,m,a] = align[b,m,a] / max_align_per_gt[b,m,1] × max_iou_per_gt[b,m,1]
  auto norm_align = align_pos / max_align_per_gt.clamp_min(1e-9) *
                     max_iou_per_gt;                              // [B, M, A]
  // Reduce to per-anchor (sum across GT, since each anchor matches ≤1 GT)
  auto target_score_per_anchor = norm_align.amax(/*dim=*/1);     // [B, A]

  auto onehot_labels = torch::nn::functional::one_hot(t_labels, nc)
                           .to(pd_scores.dtype());           // [B, A, nc]
  // Use the TAL-normalized target instead of raw iou_pa.
  auto t_scores = onehot_labels * target_score_per_anchor.unsqueeze(-1);
  t_scores = t_scores * fg_mask.unsqueeze(-1).to(t_scores.dtype());

  StalOutput o;
  o.target_labels = t_labels;
  o.target_bboxes = t_bboxes;
  o.target_scores = t_scores;
  o.fg_mask       = fg_mask;
  o.pos_iou       = pos_iou;
  return o;
}

}  // anonymous namespace

Yolo26Loss::Yolo26Loss(Yolo26LossConfig cfg) : cfg_(cfg) {}

Yolo26LossOutput Yolo26Loss::operator()(
    const std::vector<torch::Tensor>& feats, const torch::Tensor& targets,
    const std::vector<double>& strides, int /*imgsz*/, double progress) const {
  TORCH_CHECK(!feats.empty(), "no feature levels");
  auto opts = feats[0].options().dtype(torch::kFloat32);

  int nc = cfg_.nc;
  int B  = (int)feats[0].size(0);
  int no = 4 + nc;

  // ── 1. Concatenate flat predictions ────────────────────────────────────
  std::vector<torch::Tensor> flat;
  std::vector<std::pair<int, int>> sizes;
  for (auto& f : feats) {
    sizes.emplace_back((int)f.size(2), (int)f.size(3));
    flat.push_back(f.reshape({B, f.size(1), -1}));     // [B, no, h*w]
  }
  auto pred = torch::cat(flat, /*dim=*/2)            // [B, no, A]
                  .permute({0, 2, 1});               // [B, A, no]
  TORCH_CHECK(pred.size(-1) == no,
              "Yolo26Loss: feat channels (", pred.size(-1),
              ") != 4 + nc (", no, ") — wrong head?");

  auto pred_box_raw = pred.slice(-1, 0, 4);          // [B, A, 4] (l, t, r, b raw)
  auto pred_cls     = pred.slice(-1, 4, no);         // [B, A, nc]

  // Anchors live on the same device as the predictions.
  auto anc_opts = opts.device(feats[0].device());
  torch::Tensor stride_t;
  auto anc_pts = make_anchors(strides, sizes, anc_opts, &stride_t);  // [A, 2]

  // Decode predicted boxes (matches Detect26Impl::decode):
  //   distances = softplus(raw); xyxy = [anc - lt, anc + rb] * stride.
  auto box_pos = F::softplus(pred_box_raw);                // [B, A, 4]
  auto pred_xyxy = torch::cat({
      anc_pts.unsqueeze(0) - box_pos.slice(-1, 0, 2),
      anc_pts.unsqueeze(0) + box_pos.slice(-1, 2, 4)
  }, -1) * stride_t.unsqueeze(-1);                          // [B, A, 4]

  // ── 2. Build per-batch padded GT tensors ───────────────────────────────
  std::vector<int> per_batch(B, 0);
  if (targets.size(0) > 0) {
    auto bi = targets.select(1, 0).to(torch::kLong);
    for (int i = 0; i < (int)bi.size(0); ++i) {
      int b = bi[i].item<int>();
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

  // ── 3. STAL (one-to-one) assignment ────────────────────────────────────
  auto pd_scores_sig = pred_cls.detach().sigmoid();
  StalOutput sa = stal_assign(pd_scores_sig, pred_xyxy.detach(), anc_pts,
                              gt_labels, gt_bboxes, mask_gt,
                              nc, cfg_.alpha, cfg_.beta);

  // ── 4. Cls target (TAL-normalized) ─────────────────────────────────────
  // `sa.target_scores` is already the TAL-normalized per-anchor target
  // (built in stal_assign via `align / max(align) × max(iou)` per GT),
  // so the best-aligned anchor of each GT gets target ≈ max_iou ≈ 1.
  // This is the upstream YOLO recipe for matching `bce / target_sum`
  // normalization — without it, BCE pushes σ→iou (~0.3) and confidence
  // never gets high enough for NMS to keep predictions.
  (void)progress;  // ProgLoss now lives in stal_assign (top-k); the
                   // progressive iou^prog soft target was a wrong
                   // interpretation of what ProgLoss does.
  auto t_scores_prog = sa.target_scores;                     // [B, A, nc]

  // Cls loss: BCE w/ logits, normalised by sum of soft targets
  // (matches upstream v8DetectionLoss).
  auto bce = F::binary_cross_entropy_with_logits(
      pred_cls, t_scores_prog,
      F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kSum));
  auto target_scores_sum = t_scores_prog.sum().clamp_min(1.0);
  auto cls_loss = bce / target_scores_sum;

  // ── 5. Box loss (CIoU) on positives ────────────────────────────────────
  // Per-positive weight = TAL-normalized cls target (matches upstream
  // v8 box-iou loss weighting). Higher-aligned anchors of each GT get
  // higher weight, less-aligned get less — same scheme as cls loss.
  torch::Tensor box_loss = torch::zeros({}, opts);
  if (sa.fg_mask.any().item<bool>()) {
    auto fg_flat        = sa.fg_mask.reshape(-1);                  // [B*A]
    auto pred_xyxy_flat = pred_xyxy.reshape({-1, 4});               // [B*A, 4]
    auto t_bboxes_flat  = sa.target_bboxes.reshape({-1, 4});        // [B*A, 4]
    // Sum across class dim to get per-anchor weight (one column nonzero
    // per fg anchor, so this just lifts back to per-anchor).
    auto target_score_per_anchor = sa.target_scores.sum(-1);        // [B, A]
    auto w_flat = target_score_per_anchor.reshape(-1);              // [B*A]

    auto pos_idx = torch::nonzero(fg_flat).reshape(-1);
    auto pp_xyxy = pred_xyxy_flat.index_select(0, pos_idx);
    auto tt_bb   = t_bboxes_flat.index_select(0, pos_idx);
    auto w       = w_flat.index_select(0, pos_idx).clamp_min(1e-6);

    auto ciou = bbox_ciou(pp_xyxy, tt_bb);
    box_loss = ((1.0 - ciou) * w).sum() / target_scores_sum;
  }

  Yolo26LossOutput out;
  out.box   = box_loss * cfg_.box_gain;
  out.cls   = cls_loss * cfg_.cls_gain;
  out.dfl   = torch::zeros({}, opts);   // kept for log-format parity
  out.total = out.box + out.cls;
  return out;
}

}  // namespace yolocpp::losses
