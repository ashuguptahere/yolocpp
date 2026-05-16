#include "yolocpp/losses/yolo7_loss.hpp"

#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace yolocpp::losses {

namespace F = torch::nn::functional;

// ─── Autoanchor: K-means with IoU distance ───────────────────────────────
//
// Standard "autoanchor" routine matching upstream YOLO's k-means step.
// GT boxes are treated as origin-anchored w×h rectangles and clustered
// by 1 − IoU rather than Euclidean distance — Euclidean would bias
// toward larger boxes (their squared error dominates).
std::vector<std::vector<std::pair<float, float>>>
kmeans_anchors(const std::vector<std::pair<float, float>>& gt_whs,
               V7LossConfig& out_cfg, int kmeans_iters) {
  const int nl = (int)out_cfg.strides.size();
  const int na = out_cfg.na;
  const int K  = nl * na;
  std::vector<std::vector<std::pair<float, float>>> result(nl);

  if ((int)gt_whs.size() < K * 2) {
    std::cerr << "[autoanchor] only " << gt_whs.size()
              << " GT boxes; need >= " << (K * 2)
              << " for K-means. Keeping the configured anchors.\n";
    return out_cfg.anchors;
  }

  // 1) Seed K centroids by uniform random sampling without replacement.
  std::mt19937 rng(/*seed=*/0xC0FFEEu);
  std::vector<int> indices(gt_whs.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);
  std::vector<std::pair<float, float>> centers(K);
  for (int k = 0; k < K; ++k) centers[k] = gt_whs[indices[k]];

  auto iou_wh = [](float w1, float h1, float w2, float h2) {
    float iw = std::min(w1, w2);
    float ih = std::min(h1, h2);
    float inter = iw * ih;
    float u = w1 * h1 + w2 * h2 - inter;
    return u > 0 ? inter / u : 0.0f;
  };

  // 2) Lloyd's algorithm with IoU distance.
  std::vector<int> assign(gt_whs.size(), 0);
  for (int it = 0; it < kmeans_iters; ++it) {
    // Assign each GT to its closest centroid.
    for (size_t i = 0; i < gt_whs.size(); ++i) {
      float best = -1.0f; int best_k = 0;
      for (int k = 0; k < K; ++k) {
        float s = iou_wh(gt_whs[i].first, gt_whs[i].second,
                         centers[k].first, centers[k].second);
        if (s > best) { best = s; best_k = k; }
      }
      assign[i] = best_k;
    }
    // Update each centroid to the median (more robust than mean for
    // skewed GT distributions; matches upstream's `np.median`).
    for (int k = 0; k < K; ++k) {
      std::vector<float> ws, hs;
      for (size_t i = 0; i < gt_whs.size(); ++i) {
        if (assign[i] == k) {
          ws.push_back(gt_whs[i].first);
          hs.push_back(gt_whs[i].second);
        }
      }
      if (ws.empty()) continue;
      std::nth_element(ws.begin(), ws.begin() + ws.size() / 2, ws.end());
      std::nth_element(hs.begin(), hs.begin() + hs.size() / 2, hs.end());
      centers[k] = { ws[ws.size() / 2], hs[hs.size() / 2] };
    }
  }

  // 3) Sort centroids by area and assign in 3-per-level slabs.
  std::sort(centers.begin(), centers.end(),
            [](auto& a, auto& b) {
              return a.first * a.second < b.first * b.second;
            });
  for (int li = 0; li < nl; ++li) {
    for (int a = 0; a < na; ++a) {
      result[li].push_back(centers[li * na + a]);
    }
  }
  out_cfg.anchors = result;

  // 4) Log resulting per-level anchors so the user can see what the
  // reclustering produced.
  std::cerr << "[autoanchor] reclustered " << gt_whs.size()
            << " GT boxes → " << K << " anchors across " << nl << " levels:\n";
  for (int li = 0; li < nl; ++li) {
    std::cerr << "[autoanchor]   P" << (li + 3) << " (/" << out_cfg.strides[li]
              << "): ";
    for (auto& a : result[li]) {
      std::cerr << "(" << (int)a.first << "," << (int)a.second << ") ";
    }
    std::cerr << "\n";
  }
  return result;
}

namespace {

// xyxy CIoU between two [..., 4] tensors (broadcastable).
torch::Tensor bbox_ciou_xywh(const torch::Tensor& a_xywh,
                             const torch::Tensor& b_xywh,
                             double eps = 1e-7) {
  // Convert (cx, cy, w, h) to (x1, y1, x2, y2) for both.
  auto ax = a_xywh.select(-1, 0), ay = a_xywh.select(-1, 1);
  auto aw = a_xywh.select(-1, 2), ah = a_xywh.select(-1, 3);
  auto bx = b_xywh.select(-1, 0), by_ = b_xywh.select(-1, 1);
  auto bw = b_xywh.select(-1, 2), bh = b_xywh.select(-1, 3);

  auto ax1 = ax - aw / 2, ay1 = ay - ah / 2;
  auto ax2 = ax + aw / 2, ay2 = ay + ah / 2;
  auto bx1 = bx - bw / 2, by1 = by_ - bh / 2;
  auto bx2 = bx + bw / 2, by2 = by_ + bh / 2;

  auto inter = (torch::min(ax2, bx2) - torch::max(ax1, bx1)).clamp_min(0) *
               (torch::min(ay2, by2) - torch::max(ay1, by1)).clamp_min(0);
  auto union_ = aw * ah + bw * bh - inter + eps;
  auto iou    = inter / union_;

  auto cw = torch::max(ax2, bx2) - torch::min(ax1, bx1);
  auto ch = torch::max(ay2, by2) - torch::min(ay1, by1);
  auto c2 = cw.pow(2) + ch.pow(2) + eps;
  auto rho2 = ((bx - ax).pow(2) + (by_ - ay).pow(2));

  auto v = (4.0 / (M_PI * M_PI)) *
           (torch::atan(bw / (bh + eps)) - torch::atan(aw / (ah + eps))).pow(2);
  torch::Tensor alpha;
  {
    torch::NoGradGuard g;
    alpha = v / (1 - iou + v + eps);
  }
  return iou - rho2 / c2 - alpha * v;
}

}  // anonymous namespace

void V7DetectionLoss::recluster_anchors(
    const std::vector<std::pair<float, float>>& gt_whs) {
  kmeans_anchors(gt_whs, cfg_);
}

V7DetectionLoss::V7DetectionLoss(V7LossConfig cfg) : cfg_(std::move(cfg)) {
  TORCH_CHECK(cfg_.anchors.size() == cfg_.strides.size(),
              "v7 loss: anchors/strides size mismatch");
  TORCH_CHECK(cfg_.scale_xy.size() == cfg_.strides.size(),
              "v7 loss: scale_xy/strides size mismatch");
  if (cfg_.balance.size() != cfg_.strides.size()) {
    cfg_.balance.assign(cfg_.strides.size(), 1.0f);
    if (cfg_.strides.size() == 3) cfg_.balance = {4.0f, 1.0f, 0.4f};
    if (cfg_.strides.size() == 4) cfg_.balance = {4.0f, 1.0f, 0.4f, 0.1f};
  }
}

LossOutput V7DetectionLoss::operator()(
    const std::vector<torch::Tensor>& feats, const torch::Tensor& targets,
    const std::vector<double>& /*strides_arg*/, int /*imgsz*/) const {
  TORCH_CHECK(!feats.empty(), "no feature levels");
  const int nl = (int)feats.size();
  const int na = cfg_.na;
  const int nc = cfg_.nc;
  const int no = 5 + nc;
  auto opts = feats[0].options().dtype(torch::kFloat32);
  auto dev  = feats[0].device();

  // ── 1. Per-level anchor tensors (in stride-cell units) + raw pred views ──
  std::vector<torch::Tensor> anchors_cell(nl);   // [na, 2] per level
  std::vector<torch::Tensor> p(nl);              // [B, na, H, W, no] per level
  std::vector<int>           B_(nl), H_(nl), W_(nl);
  for (int i = 0; i < nl; ++i) {
    int B = (int)feats[i].size(0);
    int H = (int)feats[i].size(2);
    int W = (int)feats[i].size(3);
    B_[i] = B; H_[i] = H; W_[i] = W;
    auto t = feats[i].view({B, na, no, H, W}).permute({0, 1, 3, 4, 2})
                     .contiguous();              // [B, na, H, W, no]
    p[i] = t;

    auto a_t = torch::zeros({na, 2}, opts);
    for (int k = 0; k < na; ++k) {
      a_t[k][0] = cfg_.anchors[i][k].first  / (float)cfg_.strides[i];
      a_t[k][1] = cfg_.anchors[i][k].second / (float)cfg_.strides[i];
    }
    anchors_cell[i] = a_t.to(dev);
  }

  // ── 2. Build target list ───────────────────────────────────────────────
  // targets layout [M, 6]: (batch_idx, cls, cx, cy, w, h) in pixels.
  // We need it broadcast over `na` anchors and per-level matched.
  int M = (int)targets.size(0);
  torch::Tensor tcpu = targets.detach().to(torch::kCPU).to(torch::kFloat32);
  // Filter out padding rows where mask might be 0 (cls=0 with all-zero box?)
  // Our YoloDataset writes only real targets, so we can use as-is.

  // Loss accumulators.
  auto box_loss = torch::zeros({}, opts);
  auto cls_loss = torch::zeros({}, opts);
  auto obj_loss = torch::zeros({}, opts);

  // BCE with logits for obj/cls.
  auto bce_logits = [&](const torch::Tensor& pred, const torch::Tensor& tgt) {
    return F::binary_cross_entropy_with_logits(
        pred, tgt,
        F::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kMean));
  };

  // For each level, build obj target tensor and walk over matched (b, a, gx, gy).
  for (int li = 0; li < nl; ++li) {
    int B = B_[li], H = H_[li], W = W_[li];
    const float stride_i = (float)cfg_.strides[li];
    const float sxy      = cfg_.scale_xy[li];
    auto anc_cell = anchors_cell[li];                  // [na, 2]

    // obj_target accumulated on CPU (accessor<> only works on CPU); moved
    // to `dev` right before the obj-loss BCE call below. Keeping it on
    // the GPU and indexing with accessor<> segfaults on CUDA.
    auto obj_target = torch::zeros({B, na, H, W}, opts.device(torch::kCPU));
    auto pos_box_pred = std::vector<torch::Tensor>{};  // [P, 4] decoded xywh in cell units
    auto pos_box_tgt  = std::vector<torch::Tensor>{};  // [P, 4] gt xywh in cell units
    auto pos_cls_pred = std::vector<torch::Tensor>{};  // [P, nc]
    auto pos_cls_tgt  = std::vector<torch::Tensor>{};  // [P, nc] one-hot
    std::vector<int64_t> pos_b, pos_a, pos_gy, pos_gx;

    if (M > 0) {
      auto ta = tcpu.accessor<float, 2>();
      auto anc_cpu = anc_cell.to(torch::kCPU);
      auto anc_acc = anc_cpu.accessor<float, 2>();

      for (int t = 0; t < M; ++t) {
        int b   = (int)ta[t][0];
        int cls = (int)ta[t][1];
        if (b < 0 || b >= B || cls < 0 || cls >= nc) continue;
        // GT in CELL units of this level.
        float cx_c = ta[t][2] / stride_i;
        float cy_c = ta[t][3] / stride_i;
        float w_c  = ta[t][4] / stride_i;
        float h_c  = ta[t][5] / stride_i;
        if (w_c <= 0 || h_c <= 0) continue;

        // For each anchor, check wh-aspect-ratio match.
        for (int a = 0; a < na; ++a) {
          float aw = anc_acc[a][0];
          float a_h = anc_acc[a][1];
          float rw = w_c / aw, rh = h_c / a_h;
          float r  = std::max(std::max(rw, 1.0f / rw),
                              std::max(rh, 1.0f / rh));
          if (r >= cfg_.anchor_t) continue;

          // Center cell + neighbour cells per offset-prior.
          std::vector<std::pair<int, int>> cells;
          int gx = (int)std::floor(cx_c);
          int gy = (int)std::floor(cy_c);
          if (gx >= 0 && gx < W && gy >= 0 && gy < H) cells.push_back({gx, gy});
          if (cfg_.offset_t > 0) {
            float dx = cx_c - (float)gx;
            float dy = cy_c - (float)gy;
            if (dx < cfg_.offset_t && gx > 0)            cells.push_back({gx - 1, gy});
            if (1.0f - dx < cfg_.offset_t && gx + 1 < W) cells.push_back({gx + 1, gy});
            if (dy < cfg_.offset_t && gy > 0)            cells.push_back({gx, gy - 1});
            if (1.0f - dy < cfg_.offset_t && gy + 1 < H) cells.push_back({gx, gy + 1});
          }
          for (auto [cx_i, cy_i] : cells) {
            pos_b.push_back(b);
            pos_a.push_back(a);
            pos_gx.push_back(cx_i);
            pos_gy.push_back(cy_i);
            // Decoded pred xywh (in cell units): the actual pred values
            // per upstream's decode formula.
            auto pp = p[li].index({b, a, cy_i, cx_i});  // [no]
            auto t_xy = torch::sigmoid(pp.slice(0, 0, 2)) * sxy
                            - 0.5f * (sxy - 1.0f);     // cell-relative
            auto pred_cx = t_xy[0] + (float)cx_i;
            auto pred_cy = t_xy[1] + (float)cy_i;
            torch::Tensor pred_w, pred_h;
            if (cfg_.wh_sigmoid) {
              auto t_wh = torch::sigmoid(pp.slice(0, 2, 4)) * 2.0f;
              pred_w = t_wh[0].pow(2) * anc_cell[a][0];
              pred_h = t_wh[1].pow(2) * anc_cell[a][1];
            } else {
              pred_w = torch::exp(pp[2]) * anc_cell[a][0];
              pred_h = torch::exp(pp[3]) * anc_cell[a][1];
            }
            auto pred_xywh = torch::stack({pred_cx, pred_cy, pred_w, pred_h});

            torch::Tensor gt_xywh =
                torch::tensor({cx_c, cy_c, w_c, h_c}, opts.device(dev));
            pos_box_pred.push_back(pred_xywh.unsqueeze(0));   // [1, 4]
            pos_box_tgt.push_back(gt_xywh.unsqueeze(0));      // [1, 4]
            // cls
            auto cls_logits = pp.slice(0, 5, 5 + nc).unsqueeze(0);
            auto cls_oh = torch::zeros({1, nc}, opts.device(dev));
            cls_oh[0][cls] = 1.0f;
            pos_cls_pred.push_back(cls_logits);
            pos_cls_tgt.push_back(cls_oh);
          }
        }
      }
    }

    // Box / cls loss + obj_target accumulation.
    if (!pos_box_pred.empty()) {
      auto pbox = torch::cat(pos_box_pred, 0);   // [P, 4]
      auto tbox = torch::cat(pos_box_tgt, 0);
      auto ciou = bbox_ciou_xywh(pbox, tbox);    // [P]
      auto box_per = (1.0 - ciou).mean();
      box_loss = box_loss + box_per;
      auto pcls = torch::cat(pos_cls_pred, 0);
      auto tcls = torch::cat(pos_cls_tgt, 0);
      cls_loss = cls_loss + bce_logits(pcls, tcls);

      // Obj target = (1-gr) + gr*IoU.detach() at positives.
      auto iou_det = ciou.detach().clamp(0.0, 1.0);
      auto iou_w   = (1.0f - cfg_.gr) + cfg_.gr * iou_det;   // [P]
      // obj_target indexed by (b, a, gy, gx) — write per-positive.
      auto ow = obj_target.accessor<float, 4>();
      auto iou_w_cpu = iou_w.detach().to(torch::kCPU);
      auto iou_cpu = iou_w_cpu.accessor<float, 1>();
      for (size_t k = 0; k < pos_b.size(); ++k) {
        // If multiple positives land on the same cell, keep the higher IoU.
        float& slot = ow[pos_b[k]][pos_a[k]][pos_gy[k]][pos_gx[k]];
        slot = std::max(slot, iou_cpu[k]);
      }
    }

    // Obj loss with imbalance-aware `pos_weight`. The default mean
    // reduction over the [B, na, H, W] grid dilutes the per-positive
    // gradient by `1/N_total`; at imgsz=1280 with the P6 head this is
    // ~4× weaker than at imgsz=640 (same positives, 4× more cells),
    // which causes anchor-based v7 P6 to barely move on short training
    // budgets (mAP 0.14 vs 0.51 at 3 epochs on screen-detection).
    //
    // Fix: set BCE `pos_weight = N_neg / N_pos` so the positive term is
    // scaled up by exactly the imbalance ratio. After mean over the
    // grid, the positive contribution becomes invariant to N_total
    // (positives are weighted by N_neg, negatives by 1, both averaged
    // over N_total ≈ N_neg → each contributes a constant). Keeps the
    // same loss scale as the original mean form, so other gain
    // hyperparameters need no retuning.
    auto obj_logits = p[li].index({"...", 4});             // [B, na, H, W]
    auto obj_t      = obj_target.to(obj_logits.device());
    auto pos_mask   = (obj_t > 0).to(obj_logits.dtype());
    auto n_pos      = pos_mask.sum().clamp_min(1.0f);
    auto n_neg      = (1.0f - pos_mask).sum().clamp_min(1.0f);
    auto pos_w      = (n_neg / n_pos).clamp(1.0f, 1.0e4f);
    auto obj_per    = F::binary_cross_entropy_with_logits(
        obj_logits, obj_t,
        F::BinaryCrossEntropyWithLogitsFuncOptions()
            .pos_weight(pos_w)
            .reduction(torch::kMean));

    // Auto-balance the per-level obj weight to follow where positives
    // actually live: scale balance proportional to (this level's EMA
    // positive count / mean EMA positive count). Levels with above-
    // average positives get amplified; levels with few positives still
    // contribute (clamped floor 0.1×) to maintain negative supervision.
    //
    // On a COCO-like distribution where pos counts roughly match the
    // upstream prior, the ratio stays near 1 and behavior matches the
    // static `balance`. On a custom dataset where positives concentrate
    // at deep levels (e.g. screen-detection where most objects land at
    // P5/P6), this shifts gradient toward those levels — undoing the
    // 16–64× downweighting that upstream's [4.0, 1.0, 0.25, 0.06] prior
    // imposes on large-object data.
    if (cfg_.autobalance) {
      double n_pos_li = n_pos.item<double>();
      if ((int)ema_pos_count_.size() != nl) ema_pos_count_.assign(nl, 0.0);
      double alpha = std::min(1.0 / (step_count_ / nl + 1.0), 0.02);
      ema_pos_count_[li] =
          (1.0 - alpha) * ema_pos_count_[li] + alpha * n_pos_li;
      double mean_ema = 0.0;
      for (double v : ema_pos_count_) mean_ema += v;
      mean_ema = std::max(mean_ema / (double)nl, 1.0);
      double auto_scale = ema_pos_count_[li] / mean_ema;
      // Clamp: floor 0.1× so empty levels still train negatives;
      // ceiling 10× to prevent runaway on highly imbalanced datasets.
      auto_scale = std::clamp(auto_scale, 0.1, 10.0);
      double effective = (double)cfg_.balance[li] * auto_scale;
      obj_loss = obj_loss + obj_per * (float)effective;
    } else {
      obj_loss = obj_loss + obj_per * cfg_.balance[li];
    }
  }
  if (cfg_.autobalance) ++step_count_;

  LossOutput out;
  out.box   = box_loss * cfg_.box_gain;
  out.cls   = cls_loss * cfg_.cls_gain;
  out.dfl   = obj_loss * cfg_.obj_gain;   // reuse `dfl` slot for `obj` for log parity
  out.total = out.box + out.cls + out.dfl;
  return out;
}

}  // namespace yolocpp::losses
