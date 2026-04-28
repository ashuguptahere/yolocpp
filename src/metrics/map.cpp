#include "yolocpp/metrics/map.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

namespace yolocpp::metrics {

namespace {

double iou(const DetectionRow& d, const GroundTruthRow& g) {
  float xx1 = std::max(d.x1, g.x1), yy1 = std::max(d.y1, g.y1);
  float xx2 = std::min(d.x2, g.x2), yy2 = std::min(d.y2, g.y2);
  float w   = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
  float inter = w * h;
  float ad = (d.x2 - d.x1) * (d.y2 - d.y1);
  float ag = (g.x2 - g.x1) * (g.y2 - g.y1);
  return inter / (ad + ag - inter + 1e-9);
}

// Compute AP at a single IoU threshold for a single class.
//   class_dets and class_gts have already been filtered to one class.
double compute_ap_class(std::vector<DetectionRow> class_dets,
                        const std::vector<GroundTruthRow>& class_gts,
                        double iou_thresh) {
  if (class_dets.empty()) return 0.0;
  if (class_gts.empty())  return 0.0;

  // Sort by descending conf.
  std::sort(class_dets.begin(), class_dets.end(),
            [](const DetectionRow& a, const DetectionRow& b) {
              return a.conf > b.conf;
            });

  // Group gts by image_id for fast lookup; track matched flags.
  std::map<int, std::vector<size_t>> gts_by_img;
  for (size_t i = 0; i < class_gts.size(); ++i)
    gts_by_img[class_gts[i].image_id].push_back(i);
  std::vector<bool> matched(class_gts.size(), false);

  std::vector<int> tp(class_dets.size(), 0), fp(class_dets.size(), 0);
  for (size_t i = 0; i < class_dets.size(); ++i) {
    const auto& d = class_dets[i];
    auto it = gts_by_img.find(d.image_id);
    if (it == gts_by_img.end()) { fp[i] = 1; continue; }
    double best_iou = 0.0;
    int    best_j   = -1;
    for (auto j : it->second) {
      double v = iou(d, class_gts[j]);
      if (v > best_iou) { best_iou = v; best_j = (int)j; }
    }
    if (best_j >= 0 && best_iou >= iou_thresh && !matched[best_j]) {
      tp[i] = 1; matched[best_j] = true;
    } else {
      fp[i] = 1;
    }
  }

  // Cumulative
  int    n_gt = (int)class_gts.size();
  std::vector<double> precisions, recalls;
  int ctp = 0, cfp = 0;
  for (size_t i = 0; i < class_dets.size(); ++i) {
    ctp += tp[i]; cfp += fp[i];
    double p = (double)ctp / std::max(1, ctp + cfp);
    double r = (double)ctp / std::max(1, n_gt);
    precisions.push_back(p);
    recalls.push_back(r);
  }

  // 11-point interpolation à la Pascal VOC, simplified to all-points (COCO).
  // We use the COCO 101-point AP.
  double ap = 0.0;
  std::vector<double> rec_thresh(101);
  for (int i = 0; i < 101; ++i) rec_thresh[i] = i / 100.0;

  // Right-to-left max over precision.
  std::vector<double> p_max = precisions;
  for (int i = (int)p_max.size() - 2; i >= 0; --i)
    p_max[i] = std::max(p_max[i], p_max[i + 1]);

  for (double r : rec_thresh) {
    auto it = std::lower_bound(recalls.begin(), recalls.end(), r);
    if (it == recalls.end()) continue;
    size_t idx = it - recalls.begin();
    ap += p_max[idx];
  }
  ap /= 101.0;
  return ap;
}

}  // anonymous namespace

mAPResult compute_map(const std::vector<DetectionRow>& dets,
                      const std::vector<GroundTruthRow>& gts, int nc) {
  mAPResult r;
  r.ap_per_class_50.assign(nc, 0.0);
  r.ap_per_class_50_95.assign(nc, 0.0);

  // Bucket by class once.
  std::vector<std::vector<DetectionRow>> dc(nc);
  std::vector<std::vector<GroundTruthRow>> gc(nc);
  for (auto& d : dets) if (d.cls >= 0 && d.cls < nc) dc[d.cls].push_back(d);
  for (auto& g : gts)  if (g.cls >= 0 && g.cls < nc) gc[g.cls].push_back(g);

  std::vector<double> ious;
  for (int i = 0; i < 10; ++i) ious.push_back(0.5 + 0.05 * i);

  double sum50 = 0.0, sum5095 = 0.0;
  int n_active = 0;
  for (int c = 0; c < nc; ++c) {
    if (gc[c].empty()) continue;
    n_active++;
    double ap50  = compute_ap_class(dc[c], gc[c], 0.5);
    double mean  = 0.0;
    for (double t : ious) mean += compute_ap_class(dc[c], gc[c], t);
    mean /= (double)ious.size();
    r.ap_per_class_50[c]    = ap50;
    r.ap_per_class_50_95[c] = mean;
    sum50   += ap50;
    sum5095 += mean;
  }
  if (n_active > 0) {
    r.map_50    = sum50    / n_active;
    r.map_50_95 = sum5095 / n_active;
  }
  return r;
}

CurveData compute_curves(const std::vector<DetectionRow>& dets,
                          const std::vector<GroundTruthRow>& gts, int nc,
                          double iou_thresh) {
  CurveData c;
  c.px.resize(1000);
  for (int i = 0; i < 1000; ++i) c.px[i] = i / 999.0;
  c.pr_r.resize(101);
  for (int i = 0; i < 101; ++i) c.pr_r[i] = i / 100.0;
  c.p.assign(nc,  std::vector<double>(1000, 0.0));
  c.r.assign(nc,  std::vector<double>(1000, 0.0));
  c.f1.assign(nc, std::vector<double>(1000, 0.0));
  c.pr_p.assign(nc, std::vector<double>(101, 0.0));
  c.n_gt_per_class.assign(nc, 0);
  for (const auto& g : gts)
    if (g.cls >= 0 && g.cls < nc) ++c.n_gt_per_class[g.cls];

  // Per-class processing.
  for (int cls_id = 0; cls_id < nc; ++cls_id) {
    if (c.n_gt_per_class[cls_id] == 0) continue;
    // Filter dets and gts to this class.
    std::vector<DetectionRow> ds;
    std::vector<GroundTruthRow> gs;
    for (auto& d : dets) if (d.cls == cls_id) ds.push_back(d);
    for (auto& g : gts)  if (g.cls == cls_id) gs.push_back(g);
    if (ds.empty()) continue;

    // Sort dets by descending conf.
    std::sort(ds.begin(), ds.end(),
              [](const DetectionRow& a, const DetectionRow& b) {
                return a.conf > b.conf;
              });

    // For each det, decide TP/FP. Greedy match per image.
    std::map<int, std::vector<size_t>> gts_by_img;
    for (size_t i = 0; i < gs.size(); ++i) gts_by_img[gs[i].image_id].push_back(i);
    std::vector<bool> matched(gs.size(), false);

    std::vector<int>    tp_flag(ds.size(), 0);
    std::vector<double> conf_at(ds.size(), 0.0);
    int ctp = 0;
    for (size_t i = 0; i < ds.size(); ++i) {
      conf_at[i] = ds[i].conf;
      auto it = gts_by_img.find(ds[i].image_id);
      if (it == gts_by_img.end()) continue;
      double best = 0.0; int best_j = -1;
      for (auto j : it->second) {
        if (matched[j]) continue;
        double v = iou(ds[i], gs[j]);
        if (v > best) { best = v; best_j = (int)j; }
      }
      if (best_j >= 0 && best >= iou_thresh) {
        tp_flag[i] = 1;
        matched[best_j] = true;
        ++ctp;
      }
    }

    // Build precision/recall vs confidence: at each threshold τ in c.px,
    // count dets with conf ≥ τ as the active set.
    int n_gt = c.n_gt_per_class[cls_id];
    int j = 0;  // index into ds (sorted desc), advances as τ decreases
    int t_tp = 0, t_total = 0;
    // Iterate τ from high to low to incrementally accumulate.
    for (int ix = 999; ix >= 0; --ix) {
      double tau = c.px[ix];
      while (j < (int)ds.size() && ds[j].conf >= tau) {
        if (tp_flag[j]) ++t_tp;
        ++t_total;
        ++j;
      }
      double prec = t_total > 0 ? (double)t_tp / t_total : 1.0;
      double rec  = n_gt > 0    ? (double)t_tp / n_gt    : 0.0;
      c.p[cls_id][ix]  = prec;
      c.r[cls_id][ix]  = rec;
      c.f1[cls_id][ix] = (prec + rec) > 0 ? (2 * prec * rec / (prec + rec)) : 0.0;
    }

    // Build PR curve at the 101-recall grid (right-to-left max precision).
    // Sample (recall_at_τ, precision_at_τ) by walking τ from high to low.
    std::vector<std::pair<double, double>> rp;  // (recall, precision)
    int tp2 = 0, total2 = 0;
    for (size_t i = 0; i < ds.size(); ++i) {
      if (tp_flag[i]) ++tp2;
      ++total2;
      double rec  = (double)tp2 / n_gt;
      double prec = (double)tp2 / total2;
      rp.emplace_back(rec, prec);
    }
    // Right-to-left max precision.
    for (int i = (int)rp.size() - 2; i >= 0; --i)
      rp[i].second = std::max(rp[i].second, rp[i + 1].second);
    // Project onto the 101-point recall grid.
    for (int ri = 0; ri < 101; ++ri) {
      double r_target = c.pr_r[ri];
      auto   it = std::lower_bound(rp.begin(), rp.end(),
          std::make_pair(r_target, 0.0),
          [](const std::pair<double, double>& a, const std::pair<double, double>& b){
            return a.first < b.first;
          });
      c.pr_p[cls_id][ri] = (it == rp.end()) ? 0.0 : it->second;
    }
  }
  return c;
}

std::vector<std::vector<int>> confusion_matrix(
    const std::vector<DetectionRow>& dets,
    const std::vector<GroundTruthRow>& gts, int nc,
    double conf_thresh, double iou_thresh) {
  // (nc+1) × (nc+1) matrix; index `nc` is the background row/column.
  std::vector<std::vector<int>> m(nc + 1, std::vector<int>(nc + 1, 0));

  // Group dets by image.
  std::map<int, std::vector<const DetectionRow*>> dets_by_img;
  for (const auto& d : dets) {
    if (d.conf < conf_thresh) continue;
    if (d.cls < 0 || d.cls >= nc) continue;
    dets_by_img[d.image_id].push_back(&d);
  }
  std::map<int, std::vector<const GroundTruthRow*>> gts_by_img;
  for (const auto& g : gts) {
    if (g.cls < 0 || g.cls >= nc) continue;
    gts_by_img[g.image_id].push_back(&g);
  }

  // Process each image: greedy match dets→gts by descending IoU.
  std::set<int> all_imgs;
  for (auto& kv : dets_by_img) all_imgs.insert(kv.first);
  for (auto& kv : gts_by_img)  all_imgs.insert(kv.first);

  for (int img : all_imgs) {
    auto& ds = dets_by_img[img];
    auto& gs = gts_by_img[img];

    // Sort dets by descending conf.
    std::sort(ds.begin(), ds.end(),
              [](const DetectionRow* a, const DetectionRow* b) {
                return a->conf > b->conf;
              });

    std::vector<bool> gt_matched(gs.size(), false);
    std::vector<bool> det_matched(ds.size(), false);
    for (size_t i = 0; i < ds.size(); ++i) {
      double best_iou = 0.0;
      int    best_j   = -1;
      for (size_t j = 0; j < gs.size(); ++j) {
        if (gt_matched[j]) continue;
        double v = iou(*ds[i], *gs[j]);
        if (v > best_iou) { best_iou = v; best_j = (int)j; }
      }
      if (best_j >= 0 && best_iou >= iou_thresh) {
        // pred row = ds[i]->cls, gt col = gs[best_j]->cls.
        m[ds[i]->cls][gs[best_j]->cls] += 1;
        gt_matched[best_j] = true;
        det_matched[i]     = true;
      }
    }
    // Unmatched dets are false-positives → predicted X, gt = background (last col).
    for (size_t i = 0; i < ds.size(); ++i)
      if (!det_matched[i]) m[ds[i]->cls][nc] += 1;
    // Unmatched gts are false-negatives → predicted = background (last row),
    // gt = gs[j]->cls.
    for (size_t j = 0; j < gs.size(); ++j)
      if (!gt_matched[j]) m[nc][gs[j]->cls] += 1;
  }
  return m;
}

}  // namespace yolocpp::metrics
