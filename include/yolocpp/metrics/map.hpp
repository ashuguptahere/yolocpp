#pragma once
//
// COCO-style mean Average Precision.
//
// Inputs are accumulated across the validation set and then summarized.
//

#include <torch/torch.h>

#include <vector>

namespace yolocpp::metrics {

struct DetectionRow {
  // All in original image coordinates.
  float x1, y1, x2, y2;
  float conf;
  int   cls;
  int   image_id;   // dense index into the eval set
};

struct GroundTruthRow {
  float x1, y1, x2, y2;
  int   cls;
  int   image_id;
};

struct mAPResult {
  double map_50    = 0.0;    // mAP at IoU=0.50
  double map_50_95 = 0.0;    // mean over IoU thresholds {0.50, 0.55, ..., 0.95}
  std::vector<double> ap_per_class_50;     // size = nc
  std::vector<double> ap_per_class_50_95;  // size = nc

  // Size-stratified mAP@0.5:0.95 (#54C). COCO area buckets:
  //   small  : area ≤ 32² = 1024 px²
  //   medium : 1024 < area ≤ 96² = 9216 px²
  //   large  : area > 9216 px²
  // Each is the mean over the same {0.50, 0.55, ..., 0.95} IoU
  // grid, computed against only the GTs whose box area falls in the
  // bucket. Detections that don't match any in-bucket GT count as
  // false positives (standard COCO behaviour). Set to 0 if no GT
  // in the bucket — check `n_gt_*` to disambiguate "no GTs" from
  // "true 0 mAP".
  double map_50_95_small  = 0.0;
  double map_50_95_medium = 0.0;
  double map_50_95_large  = 0.0;
  int    n_gt_small  = 0;
  int    n_gt_medium = 0;
  int    n_gt_large  = 0;
};

// Compute mAP for the given detections and ground-truths.
//   nc:           number of classes
mAPResult compute_map(const std::vector<DetectionRow>& dets,
                      const std::vector<GroundTruthRow>& gts, int nc);

// Per-class precision / recall / F1 sampled across confidence thresholds.
//   • px:        confidence thresholds (1000 evenly-spaced points in [0,1])
//   • p[c][i]    precision for class c at threshold px[i]
//   • r[c][i]    recall    for class c at threshold px[i]
//   • f1[c][i]   F1        for class c at threshold px[i]
//   • pr_p[c][j] precision at recall-threshold j (101-point COCO recall grid)
//   • pr_r[c][j] recall threshold j (linspace 0..1)
struct CurveData {
  std::vector<double>              px;       // [1000] confidence
  std::vector<std::vector<double>> p, r, f1; // [nc][1000]
  std::vector<std::vector<double>> pr_p;     // [nc][101]
  std::vector<double>              pr_r;     // [101]
  std::vector<int>                 n_gt_per_class;  // [nc]
};

CurveData compute_curves(const std::vector<DetectionRow>& dets,
                          const std::vector<GroundTruthRow>& gts, int nc,
                          double iou_thresh = 0.5);

// Build an (nc+1) × (nc+1) confusion matrix:
//   rows    = predicted class (last row = "background"/no prediction)
//   columns = ground-truth class (last column = "background"/false positive)
// A detection matches a GT if IoU ≥ iou_thresh and class is the gt's class
// (the standard COCO confusion matrix definition).
std::vector<std::vector<int>> confusion_matrix(
    const std::vector<DetectionRow>& dets,
    const std::vector<GroundTruthRow>& gts, int nc,
    double conf_thresh = 0.25,
    double iou_thresh  = 0.45);

}  // namespace yolocpp::metrics
