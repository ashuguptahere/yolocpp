#pragma once
//
// Validator — runs inference over a YoloDataset and computes COCO-style mAP.
//

#include <torch/torch.h>

#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/inference/nms.hpp"
#include "yolocpp/metrics/map.hpp"
#include "yolocpp/models/yolo1.hpp"
#include "yolocpp/models/yolo2.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo12.hpp"
#include "yolocpp/models/yolo13.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo3.hpp"
#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/models/yolo5.hpp"
#include "yolocpp/models/yolo6.hpp"
#include "yolocpp/models/yolo7.hpp"
#include "yolocpp/models/yolo8.hpp"
#include "yolocpp/models/yolo9.hpp"
#include "yolocpp/models/yolo10.hpp"

namespace yolocpp::engine {

// Templated over the model-holder type so v8 and v5 (both expose
// forward_eval / stride) share one implementation. Explicit instantiations
// for Yolo8Detect and Yolo5Detect are in validator.cpp.
template <typename ModelHolder>
metrics::mAPResult validate(ModelHolder& model,
                            const datasets::YoloDataset& dataset,
                            torch::Device device,
                            inference::NMSConfig nms_cfg = {.conf_thresh = 0.001f,
                                                           .iou_thresh  = 0.7f,
                                                           .max_det     = 300,
                                                           .max_nms     = 30000,
                                                           .multi_label = true});

struct ValidationOutput {
  metrics::mAPResult                   map;
  std::vector<metrics::DetectionRow>   dets;
  std::vector<metrics::GroundTruthRow> gts;
};
template <typename ModelHolder>
ValidationOutput validate_with_records(
    ModelHolder& model, const datasets::YoloDataset& dataset,
    torch::Device device,
    inference::NMSConfig nms_cfg = {.conf_thresh = 0.001f,
                                   .iou_thresh  = 0.7f,
                                   .max_det     = 300,
                                   .max_nms     = 30000,
                                   .multi_label = true});  // match validate() + upstream

// Render a confusion matrix to a PNG. `names` should have length nc;
// last row/column is "background" added automatically.
void render_confusion_matrix(const std::vector<std::vector<int>>& m,
                             const std::vector<std::string>& names,
                             const std::string& out_path);

// Render upstream-style PR / F1 / P / R curves from CurveData.
//   curves[c][i] is a y-value per class c at x-axis sample i (length L).
//   xs is the L-length x-axis (confidence in [0,1] for F1/P/R, recall for PR).
//   axis labels and the all-class average ("all classes") line follow
//   the upstream visual style.
void render_curve_png(const std::vector<std::vector<double>>& curves,
                      const std::vector<double>& xs,
                      const std::vector<std::string>& names,
                      const std::vector<int>& n_gt_per_class,
                      const std::string& xlabel,
                      const std::string& ylabel,
                      const std::string& title,
                      const std::string& out_path);

// Per-class GT count histogram (upstream `labels.jpg`).
void render_labels_histogram(const std::vector<metrics::GroundTruthRow>& gts,
                             const std::vector<std::string>& names,
                             const std::string& out_path);

// Training-curve plot from results.csv (upstream `results.png`).
void render_results_png(const std::string& csv_path,
                        const std::string& out_path);

// 4×4 sanity grid of post-augmentation training images with their bbox
// targets overlaid (upstream `train_batchN.jpg`).
//   imgs:    [B, 3, H, W] float32 in [0, 1]
//   targets: [M, 6] (b_idx, cls, cx, cy, w, h) — pixel coords in the same H×W
void render_train_batch(const torch::Tensor& imgs,
                        const torch::Tensor& targets,
                        const std::vector<std::string>& names,
                        const std::string& out_path);

}  // namespace yolocpp::engine
