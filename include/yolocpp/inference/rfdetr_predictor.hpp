#pragma once
//
// RF-DETR predictor (#65E). NMS-free: each query is one prediction;
// we just threshold by confidence and (optionally) keep the top-K.
//
// `[B, 4+nc, Q]` (sigmoided cxcywh + sigmoided cls) → vector of
// `inference::Detection` in original-image coordinates.

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "yolocpp/inference/nms.hpp"        // Detection, NMSConfig (conf, max_det)
#include "yolocpp/inference/predictor.hpp"  // Detection
#include "yolocpp/models/rfdetr.hpp"

namespace yolocpp::inference {

// Decode the RF-DETR forward output to detections in letterboxed
// image coordinates. Pure tensor → struct conversion; no model
// dependency, callable from tests on synthetic forward tensors.
//
// `out`:        `[B, 4+nc, Q]` (sigmoided cxcywh + sigmoided cls).
// `imgsz`:      letterbox side (square).
// `conf`:       per-class confidence threshold.
// `max_det`:    keep at most this many detections per image (top-K
//                by score). 0 = unlimited.
//
// Returns one vector per batch image. Each detection's box is in
// (x1, y1, x2, y2) over the letterboxed `imgsz × imgsz` canvas; the
// caller unscales back to the original via `letterbox_unscale`.
std::vector<std::vector<Detection>> rfdetr_decode(
    const torch::Tensor& out, int imgsz, float conf, int max_det = 300);

// End-to-end image → detections via an `RFDetr` holder. Letterboxes
// the input, runs forward_eval, decodes, unscales. Mirrors
// `Predictor::predict` so the registry's `predict_to_file` hook
// can route through it once #65D loads weights.
std::vector<Detection> rfdetr_predict_image(
    yolocpp::models::RFDetr& m, const cv::Mat& bgr,
    int imgsz, const torch::Device& device, float conf,
    int max_det = 300);

}  // namespace yolocpp::inference
