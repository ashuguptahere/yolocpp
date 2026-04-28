#pragma once
//
// Non-maximum suppression for YOLOv8-style detections.
//
// Input pred is the model's eval output: [N, 4 + nc, A]
//   - dim 1 channels 0..3 are box xyxy in input pixels
//   - dim 1 channels 4..(4+nc) are sigmoided class scores
//
// nms_per_image returns, per image, a [k, 6] tensor with rows
//   (x1, y1, x2, y2, conf, cls).
//

#include <torch/torch.h>

#include <vector>

namespace yolocpp::inference {

struct NMSConfig {
  float conf_thresh = 0.25f;
  float iou_thresh  = 0.45f;
  int   max_det     = 300;
  int   max_nms     = 30000;  // detections to NMS at most per image
};

// Returns vector of length N, each element [k, 6].
std::vector<torch::Tensor> nms(torch::Tensor pred, NMSConfig cfg = {});

}  // namespace yolocpp::inference
