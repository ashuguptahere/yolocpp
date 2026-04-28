#pragma once
//
// YOLOv8 segment / pose / OBB heads & wrapper models.
//
// All three share the same 0–21 backbone+neck construction as detect; only
// the final layer (idx 22) differs. The head extends the Detect head with an
// additional cv4 branch (per-level Conv-Conv-Conv stack) and, for Segment,
// an extra Proto module that produces mask prototypes from the P3 feature.
//

#include <torch/torch.h>

#include <vector>

#include "yolocpp/models/yolov8.hpp"

namespace yolocpp::models {

// ─── Proto (mask prototype generator for Segment) ──────────────────────────
struct ProtoImpl : torch::nn::Module {
  Conv                  cv1{nullptr};
  torch::nn::ConvTranspose2d upsample{nullptr};
  Conv                  cv2{nullptr};
  Conv                  cv3{nullptr};
  ProtoImpl(int c1, int c_, int c2);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Proto);

// ─── Segment head (Detect + cv4 mask coefs + proto) ───────────────────────
struct SegmentImpl : torch::nn::Module {
  // Reuses the Detect head's responsibilities (cv2, cv3, dfl) by composing
  // a Detect inside; we add cv4 + proto on top.
  Detect              detect{nullptr};
  torch::nn::ModuleList cv4{nullptr};   // mask coefficients per level
  Proto               proto{nullptr};
  int nm   = 32;     // mask coef channels
  int npr  = 256;    // prototype intermediate channels (will be width-scaled)
  int nl, nc, reg_max;
  std::vector<int> ch;
  std::vector<double> stride;

  SegmentImpl(int nc, int nm, int npr_unscaled, std::vector<int> ch,
              const YoloV8Scale& scale);

  // Forward: returns (decoded_pred, mask_coefs, prototypes)
  //   decoded_pred:  [N, 4 + nc, A]    (xyxy in input pixels, sigmoided cls)
  //   mask_coefs:    [N, nm, A]
  //   prototypes:    [N, nm, h_p, w_p]
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
      forward(std::vector<torch::Tensor> x);
};
TORCH_MODULE(Segment);

// ─── Pose head (Detect + cv4 keypoint regression) ─────────────────────────
struct PoseImpl : torch::nn::Module {
  Detect detect{nullptr};
  torch::nn::ModuleList cv4{nullptr};
  int nk = 51;             // 17 keypoints × 3
  int nl, nc, reg_max;
  std::vector<int> ch;
  std::vector<double> stride;
  // kpt_shape = (num_kpts, dim) — typically (17, 3) for COCO pose
  int num_kpts = 17, kpt_dim = 3;

  PoseImpl(int nc, int num_kpts, int kpt_dim, std::vector<int> ch);

  // Forward: returns (decoded_pred [N, 4 + nc, A], keypoints [N, num_kpts*3, A])
  // keypoints layout per-anchor: [x, y, conf] × num_kpts in image pixel coords.
  std::tuple<torch::Tensor, torch::Tensor>
      forward(std::vector<torch::Tensor> x);
};
TORCH_MODULE(Pose);

// ─── OBB head (Detect + cv4 angle regression) ─────────────────────────────
struct OBBImpl : torch::nn::Module {
  Detect detect{nullptr};
  torch::nn::ModuleList cv4{nullptr};
  int ne = 1;              // number of extra (angle) channels
  int nl, nc, reg_max;
  std::vector<int> ch;
  std::vector<double> stride;

  OBBImpl(int nc, int ne, std::vector<int> ch);

  // Forward: (decoded_pred [N, 4 + nc, A], angle [N, A])  — angle in radians,
  // decoded into [-π/4, 3π/4].
  std::tuple<torch::Tensor, torch::Tensor>
      forward(std::vector<torch::Tensor> x);
};
TORCH_MODULE(OBB);

// ─── Top-level models ─────────────────────────────────────────────────────

struct YoloV8SegmentImpl : torch::nn::Module {
  YoloV8Scale scale;
  int nc;
  torch::nn::ModuleList model{nullptr};
  std::vector<double> stride;
  YoloV8SegmentImpl(YoloV8Scale s, int nc, int nm = 32, int npr_unscaled = 256);
  // Returns (decoded, mask_coefs, prototypes)
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
      forward_eval(torch::Tensor x);
  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(YoloV8Segment);

struct YoloV8PoseImpl : torch::nn::Module {
  YoloV8Scale scale;
  int nc;
  int num_kpts, kpt_dim;
  torch::nn::ModuleList model{nullptr};
  std::vector<double> stride;
  YoloV8PoseImpl(YoloV8Scale s, int nc = 1, int num_kpts = 17, int kpt_dim = 3);
  // Returns (decoded, keypoints)
  std::tuple<torch::Tensor, torch::Tensor> forward_eval(torch::Tensor x);
  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(YoloV8Pose);

struct YoloV8OBBImpl : torch::nn::Module {
  YoloV8Scale scale;
  int nc;
  int ne;
  torch::nn::ModuleList model{nullptr};
  std::vector<double> stride;
  YoloV8OBBImpl(YoloV8Scale s, int nc = 15, int ne = 1);
  std::tuple<torch::Tensor, torch::Tensor> forward_eval(torch::Tensor x);
  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(YoloV8OBB);

}  // namespace yolocpp::models
