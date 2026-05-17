#pragma once
//
// yolo2 — Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger", CVPR
// 2017. Architecture: Darknet-19 backbone (19 conv + 5 maxpool + BN +
// leaky 0.1) plus a detection head with a `reorg` passthrough fusing
// 26×26 features from layer-17 into the 13×13 head, and 5 k-means-
// clustered anchors per cell.
//
// `forward_eval` returns `[B, 4+nc, A]` (xyxy in input pixels, sigmoided
// classes / objectness multiplied through), drop-in for `inference::nms`.
//
// Scale variants:
//   • full   — Darknet-19 backbone (~50M params)
//   • tiny   — 9-conv compact variant, no passthrough, anchors=5
//

#include <torch/torch.h>

#include <vector>

namespace yolocpp::models {

enum class Yolo2Scale { Full, Tiny };

// Reorganize a (B, C, H, W) feature map by stride s into (B, C·s², H/s,
// W/s) by collecting s² strided sub-patches as additional channels.
// Matches Darknet's `[reorg]` layer (`stride=2`).
torch::Tensor reorg(const torch::Tensor& x, int stride = 2);

struct Yolo2Impl : torch::nn::Module {
  Yolo2Scale scale = Yolo2Scale::Full;
  int  nc = 20;
  // Anchors are in *grid-cell units* (Darknet convention). 5 anchors
  // by default, paired (w, h). VOC and COCO defaults are baked into
  // the constructor; caller can pass a custom set.
  std::vector<float> anchors;

  // Backbone (Darknet-19) modules. Names match the Darknet cfg's conv
  // order so the `.weights` loader can DFS over `named_children()`.
  // For the full model the backbone is split into "early" (up to the
  // layer-17 passthrough source) and "late" (layer-18 → layer-23).
  torch::nn::Sequential early{nullptr};   // produces 26×26×512
  torch::nn::Sequential late{nullptr};    // 26×26×512 → 13×13×1024
  // Detection head.
  torch::nn::Sequential head_pre{nullptr};   // 13×13×1024 → 13×13×1024 (conv24,25)
  torch::nn::Sequential head_pt{nullptr};    // 26×26×512  → 26×26×64   (conv-pt)
  torch::nn::Sequential head_post{nullptr};  // concat([reorg(pt), pre]) → final 1×1
  // Tiny variant uses a single straight-through Sequential.
  torch::nn::Sequential tiny{nullptr};

  explicit Yolo2Impl(Yolo2Scale scale = Yolo2Scale::Full, int nc = 20,
                     std::vector<float> anchors = {});

  // Raw `[B, num_anchors·(5+nc), H, W]` from the final conv — exactly
  // what Darknet's region layer feeds on (we just reshape, sigmoid,
  // exp, and project to pixel xyxy in `forward_eval`).
  torch::Tensor forward_raw(torch::Tensor x);

  // Returns `[B, 4+nc, A]` ready for `inference::nms`.
  torch::Tensor forward_eval(torch::Tensor x);

  int num_anchors() const { return (int)anchors.size() / 2; }

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo2);

// Default anchor sets in grid-cell units (13×13 grid at 416 input).
std::vector<float> yolo2_voc_anchors();
std::vector<float> yolo2_coco_anchors();

}  // namespace yolocpp::models
