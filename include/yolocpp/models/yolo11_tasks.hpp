#pragma once
//
// YOLO11 task heads — Segment, Pose, OBB, Classify.
//
// All four reuse the v11 backbone+neck (24 modules, with C2PSA at idx 10
// and Detect-style head at idx 23) and the existing v8 Segment/Pose/OBB
// task-head modules. The only structural change vs. v8 is the inner
// Detect's cv3 branch (legacy=false → DWConv→Conv nested form), which we
// thread through the task-head ctors via the `legacy` flag added to v8's
// SegmentImpl / PoseImpl / OBBImpl.
//
// Classify follows a separate, smaller architecture (10 backbone layers
// 0..9 + Classify head at 10; no SPPF, C2PSA at index 9).
//

#include <torch/torch.h>

#include <vector>

#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo8_classify.hpp"  // Classify head
#include "yolocpp/models/yolo8_tasks.hpp"     // Segment / Pose / OBB heads

namespace yolocpp::models {

// ─── Yolo11Segment ────────────────────────────────────────────────────────
struct Yolo11SegmentImpl : torch::nn::Module {
  Yolo11Scale scale;
  int          nc;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo11SegmentImpl(Yolo11Scale s, int nc, int nm = 32, int npr_unscaled = 256);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
      forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo11Segment);

// ─── Yolo11Pose ───────────────────────────────────────────────────────────
struct Yolo11PoseImpl : torch::nn::Module {
  Yolo11Scale scale;
  int          nc;
  int          num_kpts, kpt_dim;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo11PoseImpl(Yolo11Scale s, int nc = 1, int num_kpts = 17, int kpt_dim = 3);

  std::tuple<torch::Tensor, torch::Tensor> forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo11Pose);

// ─── Yolo11OBB ────────────────────────────────────────────────────────────
struct Yolo11OBBImpl : torch::nn::Module {
  Yolo11Scale scale;
  int          nc;
  int          ne;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo11OBBImpl(Yolo11Scale s, int nc = 15, int ne = 1);

  std::tuple<torch::Tensor, torch::Tensor> forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo11OBB);

// ─── Yolo11Classify ───────────────────────────────────────────────────────
//
// Topology (from yolo11-cls.yaml — confirmed against yolo11n-cls.pt):
//   model[0..1]  Conv (3→64 stride 2; 64→128 stride 2)
//   model[2]     C3k2(c_out=256, n=2, c3k=False, e=0.25)
//   model[3]     Conv (stride 2)
//   model[4]     C3k2(c_out=512, n=2, c3k=False, e=0.25)
//   model[5]     Conv (stride 2)
//   model[6]     C3k2(c_out=512, n=2, c3k=True)
//   model[7]     Conv (stride 2)
//   model[8]     C3k2(c_out=1024, n=2, c3k=True)
//   model[9]     C2PSA(c=1024, n=2)
//   model[10]    Classify head (Conv c→1280 + Linear 1280→nc)
struct Yolo11ClassifyImpl : torch::nn::Module {
  Yolo11Scale scale;
  int         nc;
  torch::nn::ModuleList model{nullptr};

  Yolo11ClassifyImpl(Yolo11Scale s, int nc);

  // Logits [N, nc].
  torch::Tensor forward(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo11Classify);

}  // namespace yolocpp::models
