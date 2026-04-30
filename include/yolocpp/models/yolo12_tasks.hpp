#pragma once
//
// YOLO12 task heads — Segment, Pose, OBB, Classify.
//
// Yolo12 detect ships from Ultralytics for all 5 scales; the four task
// heads are not in the upstream assets release as of v8.4.0, but the
// architecture is straightforward to fill in by reusing the v8/v11 task
// modules (Segment / Pose / OBB / Classify all wrap a Detect inside +
// add cv4 / proto). The backbone+neck topology mirrors the detect path
// in `yolo12.cpp`.
//

#include <torch/torch.h>

#include <vector>

#include "yolocpp/models/yolo11_tasks.hpp"  // helper builders shared
#include "yolocpp/models/yolo12.hpp"
#include "yolocpp/models/yolo8_classify.hpp"
#include "yolocpp/models/yolo8_tasks.hpp"

namespace yolocpp::models {

struct Yolo12SegmentImpl : torch::nn::Module {
  Yolo12Scale scale;
  int          nc;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo12SegmentImpl(Yolo12Scale s, int nc, int nm = 32, int npr_unscaled = 256);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
      forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo12Segment);

struct Yolo12PoseImpl : torch::nn::Module {
  Yolo12Scale scale;
  int          nc;
  int          num_kpts, kpt_dim;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo12PoseImpl(Yolo12Scale s, int nc = 1, int num_kpts = 17, int kpt_dim = 3);

  std::tuple<torch::Tensor, torch::Tensor> forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo12Pose);

struct Yolo12OBBImpl : torch::nn::Module {
  Yolo12Scale scale;
  int          nc;
  int          ne;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo12OBBImpl(Yolo12Scale s, int nc = 15, int ne = 1);

  std::tuple<torch::Tensor, torch::Tensor> forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo12OBB);

// v12-cls topology mirrors v11-cls (Conv/C3k2 chain ending in Classify).
struct Yolo12ClassifyImpl : torch::nn::Module {
  Yolo12Scale scale;
  int         nc;
  torch::nn::ModuleList model{nullptr};

  Yolo12ClassifyImpl(Yolo12Scale s, int nc);

  torch::Tensor forward(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo12Classify);

}  // namespace yolocpp::models
