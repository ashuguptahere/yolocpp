#pragma once
//
// YOLO13 task heads — Segment, Pose, OBB, Classify.
//
// Same construction as yolo12_tasks.hpp: build the v13 backbone+neck (layers
// 0..31 of the detect schedule, stopping before the Detect head) and append the
// shared Segment / Pose / OBB / Classify head from yolo8_tasks. iMoonLab ships
// no upstream task weights for v13, so these are trained on COCO ourselves (the
// #60 harness). The backbone+neck mirrors `yolo13.cpp`'s detect path verbatim.
//

#include <torch/torch.h>

#include <vector>

#include "yolocpp/models/yolo11_tasks.hpp"   // shared helper builders
#include "yolocpp/models/yolo13.hpp"
#include "yolocpp/models/yolo8_classify.hpp"
#include "yolocpp/models/yolo8_tasks.hpp"

namespace yolocpp::models {

struct Yolo13SegmentImpl : torch::nn::Module {
  Yolo13Scale scale;
  int          nc;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo13SegmentImpl(Yolo13Scale s, int nc, int nm = 32, int npr_unscaled = 256);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
      forward_eval(torch::Tensor x);
  std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>
      forward_train_seg(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo13Segment);

struct Yolo13PoseImpl : torch::nn::Module {
  Yolo13Scale scale;
  int          nc;
  int          num_kpts, kpt_dim;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo13PoseImpl(Yolo13Scale s, int nc = 1, int num_kpts = 17, int kpt_dim = 3);

  std::tuple<torch::Tensor, torch::Tensor> forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo13Pose);

struct Yolo13OBBImpl : torch::nn::Module {
  Yolo13Scale scale;
  int          nc;
  int          ne;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo13OBBImpl(Yolo13Scale s, int nc = 15, int ne = 1);

  std::tuple<torch::Tensor, torch::Tensor> forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo13OBB);

// v13-cls topology mirrors v12/v11-cls (Conv/C3k2 chain ending in Classify).
struct Yolo13ClassifyImpl : torch::nn::Module {
  Yolo13Scale scale;
  int         nc;
  torch::nn::ModuleList model{nullptr};

  Yolo13ClassifyImpl(Yolo13Scale s, int nc);

  torch::Tensor forward(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo13Classify);

}  // namespace yolocpp::models
