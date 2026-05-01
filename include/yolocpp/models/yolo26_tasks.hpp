#pragma once
//
// YOLO26 task heads — Segment, Pose, OBB, Classify.
//
// All four reuse the v11 backbone+neck (24 modules, with C2PSA at idx 10
// and a Detect-style head at idx 23). Segment26/Pose26/OBB26 each compose
// a Detect26 (DFL-free) inside, plus the standard cv4 (mask coefs / kpt /
// angle) branches and, for Segment, the Proto module.
//
// Classify uses the v11-cls backbone (10 layers, no SPPF, C2PSA at idx 9)
// with the standard Classify head — no Detect involved.
//

#include <torch/torch.h>

#include <vector>

#include "yolocpp/models/yolo11_tasks.hpp"      // for v11 task style helpers
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo8_classify.hpp"    // Classify head
#include "yolocpp/models/yolo8_tasks.hpp"       // Proto

namespace yolocpp::models {

// ─── Segment26 head (Detect26 + cv4 mask coefs + proto) ───────────────────
struct Segment26Impl : torch::nn::Module {
  Detect26              detect{nullptr};
  torch::nn::ModuleList cv4{nullptr};   // mask coefficients per level
  Proto                 proto{nullptr};
  int nm   = 32;
  int npr  = 256;
  int nl, nc;
  std::vector<int> ch;
  std::vector<double> stride;

  Segment26Impl(int nc, int nm, int npr_unscaled, std::vector<int> ch,
                const Yolo26Scale& scale);

  // Returns (decoded_pred [N, 4 + nc, A], mask_coefs [N, nm, A], protos [N, nm, h, w])
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
      forward(std::vector<torch::Tensor> x);
};
TORCH_MODULE(Segment26);

// ─── Pose26 head (Detect26 + cv4 keypoint regression) ─────────────────────
//
// Upstream v26 Pose head adds an uncertainty (sigma) branch alongside
// the standard keypoints: cv4 emits `nk + nk_sigma` channels per anchor,
// where `nk_sigma = num_kpts * 2` (one σx, σy per keypoint). The sigma
// branch is a training-only signal — for inference we emit it through the
// state_dict (so weights load) but slice it off in `forward()`. nk_sigma=0
// reproduces the v8/v11 head exactly.
struct Pose26Impl : torch::nn::Module {
  Detect26 detect{nullptr};
  torch::nn::ModuleList cv4{nullptr};
  int nk = 51;
  int nk_sigma = 34;       // 17 kpts × 2 (matches upstream's shipped v26)
  int nl, nc;
  std::vector<int> ch;
  std::vector<double> stride;
  int num_kpts = 17, kpt_dim = 3;

  Pose26Impl(int nc, int num_kpts, int kpt_dim, std::vector<int> ch,
             int nk_sigma = 34);

  // Returns (decoded_pred, keypoints) — same layout as Pose
  std::tuple<torch::Tensor, torch::Tensor>
      forward(std::vector<torch::Tensor> x);
};
TORCH_MODULE(Pose26);

// ─── OBB26 head (Detect26 + cv4 angle regression) ─────────────────────────
struct OBB26Impl : torch::nn::Module {
  Detect26 detect{nullptr};
  torch::nn::ModuleList cv4{nullptr};
  int ne = 1;
  int nl, nc;
  std::vector<int> ch;
  std::vector<double> stride;

  OBB26Impl(int nc, int ne, std::vector<int> ch);

  // Returns (decoded_pred [N, 4 + nc, A], angle [N, A]) — angle in radians
  std::tuple<torch::Tensor, torch::Tensor>
      forward(std::vector<torch::Tensor> x);
};
TORCH_MODULE(OBB26);

// ─── Top-level wrappers ───────────────────────────────────────────────────

struct Yolo26SegmentImpl : torch::nn::Module {
  Yolo26Scale scale;
  int          nc;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo26SegmentImpl(Yolo26Scale s, int nc, int nm = 32, int npr_unscaled = 256);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
      forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo26Segment);

struct Yolo26PoseImpl : torch::nn::Module {
  Yolo26Scale scale;
  int          nc;
  int          num_kpts, kpt_dim;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo26PoseImpl(Yolo26Scale s, int nc = 1, int num_kpts = 17, int kpt_dim = 3);

  std::tuple<torch::Tensor, torch::Tensor> forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo26Pose);

struct Yolo26OBBImpl : torch::nn::Module {
  Yolo26Scale scale;
  int          nc;
  int          ne;
  torch::nn::ModuleList model{nullptr};
  std::vector<double>   stride;

  Yolo26OBBImpl(Yolo26Scale s, int nc = 15, int ne = 1);

  std::tuple<torch::Tensor, torch::Tensor> forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo26OBB);

// ─── Yolo26Classify ───────────────────────────────────────────────────────
//
// Topology mirrors yolo11-cls.yaml: 10 backbone layers + Classify head.
struct Yolo26ClassifyImpl : torch::nn::Module {
  Yolo26Scale scale;
  int         nc;
  torch::nn::ModuleList model{nullptr};

  Yolo26ClassifyImpl(Yolo26Scale s, int nc);

  // Logits [N, nc].
  torch::Tensor forward(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo26Classify);

}  // namespace yolocpp::models
