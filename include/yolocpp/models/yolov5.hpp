#pragma once
//
// YOLOv5 (anchor-free Ultralytics "v5u" variant — yolov5nu.pt et al.)
//
// Architecture differs from YOLOv8 in two places:
//   1. Layer 0: 6×6 stride-2 conv (v5 stem) instead of v8's 3×3.
//   2. Backbone uses C3 blocks (one CSP path goes through Bottlenecks,
//      the other is a single 1×1 conv) — vs v8's C2f which has cumulative
//      concat across Bottleneck outputs.
//
// The Detect head is identical to v8: anchor-free, DFL projection, three
// scales. Predict / loss / export pipelines reuse v8's infrastructure.
//

#include <torch/torch.h>

#include "yolocpp/models/yolov8.hpp"

namespace yolocpp::models {

// v5 scales differ from v8: max_channels stays at 1024 for all sizes
// (v8 caps m/l/x at 768/512), and v5x has depth=1.33 / width=1.25.
constexpr YoloV8Scale kYoloV5n{0.33, 0.25, 1024};
constexpr YoloV8Scale kYoloV5s{0.33, 0.50, 1024};
constexpr YoloV8Scale kYoloV5m{0.67, 0.75, 1024};
constexpr YoloV8Scale kYoloV5l{1.00, 1.00, 1024};
constexpr YoloV8Scale kYoloV5x{1.33, 1.25, 1024};

// Pick a v5 scale from "n"/"s"/"m"/"l"/"x".
inline YoloV8Scale yolov5_scale_from_letter(const std::string& s) {
  if (s == "n") return kYoloV5n;
  if (s == "s") return kYoloV5s;
  if (s == "m") return kYoloV5m;
  if (s == "l") return kYoloV5l;
  if (s == "x") return kYoloV5x;
  return kYoloV5n;
}

// ─── C3 (CSP bottleneck with 3 conv layers) ─────────────────────────────
struct C3Impl : torch::nn::Module {
  Conv cv1{nullptr};   // 1×1 c1 → c_inner
  Conv cv2{nullptr};   // 1×1 c1 → c_inner (parallel branch)
  Conv cv3{nullptr};   // 1×1 2*c_inner → c2 (after concat)
  torch::nn::ModuleList m{nullptr};
  int c_inner = 0;
  C3Impl(int c1, int c2, int n = 1, bool shortcut = true, int g = 1,
         double e = 0.5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(C3);

// ─── YoloV5Detect (top-level model) ─────────────────────────────────────
//
// Layer order, from yolov5.yaml (Ultralytics v5u):
//   0  Conv 3→64, k=6, s=2, p=2   (P1/2)
//   1  Conv → P2/4
//   2  C3
//   3  Conv → P3/8
//   4  C3
//   5  Conv → P4/16
//   6  C3
//   7  Conv → P5/32
//   8  C3
//   9  SPPF
//   10 Conv 1×1
//   11 Upsample 2×
//   12 Concat with layer 6
//   13 C3 (no shortcut)
//   14 Conv 1×1
//   15 Upsample 2×
//   16 Concat with layer 4
//   17 C3 (no shortcut)             — P3 output
//   18 Conv 3×3 stride 2
//   19 Concat with layer 14
//   20 C3 (no shortcut)             — P4 output
//   21 Conv 3×3 stride 2
//   22 Concat with layer 10
//   23 C3 (no shortcut)             — P5 output
//   24 Detect (P3, P4, P5)          — anchor-free, DFL
struct YoloV5DetectImpl : torch::nn::Module {
  YoloV8Scale scale;
  int         nc;
  torch::nn::ModuleList model{nullptr};
  std::vector<double> stride;

  YoloV5DetectImpl(YoloV8Scale s, int nc);

  std::vector<torch::Tensor> forward_train(torch::Tensor x);
  torch::Tensor              forward_eval(torch::Tensor x);

  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(YoloV5Detect);

}  // namespace yolocpp::models
