#pragma once
//
// YOLO4 — Bochkovskiy et al. (April 2020).
//
// Backbone : CSPDarknet53 (Mish in backbone)
// Neck     : SPP (5/9/13 maxpools) + PANet (top-down + bottom-up, LeakyReLU)
// Head     : v3-style anchor-based, 3 scales × 3 anchors × (5 + nc)
//
// This header carries the architecture AND the Darknet `.weights` binary
// loader. v4 ships from AlexeyAB's release as a flat float32 blob in
// yolov4.cfg layer order (no key names) — totally unlike the upstream
// nested PyTorch `.pt` state-dicts. To make the binary stream match our
// PyTorch parameters position-for-position, every conv-bearing module is
// REGISTERED in cfg-execution order (so DFS over registered children
// yields conv blocks in the same order Darknet wrote them).
//
// CSP stage registration order (matches yolov4.cfg [convolutional] order):
//   down → cv2 (shortcut) → cv1 (main) → m[i].cv1, m[i].cv2 → cv3 → cv4
//
// Stage layout follows AlexeyAB's released yolo4.cfg:
//   stage 1 (special):  64 ch, 1 residual,  hidden=32  (CSP cv1 = full c)
//   stages 2..5 (canon): cN ch, N residuals, hidden=c/2 (CSP cv1 = c/2)
//

#include <torch/torch.h>

#include <string>
#include <vector>

namespace yolocpp::models {

// ─── ConvMish = Conv2d + BN + Mish (backbone activation) ─────────────────
struct ConvMishImpl : torch::nn::Module {
  torch::nn::Conv2d      conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  ConvMishImpl(int c_in, int c_out, int k = 1, int s = 1, int p = -1, int g = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ConvMish);

// ─── ConvLeaky = Conv2d + BN + LeakyReLU(0.1) (neck/head activation) ─────
struct ConvLeakyImpl : torch::nn::Module {
  torch::nn::Conv2d      conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  ConvLeakyImpl(int c_in, int c_out, int k = 1, int s = 1, int p = -1, int g = 1);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ConvLeaky);

// ─── DarknetResidualMish (1×1 → 3×3 with skip; Mish; configurable hidden c) ─
struct DarknetResidualMishImpl : torch::nn::Module {
  ConvMish cv1{nullptr};   // c → hidden (1×1)
  ConvMish cv2{nullptr};   // hidden → c (3×3)
  DarknetResidualMishImpl(int c, int hidden);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DarknetResidualMish);

// ─── CSPStage — CSP-wrapped Darknet stage ─────────────────────────────────
//
// Registration order (matches yolov4.cfg execution order):
//   down → cv2 (shortcut path 1×1) → cv1 (main path 1×1)
//        → m[0..N-1] (each = cv1 1×1, cv2 3×3)
//        → cv3 (post-residual 1×1) → cv4 (transition 1×1)
//
// Forward order (semantically distinct from registration):
//   y_main  = cv1(down(x));  for r in m: y_main = r(y_main);  y_main = cv3(y_main)
//   y_short = cv2(down(x))
//   out     = cv4(concat(y_main, y_short))
struct CSPStageImpl : torch::nn::Module {
  ConvMish              down{nullptr};
  ConvMish              cv2{nullptr};   // shortcut path — REGISTERED FIRST per cfg
  ConvMish              cv1{nullptr};   // main path
  torch::nn::ModuleList m{nullptr};
  ConvMish              cv3{nullptr};
  ConvMish              cv4{nullptr};
  CSPStageImpl(int c_in, int c_out, int n, bool first_stage);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CSPStage);

// ─── SPP block — 5/9/13 maxpools concatenated with identity ──────────────
struct SPPv4Impl : torch::nn::Module {
  torch::nn::MaxPool2d m5{nullptr}, m9{nullptr}, m13{nullptr};
  SPPv4Impl();
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SPPv4);

// ─── Yolo4 model ─────────────────────────────────────────────────────────
//
// Registration order (matches yolov4.cfg, top to bottom):
//   stem → s1..s5 (backbone)
//   spp_pre1..3 → spp → spp_post1..3
//   p4_td_red → p4_lateral → p4_td1..5
//   p3_td_red → p3_lateral → p3_td1..5
//   p3_out_pre → p3_out
//   p4_bu_down → p4_bu1..5 → p4_out_pre → p4_out
//   p5_bu_down → p5_bu1..5 → p5_out_pre → p5_out
// v4 has a single architecture (no scales). `Yolo4Scale` is a placeholder
// tag so v4 conforms to the trainer's `M(scale, nc)` EMA construction.
struct Yolo4Scale { int dummy = 0; };
constexpr Yolo4Scale kYolo4{};

struct Yolo4Impl : torch::nn::Module {
  Yolo4Scale scale;
  int nc;
  std::vector<double> stride;
  // Anchors in pixel units calibrated to imgsz=608 (yolov4.cfg). Per
  // scale (P3, P4, P5) × 3 anchors each. Loss/eval rescale to actual
  // imgsz at runtime.
  static constexpr int na = 3;

  // Backbone
  ConvMish  stem{nullptr};
  CSPStage  s1{nullptr}, s2{nullptr}, s3{nullptr}, s4{nullptr}, s5{nullptr};

  // SPP block (Leaky activation in neck)
  ConvLeaky spp_pre1{nullptr}, spp_pre2{nullptr}, spp_pre3{nullptr};
  SPPv4     spp{nullptr};
  ConvLeaky spp_post1{nullptr}, spp_post2{nullptr}, spp_post3{nullptr};

  // PANet top-down P5 → P4
  ConvLeaky p4_td_red{nullptr};      // from spp_post3 (512) → 256
  ConvLeaky p4_lateral{nullptr};     // from b4 (512) → 256
  ConvLeaky p4_td1{nullptr}, p4_td2{nullptr}, p4_td3{nullptr},
            p4_td4{nullptr}, p4_td5{nullptr};

  // PANet top-down P4 → P3
  ConvLeaky p3_td_red{nullptr};      // from p4_td5 (256) → 128
  ConvLeaky p3_lateral{nullptr};     // from b3 (256) → 128
  ConvLeaky p3_td1{nullptr}, p3_td2{nullptr}, p3_td3{nullptr},
            p3_td4{nullptr}, p3_td5{nullptr};

  // P3 head (output)
  ConvLeaky         p3_out_pre{nullptr};
  torch::nn::Conv2d p3_out{nullptr};

  // PANet bottom-up P3 → P4
  ConvLeaky p4_bu_down{nullptr};
  ConvLeaky p4_bu1{nullptr}, p4_bu2{nullptr}, p4_bu3{nullptr},
            p4_bu4{nullptr}, p4_bu5{nullptr};

  // P4 head (output)
  ConvLeaky         p4_out_pre{nullptr};
  torch::nn::Conv2d p4_out{nullptr};

  // PANet bottom-up P4 → P5
  ConvLeaky p5_bu_down{nullptr};
  ConvLeaky p5_bu1{nullptr}, p5_bu2{nullptr}, p5_bu3{nullptr},
            p5_bu4{nullptr}, p5_bu5{nullptr};

  // P5 head (output)
  ConvLeaky         p5_out_pre{nullptr};
  torch::nn::Conv2d p5_out{nullptr};

  // Old (nc) and new (scale, nc) ctors — the latter conforms to
  // TrainerT<M>'s `M(scale, nc)` EMA construction.
  explicit Yolo4Impl(int nc = 80);
  Yolo4Impl(Yolo4Scale s, int nc) : Yolo4Impl(nc) { scale = s; }

  // Returns 3 raw output tensors at strides 32 / 16 / 8 (P5, P4, P3).
  // Each shape: [B, 3 * (5 + nc), H_i, W_i]
  std::vector<torch::Tensor> forward(torch::Tensor x);

  // Training-mode forward: returns the same 3 raw output tensors but
  // reordered to stride-ASCENDING (P3, P4, P5) — matches V7DetectionLoss
  // input convention.
  std::vector<torch::Tensor> forward_train(torch::Tensor x);

  // Decoded inference output: [B, 4 + nc, A] with boxes in xyxy at INPUT
  // letterbox pixel coords and channels 4..(4+nc) holding obj * cls
  // (sigmoid'd). Input must be the same imgsz as letterbox preprocessing.
  torch::Tensor forward_eval(torch::Tensor x);

  // Match-by-name state_dict loader (used after we've converted
  // yolov4.weights → yolo4.pt). Returns the number of tensors copied.
  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo4);

}  // namespace yolocpp::models
