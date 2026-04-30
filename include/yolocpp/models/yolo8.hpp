#pragma once
//
// YOLO8 detection model in libtorch.
//
// Architecture matches Ultralytics yolo8.yaml exactly so that the
// Ultralytics .pt state_dict maps onto our parameters in iteration order.
//
// Variants are scaled by depth and width multipliers:
//   n: depth=0.33, width=0.25, max_channels=1024
//   s: depth=0.33, width=0.50, max_channels=1024
//   m: depth=0.67, width=0.75, max_channels=768
//   l: depth=1.00, width=1.00, max_channels=512
//   x: depth=1.00, width=1.25, max_channels=512
//

#include <torch/torch.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace yolocpp::models {

struct Yolo8Scale {
  double depth_multiple;
  double width_multiple;
  int    max_channels;
};

constexpr Yolo8Scale kYolo8n{0.33, 0.25, 1024};
constexpr Yolo8Scale kYolo8s{0.33, 0.50, 1024};
constexpr Yolo8Scale kYolo8m{0.67, 0.75, 768};
constexpr Yolo8Scale kYolo8l{1.00, 1.00, 512};
constexpr Yolo8Scale kYolo8x{1.00, 1.25, 512};

// Thread-local BN epsilon override for ConvImpl / DWConvImpl.
//
// Ultralytics' yaml-built models use BN eps=1e-3 for detect/segment/pose/
// obb, but plain PyTorch default 1e-5 for the *cls models. Switching this
// global before constructing a Yolo*Classify (and restoring afterward)
// lets all cls submodules pick up the right eps without threading a new
// parameter through every CSP block constructor.
struct BnEpsScope {
  double prev;
  BnEpsScope(double new_eps);
  ~BnEpsScope();
};
double get_default_bn_eps();

// ─── Conv = Conv2d + BN + SiLU ─────────────────────────────────────────────
//
// `conv_bias` defaults to false (the v8/v11/v26 convention: BN absorbs
// conv bias into β). Set true for v12's `pe` (the depthwise-7×7 positional
// encoding inside AAttn ships with both `pe.conv.bias` AND `pe.bn.*` — BN
// normalises post-bias, so we must keep the bias term to match training).
struct ConvImpl : torch::nn::Module {
  torch::nn::Conv2d      conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  bool                   act_silu = true;
  ConvImpl(int c_in, int c_out, int k = 1, int s = 1, int p = -1, int g = 1,
           bool act = true, bool conv_bias = false);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Conv);

// ─── Bottleneck ────────────────────────────────────────────────────────────
struct BottleneckImpl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  bool add = true;
  BottleneckImpl(int c1, int c2, bool shortcut, int g, double e,
                 std::array<int, 2> k = {3, 3});
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Bottleneck);

// ─── C2f (CSP block, "fast" 2-conv variant) ────────────────────────────────
struct C2fImpl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  torch::nn::ModuleList m{nullptr};
  int  c_inner = 0;
  C2fImpl(int c1, int c2, int n = 1, bool shortcut = false, int g = 1,
          double e = 0.5);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(C2f);

// ─── SPPF (Spatial Pyramid Pooling – Fast) ────────────────────────────────
//
// `cv1_act = true` for v8/v11 (cv1 is Conv→BN→SiLU). v26's SPPF drops
// the activation on cv1 (Identity instead of SiLU); pass false there.
//
// `shortcut = true` enables the residual `out + x` (only active when c1
// == c2). v8/v11 don't use this; v26 does.
struct SPPFImpl : torch::nn::Module {
  Conv cv1{nullptr};
  Conv cv2{nullptr};
  torch::nn::MaxPool2d m{nullptr};
  bool add = false;     // = shortcut && c1 == c2
  SPPFImpl(int c1, int c2, int k = 5, bool cv1_act = true,
           bool shortcut = false);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SPPF);

// ─── DFL projection (4 × reg_max → 4) ──────────────────────────────────────
struct DFLImpl : torch::nn::Module {
  torch::nn::Conv2d conv{nullptr};
  int               c1 = 16;  // reg_max
  explicit DFLImpl(int c1 = 16);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DFL);

// ─── Detect head ──────────────────────────────────────────────────────────
//
// Outputs differ in train vs eval:
//   train (return_features=true):  std::vector<Tensor>     length=nl,
//                                  shape [N, no, h_i, w_i] for each level i.
//   eval (return_features=false):  Tensor [N, 4 + nc, A]
//                                  where A = sum(h_i * w_i),
//                                  boxes are decoded to xyxy in input pixels,
//                                  cls is sigmoided.
//
struct DetectImpl : torch::nn::Module {
  int  nc       = 80;
  int  reg_max  = 16;
  int  nl       = 3;
  int  no       = 0;
  // legacy=true: cv3 = Conv→Conv→Conv2d (the v3/v5/v8/v9 form).
  // legacy=false: cv3 = (DWConv→Conv)→(DWConv→Conv)→Conv2d (the v11 form).
  // cv2 (box regression) is identical in both forms.
  bool legacy   = true;
  std::vector<int>     ch;          // per-level input channels
  std::vector<double>  stride;      // per-level stride (set by parent)
  torch::nn::ModuleList cv2{nullptr};   // regression branches (Sequential)
  torch::nn::ModuleList cv3{nullptr};   // classification branches (Sequential)
  DFL                  dfl{nullptr};

  DetectImpl(int nc, std::vector<int> ch, bool legacy = true);
  std::vector<torch::Tensor> forward_features(std::vector<torch::Tensor> x);
  // Returns concat-form decoded prediction. Caller passes per-level strides.
  torch::Tensor decode(const std::vector<torch::Tensor>& feats);
};
TORCH_MODULE(Detect);

// Depthwise conv (= Conv with groups = gcd(c_in, c_out); typically c_in==c_out).
// Used by v11's Detect cv3 branch and other newer YOLO heads.
struct DWConvImpl : torch::nn::Module {
  torch::nn::Conv2d      conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  bool                   act_silu = true;
  DWConvImpl(int c_in, int c_out, int k = 1, int s = 1, bool act = true);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DWConv);

// DWConvBlock = (DWConv 3×3) → (Conv 1×1). Registered children are named
// "0" and "1" so when this is pushed into another Sequential at index i, the
// resulting state_dict path becomes <prefix>.<i>.{0,1}.{conv,bn}.<...> —
// matching Ultralytics' v11 Detect cv3 nesting exactly.
//
// libtorch's nn::Sequential cannot directly hold another nn::Sequential
// (its forward is templated and that breaks AnyModule). This helper has a
// concrete forward(Tensor)→Tensor so it can.
struct DWConvBlockImpl : torch::nn::Module {
  DWConv dw{nullptr};
  Conv   pw{nullptr};
  DWConvBlockImpl(int c_in, int c_out, int k_dw = 3, bool act = true);
  torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DWConvBlock);

// ─── Whole model ───────────────────────────────────────────────────────────
//
// Order of registered child modules MATCHES Ultralytics yolo8.yaml indices.
// This is the contract that lets the .pt state_dict map cleanly:
//   children:
//     model.0  Conv
//     model.1  Conv
//     model.2  C2f
//     ...
//     model.22 Detect
//
struct Yolo8DetectImpl : torch::nn::Module {
  Yolo8Scale scale;
  int         nc;

  // Registered as "model" with index suffixes (0..22) inside it.
  torch::nn::ModuleList model{nullptr};

  // Strides (computed by a one-shot probe) — used by Detect.
  std::vector<double> stride;

  Yolo8DetectImpl(Yolo8Scale s, int nc);

  // Train mode: returns per-level raw feature maps (for loss).
  // Eval  mode: returns decoded [N, 4+nc, A] tensor.
  std::vector<torch::Tensor> forward_train(torch::Tensor x);
  torch::Tensor              forward_eval(torch::Tensor x);

  // Returns flat parameter+buffer name list in Ultralytics state_dict order.
  // Used by the .pt loader for positional mapping.
  std::vector<std::string>          state_keys() const;
  std::vector<torch::Tensor*>       state_tensors();

  // Copy weights from a StateDict-shaped map. Tensors are cast to the
  // model's existing dtype (typically float32).
  // Returns the number of tensors copied. Throws on shape mismatch.
  int load_from_state_dict(
      const std::vector<std::pair<std::string, at::Tensor>>& entries);
};
TORCH_MODULE(Yolo8Detect);

// Helper: scaled output channels (rounds, clamps to max_channels, * width).
int scale_channels(int c, const Yolo8Scale& s);
// Helper: scaled depth (rounds, clamps to ≥ 1).
int scale_depth(int n, const Yolo8Scale& s);

}  // namespace yolocpp::models
