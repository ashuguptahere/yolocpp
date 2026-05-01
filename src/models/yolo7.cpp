#include "yolocpp/models/yolo7.hpp"

#include <stdexcept>

namespace yolocpp::models {

// ─── ConvSiLU ────────────────────────────────────────────────────────────
namespace {
thread_local bool g_v7_use_leaky = false;
}
V7ActScope::V7ActScope(bool use_leaky) : prev(g_v7_use_leaky) {
  g_v7_use_leaky = use_leaky;
}
V7ActScope::~V7ActScope() { g_v7_use_leaky = prev; }

ConvSiLUImpl::ConvSiLUImpl(int c_in, int c_out, int k, int s, int p, int g) {
  if (p < 0) p = k / 2;
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, k)
                            .stride(s).padding(p).groups(g).bias(false)));
  bn = register_module(
      "bn", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c_out).eps(1e-3)));
  use_leaky = g_v7_use_leaky;   // captured at construction
}
torch::Tensor ConvSiLUImpl::forward(torch::Tensor x) {
  auto y = bn(conv(x));
  return use_leaky
             ? torch::leaky_relu(y, /*negative_slope=*/0.1)
             : torch::silu(y);
}

// ─── MP ──────────────────────────────────────────────────────────────────
MPImpl::MPImpl() {
  m = register_module("m",
                      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
}
torch::Tensor MPImpl::forward(torch::Tensor x) { return m(x); }

// ─── SPPCSPC ─────────────────────────────────────────────────────────────
//
// Layout (yolov7-base, c_out=512, c_=c_out//2=256):
//   cv1: c_in → 256 (1×1)
//   cv2: c_in → 256 (1×1)             — CSP shortcut path
//   cv3: 256 → 256 (3×3)
//   cv4: 256 → 256 (1×1)
//   m5/m9/m13: 5×5/9×9/13×13 maxpool
//   cv5: 4*256=1024 → 256 (1×1)
//   cv6: 256 → 256 (3×3)
//   cv7: 2*256=512 → 512 (1×1)        cat(cv6_out, cv2_out)
SPPCSPCImpl::SPPCSPCImpl(int c_in, int c_out) {
  // Upstream: c_ = int(2 * c_out * e) with default e=0.5, → c_ == c_out.
  const int c_ = c_out;
  cv1 = register_module("cv1", ConvSiLU(c_in, c_, 1, 1));
  cv2 = register_module("cv2", ConvSiLU(c_in, c_, 1, 1));
  cv3 = register_module("cv3", ConvSiLU(c_, c_, 3, 1));
  cv4 = register_module("cv4", ConvSiLU(c_, c_, 1, 1));
  cv5 = register_module("cv5", ConvSiLU(4 * c_, c_, 1, 1));
  cv6 = register_module("cv6", ConvSiLU(c_, c_, 3, 1));
  cv7 = register_module("cv7", ConvSiLU(2 * c_, c_out, 1, 1));
  m5  = register_module("m1", torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(5).stride(1).padding(2)));
  m9  = register_module("m2", torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(9).stride(1).padding(4)));
  m13 = register_module("m3", torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(13).stride(1).padding(6)));
}
torch::Tensor SPPCSPCImpl::forward(torch::Tensor x) {
  auto a = cv4(cv3(cv1(x)));
  auto y = cv6(cv5(torch::cat({a, m5(a), m9(a), m13(a)}, /*dim=*/1)));
  auto b = cv2(x);
  return cv7(torch::cat({y, b}, /*dim=*/1));
}

// ─── Yolo7RepConv (deploy) ────────────────────────────────────────────────────
Yolo7RepConvImpl::Yolo7RepConvImpl(int c_in, int c_out, int k, int s) {
  const int p = k / 2;
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, k)
                            .stride(s).padding(p).bias(true)));
}
torch::Tensor Yolo7RepConvImpl::forward(torch::Tensor x) {
  return torch::silu(conv(x));
}

// ─── Shortcut ────────────────────────────────────────────────────────────
torch::Tensor Yolo7ShortcutImpl::forward(const std::vector<torch::Tensor>& xs) {
  TORCH_CHECK(xs.size() == 2, "Yolo7Shortcut expects 2 inputs");
  return xs[0] + xs[1];
}

// ─── DownC ───────────────────────────────────────────────────────────────
DownCImpl::DownCImpl(int c1, int c2, int k) {
  // Upstream layout:
  //   cv1: 1×1 c1 → c_ (= c1)        — main path stage 1
  //   cv2: 3×3 stride k, c_ → c2/2   — main path stage 2 (downsamples)
  //   cv3: 1×1 c1 → c2/2             — shortcut after maxpool
  //   mp:  MaxPool stride k
  cv1 = register_module("cv1", ConvSiLU(c1, c1,     /*k=*/1, /*s=*/1));
  cv2 = register_module("cv2", ConvSiLU(c1, c2 / 2, /*k=*/3, /*s=*/k));
  cv3 = register_module("cv3", ConvSiLU(c1, c2 / 2, /*k=*/1, /*s=*/1));
  mp  = register_module("mp",  torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(k).stride(k)));
}
torch::Tensor DownCImpl::forward(torch::Tensor x) {
  auto a = cv2(cv1(x));        // 1×1 → 3×3 stride 2
  auto b = cv3(mp(x));         // maxpool stride 2 → 1×1
  return torch::cat({a, b}, /*dim=*/1);
}

// ─── ReOrg ───────────────────────────────────────────────────────────────
torch::Tensor ReOrgImpl::forward(torch::Tensor x) {
  // 4× spatial-to-depth: [B, C, H, W] → [B, 4*C, H/2, W/2]. Same as
  // PyTorch's pixel_unshuffle(downscale_factor=2). No params.
  return torch::pixel_unshuffle(x, /*downscale_factor=*/2);
}

// ─── IDetect (deploy) ────────────────────────────────────────────────────
IDetectImpl::IDetectImpl(int nc_, const std::vector<int>& ch) : nc(nc_) {
  no = 5 + nc;
  nl = (int)ch.size();          // 3 for base/tiny/x; 4 for w6/e6/d6/e6e
  m  = register_module("m", torch::nn::ModuleList());
  for (size_t i = 0; i < ch.size(); ++i) {
    m->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(ch[i], na * no, 1)));
  }
  // Initialise anchors with the right shape for nl. v7-base anchors at
  // 3 levels are used as a default; the converter overwrites with the
  // upstream values (P3-P5 for base/tiny/x; P3-P6 for w6/e6/d6/e6e).
  anchors = register_buffer(
      "anchors", torch::ones({nl, na, 2}, torch::kFloat32));
  anchor_grid = register_buffer(
      "anchor_grid", torch::ones({nl, 1, na, 1, 1, 2}, torch::kFloat32));
  stride.resize(nl);
  for (int i = 0; i < nl; ++i) stride[i] = std::pow(2.0, 3 + i);
}
torch::Tensor IDetectImpl::forward_eval(const std::vector<torch::Tensor>& feats) {
  std::vector<torch::Tensor> z;
  z.reserve(nl);
  auto dev = feats[0].device();
  for (int i = 0; i < nl; ++i) {
    auto x = m[i]->as<torch::nn::Conv2dImpl>()->forward(feats[i]);     // [B, 3*(5+nc), H, W]
    int B = (int)x.size(0);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    x = x.view({B, na, no, H, W}).permute({0, 1, 3, 4, 2}).contiguous();  // [B, na, H, W, no]

    auto y = torch::sigmoid(x);

    // grid: [1, 1, H, W, 2]  in cell units (without +0.5; v5+ uses cell-corner)
    auto gy = torch::arange(H, torch::TensorOptions().device(dev).dtype(y.dtype())).view({1, 1, H, 1});
    auto gx = torch::arange(W, torch::TensorOptions().device(dev).dtype(y.dtype())).view({1, 1, 1, W});
    auto grid_x = gx.expand({1, 1, H, W});
    auto grid_y = gy.expand({1, 1, H, W});
    auto grid = torch::stack({grid_x, grid_y}, -1).unsqueeze(1);  // [1,1,1,H,W,2]
    grid = grid.view({1, 1, H, W, 2});

    auto xy = (y.index({"...", torch::indexing::Slice(0, 2)}) * 2.0f - 0.5f + grid)
                  * (float)stride[i];                              // [B, na, H, W, 2]
    auto wh_a = anchor_grid[i].view({1, na, 1, 1, 2}).to(y.device()).to(y.dtype());
    auto wh = (y.index({"...", torch::indexing::Slice(2, 4)}) * 2.0f).pow(2) * wh_a;

    auto obj = y.index({"...", torch::indexing::Slice(4, 5)});         // [B, na, H, W, 1]
    auto cls = y.index({"...", torch::indexing::Slice(5, 5 + nc)});    // [B, na, H, W, nc]
    auto score = cls * obj;

    // xywh → xyxy
    auto x1 = xy.index({"...", 0}) - 0.5f * wh.index({"...", 0});
    auto y1 = xy.index({"...", 1}) - 0.5f * wh.index({"...", 1});
    auto x2 = xy.index({"...", 0}) + 0.5f * wh.index({"...", 0});
    auto y2 = xy.index({"...", 1}) + 0.5f * wh.index({"...", 1});
    auto box = torch::stack({x1, y1, x2, y2}, /*dim=*/-1);             // [B, na, H, W, 4]
    auto packed = torch::cat({box, score}, /*dim=*/-1);                // [B, na, H, W, 4+nc]
    packed = packed.view({B, na * H * W, 4 + nc});
    z.push_back(packed);
  }
  // [B, A, 4+nc] → [B, 4+nc, A]
  auto out = torch::cat(z, /*dim=*/1).permute({0, 2, 1}).contiguous();
  return out;
}

// ─── yolov7 yaml-walker ──────────────────────────────────────────────────
namespace {
// Private alias of the public Yolo7Spec (declared in yolo7.hpp). Keeps
// the in-file usages compact and lets `yolo7_yaml_for` return the same
// type directly.
using Spec = yolocpp::models::Yolo7Spec;

const std::vector<Spec>& v7_yaml() {
  static const std::vector<Spec> y = {
      // Backbone (0..50)
      {{-1}, "Conv", {32, 3, 1}},                          // 0
      {{-1}, "Conv", {64, 3, 2}},                          // 1 P1/2
      {{-1}, "Conv", {64, 3, 1}},                          // 2
      {{-1}, "Conv", {128, 3, 2}},                         // 3 P2/4
      {{-1}, "Conv", {64, 1, 1}},                          // 4
      {{-2}, "Conv", {64, 1, 1}},                          // 5
      {{-1}, "Conv", {64, 3, 1}},                          // 6
      {{-1}, "Conv", {64, 3, 1}},                          // 7
      {{-1}, "Conv", {64, 3, 1}},                          // 8
      {{-1}, "Conv", {64, 3, 1}},                          // 9
      {{-1, -3, -5, -6}, "Concat", {}},                    // 10
      {{-1}, "Conv", {256, 1, 1}},                         // 11
      {{-1}, "MP",   {}},                                  // 12
      {{-1}, "Conv", {128, 1, 1}},                         // 13
      {{-3}, "Conv", {128, 1, 1}},                         // 14
      {{-1}, "Conv", {128, 3, 2}},                         // 15
      {{-1, -3}, "Concat", {}},                            // 16 P3/8
      {{-1}, "Conv", {128, 1, 1}},                         // 17
      {{-2}, "Conv", {128, 1, 1}},                         // 18
      {{-1}, "Conv", {128, 3, 1}},                         // 19
      {{-1}, "Conv", {128, 3, 1}},                         // 20
      {{-1}, "Conv", {128, 3, 1}},                         // 21
      {{-1}, "Conv", {128, 3, 1}},                         // 22
      {{-1, -3, -5, -6}, "Concat", {}},                    // 23
      {{-1}, "Conv", {512, 1, 1}},                         // 24
      {{-1}, "MP",   {}},                                  // 25
      {{-1}, "Conv", {256, 1, 1}},                         // 26
      {{-3}, "Conv", {256, 1, 1}},                         // 27
      {{-1}, "Conv", {256, 3, 2}},                         // 28
      {{-1, -3}, "Concat", {}},                            // 29 P4/16
      {{-1}, "Conv", {256, 1, 1}},                         // 30
      {{-2}, "Conv", {256, 1, 1}},                         // 31
      {{-1}, "Conv", {256, 3, 1}},                         // 32
      {{-1}, "Conv", {256, 3, 1}},                         // 33
      {{-1}, "Conv", {256, 3, 1}},                         // 34
      {{-1}, "Conv", {256, 3, 1}},                         // 35
      {{-1, -3, -5, -6}, "Concat", {}},                    // 36
      {{-1}, "Conv", {1024, 1, 1}},                        // 37
      {{-1}, "MP",   {}},                                  // 38
      {{-1}, "Conv", {512, 1, 1}},                         // 39
      {{-3}, "Conv", {512, 1, 1}},                         // 40
      {{-1}, "Conv", {512, 3, 2}},                         // 41
      {{-1, -3}, "Concat", {}},                            // 42 P5/32
      {{-1}, "Conv", {256, 1, 1}},                         // 43
      {{-2}, "Conv", {256, 1, 1}},                         // 44
      {{-1}, "Conv", {256, 3, 1}},                         // 45
      {{-1}, "Conv", {256, 3, 1}},                         // 46
      {{-1}, "Conv", {256, 3, 1}},                         // 47
      {{-1}, "Conv", {256, 3, 1}},                         // 48
      {{-1, -3, -5, -6}, "Concat", {}},                    // 49
      {{-1}, "Conv", {1024, 1, 1}},                        // 50

      // Head (51..104)
      {{-1}, "SPPCSPC", {512}},                            // 51
      {{-1}, "Conv", {256, 1, 1}},                         // 52
      {{-1}, "Upsample", {2}},                             // 53
      {{37}, "Conv", {256, 1, 1}},                         // 54
      {{-1, -2}, "Concat", {}},                            // 55
      {{-1}, "Conv", {256, 1, 1}},                         // 56
      {{-2}, "Conv", {256, 1, 1}},                         // 57
      {{-1}, "Conv", {128, 3, 1}},                         // 58
      {{-1}, "Conv", {128, 3, 1}},                         // 59
      {{-1}, "Conv", {128, 3, 1}},                         // 60
      {{-1}, "Conv", {128, 3, 1}},                         // 61
      {{-1, -2, -3, -4, -5, -6}, "Concat", {}},            // 62
      {{-1}, "Conv", {256, 1, 1}},                         // 63
      {{-1}, "Conv", {128, 1, 1}},                         // 64
      {{-1}, "Upsample", {2}},                             // 65
      {{24}, "Conv", {128, 1, 1}},                         // 66
      {{-1, -2}, "Concat", {}},                            // 67
      {{-1}, "Conv", {128, 1, 1}},                         // 68
      {{-2}, "Conv", {128, 1, 1}},                         // 69
      {{-1}, "Conv", {64, 3, 1}},                          // 70
      {{-1}, "Conv", {64, 3, 1}},                          // 71
      {{-1}, "Conv", {64, 3, 1}},                          // 72
      {{-1}, "Conv", {64, 3, 1}},                          // 73
      {{-1, -2, -3, -4, -5, -6}, "Concat", {}},            // 74
      {{-1}, "Conv", {128, 1, 1}},                         // 75 → P3
      {{-1}, "MP",   {}},                                  // 76
      {{-1}, "Conv", {128, 1, 1}},                         // 77
      {{-3}, "Conv", {128, 1, 1}},                         // 78
      {{-1}, "Conv", {128, 3, 2}},                         // 79
      {{-1, -3, 63}, "Concat", {}},                        // 80
      {{-1}, "Conv", {256, 1, 1}},                         // 81
      {{-2}, "Conv", {256, 1, 1}},                         // 82
      {{-1}, "Conv", {128, 3, 1}},                         // 83
      {{-1}, "Conv", {128, 3, 1}},                         // 84
      {{-1}, "Conv", {128, 3, 1}},                         // 85
      {{-1}, "Conv", {128, 3, 1}},                         // 86
      {{-1, -2, -3, -4, -5, -6}, "Concat", {}},            // 87
      {{-1}, "Conv", {256, 1, 1}},                         // 88 → P4
      {{-1}, "MP",   {}},                                  // 89
      {{-1}, "Conv", {256, 1, 1}},                         // 90
      {{-3}, "Conv", {256, 1, 1}},                         // 91
      {{-1}, "Conv", {256, 3, 2}},                         // 92
      {{-1, -3, 51}, "Concat", {}},                        // 93
      {{-1}, "Conv", {512, 1, 1}},                         // 94
      {{-2}, "Conv", {512, 1, 1}},                         // 95
      {{-1}, "Conv", {256, 3, 1}},                         // 96
      {{-1}, "Conv", {256, 3, 1}},                         // 97
      {{-1}, "Conv", {256, 3, 1}},                         // 98
      {{-1}, "Conv", {256, 3, 1}},                         // 99
      {{-1, -2, -3, -4, -5, -6}, "Concat", {}},            // 100
      {{-1}, "Conv", {512, 1, 1}},                         // 101 → P5
      {{75},  "Yolo7RepConv", {256,  3, 1}},                    // 102
      {{88},  "Yolo7RepConv", {512,  3, 1}},                    // 103
      {{101}, "Yolo7RepConv", {1024, 3, 1}},                    // 104
      {{102, 103, 104}, "IDetect", {}},                    // 105
  };
  return y;
}
const std::vector<Spec>& v7_tiny_yaml() {
  static const std::vector<Spec> y = {
      // Backbone (LeakyReLU 0.1 throughout, set via V7ActScope)
      {{-1}, "Conv", {32, 3, 2}},                          // 0  P1/2
      {{-1}, "Conv", {64, 3, 2}},                          // 1  P2/4
      {{-1}, "Conv", {32, 1, 1}},                          // 2
      {{-2}, "Conv", {32, 1, 1}},                          // 3
      {{-1}, "Conv", {32, 3, 1}},                          // 4
      {{-1}, "Conv", {32, 3, 1}},                          // 5
      {{-1, -2, -3, -4}, "Concat", {}},                    // 6
      {{-1}, "Conv", {64, 1, 1}},                          // 7
      {{-1}, "MP",   {}},                                  // 8  P3/8
      {{-1}, "Conv", {64, 1, 1}},                          // 9
      {{-2}, "Conv", {64, 1, 1}},                          // 10
      {{-1}, "Conv", {64, 3, 1}},                          // 11
      {{-1}, "Conv", {64, 3, 1}},                          // 12
      {{-1, -2, -3, -4}, "Concat", {}},                    // 13
      {{-1}, "Conv", {128, 1, 1}},                         // 14
      {{-1}, "MP",   {}},                                  // 15 P4/16
      {{-1}, "Conv", {128, 1, 1}},                         // 16
      {{-2}, "Conv", {128, 1, 1}},                         // 17
      {{-1}, "Conv", {128, 3, 1}},                         // 18
      {{-1}, "Conv", {128, 3, 1}},                         // 19
      {{-1, -2, -3, -4}, "Concat", {}},                    // 20
      {{-1}, "Conv", {256, 1, 1}},                         // 21
      {{-1}, "MP",   {}},                                  // 22 P5/32
      {{-1}, "Conv", {256, 1, 1}},                         // 23
      {{-2}, "Conv", {256, 1, 1}},                         // 24
      {{-1}, "Conv", {256, 3, 1}},                         // 25
      {{-1}, "Conv", {256, 3, 1}},                         // 26
      {{-1, -2, -3, -4}, "Concat", {}},                    // 27
      {{-1}, "Conv", {512, 1, 1}},                         // 28
      // SPP block (no SPPCSPC — direct cv1/cv2 split + 3 SPs + cat + Conv)
      {{-1}, "Conv", {256, 1, 1}},                         // 29
      {{-2}, "Conv", {256, 1, 1}},                         // 30
      {{-1}, "SP",   {5}},                                 // 31
      {{-2}, "SP",   {9}},                                 // 32
      {{-3}, "SP",   {13}},                                // 33
      {{-1, -2, -3, -4}, "Concat", {}},                    // 34
      {{-1}, "Conv", {256, 1, 1}},                         // 35
      {{-1, -7}, "Concat", {}},                            // 36
      {{-1}, "Conv", {256, 1, 1}},                         // 37
      // Top-down P5 → P4
      {{-1}, "Conv", {128, 1, 1}},                         // 38
      {{-1}, "Upsample", {2}},                             // 39
      {{21}, "Conv", {128, 1, 1}},                         // 40
      {{-1, -2}, "Concat", {}},                            // 41
      {{-1}, "Conv", {64, 1, 1}},                          // 42
      {{-2}, "Conv", {64, 1, 1}},                          // 43
      {{-1}, "Conv", {64, 3, 1}},                          // 44
      {{-1}, "Conv", {64, 3, 1}},                          // 45
      {{-1, -2, -3, -4}, "Concat", {}},                    // 46
      {{-1}, "Conv", {128, 1, 1}},                         // 47
      // Top-down P4 → P3
      {{-1}, "Conv", {64, 1, 1}},                          // 48
      {{-1}, "Upsample", {2}},                             // 49
      {{14}, "Conv", {64, 1, 1}},                          // 50
      {{-1, -2}, "Concat", {}},                            // 51
      {{-1}, "Conv", {32, 1, 1}},                          // 52
      {{-2}, "Conv", {32, 1, 1}},                          // 53
      {{-1}, "Conv", {32, 3, 1}},                          // 54
      {{-1}, "Conv", {32, 3, 1}},                          // 55
      {{-1, -2, -3, -4}, "Concat", {}},                    // 56
      {{-1}, "Conv", {64, 1, 1}},                          // 57  → P3
      // Bottom-up P3 → P4
      {{-1}, "Conv", {128, 3, 2}},                         // 58
      {{-1, 47}, "Concat", {}},                            // 59
      {{-1}, "Conv", {64, 1, 1}},                          // 60
      {{-2}, "Conv", {64, 1, 1}},                          // 61
      {{-1}, "Conv", {64, 3, 1}},                          // 62
      {{-1}, "Conv", {64, 3, 1}},                          // 63
      {{-1, -2, -3, -4}, "Concat", {}},                    // 64
      {{-1}, "Conv", {128, 1, 1}},                         // 65  → P4
      // Bottom-up P4 → P5
      {{-1}, "Conv", {256, 3, 2}},                         // 66
      {{-1, 37}, "Concat", {}},                            // 67
      {{-1}, "Conv", {128, 1, 1}},                         // 68
      {{-2}, "Conv", {128, 1, 1}},                         // 69
      {{-1}, "Conv", {128, 3, 1}},                         // 70
      {{-1}, "Conv", {128, 3, 1}},                         // 71
      {{-1, -2, -3, -4}, "Concat", {}},                    // 72
      {{-1}, "Conv", {256, 1, 1}},                         // 73  → P5
      // Pre-IDetect Conv (NOT RepConv — tiny uses plain Conv)
      {{57}, "Conv", {128, 3, 1}},                         // 74
      {{65}, "Conv", {256, 3, 1}},                         // 75
      {{73}, "Conv", {512, 3, 1}},                         // 76
      {{74, 75, 76}, "IDetect", {}},                       // 77
  };
  return y;
}
const std::vector<Spec>& v7x_yaml() {
  // yolov7x — wider channels (40 stem) and longer ELAN cat (-1,-3,-5,-7,-8).
  static const std::vector<Spec> y = {
      {{-1}, "Conv", {40, 3, 1}},                          // 0
      {{-1}, "Conv", {80, 3, 2}},                          // 1
      {{-1}, "Conv", {80, 3, 1}},                          // 2
      {{-1}, "Conv", {160, 3, 2}},                         // 3 P2/4
      {{-1}, "Conv", {64, 1, 1}},                          // 4
      {{-2}, "Conv", {64, 1, 1}},                          // 5
      {{-1}, "Conv", {64, 3, 1}},                          // 6
      {{-1}, "Conv", {64, 3, 1}},                          // 7
      {{-1}, "Conv", {64, 3, 1}},                          // 8
      {{-1}, "Conv", {64, 3, 1}},                          // 9
      {{-1}, "Conv", {64, 3, 1}},                          // 10
      {{-1}, "Conv", {64, 3, 1}},                          // 11
      {{-1, -3, -5, -7, -8}, "Concat", {}},                // 12
      {{-1}, "Conv", {320, 1, 1}},                         // 13
      // MP-down to P3
      {{-1}, "MP",   {}},                                  // 14
      {{-1}, "Conv", {160, 1, 1}},                         // 15
      {{-3}, "Conv", {160, 1, 1}},                         // 16
      {{-1}, "Conv", {160, 3, 2}},                         // 17
      {{-1, -3}, "Concat", {}},                            // 18 P3/8
      // ELAN at P3
      {{-1}, "Conv", {128, 1, 1}},                         // 19
      {{-2}, "Conv", {128, 1, 1}},                         // 20
      {{-1}, "Conv", {128, 3, 1}},                         // 21
      {{-1}, "Conv", {128, 3, 1}},                         // 22
      {{-1}, "Conv", {128, 3, 1}},                         // 23
      {{-1}, "Conv", {128, 3, 1}},                         // 24
      {{-1}, "Conv", {128, 3, 1}},                         // 25
      {{-1}, "Conv", {128, 3, 1}},                         // 26
      {{-1, -3, -5, -7, -8}, "Concat", {}},                // 27
      {{-1}, "Conv", {640, 1, 1}},                         // 28
      // MP-down to P4
      {{-1}, "MP",   {}},                                  // 29
      {{-1}, "Conv", {320, 1, 1}},                         // 30
      {{-3}, "Conv", {320, 1, 1}},                         // 31
      {{-1}, "Conv", {320, 3, 2}},                         // 32
      {{-1, -3}, "Concat", {}},                            // 33 P4/16
      // ELAN at P4
      {{-1}, "Conv", {256, 1, 1}},                         // 34
      {{-2}, "Conv", {256, 1, 1}},                         // 35
      {{-1}, "Conv", {256, 3, 1}},                         // 36
      {{-1}, "Conv", {256, 3, 1}},                         // 37
      {{-1}, "Conv", {256, 3, 1}},                         // 38
      {{-1}, "Conv", {256, 3, 1}},                         // 39
      {{-1}, "Conv", {256, 3, 1}},                         // 40
      {{-1}, "Conv", {256, 3, 1}},                         // 41
      {{-1, -3, -5, -7, -8}, "Concat", {}},                // 42
      {{-1}, "Conv", {1280, 1, 1}},                        // 43
      // MP-down to P5
      {{-1}, "MP",   {}},                                  // 44
      {{-1}, "Conv", {640, 1, 1}},                         // 45
      {{-3}, "Conv", {640, 1, 1}},                         // 46
      {{-1}, "Conv", {640, 3, 2}},                         // 47
      {{-1, -3}, "Concat", {}},                            // 48 P5/32
      // ELAN at P5
      {{-1}, "Conv", {256, 1, 1}},                         // 49
      {{-2}, "Conv", {256, 1, 1}},                         // 50
      {{-1}, "Conv", {256, 3, 1}},                         // 51
      {{-1}, "Conv", {256, 3, 1}},                         // 52
      {{-1}, "Conv", {256, 3, 1}},                         // 53
      {{-1}, "Conv", {256, 3, 1}},                         // 54
      {{-1}, "Conv", {256, 3, 1}},                         // 55
      {{-1}, "Conv", {256, 3, 1}},                         // 56
      {{-1, -3, -5, -7, -8}, "Concat", {}},                // 57
      {{-1}, "Conv", {1280, 1, 1}},                        // 58
      // SPPCSPC
      {{-1}, "SPPCSPC", {640}},                            // 59
      // Top-down P5 → P4 (ELAN-W-style with 8 lanes)
      {{-1}, "Conv", {320, 1, 1}},                         // 60
      {{-1}, "Upsample", {2}},                             // 61
      {{43}, "Conv", {320, 1, 1}},                         // 62 from backbone P4
      {{-1, -2}, "Concat", {}},                            // 63
      {{-1}, "Conv", {256, 1, 1}},                         // 64
      {{-2}, "Conv", {256, 1, 1}},                         // 65
      {{-1}, "Conv", {256, 3, 1}},                         // 66
      {{-1}, "Conv", {256, 3, 1}},                         // 67
      {{-1}, "Conv", {256, 3, 1}},                         // 68
      {{-1}, "Conv", {256, 3, 1}},                         // 69
      {{-1}, "Conv", {256, 3, 1}},                         // 70
      {{-1}, "Conv", {256, 3, 1}},                         // 71
      {{-1, -3, -5, -7, -8}, "Concat", {}},    // 72
      {{-1}, "Conv", {320, 1, 1}},                         // 73
      // Top-down P4 → P3
      {{-1}, "Conv", {160, 1, 1}},                         // 74
      {{-1}, "Upsample", {2}},                             // 75
      {{28}, "Conv", {160, 1, 1}},                         // 76 from backbone P3
      {{-1, -2}, "Concat", {}},                            // 77
      {{-1}, "Conv", {128, 1, 1}},                         // 78
      {{-2}, "Conv", {128, 1, 1}},                         // 79
      {{-1}, "Conv", {128, 3, 1}},                         // 80
      {{-1}, "Conv", {128, 3, 1}},                         // 81
      {{-1}, "Conv", {128, 3, 1}},                         // 82
      {{-1}, "Conv", {128, 3, 1}},                         // 83
      {{-1}, "Conv", {128, 3, 1}},                         // 84
      {{-1}, "Conv", {128, 3, 1}},                         // 85
      {{-1, -3, -5, -7, -8}, "Concat", {}},    // 86
      {{-1}, "Conv", {160, 1, 1}},                         // 87 → P3
      // Bottom-up P3 → P4
      {{-1}, "MP",   {}},                                  // 88
      {{-1}, "Conv", {160, 1, 1}},                         // 89
      {{-3}, "Conv", {160, 1, 1}},                         // 90
      {{-1}, "Conv", {160, 3, 2}},                         // 91
      {{-1, -3, 73}, "Concat", {}},                        // 92
      {{-1}, "Conv", {256, 1, 1}},                         // 93
      {{-2}, "Conv", {256, 1, 1}},                         // 94
      {{-1}, "Conv", {256, 3, 1}},                         // 95
      {{-1}, "Conv", {256, 3, 1}},                         // 96
      {{-1}, "Conv", {256, 3, 1}},                         // 97
      {{-1}, "Conv", {256, 3, 1}},                         // 98
      {{-1}, "Conv", {256, 3, 1}},                         // 99
      {{-1}, "Conv", {256, 3, 1}},                         // 100
      {{-1, -3, -5, -7, -8}, "Concat", {}},    // 101
      {{-1}, "Conv", {320, 1, 1}},                         // 102 → P4
      // Bottom-up P4 → P5
      {{-1}, "MP",   {}},                                  // 103
      {{-1}, "Conv", {320, 1, 1}},                         // 104
      {{-3}, "Conv", {320, 1, 1}},                         // 105
      {{-1}, "Conv", {320, 3, 2}},                         // 106
      {{-1, -3, 59}, "Concat", {}},                        // 107
      {{-1}, "Conv", {512, 1, 1}},                         // 108
      {{-2}, "Conv", {512, 1, 1}},                         // 109
      {{-1}, "Conv", {512, 3, 1}},                         // 110
      {{-1}, "Conv", {512, 3, 1}},                         // 111
      {{-1}, "Conv", {512, 3, 1}},                         // 112
      {{-1}, "Conv", {512, 3, 1}},                         // 113
      {{-1}, "Conv", {512, 3, 1}},                         // 114
      {{-1}, "Conv", {512, 3, 1}},                         // 115
      {{-1, -3, -5, -7, -8}, "Concat", {}},    // 116
      {{-1}, "Conv", {640, 1, 1}},                         // 117 → P5
      // Pre-IDetect — yolov7x uses plain Conv (Conv+BN+SiLU), NOT
      // RepConv. The published yolov7x.pt has `.conv.weight + .bn.*`
      // keys at 118/119/120, not `.conv.weight + .conv.bias`.
      {{87},  "Conv", {320,  3, 1}},                       // 118
      {{102}, "Conv", {640,  3, 1}},                       // 119
      {{117}, "Conv", {1280, 3, 1}},                       // 120
      {{118, 119, 120}, "IDetect", {}},                    // 121
  };
  return y;
}
const std::vector<Spec>& v7_w6_yaml() {
  // yolov7-w6 deploy form: ReOrg input + P6 backbone (4 stages
  // P3/P4/P5/P6) + 4-level IDetect. The auxiliary head's pre-conv
  // layers (118-121 in the train yaml) are dropped.
  static const std::vector<Spec> y = {
      // Backbone (0..46)
      {{-1}, "ReOrg", {}},                                   // 0
      {{-1}, "Conv",  {64, 3, 1}},                           // 1  P1/2
      {{-1}, "Conv",  {128, 3, 2}},                          // 2  P2/4
      {{-1}, "Conv",  {64, 1, 1}},                           // 3
      {{-2}, "Conv",  {64, 1, 1}},                           // 4
      {{-1}, "Conv",  {64, 3, 1}},                           // 5
      {{-1}, "Conv",  {64, 3, 1}},                           // 6
      {{-1}, "Conv",  {64, 3, 1}},                           // 7
      {{-1}, "Conv",  {64, 3, 1}},                           // 8
      {{-1, -3, -5, -6}, "Concat", {}},                      // 9
      {{-1}, "Conv",  {128, 1, 1}},                          // 10
      {{-1}, "Conv",  {256, 3, 2}},                          // 11 P3/8
      {{-1}, "Conv",  {128, 1, 1}},                          // 12
      {{-2}, "Conv",  {128, 1, 1}},                          // 13
      {{-1}, "Conv",  {128, 3, 1}},                          // 14
      {{-1}, "Conv",  {128, 3, 1}},                          // 15
      {{-1}, "Conv",  {128, 3, 1}},                          // 16
      {{-1}, "Conv",  {128, 3, 1}},                          // 17
      {{-1, -3, -5, -6}, "Concat", {}},                      // 18
      {{-1}, "Conv",  {256, 1, 1}},                          // 19
      {{-1}, "Conv",  {512, 3, 2}},                          // 20 P4/16
      {{-1}, "Conv",  {256, 1, 1}},                          // 21
      {{-2}, "Conv",  {256, 1, 1}},                          // 22
      {{-1}, "Conv",  {256, 3, 1}},                          // 23
      {{-1}, "Conv",  {256, 3, 1}},                          // 24
      {{-1}, "Conv",  {256, 3, 1}},                          // 25
      {{-1}, "Conv",  {256, 3, 1}},                          // 26
      {{-1, -3, -5, -6}, "Concat", {}},                      // 27
      {{-1}, "Conv",  {512, 1, 1}},                          // 28
      {{-1}, "Conv",  {768, 3, 2}},                          // 29 P5/32
      {{-1}, "Conv",  {384, 1, 1}},                          // 30
      {{-2}, "Conv",  {384, 1, 1}},                          // 31
      {{-1}, "Conv",  {384, 3, 1}},                          // 32
      {{-1}, "Conv",  {384, 3, 1}},                          // 33
      {{-1}, "Conv",  {384, 3, 1}},                          // 34
      {{-1}, "Conv",  {384, 3, 1}},                          // 35
      {{-1, -3, -5, -6}, "Concat", {}},                      // 36
      {{-1}, "Conv",  {768, 1, 1}},                          // 37
      {{-1}, "Conv",  {1024, 3, 2}},                         // 38 P6/64
      {{-1}, "Conv",  {512, 1, 1}},                          // 39
      {{-2}, "Conv",  {512, 1, 1}},                          // 40
      {{-1}, "Conv",  {512, 3, 1}},                          // 41
      {{-1}, "Conv",  {512, 3, 1}},                          // 42
      {{-1}, "Conv",  {512, 3, 1}},                          // 43
      {{-1}, "Conv",  {512, 3, 1}},                          // 44
      {{-1, -3, -5, -6}, "Concat", {}},                      // 45
      {{-1}, "Conv",  {1024, 1, 1}},                         // 46
      // Head — top-down chain
      {{-1}, "SPPCSPC", {512}},                              // 47
      {{-1}, "Conv",  {384, 1, 1}},                          // 48
      {{-1}, "Upsample", {2}},                               // 49
      {{37}, "Conv",  {384, 1, 1}},                          // 50  route P5
      {{-1, -2}, "Concat", {}},                              // 51
      {{-1}, "Conv",  {384, 1, 1}},                          // 52
      {{-2}, "Conv",  {384, 1, 1}},                          // 53
      {{-1}, "Conv",  {192, 3, 1}},                          // 54
      {{-1}, "Conv",  {192, 3, 1}},                          // 55
      {{-1}, "Conv",  {192, 3, 1}},                          // 56
      {{-1}, "Conv",  {192, 3, 1}},                          // 57
      {{-1, -2, -3, -4, -5, -6}, "Concat", {}},              // 58
      {{-1}, "Conv",  {384, 1, 1}},                          // 59
      {{-1}, "Conv",  {256, 1, 1}},                          // 60
      {{-1}, "Upsample", {2}},                               // 61
      {{28}, "Conv",  {256, 1, 1}},                          // 62  route P4
      {{-1, -2}, "Concat", {}},                              // 63
      {{-1}, "Conv",  {256, 1, 1}},                          // 64
      {{-2}, "Conv",  {256, 1, 1}},                          // 65
      {{-1}, "Conv",  {128, 3, 1}},                          // 66
      {{-1}, "Conv",  {128, 3, 1}},                          // 67
      {{-1}, "Conv",  {128, 3, 1}},                          // 68
      {{-1}, "Conv",  {128, 3, 1}},                          // 69
      {{-1, -2, -3, -4, -5, -6}, "Concat", {}},              // 70
      {{-1}, "Conv",  {256, 1, 1}},                          // 71
      {{-1}, "Conv",  {128, 1, 1}},                          // 72
      {{-1}, "Upsample", {2}},                               // 73
      {{19}, "Conv",  {128, 1, 1}},                          // 74  route P3
      {{-1, -2}, "Concat", {}},                              // 75
      {{-1}, "Conv",  {128, 1, 1}},                          // 76
      {{-2}, "Conv",  {128, 1, 1}},                          // 77
      {{-1}, "Conv",  {64, 3, 1}},                           // 78
      {{-1}, "Conv",  {64, 3, 1}},                           // 79
      {{-1}, "Conv",  {64, 3, 1}},                           // 80
      {{-1}, "Conv",  {64, 3, 1}},                           // 81
      {{-1, -2, -3, -4, -5, -6}, "Concat", {}},              // 82
      {{-1}, "Conv",  {128, 1, 1}},                          // 83  P3 head
      // Bottom-up
      {{-1}, "Conv",  {256, 3, 2}},                          // 84
      {{-1, 71}, "Concat", {}},                              // 85
      {{-1}, "Conv",  {256, 1, 1}},                          // 86
      {{-2}, "Conv",  {256, 1, 1}},                          // 87
      {{-1}, "Conv",  {128, 3, 1}},                          // 88
      {{-1}, "Conv",  {128, 3, 1}},                          // 89
      {{-1}, "Conv",  {128, 3, 1}},                          // 90
      {{-1}, "Conv",  {128, 3, 1}},                          // 91
      {{-1, -2, -3, -4, -5, -6}, "Concat", {}},              // 92
      {{-1}, "Conv",  {256, 1, 1}},                          // 93  P4 head
      {{-1}, "Conv",  {384, 3, 2}},                          // 94
      {{-1, 59}, "Concat", {}},                              // 95
      {{-1}, "Conv",  {384, 1, 1}},                          // 96
      {{-2}, "Conv",  {384, 1, 1}},                          // 97
      {{-1}, "Conv",  {192, 3, 1}},                          // 98
      {{-1}, "Conv",  {192, 3, 1}},                          // 99
      {{-1}, "Conv",  {192, 3, 1}},                          // 100
      {{-1}, "Conv",  {192, 3, 1}},                          // 101
      {{-1, -2, -3, -4, -5, -6}, "Concat", {}},              // 102
      {{-1}, "Conv",  {384, 1, 1}},                          // 103 P5 head
      {{-1}, "Conv",  {512, 3, 2}},                          // 104
      {{-1, 47}, "Concat", {}},                              // 105
      {{-1}, "Conv",  {512, 1, 1}},                          // 106
      {{-2}, "Conv",  {512, 1, 1}},                          // 107
      {{-1}, "Conv",  {256, 3, 1}},                          // 108
      {{-1}, "Conv",  {256, 3, 1}},                          // 109
      {{-1}, "Conv",  {256, 3, 1}},                          // 110
      {{-1}, "Conv",  {256, 3, 1}},                          // 111
      {{-1, -2, -3, -4, -5, -6}, "Concat", {}},              // 112
      {{-1}, "Conv",  {512, 1, 1}},                          // 113 P6 head
      // Pre-IDetect Convs (lead head only — auxiliary 118-121 dropped at deploy).
      {{83},  "Conv", {256,  3, 1}},                         // 114
      {{93},  "Conv", {512,  3, 1}},                         // 115
      {{103}, "Conv", {768,  3, 1}},                         // 116
      {{113}, "Conv", {1024, 3, 1}},                         // 117
      {{114, 115, 116, 117}, "IDetect", {}},                 // 118
  };
  return y;
}
const std::vector<Spec>& v7_e6_yaml() {
  // yolov7-e6 deploy form (no IAuxDetect): ReOrg + DownC backbone (P2-P6
  // with 6-element-cat ELAN, 6 inner 3×3 per block) + SPPCSPC + 8-element-
  // cat ELAN-W head + 4-level IDetect. Auxiliary head pre-convs (140-143
  // in train yaml) are dropped at deploy.
  static const std::vector<Spec> y = {
      {{-1}, "ReOrg", {}},                               // 0
      {{-1}, "Conv", {80, 3, 1}},                        // 1  P1/2
      {{-1}, "DownC", {160}},                            // 2  P2/4
      {{-1}, "Conv", {64, 1, 1}},                        // 3
      {{-2}, "Conv", {64, 1, 1}},                        // 4
      {{-1}, "Conv", {64, 3, 1}},                        // 5
      {{-1}, "Conv", {64, 3, 1}},                        // 6
      {{-1}, "Conv", {64, 3, 1}},                        // 7
      {{-1}, "Conv", {64, 3, 1}},                        // 8
      {{-1}, "Conv", {64, 3, 1}},                        // 9
      {{-1}, "Conv", {64, 3, 1}},                        // 10
      {{-1, -3, -5, -7, -8}, "Concat", {}},              // 11
      {{-1}, "Conv", {160, 1, 1}},                       // 12
      {{-1}, "DownC", {320}},                            // 13 P3/8
      {{-1}, "Conv", {128, 1, 1}},                       // 14
      {{-2}, "Conv", {128, 1, 1}},                       // 15
      {{-1}, "Conv", {128, 3, 1}},                       // 16
      {{-1}, "Conv", {128, 3, 1}},                       // 17
      {{-1}, "Conv", {128, 3, 1}},                       // 18
      {{-1}, "Conv", {128, 3, 1}},                       // 19
      {{-1}, "Conv", {128, 3, 1}},                       // 20
      {{-1}, "Conv", {128, 3, 1}},                       // 21
      {{-1, -3, -5, -7, -8}, "Concat", {}},              // 22
      {{-1}, "Conv", {320, 1, 1}},                       // 23
      {{-1}, "DownC", {640}},                            // 24 P4/16
      {{-1}, "Conv", {256, 1, 1}},                       // 25
      {{-2}, "Conv", {256, 1, 1}},                       // 26
      {{-1}, "Conv", {256, 3, 1}},                       // 27
      {{-1}, "Conv", {256, 3, 1}},                       // 28
      {{-1}, "Conv", {256, 3, 1}},                       // 29
      {{-1}, "Conv", {256, 3, 1}},                       // 30
      {{-1}, "Conv", {256, 3, 1}},                       // 31
      {{-1}, "Conv", {256, 3, 1}},                       // 32
      {{-1, -3, -5, -7, -8}, "Concat", {}},              // 33
      {{-1}, "Conv", {640, 1, 1}},                       // 34
      {{-1}, "DownC", {960}},                            // 35 P5/32
      {{-1}, "Conv", {384, 1, 1}},                       // 36
      {{-2}, "Conv", {384, 1, 1}},                       // 37
      {{-1}, "Conv", {384, 3, 1}},                       // 38
      {{-1}, "Conv", {384, 3, 1}},                       // 39
      {{-1}, "Conv", {384, 3, 1}},                       // 40
      {{-1}, "Conv", {384, 3, 1}},                       // 41
      {{-1}, "Conv", {384, 3, 1}},                       // 42
      {{-1}, "Conv", {384, 3, 1}},                       // 43
      {{-1, -3, -5, -7, -8}, "Concat", {}},              // 44
      {{-1}, "Conv", {960, 1, 1}},                       // 45
      {{-1}, "DownC", {1280}},                           // 46 P6/64
      {{-1}, "Conv", {512, 1, 1}},                       // 47
      {{-2}, "Conv", {512, 1, 1}},                       // 48
      {{-1}, "Conv", {512, 3, 1}},                       // 49
      {{-1}, "Conv", {512, 3, 1}},                       // 50
      {{-1}, "Conv", {512, 3, 1}},                       // 51
      {{-1}, "Conv", {512, 3, 1}},                       // 52
      {{-1}, "Conv", {512, 3, 1}},                       // 53
      {{-1}, "Conv", {512, 3, 1}},                       // 54
      {{-1, -3, -5, -7, -8}, "Concat", {}},              // 55
      {{-1}, "Conv", {1280, 1, 1}},                      // 56
      {{-1}, "SPPCSPC", {640}},                          // 57
      // Top-down P6 → P5
      {{-1}, "Conv", {480, 1, 1}},                       // 58
      {{-1}, "Upsample", {2}},                           // 59
      {{45}, "Conv", {480, 1, 1}},                       // 60 route P5
      {{-1, -2}, "Concat", {}},                          // 61
      {{-1}, "Conv", {384, 1, 1}},                       // 62
      {{-2}, "Conv", {384, 1, 1}},                       // 63
      {{-1}, "Conv", {192, 3, 1}},                       // 64
      {{-1}, "Conv", {192, 3, 1}},                       // 65
      {{-1}, "Conv", {192, 3, 1}},                       // 66
      {{-1}, "Conv", {192, 3, 1}},                       // 67
      {{-1}, "Conv", {192, 3, 1}},                       // 68
      {{-1}, "Conv", {192, 3, 1}},                       // 69
      {{-1, -2, -3, -4, -5, -6, -7, -8}, "Concat", {}},  // 70
      {{-1}, "Conv", {480, 1, 1}},                       // 71
      // Top-down P5 → P4
      {{-1}, "Conv", {320, 1, 1}},                       // 72
      {{-1}, "Upsample", {2}},                           // 73
      {{34}, "Conv", {320, 1, 1}},                       // 74 route P4
      {{-1, -2}, "Concat", {}},                          // 75
      {{-1}, "Conv", {256, 1, 1}},                       // 76
      {{-2}, "Conv", {256, 1, 1}},                       // 77
      {{-1}, "Conv", {128, 3, 1}},                       // 78
      {{-1}, "Conv", {128, 3, 1}},                       // 79
      {{-1}, "Conv", {128, 3, 1}},                       // 80
      {{-1}, "Conv", {128, 3, 1}},                       // 81
      {{-1}, "Conv", {128, 3, 1}},                       // 82
      {{-1}, "Conv", {128, 3, 1}},                       // 83
      {{-1, -2, -3, -4, -5, -6, -7, -8}, "Concat", {}},  // 84
      {{-1}, "Conv", {320, 1, 1}},                       // 85
      // Top-down P4 → P3
      {{-1}, "Conv", {160, 1, 1}},                       // 86
      {{-1}, "Upsample", {2}},                           // 87
      {{23}, "Conv", {160, 1, 1}},                       // 88 route P3
      {{-1, -2}, "Concat", {}},                          // 89
      {{-1}, "Conv", {128, 1, 1}},                       // 90
      {{-2}, "Conv", {128, 1, 1}},                       // 91
      {{-1}, "Conv", {64, 3, 1}},                        // 92
      {{-1}, "Conv", {64, 3, 1}},                        // 93
      {{-1}, "Conv", {64, 3, 1}},                        // 94
      {{-1}, "Conv", {64, 3, 1}},                        // 95
      {{-1}, "Conv", {64, 3, 1}},                        // 96
      {{-1}, "Conv", {64, 3, 1}},                        // 97
      {{-1, -2, -3, -4, -5, -6, -7, -8}, "Concat", {}},  // 98
      {{-1}, "Conv", {160, 1, 1}},                       // 99 P3 head
      // Bottom-up P3 → P4
      {{-1}, "DownC", {320}},                            // 100
      {{-1, 85}, "Concat", {}},                          // 101
      {{-1}, "Conv", {256, 1, 1}},                       // 102
      {{-2}, "Conv", {256, 1, 1}},                       // 103
      {{-1}, "Conv", {128, 3, 1}},                       // 104
      {{-1}, "Conv", {128, 3, 1}},                       // 105
      {{-1}, "Conv", {128, 3, 1}},                       // 106
      {{-1}, "Conv", {128, 3, 1}},                       // 107
      {{-1}, "Conv", {128, 3, 1}},                       // 108
      {{-1}, "Conv", {128, 3, 1}},                       // 109
      {{-1, -2, -3, -4, -5, -6, -7, -8}, "Concat", {}},  // 110
      {{-1}, "Conv", {320, 1, 1}},                       // 111 P4 head
      // Bottom-up P4 → P5
      {{-1}, "DownC", {480}},                            // 112
      {{-1, 71}, "Concat", {}},                          // 113
      {{-1}, "Conv", {384, 1, 1}},                       // 114
      {{-2}, "Conv", {384, 1, 1}},                       // 115
      {{-1}, "Conv", {192, 3, 1}},                       // 116
      {{-1}, "Conv", {192, 3, 1}},                       // 117
      {{-1}, "Conv", {192, 3, 1}},                       // 118
      {{-1}, "Conv", {192, 3, 1}},                       // 119
      {{-1}, "Conv", {192, 3, 1}},                       // 120
      {{-1}, "Conv", {192, 3, 1}},                       // 121
      {{-1, -2, -3, -4, -5, -6, -7, -8}, "Concat", {}},  // 122
      {{-1}, "Conv", {480, 1, 1}},                       // 123 P5 head
      // Bottom-up P5 → P6
      {{-1}, "DownC", {640}},                            // 124
      {{-1, 57}, "Concat", {}},                          // 125
      {{-1}, "Conv", {512, 1, 1}},                       // 126
      {{-2}, "Conv", {512, 1, 1}},                       // 127
      {{-1}, "Conv", {256, 3, 1}},                       // 128
      {{-1}, "Conv", {256, 3, 1}},                       // 129
      {{-1}, "Conv", {256, 3, 1}},                       // 130
      {{-1}, "Conv", {256, 3, 1}},                       // 131
      {{-1}, "Conv", {256, 3, 1}},                       // 132
      {{-1}, "Conv", {256, 3, 1}},                       // 133
      {{-1, -2, -3, -4, -5, -6, -7, -8}, "Concat", {}},  // 134
      {{-1}, "Conv", {640, 1, 1}},                       // 135 P6 head
      // Pre-IDetect (lead head — auxiliary 140-143 dropped at deploy)
      {{99},  "Conv", {320,  3, 1}},                     // 136
      {{111}, "Conv", {640,  3, 1}},                     // 137
      {{123}, "Conv", {960,  3, 1}},                     // 138
      {{135}, "Conv", {1280, 3, 1}},                     // 139
      {{136, 137, 138, 139}, "IDetect", {}},             // 140
  };
  return y;
}
const std::vector<Spec>& v7_d6_yaml() {
  // yolov7-d6 deploy form: wider+deeper than e6 (96 stem, 192/384/768/
  // 1152/1536 channel widths, 8 inner 3×3 ELAN with sparse 6-element cat
  // [-1,-3,-5,-7,-9,-10] in backbone, 8 inner 3×3 ELAN-W with 10-element
  // cat in head). Same module set as e6 (DownC, ReOrg, 4-level IDetect).
  // Aux head pre-convs (162-165 in train yaml) dropped at deploy →
  // IDetect at index 162 takes [158, 159, 160, 161].
  static const std::vector<Spec> y = {
      {{-1}, "ReOrg", {}},                                     // 0
      {{-1}, "Conv", {96, 3, 1}},                              // 1  P1/2
      {{-1}, "DownC", {192}},                                  // 2  P2/4
      {{-1}, "Conv", {64, 1, 1}},                              // 3
      {{-2}, "Conv", {64, 1, 1}},                              // 4
      {{-1}, "Conv", {64, 3, 1}},                              // 5
      {{-1}, "Conv", {64, 3, 1}},                              // 6
      {{-1}, "Conv", {64, 3, 1}},                              // 7
      {{-1}, "Conv", {64, 3, 1}},                              // 8
      {{-1}, "Conv", {64, 3, 1}},                              // 9
      {{-1}, "Conv", {64, 3, 1}},                              // 10
      {{-1}, "Conv", {64, 3, 1}},                              // 11
      {{-1}, "Conv", {64, 3, 1}},                              // 12
      {{-1, -3, -5, -7, -9, -10}, "Concat", {}},               // 13
      {{-1}, "Conv", {192, 1, 1}},                             // 14
      {{-1}, "DownC", {384}},                                  // 15 P3/8
      {{-1}, "Conv", {128, 1, 1}},                             // 16
      {{-2}, "Conv", {128, 1, 1}},                             // 17
      {{-1}, "Conv", {128, 3, 1}},                             // 18
      {{-1}, "Conv", {128, 3, 1}},                             // 19
      {{-1}, "Conv", {128, 3, 1}},                             // 20
      {{-1}, "Conv", {128, 3, 1}},                             // 21
      {{-1}, "Conv", {128, 3, 1}},                             // 22
      {{-1}, "Conv", {128, 3, 1}},                             // 23
      {{-1}, "Conv", {128, 3, 1}},                             // 24
      {{-1}, "Conv", {128, 3, 1}},                             // 25
      {{-1, -3, -5, -7, -9, -10}, "Concat", {}},               // 26
      {{-1}, "Conv", {384, 1, 1}},                             // 27
      {{-1}, "DownC", {768}},                                  // 28 P4/16
      {{-1}, "Conv", {256, 1, 1}},                             // 29
      {{-2}, "Conv", {256, 1, 1}},                             // 30
      {{-1}, "Conv", {256, 3, 1}},                             // 31
      {{-1}, "Conv", {256, 3, 1}},                             // 32
      {{-1}, "Conv", {256, 3, 1}},                             // 33
      {{-1}, "Conv", {256, 3, 1}},                             // 34
      {{-1}, "Conv", {256, 3, 1}},                             // 35
      {{-1}, "Conv", {256, 3, 1}},                             // 36
      {{-1}, "Conv", {256, 3, 1}},                             // 37
      {{-1}, "Conv", {256, 3, 1}},                             // 38
      {{-1, -3, -5, -7, -9, -10}, "Concat", {}},               // 39
      {{-1}, "Conv", {768, 1, 1}},                             // 40
      {{-1}, "DownC", {1152}},                                 // 41 P5/32
      {{-1}, "Conv", {384, 1, 1}},                             // 42
      {{-2}, "Conv", {384, 1, 1}},                             // 43
      {{-1}, "Conv", {384, 3, 1}},                             // 44
      {{-1}, "Conv", {384, 3, 1}},                             // 45
      {{-1}, "Conv", {384, 3, 1}},                             // 46
      {{-1}, "Conv", {384, 3, 1}},                             // 47
      {{-1}, "Conv", {384, 3, 1}},                             // 48
      {{-1}, "Conv", {384, 3, 1}},                             // 49
      {{-1}, "Conv", {384, 3, 1}},                             // 50
      {{-1}, "Conv", {384, 3, 1}},                             // 51
      {{-1, -3, -5, -7, -9, -10}, "Concat", {}},               // 52
      {{-1}, "Conv", {1152, 1, 1}},                            // 53
      {{-1}, "DownC", {1536}},                                 // 54 P6/64
      {{-1}, "Conv", {512, 1, 1}},                             // 55
      {{-2}, "Conv", {512, 1, 1}},                             // 56
      {{-1}, "Conv", {512, 3, 1}},                             // 57
      {{-1}, "Conv", {512, 3, 1}},                             // 58
      {{-1}, "Conv", {512, 3, 1}},                             // 59
      {{-1}, "Conv", {512, 3, 1}},                             // 60
      {{-1}, "Conv", {512, 3, 1}},                             // 61
      {{-1}, "Conv", {512, 3, 1}},                             // 62
      {{-1}, "Conv", {512, 3, 1}},                             // 63
      {{-1}, "Conv", {512, 3, 1}},                             // 64
      {{-1, -3, -5, -7, -9, -10}, "Concat", {}},               // 65
      {{-1}, "Conv", {1536, 1, 1}},                            // 66
      {{-1}, "SPPCSPC", {768}},                                // 67
      // Top-down P6 → P5
      {{-1}, "Conv", {576, 1, 1}},                             // 68
      {{-1}, "Upsample", {2}},                                 // 69
      {{53}, "Conv", {576, 1, 1}},                             // 70 route P5
      {{-1, -2}, "Concat", {}},                                // 71
      {{-1}, "Conv", {384, 1, 1}},                             // 72
      {{-2}, "Conv", {384, 1, 1}},                             // 73
      {{-1}, "Conv", {192, 3, 1}},                             // 74
      {{-1}, "Conv", {192, 3, 1}},                             // 75
      {{-1}, "Conv", {192, 3, 1}},                             // 76
      {{-1}, "Conv", {192, 3, 1}},                             // 77
      {{-1}, "Conv", {192, 3, 1}},                             // 78
      {{-1}, "Conv", {192, 3, 1}},                             // 79
      {{-1}, "Conv", {192, 3, 1}},                             // 80
      {{-1}, "Conv", {192, 3, 1}},                             // 81
      {{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}, "Concat", {}}, // 82
      {{-1}, "Conv", {576, 1, 1}},                             // 83
      // Top-down P5 → P4
      {{-1}, "Conv", {384, 1, 1}},                             // 84
      {{-1}, "Upsample", {2}},                                 // 85
      {{40}, "Conv", {384, 1, 1}},                             // 86 route P4
      {{-1, -2}, "Concat", {}},                                // 87
      {{-1}, "Conv", {256, 1, 1}},                             // 88
      {{-2}, "Conv", {256, 1, 1}},                             // 89
      {{-1}, "Conv", {128, 3, 1}},                             // 90
      {{-1}, "Conv", {128, 3, 1}},                             // 91
      {{-1}, "Conv", {128, 3, 1}},                             // 92
      {{-1}, "Conv", {128, 3, 1}},                             // 93
      {{-1}, "Conv", {128, 3, 1}},                             // 94
      {{-1}, "Conv", {128, 3, 1}},                             // 95
      {{-1}, "Conv", {128, 3, 1}},                             // 96
      {{-1}, "Conv", {128, 3, 1}},                             // 97
      {{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}, "Concat", {}}, // 98
      {{-1}, "Conv", {384, 1, 1}},                             // 99
      // Top-down P4 → P3
      {{-1}, "Conv", {192, 1, 1}},                             // 100
      {{-1}, "Upsample", {2}},                                 // 101
      {{27}, "Conv", {192, 1, 1}},                             // 102 route P3
      {{-1, -2}, "Concat", {}},                                // 103
      {{-1}, "Conv", {128, 1, 1}},                             // 104
      {{-2}, "Conv", {128, 1, 1}},                             // 105
      {{-1}, "Conv", {64, 3, 1}},                              // 106
      {{-1}, "Conv", {64, 3, 1}},                              // 107
      {{-1}, "Conv", {64, 3, 1}},                              // 108
      {{-1}, "Conv", {64, 3, 1}},                              // 109
      {{-1}, "Conv", {64, 3, 1}},                              // 110
      {{-1}, "Conv", {64, 3, 1}},                              // 111
      {{-1}, "Conv", {64, 3, 1}},                              // 112
      {{-1}, "Conv", {64, 3, 1}},                              // 113
      {{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}, "Concat", {}}, // 114
      {{-1}, "Conv", {192, 1, 1}},                             // 115 P3 head
      // Bottom-up P3 → P4
      {{-1}, "DownC", {384}},                                  // 116
      {{-1, 99}, "Concat", {}},                                // 117
      {{-1}, "Conv", {256, 1, 1}},                             // 118
      {{-2}, "Conv", {256, 1, 1}},                             // 119
      {{-1}, "Conv", {128, 3, 1}},                             // 120
      {{-1}, "Conv", {128, 3, 1}},                             // 121
      {{-1}, "Conv", {128, 3, 1}},                             // 122
      {{-1}, "Conv", {128, 3, 1}},                             // 123
      {{-1}, "Conv", {128, 3, 1}},                             // 124
      {{-1}, "Conv", {128, 3, 1}},                             // 125
      {{-1}, "Conv", {128, 3, 1}},                             // 126
      {{-1}, "Conv", {128, 3, 1}},                             // 127
      {{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}, "Concat", {}}, // 128
      {{-1}, "Conv", {384, 1, 1}},                             // 129 P4 head
      // Bottom-up P4 → P5
      {{-1}, "DownC", {576}},                                  // 130
      {{-1, 83}, "Concat", {}},                                // 131
      {{-1}, "Conv", {384, 1, 1}},                             // 132
      {{-2}, "Conv", {384, 1, 1}},                             // 133
      {{-1}, "Conv", {192, 3, 1}},                             // 134
      {{-1}, "Conv", {192, 3, 1}},                             // 135
      {{-1}, "Conv", {192, 3, 1}},                             // 136
      {{-1}, "Conv", {192, 3, 1}},                             // 137
      {{-1}, "Conv", {192, 3, 1}},                             // 138
      {{-1}, "Conv", {192, 3, 1}},                             // 139
      {{-1}, "Conv", {192, 3, 1}},                             // 140
      {{-1}, "Conv", {192, 3, 1}},                             // 141
      {{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}, "Concat", {}}, // 142
      {{-1}, "Conv", {576, 1, 1}},                             // 143 P5 head
      // Bottom-up P5 → P6
      {{-1}, "DownC", {768}},                                  // 144
      {{-1, 67}, "Concat", {}},                                // 145
      {{-1}, "Conv", {512, 1, 1}},                             // 146
      {{-2}, "Conv", {512, 1, 1}},                             // 147
      {{-1}, "Conv", {256, 3, 1}},                             // 148
      {{-1}, "Conv", {256, 3, 1}},                             // 149
      {{-1}, "Conv", {256, 3, 1}},                             // 150
      {{-1}, "Conv", {256, 3, 1}},                             // 151
      {{-1}, "Conv", {256, 3, 1}},                             // 152
      {{-1}, "Conv", {256, 3, 1}},                             // 153
      {{-1}, "Conv", {256, 3, 1}},                             // 154
      {{-1}, "Conv", {256, 3, 1}},                             // 155
      {{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}, "Concat", {}}, // 156
      {{-1}, "Conv", {768, 1, 1}},                             // 157 P6 head
      // Pre-IDetect (lead — auxiliary 162-165 in train yaml dropped)
      {{115}, "Conv", {384,  3, 1}},                           // 158
      {{129}, "Conv", {768,  3, 1}},                           // 159
      {{143}, "Conv", {1152, 3, 1}},                           // 160
      {{157}, "Conv", {1536, 3, 1}},                           // 161
      {{158, 159, 160, 161}, "IDetect", {}},                   // 162
  };
  return y;
}
// Helpers for the verbose e6e yaml. Each E-ELAN stage = 1 DownC (or
// route Conv+Upsample+Conv+Concat for top-down, DownC+Concat for
// bottom-up) + ELAN1 (10 entries) + ELAN2 (10 entries, routes back via
// -11/-12 to same input as ELAN1) + 1 Shortcut.
static void e6e_elan_block_b(std::vector<Spec>& y, int c_outer, int c_inner,
                              bool is_first) {
  // ELAN block. is_first=true → cv1 from -1, cv2 from -2 (ELAN1).
  // is_first=false → cv1 from -11, cv2 from -12 (ELAN2 — routes back to
  // same input as ELAN1's cv1). Inner 3×3 channels = c_inner. Cat is
  // 5-element [-1, -3, -5, -7, -8]. Reducer to c_outer.
  y.push_back({is_first ? std::vector<int>{-1} : std::vector<int>{-11},
               "Conv", {c_inner, 1, 1}});
  y.push_back({is_first ? std::vector<int>{-2} : std::vector<int>{-12},
               "Conv", {c_inner, 1, 1}});
  for (int k = 0; k < 6; ++k) y.push_back({{-1}, "Conv", {c_inner, 3, 1}});
  y.push_back({{-1, -3, -5, -7, -8}, "Concat", {}});
  y.push_back({{-1}, "Conv", {c_outer, 1, 1}});
}
static void e6e_elan_block_h(std::vector<Spec>& y, int c_outer, int c_inner_lat,
                              int c_inner_3x3, bool is_first) {
  // Head ELAN-W block. cv1/cv2 produce c_inner_lat ch. Inner 3×3s
  // produce c_inner_3x3 ch each. Cat is 8-element [-1, -2, ..., -8].
  // Reducer to c_outer.
  y.push_back({is_first ? std::vector<int>{-1} : std::vector<int>{-11},
               "Conv", {c_inner_lat, 1, 1}});
  y.push_back({is_first ? std::vector<int>{-2} : std::vector<int>{-12},
               "Conv", {c_inner_lat, 1, 1}});
  for (int k = 0; k < 6; ++k) y.push_back({{-1}, "Conv", {c_inner_3x3, 3, 1}});
  y.push_back({{-1, -2, -3, -4, -5, -6, -7, -8}, "Concat", {}});
  y.push_back({{-1}, "Conv", {c_outer, 1, 1}});
}
static void e6e_bb_stage(std::vector<Spec>& y, int c_outer, int c_inner) {
  y.push_back({{-1}, "DownC", {c_outer}});
  e6e_elan_block_b(y, c_outer, c_inner, /*is_first=*/true);
  e6e_elan_block_b(y, c_outer, c_inner, /*is_first=*/false);
  y.push_back({{-1, -11}, "Yolo7Shortcut", {}});
}
static void e6e_head_td_stage(std::vector<Spec>& y, int c_outer,
                                int c_inner_lat, int c_inner_3x3, int route) {
  y.push_back({{-1}, "Conv", {c_outer, 1, 1}});      // reducer for upsample
  y.push_back({{-1}, "Upsample", {2}});
  y.push_back({{route}, "Conv", {c_outer, 1, 1}});  // backbone lateral
  y.push_back({{-1, -2}, "Concat", {}});
  e6e_elan_block_h(y, c_outer, c_inner_lat, c_inner_3x3, /*is_first=*/true);
  e6e_elan_block_h(y, c_outer, c_inner_lat, c_inner_3x3, /*is_first=*/false);
  y.push_back({{-1, -11}, "Yolo7Shortcut", {}});
}
static void e6e_head_bu_stage(std::vector<Spec>& y, int downc_c, int c_outer,
                                int c_inner_lat, int c_inner_3x3, int cat_idx) {
  y.push_back({{-1}, "DownC", {downc_c}});
  y.push_back({{-1, cat_idx}, "Concat", {}});
  e6e_elan_block_h(y, c_outer, c_inner_lat, c_inner_3x3, /*is_first=*/true);
  e6e_elan_block_h(y, c_outer, c_inner_lat, c_inner_3x3, /*is_first=*/false);
  y.push_back({{-1, -11}, "Yolo7Shortcut", {}});
}

const std::vector<Spec>& v7_e6e_yaml() {
  static const std::vector<Spec> y = ([]() {
    std::vector<Spec> y;
    // Backbone (0..111)
    y.push_back({{-1}, "ReOrg", {}});                      // 0
    y.push_back({{-1}, "Conv", {80, 3, 1}});               // 1
    e6e_bb_stage(y, 160,  64);   // 2..23  (P2/4)
    e6e_bb_stage(y, 320,  128);  // 24..45 (P3/8)
    e6e_bb_stage(y, 640,  256);  // 46..67 (P4/16)
    e6e_bb_stage(y, 960,  384);  // 68..89 (P5/32)
    e6e_bb_stage(y, 1280, 512);  // 90..111 (P6/64)
    // SPPCSPC + head (112..256)
    y.push_back({{-1}, "SPPCSPC", {640}});                 // 112
    e6e_head_td_stage(y, 480, 384, 192, /*route=*/89);     // 113..137 P6→P5
    e6e_head_td_stage(y, 320, 256, 128, /*route=*/67);     // 138..162 P5→P4
    e6e_head_td_stage(y, 160, 128, 64,  /*route=*/45);     // 163..187 P4→P3 (P3 head at 187)
    e6e_head_bu_stage(y, 320, 320, 256, 128, /*cat=*/162); // 188..210 P3→P4 (P4 head at 210)
    e6e_head_bu_stage(y, 480, 480, 384, 192, /*cat=*/137); // 211..233 P4→P5 (P5 head at 233)
    e6e_head_bu_stage(y, 640, 640, 512, 256, /*cat=*/112); // 234..256 P5→P6 (P6 head at 256)
    // Pre-IDetect (lead head only — auxiliary 261-264 in train yaml dropped)
    y.push_back({{187}, "Conv", {320,  3, 1}});            // 257
    y.push_back({{210}, "Conv", {640,  3, 1}});            // 258
    y.push_back({{233}, "Conv", {960,  3, 1}});            // 259
    y.push_back({{256}, "Conv", {1280, 3, 1}});            // 260
    y.push_back({{257, 258, 259, 260}, "IDetect", {}});    // 261
    return y;
  })();
  return y;
}
const std::vector<Spec>& v7_yaml_for(yolocpp::models::Yolo7Scale s) {
  using yolocpp::models::Yolo7Scale;
  switch (s) {
    case Yolo7Scale::Tiny: return v7_tiny_yaml();
    case Yolo7Scale::X:    return v7x_yaml();
    case Yolo7Scale::W6:   return v7_w6_yaml();
    case Yolo7Scale::E6:   return v7_e6_yaml();
    case Yolo7Scale::D6:   return v7_d6_yaml();
    case Yolo7Scale::E6e:  return v7_e6e_yaml();
    case Yolo7Scale::Base:
    default:               return v7_yaml();
  }
}
}  // namespace

const std::vector<Yolo7Spec>& yolo7_yaml_for(Yolo7Scale s) {
  return v7_yaml_for(s);
}

Yolo7Impl::Yolo7Impl(int nc_, Yolo7Scale s) : scale(s), nc(nc_) {
  // Tiny uses LeakyReLU(0.1); base/x use SiLU. Push the activation
  // scope BEFORE registering any ConvSiLU children so they all capture
  // the right activation at construction.
  V7ActScope act_scope(scale == Yolo7Scale::Tiny);
  model = register_module("model", torch::nn::ModuleList());
  const auto& yaml = v7_yaml_for(scale);
  std::vector<int> ch;
  const int c_in_img = 3;

  auto resolve_idx = [](int f, int i) { return f < 0 ? i + f : f; };
  auto in_ch_for = [&](size_t i) -> int {
    const auto& s = yaml[i];
    if (s.kind == "Concat") {
      int sum = 0;
      for (int f : s.from) {
        int idx = resolve_idx(f, (int)i);
        sum += (idx == -1) ? c_in_img : ch[idx];
      }
      return sum;
    }
    int f = s.from[0];
    int idx = resolve_idx(f, (int)i);
    return (idx == -1) ? c_in_img : ch[idx];
  };

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    int in_ch = in_ch_for(i);

    if (s.kind == "Conv") {
      int c_out = s.args[0], k = s.args[1], st = s.args[2];
      model->push_back(ConvSiLU(in_ch, c_out, k, st));
      ch.push_back(c_out);
    } else if (s.kind == "MP") {
      model->push_back(MP());
      ch.push_back(in_ch);
    } else if (s.kind == "ReOrg") {
      model->push_back(ReOrg());
      ch.push_back(in_ch * 4);   // 2×2 spatial-to-depth quadruples channels
    } else if (s.kind == "DownC") {
      // args=[c2]. Two-path strided downsample (k=2).
      model->push_back(DownC(in_ch, s.args[0], 2));
      ch.push_back(s.args[0]);
    } else if (s.kind == "Yolo7Shortcut") {
      // No params; just element-wise sum of two inputs. Output ch = the
      // first input's ch (both inputs must match).
      model->push_back(Yolo7Shortcut());
      int first_idx = resolve_idx(s.from[0], (int)i);
      ch.push_back(first_idx == -1 ? c_in_img : ch[first_idx]);
    } else if (s.kind == "SP") {
      // Single MaxPool with kernel=k, stride=1, padding=k/2 (used by
      // yolov7-tiny's SPP-equivalent block).
      int k = s.args[0];
      model->push_back(torch::nn::MaxPool2d(
          torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2)));
      ch.push_back(in_ch);
    } else if (s.kind == "Concat") {
      model->push_back(torch::nn::Identity());
      ch.push_back(in_ch);
    } else if (s.kind == "Upsample") {
      double sf = (double)s.args[0];
      model->push_back(torch::nn::Upsample(
          torch::nn::UpsampleOptions()
              .scale_factor(std::vector<double>{sf, sf})
              .mode(torch::kNearest)));
      ch.push_back(in_ch);
    } else if (s.kind == "SPPCSPC") {
      int c_out = s.args[0];
      model->push_back(SPPCSPC(in_ch, c_out));
      ch.push_back(c_out);
    } else if (s.kind == "Yolo7RepConv") {
      int c_out = s.args[0], k = s.args[1], st = s.args[2];
      model->push_back(Yolo7RepConv(in_ch, c_out, k, st));
      ch.push_back(c_out);
    } else if (s.kind == "IDetect") {
      std::vector<int> det_ch;
      for (int f : s.from) det_ch.push_back(ch[f]);
      model->push_back(IDetect(nc, det_ch));
      ch.push_back(0);
    } else {
      throw std::runtime_error("yolo7: unknown layer kind '" + s.kind + "'");
    }
  }
}

std::vector<torch::Tensor> Yolo7Impl::forward_features(torch::Tensor x) {
  const auto& yaml = v7_yaml_for(scale);
  std::vector<torch::Tensor> outs(yaml.size());
  auto resolve_idx = [](int f, int i) { return f < 0 ? i + f : f; };

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    torch::Tensor in;
    std::vector<torch::Tensor> in_list;
    if (s.kind == "Concat") {
      std::vector<torch::Tensor> parts;
      for (int f : s.from) {
        int idx = resolve_idx(f, (int)i);
        parts.push_back(idx == -1 ? x : outs[idx]);
      }
      in = torch::cat(parts, /*dim=*/1);
    } else if (s.kind == "Yolo7Shortcut") {
      for (int f : s.from) {
        int idx = resolve_idx(f, (int)i);
        in_list.push_back(idx == -1 ? x : outs[idx]);
      }
    } else {
      int f = s.from[0];
      int idx = resolve_idx(f, (int)i);
      in = (idx == -1) ? x : outs[idx];
    }

    if (s.kind == "Conv")    outs[i] = model[i]->as<ConvSiLUImpl>()->forward(in);
    else if (s.kind == "MP") outs[i] = model[i]->as<MPImpl>()->forward(in);
    else if (s.kind == "ReOrg") outs[i] = model[i]->as<ReOrgImpl>()->forward(in);
    else if (s.kind == "DownC") outs[i] = model[i]->as<DownCImpl>()->forward(in);
    else if (s.kind == "Yolo7Shortcut") outs[i] = model[i]->as<Yolo7ShortcutImpl>()->forward(in_list);
    else if (s.kind == "SP") outs[i] = model[i]->as<torch::nn::MaxPool2dImpl>()->forward(in);
    else if (s.kind == "Concat")   outs[i] = in;
    else if (s.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "SPPCSPC")  outs[i] = model[i]->as<SPPCSPCImpl>()->forward(in);
    else if (s.kind == "Yolo7RepConv")  outs[i] = model[i]->as<Yolo7RepConvImpl>()->forward(in);
    else if (s.kind == "IDetect") {
      // Returned via separate path in forward_eval — placeholder here.
      outs[i] = torch::Tensor();
    }
  }
  // Pull pre-IDetect feats from whichever layer indices the yaml's IDetect
  // points at (base: 102/103/104; tiny: 74/75/76; x: 118/119/120).
  size_t det_idx = yaml.size() - 1;
  std::vector<torch::Tensor> det_in;
  for (int f : yaml[det_idx].from) det_in.push_back(outs[f]);
  return det_in;
}

std::vector<torch::Tensor> Yolo7Impl::forward_train(torch::Tensor x) {
  // Pre-IDetect features (per-scale, in stride-ascending order P3→P5
  // for base/tiny/x; P3→P6 for w6/e6/d6/e6e).
  auto feats = forward_features(x);
  const auto& yaml = v7_yaml_for(scale);
  size_t det_idx = yaml.size() - 1;
  auto* d = model[det_idx]->as<IDetectImpl>();
  if (stride.empty()) {
    int img_h = (int)x.size(2);
    for (auto& t : feats) stride.push_back((double)img_h / (double)t.size(2));
    d->stride = stride;
  }
  std::vector<torch::Tensor> out;
  out.reserve(feats.size());
  for (size_t i = 0; i < feats.size(); ++i) {
    auto raw = d->m[i]->as<torch::nn::Conv2dImpl>()->forward(feats[i]);
    out.push_back(raw);
  }
  return out;
}

torch::Tensor Yolo7Impl::forward_eval(torch::Tensor x) {
  auto feats = forward_features(x);
  const auto& yaml = v7_yaml_for(scale);
  size_t det_idx = yaml.size() - 1;
  if (stride.empty()) {
    int img_h = (int)x.size(2);
    for (auto& t : feats) stride.push_back((double)img_h / (double)t.size(2));
    auto* d = model[det_idx]->as<IDetectImpl>();
    d->stride = stride;
  }
  return model[det_idx]->as<IDetectImpl>()->forward_eval(feats);
}

int Yolo7Impl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto params  = this->named_parameters(true);
  auto buffers = this->named_buffers(true);
  int n = 0;
  for (const auto& e : entries) {
    if (auto* p = params.find(e.first)) {
      if (p->sizes() != e.second.sizes()) continue;
      torch::NoGradGuard ng;
      p->copy_(e.second.to(p->device(), p->dtype()));
      ++n;
    } else if (auto* b = buffers.find(e.first)) {
      if (b->sizes() != e.second.sizes()) continue;
      torch::NoGradGuard ng;
      b->copy_(e.second.to(b->device(), b->dtype()));
      ++n;
    }
  }
  return n;
}

}  // namespace yolocpp::models
