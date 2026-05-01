#include "yolocpp/models/yolo6.hpp"

#include <algorithm>
#include <cmath>

namespace yolocpp::models {

namespace {
inline int make_div(double v, int d = 8) {
  int x = static_cast<int>(std::ceil(v / d) * d);
  return std::max(d, x);
}
inline int dep(int n, double mul) {
  return std::max(1, static_cast<int>(std::round(n * mul)));
}
}  // namespace

// ─── ConvBNReLU ──────────────────────────────────────────────────────────
namespace {
thread_local bool g_v6_use_silu = false;
}
V6ActScope::V6ActScope(bool use_silu) : prev(g_v6_use_silu) {
  g_v6_use_silu = use_silu;
}
V6ActScope::~V6ActScope() { g_v6_use_silu = prev; }

ConvBNReLUImpl::ConvBNReLUImpl(int c_in, int c_out, int k, int s, int p, int g) {
  if (p < 0) p = k / 2;
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, k)
                            .stride(s).padding(p).groups(g).bias(false)));
  // Meituan's published v6 weights are saved with `BatchNorm2d.eps = 1e-3`
  // (overridden via Meituan's training/export script — verified: every
  // saved v6 .pt has eps=0.001 across n/s/m/l/{n6,s6,m6,l6} P5+P6). Using
  // PyTorch's 1e-5 default would silently drift through the deeper l6
  // chain (5 backbone stages + 6 neck stages with BepC3 BottleReps),
  // saturating cls outputs to near-zero on bus.jpg.
  bn = register_module(
      "bn", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(c_out).eps(1e-3)));
  use_silu = g_v6_use_silu;   // captured at construction
}
torch::Tensor ConvBNReLUImpl::forward(torch::Tensor x) {
  auto y = bn(conv(x));
  return use_silu ? torch::silu(y) : torch::relu(y);
}

// ─── RepConv (deploy form) ───────────────────────────────────────────────
RepConvImpl::RepConvImpl(int c_in, int c_out, int k, int s) {
  const int p = k / 2;
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, k)
                            .stride(s).padding(p).bias(true)));
}
torch::Tensor RepConvImpl::forward(torch::Tensor x) {
  return torch::relu(conv(x));
}

// ─── RepBlock — conv1 + (n-1)×block ─────────────────────────────────────
RepBlockImpl::RepBlockImpl(int c_in, int c_out, int n) {
  conv1 = register_module("conv1", RepConv(c_in, c_out, 3, 1));
  block = register_module("block", torch::nn::ModuleList());
  for (int i = 1; i < n; ++i) block->push_back(RepConv(c_out, c_out, 3, 1));
}
torch::Tensor RepBlockImpl::forward(torch::Tensor x) {
  x = conv1(x);
  for (size_t i = 0; i < block->size(); ++i)
    x = block[i]->as<RepConvImpl>()->forward(x);
  return x;
}

// ─── BottleRep ───────────────────────────────────────────────────────────
BottleRepImpl::BottleRepImpl(int c_in, int c_out, bool use_rep) {
  use_repconv = use_rep;
  add         = (c_in == c_out);
  if (use_repconv) {
    rep_conv1 = register_module("conv1", RepConv(c_in,  c_out, 3, 1));
    rep_conv2 = register_module("conv2", RepConv(c_out, c_out, 3, 1));
  } else {
    cbr_conv1 = register_module("conv1", ConvBNReLU(c_in,  c_out, 3, 1));
    cbr_conv2 = register_module("conv2", ConvBNReLU(c_out, c_out, 3, 1));
  }
  // Always register `alpha` — upstream v6m/l have `weight=True` so this
  // shows up in their state_dicts. Initialise to 1.0 so the shortcut is
  // an identity at construction time (matches upstream init).
  alpha = register_parameter("alpha", torch::ones({1}));
}
torch::Tensor BottleRepImpl::forward(torch::Tensor x) {
  torch::Tensor y;
  if (use_repconv) y = rep_conv2(rep_conv1(x));
  else             y = cbr_conv2(cbr_conv1(x));
  return add ? (y + alpha * x) : y;
}

// ─── BottleRep3 — three-conv variant used by MBLABlock ──────────────────
BottleRep3Impl::BottleRep3Impl(int c_in, int c_out) {
  conv1 = register_module("conv1", ConvBNReLU(c_in,  c_out, 3, 1));
  conv2 = register_module("conv2", ConvBNReLU(c_out, c_out, 3, 1));
  conv3 = register_module("conv3", ConvBNReLU(c_out, c_out, 3, 1));
  add   = (c_in == c_out);
  alpha = register_parameter("alpha", torch::ones({1}));
}
torch::Tensor BottleRep3Impl::forward(torch::Tensor x) {
  auto y = conv3(conv2(conv1(x)));
  return add ? (y + alpha * x) : y;
}

// ─── MBLABlock — Multi-Branch Linear Activation Block ────────────────────
//
// Structure mirrors Meituan v6 layers/common.py:653 exactly. The branch
// list `n_list` is computed from the depth-scaled `n` upstream:
//
//   internal_n = max(n // 2, 1)
//   if internal_n == 1: n_list = [0, 1]
//   else:               n_list = [0, ⌊largest pow-2 < internal_n⌋, internal_n]
//
// cv1 outputs `branch_num * c_` channels, split equally into `branch_num`
// chunks. cv2 takes (sum(n_list) + branch_num) * c_ channels — i.e. the
// initial `branch_num` chunks PLUS one intermediate output per
// BottleRep3 inside each Sequential in m.
MBLABlockImpl::MBLABlockImpl(int c_in, int c_out, int n, double e) {
  c_inner = (int)(c_out * e);
  // Compute n_list per upstream's branching logic.
  int internal_n = std::max(1, n / 2);
  if (internal_n == 1) {
    n_list = {0, 1};
  } else {
    int extra_branch_steps = 1;
    while (extra_branch_steps * 2 < internal_n) extra_branch_steps *= 2;
    n_list = {0, extra_branch_steps, internal_n};
  }
  branch_num = (int)n_list.size();

  int sum_n = 0;
  for (int v : n_list) sum_n += v;

  cv1 = register_module("cv1", ConvBNReLU(c_in,
                                          branch_num * c_inner,
                                          /*k=*/1, /*s=*/1));
  cv2 = register_module("cv2", ConvBNReLU((sum_n + branch_num) * c_inner,
                                          c_out, /*k=*/1, /*s=*/1));

  // m is a ModuleList of (branch_num - 1) Sequentials. Each Sequential
  // holds `n_list[i+1]` BottleRep3s (so the first Sequential has 1 entry
  // when internal_n>=1; for internal_n>=2 the second Sequential has
  // internal_n entries).
  m = register_module("m", torch::nn::ModuleList());
  for (int i = 0; i < branch_num - 1; ++i) {
    auto seq = torch::nn::Sequential();
    for (int k = 0; k < n_list[i + 1]; ++k) {
      seq->push_back(BottleRep3(c_inner, c_inner));
    }
    m->push_back(seq);
  }
}
torch::Tensor MBLABlockImpl::forward(torch::Tensor x) {
  auto y = cv1(x);
  // Split y into branch_num chunks of c_inner channels each.
  std::vector<torch::Tensor> chunks;
  chunks.reserve(branch_num);
  for (int i = 0; i < branch_num; ++i) {
    chunks.push_back(y.slice(/*dim=*/1, i * c_inner, (i + 1) * c_inner));
  }

  std::vector<torch::Tensor> all_y;
  all_y.reserve((size_t)branch_num + 8);
  all_y.push_back(chunks[0]);
  for (int j = 0; j < branch_num - 1; ++j) {
    auto* seq = m[j]->as<torch::nn::SequentialImpl>();
    auto cur  = chunks[j + 1];
    all_y.push_back(cur);
    // Run each BottleRep3 in sequence, capturing the intermediate.
    for (size_t k = 0; k < seq->size(); ++k) {
      cur = seq->ptr(k)->as<BottleRep3Impl>()->forward(cur);
      all_y.push_back(cur);
    }
  }
  auto cat = torch::cat(all_y, /*dim=*/1);
  return cv2(cat);
}

// ─── RepBlockBR — RepBlock with BottleRep inner blocks ──────────────────
RepBlockBRImpl::RepBlockBRImpl(int c_in, int c_out, int n, bool use_rep) {
  use_repconv = use_rep;
  conv1 = register_module("conv1", BottleRep(c_in, c_out, use_rep));
  block = register_module("block", torch::nn::ModuleList());
  for (int i = 1; i < n; ++i) block->push_back(BottleRep(c_out, c_out, use_rep));
}
torch::Tensor RepBlockBRImpl::forward(torch::Tensor x) {
  x = conv1(x);
  for (size_t i = 0; i < block->size(); ++i)
    x = block[i]->as<BottleRepImpl>()->forward(x);
  return x;
}

// ─── BepC3 ───────────────────────────────────────────────────────────────
BepC3Impl::BepC3Impl(int c_in, int c_out, int n, bool use_rep, double e) {
  const int c_ = static_cast<int>(c_out * e);
  cv1 = register_module("cv1", ConvBNReLU(c_in, c_, 1, 1));
  cv2 = register_module("cv2", ConvBNReLU(c_in, c_, 1, 1));
  cv3 = register_module("cv3", ConvBNReLU(2 * c_, c_out, 1, 1));
  m   = register_module("m",   RepBlockBR(c_, c_, n, use_rep));
}
torch::Tensor BepC3Impl::forward(torch::Tensor x) {
  return cv3(torch::cat({m(cv1(x)), cv2(x)}, /*dim=*/1));
}

// ─── CSPSPPF ─────────────────────────────────────────────────────────────
//
// Channel widths (yolov6s, c_in=512, c_out=512): inner c_=256.
//   cv1: 512 → 256 (1×1)         main path (CSP "right")
//   cv2: 512 → 256 (1×1)         shortcut path (CSP "left")
//   cv3: 256 → 256 (3×3)
//   cv4: 256 → 256 (1×1)
//   m  : MaxPool 5×5 stride 1  ×3 chained (5,9,13 effective)
//   cv5: 1024 → 256 (1×1)        cat(y, m1, m2, m3) → 256
//   cv6: 256 → 256 (3×3)
//   cv7: 512 → 512 (1×1)         cat(cv6_out, cv2_out)
CSPSPPFImpl::CSPSPPFImpl(int c_in, int c_out, int k) {
  const int c_ = c_in / 2;
  cv1 = register_module("cv1", ConvBNReLU(c_in, c_, 1, 1));
  cv2 = register_module("cv2", ConvBNReLU(c_in, c_, 1, 1));
  cv3 = register_module("cv3", ConvBNReLU(c_, c_, 3, 1));
  cv4 = register_module("cv4", ConvBNReLU(c_, c_, 1, 1));
  cv5 = register_module("cv5", ConvBNReLU(4 * c_, c_, 1, 1));
  cv6 = register_module("cv6", ConvBNReLU(c_, c_, 3, 1));
  cv7 = register_module("cv7", ConvBNReLU(2 * c_, c_out, 1, 1));
  m   = register_module("m", torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2)));
}
torch::Tensor CSPSPPFImpl::forward(torch::Tensor x) {
  auto y = cv4(cv3(cv1(x)));
  auto y1 = m(y);
  auto y2 = m(y1);
  auto y3 = m(y2);
  auto out_main  = cv6(cv5(torch::cat({y, y1, y2, y3}, /*dim=*/1)));
  auto out_short = cv2(x);
  // Upstream order: cv7(cat([cv2(x), cv6_path])) — shortcut FIRST. The
  // first c_ channels of cv7's weight expect cv2_out, the next c_ expect
  // the main-path output.
  return cv7(torch::cat({out_short, out_main}, /*dim=*/1));
}

// ─── SPPFModule (Meituan-style SimSPPF inner) ────────────────────────────
SPPFModuleImpl::SPPFModuleImpl(int c_in, int c_out, int k) {
  const int c_ = c_in / 2;
  cv1 = register_module("cv1", ConvBNReLU(c_in,    c_,     1, 1));
  cv2 = register_module("cv2", ConvBNReLU(4 * c_,  c_out,  1, 1));
  m   = register_module("m",   torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2)));
}
torch::Tensor SPPFModuleImpl::forward(torch::Tensor x) {
  x = cv1(x);
  auto y1 = m(x);
  auto y2 = m(y1);
  auto y3 = m(y2);
  return cv2(torch::cat({x, y1, y2, y3}, /*dim=*/1));
}
SimSPPFImpl::SimSPPFImpl(int c_in, int c_out, int k) {
  sppf = register_module("sppf", SPPFModule(c_in, c_out, k));
}
torch::Tensor SimSPPFImpl::forward(torch::Tensor x) { return sppf(x); }

// ─── Transpose (ConvTranspose2d wrapped) ─────────────────────────────────
TransposeImpl::TransposeImpl(int c_in, int c_out) {
  upsample_transpose = register_module(
      "upsample_transpose",
      torch::nn::ConvTranspose2d(
          torch::nn::ConvTranspose2dOptions(c_in, c_out, 2)
              .stride(2).bias(true)));
}
torch::Tensor TransposeImpl::forward(torch::Tensor x) {
  return upsample_transpose(x);
}

// ─── BiFusionBlock ───────────────────────────────────────────────────────
//
// Constructor takes (c_high, c_lat, c_mid, c_out) where:
//   c_high : ch of the higher-stride lateral (cv1 input)
//   c_lat  : ch of the same-/lower-stride lateral (cv2 input)
//   c_mid  : the working channel count for upsample/downsample/output
//   c_out  : output ch (= c_mid in v6 baseline)
// Forward takes (x_high, x_lat, x_lower_reduced) where:
//   x_high  : feature at this scale, higher channel (e.g. P4 256 ch for Bifusion0)
//   x_lat   : feature at the FINER resolution (will be downsampled by 2)
//   x_lower : feature at the COARSER resolution, already reduced (will be upsampled)
BiFusionBlockImpl::BiFusionBlockImpl(int c_high, int c_lat, int c_mid,
                                       int c_out) {
  cv1 = register_module("cv1", ConvBNReLU(c_high, c_mid, 1, 1));
  cv2 = register_module("cv2", ConvBNReLU(c_lat, c_mid, 1, 1));
  cv3 = register_module("cv3", ConvBNReLU(3 * c_mid, c_out, 1, 1));
  upsample   = register_module("upsample",   Transpose(c_mid, c_mid));
  downsample = register_module("downsample", ConvBNReLU(c_mid, c_mid, 3, 2));
}
torch::Tensor BiFusionBlockImpl::forward(torch::Tensor x_high,
                                          torch::Tensor x_lat,
                                          torch::Tensor x_lower_reduced) {
  // Meituan's BiFusion forward concatenates as `[upsample, cv1, downsample]`
  // (see yolov6/models/reppan.py). cv3's weights are loaded straight from
  // upstream, so we must match that channel order — getting it wrong
  // mis-routes 1/3 of cv3's input channels.
  auto a = upsample(x_lower_reduced);
  auto b = cv1(x_high);
  auto c = downsample(cv2(x_lat));
  return cv3(torch::cat({a, b, c}, /*dim=*/1));
}

// ─── EffiDeHead ──────────────────────────────────────────────────────────
EffiDeHeadImpl::EffiDeHeadImpl(int nc_, int c_p3, int c_p4, int c_p5,
                                 int reg_max_, bool dfl_eval_)
    : EffiDeHeadImpl(nc_, std::vector<int>{c_p3, c_p4, c_p5},
                     reg_max_, dfl_eval_) {}

EffiDeHeadImpl::EffiDeHeadImpl(int nc_, const std::vector<int>& chans,
                               int reg_max_, bool dfl_eval_)
    : nc(nc_), reg_max(reg_max_), dfl_eval(dfl_eval_) {
  num_layers = (int)chans.size();
  const int bins = reg_max + 1;
  stems          = register_module("stems",          torch::nn::ModuleList());
  cls_convs      = register_module("cls_convs",      torch::nn::ModuleList());
  reg_convs      = register_module("reg_convs",      torch::nn::ModuleList());
  cls_preds      = register_module("cls_preds",      torch::nn::ModuleList());
  // For n/s/n6/s6 (no DFL): reg_preds is 4-ch direct ltrb at eval, plus
  // reg_preds_dist is 68-ch DFL distillation target (training only).
  // For m/l/m6/l6 (DFL): only reg_preds exists, sized 68-ch — projected
  // via DFL at eval.
  reg_preds      = register_module("reg_preds",      torch::nn::ModuleList());
  if (!dfl_eval) {
    reg_preds_dist = register_module("reg_preds_dist", torch::nn::ModuleList());
  }
  const int reg_ch   = dfl_eval ? (4 * bins) : 4;
  for (int i = 0; i < num_layers; ++i) {
    int c = chans[i];
    stems    ->push_back(ConvBNReLU(c, c, 1, 1));
    cls_convs->push_back(ConvBNReLU(c, c, 3, 1));
    reg_convs->push_back(ConvBNReLU(c, c, 3, 1));
    cls_preds->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c, nc, 1)));
    reg_preds->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c, reg_ch, 1)));
    if (!dfl_eval) {
      reg_preds_dist->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(c, 4 * bins, 1)));
    }
  }
  proj = register_buffer(
      "proj", torch::arange(0, bins, torch::kFloat32));
  proj_conv = register_module(
      "proj_conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(bins, 1, 1).bias(false)));
  torch::NoGradGuard ng;
  proj_conv->weight.copy_(proj.view({1, bins, 1, 1}));
}
std::vector<torch::Tensor> EffiDeHeadImpl::forward_eval_per_scale(
    torch::Tensor p3, torch::Tensor p4, torch::Tensor p5,
    const std::vector<int>& strides, int /*img_h*/, int /*img_w*/) {
  return forward_eval_per_scale_n({p3, p4, p5}, strides);
}

std::vector<torch::Tensor> EffiDeHeadImpl::forward_eval_per_scale_n(
    const std::vector<torch::Tensor>& feats,
    const std::vector<int>& strides) {
  std::vector<torch::Tensor> all;
  all.reserve(feats.size());
  const int bins = reg_max + 1;
  auto dev = feats[0].device();
  const int N = (int)feats.size();

  for (int i = 0; i < N; ++i) {
    auto x = stems[i]->as<ConvBNReLUImpl>()->forward(feats[i]);
    auto cls = cls_preds[i]->as<torch::nn::Conv2dImpl>()->forward(
        cls_convs[i]->as<ConvBNReLUImpl>()->forward(x));            // [B, nc, H, W]
    auto reg_feat = reg_convs[i]->as<ConvBNReLUImpl>()->forward(x);
    auto raw      = reg_preds[i]->as<torch::nn::Conv2dImpl>()->forward(reg_feat);

    int B = (int)x.size(0);
    int H = (int)x.size(2);
    int W = (int)x.size(3);
    torch::Tensor d4;
    if (dfl_eval) {
      // m/l: raw is [B, 4*bins, H, W]; project via DFL.
      auto d = raw.view({B, 4, bins, H, W});
      d  = torch::softmax(d, /*dim=*/2);
      d4 = (d * proj.view({1, 1, bins, 1, 1})).sum(/*dim=*/2);  // [B, 4, H, W]
    } else {
      // n/s: raw is already [B, 4, H, W] direct ltrb.
      d4 = raw;
    }

    // dist2bbox: l, t, r, b in stride units → xyxy in pixels.
    int s = strides[i];
    auto gy = torch::arange(H, torch::TensorOptions().device(dev).dtype(d4.dtype()))
                  .add(0.5).view({1, H, 1});
    auto gx = torch::arange(W, torch::TensorOptions().device(dev).dtype(d4.dtype()))
                  .add(0.5).view({1, 1, W});
    auto l = d4.select(/*dim=*/1, 0);
    auto t = d4.select(/*dim=*/1, 1);
    auto r = d4.select(/*dim=*/1, 2);
    auto bb = d4.select(/*dim=*/1, 3);
    auto x1 = (gx - l) * (float)s;
    auto y1 = (gy - t) * (float)s;
    auto x2 = (gx + r) * (float)s;
    auto y2 = (gy + bb) * (float)s;
    auto box = torch::stack({x1, y1, x2, y2}, /*dim=*/1);              // [B, 4, H, W]

    auto cls_sig = torch::sigmoid(cls);                                  // [B, nc, H, W]
    auto packed = torch::cat({box, cls_sig}, /*dim=*/1);                 // [B, 4+nc, H, W]
    packed = packed.view({B, 4 + nc, H * W});                            // [B, 4+nc, A_i]
    all.push_back(packed);
  }
  return all;
}

std::vector<torch::Tensor> EffiDeHeadImpl::forward_train_per_scale_n(
    const std::vector<torch::Tensor>& feats) {
  // Returns per-scale [B, 4*bins + nc, H, W] raw feature maps for the
  // V6DetectionLoss (DFL format). When the DFL distillation branch
  // `reg_preds_dist` is registered (n/s/n6/s6), use it — the upstream
  // training path also drives the DFL distribution through that branch.
  // Otherwise (m/l/m6/l6) `reg_preds` is itself the 68-ch DFL output.
  std::vector<torch::Tensor> outs;
  outs.reserve(feats.size());
  const int N = (int)feats.size();
  for (int i = 0; i < N; ++i) {
    auto x  = stems[i]->as<ConvBNReLUImpl>()->forward(feats[i]);
    auto cls = cls_preds[i]->as<torch::nn::Conv2dImpl>()->forward(
        cls_convs[i]->as<ConvBNReLUImpl>()->forward(x));
    auto reg_feat = reg_convs[i]->as<ConvBNReLUImpl>()->forward(x);
    torch::Tensor reg;
    if (reg_preds_dist && (int)reg_preds_dist->size() > i) {
      reg = reg_preds_dist[i]->as<torch::nn::Conv2dImpl>()->forward(reg_feat);
    } else {
      reg = reg_preds[i]->as<torch::nn::Conv2dImpl>()->forward(reg_feat);
    }
    outs.push_back(torch::cat({reg, cls}, /*dim=*/1));   // [B, 4*bins + nc, H, W]
  }
  return outs;
}

// ─── Yolo6Impl::Backbone ─────────────────────────────────────────────────
Yolo6Impl::BackboneImpl::BackboneImpl(int c0, int c1, int c2, int c3, int c4,
                                        int n2, int n3, int n4, int n5,
                                        bool use_rep_blocks_, bool use_repconv_,
                                        double bep_e, bool use_mbla_) {
  use_rep_blocks = use_rep_blocks_;
  use_repconv    = use_repconv_;
  use_mbla       = use_mbla_;

  // Stem & inter-stage downsamples — RepConv (single 3×3 with bias) for
  // n/s/m, ConvBNReLU for l (which uses no RepConv anywhere).
  auto reg_down = [&](const std::string& name, int ci, int co,
                      RepConv& out_rep, ConvBNReLU& out_cbr) {
    if (use_repconv) out_rep = register_module(name, RepConv(ci, co, 3, 2));
    else             out_cbr = register_module(name, ConvBNReLU(ci, co, 3, 2));
  };
  reg_down("stem",            3,  c0, stem_rep,           stem_cbr);
  reg_down("ERBlock_2_down",  c0, c1, ERBlock_2_down_rep, ERBlock_2_down_cbr);
  reg_down("ERBlock_3_down",  c1, c2, ERBlock_3_down_rep, ERBlock_3_down_cbr);
  reg_down("ERBlock_4_down",  c2, c3, ERBlock_4_down_rep, ERBlock_4_down_cbr);
  reg_down("ERBlock_5_down",  c3, c4, ERBlock_5_down_rep, ERBlock_5_down_cbr);

  // Inner blocks — RepBlock (n/s), BepC3 (m/l), or MBLABlock (*_mbla).
  if (use_mbla) {
    // MBLA: csp_e=0.5 inside MBLABlock; n2..n5 are the post-depth-scaling
    // counts passed unchanged (MBLABlock internally does n//2 itself).
    ERBlock_2_block_mbla = register_module("ERBlock_2_block", MBLABlock(c1, c1, n2, 0.5));
    ERBlock_3_block_mbla = register_module("ERBlock_3_block", MBLABlock(c2, c2, n3, 0.5));
    ERBlock_4_block_mbla = register_module("ERBlock_4_block", MBLABlock(c3, c3, n4, 0.5));
    ERBlock_5_block_mbla = register_module("ERBlock_5_block", MBLABlock(c4, c4, n5, 0.5));
  } else if (use_rep_blocks) {
    ERBlock_2_block_rb = register_module("ERBlock_2_block", RepBlock(c1, c1, n2));
    ERBlock_3_block_rb = register_module("ERBlock_3_block", RepBlock(c2, c2, n3));
    ERBlock_4_block_rb = register_module("ERBlock_4_block", RepBlock(c3, c3, n4));
    ERBlock_5_block_rb = register_module("ERBlock_5_block", RepBlock(c4, c4, n5));
  } else {
    // BepC3's BottleRep inner uses RepConv for v6m, ConvBNReLU for v6l.
    ERBlock_2_block_bep = register_module("ERBlock_2_block", BepC3(c1, c1, n2, use_repconv, bep_e));
    ERBlock_3_block_bep = register_module("ERBlock_3_block", BepC3(c2, c2, n3, use_repconv, bep_e));
    ERBlock_4_block_bep = register_module("ERBlock_4_block", BepC3(c3, c3, n4, use_repconv, bep_e));
    ERBlock_5_block_bep = register_module("ERBlock_5_block", BepC3(c4, c4, n5, use_repconv, bep_e));
  }
  // SPPF type: CSPSPPF (n/s) vs SimSPPF (m/l/MBLA).
  if (use_rep_blocks) {
    ERBlock_5_cspsppf = register_module("ERBlock_5_cspsppf", CSPSPPF(c4, c4, 5));
  } else {
    ERBlock_5_simsppf = register_module("ERBlock_5_cspsppf", SimSPPF(c4, c4, 5));
  }
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Yolo6Impl::BackboneImpl::forward(torch::Tensor /*x*/) {
  TORCH_CHECK(false, "Yolo6 BackboneImpl::forward — call sub-modules directly");
}

// ─── Yolo6Impl::Neck ─────────────────────────────────────────────────────
Yolo6Impl::NeckImpl::NeckImpl(int c2, int c3, int c4, int n_neck,
                                bool use_rep_blocks_, bool use_repconv_,
                                double bep_e, bool use_mbla_) {
  use_rep_blocks = use_rep_blocks_;
  use_repconv    = use_repconv_;
  use_mbla       = use_mbla_;
  const int c_p2 = c2 / 2;
  const int n0   = c3 / 2;
  const int n1   = c2 / 2;

  // Upstream's RepBiFPANNeck/CSPRepBiFPANNeck hardcodes plain `ConvBNReLU`
  // (ReLU) for reduce_layer*/Bifusion*/downsample* regardless of
  // training_mode. Only BepC3 stage blocks pick up the SiLU activation
  // from training_mode=conv_silu (l/x_mbla). Force ReLU in scope here.
  {
    V6ActScope force_relu(false);
    reduce_layer0 = register_module("reduce_layer0", ConvBNReLU(c4, n0, 1, 1));
    Bifusion0     = register_module("Bifusion0",
                                     BiFusionBlock(c3, c2, n0, n0));
    reduce_layer1 = register_module("reduce_layer1", ConvBNReLU(n0, n1, 1, 1));
    Bifusion1     = register_module("Bifusion1",
                                     BiFusionBlock(c2, c_p2, n1, n1));
    downsample2   = register_module("downsample2",   ConvBNReLU(n1, n1, 3, 2));
    downsample1   = register_module("downsample1",   ConvBNReLU(n0, n0, 3, 2));
  }

  if (use_mbla) {
    Rep_p4_mbla = register_module("Rep_p4", MBLABlock(n0,      n0, n_neck, 0.5));
    Rep_p3_mbla = register_module("Rep_p3", MBLABlock(n1,      n1, n_neck, 0.5));
    Rep_n3_mbla = register_module("Rep_n3", MBLABlock(n1 + n1, n0, n_neck, 0.5));
    Rep_n4_mbla = register_module("Rep_n4", MBLABlock(n0 + n0, c3, n_neck, 0.5));
  } else if (use_rep_blocks) {
    Rep_p4_rb = register_module("Rep_p4", RepBlock(n0, n0, n_neck));
    Rep_p3_rb = register_module("Rep_p3", RepBlock(n1, n1, n_neck));
    Rep_n3_rb = register_module("Rep_n3", RepBlock(n1 + n1, n0, n_neck));
    Rep_n4_rb = register_module("Rep_n4", RepBlock(n0 + n0, c3, n_neck));
  } else {
    Rep_p4_bep = register_module("Rep_p4", BepC3(n0, n0, n_neck, use_repconv, bep_e));
    Rep_p3_bep = register_module("Rep_p3", BepC3(n1, n1, n_neck, use_repconv, bep_e));
    Rep_n3_bep = register_module("Rep_n3", BepC3(n1 + n1, n0, n_neck, use_repconv, bep_e));
    Rep_n4_bep = register_module("Rep_n4", BepC3(n0 + n0, c3, n_neck, use_repconv, bep_e));
  }
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Yolo6Impl::NeckImpl::forward(torch::Tensor /*p3*/,
                              torch::Tensor /*p4*/,
                              torch::Tensor /*p5*/) {
  // Unused 3-tuple form — Yolo6Impl::forward calls neck primitives directly
  // because BiFusion needs P2 (4-tuple), which doesn't fit a 3-tuple API.
  TORCH_CHECK(false, "Yolo6 NeckImpl::forward — call sub-modules directly");
}

// ─── Yolo6Impl::BackboneP6 ──────────────────────────────────────────────
Yolo6Impl::BackboneP6Impl::BackboneP6Impl(
    int c0, int c1, int c2, int c3, int c4, int c5,
    int n2, int n3, int n4, int n5, int n6,
    bool use_rep_blocks_, bool use_repconv_, double bep_e) {
  use_rep_blocks = use_rep_blocks_;
  use_repconv    = use_repconv_;

  auto reg_down = [&](const std::string& name, int ci, int co,
                      RepConv& out_rep, ConvBNReLU& out_cbr) {
    if (use_repconv) out_rep = register_module(name, RepConv(ci, co, 3, 2));
    else             out_cbr = register_module(name, ConvBNReLU(ci, co, 3, 2));
  };
  reg_down("stem",            3,  c0, stem_rep,           stem_cbr);
  reg_down("ERBlock_2_down",  c0, c1, ERBlock_2_down_rep, ERBlock_2_down_cbr);
  reg_down("ERBlock_3_down",  c1, c2, ERBlock_3_down_rep, ERBlock_3_down_cbr);
  reg_down("ERBlock_4_down",  c2, c3, ERBlock_4_down_rep, ERBlock_4_down_cbr);
  reg_down("ERBlock_5_down",  c3, c4, ERBlock_5_down_rep, ERBlock_5_down_cbr);
  reg_down("ERBlock_6_down",  c4, c5, ERBlock_6_down_rep, ERBlock_6_down_cbr);

  if (use_rep_blocks) {
    ERBlock_2_block_rb = register_module("ERBlock_2_block", RepBlock(c1, c1, n2));
    ERBlock_3_block_rb = register_module("ERBlock_3_block", RepBlock(c2, c2, n3));
    ERBlock_4_block_rb = register_module("ERBlock_4_block", RepBlock(c3, c3, n4));
    ERBlock_5_block_rb = register_module("ERBlock_5_block", RepBlock(c4, c4, n5));
    ERBlock_6_block_rb = register_module("ERBlock_6_block", RepBlock(c5, c5, n6));
  } else {
    ERBlock_2_block_bep = register_module("ERBlock_2_block", BepC3(c1, c1, n2, use_repconv, bep_e));
    ERBlock_3_block_bep = register_module("ERBlock_3_block", BepC3(c2, c2, n3, use_repconv, bep_e));
    ERBlock_4_block_bep = register_module("ERBlock_4_block", BepC3(c3, c3, n4, use_repconv, bep_e));
    ERBlock_5_block_bep = register_module("ERBlock_5_block", BepC3(c4, c4, n5, use_repconv, bep_e));
    ERBlock_6_block_bep = register_module("ERBlock_6_block", BepC3(c5, c5, n6, use_repconv, bep_e));
  }
  // SPPF lives at ERBlock_6 for P6 (vs ERBlock_5 for P5).
  if (use_rep_blocks) {
    ERBlock_6_cspsppf = register_module("ERBlock_6_cspsppf", CSPSPPF(c5, c5, 5));
  } else {
    ERBlock_6_simsppf = register_module("ERBlock_6_cspsppf", SimSPPF(c5, c5, 5));
  }
}

// ─── Yolo6Impl::NeckP6 ──────────────────────────────────────────────────
//
// channels_list indices (matching upstream RepBiFPANNeck6):
//   c2..c5 = backbone P3..P6 outputs (256, 512, 768, 1024 yaml-scaled)
//   c6..c11 = neck channels (512, 256, 128, 256, 512, 1024 yaml-scaled)
Yolo6Impl::NeckP6Impl::NeckP6Impl(
    int c2, int c3, int c4, int c5,
    int c6, int c7, int c8, int c9, int c10, int c11,
    int n_neck, bool use_rep_blocks_, bool use_repconv_, double bep_e) {
  use_rep_blocks = use_rep_blocks_;
  use_repconv    = use_repconv_;

  // Upstream's CSPRepBiFPANNeck_P6 (and RepBiFPANNeck6) hardcodes plain
  // `ConvBNReLU` (ReLU) for the structural reduce_layer*, Bifusion* and
  // downsample* convs — independent of training_mode. Only the BepC3
  // stage blocks pick up the training_mode block (ConvBNSiLU when l6's
  // training_mode=conv_silu). Force ReLU in scope for the structural
  // convs by pushing V6ActScope(false), then restore the outer scope
  // before constructing the BepC3 blocks.
  {
    V6ActScope force_relu(false);
    reduce_layer0 = register_module("reduce_layer0", ConvBNReLU(c5, c6, 1, 1));
    Bifusion0     = register_module("Bifusion0", BiFusionBlock(c4, c3, c6, c6));
    reduce_layer1 = register_module("reduce_layer1", ConvBNReLU(c6, c7, 1, 1));
    Bifusion1     = register_module("Bifusion1", BiFusionBlock(c3, c2, c7, c7));
    reduce_layer2 = register_module("reduce_layer2", ConvBNReLU(c7, c8, 1, 1));
    Bifusion2     = register_module("Bifusion2", BiFusionBlock(c2, c2 / 2, c8, c8));
    downsample2   = register_module("downsample2", ConvBNReLU(c8,  c8, 3, 2));
    downsample1   = register_module("downsample1", ConvBNReLU(c9,  c9, 3, 2));
    downsample0   = register_module("downsample0", ConvBNReLU(c10, c10, 3, 2));
  }

  if (use_rep_blocks) {
    Rep_p5_rb = register_module("Rep_p5", RepBlock(c6, c6, n_neck));
    Rep_p4_rb = register_module("Rep_p4", RepBlock(c7, c7, n_neck));
    Rep_p3_rb = register_module("Rep_p3", RepBlock(c8, c8, n_neck));
    Rep_n4_rb = register_module("Rep_n4", RepBlock(c8 + c8, c9, n_neck));
    Rep_n5_rb = register_module("Rep_n5", RepBlock(c7 + c9, c10, n_neck));
    Rep_n6_rb = register_module("Rep_n6", RepBlock(c6 + c10, c11, n_neck));
  } else {
    Rep_p5_bep = register_module("Rep_p5", BepC3(c6, c6, n_neck, use_repconv, bep_e));
    Rep_p4_bep = register_module("Rep_p4", BepC3(c7, c7, n_neck, use_repconv, bep_e));
    Rep_p3_bep = register_module("Rep_p3", BepC3(c8, c8, n_neck, use_repconv, bep_e));
    Rep_n4_bep = register_module("Rep_n4", BepC3(c8 + c8, c9,  n_neck, use_repconv, bep_e));
    Rep_n5_bep = register_module("Rep_n5", BepC3(c7 + c9, c10, n_neck, use_repconv, bep_e));
    Rep_n6_bep = register_module("Rep_n6", BepC3(c6 + c10, c11, n_neck, use_repconv, bep_e));
  }
}

// ─── Yolo6Impl ───────────────────────────────────────────────────────────
Yolo6Impl::Yolo6Impl(int nc_, Yolo6Scale s, int reg_max_, bool p6)
    : scale(s), nc(nc_), reg_max(reg_max_), is_p6(p6) {
  const int c0 = make_div(64   * s.width_multiple);
  const int c1 = make_div(128  * s.width_multiple);
  const int c2 = make_div(256  * s.width_multiple);
  const int c3 = make_div(512  * s.width_multiple);
  const int c4 = make_div(1024 * s.width_multiple);
  // P6 adds an extra backbone stage with yaml channel 768 between c3 and c4.
  // P6 backbone channels_list = [c0, c1, c2, c3, 768*w, 1024*w] = [c0..c5].
  const int c4_p6 = make_div(768  * s.width_multiple);
  const int c5_p6 = make_div(1024 * s.width_multiple);

  // Detect scale variant from width:
  //   n: 0.25 → RepBlock + RepConv stems, direct 4-ch reg head
  //   s: 0.50 → RepBlock + RepConv stems, direct 4-ch reg head
  //   m: 0.75 → BepC3 + RepConv (in BottleRep), 68-ch DFL reg head
  //   l: 1.00 → BepC3 + ConvBNReLU (in BottleRep) + ConvBNReLU stems,
  //             68-ch DFL reg head
  const bool is_ns = (s.width_multiple <= 0.50 && s.variant == Yolo6Variant::Standard);
  const bool is_l  = (s.width_multiple >= 1.00 && s.variant == Yolo6Variant::Standard);
  const bool is_mbla = (s.variant == Yolo6Variant::MBLA);
  // use_rep_blocks=true means PLAIN RepBlock (n/s); false means BepC3 (m/l)
  // OR MBLABlock (any *_mbla scale).
  const bool use_rep_blocks = is_ns;
  // ConvBNReLU stems (no RepConv fusion): v6l + all MBLA scales (all use
  // training_mode=conv_silu, weights stored as Conv+BN, not RepVGG).
  const bool use_repconv    = !is_l && !is_mbla;
  const bool dfl_eval       = !is_ns;
  const double bep_e        = is_l ? 0.5 : (2.0 / 3.0);

  // Depth-scaled inner counts. For BepC3 we halve (each BottleRep has 2
  // convs); for MBLA we DON'T halve — MBLABlock does its own n//2 inside.
  // Standard num_repeats: backbone [1, 6, 12, 18, 6], neck [12, 12, 12, 12].
  // MBLA num_repeats: backbone [1, 4, 8, 8, 4], neck [8, 8, 8, 8].
  int n2, n3, n4, n5, n_neck;
  if (is_mbla) {
    n2     = std::max(1, (int)std::round(4 * s.depth_multiple));
    n3     = std::max(1, (int)std::round(8 * s.depth_multiple));
    n4     = std::max(1, (int)std::round(8 * s.depth_multiple));
    n5     = std::max(1, (int)std::round(4 * s.depth_multiple));
    n_neck = std::max(1, (int)std::round(8 * s.depth_multiple));
  } else {
    auto half_if_bep = [&](int x) { return use_rep_blocks ? x : std::max(1, x / 2); };
    n2     = half_if_bep(dep(6,  s.depth_multiple));
    n3     = half_if_bep(dep(12, s.depth_multiple));
    n4     = half_if_bep(dep(18, s.depth_multiple));
    n5     = half_if_bep(dep(6,  s.depth_multiple));
    n_neck = half_if_bep(dep(12, s.depth_multiple));
  }

  // v6l + all MBLA scales use training_mode=conv_silu — every
  // ConvBNReLU (including those inside BottleRep/BottleRep3, BepC3,
  // MBLABlock, the neck, and the head) uses SiLU. n/s/m use ReLU.
  V6ActScope act_scope(is_l || is_mbla);

  if (is_p6) {
    // P6 backbone: 6-stage EfficientRep6 / CSPBepBackbone_P6.
    // num_repeats: backbone [1, 6, 12, 18, 6, 6], neck [12, 12, 12, 12, 12, 12].
    auto half_if_bep = [&](int x) { return use_rep_blocks ? x : std::max(1, x / 2); };
    int n2_p = half_if_bep(dep(6,  s.depth_multiple));
    int n3_p = half_if_bep(dep(12, s.depth_multiple));
    int n4_p = half_if_bep(dep(18, s.depth_multiple));
    int n5_p = half_if_bep(dep(6,  s.depth_multiple));
    int n6_p = half_if_bep(dep(6,  s.depth_multiple));
    int n_p6_neck = half_if_bep(dep(12, s.depth_multiple));
    backbone_p6 = std::make_shared<BackboneP6Impl>(
        c0, c1, c2, c3, c4_p6, c5_p6, n2_p, n3_p, n4_p, n5_p, n6_p,
        use_rep_blocks, use_repconv, bep_e);
    register_module("backbone", backbone_p6);
    // P6 neck channels_list[6..11] = [512, 256, 128, 256, 512, 1024] * width.
    int nc6  = make_div(512  * s.width_multiple);
    int nc7  = make_div(256  * s.width_multiple);
    int nc8  = make_div(128  * s.width_multiple);
    int nc9  = make_div(256  * s.width_multiple);
    int nc10 = make_div(512  * s.width_multiple);
    int nc11 = make_div(1024 * s.width_multiple);
    neck_p6 = std::make_shared<NeckP6Impl>(
        c2, c3, c4_p6, c5_p6, nc6, nc7, nc8, nc9, nc10, nc11,
        n_p6_neck, use_rep_blocks, use_repconv, bep_e);
    register_module("neck", neck_p6);
    // 4-level head; P3..P6 → channels {nc8, nc9, nc10, nc11}.
    detect = register_module("detect",
                             EffiDeHead(nc, std::vector<int>{nc8, nc9, nc10, nc11},
                                        reg_max, dfl_eval));
    return;
  }

  backbone = std::make_shared<BackboneImpl>(
      c0, c1, c2, c3, c4, n2, n3, n4, n5,
      use_rep_blocks, use_repconv, bep_e, is_mbla);
  register_module("backbone", backbone);
  neck = std::make_shared<NeckImpl>(c2, c3, c4, n_neck,
                                    use_rep_blocks, use_repconv, bep_e, is_mbla);
  register_module("neck", neck);
  const int head_p3 = c2 / 2;
  const int head_p4 = c3 / 2;
  const int head_p5 = c3;
  detect = register_module("detect",
                           EffiDeHead(nc, head_p3, head_p4, head_p5,
                                       reg_max, dfl_eval));
}

std::vector<torch::Tensor> Yolo6Impl::forward(torch::Tensor x) {
  // Stem & inter-stage downsamples. Use the RepConv path when present
  // (n/s/m), else the ConvBNReLU path (l/MBLA).
  auto stem_fwd = [&](const torch::Tensor& t, RepConv& rp, ConvBNReLU& cbr) {
    return rp ? rp->forward(t) : cbr->forward(t);
  };

  // ─── P6 path (4 detect levels at strides [8, 16, 32, 64]) ──────────
  if (is_p6) {
    auto* bp = backbone_p6.get();
    auto block_fwd_p6 = [&](const torch::Tensor& t,
                             RepBlock& rb, BepC3& bep) {
      return rb ? rb->forward(t) : bep->forward(t);
    };
    auto y  = stem_fwd(x,  bp->stem_rep,           bp->stem_cbr);
    y       = stem_fwd(y,  bp->ERBlock_2_down_rep, bp->ERBlock_2_down_cbr);
    auto p2 = block_fwd_p6(y, bp->ERBlock_2_block_rb, bp->ERBlock_2_block_bep);
    auto p3_in = stem_fwd(p2, bp->ERBlock_3_down_rep, bp->ERBlock_3_down_cbr);
    auto p3 = block_fwd_p6(p3_in, bp->ERBlock_3_block_rb, bp->ERBlock_3_block_bep);
    auto p4_in = stem_fwd(p3, bp->ERBlock_4_down_rep, bp->ERBlock_4_down_cbr);
    auto p4 = block_fwd_p6(p4_in, bp->ERBlock_4_block_rb, bp->ERBlock_4_block_bep);
    auto p5_in = stem_fwd(p4, bp->ERBlock_5_down_rep, bp->ERBlock_5_down_cbr);
    auto p5 = block_fwd_p6(p5_in, bp->ERBlock_5_block_rb, bp->ERBlock_5_block_bep);
    auto p6_in = stem_fwd(p5, bp->ERBlock_6_down_rep, bp->ERBlock_6_down_cbr);
    auto p6_block = block_fwd_p6(p6_in, bp->ERBlock_6_block_rb, bp->ERBlock_6_block_bep);
    torch::Tensor p6_out = bp->ERBlock_6_cspsppf
                               ? bp->ERBlock_6_cspsppf->forward(p6_block)
                               : bp->ERBlock_6_simsppf->forward(p6_block);

    // Neck — 3 reduce/Bifusion/Rep_p* top-down + 3 downsample/Rep_n*.
    auto* np = neck_p6.get();
    auto fpn0      = np->reduce_layer0(p6_out);
    auto fused5    = np->Bifusion0(p5, p4, fpn0);  // (P5 lat, P4 lat, fpn) → P5
    auto out_p5_td = block_fwd_p6(fused5, np->Rep_p5_rb, np->Rep_p5_bep);

    auto fpn1      = np->reduce_layer1(out_p5_td);
    auto fused4    = np->Bifusion1(p4, p3, fpn1);
    auto out_p4_td = block_fwd_p6(fused4, np->Rep_p4_rb, np->Rep_p4_bep);

    auto fpn2      = np->reduce_layer2(out_p4_td);
    auto fused3    = np->Bifusion2(p3, p2, fpn2);
    auto out_p3    = block_fwd_p6(fused3, np->Rep_p3_rb, np->Rep_p3_bep);

    auto d2     = np->downsample2(out_p3);
    auto out_p4 = block_fwd_p6(torch::cat({d2, fpn2}, 1), np->Rep_n4_rb, np->Rep_n4_bep);
    auto d1     = np->downsample1(out_p4);
    auto out_p5 = block_fwd_p6(torch::cat({d1, fpn1}, 1), np->Rep_n5_rb, np->Rep_n5_bep);
    auto d0     = np->downsample0(out_p5);
    auto out_p6 = block_fwd_p6(torch::cat({d0, fpn0}, 1), np->Rep_n6_rb, np->Rep_n6_bep);

    return detect->forward_eval_per_scale_n(
        {out_p3, out_p4, out_p5, out_p6}, {8, 16, 32, 64});
  }
  // Inner block dispatch — exactly one of {rb, bep, mbla} is registered.
  auto block_fwd = [&](const torch::Tensor& t,
                        RepBlock& rb, BepC3& bep, MBLABlock& mbla) {
    if (rb)   return rb->forward(t);
    if (bep)  return bep->forward(t);
    return mbla->forward(t);
  };

  auto y  = stem_fwd(x,  backbone->stem_rep,           backbone->stem_cbr);
  y       = stem_fwd(y,  backbone->ERBlock_2_down_rep, backbone->ERBlock_2_down_cbr);
  auto p2 = block_fwd(y, backbone->ERBlock_2_block_rb, backbone->ERBlock_2_block_bep, backbone->ERBlock_2_block_mbla);
  auto p3_in = stem_fwd(p2, backbone->ERBlock_3_down_rep, backbone->ERBlock_3_down_cbr);
  auto p3 = block_fwd(p3_in, backbone->ERBlock_3_block_rb, backbone->ERBlock_3_block_bep, backbone->ERBlock_3_block_mbla);
  auto p4_in = stem_fwd(p3, backbone->ERBlock_4_down_rep, backbone->ERBlock_4_down_cbr);
  auto p4 = block_fwd(p4_in, backbone->ERBlock_4_block_rb, backbone->ERBlock_4_block_bep, backbone->ERBlock_4_block_mbla);
  auto p5_in = stem_fwd(p4, backbone->ERBlock_5_down_rep, backbone->ERBlock_5_down_cbr);
  auto p5_block = block_fwd(p5_in, backbone->ERBlock_5_block_rb, backbone->ERBlock_5_block_bep, backbone->ERBlock_5_block_mbla);
  torch::Tensor p5 = backbone->ERBlock_5_cspsppf
                         ? backbone->ERBlock_5_cspsppf->forward(p5_block)
                         : backbone->ERBlock_5_simsppf->forward(p5_block);

  auto fpn0      = neck->reduce_layer0(p5);
  auto fused4    = neck->Bifusion0(p4, p3, fpn0);
  auto out_p4_td = block_fwd(fused4, neck->Rep_p4_rb, neck->Rep_p4_bep, neck->Rep_p4_mbla);

  auto fpn1   = neck->reduce_layer1(out_p4_td);
  auto fused3 = neck->Bifusion1(p3, p2, fpn1);
  auto out_p3 = block_fwd(fused3, neck->Rep_p3_rb, neck->Rep_p3_bep, neck->Rep_p3_mbla);

  auto d2     = neck->downsample2(out_p3);
  auto out_p4 = block_fwd(torch::cat({d2, fpn1}, 1), neck->Rep_n3_rb, neck->Rep_n3_bep, neck->Rep_n3_mbla);
  auto d1     = neck->downsample1(out_p4);
  auto out_p5 = block_fwd(torch::cat({d1, fpn0}, 1), neck->Rep_n4_rb, neck->Rep_n4_bep, neck->Rep_n4_mbla);

  return detect->forward_eval_per_scale(out_p3, out_p4, out_p5,
                                         {8, 16, 32},
                                         (int)x.size(2), (int)x.size(3));
}

std::vector<torch::Tensor> Yolo6Impl::forward_train(torch::Tensor x) {
  // Walks the same backbone+neck as forward_eval, but ends at the head's
  // forward_train_per_scale_n which returns RAW [B, 4*bins+nc, H, W]
  // feature maps (no decode, no sigmoid) for V6DetectionLoss.
  // Mirrors forward_eval's dispatch tree exactly — only the final
  // head call differs.
  auto stem_fwd = [&](const torch::Tensor& t, RepConv& rp, ConvBNReLU& cbr) {
    return rp ? rp->forward(t) : cbr->forward(t);
  };
  auto block_fwd = [&](const torch::Tensor& t,
                        RepBlock& rb, BepC3& bep, MBLABlock& mbla) {
    if (rb)   return rb->forward(t);
    if (bep)  return bep->forward(t);
    return mbla->forward(t);
  };

  if (is_p6) {
    auto* bp = backbone_p6.get();
    auto block_fwd_p6 = [&](const torch::Tensor& t,
                             RepBlock& rb, BepC3& bep) {
      return rb ? rb->forward(t) : bep->forward(t);
    };
    auto y  = stem_fwd(x,  bp->stem_rep,           bp->stem_cbr);
    y       = stem_fwd(y,  bp->ERBlock_2_down_rep, bp->ERBlock_2_down_cbr);
    auto p2 = block_fwd_p6(y, bp->ERBlock_2_block_rb, bp->ERBlock_2_block_bep);
    auto p3_in = stem_fwd(p2, bp->ERBlock_3_down_rep, bp->ERBlock_3_down_cbr);
    auto p3 = block_fwd_p6(p3_in, bp->ERBlock_3_block_rb, bp->ERBlock_3_block_bep);
    auto p4_in = stem_fwd(p3, bp->ERBlock_4_down_rep, bp->ERBlock_4_down_cbr);
    auto p4 = block_fwd_p6(p4_in, bp->ERBlock_4_block_rb, bp->ERBlock_4_block_bep);
    auto p5_in = stem_fwd(p4, bp->ERBlock_5_down_rep, bp->ERBlock_5_down_cbr);
    auto p5 = block_fwd_p6(p5_in, bp->ERBlock_5_block_rb, bp->ERBlock_5_block_bep);
    auto p6_in = stem_fwd(p5, bp->ERBlock_6_down_rep, bp->ERBlock_6_down_cbr);
    auto p6_block = block_fwd_p6(p6_in, bp->ERBlock_6_block_rb, bp->ERBlock_6_block_bep);
    torch::Tensor p6_out = bp->ERBlock_6_cspsppf
                               ? bp->ERBlock_6_cspsppf->forward(p6_block)
                               : bp->ERBlock_6_simsppf->forward(p6_block);
    auto* np = neck_p6.get();
    auto fpn0      = np->reduce_layer0(p6_out);
    auto fused5    = np->Bifusion0(p5, p4, fpn0);
    auto out_p5_td = block_fwd_p6(fused5, np->Rep_p5_rb, np->Rep_p5_bep);
    auto fpn1      = np->reduce_layer1(out_p5_td);
    auto fused4    = np->Bifusion1(p4, p3, fpn1);
    auto out_p4_td = block_fwd_p6(fused4, np->Rep_p4_rb, np->Rep_p4_bep);
    auto fpn2      = np->reduce_layer2(out_p4_td);
    auto fused3    = np->Bifusion2(p3, p2, fpn2);
    auto out_p3    = block_fwd_p6(fused3, np->Rep_p3_rb, np->Rep_p3_bep);
    auto d2     = np->downsample2(out_p3);
    auto out_p4 = block_fwd_p6(torch::cat({d2, fpn2}, 1), np->Rep_n4_rb, np->Rep_n4_bep);
    auto d1     = np->downsample1(out_p4);
    auto out_p5 = block_fwd_p6(torch::cat({d1, fpn1}, 1), np->Rep_n5_rb, np->Rep_n5_bep);
    auto d0     = np->downsample0(out_p5);
    auto out_p6 = block_fwd_p6(torch::cat({d0, fpn0}, 1), np->Rep_n6_rb, np->Rep_n6_bep);
    if (stride.empty()) stride = {8.0, 16.0, 32.0, 64.0};
    return detect->forward_train_per_scale_n({out_p3, out_p4, out_p5, out_p6});
  }

  // P5 path
  auto y  = stem_fwd(x,  backbone->stem_rep,           backbone->stem_cbr);
  y       = stem_fwd(y,  backbone->ERBlock_2_down_rep, backbone->ERBlock_2_down_cbr);
  auto p2 = block_fwd(y, backbone->ERBlock_2_block_rb, backbone->ERBlock_2_block_bep, backbone->ERBlock_2_block_mbla);
  auto p3_in = stem_fwd(p2, backbone->ERBlock_3_down_rep, backbone->ERBlock_3_down_cbr);
  auto p3 = block_fwd(p3_in, backbone->ERBlock_3_block_rb, backbone->ERBlock_3_block_bep, backbone->ERBlock_3_block_mbla);
  auto p4_in = stem_fwd(p3, backbone->ERBlock_4_down_rep, backbone->ERBlock_4_down_cbr);
  auto p4 = block_fwd(p4_in, backbone->ERBlock_4_block_rb, backbone->ERBlock_4_block_bep, backbone->ERBlock_4_block_mbla);
  auto p5_in = stem_fwd(p4, backbone->ERBlock_5_down_rep, backbone->ERBlock_5_down_cbr);
  auto p5_block = block_fwd(p5_in, backbone->ERBlock_5_block_rb, backbone->ERBlock_5_block_bep, backbone->ERBlock_5_block_mbla);
  torch::Tensor p5 = backbone->ERBlock_5_cspsppf
                         ? backbone->ERBlock_5_cspsppf->forward(p5_block)
                         : backbone->ERBlock_5_simsppf->forward(p5_block);
  auto fpn0      = neck->reduce_layer0(p5);
  auto fused4    = neck->Bifusion0(p4, p3, fpn0);
  auto out_p4_td = block_fwd(fused4, neck->Rep_p4_rb, neck->Rep_p4_bep, neck->Rep_p4_mbla);
  auto fpn1   = neck->reduce_layer1(out_p4_td);
  auto fused3 = neck->Bifusion1(p3, p2, fpn1);
  auto out_p3 = block_fwd(fused3, neck->Rep_p3_rb, neck->Rep_p3_bep, neck->Rep_p3_mbla);
  auto d2     = neck->downsample2(out_p3);
  auto out_p4 = block_fwd(torch::cat({d2, fpn1}, 1), neck->Rep_n3_rb, neck->Rep_n3_bep, neck->Rep_n3_mbla);
  auto d1     = neck->downsample1(out_p4);
  auto out_p5 = block_fwd(torch::cat({d1, fpn0}, 1), neck->Rep_n4_rb, neck->Rep_n4_bep, neck->Rep_n4_mbla);
  if (stride.empty()) stride = {8.0, 16.0, 32.0};
  return detect->forward_train_per_scale_n({out_p3, out_p4, out_p5});
}

torch::Tensor Yolo6Impl::forward_eval(torch::Tensor x) {
  auto per = forward(x);
  // Concat over A → [B, 4+nc, A_total]
  return torch::cat(per, /*dim=*/-1).contiguous();
}

int Yolo6Impl::load_from_state_dict(
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
