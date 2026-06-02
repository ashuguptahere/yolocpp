#include "yolocpp/models/yolo8.hpp"

#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace yolocpp::models {

namespace F = torch::nn::functional;

// Auto-pad: same as the upstream autopad(k, p, d=1).
static int autopad(int k, int p, int d = 1) {
  if (p >= 0) return p;
  // d=1 always for our use — effective k stays k.
  (void)d;
  return k / 2;
}

namespace {
thread_local double g_default_bn_eps = 1e-3;
}
double get_default_bn_eps() { return g_default_bn_eps; }
BnEpsScope::BnEpsScope(double new_eps) : prev(g_default_bn_eps) {
  g_default_bn_eps = new_eps;
}
BnEpsScope::~BnEpsScope() { g_default_bn_eps = prev; }

int scale_channels(int c, const Yolo8Scale& s) {
  c = std::min(c, s.max_channels);
  // upstream rounds to multiple of 8 in some places, but for v8 the
  // YAML-defined channels are already multiples of 8 and the multiplier
  // is applied last. Their exact rule:
  //   c = make_divisible(min(c, max) * width, 8)
  auto make_divisible = [](double v, int divisor) {
    return std::max(divisor, (int)std::round(v / divisor) * divisor);
  };
  return make_divisible(c * s.width_multiple, 8);
}

int scale_depth(int n, const Yolo8Scale& s) {
  return std::max(1, (int)std::round(n * s.depth_multiple));
}

// ─── Conv ──────────────────────────────────────────────────────────────────

ConvImpl::ConvImpl(int c_in, int c_out, int k, int s, int p, int g, bool act,
                   bool conv_bias)
    : act_silu(act) {
  auto pad = autopad(k, p);
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, k)
                            .stride(s)
                            .padding(pad)
                            .groups(g)
                            .bias(conv_bias)
                            .dilation(1)));
  // Upstream overrides BN eps to 1e-3 for detect/segment/pose/obb
  // (vs PyTorch default 1e-5). The classify models use plain 1e-5
  // though — Yolo*Classify constructors push a `BnEpsScope(1e-5)` before
  // building their children so this picks up the cls value.
  bn = register_module("bn", torch::nn::BatchNorm2d(
      torch::nn::BatchNorm2dOptions(c_out).eps(g_default_bn_eps)));
}

torch::Tensor ConvImpl::forward(torch::Tensor x) {
  if (fused) {
    // Fused Conv+BN path (#95B): single conv2d with absorbed BN stats.
    // The bias slot of fused_weight is the conv bias post-folding;
    // BN module is bypassed entirely.
    const auto& s_arr = *conv->options.stride();
    const auto& p_arr = *std::get<torch::ExpandingArray<2>>(conv->options.padding());
    const auto& d_arr = *conv->options.dilation();
    std::vector<int64_t> stride{s_arr[0], s_arr[1]};
    std::vector<int64_t> padding{p_arr[0], p_arr[1]};
    std::vector<int64_t> dilation{d_arr[0], d_arr[1]};
    int groups = (int)conv->options.groups();
    x = at::conv2d(x, fused_weight, fused_bias,
                   at::IntArrayRef(stride),
                   at::IntArrayRef(padding),
                   at::IntArrayRef(dilation),
                   groups);
  } else {
    x = bn(conv(x));
  }
  if (act_silu) x = F::silu(x);
  return x;
}

void ConvImpl::fuse() {
  if (fused) return;
  torch::NoGradGuard ng;
  auto cw = conv->weight.detach();
  auto cb = conv->bias.defined() ? conv->bias.detach()
                                 : torch::zeros({cw.size(0)}, cw.options());
  auto bw = bn->weight.detach();
  auto bb = bn->bias.detach();
  auto rm = bn->running_mean.detach();
  auto rv = bn->running_var.detach();
  double eps = bn->options.eps();
  auto scale = bw / torch::sqrt(rv + eps);
  fused_weight = (cw * scale.view({-1, 1, 1, 1})).contiguous();
  fused_bias   = ((cb - rm) * scale + bb).contiguous();
  fused = true;
}

// ─── DWConv ────────────────────────────────────────────────────────────────
// Depthwise: groups = gcd(c_in, c_out). With c_in == c_out this is the
// classic depthwise conv (one filter per channel).

static int int_gcd(int a, int b) {
  while (b) { int t = b; b = a % b; a = t; }
  return a;
}

DWConvImpl::DWConvImpl(int c_in, int c_out, int k, int s, bool act)
    : act_silu(act) {
  auto pad = autopad(k, -1);
  int g = int_gcd(c_in, c_out);
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c_in, c_out, k)
                            .stride(s)
                            .padding(pad)
                            .groups(g)
                            .bias(false)
                            .dilation(1)));
  // Same BN-eps thread-local override as ConvImpl.
  bn = register_module("bn", torch::nn::BatchNorm2d(
      torch::nn::BatchNorm2dOptions(c_out).eps(g_default_bn_eps)));
}

torch::Tensor DWConvImpl::forward(torch::Tensor x) {
  if (fused) {
    const auto& s_arr = *conv->options.stride();
    const auto& p_arr = *std::get<torch::ExpandingArray<2>>(conv->options.padding());
    const auto& d_arr = *conv->options.dilation();
    std::vector<int64_t> stride{s_arr[0], s_arr[1]};
    std::vector<int64_t> padding{p_arr[0], p_arr[1]};
    std::vector<int64_t> dilation{d_arr[0], d_arr[1]};
    int groups = (int)conv->options.groups();
    x = at::conv2d(x, fused_weight, fused_bias,
                   at::IntArrayRef(stride),
                   at::IntArrayRef(padding),
                   at::IntArrayRef(dilation),
                   groups);
  } else {
    x = bn(conv(x));
  }
  if (act_silu) x = F::silu(x);
  return x;
}

void DWConvImpl::fuse() {
  if (fused) return;
  torch::NoGradGuard ng;
  auto cw = conv->weight.detach();
  auto cb = conv->bias.defined() ? conv->bias.detach()
                                 : torch::zeros({cw.size(0)}, cw.options());
  auto bw = bn->weight.detach();
  auto bb = bn->bias.detach();
  auto rm = bn->running_mean.detach();
  auto rv = bn->running_var.detach();
  double eps = bn->options.eps();
  auto scale = bw / torch::sqrt(rv + eps);
  fused_weight = (cw * scale.view({-1, 1, 1, 1})).contiguous();
  fused_bias   = ((cb - rm) * scale + bb).contiguous();
  fused = true;
}

// Recursive fuser: walks the module tree, calls fuse() on every
// ConvImpl/DWConvImpl. Other module types are descended into via
// children(). Handles modular composition (C2f/C3k2/Bottleneck/etc.)
// because they ultimately leaf into ConvImpl/DWConvImpl. (#95B.)
void fuse_model(torch::nn::Module& root) {
  if (auto* c = root.as<ConvImpl>())        { c->fuse(); return; }
  if (auto* d = root.as<DWConvImpl>())      { d->fuse(); return; }
  for (auto& child : root.children()) fuse_model(*child);
}

// ─── DWConvBlock = DWConv → Conv 1×1 (children registered as "0" / "1") ──

DWConvBlockImpl::DWConvBlockImpl(int c_in, int c_out, int k_dw, bool act) {
  dw = register_module("0", DWConv(c_in, c_in, k_dw, 1, /*act=*/true));
  pw = register_module("1", Conv(c_in, c_out, 1, 1, /*p=*/-1, /*g=*/1, act));
}

torch::Tensor DWConvBlockImpl::forward(torch::Tensor x) {
  return pw(dw(x));
}

// ─── Bottleneck ────────────────────────────────────────────────────────────

BottleneckImpl::BottleneckImpl(int c1, int c2, bool shortcut, int g, double e,
                               std::array<int, 2> k) {
  int c_ = (int)(c2 * e);
  cv1 = register_module("cv1", Conv(c1, c_, k[0], 1));
  cv2 = register_module("cv2", Conv(c_, c2, k[1], 1, -1, g));
  add = shortcut && (c1 == c2);
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
  auto y = cv2(cv1(x));
  return add ? x + y : y;
}

// ─── C2f ───────────────────────────────────────────────────────────────────

C2fImpl::C2fImpl(int c1, int c2, int n, bool shortcut, int g, double e) {
  c_inner = (int)(c2 * e);
  cv1 = register_module("cv1", Conv(c1, 2 * c_inner, 1, 1));
  cv2 = register_module("cv2", Conv((2 + n) * c_inner, c2, 1));
  m   = register_module("m", torch::nn::ModuleList());
  for (int i = 0; i < n; ++i) {
    m->push_back(Bottleneck(c_inner, c_inner, shortcut, g, 1.0,
                            std::array<int, 2>{3, 3}));
  }
}

torch::Tensor C2fImpl::forward(torch::Tensor x) {
  auto y = cv1(x).chunk(2, /*dim=*/1);
  std::vector<torch::Tensor> outs;
  outs.reserve(2 + m->size());
  outs.push_back(y[0]);
  outs.push_back(y[1]);
  for (size_t i = 0; i < m->size(); ++i) {
    auto* mod = m[i]->as<BottleneckImpl>();
    outs.push_back(mod->forward(outs.back()));
  }
  return cv2(torch::cat(outs, /*dim=*/1));
}

// ─── SPPF ──────────────────────────────────────────────────────────────────

SPPFImpl::SPPFImpl(int c1, int c2, int k, bool cv1_act, bool shortcut) {
  int c_ = c1 / 2;
  add = shortcut && (c1 == c2);
  cv1 = register_module("cv1", Conv(c1, c_, 1, 1, /*p=*/-1, /*g=*/1, cv1_act));
  cv2 = register_module("cv2", Conv(c_ * 4, c2, 1, 1));
  m   = register_module(
      "m",
      torch::nn::MaxPool2d(
          torch::nn::MaxPool2dOptions(k).stride(1).padding(k / 2)));
}

torch::Tensor SPPFImpl::forward(torch::Tensor x) {
  auto x_in = x;
  x = cv1(x);
  auto y1 = m(x);
  auto y2 = m(y1);
  auto y3 = m(y2);
  auto y  = cv2(torch::cat({x, y1, y2, y3}, 1));
  return add ? (y + x_in) : y;
}

// ─── DFL ───────────────────────────────────────────────────────────────────

DFLImpl::DFLImpl(int c1_) : c1(c1_) {
  conv = register_module(
      "conv",
      torch::nn::Conv2d(
          torch::nn::Conv2dOptions(c1, 1, 1).bias(false)));
  // Initialize weights to arange(c1) so that softmax * weight = expectation.
  torch::NoGradGuard ng;
  auto w = torch::arange(c1, torch::kFloat32).view({1, c1, 1, 1});
  conv->weight.copy_(w);
  conv->weight.set_requires_grad(false);
}

torch::Tensor DFLImpl::forward(torch::Tensor x) {
  // x: [N, 4*c1, A] → reshape to [N, 4, c1, A] → softmax(c1) → conv → [N,4,A]
  auto sizes = x.sizes();
  auto N = sizes[0];
  auto A = sizes[2];
  x = x.view({N, 4, c1, A}).transpose(2, 1).softmax(1);  // [N, c1, 4, A]
  return conv(x).view({N, 4, A});
}

// ─── Detect ────────────────────────────────────────────────────────────────

DetectImpl::DetectImpl(int nc_, std::vector<int> ch_, bool legacy_)
    : nc(nc_), legacy(legacy_), ch(std::move(ch_)) {
  nl = (int)ch.size();
  no = nc + reg_max * 4;
  // c2 = max(16, ch[0]//4, reg_max*4)
  int c2 = std::max({16, ch[0] / 4, reg_max * 4});
  // c3 = max(ch[0], min(nc, 100))
  int c3 = std::max(ch[0], std::min(nc, 100));

  cv2 = register_module("cv2", torch::nn::ModuleList());
  cv3 = register_module("cv3", torch::nn::ModuleList());

  for (int i = 0; i < nl; ++i) {
    auto reg = torch::nn::Sequential();
    reg->push_back(Conv(ch[i], c2, 3));
    reg->push_back(Conv(c2,    c2, 3));
    reg->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(c2, 4 * reg_max, 1)));
    cv2->push_back(reg);

    auto cls = torch::nn::Sequential();
    if (legacy) {
      // v3 / v5 / v8 / v9: regular Conv→Conv→Conv2d.
      cls->push_back(Conv(ch[i], c3, 3));
      cls->push_back(Conv(c3,    c3, 3));
      cls->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(c3, nc, 1)));
    } else {
      // v11+: nested (DWConv 3×3 → Conv 1×1) × 2 → Conv2d 1×1. Each pair is
      // a DWConvBlock with children named "0"/"1", so the full state_dict
      // path becomes cv3.<lvl>.<0|1>.<0|1>.{conv,bn}.<...> —
      // matching the upstream yolo11<x>.pt naming.
      cls->push_back(DWConvBlock(ch[i], c3, /*k_dw=*/3));
      cls->push_back(DWConvBlock(c3,    c3, /*k_dw=*/3));
      cls->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(c3, nc, 1)));
    }
    cv3->push_back(cls);
  }
  dfl = register_module("dfl", DFL(reg_max));
}

std::vector<torch::Tensor> DetectImpl::forward_features(std::vector<torch::Tensor> x) {
  std::vector<torch::Tensor> out;
  out.reserve(nl);
  for (int i = 0; i < nl; ++i) {
    auto* reg = cv2[i]->as<torch::nn::SequentialImpl>();
    auto* cls = cv3[i]->as<torch::nn::SequentialImpl>();
    auto a = reg->forward(x[i]);
    auto b = cls->forward(x[i]);
    out.push_back(torch::cat({a, b}, /*dim=*/1));
  }
  return out;
}

void init_detect_biases(torch::nn::Module* root) {
  if (!root) return;
  for (const auto& m : root->modules(/*include_self=*/false)) {
    if (auto* det = m->as<DetectImpl>()) {
      det->init_biases();
    }
  }
}

void DetectImpl::init_biases() {
  // Upstream Detect.bias_init (ultralytics/nn/modules/head.py):
  //   reg.bias = 2.0
  //   cls.bias[:nc] = log(5 / nc / (640 / stride[i])^2)   per-level
  // The cls formula encodes "5 objects per 640px image, uniform over
  // nc classes and (640/stride)^2 anchors" — i.e. the calibrated
  // anchor-density-aware sigmoid prior. Using a universal value like
  // log(0.01/0.99) ≈ −4.6 produces a 1 % prior across all strides,
  // which is ~60× too high for stride-8 anchors (6400 of them per
  // image → ~1.5e-4 prior). The mismatched prior makes the BCE on
  // background anchors balloon at epoch 0 and the cosine LR runs
  // out before the head can dig out — exactly the 0.0001 mAP / 0.32
  // mAP failure modes we hit before. We honour the upstream formula
  // verbatim. `stride` is populated by the parent on first forward;
  // at load_from_state_dict time it's still empty, so default to
  // [8, 16, 32] for nl=3 (the v3/v5/v8/v9/v11/v12/v13 detect form).
  torch::NoGradGuard ng;
  std::vector<double> s = stride;
  if ((int)s.size() != nl) {
    s.clear();
    if (nl == 3)      s = {8.0, 16.0, 32.0};
    else if (nl == 4) s = {4.0, 8.0, 16.0, 32.0};   // P2–P5 head
    else if (nl == 2) s = {8.0, 16.0};
    else for (int i = 0; i < nl; ++i) s.push_back(8.0 * (1 << i));
  }
  const float kImgsz = 640.0f;
  for (int i = 0; i < nl; ++i) {
    auto* reg = cv2[i]->as<torch::nn::SequentialImpl>();
    auto* cls = cv3[i]->as<torch::nn::SequentialImpl>();
    if (reg && reg->size() >= 3) {
      if (auto* last = (*reg)[reg->size() - 1]->as<torch::nn::Conv2dImpl>()) {
        if (last->bias.defined()) last->bias.fill_(2.0f);
      }
    }
    if (cls && cls->size() >= 3) {
      if (auto* last = (*cls)[cls->size() - 1]->as<torch::nn::Conv2dImpl>()) {
        if (last->bias.defined() && nc > 0) {
          const float anc_per_lvl = (kImgsz / (float)s[(std::size_t)i]);
          const float cls_bias_i =
              std::log(5.0f / (float)nc / (anc_per_lvl * anc_per_lvl));
          last->bias.slice(0, 0, nc).fill_(cls_bias_i);
        }
      }
    }
  }
}

torch::Tensor DetectImpl::decode(const std::vector<torch::Tensor>& feats) {
  // feats[i]: [N, no, h_i, w_i]
  TORCH_CHECK((int)stride.size() == nl, "Detect.stride not set");

  std::vector<torch::Tensor> flat_pred;
  std::vector<torch::Tensor> anchors;   // shape [A, 2] in input pixels
  std::vector<torch::Tensor> str_t;     // shape [A] strides

  for (int i = 0; i < nl; ++i) {
    auto t = feats[i];
    auto N = t.size(0);
    auto h = t.size(2);
    auto w = t.size(3);
    flat_pred.push_back(t.view({N, no, h * w}));

    // Build anchors at cell centers (sx, sy) = (j + 0.5, i + 0.5).
    auto opts = torch::TensorOptions().device(t.device()).dtype(t.dtype());
    auto sy = torch::arange(h, opts) + 0.5;
    auto sx = torch::arange(w, opts) + 0.5;
    auto grid_y = sy.view({h, 1}).expand({h, w});
    auto grid_x = sx.view({1, w}).expand({h, w});
    auto anc = torch::stack({grid_x, grid_y}, /*dim=*/-1).view({h * w, 2});
    anchors.push_back(anc);
    str_t.push_back(torch::full({h * w}, stride[i], opts));
  }
  auto pred = torch::cat(flat_pred, /*dim=*/2);            // [N, no, A]
  auto anc  = torch::cat(anchors,    /*dim=*/0);           // [A, 2]
  auto strd = torch::cat(str_t,      /*dim=*/0);           // [A]

  auto box = pred.slice(/*dim=*/1, 0,             4 * reg_max);
  auto cls = pred.slice(/*dim=*/1, 4 * reg_max,   no);

  // box: [N, 4*reg_max, A] → DFL → [N, 4, A] (l, t, r, b)
  auto dist = dfl(box);  // distances in feature units
  auto N    = dist.size(0);
  auto A    = dist.size(2);

  auto lt = dist.slice(/*dim=*/1, 0, 2);  // [N, 2, A]
  auto rb = dist.slice(/*dim=*/1, 2, 4);
  auto x1y1 = anc.transpose(0, 1).unsqueeze(0) - lt;       // anchors broadcast
  auto x2y2 = anc.transpose(0, 1).unsqueeze(0) + rb;
  auto xyxy = torch::cat({x1y1, x2y2}, /*dim=*/1) * strd;  // → input pixels

  cls = cls.sigmoid();
  return torch::cat({xyxy, cls}, /*dim=*/1);  // [N, 4 + nc, A]
}

// ─── Yolo8DetectImpl ──────────────────────────────────────────────────────
//
// Builds layers in the exact order of yolo8.yaml so that registration order
// = state_dict iteration order = pickle traversal order.

namespace {
// Upstream yaml decoded for v8 detect:
//   from   = which previous outputs feed this layer (-1 = previous)
//   module = layer kind
//   args   = (out_channels [pre-scale], extra...)
//   For C2f: (out_channels, n_repeats, shortcut)
//   Concat: from list, e.g. [-1, 6]
//   Detect: from list of [P3_idx, P4_idx, P5_idx] = [15, 18, 21]
struct LayerSpec {
  std::vector<int> from;        // -1 means the immediately previous layer
  std::string      kind;        // "Conv", "C2f", "SPPF", "Upsample", "Concat", "Detect"
  // Conv args: (c_out, k, s)
  // C2f  args: (c_out, n, shortcut)
  // SPPF args: (c_out, k)
  // Concat args: (dim)
  // Detect args: (nc)
  std::vector<int> a;
};

const std::vector<LayerSpec>& v8_yaml() {
  static const std::vector<LayerSpec> y = {
      // backbone (10 layers)
      {{-1}, "Conv",     {64,  3, 2}},   // 0
      {{-1}, "Conv",     {128, 3, 2}},   // 1
      {{-1}, "C2f",      {128, 3, 1}},   // 2  (shortcut=true)
      {{-1}, "Conv",     {256, 3, 2}},   // 3
      {{-1}, "C2f",      {256, 6, 1}},   // 4
      {{-1}, "Conv",     {512, 3, 2}},   // 5
      {{-1}, "C2f",      {512, 6, 1}},   // 6
      {{-1}, "Conv",     {1024,3, 2}},   // 7
      {{-1}, "C2f",      {1024,3, 1}},   // 8
      {{-1}, "SPPF",     {1024,5}},      // 9
      // head (13 layers)
      {{-1},     "Upsample", {2}},                     // 10 scale=2
      {{-1, 6},  "Concat",   {1}},                     // 11
      {{-1},     "C2f",      {512, 3, 0}},             // 12 shortcut=false
      {{-1},     "Upsample", {2}},                     // 13
      {{-1, 4},  "Concat",   {1}},                     // 14
      {{-1},     "C2f",      {256, 3, 0}},             // 15  P3
      {{-1},     "Conv",     {256, 3, 2}},             // 16
      {{-1, 12}, "Concat",   {1}},                     // 17
      {{-1},     "C2f",      {512, 3, 0}},             // 18  P4
      {{-1},     "Conv",     {512, 3, 2}},             // 19
      {{-1, 9},  "Concat",   {1}},                     // 20
      {{-1},     "C2f",      {1024,3, 0}},             // 21  P5
      {{15, 18, 21}, "Detect", {}},                    // 22
  };
  return y;
}
}  // namespace

Yolo8DetectImpl::Yolo8DetectImpl(Yolo8Scale s, int nc_) : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());

  const auto& yaml = v8_yaml();
  std::vector<int> ch;     // tracked output channels per layer index
  ch.reserve(yaml.size());
  int c_in = 3;            // image channels for layer 0

  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& spec = yaml[i];

    // Resolve input channels.
    int in_ch = 0;
    if (spec.kind == "Concat") {
      for (int f : spec.from) {
        int idx = (f == -1) ? (int)i - 1 : f;
        in_ch += ch[idx];
      }
    } else {
      int f = spec.from[0];
      int idx = (f == -1) ? (int)i - 1 : f;
      in_ch = (idx == -1) ? c_in : ch[idx];
    }

    if (spec.kind == "Conv") {
      int c_out = scale_channels(spec.a[0], scale);
      int k = spec.a[1], stride = spec.a[2];
      model->push_back(Conv(in_ch, c_out, k, stride));
      ch.push_back(c_out);
    } else if (spec.kind == "C2f") {
      int c_out = scale_channels(spec.a[0], scale);
      int n     = scale_depth(spec.a[1], scale);
      bool shortcut = (spec.a[2] != 0);
      model->push_back(C2f(in_ch, c_out, n, shortcut));
      ch.push_back(c_out);
    } else if (spec.kind == "SPPF") {
      int c_out = scale_channels(spec.a[0], scale);
      int k = spec.a[1];
      model->push_back(SPPF(in_ch, c_out, k));
      ch.push_back(c_out);
    } else if (spec.kind == "Upsample") {
      // Stateless — register as Identity placeholder so model[i] is valid.
      model->push_back(torch::nn::Upsample(
          torch::nn::UpsampleOptions()
              .scale_factor(std::vector<double>{(double)spec.a[0], (double)spec.a[0]})
              .mode(torch::kNearest)));
      ch.push_back(in_ch);
    } else if (spec.kind == "Concat") {
      // Stateless — register Identity for index alignment.
      model->push_back(torch::nn::Identity());
      ch.push_back(in_ch);
    } else if (spec.kind == "Detect") {
      std::vector<int> det_ch;
      for (int f : spec.from) det_ch.push_back(ch[f]);
      model->push_back(Detect(nc, det_ch));
      ch.push_back(0);
    } else {
      throw std::runtime_error("unknown layer kind: " + spec.kind);
    }
  }

  // Compute strides: run a dummy forward at a small input and measure
  // each detect-input feature map's spatial reduction.
  {
    torch::NoGradGuard ng;
    this->eval();
    auto x  = torch::zeros({1, 3, 256, 256});
    std::vector<torch::Tensor> outs(yaml.size());
    int   img_h = 256;
    for (size_t i = 0; i < yaml.size(); ++i) {
      const auto& spec = yaml[i];
      torch::Tensor in;
      if (spec.kind == "Concat") {
        std::vector<torch::Tensor> parts;
        for (int f : spec.from) parts.push_back(outs[f == -1 ? (int)i - 1 : f]);
        in = torch::cat(parts, 1);
      } else if (spec.kind == "Detect") {
        std::vector<torch::Tensor> det_in;
        for (int f : spec.from) det_in.push_back(outs[f]);
        // Compute strides from each input.
        std::vector<double> strides;
        for (auto& t : det_in) strides.push_back((double)img_h / (double)t.size(2));
        auto* d = model[i]->as<DetectImpl>();
        d->stride = strides;
        stride    = strides;
        outs[i]   = torch::zeros({1});
        continue;
      } else {
        int f = spec.from[0];
        in = (f == -1) ? (i == 0 ? x : outs[i - 1]) : outs[f];
      }
      // Apply layer.
      if (spec.kind == "Conv")        outs[i] = model[i]->as<ConvImpl>()->forward(in);
      else if (spec.kind == "C2f")    outs[i] = model[i]->as<C2fImpl>()->forward(in);
      else if (spec.kind == "SPPF")   outs[i] = model[i]->as<SPPFImpl>()->forward(in);
      else if (spec.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
      else if (spec.kind == "Concat")   outs[i] = in;  // already concatenated
    }
    this->train();
  }

  // Initialize the Detect head's biases to be near-zero "no object" priors.
  // Upstream: cv3 final 1x1 conv bias init to log((1 - 0.01) / 0.01) ≈ 4.6
  // is NOT what they do — they set bias to log(class_freq * stride / 8) etc.
  // For from-scratch training we leave default; for loaded weights this is
  // overwritten anyway.
}

std::vector<torch::Tensor> Yolo8DetectImpl::forward_train(torch::Tensor x) {
  const auto& yaml = v8_yaml();
  std::vector<torch::Tensor> outs(yaml.size());
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& spec = yaml[i];
    torch::Tensor in;
    if (spec.kind == "Concat") {
      std::vector<torch::Tensor> parts;
      for (int f : spec.from) parts.push_back(outs[f == -1 ? (int)i - 1 : f]);
      in = torch::cat(parts, 1);
    } else if (spec.kind == "Detect") {
      std::vector<torch::Tensor> det_in;
      for (int f : spec.from) det_in.push_back(outs[f]);
      auto* d = model[i]->as<DetectImpl>();
      return d->forward_features(det_in);
    } else {
      int f = spec.from[0];
      in = (f == -1) ? (i == 0 ? x : outs[i - 1]) : outs[f];
    }
    if (spec.kind == "Conv")        outs[i] = model[i]->as<ConvImpl>()->forward(in);
    else if (spec.kind == "C2f")    outs[i] = model[i]->as<C2fImpl>()->forward(in);
    else if (spec.kind == "SPPF")   outs[i] = model[i]->as<SPPFImpl>()->forward(in);
    else if (spec.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (spec.kind == "Concat")   outs[i] = in;
  }
  TORCH_CHECK(false, "unreachable");
}

torch::Tensor Yolo8DetectImpl::forward_eval(torch::Tensor x) {
  auto feats = forward_train(x);
  // Last layer is Detect — pull it out and decode.
  const auto& yaml = v8_yaml();
  auto* d = model[yaml.size() - 1]->as<DetectImpl>();
  return d->decode(feats);
}

std::vector<std::string> Yolo8DetectImpl::state_keys() const {
  std::vector<std::string> keys;
  for (const auto& kv : this->named_parameters())
    keys.push_back(kv.key());
  for (const auto& kv : this->named_buffers())
    keys.push_back(kv.key());
  return keys;
}

std::vector<torch::Tensor*> Yolo8DetectImpl::state_tensors() {
  std::vector<torch::Tensor*> tensors;
  for (auto& kv : this->named_parameters())
    tensors.push_back(&kv.value());
  for (auto& kv : this->named_buffers())
    tensors.push_back(&kv.value());
  return tensors;
}

int Yolo8DetectImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  // named_parameters() returns OrderedDict by value, so keep one alive for
  // the duration of the copy. The tensors inside are reference-typed —
  // copy_ on them updates the actual underlying model storage.
  auto params = this->named_parameters();
  auto buffs  = this->named_buffers();

  std::unordered_map<std::string, at::Tensor> ours;
  for (auto& kv : params) ours.emplace(kv.key(), kv.value());
  for (auto& kv : buffs)  ours.emplace(kv.key(), kv.value());

  torch::NoGradGuard ng;
  int copied = 0, skipped_shape = 0;
  for (const auto& [k, t] : entries) {
    auto it = ours.find(k);
    if (it == ours.end()) continue;
    auto& dst = it->second;
    if (dst.sizes() != t.sizes()) {
      ++skipped_shape;
      continue;
    }
    auto src = t.to(dst.dtype()).to(dst.device());
    dst.copy_(src);
    ++copied;
  }
  if (copied == 0)
    throw std::runtime_error("load_from_state_dict: copied 0 tensors");
  if (skipped_shape > 0) {
    std::cerr << "[yolo8 load] skipped " << skipped_shape
              << " tensors with shape mismatch (cls head re-purposed for custom nc); "
                 "re-initialising detect biases to the 1% sigmoid prior\n";
    init_detect_biases(this);
  }
  return copied;
}

}  // namespace yolocpp::models
