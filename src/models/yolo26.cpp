#include "yolocpp/models/yolo26.hpp"

#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace yolocpp::models {

namespace F = torch::nn::functional;

// ─── Scale helpers ────────────────────────────────────────────────────────

int scale_channels_v26(int c, const Yolo26Scale& s) {
  c = std::min(c, s.max_channels);
  auto make_divisible = [](double v, int divisor) {
    return std::max(divisor, (int)std::round(v / divisor) * divisor);
  };
  return make_divisible(c * s.width_multiple, 8);
}

int scale_depth_v26(int n, const Yolo26Scale& s) {
  return std::max(1, (int)std::round(n * s.depth_multiple));
}

Yolo26Scale yolo26_scale_from_letter(const std::string& letter) {
  if (letter == "n") return kYolo26n;
  if (letter == "s") return kYolo26s;
  if (letter == "m") return kYolo26m;
  if (letter == "l") return kYolo26l;
  if (letter == "x") return kYolo26x;
  throw std::runtime_error("yolo26: unknown scale letter '" + letter + "'");
}

Yolo26Scale yolo26_scale_from_filename(const std::string& path) {
  std::filesystem::path p(path);
  std::string base = p.filename().string();
  static const std::regex re(R"(yolo26([nsmlx])(?:-(?:cls|seg|pose|obb))?\.pt$)");
  std::smatch m;
  if (std::regex_search(base, m, re)) return yolo26_scale_from_letter(m[1].str());
  return kYolo26n;
}

// ─── C2PSAf ───────────────────────────────────────────────────────────────
// CSP-shaped block whose inner module is `Sequential(Bottleneck, PSABlock)`.
// State-dict layout matches the shipped yolo26<x>.pt at layer 22.

C2PSAfImpl::C2PSAfImpl(int c1, int c2, int n, double e) {
  c_inner = (int)((double)c2 * e);
  cv1 = register_module("cv1", Conv(c1, 2 * c_inner, 1, 1));
  cv2 = register_module("cv2", Conv((2 + n) * c_inner, c2, 1));
  m   = register_module("m", torch::nn::ModuleList());
  // PSABlock num_heads rule: max(1, c_inner / 64) (mirrors C2PSA's rule).
  int nh = std::max(1, c_inner / 64);
  for (int i = 0; i < n; ++i) {
    auto seq = torch::nn::Sequential();
    // Bottleneck: k=(3,3), e=0.5, shortcut=true (c1==c2 ensures shortcut active).
    seq->push_back(Bottleneck(c_inner, c_inner, /*shortcut=*/true,
                              /*g=*/1, /*e=*/0.5,
                              std::array<int, 2>{3, 3}));
    seq->push_back(PSABlock(c_inner, /*attn_ratio=*/0.5,
                            /*num_heads=*/nh, /*shortcut=*/true));
    m->push_back(seq);
  }
}

torch::Tensor C2PSAfImpl::forward(torch::Tensor x) {
  auto y = cv1(x).chunk(2, /*dim=*/1);
  std::vector<torch::Tensor> outs;
  outs.reserve(2 + m->size());
  outs.push_back(y[0]);
  outs.push_back(y[1]);
  for (size_t i = 0; i < m->size(); ++i) {
    outs.push_back(m[i]->as<torch::nn::SequentialImpl>()->forward(outs.back()));
  }
  return cv2(torch::cat(outs, /*dim=*/1));
}

// ─── Detect26 ─────────────────────────────────────────────────────────────
//
// DFL-free head. cv2 outputs 4 channels (l, t, r, b distances in feature
// units) directly; cv3 outputs nc class logits. Both branches use v11-style
// depthwise-separable nested DWConvBlock pairs (mobile-friendly).

Detect26Impl::Detect26Impl(int nc_, std::vector<int> ch_)
    : nc(nc_), ch(std::move(ch_)) {
  nl = (int)ch.size();
  no = 4 + nc;

  // Channel sizing rule (matches the shipped yolo26<x>.pt):
  //   c2 = max(16, ch[0]/4)               (no reg_max*4 since DFL-free)
  //   c3 = max(ch[0], min(nc, 100))
  int c2 = std::max({16, ch[0] / 4});
  int c3 = std::max(ch[0], std::min(nc, 100));

  cv2         = register_module("cv2",         torch::nn::ModuleList());
  cv3         = register_module("cv3",         torch::nn::ModuleList());
  one2one_cv2 = register_module("one2one_cv2", torch::nn::ModuleList());
  one2one_cv3 = register_module("one2one_cv3", torch::nn::ModuleList());

  // Both heads have IDENTICAL topology — only their training-time
  // assignment differs. Building them in lockstep so the upstream
  // checkpoint's `cv2/cv3/one2one_cv2/one2one_cv3` keys bind directly
  // without any remap.
  auto build_reg = [&]() {
    auto reg = torch::nn::Sequential();
    reg->push_back(Conv(ch[0], c2, 3));   // ch[0] placeholder, overwritten below per-level
    return reg;
  };
  (void)build_reg;  // suppress unused if future refactor removes per-level builders

  for (int i = 0; i < nl; ++i) {
    // ── one2many head ──────────────────────────────────────────────
    // cv2 (regression — regular Conv→Conv→Conv2d, NOT depthwise; matches
    // the v8 legacy form but with output 4 instead of 4*reg_max).
    {
      auto reg = torch::nn::Sequential();
      reg->push_back(Conv(ch[i], c2, 3));
      reg->push_back(Conv(c2,    c2, 3));
      reg->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(c2, 4, 1)));
      cv2->push_back(reg);
    }
    // cv3 (classification — depthwise-separable, v11-style nested DWConv).
    {
      auto cls = torch::nn::Sequential();
      cls->push_back(DWConvBlock(ch[i], c3, /*k_dw=*/3));
      cls->push_back(DWConvBlock(c3,    c3, /*k_dw=*/3));
      cls->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(c3, nc, 1)));
      cv3->push_back(cls);
    }
    // ── one2one head — same topology, separate parameters ───────────
    {
      auto reg = torch::nn::Sequential();
      reg->push_back(Conv(ch[i], c2, 3));
      reg->push_back(Conv(c2,    c2, 3));
      reg->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(c2, 4, 1)));
      one2one_cv2->push_back(reg);
    }
    {
      auto cls = torch::nn::Sequential();
      cls->push_back(DWConvBlock(ch[i], c3, /*k_dw=*/3));
      cls->push_back(DWConvBlock(c3,    c3, /*k_dw=*/3));
      cls->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(c3, nc, 1)));
      one2one_cv3->push_back(cls);
    }
  }
}

void Detect26Impl::init_biases() {
  // Upstream detection-prior init: cls head's final 1×1 conv bias is
  // log(p / (1 - p)) for the prior probability p ≈ 0.01 → −4.595.
  // Reg head's final bias is set to 1.0 so predicted (l, t, r, b)
  // offsets start at 1 cell wide. Applied to BOTH heads (o2m + o2o).
  torch::NoGradGuard ng;
  const float cls_bias_init = std::log(0.01f / (1.0f - 0.01f));  // ≈ −4.595
  auto init_branch = [&](torch::nn::ModuleList& reg_ml,
                          torch::nn::ModuleList& cls_ml) {
    for (int i = 0; i < nl; ++i) {
      auto* reg = reg_ml[i]->as<torch::nn::SequentialImpl>();
      auto* cls = cls_ml[i]->as<torch::nn::SequentialImpl>();
      if (reg && reg->size() >= 3) {
        if (auto* last = (*reg)[reg->size() - 1]->as<torch::nn::Conv2dImpl>()) {
          if (last->bias.defined()) last->bias.fill_(1.0f);
        }
      }
      if (cls && cls->size() >= 3) {
        if (auto* last = (*cls)[cls->size() - 1]->as<torch::nn::Conv2dImpl>()) {
          if (last->bias.defined()) last->bias.fill_(cls_bias_init);
        }
      }
    }
  };
  init_branch(cv2, cv3);                  // o2m head
  init_branch(one2one_cv2, one2one_cv3);  // o2o head
}

std::vector<torch::Tensor>
Detect26Impl::forward_features(std::vector<torch::Tensor> x) {
  // Returns 2*nl tensors:
  //   indices [0 .. nl-1]      → one2many head (cv2/cv3)
  //   indices [nl .. 2*nl-1]   → one2one head (one2one_cv2/cv3)
  // Both heads see the same feature inputs `x[i]` (shared backbone).
  std::vector<torch::Tensor> out;
  out.reserve(2 * nl);
  // First pass: one2many head.
  for (int i = 0; i < nl; ++i) {
    auto* reg = cv2[i]->as<torch::nn::SequentialImpl>();
    auto* cls = cv3[i]->as<torch::nn::SequentialImpl>();
    auto a = reg->forward(x[i]);   // [N, 4,  h, w]
    auto b = cls->forward(x[i]);   // [N, nc, h, w]
    out.push_back(torch::cat({a, b}, /*dim=*/1));  // [N, 4+nc, h, w]
  }
  // Second pass: one2one head.
  for (int i = 0; i < nl; ++i) {
    auto* reg = one2one_cv2[i]->as<torch::nn::SequentialImpl>();
    auto* cls = one2one_cv3[i]->as<torch::nn::SequentialImpl>();
    auto a = reg->forward(x[i]);
    auto b = cls->forward(x[i]);
    out.push_back(torch::cat({a, b}, /*dim=*/1));
  }
  return out;
}

torch::Tensor Detect26Impl::decode(const std::vector<torch::Tensor>& feats) {
  TORCH_CHECK((int)stride.size() == nl, "Detect26.stride not set");
  // Decode uses ONLY the one2one head (NMS-free at inference). When
  // `forward_features` returns 2*nl tensors (post the dual-head
  // upgrade), slice off the o2o tail; otherwise treat the input as
  // already-o2o for backward compat with single-head callers.
  const int feat_off = ((int)feats.size() == 2 * nl) ? nl : 0;

  std::vector<torch::Tensor> flat_pred;
  std::vector<torch::Tensor> anchors;
  std::vector<torch::Tensor> str_t;

  for (int i = 0; i < nl; ++i) {
    auto t = feats[feat_off + i];
    auto N = t.size(0);
    auto h = t.size(2);
    auto w = t.size(3);
    flat_pred.push_back(t.view({N, no, h * w}));

    auto opts = torch::TensorOptions().device(t.device()).dtype(t.dtype());
    auto sy = torch::arange(h, opts) + 0.5;
    auto sx = torch::arange(w, opts) + 0.5;
    auto grid_y = sy.view({h, 1}).expand({h, w});
    auto grid_x = sx.view({1, w}).expand({h, w});
    auto anc = torch::stack({grid_x, grid_y}, /*dim=*/-1).view({h * w, 2});
    anchors.push_back(anc);
    str_t.push_back(torch::full({h * w}, stride[i], opts));
  }
  auto pred = torch::cat(flat_pred, /*dim=*/2);            // [N, 4+nc, A]
  auto anc  = torch::cat(anchors,    /*dim=*/0);           // [A, 2]
  auto strd = torch::cat(str_t,      /*dim=*/0);           // [A]

  auto box = pred.slice(/*dim=*/1, 0,  4);                 // [N, 4, A]
  auto cls = pred.slice(/*dim=*/1, 4,  no);                // [N, nc, A]

  // Constrain regression to non-negative (l, t, r, b distances). softplus
  // gives a smooth equivalent of the v8 DFL output range.
  box = F::softplus(box);

  auto lt = box.slice(/*dim=*/1, 0, 2);   // [N, 2, A]
  auto rb = box.slice(/*dim=*/1, 2, 4);
  auto x1y1 = anc.transpose(0, 1).unsqueeze(0) - lt;       // anchors broadcast
  auto x2y2 = anc.transpose(0, 1).unsqueeze(0) + rb;
  auto xyxy = torch::cat({x1y1, x2y2}, /*dim=*/1) * strd;  // → input pixels

  cls = cls.sigmoid();
  return torch::cat({xyxy, cls}, /*dim=*/1);               // [N, 4+nc, A]
}

// ─── Yolo26DetectImpl ─────────────────────────────────────────────────────
//
// Reuses v11's yaml topology and module set (Conv / C3k2 / SPPF / C2PSA);
// only the final Detect at idx 23 is replaced with Detect26.

namespace {

struct LSpec {
  std::vector<int> from;
  std::string      kind;      // Conv / C3k2 / SPPF / C2PSA / Upsample / Concat / Detect
  std::vector<int> a;
};

const std::vector<LSpec>& v26_yaml() {
  // Differences from v11_yaml():
  //   - Layers 13, 16, 19: c3k=1 (v11 had c3k=0; force-True only at m/l/x).
  //   - Layer 22: new C2PSAf module with base n=1 (v11 was C3k2(1024, 2, 1)).
  //   The same v11 force-c3k=True rule for width >= 1.0 still applies (so
  //   v26m/l/x get c3k=True at layers 2, 4 too — matches the dump).
  //   YAML arg encoding for C2PSAf: (c_out, n, e_x100).
  static const std::vector<LSpec> y = {
      {{-1}, "Conv",   {64,   3, 2}},                  // 0
      {{-1}, "Conv",   {128,  3, 2}},                  // 1
      {{-1}, "C3k2",   {256,  2, 0, 25}},              // 2
      {{-1}, "Conv",   {256,  3, 2}},                  // 3
      {{-1}, "C3k2",   {512,  2, 0, 25}},              // 4
      {{-1}, "Conv",   {512,  3, 2}},                  // 5
      {{-1}, "C3k2",   {512,  2, 1, 50}},              // 6
      {{-1}, "Conv",   {1024, 3, 2}},                  // 7
      {{-1}, "C3k2",   {1024, 2, 1, 50}},              // 8
      {{-1}, "SPPF",   {1024, 5}},                     // 9
      {{-1}, "C2PSA",  {1024, 2}},                     // 10
      {{-1},     "Upsample", {2}},                     // 11
      {{-1, 6},  "Concat",   {1}},                     // 12
      {{-1},     "C3k2",     {512,  2, 1, 50}},        // 13  c3k=True
      {{-1},     "Upsample", {2}},                     // 14
      {{-1, 4},  "Concat",   {1}},                     // 15
      {{-1},     "C3k2",     {256,  2, 1, 50}},        // 16  c3k=True
      {{-1},     "Conv",     {256,  3, 2}},            // 17
      {{-1, 13}, "Concat",   {1}},                     // 18
      {{-1},     "C3k2",     {512,  2, 1, 50}},        // 19  c3k=True
      {{-1},     "Conv",     {512,  3, 2}},            // 20
      {{-1, 10}, "Concat",   {1}},                     // 21
      {{-1},     "C2PSAf",   {1024, 1, 50}},           // 22  new module, n=1
      {{16, 19, 22}, "Detect", {}},                    // 23
  };
  return y;
}

void build_layer(torch::nn::ModuleList& model, const LSpec& spec, int in_ch,
                 int& c_out_out, Yolo26Scale scale) {
  if (spec.kind == "Conv") {
    int c_out = scale_channels_v26(spec.a[0], scale);
    int k = spec.a[1], st = spec.a[2];
    model->push_back(Conv(in_ch, c_out, k, st));
    c_out_out = c_out;
  } else if (spec.kind == "C3k2") {
    int c_out = scale_channels_v26(spec.a[0], scale);
    int n     = scale_depth_v26(spec.a[1], scale);
    bool c3k  = (spec.a[2] != 0);
    if (scale.width_multiple >= 1.0) c3k = true;  // m/l/x force-upgrade (v11 rule)
    double e  = (double)spec.a[3] / 100.0;
    model->push_back(C3k2(in_ch, c_out, n, c3k, e));
    c_out_out = c_out;
  } else if (spec.kind == "SPPF") {
    int c_out = scale_channels_v26(spec.a[0], scale);
    // v26 SPPF: cv1 has Identity activation (not SiLU) AND adds a residual
    // shortcut (out + input) when c1 == c2.
    model->push_back(SPPF(in_ch, c_out, spec.a[1],
                          /*cv1_act=*/false, /*shortcut=*/true));
    c_out_out = c_out;
  } else if (spec.kind == "C2PSA") {
    int c_out = scale_channels_v26(spec.a[0], scale);
    int n     = scale_depth_v26(spec.a[1], scale);
    model->push_back(C2PSA(in_ch, c_out, n, /*e=*/0.5));
    c_out_out = c_out;
  } else if (spec.kind == "C2PSAf") {
    int c_out = scale_channels_v26(spec.a[0], scale);
    int n     = scale_depth_v26(spec.a[1], scale);
    double e  = (double)spec.a[2] / 100.0;
    model->push_back(C2PSAf(in_ch, c_out, n, e));
    c_out_out = c_out;
  } else if (spec.kind == "Upsample") {
    model->push_back(torch::nn::Upsample(
        torch::nn::UpsampleOptions()
            .scale_factor(std::vector<double>{(double)spec.a[0],
                                              (double)spec.a[0]})
            .mode(torch::kNearest)));
    c_out_out = in_ch;
  } else if (spec.kind == "Concat") {
    model->push_back(torch::nn::Identity());
    c_out_out = in_ch;
  } else if (spec.kind == "Detect") {
    c_out_out = 0;
  } else {
    throw std::runtime_error("yolo26: unknown layer kind: " + spec.kind);
  }
}

void forward_through(const std::vector<LSpec>& yaml,
                     torch::nn::ModuleList& model, torch::Tensor x,
                     std::vector<torch::Tensor>& outs) {
  outs.assign(yaml.size(), torch::Tensor());
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    if (s.kind == "Detect") {
      outs[i] = torch::Tensor();
      continue;
    }
    torch::Tensor in;
    if (s.kind == "Concat") {
      std::vector<torch::Tensor> parts;
      for (int f : s.from) parts.push_back(outs[f == -1 ? (int)i - 1 : f]);
      in = torch::cat(parts, /*dim=*/1);
    } else {
      int f = s.from[0];
      in = (f == -1) ? (i == 0 ? x : outs[i - 1]) : outs[f];
    }
    if      (s.kind == "Conv")     outs[i] = model[i]->as<ConvImpl>()->forward(in);
    else if (s.kind == "C3k2")     outs[i] = model[i]->as<C3k2Impl>()->forward(in);
    else if (s.kind == "SPPF")     outs[i] = model[i]->as<SPPFImpl>()->forward(in);
    else if (s.kind == "C2PSA")    outs[i] = model[i]->as<C2PSAImpl>()->forward(in);
    else if (s.kind == "C2PSAf")   outs[i] = model[i]->as<C2PSAfImpl>()->forward(in);
    else if (s.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "Concat")   outs[i] = in;
  }
}

}  // anonymous namespace

Yolo26DetectImpl::Yolo26DetectImpl(Yolo26Scale s, int nc_) : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());

  const auto& yaml = v26_yaml();
  std::vector<int> ch;
  ch.reserve(yaml.size());
  int c_in = 3;

  for (size_t i = 0; i < yaml.size() - 1; ++i) {
    const auto& spec = yaml[i];
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
    int c_out = 0;
    build_layer(model, spec, in_ch, c_out, scale);
    ch.push_back(c_out);
  }

  const auto& det = yaml.back();
  std::vector<int> det_ch;
  for (int f : det.from) det_ch.push_back(ch[f]);
  model->push_back(Detect26(nc, det_ch));
  ch.push_back(0);

  {
    torch::NoGradGuard ng;
    this->eval();
    auto x = torch::zeros({1, 3, 256, 256});
    std::vector<torch::Tensor> outs;
    forward_through(yaml, model, x, outs);
    std::vector<double> strides;
    for (int f : det.from)
      strides.push_back(256.0 / (double)outs[f].size(2));
    auto* d = model[yaml.size() - 1]->as<Detect26Impl>();
    d->stride = strides;
    stride    = strides;
    this->train();
  }
}

std::vector<torch::Tensor> Yolo26DetectImpl::forward_train(torch::Tensor x) {
  const auto& yaml = v26_yaml();
  std::vector<torch::Tensor> outs;
  forward_through(yaml, model, x, outs);
  std::vector<torch::Tensor> det_in;
  for (int f : yaml.back().from) det_in.push_back(outs[f]);
  return model[yaml.size() - 1]->as<Detect26Impl>()->forward_features(det_in);
}

torch::Tensor Yolo26DetectImpl::forward_eval(torch::Tensor x) {
  auto feats = forward_train(x);
  return model[v26_yaml().size() - 1]->as<Detect26Impl>()->decode(feats);
}

int Yolo26DetectImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto params = this->named_parameters();
  auto buffs  = this->named_buffers();
  std::unordered_map<std::string, at::Tensor> ours;
  for (auto& kv : params) ours.emplace(kv.key(), kv.value());
  for (auto& kv : buffs)  ours.emplace(kv.key(), kv.value());

  // Our Detect26Impl now has BOTH heads (cv2/cv3 for one2many,
  // one2one_cv2/cv3 for one2one) — same naming as upstream — so the
  // checkpoint's keys bind directly without any remap.

  torch::NoGradGuard ng;
  int copied = 0, skipped_shape = 0;
  for (const auto& [k, t] : entries) {
    auto it = ours.find(k);
    if (it == ours.end()) continue;
    auto& dst = it->second;
    if (dst.sizes() != t.sizes()) {
      // Shape mismatch — typically the cls head (cv3 final conv) when
      // re-purposing nc=80 upstream weights for a custom nc. Skip and
      // leave the destination at its torch-default init; the trainer
      // will re-initialize cls bias with the detection prior via
      // `Detect26Impl::init_biases()` post-load. Logging only the count
      // so it doesn't drown the training log.
      ++skipped_shape;
      continue;
    }
    dst.copy_(t.to(dst.dtype()).to(dst.device()));
    ++copied;
  }
  if (copied == 0)
    throw std::runtime_error("yolo26 load: copied 0 tensors");
  if (skipped_shape > 0) {
    std::cerr << "[yolo26 load] skipped " << skipped_shape
              << " tensors with shape mismatch "
                 "(usually cls head re-purposed for custom nc)\n";
  }
  // Apply detection-prior init to whatever cls/reg heads were skipped.
  auto* det = model[v26_yaml().size() - 1]->as<Detect26Impl>();
  if (det) det->init_biases();
  return copied;
}

// ─── Legacy stub holder ───────────────────────────────────────────────────

Yolo26Impl::Yolo26Impl(int nc_) : nc(nc_) {}

std::vector<torch::Tensor> Yolo26Impl::forward(torch::Tensor /*x*/) {
  throw std::runtime_error(
      "Yolo26Impl is a legacy stub — use Yolo26Detect / Yolo26Segment / etc.");
}

}  // namespace yolocpp::models
