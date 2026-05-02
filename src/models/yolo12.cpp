#include "yolocpp/models/yolo12.hpp"

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

int scale_channels_v12(int c, const Yolo12Scale& s) {
  c = std::min(c, s.max_channels);
  auto make_divisible = [](double v, int divisor) {
    return std::max(divisor, (int)std::round(v / divisor) * divisor);
  };
  return make_divisible(c * s.width_multiple, 8);
}

int scale_depth_v12(int n, const Yolo12Scale& s) {
  return std::max(1, (int)std::round(n * s.depth_multiple));
}

Yolo12Scale yolo12_scale_from_letter(const std::string& letter) {
  if (letter == "n") return kYolo12n;
  if (letter == "s") return kYolo12s;
  if (letter == "m") return kYolo12m;
  if (letter == "l") return kYolo12l;
  if (letter == "x") return kYolo12x;
  throw std::runtime_error("yolo12: unknown scale letter '" + letter + "'");
}

Yolo12Scale yolo12_scale_from_filename(const std::string& path) {
  std::filesystem::path p(path);
  std::string base = p.filename().string();
  static const std::regex re(R"(yolo12([nsmlx])(?:-(?:cls|seg|pose|obb))?\.pt$)");
  std::smatch m;
  if (std::regex_search(base, m, re)) return yolo12_scale_from_letter(m[1].str());
  return kYolo12n;
}

// ─── AAttn ────────────────────────────────────────────────────────────────

AAttnImpl::AAttnImpl(int dim, int num_heads_, int area_)
    : num_heads(num_heads_), area(area_) {
  TORCH_CHECK(dim % num_heads == 0,
              "AAttn: dim (", dim, ") must be divisible by num_heads (",
              num_heads, ")");
  head_dim = dim / num_heads;
  scale = std::pow((double)head_dim, -0.5);
  qkv  = register_module("qkv",  Conv(dim, dim * 3, 1, 1, /*p=*/-1, /*g=*/1,   /*act=*/false));
  proj = register_module("proj", Conv(dim, dim,     1, 1, /*p=*/-1, /*g=*/1,   /*act=*/false));
  // pe ships with conv.bias=True in v12; we must opt in (last `true` arg).
  pe   = register_module("pe",   Conv(dim, dim,     7, 1, /*p=*/3,  /*g=*/dim, /*act=*/false, /*conv_bias=*/true));
}

torch::Tensor AAttnImpl::forward(torch::Tensor x) {
  // Mirror of the upstream AAttn.forward (yolo12). Channel layout matters:
  // the qkv conv output's 3C channels are interleaved as
  // [head0(q,k,v), head1(q,k,v), ...] — `view(B, N, nh, 3*hd)` then split
  // along the per-head feature dim. Splitting the OUTER 3C with chunk(3)
  // would swap heads with q/k/v sections and produce wrong outputs.
  auto B = x.size(0);
  auto C = x.size(1);            // = num_heads * head_dim
  auto H = x.size(2);
  auto W = x.size(3);
  auto N = H * W;

  // 1) qkv conv → [B, 3C, H, W] → flatten to [B, N, 3C]
  auto qkv_flat = qkv(x).flatten(2).transpose(1, 2).contiguous();

  // 2) Optional area-windowing: reshape to [B*area, N/area, 3C]
  int64_t Bg = B, Ng = N;
  if (area > 1) {
    qkv_flat = qkv_flat.reshape({B * area, N / area, 3 * C});
    Bg = B * area;
    Ng = N / area;
  }

  // 3) View into per-head slots: [Bg, Ng, num_heads, 3*head_dim]
  //    permute → [Bg, num_heads, 3*head_dim, Ng]
  //    split along dim 2 with sizes [head_dim, head_dim, head_dim] → q, k, v
  auto qkv_h = qkv_flat.view({Bg, Ng, num_heads, 3 * head_dim})
                       .permute({0, 2, 3, 1})
                       .contiguous();
  auto parts = qkv_h.split_with_sizes(
      {(int64_t)head_dim, (int64_t)head_dim, (int64_t)head_dim},
      /*dim=*/2);
  auto q_ = parts[0];        // [Bg, nh, hd, Ng]
  auto k_ = parts[1];
  auto v_ = parts[2];

  // 4) Attention.
  auto attn = torch::matmul(q_.transpose(-2, -1), k_) * scale;   // [Bg, nh, Ng, Ng]
  attn = attn.softmax(/*dim=*/-1);
  auto out = torch::matmul(v_, attn.transpose(-2, -1));           // [Bg, nh, hd, Ng]

  // 5) Reorder back to [Bg, Ng, nh, hd] (matches Python's permute(0,3,1,2)).
  out = out.permute({0, 3, 1, 2}).contiguous();                   // [Bg, Ng, nh, hd]
  auto v_p = v_.permute({0, 3, 1, 2}).contiguous();               // [Bg, Ng, nh, hd]

  // 6) Un-window if area > 1.
  if (area > 1) {
    out = out.reshape({B, N, C});
    v_p = v_p.reshape({B, N, C});
  } else {
    out = out.reshape({B, N, C});
    v_p = v_p.reshape({B, N, C});
  }

  // 7) Reshape to spatial [B, C, H, W].
  out = out.reshape({B, H, W, C}).permute({0, 3, 1, 2}).contiguous();
  v_p = v_p.reshape({B, H, W, C}).permute({0, 3, 1, 2}).contiguous();

  // 8) Add positional encoding on V, then project.
  out = out + pe(v_p);
  return proj(out);
}

// ─── ABlock ───────────────────────────────────────────────────────────────

ABlockImpl::ABlockImpl(int dim, int num_heads, double mlp_ratio, int area_) {
  attn = register_module("attn", AAttn(dim, num_heads, area_));
  int mlp_h = (int)((double)dim * mlp_ratio);
  mlp = register_module("mlp", torch::nn::Sequential());
  mlp->push_back(Conv(dim,   mlp_h, 1, 1));
  mlp->push_back(Conv(mlp_h, dim,   1, 1, /*p=*/-1, /*g=*/1, /*act=*/false));
}

torch::Tensor ABlockImpl::forward(torch::Tensor x) {
  x = x + attn(x);
  x = x + mlp->forward(x);
  return x;
}

// ─── A2C2f ────────────────────────────────────────────────────────────────

A2C2fImpl::A2C2fImpl(int c1, int c2, int n, bool a2_, int area,
                     bool residual_, double mlp_ratio, double e)
    : a2(a2_), residual(residual_) {
  c_inner = (int)((double)c2 * e);
  cv1 = register_module("cv1", Conv(c1, c_inner, 1, 1));
  cv2 = register_module("cv2", Conv((1 + n) * c_inner, c2, 1, 1));
  m   = register_module("m", torch::nn::ModuleList());
  int nh = std::max(1, c_inner / 32);
  for (int i = 0; i < n; ++i) {
    if (a2) {
      auto seq = torch::nn::Sequential();
      seq->push_back(ABlock(c_inner, nh, mlp_ratio, area));
      seq->push_back(ABlock(c_inner, nh, mlp_ratio, area));
      m->push_back(seq);
    } else {
      m->push_back(C3k(c_inner, c_inner, /*n=*/2, /*shortcut=*/true,
                       /*g=*/1, /*e=*/0.5, /*k=*/3));
    }
  }
  // Register `gamma` as a parameter of shape [c2] when the outer residual
  // is active and channels match. Init to ones (the .pt's stored gamma will
  // overwrite via load_from_state_dict).
  if (residual && c1 == c2) {
    has_gamma = true;
    gamma = register_parameter("gamma", torch::ones({c2}));
  }
}

torch::Tensor A2C2fImpl::forward(torch::Tensor x) {
  std::vector<torch::Tensor> outs;
  outs.reserve(1 + m->size());
  outs.push_back(cv1(x));
  for (size_t i = 0; i < m->size(); ++i) {
    torch::Tensor next;
    if (a2) {
      next = m[i]->as<torch::nn::SequentialImpl>()->forward(outs.back());
    } else {
      next = m[i]->as<C3kImpl>()->forward(outs.back());
    }
    outs.push_back(next);
  }
  auto y = cv2(torch::cat(outs, /*dim=*/1));
  if (residual && x.size(1) == y.size(1)) {
    if (has_gamma) {
      // Per-channel residual gate: x + γ ⊙ y where γ is broadcast over
      // [B, C, H, W] from a [C] parameter. Without this gate the saturated
      // residual was producing 300-detection floods at conf=0.25 on v12l.
      auto g = gamma.view({1, -1, 1, 1});
      y = x + g * y;
    } else {
      y = x + y;
    }
  }
  return y;
}

// ─── Yolo12Detect ─────────────────────────────────────────────────────────

namespace {

struct LSpec {
  std::vector<int> from;
  std::string      kind;
  std::vector<int> a;
};

const std::vector<LSpec>& v12_yaml() {
  // Args:
  //   Conv:    (c_out, k, s)
  //   C3k2:    (c_out, n, c3k, e_x100)
  //   A2C2f:   (c_out, n, a2, area, residual, mlp_ratio_x100)
  //   Upsample (scale)
  //   Concat   (dim)
  //   Detect   (none — caller supplies nc)
  static const std::vector<LSpec> y = {
      {{-1}, "Conv",   {64,   3, 2}},                          // 0
      {{-1}, "Conv",   {128,  3, 2}},                          // 1
      {{-1}, "C3k2",   {256,  2, 0, 25}},                      // 2
      {{-1}, "Conv",   {256,  3, 2}},                          // 3
      {{-1}, "C3k2",   {512,  2, 0, 25}},                      // 4
      {{-1}, "Conv",   {512,  3, 2}},                          // 5
      {{-1}, "A2C2f",  {512,  4, /*a2=*/1, /*area=*/4,
                        /*residual=*/0, /*mlp=*/200}},          // 6
      {{-1}, "Conv",   {1024, 3, 2}},                          // 7
      {{-1}, "A2C2f",  {1024, 4, /*a2=*/1, /*area=*/1,
                        /*residual=*/0, /*mlp=*/200}},          // 8
      {{-1},     "Upsample", {2}},                              // 9
      {{-1, 6},  "Concat",   {1}},                              // 10
      {{-1},     "A2C2f",    {512, 2, /*a2=*/0, /*area=*/-1,
                              /*residual=*/0, /*mlp=*/200}},     // 11
      {{-1},     "Upsample", {2}},                              // 12
      {{-1, 4},  "Concat",   {1}},                              // 13
      {{-1},     "A2C2f",    {256, 2, /*a2=*/0, /*area=*/-1,
                              /*residual=*/0, /*mlp=*/200}},     // 14 (P3)
      {{-1},     "Conv",     {256,  3, 2}},                     // 15
      {{-1, 11}, "Concat",   {1}},                              // 16
      {{-1},     "A2C2f",    {512, 2, /*a2=*/0, /*area=*/-1,
                              /*residual=*/0, /*mlp=*/200}},     // 17 (P4)
      {{-1},     "Conv",     {512,  3, 2}},                     // 18
      {{-1, 8},  "Concat",   {1}},                              // 19
      {{-1},     "C3k2",     {1024, 2, /*c3k=*/1, /*e=*/50}},   // 20 (P5)
      {{14, 17, 20}, "Detect", {}},                              // 21
  };
  return y;
}

void build_layer(torch::nn::ModuleList& model, const LSpec& spec, int in_ch,
                 int& c_out_out, Yolo12Scale scale) {
  if (spec.kind == "Conv") {
    int c_out = scale_channels_v12(spec.a[0], scale);
    int k = spec.a[1], st = spec.a[2];
    model->push_back(Conv(in_ch, c_out, k, st));
    c_out_out = c_out;
  } else if (spec.kind == "C3k2") {
    int c_out = scale_channels_v12(spec.a[0], scale);
    int n     = scale_depth_v12(spec.a[1], scale);
    bool c3k  = (spec.a[2] != 0);
    if (scale.width_multiple >= 1.0) c3k = true;   // m/l/x force-upgrade
    double e  = (double)spec.a[3] / 100.0;
    model->push_back(C3k2(in_ch, c_out, n, c3k, e));
    c_out_out = c_out;
  } else if (spec.kind == "A2C2f") {
    int c_out  = scale_channels_v12(spec.a[0], scale);
    int n      = scale_depth_v12(spec.a[1], scale);
    bool a2    = (spec.a[2] != 0);
    int area   = spec.a[3];
    bool resid = (spec.a[4] != 0);
    double mlp = (double)spec.a[5] / 100.0;
    // Upstream parse_model override: scales l/x flip A2C2f to
    // `residual=True, mlp_ratio=1.2` regardless of the YAML default. l
    // and x both have depth_multiple=1.0; m (depth=0.5, width=1.0) keeps
    // the defaults.
    if (scale.depth_multiple >= 1.0 && scale.width_multiple >= 1.0) {
      resid = true;
      mlp   = 1.2;
    }
    model->push_back(A2C2f(in_ch, c_out, n, a2, area, resid, mlp, /*e=*/0.5));
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
    throw std::runtime_error("yolo12: unknown layer kind: " + spec.kind);
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
    else if (s.kind == "A2C2f")    outs[i] = model[i]->as<A2C2fImpl>()->forward(in);
    else if (s.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "Concat")   outs[i] = in;
  }
}

}  // anonymous namespace

Yolo12DetectImpl::Yolo12DetectImpl(Yolo12Scale s, int nc_) : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());

  const auto& yaml = v12_yaml();
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
  // v12 uses v11-style cv3 (DWConv-Conv nested); legacy=false.
  model->push_back(Detect(nc, det_ch, /*legacy=*/false));
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
    auto* d = model[yaml.size() - 1]->as<DetectImpl>();
    d->stride = strides;
    stride    = strides;
    this->train();
  }
}

std::vector<torch::Tensor> Yolo12DetectImpl::forward_train(torch::Tensor x) {
  const auto& yaml = v12_yaml();
  std::vector<torch::Tensor> outs;
  forward_through(yaml, model, x, outs);
  std::vector<torch::Tensor> det_in;
  for (int f : yaml.back().from) det_in.push_back(outs[f]);
  return model[yaml.size() - 1]->as<DetectImpl>()->forward_features(det_in);
}

torch::Tensor Yolo12DetectImpl::forward_eval(torch::Tensor x) {
  auto feats = forward_train(x);
  return model[v12_yaml().size() - 1]->as<DetectImpl>()->decode(feats);
}

int Yolo12DetectImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
  auto params = this->named_parameters();
  auto buffs  = this->named_buffers();
  std::unordered_map<std::string, at::Tensor> ours;
  for (auto& kv : params) ours.emplace(kv.key(), kv.value());
  for (auto& kv : buffs)  ours.emplace(kv.key(), kv.value());

  torch::NoGradGuard ng;
  int copied = 0;
  for (const auto& [k, t] : entries) {
    auto it = ours.find(k);
    if (it == ours.end()) continue;
    auto& dst = it->second;
    if (dst.sizes() != t.sizes()) {
      std::ostringstream ss;
      ss << "yolo12 load: shape mismatch for " << k << " ours=" << dst.sizes()
         << " ckpt=" << t.sizes();
      throw std::runtime_error(ss.str());
    }
    dst.copy_(t.to(dst.dtype()).to(dst.device()));
    ++copied;
  }
  if (copied == 0) throw std::runtime_error("yolo12 load: copied 0 tensors");
  return copied;
}

}  // namespace yolocpp::models
