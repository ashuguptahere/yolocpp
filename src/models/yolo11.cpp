#include "yolocpp/models/yolo11.hpp"

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
// Same rounding as v8: c = make_divisible(min(c, max_ch) * width, 8).

int scale_channels_v11(int c, const Yolo11Scale& s) {
  c = std::min(c, s.max_channels);
  auto make_divisible = [](double v, int divisor) {
    return std::max(divisor, (int)std::round(v / divisor) * divisor);
  };
  return make_divisible(c * s.width_multiple, 8);
}

int scale_depth_v11(int n, const Yolo11Scale& s) {
  return std::max(1, (int)std::round(n * s.depth_multiple));
}

Yolo11Scale yolo11_scale_from_letter(const std::string& letter) {
  if (letter == "n") return kYolo11n;
  if (letter == "s") return kYolo11s;
  if (letter == "m") return kYolo11m;
  if (letter == "l") return kYolo11l;
  if (letter == "x") return kYolo11x;
  throw std::runtime_error("yolo11: unknown scale letter '" + letter + "'");
}

Yolo11Scale yolo11_scale_from_filename(const std::string& path) {
  std::filesystem::path p(path);
  std::string base = p.filename().string();
  static const std::regex re(R"(yolov?11([nsmlx])(?:-(?:cls|seg|pose|obb))?\.pt$)");
  std::smatch m;
  if (std::regex_search(base, m, re)) return yolo11_scale_from_letter(m[1].str());
  return kYolo11n;
}

// ─── C3k ──────────────────────────────────────────────────────────────────
// Bottleneck(c_, c_, shortcut, g, k=(3,3), e=1.0) — same as v8's Bottleneck
// when called with k={3,3}. Reuse the v8 module.

C3kImpl::C3kImpl(int c1, int c2, int n, bool shortcut, int g, double e, int k) {
  int c_ = (int)(c2 * e);
  cv1 = register_module("cv1", Conv(c1,    c_, 1, 1));
  cv2 = register_module("cv2", Conv(c1,    c_, 1, 1));
  cv3 = register_module("cv3", Conv(2 * c_, c2, 1));
  m   = register_module("m",   torch::nn::ModuleList());
  for (int i = 0; i < n; ++i) {
    m->push_back(Bottleneck(c_, c_, shortcut, g, /*e=*/1.0,
                            std::array<int, 2>{k, k}));
  }
}

torch::Tensor C3kImpl::forward(torch::Tensor x) {
  auto a = cv1(x);
  for (size_t i = 0; i < m->size(); ++i) {
    a = m[i]->as<BottleneckImpl>()->forward(a);
  }
  auto b = cv2(x);
  return cv3(torch::cat({a, b}, /*dim=*/1));
}

// ─── C3k2 ─────────────────────────────────────────────────────────────────
// State-dict naming and forward shape are bit-identical to v8 C2f (only the
// inner block kind changes). c_inner here is C2f's "self.c" in Ultralytics.

C3k2Impl::C3k2Impl(int c1, int c2, int n, bool c3k_, double e, int g,
                   bool shortcut)
    : c3k(c3k_) {
  c_inner = (int)(c2 * e);
  cv1 = register_module("cv1", Conv(c1, 2 * c_inner, 1, 1));
  cv2 = register_module("cv2", Conv((2 + n) * c_inner, c2, 1));
  m   = register_module("m", torch::nn::ModuleList());
  for (int i = 0; i < n; ++i) {
    if (c3k) {
      // Ultralytics hardcodes the inner C3k's depth to 2.
      m->push_back(C3k(c_inner, c_inner, /*n=*/2, shortcut, g, /*e=*/0.5, /*k=*/3));
    } else {
      m->push_back(Bottleneck(c_inner, c_inner, shortcut, g, /*e=*/0.5,
                              std::array<int, 2>{3, 3}));
    }
  }
}

torch::Tensor C3k2Impl::forward(torch::Tensor x) {
  auto y = cv1(x).chunk(2, /*dim=*/1);
  std::vector<torch::Tensor> outs;
  outs.reserve(2 + m->size());
  outs.push_back(y[0]);
  outs.push_back(y[1]);
  for (size_t i = 0; i < m->size(); ++i) {
    torch::Tensor next;
    if (c3k) next = m[i]->as<C3kImpl>()->forward(outs.back());
    else     next = m[i]->as<BottleneckImpl>()->forward(outs.back());
    outs.push_back(next);
  }
  return cv2(torch::cat(outs, /*dim=*/1));
}

// ─── PSA Attention ────────────────────────────────────────────────────────

PSAAttentionImpl::PSAAttentionImpl(int dim, int num_heads_, double attn_ratio)
    : num_heads(num_heads_) {
  TORCH_CHECK(dim % num_heads == 0,
              "PSAAttention: dim (", dim, ") must be divisible by num_heads (",
              num_heads, ")");
  head_dim = dim / num_heads;
  key_dim  = (int)((double)head_dim * attn_ratio);
  scale    = std::pow((double)key_dim, -0.5);
  int nh_kd = key_dim * num_heads;
  int h     = dim + nh_kd * 2;
  qkv  = register_module("qkv",  Conv(dim, h,   1, 1, /*p=*/-1, /*g=*/1, /*act=*/false));
  proj = register_module("proj", Conv(dim, dim, 1, 1, /*p=*/-1, /*g=*/1, /*act=*/false));
  pe   = register_module("pe",   Conv(dim, dim, 3, 1, /*p=*/-1, /*g=*/dim, /*act=*/false));
}

torch::Tensor PSAAttentionImpl::forward(torch::Tensor x) {
  // x: [B, C, H, W]
  auto B = x.size(0);
  auto C = x.size(1);
  auto H = x.size(2);
  auto W = x.size(3);
  auto N = H * W;

  auto y = qkv(x);  // [B, h, H, W] where h = num_heads * (2*key_dim + head_dim)
  // Reshape to [B, num_heads, 2*key_dim + head_dim, N], then split.
  y = y.view({B, num_heads, 2 * key_dim + head_dim, N});
  // Match Ultralytics: q / k / v split sizes are key_dim, key_dim, head_dim.
  auto parts = y.split_with_sizes({(int64_t)key_dim, (int64_t)key_dim,
                                   (int64_t)head_dim},
                                  /*dim=*/2);
  auto q = parts[0];  // [B, nh, key_dim, N]
  auto k = parts[1];  // [B, nh, key_dim, N]
  auto v = parts[2];  // [B, nh, head_dim, N]

  // attn[b, h, n_q, n_k] = sum_d q[b,h,d,n_q] * k[b,h,d,n_k]  (transpose q on last dim).
  auto attn = torch::matmul(q.transpose(-2, -1), k) * scale;  // [B, nh, N, N]
  attn = attn.softmax(/*dim=*/-1);

  // out[b, h, d, n_q] = sum_{n_k} v[b,h,d,n_k] * attn[b,h,n_q,n_k]^T  → matmul(v, attn^T)
  auto out = torch::matmul(v, attn.transpose(-2, -1));      // [B, nh, head_dim, N]
  out = out.reshape({B, C, H, W});

  // Add depthwise positional encoding on v (reshaped back to spatial).
  auto v_spatial = v.reshape({B, C, H, W});
  out = out + pe(v_spatial);
  return proj(out);
}

// ─── PSABlock ─────────────────────────────────────────────────────────────

PSABlockImpl::PSABlockImpl(int c, double attn_ratio, int num_heads, bool shortcut)
    : add(shortcut) {
  attn = register_module("attn", PSAAttention(c, num_heads, attn_ratio));
  // ffn = Sequential(Conv(c, 2c, 1), Conv(2c, c, 1, act=False))
  ffn = register_module("ffn", torch::nn::Sequential());
  ffn->push_back(Conv(c, 2 * c, 1, 1));
  ffn->push_back(Conv(2 * c, c, 1, 1, /*p=*/-1, /*g=*/1, /*act=*/false));
}

torch::Tensor PSABlockImpl::forward(torch::Tensor x) {
  if (add) {
    x = x + attn->forward(x);
    x = x + ffn->forward(x);
  } else {
    x = attn->forward(x);
    x = ffn->forward(x);
  }
  return x;
}

// ─── C2PSA ────────────────────────────────────────────────────────────────

C2PSAImpl::C2PSAImpl(int c1, int c2, int n, double e) {
  TORCH_CHECK(c1 == c2, "C2PSA expects c1 == c2 (got ", c1, " vs ", c2, ")");
  c = (int)((double)c1 * e);
  cv1 = register_module("cv1", Conv(c1,     2 * c, 1, 1));
  cv2 = register_module("cv2", Conv(2 * c,  c1,    1));
  // num_heads = self.c // 64 (Ultralytics' rule)
  int nh = std::max(1, c / 64);
  m = register_module("m", torch::nn::Sequential());
  for (int i = 0; i < n; ++i) {
    m->push_back(PSABlock(c, /*attn_ratio=*/0.5, /*num_heads=*/nh,
                          /*shortcut=*/true));
  }
}

torch::Tensor C2PSAImpl::forward(torch::Tensor x) {
  auto y = cv1(x);                          // [B, 2c, H, W]
  auto parts = y.split_with_sizes({(int64_t)c, (int64_t)c}, /*dim=*/1);
  auto a = parts[0];                        // [B, c, H, W]
  auto b = parts[1];                        // [B, c, H, W]
  b = m->forward(b);
  return cv2(torch::cat({a, b}, /*dim=*/1));
}

// ─── Yolo11DetectImpl ─────────────────────────────────────────────────────

namespace {

// One YAML layer spec — same shape as v8's, with the v11 module set.
struct LSpec {
  std::vector<int> from;     // -1 = previous; explicit indices for Concat / Detect
  std::string      kind;     // "Conv", "C3k2", "SPPF", "C2PSA", "Upsample", "Concat", "Detect"
  std::vector<int> a;        // args (interpretation depends on kind)
  // For C3k2: a = (c_out, n, c3k, e_x100). e_x100 = 50 → e=0.5; e_x100 = 25 → e=0.25.
};

const std::vector<LSpec>& v11_yaml() {
  // Mirrors ultralytics/cfg/models/11/yolo11.yaml. e is encoded as e_x100 to
  // keep the args vector all-int.
  static const std::vector<LSpec> y = {
      // Backbone (10 layers)
      {{-1}, "Conv",  {64,   3, 2}},                  // 0   P1/2
      {{-1}, "Conv",  {128,  3, 2}},                  // 1   P2/4
      {{-1}, "C3k2",  {256,  2, /*c3k=*/0, /*e=*/25}},// 2
      {{-1}, "Conv",  {256,  3, 2}},                  // 3   P3/8
      {{-1}, "C3k2",  {512,  2, /*c3k=*/0, /*e=*/25}},// 4
      {{-1}, "Conv",  {512,  3, 2}},                  // 5   P4/16
      {{-1}, "C3k2",  {512,  2, /*c3k=*/1, /*e=*/50}},// 6
      {{-1}, "Conv",  {1024, 3, 2}},                  // 7   P5/32
      {{-1}, "C3k2",  {1024, 2, /*c3k=*/1, /*e=*/50}},// 8
      {{-1}, "SPPF",  {1024, 5}},                     // 9
      {{-1}, "C2PSA", {1024, 2}},                     // 10  (n=2 pre-depth-scale)
      // Head (12 layers + Detect = 13)
      {{-1},     "Upsample", {2}},                    // 11
      {{-1, 6},  "Concat",   {1}},                    // 12  cat backbone P4
      {{-1},     "C3k2",     {512,  2, 0, 50}},       // 13
      {{-1},     "Upsample", {2}},                    // 14
      {{-1, 4},  "Concat",   {1}},                    // 15  cat backbone P3
      {{-1},     "C3k2",     {256,  2, 0, 50}},       // 16  (P3/8-small)
      {{-1},     "Conv",     {256,  3, 2}},           // 17
      {{-1, 13}, "Concat",   {1}},                    // 18  cat head P4
      {{-1},     "C3k2",     {512,  2, 0, 50}},       // 19  (P4/16-medium)
      {{-1},     "Conv",     {512,  3, 2}},           // 20
      {{-1, 10}, "Concat",   {1}},                    // 21  cat head P5
      {{-1},     "C3k2",     {1024, 2, 1, 50}},       // 22  (P5/32-large)
      {{16, 19, 22}, "Detect", {}},                   // 23
  };
  return y;
}

void build_layer(torch::nn::ModuleList& model, const LSpec& spec, int in_ch,
                 int& c_out_out, Yolo11Scale scale, int nc) {
  if (spec.kind == "Conv") {
    int c_out = scale_channels_v11(spec.a[0], scale);
    int k = spec.a[1], st = spec.a[2];
    model->push_back(Conv(in_ch, c_out, k, st));
    c_out_out = c_out;
  } else if (spec.kind == "C3k2") {
    int c_out = scale_channels_v11(spec.a[0], scale);
    int n     = scale_depth_v11(spec.a[1], scale);
    bool c3k  = (spec.a[2] != 0);
    // Ultralytics' parse_model() forces c3k=True for m/l/x scales regardless
    // of the YAML's explicit value. Detect those scales by width >= 1.0 —
    // n=0.25, s=0.50 keep c3k as YAML; m=1.00, l=1.00, x=1.50 force-upgrade.
    if (scale.width_multiple >= 1.0) c3k = true;
    double e  = (double)spec.a[3] / 100.0;
    model->push_back(C3k2(in_ch, c_out, n, c3k, e));
    c_out_out = c_out;
  } else if (spec.kind == "SPPF") {
    int c_out = scale_channels_v11(spec.a[0], scale);
    model->push_back(SPPF(in_ch, c_out, spec.a[1]));
    c_out_out = c_out;
  } else if (spec.kind == "C2PSA") {
    int c_out = scale_channels_v11(spec.a[0], scale);
    int n     = scale_depth_v11(spec.a[1], scale);
    model->push_back(C2PSA(in_ch, c_out, n, /*e=*/0.5));
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
    // caller supplies the per-level det_ch through `in_ch`; here in_ch is
    // unused — the actual Detect is constructed by the parent because it
    // needs three input channels. We register a placeholder Identity that
    // the caller will replace with the real Detect.
    (void)nc;
    c_out_out = 0;
  } else {
    throw std::runtime_error("yolo11: unknown layer kind: " + spec.kind);
  }
}

// Forward layers 0..(yaml.size()-1) using `outs[i]` as scratch storage.
// Calls each layer with the right input tensors per its `from`. The Detect
// layer is handled by the caller (it needs the 3 inputs and decode/forward
// branching).
void forward_through(const std::vector<LSpec>& yaml,
                     torch::nn::ModuleList& model, torch::Tensor x,
                     std::vector<torch::Tensor>& outs) {
  outs.assign(yaml.size(), torch::Tensor());
  for (size_t i = 0; i < yaml.size(); ++i) {
    const auto& s = yaml[i];
    if (s.kind == "Detect") {
      outs[i] = torch::Tensor();  // caller handles it
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
    else if (s.kind == "Upsample") outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (s.kind == "Concat")   outs[i] = in;
  }
}

}  // anonymous namespace

Yolo11DetectImpl::Yolo11DetectImpl(Yolo11Scale s, int nc_) : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());

  const auto& yaml = v11_yaml();
  std::vector<int> ch;
  ch.reserve(yaml.size());
  int c_in = 3;

  // Build layers 0..22 (everything except Detect).
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
    build_layer(model, spec, in_ch, c_out, scale, nc);
    ch.push_back(c_out);
  }

  // Build Detect (layer 23) with per-level input channels from from-indices.
  const auto& det = yaml.back();
  std::vector<int> det_ch;
  for (int f : det.from) det_ch.push_back(ch[f]);
  model->push_back(Detect(nc, det_ch, /*legacy=*/false));
  ch.push_back(0);

  // Compute strides by probing the network with a dummy 256×256 input.
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

std::vector<torch::Tensor> Yolo11DetectImpl::forward_train(torch::Tensor x) {
  const auto& yaml = v11_yaml();
  std::vector<torch::Tensor> outs;
  forward_through(yaml, model, x, outs);
  std::vector<torch::Tensor> det_in;
  for (int f : yaml.back().from) det_in.push_back(outs[f]);
  return model[yaml.size() - 1]->as<DetectImpl>()->forward_features(det_in);
}

torch::Tensor Yolo11DetectImpl::forward_eval(torch::Tensor x) {
  auto feats = forward_train(x);
  return model[v11_yaml().size() - 1]->as<DetectImpl>()->decode(feats);
}

int Yolo11DetectImpl::load_from_state_dict(
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
      ss << "yolo11 load: shape mismatch for " << k << " ours=" << dst.sizes()
         << " ckpt=" << t.sizes();
      throw std::runtime_error(ss.str());
    }
    dst.copy_(t.to(dst.dtype()).to(dst.device()));
    ++copied;
  }
  if (copied == 0)
    throw std::runtime_error("yolo11 load: copied 0 tensors");
  return copied;
}

}  // namespace yolocpp::models
