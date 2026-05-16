#include "yolocpp/models/yolo5.hpp"

#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace yolocpp::models {

// ─── C3 ─────────────────────────────────────────────────────────────────
C3Impl::C3Impl(int c1, int c2, int n, bool shortcut, int g, double e) {
  c_inner = (int)(c2 * e);
  cv1 = register_module("cv1", Conv(c1, c_inner, 1, 1));
  cv2 = register_module("cv2", Conv(c1, c_inner, 1, 1));
  cv3 = register_module("cv3", Conv(2 * c_inner, c2, 1));
  m   = register_module("m",   torch::nn::ModuleList());
  for (int i = 0; i < n; ++i) {
    // Upstream C3 uses Bottleneck(k=((1,1),(3,3)), e=1.0) — 1×1 then 3×3.
    m->push_back(Bottleneck(c_inner, c_inner, shortcut, g, 1.0,
                            std::array<int, 2>{1, 3}));
  }
}

torch::Tensor C3Impl::forward(torch::Tensor x) {
  auto y1 = cv1(x);
  for (size_t i = 0; i < m->size(); ++i) {
    y1 = m[i]->as<BottleneckImpl>()->forward(y1);
  }
  auto y2 = cv2(x);
  return cv3(torch::cat({y1, y2}, /*dim=*/1));
}

// ─── Yolo5DetectImpl ───────────────────────────────────────────────────
namespace {
struct LayerSpec {
  std::vector<int> from;
  std::string      kind;
  std::vector<int> a;
};
const std::vector<LayerSpec>& v5_yaml() {
  static const std::vector<LayerSpec> y = {
      // backbone
      {{-1}, "Conv6", {64,  6, 2, 2}},   // 0  k=6 s=2 p=2
      {{-1}, "Conv",  {128, 3, 2}},      // 1
      {{-1}, "C3",    {128, 3, 1}},      // 2 (shortcut=true)
      {{-1}, "Conv",  {256, 3, 2}},      // 3
      {{-1}, "C3",    {256, 6, 1}},      // 4
      {{-1}, "Conv",  {512, 3, 2}},      // 5
      {{-1}, "C3",    {512, 9, 1}},      // 6
      {{-1}, "Conv",  {1024,3, 2}},      // 7
      {{-1}, "C3",    {1024,3, 1}},      // 8
      {{-1}, "SPPF",  {1024,5}},         // 9
      // head
      {{-1}, "Conv",  {512, 1, 1}},      // 10
      {{-1}, "Upsample", {2}},            // 11
      {{-1, 6}, "Concat", {1}},           // 12
      {{-1}, "C3",    {512, 3, 0}},      // 13 shortcut=false
      {{-1}, "Conv",  {256, 1, 1}},      // 14
      {{-1}, "Upsample", {2}},            // 15
      {{-1, 4}, "Concat", {1}},           // 16
      {{-1}, "C3",    {256, 3, 0}},      // 17 (P3)
      {{-1}, "Conv",  {256, 3, 2}},      // 18
      {{-1, 14}, "Concat", {1}},          // 19
      {{-1}, "C3",    {512, 3, 0}},      // 20 (P4)
      {{-1}, "Conv",  {512, 3, 2}},      // 21
      {{-1, 10}, "Concat", {1}},          // 22
      {{-1}, "C3",    {1024,3, 0}},      // 23 (P5)
      {{17, 20, 23}, "Detect", {}},      // 24
  };
  return y;
}
}  // anonymous namespace

Yolo5DetectImpl::Yolo5DetectImpl(Yolo8Scale s, int nc_)
    : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());
  const auto& yaml = v5_yaml();
  std::vector<int> ch;
  int c_in = 3;

  for (size_t i = 0; i < yaml.size(); ++i) {
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

    if (spec.kind == "Conv6") {
      // Layer 0: 6×6 stride-2 stem with explicit padding=2.
      int c_out = scale_channels(spec.a[0], scale);
      model->push_back(Conv(in_ch, c_out, /*k=*/spec.a[1],
                            /*s=*/spec.a[2], /*p=*/spec.a[3]));
      ch.push_back(c_out);
    } else if (spec.kind == "Conv") {
      int c_out = scale_channels(spec.a[0], scale);
      model->push_back(Conv(in_ch, c_out, spec.a[1], spec.a[2]));
      ch.push_back(c_out);
    } else if (spec.kind == "C3") {
      int c_out = scale_channels(spec.a[0], scale);
      int n     = scale_depth(spec.a[1], scale);
      bool shortcut = (spec.a[2] != 0);
      model->push_back(C3(in_ch, c_out, n, shortcut));
      ch.push_back(c_out);
    } else if (spec.kind == "SPPF") {
      int c_out = scale_channels(spec.a[0], scale);
      model->push_back(SPPF(in_ch, c_out, spec.a[1]));
      ch.push_back(c_out);
    } else if (spec.kind == "Upsample") {
      model->push_back(torch::nn::Upsample(
          torch::nn::UpsampleOptions()
              .scale_factor(std::vector<double>{(double)spec.a[0],
                                                (double)spec.a[0]})
              .mode(torch::kNearest)));
      ch.push_back(in_ch);
    } else if (spec.kind == "Concat") {
      model->push_back(torch::nn::Identity());
      ch.push_back(in_ch);
    } else if (spec.kind == "Detect") {
      std::vector<int> det_ch;
      for (int f : spec.from) det_ch.push_back(ch[f]);
      model->push_back(Detect(nc, det_ch));
      ch.push_back(0);
    }
  }

  // Compute strides via a one-shot probe (mirrors v8).
  {
    torch::NoGradGuard ng;
    this->eval();
    auto x = torch::zeros({1, 3, 256, 256});
    std::vector<torch::Tensor> outs(yaml.size());
    int img_h = 256;
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
      if (spec.kind == "Conv" || spec.kind == "Conv6")
        outs[i] = model[i]->as<ConvImpl>()->forward(in);
      else if (spec.kind == "C3")    outs[i] = model[i]->as<C3Impl>()->forward(in);
      else if (spec.kind == "SPPF")  outs[i] = model[i]->as<SPPFImpl>()->forward(in);
      else if (spec.kind == "Upsample")
        outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
      else if (spec.kind == "Concat") outs[i] = in;
    }
    this->train();
  }
}

std::vector<torch::Tensor> Yolo5DetectImpl::forward_train(torch::Tensor x) {
  const auto& yaml = v5_yaml();
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
    if (spec.kind == "Conv" || spec.kind == "Conv6")
      outs[i] = model[i]->as<ConvImpl>()->forward(in);
    else if (spec.kind == "C3")    outs[i] = model[i]->as<C3Impl>()->forward(in);
    else if (spec.kind == "SPPF")  outs[i] = model[i]->as<SPPFImpl>()->forward(in);
    else if (spec.kind == "Upsample")
      outs[i] = model[i]->as<torch::nn::UpsampleImpl>()->forward(in);
    else if (spec.kind == "Concat") outs[i] = in;
  }
  TORCH_CHECK(false, "unreachable");
}

torch::Tensor Yolo5DetectImpl::forward_eval(torch::Tensor x) {
  auto feats = forward_train(x);
  const auto& yaml = v5_yaml();
  auto* d = model[yaml.size() - 1]->as<DetectImpl>();
  return d->decode(feats);
}

int Yolo5DetectImpl::load_from_state_dict(
    const std::vector<std::pair<std::string, at::Tensor>>& entries) {
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
    dst.copy_(t.to(dst.dtype()).to(dst.device()));
    ++copied;
  }
  if (copied == 0) throw std::runtime_error("yolo5 load: copied 0 tensors");
  if (skipped_shape > 0)
    std::cerr << "[yolo5 load] skipped " << skipped_shape
              << " tensors with shape mismatch (cls head re-purposed for custom nc)\n";
  return copied;
}

}  // namespace yolocpp::models
