#include "yolocpp/models/yolov8_classify.hpp"

#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace yolocpp::models {

ClassifyImpl::ClassifyImpl(int c_in, int nc, int c_hidden) {
  conv   = register_module("conv",   Conv(c_in, c_hidden, 1, 1));
  linear = register_module("linear", torch::nn::Linear(c_hidden, nc));
}

torch::Tensor ClassifyImpl::forward(torch::Tensor x) {
  x = conv(x);
  // Adaptive avg pool to 1×1
  x = torch::adaptive_avg_pool2d(x, {1, 1});
  x = x.flatten(1);
  // Dropout (0.0 for inference) skipped
  return linear(x);
}

YoloV8ClassifyImpl::YoloV8ClassifyImpl(YoloV8Scale s, int nc_)
    : scale(s), nc(nc_) {
  model = register_module("model", torch::nn::ModuleList());

  // YAML for classify:
  //   [Conv 64,3,2] [Conv 128,3,2] [C2f 128 n=2 sc=T] [Conv 256,3,2]
  //   [C2f 256 n=4 sc=T] [Conv 512,3,2] [C2f 512 n=4 sc=T] [Conv 1024,3,2]
  //   [C2f 1024 n=2 sc=T] [Classify nc]
  struct LSpec { std::string kind; std::vector<int> a; };
  std::vector<LSpec> y = {
      {"Conv", {64,  3, 2}}, {"Conv", {128, 3, 2}},
      {"C2f",  {128, 3, 1}},
      {"Conv", {256, 3, 2}},
      {"C2f",  {256, 6, 1}},
      {"Conv", {512, 3, 2}},
      {"C2f",  {512, 6, 1}},
      {"Conv", {1024,3, 2}},
      {"C2f",  {1024,3, 1}},
      {"Classify", {}},
  };
  int c_in = 3;
  std::vector<int> ch;
  for (size_t i = 0; i < y.size(); ++i) {
    const auto& s2 = y[i];
    int prev = (i == 0) ? c_in : ch.back();
    if (s2.kind == "Conv") {
      int c_out = scale_channels(s2.a[0], scale);
      model->push_back(Conv(prev, c_out, s2.a[1], s2.a[2]));
      ch.push_back(c_out);
    } else if (s2.kind == "C2f") {
      int c_out = scale_channels(s2.a[0], scale);
      int n     = scale_depth(s2.a[1], scale);
      bool sc   = (s2.a[2] != 0);
      model->push_back(C2f(prev, c_out, n, sc));
      ch.push_back(c_out);
    } else {  // Classify
      // Hidden dim 1280 fixed, output nc.
      model->push_back(Classify(prev, nc, /*c_hidden=*/1280));
      ch.push_back(nc);
    }
  }
}

torch::Tensor YoloV8ClassifyImpl::forward(torch::Tensor x) {
  for (size_t i = 0; i < model->size(); ++i) {
    if (auto m = model[i]->as<ConvImpl>())     x = m->forward(x);
    else if (auto m = model[i]->as<C2fImpl>()) x = m->forward(x);
    else if (auto m = model[i]->as<ClassifyImpl>()) x = m->forward(x);
  }
  return x;
}

int YoloV8ClassifyImpl::load_from_state_dict(
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
      ss << "classify load: shape mismatch for " << k << " ours=" << dst.sizes()
         << " ckpt=" << t.sizes();
      throw std::runtime_error(ss.str());
    }
    dst.copy_(t.to(dst.dtype()).to(dst.device()));
    ++copied;
  }
  if (copied == 0)
    throw std::runtime_error("classify load: copied 0 tensors");
  return copied;
}

}  // namespace yolocpp::models
