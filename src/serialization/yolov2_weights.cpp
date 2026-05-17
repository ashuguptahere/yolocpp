#include "yolocpp/serialization/yolov2_weights.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "yolocpp/models/yolo2.hpp"
#include "yolocpp/models/yolo4.hpp"   // ConvLeakyImpl (BN + Conv) used by yolo2
#include "yolocpp/serialization/pt_save.hpp"

namespace yolocpp::serialization {

namespace {

void read_floats(std::ifstream& f, float* dst, std::size_t n,
                  const char* what) {
  f.read(reinterpret_cast<char*>(dst), n * sizeof(float));
  if (!f || (std::size_t)f.gcount() != n * sizeof(float)) {
    throw std::runtime_error(std::string("yolov2_weights: short read of ") +
                              std::to_string(n) + " floats while reading " +
                              what);
  }
}

void fill(std::ifstream& f, at::Tensor& t, const char* what) {
  if (!t.is_contiguous()) t = t.contiguous();
  if (t.scalar_type() != torch::kFloat32) {
    throw std::runtime_error(std::string("yolov2_weights: ") + what +
                              " is not float32");
  }
  read_floats(f, t.data_ptr<float>(), (std::size_t)t.numel(), what);
}

// One Darknet [convolutional] block with batch_normalize=1 — stream
// order is bn_bias, bn_weight, bn_mean, bn_var, conv_weight.
void load_bn_conv(std::ifstream& f, torch::nn::Conv2dImpl& conv,
                  torch::nn::BatchNorm2dImpl& bn, const std::string& name) {
  fill(f, bn.bias,         (name + ".bn.bias").c_str());
  fill(f, bn.weight,       (name + ".bn.weight").c_str());
  fill(f, bn.running_mean, (name + ".bn.running_mean").c_str());
  fill(f, bn.running_var,  (name + ".bn.running_var").c_str());
  fill(f, conv.weight,     (name + ".conv.weight").c_str());
}

void load_bare_conv(std::ifstream& f, torch::nn::Conv2dImpl& conv,
                     const std::string& name) {
  if (!conv.bias.defined() || conv.bias.numel() == 0) {
    throw std::runtime_error("yolov2_weights: bare conv " + name +
                              " has no bias tensor");
  }
  fill(f, conv.bias,   (name + ".bias").c_str());
  fill(f, conv.weight, (name + ".weight").c_str());
}

// DFS the Yolo2 model in registration order. ConvLeaky (BN+conv) is
// one Darknet block; the final 1×1 detection conv is a bare Conv2d.
void walk(std::ifstream& f, torch::nn::Module& m, const std::string& name,
          int& count) {
  using namespace yolocpp::models;
  if (auto* cl = m.as<ConvLeakyImpl>()) {
    load_bn_conv(f, *cl->conv, *cl->bn, name);
    ++count;
    return;
  }
  if (auto* c2 = m.as<torch::nn::Conv2dImpl>()) {
    load_bare_conv(f, *c2, name);
    ++count;
    return;
  }
  for (auto& kv : m.named_children()) {
    walk(f, *kv.value(), name.empty() ? kv.key() : (name + "." + kv.key()),
          count);
  }
}

}  // namespace

int convert_yolov2_weights(const std::string& weights_path,
                            const std::string& out_pt_path, int nc,
                            models::Yolo2Scale scale) {
  std::ifstream f(weights_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("yolov2_weights: cannot open " + weights_path);
  }

  std::int32_t major = 0, minor = 0, revision = 0;
  f.read(reinterpret_cast<char*>(&major),    4);
  f.read(reinterpret_cast<char*>(&minor),    4);
  f.read(reinterpret_cast<char*>(&revision), 4);
  if (!f) throw std::runtime_error("yolov2_weights: short read of header");
  if ((std::int64_t)major * 10 + minor >= 2) {
    std::int64_t seen = 0;
    f.read(reinterpret_cast<char*>(&seen), 8);
  } else {
    std::int32_t seen = 0;
    f.read(reinterpret_cast<char*>(&seen), 4);
  }
  if (!f) throw std::runtime_error("yolov2_weights: short read of seen");

  models::Yolo2 model(scale, nc);
  int count = 0;
  walk(f, *model.ptr(), "", count);

  auto here = f.tellg();
  f.seekg(0, std::ios::end);
  auto end  = f.tellg();
  if (here != end) {
    throw std::runtime_error("yolov2_weights: " +
        std::to_string((long long)(end - here)) +
        " trailing bytes after " + std::to_string(count) +
        " conv blocks (header v" + std::to_string(major) + "." +
        std::to_string(minor) + "." + std::to_string(revision) + ")");
  }

  std::vector<std::pair<std::string, at::Tensor>> entries;
  for (const auto& kv : model->named_parameters(/*recurse=*/true)) {
    entries.emplace_back(kv.key(), kv.value().detach().clone());
  }
  for (const auto& kv : model->named_buffers(/*recurse=*/true)) {
    if (kv.value().scalar_type() == torch::kFloat32) {
      entries.emplace_back(kv.key(), kv.value().detach().clone());
    }
  }
  save_state_dict(out_pt_path, entries);
  std::cerr << "[yolov2_weights] " << entries.size() << " tensors ("
            << count << " conv blocks) → " << out_pt_path
            << " (nc=" << nc
            << ", scale=" << (scale == models::Yolo2Scale::Tiny ? "tiny" : "full")
            << ")\n";
  return count;
}

}  // namespace yolocpp::serialization
