#include "yolocpp/serialization/darknet_weights.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/serialization/pt_save.hpp"

namespace yolocpp::serialization {

namespace {

void read_floats(std::ifstream& f, float* dst, std::size_t n,
                 const char* what) {
  f.read(reinterpret_cast<char*>(dst), n * sizeof(float));
  if (!f || (std::size_t)f.gcount() != n * sizeof(float)) {
    throw std::runtime_error(std::string("darknet_weights: short read of ") +
                             std::to_string(n) + " floats while reading " +
                             what);
  }
}

// Fill a (typically params'd) tensor in-place with `n` floats from `f`.
// `t` must be CPU, contiguous, float32 — guaranteed for our libtorch
// modules just after construction.
void fill_tensor(std::ifstream& f, at::Tensor& t, const char* what) {
  if (!t.is_contiguous()) t = t.contiguous();
  if (t.scalar_type() != torch::kFloat32) {
    throw std::runtime_error(std::string("darknet_weights: ") + what +
                             " is not float32 — refusing to load");
  }
  read_floats(f, t.data_ptr<float>(), (std::size_t)t.numel(), what);
}

// Treat one Conv+BN unit (as in ConvMishImpl / ConvLeakyImpl) as the
// natural Darknet [convolutional] with batch_normalize=1: stream order is
// bn.bias, bn.weight, bn.running_mean, bn.running_var, conv.weight.
void load_bn_conv(std::ifstream& f, torch::nn::Conv2dImpl& conv,
                  torch::nn::BatchNorm2dImpl& bn,
                  const std::string& name) {
  fill_tensor(f, bn.bias,         (name + ".bn.bias").c_str());
  fill_tensor(f, bn.weight,       (name + ".bn.weight").c_str());
  fill_tensor(f, bn.running_mean, (name + ".bn.running_mean").c_str());
  fill_tensor(f, bn.running_var,  (name + ".bn.running_var").c_str());
  fill_tensor(f, conv.weight,     (name + ".conv.weight").c_str());
}

// Bare Conv2d (no BN) — Darknet [convolutional] with batch_normalize=0.
// Streams bias then conv weight.
void load_bare_conv(std::ifstream& f, torch::nn::Conv2dImpl& conv,
                    const std::string& name) {
  if (!conv.bias.defined() || conv.bias.numel() == 0) {
    throw std::runtime_error("darknet_weights: bare conv " + name +
                             " has no bias tensor (expected bias=true)");
  }
  fill_tensor(f, conv.bias,   (name + ".bias").c_str());
  fill_tensor(f, conv.weight, (name + ".weight").c_str());
}

// DFS the model in registration order. When we hit a ConvMish/ConvLeaky
// node, treat it as a single BN-Conv block and short-circuit (don't recurse
// into its conv/bn children — we just consumed them). When we hit a bare
// Conv2d outside of one of those wrappers, treat as a no-BN block. All
// other module kinds (CSPStage, SPPv4, DarknetResidualMish, ModuleList,
// Yolo4Impl itself, MaxPool2d, BatchNorm2d-as-leaf-of-our-Convs) are
// either pass-through or already handled.
void walk(std::ifstream& f, torch::nn::Module& m, const std::string& name,
          int& count) {
  using namespace yolocpp::models;
  if (auto* cm = m.as<ConvMishImpl>()) {
    load_bn_conv(f, *cm->conv, *cm->bn, name);
    ++count;
    return;
  }
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
  // Recurse via named_children (preserves registration order).
  for (auto& kv : m.named_children()) {
    walk(f, *kv.value(), name.empty() ? kv.key() : (name + "." + kv.key()),
         count);
  }
}

}  // namespace

int convert_yolov4_weights(const std::string& weights_path,
                            const std::string& out_pt_path, int nc) {
  std::ifstream f(weights_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("darknet_weights: cannot open " + weights_path);
  }

  // Read header. AlexeyAB's yolov4 has major=0, minor=2, revision=5, and
  // an int64 `seen` field (5 = both >= 1 → int64 width chosen by
  // (major*10 + minor) >= 2 in Darknet's parser).
  std::int32_t major = 0, minor = 0, revision = 0;
  f.read(reinterpret_cast<char*>(&major),    4);
  f.read(reinterpret_cast<char*>(&minor),    4);
  f.read(reinterpret_cast<char*>(&revision), 4);
  if (!f) {
    throw std::runtime_error("darknet_weights: short read of header");
  }
  if ((std::int64_t)major * 10 + minor >= 2) {
    std::int64_t seen = 0;
    f.read(reinterpret_cast<char*>(&seen), 8);
  } else {
    std::int32_t seen = 0;
    f.read(reinterpret_cast<char*>(&seen), 4);
  }
  std::cerr << "[darknet_weights] header: major=" << major << " minor=" << minor
            << " revision=" << revision << "\n";

  // Build a fresh Yolo4Impl(nc). All tensors land on CPU as float32 by
  // default — exactly what the binary supplies.
  models::Yolo4 model(nc);
  model->eval();

  int count = 0;
  walk(f, *model, /*name=*/"", count);

  // Verify we consumed everything.
  std::streampos here = f.tellg();
  f.seekg(0, std::ios::end);
  std::streampos end = f.tellg();
  if (here != end) {
    throw std::runtime_error(
        "darknet_weights: " + std::to_string((long long)(end - here)) +
        " trailing bytes after consuming " + std::to_string(count) +
        " conv blocks — model topology likely doesn't match yolov4.cfg "
        "(file " + weights_path + ")");
  }

  // Snapshot named_parameters + named_buffers into a flat state-dict and
  // save with the existing upstream-compatible writer. We name the
  // tensors exactly as libtorch's named_parameters() does, so
  // load_from_state_dict can match by string.
  std::vector<std::pair<std::string, at::Tensor>> entries;
  for (const auto& kv : model->named_parameters(/*recurse=*/true)) {
    entries.emplace_back(kv.key(), kv.value().detach().clone());
  }
  for (const auto& kv : model->named_buffers(/*recurse=*/true)) {
    // Skip non-persistent buffers (num_batches_tracked) since pt_save's
    // _rebuild_tensor_v2 path only handles real float storage; running_mean
    // / running_var are float and DO want saving.
    if (kv.value().scalar_type() == torch::kFloat32) {
      entries.emplace_back(kv.key(), kv.value().detach().clone());
    }
  }

  save_state_dict(out_pt_path, entries);
  std::cerr << "[darknet_weights] wrote " << entries.size() << " tensors ("
            << count << " conv blocks) to " << out_pt_path << "\n";
  return count;
}

}  // namespace yolocpp::serialization
