#include "yolocpp/serialization/yolov1_weights.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "yolocpp/models/yolo1.hpp"
#include "yolocpp/serialization/pt_save.hpp"

namespace yolocpp::serialization {

namespace {

void read_floats(std::ifstream& f, float* dst, std::size_t n,
                  const char* what) {
  f.read(reinterpret_cast<char*>(dst), n * sizeof(float));
  if (!f || (std::size_t)f.gcount() != n * sizeof(float)) {
    throw std::runtime_error(std::string("yolov1_weights: short read of ") +
                              std::to_string(n) + " floats while reading " +
                              what);
  }
}

void fill(std::ifstream& f, at::Tensor& t, const char* what) {
  if (!t.is_contiguous()) t = t.contiguous();
  if (t.scalar_type() != torch::kFloat32) {
    throw std::runtime_error(std::string("yolov1_weights: ") + what +
                              " is not float32");
  }
  read_floats(f, t.data_ptr<float>(), (std::size_t)t.numel(), what);
}

// DFS the Yolo1 module in registration order. We expect a flat
// sequence of bare Conv2d nodes (no BN) inside `backbone`, followed
// by the two `fc1` / `fc2` Linear modules. MaxPool2d / Dropout /
// Functional carry no parameters and are skipped.
void walk(std::ifstream& f, torch::nn::Module& m, const std::string& name,
          int& count) {
  if (auto* c2 = m.as<torch::nn::Conv2dImpl>()) {
    if (!c2->bias.defined() || c2->bias.numel() == 0) {
      throw std::runtime_error("yolov1_weights: bare Conv2d " + name +
                                " has no bias tensor");
    }
    fill(f, c2->bias,   (name + ".bias").c_str());
    fill(f, c2->weight, (name + ".weight").c_str());
    ++count;
    return;
  }
  if (auto* ln = m.as<torch::nn::LinearImpl>()) {
    fill(f, ln->bias,   (name + ".bias").c_str());
    fill(f, ln->weight, (name + ".weight").c_str());
    ++count;
    return;
  }
  for (auto& kv : m.named_children()) {
    walk(f, *kv.value(), name.empty() ? kv.key() : (name + "." + kv.key()),
          count);
  }
}

}  // namespace

int convert_yolov1_weights(const std::string& weights_path,
                            const std::string& out_pt_path, int nc) {
  std::ifstream f(weights_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("yolov1_weights: cannot open " + weights_path);
  }
  // v1 header: 4× int32 (major, minor, revision, seen).
  std::int32_t major = 0, minor = 0, revision = 0, seen = 0;
  f.read(reinterpret_cast<char*>(&major),    4);
  f.read(reinterpret_cast<char*>(&minor),    4);
  f.read(reinterpret_cast<char*>(&revision), 4);
  f.read(reinterpret_cast<char*>(&seen),     4);
  if (!f) throw std::runtime_error("yolov1_weights: short read of header");

  // Construct the model and stream weights into it.
  models::Yolo1 model(nc);
  int count = 0;
  walk(f, *model.ptr(), "", count);

  // Check there are no trailing bytes.
  auto here = f.tellg();
  f.seekg(0, std::ios::end);
  auto end  = f.tellg();
  if (here != end) {
    throw std::runtime_error("yolov1_weights: " +
        std::to_string((long long)(end - here)) +
        " trailing bytes after " + std::to_string(count) +
        " blocks (header v" + std::to_string(major) + "." +
        std::to_string(minor) + "." + std::to_string(revision) + ")");
  }

  // Snapshot the model's parameters + float buffers into a flat
  // state-dict matching libtorch's named_parameters() convention so
  // `pt_loader::load_state_dict` reads it back identically.
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
  std::cerr << "[yolov1_weights] " << entries.size() << " tensors ("
            << count << " blocks) → " << out_pt_path
            << " (nc=" << nc << ", header v" << major << "." << minor
            << "." << revision << ")\n";
  return count;
}

}  // namespace yolocpp::serialization
