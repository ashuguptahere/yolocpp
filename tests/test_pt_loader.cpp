// Verify the .pt weight loader can read Ultralytics yolov8n.pt and that
// every parameter shape lines up with our YOLOv8n's named_parameters().

#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

#include "yolocpp/models/yolov8.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) {                                                 \
      std::cerr << "[FAIL] " << msg << "\n";                       \
      return 1;                                                    \
    }                                                              \
  } while (0)

int main() {
  const std::string pt_path = "data/yolov8n.pt";
  std::cout << "[load] reading " << pt_path << "\n";
  auto sd = yolocpp::serialization::load_state_dict(pt_path);
  std::cout << "[load] state_dict has " << sd.size() << " entries\n";
  EXPECT(sd.size() > 0, "state_dict is empty");

  // Print first 10 keys for inspection.
  for (size_t i = 0; i < std::min<size_t>(10, sd.entries.size()); ++i) {
    const auto& [k, v] = sd.entries[i];
    std::cout << "  " << k << "  " << v.sizes() << "  " << v.dtype() << "\n";
  }
  std::cout << "  ...\n";
  for (size_t i = sd.entries.size() > 5 ? sd.entries.size() - 5 : 0;
       i < sd.entries.size(); ++i) {
    const auto& [k, v] = sd.entries[i];
    std::cout << "  " << k << "  " << v.sizes() << "\n";
  }

  // Build our model and compare named_parameters / named_buffers.
  yolocpp::models::YoloV8Detect model(yolocpp::models::kYoloV8n, /*nc=*/80);
  model->eval();

  std::set<std::string> ckpt_keys;
  for (const auto& [k, _] : sd.entries) ckpt_keys.insert(k);

  std::set<std::string> our_keys;
  for (const auto& kv : model->named_parameters()) our_keys.insert(kv.key());
  for (const auto& kv : model->named_buffers())    our_keys.insert(kv.key());

  std::cout << "[load] checkpoint keys: " << ckpt_keys.size()
            << ", our keys: " << our_keys.size() << "\n";

  // Find missing / extra.
  std::vector<std::string> missing, extra;
  for (const auto& k : our_keys)
    if (ckpt_keys.count(k) == 0) missing.push_back(k);
  for (const auto& k : ckpt_keys)
    if (our_keys.count(k) == 0) extra.push_back(k);

  std::cout << "[load] missing in checkpoint (first 10):\n";
  for (size_t i = 0; i < std::min<size_t>(10, missing.size()); ++i)
    std::cout << "  - " << missing[i] << "\n";
  std::cout << "[load] extra in checkpoint (first 10):\n";
  for (size_t i = 0; i < std::min<size_t>(10, extra.size()); ++i)
    std::cout << "  + " << extra[i] << "\n";

  // Shape check on intersection.
  int shape_mismatches = 0;
  for (const auto& [k, t] : sd.entries) {
    if (our_keys.count(k) == 0) continue;
    at::Tensor ours;
    if (auto* p = model->named_parameters().find(k)) ours = *p;
    else if (auto* b = model->named_buffers().find(k)) ours = *b;
    if (ours.defined() && ours.sizes() != t.sizes()) {
      if (shape_mismatches < 5)
        std::cerr << "  shape mismatch: " << k << " ours=" << ours.sizes()
                  << " ckpt=" << t.sizes() << "\n";
      ++shape_mismatches;
    }
  }
  std::cout << "[load] shape mismatches: " << shape_mismatches << "\n";

  EXPECT(missing.empty(), "some of our params not present in checkpoint");
  EXPECT(shape_mismatches == 0, "shape mismatches between ckpt and our model");

  std::cout << "=== pt-loader test PASS ===\n";
  return 0;
}
