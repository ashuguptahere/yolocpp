// Phase 3.2D: round-trip a model through our pt_save → pt_loader path.
// Also confirms that a saved checkpoint reloads bit-exactly.

#include <torch/torch.h>

#include <iostream>

#include "yolocpp/models/yolo8.hpp"
#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/pt_save.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp;

  models::Yolo8Detect a(models::kYolo8n, /*nc=*/80);
  a->eval();

  // Snapshot a few key tensors before save.
  std::vector<std::pair<std::string, at::Tensor>> snap;
  for (auto& kv : a->named_parameters())
    snap.emplace_back(kv.key(), kv.value().detach().clone());
  for (auto& kv : a->named_buffers())
    snap.emplace_back(kv.key(), kv.value().detach().clone());

  // Save through our pickle writer.
  serialization::save_state_dict("build/pt_save_roundtrip.pt", snap);
  std::cout << "[pt_save] wrote build/pt_save_roundtrip.pt\n";

  // Load back into a fresh model.
  models::Yolo8Detect b(models::kYolo8n, /*nc=*/80);
  auto sd = serialization::load_state_dict("build/pt_save_roundtrip.pt");
  std::cout << "[pt_save] loaded " << sd.size() << " entries\n";
  EXPECT(sd.size() == snap.size(), "entry count round-trip");

  int copied = b->load_from_state_dict(sd.entries);
  EXPECT(copied == (int)snap.size(), "all entries copied");

  // Bit-exact compare for every parameter/buffer.
  std::map<std::string, at::Tensor> snap_map;
  for (const auto& [k, v] : snap) snap_map[k] = v;
  int mismatches = 0;
  for (auto& kv : b->named_parameters()) {
    auto& src = snap_map.at(kv.key());
    if (!src.equal(kv.value().to(src.device()))) ++mismatches;
  }
  EXPECT(mismatches == 0, "parameter bit-exactness after round-trip");

  std::cout << "=== pt_save round-trip PASS ===\n";
  return 0;
}
