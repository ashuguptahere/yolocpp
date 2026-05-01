// tests/test_registry.cpp — sanity check on the per-version registry.
//
// Asserts:
//   1. register_all_versions() seeds every supported version_id.
//   2. Each adapter has the minimum hooks wired (export_onnx + names).
//   3. supported_tasks contains "detect" everywhere; v8/v11/v26 carry
//      the full 5-task family.
//
// No model weights touched — pure metadata test.

#include <iostream>
#include <set>

#include "yolocpp/registry/version_adapter.hpp"

#define EXPECT(cond, msg) \
  do {                    \
    if (!(cond)) {        \
      std::cerr << "[FAIL] " << msg << "\n"; \
      return 1;           \
    }                     \
  } while (0)

int main() {
  using namespace yolocpp::registry;
  register_all_versions();

  auto& reg = Registry::instance();
  EXPECT(reg.ready(), "registry should be seeded after register_all_versions");

  const std::vector<std::string> expected = {
      "v3", "v4", "v5", "v6", "v7", "v8",
      "v9", "v10", "v11", "v12", "v13", "v26"};

  for (const auto& id : expected) {
    const auto* a = reg.find(id);
    EXPECT(a != nullptr, "registry should know version_id '" + id + "'");
    EXPECT(!a->display_name.empty(), id + " missing display_name");
    EXPECT(!a->default_export_basename.empty(),
           id + " missing default_export_basename");
    EXPECT(static_cast<bool>(a->export_onnx),
           id + " has no export_onnx hook");

    // predict_to_file is required for every version EXCEPT v8 (which
    // falls back to the unified inference::Predictor).
    if (id != "v8") {
      EXPECT(static_cast<bool>(a->predict_to_file),
             id + " has no predict_to_file hook");
    } else {
      EXPECT(!a->predict_to_file,
             "v8 must NOT register a predict_to_file (uses unified Predictor)");
    }

    std::set<std::string> tasks(a->supported_tasks.begin(),
                                a->supported_tasks.end());
    EXPECT(tasks.count("detect") == 1,
           id + " supported_tasks must include 'detect'");

    if (id == "v8" || id == "v11" || id == "v26") {
      for (const auto& t : {"detect", "classify", "segment", "pose", "obb"}) {
        EXPECT(tasks.count(t) == 1,
               id + " full task family: missing '" + t + "'");
      }
    }
  }

  // v10 declares the TF32 quirk — the only adapter that does today.
  EXPECT(reg.find("v10")->trt_disable_tf32,
         "v10 must declare trt_disable_tf32");
  EXPECT(!reg.find("v8")->trt_disable_tf32,
         "v8 must NOT declare trt_disable_tf32");

  // Unknown versions resolve to nullptr (caller-handled error path).
  EXPECT(reg.find("v99") == nullptr, "unknown version should be nullptr");

  std::cout << "[ok] registry has " << reg.known_ids().size()
            << " versions, all hooks wired\n";
  return 0;
}
