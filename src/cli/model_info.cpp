#include "yolocpp/cli/model_info.hpp"

#include "yolocpp/cli/resolve.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

namespace yolocpp::cli {

namespace {

// Channel count of model.0.conv.weight first dim (= base width * 8) → scale.
// Same table for v5 and v8 since both use 64*width as the stem channel count.
std::string scale_from_channels(int ch) {
  static const std::unordered_map<int, std::string> table = {
      {16, "n"}, {32, "s"}, {48, "m"}, {64, "l"}, {80, "x"},
  };
  auto it = table.find(ch);
  return it == table.end() ? std::string() : it->second;
}

// Walk the state_dict, infer (version, scale, nc) where possible.
// Empty fields mean "couldn't tell".
ModelInfo from_state_dict(const serialization::StateDict& sd) {
  ModelInfo info;
  for (const auto& [k, t] : sd.entries) {
    if (k == "model.0.conv.weight") {
      auto sz = t.sizes();
      if (sz.size() == 4) {
        int ch     = (int)sz[0];
        int kernel = (int)sz[2];
        info.version = (kernel == 6) ? "v5" : "v8";
        info.scale   = scale_from_channels(ch);
      }
    }
    // Detect-head class branch: cv3.<i>.2.weight → [nc, ...]
    if (info.nc < 0) {
      auto pos = k.rfind(".cv3.");
      if (pos != std::string::npos &&
          k.find(".2.weight", pos) != std::string::npos &&
          t.dim() >= 1) {
        info.nc = (int)t.size(0);
      }
    }
  }
  return info;
}

}  // namespace

ModelInfo infer_model_info(const std::string& pt_path) {
  ModelInfo info;

  // Primary: read the state_dict and inspect tensor shapes.
  try {
    auto sd = serialization::load_state_dict(pt_path);
    info = from_state_dict(sd);
  } catch (const std::exception& e) {
    std::cerr << "[warn] could not peek " << pt_path
              << " for auto-inference: " << e.what() << "\n";
  }

  // Fallback: filename + sibling args.yaml (handles best.pt / last.pt where
  // the basename has no version letter, but args.yaml records `model: ...`).
  if (info.version.empty()) info.version = version_from_filename(pt_path);
  if (info.scale.empty())   info.scale   = scale_from_filename(pt_path);

  if (info.version.empty())
    throw std::runtime_error(
        "could not infer YOLO version from '" + pt_path +
        "' — pass version=v5|v8 explicitly");
  if (info.scale.empty())
    throw std::runtime_error(
        "could not infer YOLO scale from '" + pt_path +
        "' — pass scale=n|s|m|l|x explicitly");

  return info;
}

}  // namespace yolocpp::cli
