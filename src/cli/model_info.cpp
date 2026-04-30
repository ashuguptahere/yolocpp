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
// v8 / v5 / v11 / v26 all use 64*width as the stem channel count, but with
// different per-letter widths:
//   v5/v8 (n,s,m,l,x):  width = 0.25, 0.50, 0.75, 1.00, 1.25 → ch 16/32/48/64/80
//   v11/v26 (n,s,m,l,x): width = 0.25, 0.50, 1.00, 1.00, 1.50 → ch 16/32/64/64/96
// 64 is ambiguous for v11/v26 (m vs l). Caller must distinguish via depth
// (model.6.m.1 exists for v11l, not v11m).
std::string scale_from_channels(int ch, const std::string& version) {
  if (version == "v11" || version == "v26") {
    static const std::unordered_map<int, std::string> table = {
        {16, "n"}, {32, "s"}, {64, "m"}, {96, "x"},
    };
    auto it = table.find(ch);
    return it == table.end() ? std::string() : it->second;
  }
  static const std::unordered_map<int, std::string> v8_table = {
      {16, "n"}, {32, "s"}, {48, "m"}, {64, "l"}, {80, "x"},
  };
  auto it = v8_table.find(ch);
  return it == v8_table.end() ? std::string() : it->second;
}

// Walk the state_dict, infer (version, scale, nc) where possible.
// Empty fields mean "couldn't tell".
ModelInfo from_state_dict(const serialization::StateDict& sd) {
  ModelInfo info;
  // First pass: find PSA marker (C2PSA's attention) → v11 or v26.
  bool has_psa = false;
  // v26 distinguishes from v11 by the absence of a DFL in the head AND a
  // direct-4 cv2 output. Probe both signals.
  bool has_dfl = false;
  int  cv2_out_dim = -1;     // first dim of any cv2.<i>.2.weight
  for (const auto& [k, t] : sd.entries) {
    if (k.find(".m.0.attn.qkv.conv.weight") != std::string::npos &&
        (k.rfind("model.9.", 0) == 0 || k.rfind("model.10.", 0) == 0)) {
      has_psa = true;
    }
    if (k.find(".dfl.conv.weight") != std::string::npos) has_dfl = true;
    if (cv2_out_dim < 0) {
      auto pos = k.rfind(".cv2.");
      if (pos != std::string::npos &&
          k.find(".2.weight", pos) != std::string::npos &&
          t.dim() >= 1) {
        cv2_out_dim = (int)t.size(0);
      }
    }
  }
  // v26 signature: PSA marker + (no DFL || cv2 outputs 4 ch directly).
  bool is_v26 = has_psa && (!has_dfl || cv2_out_dim == 4);

  for (const auto& [k, t] : sd.entries) {
    if (k == "model.0.conv.weight") {
      auto sz = t.sizes();
      if (sz.size() == 4) {
        int ch     = (int)sz[0];
        int kernel = (int)sz[2];
        if (is_v26)              info.version = "v26";
        else if (has_psa)        info.version = "v11";
        else if (kernel == 6)    info.version = "v5";
        else                     info.version = "v8";
        info.scale = scale_from_channels(ch, info.version);
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

  // v11/v26 ch=64 is ambiguous: 'm' or 'l'. Disambiguate by checking
  // whether model.6 has m.1 (depth=2 → l) or only m.0 (depth=1 → m).
  if ((info.version == "v11" || info.version == "v26") && info.scale == "m") {
    for (const auto& [k, t] : sd.entries) {
      if (k.rfind("model.6.m.1.", 0) == 0) {
        info.scale = "l";
        break;
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
