#pragma once
//
// Auto-inference of YOLO model attributes from a .pt checkpoint.
//
// Reads the state_dict and inspects layer shapes:
//   model.0.conv.weight kernel size  → version (k=6 → v5, k=3 → v8)
//   model.0.conv.weight out_channels → scale  (16/32/48/64/80 → n/s/m/l/x)
//   <head>.cv3.0.2.weight out_ch     → number of classes
//
// Falls back to filename heuristics + sibling args.yaml when the .pt
// can't be parsed (e.g., user typed a non-existent path).

#include <string>

namespace yolocpp::cli {

struct ModelInfo {
  std::string version;   // "v5" or "v8" (others as we add them)
  std::string scale;     // "n" / "s" / "m" / "l" / "x"
  int         nc = -1;   // number of classes (-1 if unknown)
};

// Infer everything we can from `pt_path`. Throws if the file isn't a
// readable Ultralytics checkpoint *and* the filename/sibling-args.yaml
// fallbacks also yield nothing.
ModelInfo infer_model_info(const std::string& pt_path);

}  // namespace yolocpp::cli
