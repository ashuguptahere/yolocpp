#pragma once
//
// Per-version registry — single point where every supported YOLO variant
// registers a handful of std::function<>s describing how to construct,
// load, run, and export it. This lets the CLI / API call
// `Registry::instance().find("v11")->export_onnx(...)` in one line
// instead of carrying a 250-line if-else chain that grows every time a
// new version lands.
//
// **How to add a new YOLO version (one-pass walkthrough):**
//
// 1. Drop your model implementation into `src/models/yolo<N>.cpp` /
//    `include/yolocpp/models/yolo<N>.hpp` (existing convention — see
//    yolo12 / yolo13 for templates).
//
// 2. Implement a `register_yolo<N>(Registry&)` helper that fills out a
//    `VersionAdapter` and pushes it. Convention: live in
//    `src/registry/register_yolo<N>.cpp` next to the existing
//    per-version registrations.
//
// 3. Add the call to `register_yolo<N>(reg)` inside
//    `register_all_versions(...)` in `src/registry/version_registry.cpp`.
//
// 4. Wire the new TU into `CMakeLists.txt`'s `yolocpp_core` source list.
//
// That's it. No edits to `cli/main.cpp`, no edits to the export
// pipeline, no edits to the predict/train dispatch (once those are
// fully on the registry — currently export is migrated; predict / val
// / train follow in subsequent tasks #46D…).

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace yolocpp::serialization {
struct OnnxExportConfig;
}
namespace yolocpp::inference {
struct NMSConfig;
}

namespace yolocpp::registry {

// Erased operations the registry knows how to do for a version. New
// hooks (predict_holder, train_holder, ...) get added here as the
// migration progresses; keep them std::function so versions that
// don't support a hook can leave it empty and the caller can detect
// the gap with `if (adapter.export_onnx)`.
struct VersionAdapter {
  // Identifiers.
  std::string version_id;     // "v3", "v8", "v11", ...
  std::string display_name;   // "yolo3", "yolo8", ...
  std::string upstream_year;  // "2018", "2024", ... (display only)

  // Default export filename basename (`runs/export/<base>.onnx`).
  std::string default_export_basename;  // "yolo3", "yolo8", ...

  // Tasks this version supports through the export pipeline. Matches
  // the CLI `task=` values: "detect", "classify", "segment", "pose",
  // "obb". Most versions are detect-only.
  std::vector<std::string> supported_tasks;

  // Resolve the natural training imgsz for a given (scale, task). 0
  // means "use caller default (640)". Examples: v4 anchors are
  // calibrated to 608; v6/v7 P6 variants are 1280; classify is 224.
  std::function<int(const std::string& scale,
                    const std::string& task)> default_imgsz;

  // Export to ONNX. Implementations construct the holder of the
  // correct concrete type, load_state_dict, optionally forward_eval to
  // populate strides, and call the per-version `export_yolo<N>_onnx`
  // emitter.
  std::function<void(const std::string& weights,
                     const std::string& scale,
                     int nc,
                     const std::string& task,
                     const std::string& onnx_path,
                     const yolocpp::serialization::OnnxExportConfig& cfg)>
      export_onnx;

  // True if TRT engine builds for this version need TF32 explicitly
  // disabled (v10 RepVGGDW saturation).
  bool trt_disable_tf32 = false;

  // Predict to a file. Returns the number of detections written.
  // Versions that fall back to the unified `inference::Predictor` (v8
  // and any anchor-free model whose state-dict shape is v8-compatible)
  // leave this empty — the CLI dispatcher routes those to
  // `cmd_predict` instead. See registry::Registry::find().
  std::function<std::size_t(const std::string& weights,
                            const std::string& source,
                            const std::string& out,
                            int imgsz,
                            const std::string& device,
                            const std::string& scale,
                            int nc,
                            const yolocpp::inference::NMSConfig& nm)>
      predict_to_file;
};

class Registry {
 public:
  static Registry& instance();

  // Add an adapter; idempotent on duplicate version_id (later wins,
  // logs once).
  void register_version(VersionAdapter a);

  // Lookup by version_id ("v3", "v8", ...). Returns nullptr if
  // unknown.
  const VersionAdapter* find(const std::string& version_id) const;

  // Sorted list of all known version ids (for diagnostics / `info`
  // output).
  std::vector<std::string> known_ids() const;

  // True if the registry has been seeded by `register_all_versions`.
  bool ready() const { return !versions_.empty(); }

 private:
  Registry() = default;
  std::unordered_map<std::string, VersionAdapter> versions_;
};

// Seed the singleton with every supported YOLO version. Idempotent —
// safe to call from multiple entry points (CLI main, public API, tests).
void register_all_versions();

}  // namespace yolocpp::registry
