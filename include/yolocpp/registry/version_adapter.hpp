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

#include <torch/torch.h>

namespace yolocpp::serialization {
struct OnnxExportConfig;
}
namespace yolocpp::inference {
struct NMSConfig;
struct Detection;
class  FramePredictor;
}
namespace yolocpp::datasets {
class YoloDataset;
}
namespace yolocpp::engine {
struct TrainConfig;
struct BenchConfig;
struct BenchResult;
}
namespace cv {
class Mat;
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
  // Returns the full detection vector for the processed image (so
  // both `YOLO::predict()` (#52A2) and the CLI's "[predict] N
  // detections, wrote ..." print line can read it). v8's hook is
  // empty — its callers fall back to `inference::Predictor`.
  std::function<std::vector<yolocpp::inference::Detection>(
      const std::string& weights,
      const std::string& source,
      const std::string& out,
      int imgsz,
      const std::string& device,
      const std::string& scale,
      int nc,
      const yolocpp::inference::NMSConfig& nm)>
      predict_to_file;

  // Validation: construct the right holder, load_state_dict, run
  // `engine::validate<M>(model, ds, device)`. Returns the full
  // mAP grid plus the S/M/L size-stratified breakdown (#54C).
  // Versions that fall back to the unified `inference::Predictor`
  // (currently v8) leave this empty.
  struct ValResult {
    double map_50          = 0.0;
    double map_50_95       = 0.0;
    double map_50_95_small = 0.0;
    double map_50_95_medium= 0.0;
    double map_50_95_large = 0.0;
    int    n_gt_small  = 0;
    int    n_gt_medium = 0;
    int    n_gt_large  = 0;
  };
  std::function<ValResult(const std::string& weights,
                          const std::string& scale,
                          int nc,
                          yolocpp::datasets::YoloDataset& ds,
                          const torch::Device& device)>
      run_val;

  // Detect-task training: construct the right holder, optionally
  // `load_state_dict` from `init_weights` (empty ⇒ from-scratch), then
  // run `engine::TrainerT<Holder>(model, ds, cfg).run()`. The unified
  // template handles every version with the matching `LossTraits<M>`
  // specialisation. v8 leaves this empty and falls back to the
  // `engine::Trainer = TrainerT<Yolo8Detect>` path.
  std::function<void(const std::string& init_weights,
                     const std::string& scale,
                     int nc,
                     yolocpp::datasets::YoloDataset train_ds,
                     const yolocpp::engine::TrainConfig& cfg)>
      run_train_detect;

  // PT FP32 benchmark — construct the right holder, wrap in
  // `engine::detail::GenericPredictor`, run `bench_one()`. Returns
  // the timed `BenchResult`. v8 leaves this empty (handled by the
  // legacy `inference::Predictor` benchmark fallback).
  std::function<yolocpp::engine::BenchResult(const yolocpp::engine::BenchConfig& cfg,
                                             const cv::Mat& img,
                                             const std::string& scale)>
      benchmark_pt;

  // Build a long-lived frame predictor (#51C2). Used by the CLI's
  // video / URL / webcam frame loop — loads weights once, then runs
  // many frames through `predict(frame, nm)`. Every non-v8 version
  // wires this; v8 has its own dedicated path via
  // `inference::Predictor::predict(cv::Mat)`.
  std::function<std::unique_ptr<yolocpp::inference::FramePredictor>(
      const std::string& weights,
      const std::string& scale,
      int nc,
      int imgsz,
      const std::string& device)>
      make_frame_predictor;
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
