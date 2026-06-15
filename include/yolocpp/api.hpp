#pragma once
//
// yolocpp public C++ API — chainable, Python-style ergonomics.
//
// Mirrors the upstream Python `YOLO("model.pt").train(...)` shape so
// users porting from a Python pipeline can transliterate calls. Under
// the hood every method routes to the same `cmd_*` bodies the CLI
// uses, so behaviour is identical to `yolocpp --mode <action> ...`.
//
// Usage:
//
//   #include <yolocpp/api.hpp>
//
//   yolocpp::YOLO m("yolo11s.pt");
//   m.predict({.source = "bus.jpg"});                       // image
//   m.predict({.source = "frames/", .out = "annotated/"});  // dir fan-out
//   m.predict({.source = "video.mp4"});                     // frame loop
//   auto v = m.val({.data = "coco/data.yaml"});
//   m.train({.data = "coco/data.yaml", .epochs = 100, .seed = 42,
//            .export_after_train = "onnx"});
//   m.export_({.format = "onnx", .precision = "fp16"});
//
// Every Args struct uses C++20 designated initialisers — fields are
// optional with sensible defaults; pass only what you need. Methods
// return either:
//   - `YOLO&` (chainable, when there's no return value to surface):
//       train(...), predict(...), export_(...), benchmark(...),
//       to(device)
//   - a result struct (val).

#include <optional>
#include <string>
#include <vector>

#include "yolocpp/inference/nms.hpp"
#include "yolocpp/inference/predictor.hpp"

namespace yolocpp {

// ─── argument structs ────────────────────────────────────────────────────
//
// Every field optional; unspecified fields fall through to the
// matching CLI default. For knobs that have no natural sentinel,
// std::optional makes "user explicitly set" distinguishable from
// "left at default".

struct PredictArgs {
  std::string source;       // image / dir / glob / video / URL / webcam-idx
  std::string out;          // output path (file or dir); default: runs/predict/
  int         imgsz = 640;
  std::string device;       // cpu | cuda | cuda:N | mps | auto (empty)
  std::string scale;        // n/s/m/l/x; auto-resolved from filename if empty
  int         nc = -1;      // -1 → task default (80 detect/seg, 1000 cls, 15 obb)
  float       conf = 0.25f;
  float       iou  = 0.45f;
  std::string task = "detect";
};

struct ValArgs {
  std::string data;         // dataset root or data.yaml
  std::string names;        // comma-separated class names
  int         imgsz = 640;
  std::string device;
  std::string scale;
  std::string task = "detect";
};

struct ValResult {
  double map_50    = 0.0;
  double map_50_95 = 0.0;
};

struct TrainArgs {
  std::string data;
  std::string names;
  int         imgsz = 640;
  int         epochs = 100;
  int         batch  = 16;
  double      lr0    = 0.01;
  std::string device;
  std::string scale;
  std::string save = "runs/train";
  int         patience = 0;
  std::uint64_t seed   = 0;
  std::string export_after_train;  // "" | "onnx" | "trt" | "onnx,trt"
  std::string task = "detect";
};

struct ExportArgs {
  std::string format    = "onnx";  // onnx | trt
  std::string out;
  int         imgsz     = 640;
  std::string scale;
  int         nc        = -1;   // -1 → recover the class count from the
                                //      checkpoint head (pass to override)
  std::string input_name = "images";
  std::string precision  = "fp16"; // fp32 | fp16 | int8 (int4/nvfp4 — #51F2)
  std::string task       = "detect";
  // INT8 PTQ (#51F2): directory of representative images (e.g. a val split)
  // used to calibrate INT8 ranges. Required when precision == "int8" + TRT.
  std::string int8_calib_dir;
  std::string int8_calib_cache;  // optional; defaults to "<out>.calib"
};

struct BenchmarkArgs {
  std::string source;
  int         imgsz = 640;
  int         warmup = 10;
  int         iters  = 100;
  std::string cache  = "build/bench_cache";
  std::string device;
};

// ─── public API ──────────────────────────────────────────────────────────

class YOLO {
 public:
  // Construct from a weights path (`.pt` or `.trt`). The file is
  // resolved through the same machinery the CLI uses (cwd, ./data,
  // ~/.cache/yolocpp/weights, then upstream URL fetch).
  explicit YOLO(std::string weights);

  // Override the default device for every subsequent call. Equivalent
  // to passing `device=...` in each Args struct. Validated.
  YOLO& to(std::string device);

  // Set the default task for every subsequent call. Args structs can
  // override per-call.
  YOLO& task(std::string task);

  // Run inference. For Image/Dir/Glob sources, returns the dets from
  // the LAST processed image. For Video/URL/Webcam, returns an empty
  // vector (use the on-disk annotated mp4 instead).
  std::vector<inference::Detection> predict(const PredictArgs& a);

  // #52A3: like predict(), but returns per-input dets keyed by input path
  // (sorted, the same order they're written to disk). For a single image
  // the vector has one entry; for a dir/glob it has one per image. Throws
  // on error. Video/URL/Webcam sources return an empty vector (per-frame
  // dets live in the on-disk mp4).
  std::vector<std::pair<std::string, std::vector<inference::Detection>>>
  predict_many(const PredictArgs& a);

  // Run validation. Returns mAP@0.5 and mAP@0.5:0.95 (detect) or
  // task-specific scalar metrics that we squeeze into the same
  // struct (top-1 accuracy fills map_50; top-5 fills map_50_95 for
  // classify, etc.). Throws on error.
  ValResult val(const ValArgs& a);

  // Train. Chainable — model state isn't held in this object (the
  // CLI command-bodies snapshot weights from disk), so calling
  // `train()` then `val()` validates the post-train weights only if
  // you pass the new `weights` path explicitly via a fresh `YOLO`
  // instance, or if `export_after_train` was set and you re-construct.
  YOLO& train(const TrainArgs& a);

  // Export. Trailing underscore matches the upstream Python API
  // (`.export(...)` is reserved-word in Python, hence `.export_`).
  YOLO& export_(const ExportArgs& a);

  // Run a latency / throughput benchmark. Prints the table; doesn't
  // return the numbers (use the CLI for scriptable output).
  YOLO& benchmark(const BenchmarkArgs& a);

  // Accessors.
  const std::string& weights() const { return weights_; }

 private:
  std::string weights_;
  std::string default_device_;
  std::string default_task_ = "detect";
};

}  // namespace yolocpp
