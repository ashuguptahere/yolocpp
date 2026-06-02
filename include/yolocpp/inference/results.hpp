#pragma once
//
// Ultralytics-compatible `Results` wrapper around `vector<Detection>`.
// Same shape every model from every backend (PT / TRT / ONNX) returns
// (task #97). Mirrors `ultralytics.engine.results.Results.boxes` so
// downstream consumers can drop in either backend with no changes to
// the calling code.
//
// Quick start:
//
//   auto r = predictor.predict_results(bgr);
//   for (size_t i = 0; i < r.boxes.size(); ++i) {
//     const auto& d = r.boxes[i];
//     fmt::print("{} {:.2f}  ({},{})-({},{})\n",
//                r.class_name(d.cls), d.conf,
//                d.x1, d.y1, d.x2, d.y2);
//   }
//   r.plot("annotated.jpg");                  // draw boxes + labels
//   r.save_txt("dets.txt");                   // CLS CONF X1 Y1 X2 Y2 per row
//   auto j = r.json();                        // serialise to JSON string
//

#include <opencv2/core.hpp>

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "yolocpp/inference/predictor.hpp"  // Detection

namespace yolocpp::inference {

struct Results {
  // Detection list — same shape every yolo version returns. Stable
  // across all backends (Predictor / GenericPredictor / TrtPredictor).
  std::vector<Detection> boxes;

  // Original image (BGR) — kept so plot() / save() can render boxes
  // onto the same pixels the predictor saw. Empty when the user
  // constructed Results directly from a tensor or didn't keep the
  // input image around.
  cv::Mat orig_img;
  int     orig_w = 0;
  int     orig_h = 0;

  // Class id → name lookup. Loaded from data.yaml's `names:` field
  // (CLI / API path) or set explicitly by the caller. Empty → uses
  // COCO80 names if available; falls back to `std::to_string(cls)`.
  std::vector<std::string> names;

  // Per-phase wall-clock (ms). Populated by the predictor — matches
  // Ultralytics' `Results.speed` dict shape.
  struct Speed {
    double preprocess_ms  = 0.0;
    double inference_ms   = 0.0;
    double postprocess_ms = 0.0;
  } speed;

  // Returns the class name for box i (falls back to int-as-string).
  std::string class_name(int cls) const;

  // ── Coordinate accessors (Ultralytics parity) ──────────────────────
  // xyxy[i] = {x1, y1, x2, y2} in original-image pixel coords.
  // xyxyn[i] = same, normalised to [0, 1] by orig_w / orig_h.
  // xywh[i] = {cx, cy, w, h} in pixels.
  // xywhn[i] = same, normalised.
  std::vector<std::array<float, 4>> xyxy()  const;
  std::vector<std::array<float, 4>> xyxyn() const;
  std::vector<std::array<float, 4>> xywh()  const;
  std::vector<std::array<float, 4>> xywhn() const;
  std::vector<float>                conf()  const;
  std::vector<int>                  cls()   const;

  // ── Rendering / serialisation ──────────────────────────────────────
  // Draw boxes + labels onto a copy of `orig_img`, return it.
  cv::Mat plot(int line_thickness = 2, double font_scale = 0.5) const;

  // Write annotated image to disk. Returns true on success.
  bool save(const std::string& path) const;

  // Write per-line "cls conf x1 y1 x2 y2" to a text file.
  bool save_txt(const std::string& path, bool normalised = false) const;

  // Serialise to a JSON string. Schema:
  //   { "boxes": [ {"cls":0, "name":"person", "conf":0.91,
  //                 "xyxy":[x1,y1,x2,y2]}, ... ],
  //     "orig_shape": [h, w],
  //     "speed": {"preprocess":0.2, "inference":0.5, "postprocess":0.1} }
  std::string json() const;

  // Quick summary printable to a stream.
  void print(std::ostream& os) const;
};

}  // namespace yolocpp::inference
