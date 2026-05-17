// End-to-end test for YOLO4: convert AlexeyAB yolov4.weights → yolo4.pt,
// load + predict on bus.jpg, sanity-check detections.
//
// Gated on `data/yolov4.weights` or the cache. Skipped (returns 0) if no
// weights are present so CI doesn't fail on machines that haven't fetched
// the 246 MiB binary.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/serialization/darknet_weights.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp;

  std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  fs::path cache_w = fs::path(home) / ".cache/yolocpp/weights";

  // Prefer the pre-converted `data/yolo4.pt` (produced by
  // `tools/convert_weights`). Only fall back to a one-shot
  // `.weights → .pt` conversion if no `.pt` exists locally.
  fs::path pt;
  for (const auto& c : std::vector<fs::path>{
           "data/yolo4.pt", cache_w / "yolo4.pt"}) {
    if (fs::exists(c)) { pt = c; break; }
  }
  if (pt.empty()) {
    fs::path src;
    for (const auto& c : std::vector<fs::path>{
             "data/yolov4.weights", "data/yolo4.weights",
             cache_w / "yolov4.weights", cache_w / "yolo4.weights"}) {
      if (fs::exists(c)) { src = c; break; }
    }
    if (src.empty()) {
      std::cout << "[v4-e2e] SKIP: no data/yolo4.pt nor yolov4.weights\n";
      return 0;
    }
    pt = "build/yolo4_e2e.pt";
    fs::remove(pt);
    std::cout << "[v4-e2e] converting " << src << " → " << pt << "\n";
    int blocks = serialization::convert_yolov4_weights(src.string(), pt.string());
    std::cout << "[v4-e2e] " << blocks << " conv blocks consumed\n";
    EXPECT(fs::exists(pt), "yolo4.pt not produced");
    EXPECT(blocks >= 100 && blocks <= 120,
           "expected 100..120 conv blocks (got " + std::to_string(blocks) + ")");
  }

  // Predict on bus.jpg.
  if (!fs::exists("data/bus.jpg")) {
    std::cout << "[v4-e2e] SKIP predict: data/bus.jpg missing\n";
    return 0;
  }
  inference::NMSConfig nm;
  nm.conf_thresh = 0.25f;
  nm.iou_thresh  = 0.45f;
  auto dets = inference::predict_v4_to_file(
      pt.string(), "data/bus.jpg", "build/v4_e2e_bus.jpg",
      /*imgsz=*/608, /*device=*/"", /*nc=*/80, nm);

  std::map<int, int> by_cls;
  for (auto& d : dets) ++by_cls[d.cls];
  std::cout << "[v4-e2e] " << dets.size() << " dets, person=" << by_cls[0]
            << " bus=" << by_cls[5] << "\n";

  EXPECT(dets.size() >= 4, "v4: expected >= 4 detections on bus.jpg");
  EXPECT(by_cls[0] >= 1, "v4: expected at least 1 person detection");
  EXPECT(by_cls[5] >= 1, "v4: expected at least 1 bus detection");

  std::cout << "=== v4 end-to-end PASS ===\n";
  return 0;
}
