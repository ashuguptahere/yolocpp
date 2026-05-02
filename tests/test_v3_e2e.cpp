// End-to-end test for YOLO3: convert upstream yolov3u.pt → yolo3.pt
// (fp16 → fp32, drop num_batches_tracked), load + predict on bus.jpg.
//
// Gated on `data/yolov3u.pt` or the cache. Skipped if no upstream weights.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/serialization/yolov3_weights.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp;

  std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  fs::path cache_w = fs::path(home) / ".cache/yolocpp/weights";

  std::vector<fs::path> w_candidates = {
      "data/yolov3u.pt", cache_w / "yolov3u.pt", "/tmp/yolov3u.pt",
  };
  fs::path src;
  for (const auto& c : w_candidates) {
    if (fs::exists(c)) { src = c; break; }
  }
  if (src.empty()) {
    std::cout << "[v3-e2e] SKIP: no yolov3u.pt available\n";
    return 0;
  }

  fs::path pt = "build/yolo3_e2e.pt";
  fs::remove(pt);
  std::cout << "[v3-e2e] converting " << src << " → " << pt << "\n";
  int n = serialization::convert_yolov3_pt(src.string(), pt.string());
  std::cout << "[v3-e2e] " << n << " tensors written\n";
  EXPECT(fs::exists(pt), "yolo3.pt not produced");
  EXPECT(n > 400, "expected > 400 tensors (got " + std::to_string(n) + ")");

  if (!fs::exists("data/bus.jpg")) {
    std::cout << "[v3-e2e] SKIP predict: data/bus.jpg missing\n";
    return 0;
  }
  inference::NMSConfig nm;
  nm.conf_thresh = 0.25f;
  nm.iou_thresh  = 0.45f;
  auto dets = inference::predict_v3_to_file(
      pt.string(), "data/bus.jpg", "build/v3_e2e_bus.jpg",
      /*imgsz=*/640, /*device=*/"", /*nc=*/80, nm);

  std::map<int, int> by_cls;
  for (auto& d : dets) ++by_cls[d.cls];
  std::cout << "[v3-e2e] " << dets.size() << " dets, person=" << by_cls[0]
            << " bus=" << by_cls[5] << " (top conf "
            << (dets.empty() ? 0.f : dets[0].conf) << ")\n";
  EXPECT(dets.size() >= 5, "v3: expected >= 5 detections on bus.jpg");
  EXPECT(by_cls[0] >= 4, "v3: expected at least 4 person detections");
  EXPECT(by_cls[5] >= 1, "v3: expected at least 1 bus detection");
  EXPECT(!dets.empty() && dets[0].conf > 0.85f,
         "v3: top detection should be >0.85 conf (parity check)");

  std::cout << "=== v3 end-to-end PASS ===\n";
  return 0;
}
