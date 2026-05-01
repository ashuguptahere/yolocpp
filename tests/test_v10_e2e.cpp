// End-to-end test for YOLO10n: convert Ultralytics yolov10n.pt → yolo10.pt
// (one2many head dropped, one2one renamed to cv2/cv3, RepVGGDW fusion +
// fp16→fp32), load + predict on bus.jpg.
//
// Gated on `data/yolov10n.pt` or the cache. Skipped if no upstream weights.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/serialization/yolov10_weights.hpp"

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
      "data/yolov10n.pt", cache_w / "yolov10n.pt", "/tmp/ult_v10n.pt", "/tmp/yolov10n.pt",
  };
  fs::path src;
  for (const auto& c : w_candidates) {
    if (fs::exists(c)) { src = c; break; }
  }
  if (src.empty()) {
    std::cout << "[v10-e2e] SKIP: no yolov10n.pt available\n";
    return 0;
  }

  fs::path pt = "build/yolo10n_e2e.pt";
  fs::remove(pt);
  std::cout << "[v10-e2e] converting " << src << " → " << pt << "\n";
  int rep = serialization::convert_yolov10_pt(src.string(), pt.string());
  std::cout << "[v10-e2e] " << rep << " RepVGGDW pairs fused\n";
  EXPECT(fs::exists(pt), "yolo10.pt not produced");
  // yolov10n has exactly 1 RepVGGDW pair (inside the single C2fCIB at
  // model.22 with lk=True, n=1).
  EXPECT(rep == 1, "expected 1 RepVGGDW pair (got " + std::to_string(rep) + ")");

  if (!fs::exists("data/bus.jpg")) {
    std::cout << "[v10-e2e] SKIP predict: data/bus.jpg missing\n";
    return 0;
  }
  inference::NMSConfig nm;
  nm.conf_thresh = 0.25f;
  nm.iou_thresh  = 0.45f;
  auto dets = inference::predict_v10_to_file(
      pt.string(), "data/bus.jpg", "build/v10_e2e_bus.jpg",
      /*imgsz=*/640, /*device=*/"", /*nc=*/80,
      models::kYolo10n, nm);

  std::map<int, int> by_cls;
  for (auto& d : dets) ++by_cls[d.cls];
  std::cout << "[v10-e2e] " << dets.size() << " dets, person=" << by_cls[0]
            << " bus=" << by_cls[5] << " (top conf "
            << (dets.empty() ? 0.f : dets[0].conf) << ")\n";
  EXPECT(dets.size() >= 5, "v10: expected >= 5 detections on bus.jpg");
  EXPECT(by_cls[0] >= 4, "v10: expected at least 4 person detections");
  EXPECT(by_cls[5] >= 1, "v10: expected at least 1 bus detection");
  EXPECT(!dets.empty() && dets[0].conf > 0.85f,
         "v10: top detection should be >0.85 conf (parity check)");

  std::cout << "=== v10 end-to-end PASS ===\n";
  return 0;
}
