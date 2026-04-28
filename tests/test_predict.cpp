// End-to-end inference test: load yolo8n.pt, run on bus.jpg, verify
// detections look reasonable (≥ 1 person, ≥ 1 bus) — same canonical sample
// Ultralytics uses to demo their releases.

#include <iostream>
#include <map>
#include <string>

#include "yolocpp/inference/predictor.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp::inference;

  Predictor pred("data/yolo8n.pt", /*imgsz=*/640);
  auto dets = pred.predict_to_file("data/bus.jpg",
                                   "build/predict_output.jpg", {});
  std::cout << "[predict] detections: " << dets.size() << "\n";

  std::map<std::string, int> counts;
  const auto& names = coco_names();
  for (const auto& d : dets) {
    auto& nm = names[d.cls];
    counts[nm]++;
    std::cout << "  " << nm << "  conf=" << d.conf
              << "  xyxy=(" << d.x1 << "," << d.y1 << ","
              << d.x2 << "," << d.y2 << ")\n";
  }

  EXPECT(dets.size() >= 3,                "expected ≥ 3 detections on bus.jpg");
  EXPECT(counts["person"] >= 1,           "expected ≥ 1 person");
  EXPECT(counts["bus"]    >= 1,           "expected ≥ 1 bus");

  std::cout << "=== predict test PASS ===\n";
  return 0;
}
