// Verify all five YOLOv8 detect scales (n/s/m/l/x) load Ultralytics weights
// and produce sensible predictions on bus.jpg. Same canonical sample
// Ultralytics uses to demo every scale.

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolov8.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp;

  struct Cfg {
    const char* path;
    models::YoloV8Scale scale;
    const char* name;
    long long expected_params;     // approximate (±1%)
  };

  Cfg cfgs[] = {
      {"data/yolov8n.pt", models::kYoloV8n, "n",  3157184},
      {"data/yolov8s.pt", models::kYoloV8s, "s", 11166560},
      {"data/yolov8m.pt", models::kYoloV8m, "m", 25902640},
      {"data/yolov8l.pt", models::kYoloV8l, "l", 43691520},
      {"data/yolov8x.pt", models::kYoloV8x, "x", 68229648},
  };

  for (const auto& c : cfgs) {
    std::cout << "[scale " << c.name << "] loading " << c.path << "...\n";
    inference::Predictor p(c.path, /*imgsz=*/640, /*device=*/"", /*nc=*/80, c.scale);

    long long params = 0;
    for (auto& t : p.model()->parameters()) params += t.numel();
    long long delta = std::abs(params - c.expected_params);
    double pct = (double)delta / (double)c.expected_params * 100.0;
    std::cout << "[scale " << c.name << "] params=" << params / 1e6
              << "M (expected " << c.expected_params / 1e6
              << "M, " << pct << "% off)\n";
    EXPECT(pct < 1.0, "param count off by >1%");

    auto dets = p.predict_to_file("data/bus.jpg",
                                  std::string("build/bus_") + c.name + ".jpg");
    std::map<std::string, int> by_class;
    const auto& names = inference::coco_names();
    for (const auto& d : dets) by_class[names[d.cls]]++;

    std::cout << "[scale " << c.name << "] " << dets.size() << " detections:";
    for (auto& [k, v] : by_class) std::cout << " " << k << "=" << v;
    std::cout << "\n";

    EXPECT(by_class["person"] >= 1, "expected ≥ 1 person");
    EXPECT(by_class["bus"]    >= 1, "expected ≥ 1 bus");
  }

  std::cout << "=== all 5 scales PASS ===\n";
  return 0;
}
