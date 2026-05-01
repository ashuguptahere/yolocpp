// End-to-end test for YOLO9c: convert Ultralytics yolov9c.pt → yolo9.pt
// (RepConv fusion + fp16→fp32), load + predict on bus.jpg.
//
// Gated on `data/yolov9c.pt` or the cache. Skipped if no upstream weights.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/serialization/yolov9_weights.hpp"

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
      "data/yolov9c.pt", cache_w / "yolov9c.pt", "/tmp/yolov9c.pt",
  };
  fs::path src;
  for (const auto& c : w_candidates) {
    if (fs::exists(c)) { src = c; break; }
  }
  if (src.empty()) {
    std::cout << "[v9-e2e] SKIP: no yolov9c.pt available\n";
    return 0;
  }

  fs::path pt = "build/yolo9_e2e.pt";
  fs::remove(pt);
  std::cout << "[v9-e2e] converting " << src << " → " << pt << "\n";
  int rep = serialization::convert_yolov9_pt(src.string(), pt.string());
  std::cout << "[v9-e2e] " << rep << " RepConv blocks fused\n";
  EXPECT(fs::exists(pt), "yolo9.pt not produced");
  // yolov9c has 16 RepConv blocks (8 RepNCSPELAN4 in backbone+neck, each
  // containing 2 RepCSP each containing 1 RepBottleneck whose cv1 is a
  // RepConv = 8 * 2 * 1 = 16).
  EXPECT(rep == 16, "expected 16 RepConv blocks (got " + std::to_string(rep) + ")");

  if (!fs::exists("data/bus.jpg")) {
    std::cout << "[v9-e2e] SKIP predict: data/bus.jpg missing\n";
    return 0;
  }
  inference::NMSConfig nm;
  nm.conf_thresh = 0.25f;
  nm.iou_thresh  = 0.45f;
  auto dets = inference::predict_v9_to_file(
      pt.string(), "data/bus.jpg", "build/v9_e2e_bus.jpg",
      /*imgsz=*/640, /*device=*/"", /*nc=*/80, models::Yolo9Scale::C, nm);

  std::map<int, int> by_cls;
  for (auto& d : dets) ++by_cls[d.cls];
  std::cout << "[v9-e2e] " << dets.size() << " dets, person=" << by_cls[0]
            << " bus=" << by_cls[5] << " (top conf "
            << (dets.empty() ? 0.f : dets[0].conf) << ")\n";
  EXPECT(dets.size() >= 5, "v9: expected >= 5 detections on bus.jpg");
  EXPECT(by_cls[0] >= 4, "v9: expected at least 4 person detections");
  EXPECT(by_cls[5] >= 1, "v9: expected at least 1 bus detection");
  EXPECT(!dets.empty() && dets[0].conf > 0.85f,
         "v9: top detection should be >0.85 conf (parity check)");

  // ─── Scale variants — t / s / m round-trip on the same converter ─────
  for (auto p : {std::pair<std::string, models::Yolo9Scale>{"t", models::Yolo9Scale::T},
                 std::pair<std::string, models::Yolo9Scale>{"s", models::Yolo9Scale::S},
                 std::pair<std::string, models::Yolo9Scale>{"m", models::Yolo9Scale::M},
                 std::pair<std::string, models::Yolo9Scale>{"e", models::Yolo9Scale::E}}) {
    std::vector<fs::path> roots = {
        "data/yolov9" + p.first + ".pt",
        cache_w / ("yolov9" + p.first + ".pt"),
        "/tmp/yolov9" + p.first + ".pt",
    };
    fs::path src_n;
    for (const auto& c : roots) if (fs::exists(c)) { src_n = c; break; }
    if (src_n.empty()) {
      std::cout << "[v9-e2e] SKIP yolo9" << p.first << " (no upstream weights)\n";
      continue;
    }
    fs::path pt_n = "build/yolo9" + p.first + "_e2e.pt";
    fs::remove(pt_n);
    serialization::convert_yolov9_pt(src_n.string(), pt_n.string());
    auto dets_n = inference::predict_v9_to_file(
        pt_n.string(), "data/bus.jpg",
        "build/v9" + p.first + "_e2e_bus.jpg",
        640, "", 80, p.second, nm);
    std::map<int, int> by;
    for (auto& d : dets_n) ++by[d.cls];
    std::cout << "[v9-e2e] yolo9" << p.first << ": " << dets_n.size()
              << " dets (top "
              << (dets_n.empty() ? 0.f : dets_n[0].conf) << ")\n";
    EXPECT(dets_n.size() >= 4, "v9" + p.first + ": >= 4 dets");
    EXPECT(by[0] >= 3, "v9" + p.first + ": >= 3 person");
    EXPECT(by[5] >= 1, "v9" + p.first + ": >= 1 bus");
  }

  std::cout << "=== v9 end-to-end PASS ===\n";
  return 0;
}
