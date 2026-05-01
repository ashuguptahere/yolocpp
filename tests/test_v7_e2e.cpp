// End-to-end test for YOLO7: convert WongKinYiu yolov7.pt → yolo7.pt
// (RepConv fusion + fp16→fp32), load + predict on bus.jpg.
//
// Gated on `data/yolov7.pt` or the cache. Skipped if no upstream weights.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/serialization/yolov7_weights.hpp"

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
      "data/yolov7.pt", cache_w / "yolov7.pt", "/tmp/yolov7.pt",
  };
  fs::path src;
  for (const auto& c : w_candidates) {
    if (fs::exists(c)) { src = c; break; }
  }
  if (src.empty()) {
    std::cout << "[v7-e2e] SKIP: no yolov7.pt available\n";
    return 0;
  }

  fs::path pt = "build/yolo7_e2e.pt";
  fs::remove(pt);
  std::cout << "[v7-e2e] converting " << src << " → " << pt << "\n";
  int rep = serialization::convert_yolov7_pt(src.string(), pt.string());
  std::cout << "[v7-e2e] " << rep << " RepConv blocks fused\n";
  EXPECT(fs::exists(pt), "yolo7.pt not produced");
  // yolov7-base has exactly 3 RepConv layers (102, 103, 104).
  EXPECT(rep == 3, "expected 3 RepConv blocks (got " + std::to_string(rep) + ")");

  if (!fs::exists("data/bus.jpg")) {
    std::cout << "[v7-e2e] SKIP predict: data/bus.jpg missing\n";
    return 0;
  }
  inference::NMSConfig nm;
  nm.conf_thresh = 0.25f;
  nm.iou_thresh  = 0.45f;
  auto dets = inference::predict_v7_to_file(
      pt.string(), "data/bus.jpg", "build/v7_e2e_bus.jpg",
      /*imgsz=*/640, /*device=*/"", /*nc=*/80, models::Yolo7Scale::Base, nm);

  std::map<int, int> by_cls;
  for (auto& d : dets) ++by_cls[d.cls];
  std::cout << "[v7-e2e] " << dets.size() << " dets, person=" << by_cls[0]
            << " bus=" << by_cls[5] << " (top conf "
            << (dets.empty() ? 0.f : dets[0].conf) << ")\n";
  EXPECT(dets.size() >= 4, "v7: expected >= 4 detections on bus.jpg");
  EXPECT(by_cls[0] >= 3, "v7: expected at least 3 person detections");
  EXPECT(by_cls[5] >= 1, "v7: expected at least 1 bus detection");
  EXPECT(!dets.empty() && dets[0].conf > 0.85f,
         "v7: top detection should be >0.85 conf (parity check)");

  // ─── tiny / x / w6 scale variants — same converter, same predict ──
  // w6 needs imgsz=1280 (4 detect levels P3-P6, 64-stride), so handle
  // separately from the 640 cases.
  for (auto p : {std::pair<std::string, std::pair<std::string, models::Yolo7Scale>>
                     {"yolov7-tiny.pt", {"yolo7-tiny.pt", models::Yolo7Scale::Tiny}},
                 std::pair<std::string, std::pair<std::string, models::Yolo7Scale>>
                     {"yolov7x.pt",     {"yolo7x.pt",     models::Yolo7Scale::X}}}) {
    std::vector<fs::path> roots = {
        "data/" + p.first, cache_w / p.first, "/tmp/" + p.first,
    };
    fs::path src_v;
    for (const auto& c : roots) if (fs::exists(c)) { src_v = c; break; }
    if (src_v.empty()) {
      std::cout << "[v7-e2e] SKIP " << p.second.first << " (no upstream)\n";
      continue;
    }
    fs::path pt_v = "build/" + p.second.first;
    fs::remove(pt_v);
    serialization::convert_yolov7_pt(src_v.string(), pt_v.string());
    auto dets_v = inference::predict_v7_to_file(
        pt_v.string(), "data/bus.jpg", "build/" + p.second.first + ".out.jpg",
        640, "", 80, p.second.second, nm);
    std::map<int, int> by_v;
    for (auto& d : dets_v) ++by_v[d.cls];
    std::cout << "[v7-e2e] " << p.second.first << ": " << dets_v.size()
              << " dets, person=" << by_v[0] << " bus=" << by_v[5] << "\n";
    EXPECT(dets_v.size() >= 4, p.second.first + ": >= 4 dets");
    EXPECT(by_v[0] >= 3, p.second.first + ": >= 3 person");
    EXPECT(by_v[5] >= 1, p.second.first + ": >= 1 bus");
  }

  // P6 variants — w6 / e6 / d6 / e6e (each at 1280×1280; 4 detect levels P3-P6).
  for (auto p : {std::pair<std::string, std::pair<std::string, models::Yolo7Scale>>
                     {"yolov7-w6.pt",  {"yolo7-w6.pt",  models::Yolo7Scale::W6}},
                 std::pair<std::string, std::pair<std::string, models::Yolo7Scale>>
                     {"yolov7-e6.pt",  {"yolo7-e6.pt",  models::Yolo7Scale::E6}},
                 std::pair<std::string, std::pair<std::string, models::Yolo7Scale>>
                     {"yolov7-d6.pt",  {"yolo7-d6.pt",  models::Yolo7Scale::D6}},
                 std::pair<std::string, std::pair<std::string, models::Yolo7Scale>>
                     {"yolov7-e6e.pt", {"yolo7-e6e.pt", models::Yolo7Scale::E6e}}}) {
    std::vector<fs::path> roots = {
        "data/" + p.first, cache_w / p.first, "/tmp/" + p.first,
    };
    fs::path src_v;
    for (const auto& c : roots) if (fs::exists(c)) { src_v = c; break; }
    if (src_v.empty()) {
      std::cout << "[v7-e2e] SKIP " << p.second.first << " (no upstream)\n";
      continue;
    }
    fs::path pt_v = "build/" + p.second.first;
    fs::remove(pt_v);
    serialization::convert_yolov7_pt(src_v.string(), pt_v.string());
    auto dets_v = inference::predict_v7_to_file(
        pt_v.string(), "data/bus.jpg", "build/" + p.second.first + ".out.jpg",
        /*imgsz=*/1280, "", 80, p.second.second, nm);
    std::map<int, int> by_v;
    for (auto& d : dets_v) ++by_v[d.cls];
    std::cout << "[v7-e2e] " << p.second.first << ": " << dets_v.size()
              << " dets, person=" << by_v[0] << " bus=" << by_v[5] << "\n";
    EXPECT(dets_v.size() >= 4, p.second.first + ": >= 4 dets");
    EXPECT(by_v[0] >= 3, p.second.first + ": >= 3 person");
    EXPECT(by_v[5] >= 1, p.second.first + ": >= 1 bus");
  }

  std::cout << "=== v7 end-to-end PASS ===\n";
  return 0;
}
