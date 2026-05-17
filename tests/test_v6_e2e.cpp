// End-to-end test for YOLO6s: convert Meituan yolov6s.pt → yolo6s.pt
// (RepVGG fusion + key rename), load + predict on bus.jpg.
//
// Gated on `data/yolov6s.pt` or the cache. Skipped (returns 0) if no
// upstream weights are present.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/serialization/yolov6_weights.hpp"

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
      "data/yolov6s.pt", cache_w / "yolov6s.pt", "/tmp/yolov6s.pt",
  };
  fs::path src;
  for (const auto& c : w_candidates) {
    if (fs::exists(c)) { src = c; break; }
  }
  if (src.empty()) {
    std::cout << "[v6-e2e] SKIP: no yolov6s.pt available\n";
    return 0;
  }

  fs::path pt = "build/yolo6s_e2e.pt";
  fs::remove(pt);
  std::cout << "[v6-e2e] converting " << src << " → " << pt << "\n";
  int rep_blocks = serialization::convert_yolov6_pt(src.string(), pt.string());
  std::cout << "[v6-e2e] " << rep_blocks << " RepVGG blocks fused\n";
  EXPECT(fs::exists(pt), "yolo6s.pt not produced");
  // yolov6s has 35 RepVGG blocks (1 stem + 1 + 2 + 1 + 4 + 1 + 6 + 1 + 2 backbone, +
  //                                 4 + 4 + 4 + 4 = 16 in neck, plus the 1x1's).
  EXPECT(rep_blocks >= 30 && rep_blocks <= 45,
         "expected 30..45 RepVGG blocks (got " + std::to_string(rep_blocks) + ")");

  if (!fs::exists("data/bus.jpg")) {
    std::cout << "[v6-e2e] SKIP predict: data/bus.jpg missing\n";
    return 0;
  }
  // Strict-parity check: at default conf=0.25/iou=0.45 we expect exactly
  // 4 persons + 1 bus on bus.jpg (matches Meituan's reference inference).
  inference::NMSConfig nm;
  nm.conf_thresh = 0.25f;
  nm.iou_thresh  = 0.45f;
  auto dets = inference::predict_v6_to_file(
      pt.string(), "data/bus.jpg", "build/v6_e2e_bus.jpg",
      /*imgsz=*/640, /*device=*/"", /*nc=*/80, models::kYolo6s, /*p6=*/false, nm);

  std::map<int, int> by_cls;
  for (auto& d : dets) ++by_cls[d.cls];
  std::cout << "[v6-e2e] " << dets.size() << " dets, person=" << by_cls[0]
            << " bus=" << by_cls[5] << "\n";
  EXPECT(dets.size() >= 4, "v6: expected >= 4 detections on bus.jpg");
  EXPECT(by_cls[0] >= 3, "v6: expected at least 3 person detections");
  EXPECT(by_cls[5] >= 1, "v6: expected at least 1 bus detection");

  // ─── v6n / v6m round-trip — same converter; n/s/m use the same fusion
  // (RepConv / RepBlock for n/s; BepC3 + RepConv-inside-BottleRep for m).
  // v6l uses MBLABlock (m operates on full in_channels) — deferred.
  for (auto p : {std::pair<std::string, models::Yolo6Scale>{"n", models::kYolo6n},
                 std::pair<std::string, models::Yolo6Scale>{"m", models::kYolo6m},
                 std::pair<std::string, models::Yolo6Scale>{"l", models::kYolo6l}}) {
    std::vector<fs::path> roots = {
        "data/yolov6" + p.first + ".pt",
        cache_w / ("yolov6" + p.first + ".pt"),
        "/tmp/yolov6" + p.first + ".pt",
    };
    fs::path src_n;
    for (const auto& c : roots) if (fs::exists(c)) { src_n = c; break; }
    if (src_n.empty()) {
      std::cout << "[v6-e2e] SKIP v6" << p.first << " (no upstream weights)\n";
      continue;
    }
    fs::path pt_n = "build/yolo6" + p.first + "_e2e.pt";
    fs::remove(pt_n);
    serialization::convert_yolov6_pt(src_n.string(), pt_n.string());
    auto dets_n = inference::predict_v6_to_file(
        pt_n.string(), "data/bus.jpg",
        "build/v6" + p.first + "_e2e_bus.jpg",
        640, "", 80, p.second, /*p6=*/false, nm);
    std::map<int, int> by;
    for (auto& d : dets_n) ++by[d.cls];
    std::cout << "[v6-e2e] v6" << p.first << ": " << dets_n.size()
              << " dets, person=" << by[0] << " bus=" << by[5] << "\n";
    EXPECT(dets_n.size() >= 4, "v6" + p.first + ": >= 4 dets");
    EXPECT(by[0] >= 3, "v6" + p.first + ": >= 3 person");
    EXPECT(by[5] >= 1, "v6" + p.first + ": >= 1 bus");
  }

  std::cout << "=== v6 end-to-end PASS ===\n";
  return 0;
}
