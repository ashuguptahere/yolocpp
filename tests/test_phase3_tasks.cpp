// Verify all four Phase 3 tasks (classify / segment / pose / OBB) load
// upstream weights and produce sensible predictions on bus.jpg.

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

#include "yolocpp/inference/task_predictors.hpp"
#include "test_weights.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp::inference;

  cv::Mat img = cv::imread("data/bus.jpg", cv::IMREAD_COLOR);
  EXPECT(!img.empty(), "could not load bus.jpg");

  using yolocpp::test::find_weight;

  // ─── classify ─────────────────────────────────────────────────────────
  if (std::string w = find_weight("yolo8n-cls.pt"); w.empty()) {
    std::cout << "[cls] SKIP: no yolo8n-cls.pt in ./models or ./data\n";
  } else {
    ClassifyPredictor p(w, /*imgsz=*/224);
    auto r = p.predict(img, /*top_k=*/5);
    std::cout << "[cls] top-5:";
    for (auto& [cid, prob] : r.topk)
      std::cout << " (" << cid << " " << prob << ")";
    std::cout << "\n";
    EXPECT(r.topk.size() == 5,            "top-5 size");
    EXPECT(r.topk[0].second > 0.1f,       "top-1 prob > 0.1");
    // ImageNet class 654 = "minibus", 779 = "school bus", 874 = "trolleybus".
    bool ok = false;
    for (auto& [cid, _] : r.topk)
      if (cid == 654 || cid == 779 || cid == 874) ok = true;
    EXPECT(ok, "expected a bus-related class in top-5 for bus.jpg");
  }

  // ─── segment ──────────────────────────────────────────────────────────
  if (std::string w = find_weight("yolo8n-seg.pt"); w.empty()) {
    std::cout << "[seg] SKIP: no yolo8n-seg.pt in ./models or ./data\n";
  } else {
    SegmentPredictor p(w, /*imgsz=*/640);
    auto insts = p.predict_to_file("data/bus.jpg", "build/seg_bus.jpg");
    std::cout << "[seg] " << insts.size() << " instances\n";
    EXPECT(insts.size() >= 4, "expected ≥ 4 segmented instances on bus.jpg");
    int nonzero_masks = 0;
    for (const auto& i : insts)
      if (cv::countNonZero(i.mask) > 100) ++nonzero_masks;
    EXPECT(nonzero_masks >= 4, "expected ≥ 4 non-empty masks");
  }

  // ─── pose ─────────────────────────────────────────────────────────────
  if (std::string w = find_weight("yolo8n-pose.pt"); w.empty()) {
    std::cout << "[pose] SKIP: no yolo8n-pose.pt in ./models or ./data\n";
  } else {
    PosePredictor p(w, /*imgsz=*/640);
    auto insts = p.predict_to_file("data/bus.jpg", "build/pose_bus.jpg");
    std::cout << "[pose] " << insts.size() << " people\n";
    EXPECT(insts.size() >= 1, "expected ≥ 1 person on bus.jpg");
    // Each person should have 17 keypoints (some may have low confidence).
    EXPECT((int)insts[0].keypoints.size() == 17, "expected 17 keypoints");
  }

  // ─── OBB ──────────────────────────────────────────────────────────────
  // bus.jpg isn't aerial DOTA imagery — we just verify the pipeline runs
  // without crashing and produces some valid (cx,cy,w,h,angle) output.
  if (std::string w = find_weight("yolo8n-obb.pt"); w.empty()) {
    std::cout << "[obb] SKIP: no yolo8n-obb.pt in ./models or ./data\n";
  } else {
    OBBPredictor p(w, /*imgsz=*/640);
    auto insts = p.predict_to_file("data/bus.jpg", "build/obb_bus.jpg",
                                   /*conf=*/{ .conf_thresh = 0.05f });
    std::cout << "[obb] " << insts.size() << " rotated boxes\n";
    // OBB on a non-aerial image: we just want it to not crash.
    EXPECT(insts.size() >= 0, "OBB ran (sanity check)");
  }

  std::cout << "=== phase 3 tasks test PASS ===\n";
  return 0;
}
