// Unit smoke for the draw-on-Mat task helpers (draw_segments / draw_poses /
// draw_obbs / draw_classify) that back the non-detect video frame loop. Loads
// the v8 task weights when present, predicts on bus.jpg, draws onto a clone,
// and asserts the clone was actually modified (annotation applied). SKIP-gated
// on missing weights so it's a pass in CI without the .pt cache.

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

#include "yolocpp/inference/task_predictors.hpp"
#include "test_weights.hpp"

#define EXPECT(cond, msg)                                              \
  do {                                                                 \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; }  \
  } while (0)

namespace {
// Sum of absolute per-pixel difference — > 0 means the draw mutated the frame.
double diff(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat d; cv::absdiff(a, b, d);
  auto s = cv::sum(d);
  return s[0] + s[1] + s[2];
}
}  // namespace

int main() {
  using namespace yolocpp::inference;
  using yolocpp::test::find_weight;

  cv::Mat img = cv::imread("data/bus.jpg", cv::IMREAD_COLOR);
  EXPECT(!img.empty(), "could not load data/bus.jpg");

  int ran = 0;

  if (std::string w = find_weight("yolov8n-seg.pt"); !w.empty()) {
    SegmentPredictor p(w, 640);
    auto insts = p.predict(img);
    cv::Mat canvas = img.clone();
    draw_segments(canvas, insts);
    EXPECT(diff(canvas, img) > 0.0, "draw_segments left the frame unchanged");
    std::cout << "[seg]  " << insts.size() << " insts → frame annotated\n";
    ++ran;
  } else std::cout << "[seg]  SKIP (no yolov8n-seg.pt)\n";

  if (std::string w = find_weight("yolov8n-pose.pt"); !w.empty()) {
    PosePredictor p(w, 640);
    auto insts = p.predict(img);
    cv::Mat canvas = img.clone();
    draw_poses(canvas, insts);
    EXPECT(diff(canvas, img) > 0.0, "draw_poses left the frame unchanged");
    std::cout << "[pose] " << insts.size() << " people → frame annotated\n";
    ++ran;
  } else std::cout << "[pose] SKIP (no yolov8n-pose.pt)\n";

  if (std::string w = find_weight("yolov8n-obb.pt"); !w.empty()) {
    OBBPredictor p(w, 1024);
    auto insts = p.predict(img, {.conf_thresh = 0.05f});
    cv::Mat canvas = img.clone();
    draw_obbs(canvas, insts);
    // OBB on a non-aerial image may find nothing; only assert mutation when it
    // did detect something — the goal is to verify the draw path, not detection.
    if (!insts.empty())
      EXPECT(diff(canvas, img) > 0.0, "draw_obbs left the frame unchanged");
    std::cout << "[obb]  " << insts.size() << " boxes → draw ran\n";
    ++ran;
  } else std::cout << "[obb]  SKIP (no yolov8n-obb.pt)\n";

  if (std::string w = find_weight("yolov8n-cls.pt"); !w.empty()) {
    ClassifyPredictor p(w, 224);
    auto r = p.predict(img, 5);
    cv::Mat canvas = img.clone();
    draw_classify(canvas, r);
    EXPECT(diff(canvas, img) > 0.0, "draw_classify left the frame unchanged");
    std::cout << "[cls]  top-1=" << (r.topk.empty() ? -1 : r.topk[0].first)
              << " → banner drawn\n";
    ++ran;
  } else std::cout << "[cls]  SKIP (no yolov8n-cls.pt)\n";

  std::cout << "=== task draw smoke " << (ran ? "PASS" : "SKIP (no weights)")
            << " (" << ran << "/4 ran) ===\n";
  return 0;
}
