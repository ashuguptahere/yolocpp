// End-to-end Phase 2 test:
//   1. Load yolov8n.pt
//   2. Export to ONNX
//   3. Build a TRT engine from the ONNX (FP16 on Blackwell)
//   4. Run inference on bus.jpg via the TRT runtime
//   5. Compare TRT detections to libtorch detections — labels and box
//      centers should match within tolerance.
//
// All artifacts are written under build/ for inspection.

#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/inference/trt_predictor.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/trt_export.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp;

  const std::string pt   = "data/yolov8n.pt";
  const std::string img  = "data/bus.jpg";
  const std::string onnx = "build/yolov8n.test.onnx";
  const std::string trt  = "build/yolov8n.test.trt";

  // 1. Build the model from .pt.
  models::YoloV8Detect model(models::kYoloV8n, /*nc=*/80);
  auto sd = serialization::load_state_dict(pt);
  model->load_from_state_dict(sd.entries);
  model->eval();

  // 2. Export ONNX.
  serialization::OnnxExportConfig ocfg; ocfg.imgsz = 640;
  serialization::export_yolov8_onnx(model, onnx, ocfg);
  std::cout << "[trt-test] wrote " << onnx << "\n";

  // 3. Build TRT engine.
  serialization::TrtBuildConfig tcfg;
  tcfg.imgsz = 640;
  tcfg.fp16  = true;
  tcfg.builder_opt_level = 1;   // keep test fast
  serialization::build_trt_engine(onnx, trt, tcfg);

  // 4. Inference via libtorch (reference) AND via TRT.
  inference::Predictor    lt(pt, 640);
  inference::TrtPredictor rt(trt, 640);

  cv::Mat src = cv::imread(img, cv::IMREAD_COLOR);
  EXPECT(!src.empty(), "could not load bus.jpg");
  auto lt_d = lt.predict(src);
  auto rt_d = rt.predict(src);

  // 5. Compare. Group by class label and check that TRT finds at least each
  // class libtorch found, with similar box centers.
  std::map<int, std::vector<inference::Detection>> by_cls_lt, by_cls_rt;
  for (auto& d : lt_d) by_cls_lt[d.cls].push_back(d);
  for (auto& d : rt_d) by_cls_rt[d.cls].push_back(d);

  std::cout << "[trt-test] libtorch dets:\n";
  for (auto& [c, v] : by_cls_lt) {
    std::cout << "  cls " << c << " (" << inference::coco_names()[c] << "): ";
    for (auto& d : v) std::cout << "(" << (int)d.x1 << "," << (int)d.y1 << ","
                                << (int)d.x2 << "," << (int)d.y2
                                << " " << d.conf << ") ";
    std::cout << "\n";
  }
  std::cout << "[trt-test] tensorrt dets:\n";
  for (auto& [c, v] : by_cls_rt) {
    std::cout << "  cls " << c << " (" << inference::coco_names()[c] << "): ";
    for (auto& d : v) std::cout << "(" << (int)d.x1 << "," << (int)d.y1 << ","
                                << (int)d.x2 << "," << (int)d.y2
                                << " " << d.conf << ") ";
    std::cout << "\n";
  }

  // For every libtorch detection, find a TRT detection of the same class
  // whose center is within 30 pixels and confidence is within 0.20.
  int matched = 0;
  for (auto& d : lt_d) {
    float cx = (d.x1 + d.x2) / 2, cy = (d.y1 + d.y2) / 2;
    auto it = by_cls_rt.find(d.cls);
    if (it == by_cls_rt.end()) continue;
    bool ok = false;
    for (auto& r : it->second) {
      float rx = (r.x1 + r.x2) / 2, ry = (r.y1 + r.y2) / 2;
      if (std::hypot(cx - rx, cy - ry) <= 30.0f &&
          std::abs(d.conf - r.conf) <= 0.20f) {
        ok = true; break;
      }
    }
    if (ok) ++matched;
  }
  std::cout << "[trt-test] matched " << matched << "/" << lt_d.size()
            << " libtorch detections within tolerance\n";

  EXPECT(lt_d.size() >= 4,             "libtorch should find ≥ 4 detections");
  EXPECT(matched >= (int)lt_d.size() - 1,
         "TRT engine should match all but at most one libtorch detection");

  // Write TRT-annotated image for visual inspection.
  rt.predict_to_file(img, "build/predict_bus_trt.jpg");

  std::cout << "=== trt export+predict test PASS ===\n";
  return 0;
}
