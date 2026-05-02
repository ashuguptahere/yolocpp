// Cross-backend parity assert (#53A).
//
// For every model whose `.pt` is present locally, run inference on
// `data/bus.jpg` through three backends:
//
//   1. libtorch FP32  (.pt via the registry's FramePredictor)
//   2. TRT FP32       (built from our ONNX exporter)
//   3. TRT FP16       (same ONNX, fp16 builder flag)
//
// Assert:
//   - det count matches libtorch within ±1 (FP precision can shift
//     one border-conf detection in or out)
//   - for every libtorch detection there is a same-class TRT
//     detection with IoU ≥ 0.50 (cls + bbox parity holds across
//     backends)
//
// Discovery-driven: SKIPs cleanly when a model's `.pt` isn't on
// disk, so the test passes on a fresh checkout.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "yolocpp/inference/frame_predictor.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/inference/trt_predictor.hpp"
#include "yolocpp/registry/version_adapter.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/trt_export.hpp"

namespace fs = std::filesystem;

namespace {

float box_iou(const yolocpp::inference::Detection& a,
              const yolocpp::inference::Detection& b) {
  float x1 = std::max(a.x1, b.x1);
  float y1 = std::max(a.y1, b.y1);
  float x2 = std::min(a.x2, b.x2);
  float y2 = std::min(a.y2, b.y2);
  float iw = std::max(0.f, x2 - x1);
  float ih = std::max(0.f, y2 - y1);
  float inter = iw * ih;
  float aa = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
  float ba = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
  float un = aa + ba - inter;
  return un > 0 ? inter / un : 0.f;
}

int match_iou(const std::vector<yolocpp::inference::Detection>& ref,
              const std::vector<yolocpp::inference::Detection>& test,
              float iou_thresh) {
  std::map<int, std::vector<yolocpp::inference::Detection>> by_cls;
  for (const auto& d : test) by_cls[d.cls].push_back(d);
  int matched = 0;
  for (const auto& r : ref) {
    auto it = by_cls.find(r.cls);
    if (it == by_cls.end()) continue;
    for (const auto& t : it->second) {
      if (box_iou(r, t) >= iou_thresh) { ++matched; break; }
    }
  }
  return matched;
}

// Returns 0 PASS, 1 FAIL, 2 SKIP.
int run_cell(const std::string& pt_path, const std::string& version,
             const std::string& scale, int nc, int imgsz) {
  using namespace yolocpp;

  if (!fs::exists(pt_path)) {
    std::cout << "[parity] SKIP " << pt_path << " (not on disk)\n";
    return 2;
  }
  if (!fs::exists("data/bus.jpg")) {
    std::cout << "[parity] SKIP — data/bus.jpg missing\n";
    return 2;
  }

  std::string stem = fs::path(pt_path).stem().string();
  fs::create_directories("build/parity_cache");
  std::string onnx_path = "build/parity_cache/" + stem + ".onnx";
  std::string trt_fp32  = "build/parity_cache/" + stem + ".fp32.trt";
  std::string trt_fp16  = "build/parity_cache/" + stem + ".fp16.trt";

  // (A) Export ONNX via the registry — same path as `--mode export`.
  registry::register_all_versions();
  const auto* adapter = registry::Registry::instance().find(version);
  if (!adapter || !adapter->export_onnx) {
    std::cerr << "[parity] " << version << ": no export_onnx hook\n";
    return 1;
  }
  serialization::OnnxExportConfig ocfg; ocfg.imgsz = imgsz;
  adapter->export_onnx(pt_path, scale, nc, /*task=*/"detect", onnx_path, ocfg);

  // (B) Build two TRT engines (fp32, fp16). Cache to disk so repeat
  // ctest runs reuse them. opt_level=1 keeps the test fast.
  auto build_trt = [&](const std::string& out, bool fp16) {
    if (fs::exists(out)) return;
    serialization::TrtBuildConfig tcfg;
    tcfg.imgsz = imgsz;
    tcfg.fp16 = fp16;
    tcfg.tf32 = !adapter->trt_disable_tf32;
    tcfg.builder_opt_level = 1;
    serialization::build_trt_engine(onnx_path, out, tcfg);
  };
  build_trt(trt_fp32, /*fp16=*/false);
  build_trt(trt_fp16, /*fp16=*/true);

  // (C) Run all three backends on the same image.
  cv::Mat img = cv::imread("data/bus.jpg", cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "[parity] cannot read data/bus.jpg\n";
    return 1;
  }

  std::vector<inference::Detection> dets_pt;
  if (adapter->make_frame_predictor) {
    auto pred = adapter->make_frame_predictor(pt_path, scale, nc, imgsz, "auto");
    dets_pt = pred->predict(img);
  } else {
    // v8 fallback: unified Predictor.
    inference::Predictor p(pt_path, imgsz, /*device=*/"", nc,
                            yolocpp::models::kYolo8n);
    dets_pt = p.predict(img);
  }

  inference::TrtPredictor rt32(trt_fp32, imgsz);
  auto dets_t32 = rt32.predict(img);
  inference::TrtPredictor rt16(trt_fp16, imgsz);
  auto dets_t16 = rt16.predict(img);

  int n_pt  = (int)dets_pt.size();
  int n_t32 = (int)dets_t32.size();
  int n_t16 = (int)dets_t16.size();
  int m32   = match_iou(dets_pt, dets_t32, /*iou=*/0.50f);
  int m16   = match_iou(dets_pt, dets_t16, /*iou=*/0.50f);

  std::cout << "[parity] " << version << "/" << (scale.empty() ? "-" : scale)
            << " pt=" << n_pt
            << " trt-fp32=" << n_t32 << " (matched " << m32 << "/" << n_pt << ")"
            << " trt-fp16=" << n_t16 << " (matched " << m16 << "/" << n_pt << ")\n";

  if (n_pt < 1) {
    std::cerr << "[parity] " << version
              << ": libtorch produced 0 detections — model unusable\n";
    return 1;
  }
  if (std::abs(n_pt - n_t32) > 1 || std::abs(n_pt - n_t16) > 1) {
    std::cerr << "[parity] " << version
              << ": det count drift > ±1 — pt=" << n_pt
              << " trt-fp32=" << n_t32 << " trt-fp16=" << n_t16 << "\n";
    return 1;
  }
  if (m32 < n_pt - 1 || m16 < n_pt - 1) {
    std::cerr << "[parity] " << version
              << ": IoU-0.50 match too low — fp32=" << m32
              << " fp16=" << m16 << " of " << n_pt << "\n";
    return 1;
  }
  return 0;
}

}  // namespace

int main() {
  struct Cell {
    std::string pt;
    std::string version;
    std::string scale;
    int nc;
    int imgsz;
  };
  // Representative subset across the registry. Each cell SKIPs
  // cleanly if the .pt isn't downloaded.
  //
  // **v26 deliberately excluded.** Its DFL-free + NMS-free deploy
  // form emits a different output tensor shape that the standard
  // `inference::TrtPredictor::predict` post-process doesn't unpack
  // the same way as the other versions — running NMS again on
  // already-NMS'd output collapses to a single det. The
  // `test_v26_e2e` ctest + the full-matrix sweep's v26 cells cover
  // its TRT path with v26-aware logic; this generic parity assert
  // doesn't add coverage there.
  std::vector<Cell> cells = {
      {"data/yolo8n.pt",  "v8",  "n", 80, 640},
      {"data/yolo11s.pt", "v11", "s", 80, 640},
      {"data/yolo12s.pt", "v12", "s", 80, 640},
      {"data/yolo13s.pt", "v13", "s", 80, 640},
  };

  int n_pass = 0, n_fail = 0, n_skip = 0;
  for (const auto& c : cells) {
    int rc = run_cell(c.pt, c.version, c.scale, c.nc, c.imgsz);
    if (rc == 0) ++n_pass;
    else if (rc == 2) ++n_skip;
    else ++n_fail;
  }
  std::cout << "[parity] PASS=" << n_pass
            << " FAIL=" << n_fail
            << " SKIP=" << n_skip << "\n";

  if (n_pass == 0 && n_fail == 0) {
    std::cout << "[parity] no weights available — test trivially passes\n";
    return 0;
  }
  return n_fail == 0 ? 0 : 1;
}
