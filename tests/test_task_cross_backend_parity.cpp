// #53C — cross-backend (PyTorch ↔ TensorRT) numerical parity for the TASK
// heads (segment / pose / obb / classify), the task-head counterpart to the
// detect-only test_cross_backend_parity (#53A).
//
// For each (version × task) cell: build the libtorch model, export its ONNX
// directly, build an FP32 TRT engine (TF32 off — faithful to CPU fp32), run
// the SAME input through libtorch `forward_eval` and through the TRT engine's
// raw multi-output binding (`make_trt_multi_forward`), and assert each named
// output tensor (output / coefs / protos / keypoints / angle) matches in shape
// and within a relative-L2 tolerance. A silent decode bug (wrong anchor table,
// cv4 wiring, proto, angle shift) that the structural smoke can't see shows up
// here as a blown relative-L2.
//
// Weights are irrelevant to a cross-backend check — both backends run the
// identical graph on identical weights — so every cell mints from random init
// (no external .pt / data needed) and v12/v13 (no upstream task weights) are
// covered alongside v8/v11.
//
// Gated behind YOLOCPP_TRT_PARITY=1: it builds 16 TRT engines (~minutes), too
// heavy for the default fast suite. Engines cache under build/parity_cache_task/
// so reruns are quick.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "yolocpp/inference/trt_task_eval.hpp"
#include "yolocpp/models/yolo8_tasks.hpp"
#include "yolocpp/models/yolo8_classify.hpp"
#include "yolocpp/models/yolo11_tasks.hpp"
#include "yolocpp/models/yolo12_tasks.hpp"
#include "yolocpp/models/yolo13_tasks.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/trt_export.hpp"

namespace {

namespace fs  = std::filesystem;
namespace ser = yolocpp::serialization;

const std::string kCache = "build/parity_cache_task/";
constexpr int     kSz    = 256;     // small input → cheap engines, all 3 levels
// Relative-L2 ceiling. Observed across all 16 cells: output ~5e-8, coefs ~1e-7,
// protos ~4e-7, keypoints ~8e-9, angle ~1e-7, classify exactly 0 (fp32 CPU vs
// fp32 TF32-off TRT). 1e-3 is 3+ orders of margin yet still catches any real
// decode bug (wrong anchor/cv4/proto/angle → relL2 O(0.1–10)).
constexpr double  kTol   = 1e-3;

// Build an FP32, TF32-off engine (most faithful to libtorch CPU fp32). Cached.
bool build_engine(const std::string& onnx, const std::string& eng) {
  if (fs::exists(eng)) return true;
  ser::TrtBuildConfig c;
  c.imgsz = kSz; c.fp16 = false; c.tf32 = false;
  c.builder_opt_level = 1; c.input_name = "images";
  try {
    ser::build_trt_engine(onnx, eng, c);
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] TRT build " << eng << ": " << e.what() << "\n";
    return false;
  }
  return fs::exists(eng);
}

// Compare one named output: shapes must match, relative-L2 must be < kTol.
bool cmp(const std::string& name, torch::Tensor a, torch::Tensor b) {
  a = a.to(torch::kCPU).to(torch::kFloat).contiguous();
  b = b.to(torch::kCPU).to(torch::kFloat).contiguous();
  if (a.sizes() != b.sizes()) {
    std::cerr << "[FAIL] " << name << " shape " << a.sizes() << " vs "
              << b.sizes() << "\n";
    return false;
  }
  double rel = (a - b).norm().item<double>() /
               (b.norm().item<double>() + 1e-9);
  std::cout << "    " << name << "  relL2=" << rel << "  " << a.sizes() << "\n";
  if (rel > kTol) {
    std::cerr << "[FAIL] " << name << " relL2 " << rel << " > " << kTol << "\n";
    return false;
  }
  return true;
}

torch::Tensor fixed_input() {
  torch::manual_seed(7);
  return torch::randn({1, 3, kSz, kSz});
}

// ─── task runners (templated over holder; export passed as a thunk) ─────────

template <class Seg, class ExportFn>
bool run_seg(const std::string& tag, Seg m, ExportFn exp) {
  std::cout << "[" << tag << "]\n";
  m->eval();
  auto x = fixed_input();
  auto [o, c, p] = m->forward_eval(x);
  auto onnx = kCache + tag + ".onnx", eng = kCache + tag + ".trt";
  exp(onnx);
  if (!build_engine(onnx, eng)) return false;
  int isz = kSz;
  auto t = yolocpp::inference::make_trt_multi_forward(eng, isz)(x);
  bool ok = true;
  ok &= cmp(tag + ".output", o, t.at("output"));
  ok &= cmp(tag + ".coefs",  c, t.at("coefs"));
  ok &= cmp(tag + ".protos", p, t.at("protos"));
  return ok;
}

template <class Pose, class ExportFn>
bool run_pose(const std::string& tag, Pose m, ExportFn exp) {
  std::cout << "[" << tag << "]\n";
  m->eval();
  auto x = fixed_input();
  auto [o, k] = m->forward_eval(x);
  auto onnx = kCache + tag + ".onnx", eng = kCache + tag + ".trt";
  exp(onnx);
  if (!build_engine(onnx, eng)) return false;
  int isz = kSz;
  auto t = yolocpp::inference::make_trt_multi_forward(eng, isz)(x);
  bool ok = true;
  ok &= cmp(tag + ".output",    o, t.at("output"));
  ok &= cmp(tag + ".keypoints", k, t.at("keypoints"));
  return ok;
}

template <class OBB, class ExportFn>
bool run_obb(const std::string& tag, OBB m, ExportFn exp) {
  std::cout << "[" << tag << "]\n";
  m->eval();
  auto x = fixed_input();
  auto [o, a] = m->forward_eval(x);
  auto onnx = kCache + tag + ".onnx", eng = kCache + tag + ".trt";
  exp(onnx);
  if (!build_engine(onnx, eng)) return false;
  int isz = kSz;
  auto t = yolocpp::inference::make_trt_multi_forward(eng, isz)(x);
  bool ok = true;
  ok &= cmp(tag + ".output", o, t.at("output"));
  ok &= cmp(tag + ".angle",  a, t.at("angle"));
  return ok;
}

template <class Cls, class ExportFn>
bool run_cls(const std::string& tag, Cls m, ExportFn exp) {
  std::cout << "[" << tag << "]\n";
  m->eval();
  auto x = fixed_input();
  auto o = m->forward(x);
  auto onnx = kCache + tag + ".onnx", eng = kCache + tag + ".trt";
  exp(onnx);
  if (!build_engine(onnx, eng)) return false;
  int isz = kSz;
  auto t = yolocpp::inference::make_trt_multi_forward(eng, isz)(x);
  return cmp(tag + ".output", o, t.at("output"));
}

}  // namespace

int main() {
  if (!std::getenv("YOLOCPP_TRT_PARITY")) {
    std::cout << "SKIP: set YOLOCPP_TRT_PARITY=1 to run "
                 "(builds 16 task TRT engines)\n";
    return 0;
  }
  torch::NoGradGuard ng;
  fs::create_directories(kCache);
  ser::OnnxExportConfig cfg;
  cfg.imgsz = kSz;

  using namespace yolocpp::models;
  auto s8  = kYolo8n;
  auto s11 = yolo11_scale_from_letter("n");
  auto s12 = yolo12_scale_from_letter("n");
  auto s13 = yolo13_scale_from_letter("n");

  bool ok = true;
  // Seed before EACH construction so a cell's random weights depend only on its
  // own seed — reproducible across process runs (the on-disk engine cache stays
  // valid) and independent of cell order. (Without this, the first-constructed
  // cell drew from the non-deterministic initial RNG state, so its cached engine
  // went stale on the next run → false mismatch.)
  auto seed = [](int s) { torch::manual_seed(s); };

  // ─── segment ──────────────────────────────────────────────────────────
  seed(101); { Yolo8Segment  m(s8,  80); ok &= run_seg("v8-seg",  m,
      [&](const std::string& p){ ser::export_yolo8_segment_onnx(m, p, cfg); }); }
  seed(102); { Yolo11Segment m(s11, 80); ok &= run_seg("v11-seg", m,
      [&](const std::string& p){ ser::export_yolo11_segment_onnx(m, p, cfg); }); }
  seed(103); { Yolo12Segment m(s12, 80); ok &= run_seg("v12-seg", m,
      [&](const std::string& p){ ser::export_yolo12_segment_onnx(m, p, cfg); }); }
  seed(104); { Yolo13Segment m(s13, 80); ok &= run_seg("v13-seg", m,
      [&](const std::string& p){ ser::export_yolo13_segment_onnx(m, p, cfg); }); }

  // ─── pose ─────────────────────────────────────────────────────────────
  seed(105); { Yolo8Pose  m(s8,  1, 17, 3); ok &= run_pose("v8-pose",  m,
      [&](const std::string& p){ ser::export_yolo8_pose_onnx(m, p, cfg); }); }
  seed(106); { Yolo11Pose m(s11, 1, 17, 3); ok &= run_pose("v11-pose", m,
      [&](const std::string& p){ ser::export_yolo11_pose_onnx(m, p, cfg); }); }
  seed(107); { Yolo12Pose m(s12, 1, 17, 3); ok &= run_pose("v12-pose", m,
      [&](const std::string& p){ ser::export_yolo12_pose_onnx(m, p, cfg); }); }
  seed(108); { Yolo13Pose m(s13, 1, 17, 3); ok &= run_pose("v13-pose", m,
      [&](const std::string& p){ ser::export_yolo13_pose_onnx(m, p, cfg); }); }

  // ─── obb ──────────────────────────────────────────────────────────────
  seed(109); { Yolo8OBB  m(s8,  15, 1); ok &= run_obb("v8-obb",  m,
      [&](const std::string& p){ ser::export_yolo8_obb_onnx(m, p, cfg); }); }
  seed(110); { Yolo11OBB m(s11, 15, 1); ok &= run_obb("v11-obb", m,
      [&](const std::string& p){ ser::export_yolo11_obb_onnx(m, p, cfg); }); }
  seed(111); { Yolo12OBB m(s12, 15, 1); ok &= run_obb("v12-obb", m,
      [&](const std::string& p){ ser::export_yolo12_obb_onnx(m, p, cfg); }); }
  seed(112); { Yolo13OBB m(s13, 15, 1); ok &= run_obb("v13-obb", m,
      [&](const std::string& p){ ser::export_yolo13_obb_onnx(m, p, cfg); }); }

  // ─── classify ─────────────────────────────────────────────────────────
  seed(113); { Yolo8Classify  m(s8,  1000); ok &= run_cls("v8-cls",  m,
      [&](const std::string& p){ ser::export_yolo8_classify_onnx(m, p, cfg); }); }
  seed(114); { Yolo11Classify m(s11, 1000); ok &= run_cls("v11-cls", m,
      [&](const std::string& p){ ser::export_yolo11_classify_onnx(m, p, cfg); }); }
  seed(115); { Yolo12Classify m(s12, 1000); ok &= run_cls("v12-cls", m,
      [&](const std::string& p){ ser::export_yolo12_classify_onnx(m, p, cfg); }); }
  seed(116); { Yolo13Classify m(s13, 1000); ok &= run_cls("v13-cls", m,
      [&](const std::string& p){ ser::export_yolo13_classify_onnx(m, p, cfg); }); }

  if (!ok) { std::cerr << "=== task cross-backend parity FAIL ===\n"; return 1; }
  std::cout << "=== task cross-backend parity PASS (16 cells) ===\n";
  return 0;
}
