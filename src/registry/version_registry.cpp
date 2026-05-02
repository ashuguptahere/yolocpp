// Per-version registry — see include/yolocpp/registry/version_adapter.hpp
// for the contract and the "how to add a new YOLO version" walkthrough.
//
// This TU owns:
//   1. The Registry singleton (idempotent register/find).
//   2. Per-version registration helpers (`register_yolo<N>`).
//   3. `register_all_versions()` — the one entry point CLI / API call.
//
// First migration target: ONNX export (clearest concentration of
// per-version dispatch, ~250 lines in cmd_export). Predict / val /
// train will follow in subsequent tasks (#46D, #46E, #46F).

#include "yolocpp/registry/version_adapter.hpp"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include <torch/torch.h>

#include "yolocpp/models/yolo3.hpp"
#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/models/yolo5.hpp"
#include "yolocpp/models/yolo6.hpp"
#include "yolocpp/models/yolo7.hpp"
#include "yolocpp/models/yolo8.hpp"
#include "yolocpp/models/yolo8_classify.hpp"
#include "yolocpp/models/yolo8_tasks.hpp"
#include "yolocpp/models/yolo9.hpp"
#include "yolocpp/models/yolo10.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo11_tasks.hpp"
#include "yolocpp/models/yolo12.hpp"
#include "yolocpp/models/yolo13.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo26_tasks.hpp"
#include "yolocpp/models/rfdetr.hpp"
#include "yolocpp/inference/rfdetr_predictor.hpp"

#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/engine/benchmark.hpp"
#include "yolocpp/engine/benchmark_internal.hpp"
#include "yolocpp/engine/trainer.hpp"
#include "yolocpp/engine/validator.hpp"
#include "yolocpp/inference/frame_predictor.hpp"
#include "yolocpp/inference/nms.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/inference/task_predictors.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

namespace yolocpp::registry {

Registry& Registry::instance() {
  static Registry r;
  return r;
}

void Registry::register_version(VersionAdapter a) {
  versions_[a.version_id] = std::move(a);
}

const VersionAdapter* Registry::find(const std::string& version_id) const {
  auto it = versions_.find(version_id);
  return it == versions_.end() ? nullptr : &it->second;
}

std::vector<std::string> Registry::known_ids() const {
  std::vector<std::string> ids;
  ids.reserve(versions_.size());
  for (const auto& kv : versions_) ids.push_back(kv.first);
  std::sort(ids.begin(), ids.end());
  return ids;
}

namespace {

// Helpers — keep the per-version blocks below readable.

void load_and_eval(auto& m, const std::string& weights, int imgsz,
                   bool needs_warmup) {
  auto sd = serialization::load_state_dict(weights);
  m->load_from_state_dict(sd.entries);
  if (needs_warmup) {
    torch::NoGradGuard ng;
    auto x = torch::zeros({1, 3, imgsz, imgsz});
    m->forward_eval(x);  // populates strides for v3/v7/v9/v10
  }
  m->eval();
}

// v6 has the most complex scale/variant matrix — pull it out so the
// adapter lambda stays tight.
struct V6Resolved {
  models::Yolo6Variant variant_kind = models::Yolo6Variant::Standard;
  models::Yolo6Scale scale = models::kYolo6s;
  bool p6 = false;
};

V6Resolved resolve_v6(const std::string& s) {
  V6Resolved r;
  if      (s == "n")        r.scale = models::kYolo6n;
  else if (s == "s" || s.empty()) r.scale = models::kYolo6s;
  else if (s == "m")        r.scale = models::kYolo6m;
  else if (s == "l")        r.scale = models::kYolo6l;
  else if (s == "n6")       { r.scale = models::kYolo6n; r.p6 = true; }
  else if (s == "s6")       { r.scale = models::kYolo6s; r.p6 = true; }
  else if (s == "m6")       { r.scale = models::kYolo6m; r.p6 = true; }
  else if (s == "l6")       { r.scale = models::kYolo6l; r.p6 = true; }
  else if (s == "s_mbla")   r.scale = models::kYolo6s_mbla;
  else if (s == "m_mbla")   r.scale = models::kYolo6m_mbla;
  else if (s == "l_mbla")   r.scale = models::kYolo6l_mbla;
  else if (s == "x_mbla")   r.scale = models::kYolo6x_mbla;
  return r;
}

// v8's scale-letter helper lives in cli/main.cpp; inline a copy here so
// the registry doesn't depend on CLI internals. Mirrors the v8.hpp
// constexpr scale enum (kYolo8n / s / m / l / x).
models::Yolo8Scale v8_scale_from_letter(const std::string& s) {
  if (s == "n") return models::kYolo8n;
  if (s == "s" || s.empty()) return models::kYolo8s;
  if (s == "m") return models::kYolo8m;
  if (s == "l") return models::kYolo8l;
  if (s == "x") return models::kYolo8x;
  return models::kYolo8n;
}

// Shared val driver: loads weights, moves to device, runs the
// templated `engine::validate<M>` and unwraps the mAPResult into the
// adapter's ValResult. Every version's `run_val` lambda calls this
// after constructing its concrete holder.
template <typename Holder>
VersionAdapter::ValResult run_val_with(Holder& m,
                                       const std::string& weights,
                                       datasets::YoloDataset& ds,
                                       const torch::Device& device) {
  auto sd = serialization::load_state_dict(weights);
  m->load_from_state_dict(sd.entries);
  m->to(device);
  m->eval();
  auto r = engine::validate(m, ds, device);
  return VersionAdapter::ValResult{
      r.map_50, r.map_50_95,
      r.map_50_95_small, r.map_50_95_medium, r.map_50_95_large,
      r.n_gt_small, r.n_gt_medium, r.n_gt_large};
}

// Shared detect-train driver: optionally load init weights, construct
// the matching `TrainerT<Holder>`, run. Every version's
// `run_train_detect` lambda calls this after building its holder.
template <typename Holder>
void run_train_with(Holder& m,
                    const std::string& init_weights,
                    datasets::YoloDataset train_ds,
                    const engine::TrainConfig& cfg) {
  if (!init_weights.empty()) {
    auto sd = serialization::load_state_dict(init_weights);
    int copied = m->load_from_state_dict(sd.entries);
    std::cout << "[train] loaded " << copied << " weights from "
              << init_weights << "\n";
  }
  engine::TrainerT<Holder> trainer(m, std::move(train_ds), cfg);
  trainer.run();
}

// Shared PT-FP32 benchmark driver: load weights into the supplied
// holder, wrap in GenericPredictor, time it via bench_one().
template <typename Holder>
engine::BenchResult run_bench_pt_with(Holder& m,
                                      const engine::BenchConfig& cfg,
                                      const cv::Mat& img) {
  auto sd = serialization::load_state_dict(cfg.weights);
  m->load_from_state_dict(sd.entries);
  auto dev = engine::detail::pick_device(cfg.device);
  engine::detail::GenericPredictor<Holder> p(std::move(m), cfg.imgsz, dev);
  return engine::detail::bench_one("PT (libtorch FP32)", img, p,
                                    cfg.warmup_iters, cfg.iters);
}

// Long-lived frame predictor wrapping any holder. Reuses the
// `engine::detail::GenericPredictor<Holder>` pipeline (letterbox →
// forward_eval → NMS → scale_boxes) but holds the model + device
// across many `predict(frame)` calls.
template <typename Holder>
class GenericFramePredictor : public inference::FramePredictor {
 public:
  GenericFramePredictor(Holder m, int imgsz, torch::Device dev)
      : p_(std::move(m), imgsz, dev) {}
  std::vector<inference::Detection>
  predict(const cv::Mat& frame, inference::NMSConfig nm) override {
    return p_.predict(frame, nm);
  }
 private:
  engine::detail::GenericPredictor<Holder> p_;
};

// Build a `GenericFramePredictor<Holder>` with weights pre-loaded.
// Each version's `make_frame_predictor` lambda calls this after
// constructing its concrete holder.
template <typename Holder>
std::unique_ptr<inference::FramePredictor>
make_frame_pred_with(Holder& m, const std::string& weights, int imgsz,
                     const std::string& device) {
  auto sd = serialization::load_state_dict(weights);
  m->load_from_state_dict(sd.entries);
  auto dev = engine::detail::pick_device(device);
  return std::make_unique<GenericFramePredictor<Holder>>(std::move(m),
                                                          imgsz, dev);
}

// Default imgsz lookups shared by multiple versions.
int detect_imgsz_default(const std::string& /*scale*/,
                         const std::string& task) {
  if (task == "classify") return 224;
  return 0;  // 0 ⇒ caller default (640)
}

// ---- v3 ------------------------------------------------------------------
VersionAdapter make_v3() {
  VersionAdapter a;
  a.version_id = "v3";
  a.display_name = "yolo3";
  a.upstream_year = "2018";
  a.default_export_basename = "yolo3";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string&,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo3 export: only 'detect' supported");
    models::Yolo3 m(models::kYolo3, nc);
    load_and_eval(m, weights, cfg.imgsz, /*needs_warmup=*/true);
    serialization::export_yolo3_onnx(m, path, cfg);
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string&,
                         int nc, const inference::NMSConfig& nm) {
    return inference::predict_v3_to_file(weights, src, out, imgsz, device,
                                         nc, nm);
  };
  a.run_val = [](const std::string& weights, const std::string&,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    models::Yolo3 m(models::kYolo3, nc);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string&,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    models::Yolo3 m(models::kYolo3, nc);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string&) {
    models::Yolo3 m(models::kYolo3, cfg.nc);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string&,
                              int nc, int imgsz, const std::string& device) {
    models::Yolo3 m(models::kYolo3, nc);
    return make_frame_pred_with(m, weights, imgsz, device);
  };
  return a;
}

// ---- v4 ------------------------------------------------------------------
VersionAdapter make_v4() {
  VersionAdapter a;
  a.version_id = "v4";
  a.display_name = "yolo4";
  a.upstream_year = "2020";
  a.default_export_basename = "yolo4";
  a.supported_tasks = {"detect"};
  a.default_imgsz = [](const std::string&, const std::string& task) {
    if (task == "classify") return 224;
    return 608;  // v4 anchors are calibrated to 608²
  };
  a.export_onnx = [](const std::string& weights, const std::string&,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo4 export: only 'detect' supported");
    models::Yolo4 m(nc);
    auto sd = serialization::load_state_dict(weights);
    m->load_from_state_dict(sd.entries);
    m->eval();
    serialization::export_yolo4_onnx(m, path, cfg);
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string&,
                         int nc, const inference::NMSConfig& nm) {
    int v4_imgsz = (imgsz == 640) ? 608 : imgsz;  // v4 anchor calibration
    return inference::predict_v4_to_file(weights, src, out, v4_imgsz,
                                         device, nc, nm);
  };
  a.run_val = [](const std::string& weights, const std::string&,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    models::Yolo4 m(nc);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string&,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    models::Yolo4 m(nc);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string&) {
    models::Yolo4 m(cfg.nc);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string&,
                              int nc, int imgsz, const std::string& device) {
    models::Yolo4 m(nc);
    return make_frame_pred_with(m, weights, imgsz, device);
  };
  return a;
}

// ---- v5 ------------------------------------------------------------------
VersionAdapter make_v5() {
  VersionAdapter a;
  a.version_id = "v5";
  a.display_name = "yolo5";
  a.upstream_year = "2020";
  a.default_export_basename = "yolo5";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo5 export: only 'detect' supported");
    models::Yolo5Detect m(models::yolo5_scale_from_letter(scale), nc);
    auto sd = serialization::load_state_dict(weights);
    m->load_from_state_dict(sd.entries);
    m->eval();
    serialization::export_yolo5_onnx(m, path, cfg);
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string& scale,
                         int nc, const inference::NMSConfig& nm) {
    return inference::predict_v5_to_file(weights, src, out, imgsz, device,
                                         nc, models::yolo5_scale_from_letter(scale),
                                         nm);
  };
  a.run_val = [](const std::string& weights, const std::string& scale,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    models::Yolo5Detect m(models::yolo5_scale_from_letter(scale), nc);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string& scale,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    models::Yolo5Detect m(models::yolo5_scale_from_letter(scale), nc);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string& scale) {
    models::Yolo5Detect m(models::yolo5_scale_from_letter(scale), cfg.nc);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string& scale,
                              int nc, int imgsz, const std::string& device) {
    models::Yolo5Detect m(models::yolo5_scale_from_letter(scale), nc);
    return make_frame_pred_with(m, weights, imgsz, device);
  };
  return a;
}

// ---- v6 ------------------------------------------------------------------
VersionAdapter make_v6() {
  VersionAdapter a;
  a.version_id = "v6";
  a.display_name = "yolo6";
  a.upstream_year = "2022";
  a.default_export_basename = "yolo6";
  a.supported_tasks = {"detect"};
  a.default_imgsz = [](const std::string& scale, const std::string& task) {
    if (task == "classify") return 224;
    if (scale == "n6" || scale == "s6" || scale == "m6" || scale == "l6")
      return 1280;  // P6 variants
    return 0;
  };
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo6 export: only 'detect' supported");
    auto r = resolve_v6(scale);
    models::Yolo6 m(nc, r.scale, /*reg_max=*/16, /*p6=*/r.p6);
    auto sd = serialization::load_state_dict(weights);
    m->load_from_state_dict(sd.entries);
    m->eval();
    serialization::export_yolo6_onnx(m, path, cfg);
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string& scale,
                         int nc, const inference::NMSConfig& nm) {
    auto r = resolve_v6(scale);
    int v6_imgsz = (r.p6 && imgsz == 640) ? 1280 : imgsz;
    return inference::predict_v6_to_file(weights, src, out, v6_imgsz, device,
                                         nc, r.scale, r.p6, nm);
  };
  a.run_val = [](const std::string& weights, const std::string& scale,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    auto r = resolve_v6(scale);
    models::Yolo6 m(nc, r.scale, /*reg_max=*/16, /*p6=*/r.p6);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string& scale,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    auto r = resolve_v6(scale);
    models::Yolo6 m(nc, r.scale, /*reg_max=*/16, /*p6=*/r.p6);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string& scale) {
    auto r = resolve_v6(scale);
    models::Yolo6 m(cfg.nc, r.scale, /*reg_max=*/16, /*p6=*/r.p6);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string& scale,
                              int nc, int imgsz, const std::string& device) {
    auto r = resolve_v6(scale);
    int v6_imgsz = (r.p6 && imgsz == 640) ? 1280 : imgsz;
    models::Yolo6 m(nc, r.scale, /*reg_max=*/16, /*p6=*/r.p6);
    return make_frame_pred_with(m, weights, v6_imgsz, device);
  };
  return a;
}

// ---- v7 ------------------------------------------------------------------
VersionAdapter make_v7() {
  VersionAdapter a;
  a.version_id = "v7";
  a.display_name = "yolo7";
  a.upstream_year = "2022";
  a.default_export_basename = "yolo7";
  a.supported_tasks = {"detect"};
  a.default_imgsz = [](const std::string& scale, const std::string& task) {
    if (task == "classify") return 224;
    if (scale == "w6" || scale == "e6" || scale == "d6" || scale == "e6e")
      return 1280;  // P6 variants
    return 0;
  };
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo7 export: only 'detect' supported");
    models::Yolo7 m(models::yolo7_scale_from_letter(scale), nc);
    load_and_eval(m, weights, cfg.imgsz, /*needs_warmup=*/true);
    serialization::export_yolo7_onnx(m, path, cfg);
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string& scale,
                         int nc, const inference::NMSConfig& nm) {
    return inference::predict_v7_to_file(weights, src, out, imgsz, device,
                                         nc, models::yolo7_scale_from_letter(scale),
                                         nm);
  };
  a.run_val = [](const std::string& weights, const std::string& scale,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    models::Yolo7 m(models::yolo7_scale_from_letter(scale), nc);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string& scale,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    models::Yolo7 m(models::yolo7_scale_from_letter(scale), nc);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string& scale) {
    models::Yolo7 m(models::yolo7_scale_from_letter(scale), cfg.nc);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string& scale,
                              int nc, int imgsz, const std::string& device) {
    models::Yolo7 m(models::yolo7_scale_from_letter(scale), nc);
    return make_frame_pred_with(m, weights, imgsz, device);
  };
  return a;
}

// ---- v8 (full task family) ----------------------------------------------
VersionAdapter make_v8() {
  VersionAdapter a;
  a.version_id = "v8";
  a.display_name = "yolo8";
  a.upstream_year = "2023";
  a.default_export_basename = "yolo8";
  a.supported_tasks = {"detect", "classify", "segment", "pose", "obb"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg_in) {
    auto cfg = cfg_in;
    auto sc = v8_scale_from_letter(scale);
    auto sd = serialization::load_state_dict(weights);
    if (task == "detect") {
      models::Yolo8Detect m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo8_onnx(m, path, cfg);
    } else if (task == "classify") {
      if (cfg.imgsz == 640) cfg.imgsz = 224;
      int cls_nc = (nc < 0 || nc == 80) ? 1000 : nc;
      models::Yolo8Classify m(sc, cls_nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo8_classify_onnx(m, path, cfg);
    } else if (task == "segment") {
      models::Yolo8Segment m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo8_segment_onnx(m, path, cfg);
    } else if (task == "pose") {
      models::Yolo8Pose m(sc, /*nc=*/1, /*num_kpts=*/17, /*kpt_dim=*/3);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo8_pose_onnx(m, path, cfg);
    } else if (task == "obb") {
      int obb_nc = (nc == 80) ? 15 : nc;
      models::Yolo8OBB m(sc, obb_nc, /*ne=*/1);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo8_obb_onnx(m, path, cfg);
    } else {
      throw std::runtime_error("yolo8 export: unknown task '" + task + "'");
    }
  };
  return a;
}

// ---- v9 ------------------------------------------------------------------
VersionAdapter make_v9() {
  VersionAdapter a;
  a.version_id = "v9";
  a.display_name = "yolo9";
  a.upstream_year = "2024";
  a.default_export_basename = "yolo9";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo9 export: only 'detect' supported");
    models::Yolo9 m(models::yolo9_scale_from_letter(scale), nc);
    load_and_eval(m, weights, cfg.imgsz, /*needs_warmup=*/true);
    serialization::export_yolo9_onnx(m, path, cfg);
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string& scale,
                         int nc, const inference::NMSConfig& nm) {
    return inference::predict_v9_to_file(weights, src, out, imgsz, device,
                                         nc, models::yolo9_scale_from_letter(scale),
                                         nm);
  };
  a.run_val = [](const std::string& weights, const std::string& scale,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    models::Yolo9 m(models::yolo9_scale_from_letter(scale), nc);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string& scale,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    models::Yolo9 m(models::yolo9_scale_from_letter(scale), nc);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string& scale) {
    models::Yolo9 m(models::yolo9_scale_from_letter(scale), cfg.nc);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string& scale,
                              int nc, int imgsz, const std::string& device) {
    models::Yolo9 m(models::yolo9_scale_from_letter(scale), nc);
    return make_frame_pred_with(m, weights, imgsz, device);
  };
  return a;
}

// ---- v10 -----------------------------------------------------------------
VersionAdapter make_v10() {
  VersionAdapter a;
  a.version_id = "v10";
  a.display_name = "yolo10";
  a.upstream_year = "2024";
  a.default_export_basename = "yolo10";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.trt_disable_tf32 = true;  // RepVGGDW saturation across all scales
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error("yolo10 export: only 'detect' supported");
    models::Yolo10 m(models::yolo10_scale_from_letter(scale), nc);
    load_and_eval(m, weights, cfg.imgsz, /*needs_warmup=*/true);
    serialization::export_yolo10_onnx(m, path, cfg);
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string& scale,
                         int nc, const inference::NMSConfig& nm) {
    return inference::predict_v10_to_file(weights, src, out, imgsz, device,
                                          nc, models::yolo10_scale_from_letter(scale),
                                          nm);
  };
  a.run_val = [](const std::string& weights, const std::string& scale,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    models::Yolo10 m(models::yolo10_scale_from_letter(scale), nc);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string& scale,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    models::Yolo10 m(models::yolo10_scale_from_letter(scale), nc);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string& scale) {
    models::Yolo10 m(models::yolo10_scale_from_letter(scale), cfg.nc);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string& scale,
                              int nc, int imgsz, const std::string& device) {
    models::Yolo10 m(models::yolo10_scale_from_letter(scale), nc);
    return make_frame_pred_with(m, weights, imgsz, device);
  };
  return a;
}

// ---- v11 (full task family) ---------------------------------------------
VersionAdapter make_v11() {
  VersionAdapter a;
  a.version_id = "v11";
  a.display_name = "yolo11";
  a.upstream_year = "2024";
  a.default_export_basename = "yolo11";
  a.supported_tasks = {"detect", "classify", "segment", "pose", "obb"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg_in) {
    auto cfg = cfg_in;
    auto sc = models::yolo11_scale_from_letter(scale);
    auto sd = serialization::load_state_dict(weights);
    if (task == "detect") {
      models::Yolo11Detect m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo11_onnx(m, path, cfg);
    } else if (task == "classify") {
      if (cfg.imgsz == 640) cfg.imgsz = 224;
      int cls_nc = (nc < 0 || nc == 80) ? 1000 : nc;
      models::Yolo11Classify m(sc, cls_nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo11_classify_onnx(m, path, cfg);
    } else if (task == "segment") {
      models::Yolo11Segment m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo11_segment_onnx(m, path, cfg);
    } else if (task == "pose") {
      models::Yolo11Pose m(sc, /*nc=*/1, /*num_kpts=*/17, /*kpt_dim=*/3);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo11_pose_onnx(m, path, cfg);
    } else if (task == "obb") {
      int obb_nc = (nc == 80) ? 15 : nc;
      models::Yolo11OBB m(sc, obb_nc, /*ne=*/1);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo11_obb_onnx(m, path, cfg);
    } else {
      throw std::runtime_error("yolo11 export: unknown task '" + task + "'");
    }
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string& scale,
                         int nc, const inference::NMSConfig& nm) {
    return inference::predict_v11_to_file(weights, src, out, imgsz, device,
                                          nc, models::yolo11_scale_from_letter(scale),
                                          nm);
  };
  a.run_val = [](const std::string& weights, const std::string& scale,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    models::Yolo11Detect m(models::yolo11_scale_from_letter(scale), nc);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string& scale,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    models::Yolo11Detect m(models::yolo11_scale_from_letter(scale), nc);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string& scale) {
    models::Yolo11Detect m(models::yolo11_scale_from_letter(scale), cfg.nc);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string& scale,
                              int nc, int imgsz, const std::string& device) {
    models::Yolo11Detect m(models::yolo11_scale_from_letter(scale), nc);
    return make_frame_pred_with(m, weights, imgsz, device);
  };
  return a;
}

// ---- v12 -----------------------------------------------------------------
VersionAdapter make_v12() {
  VersionAdapter a;
  a.version_id = "v12";
  a.display_name = "yolo12";
  a.upstream_year = "2025";
  a.default_export_basename = "yolo12";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error(
          "yolo12 export: only 'detect' supported (task heads scaffolded; "
          "weights pending — see TODO #60)");
    models::Yolo12Detect m(models::yolo12_scale_from_letter(scale), nc);
    auto sd = serialization::load_state_dict(weights);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo12_onnx(m, path, cfg);
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string& scale,
                         int nc, const inference::NMSConfig& nm) {
    return inference::predict_v12_to_file(weights, src, out, imgsz, device,
                                          nc, models::yolo12_scale_from_letter(scale),
                                          nm);
  };
  a.run_val = [](const std::string& weights, const std::string& scale,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    models::Yolo12Detect m(models::yolo12_scale_from_letter(scale), nc);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string& scale,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    models::Yolo12Detect m(models::yolo12_scale_from_letter(scale), nc);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string& scale) {
    models::Yolo12Detect m(models::yolo12_scale_from_letter(scale), cfg.nc);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string& scale,
                              int nc, int imgsz, const std::string& device) {
    models::Yolo12Detect m(models::yolo12_scale_from_letter(scale), nc);
    return make_frame_pred_with(m, weights, imgsz, device);
  };
  return a;
}

// ---- v13 -----------------------------------------------------------------
VersionAdapter make_v13() {
  VersionAdapter a;
  a.version_id = "v13";
  a.display_name = "yolo13";
  a.upstream_year = "2025";
  a.default_export_basename = "yolo13";
  a.supported_tasks = {"detect"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg) {
    if (task != "detect")
      throw std::runtime_error(
          "yolo13 export: only 'detect' supported (no `m` scale upstream)");
    models::Yolo13Detect m(models::yolo13_scale_from_letter(scale), nc);
    auto sd = serialization::load_state_dict(weights);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo13_onnx(m, path, cfg);
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string& scale,
                         int nc, const inference::NMSConfig& nm) {
    return inference::predict_v13_to_file(weights, src, out, imgsz, device,
                                          nc, models::yolo13_scale_from_letter(scale),
                                          nm);
  };
  a.run_val = [](const std::string& weights, const std::string& scale,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    models::Yolo13Detect m(models::yolo13_scale_from_letter(scale), nc);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string& scale,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    models::Yolo13Detect m(models::yolo13_scale_from_letter(scale), nc);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string& scale) {
    models::Yolo13Detect m(models::yolo13_scale_from_letter(scale), cfg.nc);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string& scale,
                              int nc, int imgsz, const std::string& device) {
    models::Yolo13Detect m(models::yolo13_scale_from_letter(scale), nc);
    return make_frame_pred_with(m, weights, imgsz, device);
  };
  return a;
}

// ---- v26 (full task family) ---------------------------------------------
VersionAdapter make_v26() {
  VersionAdapter a;
  a.version_id = "v26";
  a.display_name = "yolo26";
  a.upstream_year = "2025";
  a.default_export_basename = "yolo26";
  a.supported_tasks = {"detect", "classify", "segment", "pose", "obb"};
  a.default_imgsz = detect_imgsz_default;
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                     int nc, const std::string& task,
                     const std::string& path,
                     const serialization::OnnxExportConfig& cfg_in) {
    auto cfg = cfg_in;
    auto sc = models::yolo26_scale_from_letter(scale);
    auto sd = serialization::load_state_dict(weights);
    if (task == "detect") {
      models::Yolo26Detect m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo26_onnx(m, path, cfg);
    } else if (task == "classify") {
      if (cfg.imgsz == 640) cfg.imgsz = 224;
      int cls_nc = (nc < 0 || nc == 80) ? 1000 : nc;
      models::Yolo26Classify m(sc, cls_nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo26_classify_onnx(m, path, cfg);
    } else if (task == "segment") {
      models::Yolo26Segment m(sc, nc);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo26_segment_onnx(m, path, cfg);
    } else if (task == "pose") {
      models::Yolo26Pose m(sc, /*nc=*/1, /*num_kpts=*/17, /*kpt_dim=*/3);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo26_pose_onnx(m, path, cfg);
    } else if (task == "obb") {
      int obb_nc = (nc == 80) ? 15 : nc;
      models::Yolo26OBB m(sc, obb_nc, /*ne=*/1);
      m->load_from_state_dict(sd.entries); m->eval();
      serialization::export_yolo26_obb_onnx(m, path, cfg);
    } else {
      throw std::runtime_error("yolo26 export: unknown task '" + task + "'");
    }
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                         const std::string& out, int imgsz,
                         const std::string& device, const std::string& scale,
                         int nc, const inference::NMSConfig& nm) {
    return inference::predict_v26_to_file(weights, src, out, imgsz, device,
                                          nc, models::yolo26_scale_from_letter(scale),
                                          nm);
  };
  a.run_val = [](const std::string& weights, const std::string& scale,
                 int nc, datasets::YoloDataset& ds,
                 const torch::Device& device) {
    models::Yolo26Detect m(models::yolo26_scale_from_letter(scale), nc);
    return run_val_with(m, weights, ds, device);
  };
  a.run_train_detect = [](const std::string& init, const std::string& scale,
                          int nc, datasets::YoloDataset ds,
                          const engine::TrainConfig& cfg) {
    models::Yolo26Detect m(models::yolo26_scale_from_letter(scale), nc);
    run_train_with(m, init, std::move(ds), cfg);
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat& img,
                      const std::string& scale) {
    models::Yolo26Detect m(models::yolo26_scale_from_letter(scale), cfg.nc);
    return run_bench_pt_with(m, cfg, img);
  };
  a.make_frame_predictor = [](const std::string& weights, const std::string& scale,
                              int nc, int imgsz, const std::string& device) {
    models::Yolo26Detect m(models::yolo26_scale_from_letter(scale), nc);
    return make_frame_pred_with(m, weights, imgsz, device);
  };
  return a;
}

// ─── RF-DETR (#65) ───────────────────────────────────────────────────────
// Scaffold-only adapter. Every hook routes to the throw-stubs in
// `models::RFDetr` which surface "not yet implemented — see #65X"
// rather than silently producing wrong output. Once #65A..#65L
// land slice by slice, each hook gets a real implementation; the
// registry surface itself doesn't need to change.
VersionAdapter make_rfdetr() {
  VersionAdapter a;
  a.version_id              = "rfdetr";
  a.display_name            = "rfdetr";
  a.upstream_year           = "2025";
  a.default_export_basename = "rfdetr";
  a.supported_tasks         = {"detect", "segment"};
  a.default_imgsz = [](const std::string& scale, const std::string& task) {
    if (task == "classify") return 224;  // not supported, but keep symmetry
    auto s = models::rfdetr_scale_from_letter(scale);
    return models::rfdetr_default_imgsz(s);
  };
  // Every hook below constructs an RFDetr / RFDetrSegment holder and
  // calls into a method that throws. Caller gets a clear error
  // pointing at the slice that owns the missing piece. This is the
  // honest version of "scaffolded" — the dispatch surface compiles
  // and links, but no path silently mis-loads weights or returns
  // garbage detections.
  a.export_onnx = [](const std::string& weights, const std::string& scale,
                      int nc, const std::string& task,
                      const std::string& path,
                      const serialization::OnnxExportConfig& cfg) {
    (void)weights; (void)nc; (void)path; (void)cfg;
    if (task == "segment") {
      models::RFDetrSegment m(models::rfdetr_scale_from_letter(scale), nc);
      m->load_from_state_dict({});  // throws
    } else {
      models::RFDetr m(models::rfdetr_scale_from_letter(scale), nc);
      m->load_from_state_dict({});  // throws
    }
    throw std::runtime_error("rfdetr export_onnx: scaffolded only — see #65I");
  };
  a.predict_to_file = [](const std::string& weights, const std::string& src,
                          const std::string& out_path, int imgsz,
                          const std::string& device,
                          const std::string& scale, int nc,
                          const inference::NMSConfig& nm) {
    auto rfscale = models::rfdetr_scale_from_letter(scale);
    models::RFDetr m(rfscale, nc);
    if (!weights.empty()) {
      m->load_from_upstream_pt(weights, /*strict=*/false);
    }
    auto dev = torch::Device(torch::kCPU);
    if (device == "cuda" && torch::cuda::is_available())
        dev = torch::Device(torch::kCUDA);
    m->to(dev);
    m->eval();

    auto bgr = cv::imread(src);
    if (bgr.empty())
        throw std::runtime_error("predict_to_file: cannot read source: " + src);
    // RF-DETR per-variant resolution. Each of the 12 variants has
    // its own pretrained input size baked into `RFDetrScale.resolution`
    // (n=384, s=512, m=576, b=560, l=704, seg-n=368, seg-s=512,
    // seg-m=576, seg-l=672, seg-xl=624, seg-xxl=768, seg-prv=432).
    // The windowed-attention embeddings layer requires the input
    // dims to be divisible by `patch_size × num_windows`. If the
    // caller supplies `--imgsz N`, honour it iff it satisfies that
    // constraint; otherwise fall back to the variant default.
    int default_side = models::rfdetr_default_imgsz(rfscale);
    int side = default_side;
    if (imgsz > 0 && imgsz != default_side) {
      auto& bcfg = yolocpp::models::rfdetr::dinov2_cfg_for(
          rfscale.upstream_id, rfscale.patch_size, rfscale.pretrain_grid,
          rfscale.backbone_embed);
      int stride = bcfg.patch_size * std::max(1, bcfg.num_windows);
      if (imgsz % stride == 0) {
        side = imgsz;
      } else {
        std::cerr << "[warn] rfdetr-" << rfscale.upstream_id
                  << ": --imgsz=" << imgsz
                  << " not divisible by patch×num_windows=" << stride
                  << "; falling back to variant default "
                  << default_side << "\n";
      }
    }
    auto dets = inference::rfdetr_predict_image(
        m, bgr, side, dev, nm.conf_thresh, /*max_det=*/300);

    // Render annotated boxes onto the source image.
    if (!out_path.empty()) {
      auto img = bgr.clone();
      for (const auto& d : dets) {
        cv::rectangle(img,
                       cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                       cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)),
                       cv::Scalar(0, 255, 0), 2);
        std::ostringstream lab;
        lab.precision(2); lab << std::fixed << "cls=" << d.cls
            << " " << d.conf;
        cv::putText(img, lab.str(),
                     cv::Point(static_cast<int>(d.x1),
                                std::max(0, static_cast<int>(d.y1) - 4)),
                     cv::FONT_HERSHEY_SIMPLEX, 0.5,
                     cv::Scalar(0, 255, 0), 1);
      }
      std::filesystem::create_directories(
          std::filesystem::path(out_path).parent_path());
      cv::imwrite(out_path, img);
    }
    return dets;
  };
  a.run_val = [](const std::string&, const std::string& scale, int nc,
                  datasets::YoloDataset&, const torch::Device&) {
    models::RFDetr m(models::rfdetr_scale_from_letter(scale), nc);
    m->load_from_state_dict({});  // throws via #65D
    return VersionAdapter::ValResult{};
  };
  a.run_train_detect = [](const std::string& weights, const std::string& scale,
                           int nc, datasets::YoloDataset ds,
                           const engine::TrainConfig& tc) {
    models::RFDetr m(models::rfdetr_scale_from_letter(scale), nc);
    if (!weights.empty()) {
      // #65D pending: throws with a slice tag.
      m->load_from_state_dict({});
    }
    // Random-init training is meaningful (Hungarian loss runs);
    // it just won't converge in a single session.
    engine::TrainerRFDetr t(m, std::move(ds), tc);
    t.run();
  };
  a.benchmark_pt = [](const engine::BenchConfig& cfg, const cv::Mat&,
                      const std::string& scale) {
    models::RFDetr m(models::rfdetr_scale_from_letter(scale), cfg.nc);
    m->forward_eval(torch::zeros({1, 3, cfg.imgsz, cfg.imgsz}));  // throws
    return engine::BenchResult{};
  };
  a.make_frame_predictor = [](const std::string&, const std::string& scale,
                               int nc, int, const std::string&) {
    models::RFDetr m(models::rfdetr_scale_from_letter(scale), nc);
    m->forward_eval(torch::zeros({1, 3, 32, 32}));  // throws via #65C
    return std::unique_ptr<inference::FramePredictor>{};
  };
  return a;
}

}  // namespace

void register_all_versions() {
  static std::once_flag once;
  std::call_once(once, []() {
    auto& r = Registry::instance();
    r.register_version(make_v3());
    r.register_version(make_v4());
    r.register_version(make_v5());
    r.register_version(make_v6());
    r.register_version(make_v7());
    r.register_version(make_v8());
    r.register_version(make_v9());
    r.register_version(make_v10());
    r.register_version(make_v11());
    r.register_version(make_v12());
    r.register_version(make_v13());
    r.register_version(make_v26());
    r.register_version(make_rfdetr());
  });
}

}  // namespace yolocpp::registry
