// yolocpp CLI — supports two argument styles, dispatched automatically:
//
// 1. kv-style key=value args (canonical, drop-in for upstream tooling):
//      yolocpp task=detect mode=train  model=yolo8n.pt data=coco/ epochs=100
//      yolocpp task=detect mode=val    model=yolo8n.pt data=coco/
//      yolocpp task=detect mode=predict model=yolo8n.pt source=bus.jpg
//      yolocpp task=detect mode=export model=yolo8n.pt format=trt
//      yolocpp mode=benchmark model=yolo8n.pt source=bus.jpg
//
// 2. Legacy subcommand style:
//      yolocpp predict --weights=yolo8n.pt --source=bus.jpg
//      yolocpp train   --data=...
//      yolocpp info
//
// Style is auto-detected: if any arg has the form "key=value" and there is
// no leading flag-style subcommand, we use the new parser.

#include <CLI11.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <memory>

#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <yolocpp/config.hpp>

#include "yolocpp/cli/args.hpp"
#include "yolocpp/registry/version_adapter.hpp"
#include "yolocpp/cli/data_yaml.hpp"
#include "yolocpp/cli/model_info.hpp"
#include "yolocpp/cli/resolve.hpp"
#include "yolocpp/core/device.hpp"
#include "yolocpp/core/version.hpp"
#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/engine/benchmark.hpp"
#include "yolocpp/engine/trainer.hpp"
#include "yolocpp/engine/validator.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/inference/task_predictors.hpp"
#include "yolocpp/inference/trt_predictor.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo11_tasks.hpp"
#include "yolocpp/models/yolo12.hpp"
#include "yolocpp/models/yolo13.hpp"
#include "yolocpp/models/yolo12_tasks.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo26_tasks.hpp"
#include "yolocpp/models/yolo5.hpp"
#include "yolocpp/models/yolo8.hpp"
#include "yolocpp/models/yolo8_classify.hpp"
#include "yolocpp/models/yolo8_tasks.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/trt_export.hpp"
#include "yolocpp/tasks/classify_train.hpp"
#include "yolocpp/tasks/pose_obb_train.hpp"
#include "yolocpp/tasks/segment_train.hpp"

namespace {

using yolocpp::models::Yolo8Scale;

Yolo8Scale parse_scale(const std::string& s) {
  if (s == "n") return yolocpp::models::kYolo8n;
  if (s == "s") return yolocpp::models::kYolo8s;
  if (s == "m") return yolocpp::models::kYolo8m;
  if (s == "l") return yolocpp::models::kYolo8l;
  if (s == "x") return yolocpp::models::kYolo8x;
  throw std::runtime_error("unknown YOLO8 scale: " + s);
}

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    while (!tok.empty() && std::isspace((unsigned char)tok.front())) tok.erase(0, 1);
    while (!tok.empty() && std::isspace((unsigned char)tok.back()))  tok.pop_back();
    if (!tok.empty()) out.push_back(std::move(tok));
  }
  return out;
}

// ── shared command bodies ────────────────────────────────────────────────

int cmd_info() {
  const auto info = yolocpp::build_info();
  std::cout << "yolocpp     " << info.yolocpp_version       << "\n"
            << "libtorch    " << info.libtorch_version      << "\n"
            << "cuda_tk     " << info.cuda_toolkit_version  << "\n"
            << "cuda_rt     " << info.cuda_runtime_version  << "\n"
            << "tensorrt    " << info.tensorrt_version      << "\n"
            << "opencv      " << info.opencv_version        << "\n";
  std::cout << "\nCUDA available: "
            << (yolocpp::cuda_available() ? "yes" : "no") << "\n";
  for (const auto& d : yolocpp::list_cuda_devices()) {
    std::cout << "  [" << d.index << "] " << d.name
              << " (sm_" << d.compute_capability_major << d.compute_capability_minor
              << ", " << (d.total_memory_bytes >> 20) << " MiB, "
              << d.multi_processor_count << " SMs)\n";
  }
  std::cout << "\nSupported tasks (predict mode):\n"
            << "  detect    — train, val, predict, export, benchmark\n"
            << "  classify  — predict (train/val deferred)\n"
            << "  segment   — predict (train/val deferred)\n"
            << "  pose      — predict (train/val deferred)\n"
            << "  obb       — predict (train/val deferred)\n";
  return 0;
}

// Write val results to runs/val/<weights_stem>_results.txt and echo
// the path. Mirrors runs/predict and runs/export so val artifacts have
// the same `runs/<mode>` home as everything else.
void write_val_results(const std::string& weights, const std::string& data,
                       int imgsz, double map_50, double map_50_95) {
  std::filesystem::create_directories("runs/val");
  auto stem = std::filesystem::path(weights).stem().string();
  if (stem.empty()) stem = "out";
  auto out_path = "runs/val/" + stem + "_results.txt";
  std::ofstream f(out_path);
  f << "weights=" << weights << "\n"
    << "data="    << data    << "\n"
    << "imgsz="   << imgsz   << "\n"
    << "mAP@0.5      = " << map_50    << "\n"
    << "mAP@0.5:0.95 = " << map_50_95 << "\n";
  std::cout << "[val] wrote " << out_path << "\n";
}

int cmd_predict(const std::string& weights, const std::string& source,
                const std::string& out, int imgsz, std::string device,
                std::string scale_s, int nc, float conf, float iou) {
  yolocpp::inference::NMSConfig c;
  c.conf_thresh = conf;
  c.iou_thresh  = iou;
  // Default output path: runs/predict/<source_stem>.jpg — mirrors the
  // train mode's `runs/train` convention. Caller's explicit `out=` wins.
  std::string out_path;
  if (!out.empty()) {
    out_path = out;
  } else {
    std::filesystem::create_directories("runs/predict");
    auto stem = std::filesystem::path(source).stem().string();
    if (stem.empty()) stem = "out";
    out_path = "runs/predict/" + stem + ".jpg";
  }
  if (weights.size() >= 4 && weights.substr(weights.size() - 4) == ".trt") {
    yolocpp::inference::TrtPredictor p(weights, imgsz);
    auto dets = p.predict_to_file(source, out_path, c);
    std::cout << "[predict] (trt) " << dets.size() << " detections, wrote "
              << out_path << "\n";
    return 0;
  }
  // Auto-resolve scale from filename when --scale is not passed.
  // Otherwise the libtorch Predictor falls back to scale=N for v10s/m/b/l/x
  // and silently load_state_dict's mismatched-shape weights.
  if (scale_s.empty()) {
    auto fs_scale = yolocpp::cli::scale_from_filename(weights);
    if (!fs_scale.empty()) scale_s = fs_scale;
  }
  yolocpp::inference::Predictor p(weights, imgsz, std::move(device), nc,
                                   parse_scale(scale_s));
  auto dets = p.predict_to_file(source, out_path, c);
  std::cout << "[predict] (libtorch) " << dets.size() << " detections, wrote "
            << out_path << "\n";
  return 0;
}

int cmd_val(const std::string& weights, const std::string& root,
            const std::string& names_csv, int imgsz, std::string device,
            std::string scale_s) {
  auto names = split_csv(names_csv);
  if (names.empty()) names = yolocpp::inference::coco_names();
  int nc = (int)names.size();
  yolocpp::datasets::AugConfig aug; aug.augment = false;
  yolocpp::datasets::YoloDataset ds(root, "val", imgsz, names, aug);
  auto torch_dev = (device == "cpu") ? torch::Device(torch::kCPU)
                  : torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                                : torch::Device(torch::kCPU);

  if (scale_s.empty()) {
    auto fs_scale = yolocpp::cli::scale_from_filename(weights);
    if (!fs_scale.empty()) scale_s = fs_scale;
  }
  auto v_hint = yolocpp::cli::version_from_filename(weights);

  // Registry-driven dispatch: each non-v8 adapter wires `run_val`
  // with its concrete holder type. v8 leaves it empty and falls back
  // to the unified `inference::Predictor` (which already holds
  // Yolo8Detect — the unified path's only supported architecture).
  yolocpp::registry::register_all_versions();
  if (const auto* adapter =
          yolocpp::registry::Registry::instance().find(v_hint);
      adapter && adapter->run_val) {
    auto r = adapter->run_val(weights, scale_s, nc, ds, torch_dev);
    std::cout << "mAP@0.5      = " << r.map_50    << "\n"
              << "mAP@0.5:0.95 = " << r.map_50_95 << "\n";
    write_val_results(weights, root, imgsz, r.map_50, r.map_50_95);
    return 0;
  }

  yolocpp::inference::Predictor p(weights, imgsz, device, nc,
                                   parse_scale(scale_s));
  auto res = yolocpp::engine::validate(p.model(), ds, p.device());
  std::cout << "mAP@0.5      = " << res.map_50    << "\n"
            << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
  write_val_results(weights, root, imgsz, res.map_50, res.map_50_95);
  return 0;
}

int cmd_train(const std::string& root, const std::string& names_csv,
              int imgsz, int epochs, int batch_size, double lr0,
              std::string device, std::string scale_s,
              const std::string& save_dir,
              const std::string& init_weights,
              int patience = 0,
              std::vector<std::pair<std::string, std::string>> args_for_yaml = {}) {
  auto names = split_csv(names_csv);
  if (names.empty()) names = yolocpp::inference::coco_names();
  int nc = (int)names.size();

  yolocpp::datasets::YoloDataset train_ds(root, "train", imgsz, names);

  yolocpp::engine::TrainConfig cfg;
  cfg.epochs     = epochs;
  cfg.batch_size = batch_size;
  cfg.imgsz      = imgsz;
  cfg.lr0        = lr0;
  cfg.device     = std::move(device);
  cfg.save_dir   = save_dir;
  cfg.patience   = patience;
  cfg.args_for_yaml = std::move(args_for_yaml);

  // Auto-attach val split for best.pt tracking.
  std::string val_dir = root + "/images/val";
  if (std::filesystem::exists(val_dir) &&
      !std::filesystem::is_empty(val_dir)) {
    yolocpp::datasets::AugConfig vaug; vaug.augment = false;
    cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
        root, "val", imgsz, names, vaug);
    cfg.val_every   = 1;
    std::cout << "[train] val split detected (" << cfg.val_dataset->size()
              << " imgs); will track best.pt by mAP@0.5:0.95\n";
  }

  std::string v_hint =
      init_weights.empty() ? "" : yolocpp::cli::version_from_filename(init_weights);

  auto load_init = [&](auto& model) {
    if (init_weights.empty()) return;
    auto sd = yolocpp::serialization::load_state_dict(init_weights);
    int copied = model->load_from_state_dict(sd.entries);
    std::cout << "[train] loaded " << copied << " weights from "
              << init_weights << "\n";
  };

  if (v_hint == "v12") {
    yolocpp::models::Yolo12Detect model(
        yolocpp::models::yolo12_scale_from_letter(scale_s), nc);
    load_init(model);
    yolocpp::engine::TrainerV12 trainer(model, train_ds, cfg);
    trainer.run();
    return 0;
  }
  if (v_hint == "v13") {
    yolocpp::models::Yolo13Detect model(
        yolocpp::models::yolo13_scale_from_letter(scale_s), nc);
    load_init(model);
    yolocpp::engine::TrainerV13 trainer(model, train_ds, cfg);
    trainer.run();
    return 0;
  }
  // Default v8 path.
  auto scale = parse_scale(scale_s);
  yolocpp::models::Yolo8Detect model(scale, nc);
  load_init(model);
  yolocpp::engine::Trainer trainer(model, train_ds, cfg);
  trainer.run();
  return 0;
}

int cmd_export(const std::string& weights, const std::string& format,
               const std::string& out, int imgsz, const std::string& scale_s_in,
               int nc, const std::string& input_name, bool fp16,
               const std::string& version_hint = "",
               const std::string& task = "detect") {
  // Determine version: explicit hint (from CLI) or filename inference.
  std::string version = version_hint.empty()
                            ? yolocpp::cli::version_from_filename(weights)
                            : version_hint;
  // Resolve scale: explicit CLI flag wins; otherwise infer from filename.
  // Without this, `yolo10s.pt` exports as scale=N (the default of
  // yolo10_scale_from_letter("")) and silently load_state_dict's the
  // wrong-shaped weights, producing garbage ONNX/TRT for s/m/b/l/x.
  std::string scale_s = scale_s_in;
  if (scale_s.empty()) {
    auto fs_scale = yolocpp::cli::scale_from_filename(weights);
    if (!fs_scale.empty()) {
      scale_s = fs_scale;
      std::cerr << "[hint] inferred scale=" << scale_s
                << " from " << weights << " (pass --scale to override)\n";
    }
  }
  // Resolve the per-version adapter from the registry; this replaces
  // ~250 lines of if-else dispatch (per-version × per-task ctor +
  // load_state_dict + warmup + emitter call). See
  // include/yolocpp/registry/version_adapter.hpp for the contract and
  // the "how to add a new YOLO version" walkthrough.
  yolocpp::registry::register_all_versions();
  const auto* adapter =
      yolocpp::registry::Registry::instance().find(version);
  if (!adapter) {
    throw std::runtime_error(
        "export: unknown version '" + version + "' — registry has " +
        std::to_string(
            yolocpp::registry::Registry::instance().known_ids().size()) +
        " known");
  }

  // Per-version + per-task imgsz default (v4=608, v6/v7-P6=1280,
  // classify=224). Honours explicit caller imgsz when not the
  // 640 default.
  if (imgsz == 640 && adapter->default_imgsz) {
    int v = adapter->default_imgsz(scale_s, task);
    if (v > 0) imgsz = v;
  }

  auto write_onnx = [&](const std::string& onnx_path) {
    yolocpp::serialization::OnnxExportConfig ocfg;
    ocfg.imgsz = imgsz; ocfg.input_name = input_name;
    if (!adapter->export_onnx) {
      throw std::runtime_error(
          "export: version '" + version + "' has no ONNX exporter wired");
    }
    adapter->export_onnx(weights, scale_s, nc, task, onnx_path, ocfg);
  };

  auto base_name = [&]() -> std::string {
    std::string base = adapter->default_export_basename;
    if (task != "detect") base += "_" + task;
    return base;
  }();

  // Default output dir: runs/export/. Mirrors the train mode's
  // `runs/train` convention. Caller's explicit `out=` wins.
  if (format == "onnx") {
    std::string path;
    if (!out.empty()) path = out;
    else {
      std::filesystem::create_directories("runs/export");
      path = "runs/export/" + base_name + ".onnx";
    }
    write_onnx(path);
    std::cout << "[export] (" << version << "/" << task << ") wrote "
              << path << "\n";
    return 0;
  }
  if (format == "trt" || format == "engine") {
    std::string path;
    std::string onnx_tmp;
    if (!out.empty()) {
      path     = out;
      onnx_tmp = out + ".tmp.onnx";
    } else {
      std::filesystem::create_directories("runs/export");
      path     = "runs/export/" + base_name + ".trt";
      onnx_tmp = "runs/export/" + base_name + ".tmp.onnx";
    }
    write_onnx(onnx_tmp);
    yolocpp::serialization::TrtBuildConfig tcfg;
    tcfg.imgsz = imgsz; tcfg.fp16 = fp16; tcfg.input_name = input_name;
    // v10's RepVGGDW 7×7 dwconv-with-bias stack saturates cls under TF32
    // accumulation; the registry adapter declares this with
    // `trt_disable_tf32`. Generic switch covers any future version that
    // hits the same class of issue.
    if (adapter->trt_disable_tf32) {
      tcfg.tf32 = false;
    }
    yolocpp::serialization::build_trt_engine(onnx_tmp, path, tcfg);
    // Clean up the throwaway intermediate ONNX after the TRT engine is
    // built — only the final `.trt` is user-visible. Caller's explicit
    // `out=` may have used a non-`.tmp.onnx` filename; use error_code
    // overload so a missing file isn't an exception.
    std::error_code ec;
    std::filesystem::remove(onnx_tmp, ec);
    std::cout << "[export] (" << version << "/" << task << ") wrote "
              << path << "\n";
    return 0;
  }
  std::cerr << "[export] unknown format: " << format
            << " (supported: onnx, trt)\n";
  return 2;
}

int cmd_benchmark(const std::string& weights, const std::string& source,
                  int imgsz, int warmup, int iters,
                  const std::string& cache, const std::string& device) {
  yolocpp::engine::BenchConfig cfg;
  cfg.weights      = weights;
  cfg.source       = source;
  cfg.imgsz        = imgsz;
  cfg.warmup_iters = warmup;
  cfg.iters        = iters;
  cfg.cache_dir    = cache;
  cfg.device       = device;
  auto rows = yolocpp::engine::run_benchmark(cfg);
  yolocpp::engine::print_benchmark(rows);
  return 0;
}

// ── New-style (kv) parser dispatcher ─────────────────────────────────────

constexpr const char* kSupportedTasks[] = {"detect", "classify", "segment", "pose", "obb"};
constexpr const char* kSupportedModes[] = {"train", "val", "predict", "export", "benchmark", "info"};

bool task_implemented(const std::string& task) {
  return task == "detect" || task == "classify" || task == "segment" ||
         task == "pose"   || task == "obb";
}

int cmd_predict_task(const std::string& task, const std::string& weights,
                     const std::string& source, std::string out, int imgsz,
                     std::string device, std::string scale_s, int nc,
                     float conf, float iou,
                     const std::string& version_hint = "") {
  yolocpp::inference::NMSConfig c;
  c.conf_thresh = conf; c.iou_thresh = iou;
  // Default output path: runs/predict/<source_stem>_<task>.jpg.
  // Mirrors train's `runs/train` convention. Caller's `out=` wins.
  if (out.empty()) {
    std::filesystem::create_directories("runs/predict");
    auto stem = std::filesystem::path(source).stem().string();
    if (stem.empty()) stem = "out";
    out = "runs/predict/" + stem + "_" + task + ".jpg";
  }

  if (task == "detect") {
    // .trt engines are version-agnostic — the serialized graph already
    // bakes in the architecture, so any task=detect TRT engine routes
    // through the TrtPredictor regardless of which YOLO version produced it.
    if (weights.size() >= 4 &&
        weights.substr(weights.size() - 4) == ".trt") {
      return cmd_predict(weights, source, out, imgsz, device, scale_s, nc, conf, iou);
    }
    // version_hint comes from the dispatcher's state-dict-based inference;
    // fall back to filename heuristics if the caller didn't pass one.
    auto version = version_hint.empty()
                       ? yolocpp::cli::version_from_filename(weights)
                       : version_hint;

    // Registry-driven dispatch: replaces ~120 lines of per-version
    // if-else. Each adapter's `predict_to_file` hook resolves its own
    // scale enum + per-version imgsz quirks (v4=608, v6 P6=1280, …)
    // and calls into `inference::predict_v<N>_to_file`. v8 has no
    // dedicated helper — it falls back to `cmd_predict` below.
    yolocpp::registry::register_all_versions();
    if (const auto* adapter =
            yolocpp::registry::Registry::instance().find(version);
        adapter && adapter->predict_to_file) {
      yolocpp::inference::NMSConfig nm;
      nm.conf_thresh = conf;
      nm.iou_thresh  = iou;
      auto n = adapter->predict_to_file(weights, source, out, imgsz,
                                         device, scale_s, nc, nm);
      std::cout << "[predict] (" << version << ") " << n
                << " detections, wrote " << out << "\n";
      return 0;
    }

    static const std::vector<std::string> kKnown = {
        "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v26", "rtdetr"};
    if (std::find(kKnown.begin(), kKnown.end(), version) == kKnown.end()) {
      std::cerr << "[error] unrecognised YOLO version '" << version
                << "' — supported set: yolo3..yolo13, yolo26\n";
      return 2;
    }
    // v8 (and any anchor-free model whose state-dict shape is v8-shape)
    // has no dedicated `predict_v<N>_to_file` — fall back to the
    // unified Predictor.
    return cmd_predict(weights, source, out, imgsz, device, scale_s, nc, conf, iou);
  }
  // For task variants we route on version_hint (or filename inference) too —
  // v11 / v26 task models have a different module layout and can't load
  // through the v8 task predictor classes.
  auto task_version = version_hint.empty()
                          ? yolocpp::cli::version_from_filename(weights)
                          : version_hint;
  bool is_v11 = (task_version == "v11");
  bool is_v26 = (task_version == "v26");

  if (task == "classify") {
    int sz = (imgsz == 640) ? 224 : imgsz;  // classify default 224
    if (is_v26) {
      yolocpp::inference::Yolo26ClassifyPredictor p(
          weights, sz, device, /*nc=*/1000,
          yolocpp::models::yolo26_scale_from_letter(scale_s));
      cv::Mat img = cv::imread(source, cv::IMREAD_COLOR);
      if (img.empty()) throw std::runtime_error("could not read " + source);
      auto r = p.predict(img, /*top_k=*/5);
      std::cout << "[classify] (v26) top-5:\n";
      for (auto& [cid, prob] : r.topk)
        std::cout << "  " << cid << "  " << prob << "\n";
      return 0;
    }
    if (is_v11) {
      yolocpp::inference::Yolo11ClassifyPredictor p(
          weights, sz, device, /*nc=*/1000,
          yolocpp::models::yolo11_scale_from_letter(scale_s));
      cv::Mat img = cv::imread(source, cv::IMREAD_COLOR);
      if (img.empty()) throw std::runtime_error("could not read " + source);
      auto r = p.predict(img, /*top_k=*/5);
      std::cout << "[classify] (v11) top-5:\n";
      for (auto& [cid, prob] : r.topk)
        std::cout << "  " << cid << "  " << prob << "\n";
      return 0;
    }
    yolocpp::inference::ClassifyPredictor p(weights, sz, device, /*nc=*/1000,
                                             parse_scale(scale_s));
    cv::Mat img = cv::imread(source, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("could not read " + source);
    auto r = p.predict(img, /*top_k=*/5);
    std::cout << "[classify] top-5:\n";
    for (auto& [cid, prob] : r.topk)
      std::cout << "  " << cid << "  " << prob << "\n";
    return 0;
  }
  if (task == "segment") {
    if (is_v26) {
      yolocpp::inference::Yolo26SegmentPredictor p(
          weights, imgsz, device, nc,
          yolocpp::models::yolo26_scale_from_letter(scale_s));
      auto insts = p.predict_to_file(source, out, c);
      std::cout << "[segment] (v26) " << insts.size() << " instances, wrote " << out << "\n";
      return 0;
    }
    if (is_v11) {
      yolocpp::inference::Yolo11SegmentPredictor p(
          weights, imgsz, device, nc,
          yolocpp::models::yolo11_scale_from_letter(scale_s));
      auto insts = p.predict_to_file(source, out, c);
      std::cout << "[segment] (v11) " << insts.size() << " instances, wrote " << out << "\n";
      return 0;
    }
    yolocpp::inference::SegmentPredictor p(weights, imgsz, device, nc,
                                            parse_scale(scale_s));
    auto insts = p.predict_to_file(source, out, c);
    std::cout << "[segment] " << insts.size() << " instances, wrote " << out << "\n";
    return 0;
  }
  if (task == "pose") {
    if (is_v26) {
      yolocpp::inference::Yolo26PosePredictor p(
          weights, imgsz, device, /*num_kpts=*/17, /*kpt_dim=*/3,
          yolocpp::models::yolo26_scale_from_letter(scale_s));
      auto insts = p.predict_to_file(source, out, c);
      std::cout << "[pose] (v26) " << insts.size() << " people, wrote " << out << "\n";
      return 0;
    }
    if (is_v11) {
      yolocpp::inference::Yolo11PosePredictor p(
          weights, imgsz, device, /*num_kpts=*/17, /*kpt_dim=*/3,
          yolocpp::models::yolo11_scale_from_letter(scale_s));
      auto insts = p.predict_to_file(source, out, c);
      std::cout << "[pose] (v11) " << insts.size() << " people, wrote " << out << "\n";
      return 0;
    }
    yolocpp::inference::PosePredictor p(weights, imgsz, device,
                                         /*num_kpts=*/17, /*kpt_dim=*/3,
                                         parse_scale(scale_s));
    auto insts = p.predict_to_file(source, out, c);
    std::cout << "[pose] " << insts.size() << " people, wrote " << out << "\n";
    return 0;
  }
  if (task == "obb") {
    int sz = (imgsz == 640) ? 1024 : imgsz;  // OBB default 1024
    if (is_v26) {
      yolocpp::inference::Yolo26OBBPredictor p(
          weights, sz, device, /*nc=*/15,
          yolocpp::models::yolo26_scale_from_letter(scale_s));
      auto insts = p.predict_to_file(source, out, c);
      std::cout << "[obb] (v26) " << insts.size() << " rotated boxes, wrote " << out << "\n";
      return 0;
    }
    if (is_v11) {
      yolocpp::inference::Yolo11OBBPredictor p(
          weights, sz, device, /*nc=*/15,
          yolocpp::models::yolo11_scale_from_letter(scale_s));
      auto insts = p.predict_to_file(source, out, c);
      std::cout << "[obb] (v11) " << insts.size() << " rotated boxes, wrote " << out << "\n";
      return 0;
    }
    yolocpp::inference::OBBPredictor p(weights, sz, device, /*nc=*/15,
                                        parse_scale(scale_s));
    auto insts = p.predict_to_file(source, out, c);
    std::cout << "[obb] " << insts.size() << " rotated boxes, wrote " << out << "\n";
    return 0;
  }
  std::cerr << "[error] unknown task: " << task << "\n";
  return 2;
}

int dispatch_kv(const yolocpp::cli::Args& a) {
  static const std::vector<std::string> kCanonical = {
      "task", "mode", "model", "weights", "source", "data", "format",
      "imgsz", "epochs", "batch", "lr0", "device", "scale", "names", "nc",
      "conf", "iou", "out", "save", "input_name", "fp16",
      "warmup", "iters", "cache", "patience", "version",
  };
  a.warn_unknown(kCanonical);

  std::string task = a.get_str("task", "detect");
  std::string mode = a.get_str("mode", "");
  // "model" is the canonical kv-style name; we also accept "weights".
  std::string weights = a.get_str("model", a.get_str("weights", ""));

  // Default mode if absent: predict if model+source given; info otherwise.
  if (mode.empty()) {
    if (!weights.empty() && !a.get_str("source").empty()) mode = "predict";
    else                                                  mode = "info";
  }
  if (std::find(std::begin(kSupportedTasks), std::end(kSupportedTasks), task)
        == std::end(kSupportedTasks)) {
    std::cerr << "[error] unknown task: " << task << "\n";
    return 2;
  }
  if (std::find(std::begin(kSupportedModes), std::end(kSupportedModes), mode)
        == std::end(kSupportedModes)) {
    std::cerr << "[error] unknown mode: " << mode << "\n";
    return 2;
  }
  if (!task_implemented(task) && mode != "info") {
    std::cerr << "[error] task='" << task << "' is not yet supported.\n";
    return 2;
  }

  // Common args.
  int    imgsz   = a.get_int   ("imgsz",   640);
  int    epochs  = a.get_int   ("epochs",  100);
  int    batch   = a.get_int   ("batch",   16);
  double lr0     = a.get_double("lr0",     0.01);
  int    nc      = a.get_int   ("nc",      80);
  float  conf    = (float)a.get_double("conf", 0.25);
  float  iou     = (float)a.get_double("iou",  0.45);
  bool   fp16    = a.get_bool ("fp16",   true);
  std::string device     = a.get_str("device", "");
  std::string scale_s    = a.get_str("scale",  "n");
  std::string source     = a.get_str("source", "");
  std::string data       = a.get_str("data",   "");
  std::string names_csv  = a.get_str("names",  "");
  std::string out        = a.get_str("out",    "");
  std::string save_dir   = a.get_str("save",   "runs/train");
  std::string input_name = a.get_str("input_name", "images");
  std::string format     = a.get_str("format", "");

  // Auto-resolve weights (search cwd / data / cache, download upstream
  // assets if recognised) and datasets (cwd / data; setup coco/coco8 if
  // recognised).
  if (!weights.empty()) weights = yolocpp::cli::resolve_weights(weights);

  // `data=` is yaml-only. resolve_dataset performs auto-download via the
  // yaml's `download:` URL when the dataset is missing locally.
  if (!data.empty()) {
    std::string yaml_spec = data;
    data = yolocpp::cli::resolve_dataset(yaml_spec);
    // Pull `names:` from the yaml when the user didn't pass names= explicitly.
    if (names_csv.empty()) {
      try {
        auto dy = yolocpp::cli::parse_data_yaml(yaml_spec);
        if (!dy.names.empty()) {
          std::ostringstream oss;
          for (size_t i = 0; i < dy.names.size(); ++i) {
            if (i) oss << ',';
            oss << dy.names[i];
          }
          names_csv = oss.str();
        }
      } catch (const std::exception& e) {
        std::cerr << "[warn] yaml names parse failed: " << e.what() << "\n";
      }
    }
  }

  // Auto-infer (version, scale, nc) from the .pt's actual layer shapes —
  // works for renamed checkpoints (best.pt, last.pt) where filename carries
  // no version letter. User can still override with version=/scale=/nc=.
  std::string inferred_version;
  if (!weights.empty()) {
    // For v4/v6/v7 the filename hint is authoritative — those models have
    // their own state-dict layouts that infer_model_info (which probes for
    // upstream-shaped keys) will mis-classify as v8.
    auto v_hint_pre = yolocpp::cli::version_from_filename(weights);
    bool legacy_pre = (v_hint_pre == "v3" || v_hint_pre == "v4" ||
                       v_hint_pre == "v6" || v_hint_pre == "v7" ||
                       v_hint_pre == "v9" || v_hint_pre == "v10");
    if (legacy_pre) {
      inferred_version = v_hint_pre;
      // For v10 specifically, the filename scale letter is authoritative —
      // infer_model_info shares stem channels between e.g. v10b and v10l
      // (both ch=64 with width=1.0) and will mis-classify b as l.
      // Prefer the filename's scale before infer_model_info gets called.
      if (v_hint_pre == "v10" && !a.has("scale")) {
        auto fs_scale = yolocpp::cli::scale_from_filename(weights);
        if (!fs_scale.empty()) scale_s = fs_scale;
      }
    }
    try {
      auto mi = yolocpp::cli::infer_model_info(weights);
      if (!a.has("version") && inferred_version.empty()) inferred_version = mi.version;
      // Don't let state-dict-derived scale override v10's filename hint —
      // mi.scale comes from the v8/v11 stem-channel table which doesn't
      // know about v10's `b` scale.
      bool keep_filename_scale_for_v10 =
          (inferred_version == "v10" && !a.has("scale") && !scale_s.empty());
      if (!a.has("scale") && !mi.scale.empty() && !keep_filename_scale_for_v10) {
        if (mi.scale != scale_s)
          std::cerr << "[hint] inferred scale=" << mi.scale
                    << " from " << weights << " (pass scale=... to override)\n";
        scale_s = mi.scale;
      }
      if (!a.has("nc") && mi.nc > 0) nc = mi.nc;
      std::cerr << "[hint] inferred " << (inferred_version.empty()
                                          ? a.get_str("version", "?")
                                          : inferred_version)
                << "/" << scale_s << " nc=" << nc
                << " from " << weights << "\n";
    } catch (const std::exception& e) {
      // v4 has a single architecture (no scales) and no upstream-style
      // marker keys, so infer_model_info will throw — silence that case.
      auto v_hint = yolocpp::cli::version_from_filename(weights);
      if (v_hint != "v3" && v_hint != "v4" && v_hint != "v6"
          && v_hint != "v7" && v_hint != "v9" && v_hint != "v10") {
        std::cerr << "[warn] auto-inference failed: " << e.what() << "\n";
      } else {
        inferred_version = v_hint;
      }
    }
  }

  // Auto finetune-LR: if user supplies pretrained weights AND didn't set
  // lr0 explicitly, drop lr0 to 0.001 (upstream finetune default).
  // Without this, lr0=0.01 destroys pretrained features in <100 steps.
  bool lr_explicit = a.has("lr0");
  if (!lr_explicit && !weights.empty() && mode == "train") {
    lr0 = 0.001;
    std::cerr << "[hint] using finetune lr0=0.001 (model=*.pt supplied);"
              << " pass lr0=... to override\n";
  }

  if (mode == "info")      return cmd_info();
  if (mode == "predict") {
    if (weights.empty() || source.empty()) {
      std::cerr << "[error] predict needs model=... source=...\n";
      return 2;
    }
    std::string ver = a.has("version") ? a.get_str("version")
                                       : inferred_version;
    return cmd_predict_task(task, weights, source, out, imgsz, device,
                            scale_s, nc, conf, iou, ver);
  }
  if (mode == "val") {
    if (data.empty()) {
      std::cerr << "[error] val needs data=...\n";
      return 2;
    }
    if (task == "detect") {
      if (weights.empty()) {
        std::cerr << "[error] val needs model=...\n"; return 2;
      }
      // v5 weights → instantiate Yolo5Detect and run the templated validator.
      auto version = a.has("version") ? a.get_str("version")
                                       : inferred_version;
      if (version == "v5") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        yolocpp::datasets::AugConfig vaug; vaug.augment = false;
        yolocpp::datasets::YoloDataset ds(data, "val", imgsz, names, vaug);
        auto v5_scale = yolocpp::models::yolo5_scale_from_letter(scale_s);
        yolocpp::models::Yolo5Detect m(v5_scale, (int)names.size());
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
        auto dev = device.empty()
                       ? (torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                      : torch::Device(torch::kCPU))
                       : torch::Device(device);
        auto res = yolocpp::engine::validate(m, ds, dev);
        std::cout << "mAP@0.5      = " << res.map_50    << "\n" << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
        write_val_results(weights, data, imgsz, res.map_50, res.map_50_95);
        return 0;
      }
      if (version == "v11") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        yolocpp::datasets::AugConfig vaug; vaug.augment = false;
        yolocpp::datasets::YoloDataset ds(data, "val", imgsz, names, vaug);
        auto v11_scale = yolocpp::models::yolo11_scale_from_letter(scale_s);
        yolocpp::models::Yolo11Detect m(v11_scale, (int)names.size());
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
        auto dev = device.empty()
                       ? (torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                      : torch::Device(torch::kCPU))
                       : torch::Device(device);
        auto res = yolocpp::engine::validate(m, ds, dev);
        std::cout << "mAP@0.5      = " << res.map_50    << "\n" << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
        write_val_results(weights, data, imgsz, res.map_50, res.map_50_95);
        return 0;
      }
      if (version == "v26") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        yolocpp::datasets::AugConfig vaug; vaug.augment = false;
        yolocpp::datasets::YoloDataset ds(data, "val", imgsz, names, vaug);
        auto v26_scale = yolocpp::models::yolo26_scale_from_letter(scale_s);
        yolocpp::models::Yolo26Detect m(v26_scale, (int)names.size());
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
        auto dev = device.empty()
                       ? (torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                      : torch::Device(torch::kCPU))
                       : torch::Device(device);
        auto res = yolocpp::engine::validate(m, ds, dev);
        std::cout << "mAP@0.5      = " << res.map_50    << "\n" << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
        write_val_results(weights, data, imgsz, res.map_50, res.map_50_95);
        return 0;
      }
      auto pick_dev = [&]() {
        return device.empty()
                   ? (torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                  : torch::Device(torch::kCPU))
                   : torch::Device(device);
      };
      if (version == "v3") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        yolocpp::datasets::AugConfig vaug; vaug.augment = false;
        yolocpp::datasets::YoloDataset ds(data, "val", imgsz, names, vaug);
        yolocpp::models::Yolo3 m(yolocpp::models::kYolo3, (int)names.size());
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
        auto res = yolocpp::engine::validate(m, ds, pick_dev());
        std::cout << "mAP@0.5      = " << res.map_50    << "\n" << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
        write_val_results(weights, data, imgsz, res.map_50, res.map_50_95);
        return 0;
      }
      if (version == "v4") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        // v4 anchors calibrate to 608×608; default unless caller overrode.
        int v4_imgsz = (imgsz == 640) ? 608 : imgsz;
        yolocpp::datasets::AugConfig vaug; vaug.augment = false;
        yolocpp::datasets::YoloDataset ds(data, "val", v4_imgsz, names, vaug);
        yolocpp::models::Yolo4 m((int)names.size());
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
        auto res = yolocpp::engine::validate(m, ds, pick_dev());
        std::cout << "mAP@0.5      = " << res.map_50    << "\n" << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
        write_val_results(weights, data, imgsz, res.map_50, res.map_50_95);
        return 0;
      }
      if (version == "v6") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        auto v6_scale = yolocpp::models::kYolo6s;
        bool v6_p6 = false;
        if      (scale_s == "n") v6_scale = yolocpp::models::kYolo6n;
        else if (scale_s == "m") v6_scale = yolocpp::models::kYolo6m;
        else if (scale_s == "l") v6_scale = yolocpp::models::kYolo6l;
        else if (scale_s == "s_mbla") v6_scale = yolocpp::models::kYolo6s_mbla;
        else if (scale_s == "m_mbla") v6_scale = yolocpp::models::kYolo6m_mbla;
        else if (scale_s == "l_mbla") v6_scale = yolocpp::models::kYolo6l_mbla;
        else if (scale_s == "x_mbla") v6_scale = yolocpp::models::kYolo6x_mbla;
        else if (scale_s == "n6") { v6_scale = yolocpp::models::kYolo6n; v6_p6 = true; }
        else if (scale_s == "s6") { v6_scale = yolocpp::models::kYolo6s; v6_p6 = true; }
        else if (scale_s == "m6") { v6_scale = yolocpp::models::kYolo6m; v6_p6 = true; }
        else if (scale_s == "l6") { v6_scale = yolocpp::models::kYolo6l; v6_p6 = true; }
        // P6 variants need imgsz=1280 unless caller overrides.
        int v6_imgsz = (v6_p6 && imgsz == 640) ? 1280 : imgsz;
        yolocpp::datasets::AugConfig vaug; vaug.augment = false;
        yolocpp::datasets::YoloDataset ds(data, "val", v6_imgsz, names, vaug);
        yolocpp::models::Yolo6 m((int)names.size(), v6_scale, /*reg_max=*/16, v6_p6);
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
        auto res = yolocpp::engine::validate(m, ds, pick_dev());
        std::cout << "mAP@0.5      = " << res.map_50    << "\n" << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
        write_val_results(weights, data, imgsz, res.map_50, res.map_50_95);
        return 0;
      }
      if (version == "v7") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        // P6 variants (w6/e6/d6/e6e) need 1280² unless caller overrode.
        auto v7_scale = yolocpp::models::yolo7_scale_from_letter(scale_s);
        bool p6 = (v7_scale == yolocpp::models::Yolo7Scale::W6 ||
                   v7_scale == yolocpp::models::Yolo7Scale::E6 ||
                   v7_scale == yolocpp::models::Yolo7Scale::D6 ||
                   v7_scale == yolocpp::models::Yolo7Scale::E6e);
        int v7_imgsz = (p6 && imgsz == 640) ? 1280 : imgsz;
        yolocpp::datasets::AugConfig vaug; vaug.augment = false;
        yolocpp::datasets::YoloDataset ds(data, "val", v7_imgsz, names, vaug);
        yolocpp::models::Yolo7 m((int)names.size(), v7_scale);
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
        auto res = yolocpp::engine::validate(m, ds, pick_dev());
        std::cout << "mAP@0.5      = " << res.map_50    << "\n" << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
        write_val_results(weights, data, imgsz, res.map_50, res.map_50_95);
        return 0;
      }
      if (version == "v9") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        yolocpp::datasets::AugConfig vaug; vaug.augment = false;
        yolocpp::datasets::YoloDataset ds(data, "val", imgsz, names, vaug);
        auto v9_scale = yolocpp::models::yolo9_scale_from_letter(scale_s);
        yolocpp::models::Yolo9 m(v9_scale, (int)names.size());
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
        auto res = yolocpp::engine::validate(m, ds, pick_dev());
        std::cout << "mAP@0.5      = " << res.map_50    << "\n" << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
        write_val_results(weights, data, imgsz, res.map_50, res.map_50_95);
        return 0;
      }
      if (version == "v10") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        yolocpp::datasets::AugConfig vaug; vaug.augment = false;
        yolocpp::datasets::YoloDataset ds(data, "val", imgsz, names, vaug);
        auto v10_scale = yolocpp::models::yolo10_scale_from_letter(scale_s);
        yolocpp::models::Yolo10 m(v10_scale, (int)names.size());
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
        auto res = yolocpp::engine::validate(m, ds, pick_dev());
        std::cout << "mAP@0.5      = " << res.map_50    << "\n" << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
        write_val_results(weights, data, imgsz, res.map_50, res.map_50_95);
        return 0;
      }
      return cmd_val(weights, data, names_csv, imgsz, device, scale_s);
    }
    if (task == "classify") {
      int sz = (imgsz == 640) ? 224 : imgsz;
      yolocpp::tasks::ClassifyDataset ds(data, "val", sz, /*augment=*/false);
      yolocpp::models::Yolo8Classify m(parse_scale(scale_s), ds.num_classes());
      if (!weights.empty()) {
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
      }
      auto dev = (device.empty() && torch::cuda::is_available()) ?
                  torch::Device(torch::kCUDA) :
                  (device.empty() ? torch::Device(torch::kCPU)
                                  : torch::Device(device));
      auto r = yolocpp::tasks::validate_classify(m, ds, dev);
      std::cout << "top1=" << r.top1_acc << " top5=" << r.top5_acc
                << " (n=" << r.n_total << ")\n";
      return 0;
    }
    if (task == "segment") {
      auto names = split_csv(names_csv);
      if (names.empty()) names = yolocpp::inference::coco_names();
      yolocpp::tasks::SegDataset ds(data, "val", imgsz, names, /*augment=*/false);
      yolocpp::models::Yolo8Segment m(parse_scale(scale_s), ds.num_classes());
      if (!weights.empty()) {
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
      }
      auto dev = device.empty()
                     ? (torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                    : torch::Device(torch::kCPU))
                     : torch::Device(device);
      auto r = yolocpp::tasks::validate_segment(m, ds, dev);
      std::cout << "mask mAP@0.5=" << r.map_50
                << " (pred=" << r.n_predictions
                << " gt=" << r.n_ground_truths << ")\n";
      return 0;
    }
    if (task == "pose") {
      yolocpp::tasks::PoseDataset ds(data, "val", imgsz, /*num_kpts=*/17, /*kpt_dim=*/3,
                                      /*augment=*/false);
      yolocpp::models::Yolo8Pose m(parse_scale(scale_s), /*nc=*/1, 17, 3);
      if (!weights.empty()) {
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
      }
      auto dev = device.empty()
                     ? (torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                    : torch::Device(torch::kCPU))
                     : torch::Device(device);
      auto r = yolocpp::tasks::validate_pose(m, ds, dev);
      std::cout << "OKS mAP@0.5=" << r.oks_map_50 << "\n";
      return 0;
    }
    if (task == "obb") {
      auto names = split_csv(names_csv);
      if (names.empty()) names = yolocpp::inference::dota_names();
      yolocpp::tasks::OBBDataset ds(data, "val", imgsz, names, /*augment=*/false);
      yolocpp::models::Yolo8OBB m(parse_scale(scale_s), ds.num_classes(), /*ne=*/1);
      if (!weights.empty()) {
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
      }
      auto dev = device.empty()
                     ? (torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                    : torch::Device(torch::kCPU))
                     : torch::Device(device);
      auto r = yolocpp::tasks::validate_obb(m, ds, dev);
      std::cout << "rotated mAP@0.5=" << r.map_50 << "\n";
      return 0;
    }
  }
  if (mode == "train") {
    if (data.empty()) {
      std::cerr << "[error] train needs data=...\n"; return 2;
    }
    if (task == "detect") {
      int patience = a.get_int("patience", 0);
      std::vector<std::pair<std::string, std::string>> yaml_args(
          a.kv().begin(), a.kv().end());

      // v5 → build Yolo5Detect + TrainerV5 inline (cmd_train is v8-only).
      auto version = a.has("version") ? a.get_str("version")
                                       : inferred_version;
      if (version == "v5") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        int nc_v5 = (int)names.size();
        yolocpp::datasets::YoloDataset train_ds(data, "train", imgsz, names);
        auto v5_scale = yolocpp::models::yolo5_scale_from_letter(scale_s);
        yolocpp::models::Yolo5Detect model(v5_scale, nc_v5);
        if (!weights.empty()) {
          auto sd = yolocpp::serialization::load_state_dict(weights);
          int copied = model->load_from_state_dict(sd.entries);
          std::cout << "[train] (v5) loaded " << copied
                    << " weights from " << weights << "\n";
        }
        yolocpp::engine::TrainConfig cfg;
        cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
        cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
        cfg.patience = patience; cfg.args_for_yaml = std::move(yaml_args);
        // Auto-attach val split for best.pt tracking.
        std::string val_dir = data + "/images/val";
        if (std::filesystem::exists(val_dir) &&
            !std::filesystem::is_empty(val_dir)) {
          yolocpp::datasets::AugConfig vaug; vaug.augment = false;
          cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
              data, "val", imgsz, names, vaug);
          cfg.val_every = 1;
          std::cout << "[train] val split detected (" << cfg.val_dataset->size()
                    << " imgs); will track best.pt by mAP@0.5:0.95\n";
        }
        yolocpp::engine::TrainerV5 trainer(model, train_ds, cfg);
        trainer.run();
        return 0;
      }
      if (version == "v26") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        int nc_v26 = (int)names.size();
        yolocpp::datasets::YoloDataset train_ds(data, "train", imgsz, names);
        auto v26_scale = yolocpp::models::yolo26_scale_from_letter(scale_s);
        yolocpp::models::Yolo26Detect model(v26_scale, nc_v26);
        if (!weights.empty()) {
          auto sd = yolocpp::serialization::load_state_dict(weights);
          int copied = model->load_from_state_dict(sd.entries);
          std::cout << "[train] (v26) loaded " << copied
                    << " weights from " << weights << "\n";
        }
        yolocpp::engine::TrainConfig cfg;
        cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
        cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
        cfg.patience = patience; cfg.args_for_yaml = std::move(yaml_args);
        std::string val_dir = data + "/images/val";
        if (std::filesystem::exists(val_dir) &&
            !std::filesystem::is_empty(val_dir)) {
          yolocpp::datasets::AugConfig vaug; vaug.augment = false;
          cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
              data, "val", imgsz, names, vaug);
          cfg.val_every = 1;
          std::cout << "[train] val split detected (" << cfg.val_dataset->size()
                    << " imgs); will track best.pt by mAP@0.5:0.95\n";
        }
        yolocpp::engine::TrainerV26 trainer(model, train_ds, cfg);
        trainer.run();
        return 0;
      }
      if (version == "v11") {
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        int nc_v11 = (int)names.size();
        yolocpp::datasets::YoloDataset train_ds(data, "train", imgsz, names);
        auto v11_scale = yolocpp::models::yolo11_scale_from_letter(scale_s);
        yolocpp::models::Yolo11Detect model(v11_scale, nc_v11);
        if (!weights.empty()) {
          auto sd = yolocpp::serialization::load_state_dict(weights);
          int copied = model->load_from_state_dict(sd.entries);
          std::cout << "[train] (v11) loaded " << copied
                    << " weights from " << weights << "\n";
        }
        yolocpp::engine::TrainConfig cfg;
        cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
        cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
        cfg.patience = patience; cfg.args_for_yaml = std::move(yaml_args);
        std::string val_dir = data + "/images/val";
        if (std::filesystem::exists(val_dir) &&
            !std::filesystem::is_empty(val_dir)) {
          yolocpp::datasets::AugConfig vaug; vaug.augment = false;
          cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
              data, "val", imgsz, names, vaug);
          cfg.val_every = 1;
          std::cout << "[train] val split detected (" << cfg.val_dataset->size()
                    << " imgs); will track best.pt by mAP@0.5:0.95\n";
        }
        yolocpp::engine::TrainerV11 trainer(model, train_ds, cfg);
        trainer.run();
        return 0;
      }
      if (version == "v9") {
        // v9 reuses v8's anchor-free DFL Detect head (legacy=true) so it
        // plugs into the default V8DetectionLoss via TrainerT — no v9-
        // specific loss class needed. Note: PGI auxiliary branch is
        // training-only in upstream v9; we train the deploy head only,
        // which still converges (mAP plateaus a few points lower than
        // upstream's full PGI path; sufficient for finetune scenarios).
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        int nc_v9 = (int)names.size();
        yolocpp::datasets::YoloDataset train_ds(data, "train", imgsz, names);
        auto v9_scale = yolocpp::models::yolo9_scale_from_letter(scale_s);
        yolocpp::models::Yolo9 model(v9_scale, nc_v9);
        if (!weights.empty()) {
          auto sd = yolocpp::serialization::load_state_dict(weights);
          int copied = model->load_from_state_dict(sd.entries);
          std::cout << "[train] (v9) loaded " << copied
                    << " weights from " << weights << "\n";
        }
        yolocpp::engine::TrainConfig cfg;
        cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
        cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
        cfg.patience = patience; cfg.args_for_yaml = std::move(yaml_args);
        std::string val_dir = data + "/images/val";
        if (std::filesystem::exists(val_dir) &&
            !std::filesystem::is_empty(val_dir)) {
          yolocpp::datasets::AugConfig vaug; vaug.augment = false;
          cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
              data, "val", imgsz, names, vaug);
          cfg.val_every = 1;
          std::cout << "[train] val split detected (" << cfg.val_dataset->size()
                    << " imgs); will track best.pt by mAP@0.5:0.95\n";
        }
        yolocpp::engine::TrainerV9 trainer(model, train_ds, cfg);
        trainer.run();
        return 0;
      }
      if (version == "v3") {
        // yolov3u uses Darknet-53 + v8 anchor-free DFL Detect head
        // (legacy=true). Plugs into the default V8DetectionLoss via
        // TrainerT exactly like v9 — no v3-specific loss class needed.
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        int nc_v3 = (int)names.size();
        yolocpp::datasets::YoloDataset train_ds(data, "train", imgsz, names);
        yolocpp::models::Yolo3 model(yolocpp::models::kYolo3, nc_v3);
        if (!weights.empty()) {
          auto sd = yolocpp::serialization::load_state_dict(weights);
          int copied = model->load_from_state_dict(sd.entries);
          std::cout << "[train] (v3) loaded " << copied
                    << " weights from " << weights << "\n";
        }
        yolocpp::engine::TrainConfig cfg;
        cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
        cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
        cfg.patience = patience; cfg.args_for_yaml = std::move(yaml_args);
        std::string val_dir = data + "/images/val";
        if (std::filesystem::exists(val_dir) &&
            !std::filesystem::is_empty(val_dir)) {
          yolocpp::datasets::AugConfig vaug; vaug.augment = false;
          cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
              data, "val", imgsz, names, vaug);
          cfg.val_every = 1;
          std::cout << "[train] val split detected (" << cfg.val_dataset->size()
                    << " imgs); will track best.pt by mAP@0.5:0.95\n";
        }
        yolocpp::engine::TrainerV3 trainer(model, train_ds, cfg);
        trainer.run();
        return 0;
      }
      if (version == "v6") {
        // v6 uses VFL (Varifocal Loss) + SIoU + TAL via V6DetectionLoss.
        // Targets the DFL-headed variants (m/l/m6/l6 with reg_max=16).
        // For n/s/n6/s6, the head's reg_preds_dist branch carries the
        // 68-ch DFL distribution and feeds the same loss path — at the
        // deploy-form converter the dist branch is preserved.
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        int nc_v6 = (int)names.size();
        auto v6_scale = yolocpp::models::kYolo6s;
        bool v6_p6 = false;
        if      (scale_s == "n") v6_scale = yolocpp::models::kYolo6n;
        else if (scale_s == "m") v6_scale = yolocpp::models::kYolo6m;
        else if (scale_s == "l") v6_scale = yolocpp::models::kYolo6l;
        else if (scale_s == "s_mbla") v6_scale = yolocpp::models::kYolo6s_mbla;
        else if (scale_s == "m_mbla") v6_scale = yolocpp::models::kYolo6m_mbla;
        else if (scale_s == "l_mbla") v6_scale = yolocpp::models::kYolo6l_mbla;
        else if (scale_s == "x_mbla") v6_scale = yolocpp::models::kYolo6x_mbla;
        else if (scale_s == "n6") { v6_scale = yolocpp::models::kYolo6n; v6_p6 = true; }
        else if (scale_s == "s6") { v6_scale = yolocpp::models::kYolo6s; v6_p6 = true; }
        else if (scale_s == "m6") { v6_scale = yolocpp::models::kYolo6m; v6_p6 = true; }
        else if (scale_s == "l6") { v6_scale = yolocpp::models::kYolo6l; v6_p6 = true; }
        int v6_imgsz = (v6_p6 && imgsz == 640) ? 1280 : imgsz;
        yolocpp::datasets::YoloDataset train_ds(data, "train", v6_imgsz, names);
        yolocpp::models::Yolo6 model(nc_v6, v6_scale, /*reg_max=*/16, v6_p6);
        if (!weights.empty()) {
          auto sd = yolocpp::serialization::load_state_dict(weights);
          int copied = model->load_from_state_dict(sd.entries);
          std::cout << "[train] (v6) loaded " << copied
                    << " weights from " << weights << "\n";
        }
        yolocpp::engine::TrainConfig cfg;
        cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = v6_imgsz;
        cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
        cfg.patience = patience; cfg.args_for_yaml = std::move(yaml_args);
        std::string val_dir = data + "/images/val";
        if (std::filesystem::exists(val_dir) &&
            !std::filesystem::is_empty(val_dir)) {
          yolocpp::datasets::AugConfig vaug; vaug.augment = false;
          cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
              data, "val", v6_imgsz, names, vaug);
          cfg.val_every = 1;
          std::cout << "[train] val split detected (" << cfg.val_dataset->size()
                    << " imgs); will track best.pt by mAP@0.5:0.95\n";
        }
        yolocpp::engine::TrainerV6 trainer(model, train_ds, cfg);
        trainer.run();
        return 0;
      }
      if (version == "v4") {
        // v4 anchor-based v3-style loss via V7DetectionLoss with v4-
        // specific anchors + scale_xy=[1.2, 1.1, 1.05] + exp() wh decode.
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        int v4_imgsz = (imgsz == 640) ? 608 : imgsz;
        yolocpp::datasets::YoloDataset train_ds(data, "train", v4_imgsz, names);
        yolocpp::models::Yolo4 model((int)names.size());
        if (!weights.empty()) {
          auto sd = yolocpp::serialization::load_state_dict(weights);
          int copied = model->load_from_state_dict(sd.entries);
          std::cout << "[train] (v4) loaded " << copied
                    << " weights from " << weights << "\n";
        }
        yolocpp::engine::TrainConfig cfg;
        cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = v4_imgsz;
        cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
        cfg.patience = patience; cfg.args_for_yaml = std::move(yaml_args);
        std::string val_dir = data + "/images/val";
        if (std::filesystem::exists(val_dir) &&
            !std::filesystem::is_empty(val_dir)) {
          yolocpp::datasets::AugConfig vaug; vaug.augment = false;
          cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
              data, "val", v4_imgsz, names, vaug);
          cfg.val_every = 1;
          std::cout << "[train] val split detected (" << cfg.val_dataset->size()
                    << " imgs); will track best.pt by mAP@0.5:0.95\n";
        }
        yolocpp::engine::TrainerV4 trainer(model, train_ds, cfg);
        trainer.run();
        return 0;
      }
      if (version == "v7") {
        // v7 anchor-based loss via V7DetectionLoss with v7 anchors
        // + scale_xy=2.0 + (sigmoid*2)^2 wh decode.
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        int nc_v7 = (int)names.size();
        auto v7_scale = yolocpp::models::yolo7_scale_from_letter(scale_s);
        bool p6 = (v7_scale == yolocpp::models::Yolo7Scale::W6 ||
                   v7_scale == yolocpp::models::Yolo7Scale::E6 ||
                   v7_scale == yolocpp::models::Yolo7Scale::D6 ||
                   v7_scale == yolocpp::models::Yolo7Scale::E6e);
        int v7_imgsz = (p6 && imgsz == 640) ? 1280 : imgsz;
        yolocpp::datasets::YoloDataset train_ds(data, "train", v7_imgsz, names);
        yolocpp::models::Yolo7 model(v7_scale, nc_v7);
        if (!weights.empty()) {
          auto sd = yolocpp::serialization::load_state_dict(weights);
          int copied = model->load_from_state_dict(sd.entries);
          std::cout << "[train] (v7) loaded " << copied
                    << " weights from " << weights << "\n";
        }
        yolocpp::engine::TrainConfig cfg;
        cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = v7_imgsz;
        cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
        cfg.patience = patience; cfg.args_for_yaml = std::move(yaml_args);
        std::string val_dir = data + "/images/val";
        if (std::filesystem::exists(val_dir) &&
            !std::filesystem::is_empty(val_dir)) {
          yolocpp::datasets::AugConfig vaug; vaug.augment = false;
          cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
              data, "val", v7_imgsz, names, vaug);
          cfg.val_every = 1;
          std::cout << "[train] val split detected (" << cfg.val_dataset->size()
                    << " imgs); will track best.pt by mAP@0.5:0.95\n";
        }
        yolocpp::engine::TrainerV7 trainer(model, train_ds, cfg);
        trainer.run();
        return 0;
      }
      if (version == "v10") {
        // v10 train modes:
        //   default (dual_head=false) — trains the deploy one2one head
        //   only with V8DetectionLoss (legacy=false matches v11's cv3
        //   form). Used when finetuning from upstream's converted .pt
        //   (which strips the one2many keys at conversion time).
        //   `dual_head=true` — paper §3.1 consistent assignment: builds
        //   a parallel v8-style one2many head (legacy=true cv3) and
        //   sums V8DetectionLoss(o2m, topk=10) + V8DetectionLoss(o2o,
        //   topk=1) via Yolo10LossAdapter. Use for from-scratch
        //   training; loading upstream's pretrained one2many keys
        //   into the parallel head requires a converter pass that
        //   preserves `model.<head>.cv2/cv3.*` (currently dropped).
        bool v10_dual = a.get_bool("dual_head", false);
        auto names = split_csv(names_csv);
        if (names.empty()) names = yolocpp::inference::coco_names();
        int nc_v10 = (int)names.size();
        yolocpp::datasets::YoloDataset train_ds(data, "train", imgsz, names);
        auto v10_scale = yolocpp::models::yolo10_scale_from_letter(scale_s);
        yolocpp::models::Yolo10 model(v10_scale, nc_v10, v10_dual);
        if (v10_dual) {
          std::cout << "[train] (v10) dual-head training enabled — "
                       "running V10DualLoss (one2many topk=10 + "
                       "one2one topk=1).\n";
        }
        if (!weights.empty()) {
          auto sd = yolocpp::serialization::load_state_dict(weights);
          int copied = model->load_from_state_dict(sd.entries);
          std::cout << "[train] (v10) loaded " << copied
                    << " weights from " << weights << "\n";
        }
        yolocpp::engine::TrainConfig cfg;
        cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
        cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
        cfg.patience = patience; cfg.args_for_yaml = std::move(yaml_args);
        std::string val_dir = data + "/images/val";
        if (std::filesystem::exists(val_dir) &&
            !std::filesystem::is_empty(val_dir)) {
          yolocpp::datasets::AugConfig vaug; vaug.augment = false;
          cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
              data, "val", imgsz, names, vaug);
          cfg.val_every = 1;
          std::cout << "[train] val split detected (" << cfg.val_dataset->size()
                    << " imgs); will track best.pt by mAP@0.5:0.95\n";
        }
        yolocpp::engine::TrainerV10 trainer(model, train_ds, cfg);
        trainer.run();
        return 0;
      }
      return cmd_train(data, names_csv, imgsz, epochs, batch, lr0,
                       device, scale_s, save_dir, weights, patience,
                       std::move(yaml_args));
    }
    if (task == "classify") {
      int sz = (imgsz == 640) ? 224 : imgsz;
      yolocpp::tasks::ClassifyDataset tr(data, "train", sz, /*augment=*/true);
      yolocpp::models::Yolo8Classify m(parse_scale(scale_s), tr.num_classes());
      if (!weights.empty()) {
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
      }
      yolocpp::tasks::ClassifyTrainConfig cfg;
      cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = sz;
      cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
      yolocpp::tasks::train_classify(m, tr, /*val=*/nullptr, cfg);
      return 0;
    }
    if (task == "segment") {
      auto names = split_csv(names_csv);
      if (names.empty()) names = yolocpp::inference::coco_names();
      yolocpp::tasks::SegDataset tr(data, "train", imgsz, names, /*augment=*/true);
      yolocpp::models::Yolo8Segment m(parse_scale(scale_s), tr.num_classes());
      if (!weights.empty()) {
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
      }
      yolocpp::tasks::SegTrainConfig cfg;
      cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
      cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
      yolocpp::tasks::train_segment(m, tr, /*val=*/nullptr, cfg);
      return 0;
    }
    if (task == "pose") {
      yolocpp::tasks::PoseDataset tr(data, "train", imgsz, 17, 3, /*augment=*/true);
      yolocpp::models::Yolo8Pose m(parse_scale(scale_s), /*nc=*/1, 17, 3);
      if (!weights.empty()) {
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
      }
      yolocpp::tasks::PoseTrainConfig cfg;
      cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
      cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
      yolocpp::tasks::train_pose(m, tr, /*val=*/nullptr, cfg);
      return 0;
    }
    if (task == "obb") {
      auto names = split_csv(names_csv);
      if (names.empty()) names = yolocpp::inference::dota_names();
      yolocpp::tasks::OBBDataset tr(data, "train", imgsz, names, /*augment=*/true);
      yolocpp::models::Yolo8OBB m(parse_scale(scale_s), tr.num_classes(), /*ne=*/1);
      if (!weights.empty()) {
        auto sd = yolocpp::serialization::load_state_dict(weights);
        m->load_from_state_dict(sd.entries);
      }
      yolocpp::tasks::OBBTrainConfig cfg;
      cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
      cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
      yolocpp::tasks::train_obb(m, tr, /*val=*/nullptr, cfg);
      return 0;
    }
  }
  if (mode == "export") {
    if (weights.empty() || format.empty()) {
      std::cerr << "[error] export needs model=... format=onnx|trt\n";
      return 2;
    }
    return cmd_export(weights, format, out, imgsz, scale_s, nc, input_name, fp16,
                      a.has("version") ? a.get_str("version") : inferred_version,
                      task);
  }
  if (mode == "benchmark") {
    if (weights.empty() || source.empty()) {
      std::cerr << "[error] benchmark needs model=... source=...\n";
      return 2;
    }
    int warmup = a.get_int("warmup", 10);
    int iters  = a.get_int("iters",  100);
    std::string cache = a.get_str("cache", "build/bench_cache");
    return cmd_benchmark(weights, source, imgsz, warmup, iters, cache, device);
  }
  return 2;
}

bool looks_like_kv_style(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (!s.empty() && s.find('=') != std::string::npos &&
        s[0] != '-') {
      return true;
    }
  }
  return false;
}

}  // anonymous namespace

int main(int argc, char** argv) {
  // --version / -v / -V short-circuit. Reads YOLOCPP_VERSION_STRING
  // from the CMake-stamped config.hpp (which CMake derives from the
  // top-level ./VERSION file — single source of truth).
  if (argc == 2 && (std::string(argv[1]) == "--version" ||
                    std::string(argv[1]) == "-v" ||
                    std::string(argv[1]) == "-V")) {
    std::cout << "yolocpp " << YOLOCPP_VERSION_STRING << "\n";
    return 0;
  }

  // Print top-level help if user asks.
  if (argc <= 1 ||
      (argc == 2 && (std::string(argv[1]) == "--help" ||
                     std::string(argv[1]) == "-h"))) {
    std::cout <<
      "yolocpp — pure C++ computer vision suite\n"
      "\n"
      "Usage (kv-style):\n"
      "  yolocpp task=detect mode=train  model=yolo8n.pt data=DATA epochs=100\n"
      "  yolocpp task=detect mode=val    model=yolo8n.pt data=DATA\n"
      "  yolocpp task=detect mode=predict model=yolo8n.pt source=IMG\n"
      "  yolocpp task=detect mode=export model=yolo8n.pt format=trt\n"
      "  yolocpp mode=benchmark model=yolo8n.pt source=IMG\n"
      "\n"
      "Usage (legacy subcommand-style):\n"
      "  yolocpp {info|train|val|predict|export} [--key=value ...]\n"
      "\n"
      "Tasks   : detect (only — classify/segment/pose/obb planned for Phase 3)\n"
      "Modes   : train, val, predict, export, benchmark, info\n"
      "Formats : onnx, trt\n";
    return 0;
  }

  try {
    if (looks_like_kv_style(argc, argv)) {
      auto a = yolocpp::cli::Args::parse(argc, argv);
      return dispatch_kv(a);
    }

    // Legacy subcommand style (CLI11).
    CLI::App app{"yolocpp"};
    app.require_subcommand(1);

    // scale_s default is empty — cmd_predict/cmd_val/cmd_export auto-resolve
    // from the weights filename when not passed explicitly. Defaulting to
    // "n" here silently mis-classifies non-n scales (`yolo10s.pt` exports
    // as scale=N → mismatched-shape weights → garbage ONNX/TRT).
    std::string weights, source, out, root, names_csv, device, scale_s,
                save_dir = "runs/train", init_weights;
    int    imgsz = 640, epochs = 100, batch_size = 16, nc = 80;
    double lr0   = 0.01;
    float  conf  = 0.25f, iou = 0.45f;

    auto* t = app.add_subcommand("train", "Train a YOLO8 model");
    t->add_option("--data",    root)->required();
    t->add_option("--names",   names_csv);
    t->add_option("--imgsz",   imgsz);
    t->add_option("--epochs",  epochs);
    t->add_option("--batch",   batch_size);
    t->add_option("--lr0",     lr0);
    t->add_option("--device",  device);
    t->add_option("--scale",   scale_s);
    t->add_option("--save",    save_dir);
    t->add_option("--weights", init_weights);
    int legacy_patience = 0;
    t->add_option("--patience", legacy_patience,
                  "stop if val mAP@0.5:0.95 doesn't improve for N epochs");

    auto* v = app.add_subcommand("val", "Validate (mAP)");
    v->add_option("--weights", weights)->required();
    v->add_option("--data",    root)->required();
    v->add_option("--names",   names_csv);
    v->add_option("--imgsz",   imgsz);
    v->add_option("--device",  device);
    v->add_option("--scale",   scale_s);

    auto* p = app.add_subcommand("predict", "Run inference");
    p->add_option("--weights", weights)->required();
    p->add_option("--source",  source)->required();
    p->add_option("--out",     out);
    p->add_option("--imgsz",   imgsz);
    p->add_option("--device",  device);
    p->add_option("--scale",   scale_s);
    p->add_option("--nc",      nc);
    p->add_option("--conf",    conf);
    p->add_option("--iou",     iou);

    auto* e = app.add_subcommand("export", "Export to ONNX / TRT");
    std::string export_fmt, export_input_name = "images";
    bool   export_fp16 = true;
    e->add_option("--format",     export_fmt)->required();
    e->add_option("--weights",    weights)->required();
    e->add_option("--out",        out);
    e->add_option("--imgsz",      imgsz);
    e->add_option("--scale",      scale_s);
    e->add_option("--nc",         nc);
    e->add_option("--input-name", export_input_name);
    e->add_flag  ("--fp16,!--no-fp16", export_fp16);

    auto* b = app.add_subcommand("benchmark", "Latency / throughput benchmark");
    std::string cache = "build/bench_cache";
    int warmup = 10, iters = 100;
    b->add_option("--weights", weights)->required();
    b->add_option("--source",  source)->required();
    b->add_option("--imgsz",   imgsz);
    b->add_option("--warmup",  warmup);
    b->add_option("--iters",   iters);
    b->add_option("--cache",   cache);
    b->add_option("--device",  device);

    auto* i = app.add_subcommand("info", "Build / device info");
    (void)i;

    CLI11_PARSE(app, argc, argv);

    if (app.got_subcommand("info"))    return cmd_info();
    if (app.got_subcommand("predict"))
      return cmd_predict(weights, source, out, imgsz, device, scale_s, nc, conf, iou);
    if (app.got_subcommand("val"))
      return cmd_val(weights, root, names_csv, imgsz, device, scale_s);
    if (app.got_subcommand("train"))
      return cmd_train(root, names_csv, imgsz, epochs, batch_size, lr0,
                       device, scale_s, save_dir, init_weights,
                       legacy_patience);
    if (app.got_subcommand("export"))
      return cmd_export(weights, export_fmt, out, imgsz, scale_s, nc,
                        export_input_name, export_fp16);
    if (app.got_subcommand("benchmark"))
      return cmd_benchmark(weights, source, imgsz, warmup, iters, cache, device);
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 1;
  }
  return 1;
}
