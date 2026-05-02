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
#include <regex>
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

// Normalise + validate a `--device` spec. Accepted forms:
//   ""           → leave empty (callers pick CUDA when available, else CPU)
//   "auto"       → same as ""
//   "cpu"        → CPU
//   "cuda"       → first CUDA device
//   "cuda:N"     → CUDA device N (validated against torch::cuda::device_count())
//   "cuda:a,b"   → multi-device DDP launch (only the first index is honoured
//                  by the inference path; trainer's DDP launcher consumes
//                  the full list)
//   "mps"        → Apple Metal (returns the literal "mps" — torch CPU
//                  fallback handles routing on non-Apple platforms with
//                  a warning)
// Returns the canonicalised string. Throws on unparseable input or an
// out-of-range CUDA index.
std::string normalise_device(std::string d) {
  // strip whitespace
  while (!d.empty() && std::isspace((unsigned char)d.front())) d.erase(0, 1);
  while (!d.empty() && std::isspace((unsigned char)d.back()))  d.pop_back();
  if (d.empty() || d == "auto") return "";
  if (d == "cpu" || d == "mps") return d;
  if (d == "cuda") {
    if (!torch::cuda::is_available())
      throw std::runtime_error("--device=cuda requested but CUDA isn't available");
    return "cuda";
  }
  if (d.rfind("cuda:", 0) == 0) {
    auto idx_str = d.substr(5);
    if (idx_str.empty()) throw std::runtime_error("--device=cuda: missing device index");
    if (!torch::cuda::is_available())
      throw std::runtime_error("--device=" + d + " requested but CUDA isn't available");
    auto count = torch::cuda::device_count();
    // Validate every comma-separated index.
    std::stringstream ss(idx_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      try {
        int idx = std::stoi(tok);
        if (idx < 0 || idx >= (int)count)
          throw std::runtime_error("--device=" + d + ": index " + tok +
                                    " out of range (have " + std::to_string(count) + ")");
      } catch (const std::invalid_argument&) {
        throw std::runtime_error("--device=" + d + ": '" + tok + "' is not an integer");
      }
    }
    return d;
  }
  throw std::runtime_error(
      "--device='" + d +
      "' not recognised; expected cpu | cuda | cuda:N | cuda:0,1,... | mps | auto");
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
              std::vector<std::pair<std::string, std::string>> args_for_yaml = {},
              uint64_t seed = 0) {
  auto names = split_csv(names_csv);
  if (names.empty()) names = yolocpp::inference::coco_names();
  int nc = (int)names.size();

  // Auto-resolve scale from the init weights filename when not passed
  // (mirrors cmd_predict / cmd_val / cmd_export). Without this, the
  // registry-driven dispatch errors out on `scale_letter == ""` for
  // any version that requires a scale (v5/v6/v7/v9/v10/v11/v12/v13/v26).
  if (scale_s.empty() && !init_weights.empty()) {
    auto fs_scale = yolocpp::cli::scale_from_filename(init_weights);
    if (!fs_scale.empty()) scale_s = fs_scale;
  }

  yolocpp::datasets::YoloDataset train_ds(root, "train", imgsz, names);

  yolocpp::engine::TrainConfig cfg;
  cfg.epochs     = epochs;
  cfg.batch_size = batch_size;
  cfg.imgsz      = imgsz;
  cfg.lr0        = lr0;
  cfg.device     = std::move(device);
  cfg.save_dir   = save_dir;
  cfg.patience   = patience;
  cfg.seed       = seed;
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

  // Registry-driven dispatch: each non-v8 adapter wires
  // `run_train_detect` with its concrete holder type + matching
  // TrainerT<Holder>. v8 falls through to the explicit
  // `engine::Trainer = TrainerT<Yolo8Detect>` path below.
  yolocpp::registry::register_all_versions();
  if (const auto* adapter =
          yolocpp::registry::Registry::instance().find(v_hint);
      adapter && adapter->run_train_detect) {
    adapter->run_train_detect(init_weights, scale_s, nc,
                               std::move(train_ds), cfg);
    return 0;
  }

  auto scale = parse_scale(scale_s);
  yolocpp::models::Yolo8Detect model(scale, nc);
  if (!init_weights.empty()) {
    auto sd = yolocpp::serialization::load_state_dict(init_weights);
    int copied = model->load_from_state_dict(sd.entries);
    std::cout << "[train] loaded " << copied << " weights from "
              << init_weights << "\n";
  }
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

// ─── --source classification ─────────────────────────────────────────────
// The predict pipeline accepts a much broader `--source` spec than just a
// single image path. Classification rules (in order; first match wins):
//
//   "0", "1", … (digits only)   → webcam index           [SourceKind::Webcam]
//   matches `^[a-z][a-z0-9]*://` → URL stream            [SourceKind::Url]
//   ends in a video extension   → local video file       [SourceKind::Video]
//   contains '*' or '?' glob    → glob over images       [SourceKind::Glob]
//   directory on disk           → all images in dir      [SourceKind::Dir]
//   file with image extension   → single image           [SourceKind::Image]
//
// Classification is purely string-based — no I/O — so callers can detect
// invalid specs early. Webcam / Url / Video routing is gated behind the
// frame-level predict hook (TODO #51C2); for now those return a clear
// error pointing at the gap.
enum class SourceKind { Image, Dir, Glob, Video, Url, Webcam, Unknown };

bool ends_with_lower(const std::string& s, const std::string& suf) {
  if (s.size() < suf.size()) return false;
  for (std::size_t i = 0; i < suf.size(); ++i) {
    if (std::tolower((unsigned char)s[s.size() - suf.size() + i]) != suf[i])
      return false;
  }
  return true;
}

bool has_image_extension(const std::string& s) {
  for (const auto* ext : {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
    if (ends_with_lower(s, ext)) return true;
  return false;
}

bool has_video_extension(const std::string& s) {
  for (const auto* ext : {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
    if (ends_with_lower(s, ext)) return true;
  return false;
}

bool looks_like_url(const std::string& s) {
  // RTSP / HTTP / HTTPS / FTP / RTMP — any scheme:// prefix is a stream.
  auto p = s.find("://");
  if (p == std::string::npos || p == 0) return false;
  for (std::size_t i = 0; i < p; ++i) {
    char ch = s[i];
    if (!(std::isalpha((unsigned char)ch) ||
          std::isdigit((unsigned char)ch) || ch == '+' || ch == '-' || ch == '.'))
      return false;
  }
  return true;
}

bool looks_like_webcam_index(const std::string& s) {
  if (s.empty()) return false;
  for (char ch : s) if (!std::isdigit((unsigned char)ch)) return false;
  return true;
}

bool looks_like_glob(const std::string& s) {
  return s.find('*') != std::string::npos || s.find('?') != std::string::npos;
}

SourceKind classify_source(const std::string& spec) {
  if (spec.empty()) return SourceKind::Unknown;
  if (looks_like_webcam_index(spec)) return SourceKind::Webcam;
  if (looks_like_url(spec))          return SourceKind::Url;
  if (has_video_extension(spec))     return SourceKind::Video;
  if (looks_like_glob(spec))         return SourceKind::Glob;
  std::error_code ec;
  if (std::filesystem::is_directory(spec, ec)) return SourceKind::Dir;
  if (has_image_extension(spec))     return SourceKind::Image;
  if (std::filesystem::exists(spec, ec)) return SourceKind::Image;  // fall-through trust
  return SourceKind::Unknown;
}

// Expand an image-type source (single file / directory / glob) into a
// concrete list of image paths sorted lexicographically. Throws if the
// expansion yields zero matches (so callers get a clear error rather
// than silently doing nothing).
std::vector<std::string> expand_image_source(const std::string& spec) {
  std::vector<std::string> out;
  auto kind = classify_source(spec);
  namespace fs = std::filesystem;

  if (kind == SourceKind::Image) {
    out.push_back(spec);
  } else if (kind == SourceKind::Dir) {
    for (const auto& e : fs::directory_iterator(spec)) {
      if (e.is_regular_file() && has_image_extension(e.path().string()))
        out.push_back(e.path().string());
    }
  } else if (kind == SourceKind::Glob) {
    fs::path p(spec);
    auto parent = p.parent_path();
    if (parent.empty()) parent = ".";
    auto pat = p.filename().string();
    // Translate the glob to a small regex (only `*` and `?` supported).
    std::string re_str = "^";
    for (char c : pat) {
      if (c == '*') re_str += ".*";
      else if (c == '?') re_str += ".";
      else if (std::isalnum((unsigned char)c)) re_str += c;
      else { re_str += "\\"; re_str += c; }
    }
    re_str += "$";
    std::regex re(re_str, std::regex::icase);
    for (const auto& e : fs::directory_iterator(parent)) {
      if (e.is_regular_file() &&
          std::regex_match(e.path().filename().string(), re) &&
          has_image_extension(e.path().string()))
        out.push_back(e.path().string());
    }
  } else {
    throw std::runtime_error(
        "expand_image_source: '" + spec +
        "' is not a single image, directory, or glob");
  }
  std::sort(out.begin(), out.end());
  if (out.empty())
    throw std::runtime_error("source '" + spec + "' matched no images");
  return out;
}

// Inner predict-one-image path: runs the registered `predict_to_file`
// hook (or the unified Predictor fallback) on a single image. Returns
// the CLI exit code (0 = ok, 2 = unknown version).
int predict_one_image(const std::string& task, const std::string& weights,
                      const std::string& source, const std::string& out,
                      int imgsz, std::string device, std::string scale_s,
                      int nc, float conf, float iou,
                      const std::string& version_hint);

int cmd_predict_task(const std::string& task, const std::string& weights,
                     const std::string& source, std::string out, int imgsz,
                     std::string device, std::string scale_s, int nc,
                     float conf, float iou,
                     const std::string& version_hint = "") {
  // Auto-resolve scale from the weights filename when the caller
  // didn't pass --scale. The registry's per-version `predict_to_file`
  // hooks need a scale letter (`yolo11s.pt` → "s") to construct the
  // right holder; without this the legacy CLI11 subcommand path errors
  // with "unknown scale letter ''" on every non-v8 model.
  if (scale_s.empty()) {
    auto fs_scale = yolocpp::cli::scale_from_filename(weights);
    if (!fs_scale.empty()) scale_s = fs_scale;
  }

  // Classify the source. Image / Dir / Glob fan out to a list of image
  // paths and run inference per-image. Video / URL / Webcam are filed
  // under TODO #51C2 (needs a frame-level adapter hook + VideoCapture
  // loop) — surface a clear error instead of silently routing through
  // cv::imread which would fail with an unhelpful "image not found".
  auto kind = classify_source(source);
  if (kind == SourceKind::Video || kind == SourceKind::Url ||
      kind == SourceKind::Webcam) {
    std::cerr << "[error] --source='" << source
              << "' is a "
              << (kind == SourceKind::Video  ? "video file"
                : kind == SourceKind::Url    ? "URL stream"
                                              : "webcam index")
              << " — frame-level inference not yet wired (TODO #51C2). "
                 "Workaround: split to frames first (e.g. `ffmpeg -i "
                 "input.mp4 frames/%06d.jpg`) and pass `--source=frames/`.\n";
    return 2;
  }
  if (kind == SourceKind::Unknown) {
    std::cerr << "[error] --source='" << source
              << "' not recognised: not an existing file, directory, "
                 "glob, video, URL, or webcam index\n";
    return 2;
  }

  // Image / Dir / Glob → expand to a sorted list and iterate.
  std::vector<std::string> inputs;
  try {
    inputs = expand_image_source(source);
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 2;
  }

  // For multi-input cases (dir / glob), `out` becomes an OUTPUT
  // DIRECTORY rather than a single file. We accept the user's --out
  // either as a directory (mkdir -p, write `<basename>_<task>.jpg` per
  // input) or, for single inputs, as a literal output file (existing
  // behaviour).
  bool multi = (inputs.size() > 1);
  std::string out_dir;
  if (multi) {
    if (out.empty()) {
      out_dir = "runs/predict";
    } else {
      out_dir = out;
    }
    std::filesystem::create_directories(out_dir);
  }

  int rc = 0;
  for (const auto& in : inputs) {
    std::string this_out;
    if (multi) {
      auto stem = std::filesystem::path(in).stem().string();
      if (stem.empty()) stem = "out";
      this_out = out_dir + "/" + stem + "_" + task + ".jpg";
    } else if (out.empty()) {
      std::filesystem::create_directories("runs/predict");
      auto stem = std::filesystem::path(in).stem().string();
      if (stem.empty()) stem = "out";
      this_out = "runs/predict/" + stem + "_" + task + ".jpg";
    } else {
      this_out = out;
    }
    int sub_rc = predict_one_image(task, weights, in, this_out, imgsz,
                                    device, scale_s, nc, conf, iou,
                                    version_hint);
    if (sub_rc != 0) rc = sub_rc;  // last non-zero wins, but keep going
  }
  return rc;
}

int predict_one_image(const std::string& task, const std::string& weights,
                      const std::string& source, const std::string& out,
                      int imgsz, std::string device, std::string scale_s,
                      int nc, float conf, float iou,
                      const std::string& version_hint) {
  yolocpp::inference::NMSConfig c;
  c.conf_thresh = conf; c.iou_thresh = iou;

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
  std::string device     = normalise_device(a.get_str("device", ""));
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
      // The kv-style val path used to reimplement per-version
      // dispatch (v3..v11/v26 each rebuilt the holder + dataset +
      // validator inline). Since #46E that logic lives in
      // `cmd_val()` which routes through the registry's `run_val`
      // hook for every version. One call replaces ~170 lines.
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
      uint64_t kv_seed = static_cast<uint64_t>(a.get_int("seed", 0));
      std::vector<std::pair<std::string, std::string>> yaml_args(
          a.kv().begin(), a.kv().end());

      // The kv-style train path used to reimplement per-version dispatch
      // (v5/v26/v11/v9/v3/v6/v4/v7 each rebuilt the holder + dataset +
      // TrainerT inline). Since #46F that logic lives in `cmd_train()`
      // which routes through the registry's `run_train_detect` hook for
      // every non-v8 version. The v10 dual-head case below stays inline
      // because its (scale, nc, dual_head) ctor isn't on the standard
      // adapter signature — could be promoted to a `dual_head` arg in a
      // future #46Fx if more versions need ctor variants.
      auto version = a.has("version") ? a.get_str("version")
                                       : inferred_version;
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
                       std::move(yaml_args), kv_seed);
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

// True if the user passed `--mode <X>` (or `--mode=X`) anywhere on
// the command line. The flag-style is the canonical entry point —
// every argument is a top-level flag and the action is selected by
// `--mode`. See `cmd_dispatch_flag_style`.
bool looks_like_mode_style(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--mode" || s.rfind("--mode=", 0) == 0) return true;
  }
  return false;
}

// Flag-style dispatcher: every option is a top-level flag, `--mode`
// picks the action. This is the canonical CLI shape; the kv-style
// (`mode=train task=detect ...`) survives as a drop-in for upstream
// tooling, and the legacy subcommand-style (`yolocpp train ...`)
// remains as a deprecated fallback for one release.
int cmd_dispatch_flag_style(int argc, char** argv) {
  CLI::App app{"yolocpp — pure C++ computer vision suite"};

  std::string mode;
  app.add_option("--mode", mode,
                  "train | predict | val | export | benchmark | info | download")
      ->required();

  std::string task = "detect";
  app.add_option("--task", task,
                  "detect | classify | segment | pose | obb (default: detect)");

  // Common knobs used by multiple modes. We accept the union here and
  // validate per-mode at dispatch time so the help text shows
  // everything in one place.
  std::string weights, source, out, data, names_csv, device, scale_s;
  std::string export_fmt, export_input_name = "images";
  std::string export_precision, export_after_train, dl_target, cache_dir = "build/bench_cache";
  int    imgsz = 640, epochs = 100, batch_size = 16, nc = 80;
  int    patience = 0, warmup = 10, iters = 100;
  double lr0   = 0.01;
  float  conf  = 0.25f, iou = 0.45f;
  bool   export_fp16 = true;
  uint64_t seed = 0;
  std::string save_dir = "runs/train";

  app.add_option("--model,-m,--weights", weights,
                  "weights `.pt` / `.trt` (alias: --weights)");
  app.add_option("--source,-s",  source,
                  "image, video, dir, glob, URL, or webcam index");
  app.add_option("--data,-d",    data,
                  "dataset root or data.yaml path");
  app.add_option("--out,-o",     out,         "output path / directory");
  app.add_option("--imgsz,-i",   imgsz,       "input image size (default 640)");
  app.add_option("--epochs,-e",  epochs,      "epochs (train)");
  app.add_option("--batch,-b",   batch_size,  "batch size (train)");
  app.add_option("--lr0",        lr0,         "initial LR (train)");
  app.add_option("--device,-D",  device,
                  "cpu | cuda | cuda:N | cuda:0,1,... | mps | auto");
  app.add_option("--scale",      scale_s,
                  "model scale letter (n/s/m/l/x; auto from filename)");
  app.add_option("--save",       save_dir,    "output directory under runs/");
  app.add_option("--names",      names_csv,   "comma-separated class names");
  app.add_option("--nc,-n",      nc,          "number of classes (default 80 = COCO)");
  app.add_option("--conf,-c",    conf,        "confidence threshold (default 0.25)");
  app.add_option("--iou",        iou,         "NMS IoU threshold (default 0.45)");
  app.add_option("--patience",   patience,
                  "stop if val mAP@0.5:0.95 doesn't improve for N epochs");
  app.add_option("--seed",       seed,
                  "deterministic-training seed (0 = non-deterministic)");
  app.add_option("--export-after-train", export_after_train,
                  "post-train, export best.pt as 'onnx', 'trt', or 'onnx,trt'");
  app.add_option("--format,-f",  export_fmt,
                  "export: onnx | trt");
  app.add_option("--precision,-p", export_precision,
                  "export: fp32 | fp16 | int8 | int4 | nvfp4 (only fp32/fp16 wired today)");
  app.add_option("--input-name", export_input_name,
                  "ONNX graph input tensor name");
  app.add_flag  ("--fp16,!--no-fp16", export_fp16,
                  "TRT FP16 (legacy alias for --precision=fp16/fp32)");
  app.add_option("--warmup",     warmup,      "benchmark warmup iters");
  app.add_option("--iters",      iters,       "benchmark timed iters");
  app.add_option("--cache",      cache_dir,   "TRT engine cache directory");
  app.add_option("--dataset",    dl_target,
                  "download mode: dataset short-name (coco8, VOC, ...) or .zip URL");

  CLI11_PARSE(app, argc, argv);
  device = normalise_device(device);

  // Mode dispatch. Per-mode validation lives here (single place to
  // read for "what does each mode need?") rather than scattered
  // across cmd_*().
  auto need = [](const std::string& mode_s, const std::string& field,
                  bool ok) {
    if (!ok) {
      std::cerr << "[error] --mode=" << mode_s << " needs --" << field << "\n";
    }
    return ok;
  };

  if (mode == "info") return cmd_info();

  if (mode == "predict") {
    if (!need(mode, "model",  !weights.empty())) return 2;
    if (!need(mode, "source", !source.empty()))  return 2;
    return cmd_predict_task(task, weights, source, out, imgsz, device,
                             scale_s, nc, conf, iou);
  }

  if (mode == "val") {
    if (!need(mode, "model", !weights.empty())) return 2;
    if (!need(mode, "data",  !data.empty()))    return 2;
    return cmd_val(weights, data, names_csv, imgsz, device, scale_s);
  }

  if (mode == "train") {
    if (!need(mode, "data", !data.empty())) return 2;
    // Pre-resolve scale + version_hint from init weights so post-train
    // export below sees the right shape (best.pt has no scale letter).
    std::string train_version_hint;
    if (!weights.empty()) {
      if (scale_s.empty()) {
        auto fs_scale = yolocpp::cli::scale_from_filename(weights);
        if (!fs_scale.empty()) scale_s = fs_scale;
      }
      train_version_hint = yolocpp::cli::version_from_filename(weights);
    }
    int rc = cmd_train(data, names_csv, imgsz, epochs, batch_size, lr0,
                        device, scale_s, save_dir, weights,
                        patience, /*args_for_yaml=*/{}, seed);
    if (rc != 0) return rc;
    if (!export_after_train.empty()) {
      std::filesystem::path src = std::filesystem::path(save_dir) / "best.pt";
      if (!std::filesystem::exists(src))
        src = std::filesystem::path(save_dir) / "last.pt";
      if (!std::filesystem::exists(src)) {
        std::cerr << "[warn] --export-after-train: no best.pt or last.pt under "
                  << save_dir << "; skipping export\n";
      } else {
        for (const auto& fmt : split_csv(export_after_train)) {
          if (fmt != "onnx" && fmt != "trt") {
            std::cerr << "[error] --export-after-train='" << fmt
                      << "' not recognised (expected: onnx, trt)\n";
            return 2;
          }
          std::filesystem::path out_path = src;
          out_path.replace_extension("." + fmt);
          int xrc = cmd_export(src.string(), fmt, out_path.string(),
                                imgsz, scale_s, nc,
                                /*input_name=*/"images",
                                /*fp16=*/true, train_version_hint);
          if (xrc != 0) return xrc;
        }
      }
    }
    return 0;
  }

  if (mode == "export") {
    if (!need(mode, "model",  !weights.empty()))   return 2;
    if (!need(mode, "format", !export_fmt.empty())) return 2;
    if (!export_precision.empty()) {
      if (export_precision == "fp32") export_fp16 = false;
      else if (export_precision == "fp16") export_fp16 = true;
      else if (export_precision == "int8" || export_precision == "int4" ||
               export_precision == "nvfp4") {
        std::cerr << "[error] --precision=" << export_precision
                  << " not yet wired (TODO #51F2)\n";
        return 2;
      } else {
        std::cerr << "[error] unknown --precision='" << export_precision
                  << "' (expected fp32 | fp16 | int8 | int4 | nvfp4)\n";
        return 2;
      }
    }
    return cmd_export(weights, export_fmt, out, imgsz, scale_s, nc,
                       export_input_name, export_fp16);
  }

  if (mode == "benchmark") {
    if (!need(mode, "model",  !weights.empty())) return 2;
    if (!need(mode, "source", !source.empty()))  return 2;
    return cmd_benchmark(weights, source, imgsz, warmup, iters, cache_dir, device);
  }

  if (mode == "download") {
    if (dl_target.empty()) {
      std::cerr << "[error] --mode=download needs --dataset=<name|url>\n";
      return 2;
    }
    auto path = yolocpp::cli::download_known_dataset(dl_target);
    std::cout << path << "\n";
    return 0;
  }

  std::cerr << "[error] --mode='" << mode
            << "' not recognised (expected: train | predict | val | export | "
               "benchmark | info | download)\n";
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
      "Usage (canonical: `--mode <action>` with flat top-level flags):\n"
      "  yolocpp --mode predict   -m yolo11s.pt -s bus.jpg [-D auto]\n"
      "  yolocpp --mode predict   -m yolo11s.pt -s images/    -o runs/predict/\n"
      "  yolocpp --mode predict   -m yolo11s.pt -s 'frames/*.jpg'\n"
      "  yolocpp --mode val       -m yolo11s.pt -d coco/data.yaml\n"
      "  yolocpp --mode train     -m yolo11s.pt -d coco/data.yaml -e 100 -b 16 \\\n"
      "                           --seed 42 --export-after-train onnx\n"
      "  yolocpp --mode export    -m yolo11s.pt -f onnx -p fp16\n"
      "  yolocpp --mode benchmark -m yolo11s.pt -s bus.jpg --warmup 10 --iters 100\n"
      "  yolocpp --mode download  --dataset coco8\n"
      "  yolocpp --mode info\n"
      "  yolocpp --version\n"
      "\n"
      "Usage (kv-style, drop-in for upstream tooling):\n"
      "  yolocpp task=detect mode=train  model=yolo11s.pt data=DATA epochs=100\n"
      "  yolocpp task=detect mode=predict model=yolo11s.pt source=IMG\n"
      "  yolocpp mode=export model=yolo11s.pt format=trt\n"
      "  yolocpp mode=benchmark model=yolo11s.pt source=IMG\n"
      "\n"
      "Tasks  : detect (default), classify, segment, pose, obb\n"
      "Modes  : train, val, predict, export, benchmark, info, download\n"
      "Formats: onnx, trt\n"
      "Precs  : fp32, fp16   (int8/int4/nvfp4 — TODO #51F2)\n"
      "Devices: cpu, cuda, cuda:N, cuda:0,1,..., mps, auto\n"
      "Source : single image / directory / glob (e.g. 'frames/*.jpg').\n"
      "         Video / RTSP / HTTP / webcam-index — frame loop is\n"
      "         tracked under #51C2 in TODO.md.\n"
      "\n"
      "The legacy subcommand-style (`yolocpp predict -m ...`) still works\n"
      "but is deprecated; pass `--mode predict` instead.\n";
    return 0;
  }

  try {
    // Routing precedence:
    //   1. kv-style                — drop-in for upstream tooling.
    //   2. flag-style (--mode=...) — canonical entry point as of #51J.
    //   3. legacy subcommand-style — kept as a deprecated fallback for
    //      one release; emits a one-line warning.
    if (looks_like_kv_style(argc, argv)) {
      auto a = yolocpp::cli::Args::parse(argc, argv);
      return dispatch_kv(a);
    }
    if (looks_like_mode_style(argc, argv)) {
      return cmd_dispatch_flag_style(argc, argv);
    }

    // Legacy subcommand style (CLI11). Deprecated — prefer
    // `yolocpp --mode <action> ...` (canonical) or
    // `yolocpp action=value ...` (kv-style, drop-in for upstream).
    std::cerr << "[deprecation] subcommand-style CLI is deprecated; "
                 "use `yolocpp --mode <action> ...` instead. "
                 "See `yolocpp --help`.\n";
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

    // Every subcommand below registers short + long flags. The
    // canonical name is `--model` (taking a `.pt` weights path); the
    // legacy `--weights` form is kept as a CLI11 alias on every
    // subcommand that accepts it. Short forms follow upstream
    // convention where unambiguous: -m model, -d data, -s source,
    // -D device, -e epochs, -b batch, -i imgsz, -o out, -c conf,
    // -n nc.
    auto* t = app.add_subcommand("train", "Train a YOLO model");
    t->add_option("--data,-d",    root, "dataset root (or data.yaml path)")->required();
    t->add_option("--names",      names_csv, "comma-separated class names");
    t->add_option("--imgsz,-i",   imgsz, "input image size (default 640)");
    t->add_option("--epochs,-e",  epochs, "number of epochs");
    t->add_option("--batch,-b",   batch_size, "batch size");
    t->add_option("--lr0",        lr0, "initial LR (default 0.01)");
    t->add_option("--device,-D",  device, "cpu | cuda | cuda:N | cuda:0,1,... | mps | auto");
    t->add_option("--scale",      scale_s, "model scale letter (n/s/m/l/x; auto from filename)");
    t->add_option("--save",       save_dir, "output directory under runs/");
    t->add_option("--model,-m,--weights", init_weights, "init weights `.pt` (alias: --weights)");
    int legacy_patience = 0;
    t->add_option("--patience",   legacy_patience,
                  "stop if val mAP@0.5:0.95 doesn't improve for N epochs");
    uint64_t cli_seed = 0;
    t->add_option("--seed",       cli_seed,
                  "deterministic-training seed (0 = non-deterministic)");
    std::string export_after_train;
    t->add_option("--export-after-train", export_after_train,
                  "post-train, export best.pt as 'onnx', 'trt', or 'onnx,trt'");

    auto* v = app.add_subcommand("val", "Validate (mAP)");
    v->add_option("--model,-m,--weights", weights, "weights `.pt` (alias: --weights)")->required();
    v->add_option("--data,-d",    root, "dataset root (or data.yaml path)")->required();
    v->add_option("--names",      names_csv, "comma-separated class names");
    v->add_option("--imgsz,-i",   imgsz, "input image size");
    v->add_option("--device,-D",  device, "cpu | cuda | cuda:N | mps | auto");
    v->add_option("--scale",      scale_s, "model scale letter (auto from filename)");

    auto* p = app.add_subcommand("predict", "Run inference");
    p->add_option("--model,-m,--weights", weights, "weights `.pt` / `.trt` (alias: --weights)")->required();
    p->add_option("--source,-s",  source, "image, video, dir, glob, URL, or webcam index")->required();
    p->add_option("--out,-o",     out, "output path (default: runs/predict/<stem>.jpg)");
    p->add_option("--imgsz,-i",   imgsz, "input image size");
    p->add_option("--device,-D",  device, "cpu | cuda | cuda:N | mps | auto");
    p->add_option("--scale",      scale_s, "model scale letter");
    p->add_option("--nc,-n",      nc, "number of classes (default 80 = COCO)");
    p->add_option("--conf,-c",    conf, "confidence threshold (default 0.25)");
    p->add_option("--iou",        iou, "NMS IoU threshold (default 0.45)");

    auto* e = app.add_subcommand("export", "Export to ONNX / TRT");
    std::string export_fmt, export_input_name = "images";
    bool   export_fp16 = true;
    std::string export_precision;
    e->add_option("--format,-f",  export_fmt, "onnx | trt")->required();
    e->add_option("--model,-m,--weights", weights, "weights `.pt` (alias: --weights)")->required();
    e->add_option("--out,-o",     out, "output path");
    e->add_option("--imgsz,-i",   imgsz, "input image size");
    e->add_option("--scale",      scale_s, "model scale letter");
    e->add_option("--nc,-n",      nc, "number of classes");
    e->add_option("--input-name", export_input_name, "ONNX graph input tensor name");
    e->add_flag  ("--fp16,!--no-fp16", export_fp16, "TRT FP16 (legacy alias for --precision=fp16/fp32)");
    e->add_option("--precision,-p", export_precision,
                  "fp32 | fp16 | int8 | int4 | nvfp4 (only fp32/fp16 wired today)");

    auto* b = app.add_subcommand("benchmark", "Latency / throughput benchmark");
    std::string cache = "build/bench_cache";
    int warmup = 10, iters = 100;
    b->add_option("--model,-m,--weights", weights, "weights `.pt` (alias: --weights)")->required();
    b->add_option("--source,-s",  source, "image / video / URL / webcam index")->required();
    b->add_option("--imgsz,-i",   imgsz, "input image size");
    b->add_option("--warmup",     warmup, "warmup iterations");
    b->add_option("--iters",      iters, "timed iterations");
    b->add_option("--cache",      cache, "TRT engine cache directory");
    b->add_option("--device,-D",  device, "cpu | cuda | cuda:N");

    auto* i = app.add_subcommand("info", "Build / device info");
    (void)i;

    // `yolocpp download <name|url>` — fetch a known dataset (coco8,
    // coco128, dota8, VOC, xView, …) into ./data/<name>/, or any
    // direct .zip URL into ./data/<stem>/. Prints the resolved
    // directory on success so it can be piped into `--data`.
    std::string dl_target;
    auto* dl = app.add_subcommand("download",
        "Download a known dataset (coco8, coco128, VOC, dota8, xView, ...) or a direct URL");
    dl->add_option("name", dl_target,
                    "dataset short-name or .zip URL")->required();

    CLI11_PARSE(app, argc, argv);

    // Validate / canonicalise --device once, before any subcommand
    // dispatch. Throws with a clear message on bad input (caught by
    // the outer try/catch and reported via stderr + exit 1).
    device = normalise_device(device);

    if (app.got_subcommand("info"))    return cmd_info();
    if (app.got_subcommand("predict")) {
      // Route through cmd_predict_task so the registry's per-version
      // `predict_to_file` hook handles non-v8 models. cmd_predict
      // (the v8-only Yolo8Detect Predictor path) is reachable as the
      // fallback inside cmd_predict_task.
      return cmd_predict_task(/*task=*/"detect", weights, source, out,
                              imgsz, device, scale_s, nc, conf, iou);
    }
    if (app.got_subcommand("val"))
      return cmd_val(weights, root, names_csv, imgsz, device, scale_s);
    if (app.got_subcommand("train")) {
      // Pre-resolve scale + version from init_weights filename so they
      // are available to BOTH cmd_train and the post-train auto-export
      // below. Without this, `--export-after-train=onnx` on a
      // `yolo11s.pt`-init run would see `scale_s=""` and `best.pt`'s
      // filename would resolve to `v8` (the default) — causing
      // shape-mismatch on load.
      std::string train_version_hint;
      if (!init_weights.empty()) {
        if (scale_s.empty()) {
          auto fs_scale = yolocpp::cli::scale_from_filename(init_weights);
          if (!fs_scale.empty()) scale_s = fs_scale;
        }
        train_version_hint = yolocpp::cli::version_from_filename(init_weights);
      }
      int rc = cmd_train(root, names_csv, imgsz, epochs, batch_size, lr0,
                          device, scale_s, save_dir, init_weights,
                          legacy_patience, /*args_for_yaml=*/{}, cli_seed);
      if (rc != 0) return rc;
      // Post-train auto-export: walks `--export-after-train`'s
      // comma-separated formats. Looks for `<save_dir>/best.pt`
      // first, falls back to `<save_dir>/last.pt`.
      if (!export_after_train.empty()) {
        std::filesystem::path src = std::filesystem::path(save_dir) / "best.pt";
        if (!std::filesystem::exists(src)) {
          src = std::filesystem::path(save_dir) / "last.pt";
        }
        if (!std::filesystem::exists(src)) {
          std::cerr << "[warn] --export-after-train: no best.pt or last.pt under "
                    << save_dir << "; skipping export\n";
        } else {
          for (const auto& fmt : split_csv(export_after_train)) {
            if (fmt != "onnx" && fmt != "trt") {
              std::cerr << "[error] --export-after-train='" << fmt
                        << "' not recognised (expected: onnx, trt)\n";
              return 2;
            }
            // Place the exported artefact next to the source `best.pt`
            // (e.g. `<save_dir>/best.onnx`, `<save_dir>/best.trt`). The
            // default `runs/export/<base>.onnx` location wouldn't tie the
            // exported file to the train run that produced it.
            std::filesystem::path out_path = src;
            out_path.replace_extension("." + fmt);
            int xrc = cmd_export(src.string(), fmt, out_path.string(),
                                  imgsz, scale_s, nc,
                                  /*input_name=*/"images",
                                  /*fp16=*/true, train_version_hint);
            if (xrc != 0) return xrc;
          }
        }
      }
      return 0;
    }
    if (app.got_subcommand("export")) {
      // --precision wins over --fp16 when both are given. fp32 and
      // fp16 are wired through the existing `TrtBuildConfig.fp16`
      // toggle; int8 / int4 / nvfp4 are not yet implemented and
      // surface a clear error pointing at the gap.
      if (!export_precision.empty()) {
        if (export_precision == "fp32") {
          export_fp16 = false;
        } else if (export_precision == "fp16") {
          export_fp16 = true;
        } else if (export_precision == "int8" ||
                   export_precision == "int4" ||
                   export_precision == "nvfp4") {
          std::cerr << "[error] --precision=" << export_precision
                    << " not yet wired (TODO #51F2: add INT8 calibration "
                       "+ INT4/NVFP4 routing through TrtBuildConfig)\n";
          return 2;
        } else {
          std::cerr << "[error] unknown --precision='" << export_precision
                    << "' (expected fp32 | fp16 | int8 | int4 | nvfp4)\n";
          return 2;
        }
      }
      return cmd_export(weights, export_fmt, out, imgsz, scale_s, nc,
                        export_input_name, export_fp16);
    }
    if (app.got_subcommand("benchmark"))
      return cmd_benchmark(weights, source, imgsz, warmup, iters, cache, device);
    if (app.got_subcommand("download")) {
      auto path = yolocpp::cli::download_known_dataset(dl_target);
      std::cout << path << "\n";
      return 0;
    }
  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 1;
  }
  return 1;
}
