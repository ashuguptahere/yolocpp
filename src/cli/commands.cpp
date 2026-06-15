// Implementation of the CLI command bodies — every `cmd_*`
// function declared in `include/yolocpp/cli/commands.hpp`,
// plus the helpers they share. Moved out of `src/cli/main.cpp`
// under #52 so the public chainable C++ API in `src/cli/api.cpp`
// can call into the same dispatch logic the CLI uses.
//
// Both `main.cpp` (the driver) and `api.cpp` (the public API)
// link against this TU through the core static library.

//
//   yolocpp --mode train     -m yolo11s.pt -d coco/data.yaml -e 100 -b 16
//   yolocpp --mode val       -m yolo11s.pt -d coco/data.yaml
//   yolocpp --mode predict   -m yolo11s.pt -s bus.jpg
//   yolocpp --mode export    -m yolo11s.pt -f onnx -p fp16
//   yolocpp --mode benchmark -m yolo11s.pt -s bus.jpg
//   yolocpp --mode download  --dataset coco8
//   yolocpp --mode info
//   yolocpp --version
//
// Removed under #51K (maintainer request): the kv-style parser
// (`task=detect mode=train ...`) and the legacy subcommand-style
// parser (`yolocpp train --data ...`). Flag-style is now the only
// entry point.

// CLI11 is consumed only by `cmd_dispatch_flag_style` which lives in
// main.cpp — keep it off the core lib's include path.
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <regex>
#include <fstream>
#include <memory>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <functional>

#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <yolocpp/config.hpp>

#include "yolocpp/registry/version_adapter.hpp"
#include "yolocpp/cli/data_yaml.hpp"
#include "yolocpp/cli/model_info.hpp"
#include "yolocpp/cli/resolve.hpp"
#include "yolocpp/core/device.hpp"
#include "yolocpp/core/version.hpp"
#include "yolocpp/datasets/coco_dataset.hpp"
#include "yolocpp/datasets/flat_dataset.hpp"
#include "yolocpp/datasets/voc_dataset.hpp"
#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/core/log.hpp"
#include "yolocpp/engine/benchmark.hpp"
#include "yolocpp/engine/trainer.hpp"
#include "yolocpp/engine/validator.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/inference/frame_predictor.hpp"
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
#include "yolocpp/inference/trt_task_eval.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/trt_export.hpp"
#include "yolocpp/tasks/classify_train.hpp"
#include "yolocpp/tasks/pose_obb_train.hpp"
#include "yolocpp/tasks/segment_train.hpp"

// All helpers + `cmd_*` bodies live in `yolocpp::cli`. The public C++
// API (`yolocpp::YOLO` in api.cpp) routes here through
// `include/yolocpp/cli/commands.hpp`. Previously this was an
// anonymous namespace — promoted under #52 so the API can reuse the
// same dispatch logic the CLI uses.
#include "yolocpp/cli/commands.hpp"

namespace yolocpp::cli {

using yolocpp::models::Yolo8Scale;

Yolo8Scale parse_scale(const std::string& s) {
  if (s == "n") return yolocpp::models::kYolo8n;
  if (s == "s") return yolocpp::models::kYolo8s;
  if (s == "m") return yolocpp::models::kYolo8m;
  if (s == "l") return yolocpp::models::kYolo8l;
  if (s == "x") return yolocpp::models::kYolo8x;
  // Deliberately do NOT default an empty/unknown scale to 'n': loading an x/l
  // checkpoint as 'n' builds the wrong architecture and silently mis-predicts
  // (the class of bug the load_from_state_dict backstop now catches loudly).
  // Fail with an actionable message instead.
  if (s.empty())
    throw std::runtime_error(
        "YOLO scale could not be inferred from the weights filename — pass "
        "--scale=n|s|m|l|x explicitly");
  throw std::runtime_error("unknown YOLO scale '" + s +
                           "' (expected n|s|m|l|x)");
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

// Pick a torch device honouring the CLI `--device` value (already normalised
// by `normalise_device`). Empty / "auto" / "cuda" → CUDA:0 when available else
// CPU; "cuda:N" / "cuda:N,M" → CUDA:N (the inference/val path honours the first
// index); "cpu" → CPU; anything else (e.g. "mps") → torch::Device(string).
torch::Device pick_device(const std::string& device) {
  if (device == "cpu") return torch::Device(torch::kCPU);
  if (device.empty() || device == "auto" || device == "cuda")
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                       : torch::Device(torch::kCPU);
  if (device.rfind("cuda:", 0) == 0) {
    if (!torch::cuda::is_available()) return torch::Device(torch::kCPU);
    return torch::Device(torch::kCUDA,
                         std::stoi(device.substr(5)));  // first index of N[,M…]
  }
  return torch::Device(device);  // mps, etc.
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
                       int imgsz, double map_50, double map_50_95,
                       const yolocpp::registry::VersionAdapter::ValResult* sml = nullptr) {
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
  if (sml) {
    f << "mAP@0.5:0.95 (small,  n_gt=" << sml->n_gt_small  << ") = "
      << sml->map_50_95_small  << "\n"
      << "mAP@0.5:0.95 (medium, n_gt=" << sml->n_gt_medium << ") = "
      << sml->map_50_95_medium << "\n"
      << "mAP@0.5:0.95 (large,  n_gt=" << sml->n_gt_large  << ") = "
      << sml->map_50_95_large  << "\n";
  }
  std::cout << "[val] wrote " << out_path << "\n";

  // Append a row to runs/val/validate.csv — the val analogue of the
  // trainer's per-epoch train.csv. Each `--mode val` run adds one row so
  // you can track how a model's val metrics change across checkpoints/runs.
  // Header is written once when the file is new.
  auto csv_path = std::string("runs/val/validate.csv");
  bool fresh = !std::filesystem::exists(csv_path);
  std::ofstream c(csv_path, std::ios::app);
  if (fresh)
    c << "timestamp,weights,data,imgsz,mAP50,mAP50-95,"
         "mAP50-95_small,mAP50-95_medium,mAP50-95_large\n";
  std::time_t now = std::time(nullptr);
  char ts[32] = {0};
  std::strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  c << ts << ',' << weights << ',' << data << ',' << imgsz << ','
    << map_50 << ',' << map_50_95 << ',';
  if (sml)
    c << sml->map_50_95_small << ',' << sml->map_50_95_medium << ','
      << sml->map_50_95_large << '\n';
  else
    c << ",,\n";
  std::cout << "[val] appended row to " << csv_path << "\n";
}

int cmd_predict(const std::string& weights, const std::string& source,
                const std::string& out, int imgsz, std::string device,
                std::string scale_s, int nc, float conf, float iou,
                std::vector<yolocpp::inference::Detection>* out_dets) {
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
    if (out_dets) *out_dets = std::move(dets);
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
  if (out_dets) *out_dets = std::move(dets);
  return 0;
}

int cmd_val(const std::string& weights, const std::string& root,
            const std::string& names_csv, int imgsz, std::string device,
            std::string scale_s) {
  auto names = split_csv(names_csv);
  if (names.empty()) names = yolocpp::inference::coco_names();
  int nc = (int)names.size();
  yolocpp::datasets::AugConfig aug; aug.augment = false;
  // Format-aware dispatch: `root` may be a YOLO-layout directory,
  // a Pascal VOC root, a `.csv`/`.tsv` flat file, a COCO `.json`,
  // or a `data.yaml` (#54B → CLI).
  auto ds = make_dataset(root, "val", imgsz, names, aug);
  auto torch_dev = pick_device(device);  // honours cuda:N (was hardcoded to 0)

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
              << "mAP@0.5:0.95 = " << r.map_50_95 << "\n"
              << "precision    = " << r.precision << "\n"
              << "recall       = " << r.recall    << "\n"
              << "F1           = " << r.f1         << "\n"
              << "  small  (n_gt=" << r.n_gt_small  << "): "
              << r.map_50_95_small  << "\n"
              << "  medium (n_gt=" << r.n_gt_medium << "): "
              << r.map_50_95_medium << "\n"
              << "  large  (n_gt=" << r.n_gt_large  << "): "
              << r.map_50_95_large  << "\n";
    write_val_results(weights, root, imgsz, r.map_50, r.map_50_95, &r);
    return 0;
  }

  yolocpp::inference::Predictor p(weights, imgsz, device, nc,
                                   parse_scale(scale_s));
  auto res = yolocpp::engine::validate(p.model(), ds, p.device());
  std::cout << "mAP@0.5      = " << res.map_50    << "\n"
            << "mAP@0.5:0.95 = " << res.map_50_95 << "\n"
            << "precision    = " << res.precision << "\n"
            << "recall       = " << res.recall    << "\n"
            << "F1           = " << res.f1         << "\n";
  write_val_results(weights, root, imgsz, res.map_50, res.map_50_95);
  return 0;
}

int cmd_train(const std::string& root, const std::string& names_csv,
              int imgsz, int epochs, int batch_size, double lr0,
              std::string device, std::string scale_s,
              const std::string& save_dir,
              const std::string& init_weights,
              int patience,
              std::vector<std::pair<std::string, std::string>> args_for_yaml,
              uint64_t seed,
              double lrf,
              const std::string& optimizer,
              int workers,
              bool cache_ram,
              bool deterministic,
              int close_mosaic) {
  auto names = split_csv(names_csv);
  // If --names wasn't passed, prefer the data.yaml's `names:` over the
  // COCO 80 fallback. Without this, training on a 5-class dataset
  // builds an nc=80 model (cls head retains all upstream channels) and
  // the val mAP collapses because the predictions span classes that
  // don't exist in the dataset.
  if (names.empty()) {
    namespace fs = std::filesystem;
    fs::path rp(root);
    std::string rext = rp.extension().string();
    for (auto& c : rext) c = (char)std::tolower((unsigned char)c);
    if (rext == ".yaml" || rext == ".yml") {
      try {
        auto dy = yolocpp::cli::parse_data_yaml(root);
        if (!dy.names.empty()) names = dy.names;
      } catch (const std::exception&) { /* fall through to COCO */ }
    }
  }
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

  // Adapter-driven imgsz fallback: when the CLI passed the default
  // 640 and the adapter has an opinion (e.g. v1=448, v4=608, v6-P6=
  // 1280), honour the adapter. Without this, training v1 with default
  // 640 fails at the FC layer because the backbone outputs 10×10
  // instead of 7×7. Mirrors `cmd_export`'s same imgsz dispatch.
  {
    std::string v_h = init_weights.empty()
        ? "" : yolocpp::cli::version_from_filename(init_weights);
    if (imgsz == 640 && !v_h.empty()) {
      yolocpp::registry::register_all_versions();
      if (const auto* adapter =
              yolocpp::registry::Registry::instance().find(v_h);
          adapter && adapter->default_imgsz) {
        int v = adapter->default_imgsz(scale_s, "detect");
        if (v > 0 && v != imgsz) {
          std::cerr << "[cmd_train] adapter default_imgsz=" << v
                    << " for version=" << v_h
                    << " — overriding the 640 default\n";
          imgsz = v;
        }
      }
    }
  }

  // Format-aware dispatch: `root` accepts every loader the
  // dispatcher knows (#54B → CLI). Augmentation defaults mirror
  // Ultralytics' detect-train defaults: mosaic_p=1.0 (4-in-1
  // composite per sample — the single biggest quality lever in 1-2
  // epoch comparisons), mixup off (0.0), HSV + flip kept at the
  // existing AugConfig defaults (h=0.015, s=0.7, v=0.4, flip=0.5).
  // AugConfig{} defaults stay mosaic_p=0.0 so other callers (e.g.
  // overfit smoke tests) aren't affected.
  yolocpp::datasets::AugConfig train_aug;
  train_aug.mosaic_p  = 1.0f;   // Ultralytics default
  train_aug.translate = 0.1f;   // Ultralytics detect default
  train_aug.scale_amp = 0.5f;   // Ultralytics detect default — scale in [0.5, 1.5]
  train_aug.cache_ram = cache_ram;
  // GPU augmentation: hsv_jitter + horizontal flip move from CPU
  // (per-sample) to GPU (per-batch tensor ops). Saves ~5 ms/batch of
  // worker CPU on small models where the data-prep pipeline is the
  // bottleneck. Enabled by default when CUDA is available; users
  // can disable via the env (CPU-only path stays bit-identical).
  train_aug.gpu_aug = torch::cuda::is_available();
  // degrees/shear stay 0 — also the Ultralytics default for detect.
  auto train_ds = make_dataset(root, "train", imgsz, names,
                                train_aug, seed);

  yolocpp::engine::TrainConfig cfg;
  cfg.epochs     = epochs;
  cfg.batch_size = batch_size;
  cfg.imgsz      = imgsz;
  cfg.lr0        = lr0;
  if (lrf >= 0.0) cfg.lrf = lrf;          // CLI override; -1 = trainer default
  cfg.device     = std::move(device);
  cfg.save_dir   = save_dir;
  cfg.patience   = patience;
  cfg.seed       = seed;
  cfg.optimizer  = optimizer;
  cfg.workers    = workers;
  cfg.deterministic = deterministic;
  cfg.close_mosaic  = close_mosaic;
  cfg.args_for_yaml = std::move(args_for_yaml);

  // Auto-attach val split for best.pt tracking. Try `make_dataset`
  // with split="val"; on any error (no val split in the source
  // format, ImageSets/val.txt missing, etc.) we silently skip the
  // val pass and just save last.pt at the end.
  try {
    yolocpp::datasets::AugConfig vaug; vaug.augment = false;
    auto val_loaded = make_dataset(root, "val", imgsz, names, vaug);
    cfg.val_dataset = std::make_shared<yolocpp::datasets::YoloDataset>(
        std::move(val_loaded));
    cfg.val_every = 1;
    std::cout << "[train] val split detected (" << cfg.val_dataset->size()
              << " imgs); will track best.pt by mAP@0.5:0.95\n";
  } catch (const std::exception&) {
    // No val split available — train without best-tracking. The
    // user can run `--mode val` separately on `last.pt`.
  }

  // Extract v_hint from the ORIGINAL init_weights spec (not init_eff
  // — even if the path doesn't resolve to a file, the bare name
  // `yolo1` / `yolo2-tiny` carries the version hint).
  std::string v_hint =
      init_weights.empty() ? "" : yolocpp::cli::version_from_filename(init_weights);

  // If the caller passed a bare version spec (e.g. `-m yolo1`,
  // `-m yolo2-tiny`) that the resolver couldn't map to a real file,
  // treat training as "from scratch": pass an empty string to the
  // adapter so it doesn't try to load_state_dict from a non-existent
  // path. The version + scale hints we've already extracted remain.
  std::string init_eff = init_weights;
  if (!init_eff.empty() && !std::filesystem::exists(init_eff)) {
    std::cerr << "[cmd_train] '" << init_eff
              << "' is not a file on disk — training from scratch "
              << "(version=" << v_hint << ", scale=" << scale_s << ")\n";
    init_eff.clear();
  }

  // Registry-driven dispatch: each non-v8 adapter wires
  // `run_train_detect` with its concrete holder type + matching
  // TrainerT<Holder>. v8 falls through to the explicit
  // `engine::Trainer = TrainerT<Yolo8Detect>` path below.
  yolocpp::registry::register_all_versions();
  if (const auto* adapter =
          yolocpp::registry::Registry::instance().find(v_hint);
      adapter && adapter->run_train_detect) {
    adapter->run_train_detect(init_eff, scale_s, nc,
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


// Validate a non-detect task. The detect path stays in `cmd_val` —
// it's registry-routed and supports every YOLO version. The
// classify/segment/pose/obb paths use the v8 task families
// (Yolo8Classify / Yolo8Segment / Yolo8Pose / Yolo8OBB) — those are
// the architectures whose task heads ship with weights upstream.
int cmd_val_task(const std::string& task, const std::string& weights,
                 const std::string& data, const std::string& names_csv,
                 int imgsz, const std::string& device,
                 const std::string& scale_s) {
  auto load = [&](auto& m) {
    if (!weights.empty()) {
      auto sd = yolocpp::serialization::load_state_dict(weights);
      m->load_from_state_dict(sd.entries);
    }
  };
  auto dev = pick_device(device);
  // Recover the backbone version + scale. A trained task checkpoint (best.pt /
  // last.pt) carries no version/scale token in its name, so probe the
  // state-dict architecture via infer_model_info; fall back to the filename.
  // --scale always overrides the scale. (Honours "never default scale; resolve"
  // — CLAUDE.md.)
  std::string version = "v8", scale = scale_s;
  if (!weights.empty()) {
    try {
      auto info = yolocpp::cli::infer_model_info(weights);
      version = info.version;
      if (scale.empty()) scale = info.scale;
    } catch (...) {
      version = yolocpp::cli::version_from_filename(weights);
      if (scale.empty()) scale = yolocpp::cli::scale_from_filename(weights);
    }
  }

  if (task == "classify") {
    int sz = (imgsz == 640) ? 224 : imgsz;
    yolocpp::tasks::ClassifyDataset ds(data, "val", sz, /*augment=*/false);
    auto run = [&](auto m) { load(m); return yolocpp::tasks::validate_classify(m, ds, dev); };
    yolocpp::tasks::ClassifyValResult r;
    if (version == "v12")      r = run(yolocpp::models::Yolo12Classify(yolocpp::models::yolo12_scale_from_letter(scale), ds.num_classes()));
    else if (version == "v11") r = run(yolocpp::models::Yolo11Classify(yolocpp::models::yolo11_scale_from_letter(scale), ds.num_classes()));
    else if (version == "v13") r = run(yolocpp::models::Yolo13Classify(yolocpp::models::yolo13_scale_from_letter(scale), ds.num_classes()));
    else                       r = run(yolocpp::models::Yolo8Classify(parse_scale(scale), ds.num_classes()));
    std::cout << "top1=" << r.top1_acc << " top5=" << r.top5_acc
              << " (n=" << r.n_total << ")\n";
    return 0;
  }
  if (task == "segment") {
    auto names = split_csv(names_csv);
    if (names.empty()) names = yolocpp::inference::coco_names();
    yolocpp::tasks::SegDataset ds(data, "val", imgsz, names, /*augment=*/false);
    auto run = [&](auto m) { load(m); return yolocpp::tasks::validate_segment(m, ds, dev); };
    yolocpp::tasks::SegValResult r;
    if (version == "v12")      r = run(yolocpp::models::Yolo12Segment(yolocpp::models::yolo12_scale_from_letter(scale), ds.num_classes()));
    else if (version == "v11") r = run(yolocpp::models::Yolo11Segment(yolocpp::models::yolo11_scale_from_letter(scale), ds.num_classes()));
    else if (version == "v13") r = run(yolocpp::models::Yolo13Segment(yolocpp::models::yolo13_scale_from_letter(scale), ds.num_classes()));
    else                       r = run(yolocpp::models::Yolo8Segment(parse_scale(scale), ds.num_classes()));
    std::cout << "mask mAP@0.5=" << r.map_50
              << " (pred=" << r.n_predictions
              << " gt=" << r.n_ground_truths << ")\n";
    return 0;
  }
  if (task == "pose") {
    yolocpp::tasks::PoseDataset ds(data, "val", imgsz, /*num_kpts=*/17,
                                    /*kpt_dim=*/3, /*augment=*/false);
    auto run = [&](auto m) { load(m); return yolocpp::tasks::validate_pose(m, ds, dev); };
    yolocpp::tasks::PoseValResult r;
    if (version == "v12")      r = run(yolocpp::models::Yolo12Pose(yolocpp::models::yolo12_scale_from_letter(scale), 1, 17, 3));
    else if (version == "v11") r = run(yolocpp::models::Yolo11Pose(yolocpp::models::yolo11_scale_from_letter(scale), 1, 17, 3));
    else if (version == "v13") r = run(yolocpp::models::Yolo13Pose(yolocpp::models::yolo13_scale_from_letter(scale), 1, 17, 3));
    else                       r = run(yolocpp::models::Yolo8Pose(parse_scale(scale), 1, 17, 3));
    std::cout << "OKS mAP@0.5=" << r.oks_map_50 << "\n";
    return 0;
  }
  if (task == "obb") {
    int sz = (imgsz == 640) ? 1024 : imgsz;  // OBB/DOTA default 1024 (cf. predict)
    auto names = split_csv(names_csv);
    if (names.empty()) names = yolocpp::inference::dota_names();
    yolocpp::tasks::OBBDataset ds(data, "val", sz, names, /*augment=*/false);
    auto run = [&](auto m) { load(m); return yolocpp::tasks::validate_obb(m, ds, dev); };
    yolocpp::tasks::OBBValResult r;
    if (version == "v12")      r = run(yolocpp::models::Yolo12OBB(yolocpp::models::yolo12_scale_from_letter(scale), ds.num_classes(), 1));
    else if (version == "v11") r = run(yolocpp::models::Yolo11OBB(yolocpp::models::yolo11_scale_from_letter(scale), ds.num_classes(), 1));
    else if (version == "v13") r = run(yolocpp::models::Yolo13OBB(yolocpp::models::yolo13_scale_from_letter(scale), ds.num_classes(), 1));
    else                       r = run(yolocpp::models::Yolo8OBB(parse_scale(scale), ds.num_classes(), 1));
    std::cout << "rotated mAP@0.5=" << r.map_50 << "\n";
    return 0;
  }
  std::cerr << "[error] cmd_val_task: unknown task '" << task << "'\n";
  return 2;
}

// Benchmark a non-detect task (classify/segment/pose/obb) — #5. The detect
// benchmark stays in `cmd_benchmark` (registry-routed, multi-format PT + TRT
// fp32/fp16/int8 + ONNX). Non-detect TRT/ONNX would need task-specific output
// decode in `TrtPredictor` (masks / keypoints / rotated boxes) — a separate
// extension — so this path is PT-only: it times the forward pass (synced via a
// host copy, same as the detect bench) and, when `--data` is given, reports the
// task's accuracy metric via the existing `validate_*` functions. Uses the v8
// task families (the architectures whose task heads ship with weights upstream).
namespace {

// One row of the per-format benchmark table.
struct FmtRow {
  std::string name;
  double      size_mb   = 0.0;
  double      ms        = -1.0;
  double      metric    = -1.0;
  bool        has_metric = false;
};

// Result of building + timing one non-detect task TRT engine.
struct TaskTrtRunner {
  yolocpp::inference::TrtMultiForward fwd;
  double ms        = -1.0;
  double size_mb   = 0.0;
  int    eng_imgsz = 0;
  bool   ok        = false;
};

// Build a non-detect task TRT engine at `precision` ("fp16"/"int8") via the
// registry's task-aware ONNX export + build_trt_engine, then time the forward.
// Best-effort: any failure (e.g. INT8 needs a calibration image dir that isn't
// present) returns {.ok=false} so the benchmark degrades gracefully.
TaskTrtRunner build_task_trt_runner(
    const yolocpp::registry::VersionAdapter* adapter,
    const std::string& weights, const std::string& scale_s, int nc,
    const std::string& task, int sz, const std::string& precision,
    const std::string& calib_dir, torch::Device dev, int warmup, int iters) {
  namespace fs = std::filesystem;
  TaskTrtRunner r;
  try {
    fs::create_directories("runs/export");
    const std::string base     = "runs/export/bench_" + task + "_" + precision;
    const std::string onnx_tmp = base + ".tmp.onnx";
    const std::string engine   = base + ".trt";

    yolocpp::serialization::OnnxExportConfig ocfg;
    ocfg.imgsz = sz;
    adapter->export_onnx(weights, scale_s, nc, task, onnx_tmp, ocfg);

    yolocpp::serialization::TrtBuildConfig tcfg;
    tcfg.imgsz = sz;
    tcfg.input_name = ocfg.input_name;
    if (precision == "int8") {
      if (calib_dir.empty() || !fs::is_directory(calib_dir)) {
        std::error_code ec; fs::remove(onnx_tmp, ec);
        return r;  // no calibration images → skip INT8 gracefully
      }
      tcfg.int8 = true;  tcfg.fp16 = false;
      tcfg.calib_image_dir = calib_dir;
      tcfg.calib_cache     = engine + ".calib";
    } else {
      tcfg.fp16 = true;
    }
    if (adapter->trt_disable_tf32) tcfg.tf32 = false;
    yolocpp::serialization::build_trt_engine(onnx_tmp, engine, tcfg);
    { std::error_code ec; fs::remove(onnx_tmp, ec); }

    int eng_imgsz = sz;
    r.fwd = yolocpp::inference::make_trt_multi_forward(engine, eng_imgsz);
    r.eng_imgsz = eng_imgsz;
    { std::error_code ec; auto n = fs::file_size(engine, ec);
      if (!ec) r.size_mb = static_cast<double>(n) / 1e6; }

    // Median ms over `iters`, matching the PT timing harness above.
    auto x = torch::zeros({1, 3, eng_imgsz, eng_imgsz},
        torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    for (int i = 0; i < warmup; ++i) r.fwd(x);
    std::vector<double> ts; ts.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      r.fwd(x);
      auto t1 = std::chrono::steady_clock::now();
      ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    r.ms = ts[ts.size() / 2];
    r.ok = true;
  } catch (const std::exception& e) {
    LOG_WARN("bench") << "TRT " << precision << " for " << task
                      << " unavailable: " << e.what();
    r.ok = false;
  }
  return r;
}

}  // namespace

int cmd_benchmark_task(const std::string& task, const std::string& weights,
                       const std::string& data, const std::string& names_csv,
                       int imgsz, const std::string& device,
                       const std::string& scale_s, int warmup, int iters) {
  namespace fs = std::filesystem;
  auto dev = pick_device(device);
  // Auto-resolve scale from the weights filename when --scale isn't given
  // (yolov8n-seg.pt → "n"), mirroring cmd_export / the detect benchmark.
  std::string scale = scale_s.empty()
                          ? yolocpp::cli::scale_from_filename(weights)
                          : scale_s;
  if (scale.empty()) scale = "n";
  auto load = [&](auto& m) {
    if (!weights.empty()) {
      auto sd = yolocpp::serialization::load_state_dict(weights);
      int n = m->load_from_state_dict(sd.entries);
      LOG_INFO("bench") << "loaded " << n << " weights from " << weights;
    }
  };
  double size_mb = 0.0;
  { std::error_code ec; auto n = fs::file_size(weights, ec);
    if (!ec) size_mb = static_cast<double>(n) / 1e6; }
  const bool have_map = !data.empty();
  if (warmup <= 0) warmup = 5;
  if (iters  <= 0) iters  = 50;

  // Resolve a `.yaml` --data to its root directory — the task datasets want a
  // directory (same as the task trainers/validators).
  std::string data_root = data;
  if (have_map) {
    std::string ext = fs::path(data).extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower((unsigned char)c));
    if (ext == ".yaml" || ext == ".yml") {
      try { data_root = yolocpp::cli::resolve_dataset(data); }
      catch (const std::exception& e) { LOG_ERROR("bench") << e.what(); return 2; }
    }
  }

  // Median ms over `iters` forwards; each run_once() ends with a host copy
  // (.cpu()) so GPU work is synchronised before the timer stops — matching the
  // detect bench (whose predict()'s final .cpu() in NMS provides the barrier).
  auto time_fwd = [&](const std::function<void()>& run_once) -> double {
    for (int i = 0; i < warmup; ++i) run_once();
    std::vector<double> ts; ts.reserve(iters);
    for (int i = 0; i < iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      run_once();
      auto t1 = std::chrono::steady_clock::now();
      ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
  };
  auto input = [&](int s) {
    return torch::zeros({1, 3, s, s},
        torch::TensorOptions().dtype(torch::kFloat32).device(dev));
  };

  int    sz = imgsz;
  double ms = -1.0, metric = -1.0;
  std::string metric_name = "metric";
  // Extra TRT format rows (segment/pose/obb) appended below the PyTorch row.
  std::vector<FmtRow> trt_rows;

  if (task == "classify") {
    sz = (imgsz == 640) ? 224 : imgsz;
    int nc = 1000;  // ImageNet default when no --data
    std::unique_ptr<yolocpp::tasks::ClassifyDataset> ds;
    if (have_map) {
      ds = std::make_unique<yolocpp::tasks::ClassifyDataset>(data_root, "val", sz, false);
      nc = ds->num_classes();
    }
    yolocpp::models::Yolo8Classify m(parse_scale(scale), nc);
    load(m); m->to(dev); m->eval();
    auto x = input(sz);
    ms = time_fwd([&]{ c10::InferenceMode g; m->forward(x).cpu(); });
    metric_name = "top1-acc";
    if (have_map) metric = yolocpp::tasks::validate_classify(m, *ds, dev).top1_acc;
    if (dev.is_cuda()) {
      yolocpp::registry::register_all_versions();
      const auto* adapter = yolocpp::registry::Registry::instance().find("v8");
      // Classify val images live in per-class subdirs, not a flat dir the INT8
      // calibrator can read — empty calib ⇒ only the fp16 row builds.
      for (const char* prec : {"fp16"}) {
        auto rr = build_task_trt_runner(adapter, weights, scale, nc, "classify",
                                        sz, prec, /*calib=*/"", dev, warmup, iters);
        if (!rr.ok) continue;
        double mm = -1.0;
        if (have_map) {
          yolocpp::inference::TrtClassifyModel tm{rr.fwd};
          mm = yolocpp::tasks::validate_classify_t(tm, *ds, dev).top1_acc;
        }
        trt_rows.push_back({std::string("TRT-") + prec, rr.size_mb, rr.ms, mm,
                            have_map});
      }
    }
  } else if (task == "segment") {
    auto names = split_csv(names_csv);
    if (names.empty()) names = yolocpp::inference::coco_names();
    int nc = static_cast<int>(names.size());
    std::unique_ptr<yolocpp::tasks::SegDataset> ds;
    if (have_map) {
      ds = std::make_unique<yolocpp::tasks::SegDataset>(data_root, "val", imgsz, names, false);
      nc = ds->num_classes();
    }
    yolocpp::models::Yolo8Segment m(parse_scale(scale), nc);
    load(m); m->to(dev); m->eval();
    auto x = input(sz);
    ms = time_fwd([&]{ c10::InferenceMode g;
                       auto [d, c, p] = m->forward_eval(x); (void)c; (void)p; d.cpu(); });
    metric_name = "mask-mAP@0.5";
    if (have_map) metric = yolocpp::tasks::validate_segment(m, *ds, dev).map_50;
    if (dev.is_cuda()) {
      yolocpp::registry::register_all_versions();
      const auto* adapter = yolocpp::registry::Registry::instance().find("v8");
      const std::string calib = data_root + "/images/val";
      for (const char* prec : {"fp16", "int8"}) {
        auto rr = build_task_trt_runner(adapter, weights, scale, nc, "segment",
                                        sz, prec, calib, dev, warmup, iters);
        if (!rr.ok) continue;
        double mm = -1.0;
        if (have_map) {
          yolocpp::inference::TrtSegModel tm{rr.fwd};
          mm = yolocpp::tasks::validate_segment_t(tm, *ds, dev).map_50;
        }
        trt_rows.push_back({std::string("TRT-") + prec, rr.size_mb, rr.ms, mm,
                            have_map});
      }
    }
  } else if (task == "pose") {
    std::unique_ptr<yolocpp::tasks::PoseDataset> ds;
    if (have_map)
      ds = std::make_unique<yolocpp::tasks::PoseDataset>(data_root, "val", imgsz, 17, 3, false);
    yolocpp::models::Yolo8Pose m(parse_scale(scale), /*nc=*/1, 17, 3);
    load(m); m->to(dev); m->eval();
    auto x = input(sz);
    ms = time_fwd([&]{ c10::InferenceMode g;
                       auto [d, k] = m->forward_eval(x); (void)k; d.cpu(); });
    metric_name = "OKS-mAP@0.5";
    if (have_map) metric = yolocpp::tasks::validate_pose(m, *ds, dev).oks_map_50;
    if (dev.is_cuda()) {
      yolocpp::registry::register_all_versions();
      const auto* adapter = yolocpp::registry::Registry::instance().find("v8");
      const std::string calib = data_root + "/images/val";
      for (const char* prec : {"fp16", "int8"}) {
        auto rr = build_task_trt_runner(adapter, weights, scale, /*nc=*/1, "pose",
                                        sz, prec, calib, dev, warmup, iters);
        if (!rr.ok) continue;
        double mm = -1.0;
        if (have_map) {
          yolocpp::inference::TrtPoseModel tm{rr.fwd};
          mm = yolocpp::tasks::validate_pose_t(tm, *ds, dev).oks_map_50;
        }
        trt_rows.push_back({std::string("TRT-") + prec, rr.size_mb, rr.ms, mm,
                            have_map});
      }
    }
  } else if (task == "obb") {
    sz = (imgsz == 640) ? 1024 : imgsz;  // OBB/DOTA default 1024
    auto names = split_csv(names_csv);
    if (names.empty()) names = yolocpp::inference::dota_names();
    int nc = static_cast<int>(names.size());
    std::unique_ptr<yolocpp::tasks::OBBDataset> ds;
    if (have_map) {
      ds = std::make_unique<yolocpp::tasks::OBBDataset>(data_root, "val", sz, names, false);
      nc = ds->num_classes();
    }
    yolocpp::models::Yolo8OBB m(parse_scale(scale), nc, /*ne=*/1);
    load(m); m->to(dev); m->eval();
    auto x = input(sz);
    ms = time_fwd([&]{ c10::InferenceMode g;
                       auto [d, a] = m->forward_eval(x); (void)a; d.cpu(); });
    metric_name = "rotated-mAP@0.5";
    if (have_map) metric = yolocpp::tasks::validate_obb(m, *ds, dev).map_50;
    if (dev.is_cuda()) {
      yolocpp::registry::register_all_versions();
      const auto* adapter = yolocpp::registry::Registry::instance().find("v8");
      const std::string calib = data_root + "/images/val";
      for (const char* prec : {"fp16", "int8"}) {
        auto rr = build_task_trt_runner(adapter, weights, scale, nc, "obb",
                                        sz, prec, calib, dev, warmup, iters);
        if (!rr.ok) continue;
        double mm = -1.0;
        if (have_map) {
          yolocpp::inference::TrtOBBModel tm{rr.fwd};
          mm = yolocpp::tasks::validate_obb_t(tm, *ds, dev).map_50;
        }
        trt_rows.push_back({std::string("TRT-") + prec, rr.size_mb, rr.ms, mm,
                            have_map});
      }
    }
  } else {
    LOG_ERROR("bench") << "cmd_benchmark_task: unknown task '" << task << "'";
    return 2;
  }

  // Assemble the format table: PyTorch first, then any TRT rows that built.
  std::vector<FmtRow> rows;
  rows.push_back({"PyTorch", size_mb, ms, metric, have_map});
  for (const auto& t : trt_rows) rows.push_back(t);

  std::cout << "\n  Task: " << task << "   model: " << weights
            << "   scale: " << scale
            << "   imgsz: " << sz
            << "   device: " << (dev.is_cuda() ? "cuda" : "cpu") << "\n"
            << "  Format     Size(MB)   ms/im     img/s    " << metric_name << "\n"
            << "  ────────   ────────   ──────    ──────   ────────────\n";
  for (const auto& r : rows) {
    const double imgps = r.ms > 0 ? 1000.0 / r.ms : 0.0;
    std::cout << std::fixed << std::setprecision(2)
              << "  " << std::left << std::setw(9) << r.name << std::right
              << std::setw(8) << r.size_mb
              << "   " << std::setw(6) << r.ms
              << "   " << std::setw(6) << imgps << "   ";
    if (r.has_metric) std::cout << std::setprecision(4) << r.metric << "\n";
    else              std::cout << "—  (pass --data for " << metric_name << ")\n";
  }
  if (trt_rows.empty() && dev.is_cuda())
    std::cout << "  [note] no TRT rows — engine build/calibration unavailable "
                 "for this task (ONNX runtime is gated on #70).\n";
  return 0;
}

// Train a non-detect task. Same v8-family caveat as `cmd_val_task`.
int cmd_train_task(const std::string& task, const std::string& data,
                   const std::string& names_csv, int imgsz, int epochs,
                   int batch, double lr0, const std::string& device,
                   const std::string& scale_s, const std::string& save_dir,
                   const std::string& weights) {
  // Resolve .yaml input to a dataset-root directory so the task-
  // specific dataset constructors (Seg/Pose/OBB/Classify) — which
  // expect a directory, not a yaml file — get the path they need.
  // Previously cmd_train_task passed `data` straight through and the
  // dataset ctor tried to open `<data.yaml-path>/images/train`.
  std::string data_root = data;
  {
    namespace fs = std::filesystem;
    fs::path p(data);
    std::string ext = p.extension().string();
    for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
    if (ext == ".yaml" || ext == ".yml") {
      try {
        data_root = yolocpp::cli::resolve_dataset(data);
      } catch (const std::exception& e) {
        std::cerr << "[error] cmd_train_task: " << e.what() << "\n";
        return 2;
      }
    }
  }

  // Load init weights only when the spec is a real file. A bare scratch spec
  // (e.g. `-m yolo12n-seg`) reaches here as a non-file string carrying just the
  // version + scale hint — train from random init in that case.
  auto load = [&](auto& m) {
    if (!weights.empty() && std::filesystem::exists(weights)) {
      auto sd = yolocpp::serialization::load_state_dict(weights);
      m->load_from_state_dict(sd.entries);
    }
  };
  // Auto-resolve scale from the init-weights filename when --scale isn't given
  // (mirrors detect train / val / export). From-scratch training with neither
  // --scale nor init weights still errors clearly at parse_scale.
  std::string scale = (scale_s.empty() && !weights.empty())
                          ? yolocpp::cli::scale_from_filename(weights)
                          : scale_s;
  // Task-backbone version (v8 / v11 / v12) from the init/weights spec filename —
  // selects which task-family architecture to build. v8 is the default (and the
  // fallback for any version without task heads).
  const std::string version =
      weights.empty() ? "v8" : yolocpp::cli::version_from_filename(weights);

  if (task == "classify") {
    int sz = (imgsz == 640) ? 224 : imgsz;
    yolocpp::tasks::ClassifyDataset tr(data_root, "train", sz, /*augment=*/true);
    yolocpp::tasks::ClassifyTrainConfig cfg;
    cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = sz;
    cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
    auto run = [&](auto m) {
      load(m); yolocpp::tasks::train_classify(std::move(m), tr, nullptr, cfg);
    };
    if (version == "v12")
      run(yolocpp::models::Yolo12Classify(
          yolocpp::models::yolo12_scale_from_letter(scale), tr.num_classes()));
    else if (version == "v11")
      run(yolocpp::models::Yolo11Classify(
          yolocpp::models::yolo11_scale_from_letter(scale), tr.num_classes()));
    else if (version == "v13")
      run(yolocpp::models::Yolo13Classify(
          yolocpp::models::yolo13_scale_from_letter(scale), tr.num_classes()));
    else
      run(yolocpp::models::Yolo8Classify(parse_scale(scale), tr.num_classes()));
    return 0;
  }
  if (task == "segment") {
    auto names = split_csv(names_csv);
    if (names.empty()) names = yolocpp::inference::coco_names();
    yolocpp::tasks::SegDataset tr(data_root, "train", imgsz, names, /*augment=*/true);
    yolocpp::tasks::SegTrainConfig cfg;
    cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
    cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
    auto run = [&](auto m) {
      load(m); yolocpp::tasks::train_segment(std::move(m), tr, nullptr, cfg);
    };
    if (version == "v12")
      run(yolocpp::models::Yolo12Segment(
          yolocpp::models::yolo12_scale_from_letter(scale), tr.num_classes()));
    else if (version == "v11")
      run(yolocpp::models::Yolo11Segment(
          yolocpp::models::yolo11_scale_from_letter(scale), tr.num_classes()));
    else if (version == "v13")
      run(yolocpp::models::Yolo13Segment(
          yolocpp::models::yolo13_scale_from_letter(scale), tr.num_classes()));
    else
      run(yolocpp::models::Yolo8Segment(parse_scale(scale), tr.num_classes()));
    return 0;
  }
  if (task == "pose") {
    yolocpp::tasks::PoseDataset tr(data_root, "train", imgsz, 17, 3, /*augment=*/true);
    yolocpp::tasks::PoseTrainConfig cfg;
    cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
    cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
    auto run = [&](auto m) {
      load(m); yolocpp::tasks::train_pose(std::move(m), tr, nullptr, cfg);
    };
    if (version == "v12")
      run(yolocpp::models::Yolo12Pose(
          yolocpp::models::yolo12_scale_from_letter(scale), /*nc=*/1, 17, 3));
    else if (version == "v11")
      run(yolocpp::models::Yolo11Pose(
          yolocpp::models::yolo11_scale_from_letter(scale), /*nc=*/1, 17, 3));
    else if (version == "v13")
      run(yolocpp::models::Yolo13Pose(
          yolocpp::models::yolo13_scale_from_letter(scale), /*nc=*/1, 17, 3));
    else
      run(yolocpp::models::Yolo8Pose(parse_scale(scale), /*nc=*/1, 17, 3));
    return 0;
  }
  if (task == "obb") {
    auto names = split_csv(names_csv);
    if (names.empty()) names = yolocpp::inference::dota_names();
    yolocpp::tasks::OBBDataset tr(data_root, "train", imgsz, names, /*augment=*/true);
    yolocpp::tasks::OBBTrainConfig cfg;
    cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
    cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
    auto run = [&](auto m) {
      load(m); yolocpp::tasks::train_obb(std::move(m), tr, nullptr, cfg);
    };
    if (version == "v12")
      run(yolocpp::models::Yolo12OBB(
          yolocpp::models::yolo12_scale_from_letter(scale), tr.num_classes(), 1));
    else if (version == "v11")
      run(yolocpp::models::Yolo11OBB(
          yolocpp::models::yolo11_scale_from_letter(scale), tr.num_classes(), 1));
    else if (version == "v13")
      run(yolocpp::models::Yolo13OBB(
          yolocpp::models::yolo13_scale_from_letter(scale), tr.num_classes(), 1));
    else
      run(yolocpp::models::Yolo8OBB(parse_scale(scale), tr.num_classes(), 1));
    return 0;
  }
  std::cerr << "[error] cmd_train_task: unknown task '" << task << "'\n";
  return 2;
}

int cmd_export(const std::string& weights, const std::string& format,
               const std::string& out, int imgsz, const std::string& scale_s_in,
               int nc, const std::string& input_name, bool fp16,
               const std::string& version_hint,
               const std::string& task,
               bool int8,
               const std::string& int8_calib_dir,
               const std::string& int8_calib_cache) {
  // Determine version: explicit hint (from CLI) or filename inference.
  std::string version = version_hint.empty()
                            ? yolocpp::cli::version_from_filename(weights)
                            : version_hint;
  // If the resolver let a bare version spec through (e.g. `-m yolo1`
  // when no .pt exists), fall through with empty weights so the
  // adapter constructs a fresh random-init model. The exported ONNX
  // is then graph-correct but its weights are uninitialised.
  std::string weights_eff = weights;
  if (!weights_eff.empty() && !std::filesystem::exists(weights_eff)) {
    std::cerr << "[cmd_export] '" << weights_eff
              << "' is not a file on disk — exporting from random init "
              << "(version=" << version << ")\n";
    weights_eff.clear();
  }
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
    adapter->export_onnx(weights_eff, scale_s, nc, task, onnx_path, ocfg);
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
    tcfg.imgsz = imgsz; tcfg.input_name = input_name;
    // INT8 PTQ (#51F2): INT8-only build (FP16 off, matching the benchmark
    // path + Ultralytics' engine.py), calibrated on the user-supplied image
    // directory. The calibration table caches to "<out>.calib" by default so
    // a re-export at the same shape skips re-sampling.
    if (int8) {
      tcfg.int8 = true;
      tcfg.fp16 = false;
      tcfg.calib_image_dir = int8_calib_dir;
      tcfg.calib_cache     = int8_calib_cache.empty() ? (path + ".calib")
                                                      : int8_calib_cache;
    } else {
      tcfg.fp16 = fp16;
    }
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

namespace {

// Trainable params (millions) from the checkpoint: sum tensor numel, skipping
// BN running stats / counters. Approximate but version-agnostic. -1 on failure.
double bench_params_m(const std::string& weights) {
  try {
    auto sd = yolocpp::serialization::load_state_dict(weights);
    long long n = 0;
    for (const auto& [k, t] : sd.entries) {
      if (k.find("running_mean") != std::string::npos ||
          k.find("running_var")  != std::string::npos ||
          k.find("num_batches")  != std::string::npos) continue;
      n += t.numel();
    }
    return static_cast<double>(n) / 1e6;
  } catch (...) { return -1.0; }
}

// One reference mAP (50, 50:95) for a model on `data`. Uses the registry
// adapter's run_val where available, else the v8 Predictor+validate path —
// the same dispatch cmd_val uses. {-1,-1} on failure.
std::pair<double, double> bench_map(const std::string& weights, const std::string& data,
                                    int imgsz, const std::string& device,
                                    std::string scale_s, const std::string& names_csv) {
  try {
    auto names = split_csv(names_csv);
    if (names.empty()) names = yolocpp::inference::coco_names();
    int nc = static_cast<int>(names.size());
    yolocpp::datasets::AugConfig aug; aug.augment = false;
    auto ds = make_dataset(data, "val", imgsz, names, aug);
    auto dev = pick_device(device);  // honours cuda:N (was hardcoded to 0)
    if (scale_s.empty()) scale_s = yolocpp::cli::scale_from_filename(weights);
    auto vh = yolocpp::cli::version_from_filename(weights);
    yolocpp::registry::register_all_versions();
    if (const auto* a = yolocpp::registry::Registry::instance().find(vh); a && a->run_val) {
      auto r = a->run_val(weights, scale_s, nc, ds, dev);
      return {r.map_50, r.map_50_95};
    }
    yolocpp::inference::Predictor pr(weights, imgsz, device, nc, parse_scale(scale_s));
    auto res = yolocpp::engine::validate(pr.model(), ds, pr.device());
    return {res.map_50, res.map_50_95};
  } catch (const std::exception& e) {
    LOG_WARN("bench") << "mAP unavailable for " << weights << ": " << e.what();
    return {-1.0, -1.0};
  }
}

struct ModelBench {
  std::string name;
  double params_m = -1, map50 = -1, map95 = -1;
  std::vector<yolocpp::engine::BenchResult> formats;
};

void print_model_formats(const ModelBench& r, bool have_map) {
  std::cout << "\n\033[1m" << r.name << "\033[0m";
  if (r.params_m >= 0) { char b[48]; std::snprintf(b, sizeof b, "  %.1fM params", r.params_m); std::cout << b; }
  std::cout << "\n  Format                Size(MB)   ms/im      img/s    dets";
  if (have_map) std::cout << "    mAP50  mAP50-95";
  std::cout << "\n  ────────────────────  ────────  ────────  ────────  ─────";
  if (have_map) std::cout << "    ─────  ────────";
  std::cout << "\n";
  for (const auto& f : r.formats) {
    // ms/im is per-image (throughput-normalized) so a batched TRT row is
    // comparable to single-image PT, not a per-call number.
    char line[320];
    int off;
    if (f.throughput_imgps > 0) {
      double ms_im = 1000.0 / f.throughput_imgps;
      off = std::snprintf(line, sizeof line, "  %-20s  %8.1f  %8.3f  %8.1f  %5d",
                          f.backend.c_str(), f.size_mb, ms_im, f.throughput_imgps,
                          f.num_detections);
    } else {  // not timed (e.g. cv::dnn couldn't load the ONNX graph)
      off = std::snprintf(line, sizeof line, "  %-20s  %8.1f  %8s  %8s  %5s",
                          f.backend.c_str(), f.size_mb, "-", "-", "-");
    }
    if (have_map) {
      if (f.map_50 >= 0) off += std::snprintf(line + off, sizeof line - off,
                                              "    %.3f     %.3f", f.map_50, f.map_50_95);
      else off += std::snprintf(line + off, sizeof line - off, "        -         -");
    }
    std::snprintf(line + off, sizeof line - off, "\n");
    std::cout << line;
  }
}

void print_leaderboard(const std::vector<ModelBench>& reports, bool have_map) {
  std::cout << "\n\033[1m=== Leaderboard (" << reports.size() << " models) ===\033[0m\n";
  std::cout << "  model         params(M)";
  if (have_map) std::cout << "   mAP50  mAP50-95";
  std::cout << "   PT ms/im   best ms/im   best img/s\n";
  std::cout << "  ────────────  ─────────";
  if (have_map) std::cout << "   ─────  ────────";
  std::cout << "   ────────   ──────────   ──────────\n";
  for (const auto& r : reports) {
    double pt_ms = 0, best_ips = 0;
    for (const auto& f : r.formats) {
      double ms_im = f.throughput_imgps > 0 ? 1000.0 / f.throughput_imgps : f.median_ms;
      if (f.backend.rfind("PT", 0) == 0) pt_ms = ms_im;
      best_ips = std::max(best_ips, f.throughput_imgps);
    }
    double best_ms = best_ips > 0 ? 1000.0 / best_ips : 0;
    char line[320];
    int off = std::snprintf(line, sizeof line, "  %-12s  %9.2f", r.name.c_str(), r.params_m);
    if (have_map)
      off += std::snprintf(line + off, sizeof line - off, "   %.3f   %.3f", r.map50, r.map95);
    std::snprintf(line + off, sizeof line - off, "   %8.2f   %10.2f   %10.1f\n",
                  pt_ms, best_ms, best_ips);
    std::cout << line;
  }
}

}  // namespace

int cmd_benchmark(const std::string& weights, const std::string& source,
                  int imgsz, int warmup, int iters,
                  const std::string& cache, const std::string& device,
                  int batch_size, const std::string& precision_csv,
                  const std::string& int8_calib_dir,
                  const std::string& int8_calib_cache,
                  const std::string& data, const std::string& names_csv,
                  const std::string& scale_cli) {
  // Parse format/precision CSV: fp32|fp16|int8 → TRT precisions, onnx → ONNX.
  bool fp32 = false, fp16 = false, int8 = false, onnx = false;
  for (const auto& tok : split_csv(precision_csv)) {
    if      (tok == "fp32") fp32 = true;
    else if (tok == "fp16") fp16 = true;
    else if (tok == "int8") int8 = true;
    else if (tok == "onnx") onnx = true;
    else if (!tok.empty()) LOG_WARN("bench") << "unknown format '" << tok
                                             << "' (expected fp32|fp16|int8|onnx)";
  }

  // `weights` may be a comma-separated list — benchmark each model.
  auto models = split_csv(weights);
  if (models.empty()) { LOG_ERROR("bench") << "no model given"; return 2; }
  const bool have_map = !data.empty();

  // Build the eval dataset once (shared across models). PT mAP comes from the
  // registry validator; TRT + ONNX mAP are measured per format inside
  // run_benchmark via cfg.eval_ds.
  std::unique_ptr<yolocpp::datasets::YoloDataset> eval_ds;
  int eval_nc = 80;
  if (have_map) {
    try {
      auto names = split_csv(names_csv);
      if (names.empty()) names = yolocpp::inference::coco_names();
      eval_nc = static_cast<int>(names.size());
      yolocpp::datasets::AugConfig aug; aug.augment = false;
      eval_ds = std::make_unique<yolocpp::datasets::YoloDataset>(
          make_dataset(data, "val", imgsz, names, aug));
      LOG_INFO("bench") << "per-format mAP on " << data << " (" << eval_ds->size()
                        << " val images, nc=" << eval_nc << ")";
    } catch (const std::exception& e) {
      LOG_WARN("bench") << "mAP disabled — dataset load failed: " << e.what();
    }
  }
  const bool map_on = have_map && eval_ds;

  std::vector<ModelBench> reports;
  for (const auto& spec : models) {
    std::string w;
    try { w = resolve_weights(spec); }
    catch (const std::exception& e) { LOG_ERROR("bench") << "skip " << spec << ": " << e.what(); continue; }

    yolocpp::engine::BenchConfig cfg;
    cfg.weights = w; cfg.source = source; cfg.imgsz = imgsz;
    cfg.warmup_iters = warmup; cfg.iters = iters; cfg.cache_dir = cache;
    cfg.device = device; cfg.batch_size = batch_size; cfg.scale = scale_cli;
    cfg.int8_calib_dir = int8_calib_dir; cfg.int8_calib_cache = int8_calib_cache;
    cfg.run_trt_fp32 = fp32; cfg.run_trt_fp16 = fp16; cfg.run_trt_int8 = int8;
    cfg.run_onnx = onnx;
    if (map_on) { cfg.eval_ds = eval_ds.get(); cfg.nc_eval = eval_nc; }

    ModelBench rep;
    rep.name = std::filesystem::path(spec).stem().string();
    try { rep.formats = yolocpp::engine::run_benchmark(cfg); }
    catch (const std::exception& e) { LOG_ERROR("bench") << "skip " << spec << ": " << e.what(); continue; }
    rep.params_m = bench_params_m(w);

    // PT mAP via the registry validator (TRT/ONNX rows were scored in
    // run_benchmark). Stamp it onto the PT format row + the model summary.
    if (map_on) {
      std::tie(rep.map50, rep.map95) = bench_map(w, data, imgsz, device, scale_cli, names_csv);
      for (auto& f : rep.formats)
        if (f.backend.rfind("PT", 0) == 0) { f.map_50 = rep.map50; f.map_50_95 = rep.map95; }
    }

    print_model_formats(rep, map_on);
    reports.push_back(std::move(rep));
  }

  if (reports.size() > 1) print_leaderboard(reports, map_on);
  std::cout << "\n";
  return reports.empty() ? 1 : 0;
}

// ── New-style (kv) parser dispatcher ─────────────────────────────────────

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
// the CLI exit code (0 = ok, 2 = unknown version). When `out_dets`
// is non-null, it's populated with the predicted detections (#52A2).
int predict_one_image(const std::string& task, const std::string& weights,
                      const std::string& source, const std::string& out,
                      int imgsz, std::string device, std::string scale_s,
                      int nc, float conf, float iou,
                      const std::string& version_hint,
                      std::vector<yolocpp::inference::Detection>* out_dets);

// Frame loop for a non-detect task over video / URL / webcam. `predict`
// runs one frame → task result; `draw` annotates the frame in place. Mirrors
// the detect video path (lazy VideoWriter, webcam frame cap, runtime-I/O exit
// codes) but is generic over the task predictor + result type. (#51C2 follow-up.)
template <class PredictFn, class DrawFn>
int run_task_video(PredictFn predict, DrawFn draw, const std::string& source,
                   SourceKind kind, std::string out, const std::string& task,
                   const std::string& tag) {
  cv::VideoCapture cap;
  if (kind == SourceKind::Webcam) cap.open(std::stoi(source));
  else                            cap.open(source);
  if (!cap.isOpened()) {
    std::cerr << "[error] could not open source: " << source << "\n";
    return 1;  // runtime I/O failure (#51I2)
  }
  std::string out_path;
  if (!out.empty()) {
    out_path = out;
  } else {
    std::filesystem::create_directories("runs/predict");
    auto stem = (kind == SourceKind::Webcam)
                    ? ("webcam" + source)
                    : std::filesystem::path(source).stem().string();
    if (stem.empty()) stem = "out";
    out_path = "runs/predict/" + stem + "_" + task + ".mp4";
  }
  cv::VideoWriter writer;
  int n_frames = 0;
  cv::Mat frame;
  while (cap.read(frame) && !frame.empty()) {
    auto res = predict(frame);
    draw(frame, res);
    if (!writer.isOpened()) {
      double fps = cap.get(cv::CAP_PROP_FPS);
      if (fps <= 1.0 || std::isnan(fps)) fps = 25.0;  // webcams often report 0
      int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
      if (!writer.open(out_path, fourcc, fps, frame.size())) {
        std::cerr << "[error] could not open output writer: " << out_path
                  << "\n";
        return 1;
      }
    }
    writer.write(frame);
    ++n_frames;
    if (kind == SourceKind::Webcam && n_frames >= 600) break;  // disk guard
  }
  std::cout << "[" << task << "] (" << tag << "/video) " << n_frames
            << " frames, wrote " << out_path << "\n";
  return 0;
}

int cmd_predict_task(const std::string& task, const std::string& weights,
                     const std::string& source, std::string out, int imgsz,
                     std::string device, std::string scale_s, int nc,
                     float conf, float iou,
                     const std::string& version_hint,
                     std::vector<yolocpp::inference::Detection>* out_dets,
                     std::vector<std::pair<std::string,
                         std::vector<yolocpp::inference::Detection>>>*
                         out_dets_per_image) {
  // Auto-resolve scale from the weights filename when the caller
  // didn't pass --scale. The registry's per-version `predict_to_file`
  // hooks need a scale letter (`yolo11s.pt` → "s") to construct the
  // right holder; without this the legacy CLI11 subcommand path errors
  // with "unknown scale letter ''" on every non-v8 model.
  if (scale_s.empty()) {
    auto fs_scale = yolocpp::cli::scale_from_filename(weights);
    if (!fs_scale.empty()) scale_s = fs_scale;
  }

  // Classify the source. Image / Dir / Glob fan out to a per-image
  // loop. Video / URL / Webcam open `cv::VideoCapture`, run frames
  // through a long-lived `FramePredictor`, and write annotated frames
  // via `cv::VideoWriter` (closed in #51C2).
  auto kind = classify_source(source);
  if (kind == SourceKind::Video || kind == SourceKind::Url ||
      kind == SourceKind::Webcam) {
    if (task != "detect") {
      // Non-detect tasks: resolve version (filename token → state-dict
      // architecture for versionless best.pt/last.pt), construct the matching
      // per-version task predictor, and run the shared frame loop with the
      // task's draw helper.
      namespace inf = yolocpp::inference;
      namespace mdl = yolocpp::models;
      auto tver = version_hint.empty()
                      ? yolocpp::cli::version_from_filename(weights)
                      : version_hint;
      if (tver == "v8" && !weights.empty()) {
        try { tver = yolocpp::cli::infer_model_info(weights).version; }
        catch (...) {}
      }
      const bool v11 = tver == "v11", v12 = tver == "v12",
                 v13 = tver == "v13", v26 = tver == "v26";
      inf::NMSConfig nm; nm.conf_thresh = conf; nm.iou_thresh = iou;

      if (task == "segment") {
        auto draw = [&](cv::Mat& f, const std::vector<inf::SegInstance>& v) {
          inf::draw_segments(f, v);
        };
        auto run = [&](auto p) {
          return run_task_video([&](cv::Mat& f) { return p.predict(f, nm); },
                                draw, source, kind, out, "segment", tver);
        };
        if (v11) return run(inf::Yolo11SegmentPredictor(weights, imgsz, device, nc, mdl::yolo11_scale_from_letter(scale_s)));
        if (v12) return run(inf::Yolo12SegmentPredictor(weights, imgsz, device, nc, mdl::yolo12_scale_from_letter(scale_s)));
        if (v13) return run(inf::Yolo13SegmentPredictor(weights, imgsz, device, nc, mdl::yolo13_scale_from_letter(scale_s)));
        if (v26) return run(inf::Yolo26SegmentPredictor(weights, imgsz, device, nc, mdl::yolo26_scale_from_letter(scale_s)));
        return run(inf::SegmentPredictor(weights, imgsz, device, nc, parse_scale(scale_s)));
      }
      if (task == "pose") {
        auto draw = [&](cv::Mat& f, const std::vector<inf::PoseInstance>& v) {
          inf::draw_poses(f, v);
        };
        auto run = [&](auto p) {
          return run_task_video([&](cv::Mat& f) { return p.predict(f, nm); },
                                draw, source, kind, out, "pose", tver);
        };
        if (v11) return run(inf::Yolo11PosePredictor(weights, imgsz, device, 17, 3, mdl::yolo11_scale_from_letter(scale_s)));
        if (v12) return run(inf::Yolo12PosePredictor(weights, imgsz, device, 17, 3, mdl::yolo12_scale_from_letter(scale_s)));
        if (v13) return run(inf::Yolo13PosePredictor(weights, imgsz, device, 17, 3, mdl::yolo13_scale_from_letter(scale_s)));
        if (v26) return run(inf::Yolo26PosePredictor(weights, imgsz, device, 17, 3, mdl::yolo26_scale_from_letter(scale_s)));
        return run(inf::PosePredictor(weights, imgsz, device, 17, 3, parse_scale(scale_s)));
      }
      if (task == "obb") {
        int sz = (imgsz == 640) ? 1024 : imgsz;          // OBB default 1024
        int obb_nc = (nc < 0 || nc == 80) ? 15 : nc;     // DOTA default 15
        auto draw = [&](cv::Mat& f, const std::vector<inf::OBBInstance>& v) {
          inf::draw_obbs(f, v);
        };
        auto run = [&](auto p) {
          return run_task_video([&](cv::Mat& f) { return p.predict(f, nm); },
                                draw, source, kind, out, "obb", tver);
        };
        if (v11) return run(inf::Yolo11OBBPredictor(weights, sz, device, obb_nc, mdl::yolo11_scale_from_letter(scale_s)));
        if (v12) return run(inf::Yolo12OBBPredictor(weights, sz, device, obb_nc, mdl::yolo12_scale_from_letter(scale_s)));
        if (v13) return run(inf::Yolo13OBBPredictor(weights, sz, device, obb_nc, mdl::yolo13_scale_from_letter(scale_s)));
        if (v26) return run(inf::Yolo26OBBPredictor(weights, sz, device, obb_nc, mdl::yolo26_scale_from_letter(scale_s)));
        return run(inf::OBBPredictor(weights, sz, device, obb_nc, parse_scale(scale_s)));
      }
      if (task == "classify") {
        int sz = (imgsz == 640) ? 224 : imgsz;           // classify default 224
        int cls_nc = (nc < 0 || nc == 80) ? 1000 : nc;   // ImageNet default
        auto draw = [&](cv::Mat& f, const inf::ClassifyResult& r) {
          inf::draw_classify(f, r);
        };
        auto run = [&](auto p) {
          return run_task_video([&](cv::Mat& f) { return p.predict(f, 5); },
                                draw, source, kind, out, "classify", tver);
        };
        if (v11) return run(inf::Yolo11ClassifyPredictor(weights, sz, device, cls_nc, mdl::yolo11_scale_from_letter(scale_s)));
        if (v12) return run(inf::Yolo12ClassifyPredictor(weights, sz, device, cls_nc, mdl::yolo12_scale_from_letter(scale_s)));
        if (v13) return run(inf::Yolo13ClassifyPredictor(weights, sz, device, cls_nc, mdl::yolo13_scale_from_letter(scale_s)));
        if (v26) return run(inf::Yolo26ClassifyPredictor(weights, sz, device, cls_nc, mdl::yolo26_scale_from_letter(scale_s)));
        return run(inf::ClassifyPredictor(weights, sz, device, cls_nc, parse_scale(scale_s)));
      }
      std::cerr << "[error] --task=" << task << " unknown\n";
      return 2;
    }
    auto version = version_hint.empty()
                       ? yolocpp::cli::version_from_filename(weights)
                       : version_hint;
    yolocpp::registry::register_all_versions();
    const auto* adapter =
        yolocpp::registry::Registry::instance().find(version);
    std::unique_ptr<yolocpp::inference::FramePredictor> pred;
    if (adapter && adapter->make_frame_predictor) {
      pred = adapter->make_frame_predictor(weights, scale_s, nc, imgsz, device);
    } else {
      // v8 fallback path: wrap the unified Predictor so we can call
      // predict(cv::Mat) per-frame without re-loading weights.
      class V8FramePredictor : public yolocpp::inference::FramePredictor {
       public:
        V8FramePredictor(const std::string& w, int sz, std::string dev,
                          int nc, std::string scale)
            : p_(w, sz, std::move(dev), nc, parse_scale(scale)) {}
        std::vector<yolocpp::inference::Detection>
        predict(const cv::Mat& f, yolocpp::inference::NMSConfig nm) override {
          return p_.predict(f, nm);
        }
       private:
        yolocpp::inference::Predictor p_;
      };
      pred = std::make_unique<V8FramePredictor>(weights, imgsz, device,
                                                  nc, scale_s);
    }

    cv::VideoCapture cap;
    if (kind == SourceKind::Webcam) cap.open(std::stoi(source));
    else                            cap.open(source);
    if (!cap.isOpened()) {
      // Runtime I/O failure (the spec classified fine but the device/stream
      // won't open) → exit 1, not a user-input error. (#51I2)
      std::cerr << "[error] could not open source: " << source << "\n";
      return 1;
    }

    // Default output: runs/predict/<stem>.mp4. Webcam stems into
    // `webcam<idx>` since `<source>` is just an integer.
    std::string out_path;
    if (!out.empty()) {
      out_path = out;
    } else {
      std::filesystem::create_directories("runs/predict");
      auto stem = (kind == SourceKind::Webcam)
                      ? ("webcam" + source)
                      : std::filesystem::path(source).stem().string();
      if (stem.empty()) stem = "out";
      out_path = "runs/predict/" + stem + ".mp4";
    }

    cv::VideoWriter writer;
    yolocpp::inference::NMSConfig nm;
    nm.conf_thresh = conf;
    nm.iou_thresh  = iou;
    int n_frames = 0;
    std::size_t total_dets = 0;
    cv::Mat frame;
    while (cap.read(frame) && !frame.empty()) {
      auto dets = pred->predict(frame, nm);
      yolocpp::inference::draw_detections(frame, dets);
      if (!writer.isOpened()) {
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 1.0 || std::isnan(fps)) fps = 25.0;  // webcams often report 0
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        if (!writer.open(out_path, fourcc, fps, frame.size())) {
          // Runtime I/O failure (codec/permission) → exit 1. (#51I2)
          std::cerr << "[error] could not open output writer: "
                    << out_path << "\n";
          return 1;
        }
      }
      writer.write(frame);
      ++n_frames;
      total_dets += dets.size();
      // Webcams have no natural EOF — cap at a generous frame budget
      // so a forgotten Ctrl-C doesn't fill the disk. Override with
      // an explicit `--iters` (unused today; future #51C3).
      if (kind == SourceKind::Webcam && n_frames >= 600) break;
    }
    std::cout << "[predict] (" << version << "/video) " << n_frames
              << " frames, " << total_dets << " total dets, wrote "
              << out_path << "\n";
    return 0;
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
    // #52A3: when a per-image collector is supplied, capture this image's
    // dets into a local sink and key them by input path; otherwise fall back
    // to the single `out_dets` (which holds the LAST image for multi-input).
    std::vector<yolocpp::inference::Detection> this_dets;
    std::vector<yolocpp::inference::Detection>* sink =
        out_dets_per_image ? &this_dets : out_dets;
    int sub_rc = predict_one_image(task, weights, in, this_out, imgsz,
                                    device, scale_s, nc, conf, iou,
                                    version_hint, sink);
    if (out_dets_per_image)
      out_dets_per_image->emplace_back(in, std::move(this_dets));
    if (sub_rc != 0) rc = sub_rc;  // last non-zero wins, but keep going
  }
  return rc;
}

int predict_one_image(const std::string& task, const std::string& weights,
                      const std::string& source, const std::string& out,
                      int imgsz, std::string device, std::string scale_s,
                      int nc, float conf, float iou,
                      const std::string& version_hint,
                      std::vector<yolocpp::inference::Detection>* out_dets) {
  yolocpp::inference::NMSConfig c;
  c.conf_thresh = conf; c.iou_thresh = iou;

  if (task == "detect") {
    // .trt engines are version-agnostic — the serialized graph already
    // bakes in the architecture, so any task=detect TRT engine routes
    // through the TrtPredictor regardless of which YOLO version produced it.
    if (weights.size() >= 4 &&
        weights.substr(weights.size() - 4) == ".trt") {
      return cmd_predict(weights, source, out, imgsz, device, scale_s, nc,
                          conf, iou, out_dets);
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
      auto dets = adapter->predict_to_file(weights, source, out, imgsz,
                                            device, scale_s, nc, nm);
      std::cout << "[predict] (" << version << ") " << dets.size()
                << " detections, wrote " << out << "\n";
      if (out_dets) *out_dets = std::move(dets);
      return 0;
    }

    static const std::vector<std::string> kKnown = {
        "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v26"};
    if (std::find(kKnown.begin(), kKnown.end(), version) == kKnown.end()) {
      std::cerr << "[error] unrecognised YOLO version '" << version
                << "' — supported set: yolo3..yolo13, yolo26\n";
      return 2;
    }
    // v8 (and any anchor-free model whose state-dict shape is v8-shape)
    // has no dedicated `predict_v<N>_to_file` — fall back to the
    // unified Predictor.
    return cmd_predict(weights, source, out, imgsz, device, scale_s, nc,
                        conf, iou, out_dets);
  }
  // For task variants we route on version_hint (or filename inference) too —
  // v11 / v26 task models have a different module layout and can't load
  // through the v8 task predictor classes.
  auto task_version = version_hint.empty()
                          ? yolocpp::cli::version_from_filename(weights)
                          : version_hint;
  // Versionless trained checkpoints (best.pt / last.pt) carry no version token
  // (version_from_filename defaults to v8) — recover from the state-dict
  // architecture so a trained v12/v13 task model predicts with the right class.
  if (task_version == "v8" && !weights.empty()) {
    try { task_version = yolocpp::cli::infer_model_info(weights).version; }
    catch (...) {}
  }
  bool is_v11 = (task_version == "v11");
  bool is_v26 = (task_version == "v26");
  bool is_v12 = (task_version == "v12");
  bool is_v13 = (task_version == "v13");

  if (task == "classify") {
    int sz = (imgsz == 640) ? 224 : imgsz;  // classify default 224
    // Honour --nc; fall back to the ImageNet 1000 default when --nc wasn't set
    // (the global default is the detect 80). Was hardcoded to 1000, ignoring --nc.
    int cls_nc = (nc < 0 || nc == 80) ? 1000 : nc;
    if (is_v26) {
      yolocpp::inference::Yolo26ClassifyPredictor p(
          weights, sz, device, /*nc=*/cls_nc,
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
          weights, sz, device, /*nc=*/cls_nc,
          yolocpp::models::yolo11_scale_from_letter(scale_s));
      cv::Mat img = cv::imread(source, cv::IMREAD_COLOR);
      if (img.empty()) throw std::runtime_error("could not read " + source);
      auto r = p.predict(img, /*top_k=*/5);
      std::cout << "[classify] (v11) top-5:\n";
      for (auto& [cid, prob] : r.topk)
        std::cout << "  " << cid << "  " << prob << "\n";
      return 0;
    }
    if (is_v12 || is_v13) {
      cv::Mat img = cv::imread(source, cv::IMREAD_COLOR);
      if (img.empty()) throw std::runtime_error("could not read " + source);
      yolocpp::inference::ClassifyResult r;
      if (is_v12) r = yolocpp::inference::Yolo12ClassifyPredictor(
          weights, sz, device, cls_nc,
          yolocpp::models::yolo12_scale_from_letter(scale_s)).predict(img, 5);
      else        r = yolocpp::inference::Yolo13ClassifyPredictor(
          weights, sz, device, cls_nc,
          yolocpp::models::yolo13_scale_from_letter(scale_s)).predict(img, 5);
      std::cout << "[classify] (" << task_version << ") top-5:\n";
      for (auto& [cid, prob] : r.topk) std::cout << "  " << cid << "  " << prob << "\n";
      return 0;
    }
    yolocpp::inference::ClassifyPredictor p(weights, sz, device, /*nc=*/cls_nc,
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
    if (is_v12 || is_v13) {
      auto insts = is_v12
          ? yolocpp::inference::Yolo12SegmentPredictor(weights, imgsz, device, nc,
                yolocpp::models::yolo12_scale_from_letter(scale_s)).predict_to_file(source, out, c)
          : yolocpp::inference::Yolo13SegmentPredictor(weights, imgsz, device, nc,
                yolocpp::models::yolo13_scale_from_letter(scale_s)).predict_to_file(source, out, c);
      std::cout << "[segment] (" << task_version << ") " << insts.size()
                << " instances, wrote " << out << "\n";
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
    if (is_v12 || is_v13) {
      auto insts = is_v12
          ? yolocpp::inference::Yolo12PosePredictor(weights, imgsz, device, 17, 3,
                yolocpp::models::yolo12_scale_from_letter(scale_s)).predict_to_file(source, out, c)
          : yolocpp::inference::Yolo13PosePredictor(weights, imgsz, device, 17, 3,
                yolocpp::models::yolo13_scale_from_letter(scale_s)).predict_to_file(source, out, c);
      std::cout << "[pose] (" << task_version << ") " << insts.size()
                << " people, wrote " << out << "\n";
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
    // Honour --nc; fall back to the DOTA 15 default when --nc wasn't set.
    int obb_nc = (nc < 0 || nc == 80) ? 15 : nc;
    if (is_v26) {
      yolocpp::inference::Yolo26OBBPredictor p(
          weights, sz, device, /*nc=*/obb_nc,
          yolocpp::models::yolo26_scale_from_letter(scale_s));
      auto insts = p.predict_to_file(source, out, c);
      std::cout << "[obb] (v26) " << insts.size() << " rotated boxes, wrote " << out << "\n";
      return 0;
    }
    if (is_v11) {
      yolocpp::inference::Yolo11OBBPredictor p(
          weights, sz, device, /*nc=*/obb_nc,
          yolocpp::models::yolo11_scale_from_letter(scale_s));
      auto insts = p.predict_to_file(source, out, c);
      std::cout << "[obb] (v11) " << insts.size() << " rotated boxes, wrote " << out << "\n";
      return 0;
    }
    if (is_v12 || is_v13) {
      auto insts = is_v12
          ? yolocpp::inference::Yolo12OBBPredictor(weights, sz, device, obb_nc,
                yolocpp::models::yolo12_scale_from_letter(scale_s)).predict_to_file(source, out, c)
          : yolocpp::inference::Yolo13OBBPredictor(weights, sz, device, obb_nc,
                yolocpp::models::yolo13_scale_from_letter(scale_s)).predict_to_file(source, out, c);
      std::cout << "[obb] (" << task_version << ") " << insts.size()
                << " rotated boxes, wrote " << out << "\n";
      return 0;
    }
    yolocpp::inference::OBBPredictor p(weights, sz, device, /*nc=*/obb_nc,
                                        parse_scale(scale_s));
    auto insts = p.predict_to_file(source, out, c);
    std::cout << "[obb] " << insts.size() << " rotated boxes, wrote " << out << "\n";
    return 0;
  }
  std::cerr << "[error] unknown task: " << task << "\n";
  return 2;
}


// Flag-style dispatcher: every option is a top-level flag, `--mode`
// picks the action. This is the only CLI parser as of #51K — the
// kv-style and legacy subcommand-style parsers were removed by the
// maintainer's request.
// `cmd_dispatch_flag_style` lives in `src/cli/main.cpp` — it's the
// only CLI11 consumer in the codebase, and pulling CLI11 into the
// core static library would balloon yolocpp_core's include surface.

// ─── Dataset format dispatcher (#54B → CLI) ──────────────────────────────
// Auto-detects which loader to use from the `--data` value and
// funnels everything through `YoloDataset`'s pre-loaded ctor so the
// trainer + validator can stay typed on a single dataset class. The
// detection rules are intentionally simple — extension/sniff first,
// existence check second; ambiguous specs error.
datasets::YoloDataset make_dataset(
    const std::string& spec, const std::string& split, int imgsz,
    const std::vector<std::string>& names,
    const datasets::AugConfig& aug, std::uint64_t seed) {
  namespace fs = std::filesystem;

  auto lower_ext = [](const fs::path& p) {
    auto s = p.extension().string();
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
  };

  // (1) data.yaml — resolve through the existing dataset resolver
  // (which honours `path:` / `train:` / `val:` / `download:` /
  // `names:`) and recurse on the resolved root directory.
  if (auto ext = lower_ext(spec); ext == ".yaml" || ext == ".yml") {
    auto root = yolocpp::cli::resolve_dataset(spec);
    return make_dataset(root, split, imgsz, names, aug, seed);
  }

  // (2) Flat CSV / TSV.
  if (auto ext = lower_ext(spec); ext == ".csv" || ext == ".tsv") {
    datasets::FlatDataset f(spec, split, imgsz, names, aug, seed);
    // Funnel through YoloDataset's pre-loaded ctor.
    return datasets::YoloDataset(f.paths(), f.labels(), imgsz, names, aug);
  }

  // (3) COCO JSON.
  if (auto ext = lower_ext(spec); ext == ".json") {
    // images_dir defaults to <json's parent>; users can override by
    // sym-linking the json next to the images. (#54B2 will accept a
    // sibling `coco_images_dir:` key in data.yaml.)
    datasets::CocoDataset c(spec, /*images_dir=*/"", imgsz, aug);
    auto coco_names = c.names();  // dense [0,N) names from the JSON itself
    return datasets::YoloDataset(c.paths(), c.labels(), imgsz,
                                  coco_names, aug);
  }

  // (4) Pascal VOC layout (directory containing
  //     JPEGImages/ + Annotations/ + ImageSets/Main/<split>.txt).
  if (fs::is_directory(spec)) {
    fs::path root(spec);
    if (fs::is_directory(root / "JPEGImages") &&
        fs::is_directory(root / "Annotations") &&
        fs::is_directory(root / "ImageSets" / "Main")) {
      // names default to canonical 20 unless caller passed something.
      auto voc_names = names.empty() ? datasets::voc_default_names() : names;
      datasets::VocDataset v(spec, split, imgsz, voc_names, aug);
      return datasets::YoloDataset(v.paths(), v.labels(), imgsz,
                                    voc_names, aug);
    }
    // (5) YOLO-format directory layout — existing path. Pass through
    // to the original ctor.
    return datasets::YoloDataset(spec, split, imgsz, names, aug);
  }

  throw std::runtime_error(
      "make_dataset: '" + spec +
      "' isn't recognised — expected a directory (YOLO / VOC layout), "
      "a .csv/.tsv (flat format), a .json (COCO), or a .yaml/.yml "
      "(data.yaml).");
}

}  // namespace yolocpp::cli
