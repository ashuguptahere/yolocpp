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
              << "mAP@0.5:0.95 = " << r.map_50_95 << "\n"
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
            << "mAP@0.5:0.95 = " << res.map_50_95 << "\n";
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
              const std::string& optimizer) {
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

// Pick a torch device honouring the CLI `--device` value (already
// normalised by `normalise_device`). Empty / unrecognised → CUDA when
// available, else CPU.
torch::Device pick_device(const std::string& device) {
  if (device == "cpu") return torch::Device(torch::kCPU);
  if (device.empty() || device == "auto" || device == "cuda" ||
      device.rfind("cuda:", 0) == 0) {
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                        : torch::Device(torch::kCPU);
  }
  return torch::Device(device);
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

  if (task == "classify") {
    int sz = (imgsz == 640) ? 224 : imgsz;
    yolocpp::tasks::ClassifyDataset ds(data, "val", sz, /*augment=*/false);
    yolocpp::models::Yolo8Classify m(parse_scale(scale_s), ds.num_classes());
    load(m);
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
    load(m);
    auto r = yolocpp::tasks::validate_segment(m, ds, dev);
    std::cout << "mask mAP@0.5=" << r.map_50
              << " (pred=" << r.n_predictions
              << " gt=" << r.n_ground_truths << ")\n";
    return 0;
  }
  if (task == "pose") {
    yolocpp::tasks::PoseDataset ds(data, "val", imgsz, /*num_kpts=*/17,
                                    /*kpt_dim=*/3, /*augment=*/false);
    yolocpp::models::Yolo8Pose m(parse_scale(scale_s), /*nc=*/1, 17, 3);
    load(m);
    auto r = yolocpp::tasks::validate_pose(m, ds, dev);
    std::cout << "OKS mAP@0.5=" << r.oks_map_50 << "\n";
    return 0;
  }
  if (task == "obb") {
    auto names = split_csv(names_csv);
    if (names.empty()) names = yolocpp::inference::dota_names();
    yolocpp::tasks::OBBDataset ds(data, "val", imgsz, names, /*augment=*/false);
    yolocpp::models::Yolo8OBB m(parse_scale(scale_s), ds.num_classes(), /*ne=*/1);
    load(m);
    auto r = yolocpp::tasks::validate_obb(m, ds, dev);
    std::cout << "rotated mAP@0.5=" << r.map_50 << "\n";
    return 0;
  }
  std::cerr << "[error] cmd_val_task: unknown task '" << task << "'\n";
  return 2;
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

  auto load = [&](auto& m) {
    if (!weights.empty()) {
      auto sd = yolocpp::serialization::load_state_dict(weights);
      m->load_from_state_dict(sd.entries);
    }
  };

  if (task == "classify") {
    int sz = (imgsz == 640) ? 224 : imgsz;
    yolocpp::tasks::ClassifyDataset tr(data_root, "train", sz, /*augment=*/true);
    yolocpp::models::Yolo8Classify m(parse_scale(scale_s), tr.num_classes());
    load(m);
    yolocpp::tasks::ClassifyTrainConfig cfg;
    cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = sz;
    cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
    yolocpp::tasks::train_classify(m, tr, /*val=*/nullptr, cfg);
    return 0;
  }
  if (task == "segment") {
    auto names = split_csv(names_csv);
    if (names.empty()) names = yolocpp::inference::coco_names();
    yolocpp::tasks::SegDataset tr(data_root, "train", imgsz, names, /*augment=*/true);
    yolocpp::models::Yolo8Segment m(parse_scale(scale_s), tr.num_classes());
    load(m);
    yolocpp::tasks::SegTrainConfig cfg;
    cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
    cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
    yolocpp::tasks::train_segment(m, tr, /*val=*/nullptr, cfg);
    return 0;
  }
  if (task == "pose") {
    yolocpp::tasks::PoseDataset tr(data_root, "train", imgsz, 17, 3, /*augment=*/true);
    yolocpp::models::Yolo8Pose m(parse_scale(scale_s), /*nc=*/1, 17, 3);
    load(m);
    yolocpp::tasks::PoseTrainConfig cfg;
    cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
    cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
    yolocpp::tasks::train_pose(m, tr, /*val=*/nullptr, cfg);
    return 0;
  }
  if (task == "obb") {
    auto names = split_csv(names_csv);
    if (names.empty()) names = yolocpp::inference::dota_names();
    yolocpp::tasks::OBBDataset tr(data_root, "train", imgsz, names, /*augment=*/true);
    yolocpp::models::Yolo8OBB m(parse_scale(scale_s), tr.num_classes(), /*ne=*/1);
    load(m);
    yolocpp::tasks::OBBTrainConfig cfg;
    cfg.epochs = epochs; cfg.batch_size = batch; cfg.imgsz = imgsz;
    cfg.lr0 = lr0; cfg.device = device; cfg.save_dir = save_dir;
    yolocpp::tasks::train_obb(m, tr, /*val=*/nullptr, cfg);
    return 0;
  }
  std::cerr << "[error] cmd_train_task: unknown task '" << task << "'\n";
  return 2;
}

int cmd_export(const std::string& weights, const std::string& format,
               const std::string& out, int imgsz, const std::string& scale_s_in,
               int nc, const std::string& input_name, bool fp16,
               const std::string& version_hint,
               const std::string& task) {
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

int cmd_predict_task(const std::string& task, const std::string& weights,
                     const std::string& source, std::string out, int imgsz,
                     std::string device, std::string scale_s, int nc,
                     float conf, float iou,
                     const std::string& version_hint,
                     std::vector<yolocpp::inference::Detection>* out_dets) {
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
      std::cerr << "[error] --task=" << task
                << " not yet wired for video/URL/webcam frame loop "
                   "(only detect is). Filed as a follow-up to #51C2.\n";
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
      std::cerr << "[error] could not open source: " << source << "\n";
      return 2;
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
          std::cerr << "[error] could not open output writer: "
                    << out_path << "\n";
          return 2;
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
    int sub_rc = predict_one_image(task, weights, in, this_out, imgsz,
                                    device, scale_s, nc, conf, iou,
                                    version_hint, out_dets);
    if (sub_rc != 0) rc = sub_rc;  // last non-zero wins, but keep going
  }
  // For multi-input runs, `out_dets` ends up holding the LAST
  // processed image's dets (each loop iter overwrites). The CLI
  // ignores it; the API uses it for single-image predict + treats
  // multi-input as "intermediate writes to disk only". Future
  // enhancement (#52A3) could collect per-input dets in a map.
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
