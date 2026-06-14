// yolocpp CLI driver — argument parsing + dispatch lives in
// `src/cli/commands.cpp` (`cmd_dispatch_flag_style`). This file
// is just the entry-point shell: --version short-circuit, top-level
// help, and the dispatch call.
//
// Exit-code convention (#51I2) — every CLI path returns one of:
//   0  success.
//   2  user error: bad or missing CLI input — missing required flag,
//      unknown/unsupported --mode / --task / --format / --precision /
//      --device value, unclassifiable --source, un-inferable version.
//      Detected at arg-handling time, before (or instead of) doing work.
//   1  runtime error: a failure while executing valid input — a file /
//      stream / device that won't open, a model that won't load, a TRT
//      build failure, or any uncaught exception (caught by main()).
// New code MUST follow this: validation failures → 2, execution failures
// → 1. (The public C++ API throws instead of returning codes.)

#include <cstdlib>
#include <iostream>
#include <string>

#include <yolocpp/config.hpp>
#include "yolocpp/cli/commands.hpp"
#include "yolocpp/core/log.hpp"
#include "yolocpp/core/profile.hpp"
#include "yolocpp/inference/results.hpp"

#include <fstream>

#include <CLI11.hpp>

#include <filesystem>
#include <set>

#include "yolocpp/cli/resolve.hpp"
#include "yolocpp/serialization/trt_export.hpp"

namespace yolocpp::cli {

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
  std::string bench_precision = "fp32,fp16";  // CSV of fp32 | fp16 | int8
  std::string int8_calib_dir, int8_calib_cache;
  double lr0   = 0.01;
  // -1 → use trainer default (0.01 = cosine to 1% of lr0). Set to
  // 1.0 for constant LR (no cosine decay). Threaded through cmd_train
  // → TrainConfig.lrf.
  double lrf   = -1.0;
  std::string optimizer = "auto";  // auto → AdamW for batch<64, else SGD
  int workers = 4;  // BatchPrefetcher background threads (0 = synchronous)
  bool cache_ram = true;  // pre-decode all images into RAM at train start
  bool strict_det = false;  // bit-exact reproducibility at ~30-50% perf cost
  int close_mosaic = 10;  // #57G: disable mosaic+mixup last N epochs
  float  conf  = 0.25f, iou = 0.45f;
  bool   export_fp16 = true;
  uint64_t seed = 0;
  std::string save_dir = "runs/train";
  bool   profile_enabled = false;
  bool   debug_log = false;  // --debug/--verbose → YOLOCPP_LOG=debug
  // Predict output options (#97 Results integration).
  std::string save_json;       // --save-json <path>: dump Results.json()
  std::string save_txt;        // --save-txt  <path>: dump "cls conf x1 y1 x2 y2"

  app.add_option("--model,-m,--weights", weights,
                  "weights `.pt` / `.trt` (alias: --weights)");
  app.add_option("--source,-s",  source,
                  "image, video, dir, glob, URL, or webcam index");
  app.add_option("--data,-d",    data,
                  "dataset root or data.yaml path");
  app.add_option("--out,-o",     out,         "output path / directory");
  app.add_option("--imgsz,-i",   imgsz,
                  "input image size (default 640). Use space-separated form: "
                  "--imgsz 640");
  app.add_option("--epochs,-e",  epochs,      "epochs (train)");
  app.add_option("--batch,-b",   batch_size,  "batch size (train)");
  app.add_option("--lr0",        lr0,         "initial LR (train)");
  app.add_option("--optimizer",  optimizer,
                  "train optimizer: auto (default) | sgd | adamw");
  app.add_option("--workers",    workers,
                  "background data-prep threads (default 4; 0 = synchronous)");
  app.add_flag  ("--cache-ram,!--no-cache-ram", cache_ram,
                  "pre-decode all training images into RAM at start "
                  "(default on; --no-cache-ram to disable for huge datasets)");
  app.add_flag  ("--strict-deterministic", strict_det,
                  "bit-exact reproducibility: workers=0, no cuDNN benchmark, "
                  "torch deterministic algorithms (~30-50% slower)");
  app.add_option("--close-mosaic", close_mosaic,
                  "train: disable mosaic+mixup for the last N epochs "
                  "(default 10; 0 = keep mosaic the whole run)");
  app.add_option("--lrf",        lrf,
                  "final LR fraction of lr0 (cosine schedule end). "
                  "Default 0.01 = decay to 1%% of lr0; pass 1.0 for constant LR.");
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
                  "export: fp32 | fp16 | int8 (TRT; int8 needs --int8-calib) "
                  "| int4 | nvfp4 (int4/nvfp4 not yet wired — #51F2)");
  app.add_option("--input-name", export_input_name,
                  "ONNX graph input tensor name");
  app.add_flag  ("--fp16,!--no-fp16", export_fp16,
                  "TRT FP16 (legacy alias for --precision=fp16/fp32)");
  app.add_option("--warmup",     warmup,      "benchmark warmup iters");
  app.add_option("--iters",      iters,       "benchmark timed iters");
  app.add_option("--cache",      cache_dir,   "TRT engine cache directory");
  app.add_option("--bench-precision", bench_precision,
                 "benchmark precision list (csv: fp32,fp16,int8). int8 needs --int8-calib");
  app.add_option("--int8-calib", int8_calib_dir,
                 "INT8 calibration image directory (val split recommended)");
  app.add_option("--int8-calib-cache", int8_calib_cache,
                 "INT8 calibration cache file (default: <engine>.calib)");
  app.add_flag  ("--profile", profile_enabled,
                  "enable per-phase wall-clock profiler (every mode, every model)");
  app.add_flag  ("--debug,--verbose", debug_log,
                  "verbose debug logging (equivalent to YOLOCPP_LOG=debug)");
  app.add_option("--save-json", save_json,
                  "predict: dump Results.json() (boxes + xyxy + names + speed) to file");
  app.add_option("--save-txt", save_txt,
                  "predict: dump 'cls conf x1 y1 x2 y2' lines to file (Ultralytics-style)");
  app.add_option("--dataset",    dl_target,
                  "download mode: dataset short-name (coco8, VOC, ...) or .zip URL");

  CLI11_PARSE(app, argc, argv);
  if (debug_log) yolocpp::log::set_level(yolocpp::log::Level::Debug);
  device = normalise_device(device);
  // Flip the global profile switch as early as possible — every
  // PROFILE_SCOPE downstream will then start recording. atexit
  // hook prints the summary on clean exit.
  if (profile_enabled) {
    yolocpp::core::Profile::instance().set_enabled(true);
    std::atexit([]() {
      yolocpp::core::Profile::instance().print_summary(std::cout);
    });
    std::cerr << "[profile] enabled — summary will print at exit\n";
  }

  // Auto-resolve weights: searches cwd, ./data/, ~/.cache/yolocpp/weights/,
  // and falls back to downloading from upstream for recognised
  // basenames (e.g. yolo3n.pt → cached under canonical name). Also
  // routes yolo4 .weights → .pt conversion and yolo6 upstream-pt →
  // canonical-pt conversion. Skipped for trt engines (.trt) and any
  // file that already exists at the given path.
  // benchmark resolves each model itself (its --model may be a comma-separated
  // list a.pt,b.pt,…), so skip the single-spec resolve here for that mode.
  if (!weights.empty() && mode != "download" && mode != "benchmark") {
    try {
      weights = resolve_weights(weights);
    } catch (const std::exception& e) {
      // For mode=train, allow a bare version+scale spec (e.g.
      // `-m yolo1`, `-m yolo2-tiny`) to fall through to "train from
      // scratch": pass `weights` along so the version/scale resolver
      // can extract its hint, but the trainer will see no init file
      // and start from random weights.
      auto vh = yolocpp::cli::version_from_filename(weights);
      if ((mode == "train" || mode == "export") && !vh.empty()) {
        std::cerr << "[info] no weights file found for '" << weights
                  << "' — proceeding with random-init for mode=" << mode
                  << " (version=" << vh << ")\n";
      } else {
        std::cerr << "[error] resolve weights: " << e.what() << "\n";
        return 2;
      }
    }
  }

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

  // Validate --task once for every mode that consumes it. Detect is
  // always supported; classify/segment/pose/obb route through the v8
  // task families (only architecture whose task heads ship upstream).
  static const std::set<std::string> kKnownTasks = {
      "detect", "classify", "segment", "pose", "obb"};
  if (!kKnownTasks.count(task)) {
    std::cerr << "[error] --task='" << task
              << "' not recognised (expected: detect | classify | segment | "
                 "pose | obb)\n";
    return 2;
  }

  if (mode == "info") return cmd_info();

  if (mode == "predict") {
    if (!need(mode, "model",  !weights.empty())) return 2;
    if (!need(mode, "source", !source.empty()))  return 2;
    // #97: collect last-image detections so we can emit Ultralytics-
    // style sidecar files (.json / .txt) when the user asks for them.
    std::vector<yolocpp::inference::Detection> last_dets;
    int rc = cmd_predict_task(task, weights, source, out, imgsz, device,
                              scale_s, nc, conf, iou,
                              /*version_hint=*/"", &last_dets);
    if (rc == 0 && (!save_json.empty() || !save_txt.empty())) {
      // Wrap last-image detections in a Results object so callers get
      // the same shape Ultralytics returns. orig_img isn't carried
      // through cmd_predict_task — Results.json() works without it
      // (orig_shape will be [0, 0]); Results.save_txt() works too.
      yolocpp::inference::Results r;
      r.boxes = std::move(last_dets);
      // Parse names_csv (already set above for the CLI dispatcher).
      if (!names_csv.empty()) {
        for (const auto& nm : yolocpp::cli::split_csv(names_csv))
          r.names.push_back(nm);
      }
      if (!save_json.empty()) {
        std::ofstream f(save_json);
        f << r.json() << "\n";
        std::cerr << "[predict] wrote " << save_json << "\n";
      }
      if (!save_txt.empty()) {
        r.save_txt(save_txt);
        std::cerr << "[predict] wrote " << save_txt << "\n";
      }
    }
    return rc;
  }

  if (mode == "val") {
    if (!need(mode, "model", !weights.empty())) return 2;
    if (!need(mode, "data",  !data.empty()))    return 2;
    if (task == "detect") {
      // Detect val is registry-routed and supports every YOLO version.
      return cmd_val(weights, data, names_csv, imgsz, device, scale_s);
    }
    // Non-detect val uses the v8 task families.
    return cmd_val_task(task, weights, data, names_csv, imgsz, device,
                        scale_s);
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
    int rc;
    if (task == "detect") {
      // Detect train is registry-routed.
      rc = cmd_train(data, names_csv, imgsz, epochs, batch_size, lr0,
                      device, scale_s, save_dir, weights,
                      patience, /*args_for_yaml=*/{}, seed, lrf, optimizer,
                      workers, cache_ram, strict_det, close_mosaic);
    } else {
      // Non-detect train uses the v8 task families.
      rc = cmd_train_task(task, data, names_csv, imgsz, epochs, batch_size,
                           lr0, device, scale_s, save_dir, weights);
    }
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
    bool export_int8 = false;
    if (!export_precision.empty()) {
      if (export_precision == "fp32") export_fp16 = false;
      else if (export_precision == "fp16") export_fp16 = true;
      else if (export_precision == "int8") {
        // INT8 PTQ (#51F2) is a TRT-engine feature — the ONNX graph is
        // always fp32. Require a calibration image directory.
        if (export_fmt != "trt" && export_fmt != "engine") {
          std::cerr << "[error] --precision=int8 applies to TRT export only "
                       "(use -f trt)\n";
          return 2;
        }
        if (int8_calib_dir.empty()) {
          std::cerr << "[error] --precision=int8 needs --int8-calib <dir> "
                       "(a folder of representative images, e.g. a val split)\n";
          return 2;
        }
        export_int8 = true;
        export_fp16 = false;
      }
      else if (export_precision == "int4" || export_precision == "nvfp4") {
        std::cerr << "[error] --precision=" << export_precision
                  << " not yet wired — needs Blackwell TRT 10.4+ low-bit APIs "
                     "(TODO #51F2)\n";
        return 2;
      } else {
        std::cerr << "[error] unknown --precision='" << export_precision
                  << "' (expected fp32 | fp16 | int8 | int4 | nvfp4)\n";
        return 2;
      }
    }
    return cmd_export(weights, export_fmt, out, imgsz, scale_s, nc,
                       export_input_name, export_fp16,
                       /*version_hint=*/"", task,
                       export_int8, int8_calib_dir, int8_calib_cache);
  }

  if (mode == "benchmark") {
    if (!need(mode, "model",  !weights.empty())) return 2;
    // Non-detect tasks route to the PT task benchmark (#5) — forward timing +
    // optional --data metric. No --source needed (timing uses a synthetic
    // input; the metric comes from --data).
    if (task != "detect") {
      return cmd_benchmark_task(task, weights, data, names_csv, imgsz, device,
                                scale_s, warmup, iters);
    }
    if (!need(mode, "source", !source.empty()))  return 2;
    // Benchmark defaults to batch=1 (single-image latency, Ultralytics-style)
    // unless --batch was given explicitly — the global default of 16 is a
    // training default, not a latency-benchmark one.
    int bench_batch = app.count("--batch") ? batch_size : 1;
    return cmd_benchmark(weights, source, imgsz, warmup, iters, cache_dir, device,
                         bench_batch, bench_precision, int8_calib_dir, int8_calib_cache,
                         data, names_csv, scale_s);
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

}  // namespace yolocpp::cli


int main(int argc, char** argv) {
  // libtorch's intra-op OMP threadpool uses ACTIVE wait by default
  // (KMP_BLOCKTIME=200ms on Intel OpenMP) which keeps worker threads
  // *spinning* on the CPU between micro-ops. For training where each
  // step launches hundreds of small CUDA kernels, those threads
  // barely sleep — and we observed ~12 cores busy on yolo11n with
  // only 4 prefetcher threads + 4 intra-op threads + main = 9 logical
  // threads, the rest of the CPU usage being spin-wait. setenv with
  // overwrite=0 respects user-supplied values; must run BEFORE the
  // first libtorch call so OMP picks the policy up at init time.
  setenv("OMP_WAIT_POLICY", "PASSIVE", /*overwrite=*/0);
  setenv("KMP_BLOCKTIME",   "0",       /*overwrite=*/0);
  yolocpp::log::init_from_env();  // YOLOCPP_LOG baseline (--debug overrides post-parse)
  using namespace yolocpp::cli;
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
      "  yolocpp --mode predict --task segment -m yolo11s-seg.pt  -s bus.jpg\n"
      "  yolocpp --mode predict --task pose    -m yolo11s-pose.pt -s bus.jpg\n"
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
      "Tasks  : detect (default), classify, segment, pose, obb\n"
      "         (--task <name>; classify/segment/pose/obb use the v8\n"
      "          task families — the only architectures whose task\n"
      "          heads ship with weights upstream)\n"
      "Modes  : train, val, predict, export, benchmark, info, download\n"
      "Formats: onnx, trt\n"
      "Precs  : fp32, fp16, int8 (TRT; int8 needs --int8-calib <dir>)\n"
      "         (int4/nvfp4 — TODO #51F2)\n"
      "Devices: cpu, cuda, cuda:N, cuda:0,1,..., mps, auto\n"
      "Source : single image / directory / glob (e.g. 'frames/*.jpg').\n"
      "         Video / RTSP / HTTP / webcam-index — frame loop is\n"
      "         tracked under #51C2 in TODO.md.\n";
    return 0;
  }

  try {
    // Single canonical CLI parser: flag-style.
    //   yolocpp --mode <action> [-m model] [-s source] ...
    // The kv-style and legacy subcommand-style parsers were
    // removed under #51K — flag-style is now the only entry point.
    return cmd_dispatch_flag_style(argc, argv);

  } catch (const std::exception& e) {
    std::cerr << "[error] " << e.what() << "\n";
    return 1;
  }
  return 1;
}
