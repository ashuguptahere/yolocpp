// Public chainable C++ API (#52) — `yolocpp::YOLO`.
//
// Every method routes to the same `cmd_*` body the CLI uses; see
// `include/yolocpp/cli/commands.hpp` for the dispatch surface. This
// file contains zero domain logic of its own — it's purely an
// argument-shape adapter so callers can write
//
//   yolocpp::YOLO("yolo11s.pt").predict({.source = "bus.jpg"});
//
// instead of stringing together CLI flags. Keep this file thin: when
// new CLI flags land, add the corresponding field to the matching
// Args struct in `api.hpp` and forward it here.

#include "yolocpp/api.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

#include "yolocpp/cli/commands.hpp"

namespace yolocpp {

namespace {

// Pick the device that should reach the underlying cmd_*. Per-call
// override wins; falls back to the model-level default set via
// `to(...)`. Both pass through `cli::normalise_device` so bad values
// throw with the same error the CLI surfaces.
std::string resolve_device(const std::string& per_call,
                            const std::string& model_default) {
  return cli::normalise_device(per_call.empty() ? model_default : per_call);
}

// Same idea for `task` — per-call override > model default > "detect".
std::string resolve_task(const std::string& per_call,
                          const std::string& model_default) {
  if (!per_call.empty()) return per_call;
  if (!model_default.empty()) return model_default;
  return "detect";
}

}  // namespace

YOLO::YOLO(std::string weights) : weights_(std::move(weights)) {}

YOLO& YOLO::to(std::string device) {
  // Validate now so a bad string is reported at the call site that
  // set it, not buried inside the next predict/train call.
  default_device_ = cli::normalise_device(std::move(device));
  return *this;
}

YOLO& YOLO::task(std::string task) {
  default_task_ = std::move(task);
  return *this;
}

std::vector<inference::Detection> YOLO::predict(const PredictArgs& a) {
  auto device = resolve_device(a.device, default_device_);
  auto task   = resolve_task(a.task, default_task_);
  // `cmd_predict_task` drives the source classifier and the
  // per-version `predict_to_file` hook (or v8 fallback). When called
  // with an out-param it also threads the last processed image's
  // dets back through (#52A2). Video/URL/Webcam frame loops still
  // return empty here — per-frame dets live in the on-disk mp4.
  std::vector<inference::Detection> dets;
  int rc = cli::cmd_predict_task(task, weights_, a.source, a.out, a.imgsz,
                                  device, a.scale, a.nc, a.conf, a.iou,
                                  /*version_hint=*/"", &dets);
  if (rc != 0) {
    throw std::runtime_error("yolocpp::YOLO::predict failed (rc=" +
                              std::to_string(rc) + ")");
  }
  return dets;
}

ValResult YOLO::val(const ValArgs& a) {
  auto device = resolve_device(a.device, default_device_);
  auto task   = resolve_task(a.task, default_task_);
  // cmd_val / cmd_val_task print the result directly; we don't have
  // a structured return path yet. Future #52B: thread mAP back so
  // the API can return populated ValResult fields. For now this
  // returns zeros; users get the printed numbers + the
  // `runs/val/<stem>_results.txt` artefact.
  int rc = (task == "detect")
               ? cli::cmd_val(weights_, a.data, a.names, a.imgsz, device,
                               a.scale)
               : cli::cmd_val_task(task, weights_, a.data, a.names, a.imgsz,
                                    device, a.scale);
  if (rc != 0) {
    throw std::runtime_error("yolocpp::YOLO::val failed (rc=" +
                              std::to_string(rc) + ")");
  }
  return {};
}

YOLO& YOLO::train(const TrainArgs& a) {
  auto device = resolve_device(a.device, default_device_);
  auto task   = resolve_task(a.task, default_task_);
  int rc = (task == "detect")
               ? cli::cmd_train(a.data, a.names, a.imgsz, a.epochs, a.batch,
                                 a.lr0, device, a.scale, a.save, weights_,
                                 a.patience, /*args_for_yaml=*/{}, a.seed)
               : cli::cmd_train_task(task, a.data, a.names, a.imgsz,
                                      a.epochs, a.batch, a.lr0, device,
                                      a.scale, a.save, weights_);
  if (rc != 0) {
    throw std::runtime_error("yolocpp::YOLO::train failed (rc=" +
                              std::to_string(rc) + ")");
  }
  // Post-train auto-export. Mirrors the CLI's --export-after-train
  // flag, exporting `<save>/best.pt` (or last.pt) next to the run.
  if (!a.export_after_train.empty()) {
    namespace fs = std::filesystem;
    fs::path src = fs::path(a.save) / "best.pt";
    if (!fs::exists(src)) src = fs::path(a.save) / "last.pt";
    if (fs::exists(src)) {
      // Comma-separated formats, e.g. "onnx,trt".
      std::string s = a.export_after_train;
      std::string buf;
      buf.reserve(s.size() + 1);
      auto run_one = [&](const std::string& fmt) {
        fs::path out_path = src; out_path.replace_extension("." + fmt);
        ExportArgs xa;
        xa.format = fmt;
        xa.out    = out_path.string();
        xa.imgsz  = a.imgsz;
        xa.scale  = a.scale;
        xa.precision = "fp16";
        xa.task   = task;
        // Route through cmd_export directly so we pass version_hint
        // resolved from the *original* weights (best.pt has no scale
        // letter in name).
        std::string vh; // resolved by cmd_export from filename if empty.
        int xrc = cli::cmd_export(src.string(), fmt, out_path.string(),
                                   a.imgsz, a.scale, /*nc=*/80,
                                   /*input_name=*/"images",
                                   /*fp16=*/true, vh, task);
        if (xrc != 0) {
          throw std::runtime_error("yolocpp::YOLO::train export-after-train "
                                    "failed (rc=" + std::to_string(xrc) + ")");
        }
      };
      std::size_t p = 0;
      for (std::size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',') {
          if (i > p) run_one(s.substr(p, i - p));
          p = i + 1;
        }
      }
    }
  }
  return *this;
}

YOLO& YOLO::export_(const ExportArgs& a) {
  auto task = resolve_task(a.task, default_task_);
  bool fp16 = (a.precision == "fp16");
  if (a.precision == "int8" || a.precision == "int4" ||
      a.precision == "nvfp4") {
    throw std::runtime_error("yolocpp::YOLO::export_: --precision=" +
                              a.precision +
                              " not yet wired (TODO #51F2)");
  }
  if (!a.precision.empty() && a.precision != "fp16" && a.precision != "fp32") {
    throw std::runtime_error("yolocpp::YOLO::export_: unknown precision='" +
                              a.precision + "'");
  }
  int rc = cli::cmd_export(weights_, a.format, a.out, a.imgsz, a.scale, a.nc,
                            a.input_name, fp16, /*version_hint=*/"", task);
  if (rc != 0) {
    throw std::runtime_error("yolocpp::YOLO::export_ failed (rc=" +
                              std::to_string(rc) + ")");
  }
  return *this;
}

YOLO& YOLO::benchmark(const BenchmarkArgs& a) {
  auto device = resolve_device(a.device, default_device_);
  int rc = cli::cmd_benchmark(weights_, a.source, a.imgsz, a.warmup, a.iters,
                                a.cache, device);
  if (rc != 0) {
    throw std::runtime_error("yolocpp::YOLO::benchmark failed (rc=" +
                              std::to_string(rc) + ")");
  }
  return *this;
}

}  // namespace yolocpp
