#include "yolocpp/engine/benchmark.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "yolocpp/cli/resolve.hpp"
#include "yolocpp/inference/letterbox.hpp"
#include "yolocpp/inference/nms.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/inference/trt_predictor.hpp"
#include "yolocpp/engine/benchmark_internal.hpp"
#include "yolocpp/models/yolo8.hpp"  // v8 fallback path uses Yolo8Scale
#include "yolocpp/registry/version_adapter.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/trt_export.hpp"

namespace fs = std::filesystem;

namespace yolocpp::engine {

namespace {

// Templates + helpers shared with the registry live in
// `yolocpp/engine/benchmark_internal.hpp`.
using detail::bench_one;
using detail::GenericPredictor;
using detail::pick_device;

std::string base_name(const std::string& path) {
  fs::path p(path);
  return p.stem().string();
}

models::Yolo8Scale parse_yolo8_scale(const std::string& s) {
  if (s == "s") return models::kYolo8s;
  if (s == "m") return models::kYolo8m;
  if (s == "l") return models::kYolo8l;
  if (s == "x") return models::kYolo8x;
  return models::kYolo8n;
}

// Run the PT FP32 benchmark for a specific version via the registry.
// Returns nullopt for v8 / unknown versions — caller falls back to the
// legacy `inference::Predictor` path.
std::optional<BenchResult> bench_pt(const BenchConfig& cfg,
                                    const cv::Mat& img,
                                    const std::string& version,
                                    const std::string& scale_s) {
  registry::register_all_versions();
  const auto* adapter = registry::Registry::instance().find(version);
  if (!adapter || !adapter->benchmark_pt) return std::nullopt;
  return adapter->benchmark_pt(cfg, img, scale_s);
}

// Build the ONNX file for a given version via the registry. Replaces
// ~95 lines of per-version dispatch — the same emitter the CLI's
// `cmd_export` uses, just routed by adapter rather than by if-else.
// v8 is registered too via its `export_onnx` hook (kept the lambda
// form even though v8 has no dedicated `predict_v8_to_file`).
void build_onnx_for(const BenchConfig& cfg, const std::string& version,
                    const std::string& scale_s,
                    const std::string& onnx_path) {
  registry::register_all_versions();
  const auto* adapter = registry::Registry::instance().find(version);
  if (!adapter || !adapter->export_onnx) {
    throw std::runtime_error(
        "benchmark: no ONNX exporter registered for version '" + version + "'");
  }
  serialization::OnnxExportConfig ocfg;
  ocfg.imgsz = cfg.imgsz;
  adapter->export_onnx(cfg.weights, scale_s, cfg.nc, /*task=*/"detect",
                       onnx_path, ocfg);
}

}  // anonymous namespace

std::vector<BenchResult> run_benchmark(const BenchConfig& cfg) {
  if (cfg.weights.empty() || cfg.source.empty())
    throw std::runtime_error("benchmark needs both weights and source");

  cv::Mat img = cv::imread(cfg.source, cv::IMREAD_COLOR);
  if (img.empty())
    throw std::runtime_error("cannot read image: " + cfg.source);

  // Resolve version + scale: explicit cfg overrides win; otherwise infer
  // from the weights filename. Falls back to v8 / scale=n for unknown
  // patterns (preserves the legacy single-version behavior).
  std::string version = cfg.version;
  if (version.empty()) version = cli::version_from_filename(cfg.weights);
  if (version.empty()) version = "v8";
  std::string scale_s = cfg.scale;
  if (scale_s.empty()) {
    auto fs_scale = cli::scale_from_filename(cfg.weights);
    if (!fs_scale.empty()) scale_s = fs_scale;
  }
  if (scale_s.empty()) scale_s = "n";

  fs::create_directories(cfg.cache_dir);
  auto stem      = base_name(cfg.weights);
  auto onnx_path = (fs::path(cfg.cache_dir) /
                    (stem + "." + std::to_string(cfg.imgsz) + ".onnx")).string();
  auto trt32     = (fs::path(cfg.cache_dir) /
                    (stem + "." + std::to_string(cfg.imgsz) + ".fp32.trt")).string();
  auto trt16     = (fs::path(cfg.cache_dir) /
                    (stem + "." + std::to_string(cfg.imgsz) + ".fp16.trt")).string();

  std::vector<BenchResult> results;

  // ── PT (libtorch) ───────────────────────────────────────────────────────
  if (cfg.run_pt) {
    auto pt_res = bench_pt(cfg, img, version, scale_s);
    if (pt_res) {
      results.push_back(std::move(*pt_res));
    } else {
      // v8 (and unknown) fall through to the legacy `inference::Predictor`
      // path which handles v8 directly.
      inference::Predictor p(cfg.weights, cfg.imgsz, cfg.device,
                             cfg.nc, parse_yolo8_scale(scale_s));
      results.push_back(bench_one("PT (libtorch FP32)", img, p,
                                  cfg.warmup_iters, cfg.iters));
    }
  }

  // Build/cache ONNX once if any TRT path is requested.
  bool need_onnx = cfg.run_trt_fp32 || cfg.run_trt_fp16;
  if (need_onnx && !fs::exists(onnx_path)) {
    build_onnx_for(cfg, version, scale_s, onnx_path);
  }

  // ── TRT FP32 ────────────────────────────────────────────────────────────
  if (cfg.run_trt_fp32) {
    if (!fs::exists(trt32)) {
      serialization::TrtBuildConfig tcfg;
      tcfg.imgsz = cfg.imgsz;
      tcfg.fp16  = false;
      tcfg.builder_opt_level = 1;
      // v10 saturates cls under TF32 — clear the kTF32 builder flag to
      // force true FP32 math (mirrors `cmd_export`'s v10 override).
      if (version == "v10") tcfg.tf32 = false;
      serialization::build_trt_engine(onnx_path, trt32, tcfg);
    }
    inference::TrtPredictor p(trt32, cfg.imgsz);
    results.push_back(bench_one("TRT FP32", img, p,
                                cfg.warmup_iters, cfg.iters));
  }

  // ── TRT FP16 ────────────────────────────────────────────────────────────
  if (cfg.run_trt_fp16) {
    if (!fs::exists(trt16)) {
      serialization::TrtBuildConfig tcfg;
      tcfg.imgsz = cfg.imgsz;
      tcfg.fp16  = true;
      tcfg.builder_opt_level = 1;
      if (version == "v10") tcfg.tf32 = false;
      serialization::build_trt_engine(onnx_path, trt16, tcfg);
    }
    inference::TrtPredictor p(trt16, cfg.imgsz);
    results.push_back(bench_one("TRT FP16", img, p,
                                cfg.warmup_iters, cfg.iters));
  }

  return results;
}

void print_benchmark(const std::vector<BenchResult>& rows) {
  std::cout << "\n";
  std::cout << "  backend                  median (ms)   p95 (ms)   mean (ms)   img/s    dets\n";
  std::cout << "  ──────────────────────  ────────────  ─────────  ──────────  ───────  ─────\n";
  for (const auto& r : rows) {
    char line[256];
    std::snprintf(line, sizeof(line),
                  "  %-22s  %12.2f  %9.2f  %10.2f  %7.1f  %5d\n",
                  r.backend.c_str(), r.median_ms, r.p95_ms, r.mean_ms,
                  r.throughput_imgps, r.num_detections);
    std::cout << line;
  }
  // Speedup vs PT (first row).
  if (rows.size() > 1) {
    std::cout << "\n  Speedup vs PT:\n";
    double base = rows.front().median_ms;
    for (size_t i = 1; i < rows.size(); ++i) {
      char line[128];
      std::snprintf(line, sizeof(line), "    %-22s  %.2fx\n",
                    rows[i].backend.c_str(), base / rows[i].median_ms);
      std::cout << line;
    }
  }
  std::cout << "\n";
}

}  // namespace yolocpp::engine
