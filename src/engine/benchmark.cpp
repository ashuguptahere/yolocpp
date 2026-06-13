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

namespace {
double file_mb(const std::string& path) {
  std::error_code ec;
  auto n = fs::file_size(path, ec);
  return ec ? 0.0 : static_cast<double>(n) / (1024.0 * 1024.0);
}
}  // namespace

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
  // Engine caches encode imgsz, batch, precision so the same model at
  // different batch sizes coexists (b=1 single-image latency vs b=32
  // throughput).
  auto bs_tag    = "b" + std::to_string(cfg.batch_size);
  auto onnx_path = (fs::path(cfg.cache_dir) /
                    (stem + "." + std::to_string(cfg.imgsz) + ".onnx")).string();
  auto trt32     = (fs::path(cfg.cache_dir) /
                    (stem + "." + std::to_string(cfg.imgsz) + "." + bs_tag + ".fp32.trt")).string();
  auto trt16     = (fs::path(cfg.cache_dir) /
                    (stem + "." + std::to_string(cfg.imgsz) + "." + bs_tag + ".fp16.trt")).string();
  auto trt8      = (fs::path(cfg.cache_dir) /
                    (stem + "." + std::to_string(cfg.imgsz) + "." + bs_tag + ".int8.trt")).string();

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
    if (!results.empty()) results.back().size_mb = file_mb(cfg.weights);
  }

  // Build/cache ONNX once if any TRT path is requested.
  bool need_onnx = cfg.run_trt_fp32 || cfg.run_trt_fp16 || cfg.run_trt_int8;
  if (need_onnx && !fs::exists(onnx_path)) {
    build_onnx_for(cfg, version, scale_s, onnx_path);
  }

  // Helper that fills in the per-batch profile + v10 TF32 quirk.
  auto with_batch_profile = [&](serialization::TrtBuildConfig& tcfg) {
    tcfg.imgsz = cfg.imgsz;
    tcfg.builder_opt_level = 1;
    tcfg.batch_min = 1;
    tcfg.batch_opt = cfg.batch_size;
    tcfg.batch_max = cfg.batch_size;
    if (version == "v10") tcfg.tf32 = false;
  };

  // True batched throughput at batch=N: call predict_batch(N copies of
  // img) so a single enqueueV3 processes all N images. Reports median
  // call latency (ms) and per-image throughput = N / latency_s. (#88B.)
  auto bench_trt = [&](const std::string& tag, const std::string& plan_path) {
    inference::TrtPredictor p(plan_path, cfg.imgsz, cfg.batch_size);
    if (cfg.batch_size <= 1) {
      results.push_back(bench_one(tag, img, p,
                                  cfg.warmup_iters, cfg.iters));
      return;
    }
    std::vector<cv::Mat> batch(cfg.batch_size, img);
    for (int i = 0; i < cfg.warmup_iters; ++i) {
      auto _ = p.predict_batch(batch); (void)_;
    }
    std::vector<double> times_ms;
    times_ms.reserve(cfg.iters);
    std::vector<std::vector<inference::Detection>> last;
    for (int i = 0; i < cfg.iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      last = p.predict_batch(batch);
      auto t1 = std::chrono::steady_clock::now();
      times_ms.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    BenchResult r;
    r.backend          = tag + " (b=" + std::to_string(cfg.batch_size) + ")";
    r.median_ms        = detail::percentile(times_ms, 50);
    r.p95_ms           = detail::percentile(times_ms, 95);
    r.mean_ms          = detail::mean(times_ms);
    // Per-image throughput: N images processed per call.
    r.throughput_imgps = 1000.0 * cfg.batch_size / r.median_ms;
    r.num_detections   = last.empty() ? 0 : (int)last[0].size();
    if (!last.empty()) r.dets = std::move(last[0]);
    results.push_back(std::move(r));
  };

  // ── TRT FP32 ────────────────────────────────────────────────────────────
  if (cfg.run_trt_fp32) {
    if (!fs::exists(trt32)) {
      serialization::TrtBuildConfig tcfg;
      with_batch_profile(tcfg);
      tcfg.fp16 = false;
      serialization::build_trt_engine(onnx_path, trt32, tcfg);
    }
    bench_trt("TRT FP32", trt32);
    results.back().size_mb = file_mb(trt32);
  }

  // ── TRT FP16 ────────────────────────────────────────────────────────────
  if (cfg.run_trt_fp16) {
    if (!fs::exists(trt16)) {
      serialization::TrtBuildConfig tcfg;
      with_batch_profile(tcfg);
      tcfg.fp16 = true;
      serialization::build_trt_engine(onnx_path, trt16, tcfg);
    }
    bench_trt("TRT FP16", trt16);
    results.back().size_mb = file_mb(trt16);
  }

  // ── TRT INT8 ────────────────────────────────────────────────────────────
  if (cfg.run_trt_int8) {
    if (cfg.int8_calib_dir.empty()) {
      std::cerr << "[bench] INT8 requested but --int8-calib unset; skipping\n";
    } else {
      if (!fs::exists(trt8)) {
        serialization::TrtBuildConfig tcfg;
        with_batch_profile(tcfg);
        tcfg.int8 = true;
        tcfg.fp16 = false;  // INT8-only — matches Ultralytics' engine.py
        // (line 283: they only set FP16 flag in the `elif half` branch).
        // Mixing FP16+INT8 made TRT pick more FP16 fallback tactics
        // than necessary, costing ~5% throughput on small models.
        tcfg.calib_image_dir = cfg.int8_calib_dir;
        tcfg.calib_cache     = cfg.int8_calib_cache.empty()
            ? trt8 + ".calib"
            : cfg.int8_calib_cache;
        serialization::build_trt_engine(onnx_path, trt8, tcfg);
      }
      bench_trt("TRT INT8", trt8);
      results.back().size_mb = file_mb(trt8);
    }
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
