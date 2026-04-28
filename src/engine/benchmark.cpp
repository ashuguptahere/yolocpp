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

#include "yolocpp/inference/trt_predictor.hpp"
#include "yolocpp/serialization/onnx_export.hpp"
#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/trt_export.hpp"

namespace fs = std::filesystem;

namespace yolocpp::engine {

namespace {

double percentile(std::vector<double> xs, double p) {
  if (xs.empty()) return 0.0;
  std::sort(xs.begin(), xs.end());
  double r = (p / 100.0) * (xs.size() - 1);
  size_t lo = (size_t)std::floor(r), hi = (size_t)std::ceil(r);
  if (lo == hi) return xs[lo];
  double frac = r - (double)lo;
  return xs[lo] * (1.0 - frac) + xs[hi] * frac;
}
double mean(const std::vector<double>& xs) {
  if (xs.empty()) return 0.0;
  double s = 0.0;
  for (auto v : xs) s += v;
  return s / xs.size();
}

template <typename Pred>
BenchResult bench_one(const std::string& name, const cv::Mat& img,
                      Pred& predictor, int warmup, int iters) {
  // Warmup
  for (int i = 0; i < warmup; ++i) {
    auto _ = predictor.predict(img);
    (void)_;
  }
  std::vector<double> times_ms;
  times_ms.reserve(iters);
  std::vector<inference::Detection> last_dets;
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    last_dets = predictor.predict(img);
    auto t1 = std::chrono::steady_clock::now();
    times_ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  BenchResult r;
  r.backend           = name;
  r.median_ms         = percentile(times_ms, 50);
  r.p95_ms            = percentile(times_ms, 95);
  r.mean_ms           = mean(times_ms);
  r.throughput_imgps  = 1000.0 / r.median_ms;
  r.num_detections    = (int)last_dets.size();
  r.dets              = std::move(last_dets);
  return r;
}

std::string base_name(const std::string& path) {
  fs::path p(path);
  return p.stem().string();
}

}  // anonymous namespace

std::vector<BenchResult> run_benchmark(const BenchConfig& cfg) {
  if (cfg.weights.empty() || cfg.source.empty())
    throw std::runtime_error("benchmark needs both weights and source");

  cv::Mat img = cv::imread(cfg.source, cv::IMREAD_COLOR);
  if (img.empty())
    throw std::runtime_error("cannot read image: " + cfg.source);

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
    inference::Predictor p(cfg.weights, cfg.imgsz, cfg.device);
    results.push_back(bench_one("PT (libtorch FP32)", img, p,
                                cfg.warmup_iters, cfg.iters));
  }

  // Build/cache ONNX once if any TRT path is requested.
  bool need_onnx = cfg.run_trt_fp32 || cfg.run_trt_fp16;
  if (need_onnx && !fs::exists(onnx_path)) {
    models::Yolo8Detect model(models::kYolo8n, /*nc=*/80);
    auto sd = serialization::load_state_dict(cfg.weights);
    model->load_from_state_dict(sd.entries);
    model->eval();
    serialization::OnnxExportConfig ocfg; ocfg.imgsz = cfg.imgsz;
    serialization::export_yolo8_onnx(model, onnx_path, ocfg);
  }

  // ── TRT FP32 ────────────────────────────────────────────────────────────
  if (cfg.run_trt_fp32) {
    if (!fs::exists(trt32)) {
      serialization::TrtBuildConfig tcfg;
      tcfg.imgsz = cfg.imgsz;
      tcfg.fp16  = false;
      tcfg.builder_opt_level = 1;
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
