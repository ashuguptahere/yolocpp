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
#include "yolocpp/models/yolo10.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo12.hpp"
#include "yolocpp/models/yolo13.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo3.hpp"
#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/models/yolo5.hpp"
#include "yolocpp/models/yolo6.hpp"
#include "yolocpp/models/yolo7.hpp"
#include "yolocpp/models/yolo8.hpp"
#include "yolocpp/models/yolo9.hpp"
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

// Generic PT predictor adapter — wraps any model holder whose `forward_eval`
// returns `[B, 4+nc, A]` (xyxy + sigmoided cls). Mirrors
// `inference::Predictor::predict`'s shape: letterbox → forward_eval →
// NMS → scale_boxes back to original-image coordinates. Used by the
// PT FP32 benchmark path so we can time any of the 12 supported
// versions, not just v8.
template <typename ModelHolder>
struct GenericPredictor {
  ModelHolder model;
  int         imgsz;
  torch::Device device = torch::Device(torch::kCPU);

  GenericPredictor(ModelHolder m, int sz, torch::Device dev)
      : model(std::move(m)), imgsz(sz), device(dev) {
    model->to(device);
    model->eval();
  }

  std::vector<inference::Detection> predict(const cv::Mat& bgr,
                                             inference::NMSConfig nms = {}) const {
    auto lb = inference::letterbox(bgr, imgsz);
    auto x  = inference::image_to_tensor(lb.img).unsqueeze(0).to(device);
    torch::Tensor pred;
    {
      torch::NoGradGuard ng;
      pred = const_cast<ModelHolder&>(model)->forward_eval(x);
    }
    auto outs = inference::nms(pred, nms);
    if (outs.empty() || outs[0].size(0) == 0) return {};
    auto det = outs[0].to(torch::kCPU);
    auto boxes = det.slice(1, 0, 4).contiguous();
    inference::scale_boxes(boxes, lb);
    det.slice(1, 0, 4) = boxes;
    std::vector<inference::Detection> result;
    result.reserve(det.size(0));
    auto a = det.accessor<float, 2>();
    for (int i = 0; i < det.size(0); ++i) {
      inference::Detection d;
      d.x1   = a[i][0]; d.y1 = a[i][1]; d.x2 = a[i][2]; d.y2 = a[i][3];
      d.conf = a[i][4]; d.cls = (int)a[i][5];
      result.push_back(d);
    }
    return result;
  }
};

models::Yolo8Scale parse_yolo8_scale(const std::string& s) {
  if (s == "s") return models::kYolo8s;
  if (s == "m") return models::kYolo8m;
  if (s == "l") return models::kYolo8l;
  if (s == "x") return models::kYolo8x;
  return models::kYolo8n;
}

torch::Device pick_device(const std::string& d) {
  if (d == "cpu") return torch::Device(torch::kCPU);
  if (d == "cuda" || d.empty()) {
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                        : torch::Device(torch::kCPU);
  }
  return torch::Device(torch::kCPU);
}

// Run the PT FP32 benchmark for a specific version. Returns nullopt if
// the version isn't supported (caller falls back to a clear error).
std::optional<BenchResult> bench_pt(const BenchConfig& cfg,
                                    const cv::Mat& img,
                                    const std::string& version,
                                    const std::string& scale_s) {
  auto dev = pick_device(cfg.device);
  auto sd  = serialization::load_state_dict(cfg.weights);

  auto run = [&](auto& pred) {
    return bench_one("PT (libtorch FP32)", img, pred,
                     cfg.warmup_iters, cfg.iters);
  };

  if (version == "v3") {
    models::Yolo3 m(models::kYolo3, cfg.nc);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo3> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  if (version == "v4") {
    models::Yolo4 m(cfg.nc);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo4> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  if (version == "v5") {
    models::Yolo5Detect m(models::yolo5_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo5Detect> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  if (version == "v6") {
    auto v6_scale = models::kYolo6n;
    bool p6 = false;
    if      (scale_s == "n")        v6_scale = models::kYolo6n;
    else if (scale_s == "s")        v6_scale = models::kYolo6s;
    else if (scale_s == "m")        v6_scale = models::kYolo6m;
    else if (scale_s == "l")        v6_scale = models::kYolo6l;
    else if (scale_s == "n6")       { v6_scale = models::kYolo6n; p6 = true; }
    else if (scale_s == "s6")       { v6_scale = models::kYolo6s; p6 = true; }
    else if (scale_s == "m6")       { v6_scale = models::kYolo6m; p6 = true; }
    else if (scale_s == "l6")       { v6_scale = models::kYolo6l; p6 = true; }
    else if (scale_s == "s_mbla")   v6_scale = models::kYolo6s_mbla;
    else if (scale_s == "m_mbla")   v6_scale = models::kYolo6m_mbla;
    else if (scale_s == "l_mbla")   v6_scale = models::kYolo6l_mbla;
    else if (scale_s == "x_mbla")   v6_scale = models::kYolo6x_mbla;
    models::Yolo6 m(cfg.nc, v6_scale, /*reg_max=*/16, p6);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo6> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  if (version == "v7") {
    models::Yolo7 m(models::yolo7_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo7> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  if (version == "v9") {
    models::Yolo9 m(models::yolo9_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo9> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  if (version == "v10") {
    models::Yolo10 m(models::yolo10_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo10> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  if (version == "v11") {
    models::Yolo11Detect m(models::yolo11_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo11Detect> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  if (version == "v12") {
    models::Yolo12Detect m(models::yolo12_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo12Detect> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  if (version == "v13") {
    models::Yolo13Detect m(models::yolo13_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo13Detect> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  if (version == "v26") {
    models::Yolo26Detect m(models::yolo26_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    GenericPredictor<models::Yolo26Detect> p(std::move(m), cfg.imgsz, dev);
    return run(p);
  }
  // v8 and unknowns fall through to the legacy `inference::Predictor`
  // path (which handles v8 directly).
  return std::nullopt;
}

// Build the ONNX file for a given version. Mirrors `cmd_export`'s
// per-version dispatch in src/cli/main.cpp.
void build_onnx_for(const BenchConfig& cfg, const std::string& version,
                    const std::string& scale_s,
                    const std::string& onnx_path) {
  auto sd = serialization::load_state_dict(cfg.weights);
  serialization::OnnxExportConfig ocfg; ocfg.imgsz = cfg.imgsz;

  if (version == "v3") {
    models::Yolo3 m(models::kYolo3, cfg.nc);
    m->load_from_state_dict(sd.entries);
    {
      torch::NoGradGuard ng;
      auto x = torch::zeros({1, 3, ocfg.imgsz, ocfg.imgsz});
      m->forward_eval(x);
    }
    m->eval();
    serialization::export_yolo3_onnx(m, onnx_path, ocfg);
  } else if (version == "v4") {
    models::Yolo4 m(cfg.nc);
    m->load_from_state_dict(sd.entries);
    m->eval();
    serialization::export_yolo4_onnx(m, onnx_path, ocfg);
  } else if (version == "v5") {
    models::Yolo5Detect m(models::yolo5_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo5_onnx(m, onnx_path, ocfg);
  } else if (version == "v6") {
    auto v6_scale = models::kYolo6n;
    bool p6 = false;
    if      (scale_s == "n")        v6_scale = models::kYolo6n;
    else if (scale_s == "s")        v6_scale = models::kYolo6s;
    else if (scale_s == "m")        v6_scale = models::kYolo6m;
    else if (scale_s == "l")        v6_scale = models::kYolo6l;
    else if (scale_s == "n6")       { v6_scale = models::kYolo6n; p6 = true; }
    else if (scale_s == "s6")       { v6_scale = models::kYolo6s; p6 = true; }
    else if (scale_s == "m6")       { v6_scale = models::kYolo6m; p6 = true; }
    else if (scale_s == "l6")       { v6_scale = models::kYolo6l; p6 = true; }
    else if (scale_s == "s_mbla")   v6_scale = models::kYolo6s_mbla;
    else if (scale_s == "m_mbla")   v6_scale = models::kYolo6m_mbla;
    else if (scale_s == "l_mbla")   v6_scale = models::kYolo6l_mbla;
    else if (scale_s == "x_mbla")   v6_scale = models::kYolo6x_mbla;
    models::Yolo6 m(cfg.nc, v6_scale, /*reg_max=*/16, p6);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo6_onnx(m, onnx_path, ocfg);
  } else if (version == "v7") {
    models::Yolo7 m(models::yolo7_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    {
      torch::NoGradGuard ng;
      auto x = torch::zeros({1, 3, ocfg.imgsz, ocfg.imgsz});
      m->forward_eval(x);
    }
    m->eval();
    serialization::export_yolo7_onnx(m, onnx_path, ocfg);
  } else if (version == "v9") {
    models::Yolo9 m(models::yolo9_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    {
      torch::NoGradGuard ng;
      auto x = torch::zeros({1, 3, ocfg.imgsz, ocfg.imgsz});
      m->forward_eval(x);
    }
    m->eval();
    serialization::export_yolo9_onnx(m, onnx_path, ocfg);
  } else if (version == "v10") {
    models::Yolo10 m(models::yolo10_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries);
    {
      torch::NoGradGuard ng;
      auto x = torch::zeros({1, 3, ocfg.imgsz, ocfg.imgsz});
      m->forward_eval(x);
    }
    m->eval();
    serialization::export_yolo10_onnx(m, onnx_path, ocfg);
  } else if (version == "v11") {
    models::Yolo11Detect m(models::yolo11_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo11_onnx(m, onnx_path, ocfg);
  } else if (version == "v12") {
    models::Yolo12Detect m(models::yolo12_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo12_onnx(m, onnx_path, ocfg);
  } else if (version == "v13") {
    models::Yolo13Detect m(models::yolo13_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo13_onnx(m, onnx_path, ocfg);
  } else if (version == "v26") {
    models::Yolo26Detect m(models::yolo26_scale_from_letter(scale_s), cfg.nc);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo26_onnx(m, onnx_path, ocfg);
  } else {
    // v8 default
    auto bench_scale = models::kYolo8n;
    if      (scale_s == "s") bench_scale = models::kYolo8s;
    else if (scale_s == "m") bench_scale = models::kYolo8m;
    else if (scale_s == "l") bench_scale = models::kYolo8l;
    else if (scale_s == "x") bench_scale = models::kYolo8x;
    models::Yolo8Detect m(bench_scale, cfg.nc);
    m->load_from_state_dict(sd.entries); m->eval();
    serialization::export_yolo8_onnx(m, onnx_path, ocfg);
  }
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
