#pragma once
//
// Internal helpers shared between `engine/benchmark.cpp` (the public
// benchmark entry point) and `registry/version_registry.cpp` (the
// per-version `benchmark_pt` adapter hook).
//
// `GenericPredictor<Holder>` and `bench_one()` live here because:
//   - they're templated (must be visible at the registry's TU);
//   - they wrap the same letterbox → forward_eval → NMS → scale_boxes
//     pipeline as `inference::Predictor`, but generic over any holder
//     whose `forward_eval` returns `[B, 4+nc, A]` (xyxy + sigmoided
//     cls). That covers every detect-shape model (v3..v13, v26).
//
// Not part of the public API surface; included by core TUs only.

#include <chrono>
#include <opencv2/core.hpp>
#include <torch/torch.h>
#include <utility>
#include <vector>

#include "yolocpp/engine/benchmark.hpp"
#include "yolocpp/inference/letterbox.hpp"
#include "yolocpp/inference/nms.hpp"
#include "yolocpp/models/yolo8.hpp"  // fuse_model()

namespace yolocpp::engine::detail {

inline double percentile(std::vector<double> xs, double p) {
  if (xs.empty()) return 0.0;
  std::sort(xs.begin(), xs.end());
  const double idx = (p / 100.0) * (xs.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(idx);
  const std::size_t hi = std::min(lo + 1, xs.size() - 1);
  const double frac = idx - lo;
  return xs[lo] + (xs[hi] - xs[lo]) * frac;
}

inline double mean(const std::vector<double>& xs) {
  if (xs.empty()) return 0.0;
  double s = 0.0;
  for (auto v : xs) s += v;
  return s / xs.size();
}

template <typename Pred>
BenchResult bench_one(const std::string& name, const cv::Mat& img,
                      Pred& predictor, int warmup, int iters) {
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
  r.backend          = name;
  r.median_ms        = percentile(times_ms, 50);
  r.p95_ms           = percentile(times_ms, 95);
  r.mean_ms          = mean(times_ms);
  r.throughput_imgps = 1000.0 / r.median_ms;
  r.num_detections   = static_cast<int>(last_dets.size());
  r.dets             = std::move(last_dets);
  return r;
}

// Generic PT predictor adapter — wraps any model holder whose
// `forward_eval` returns `[B, 4+nc, A]` (xyxy + sigmoided cls).
// Mirrors `inference::Predictor::predict`'s pipeline.
template <typename ModelHolder>
struct GenericPredictor {
  ModelHolder model;
  int         imgsz;
  torch::Device device = torch::Device(torch::kCPU);

  GenericPredictor(ModelHolder m, int sz, torch::Device dev)
      : model(std::move(m)), imgsz(sz), device(dev) {
    model->to(device);
    model->eval();
    // Fuse Conv+BN for the eval-only PT path (#95B). Ultralytics calls
    // `model.fuse()` at predict-time which folds BN's running stats
    // into the preceding conv's weight/bias; without it our PT FP32
    // path was ~40% slower than theirs. `fuse_model` is recursive and
    // a no-op for module types without a `fuse()` method (e.g. v6's
    // ConvBNReLU — those still go through the unfused path until a
    // version-specific fuser lands).
    models::fuse_model(*model);
  }

  std::vector<inference::Detection>
  predict(const cv::Mat& bgr,
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
      d.conf = a[i][4]; d.cls = static_cast<int>(a[i][5]);
      result.push_back(d);
    }
    return result;
  }
};

inline torch::Device pick_device(const std::string& d) {
  if (d == "cpu") return torch::Device(torch::kCPU);
  if (d == "cuda" || d.empty()) {
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                       : torch::Device(torch::kCPU);
  }
  return torch::Device(torch::kCPU);
}

}  // namespace yolocpp::engine::detail
