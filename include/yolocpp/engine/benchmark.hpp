#pragma once
//
// Latency / throughput benchmark across available backends.
//
// We time three configurations end-to-end (preprocess → forward → NMS):
//   • PT       — libtorch eager (FP32 on GPU)
//   • TRT-FP32 — engine built without the FP16 builder flag
//   • TRT-FP16 — engine built with FP16 enabled
//
// Engines are cached at <cache_dir>/yolo8n.<imgsz>.{onnx,fp32.trt,fp16.trt}
// so repeated runs only pay for warmup + measurement.
//

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "yolocpp/inference/predictor.hpp"

namespace yolocpp::engine {

struct BenchConfig {
  std::string weights;            // .pt source
  std::string source;             // image (single jpg/png)
  int         imgsz       = 640;
  int         warmup_iters = 10;
  int         iters       = 100;
  std::string cache_dir   = "build/bench_cache";
  std::string device      = "";
  bool        run_pt      = true;
  bool        run_trt_fp32 = true;
  bool        run_trt_fp16 = true;
};

struct BenchResult {
  std::string backend;
  double      median_ms;
  double      p95_ms;
  double      mean_ms;
  double      throughput_imgps;   // 1000 / median
  int         num_detections;
  // Detections from this backend (used for accuracy delta vs PT).
  std::vector<inference::Detection> dets;
};

std::vector<BenchResult> run_benchmark(const BenchConfig& cfg);

// Pretty-print a benchmark table to stdout.
void print_benchmark(const std::vector<BenchResult>& rows);

}  // namespace yolocpp::engine
