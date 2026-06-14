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

namespace yolocpp::datasets { class YoloDataset; }

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
  bool        run_trt_int8 = false;  // requires `int8_calib_dir`
  bool        run_onnx     = false;  // ONNX (cv::dnn) timing/size row
  // YOLO version + scale dispatch. When empty, inferred from the
  // weights filename (`cli::version_from_filename` / `scale_from_filename`).
  // The PT path constructs the correct model (Yolo3/4/5/6/7/8/9/10/11/
  // 12/13/26 detect) and the ONNX export step calls the matching
  // `export_yolo*_onnx`. TRT inference is graph-agnostic (TrtPredictor).
  std::string version;            // "v3", "v5", "v6", ..., "v26"
  std::string scale;              // "n", "s", "m", "l", "x", or version-specific
  int         nc = 80;
  // Batch size used for TRT engine + inference. b=1 measures
  // single-image latency (the legacy default); larger batches
  // measure peak throughput. PT path always runs single-image
  // because we time the full preprocess→forward→NMS pipeline.
  int         batch_size = 1;
  // INT8 calibration image directory. Required when `run_trt_int8`
  // is true. Standard choice: val split of the deploy dataset.
  std::string int8_calib_dir = "";
  std::string int8_calib_cache = "";
  // When set, each non-PT format is scored over this eval set (per-format
  // mAP). PT mAP is filled by the caller (registry validator). Null ⇒ speed
  // only.
  const datasets::YoloDataset* eval_ds = nullptr;
  int nc_eval = 80;
};

struct BenchResult {
  std::string backend;
  double      median_ms = 0.0;
  double      p95_ms = 0.0;
  double      mean_ms = 0.0;
  double      throughput_imgps = 0.0;   // 1000 / median
  int         num_detections = 0;
  double      size_mb = 0.0;      // on-disk size of this format's artefact
  std::string artifact;           // the file backing this format (.pt/.trt/.onnx)
  double      map_50    = -1.0;    // per-format mAP (<0 = not measured)
  double      map_50_95 = -1.0;
  // Detections from this backend (used for accuracy delta vs PT).
  std::vector<inference::Detection> dets;
};

std::vector<BenchResult> run_benchmark(const BenchConfig& cfg);

// Pretty-print a benchmark table to stdout.
void print_benchmark(const std::vector<BenchResult>& rows);

}  // namespace yolocpp::engine
