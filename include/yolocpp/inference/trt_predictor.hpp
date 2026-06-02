#pragma once
//
// TensorRT runtime predictor — same interface as the libtorch Predictor
// but the forward pass runs on a deserialized TRT engine.
//

#include <opencv2/core.hpp>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#include "yolocpp/inference/nms.hpp"
#include "yolocpp/inference/predictor.hpp"  // for Detection struct
#include "yolocpp/inference/results.hpp"    // for Results wrapper

namespace yolocpp::inference {

class TrtPredictor {
 public:
  // `max_batch` sizes the device buffers and is what the runtime input
  // shape is set to. Engine must be built with an optimisation profile
  // whose kMAX covers max_batch. Default 1 matches the historical
  // single-image latency path. (Task #88B for true batched throughput.)
  TrtPredictor(const std::string& engine_path,
               int                imgsz = 640,
               int                max_batch = 1,
               std::string        input_name  = "images",
               std::string        output_name = "output");

  ~TrtPredictor();

  std::vector<Detection> predict(const cv::Mat& bgr,
                                 NMSConfig conf = {}) const;

  // True batched inference. `bgrs.size()` must be <= max_batch given to
  // the constructor. Returns per-image detections. Latency for the
  // whole call is what the benchmark measures; throughput =
  // bgrs.size() / latency.
  std::vector<std::vector<Detection>>
  predict_batch(const std::vector<cv::Mat>& bgrs,
                NMSConfig conf = {}) const;

  // Ultralytics-API parity entry point (#97). Returns a Results object
  // wrapping boxes + the original image + speed timings + names. Same
  // detections as predict() — convenience wrapper. `names` is optional
  // (defaults to empty → class ids stringify).
  Results predict_results(const cv::Mat& bgr,
                          const std::vector<std::string>& names = {},
                          NMSConfig conf = {}) const;

  // Batched Results — one per input image.
  std::vector<Results>
  predict_results_batch(const std::vector<cv::Mat>& bgrs,
                        const std::vector<std::string>& names = {},
                        NMSConfig conf = {}) const;

  std::vector<Detection> predict_to_file(const std::string& in_path,
                                         const std::string& out_path,
                                         NMSConfig conf = {},
                                         const std::vector<std::string>& names = {}) const;

  int imgsz() const { return imgsz_; }
  int max_batch() const { return max_batch_; }

 private:
  struct Impl;                    // hides nvinfer types from header consumers
  std::unique_ptr<Impl> impl_;
  int                  imgsz_;
  int                  max_batch_;
};

}  // namespace yolocpp::inference
