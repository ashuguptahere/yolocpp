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
