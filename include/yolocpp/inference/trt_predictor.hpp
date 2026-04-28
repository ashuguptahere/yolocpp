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
  TrtPredictor(const std::string& engine_path,
               int                imgsz = 640,
               std::string        input_name  = "images",
               std::string        output_name = "output");

  ~TrtPredictor();

  std::vector<Detection> predict(const cv::Mat& bgr,
                                 NMSConfig conf = {}) const;

  std::vector<Detection> predict_to_file(const std::string& in_path,
                                         const std::string& out_path,
                                         NMSConfig conf = {},
                                         const std::vector<std::string>& names = {}) const;

  int imgsz() const { return imgsz_; }

 private:
  struct Impl;                    // hides nvinfer types from header consumers
  std::unique_ptr<Impl> impl_;
  int                  imgsz_;
};

}  // namespace yolocpp::inference
