#pragma once
//
// Format-agnostic mAP: run any per-image predictor over the dataset's
// letterboxed eval images and score it with the shared `compute_map`. Used by
// the benchmark to measure mAP per export format (TRT fp32/fp16/int8, ONNX)
// rather than reporting one reference number — so e.g. INT8's accuracy drop is
// actually visible. Each format is fed the same imgsz letterboxed image, so its
// detections and the ground truth share one coordinate space.
//
#include <functional>
#include <string>
#include <vector>

#include "yolocpp/metrics/map.hpp"

namespace cv { class Mat; }
namespace yolocpp::inference { struct Detection; struct NMSConfig; }
namespace yolocpp::datasets { class YoloDataset; }

namespace yolocpp::engine {

// (letterboxed BGR imgsz×imgsz, NMS config) → detections in imgsz coords.
using ImagePredictor = std::function<std::vector<inference::Detection>(
    const cv::Mat&, const inference::NMSConfig&)>;

// Score `predict` over `ds`'s eval split. `conf` is the low score floor for the
// PR sweep (Ultralytics uses ~1e-3); `iou` is the NMS IoU.
metrics::mAPResult eval_predictor(const ImagePredictor& predict,
                                  const datasets::YoloDataset& ds, int nc,
                                  float conf = 0.001f, float iou = 0.7f);

// cv::dnn-backed ONNX predictor (OpenCV's ONNX importer — no onnxruntime dep).
// Returns an empty std::function if OpenCV can't load/run the graph; the caller
// treats that as "mAP not measured for ONNX".
ImagePredictor make_onnx_predictor(const std::string& onnx_path, int imgsz, int nc);

}  // namespace yolocpp::engine
