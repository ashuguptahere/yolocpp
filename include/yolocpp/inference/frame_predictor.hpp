#pragma once
//
// Frame-level predictor abstraction (#51C2).
//
// Image-mode predict loads weights, runs once, and writes a JPG.
// Video / URL / Webcam mode loads weights ONCE and runs many frames
// through the same model — passing a path through `predict_to_file`
// per frame would re-load the `.pt` every call.
//
// `FramePredictor` is the thin polymorphic boundary between
// per-version model holders and the CLI's frame loop. The registry
// hands out `unique_ptr<FramePredictor>` instances; the loop calls
// `predict(frame, nm)` per decoded frame.

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

#include "yolocpp/inference/nms.hpp"
#include "yolocpp/inference/predictor.hpp"  // Detection

namespace yolocpp::inference {

class FramePredictor {
 public:
  virtual ~FramePredictor() = default;
  virtual std::vector<Detection> predict(const cv::Mat& frame,
                                          NMSConfig nm = {}) = 0;
};

}  // namespace yolocpp::inference
