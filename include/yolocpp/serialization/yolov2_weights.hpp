#pragma once
//
// Darknet `yolov2.weights` / `yolov2-tiny.weights` → our `.pt` state-
// dict converter. Lets us run YOLOv2 inference without any Darknet
// runtime dependency.
//
// Header layout (pjreddie / AlexeyAB, "NEW" format, v2-onwards):
//   int32 major, int32 minor, int32 revision           (12 B)
//   int64 seen                                          (8 B if major*10+minor >= 2)
//   int32 seen (legacy)                                 (4 B otherwise)
//
// Body: for each conv layer in cfg order, with batch_normalize=1:
//   bn_bias[out_c], bn_scale[out_c], bn_mean[out_c], bn_var[out_c],
//   conv_weight[out_c · in_c · kh · kw]
// For the final 1×1 detection conv (batch_normalize=0):
//   bias[out_c], conv_weight[out_c · in_c · 1 · 1]
// No fully-connected layers in v2.
//

#include <string>

#include "yolocpp/models/yolo2.hpp"

namespace yolocpp::serialization {

// Reads `weights_path`, constructs a Yolo2Impl(scale, nc), copies the
// streamed conv blocks into it, and writes the resulting state-dict
// to `out_pt_path`. Returns the number of conv blocks consumed.
int convert_yolov2_weights(const std::string& weights_path,
                            const std::string& out_pt_path,
                            int                nc    = 20,
                            models::Yolo2Scale scale = models::Yolo2Scale::Full);

}  // namespace yolocpp::serialization
