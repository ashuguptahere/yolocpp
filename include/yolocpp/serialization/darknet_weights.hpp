#pragma once
//
// Darknet `.weights` binary → our PyTorch `.pt` state-dict converter.
//
// AlexeyAB's yolov4.weights ships as:
//   header: int32 major, int32 minor, int32 revision, int64 seen      (20 B)
//   body  : flat float32 blob, one [convolutional] block at a time in
//           yolov4.cfg execution order. Each block is:
//             with batch_normalize:  bn_bias, bn_scale, bn_mean, bn_var,
//                                    conv_weight (out_c, in_c, kh, kw)
//             without batch_normalize: bias, conv_weight
//
// Yolo4Impl's modules are registered in matching cfg order, so a DFS over
// registered children visits conv blocks in the same order Darknet wrote
// them. We walk the model, pull bytes per block, fill PyTorch tensors,
// then write a state-dict via save_state_dict() that the existing
// pt_loader can read back like any other upstream-format weight file.
//

#include <string>

namespace yolocpp::serialization {

// Read yolov4.weights from `weights_path`, build a Yolo4Impl(nc), copy
// weights into it, and write the resulting state-dict to `out_pt_path`.
// Returns the number of conv blocks consumed. Throws on header mismatch,
// short read, or trailing bytes.
int convert_yolov4_weights(const std::string& weights_path,
                           const std::string& out_pt_path,
                           int                nc = 80);

}  // namespace yolocpp::serialization
