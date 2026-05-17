#pragma once
//
// Darknet `yolov1.weights` (or `yolov1-tiny.weights`) → our `.pt`
// state-dict converter. Lets us run YOLOv1 inference without any
// Darknet runtime dependency — we parse the binary ourselves.
//
// Header layout (pjreddie's Darknet, "OLD" format predating v2):
//   int32 major, int32 minor, int32 revision, int32 seen        (16 B)
//   (no int64 seen — that arrived with v2; v1 uses int32 throughout)
//
// Body: for each conv layer in cfg order:
//   bias[out_c], weight[out_c · in_c · kh · kw]
// Then for each fully-connected layer:
//   bias[outputs], weight[outputs · inputs]   (row-major, no transpose
//   needed — Darknet's `connected_layer` matches PyTorch's
//   `nn.Linear` weight shape `(out_features, in_features)`).
//

#include <string>

namespace yolocpp::serialization {

// Reads `weights_path`, builds a Yolo1Impl(nc), copies in the
// 24 conv blocks + 2 FC layers, and writes the resulting state-dict
// to `out_pt_path`. Returns the total number of streamed (conv+fc)
// blocks. Throws on header mismatch, short read, or trailing bytes.
int convert_yolov1_weights(const std::string& weights_path,
                            const std::string& out_pt_path,
                            int                nc = 20);

}  // namespace yolocpp::serialization
