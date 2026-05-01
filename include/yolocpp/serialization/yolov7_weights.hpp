#pragma once
//
// WongKinYiu YOLOv7 `.pt` → our `.pt` converter.
//
// Upstream `.pt` ships in TRAIN form for the head's RepConv layers (102,
// 103, 104):
//   model.<i>.rbr_dense.0  — 3×3 Conv (no bias)
//   model.<i>.rbr_dense.1  — BN
//   model.<i>.rbr_1x1.0    — 1×1 Conv (no bias)
//   model.<i>.rbr_1x1.1    — BN
//   model.<i>.rbr_identity (BN only, optional — only if c_in==c_out && s==1)
//
// We fuse those into a single 3×3 Conv with bias (deploy form) at
// `model.<i>.conv.{weight,bias}`. The 1×1 branch's W is zero-padded to
// 3×3; the identity branch (when present) is encoded as a per-out-channel
// identity 1×1 Conv before BN-fuse.
//
// Everything else maps 1:1 (key names already match because our
// Yolo7Impl uses a flat ModuleList named "model"). fp16 → fp32.
//

#include <string>

namespace yolocpp::serialization {

int convert_yolov7_pt(const std::string& src_pt_path,
                      const std::string& out_pt_path,
                      int                nc = 80);

}  // namespace yolocpp::serialization
