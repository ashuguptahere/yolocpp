#pragma once
//
// yolov3u.pt → our yolo3.pt converter (consumes the upstream form).
//
// No fusion needed (the v3 deploy form has no RepConv branches).
// Just casts fp16 → fp32 and drops `num_batches_tracked` for a uniform
// round-trip through our pt_save / pt_loader.
//

#include <string>

namespace yolocpp::serialization {

int convert_yolov3_pt(const std::string& src_pt_path,
                      const std::string& out_pt_path,
                      int                nc = 80);

}  // namespace yolocpp::serialization
