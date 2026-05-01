#pragma once
//
// Meituan YOLOv6 (v3.0 / release 0.4.0) `.pt` → our `.pt` converter.
//
// Upstream `.pt` files ship in TRAIN form: every RepVGG block is stored
// as three branches:
//   <prefix>.rbr_dense  (3×3 Conv + BN)
//   <prefix>.rbr_1x1    (1×1 Conv + BN)
//   <prefix>.rbr_identity (BN only — present iff in_c==out_c and stride==1)
//
// The converter applies RepVGG re-parameterization to fuse those three
// branches into a single 3×3 Conv with bias, matching our deploy-form
// `Yolo6Impl::RepConvImpl` layout (`<prefix>.conv.{weight,bias}`).
//
// Non-RepVGG modules are stored as `<prefix>.block.conv.weight` etc;
// the converter strips the `.block.` infix to match our naming.
// Other rewrites:
//   * upstream `backbone.ERBlock_2.0.*`  → ours `backbone.ERBlock_2_down.*`
//   * upstream `backbone.ERBlock_2.1.*`  → ours `backbone.ERBlock_2_block.*`
//   * upstream `backbone.ERBlock_5.2.cspsppf.*` → ours `backbone.ERBlock_5_cspsppf.*`
//   * upstream `RepBlock.conv1` and `RepBlock.block.<i>` are kept as-is
//   * upstream stores fp16; we cast to fp32.
//

#include <string>

namespace yolocpp::serialization {

int convert_yolov6_pt(const std::string& src_pt_path,
                      const std::string& out_pt_path,
                      int                nc = 80);

}  // namespace yolocpp::serialization
