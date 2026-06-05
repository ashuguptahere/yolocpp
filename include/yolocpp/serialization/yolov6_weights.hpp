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
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace yolocpp::serialization {

int convert_yolov6_pt(const std::string& src_pt_path,
                      const std::string& out_pt_path,
                      int                nc = 80);

// In-memory reparameterization: raw Meituan training-form v6 state-dict
// (RepVGG rbr_dense/rbr_1x1/rbr_identity + upstream module naming) → the
// deploy form our Yolo6Impl loads (fused `.conv` + our prefixes). Used by the
// converter above and by Yolo6Impl::load_from_state_dict so the original `.pt`
// loads directly. Pass-through for already-deploy checkpoints.
std::vector<std::pair<std::string, at::Tensor>>
reparam_yolov6(const std::vector<std::pair<std::string, at::Tensor>>& src);

}  // namespace yolocpp::serialization
