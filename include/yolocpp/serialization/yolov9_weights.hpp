#pragma once
//
// YOLOv9 `.pt` → our `.pt` converter (consumes the upstream form).
//
// Upstream's `RepConv` ships in train form: each block has two parallel
// convolutions (`conv1` 3×3 + `conv2` 1×1), each with its own BN. We
// fuse them into a single 3×3 Conv with bias for deploy:
//   * `<prefix>.conv1.conv.weight` (3×3) + `<prefix>.conv1.bn.{...}`
//   * `<prefix>.conv2.conv.weight` (1×1) + `<prefix>.conv2.bn.{...}`
//   * optional `<prefix>.bn.{...}` (identity branch, only when c1==c2 &&
//     stride==1 && bn=True; rare in v9c — upstream builds RepConv with
//     bn=False by default)
//
// Result is written as `<prefix>.conv.weight` and `<prefix>.conv.bias`.
// All other keys pass through unchanged (model module names already
// match because both upstream and our `Yolo9Impl` use a flat ModuleList
// named "model"). fp16 → fp32, num_batches_tracked dropped.
//

#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace yolocpp::serialization {

int convert_yolov9_pt(const std::string& src_pt_path,
                      const std::string& out_pt_path,
                      int                nc = 80);

// In-memory reparameterization: transform a raw upstream/Ultralytics
// training-form state-dict (RepConv conv1/conv2 + BN) into the deploy form
// our Yolo9Impl loads (`<prefix>.conv.weight` / `.conv.bias`). Used both by
// the offline converter above and by Yolo9Impl::load_from_state_dict, so the
// original `.pt` loads directly. Pass-through for already-deploy checkpoints.
std::vector<std::pair<std::string, at::Tensor>>
reparam_yolov9(const std::vector<std::pair<std::string, at::Tensor>>& src);

}  // namespace yolocpp::serialization
