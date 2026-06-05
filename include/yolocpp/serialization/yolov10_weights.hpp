#pragma once
//
// THU-MIG YOLOv10 `.pt` → our `.pt` converter (consumes the upstream form).
//
// Two transformations:
// 1. Drop the one2many `cv2`/`cv3` keys at `model.<head_idx>.cv2.*` /
//    `.cv3.*` (training-only). Rename `one2one_cv2.*` → `cv2.*` and
//    `one2one_cv3.*` → `cv3.*` so they fit our standard
//    `Detect(legacy=false)` slots.
// 2. Fuse RepVGGDW pairs found at `<prefix>.conv.*` (7×7 dwconv) +
//    `<prefix>.conv1.*` (3×3 dwconv) into a single 7×7 dwconv with bias.
//    Each branch already has its own BN; both fold in via the standard
//    `(γ/√(σ²+ε)) · W` scale-and-shift fusion. The 3×3 weight is
//    zero-padded to 7×7 before summing.
//
// fp16 → fp32. `num_batches_tracked` dropped.
//

#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace yolocpp::serialization {

int convert_yolov10_pt(const std::string& src_pt_path,
                       const std::string& out_pt_path,
                       int                nc = 80);

// In-memory reparameterization: raw upstream training-form v10 state-dict
// (RepVGGDW conv/conv1 + BN, dual one2one/one2many head) → the deploy form
// our Yolo10Impl loads. Used by the converter above and by
// Yolo10Impl::load_from_state_dict so the original `.pt` loads directly.
std::vector<std::pair<std::string, at::Tensor>>
reparam_yolov10(const std::vector<std::pair<std::string, at::Tensor>>& src);

// Dual-head variant — keeps the upstream one2many head as a parallel
// module rooted at `o2m_detect.cv2.*` / `o2m_detect.cv3.*` (the v8-style
// legacy=true Detect we register on `Yolo10Impl` when `dual_head=true`).
// `one2one_cv2/3` still renames to `cv2/3` inside `model[head_idx].*`.
// Used only by `mode=train ... dual_head=true` flows; the deploy path
// continues to use `convert_yolov10_pt` (which strips the one2many head).
int convert_yolov10_dual_pt(const std::string& src_pt_path,
                            const std::string& out_pt_path,
                            int                nc = 80);

}  // namespace yolocpp::serialization
