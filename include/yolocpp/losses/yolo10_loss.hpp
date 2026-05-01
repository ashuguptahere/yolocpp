#pragma once
//
// V10DualLoss — Wang et al. 2024 §3.1 consistent dual-assignment.
//
// Trains v10's two parallel heads (one2many + one2one) under a SHARED
// metric (CIoU + classification weighting from V8DetectionLoss / TAL)
// but with different topK assignments:
//   - one2many head: TAL with topk=10 (rich gradient, drives backbone).
//   - one2one  head: TAL with topk=1  (forces a single positive per GT
//                    so the deploy graph is NMS-free).
//
// The "consistent" property comes from BOTH heads sharing the V8 TAL
// metric definition; the topk difference is the only knob. The two
// losses are independent — gradients flow through their own weights
// and into the shared backbone/neck via summation.
//
// Input feats vector layout (matches Yolo10Impl::forward_train when
// `dual_head=true`):
//   feats[0..2] : one2many P3..P5 raw [B, 4*reg_max+nc, H_i, W_i]
//   feats[3..5] : one2one  P3..P5 raw, same layout
//

#include <torch/torch.h>

#include "yolocpp/losses/yolo8_loss.hpp"

namespace yolocpp::losses {

struct V10DualLossConfig {
  // Shared LossConfig overrides (nc, reg_max, gain weights). The topk
  // fields below override LossConfig.topk on the respective branch.
  LossConfig base;
  int  topk_o2m = 10;
  int  topk_o2o = 1;
};

class V10DualLoss {
 public:
  explicit V10DualLoss(V10DualLossConfig cfg = {});

  // Returns combined LossOutput where:
  //   .total = o2m.total + o2o.total
  //   .box   = o2m.box   + o2o.box
  //   .cls   = o2m.cls   + o2o.cls
  //   .dfl   = o2m.dfl   + o2o.dfl
  // (matches v10 paper's Equation 4 — equal-weight sum.)
  LossOutput operator()(const std::vector<torch::Tensor>& feats,
                        const torch::Tensor& targets,
                        const std::vector<double>& strides,
                        int imgsz) const;

  const V10DualLossConfig& config() const { return cfg_; }

 private:
  V10DualLossConfig cfg_;
  V8DetectionLoss   o2m_;
  V8DetectionLoss   o2o_;
};

// Yolo10 trainer adapter — picks V10DualLoss when feats.size()==6,
// falls back to V8DetectionLoss (one2one only) when feats.size()==3.
// The runtime branch lets us reuse the existing single-head training
// path (V8DetectionLoss) without forcing the dual-head one2many parameters
// to exist when not requested.
class Yolo10LossAdapter {
 public:
  explicit Yolo10LossAdapter(LossConfig base);

  LossOutput operator()(const std::vector<torch::Tensor>& feats,
                        const torch::Tensor& targets,
                        const std::vector<double>& strides,
                        int imgsz) const;

 private:
  V8DetectionLoss single_;
  V10DualLoss     dual_;
};

}  // namespace yolocpp::losses
