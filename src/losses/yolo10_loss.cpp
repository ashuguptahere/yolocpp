#include "yolocpp/losses/yolo10_loss.hpp"

#include <stdexcept>

namespace yolocpp::losses {

namespace {
LossConfig with_topk(LossConfig c, int k) {
  c.topk = k;
  return c;
}
}  // namespace

V10DualLoss::V10DualLoss(V10DualLossConfig cfg)
    : cfg_(cfg),
      o2m_(with_topk(cfg.base, cfg.topk_o2m)),
      o2o_(with_topk(cfg.base, cfg.topk_o2o)) {}

Yolo10LossAdapter::Yolo10LossAdapter(LossConfig base)
    : single_(base),
      dual_([&]() {
        V10DualLossConfig c;
        c.base = base;
        return c;
      }()) {}

LossOutput Yolo10LossAdapter::operator()(
    const std::vector<torch::Tensor>& feats,
    const torch::Tensor& targets,
    const std::vector<double>& strides,
    int imgsz) const {
  if (feats.size() == 6) return dual_(feats, targets, strides, imgsz);
  return single_(feats, targets, strides, imgsz);
}

LossOutput V10DualLoss::operator()(
    const std::vector<torch::Tensor>& feats,
    const torch::Tensor& targets,
    const std::vector<double>& strides,
    int imgsz) const {
  if (feats.size() != 6) {
    throw std::runtime_error(
        "V10DualLoss expects 6 feature tensors "
        "(one2many P3..P5 + one2one P3..P5); got " +
        std::to_string(feats.size()));
  }
  std::vector<torch::Tensor> o2m_feats(feats.begin(), feats.begin() + 3);
  std::vector<torch::Tensor> o2o_feats(feats.begin() + 3, feats.end());

  auto out_m = o2m_(o2m_feats, targets, strides, imgsz);
  auto out_o = o2o_(o2o_feats, targets, strides, imgsz);

  LossOutput total;
  total.total = out_m.total + out_o.total;
  total.box   = out_m.box   + out_o.box;
  total.cls   = out_m.cls   + out_o.cls;
  total.dfl   = out_m.dfl   + out_o.dfl;
  return total;
}

}  // namespace yolocpp::losses
