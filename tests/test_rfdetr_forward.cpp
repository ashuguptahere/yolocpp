// #65F2 — pins the real-arch RFDetrImpl::forward_eval shape contract.
// With random init the detection values aren't meaningful but the
// shape `[B, 4+nc, num_queries]` (xyxy in pixel coords + sigmoided
// cls) is what `inference::nms` and `rfdetr_decode` consume.

#include <torch/torch.h>

#include <iostream>

#include "yolocpp/models/rfdetr.hpp"

using yolocpp::models::RFDetr;

namespace {

void check_eval(const std::string& letter) {
  torch::NoGradGuard ng;
  torch::manual_seed(0);
  auto scale = yolocpp::models::rfdetr_scale_from_letter(letter);
  RFDetr m(scale, /*nc=*/80);
  m->eval();

  // Use the pretrain grid × patch_size so the position embedding
  // doesn't need interpolation.
  int side = scale.pretrain_grid * scale.patch_size;
  auto x = torch::randn({1, 3, side, side});
  auto out = m->forward_eval(x);
  if (out.dim() != 3 || out.size(0) != 1 ||
      out.size(1) != 4 + 80 || out.size(2) != scale.num_queries) {
    std::cerr << "[FAIL] " << letter << " forward_eval shape: "
              << out.sizes() << " expected [1, " << 4 + 80 << ", "
              << scale.num_queries << "]\n";
    std::exit(1);
  }
  // cls channels (4..) sigmoided so values in [0, 1].
  auto cls_lo = out.slice(1, 4).min().item<float>();
  auto cls_hi = out.slice(1, 4).max().item<float>();
  if (cls_lo < 0.0f || cls_hi > 1.0f) {
    std::cerr << "[FAIL] " << letter << " cls out of [0,1]: "
              << cls_lo << ", " << cls_hi << "\n";
    std::exit(1);
  }
  if (!std::isfinite(out.abs().sum().item<float>())) {
    std::cerr << "[FAIL] " << letter << " non-finite output\n";
    std::exit(1);
  }
  std::cout << "[PASS] rfdetr-" << letter
            << " forward_eval Q=" << scale.num_queries
            << " hidden=" << scale.hidden_dim
            << " dec_layers=" << scale.num_dec_layers << "\n";
}

}  // namespace

int main() {
  // Only nano (smallest) — the bigger variants instantiate ~30M
  // params each which makes the unit test slow + memory-heavy.
  // The full forward shape is identical across variants by
  // architecture; bigger variants are exercised by #65L's parity
  // suite.
  check_eval("n");
  std::cout << "rfdetr forward_eval shape: OK\n";
  return 0;
}
