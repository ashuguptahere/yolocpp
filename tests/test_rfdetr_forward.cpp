// Pins full RF-DETR forward (backbone + encoder + decoder + head)
// shape contract per scale (#65C). With weights uninitialised the
// numerical output is meaningless, but the YOLO-style channel order
// `[B, 4+nc, num_queries]` is the contract downstream NMS-free
// decoding (#65E) relies on.

#include <torch/torch.h>

#include <iostream>

#include "yolocpp/models/rfdetr.hpp"
#include "yolocpp/models/rfdetr_backbone.hpp"

using yolocpp::models::RFDetr;
using yolocpp::models::rfdetr::backbone_cfg_from_name;

namespace {

void check_forward(const std::string& letter) {
  torch::NoGradGuard ng;
  torch::manual_seed(0);

  auto scale = yolocpp::models::rfdetr_scale_from_letter(letter);
  RFDetr m(scale, /*nc=*/80);
  m->eval();

  // Old scaffold's backbone is now placeholder-only (see #65A2).
  // Run forward at a hard-coded letterbox size; the real
  // per-variant resolution is `scale.resolution`.
  (void)backbone_cfg_from_name;
  auto x = torch::randn({1, 3, 640, 640});

  // Eval — YOLO contract.
  auto out = m->forward_eval(x);
  if (out.dim() != 3 || out.size(0) != 1 || out.size(1) != 4 + 80 ||
      out.size(2) != scale.num_queries) {
    std::cerr << "[FAIL] " << letter << " forward_eval shape: "
              << out.sizes() << " expected [1, " << 4 + 80 << ", "
              << scale.num_queries << "]\n";
    std::exit(1);
  }
  // forward_eval matches the YOLO contract: bbox is xyxy in PIXEL
  // coords (caller's input H/W); cls is sigmoided in [0, 1].
  auto cls_slice = out.slice(1, 4);
  auto c_lo = cls_slice.min().item<float>();
  auto c_hi = cls_slice.max().item<float>();
  if (c_lo < 0.0f || c_hi > 1.0f) {
    std::cerr << "[FAIL] " << letter << " cls out of [0,1]: "
              << c_lo << ", " << c_hi << "\n";
    std::exit(1);
  }
  // bbox channels: x1,y1 in [-imgsz, imgsz]; x2,y2 too. Just check
  // finiteness; range will be tight once weights load.
  if (!std::isfinite(out.slice(1, 0, 4).abs().sum().item<float>())) {
    std::cerr << "[FAIL] " << letter << " bbox non-finite\n";
    std::exit(1);
  }

  // Train — per-layer logits + bbox deltas.
  auto outs = m->forward_train(x);
  if (static_cast<int>(outs.size()) != scale.num_dec_layers * 2) {
    std::cerr << "[FAIL] " << letter << " forward_train layer count: "
              << outs.size() << " expected "
              << scale.num_dec_layers * 2 << "\n";
    std::exit(1);
  }
  std::cout << "[PASS] rfdetr-" << letter
            << " forward Q=" << scale.num_queries
            << " hidden=" << scale.hidden_dim
            << " dec_layers=" << scale.num_dec_layers << "\n";
}

}  // namespace

int main() {
  for (const char* s : {"n", "s"}) check_forward(s);
  std::cout << "rfdetr forward shapes: OK\n";
  return 0;
}
