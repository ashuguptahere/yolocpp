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

  const auto& cfg = backbone_cfg_from_name(scale.backbone);
  auto x = torch::randn({1, 3, cfg.img_size, cfg.img_size});

  // Eval — YOLO contract.
  auto out = m->forward_eval(x);
  if (out.dim() != 3 || out.size(0) != 1 || out.size(1) != 4 + 80 ||
      out.size(2) != scale.num_queries) {
    std::cerr << "[FAIL] " << letter << " forward_eval shape: "
              << out.sizes() << " expected [1, " << 4 + 80 << ", "
              << scale.num_queries << "]\n";
    std::exit(1);
  }
  // Bbox channels (0..3) sigmoided so values in [0, 1]; cls (4..) too.
  auto b_lo = out.slice(1, 0, 4).min().item<float>();
  auto b_hi = out.slice(1, 0, 4).max().item<float>();
  if (b_lo < 0.0f || b_hi > 1.0f) {
    std::cerr << "[FAIL] " << letter << " bbox out of [0,1]: "
              << b_lo << ", " << b_hi << "\n";
    std::exit(1);
  }

  // Train — per-layer logits + bbox deltas.
  auto outs = m->forward_train(x);
  if (static_cast<int>(outs.size()) != scale.num_decoder_layers * 2) {
    std::cerr << "[FAIL] " << letter << " forward_train layer count: "
              << outs.size() << " expected "
              << scale.num_decoder_layers * 2 << "\n";
    std::exit(1);
  }
  std::cout << "[PASS] rfdetr-" << letter
            << " forward Q=" << scale.num_queries
            << " enc_layers=" << scale.num_encoder_layers
            << " dec_layers=" << scale.num_decoder_layers << "\n";
}

}  // namespace

int main() {
  for (const char* s : {"n", "s"}) check_forward(s);
  std::cout << "rfdetr forward shapes: OK\n";
  return 0;
}
