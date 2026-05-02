// Pins encoder forward shapes per RF-DETR scale (#65B). Backbone +
// encoder run end-to-end on random input; decoder/head still throws.

#include <torch/torch.h>

#include <iostream>

#include "yolocpp/models/rfdetr.hpp"
#include "yolocpp/models/rfdetr_backbone.hpp"

using yolocpp::models::RFDetr;
using yolocpp::models::rfdetr::backbone_cfg_from_name;

namespace {

void check_encoder(const std::string& letter) {
  torch::NoGradGuard ng;
  torch::manual_seed(0);

  auto scale = yolocpp::models::rfdetr_scale_from_letter(letter);
  RFDetr m(scale, /*nc=*/80);
  m->eval();

  const auto& cfg = backbone_cfg_from_name(scale.backbone);
  // Backbone's positional embedding is sized for cfg.img_size, so
  // that's the only size guaranteed to match without interpolation.
  // Pos-embed interpolation for variable-size inference lands later
  // (it's not on the #65 critical path for now).
  int side = cfg.img_size;
  int grid = side / cfg.patch_size;

  auto x   = torch::randn({1, 3, side, side});
  auto out = m->forward_encoder(x);

  int64_t expected_tokens = static_cast<int64_t>(grid) * grid *
                            cfg.tap_blocks.size();
  if (out.memory.dim() != 3 || out.memory.size(0) != 1 ||
      out.memory.size(1) != expected_tokens ||
      out.memory.size(2) != scale.hidden_dim) {
    std::cerr << "[FAIL] " << letter << " memory shape: "
              << out.memory.sizes() << " expected [1, " << expected_tokens
              << ", " << scale.hidden_dim << "]\n";
    std::exit(1);
  }
  if (!std::isfinite(out.memory.abs().sum().item<float>())) {
    std::cerr << "[FAIL] " << letter << " memory has non-finite values\n";
    std::exit(1);
  }
  if (out.spatial_shapes.size(0) !=
          static_cast<int64_t>(cfg.tap_blocks.size()) ||
      out.spatial_shapes.size(1) != 2) {
    std::cerr << "[FAIL] " << letter << " spatial_shapes shape\n";
    std::exit(1);
  }
  std::cout << "[PASS] rfdetr-" << letter
            << " encoder hidden=" << scale.hidden_dim
            << " layers=" << scale.num_encoder_layers
            << " tokens=" << expected_tokens << "\n";
}

}  // namespace

int main() {
  // n/s only — b/m at 224×224 with 10/12-block backbones still
  // costs >2 GB. Bigger scales are exercised by #65L's full smoke
  // suite once weights land.
  for (const char* s : {"n", "s"}) check_encoder(s);
  std::cout << "rfdetr encoder shapes: OK\n";
  return 0;
}
