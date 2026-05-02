// #65A2 — pins the real HF-DINOv2 backbone forward shape (small
// variant, C=384, depth=12, patch=14). Exercises the parameter
// dotted-paths the loader uses (`embeddings.cls_token`,
// `embeddings.patch_embeddings.projection.weight`,
// `encoder.layer.0.attention.attention.query.weight`, etc.).

#include <torch/torch.h>

#include <iostream>

#include "yolocpp/models/rfdetr_backbone.hpp"

using yolocpp::models::rfdetr::Dinov2Cfg;
using yolocpp::models::rfdetr::Dinov2Model;
using yolocpp::models::rfdetr::Dinov2WrapperOuter;

int main() {
  torch::NoGradGuard ng;
  torch::manual_seed(0);

  Dinov2Cfg cfg = yolocpp::models::rfdetr::kDinov2Small;
  cfg.patch_size = 14;
  cfg.pretrain_grid = 16;  // smaller for test speed
  cfg.num_layers = 4;       // smaller depth for test speed
  cfg.tap_blocks = {0, 1, 2, 3};

  Dinov2Model m(cfg);
  m->eval();

  // Verify the parameter dotted-paths exist with the right shapes.
  auto params = m->named_parameters();
  std::vector<std::string> required = {
      "embeddings.cls_token",
      "embeddings.mask_token",
      "embeddings.position_embeddings",
      "embeddings.patch_embeddings.projection.weight",
      "embeddings.patch_embeddings.projection.bias",
      "encoder.layer.0.norm1.weight",
      "encoder.layer.0.attention.attention.query.weight",
      "encoder.layer.0.attention.attention.key.weight",
      "encoder.layer.0.attention.attention.value.weight",
      "encoder.layer.0.attention.output.dense.weight",
      "encoder.layer.0.layer_scale1.lambda1",
      "encoder.layer.0.norm2.weight",
      "encoder.layer.0.mlp.fc1.weight",
      "encoder.layer.0.mlp.fc2.weight",
      "encoder.layer.0.layer_scale2.lambda1",
      "layernorm.weight",
      "layernorm.bias",
  };
  for (const auto& key : required) {
    if (!params.contains(key)) {
      std::cerr << "[FAIL] backbone missing param '" << key << "'\n";
      std::cerr << "  available keys (first 25):\n";
      int n = 0;
      for (const auto& kv : params) {
        if (n++ >= 25) break;
        std::cerr << "    " << kv.key() << "\n";
      }
      return 1;
    }
  }

  // Forward at a divisible-by-patch resolution. Pretrain grid is
  // 16×16; input grid here is 4×4 so position embedding is
  // bicubic-interpolated.
  auto x = torch::randn({1, 3, 4 * cfg.patch_size, 4 * cfg.patch_size});
  auto taps = m->forward(x);
  if (taps.size() != cfg.tap_blocks.size()) {
    std::cerr << "[FAIL] taps " << taps.size() << " expected "
              << cfg.tap_blocks.size() << "\n";
    return 1;
  }
  for (auto& t : taps) {
    if (t.dim() != 4 || t.size(0) != 1 || t.size(1) != cfg.hidden_size ||
        t.size(2) != 4 || t.size(3) != 4) {
      std::cerr << "[FAIL] tap shape " << t.sizes() << " expected [1, "
                << cfg.hidden_size << ", 4, 4]\n";
      return 1;
    }
  }
  if (!std::isfinite(taps.back().abs().sum().item<float>())) {
    std::cerr << "[FAIL] non-finite tap output\n";
    return 1;
  }

  // The wrapper-outer / wrapper / model nesting reproduces upstream's
  // `backbone.0.encoder.encoder.embeddings.*` path. Verify the
  // OUTER wrapper exposes that.
  Dinov2WrapperOuter outer(cfg);
  auto outer_params = outer->named_parameters();
  if (!outer_params.contains("encoder.encoder.embeddings.cls_token")) {
    std::cerr << "[FAIL] wrapper-outer path missing\n";
    return 1;
  }

  std::cout << "[PASS] dinov2 backbone (C=" << cfg.hidden_size
            << ", depth=" << cfg.num_layers << ", taps="
            << taps.size() << "); wrapper exposes "
            << "encoder.encoder.embeddings.cls_token\n";
  return 0;
}
