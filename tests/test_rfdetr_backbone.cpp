// Pins backbone forward shapes per RF-DETR scale (#65A). The
// backbone is the only RF-DETR slice whose forward path is wired
// today; everything downstream still throws. This test verifies
// the whole ViT runs end-to-end on random input and produces the
// expected multi-scale tap shapes.

#include <torch/torch.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "yolocpp/models/rfdetr.hpp"
#include "yolocpp/models/rfdetr_backbone.hpp"

using yolocpp::models::RFDetr;
using yolocpp::models::rfdetr::backbone_cfg_from_name;

namespace {

void check_backbone(const std::string& letter) {
  torch::NoGradGuard ng;
  torch::manual_seed(0);

  auto scale = yolocpp::models::rfdetr_scale_from_letter(letter);
  RFDetr m(scale, /*nc=*/80);
  m->eval();

  const auto& cfg = backbone_cfg_from_name(scale.backbone);
  int img        = cfg.img_size;
  int grid       = img / cfg.patch_size;

  auto x   = torch::randn({1, 3, img, img});
  auto out = m->forward_backbone(x);

  if (out.size() != cfg.tap_blocks.size()) {
    std::cerr << "[FAIL] " << letter << " tap count: got " << out.size()
              << " expected " << cfg.tap_blocks.size() << "\n";
    std::exit(1);
  }
  for (auto& f : out) {
    if (f.dim() != 4 || f.size(0) != 1 || f.size(1) != cfg.embed_dim ||
        f.size(2) != grid || f.size(3) != grid) {
      std::cerr << "[FAIL] " << letter << " tap shape: " << f.sizes()
                << " expected [1, " << cfg.embed_dim << ", " << grid
                << ", " << grid << "]\n";
      std::exit(1);
    }
    if (!std::isfinite(f.abs().sum().item<float>())) {
      std::cerr << "[FAIL] " << letter << " tap has non-finite values\n";
      std::exit(1);
    }
  }
  std::cout << "[PASS] rfdetr-" << letter << " backbone " << scale.backbone
            << " (depth=" << cfg.depth << ", embed=" << cfg.embed_dim
            << ", img=" << img << ", taps=" << cfg.tap_blocks.size() << ")\n";
}

}  // namespace

int main() {
  // DINOv2-large is gigantic (~300M params, 24 blocks of embed=1024)
  // — instantiating it in a unit test costs ~3 GB of host memory
  // for the parameter tensors alone. Skip in this test; the lw-detr
  // family covers the same forward path.
  for (const char* s : {"n", "s", "b", "m"}) {
    check_backbone(s);
  }
  std::cout << "rfdetr backbone shapes: OK\n";
  return 0;
}
