// Tests for the RF-DETR set loss (#65F):
//   1. Hungarian matcher correctness — known small cases.
//   2. Full set-loss runs end-to-end on the random-init RF-DETR
//      forward and produces finite gradients.

#include <torch/torch.h>

#include <iostream>
#include <vector>

#include "yolocpp/losses/hungarian.hpp"
#include "yolocpp/losses/rfdetr_loss.hpp"
#include "yolocpp/models/rfdetr.hpp"
#include "yolocpp/models/rfdetr_backbone.hpp"

using yolocpp::losses::hungarian_assign;
using yolocpp::losses::RFDetrTarget;
using yolocpp::losses::rfdetr_set_loss;
using yolocpp::models::RFDetr;
using yolocpp::models::rfdetr::backbone_cfg_from_name;

namespace {

void test_matcher() {
  // Known optimum: identity assignment costs 1+5+9 = 15;
  // anti-identity costs 3+5+7 = 15. Algorithm is correct as long as
  // it picks any optimum.
  std::vector<float> cost = {1, 2, 3,
                              4, 5, 6,
                              7, 8, 9};
  auto a = hungarian_assign(cost.data(), 3, 3);
  // Each col should be assigned to a unique row.
  std::vector<int> seen(3, 0);
  float total = 0.0f;
  for (int j = 0; j < 3; ++j) {
    if (a[j] < 0 || a[j] >= 3) {
      std::cerr << "[FAIL] matcher: col " << j << " unassigned\n";
      std::exit(1);
    }
    seen[a[j]]++;
    total += cost[a[j] * 3 + j];
  }
  for (int r : seen) {
    if (r != 1) {
      std::cerr << "[FAIL] matcher: row used " << r << " times\n";
      std::exit(1);
    }
  }
  if (std::abs(total - 15.0f) > 1e-6f) {
    std::cerr << "[FAIL] matcher: optimal cost expected 15, got " << total
              << "\n";
    std::exit(1);
  }

  // Rectangular case: 5 rows, 2 cols. Pick the cheapest row per col.
  std::vector<float> rect = {10, 20,
                              1,  2,
                              30, 40,
                              50, 60,
                              70, 80};
  auto r = hungarian_assign(rect.data(), 5, 2);
  // Optimal: col 0 → row 0 (cost 10), col 1 → row 1 (cost 2). Total 12.
  float rt = rect[r[0] * 2 + 0] + rect[r[1] * 2 + 1];
  if (rt != 12.0f) {
    std::cerr << "[FAIL] matcher rect: total expected 12 got " << rt << "\n";
    std::exit(1);
  }
  std::cout << "[PASS] hungarian matcher (3×3 + 5×2 cases)\n";
}

void test_full_loss() {
  torch::manual_seed(0);
  auto scale = yolocpp::models::rfdetr_scale_from_letter("n");
  RFDetr m(scale, /*nc=*/80);
  m->train();

  (void)backbone_cfg_from_name;
  // Hardcoded test resolution; #65A2 removed per-variant backbone
  // string. Real RF-DETR per-variant resolution is `scale.resolution`.
  struct { int img_size; } cfg{640};
  auto x   = torch::randn({2, 3, cfg.img_size, cfg.img_size});
  auto out = m->forward_train(x);

  // out is [cls0, bbox0, cls1, bbox1, ..., clsN-1, bboxN-1].
  std::vector<torch::Tensor> cls_per_layer, bbox_per_layer;
  for (size_t i = 0; i < out.size(); i += 2) {
    cls_per_layer.push_back(out[i]);
    bbox_per_layer.push_back(out[i + 1]);
  }

  // Two-image batch: 2 GTs in image 0, 1 GT in image 1.
  std::vector<std::vector<RFDetrTarget>> targets = {
      {{3, 0.3f, 0.4f, 0.2f, 0.3f},
       {17, 0.7f, 0.8f, 0.1f, 0.2f}},
      {{42, 0.5f, 0.5f, 0.4f, 0.4f}},
  };

  auto L = rfdetr_set_loss(cls_per_layer, bbox_per_layer, targets);
  if (!std::isfinite(L.total.item<float>()) ||
      !std::isfinite(L.cls.item<float>()) ||
      !std::isfinite(L.l1.item<float>()) ||
      !std::isfinite(L.giou.item<float>())) {
    std::cerr << "[FAIL] loss has non-finite components: "
              << "total=" << L.total.item<float>()
              << " cls="  << L.cls.item<float>()
              << " l1="   << L.l1.item<float>()
              << " giou=" << L.giou.item<float>() << "\n";
    std::exit(1);
  }
  if (L.total.item<float>() <= 0.0f) {
    std::cerr << "[FAIL] loss total non-positive: "
              << L.total.item<float>() << "\n";
    std::exit(1);
  }

  // Gradients flow through to model parameters.
  L.total.backward();
  bool any_grad = false;
  for (auto& p : m->parameters()) {
    if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0) {
      any_grad = true;
      break;
    }
  }
  if (!any_grad) {
    std::cerr << "[FAIL] no parameter received non-zero gradient\n";
    std::exit(1);
  }

  std::cout << "[PASS] rfdetr-n full loss"
            << " total=" << L.total.item<float>()
            << " cls="   << L.cls.item<float>()
            << " l1="    << L.l1.item<float>()
            << " giou="  << L.giou.item<float>() << "\n";
}

}  // namespace

int main() {
  test_matcher();
  test_full_loss();
  std::cout << "rfdetr loss: OK\n";
  return 0;
}
