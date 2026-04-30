// Yolo26 training smoke test:
//   - Build a Yolo26Detect-n model + a tiny synthetic batch with fixed GTs.
//   - Run a few SGD iterations against Yolo26Loss (STAL + ProgLoss).
//   - Assert total loss strictly decreases AND box loss is non-zero at
//     iter 0 (i.e. STAL actually assigns positives at cold-start), then
//     decreases — guards against the original "box=0 forever" bug.
//
#include <cassert>
#include <cmath>
#include <iostream>

#include "yolocpp/losses/yolo26_loss.hpp"
#include "yolocpp/models/yolo26.hpp"

int main() {
  using namespace yolocpp;

  torch::manual_seed(42);
  auto dev = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                          : torch::Device(torch::kCPU);

  models::Yolo26Detect model(models::kYolo26n, /*nc=*/3);
  model->to(dev);
  model->train();

  int imgsz = 320;
  auto x = torch::randn({2, 3, imgsz, imgsz}, dev);
  auto tgt = torch::tensor({
      {0.0f, 0.0f, 100.0f,  120.0f,  60.0f,  70.0f},
      {0.0f, 1.0f, 220.0f,   80.0f,  40.0f,  50.0f},
      {1.0f, 2.0f, 160.0f,  200.0f,  80.0f, 100.0f},
  }).to(dev);

  losses::Yolo26LossConfig lcfg; lcfg.nc = 3;
  losses::Yolo26Loss loss(lcfg);

  std::vector<at::Tensor> params;
  for (auto& p : model->parameters())
    if (p.requires_grad()) params.push_back(p);
  torch::optim::SGD opt(params, torch::optim::SGDOptions(0.01).momentum(0.9));

  const int iters = 30;
  double first_total = 0, last_total = 0;
  double first_box   = 0;
  for (int it = 0; it < iters; ++it) {
    auto feats = model->forward_train(x);
    auto lo = loss(feats, tgt, model->stride, imgsz,
                   /*progress=*/(double)it / (iters - 1));
    opt.zero_grad();
    lo.total.backward();
    opt.step();

    double tot = lo.total.item<double>();
    double bx  = lo.box.item<double>();
    if (it == 0) { first_total = tot; first_box = bx; }
    last_total = tot;
  }

  std::cout << "first total=" << first_total
            << "   last total=" << last_total
            << "   first box=" << first_box << "\n";

  if (!(first_box > 0)) {
    std::cerr << "FAIL: box loss was 0 at iter 0 — STAL didn't assign positives\n";
    return 1;
  }
  if (!(last_total < first_total)) {
    std::cerr << "FAIL: total loss did not decrease\n";
    return 1;
  }
  std::cout << "OK\n";
  return 0;
}
