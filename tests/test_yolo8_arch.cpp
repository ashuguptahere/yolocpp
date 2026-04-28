// Architecture-only smoke test for YOLO8n.
// - Builds the model
// - Runs a random forward pass on CPU then on CUDA
// - Verifies output shapes and that strides are {8, 16, 32}
// - Verifies parameter count is in the expected ballpark for v8n (~3.2M)

#include <torch/torch.h>

#include <iostream>
#include <vector>

#include "yolocpp/models/yolo8.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) {                                                 \
      std::cerr << "[FAIL] " << msg << "\n";                       \
      return 1;                                                    \
    }                                                              \
  } while (0)

int main() {
  using namespace yolocpp::models;

  Yolo8Detect model(kYolo8n, /*nc=*/80);
  model->eval();
  std::cout << "[arch] built YOLO8n\n";

  // Strides set?
  EXPECT(model->stride.size() == 3, "expected 3 detection levels");
  EXPECT(model->stride[0] == 8.0 && model->stride[1] == 16.0 && model->stride[2] == 32.0,
         "strides should be {8, 16, 32}");
  std::cout << "[arch] strides: " << model->stride[0] << ", "
            << model->stride[1] << ", " << model->stride[2] << "\n";

  // Param count (only trainable).
  long long n_params = 0;
  for (const auto& p : model->parameters())
    if (p.requires_grad()) n_params += p.numel();
  std::cout << "[arch] trainable params: " << n_params << "\n";
  // YOLO8n is ~3.15M params with COCO head (nc=80). Allow ±10%.
  EXPECT(n_params > 2'700'000 && n_params < 3'600'000,
         "param count outside expected range for v8n");

  // Forward eval shape check on CPU.
  {
    auto x = torch::randn({1, 3, 640, 640});
    auto y = model->forward_eval(x);
    std::cout << "[arch] eval out shape: " << y.sizes() << "\n";
    EXPECT(y.dim() == 3, "eval output should be 3-D");
    EXPECT(y.size(0) == 1,        "batch == 1");
    EXPECT(y.size(1) == 4 + 80,   "channel == 4 + nc");
    // Anchors at 640 input: 80*80 + 40*40 + 20*20 = 8400
    EXPECT(y.size(2) == 8400,     "anchor count == 8400 for 640 input");
  }

  // Forward train shape check on CPU.
  {
    auto x = torch::randn({2, 3, 640, 640});
    auto feats = model->forward_train(x);
    EXPECT((int)feats.size() == 3, "train output should have 3 levels");
    EXPECT(feats[0].sizes() == torch::IntArrayRef({2, 4 * 16 + 80, 80, 80}),
           "P3 train shape mismatch");
    EXPECT(feats[1].sizes() == torch::IntArrayRef({2, 4 * 16 + 80, 40, 40}),
           "P4 train shape mismatch");
    EXPECT(feats[2].sizes() == torch::IntArrayRef({2, 4 * 16 + 80, 20, 20}),
           "P5 train shape mismatch");
    std::cout << "[arch] train levels: " << feats[0].sizes() << ", "
              << feats[1].sizes() << ", " << feats[2].sizes() << "\n";
  }

  // CUDA round-trip.
  if (torch::cuda::is_available()) {
    model->to(torch::kCUDA);
    model->eval();
    auto x = torch::randn({1, 3, 640, 640}).to(torch::kCUDA);
    auto y = model->forward_eval(x);
    std::cout << "[arch] cuda eval out: " << y.sizes() << " on " << y.device() << "\n";
    EXPECT(y.size(2) == 8400, "cuda anchor count");
  } else {
    std::cout << "[arch] CUDA unavailable — skipping GPU forward\n";
  }

  std::cout << "=== arch test PASS ===\n";
  return 0;
}
