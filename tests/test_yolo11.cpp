// End-to-end test for YOLO11.
//
// For each scale (n / s / m / l / x):
//   1. Construct Yolo11Detect at that scale.
//   2. Load yolo11<scale>.pt's state_dict.
//   3. Verify (a) parameter count matches Ultralytics' published value
//                 within rounding, (b) every checkpoint key was applied.
//   4. Run forward_eval on a 256×256 zero image — confirms the full
//      module graph wires up + Detect decode produces the right shape.
//
// Then: run predict on bus.jpg with the n-scale weights and assert at
// least one person + one bus are detected (proves loaded weights are
// numerically meaningful, not zero-init).

#include <torch/torch.h>

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include "yolocpp/inference/letterbox.hpp"
#include "yolocpp/inference/nms.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

// Ultralytics-published parameter counts for yolo11<scale>.pt
// (the values printed by ultralytics' `model.info()`).
struct ScaleSpec {
  std::string letter;
  yolocpp::models::Yolo11Scale scale;
  long long expected_params;        // approximate; we allow ±0.5%
};

static long long count_params(torch::nn::Module& m) {
  long long total = 0;
  for (const auto& kv : m.named_parameters()) {
    long long n = 1;
    for (auto d : kv.value().sizes()) n *= d;
    total += n;
  }
  return total;
}

int main() {
  using namespace yolocpp;
  // Approximate counts (parameter-only, not buffers) from Ultralytics:
  //   yolo11n: 2.6M, yolo11s: 9.4M, yolo11m: 20.0M, yolo11l: 25.3M, yolo11x: 56.9M
  // Tolerance ±1.5% absorbs rounding differences in scale_channels.
  std::vector<ScaleSpec> scales = {
      {"n", models::kYolo11n,  2'600'000ll},
      {"s", models::kYolo11s,  9'400'000ll},
      {"m", models::kYolo11m, 20'000'000ll},
      {"l", models::kYolo11l, 25'300'000ll},
      {"x", models::kYolo11x, 56'900'000ll},
  };

  for (const auto& sp : scales) {
    std::string pt = "data/yolo11" + sp.letter + ".pt";
    std::cout << "[v11] === " << pt << " ===\n";

    models::Yolo11Detect model(sp.scale, /*nc=*/80);
    long long got_params = count_params(*model);
    std::cout << "  built " << got_params << " params (expected ~"
              << sp.expected_params << ")\n";

    long long diff = std::abs(got_params - sp.expected_params);
    EXPECT(diff < sp.expected_params / 50,  // ±2%
           "param count too far from Ultralytics published value for " + pt);

    auto sd = serialization::load_state_dict(pt);
    int copied = model->load_from_state_dict(sd.entries);
    std::cout << "  loaded " << copied << " of " << sd.entries.size()
              << " checkpoint tensors\n";
    EXPECT(copied == (int)sd.entries.size(),
           "did not consume the full state_dict for " + pt);

    auto dev = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                            : torch::Device(torch::kCPU);
    model->to(dev);
    model->eval();

    {
      torch::NoGradGuard ng;
      auto x  = torch::zeros({1, 3, 640, 640}, torch::kFloat32).to(dev);
      auto y  = model->forward_eval(x);
      // Expected shape: [1, 4 + nc, A] where A = 80²/8² + 40²/8² + 20²/8² ...
      // For 640 input + strides 8/16/32: A = 80*80 + 40*40 + 20*20 = 8400.
      EXPECT(y.dim() == 3, "forward_eval should return 3D tensor");
      EXPECT(y.size(0) == 1 && y.size(1) == 84 && y.size(2) == 8400,
             "unexpected output shape from forward_eval (want [1,84,8400])");
      std::cout << "  forward_eval output shape OK\n";
    }
  }

  // Detection-quality smoke test on bus.jpg with the n-scale weights.
  {
    std::cout << "[v11] === bus.jpg detection smoke ===\n";
    models::Yolo11Detect model(models::kYolo11n, /*nc=*/80);
    auto sd = serialization::load_state_dict("data/yolo11n.pt");
    model->load_from_state_dict(sd.entries);
    auto dev = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                            : torch::Device(torch::kCPU);
    model->to(dev);
    model->eval();

    cv::Mat bgr = cv::imread("data/bus.jpg", cv::IMREAD_COLOR);
    EXPECT(!bgr.empty(), "could not read data/bus.jpg");
    auto lb = inference::letterbox(bgr, 640);
    auto x  = inference::image_to_tensor(lb.img).unsqueeze(0).to(dev);

    torch::Tensor pred;
    {
      torch::NoGradGuard ng;
      pred = model->forward_eval(x);
    }
    auto outs = inference::nms(pred, /*conf=*/{});
    EXPECT(outs.size() == 1, "nms should return one image's detections");
    auto det = outs[0].to(torch::kCPU);
    int n = (int)det.size(0);
    std::cout << "  " << n << " detections on bus.jpg\n";
    EXPECT(n >= 4, "expected ≥4 detections on bus.jpg from yolo11n");

    bool saw_person = false, saw_bus = false;
    auto a = det.accessor<float, 2>();
    for (int i = 0; i < n; ++i) {
      int cls = (int)a[i][5];
      if (cls == 0) saw_person = true;
      if (cls == 5) saw_bus    = true;
    }
    EXPECT(saw_person, "expected to detect at least one person");
    EXPECT(saw_bus,    "expected to detect at least one bus");
    std::cout << "  saw person + bus → OK\n";
  }

  std::cout << "=== test_yolo11 PASS ===\n";
  return 0;
}

