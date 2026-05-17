// End-to-end test for YOLO2: convert pjreddie's yolov2.weights → yolo2.pt
// (no Darknet runtime), construct Yolo2 (full + tiny), forward + decode
// sanity, optional predict on bus.jpg. Forward-shape sanity always runs;
// .weights round-trip is gated on cache availability.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolo2.hpp"
#include "yolocpp/serialization/yolov2_weights.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp;

  // 1) Full Darknet-19 + reorg + region forward sanity.
  //    Output: [B, 4+nc, na·H·W] = [1, 4+20, 5·13·13] = [1, 24, 845].
  {
    models::Yolo2 m(models::Yolo2Scale::Full, /*nc=*/20);
    m->eval();
    auto x   = torch::zeros({1, 3, 416, 416});
    auto out = m->forward_eval(x);
    EXPECT(out.dim() == 3,                  "v2 forward_eval rank");
    EXPECT(out.size(0) == 1,                "v2 batch");
    EXPECT(out.size(1) == 4 + 20,           "v2 (4+nc) channels");
    EXPECT(out.size(2) == 5 * 13 * 13,      "v2 A = na·H·W");
    std::cout << "[v2-e2e] full forward shape OK (out=" << out.sizes() << ")\n";
  }
  // 2) Reorg invariant: numel preserved, shape collapsed by stride².
  {
    auto x = torch::arange(64 * 26 * 26).reshape({1, 64, 26, 26}).to(torch::kFloat32);
    auto r = models::reorg(x, /*stride=*/2);
    EXPECT(r.sizes() == torch::IntArrayRef({1, 256, 13, 13}),
            "v2 reorg shape (1, 64, 26, 26) → (1, 256, 13, 13)");
    EXPECT(r.numel() == x.numel(), "v2 reorg preserves numel");
    std::cout << "[v2-e2e] reorg layout OK\n";
  }
  // 3) Tiny variant forward sanity. Final spatial = 13 due to the
  //    stride-1 fake-pool keeping the resolution after the last
  //    full /2 chain.
  {
    models::Yolo2 m(models::Yolo2Scale::Tiny, /*nc=*/20);
    m->eval();
    auto x   = torch::zeros({1, 3, 416, 416});
    auto out = m->forward_eval(x);
    EXPECT(out.size(0) == 1,                "v2-tiny batch");
    EXPECT(out.size(1) == 4 + 20,           "v2-tiny (4+nc) channels");
    EXPECT(out.size(2) == 5 * 13 * 13,      "v2-tiny A");
    std::cout << "[v2-e2e] tiny forward shape OK (out=" << out.sizes() << ")\n";
  }

  // 4) Optional .weights round-trip.
  std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  fs::path cache_w = fs::path(home) / ".cache/yolocpp/weights";
  std::vector<fs::path> w_candidates = {
      "data/yolov2.weights", cache_w / "yolov2.weights",
      "/tmp/yolov2.weights",
  };
  fs::path src;
  for (const auto& c : w_candidates) {
    if (fs::exists(c)) { src = c; break; }
  }
  if (src.empty()) {
    std::cout << "[v2-e2e] SKIP weights round-trip (no yolov2.weights)\n";
    return 0;
  }
  fs::path pt = "build/yolo2_e2e.pt";
  fs::remove(pt);
  int blocks = serialization::convert_yolov2_weights(src.string(), pt.string(),
                                                      /*nc=*/80,
                                                      models::Yolo2Scale::Full);
  EXPECT(blocks >= 23, "v2 expected ≥ 23 conv blocks");

  fs::path bus = "data/bus.jpg";
  if (!fs::exists(bus)) {
    std::cout << "[v2-e2e] SKIP predict (no data/bus.jpg)\n";
    return 0;
  }
  auto dets = inference::predict_v2_to_file(
      pt.string(), bus.string(), "build/v2_e2e_bus.jpg",
      /*imgsz=*/416, /*device=*/"", /*nc=*/80,
      models::Yolo2Scale::Full);
  std::cout << "[v2-e2e] " << dets.size() << " dets on bus.jpg\n";
  return 0;
}
