// End-to-end test for YOLO1: convert pjreddie's yolov1.weights → yolo1.pt
// (no Darknet runtime), construct Yolo1, load, run forward shape + decode
// + NMS sanity. Gated on `data/yolov1.weights` or the local cache; the
// pure forward-shape sanity (no weights) is always run.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolo1.hpp"
#include "yolocpp/serialization/yolov1_weights.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp;

  // 1) Pure forward-shape sanity (always runs). Verifies the Yolo1
  //    holder constructs cleanly and forward_eval returns
  //    `[B, 4+nc, S·S·B] = [1, 24, 98]` on a 1×3×448×448 input.
  {
    models::Yolo1 m(/*nc=*/20);
    m->eval();
    auto x   = torch::zeros({1, 3, 448, 448});
    auto out = m->forward_eval(x);
    EXPECT(out.dim() == 3,             "v1 forward_eval rank");
    EXPECT(out.size(0) == 1,           "v1 batch dim");
    EXPECT(out.size(1) == 4 + 20,      "v1 (4+nc) channels");
    EXPECT(out.size(2) == 7 * 7 * 2,   "v1 A = S·S·B");
    std::cout << "[v1-e2e] forward-shape sanity OK (out="
              << out.sizes() << ")\n";
  }

  // 2) Prefer the pre-converted `data/yolo1.pt` (this is the canonical
  //    runtime form). Only fall back to converting from `.weights` if
  //    no `.pt` exists — the conversion is a one-shot maintenance step
  //    (see `tools/convert_weights`) rather than something we want
  //    every test run to redo.
  std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  fs::path cache_w = fs::path(home) / ".cache/yolocpp/weights";
  fs::path pt;
  for (const auto& c : std::vector<fs::path>{
           "data/yolo1.pt", cache_w / "yolo1.pt"}) {
    if (fs::exists(c)) { pt = c; break; }
  }
  if (pt.empty()) {
    fs::path wsrc;
    for (const auto& c : std::vector<fs::path>{
             "data/yolov1.weights", cache_w / "yolov1.weights",
             "/tmp/yolov1.weights"}) {
      if (fs::exists(c)) { wsrc = c; break; }
    }
    if (wsrc.empty()) {
      std::cout << "[v1-e2e] SKIP predict (no data/yolo1.pt nor yolov1.weights)\n";
      return 0;
    }
    pt = "build/yolo1_e2e.pt";
    fs::remove(pt);
    int blocks = serialization::convert_yolov1_weights(
        wsrc.string(), pt.string(), /*nc=*/20);
    EXPECT(blocks >= 26, "v1 expected ≥ 26 blocks (24 conv + 2 fc)");
  }

  // 3) Predict on bus.jpg if it's around.
  fs::path bus = "data/bus.jpg";
  if (!fs::exists(bus)) {
    std::cout << "[v1-e2e] SKIP predict (no data/bus.jpg)\n";
    return 0;
  }
  auto dets = inference::predict_v1_to_file(
      pt.string(), bus.string(), "build/v1_e2e_bus.jpg",
      /*imgsz=*/448, /*device=*/"", /*nc=*/20);
  std::cout << "[v1-e2e] " << dets.size() << " dets on bus.jpg\n";
  return 0;
}
