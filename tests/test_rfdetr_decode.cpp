// Tests RF-DETR's NMS-free decode (#65E) on synthetic forward
// output. Independent of the random-init model — drives the decode
// logic directly with a hand-crafted `[B, 4+nc, Q]` tensor where
// the expected detections are known a priori.

#include <torch/torch.h>

#include <iostream>

#include "yolocpp/inference/rfdetr_predictor.hpp"

using yolocpp::inference::Detection;
using yolocpp::inference::rfdetr_decode;

int main() {
  int B = 1, nc = 5, Q = 4, imgsz = 640;
  // Build a `[1, 4+nc, Q]` tensor with explicit per-query values.
  // Query 0: cls=2 score=0.9, box=(0.5, 0.5, 0.2, 0.2)
  // Query 1: cls=4 score=0.3 (below threshold 0.5)
  // Query 2: cls=0 score=0.7, box=(0.1, 0.1, 0.05, 0.05)
  // Query 3: cls=1 score=0.85, box=(0.9, 0.9, 0.05, 0.05)
  auto t = torch::zeros({B, 4 + nc, Q});
  // boxes
  t[0][0][0] = 0.5; t[0][1][0] = 0.5; t[0][2][0] = 0.2; t[0][3][0] = 0.2;
  t[0][0][1] = 0.0; t[0][1][1] = 0.0; t[0][2][1] = 0.0; t[0][3][1] = 0.0;
  t[0][0][2] = 0.1; t[0][1][2] = 0.1; t[0][2][2] = 0.05; t[0][3][2] = 0.05;
  t[0][0][3] = 0.9; t[0][1][3] = 0.9; t[0][2][3] = 0.05; t[0][3][3] = 0.05;
  // cls scores
  t[0][4 + 2][0] = 0.9;
  t[0][4 + 4][1] = 0.3;
  t[0][4 + 0][2] = 0.7;
  t[0][4 + 1][3] = 0.85;

  auto dets_per_image = rfdetr_decode(t, imgsz, /*conf=*/0.5f);
  if (dets_per_image.size() != 1) {
    std::cerr << "[FAIL] expected 1 image's dets\n"; return 1;
  }
  auto& dets = dets_per_image[0];
  // Q1 below threshold. Top-K by score: q0(0.9), q3(0.85), q2(0.7).
  if (dets.size() != 3) {
    std::cerr << "[FAIL] expected 3 dets after thresholding, got "
              << dets.size() << "\n";
    return 1;
  }
  if (dets[0].cls != 2 || std::abs(dets[0].conf - 0.9f) > 1e-6f) {
    std::cerr << "[FAIL] top det wrong: cls=" << dets[0].cls
              << " conf=" << dets[0].conf << "\n";
    return 1;
  }
  if (dets[1].cls != 1) {
    std::cerr << "[FAIL] det[1] cls expected 1 got " << dets[1].cls << "\n";
    return 1;
  }
  if (dets[2].cls != 0) {
    std::cerr << "[FAIL] det[2] cls expected 0 got " << dets[2].cls << "\n";
    return 1;
  }
  // Verify cxcywh → xyxy on q0: (0.5, 0.5, 0.2, 0.2) at imgsz=640
  // → cx=320, cy=320, w=128, h=128 → (256, 256, 384, 384).
  if (std::abs(dets[0].x1 - 256.0f) > 1e-3f ||
      std::abs(dets[0].y1 - 256.0f) > 1e-3f ||
      std::abs(dets[0].x2 - 384.0f) > 1e-3f ||
      std::abs(dets[0].y2 - 384.0f) > 1e-3f) {
    std::cerr << "[FAIL] q0 box xyxy: "
              << dets[0].x1 << "," << dets[0].y1 << "," << dets[0].x2
              << "," << dets[0].y2 << "\n";
    return 1;
  }

  // max_det=2 should keep only the top 2.
  auto capped = rfdetr_decode(t, imgsz, 0.5f, /*max_det=*/2);
  if (capped[0].size() != 2) {
    std::cerr << "[FAIL] max_det=2: got " << capped[0].size() << "\n";
    return 1;
  }

  std::cout << "[PASS] rfdetr decode: threshold + top-K + cxcywh→xyxy\n";
  return 0;
}
