// Architecture-only shape verification for YOLO4 / YOLO6 / YOLO7.
// Weight-loader work is tracked separately (see CLAUDE.md task #4).

#include <torch/torch.h>

#include <iostream>

#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/models/yolo6.hpp"
#include "yolocpp/models/yolo7.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp::models;

  // ─── YOLO4 — CSPDarknet53 + SPP + PANet + v3-style head ────────────────
  // Expected param count: ~64M for nc=80 at full scale.
  {
    Yolo4 m(/*nc=*/80);
    m->eval();
    long long n = 0;
    for (auto& p : m->parameters()) if (p.requires_grad()) n += p.numel();
    std::cout << "[v4] params=" << n / 1e6 << "M\n";
    EXPECT(n > 50'000'000 && n < 75'000'000, "v4 param count outside 50–75M");

    auto outs = m->forward(torch::randn({1, 3, 416, 416}));
    EXPECT((int)outs.size() == 3, "v4: 3 scales");
    EXPECT(outs[0].sizes() == torch::IntArrayRef({1, 255, 13, 13}), "v4 P5 (stride 32)");
    EXPECT(outs[1].sizes() == torch::IntArrayRef({1, 255, 26, 26}), "v4 P4 (stride 16)");
    EXPECT(outs[2].sizes() == torch::IntArrayRef({1, 255, 52, 52}), "v4 P3 (stride 8)");
  }

  // ─── YOLO6 — Meituan v3.0 deploy form (DFL + dual reg + RepBiFPANNeck) ─
  // forward() now returns the DECODED per-scale [B, 4+nc, A_i] (after DFL
  // + dist2bbox), so the assertion is on shape post-decode.
  {
    Yolo6 m(/*nc=*/80, kYolo6s);
    m->eval();
    long long n = 0;
    for (auto& p : m->parameters()) if (p.requires_grad()) n += p.numel();
    std::cout << "[v6s] params=" << n / 1e6 << "M\n";

    auto outs = m->forward(torch::randn({1, 3, 640, 640}));
    EXPECT((int)outs.size() == 3, "v6: 3 scales");
    const int oc = 4 + 80;  // 84 (xyxy + sigmoid'd cls)
    EXPECT(outs[0].sizes() == torch::IntArrayRef({1, oc, 80 * 80}), "v6 P3");
    EXPECT(outs[1].sizes() == torch::IntArrayRef({1, oc, 40 * 40}), "v6 P4");
    EXPECT(outs[2].sizes() == torch::IntArrayRef({1, oc, 20 * 20}), "v6 P5");
  }

  // ─── YOLO7 — yolov7-base deploy form (yaml-walker, anchor-based IDetect) ─
  // Expected ~37M params for nc=80. forward_eval returns the decoded
  // [B, 4+nc, A] for NMS; we just check shape on a synthetic input.
  {
    Yolo7 m(/*nc=*/80);
    m->eval();
    long long n = 0;
    for (auto& p : m->parameters()) if (p.requires_grad()) n += p.numel();
    std::cout << "[v7] params=" << n / 1e6 << "M\n";

    auto out = m->forward_eval(torch::randn({1, 3, 640, 640}));
    EXPECT(out.dim() == 3, "v7: forward_eval rank 3 ([B, 4+nc, A])");
    EXPECT(out.size(0) == 1, "v7: B=1");
    EXPECT(out.size(1) == 4 + 80, "v7: channel=4+nc");
    // A = sum over scales of na*H*W = 3*(80²+40²+20²) = 3*8400 = 25200
    EXPECT(out.size(2) == 25200, "v7: A=25200 anchors");
  }

  std::cout << "=== v4 + v6 + v7 PASS ===\n";
  return 0;
}
