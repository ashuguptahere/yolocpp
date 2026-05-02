// Phase 5A/5B regression: YOLO5 end-to-end on bus.jpg + YOLO3 architecture
// shape check.

#include <torch/torch.h>

#include <iostream>
#include <map>

#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolo3.hpp"
#include "yolocpp/models/yolo5.hpp"

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

int main() {
  using namespace yolocpp;

  // ─── YOLO3 (yolov3u — anchor-free v8-style DFL head):
  //     Darknet-53 backbone, ~103M params at nc=80, decoded forward
  //     output [1, 4+nc, A] for NMS (A = 8400 at imgsz=640).
  {
    models::Yolo3 m(models::kYolo3, /*nc=*/80);
    m->eval();
    long long n = 0;
    for (auto& p : m->parameters()) if (p.requires_grad()) n += p.numel();
    std::cout << "[v3] params=" << n / 1e6 << "M\n";
    EXPECT(n > 95'000'000 && n < 110'000'000, "v3 param count outside 95–110M");

    auto out = m->forward_eval(torch::randn({1, 3, 640, 640}));
    EXPECT(out.dim() == 3, "v3 forward_eval rank 3");
    EXPECT(out.size(1) == 84, "v3 channels = 4 + nc");
    EXPECT(out.size(2) == 8400, "v3 anchors = 80² + 40² + 20² = 8400");
  }

  // ─── YOLO5 — verify all five scales (n/s/m/l/x) end-to-end ──────────
  struct V5Cfg {
    const char* path;
    models::Yolo8Scale scale;
    const char* name;
    long long  expected_params_M_x10;  // expected param count × 10 (so 1.86M → 19)
  };
  V5Cfg v5cfgs[] = {
      {"data/yolo5n.pt", models::kYolo5n, "n",  19},   //  1.9M
      {"data/yolo5s.pt", models::kYolo5s, "s",  72},   //  7.2M
      {"data/yolo5m.pt", models::kYolo5m, "m", 211},   // 21.1M
      {"data/yolo5l.pt", models::kYolo5l, "l", 463},   // 46.3M
      {"data/yolo5x.pt", models::kYolo5x, "x", 868},   // 86.8M
  };
  for (const auto& cfg : v5cfgs) {
    auto dets = inference::predict_v5_to_file(
        cfg.path, "data/bus.jpg",
        std::string("build/v5_test_bus_") + cfg.name + ".jpg",
        /*imgsz=*/640, /*device=*/"", /*nc=*/80, cfg.scale);
    EXPECT(dets.size() >= 4,
           std::string("v5") + cfg.name + ": ≥ 4 detections on bus.jpg");
    std::map<int, int> by_cls;
    for (auto& d : dets) ++by_cls[d.cls];
    EXPECT(by_cls[0] >= 1, std::string("v5") + cfg.name + ": ≥ 1 person");
    EXPECT(by_cls[5] >= 1, std::string("v5") + cfg.name + ": ≥ 1 bus");
    std::cout << "[v5" << cfg.name << "] " << dets.size()
              << " dets, person=" << by_cls[0] << " bus=" << by_cls[5] << "\n";
  }

  std::cout << "=== v3 + v5 PASS ===\n";
  return 0;
}
