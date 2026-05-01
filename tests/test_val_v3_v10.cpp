// End-to-end val test for v3 / v4 / v6 / v7 / v9 / v10 against coco8.
//
// Each holder exposes `forward_eval(x) → [B, 4+nc, A]` (xyxy + sigmoid'd cls)
// so they slot into the templated `engine::validate` runner unchanged. This
// test exists to lock in the explicit template instantiations + CLI dispatch
// added under task #17 — it does NOT assert specific mAP values (coco8 is
// 4 images and any forward-path drift would already be caught by the
// matching `test_v<N>_e2e` predict suites). It only asserts the validator
// runs without errors and produces non-NaN, non-zero mAP for each version
// where the converted weights are present in the cache.
//
// Skipped (without failure) when the upstream-converted `.pt` file is not
// in `~/.cache/yolocpp/weights/` and `data/coco8/` is missing.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/engine/validator.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolo3.hpp"
#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/models/yolo6.hpp"
#include "yolocpp/models/yolo7.hpp"
#include "yolocpp/models/yolo9.hpp"
#include "yolocpp/models/yolo10.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg) \
  do { if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } } while (0)

static fs::path resolve_weight(const std::string& name) {
  std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  fs::path cache = fs::path(home) / ".cache/yolocpp/weights" / name;
  if (fs::exists(cache)) return cache;
  fs::path local = fs::path("data") / name;
  if (fs::exists(local)) return local;
  return {};
}

template <typename ModelHolder>
static int run_val(const std::string& tag, ModelHolder& m,
                   const std::string& weights, int imgsz) {
  auto sd = yolocpp::serialization::load_state_dict(weights);
  m->load_from_state_dict(sd.entries);
  auto names = yolocpp::inference::coco_names();
  yolocpp::datasets::AugConfig vaug; vaug.augment = false;
  yolocpp::datasets::YoloDataset ds("data/coco8", "val", imgsz, names, vaug);
  auto dev = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                         : torch::Device(torch::kCPU);
  auto res = yolocpp::engine::validate(m, ds, dev);
  std::cout << "[val-" << tag << "] mAP@0.5=" << res.map_50
            << " mAP@0.5:0.95=" << res.map_50_95 << "\n";
  EXPECT(std::isfinite(res.map_50)    && res.map_50    > 0.0f, tag + ": map_50 must be finite > 0");
  EXPECT(std::isfinite(res.map_50_95) && res.map_50_95 > 0.0f, tag + ": map_50_95 must be finite > 0");
  return 0;
}

int main() {
  using namespace yolocpp;

  if (!fs::exists("data/coco8/images/val")) {
    std::cout << "[val-suite] SKIP: data/coco8/images/val missing\n";
    return 0;
  }

  // ── v3 ── (anchorless yolov3u; reuses v8 Detect head) ─────────────────
  if (auto w = resolve_weight("yolo3.pt"); !w.empty()) {
    models::Yolo3 m(models::kYolo3, 80);
    if (int rc = run_val("v3", m, w.string(), 640); rc) return rc;
  } else { std::cout << "[val-suite] SKIP v3: yolo3.pt not present\n"; }

  // ── v4 ── (Darknet CSPDarknet53 + anchor head, 608² imgsz) ────────────
  if (auto w = resolve_weight("yolo4.pt"); !w.empty()) {
    models::Yolo4 m(80);
    if (int rc = run_val("v4", m, w.string(), 608); rc) return rc;
  } else { std::cout << "[val-suite] SKIP v4: yolo4.pt not present\n"; }

  // ── v6 n/s + s/m/l/x_mbla + n6/s6/m6 P6 ────────────────────────────────
  for (auto p : {std::pair<std::string, models::Yolo6Scale>{"yolo6n.pt",      models::kYolo6n},
                 std::pair<std::string, models::Yolo6Scale>{"yolo6s.pt",      models::kYolo6s},
                 std::pair<std::string, models::Yolo6Scale>{"yolo6s_mbla.pt", models::kYolo6s_mbla},
                 std::pair<std::string, models::Yolo6Scale>{"yolo6m_mbla.pt", models::kYolo6m_mbla},
                 std::pair<std::string, models::Yolo6Scale>{"yolo6l_mbla.pt", models::kYolo6l_mbla},
                 std::pair<std::string, models::Yolo6Scale>{"yolo6x_mbla.pt", models::kYolo6x_mbla}}) {
    if (auto w = resolve_weight(p.first); !w.empty()) {
      models::Yolo6 m(80, p.second);
      if (int rc = run_val("v6:" + p.first, m, w.string(), 640); rc) return rc;
    } else { std::cout << "[val-suite] SKIP " << p.first << "\n"; }
  }
  // P6 variants — 1280² imgsz. l6 has a parity gap (#42), excluded from suite.
  for (auto p : {std::pair<std::string, models::Yolo6Scale>{"yolo6n6.pt", models::kYolo6n},
                 std::pair<std::string, models::Yolo6Scale>{"yolo6s6.pt", models::kYolo6s},
                 std::pair<std::string, models::Yolo6Scale>{"yolo6m6.pt", models::kYolo6m}}) {
    if (auto w = resolve_weight(p.first); !w.empty()) {
      models::Yolo6 m(80, p.second, /*reg_max=*/16, /*p6=*/true);
      if (int rc = run_val("v6:" + p.first, m, w.string(), 1280); rc) return rc;
    } else { std::cout << "[val-suite] SKIP " << p.first << "\n"; }
  }

  // ── v7 base ─ (1 P5 variant; P6 variants would need imgsz=1280) ───────
  if (auto w = resolve_weight("yolo7.pt"); !w.empty()) {
    models::Yolo7 m(80, models::Yolo7Scale::Base);
    if (int rc = run_val("v7", m, w.string(), 640); rc) return rc;
  } else { std::cout << "[val-suite] SKIP v7: yolo7.pt not present\n"; }

  // ── v9c ── (GELAN + RepNCSPELAN4) ─────────────────────────────────────
  if (auto w = resolve_weight("yolo9c.pt"); !w.empty()) {
    models::Yolo9 m(models::Yolo9Scale::C, 80);
    if (int rc = run_val("v9c", m, w.string(), 640); rc) return rc;
  } else { std::cout << "[val-suite] SKIP v9c: yolo9c.pt not present\n"; }

  // ── v10 n/s/m/b/l/x ── (NMS-free deploy form via one2one head) ────────
  // n is also stored as the historical "yolo10.pt" filename without a
  // letter; the rest are yolo10<L>.pt.
  for (auto p : {std::pair<std::string, models::Yolo10Scale>{"yolo10.pt",  models::kYolo10n},
                 std::pair<std::string, models::Yolo10Scale>{"yolo10s.pt", models::kYolo10s},
                 std::pair<std::string, models::Yolo10Scale>{"yolo10m.pt", models::kYolo10m},
                 std::pair<std::string, models::Yolo10Scale>{"yolo10b.pt", models::kYolo10b},
                 std::pair<std::string, models::Yolo10Scale>{"yolo10l.pt", models::kYolo10l},
                 std::pair<std::string, models::Yolo10Scale>{"yolo10x.pt", models::kYolo10x}}) {
    if (auto w = resolve_weight(p.first); !w.empty()) {
      models::Yolo10 m(p.second, 80);
      if (int rc = run_val("v10:" + p.first, m, w.string(), 640); rc) return rc;
    } else { std::cout << "[val-suite] SKIP " << p.first << " (not present)\n"; }
  }

  std::cout << "=== val v3/v4/v6/v7/v9/v10 PASS ===\n";
  return 0;
}
