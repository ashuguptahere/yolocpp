// v4 + v7 train smoke — exercises the V7DetectionLoss + TrainerT<Yolo4/7>
// pipelines end-to-end on coco8 from pretrained weights.
//
// v4 uses scale_xy=[1.2, 1.1, 1.05] + exp() wh decode (Darknet).
// v7 uses scale_xy=2.0 uniform + (sigmoid*2)^2 wh decode.
// Both share V7DetectionLoss with per-version anchor/decode config.
//
// Skipped per-version if the converted .pt is not in cache.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/engine/trainer.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/models/yolo7.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg) \
  do { if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } } while (0)

template <typename ModelHolder, typename Trainer>
static int run_smoke(const std::string& tag,
                     ModelHolder& model,
                     const fs::path& weights, int imgsz) {
  using namespace yolocpp;
  auto sd = serialization::load_state_dict(weights.string());
  int copied = model->load_from_state_dict(sd.entries);
  EXPECT(copied > 0, tag + ": no weights copied");

  auto names = inference::coco_names();
  datasets::YoloDataset train_ds("data/coco8", "train", imgsz, names);

  engine::TrainConfig cfg;
  cfg.epochs       = 2;
  cfg.batch_size   = 2;
  cfg.imgsz        = imgsz;
  cfg.lr0          = 0.0005;
  cfg.device       = "cpu";
  cfg.save_dir     = "/tmp/yolocpp_" + tag + "_train_smoke";
  cfg.log_every    = 100;
  cfg.warmup_epochs = 0;

  fs::remove_all(cfg.save_dir);
  Trainer trainer(model, train_ds, cfg);
  trainer.run();

  bool found_ckpt = false;
  if (fs::exists(cfg.save_dir)) {
    for (auto& e : fs::recursive_directory_iterator(cfg.save_dir)) {
      auto n = e.path().filename().string();
      if (n == "last.pt" || n == "best.pt") { found_ckpt = true; break; }
    }
  }
  EXPECT(found_ckpt, tag + ": no checkpoint written");
  std::cout << "[" << tag << "-train] PASS\n";
  return 0;
}

int main() {
  using namespace yolocpp;

  if (!fs::exists("data/coco8/images/train")) {
    std::cout << "[v4v7-train] SKIP: data/coco8 missing\n";
    return 0;
  }

  std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  fs::path w_dir = fs::path(home) / ".cache/yolocpp/weights";

  // v4 (608² imgsz).
  auto w_v4 = w_dir / "yolo4.pt";
  if (fs::exists(w_v4)) {
    models::Yolo4 m(80);
    if (int rc = run_smoke<models::Yolo4, engine::TrainerV4>("v4", m, w_v4, 608); rc) return rc;
  } else { std::cout << "[v4-train] SKIP (no yolo4.pt)\n"; }

  // v7-base (640² imgsz).
  auto w_v7 = w_dir / "yolo7.pt";
  if (fs::exists(w_v7)) {
    models::Yolo7 m(models::Yolo7Scale::Base, 80);
    if (int rc = run_smoke<models::Yolo7, engine::TrainerV7>("v7", m, w_v7, 640); rc) return rc;
  } else { std::cout << "[v7-train] SKIP (no yolo7.pt)\n"; }

  std::cout << "=== v4 / v7 train PASS ===\n";
  return 0;
}
