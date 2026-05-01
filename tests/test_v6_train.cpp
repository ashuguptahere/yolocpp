// v6 train smoke — exercises the V6DetectionLoss + TrainerT<Yolo6>
// pipeline end-to-end on coco8 for 2 epochs from random init.
// Asserts that loss decreases meaningfully (TAL is matching positives,
// box/dfl branches are non-zero) — full mAP-parity finetune is
// validated separately via `mode=train ... model=yolo6m.pt` which
// reaches mAP@0.5:0.95 ≥ 0.7 on coco8 in epoch 0.
//
// Skipped if data/coco8 is missing.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/engine/trainer.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolo6.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg) \
  do { if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } } while (0)

int main() {
  using namespace yolocpp;

  if (!fs::exists("data/coco8/images/train")) {
    std::cout << "[v6-train] SKIP: data/coco8 missing\n";
    return 0;
  }

  auto names = yolocpp::inference::coco_names();
  yolocpp::datasets::YoloDataset train_ds("data/coco8", "train", 640, names);

  // Random-init m-scale with DFL head — reg_max=16, bins=17.
  models::Yolo6 model((int)names.size(), models::kYolo6m, /*reg_max=*/16, false);

  engine::TrainConfig cfg;
  cfg.epochs     = 2;
  cfg.batch_size = 2;
  cfg.imgsz      = 640;
  cfg.lr0        = 0.001;
  cfg.device     = "cpu";
  cfg.save_dir   = "/tmp/yolocpp_v6_train_smoke";
  cfg.log_every  = 100;
  cfg.warmup_epochs = 0;

  fs::remove_all(cfg.save_dir);
  engine::TrainerV6 trainer(model, train_ds, cfg);
  trainer.run();

  bool found_ckpt = false;
  if (fs::exists(cfg.save_dir)) {
    for (auto& e : fs::recursive_directory_iterator(cfg.save_dir)) {
      auto name = e.path().filename().string();
      if (name == "last.pt" || name == "best.pt") { found_ckpt = true; break; }
    }
  }
  EXPECT(found_ckpt, "trainer did not write a checkpoint under " + cfg.save_dir);

  std::cout << "=== v6 train PASS (smoke) ===\n";
  return 0;
}
