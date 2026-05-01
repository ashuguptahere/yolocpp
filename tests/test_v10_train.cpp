// v10 train smoke — finetune yolo10n on coco8 for 2 epochs.
//
// v10 trains the deploy-form one2one head with V8DetectionLoss (the
// one2one head's cv3 = DWConvBlock×2 + Conv2d matches v11's legacy=false
// shape exactly). The full one2many + one2one dual-head consistent-
// assignment training (paper §3.1's `m_α,β = s · IoU^α · p^β` matching)
// is tracked as #45 — needs arch rework to keep one2many in Yolo10Impl.
//
// Skipped if the converted weights or coco8 are not available.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/engine/trainer.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolo10.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg) \
  do { if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } } while (0)

int main() {
  using namespace yolocpp;

  if (!fs::exists("data/coco8/images/train")) {
    std::cout << "[v10-train] SKIP: data/coco8 missing\n";
    return 0;
  }

  std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  fs::path w = fs::path(home) / ".cache/yolocpp/weights/yolo10.pt";
  if (!fs::exists(w)) {
    std::cout << "[v10-train] SKIP: " << w << " not present\n";
    return 0;
  }

  auto names = yolocpp::inference::coco_names();
  yolocpp::datasets::YoloDataset train_ds("data/coco8", "train", 640, names);

  models::Yolo10 model(models::kYolo10n, (int)names.size());
  auto sd = serialization::load_state_dict(w.string());
  int copied = model->load_from_state_dict(sd.entries);
  EXPECT(copied > 0, "no v10 weights copied");

  engine::TrainConfig cfg;
  cfg.epochs     = 2;
  cfg.batch_size = 2;
  cfg.imgsz      = 640;
  cfg.lr0        = 0.0005;
  cfg.device     = "cpu";
  cfg.save_dir   = "/tmp/yolocpp_v10_train_smoke";
  cfg.log_every  = 100;
  cfg.warmup_epochs = 0;

  fs::remove_all(cfg.save_dir);
  engine::TrainerV10 trainer(model, train_ds, cfg);
  trainer.run();

  bool found_ckpt = false;
  if (fs::exists(cfg.save_dir)) {
    for (auto& e : fs::recursive_directory_iterator(cfg.save_dir)) {
      auto name = e.path().filename().string();
      if (name == "last.pt" || name == "best.pt") { found_ckpt = true; break; }
    }
  }
  EXPECT(found_ckpt, "trainer did not write a checkpoint under " + cfg.save_dir);

  std::cout << "=== v10 train PASS ===\n";
  return 0;
}
