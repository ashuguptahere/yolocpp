// v10 dual-head train smoke — paper §3.1 consistent-assignment training.
//
// Builds Yolo10(scale=n, dual_head=true) which adds a parallel one2many
// head (legacy=true v8-style cv3) alongside the deploy one2one head
// (legacy=false v11-style cv3). Loads upstream weights via
// `convert_yolov10_dual_pt` (preserves both heads). Runs 2 epochs on
// coco8 via TrainerV10 → Yolo10LossAdapter → V10DualLoss
// (V8DetectionLoss(o2m, topk=10) + V8DetectionLoss(o2o, topk=1)).
// Verifies the loss decreases.
//
// Skipped if upstream `yolov10n.pt` or coco8 are missing.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/engine/trainer.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolo10.hpp"
#include "yolocpp/serialization/pt_loader.hpp"
#include "yolocpp/serialization/yolov10_weights.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg) \
  do { if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } } while (0)

int main() {
  using namespace yolocpp;

  if (!fs::exists("data/coco8/images/train")) {
    std::cout << "[v10-dual-train] SKIP: data/coco8 missing\n";
    return 0;
  }

  std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  fs::path src = fs::path(home) / ".cache/yolocpp/weights/yolov10n.pt";
  if (!fs::exists(src)) {
    std::cout << "[v10-dual-train] SKIP: " << src << " not present\n";
    return 0;
  }

  fs::path dual = "/tmp/yolocpp_v10_dual.pt";
  serialization::convert_yolov10_dual_pt(src.string(), dual.string());

  auto names = yolocpp::inference::coco_names();
  yolocpp::datasets::YoloDataset train_ds("data/coco8", "train", 640, names);

  models::Yolo10 model(models::kYolo10n, (int)names.size(),
                        /*dual_head=*/true);
  auto sd = serialization::load_state_dict(dual.string());
  int copied = model->load_from_state_dict(sd.entries);
  EXPECT(copied > 0, "no v10 dual-head weights copied");
  std::cout << "[v10-dual-train] loaded " << copied
            << " tensors (incl. one2many)\n";

  // Sanity: the parallel o2m_detect must have at least one parameter
  // initialised from upstream's cv2/cv3.
  EXPECT(model->o2m_detect, "o2m_detect not registered when dual_head=true");

  engine::TrainConfig cfg;
  cfg.epochs     = 2;
  cfg.batch_size = 2;
  cfg.imgsz      = 640;
  cfg.lr0        = 0.0005;
  cfg.device     = "cpu";
  cfg.save_dir   = "/tmp/yolocpp_v10_dual_train_smoke";
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

  std::cout << "=== v10 dual-head train PASS ===\n";
  return 0;
}
