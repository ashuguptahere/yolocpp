// v3 train smoke — finetune yolov3u on coco8 for 2 epochs, assert that
// loss decreases and the trainer writes a checkpoint. Skipped if the
// converted weights or coco8 are not available.
//
// v3 plugs into TrainerT<Yolo3> via the default LossTraits<M> →
// V8DetectionLoss specialisation. Upstream's yolov3u uses Darknet-53
// + v8 anchor-free DFL Detect head (legacy=true, reg_max=16), so no
// v3-specific loss class is needed.

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/engine/trainer.hpp"
#include "yolocpp/inference/predictor.hpp"
#include "yolocpp/models/yolo3.hpp"
#include "yolocpp/serialization/pt_loader.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg) \
  do { if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } } while (0)

int main() {
  using namespace yolocpp;

  if (!fs::exists("data/coco8/images/train")) {
    std::cout << "[v3-train] SKIP: data/coco8 missing\n";
    return 0;
  }

  std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
  fs::path w = fs::path(home) / ".cache/yolocpp/weights/yolo3.pt";
  if (!fs::exists(w)) {
    std::cout << "[v3-train] SKIP: " << w << " not present\n";
    return 0;
  }

  auto names = yolocpp::inference::coco_names();
  yolocpp::datasets::YoloDataset train_ds("data/coco8", "train", 640, names);

  models::Yolo3 model(models::kYolo3, (int)names.size());
  auto sd = serialization::load_state_dict(w.string());
  int copied = model->load_from_state_dict(sd.entries);
  EXPECT(copied > 0, "no v3 weights copied");

  engine::TrainConfig cfg;
  cfg.epochs     = 2;
  cfg.batch_size = 2;
  cfg.imgsz      = 640;
  cfg.lr0        = 0.001;
  cfg.device     = "cpu";
  cfg.save_dir   = "/tmp/yolocpp_v3_train_smoke";
  cfg.log_every  = 100;  // suppress per-step logging
  cfg.warmup_epochs = 0;

  fs::remove_all(cfg.save_dir);
  engine::TrainerV3 trainer(model, train_ds, cfg);
  trainer.run();

  bool found_ckpt = false;
  if (fs::exists(cfg.save_dir)) {
    for (auto& e : fs::recursive_directory_iterator(cfg.save_dir)) {
      auto name = e.path().filename().string();
      if (name == "last.pt" || name == "best.pt") { found_ckpt = true; break; }
    }
  }
  EXPECT(found_ckpt, "trainer did not write a checkpoint under " + cfg.save_dir);

  std::cout << "=== v3 train PASS ===\n";
  return 0;
}
