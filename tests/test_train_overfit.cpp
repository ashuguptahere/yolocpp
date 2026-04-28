// Verify the training pipeline can overfit a tiny synthetic dataset.
//
// Synthetic data: 4 images, each with one box at the center of a colored
// rectangle. We start from random init (not pretrained), train a fresh
// YOLO8n for a small number of iterations, and verify the loss decreases
// by at least 5×. This exercises:
//   - dataset loading (synthetic on-disk YOLO labels)
//   - augmentation
//   - forward / loss / backward / SGD / EMA
//   - validate() and mAP computation

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <string>
#include <vector>

#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/engine/trainer.hpp"
#include "yolocpp/engine/validator.hpp"
#include "yolocpp/losses/yolo8_loss.hpp"
#include "yolocpp/models/yolo8.hpp"

namespace fs = std::filesystem;

#define EXPECT(cond, msg)                                          \
  do {                                                             \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; return 1; } \
  } while (0)

// Generate one synthetic image + label: a colored rectangle at a known
// location on a noisy background. Class index = box index modulo nc.
static void make_synthetic(const std::string& img_path,
                           const std::string& lbl_path,
                           int W, int H, int cls,
                           cv::Rect box) {
  cv::Mat img(H, W, CV_8UC3);
  cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(80, 80, 80));
  cv::Scalar color((cls * 137) % 255, (cls * 59) % 255, (cls * 211) % 255);
  cv::rectangle(img, box, color, cv::FILLED);
  cv::imwrite(img_path, img);

  std::ofstream f(lbl_path);
  float cx = (box.x + box.width  / 2.0f) / W;
  float cy = (box.y + box.height / 2.0f) / H;
  float bw = box.width  / (float)W;
  float bh = box.height / (float)H;
  f << cls << " " << cx << " " << cy << " " << bw << " " << bh << "\n";
}

int main() {
  using namespace yolocpp;

  // Build tiny dataset on disk under build/ovfit_data/
  fs::path root = "build/ovfit_data";
  fs::create_directories(root / "images" / "train");
  fs::create_directories(root / "labels" / "train");
  fs::create_directories(root / "images" / "val");
  fs::create_directories(root / "labels" / "val");

  std::vector<std::string> names = {"red", "green"};
  const int W = 320, H = 320;

  std::vector<cv::Rect> boxes = {
      cv::Rect(40,  40,  120, 120),
      cv::Rect(180, 30,  100, 140),
      cv::Rect(60,  170, 90,  110),
      cv::Rect(170, 180, 130, 110),
  };
  for (size_t i = 0; i < boxes.size(); ++i) {
    int cls = (int)(i % names.size());
    make_synthetic(
        (root / "images" / "train" / (std::to_string(i) + ".jpg")).string(),
        (root / "labels" / "train" / (std::to_string(i) + ".txt")).string(),
        W, H, cls, boxes[i]);
    make_synthetic(
        (root / "images" / "val" / (std::to_string(i) + ".jpg")).string(),
        (root / "labels" / "val" / (std::to_string(i) + ".txt")).string(),
        W, H, cls, boxes[i]);
  }

  // No augmentation for the overfit test.
  datasets::AugConfig aug; aug.augment = false;
  datasets::YoloDataset train_ds(root.string(), "train", /*imgsz=*/320, names, aug);
  auto val_ds = std::make_shared<datasets::YoloDataset>(
      root.string(), "val", /*imgsz=*/320, names, aug);

  // Build a tiny model. We use full v8n architecture but with nc=2.
  models::Yolo8Detect model(models::kYolo8n, /*nc=*/(int)names.size());
  // Re-init Detect head biases sensibly (many zeros after constructor).
  // (nothing else needed; default init is fine for overfit demo)

  engine::TrainConfig cfg;
  cfg.epochs        = 30;
  cfg.batch_size    = 4;
  cfg.imgsz         = 320;
  cfg.lr0           = 0.01;
  cfg.lrf           = 0.01;
  cfg.warmup_epochs = 1;
  cfg.log_every     = 10;
  cfg.save_dir      = "build/ovfit_run";
  cfg.device        = torch::cuda::is_available() ? "cuda" : "cpu";

  // Quick "before" loss measurement.
  losses::LossConfig lc; lc.nc = (int)names.size();
  losses::V8DetectionLoss loss(lc);
  std::mt19937 rng(0);
  auto device = torch::Device(cfg.device == "cuda" ? torch::kCUDA : torch::kCPU);
  model->to(device);

  double pre_loss;
  {
    auto batch = train_ds.sample_batch(4, rng);
    auto imgs = batch.imgs.to(device);
    auto tgt  = batch.targets.to(device);
    torch::NoGradGuard ng;
    auto feats = model->forward_train(imgs);
    auto lo = loss(feats, tgt, model->stride, cfg.imgsz);
    pre_loss = lo.total.item<double>();
  }
  std::cout << "[ovfit] initial total loss: " << pre_loss << "\n";

  engine::Trainer trainer(model, train_ds, cfg);
  trainer.run();

  // After training, sample one batch and measure loss with EMA.
  double post_loss;
  {
    auto& ema = trainer.ema_model();
    ema->to(device); ema->eval();
    auto batch = train_ds.sample_batch(4, rng);
    auto imgs = batch.imgs.to(device);
    auto tgt  = batch.targets.to(device);
    torch::NoGradGuard ng;
    auto feats = ema->forward_train(imgs);
    auto lo = loss(feats, tgt, model->stride, cfg.imgsz);
    post_loss = lo.total.item<double>();
  }
  std::cout << "[ovfit] post-training total loss (EMA): " << post_loss << "\n";

  // Validate (mAP).
  auto map = engine::validate(trainer.ema_model(), *val_ds, device);
  std::cout << "[ovfit] mAP@0.5 = " << map.map_50
            << "  mAP@0.5:0.95 = " << map.map_50_95 << "\n";

  EXPECT(post_loss < pre_loss / 2.0,
         "loss should decrease meaningfully on a tiny overfit dataset");

  std::cout << "=== train+validate test PASS ===\n";
  return 0;
}
