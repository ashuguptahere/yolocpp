#pragma once
//
// Trainer for YOLO8 detection.
//
// Implements:
//   - SGD with momentum + weight decay (with no-decay group for biases / BN)
//   - Linear warmup → cosine LR schedule
//   - EMA of model weights
//   - Per-iteration logging (box / cls / dfl / total)
//
// Saves a checkpoint to <save_dir>/last.pt — note: written via libtorch's
// Module::save (TorchScript-compatible archive), not the upstream `.pt`
// shape. To resume training in this codebase, use load_checkpoint_pt below.
//

#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#include "yolocpp/datasets/yolo_dataset.hpp"
#include "yolocpp/losses/yolo10_loss.hpp"
#include "yolocpp/losses/yolo26_loss.hpp"
#include "yolocpp/losses/yolo6_loss.hpp"
#include "yolocpp/losses/yolo7_loss.hpp"
#include "yolocpp/losses/yolo8_loss.hpp"
#include "yolocpp/models/yolo11.hpp"
#include "yolocpp/models/yolo12.hpp"
#include "yolocpp/models/yolo13.hpp"
#include "yolocpp/models/yolo26.hpp"
#include "yolocpp/models/yolo10.hpp"
#include "yolocpp/models/yolo3.hpp"
#include "yolocpp/models/yolo4.hpp"
#include "yolocpp/models/yolo5.hpp"
#include "yolocpp/models/yolo6.hpp"
#include "yolocpp/models/yolo7.hpp"
#include "yolocpp/models/yolo8.hpp"
#include "yolocpp/models/yolo9.hpp"

namespace yolocpp::engine {

struct TrainConfig {
  int    epochs        = 100;
  int    batch_size    = 16;
  int    imgsz         = 640;
  double lr0           = 0.01;
  double lrf           = 0.01;     // final lr = lr0 * lrf (cosine)
  double momentum      = 0.937;
  double weight_decay  = 0.0005;
  int    warmup_epochs = 3;
  double warmup_bias_lr= 0.1;
  double warmup_momentum = 0.8;
  double ema_decay     = 0.9999;
  int    ema_warmup    = 2000;
  std::string device   = "";       // auto; comma-separated lists (e.g. "cuda:0,1")
                                   // currently produce a clear "not supported" error
  std::string save_dir = "runs/train";
  int        log_every  = 10;
  // Optional: validate every N epochs (0 = never)
  int        val_every  = 0;
  std::shared_ptr<datasets::YoloDataset> val_dataset;  // optional
  // Early stopping: stop training if best val mAP@0.5:0.95 doesn't improve
  // for `patience` consecutive validation passes. 0 disables.
  int        patience   = 0;
  // Free-form (key, value) pairs dumped to <save_dir>/args.yaml at run start
  // for reproducibility — typically the CLI args verbatim.
  std::vector<std::pair<std::string, std::string>> args_for_yaml;
};

// Trainer is templated over the model-holder type (Yolo8Detect or
// Yolo5Detect). Both share the same interface (forward_train, forward_eval,
// stride, nc, scale, load_from_state_dict). Explicit instantiations live in
// trainer.cpp.
template <typename ModelHolder>
class TrainerT {
 public:
  TrainerT(ModelHolder model, datasets::YoloDataset train, TrainConfig cfg);

  // Run the full training schedule.
  void run();

  ModelHolder& model()      { return model_; }
  ModelHolder& ema_model()  { return ema_; }

 private:
  void ema_update(double decay);

  ModelHolder              model_;
  ModelHolder              ema_;
  datasets::YoloDataset    train_;
  TrainConfig              cfg_;
  torch::Device            device_;
};

using Trainer    = TrainerT<models::Yolo8Detect>;
using TrainerV3  = TrainerT<models::Yolo3>;
using TrainerV4  = TrainerT<models::Yolo4>;
using TrainerV5  = TrainerT<models::Yolo5Detect>;
using TrainerV6  = TrainerT<models::Yolo6>;
using TrainerV7  = TrainerT<models::Yolo7>;
using TrainerV9  = TrainerT<models::Yolo9>;
using TrainerV10 = TrainerT<models::Yolo10>;
using TrainerV11 = TrainerT<models::Yolo11Detect>;
using TrainerV12 = TrainerT<models::Yolo12Detect>;
using TrainerV13 = TrainerT<models::Yolo13Detect>;
using TrainerV26 = TrainerT<models::Yolo26Detect>;

}  // namespace yolocpp::engine
