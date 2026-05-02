#pragma once
//
// Train + validate for image classification.
//
// Dataset layout (matches the upstream image-folder convention):
//   <root>/<split>/<class_name>/<img>.jpg
// Class indices are assigned by sorted-string order of class subdirectories.
//

#include <torch/torch.h>

#include <random>
#include <string>
#include <utility>
#include <vector>

#include "yolocpp/models/yolo11_tasks.hpp"
#include "yolocpp/models/yolo8_classify.hpp"

namespace yolocpp::tasks {

struct ClassifyExample {
  torch::Tensor img;      // [3, H, W] float32 in [0, 1]
  int           label;
  std::string   path;
};

class ClassifyDataset {
 public:
  ClassifyDataset(std::string root, std::string split, int imgsz,
                  bool augment = true);

  std::size_t size() const { return paths_.size(); }
  int         num_classes() const { return (int)class_names_.size(); }
  const std::vector<std::string>& class_names() const { return class_names_; }

  ClassifyExample get(std::size_t idx, uint64_t seed = 0) const;

  struct Batch {
    torch::Tensor imgs;     // [B, 3, H, W]
    torch::Tensor labels;   // [B]
  };
  Batch sample_batch(std::size_t bsz, std::mt19937& rng) const;

 private:
  std::vector<std::string> paths_;
  std::vector<int>         labels_;
  std::vector<std::string> class_names_;
  int                      imgsz_;
  bool                     augment_;
};

struct ClassifyTrainConfig {
  int    epochs        = 30;
  int    batch_size    = 32;
  int    imgsz         = 224;
  double lr0           = 0.01;
  double lrf           = 0.01;
  double momentum      = 0.937;
  double weight_decay  = 0.0005;
  int    warmup_epochs = 1;
  std::string device   = "";
  std::string save_dir = "runs/classify";
  int        log_every = 20;
  // Optional val every N epochs.
  int        val_every = 0;
};

struct ClassifyValResult {
  double top1_acc;
  double top5_acc;
  int    n_total;
};

// Internal template impl — defined in classify_train.cpp with explicit
// instantiations for Yolo8Classify and Yolo11Classify.
template <typename M>
void train_classify_t(M model, const ClassifyDataset& train,
                      const ClassifyDataset* val, ClassifyTrainConfig cfg);
template <typename M>
ClassifyValResult validate_classify_t(M& model, const ClassifyDataset& dataset,
                                      torch::Device device);

inline void train_classify(models::Yolo8Classify model,
                           const ClassifyDataset& train,
                           const ClassifyDataset* val,
                           ClassifyTrainConfig cfg) {
  train_classify_t(std::move(model), train, val, std::move(cfg));
}
inline void train_classify(models::Yolo11Classify model,
                           const ClassifyDataset& train,
                           const ClassifyDataset* val,
                           ClassifyTrainConfig cfg) {
  train_classify_t(std::move(model), train, val, std::move(cfg));
}
inline ClassifyValResult validate_classify(models::Yolo8Classify& model,
                                           const ClassifyDataset& dataset,
                                           torch::Device device) {
  return validate_classify_t(model, dataset, device);
}
inline ClassifyValResult validate_classify(models::Yolo11Classify& model,
                                           const ClassifyDataset& dataset,
                                           torch::Device device) {
  return validate_classify_t(model, dataset, device);
}

}  // namespace yolocpp::tasks
